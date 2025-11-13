import argparse
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from scipy.sparse import csr_matrix
import yaml

from config.configurator import configs
from data_utils.build_data_handler import build_data_handler
from models.bulid_model import build_model
from trainer.metrics import Metric


BASE = Path(__file__).resolve().parents[1]
EXPORT_DIR = BASE / "encoder" / "exports"


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _load_modelconf(model_full: str, dataset: str) -> None:
    yml = BASE / "encoder" / "config" / "modelconf" / f"{model_full}.yml"
    if not yml.exists():
        raise FileNotFoundError(f"Model config not found: {yml}")
    with open(yml, "r") as f:
        conf = yaml.safe_load(f)
    # merge the yaml into configs
    _deep_update(configs, conf)


def _choose_checkpoint(model_full: str, dataset: str, explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p
    ckpt_dir = BASE / "encoder" / "checkpoint" / model_full
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")
    cands = sorted(ckpt_dir.glob(f"{model_full}-{dataset}-*.pth"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        cands = sorted(ckpt_dir.glob("*.pth"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No .pth checkpoints in {ckpt_dir}")
    return cands[0]


def _per_user_metrics(top_items: np.ndarray, gt_items: List[int], ks=(5, 10, 20)) -> Tuple[List[float], List[float]]:
    recalls, ndcgs = [], []
    m = len(gt_items)
    if m == 0:
        return [0.0] * len(ks), [0.0] * len(ks)

    maxK = int(max(ks))
    discounts = 1.0 / np.log2(np.arange(2, maxK + 2))
    hits = np.isin(top_items[:maxK], gt_items).astype(np.float32)

    for K in ks:
        hK = hits[:K]
        # recall@K
        recalls.append(float(hK.sum() / m))
        # ndcg@K with IDCG computed for this K
        ideal_len = min(m, K)
        ideal_dcg = discounts[:ideal_len].sum() if ideal_len > 0 else 1.0
        dcg = float((hK * discounts[:K]).sum())
        ndcgs.append(float(dcg / (ideal_dcg if ideal_dcg > 0 else 1.0)))

    return recalls, ndcgs


def main():
    ap = argparse.ArgumentParser(description="Dump per-user recs & metrics for a saved checkpoint.")
    ap.add_argument("--dataset", default="amazon")
    ap.add_argument("--model", default="lightgcn")
    ap.add_argument("--paradigm", choices=["baseline", "plus", "gene"], required=True)
    ap.add_argument("--ckpt", default=None, help="optional explicit path to .pth")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    dataset = args.dataset
    model_full = args.model if args.paradigm == "baseline" else f"{args.model}_{args.paradigm}"

    # baseline config + YAML modelconf (so hyper params like kd_weight exist)
    configs['model']['name'] = model_full
    configs['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load the modelconf YAML to populate hyper_config etc.
    _load_modelconf(model_full, dataset)
    # Override dataset name AFTER loading model config (YAML may have default dataset)
    configs['data']['name'] = dataset
    # make sure test ks exist
    configs['test']['k'] = [5, 10, 20]
    configs['test']['metrics'] = ['recall', 'ndcg']

    data_handler = build_data_handler()
    data_handler.load_data()
    model = build_model(data_handler).to(configs['device'])

    ckpt_path = _choose_checkpoint(model_full, dataset, args.ckpt)
    state = torch.load(ckpt_path, map_location=configs['device'])
    model.load_state_dict(state)
    model.eval()
    print(f"\nLoaded checkpoint: {ckpt_path}")

    trn_csr: csr_matrix = data_handler.trn_mat.tocsr()
    user_pos_lists = data_handler.test_dataloader.dataset.user_pos_lists

    dl = data_handler.test_dataloader
    Kdump = int(args.topk)
    ks_metrics = (5, 10, 20)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = EXPORT_DIR / f"recs_{model_full}_{dataset}.csv"

    header = [
        "user_id_int",
        "n_train_interactions",
        "train_item_ids", "test_item_ids",
        "topk_item_ids", "topk_scores",
        "recall@5", "recall@10", "recall@20",
        "ndcg@5", "ndcg@10", "ndcg@20",
    ]

    # accumulators for our own aggregate metrics for sanity check
    sum_recalls = np.zeros(3, dtype=np.float64)  # for @5,@10,@20
    sum_ndcgs = np.zeros(3, dtype=np.float64)
    n_users_eval = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for _, tem in enumerate(dl):
            if not isinstance(tem, list):
                tem = [tem]
            users_batch = tem[0].numpy().tolist()
            batch_data = [x.long().to(configs['device']) for x in tem]

            with torch.no_grad():
                scores: torch.Tensor = model.full_predict(batch_data)  # [B, n_items]

            top_scores, top_items = torch.topk(scores, k=Kdump, dim=1)
            top_items = top_items.detach().cpu().numpy()
            top_scores = top_scores.detach().cpu().numpy()

            for i, u in enumerate(users_batch):
                train_items = trn_csr[u].indices.tolist()
                test_items = user_pos_lists[u]
                if len(test_items) == 0:
                    continue

                recalls, ndcgs = _per_user_metrics(top_items[i], test_items, ks=ks_metrics)

                # accumulate our aggregates
                sum_recalls += np.array(recalls, dtype=np.float64)
                sum_ndcgs += np.array(ndcgs, dtype=np.float64)
                n_users_eval += 1

                train_str = "|".join(map(str, train_items))
                test_str = "|".join(map(str, test_items))
                rec_ids = "|".join(map(str, top_items[i]))
                rec_scs = "|".join(f"{x:.6f}" for x in top_scores[i])

                w.writerow([
                    u,
                    len(train_items),
                    train_str, test_str,
                    rec_ids, rec_scs,
                    f"{recalls[0]:.6f}", f"{recalls[1]:.6f}", f"{recalls[2]:.6f}",
                    f"{ndcgs[0]:.6f}",  f"{ndcgs[1]:.6f}",  f"{ndcgs[2]:.6f}",
                ])

    print(f"\nWrote CSV to {out_csv}")

    # Sanity check
    agg_recalls = (sum_recalls / n_users_eval)
    agg_ndcgs = (sum_ndcgs / n_users_eval)
    print("[Our aggregate]")
    print(f"  recall@5={agg_recalls[0]:.6f}  recall@10={agg_recalls[1]:.6f}  recall@20={agg_recalls[2]:.6f}")
    print(f"  ndcg@5={agg_ndcgs[0]:.6f}    ndcg@10={agg_ndcgs[1]:.6f}    ndcg@20={agg_ndcgs[2]:.6f}")

    official = Metric().eval(model, data_handler.test_dataloader)
    print("[Official Metric().eval]")
    print(f"  recall@5={official['recall'][0]:.6f}  recall@10={official['recall'][1]:.6f}  recall@20={official['recall'][2]:.6f}")
    print(f"  ndcg@5={official['ndcg'][0]:.6f}    ndcg@10={official['ndcg'][1]:.6f}    ndcg@20={official['ndcg'][2]:.6f}")


if __name__ == "__main__":
    main()
