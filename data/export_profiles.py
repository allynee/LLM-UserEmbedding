"""
Export user and item profiles (Amazon, Steam, Yelp, or other supported dataset) to CSVs,
including raw IDs via mapper files.

Assumed constant paths:
  Profiles:
    data/{dataset}/usr_prf.pkl
    data/{dataset}/itm_prf.pkl
  Mappers (JSON):
    data/mapper/*{dataset}*user*.json
    data/mapper/*{dataset}*item*.json
Outputs:
  data/exports/{dataset}_user_profiles.csv
  data/exports/{dataset}_item_profiles.csv

Columns:
  Users: user_id_int, user_id_raw, profile_text
  Items: item_id_int, item_id_raw, profile_text

Dataset-specific ID fields:
  Amazon: reviewerID (users), asin (items)
  Steam: username (users), product_id (items)
  Yelp: user_id (users), business_id (items)
"""

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict


BASE_DATA_DIR = Path("data")
MAPPER_DIR = BASE_DATA_DIR / "mapper"
EXPORT_DIR = BASE_DATA_DIR / "exports"


def _safe_profile_text(x: Any) -> str:
    if isinstance(x, dict):
        for k in ("PROFILE", "profile", "text", "desc", "description"):
            if k in x and isinstance(x[k], str):
                return x[k]
        return json.dumps(x, ensure_ascii=False)
    if isinstance(x, (list, tuple)):
        try:
            return " ".join(str(t) for t in x)
        except Exception:
            return str(x)
    if isinstance(x, str):
        return x
    return str(x)


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_mapper_json(path: Path, key_field: str, val_field: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if key_field not in obj or val_field not in obj:
                continue
            k = str(int(obj[key_field])) if isinstance(obj[key_field], (int, float, str)) else str(obj[key_field])
            v = str(obj[val_field])
            mapping[k] = v
    if not mapping:
        raise ValueError(f"No mappings parsed from {path} using fields '{key_field}' and '{val_field}'.")
    return mapping


def _iter_profiles(prof: Any):
    if isinstance(prof, list):
        for i, v in enumerate(prof):
            yield i, _safe_profile_text(v)
    elif isinstance(prof, dict):
        for k, v in prof.items():
            try:
                i = int(k)
            except Exception:
                continue
            yield i, _safe_profile_text(v)
    else:
        raise ValueError("Unexpected profile structure; expected list or dict.")


def export_user_profiles(usr_prf_path: Path, user_mapper: Dict[str, str], out_csv: Path) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id_int", "user_id_raw", "profile_text"])
        usr_prf = _load_pickle(usr_prf_path)
        for u_int, text in _iter_profiles(usr_prf):
            user_id = user_mapper.get(str(u_int), "")
            w.writerow([u_int, user_id, text])
            n += 1
    return n


def export_item_profiles(itm_prf_path: Path, item_mapper: Dict[str, str], out_csv: Path) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id_int", "item_id_raw", "profile_text"])
        itm_prf = _load_pickle(itm_prf_path)
        for i_int, text in _iter_profiles(itm_prf):
            item_id = item_mapper.get(str(i_int), "")
            w.writerow([i_int, item_id, text])
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="Export user/item profiles")
    ap.add_argument("--dataset", default="amazon", help="Dataset name (default: amazon)")
    args = ap.parse_args()

    dataset = args.dataset
    user_prof_path = BASE_DATA_DIR / dataset / "usr_prf.pkl"
    item_prof_path = BASE_DATA_DIR / dataset / "itm_prf.pkl"

    if not user_prof_path.exists():
        raise FileNotFoundError(f"User profile pickle not found: {user_prof_path}")
    if not item_prof_path.exists():
        raise FileNotFoundError(f"Item profile pickle not found: {item_prof_path}")

    user_mapper_path = MAPPER_DIR / f"{dataset}_user.json"
    item_mapper_path = MAPPER_DIR / f"{dataset}_item.json"

    if dataset == "steam":
        user_val_field = "username"
        item_val_field = "product_id"
    elif dataset == "amazon":
        user_val_field = "reviewerID"
        item_val_field = "asin"
    elif dataset == "yelp":
        user_val_field = "user_id"
        item_val_field = "business_id"
    else:
        raise ValueError(f"Unsupported dataset for export: {dataset}")

    user_mapper = _load_mapper_json(user_mapper_path, key_field="uid", val_field=user_val_field)
    item_mapper = _load_mapper_json(item_mapper_path, key_field="iid", val_field=item_val_field)

    out_users = EXPORT_DIR / f"{dataset}_user_profiles.csv"
    out_items = EXPORT_DIR / f"{dataset}_item_profiles.csv"

    n_users = export_user_profiles(user_prof_path, user_mapper, out_users)
    n_items = export_item_profiles(item_prof_path, item_mapper, out_items)

    print(f"Wrote {n_users} users to {out_users}")
    print(f"Wrote {n_items} items to {out_items}")


if __name__ == "__main__":
    main()
