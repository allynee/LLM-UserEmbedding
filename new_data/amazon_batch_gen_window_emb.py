"""
Test: python new_data/amazon_batch_gen_window_emb.py \
  --input_prf_windows ./data/amazon/usr_prf_windows.pkl \
  --output_dir ./data/amazon/long_embs_test \
  --test

Full: python new_data/amazon_batch_gen_window_emb.py \
  --input_prf_windows ./data/amazon/usr_prf_windows.pkl \
  --output_dir ./data/amazon/long_embs \
  --batch_size 64
"""


import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def mean_pooling(model_output, attention_mask):
    """
    Standard mean pooling over token embeddings, masking out padding.
    """
    token_embeddings = model_output[0]  # [batch, seq_len, hidden]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    counts = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def get_embeddings_batched(texts, tokenizer, model, device, max_length=512):
    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.cpu().numpy()


def build_user_window_embs(flattened, all_embs):
    """
    flattened: list of (uid, widx, is_short, text)
    all_embs:  [N, d] numpy array, in same order as flattened
    Returns:
      window_embs: uid -> list[{window_idx, is_short_term, embedding}]
    """
    window_embs = defaultdict(list)
    for (uid, widx, is_short, _), emb in zip(flattened, all_embs):
        entry = {
            "window_idx": int(widx),
            "is_short_term": bool(is_short),
            "embedding": emb.astype(np.float32),
        }
        window_embs[uid].append(entry)

    # ensure chronological order by window_idx
    for uid in window_embs:
        window_embs[uid].sort(key=lambda e: e["window_idx"])

    return dict(window_embs)


def recency_weighted(E, alpha=1.0):
    """
    E: [T, d] embeddings sorted from oldest -> newest.
    alpha controls how aggressively we upweight recent windows.
    """
    T = E.shape[0]
    if T == 1:
        return E[0]

    # earliest gets exp(-alpha), latest gets exp(0)=1
    weights = np.exp(np.linspace(-alpha, 0.0, T).astype(np.float32))
    weights /= weights.sum()
    return (weights[:, None] * E).sum(axis=0)


def attention_pool(E):
    """
    Simple content-based attention over windows:
    - Query q = mean of window embeddings.
    - Scores = E @ q
    - Weights = softmax(scores)
    - Output = weighted sum
    """
    T, d = E.shape
    if T == 1:
        return E[0]

    q = E.mean(axis=0)  # [d]
    scores = E @ q      # [T]
    scores = scores - scores.max()  # stability
    weights = np.exp(scores)
    weights /= weights.sum()
    return (weights[:, None] * E).sum(axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_prf_windows",
        type=str,
        required=True,
        help="Path to usr_prf_windows.pkl (user -> list[window_dict]).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save long-term user emb variants (avg/sum/recency/attn).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for encoding.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="If set, embed just one user with multiple windows and print sample.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/embeddinggemma-300m",
        help="HF model name for text embeddings.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading window-level profiles from {args.input_prf_windows} ...")
    profiles = load_pickle(args.input_prf_windows)
    if not profiles:
        print("No profiles found.")
        raise SystemExit

    # Flatten to (uid, window_idx, is_short_term, text)
    print("Preparing window texts...")
    flattened = []
    for uid, win_list in profiles.items():
        if not isinstance(win_list, list):
            continue
        for w in win_list:
            widx = w.get("window_idx")
            is_short = bool(w.get("is_short_term", False))
            text = (w.get("summarization", "") or "").strip()
            if not text:
                continue
            flattened.append((int(uid), int(widx), is_short, text))

    if not flattened:
        print("No windows with non-empty summarization found.")
        raise SystemExit

    print(f"Total windows with text: {len(flattened)}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load encoder
    model_name = args.model_name
    print(f"Loading model {model_name} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        raise SystemExit

    # TEST MODE: embed all windows for one user & show aggregated variants
    if args.test:
        windows_by_user = {}
        for uid, widx, is_short, text in flattened:
            windows_by_user.setdefault(uid, []).append((widx, is_short, text))

        candidate_users = [u for u, ws in windows_by_user.items() if len(ws) >= 3]
        if not candidate_users:
            print("No users with >= 3 windows, falling back to first user in data.")
            u0 = sorted(windows_by_user.keys())[0]
        else:
            u0 = sorted(candidate_users)[0]

        win_list = sorted(windows_by_user[u0], key=lambda x: x[0])
        print(f"[TEST] User {u0} has {len(win_list)} windows. Embedding all of them...")

        texts = [w[2] for w in win_list]
        embs = get_embeddings_batched(texts, tokenizer, model, device)

        for (widx, is_short, text), emb in zip(win_list, embs):
            print(f"\n----- Window idx {widx} (is_short_term={is_short}) -----")
            print("Text snippet:", text[:200].replace("\n", " "), "...")
            print("Embedding shape:", emb.shape)
            print("First 5 dims:", emb[:5])

        # Show fused long-term variants for this user
        E = embs  # [T, d]
        avg_vec = E.mean(axis=0)
        sum_vec = E.sum(axis=0)
        rec_vec = recency_weighted(E, alpha=1.0)
        att_vec = attention_pool(E)

        print("\n[TEST] Aggregated variants for this user:")
        print("avg  first 5 dims:", avg_vec[:5])
        print("sum  first 5 dims:", sum_vec[:5])
        print("rec  first 5 dims:", rec_vec[:5])
        print("attn first 5 dims:", att_vec[:5])
        print("\n[TEST] Done.")
        raise SystemExit

    # FULL RUN: encode all windows
    print("Encoding all windows...")
    all_texts = [x[3] for x in flattened]
    batch_size = args.batch_size
    all_embs_parts = []

    for i in tqdm(range(0, len(all_texts), batch_size), desc="Encoding"):
        batch_texts = all_texts[i : i + batch_size]
        batch_embs = get_embeddings_batched(batch_texts, tokenizer, model, device)
        all_embs_parts.append(batch_embs)

    all_embs = np.concatenate(all_embs_parts, axis=0)
    assert all_embs.shape[0] == len(flattened)
    emb_dim = all_embs.shape[1]
    print(f"Encoded window embeddings shape: {all_embs.shape}")

    # Build user -> list of window embeddings
    print("Assembling user->window embeddings dict...")
    window_embs = build_user_window_embs(flattened, all_embs)

    # Save window-level embeddings too (nice to have)
    window_embs_path = os.path.join(args.output_dir, "window_embs.pkl")
    print(f"Saving window-level embeddings to {window_embs_path} ...")
    with open(window_embs_path, "wb") as f:
        pickle.dump(window_embs, f)

    # Now build usr_emb_np-style matrices for each fusion:
    print("Building long-term user embedding matrices (avg/sum/recency/attn) ...")

    max_uid = max(profiles.keys())  # assumes keys are user_ints 0..N-1
    num_users = max_uid + 1

    avg_mat = np.zeros((num_users, emb_dim), dtype=np.float32)
    sum_mat = np.zeros((num_users, emb_dim), dtype=np.float32)
    rec_mat = np.zeros((num_users, emb_dim), dtype=np.float32)
    att_mat = np.zeros((num_users, emb_dim), dtype=np.float32)

    for uid in range(num_users):
        if uid not in window_embs:
            # user with no windows or no text -> stays as zero vector
            continue

        E = np.stack([e["embedding"] for e in window_embs[uid]], axis=0)  # [T, d]

        avg_vec = E.mean(axis=0)
        sum_vec = E.sum(axis=0)
        rec_vec = recency_weighted(E, alpha=1.0)
        att_vec = attention_pool(E)

        avg_mat[uid] = avg_vec
        sum_mat[uid] = sum_vec
        rec_mat[uid] = rec_vec
        att_mat[uid] = att_vec

    # Save each as its own usr_emb_np.pkl-style file
    out_avg = os.path.join(args.output_dir, "long_emb_avg_usr_emb_np.pkl")
    out_sum = os.path.join(args.output_dir, "long_emb_sum_usr_emb_np.pkl")
    out_rec = os.path.join(args.output_dir, "long_emb_recency_usr_emb_np.pkl")
    out_att = os.path.join(args.output_dir, "long_emb_attn_usr_emb_np.pkl")

    print(f"Saving avg embeddings to {out_avg} ...")
    with open(out_avg, "wb") as f:
        pickle.dump(avg_mat, f)

    print(f"Saving sum embeddings to {out_sum} ...")
    with open(out_sum, "wb") as f:
        pickle.dump(sum_mat, f)

    print(f"Saving recency-weighted embeddings to {out_rec} ...")
    with open(out_rec, "wb") as f:
        pickle.dump(rec_mat, f)

    print(f"Saving attention-pooled embeddings to {out_att} ...")
    with open(out_att, "wb") as f:
        pickle.dump(att_mat, f)

    print("All done.")
