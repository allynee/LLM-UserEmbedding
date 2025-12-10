"""
python new_data/gen_short_profile_from_items.py \
  --short_mat ./data/amazon/trn_short_mat.pkl \
  --itm_emb  ./data/amazon/itm_emb_np.pkl \
  --itm_prf  ./data/amazon/itm_prf.pkl \
  --output_dir ./data/amazon/short_embs \
  --mode both \
  --batch_size 64

# Yelp
python new_data/gen_short_profile_from_items.py \
  --short_mat ./data/yelp/trn_short_mat.pkl \
  --itm_emb  ./data/yelp/itm_emb_np.pkl \
  --itm_prf  ./data/yelp/itm_prf.pkl \
  --output_dir ./data/yelp/short_embs \
  --mode both \
  --batch_size 64

# Steam
python new_data/gen_short_profile_from_items.py \
  --short_mat ./data/steam/trn_short_mat.pkl \
  --itm_emb  ./data/steam/itm_emb_np.pkl \
  --itm_prf  ./data/steam/itm_prf.pkl \
  --output_dir ./data/steam/short_embs \
  --mode both \
  --batch_size 64
"""

import argparse
import os
import pickle

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


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


def get_embeddings_batched(texts, tokenizer, model, device, max_length=512, batch_size=64):
    """
    Encode a list of texts into embeddings using mean pooling.
    Returns numpy array [len(texts), dim].
    """
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding short-text"):
        batch_texts = texts[i : i + batch_size]
        encoded_input = tokenizer(
            batch_texts,
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
        all_embs.append(sentence_embeddings.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def build_avg_item_embs(short_mat, item_embs):
    """
    short_mat: scipy sparse [num_users, num_items], last-K interactions
    item_embs: numpy [num_items, dim]

    Returns:
      usr_emb_avg: numpy [num_users, dim], mean of item embeddings for non-zero entries.
    """
    short_csr = short_mat.tocsr()
    num_users, num_items = short_csr.shape
    dim = item_embs.shape[1]

    usr_emb_avg = np.zeros((num_users, dim), dtype=np.float32)

    for u in tqdm(range(num_users), desc="Averaging item embeddings"):
        row = short_csr.getrow(u)
        idx = row.indices
        if idx.size == 0:
            continue
        usr_emb_avg[u] = item_embs[idx].mean(axis=0)

    return usr_emb_avg


def build_text_concat_embs(short_mat, item_profiles, model_name, batch_size):
    """
    Build short-term *text* embeddings by concatenating the text profiles
    of the recent items for each user and encoding them in one go.

    short_mat: scipy sparse [num_users, num_items]
    item_profiles: dict[iid] -> dict with 'summarization' / 'profile' / 'description' etc.
    model_name: HF model to use (e.g. google/embeddinggemma-300m)

    Returns:
      usr_emb_text: numpy [num_users, dim]
    """
    short_csr = short_mat.tocsr()
    num_users, num_items = short_csr.shape

    # 1) Build per-user text
    user_texts = {}
    for u in tqdm(range(num_users), desc="Building per-user short texts"):
        row = short_csr.getrow(u)
        idx = row.indices
        if idx.size == 0:
            continue

        chunks = []
        for i in idx:
            prof = item_profiles.get(i)
            if isinstance(prof, dict):
                txt = (
                    prof.get("summarization")
                    or prof.get("profile")
                    or prof.get("description")
                    or ""
                )
            else:
                txt = str(prof) if prof is not None else ""

            txt = txt.strip()
            if not txt:
                continue

            # Simple tag per item (optional, but gives structure):
            chunks.append(f"[ITEM {i}] {txt}")

        if not chunks:
            continue

        long_text = "\n\n".join(chunks)
        user_texts[u] = long_text

    if not user_texts:
        print("No users with any short-text available. Returning zeros.")
        # We don't know dim yet, will infer after encoder; but if here, we bail.
        return None

    uids = sorted(user_texts.keys())
    texts = [user_texts[u] for u in uids]

    # 2) Load encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[text_concat] Using device: {device}")
    print(f"[text_concat] Loading model {model_name} ...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return None

    # 3) Encode
    print(f"[text_concat] Encoding {len(texts)} user short-text profiles ...")
    all_embs = get_embeddings_batched(
        texts, tokenizer, model, device, batch_size=batch_size
    )
    dim = all_embs.shape[1]

    # 4) Build full matrix [num_users, dim] with zeros for missing
    usr_emb_text = np.zeros((num_users, dim), dtype=np.float32)
    for i, u in enumerate(uids):
        usr_emb_text[u] = all_embs[i]

    return usr_emb_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--short_mat",
        type=str,
        required=True,
        help="Path to trn_short_mat.pkl (last-K interactions per user)",
    )
    parser.add_argument(
        "--itm_emb",
        type=str,
        required=True,
        help="Path to itm_emb_np.pkl (item embedding matrix)",
    )
    parser.add_argument(
        "--itm_prf",
        type=str,
        default=None,
        help="Path to itm_prf.pkl (item text profiles, needed for text_concat mode)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save short-term user embeddings (e.g. ./data/amazon/short_embs)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="avg",
        choices=["avg", "text_concat", "both"],
        help="Which short-term representation(s) to compute.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/embeddinggemma-300m",
        help="HF model name for text_concat mode.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for encoding text_concat.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="If set, run on a single user and print samples instead of saving.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading short-term matrix from {args.short_mat} ...")
    short_mat = load_pickle(args.short_mat)
    if not sp.isspmatrix(short_mat):
        short_mat = sp.coo_matrix(short_mat)
    num_users, num_items = short_mat.shape
    print(f"short_mat shape: {short_mat.shape}")

    print(f"Loading item embeddings from {args.itm_emb} ...")
    item_embs = load_pickle(args.itm_emb)
    print(f"item_embs shape: {item_embs.shape}")

    # --- TEST MODE: inspect a single user ---
    if args.test:
        short_csr = short_mat.tocsr()
        # pick first user with at least one short interaction
        u0 = None
        for u in range(num_users):
            if short_csr.getrow(u).nnz > 0:
                u0 = u
                break

        if u0 is None:
            print("No users with short-term interactions found.")
            exit()

        row = short_csr.getrow(u0)
        idx = row.indices
        print(f"[TEST] User {u0} short-term items: {idx.tolist()}")

        if args.mode in ["avg", "both"]:
            avg_emb = item_embs[idx].mean(axis=0)
            print("[TEST] avg embedding shape:", avg_emb.shape)
            print("[TEST] avg first 5 dims:", avg_emb[:5])

        if args.mode in ["text_concat", "both"]:
            if args.itm_prf is None:
                print("[TEST] text_concat requested but --itm_prf not provided.")
            else:
                print(f"Loading item profiles from {args.itm_prf} ...")
                item_profiles = load_pickle(args.itm_prf)
                # Build one user text manually
                chunks = []
                for i in idx:
                    prof = item_profiles.get(i)
                    if isinstance(prof, dict):
                        txt = (
                            prof.get("summarization")
                            or prof.get("profile")
                            or prof.get("description")
                            or ""
                        )
                    else:
                        txt = str(prof) if prof is not None else ""
                    txt = txt.strip()
                    if not txt:
                        continue
                    chunks.append(f"[ITEM {i}] {txt}")
                long_text = "\n\n".join(chunks)
                print("\n[TEST] short-text for this user (first 300 chars):")
                print(long_text[:300].replace("\n", " ") + "...")

                # Encode once
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                tokenizer = AutoTokenizer.from_pretrained(args.model_name)
                model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).to(device)
                model.eval()
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                emb = get_embeddings_batched([long_text], tokenizer, model, device, batch_size=1)
                print("[TEST] text_concat emb shape:", emb.shape)
                print("[TEST] text_concat first 5 dims:", emb[0][:5])

        print("[TEST] Done.")
        exit()

    # --- FULL RUN ---

    usr_emb_avg = None
    usr_emb_text = None

    # 1) Average of item embeddings
    if args.mode in ["avg", "both"]:
        usr_emb_avg = build_avg_item_embs(short_mat, item_embs)
        out_path_avg = os.path.join(args.output_dir, "usr_emb_np_short_avg.pkl")
        print(f"Saving avg-based short user embeddings to {out_path_avg} ...")
        with open(out_path_avg, "wb") as f:
            pickle.dump(usr_emb_avg, f)

    # 2) Text-concat embeddings
    if args.mode in ["text_concat", "both"]:
        if args.itm_prf is None:
            print("ERROR: text_concat mode requires --itm_prf (item profiles).")
        else:
            print(f"Loading item profiles from {args.itm_prf} ...")
            item_profiles = load_pickle(args.itm_prf)
            usr_emb_text = build_text_concat_embs(
                short_mat,
                item_profiles,
                model_name=args.model_name,
                batch_size=args.batch_size,
            )
            if usr_emb_text is not None:
                out_path_text = os.path.join(args.output_dir, "usr_emb_np_short_text_concat.pkl")
                print(f"Saving text-concat short user embeddings to {out_path_text} ...")
                with open(out_path_text, "wb") as f:
                    pickle.dump(usr_emb_text, f)

    print("Done.")
