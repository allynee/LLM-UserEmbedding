# new_data/amazon_batch_gen_user_longtext_emb.py
"""
Usage:

TEST (just inspect one user):
python new_data/amazon_batch_gen_user_longtext_emb.py \
  --input_prf_windows ./data/amazon/usr_prf_windows.pkl \
  --output_emb ./data/amazon/long_embs/usr_emb_np_long_text_test.pkl \
  --test

FULL:
python new_data/amazon_batch_gen_user_longtext_emb.py \
  --input_prf_windows ./data/amazon/usr_prf_windows.pkl \
  --output_emb ./data/amazon/long_embs/usr_emb_np_long_text.pkl \
  --batch_size 64
"""

import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_prf_windows",
        type=str,
        required=True,
        help="Path to usr_prf_windows.pkl (user_int -> list[window_dict])",
    )
    parser.add_argument(
        "--output_emb",
        type=str,
        required=True,
        help="Path to output usr_emb_np.pkl (user-level long text embeddings)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/embeddinggemma-300m",
        help="HF model name for text embeddings",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="If set, embed just one user and print sample.",
    )
    args = parser.parse_args()

    print(f"Loading window-level profiles from {args.input_prf_windows} ...")
    profiles = load_pickle(args.input_prf_windows)
    if not profiles:
        print("No profiles found.")
        exit()

    # Build per-user long text by concatenating all window summaries
    # in order of window_idx.
    print("Building per-user long texts from windows...")
    user_texts = {}  # uid -> concatenated text

    for uid, win_list in profiles.items():
        if not isinstance(win_list, list):
            continue
        # sort by window_idx
        sorted_wins = sorted(
            win_list, key=lambda w: int(w.get("window_idx", 0))
        )

        chunks = []
        for w in sorted_wins:
            widx = w.get("window_idx", 0)
            is_short = bool(w.get("is_short_term", False))
            summary = (w.get("summarization") or "").strip()
            if not summary:
                continue
            # Optional tagging so the model can see some structure
            tag = "RECENT" if is_short else "PAST"
            chunks.append(f"[{tag} window {widx}] {summary}")

        if not chunks:
            continue

        long_text = "\n\n".join(chunks)
        user_texts[int(uid)] = long_text

    if not user_texts:
        print("No users with non-empty long texts were found.")
        exit()

    uids = sorted(user_texts.keys())
    all_texts = [user_texts[uid] for uid in uids]

    print(f"Total users with text: {len(uids)}")

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
        exit()

    # TEST MODE: inspect a single user's long text + embedding
    if args.test:
        uid0 = uids[0]
        text0 = user_texts[uid0]
        print(f"[TEST] User {uid0} long text (first 300 chars):")
        print(text0[:300].replace("\n", " ") + "...")
        emb = get_embeddings_batched([text0], tokenizer, model, device)
        print("Embedding shape:", emb.shape)
        print("First 5 dims:", emb[0][:5])
        print("[TEST] Done.")
        exit()

    # FULL RUN
    print("Encoding all user long texts...")

    batch_size = args.batch_size
    all_embs_chunks = []

    for i in tqdm(range(0, len(all_texts), batch_size), desc="Encoding"):
        batch_texts = all_texts[i : i + batch_size]
        batch_embs = get_embeddings_batched(batch_texts, tokenizer, model, device)
        all_embs_chunks.append(batch_embs)

    all_embs = np.concatenate(all_embs_chunks, axis=0)
    assert all_embs.shape[0] == len(uids)

    # Build final [num_users, dim] matrix, filling missing users with zeros
    max_uid = max(uids)
    num_users = max_uid + 1
    emb_dim = all_embs.shape[1]

    print(f"Creating embedding matrix of shape ({num_users}, {emb_dim})...")
    emb_matrix = np.zeros((num_users, emb_dim), dtype=np.float32)

    for i, uid in enumerate(uids):
        emb_matrix[uid] = all_embs[i]

    print(f"Saving user-level long-text embeddings to {args.output_emb} ...")
    os.makedirs(os.path.dirname(args.output_emb), exist_ok=True)
    with open(args.output_emb, "wb") as f:
        pickle.dump(emb_matrix, f)

    print("Done.")
