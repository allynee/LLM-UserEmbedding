"""
Prepare Yelp for:
  - short-term vs long-term interaction graph
  - windowed user histories for LLM profiles

Outputs (under --data-dir, default ./data/yelp):

  - trn_short_mat.pkl
      sparse matrix: (num_users x num_items), only last K training
      interactions per user, by timestamp.

  - user_windows.jsonl
      one JSON per user:
      {
        "user_int": <int>,
        "windows": [
          {
            "window_idx": 0,
            "is_short_term": false,
            "interactions": [
              {
                "item_int": <int>,
                "timestamp": <int>,
                "reviewText": "<str>"
              },
              ...
            ]
          },
          ...
        ]
      }
"""

import argparse
import gzip
import json
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def smart_open(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_and_invert_mapping(path: Path, id_field: str, orig_field: str):
    """
    For files like:
      {"uid": 0, "user_id": "..."}
      {"iid": 0, "business_id": "..."}

    Returns a dict:
        original_id (str) -> internal_id (int)
    """
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                internal = int(obj[id_field])
                orig = str(obj[orig_field])
                mapping[orig] = internal
            except json.JSONDecodeError:
                continue
    if not mapping:
        raise ValueError(f"No mappings loaded from {path}")
    return mapping


def build_short_train_mat(events_by_user, num_users, num_items, k):
    """
    events_by_user: user_int -> list[(timestamp, item_int)]

    For each user, keep last k items (by time).
    Returns a binary COO matrix of shape (num_users, num_items).
    """
    rows, cols, data = [], [], []

    for u, evts in events_by_user.items():
        if not evts:
            continue
        evts = sorted(evts, key=lambda x: x[0])
        last = evts[-k:] if len(evts) > k else evts

        # deduplicate items, keeping the latest occurrence
        seen = set()
        dedup_last = []
        for ts, i in reversed(last):
            if i in seen:
                continue
            seen.add(i)
            dedup_last.append((ts, i))
        dedup_last.reverse()

        for _, i in dedup_last:
            rows.append(u)
            cols.append(i)
            data.append(1.0)

    mat = coo_matrix(
        (np.array(data, dtype=np.float32), (np.array(rows), np.array(cols))),
        shape=(num_users, num_items),
        dtype=np.float32,
    )
    return mat


def build_windows(events_by_user, window_size):
    """
    events_by_user: user_int -> list[(timestamp, item_int, reviewText)]

    Returns a dict:
      user_int -> list[{
         "window_idx": int,
         "is_short_term": bool,
         "interactions": [ {item_int, timestamp, reviewText}, ... ]
      }]
    """
    user_windows = {}

    for u, evts in events_by_user.items():
        if not evts:
            continue
        evts = sorted(evts, key=lambda x: x[0])  # by ts

        windows = []
        # non-overlapping chunks of size window_size
        for w_idx in range(0, len(evts), window_size):
            chunk = evts[w_idx:w_idx + window_size]
            windows.append(chunk)

        if not windows:
            continue

        last_idx = len(windows) - 1
        win_objs = []
        for j, chunk in enumerate(windows):
            win_objs.append({
                "window_idx": j,
                "is_short_term": (j == last_idx),
                "interactions": [
                    {
                        "item_int": i,
                        "timestamp": int(ts),
                        "reviewText": txt,
                    }
                    for (ts, i, txt) in chunk
                ]
            })

        user_windows[u] = win_objs

    return user_windows


def parse_yelp_date(date_str):
    """
    Parses date string like '2018-07-07 22:09:11' to unix timestamp.
    Returns 0 if parsing fails.
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        return int(dt.timestamp())
    except (ValueError, TypeError):
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw",
        default="./Yelp JSON/yelp_dataset/yelp_academic_dataset_review.json",
        help="Path to Yelp reviews raw file (yelp_academic_dataset_review.json or .json.gz)",
    )
    ap.add_argument(
        "--data-dir",
        default="./data/yelp",
        help="Dir containing trn_mat.pkl",
    )
    ap.add_argument(
        "--user-map",
        default="./data/mapper/yelp_user.json",
        help="JSON mapping internal_user_id -> original user_id",
    )
    ap.add_argument(
        "--item-map",
        default="./data/mapper/yelp_item.json",
        help="JSON mapping internal_item_id -> original business_id",
    )
    ap.add_argument(
        "--short-k",
        type=int,
        default=10,
        help="K for last-K short-term graph per user",
    )
    ap.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Number of interactions per window for LLM profiles",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    raw_path = Path(args.raw)
    user_map_path = Path(args.user_map)
    item_map_path = Path(args.item_map)

    # 1) load train/val/test matrices
    trn_pkl = data_dir / "trn_mat.pkl"
    if not trn_pkl.exists():
        raise FileNotFoundError(f"Missing {trn_pkl}")

    print(f"Loading matrices from {data_dir} ...")
    with open(trn_pkl, "rb") as f:
        trn_mat = pickle.load(f)

    trn_csr: csr_matrix = trn_mat.tocsr()
    num_users, num_items = trn_mat.shape
    print(f"Train matrix shape: {trn_mat.shape}")

    # 2) invert mappings
    print("Loading ID mappings ...")
    orig2user = load_and_invert_mapping(user_map_path, id_field="uid", orig_field="user_id")
    orig2item = load_and_invert_mapping(item_map_path, id_field="iid", orig_field="business_id")

    # 3) stream raw data
    print(f"Streaming raw Yelp file from {raw_path} ...")
    short_events_by_user = defaultdict(list)   # for last-K graph
    window_events_by_user = defaultdict(list)  # for windows

    n_lines = n_mapped = n_train_edges = 0

    with smart_open(raw_path) as f:
        for line in f:
            n_lines += 1
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            u_orig = obj.get("user_id")
            i_orig = obj.get("business_id")

            if u_orig is None or i_orig is None:
                continue

            if u_orig not in orig2user or i_orig not in orig2item:
                continue
            u = orig2user[u_orig]
            i = orig2item[i_orig]
            if u >= num_users or i >= num_items:
                continue

            n_mapped += 1

            date_str = obj.get("date")
            ts = parse_yelp_date(date_str)

            review = obj.get("text", "")

            # only TRAIN edges should affect short-term graph + windows
            if trn_csr[u, i] != 0:
                n_train_edges += 1
                short_events_by_user[u].append((ts, i))
                window_events_by_user[u].append((ts, i, review))

    print(f"Lines read: {n_lines}")
    print(f"Mapped to processed IDs: {n_mapped}")
    print(f"Train edges with timestamps: {n_train_edges}")

    # 4) build last-K short-term matrix
    print(f"Building last-K train matrix (K={args.short_k}) ...")
    short_mat = build_short_train_mat(
        short_events_by_user,
        num_users=num_users,
        num_items=num_items,
        k=args.short_k,
    )
    short_pkl = data_dir / "trn_short_mat.pkl"
    with open(short_pkl, "wb") as f:
        pickle.dump(short_mat, f)
    print(f"Saved short-term matrix to {short_pkl} (shape={short_mat.shape})")

    # 5) build per-user windows for LLM
    print(f"Building windows of size {args.window_size} for LLM profiles ...")
    user_windows = build_windows(window_events_by_user, window_size=args.window_size)

    windows_path = data_dir / "user_windows.jsonl"
    with open(windows_path, "w", encoding="utf-8") as f:
        for u, wins in user_windows.items():
            obj = {"user_int": u, "windows": wins}
            f.write(json.dumps(obj) + "\n")

    print(f"Wrote per-user windows to {windows_path}")
    print("Done.")


if __name__ == "__main__":
    main()
