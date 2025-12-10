import argparse
import pickle
from pathlib import Path
from io import StringIO
from contextlib import redirect_stdout

import numpy as np
import pandas as pd


# ----------------- Helpers ----------------- #

def parse_id_list(s):
    """Parse a '1|2|3' style string into a list of ints."""
    if pd.isna(s) or s is None or str(s).strip() == "":
        return []
    return [int(x) for x in str(s).split("|") if x != ""]


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_item_profile(itm_prf_dict, item_id_int):
    """
    Get item title/profile text by integer ID from itm_prf.pkl.

    itm_prf entry is typically:
      { 'summarization': ..., 'profile': ..., 'description': ..., ... }
    but we handle other shapes defensively.
    """
    entry = itm_prf_dict.get(item_id_int)

    if isinstance(entry, dict):
        title = (
            entry.get("title")
            or entry.get("name")
            or entry.get("raw_title")
            or f"item_{item_id_int}"
        )
        txt = (
            entry.get("summarization")
            or entry.get("profile")
            or entry.get("description")
            or ""
        )
    else:
        title = f"item_{item_id_int}"
        txt = str(entry) if entry is not None else ""

    if not txt:
        txt = "(no profile text available)"

    return {"item_id_raw": title, "profile": txt}


def get_user_full_profile(usr_prf_dict, user_id_int):
    """
    Get full user profile from usr_prf.pkl.

    Expected shape:
      { user_id_int: { 'profile': ..., 'reasoning': ... }, ... }
    but we handle plain strings as well.
    """
    entry = usr_prf_dict.get(user_id_int)
    if entry is None:
        return {"profile": "N/A", "reasoning": "N/A"}

    if isinstance(entry, dict):
        return {
            "profile": entry.get("profile", "N/A"),
            "reasoning": entry.get("reasoning", "N/A"),
        }

    # Fallback: treat as string
    return {"profile": str(entry), "reasoning": "N/A"}


def print_user_comparison(
    user_row,
    usr_prf_dict,
    itm_prf_dict,
    old_model_name="RLMRec-Plus",
    new_model_name="SegRLMRec-Plus",
    top_display=5,
    hist_K=5,
):
    """
    Print qualitative comparison for a single user:
      - metrics for old vs new
      - history (earliest K + latest K)
      - old vs new top-K recs
      - ground-truth test items
    """
    user_id = int(user_row["user_id_int"])

    # train interactions - should match for old/new
    n_train_old = int(user_row["n_train_interactions_old"])
    n_train_new = int(user_row["n_train_interactions_new"])
    n_train = n_train_old
    if n_train_old != n_train_new:
        print(
            f"[WARN] n_train_interactions mismatch for user {user_id}: "
            f"{n_train_old} vs {n_train_new}"
        )

    # Metrics
    r5_old = float(user_row["recall@5_old"])
    r10_old = float(user_row["recall@10_old"])
    r20_old = float(user_row["recall@20_old"])
    nd5_old = float(user_row["ndcg@5_old"])
    nd10_old = float(user_row["ndcg@10_old"])
    nd20_old = float(user_row["ndcg@20_old"])

    r5_new = float(user_row["recall@5_new"])
    r10_new = float(user_row["recall@10_new"])
    r20_new = float(user_row["recall@20_new"])
    nd5_new = float(user_row["ndcg@5_new"])
    nd10_new = float(user_row["ndcg@10_new"])
    nd20_new = float(user_row["ndcg@20_new"])

    delta_nd10 = nd10_new - nd10_old

    print("=" * 120)
    print(f"USER ID (int): {user_id}")
    print(f"Number of train interactions: {n_train}")
    print("-" * 120)
    print(
        f"{old_model_name}: "
        f"R@5={r5_old:.4f}, R@10={r10_old:.4f}, R@20={r20_old:.4f}, "
        f"N@5={nd5_old:.4f}, N@10={nd10_old:.4f}, N@20={nd20_old:.4f}"
    )
    print(
        f"{new_model_name}: "
        f"R@5={r5_new:.4f}, R@10={r10_new:.4f}, R@20={r20_new:.4f}, "
        f"N@5={nd5_new:.4f}, N@10={nd10_new:.4f}, N@20={nd20_new:.4f}"
    )
    print(f"ΔNDCG@10 (new - old): {delta_nd10:+.4f}")
    print("=" * 120)

    # User profile / reasoning
    user_full_profile = get_user_full_profile(usr_prf_dict, user_id)
    print(f"\n{'USER PROFILE:':<25}")
    print(user_full_profile.get("profile", "N/A"))
    print(f"\n{'USER REASONING:':<25}")
    print(user_full_profile.get("reasoning", "N/A"))

    # History and test
    train_items = parse_id_list(user_row["train_item_ids_old"])
    test_items = parse_id_list(user_row["test_item_ids_old"])

    # Predicted items & scores
    pred_items_old = parse_id_list(user_row["topk_item_ids_old"])
    pred_scores_old = (
        [float(x) for x in str(user_row["topk_scores_old"]).split("|")]
        if not pd.isna(user_row["topk_scores_old"])
        else []
    )

    pred_items_new = parse_id_list(user_row["topk_item_ids_new"])
    pred_scores_new = (
        [float(x) for x in str(user_row["topk_scores_new"]).split("|")]
        if not pd.isna(user_row["topk_scores_new"])
        else []
    )

    # -------- History --------
    print(f"\n{'-' * 120}")
    print(f"HISTORICAL INTERACTIONS (TRAIN): {len(train_items)} items")
    print(f"{'-' * 120}")

    if len(train_items) <= 2 * hist_K:
        # show all
        for idx, item_id in enumerate(train_items, 1):
            item_info = get_item_profile(itm_prf_dict, item_id)
            print(f"\n[{idx}] Item ID: {item_id} (Raw: {item_info['item_id_raw']})")
            print(f"    {item_info['profile']}")
    else:
        # earliest K
        print(f"\nEarliest {hist_K} train items:")
        first_part = train_items[:hist_K]
        for offset, item_id in enumerate(first_part, 1):
            item_info = get_item_profile(itm_prf_dict, item_id)
            print(f"\n[{offset}] Item ID: {item_id} (Raw: {item_info['item_id_raw']})")
            print(f"    {item_info['profile']}")

        # latest K
        print(f"\nLatest {hist_K} train items:")
        last_part = train_items[-hist_K:]
        start_idx = len(train_items) - hist_K + 1
        for idx, item_id in zip(range(start_idx, len(train_items) + 1), last_part):
            item_info = get_item_profile(itm_prf_dict, item_id)
            print(f"\n[{idx}] Item ID: {item_id} (Raw: {item_info['item_id_raw']})")
            print(f"    {item_info['profile']}")

        print("\n... showing earliest and latest items only")

    # -------- OLD model recs --------
    print(f"\n{'-' * 120}")
    print(f"{old_model_name.upper()} PREDICTED ITEMS (TOP-{top_display})")
    print(f"{'-' * 120}")
    for i, (item_id, score) in enumerate(
        zip(pred_items_old[:top_display], pred_scores_old[:top_display]), 1
    ):
        item_info = get_item_profile(itm_prf_dict, item_id)
        in_test = "✓ HIT" if item_id in test_items else "✗ MISS"
        print(
            f"\n[{i}] Item ID: {item_id} (Raw: {item_info['item_id_raw']}) "
            f"| Score: {score:.4f} | {in_test}"
        )
        print(f"    {item_info['profile']}")

    # -------- NEW model recs --------
    print(f"\n{'-' * 120}")
    print(f"{new_model_name.upper()} PREDICTED ITEMS (TOP-{top_display})")
    print(f"{'-' * 120}")
    for i, (item_id, score) in enumerate(
        zip(pred_items_new[:top_display], pred_scores_new[:top_display]), 1
    ):
        item_info = get_item_profile(itm_prf_dict, item_id)
        in_test = "✓ HIT" if item_id in test_items else "✗ MISS"
        print(
            f"\n[{i}] Item ID: {item_id} (Raw: {item_info['item_id_raw']}) "
            f"| Score: {score:.4f} | {in_test}"
        )
        print(f"    {item_info['profile']}")

    # -------- Ground truth --------
    print(f"\n{'-' * 120}")
    print(f"GROUND TRUTH (TEST ITEMS): {len(test_items)} items")
    print(f"{'-' * 120}")
    for i, item_id in enumerate(test_items, 1):
        item_info = get_item_profile(itm_prf_dict, item_id)
        in_old = "✓ OLD" if item_id in pred_items_old else "✗ OLD"
        in_new = "✓ NEW" if item_id in pred_items_new else "✗ NEW"
        print(
            f"\n[{i}] Item ID: {item_id} (Raw: {item_info['item_id_raw']}) "
            f"| {in_old} | {in_new}"
        )
        print(f"    {item_info['profile']}")

    print("\n" + "=" * 120 + "\n")


# ----------------- Main ----------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Compare two rec models (old vs new) and dump top per-user examples."
    )
    ap.add_argument(
        "--dataset",
        default="amazon",
        help="Dataset label (only used for naming)."
    )
    ap.add_argument(
        "--old_csv",
        type=str,
        required=True,
        help="Path to old model recs CSV (e.g., recs_lightgcn_plus_amazon.csv)."
    )
    ap.add_argument(
        "--new_csv",
        type=str,
        required=True,
        help="Path to new model recs CSV (e.g., recs_lightgcn_plus_fusion_amazon.csv)."
    )
    ap.add_argument(
        "--old_model_name",
        type=str,
        default="RLMRec-Plus",
        help="Pretty name for old model in prints."
    )
    ap.add_argument(
        "--new_model_name",
        type=str,
        default="SegRLMRec-Plus",
        help="Pretty name for new model in prints."
    )
    ap.add_argument(
        "--usr_prf_pkl",
        type=str,
        required=True,
        help="Path to usr_prf.pkl (user profiles)."
    )
    ap.add_argument(
        "--itm_prf_pkl",
        type=str,
        required=True,
        help="Path to itm_prf.pkl (item profiles)."
    )
    ap.add_argument(
        "--num_examples",
        type=int,
        default=3,
        help="Number of examples in each direction (new>>old and old>>new)."
    )
    ap.add_argument(
        "--top_display",
        type=int,
        default=5,
        help="Number of top recommendations to display per model."
    )
    ap.add_argument(
        "--hist_K",
        type=int,
        default=5,
        help="How many earliest/latest history items to show."
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional path for output .txt. "
            "Default: user_comparison_<old>_vs_<new>_<dataset>.txt"
        ),
    )

    args = ap.parse_args()

    old_csv_path = Path(args.old_csv)
    new_csv_path = Path(args.new_csv)
    usr_prf_pkl_path = Path(args.usr_prf_pkl)
    itm_prf_pkl_path = Path(args.itm_prf_pkl)

    # Load rec CSVs
    df_old = pd.read_csv(old_csv_path)
    df_new = pd.read_csv(new_csv_path)

    # Merge on user_id_int
    df_merge = df_old.merge(df_new, on="user_id_int", suffixes=("_old", "_new"))

    # Check for expected columns
    required_cols = [
        "ndcg@10_old", "ndcg@10_new",
        "n_train_interactions_old", "n_train_interactions_new",
        "train_item_ids_old", "test_item_ids_old",
        "topk_item_ids_old", "topk_scores_old",
        "topk_item_ids_new", "topk_scores_new",
        "recall@5_old", "recall@10_old", "recall@20_old",
        "recall@5_new", "recall@10_new", "recall@20_new",
        "ndcg@5_old", "ndcg@20_old",
        "ndcg@5_new", "ndcg@20_new",
    ]
    for c in required_cols:
        if c not in df_merge.columns:
            raise ValueError(
                f"Expected column '{c}' not found after merge. "
                f"Check your CSV headers and suffixes."
            )

    # ---------------- FILTER: only users with > 10 train interactions ----------------
    df_merge = df_merge[
        (df_merge["n_train_interactions_old"] > 10)
        & (df_merge["n_train_interactions_new"] > 10)
    ].copy()

    if df_merge.empty:
        raise ValueError(
            "After filtering for users with > 10 train interactions, "
            "no users remain. Check your data or relax the threshold."
        )
    # -------------------------------------------------------------------------------

    # ΔNDCG@10 = new - old
    df_merge["delta_ndcg10"] = df_merge["ndcg@10_new"] - df_merge["ndcg@10_old"]

    # Load profiles
    usr_prf_dict = load_pkl(usr_prf_pkl_path)
    itm_prf_dict = load_pkl(itm_prf_pkl_path)

    # Filter to users where there is at least some difference (optional)
    df_nonzero = df_merge[df_merge["delta_ndcg10"] != 0].copy()
    if df_nonzero.empty:
        df_nonzero = df_merge.copy()

    # 1) New >> Old
    df_new_better = df_nonzero.sort_values("delta_ndcg10", ascending=False).head(
        args.num_examples
    )

    # 2) Old >> New
    df_old_better = df_nonzero.sort_values("delta_ndcg10", ascending=True).head(
        args.num_examples
    )

    buf = StringIO()
    with redirect_stdout(buf):
        print("\n" + "#" * 120)
        print(
            f"TOP {args.num_examples} USERS WHERE NEW MODEL >> OLD MODEL (by ΔNDCG@10)"
        )
        print("#" * 120 + "\n")

        for idx, (_, row) in enumerate(df_new_better.iterrows(), 1):
            delta = float(row["delta_ndcg10"])
            print(
                f"\n>>> [NEW BETTER] Example {idx}/{len(df_new_better)} "
                f"(ΔNDCG@10 = {delta:+.4f}) <<<\n"
            )
            print_user_comparison(
                row,
                usr_prf_dict,
                itm_prf_dict,
                old_model_name=args.old_model_name,
                new_model_name=args.new_model_name,
                top_display=args.top_display,
                hist_K=args.hist_K,
            )

        print("\n" + "#" * 120)
        print(
            f"TOP {args.num_examples} USERS WHERE OLD MODEL >> NEW MODEL (by ΔNDCG@10)"
        )
        print("#" * 120 + "\n")

        for idx, (_, row) in enumerate(df_old_better.iterrows(), 1):
            delta = float(row["delta_ndcg10"])
            print(
                f"\n>>> [OLD BETTER] Example {idx}/{len(df_old_better)} "
                f"(ΔNDCG@10 = {delta:+.4f}) <<<\n"
            )
            print_user_comparison(
                row,
                usr_prf_dict,
                itm_prf_dict,
                old_model_name=args.old_model_name,
                new_model_name=args.new_model_name,
                top_display=args.top_display,
                hist_K=args.hist_K,
            )

    output_text = buf.getvalue()

    # Output path
    if args.output is None:
        out_path = Path(
            f"user_comparison_{args.old_model_name}_vs_"
            f"{args.new_model_name}_{args.dataset}.txt"
        )
    else:
        out_path = Path(args.output)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            f"User comparison: {args.old_model_name} vs {args.new_model_name} "
            f"on {args.dataset}\n"
        )
        f.write("=" * 120 + "\n\n")
        f.write(output_text)

    print("\n" + "=" * 120)
    print(f"User comparison examples saved to: {out_path}")
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
