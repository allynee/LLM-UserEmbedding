import argparse
import json
from collections import Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--windows_jsonl",
        type=str,
        required=True,
        help="Path to user_windows.jsonl",
    )
    args = parser.parse_args()

    counts = []  # number of windows per user

    print(f"Loading {args.windows_jsonl} ...")
    with open(args.windows_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            win_list = obj.get("windows", [])
            counts.append(len(win_list))

    if not counts:
        print("No users found.")
        return

    total_users = len(counts)
    total_windows = sum(counts)
    avg_windows = total_windows / total_users
    min_windows = min(counts)
    max_windows = max(counts)

    dist = Counter(counts)

    print("\n=== User-window statistics ===")
    print(f"Total users:          {total_users}")
    print(f"Total windows:        {total_windows}")
    print(f"Avg windows / user:   {avg_windows:.3f}")
    print(f"Min windows / user:   {min_windows}")
    print(f"Max windows / user:   {max_windows}")

    print("\nDistribution (windows_per_user -> num_users):")
    # Print small k explicitly, then bucket the rest
    max_k_to_print = 10
    for k in range(1, max_k_to_print + 1):
        print(f"  {k}: {dist.get(k, 0)}")

    higher = [(k, v) for k, v in dist.items() if k > max_k_to_print]
    if higher:
        print(f"\nUsers with > {max_k_to_print} windows:")
        for k, v in sorted(higher):
            print(f"  {k}: {v}")

    print("================================\n")

if __name__ == "__main__":
    main()
