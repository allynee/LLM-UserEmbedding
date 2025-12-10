import argparse
import ast
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import litellm
import numpy as np
from tqdm import tqdm

os.environ["OPENAI_API_BASE"] = "https://ai-gateway.andrew.cmu.edu/"
os.environ["OPENAI_API_KEY"] = "sk-_-T7rwqh0roeL1KXxpsafQ"


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_jsonl_mapper(path):
    """
    For mapper files like amazon_item.json / amazon_user.json that are JSONL:
    - if 'uid' present, map uid -> reviewerID
    - if 'iid' present, map iid -> asin
    """
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "uid" in obj:
                val = obj.get("username", obj.get("reviewerID"))
                if val is not None:
                    mapping[int(obj["uid"])] = val
            elif "iid" in obj:
                val = obj.get("product_id", obj.get("asin"))
                if val is not None:
                    mapping[int(obj["iid"])] = val
    return mapping


def load_asin2title(path):
    """
    Load precomputed asin -> title mapping (JSON).
    This is the output from your meta_Books -> asin2title script.
    """
    if path is None or not os.path.exists(path):
        print(f"[WARN] asin2title JSON not found at {path}. Titles will be 'None'.")
        return {}

    print(f"Loading asin->title mapping from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        asin2title = json.load(f)
    print(f"Loaded {len(asin2title)} asin->title entries.")
    return asin2title


def load_user_windows(path):
    """
    Reads user_windows.jsonl:
      {"user_int": int, "windows": [ {window_idx, is_short_term, interactions: [...]}, ... ]}
    Returns dict: user_int -> list[window_obj]
    """
    windows_by_user = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            u = int(obj["user_int"])
            windows_by_user[u] = obj["windows"]
    print(f"Loaded windows for {len(windows_by_user)} users from {path}")
    return windows_by_user


def build_purchased_items_for_window(
    window,
    iid_to_asin,
    asin2title,
    item_profiles,
    default_title="None",
):
    """
    window: dict with 'interactions': list[{item_int, timestamp, reviewText}]
    Returns list of dicts:
      {
        "title": str,
        "description": str,
        "review": str
      }
    Exactly matches the format described in amazon_user.txt.
    """
    purchased_items = []
    for inter in window.get("interactions", []):
        iid = int(inter["item_int"])
        review = inter.get("reviewText") or "None"

        asin = iid_to_asin.get(iid)
        title = asin2title.get(asin, default_title) if asin is not None else default_title

        desc = "None"
        if item_profiles is not None:
            prof = item_profiles.get(iid)
            if isinstance(prof, dict):
                desc = (
                    prof.get("summarization")
                    or prof.get("profile")
                    or prof.get("description")
                    or "None"
                )
            else:
                # If item_profiles is something else (e.g., raw text)
                desc = str(prof)

        purchased_items.append(
            {
                "title": title if title is not None else "None",
                "description": desc,
                "review": review,
            }
        )

    return purchased_items


def call_llm(system_prompt, purchased_items, max_retries=3, sleep_sec=2.0):
    """
    Send one window to the LLM using litellm, return parsed JSON (or fallback dict).
    This matches the RLMRec prompt convention:
      - system: instruction text (amazon_user.txt)
      - user: "PURCHASED ITEMS: [ {...}, {...}, ... ]"
    """
    user_content = "PURCHASED ITEMS: " + json.dumps(purchased_items, ensure_ascii=False)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(max_retries):
        try:
            resp = litellm.completion(
                model="gpt-4o-mini-2024-07-18",
                messages=messages,
                api_base="https://ai-gateway.andrew.cmu.edu/",
                api_key="sk-_-T7rwqh0roeL1KXxpsafQ",
                temperature=0,
            )
            content = resp.choices[0].message.content

            # Try to extract JSON block
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end != -1:
                json_str = content[start:end]
                try:
                    data = json.loads(json_str)
                    return data
                except json.JSONDecodeError:
                    # Fall through to fallback below
                    pass

            # Fallback if JSON parsing fails
            return {
                "summarization": content,
                "reasoning": "Failed to parse strict JSON; storing raw content.",
            }

        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "summarization": "None",
                    "reasoning": f"LLM error after {max_retries} attempts: {e}",
                }
            time.sleep(sleep_sec)


def process_one_window(
    user_int,
    window,
    iid_to_asin,
    asin2title,
    item_profiles,
    system_prompt,
):
    purchased_items = build_purchased_items_for_window(
        window, iid_to_asin, asin2title, item_profiles
    )
    if not purchased_items:
        return None

    llm_out = call_llm(system_prompt, purchased_items)
    if llm_out is None:
        return None

    return {
        "window_idx": window["window_idx"],
        "is_short_term": bool(window.get("is_short_term", False)),
        "summarization": llm_out.get("summarization", "None"),
        "reasoning": llm_out.get("reasoning", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--windows_jsonl",
        type=str,
        required=True,
        help="Path to user_windows.jsonl",
    )
    parser.add_argument(
        "--item_prf",
        type=str,
        required=True,
        help="Path to itm_prf.pkl (item profiles, used for 'description')",
    )
    parser.add_argument(
        "--item_map",
        type=str,
        required=True,
        help="Path to amazon_item.json (JSONL mapper: iid -> asin)",
    )
    parser.add_argument(
        "--asin2title_json",
        type=str,
        required=True,
        help="Path to asin2title.json (asin -> title mapping)",
    )
    parser.add_argument(
        "--instruction_file",
        type=str,
        required=True,
        help="Path to amazon_user.txt system prompt (RLMRec style)",
    )
    parser.add_argument(
        "--output_pkl",
        type=str,
        required=True,
        help="Where to save user-window profiles (pickle)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Number of parallel threads for LLM calls",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="If set, only process first user & first window and print result.",
    )
    args = parser.parse_args()

    # Make sure API key is in env (do NOT hardcode)
    if "OPENAI_API_KEY" not in os.environ:
        print(
            "[WARN] OPENAI_API_KEY not set in environment. "
            "Set it via `export OPENAI_API_KEY=...` before running."
        )

    # Read system prompt (this *is* the original RLMRec prompt for Amazon)
    with open(args.instruction_file, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    # Load data
    print("Loading user_windows...")
    windows_by_user = load_user_windows(args.windows_jsonl)

    print("Loading item profiles from", args.item_prf)
    item_profiles = load_pickle(args.item_prf)

    print("Loading item mapper from", args.item_map)
    iid_to_asin = load_jsonl_mapper(args.item_map)

    asin2title = load_asin2title(args.asin2title_json)

        # TEST MODE: pick a user with at least 3 windows and show + save their window profiles
    if args.test:
        # Find users with >= 3 windows
        candidate_users = [
            u for u, win_list in windows_by_user.items()
            if len(win_list) >= 3
        ]

        if not candidate_users:
            print("No users with >= 3 windows found. Falling back to the first user.")
            user_ints = sorted(windows_by_user.keys())
            if not user_ints:
                print("No users found in windows file.")
                return
            u0 = user_ints[0]
        else:
            # Just take the first candidate (or you could random.choice)
            u0 = sorted(candidate_users)[0]

        win_list = windows_by_user[u0]
        print(f"[TEST] User {u0} has {len(win_list)} windows.")

        # If you only want to inspect first 3, uncomment this:
        # win_list = win_list[:3]

        results = {u0: []}

        for w in win_list:
            print(
                f"\n----- Window idx {w['window_idx']} "
                f"(is_short_term={w.get('is_short_term', False)}) -----"
            )
            res = process_one_window(
                u0,
                w,
                iid_to_asin,
                asin2title,
                item_profiles,
                system_prompt,
            )
            print(json.dumps(res, indent=2, ensure_ascii=False))

            if res is not None:
                results[u0].append(res)

        print(
            f"\n[TEST] Finished test user; collected "
            f"{len(results[u0])} window-level profiles."
        )

        # Save test results using the same structure as full batch gen
        print(f"[TEST] Saving test results to {args.output_pkl} ...")
        with open(args.output_pkl, "wb") as f:
            pickle.dump(results, f)
        print("[TEST] Done.")
        return

    # Full run
    results = {}  # user_int -> list of window summaries

    # Flatten tasks: one future per (user, window)
    tasks = []
    for u, win_list in windows_by_user.items():
        for w in win_list:
            tasks.append((u, w))

    print(f"Starting generation for {len(tasks)} user-window pairs...")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_key = {
            executor.submit(
                process_one_window,
                u,
                w,
                iid_to_asin,
                asin2title,
                item_profiles,
                system_prompt,
            ): (u, w["window_idx"])
            for (u, w) in tasks
        }

        for fut in tqdm(as_completed(future_to_key), total=len(future_to_key)):
            u, widx = future_to_key[fut]
            try:
                out = fut.result()
            except Exception as e:
                print(f"[ERROR] Exception for user {u}, window {widx}: {e}")
                out = None

            if out is None:
                continue

            if u not in results:
                results[u] = []
            results[u].append(out)

    print(f"Saving window-level profiles for {len(results)} users to {args.output_pkl} ...")
    with open(args.output_pkl, "wb") as f:
        pickle.dump(results, f)
    print("Done.")


if __name__ == "__main__":
    main()
