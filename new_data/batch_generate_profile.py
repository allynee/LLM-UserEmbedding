import argparse
import pickle
import json
import ast
import numpy as np
import scipy.sparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import litellm
import time
import os
import threading
from tqdm import tqdm

# Configure litellm
# litellm.api_base = "https://ai-gateway.andrew.cmu.edu/" # This might need to be passed in completion or set via env
os.environ["OPENAI_API_BASE"] = "https://ai-gateway.andrew.cmu.edu/"
os.environ["OPENAI_API_KEY"] = "sk-erWkuPo4ryQ0b5pjXu1WwQ" # litellm might require a non-empty key even if not used

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_jsonl_mapper(path):
    # Mappers like steam_item.json seem to be JSONL
    mapping = {}
    with open(path, 'r') as f:
        for line in f:
            try:
                obj = json.loads(line)
                # Assuming uid/iid -> id mapping
                if 'uid' in obj:
                    # Try 'username' (Steam) or 'reviewerID' (Amazon)
                    val = obj.get('username', obj.get('reviewerID'))
                    if val:
                        mapping[obj['uid']] = val
                elif 'iid' in obj:
                    # Try 'product_id' (Steam) or 'asin' (Amazon)
                    val = obj.get('product_id', obj.get('asin'))
                    if val:
                        mapping[obj['iid']] = val
            except:
                pass
    return mapping

def load_games_info(path, dataset='steam'):
    print(f"Loading item info from {path}...")
    items = {}
    with open(path, 'r') as f:
        for line in f:
            try:
                if dataset == 'steam':
                    item = ast.literal_eval(line)
                    if 'id' in item:
                        items[item['id']] = item
                elif dataset == 'amazon':
                    item = json.loads(line)
                    if 'asin' in item:
                        items[item['asin']] = item
            except:
                pass
    return items

def load_reviews_filtered(path, valid_usernames, dataset='steam'):
    print(f"Loading reviews from {path} (filtering for {len(valid_usernames)} users)...")
    reviews = {} # username -> {product_id -> text}
    
    with open(path, 'r') as f:
        for line in tqdm(f, desc="Loading reviews"):
            try:
                if dataset == 'steam':
                    review = ast.literal_eval(line)
                    username = review.get('username')
                    pid = review.get('product_id')
                    text = review.get('text')
                elif dataset == 'amazon':
                    review = json.loads(line)
                    username = review.get('reviewerID')
                    pid = review.get('asin')
                    text = review.get('reviewText')

                if username in valid_usernames:
                    if username not in reviews:
                        reviews[username] = {}
                    reviews[username][pid] = text
            except:
                pass
    return reviews

def generate_prompt_for_user(user_idx, trn_mat, uid_to_username, iid_to_productid, games_info, item_profiles, user_reviews, system_prompt_template):
    # Get user interactions
    # trn_mat is COO or CSR. CSR is better for row slicing.
    if not scipy.sparse.isspmatrix_csr(trn_mat):
        trn_mat = trn_mat.tocsr()
    
    row = trn_mat.getrow(user_idx)
    item_indices = row.indices
    
    username = uid_to_username.get(user_idx)
    if not username:
        return None # Should not happen if mapper is complete

    played_games = []
    for iid in item_indices:
        pid = iid_to_productid.get(iid)
        if not pid:
            continue
        
        game = games_info.get(pid, {})
        title = game.get('title', game.get('app_name', 'None'))
        
        # Description from item profiles
        description = "None"
        if item_profiles and iid in item_profiles:
            prof = item_profiles[iid]
            description = prof.get('summarization', prof.get('profile', 'None'))
        
        review = "None"
        if username in user_reviews and pid in user_reviews[username]:
            review = user_reviews[username][pid]
            
        played_games.append({
            "title": title,
            "description": description,
            "review": review
        })
    
    # Construct the final prompt
    user_content = f"PLAYED GAMES: {json.dumps(played_games)}"
    return user_content

def process_user(user_idx, trn_mat, uid_to_username, iid_to_productid, games_info, item_profiles, user_reviews, system_prompt):
    user_content = generate_prompt_for_user(user_idx, trn_mat, uid_to_username, iid_to_productid, games_info, item_profiles, user_reviews, system_prompt)
    if not user_content:
        return None

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    retries = 3
    for attempt in range(retries):
        try:
            response = litellm.completion(
                model="gpt-4o-mini-2024-07-18",
                messages=messages,
                api_base="https://ai-gateway.andrew.cmu.edu/",
                api_key="sk-erWkuPo4ryQ0b5pjXu1WwQ",
                temperature=0,
            )
            content = response.choices[0].message.content
            # Try to parse JSON to ensure validity?
            # The instruction asks for JSON.
            try:
                # Find JSON substring if there is extra text
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = content[start:end]
                    data = json.loads(json_str)
                    return data
                else:
                    # If strict JSON parsing fails, just return the content or a dict with raw content
                    return {"summarization": content, "reasoning": "Failed to parse JSON"}
            except:
                 return {"summarization": content, "reasoning": "Failed to parse JSON"}

        except Exception as e:
            if attempt == retries - 1:
                print(f"Error processing user {user_idx}: {e}")
                return None
            time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mat", type=str, required=True, help="Path to trn_mat.pkl or trn_short_mat.pkl")
    parser.add_argument("--output_pkl", type=str, required=True, help="Path to output usr_prf.pkl")
    parser.add_argument("--test", action="store_true", help="Run test for the first user")
    parser.add_argument("--dataset", type=str, default="steam", choices=["steam", "amazon"], help="Dataset name")
    parser.add_argument("--raw_item_file", type=str, help="Path to raw item metadata file (for titles)")
    parser.add_argument("--raw_review_file", type=str, help="Path to raw review file")
    parser.add_argument("--item_prf", type=str, required=True, help="Path to itm_prf.pkl")
    args = parser.parse_args()

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust base_dir if script is in new_data
    if os.path.basename(base_dir) == 'new_data':
        base_dir = os.path.dirname(base_dir)
        
    data_dir = os.path.join(base_dir, 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    mapper_dir = os.path.join(data_dir, 'mapper')
    
    if args.dataset == 'steam':
        instruction_path = os.path.join(base_dir, 'generation', 'instruction', 'steam_user.txt')
        user_map_path = os.path.join(mapper_dir, 'steam_user.json')
        item_map_path = os.path.join(mapper_dir, 'steam_item.json')
        raw_item_path = args.raw_item_file if args.raw_item_file else os.path.join(raw_dir, 'steam_games.json')
        raw_review_path = args.raw_review_file if args.raw_review_file else os.path.join(raw_dir, 'steam_reviews.json')
    elif args.dataset == 'amazon':
        instruction_path = os.path.join(base_dir, 'generation', 'instruction', 'amazon_user.txt')
        user_map_path = os.path.join(mapper_dir, 'amazon_user.json')
        item_map_path = os.path.join(mapper_dir, 'amazon_item.json')
        raw_item_path = args.raw_item_file # Must be provided or we fail/warn
        raw_review_path = args.raw_review_file # Must be provided or we fail/warn

    # Load Data
    print("Loading data...")
    trn_mat = load_pickle(args.input_mat)
    uid_to_username = load_jsonl_mapper(user_map_path)
    iid_to_productid = load_jsonl_mapper(item_map_path)
    
    item_profiles = load_pickle(args.item_prf)
    
    games_info = {}
    if raw_item_path and os.path.exists(raw_item_path):
        games_info = load_games_info(raw_item_path, dataset=args.dataset)
    else:
        print(f"Warning: Raw item file {raw_item_path} not found. Titles will be 'None'.")
    
    # Filter users present in trn_mat
    if scipy.sparse.isspmatrix(trn_mat):
        valid_user_indices = range(trn_mat.shape[0])
    else:
        valid_user_indices = range(len(trn_mat)) # Assuming it might be something else? No, it's sparse matrix.
    
    valid_usernames = set()
    for idx in valid_user_indices:
        uname = uid_to_username.get(idx)
        if uname:
            valid_usernames.add(uname)
            
    user_reviews = {}
    if raw_review_path and os.path.exists(raw_review_path):
        user_reviews = load_reviews_filtered(raw_review_path, valid_usernames, dataset=args.dataset)
    else:
        print(f"Warning: Raw review file {raw_review_path} not found. Reviews will be 'None'.")

    # Load System Prompt
    with open(instruction_path, 'r') as f:
        system_prompt = f.read()

    # Processing
    results = {}

    if args.test:
        print("Running in TEST mode for the first user...")
        uid = 0
        print(f"Processing user {uid}...")
        res = process_user(uid, trn_mat, uid_to_username, iid_to_productid, games_info, item_profiles, user_reviews, system_prompt)
        print("Output:")
        print(json.dumps(res, indent=2))
        exit()

    print(f"Starting generation for {trn_mat.shape[0]} users...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_uid = {
            executor.submit(process_user, uid, trn_mat, uid_to_username, iid_to_productid, games_info, item_profiles, user_reviews, system_prompt): uid 
            for uid in range(trn_mat.shape[0])
        }
        
        for future in tqdm(as_completed(future_to_uid), total=len(future_to_uid)):
            uid = future_to_uid[future]
            try:
                res = future.result()
                if res:
                    results[uid] = res
            except Exception as e:
                print(f"Exception for user {uid}: {e}")

    # Save
    print(f"Saving results to {args.output_pkl}...")
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(results, f)
    print("Done.")
