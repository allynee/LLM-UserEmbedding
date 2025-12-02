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
                # steam_user.json: {"uid": 0, "username": "..."}
                # steam_item.json: {"iid": 0, "product_id": "..."}
                if 'uid' in obj:
                    mapping[obj['uid']] = obj['username']
                elif 'iid' in obj:
                    mapping[obj['iid']] = obj['product_id']
            except:
                pass
    return mapping

def load_games_info(path):
    print(f"Loading games info from {path}...")
    games = {}
    with open(path, 'r') as f:
        for line in f:
            try:
                game = ast.literal_eval(line)
                # Map id -> game info
                if 'id' in game:
                    games[game['id']] = game
            except:
                pass
    return games

def load_reviews_filtered(path, valid_usernames):
    print(f"Loading reviews from {path} (filtering for {len(valid_usernames)} users)...")
    reviews = {} # username -> {product_id -> text}
    
    # Get total lines for tqdm if possible (optional, but nice)
    # Using wc -l might be slow for huge files, but let's try to just iterate with tqdm without total first
    # or just use a large number estimate if we want a bar, but simple tqdm(f) works if we don't need percentage
    
    with open(path, 'r') as f:
        for line in tqdm(f, desc="Loading reviews"):
            try:
                review = ast.literal_eval(line)
                username = review.get('username')
                if username in valid_usernames:
                    pid = review.get('product_id')
                    text = review.get('text')
                    if username not in reviews:
                        reviews[username] = {}
                    reviews[username][pid] = text
            except:
                pass
    return reviews

def generate_prompt_for_user(user_idx, trn_mat, uid_to_username, iid_to_productid, games_info, user_reviews, system_prompt_template):
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
        # Description: 'desc_snippet' or 'detailed_description' or 'short_description'?
        # steam_games.json usually has these. Let's check keys later if needed. 
        # For now assume 'title' is key. I'll check keys in game dict.
        # Based on inspect output: 'publisher', 'genres', 'app_name', 'title', 'tags', 'specs', 'price', 'early_access', 'id', 'developer', 'sentiment'
        # It doesn't show description in the head output. Maybe it's missing or named differently?
        # I'll use 'tags' and 'genres' as description if description is missing, or just "None".
        # Wait, the instruction says: "description": "a description of what types of users will like this game"
        # I can construct this from genres and tags.
        
        description = "None"
        if 'genres' in game or 'tags' in game:
            genres = game.get('genres', [])
            tags = game.get('tags', [])
            description = f"Genres: {', '.join(genres)}. Tags: {', '.join(tags)}."
        
        review = "None"
        if username in user_reviews and pid in user_reviews[username]:
            review = user_reviews[username][pid]
            
        played_games.append({
            "title": title,
            "description": description,
            "review": review
        })
    
    # Construct the final prompt
    # The system prompt contains the instructions.
    # The user message contains "PLAYED GAMES: ..."
    
    user_content = f"PLAYED GAMES: {json.dumps(played_games)}"
    return user_content

def process_user(user_idx, trn_mat, uid_to_username, iid_to_productid, games_info, user_reviews, system_prompt):
    user_content = generate_prompt_for_user(user_idx, trn_mat, uid_to_username, iid_to_productid, games_info, user_reviews, system_prompt)
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
    args = parser.parse_args()

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    mapper_dir = os.path.join(data_dir, 'mapper')
    instruction_path = os.path.join(base_dir, 'generation', 'instruction', 'steam_user.txt')

    # Load Data
    print("Loading data...")
    trn_mat = load_pickle(args.input_mat)
    uid_to_username = load_jsonl_mapper(os.path.join(mapper_dir, 'steam_user.json'))
    iid_to_productid = load_jsonl_mapper(os.path.join(mapper_dir, 'steam_item.json'))
    games_info = load_games_info(os.path.join(raw_dir, 'steam_games.json'))
    
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
            
    user_reviews = load_reviews_filtered(os.path.join(raw_dir, 'steam_reviews.json'), valid_usernames)

    # Load System Prompt
    with open(instruction_path, 'r') as f:
        system_prompt = f.read()

    # Processing
    results = {}

    if args.test:
        print("Running in TEST mode for the first user...")
        uid = 0
        print(f"Processing user {uid}...")
        res = process_user(uid, trn_mat, uid_to_username, iid_to_productid, games_info, user_reviews, system_prompt)
        print("Output:")
        print(json.dumps(res, indent=2))
        exit()

    print(f"Starting generation for {trn_mat.shape[0]} users...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_uid = {
            executor.submit(process_user, uid, trn_mat, uid_to_username, iid_to_productid, games_info, user_reviews, system_prompt): uid 
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
