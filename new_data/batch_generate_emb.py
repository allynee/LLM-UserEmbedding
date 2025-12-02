import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeddings_batched(texts, tokenizer, model, device):
    # Tokenize
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    
    return sentence_embeddings.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_prf", type=str, required=True, help="Path to usr_prf.pkl")
    parser.add_argument("--output_emb", type=str, required=True, help="Path to usr_emb_np.pkl")
    parser.add_argument("--test", action="store_true", help="Run test for the first user")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    args = parser.parse_args()

    print("Loading profiles...")
    profiles = load_pickle(args.input_prf)
    
    if not profiles:
        print("No profiles found.")
        exit()

    uids = sorted(list(profiles.keys()))
    
    # Prepare data
    data_list = []
    for uid in uids:
        data = profiles[uid]
        text = data.get('summarization', data.get('profile', ''))
        if text:
            data_list.append((uid, text))
            
    if not data_list:
        print("No valid profiles with text found.")
        exit()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model_name = "google/embeddinggemma-300m" 
    print(f"Loading model {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        model.eval()
        
        # Set pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        exit()

    if args.test:
        print("Running in TEST mode for the first user...")
        uid, text = data_list[0]
        print(f"Processing user {uid}...")
        print(f"Text: {text[:100]}...")
        
        emb = get_embeddings_batched([text], tokenizer, model, device)
        
        print(f"Embedding shape: {emb.shape}")
        print(f"Embedding sample: {emb[0][:5]}")
        exit()

    # Batch Processing
    print(f"Generating embeddings for {len(data_list)} users...")
    
    all_texts = [x[1] for x in data_list]
    all_uids = [x[0] for x in data_list]
    
    embeddings = []
    batch_size = args.batch_size
    
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Encoding"):
        batch_texts = all_texts[i:i+batch_size]
        batch_embs = get_embeddings_batched(batch_texts, tokenizer, model, device)
        embeddings.append(batch_embs)
        
    embeddings = np.concatenate(embeddings, axis=0)
    
    # Create final matrix
    max_uid = max(all_uids)
    num_users = max_uid + 1
    emb_dim = embeddings.shape[1]
    
    print(f"Creating embedding matrix of shape ({num_users}, {emb_dim})...")
    emb_matrix = np.zeros((num_users, emb_dim))
    
    for i, uid in enumerate(all_uids):
        emb_matrix[uid] = embeddings[i]

    print(f"Saving embeddings to {args.output_emb}...")
    with open(args.output_emb, 'wb') as f:
        pickle.dump(emb_matrix, f)
    print("Done.")
