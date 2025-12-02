import argparse
import json
import ast
import os
from tqdm import tqdm

def parse_line(line):
    # Try JSON first
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        pass
    
    # Try ast.literal_eval (for Python dictionary string representation)
    try:
        return ast.literal_eval(line)
    except:
        return None

def process_file(input_path, output_path):
    print(f"Processing {input_path} -> {output_path}")
    count = 0
    errors = 0
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in tqdm(f_in):
            line = line.strip()
            if not line:
                continue
                
            obj = parse_line(line)
            
            if obj:
                # Write as valid JSONL
                f_out.write(json.dumps(obj) + '\n')
                count += 1
            else:
                errors += 1
                
    print(f"Finished. Processed {count} lines. Failed to parse {errors} lines.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw data files (JSON or Python literals) to clean JSONL.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to raw input file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSONL file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist.")
        exit(1)
        
    process_file(args.input_file, args.output_file)
