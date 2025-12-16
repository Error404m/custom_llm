"""
data/prepare.py

Handles downloading the TinyStories dataset, tokenizing it using GPT-2 BPE, 
and saving the results as memory-mapped binary files (train.bin, val.bin).
"""
import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm.auto import tqdm
from config import DATA_DIR, BLOCK_SIZE

def process():
    # Create the data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Load Dataset
    print("Loading TinyStories dataset from HuggingFace...")
    raw_datasets = load_dataset("roneneldan/TinyStories")

    # 2. Initialize Tokenizer (GPT-2 BPE)
    enc = tiktoken.get_encoding("gpt2")
    
    # 3. Process splits
    for split, raw_data in raw_datasets.items():
        print(f"Processing '{split}' split...")
        
        # Concatenate all text into a single long string
        text_data = "\n".join(raw_data['text'])
        
        # Tokenize the entire text
        tokens = enc.encode_ordinary(text_data)
        
        # Convert to numpy array of 16-bit integers
        token_ids = np.array(tokens, dtype=np.uint16)
        
        # Save to binary file
        output_file = os.path.join(DATA_DIR, f"{split}.bin")
        token_ids.tofile(output_file)
        
        print(f"Saved {len(token_ids):,} tokens to {output_file}")

    print("\nData preparation complete.")
    print("Run 'python train.py' to begin training MrityunjayaGPT.")

if __name__ == '__main__':
    process()
