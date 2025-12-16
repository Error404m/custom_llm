"""
generate.py

Loads the best-performing model checkpoint and runs text generation.
"""
import torch
import tiktoken
import argparse
from config import GPT_CONFIG, CHECKPOINT_PATH, DEVICE
from model import GPT

@torch.no_grad()
def generate_text(prompt, max_new_tokens, num_samples, temperature, top_k):
    # 1. Load Tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # 2. Load Model
    model = GPT(GPT_CONFIG).to(DEVICE)
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Model checkpoint not found at {CHECKPOINT_PATH}. Please run 'python train.py' first.")
        return
        
    print(f"Loading model checkpoint from {CHECKPOINT_PATH}...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=torch.device(DEVICE)))
    model.eval()

    # 3. Tokenize the prompt
    start_ids = enc.encode_ordinary(prompt)
    x = (torch.tensor(start_ids, dtype=torch.long, device=DEVICE)[None, ...])

    # 4. Generate Samples
    print(f"\n--- Generating {num_samples} Samples ---")
    for i in range(num_samples):
        print(f"\n--- Sample {i+1} ---")
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        output_tokens = y[0].tolist()
        output_text = enc.decode(output_tokens)
        print(output_text)
    
    print("\nGeneration complete.")

if __name__ == '__main__':
    # Setup command line arguments for a professional feel
    parser = argparse.ArgumentParser(description="Generate text using MrityunjayaGPT.")
    parser.add_argument("--prompt", type=str, default="Once upon a time there was a cat.", 
                        help="The starting prompt for text generation.")
    parser.add_argument("--max_tokens", type=int, default=100, 
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--num_samples", type=int, default=3, 
                        help="Number of independent samples to generate.")
    parser.add_argument("--temp", type=float, default=0.8, 
                        help="Sampling temperature (lower is less random).")
    parser.add_argument("--top_k", type=int, default=50, 
                        help="Top-K sampling limit.")
    
    args = parser.parse_args()

    generate_text(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        num_samples=args.num_samples,
        temperature=args.temp,
        top_k=args.top_k
    )
