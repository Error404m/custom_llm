"""
train.py

Main training script for MrityunjayaGPT.
Uses the configuration in config.py and the model definition in model.py.
Author: Mrityunjaya Tiwari
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from config import GPT_CONFIG, CHECKPOINT_PATH, DEVICE, MAX_ITERS, BATCH_SIZE, BLOCK_SIZE, \
                   LEARNING_RATE, GRADIENT_CLIP, GRADIENT_ACCUMULATION_STEPS, EVAL_INTERVAL, LOG_INTERVAL
from model import GPT

# --- Dataset Class ---
class TinyStoriesDataset(Dataset):
    def __init__(self, split):
        data_file = os.path.join(os.path.dirname(__file__), f"data/{split}.bin")
        if not os.path.exists(data_file):
             raise FileNotFoundError(f"Data file not found: {data_file}. Please run 'python data/prepare.py' first.")
        
        # Load the data as a memory-mapped numpy array (efficient)
        self.data = np.memmap(data_file, dtype=np.uint16, mode='r')

    def __len__(self):
        # The number of blocks we can extract from the data
        return len(self.data) // BLOCK_SIZE

    def __getitem__(self, idx):
        # Determine the start index for the block
        start_index = idx * BLOCK_SIZE
        
        # Get the input (x) and target (y) tokens
        # x: tokens 0 to BLOCK_SIZE-1
        # y: tokens 1 to BLOCK_SIZE (shifted by one for next-token prediction)
        x = torch.from_numpy(self.data[start_index : start_index + BLOCK_SIZE].astype(np.int64))
        y = torch.from_numpy(self.data[start_index + 1 : start_index + BLOCK_SIZE + 1].astype(np.int64))
        return x, y

# --- Helper Functions ---
@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters):
    """Evaluates the loss on a small batch of train/val data."""
    out = {}
    model.eval()
    
    for loader, split in zip([train_loader, val_loader], ['train', 'val']):
        losses = torch.zeros(eval_iters)
        for k, (X, Y) in enumerate(loader):
            if k >= eval_iters:
                break
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    
    model.train()
    return out

def get_lr(it):
    """Cosine learning rate decay scheduler."""
    # 1. Warmup (Linear increase)
    if it < 100:
        return LEARNING_RATE * it / 100
    # 2. Constant rate (The 0.1 is a small constant to prevent division by zero)
    if it > MAX_ITERS:
        return 0.1 * LEARNING_RATE
    # 3. Cosine decay
    decay_ratio = (it - 100) / (MAX_ITERS - 100)
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return LEARNING_RATE * coeff

# --- Main Training Function ---
def main():
    # 1. Setup DataLoaders
    train_dataset = TinyStoriesDataset('train')
    val_dataset = TinyStoriesDataset('validation')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Use iterable data loaders for efficiency with huge datasets
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    def get_batch(split):
        nonlocal train_iter, val_iter
        loader, iterator = (train_loader, train_iter) if split == 'train' else (val_loader, val_iter)
        try:
            X, Y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            X, Y = next(iterator)
        
        if split == 'train': train_iter = iterator
        else: val_iter = iterator
        
        return X.to(DEVICE), Y.to(DEVICE)

    # 2. Setup Model
    model = GPT(GPT_CONFIG).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-1)
    
    # Use Automatic Mixed Precision (AMP) for faster training
    scaler = torch.cuda.amp.GradScaler() if DEVICE == 'cuda' else None

    # 3. Training Loop
    best_val_loss = float('inf')
    
    print("\nStarting Training of MrityunjayaGPT...")
    for iter_num in tqdm(range(MAX_ITERS)):
        # Determine learning rate and set it
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluate loss occasionally
        if iter_num % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_loader, val_loader, eval_iters=20)
            print(f"Step {iter_num}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")
            
            # Save the best model checkpoint
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(f"--- Saved new best model with Val Loss: {best_val_loss:.4f} ---")

        # Get the batch
        X, Y = get_batch('train')

        # Forward pass with AMP
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE == 'cuda')):
            logits, loss = model(X, Y)
            loss = loss / GRADIENT_ACCUMULATION_STEPS # Normalize loss

        # Backward pass and optimization
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step only after accumulation steps
        if (iter_num + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            if scaler:
                scaler.unscale_(optimizer) # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)

        if iter_num % LOG_INTERVAL == 0:
             print(f"Step {iter_num}: Loss {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f} | LR {lr:.2e}")


if __name__ == '__main__':
    main()
