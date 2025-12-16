"""
model.py

Defines the core GPT (Generative Pre-trained Transformer) architecture.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in use in GPT-2 and BERT.
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            (torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3)))
        ))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer.
    """
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        
        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'], bias=config['bias'])
        # Output projection
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'], bias=config['bias'])
        # Regularization dropout
        self.attn_dropout = nn.Dropout(config['dropout'])
        self.resid_dropout = nn.Dropout(config['dropout'])
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        self.dropout = config['dropout']
        
        # Causal mask to prevent attending to future tokens
        self.register_buffer("bias", torch.tril(torch.ones(config['block_size'], config['block_size']))
                                     .view(1, 1, config['block_size'], config['block_size']))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimension (n_embd)

        # Calculate query, key, values for all heads in parallel
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Causal self-attention; (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))
        
        # Apply the causal mask (Tril)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Final weighted average
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """
    Standard Feed-Forward Network within a Transformer block.
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config['n_embd'], 4 * config['n_embd'], bias=config['bias'])
        self.gelu    = NewGELU()
        self.c_proj  = nn.Linear(4 * config['n_embd'], config['n_embd'], bias=config['bias'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """
    A single Transformer block (Attention + MLP). Uses Pre-Norm structure.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config['n_embd'])
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-Norm with Residual Connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    """
    The main Generative Pre-trained Transformer model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config['vocab_size'], config['n_embd']),
            wpe = nn.Embedding(config['block_size'], config['n_embd']),
            drop = nn.Dropout(config['dropout']),
            h = nn.ModuleList([Block(config) for _ in range(config['n_layer'])]),
            ln_f = nn.LayerNorm(config['n_embd']),
        ))
        
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        
        # Weight tying (a standard optimization)
        self.transformer.wte.weight = self.lm_head.weight 

        # Initialize weights
        self.apply(self._init_weights)
        print(f"MrityunjayaGPT initialized with {self.get_num_params()/1e6:.2f} Million parameters.")

    def get_num_params(self, non_embedding=True):
        """ Return the number of parameters in the model. """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # subtract position and token embeddings
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        
        assert T <= self.config['block_size'], f"Cannot forward sequence of length {T}, block size is only {self.config['block_size']}"
        
        # Token and positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings (B, T, n_embd)
        
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        # Final layer normalization
        x = self.transformer.ln_f(x)

        if targets is not None:
            # If we are training, compute the loss
            logits = self.lm_head(x)
            # Reshape (B, T, C) -> (B*T, C) for CrossEntropyLoss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            # If we are generating, only compute logits for the last time step
            logits = self.lm_head(x[:, [-1], :]) # (B, 1, vocab_size)
            loss = None
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Takes a sequence of indices (idx) and generates tokens one by one.
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block size
            idx_cond = idx if idx.size(1) <= self.config['block_size'] else idx[:, -self.config['block_size']:]
            
            # Forward pass: get the logits (predictions)
            logits, _ = self(idx_cond)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k sampling if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
