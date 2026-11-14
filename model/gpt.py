"""
MiniGPT-X Complete Language Model Architecture.

Configurable decoder-only Transformer with:
- RoPE or learned absolute positional embeddings
- RMSNorm or LayerNorm
- SwiGLU or ReLU MLP
- SDPA or manual causal attention
- Tied or untied input/output embeddings
- Weight decay grouping helper for 2D weight tensors
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embeddings import precompute_rope_freqs
from model.normalization import get_norm
from model.transformer import TransformerBlock


@dataclass
class GPTConfig:
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    vocab_size: int = 4096
    dropout: float = 0.0
    use_rope: bool = True
    norm_type: str = "rmsnorm"
    activation: str = "swiglu"
    use_sdpa: bool = True
    tie_embeddings: bool = True

    @classmethod
    def from_yaml(cls, path: str, overrides: Optional[Dict] = None) -> "GPTConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        m_cfg = raw.get("model", {})
        if overrides:
            m_cfg.update(overrides)
        return cls(**m_cfg)


class MiniGPTX(nn.Module):
    """
    MiniGPT-X Language Model.
    
    Tensor Shapes across forward pass:
        Input idx:           (B, T) token indices
        Token Embeddings:    (B, T, C)
        + Learned Pos Emb:   (B, T, C) if use_rope is False
        Transformer Blocks:  (B, T, C)
        Final Norm:          (B, T, C)
        LM Head Projection:  (B, T, vocab_size) logits
        Cross Entropy Loss:  scalar float
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size

        # 1. Token Embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)

        # 2. Position Embeddings (RoPE vs Learned)
        if config.use_rope:
            head_dim = config.n_embd // config.n_head
            cos, sin = precompute_rope_freqs(head_dim, config.block_size)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
            self.pos_embeddings = None
        else:
            self.pos_embeddings = nn.Embedding(config.block_size, config.n_embd)
            self.rope_cos = None
            self.rope_sin = None

        self.drop = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

        # 3. Stack of Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                n_embd=config.n_embd,
                n_head=config.n_head,
                block_size=config.block_size,
                dropout=config.dropout,
                norm_type=config.norm_type,
                activation=config.activation,
                use_rope=config.use_rope,
                use_sdpa=config.use_sdpa,
            )
            for _ in range(config.n_layer)
        ])

        # 4. Final Normalization
        self.ln_f = get_norm(config.norm_type, config.n_embd)

        # 5. Language Model Head Projection
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 6. Weight Tying
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        # 7. Parameter initialization
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("w_down.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = True) -> int:
        """Counts total trainable model parameters."""
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, betas: Tuple[float, float], device_type: str
    ) -> torch.optim.AdamW:
        decay_params = []
        nodecay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        offset: int = 0,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        device = idx.device
        B, T = idx.shape
        assert (
            offset + T <= self.block_size
        ), f"Cannot forward sequence of length {offset + T}, block_size is {self.block_size}"

        # 1. Embeddings
        x = self.tok_embeddings(idx)
        if self.pos_embeddings is not None:
            positions = torch.arange(offset, offset + T, dtype=torch.long, device=device)
            x = x + self.pos_embeddings(positions)
        x = self.drop(x)

        # 2. RoPE frequencies
        cos = self.rope_cos if self.config.use_rope else None
        sin = self.rope_sin if self.config.use_rope else None

        # 3. Transformer blocks
        new_kv_caches = [] if (kv_caches is not None or use_cache) else None
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, updated_cache = block(
                x, cos=cos, sin=sin, kv_cache=layer_cache, offset=offset, use_cache=use_cache
            )
            if new_kv_caches is not None:
                new_kv_caches.append(updated_cache)

        # 4. Final normalization
        x = self.ln_f(x)

        # 5. LM Head
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x)
            loss = None

        return logits, loss, new_kv_caches
