from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @classmethod
    def from_chars(cls, chars: list[str]) -> "CharTokenizer":
        return cls(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 32
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


class Attention(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.n_head = c.n_head
        self.dropout = c.dropout
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)
        self.proj = nn.Linear(c.n_embd, c.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = rearrange(self.qkv(x), "b t (qkv h d) -> qkv b h t d", qkv=3, h=self.n_head).unbind(0)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self.proj(rearrange(y, "b h t d -> b t (h d)"))


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.attn = Attention(c)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.ff = nn.Sequential(
            nn.Linear(c.n_embd, 4 * c.n_embd),
            nn.GELU(),
            nn.Linear(4 * c.n_embd, c.n_embd),
            nn.Dropout(c.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.config = c
        self.tok = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos = nn.Embedding(c.block_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))[None]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln(x))

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            rearrange(logits, "b t v -> (b t) v"),
            rearrange(targets, "b t -> (b t)"),
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        stop_token_id: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.config.block_size :])
            logits = logits[:, -1] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -torch.inf
            next_id = torch.multinomial(logits.softmax(dim=-1), 1)
            idx = torch.cat((idx, next_id), dim=1)
            if stop_token_id is not None and torch.all(next_id == stop_token_id):
                break
        return idx


@torch.no_grad()
def load_checkpoint(path: str, device: str) -> tuple[GPT, CharTokenizer, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    tok = CharTokenizer.from_chars(ckpt["chars"])
    cfg = GPTConfig(**ckpt["model_config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tok, ckpt


def checkpoint_payload(
    model: GPT,
    tokenizer: CharTokenizer,
    config: GPTConfig,
    step: int,
    train_config: dict[str, Any],
    best_val_loss: float,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "model_config": asdict(config),
        "chars": tokenizer.chars,
        "step": step,
        "train_config": train_config,
        "best_val_loss": best_val_loss,
    }
