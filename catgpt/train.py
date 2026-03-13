from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import trange

from .model import CharTokenizer, GPT, GPTConfig, checkpoint_payload
from .runtime import pick_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train tiny CatGPT")
    p.add_argument("--data", default="data/cat_corpus.txt")
    p.add_argument("--out", default="checkpoints/catgpt.pt")
    p.add_argument("--steps", type=int, default=3_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=40)
    p.add_argument("--sample-prompt", default="<USER>hello cat</USER><THINK>")
    p.add_argument("--sample-tokens", type=int, default=48)
    p.add_argument("--sample-temperature", type=float, default=0.9)
    p.add_argument("--sample-top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    print(f"Using device: {device}")

    text = Path(args.data).read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)

    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    print(
        f"Dataset chars: {len(data):,} | vocab: {tok.vocab_size} | "
        f"train chars: {len(train_data):,} | val chars: {len(val_data):,}"
    )

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.hidden,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def get_batch(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(0, len(source) - args.block_size - 1, (args.batch_size,))
        x = torch.stack([source[i : i + args.block_size] for i in ix]).to(device)
        y = torch.stack([source[i + 1 : i + args.block_size + 1] for i in ix]).to(device)
        return x, y

    @torch.no_grad()
    def eval_loss(source: torch.Tensor) -> float:
        model.eval()
        losses = torch.zeros(args.eval_iters)
        for i in range(args.eval_iters):
            _, loss = model(*get_batch(source))
            losses[i] = loss
        model.train()
        return losses.mean().item()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    t0 = time.time()
    train_str = "--"
    val_str = "--"

    progress = trange(1, args.steps + 1, desc="Training", ncols=100)
    for step in progress:
        _, loss = model(*get_batch(train_data))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            train_loss = eval_loss(train_data)
            val_loss = eval_loss(val_data)
            train_str = f"{train_loss:.4f}"
            val_str = f"{val_loss:.4f}"
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    checkpoint_payload(model, tok, cfg, step, vars(args), best_val),
                    out,
                )
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            train=train_str,
            val=val_str,
            best=f"{best_val:.4f}" if best_val < float("inf") else "--",
            elapsed=f"{time.time() - t0:.1f}s",
        )

    print(f"Saved best checkpoint to {out} (val_loss={best_val:.4f})")

    with torch.no_grad():
        x = torch.tensor([tok.encode(args.sample_prompt)], dtype=torch.long, device=device)
        y = model.generate(
            x,
            max_new_tokens=args.sample_tokens,
            temperature=args.sample_temperature,
            top_k=args.sample_top_k,
            stop_token_id=tok.stoi["\n"],
        )
    print("--- sample ---")
    print(tok.decode(y[0].tolist()))


if __name__ == "__main__":
    main()
