from __future__ import annotations

import argparse
import random

import torch

from .model import load_checkpoint
from .mood import initial_mood, next_mood
from .runtime import pick_device


def sample_reply(
    model,
    tok,
    mood: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
) -> str:
    prefix = f"<MOOD={mood}> "
    x = torch.tensor([tok.encode(prefix)], dtype=torch.long, device=device)
    y = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_token_id=tok.stoi["\n"],
    )
    return tok.decode(y[0].tolist())[len(prefix) :].split("\n", 1)[0].strip() or "mrrp"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chat with CatGPT")
    p.add_argument("--checkpoint", default="checkpoints/catgpt.pt")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--message", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--force-mood", default=None, choices=["HUNGRY", "SLEEPY", "GRUMPY", "PLAYFUL"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    device = pick_device(args.device)
    model, tok, _ = load_checkpoint(args.checkpoint, device)

    if args.message is not None:
        mood = args.force_mood or next_mood("PLAYFUL", args.message, rng)
        print(f"CatGPT ({mood}): {sample_reply(model, tok, mood, args.max_new_tokens, args.temperature, args.top_k, device)}")
        return

    print("Type messages and press enter. Use 'quit' to exit.")
    mood = args.force_mood or initial_mood(rng)
    while True:
        msg = input("You: ").strip()
        if not msg:
            continue
        if msg.lower() in {"quit", "exit"}:
            break
        if not args.force_mood:
            mood = next_mood(mood, msg, rng)
        print(f"CatGPT ({mood}): {sample_reply(model, tok, mood, args.max_new_tokens, args.temperature, args.top_k, device)}")


if __name__ == "__main__":
    main()
