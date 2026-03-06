from __future__ import annotations

import argparse
import random

import torch

from .model import load_checkpoint
from .mood import initial_mood, next_mood
from .runtime import pick_device
from .ttc import missing_reasoning_chars, ttc_turn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chat with CatGPT")
    p.add_argument("--checkpoint", default="checkpoints/catgpt.pt")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--mode", default="reasoning", choices=["reasoning", "instant"])
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--message", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--reasoning-rollouts", type=int, default=8)
    p.add_argument("--mood-inertia", type=float, default=0.35)
    p.add_argument("--show-reasoning", action="store_true")
    return p.parse_args()


def sample_instant_reply(
    model,
    tok,
    mood: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    prefix = f"<MOOD={mood}> "
    encoded = tok.encode(prefix)
    x = torch.tensor([encoded], dtype=torch.long, device=device)
    y = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_token_id=tok.stoi.get("\n"),
    )
    return tok.decode(y[0].tolist())[len(prefix) :].split("\n", 1)[0].strip() or "mrrp"


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = pick_device(args.device)
    model, tok, _ = load_checkpoint(args.checkpoint, device)

    if args.mode == "reasoning":
        missing = missing_reasoning_chars(tok)
        if missing:
            joined = "".join(missing)
            raise SystemExit(
                f"Checkpoint tokenizer is missing reasoning chars: {joined!r}. "
                "Regenerate corpus with reasoning traces and retrain, or run --mode instant."
            )

    if args.message is not None:
        if args.mode == "reasoning":
            d = ttc_turn(
                model=model,
                tokenizer=tok,
                device=device,
                history="",
                user_message=args.message,
                rollouts=args.reasoning_rollouts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                mood_inertia=args.mood_inertia,
            )
            print(f"CatGPT ({d.mood}): {d.reply}")
            if args.show_reasoning:
                print(f"[think={d.think}; action={d.action}; consensus={d.consensus}/{d.used_samples}]")
        else:
            mood = next_mood("PLAYFUL", args.message, rng)
            reply = sample_instant_reply(
                model,
                tok,
                mood=mood,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            print(f"CatGPT ({mood}): {reply}")
        return

    print("Type messages and press enter. Use 'quit' to exit.")
    history = ""
    mood = initial_mood(rng)

    while True:
        msg = input("You: ").strip()
        if not msg:
            continue
        if msg.lower() in {"quit", "exit"}:
            break

        if args.mode == "reasoning":
            d = ttc_turn(
                model=model,
                tokenizer=tok,
                device=device,
                history=history,
                user_message=msg,
                rollouts=args.reasoning_rollouts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                mood_inertia=args.mood_inertia,
            )
            history = d.history
            print(f"CatGPT ({d.mood}): {d.reply}")
            if args.show_reasoning:
                print(f"[think={d.think}; action={d.action}; consensus={d.consensus}/{d.used_samples}]")
        else:
            mood = next_mood(mood, msg, rng)
            reply = sample_instant_reply(
                model,
                tok,
                mood=mood,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            print(f"CatGPT ({mood}): {reply}")


if __name__ == "__main__":
    main()
