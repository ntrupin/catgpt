from __future__ import annotations

import argparse
import random
from pathlib import Path

from tqdm import tqdm


BASE_SOUNDS: list[tuple[str, float]] = [
    ("meow", 20),
    ("meow!", 8),
    ("meow?", 4),
    ("meow...", 4),
    ("MEOW", 2),
    ("MEEEEOW!!!", 1),
    ("meeeeow", 4),
    ("mew", 5),
    ("mrrp", 10),
    ("mrrp?", 3),
    ("mrrow", 8),
    ("mrrrrrow", 2),
    ("prrr", 9),
    ("prrrr", 5),
    ("hiss", 2),
    ("hiss!", 2),
    ("brrp", 6),
    ("nya", 6),
]

MOOD_SOUNDS: dict[str, list[tuple[str, float]]] = {
    "HUNGRY": [
        ("meow", 20),
        ("meow!", 14),
        ("MEEEEOW!!!", 6),
        ("mew", 8),
        ("brrp", 3),
        ("mrrp?", 4),
    ],
    "SLEEPY": [
        ("prrr", 18),
        ("prrrr", 14),
        ("mrrp", 7),
        ("meow...", 7),
        ("mew", 4),
    ],
    "GRUMPY": [
        ("hiss", 16),
        ("hiss!", 12),
        ("mrrow", 12),
        ("mrrrrrow", 6),
        ("meow?", 3),
    ],
    "PLAYFUL": [
        ("brrp", 12),
        ("nya", 12),
        ("mrrp", 10),
        ("meow?", 8),
        ("meow", 8),
        ("mew", 7),
    ],
}


def pick(rng: random.Random, items: list[tuple[str, float]]) -> str:
    sounds, weights = zip(*items)
    return rng.choices(sounds, weights=weights, k=1)[0]


def utterance(rng: random.Random, mood: str | None, lo: int, hi: int) -> str:
    pool = MOOD_SOUNDS[mood] if mood else BASE_SOUNDS
    return " ".join(pick(rng, pool) for _ in range(rng.randint(lo, hi)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic cat speech corpus")
    p.add_argument("--out", default="data/cat_corpus.txt")
    p.add_argument("--lines", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--mood-prob", type=float, default=0.4)
    p.add_argument("--min-sounds", type=int, default=1)
    p.add_argument("--max-sounds", type=int, default=6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    moods = list(MOOD_SOUNDS)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for _ in tqdm(range(args.lines), total=args.lines, desc="Generating"):
            mood = rng.choice(moods) if rng.random() < args.mood_prob else None
            line = utterance(rng, mood, args.min_sounds, args.max_sounds)
            f.write(f"<MOOD={mood}> {line}\n" if mood else f"{line}\n")

    print(f"Wrote {args.lines:,} lines to {out}")


if __name__ == "__main__":
    main()
