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

SCENARIOS = [
    {
        "mood": "HUNGRY",
        "action": "seek_food",
        "cues": ["food", "treat", "can_open", "dinner_time"],
        "drives": ["hunger_high", "patience_low", "stomach_loud"],
        "prompts": [
            "do you want food",
            "time for dinner kitty",
            "i opened a tuna can",
            "want a treat",
            "are you hungry",
        ],
    },
    {
        "mood": "SLEEPY",
        "action": "rest",
        "cues": ["bed", "blanket", "quiet_room", "warm_lap"],
        "drives": ["sleepiness_high", "stress_low", "cozy_seek"],
        "prompts": [
            "come nap on the blanket",
            "it is sleepy time",
            "cozy bed is ready",
            "want to rest",
            "you look tired",
        ],
    },
    {
        "mood": "PLAYFUL",
        "action": "play",
        "cues": ["toy", "laser", "string", "zoomies"],
        "drives": ["play_drive_high", "energy_up", "curious"],
        "prompts": [
            "want to chase the laser",
            "i found your toy mouse",
            "play with this string",
            "zoomies time",
            "come play",
        ],
    },
    {
        "mood": "PLAYFUL",
        "action": "seek_affection",
        "cues": ["lap", "petting", "cuddle", "gentle_voice"],
        "drives": ["trust_high", "social_seek", "stress_low"],
        "prompts": [
            "come cuddle on my lap",
            "you are such a sweet cat",
            "want gentle pets",
            "come here little friend",
            "good kitty",
        ],
    },
    {
        "mood": "GRUMPY",
        "action": "hide",
        "cues": ["vacuum", "loud_noise", "stranger", "bath"],
        "drives": ["stress_high", "safety_seek", "trust_drop"],
        "prompts": [
            "the vacuum is loud",
            "a stranger is here",
            "time for a bath",
            "that noise is scary",
            "you seem upset",
        ],
    },
]


def pick(rng: random.Random, items: list[tuple[str, float]]) -> str:
    sounds, weights = zip(*items)
    return rng.choices(sounds, weights=weights, k=1)[0]


def utterance(rng: random.Random, mood: str | None, lo: int, hi: int) -> str:
    pool = MOOD_SOUNDS[mood] if mood else BASE_SOUNDS
    return " ".join(pick(rng, pool) for _ in range(rng.randint(lo, hi)))


def reasoning_line(rng: random.Random, lo: int, hi: int) -> str:
    s = rng.choice(SCENARIOS)
    mood = s["mood"]
    prompt = rng.choice(s["prompts"])
    if rng.random() < 0.25:
        prompt += "?"
    think = (
        f"cue={rng.choice(s['cues'])} drive={rng.choice(s['drives'])} "
        f"action={s['action']} mood={mood}"
    )
    return f"<USER>{prompt}</USER><THINK>{think}</THINK><MOOD={mood}> {utterance(rng, mood, lo, hi)}"


def free_cat_line(rng: random.Random, lo: int, hi: int, mood_prob: float) -> str:
    mood = rng.choice(list(MOOD_SOUNDS)) if rng.random() < mood_prob else None
    line = utterance(rng, mood, lo, hi)
    return f"<MOOD={mood}> {line}" if mood else line


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate CatGPT corpus with reasoning traces")
    p.add_argument("--out", default="data/cat_corpus.txt")
    p.add_argument("--lines", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--reasoning-prob", type=float, default=0.8)
    p.add_argument("--free-mood-prob", type=float, default=0.5)
    p.add_argument("--min-sounds", type=int, default=1)
    p.add_argument("--max-sounds", type=int, default=6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for _ in tqdm(range(args.lines), total=args.lines, desc="Generating"):
            if rng.random() < args.reasoning_prob:
                line = reasoning_line(rng, args.min_sounds, args.max_sounds)
            else:
                line = free_cat_line(rng, args.min_sounds, args.max_sounds, args.free_mood_prob)
            f.write(line + "\n")

    print(f"Wrote {args.lines:,} lines to {out}")


if __name__ == "__main__":
    main()
