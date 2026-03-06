from __future__ import annotations

import random
import re

MOODS = ("PLAYFUL", "HUNGRY", "SLEEPY", "GRUMPY")
HINTS = {
    "HUNGRY": {"food", "hungry", "feed", "treat", "dinner", "breakfast", "lunch"},
    "SLEEPY": {"sleep", "sleepy", "nap", "tired", "bed", "cozy", "rest"},
    "GRUMPY": {"bad", "no", "stop", "angry", "mad", "shoo", "scold"},
    "PLAYFUL": {"play", "toy", "laser", "string", "zoomies", "pet", "cute"},
}
TRANSITIONS = {
    "PLAYFUL": {"PLAYFUL": 0.55, "HUNGRY": 0.2, "SLEEPY": 0.15, "GRUMPY": 0.1},
    "HUNGRY": {"PLAYFUL": 0.2, "HUNGRY": 0.6, "SLEEPY": 0.1, "GRUMPY": 0.1},
    "SLEEPY": {"PLAYFUL": 0.2, "HUNGRY": 0.1, "SLEEPY": 0.65, "GRUMPY": 0.05},
    "GRUMPY": {"PLAYFUL": 0.2, "HUNGRY": 0.15, "SLEEPY": 0.1, "GRUMPY": 0.55},
}
INITIAL = {"PLAYFUL": 0.5, "HUNGRY": 0.2, "SLEEPY": 0.2, "GRUMPY": 0.1}
WORD_RE = re.compile(r"[a-z']+")


def pick(weights: dict[str, float], rng: random.Random) -> str:
    return rng.choices(MOODS, [weights[m] for m in MOODS], k=1)[0]


def initial_mood(rng: random.Random) -> str:
    return pick(INITIAL, rng)


def next_mood(prev: str, text: str, rng: random.Random) -> str:
    weights = dict(TRANSITIONS[prev])
    words = set(WORD_RE.findall(text.lower()))

    for mood, hints in HINTS.items():
        hits = len(words & hints)
        if hits:
            weights[mood] += min(0.9, 0.3 * hits)

    if "!" in text:
        weights["HUNGRY"] += 0.08
        weights["GRUMPY"] += 0.08

    for mood in MOODS:
        weights[mood] += rng.uniform(0.0, 0.03)

    return pick(weights, rng)
