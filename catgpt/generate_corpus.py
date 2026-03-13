from __future__ import annotations

import argparse
from dataclasses import dataclass
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

LEVELS = ("low", "mid", "high", "peak")
LEVEL_INDEX = {level: i for i, level in enumerate(LEVELS)}
ROOMS = ("kitchen", "living_room", "hallway", "window", "bedroom", "sofa", "desk")
FOOD_CUES = ("can_open", "treat_bag", "bowl_rattle", "kibble_scent")
PLAY_CUES = ("laser_dot", "string_twitch", "toy_mouse", "feather_wand")
SCARE_CUES = ("vacuum_noise", "stranger_scent", "bath_water", "door_bang")
REST_CUES = ("warm_blanket", "sunbeam", "quiet_room", "lap_heat")
MISCHIEF_CUES = ("keyboard_warm", "glass_edge", "plant_leaves", "counter_corner", "box_flap")
DREAM_CUES = ("moon_moth", "ghost_mouse", "cloud_string", "starlight_bowl", "floating_box")
FOLLOW_UP_PROMPTS = (
    "where is it now",
    "what are you staring at",
    "still thinking about that",
    "what did you do over there",
    "is it still there",
)


@dataclass
class EpisodeState:
    hunger: str
    energy: str
    trust: str
    mischief: str
    room: str
    bowl: str
    toy: str
    vacuum: str
    sunbeam: str
    box: str
    dream: str
    focus: str
    memory: str
    last_action: str


@dataclass
class Turn:
    prompt: str
    cue: str
    action: str
    plan: str
    mood: str


def pick(rng: random.Random, items: list[tuple[str, float]]) -> str:
    sounds, weights = zip(*items)
    return rng.choices(sounds, weights=weights, k=1)[0]


def weighted_choice(rng: random.Random, items: list[tuple[str, float]]) -> str:
    labels, weights = zip(*items)
    return rng.choices(labels, weights=weights, k=1)[0]


def utterance(rng: random.Random, mood: str | None, lo: int, hi: int) -> str:
    pool = MOOD_SOUNDS[mood] if mood else BASE_SOUNDS
    return " ".join(pick(rng, pool) for _ in range(rng.randint(lo, hi)))


def shift_level(value: str, delta: int) -> str:
    idx = min(len(LEVELS) - 1, max(0, LEVEL_INDEX[value] + delta))
    return LEVELS[idx]


def level(value: str) -> int:
    return LEVEL_INDEX[value]


def dominant_drive(state: EpisodeState, action: str) -> str:
    drives = {
        "hunger": level(state.hunger) + (3 if action == "seek_food" else 0),
        "rest": (3 - level(state.energy)) + (3 if action == "rest" else 0),
        "bond": level(state.trust) + (3 if action == "seek_affection" else 0),
        "mischief": level(state.mischief) + (3 if action in {"play", "make_mischief"} else 0),
    }
    return max(drives.items(), key=lambda item: (item[1], item[0]))[0]


def derive_mood(state: EpisodeState) -> str:
    if state.dream != "awake":
        return "SLEEPY"
    if state.focus == "vacuum" or (state.vacuum == "hallway" and level(state.trust) <= 1):
        return "GRUMPY"
    if level(state.hunger) >= 2 and state.bowl != "full":
        return "HUNGRY"
    if level(state.energy) <= 1 and state.sunbeam != "gone":
        return "SLEEPY"
    if level(state.mischief) >= 2 or state.focus in {"toy", "keyboard", "glass", "plant", "box"}:
        return "PLAYFUL"
    if level(state.trust) <= 1:
        return "GRUMPY"
    return "PLAYFUL"


def random_state(rng: random.Random) -> EpisodeState:
    return EpisodeState(
        hunger=weighted_choice(rng, [("low", 1), ("mid", 3), ("high", 3), ("peak", 1)]),
        energy=weighted_choice(rng, [("low", 1), ("mid", 3), ("high", 3), ("peak", 1)]),
        trust=weighted_choice(rng, [("low", 1), ("mid", 3), ("high", 2), ("peak", 1)]),
        mischief=weighted_choice(rng, [("low", 1), ("mid", 3), ("high", 3), ("peak", 1)]),
        room=rng.choice(ROOMS),
        bowl=rng.choice(("empty", "half", "full")),
        toy=rng.choice(("basket", "hallway", "under_couch", "sofa", "desk")),
        vacuum=rng.choice(("closet", "closet", "hallway")),
        sunbeam=rng.choice(("window", "sofa", "gone")),
        box=rng.choice(("hallway", "sofa", "desk", "claimed")),
        dream=weighted_choice(rng, [("awake", 6), ("drifting", 2), ("deep", 1)]),
        focus=rng.choice(("bowl", "toy", "sunbeam", "box", "human")),
        memory=rng.choice(("quiet_house", "footsteps", "treat_rustle", "toy_roll")),
        last_action="observe",
    )


def food_turn(state: EpisodeState, rng: random.Random) -> Turn:
    state.room = "kitchen"
    state.focus = "bowl"
    state.memory = rng.choice(("heard_can_open", "smelled_tuna", "bowl_echo", "cabinet_watch"))
    state.bowl = rng.choice(("half", "full")) if state.bowl == "empty" else rng.choice(("empty", "half"))
    state.dream = "awake"
    state.hunger = shift_level(state.hunger, -1)
    state.energy = shift_level(state.energy, -1)
    state.trust = shift_level(state.trust, 1)
    state.mischief = shift_level(state.mischief, -1)
    state.last_action = "seek_food"
    prompt = rng.choice(
        (
            "i opened a tuna can",
            "your bowl is ready",
            "want a treat",
            "dinner time kitty",
            "do you hear the food scoop",
        )
    )
    mood = derive_mood(state)
    return Turn(
        prompt=prompt,
        cue=rng.choice(FOOD_CUES),
        action="seek_food",
        plan=rng.choice(("trot>figure_eight>stare", "chirp>escort>eat", "tail_up>cabinet_check>munch")),
        mood=mood,
    )


def play_turn(state: EpisodeState, rng: random.Random) -> Turn:
    state.room = rng.choice(("living_room", "hallway", "desk"))
    state.focus = "toy"
    state.memory = rng.choice(("toy_skitter", "laser_flash", "feather_bounce", "tiny_prey"))
    state.toy = rng.choice(("hallway", "under_couch", "sofa", "desk"))
    state.dream = "awake"
    state.energy = shift_level(state.energy, -1)
    state.hunger = shift_level(state.hunger, 1)
    state.mischief = shift_level(state.mischief, 1)
    state.last_action = "play"
    prompt = rng.choice(
        (
            "want the feather wand",
            "i found your toy mouse",
            "chase the laser dot",
            "play with this string",
            "zoomies time",
        )
    )
    mood = derive_mood(state)
    return Turn(
        prompt=prompt,
        cue=rng.choice(PLAY_CUES),
        action="play",
        plan=rng.choice(("stalk>wiggle>pounce", "skid>spin>bat", "listen>paw_probe>hook")),
        mood=mood,
    )


def cuddle_turn(state: EpisodeState, rng: random.Random) -> Turn:
    state.room = rng.choice(("sofa", "bedroom"))
    state.focus = "human"
    state.memory = rng.choice(("warm_lap", "gentle_voice", "slow_blink", "soft_blanket"))
    state.dream = "awake"
    state.energy = shift_level(state.energy, 1)
    state.trust = shift_level(state.trust, 1)
    state.mischief = shift_level(state.mischief, -1)
    state.last_action = "seek_affection"
    prompt = rng.choice(
        (
            "come cuddle on my lap",
            "you are such a sweet cat",
            "want gentle pets",
            "come here little friend",
            "good kitty",
        )
    )
    mood = derive_mood(state)
    return Turn(
        prompt=prompt,
        cue="warm_human",
        action="seek_affection",
        plan=rng.choice(("slowblink>head_bonk>settle", "approach>circle>knead", "chirp>lap_hop>purr")),
        mood=mood,
    )


def rest_turn(state: EpisodeState, rng: random.Random) -> Turn:
    was_sleepy = level(state.energy) <= 1
    state.room = "window" if state.sunbeam == "window" else rng.choice(("sofa", "bedroom", "window"))
    state.focus = "sunbeam" if state.sunbeam != "gone" else "blanket"
    state.memory = rng.choice(("warm_patch", "quiet_blanket", "lap_heat", "soft_pillow"))
    state.dream = weighted_choice(rng, [("awake", 2), ("drifting", 4), ("deep", 2 if was_sleepy else 1)])
    state.energy = shift_level(state.energy, 1)
    state.hunger = shift_level(state.hunger, 1)
    state.mischief = shift_level(state.mischief, -1)
    state.last_action = "rest"
    prompt = rng.choice(
        (
            "the blanket is warm",
            "want to nap by the window",
            "cozy bed is ready",
            "you look sleepy",
            "rest in the sunbeam",
        )
    )
    mood = derive_mood(state)
    return Turn(
        prompt=prompt,
        cue=rng.choice(REST_CUES),
        action="rest",
        plan=rng.choice(("yawn>knead>curl", "slowblink>loaf>purr", "stretch>circle>doze")),
        mood=mood,
    )


def scare_turn(state: EpisodeState, rng: random.Random) -> Turn:
    state.room = rng.choice(("bedroom", "hallway", "living_room"))
    state.focus = "vacuum"
    state.memory = rng.choice(("loud_corner", "door_shadow", "wet_room", "shaky_whiskers"))
    state.vacuum = "hallway"
    state.dream = "awake"
    state.trust = shift_level(state.trust, -1)
    state.energy = shift_level(state.energy, 1)
    state.mischief = shift_level(state.mischief, -1)
    state.last_action = "hide"
    prompt = rng.choice(
        (
            "the vacuum is out",
            "a stranger is here",
            "time for a bath",
            "that noise is scary",
            "did the door just slam",
        )
    )
    mood = derive_mood(state)
    return Turn(
        prompt=prompt,
        cue=rng.choice(SCARE_CUES),
        action="hide",
        plan=rng.choice(("freeze>flatten_ears>vanish", "hiss>backstep>under_bed", "crouch>tail_puff>retreat")),
        mood=mood,
    )


def mischief_turn(state: EpisodeState, rng: random.Random) -> Turn:
    focus = rng.choice(("keyboard", "glass", "plant", "box", "counter"))
    state.room = "desk" if focus in {"keyboard", "glass"} else rng.choice(("kitchen", "living_room", "desk"))
    state.focus = focus
    state.memory = rng.choice(("forbidden_edge", "tiny_rebellion", "silent_target", "chaos_idea"))
    state.dream = "awake"
    if focus == "box":
        state.box = "claimed"
    state.mischief = shift_level(state.mischief, 1)
    state.energy = shift_level(state.energy, -1)
    state.last_action = "make_mischief"
    prompt = rng.choice(
        (
            "please leave the keyboard alone",
            "do not knock that glass over",
            "stay off the counter",
            "what are you doing to the plant",
            "that box is not for biting",
        )
    )
    mood = derive_mood(state)
    return Turn(
        prompt=prompt,
        cue=rng.choice(MISCHIEF_CUES),
        action="make_mischief",
        plan=rng.choice(("creep>paw_test>push", "stare>swat>bolt", "wiggle>launch>scatter")),
        mood=mood,
    )


def dream_turn(state: EpisodeState, rng: random.Random) -> Turn:
    state.room = rng.choice(("window", "sofa", "bedroom"))
    state.focus = rng.choice(("moon_moth", "ghost_mouse", "cloud_string", "starlight_bowl", "floating_box"))
    state.memory = rng.choice(("silver_birds", "soft_static", "tiny_kicks", "night_zoomies"))
    state.dream = weighted_choice(rng, [("drifting", 3), ("deep", 5)])
    state.energy = shift_level(state.energy, -1)
    state.mischief = shift_level(state.mischief, -1)
    state.last_action = "dream"
    prompt = rng.choice(
        (
            "what are you dreaming about",
            "still asleep in the sunbeam",
            "did you chase something in your sleep",
            "little paws are twitching",
            "what did you see in that nap",
        )
    )
    mood = derive_mood(state)
    return Turn(
        prompt=prompt,
        cue=rng.choice(DREAM_CUES),
        action="dream",
        plan=rng.choice(("doze>moon_pounce>twitch", "float>tail_flick>purr", "snore>paw_kick>slowblink", "drift>cloud_chase>mrrp")),
        mood=mood,
    )


def follow_up_turn(state: EpisodeState, rng: random.Random) -> Turn:
    prompt = rng.choice(FOLLOW_UP_PROMPTS)
    action = state.last_action if state.last_action in {
        "seek_food",
        "play",
        "seek_affection",
        "rest",
        "hide",
        "make_mischief",
        "dream",
    } else "observe"
    if state.dream != "awake":
        action = "dream"
    if action == "seek_food":
        plan = rng.choice(("sniff>check_bowl>stare", "listen>kitchen_trot>meow", "circle>pause>munch"))
        state.room = "kitchen"
        state.focus = "bowl"
    elif action == "play":
        plan = rng.choice(("listen>paw_probe>hook", "peek>spring>bat", "crouch>dart>grab"))
        state.focus = "toy"
    elif action == "seek_affection":
        plan = rng.choice(("slowblink>step_close>lean", "chirp>lap_hop>settle", "purr>knead>curl"))
        state.focus = "human"
    elif action == "rest":
        plan = rng.choice(("blink>stretch>loaf", "yawn>curl>purr", "loaf>tuck_paws>doze"))
    elif action == "dream":
        plan = rng.choice(("doze>moon_pounce>twitch", "float>tail_flick>purr", "snore>paw_kick>slowblink"))
        state.focus = rng.choice(("moon_moth", "ghost_mouse", "cloud_string", "floating_box"))
        state.dream = weighted_choice(rng, [("drifting", 2), ("deep", 4)])
    elif action == "hide":
        plan = rng.choice(("listen>stay_small>peek", "crouch>pause>retreat", "freeze>peek>retreat"))
        state.focus = "vacuum"
    else:
        plan = rng.choice(("peek>paw_test>bolt", "stare>tiny_swat>flee", "inspect>tap>scatter"))
    state.memory = f"still_{state.focus}"
    state.last_action = action
    mood = derive_mood(state)
    cue = state.focus if state.focus != "human" else "warm_human"
    return Turn(prompt=prompt, cue=cue, action=action, plan=plan, mood=mood)


def choose_turn(state: EpisodeState, rng: random.Random) -> Turn:
    options = [
        ("food", 1.0 + level(state.hunger) + (1.5 if state.bowl == "empty" else 0.0)),
        ("play", 1.0 + level(state.energy) + level(state.mischief)),
        ("cuddle", 0.8 + level(state.trust)),
        ("rest", 1.0 + (3 - level(state.energy)) + (0.8 if state.sunbeam != "gone" else 0.0)),
        ("dream", 0.5 + (2.4 if state.dream != "awake" else 0.0) + (1.8 if level(state.energy) <= 1 else 0.0)),
        ("scare", 0.5 + (1.8 if state.vacuum == "hallway" else 0.0)),
        ("mischief", 0.9 + level(state.mischief) + (0.6 if state.room == "desk" else 0.0)),
        ("follow_up", 0.5 + (1.8 if state.focus in {"toy", "bowl", "vacuum", "box", "keyboard", "glass", "plant"} else 0.0)),
    ]
    choice = weighted_choice(rng, options)
    if choice == "food":
        return food_turn(state, rng)
    if choice == "play":
        return play_turn(state, rng)
    if choice == "cuddle":
        return cuddle_turn(state, rng)
    if choice == "rest":
        return rest_turn(state, rng)
    if choice == "dream":
        return dream_turn(state, rng)
    if choice == "scare":
        return scare_turn(state, rng)
    if choice == "mischief":
        return mischief_turn(state, rng)
    return follow_up_turn(state, rng)


def drift_state(state: EpisodeState, rng: random.Random) -> None:
    if rng.random() < 0.55:
        state.hunger = shift_level(state.hunger, 1)
    if state.last_action not in {"rest", "seek_affection"} and rng.random() < 0.45:
        state.energy = shift_level(state.energy, -1)
    if state.bowl == "full" and rng.random() < 0.4:
        state.bowl = "half"
    elif state.bowl == "half" and rng.random() < 0.35:
        state.bowl = "empty"
    if state.vacuum == "hallway" and rng.random() < 0.35:
        state.vacuum = "closet"
    if state.sunbeam != "gone" and rng.random() < 0.28:
        state.sunbeam = rng.choice(("window", "sofa", "gone"))
    if state.dream == "deep" and rng.random() < 0.32:
        state.dream = "drifting"
    elif state.dream == "drifting" and rng.random() < 0.4:
        state.dream = "awake"
    if state.focus == "toy" and rng.random() < 0.3:
        state.toy = rng.choice(("hallway", "under_couch", "sofa", "desk"))


def think_fields(state: EpisodeState, turn: Turn) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = [
        ("q", turn.cue),
        ("d", dominant_drive(state, turn.action)),
        ("a", turn.action),
        ("p", turn.plan),
        ("h", state.hunger),
        ("e", state.energy),
        ("t", state.trust),
        ("m", state.mischief),
        ("r", state.room),
        ("f", state.focus),
        ("n", state.memory),
    ]

    if state.dream != "awake" or turn.action in {"rest", "dream"}:
        fields.append(("z", state.dream))

    if turn.action == "seek_food" or state.focus == "bowl":
        fields.append(("b", state.bowl))
    if turn.action == "play" or state.focus == "toy":
        fields.append(("y", state.toy))
    if turn.action == "hide" or state.focus == "vacuum":
        fields.append(("v", state.vacuum))
    if turn.action == "rest" or state.focus in {"sunbeam", "blanket"}:
        fields.append(("s", state.sunbeam))
    if turn.action == "make_mischief" or state.focus == "box":
        fields.append(("x", state.box))

    return fields


def reasoning_line(state: EpisodeState, rng: random.Random, lo: int, hi: int) -> str:
    turn = choose_turn(state, rng)
    think = " ".join(f"{key}={value}" for key, value in think_fields(state, turn))
    reply = utterance(rng, turn.mood, lo, hi)
    drift_state(state, rng)
    return f"<USER>{turn.prompt}</USER><THINK>{think}</THINK><MOOD={turn.mood}> {reply}"


def free_cat_line(rng: random.Random, lo: int, hi: int, mood_prob: float) -> str:
    mood = rng.choice(list(MOOD_SOUNDS)) if rng.random() < mood_prob else None
    line = utterance(rng, mood, lo, hi)
    return f"<MOOD={mood}> {line}" if mood else line


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate CatGPT corpus with reasoning traces")
    p.add_argument("--out", default="data/cat_corpus.txt")
    p.add_argument("--lines", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--reasoning-prob", type=float, default=0.86)
    p.add_argument("--free-mood-prob", type=float, default=0.5)
    p.add_argument("--min-sounds", type=int, default=1)
    p.add_argument("--max-sounds", type=int, default=6)
    p.add_argument("--episode-min-turns", type=int, default=3)
    p.add_argument("--episode-max-turns", type=int, default=7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out.open("w", encoding="utf-8") as f:
        with tqdm(total=args.lines, desc="Generating") as bar:
            while written < args.lines:
                if rng.random() >= args.reasoning_prob:
                    line = free_cat_line(rng, args.min_sounds, args.max_sounds, args.free_mood_prob)
                    f.write(line + "\n")
                    written += 1
                    bar.update(1)
                    continue

                state = random_state(rng)
                turns = rng.randint(args.episode_min_turns, args.episode_max_turns)
                for _ in range(turns):
                    if written >= args.lines:
                        break
                    line = reasoning_line(state, rng, args.min_sounds, args.max_sounds)
                    f.write(line + "\n")
                    written += 1
                    bar.update(1)

    print(f"Wrote {written:,} lines to {out}")


if __name__ == "__main__":
    main()
