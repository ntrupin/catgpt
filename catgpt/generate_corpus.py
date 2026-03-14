from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random

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
TIMES = ("dawn", "morning", "afternoon", "evening", "night")
TIME_INDEX = {name: i for i, name in enumerate(TIMES)}
ATTENTION_LEVELS = ("loose", "curious", "tracking", "locked")
ATTENTION_INDEX = {level: i for i, level in enumerate(ATTENTION_LEVELS)}
RESPONSE_LEVELS = ("ignoring", "selective", "present", "eager")
RESPONSE_INDEX = {level: i for i, level in enumerate(RESPONSE_LEVELS)}

ROOMS = ("kitchen", "living_room", "hallway", "window", "bedroom", "sofa", "desk")
FOOD_CUES = ("can_open", "treat_bag", "bowl_rattle", "kibble_scent")
PLAY_CUES = ("laser_dot", "string_twitch", "toy_mouse", "feather_wand")
SCARE_CUES = ("vacuum_noise", "stranger_scent", "bath_water", "door_bang")
REST_CUES = ("warm_blanket", "sunbeam", "quiet_room", "lap_heat")
MISCHIEF_CUES = ("keyboard_warm", "glass_edge", "plant_leaves", "counter_corner", "box_flap")
DREAM_CUES = ("moon_moth", "ghost_mouse", "cloud_string", "starlight_bowl", "floating_box")
OBSERVE_CUES = ("window_bird", "hall_rustle", "radiator_tick", "ceiling_moth", "door_shadow")
IGNORE_CUES = ("name_call", "room_invite", "kissy_noise", "phone_glow", "distant_psst")
FOLLOW_UP_PROMPTS = (
    "where is it now",
    "what are you staring at",
    "still thinking about that",
    "what did you do over there",
    "is it still there",
)
REDIRECT_PROMPTS = (
    "psst cat",
    "come here for a second",
    "hello tiny menace",
    "you can stop that now",
    "are you ignoring me",
)

REST_BODIES = ("loaf", "curl", "sprawl", "sun_loaf")
DREAM_BODIES = ("curl", "sprawl", "twitch")
HIDE_BODIES = ("crouch", "under_bed", "flat_ears")
PLAY_BODIES = ("crouch", "pounce_ready", "perch")
MISCHIEF_BODIES = ("prowl", "perch", "tail_wrap")
FOOD_BODIES = ("tail_up", "kitchen_orbit", "counter_peek")
AFFECTION_BODIES = ("lap_loaf", "lean", "knead")
OBSERVE_BODIES = ("perch", "desk_sphinx", "loaf")
FOCUS_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("bowl", 1.0),
    ("toy", 1.2),
    ("sunbeam", 1.0),
    ("box", 0.9),
    ("human", 1.1),
    ("bird", 0.8),
)


@dataclass
class EpisodeState:
    hunger: str
    energy: str
    trust: str
    mischief: str
    inertia: str
    time: str
    room: str
    bowl: str
    toy: str
    vacuum: str
    sunbeam: str
    box: str
    dream: str
    focus: str
    attention: str
    responsiveness: str
    body: str
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


def weighted_choice(rng: random.Random, items: list[tuple[str, float]] | tuple[tuple[str, float], ...]) -> str:
    labels, weights = zip(*items)
    return rng.choices(labels, weights=weights, k=1)[0]


def utterance(rng: random.Random, mood: str | None, lo: int, hi: int) -> str:
    pool = MOOD_SOUNDS[mood] if mood else BASE_SOUNDS
    return " ".join(pick(rng, pool) for _ in range(rng.randint(lo, hi)))


def shift_named(value: str, levels: tuple[str, ...], index: dict[str, int], delta: int) -> str:
    idx = min(len(levels) - 1, max(0, index[value] + delta))
    return levels[idx]


def shift_level(value: str, delta: int) -> str:
    return shift_named(value, LEVELS, LEVEL_INDEX, delta)


def shift_attention(value: str, delta: int) -> str:
    return shift_named(value, ATTENTION_LEVELS, ATTENTION_INDEX, delta)


def shift_response(value: str, delta: int) -> str:
    return shift_named(value, RESPONSE_LEVELS, RESPONSE_INDEX, delta)


def level(value: str) -> int:
    return LEVEL_INDEX[value]


def attention_level(value: str) -> int:
    return ATTENTION_INDEX[value]


def response_level(value: str) -> int:
    return RESPONSE_INDEX[value]


def advance_time(value: str, steps: int = 1) -> str:
    idx = (TIME_INDEX[value] + steps) % len(TIMES)
    return TIMES[idx]


def sunbeam_for_time(rng: random.Random, time_of_day: str) -> str:
    if time_of_day == "dawn":
        return "window"
    if time_of_day == "morning":
        return rng.choice(("window", "window", "sofa"))
    if time_of_day == "afternoon":
        return rng.choice(("window", "sofa", "sofa", "gone"))
    return "gone"


def room_for_focus(rng: random.Random, focus: str, current_room: str) -> str:
    if focus in {"bowl", "counter"}:
        return "kitchen"
    if focus in {"keyboard", "glass"}:
        return "desk"
    if focus in {"bird", "sunbeam"}:
        return "window"
    if focus in {"vacuum", "hall_noise"}:
        return "hallway"
    if focus == "human":
        return rng.choice(("sofa", "bedroom"))
    if focus == "toy":
        return rng.choice(("living_room", "hallway", "desk"))
    if focus == "plant":
        return rng.choice(("living_room", "desk"))
    if focus == "box":
        return rng.choice(("hallway", "sofa", "desk"))
    if focus in {"moon_moth", "ghost_mouse", "cloud_string", "starlight_bowl", "floating_box"}:
        return rng.choice(("window", "sofa", "bedroom"))
    return current_room


def cue_for_focus(focus: str) -> str:
    mapping = {
        "human": "warm_human",
        "bird": "window_bird",
        "hall_noise": "hall_rustle",
    }
    return mapping.get(focus, focus)


def environment_focus(state: EpisodeState, rng: random.Random) -> str:
    options: list[tuple[str, float]] = [
        ("bird", 1.6 if state.room == "window" else 0.6),
        ("hall_noise", 1.3 if state.room == "hallway" else 0.6),
        ("toy", 1.2),
        ("box", 1.0),
        ("sunbeam", 1.0 if state.sunbeam != "gone" else 0.2),
        ("keyboard", 0.8 if state.room == "desk" else 0.3),
        ("plant", 0.7 if state.room in {"living_room", "desk"} else 0.2),
    ]
    if state.focus != "human":
        options.append((state.focus, 1.2))
    return weighted_choice(rng, options)


def apply_behavior(
    state: EpisodeState,
    rng: random.Random,
    *,
    body_choices: tuple[str, ...] | None = None,
    attention_choices: list[tuple[str, float]] | None = None,
    response_choices: list[tuple[str, float]] | None = None,
    inertia_delta: int = 0,
) -> None:
    if body_choices:
        state.body = rng.choice(body_choices)
    if attention_choices:
        state.attention = weighted_choice(rng, attention_choices)
    if response_choices:
        state.responsiveness = weighted_choice(rng, response_choices)
    if inertia_delta:
        state.inertia = shift_level(state.inertia, inertia_delta)


def ensure_consistency(state: EpisodeState, rng: random.Random) -> None:
    state.room = room_for_focus(rng, state.focus, state.room)
    state.sunbeam = sunbeam_for_time(rng, state.time) if state.sunbeam != "gone" or state.time in {"dawn", "morning", "afternoon"} else "gone"

    if state.dream != "awake":
        if state.body not in DREAM_BODIES and state.body not in REST_BODIES:
            state.body = rng.choice(DREAM_BODIES)
        if response_level(state.responsiveness) > RESPONSE_INDEX["selective"]:
            state.responsiveness = "selective"
        if state.last_action == "dream" and state.focus not in {
            "moon_moth",
            "ghost_mouse",
            "cloud_string",
            "starlight_bowl",
            "floating_box",
        }:
            state.focus = rng.choice(("moon_moth", "ghost_mouse", "cloud_string", "floating_box"))

    if state.last_action == "hide":
        state.focus = "vacuum"
        if state.body not in HIDE_BODIES:
            state.body = rng.choice(HIDE_BODIES)
        if response_level(state.responsiveness) > RESPONSE_INDEX["selective"]:
            state.responsiveness = "selective"

    if state.last_action == "seek_affection":
        state.focus = "human"
        if state.body not in AFFECTION_BODIES:
            state.body = rng.choice(AFFECTION_BODIES)
        if response_level(state.responsiveness) < RESPONSE_INDEX["present"]:
            state.responsiveness = "present"

    if state.last_action in {"play", "make_mischief"} and level(state.energy) >= 1:
        if state.body in REST_BODIES:
            state.body = rng.choice(PLAY_BODIES if state.last_action == "play" else MISCHIEF_BODIES)

    if state.time == "afternoon" and state.last_action not in {"play", "make_mischief"} and level(state.energy) <= 1:
        if state.body not in REST_BODIES:
            state.body = rng.choice(REST_BODIES)


def apply_circadian_context(state: EpisodeState, rng: random.Random, *, initial: bool = False) -> None:
    state.sunbeam = sunbeam_for_time(rng, state.time)

    if state.time == "dawn":
        if initial or rng.random() < 0.55:
            state.hunger = shift_level(state.hunger, 1)
        if state.last_action in {"rest", "dream"} and rng.random() < 0.35:
            state.energy = shift_level(state.energy, 1)
        if initial or rng.random() < 0.3:
            state.mischief = shift_level(state.mischief, 1)
    elif state.time == "morning":
        if initial or rng.random() < 0.4:
            state.hunger = shift_level(state.hunger, 1)
        if response_level(state.responsiveness) < RESPONSE_INDEX["present"] and rng.random() < 0.3:
            state.responsiveness = shift_response(state.responsiveness, 1)
    elif state.time == "afternoon":
        if initial or rng.random() < 0.5:
            state.energy = shift_level(state.energy, -1)
        if state.dream == "awake" and level(state.energy) <= 1 and rng.random() < 0.25:
            state.dream = "drifting"
    elif state.time == "evening":
        if initial or rng.random() < 0.45:
            state.hunger = shift_level(state.hunger, 1)
        if rng.random() < 0.25:
            state.trust = shift_level(state.trust, 1)
    elif state.time == "night":
        state.sunbeam = "gone"
        if initial or rng.random() < 0.45:
            state.mischief = shift_level(state.mischief, 1)
        if response_level(state.responsiveness) > RESPONSE_INDEX["selective"] and rng.random() < 0.45:
            state.responsiveness = shift_response(state.responsiveness, -1)
        if state.dream == "awake" and level(state.energy) <= 1 and rng.random() < 0.35:
            state.dream = weighted_choice(rng, [("drifting", 3), ("deep", 2)])

    ensure_consistency(state, rng)


def dominant_drive(state: EpisodeState, action: str) -> str:
    drives = {
        "hunger": level(state.hunger) + (3 if action == "seek_food" else 0),
        "rest": (3 - level(state.energy)) + (3 if action in {"rest", "dream"} else 0),
        "bond": level(state.trust) + (3 if action == "seek_affection" else 0),
        "mischief": level(state.mischief) + (3 if action in {"play", "make_mischief"} else 0),
        "watch": attention_level(state.attention) + (level(state.inertia) // 2) + (4 if action in {"observe", "ignore"} else 0),
    }
    return max(drives.items(), key=lambda item: (item[1], item[0]))[0]


def derive_mood(state: EpisodeState) -> str:
    if state.dream != "awake":
        return "SLEEPY"
    if state.focus == "vacuum" or state.body in {"under_bed", "flat_ears"}:
        return "GRUMPY"
    if level(state.hunger) >= 2 and state.bowl != "full" and state.time in {"dawn", "morning", "evening"}:
        return "HUNGRY"
    if level(state.energy) <= 1 and (state.body in REST_BODIES or state.sunbeam != "gone" or state.time == "afternoon"):
        return "SLEEPY"
    if level(state.mischief) >= 2 or state.focus in {"toy", "keyboard", "glass", "plant", "box", "counter", "bird"}:
        return "PLAYFUL"
    if response_level(state.responsiveness) <= RESPONSE_INDEX["selective"] and level(state.trust) <= 1:
        return "GRUMPY"
    return "PLAYFUL"


def random_state(rng: random.Random) -> EpisodeState:
    state = EpisodeState(
        hunger=weighted_choice(rng, [("low", 1), ("mid", 3), ("high", 3), ("peak", 1)]),
        energy=weighted_choice(rng, [("low", 1), ("mid", 3), ("high", 3), ("peak", 1)]),
        trust=weighted_choice(rng, [("low", 1), ("mid", 3), ("high", 2), ("peak", 1)]),
        mischief=weighted_choice(rng, [("low", 1), ("mid", 3), ("high", 3), ("peak", 1)]),
        inertia=weighted_choice(rng, [("low", 1), ("mid", 3), ("high", 3), ("peak", 1)]),
        time=weighted_choice(rng, [("dawn", 1), ("morning", 3), ("afternoon", 3), ("evening", 2), ("night", 2)]),
        room=rng.choice(ROOMS),
        bowl=rng.choice(("empty", "half", "full")),
        toy=rng.choice(("basket", "hallway", "under_couch", "sofa", "desk")),
        vacuum=rng.choice(("closet", "closet", "hallway")),
        sunbeam=rng.choice(("window", "sofa", "gone")),
        box=rng.choice(("hallway", "sofa", "desk", "claimed")),
        dream=weighted_choice(rng, [("awake", 6), ("drifting", 2), ("deep", 1)]),
        focus=weighted_choice(rng, FOCUS_WEIGHTS),
        attention=weighted_choice(rng, [("loose", 1), ("curious", 3), ("tracking", 3), ("locked", 1)]),
        responsiveness=weighted_choice(rng, [("ignoring", 1), ("selective", 3), ("present", 3), ("eager", 1)]),
        body=rng.choice(("loaf", "perch", "crouch", "sprawl", "desk_sphinx")),
        memory=rng.choice(("quiet_house", "footsteps", "treat_rustle", "toy_roll", "window_motion")),
        last_action="observe",
    )
    apply_circadian_context(state, rng, initial=True)
    ensure_consistency(state, rng)
    return state


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
    apply_behavior(
        state,
        rng,
        body_choices=FOOD_BODIES,
        attention_choices=[("tracking", 2), ("locked", 4)],
        response_choices=[("selective", 1), ("present", 2), ("eager", 4)],
        inertia_delta=1,
    )
    state.last_action = "seek_food"
    if state.time in {"dawn", "morning"}:
        prompt = rng.choice(("breakfast time kitty", "i opened your breakfast can", "morning snack for you"))
    elif state.time == "evening":
        prompt = rng.choice(("dinner time kitty", "your evening bowl is ready", "want your dinner scoop"))
    else:
        prompt = rng.choice(("i opened a tuna can", "your bowl is ready", "want a treat", "do you hear the food scoop"))
    mood = derive_mood(state)
    return Turn(
        prompt=prompt,
        cue=rng.choice(FOOD_CUES),
        action="seek_food",
        plan=rng.choice(("trot>figure_eight>stare", "chirp>escort>eat", "tail_up>check_bowl>munch")),
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
    apply_behavior(
        state,
        rng,
        body_choices=PLAY_BODIES,
        attention_choices=[("tracking", 2), ("locked", 4)],
        response_choices=[("selective", 3), ("present", 2)],
        inertia_delta=1,
    )
    state.last_action = "play"
    if state.time in {"dawn", "night"}:
        prompt = rng.choice(("midnight zoomies time", "did you hear that toy at dawn", "want to chase something fast"))
    else:
        prompt = rng.choice(("want the feather wand", "i found your toy mouse", "chase the laser dot", "play with this string"))
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
    apply_behavior(
        state,
        rng,
        body_choices=AFFECTION_BODIES,
        attention_choices=[("curious", 1), ("tracking", 3), ("locked", 2)],
        response_choices=[("present", 2), ("eager", 4)],
        inertia_delta=1,
    )
    state.last_action = "seek_affection"
    prompt = rng.choice(("come cuddle on my lap", "you are such a sweet cat", "want gentle pets", "good kitty"))
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
    apply_behavior(
        state,
        rng,
        body_choices=REST_BODIES,
        attention_choices=[("loose", 2), ("curious", 3), ("tracking", 1)],
        response_choices=[("ignoring", 2), ("selective", 4), ("present", 1)],
        inertia_delta=1,
    )
    state.last_action = "rest"
    if state.time == "afternoon":
        prompt = rng.choice(("afternoon sunbeam is waiting", "want your afternoon nap", "cozy patch by the window"))
    elif state.time == "night":
        prompt = rng.choice(("bedtime loaf", "night blanket is warm", "ready for a sleepy curl"))
    else:
        prompt = rng.choice(("the blanket is warm", "want to nap by the window", "cozy bed is ready", "you look sleepy"))
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
    apply_behavior(
        state,
        rng,
        body_choices=HIDE_BODIES,
        attention_choices=[("tracking", 1), ("locked", 5)],
        response_choices=[("ignoring", 3), ("selective", 3)],
        inertia_delta=1,
    )
    state.last_action = "hide"
    prompt = rng.choice(("the vacuum is out", "a stranger is here", "time for a bath", "that noise is scary"))
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
    state.focus = focus
    state.room = room_for_focus(rng, focus, state.room)
    state.memory = rng.choice(("forbidden_edge", "tiny_rebellion", "silent_target", "chaos_idea"))
    state.dream = "awake"
    if focus == "box":
        state.box = "claimed"
    state.mischief = shift_level(state.mischief, 1)
    state.energy = shift_level(state.energy, -1)
    apply_behavior(
        state,
        rng,
        body_choices=MISCHIEF_BODIES,
        attention_choices=[("tracking", 1), ("locked", 5)],
        response_choices=[("ignoring", 2), ("selective", 4)],
        inertia_delta=1,
    )
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
    apply_behavior(
        state,
        rng,
        body_choices=DREAM_BODIES,
        attention_choices=[("curious", 1), ("tracking", 2), ("locked", 2)],
        response_choices=[("ignoring", 4), ("selective", 2)],
        inertia_delta=1,
    )
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


def observe_turn(state: EpisodeState, rng: random.Random) -> Turn:
    focus = environment_focus(state, rng)
    state.focus = focus
    state.room = room_for_focus(rng, focus, state.room)
    state.memory = rng.choice(("tiny_motion", "far_rustle", "window_spark", "still_target"))
    state.dream = "awake"
    apply_behavior(
        state,
        rng,
        body_choices=OBSERVE_BODIES,
        attention_choices=[("curious", 1), ("tracking", 3), ("locked", 3)],
        response_choices=[("selective", 3), ("present", 2)],
        inertia_delta=1,
    )
    state.last_action = "observe"
    prompt = rng.choice(("what are you staring at", "see something over there", "still watching that", "what did you hear"))
    mood = derive_mood(state)
    return Turn(
        prompt=prompt,
        cue=rng.choice(OBSERVE_CUES) if focus in {"bird", "hall_noise"} else cue_for_focus(focus),
        action="observe",
        plan=rng.choice(("perch>ear_flick>stare", "listen>track>still_tail", "watch>blink>ear_turn")),
        mood=mood,
    )


def ignore_turn(state: EpisodeState, rng: random.Random) -> Turn:
    if state.focus == "human":
        state.focus = environment_focus(state, rng)
    state.room = room_for_focus(rng, state.focus, state.room)
    state.memory = rng.choice(("heard_you", "not_now", "busy_watching", "tiny_refusal"))
    state.dream = "awake" if state.dream == "awake" else state.dream
    apply_behavior(
        state,
        rng,
        body_choices=(state.body,) if state.body else OBSERVE_BODIES,
        attention_choices=[("tracking", 2), ("locked", 4)],
        response_choices=[("ignoring", 4), ("selective", 3)],
        inertia_delta=1,
    )
    state.last_action = "ignore"
    prompt = rng.choice(REDIRECT_PROMPTS)
    mood = derive_mood(state)
    return Turn(
        prompt=prompt,
        cue=rng.choice(IGNORE_CUES),
        action="ignore",
        plan=rng.choice(("ear_flick>look_away>stay_put", "tail_flick>keep_watch>tiny_mrrp", "slowblink>turn_ear>resume_stare")),
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
        "observe",
        "ignore",
    } else "observe"
    state.inertia = shift_level(state.inertia, 1)

    if action == "seek_food":
        plan = rng.choice(("sniff>check_bowl>stare", "listen>kitchen_trot>meow", "circle>pause>munch"))
        state.room = "kitchen"
        state.focus = "bowl"
        apply_behavior(
            state,
            rng,
            body_choices=FOOD_BODIES,
            attention_choices=[("tracking", 2), ("locked", 3)],
            response_choices=[("selective", 1), ("present", 2), ("eager", 3)],
        )
    elif action == "play":
        plan = rng.choice(("listen>paw_probe>hook", "peek>spring>bat", "crouch>dart>grab"))
        state.focus = "toy"
        state.room = room_for_focus(rng, "toy", state.room)
        apply_behavior(
            state,
            rng,
            body_choices=PLAY_BODIES,
            attention_choices=[("tracking", 2), ("locked", 4)],
            response_choices=[("selective", 3), ("present", 2)],
        )
    elif action == "seek_affection":
        plan = rng.choice(("slowblink>step_close>lean", "chirp>lap_hop>settle", "purr>knead>curl"))
        state.focus = "human"
        apply_behavior(
            state,
            rng,
            body_choices=AFFECTION_BODIES,
            attention_choices=[("tracking", 3), ("locked", 2)],
            response_choices=[("present", 2), ("eager", 4)],
        )
    elif action == "rest":
        plan = rng.choice(("blink>stretch>loaf", "yawn>curl>purr", "loaf>tuck_paws>doze"))
        apply_behavior(
            state,
            rng,
            body_choices=REST_BODIES,
            attention_choices=[("loose", 2), ("curious", 3)],
            response_choices=[("ignoring", 2), ("selective", 4), ("present", 1)],
        )
    elif action == "dream":
        plan = rng.choice(("doze>moon_pounce>twitch", "float>tail_flick>purr", "snore>paw_kick>slowblink"))
        state.focus = rng.choice(("moon_moth", "ghost_mouse", "cloud_string", "floating_box"))
        state.dream = weighted_choice(rng, [("drifting", 2), ("deep", 4)])
        apply_behavior(
            state,
            rng,
            body_choices=DREAM_BODIES,
            attention_choices=[("tracking", 2), ("locked", 2)],
            response_choices=[("ignoring", 4), ("selective", 2)],
        )
    elif action == "hide":
        plan = rng.choice(("listen>stay_small>peek", "crouch>pause>retreat", "freeze>peek>retreat"))
        state.focus = "vacuum"
        apply_behavior(
            state,
            rng,
            body_choices=HIDE_BODIES,
            attention_choices=[("tracking", 1), ("locked", 5)],
            response_choices=[("ignoring", 3), ("selective", 3)],
        )
    elif action == "make_mischief":
        plan = rng.choice(("peek>paw_test>bolt", "stare>tiny_swat>flee", "inspect>tap>scatter"))
        if state.focus not in {"keyboard", "glass", "plant", "box", "counter"}:
            state.focus = rng.choice(("keyboard", "glass", "plant", "box", "counter"))
        state.room = room_for_focus(rng, state.focus, state.room)
        apply_behavior(
            state,
            rng,
            body_choices=MISCHIEF_BODIES,
            attention_choices=[("tracking", 2), ("locked", 4)],
            response_choices=[("ignoring", 2), ("selective", 4)],
        )
    elif action == "ignore":
        plan = rng.choice(("ear_flick>look_away>stay_put", "tail_flick>keep_watch>tiny_mrrp", "slowblink>resume_stare>stay_put"))
        if state.focus == "human":
            state.focus = environment_focus(state, rng)
        apply_behavior(
            state,
            rng,
            body_choices=(state.body,) if state.body else OBSERVE_BODIES,
            attention_choices=[("tracking", 2), ("locked", 4)],
            response_choices=[("ignoring", 4), ("selective", 2)],
        )
    else:
        action = "observe"
        plan = rng.choice(("perch>ear_flick>stare", "listen>track>still_tail", "watch>blink>ear_turn"))
        state.focus = environment_focus(state, rng)
        state.room = room_for_focus(rng, state.focus, state.room)
        apply_behavior(
            state,
            rng,
            body_choices=OBSERVE_BODIES,
            attention_choices=[("curious", 1), ("tracking", 3), ("locked", 2)],
            response_choices=[("selective", 3), ("present", 2)],
        )

    state.memory = f"still_{state.focus}"
    state.last_action = action
    mood = derive_mood(state)
    return Turn(prompt=prompt, cue=cue_for_focus(state.focus), action=action, plan=plan, mood=mood)


def choose_turn(state: EpisodeState, rng: random.Random) -> Turn:
    attention_bias = attention_level(state.attention)
    response_bias = response_level(state.responsiveness)
    inertia_bias = level(state.inertia)
    options = [
        ("food", 0.9 + level(state.hunger) + (1.5 if state.bowl == "empty" else 0.0) + (1.2 if state.time in {"dawn", "morning", "evening"} else 0.0)),
        ("play", 0.7 + level(state.energy) + level(state.mischief) + (1.2 if state.time in {"dawn", "night"} else 0.2 if state.time == "evening" else 0.0)),
        ("cuddle", 0.5 + level(state.trust) + (0.8 if state.time == "evening" else 0.3 if state.time == "night" else 0.0) + (0.4 if response_bias >= RESPONSE_INDEX["present"] else 0.0)),
        ("rest", 0.9 + (3 - level(state.energy)) + (0.9 if state.time == "afternoon" else 0.4 if state.time == "night" else 0.0) + (0.8 if state.sunbeam != "gone" else 0.0)),
        ("dream", 0.4 + (2.5 if state.dream != "awake" else 0.0) + (1.8 if level(state.energy) <= 1 else 0.0) + (0.8 if state.time == "night" else 0.0)),
        ("scare", 0.3 + (1.8 if state.vacuum == "hallway" else 0.0)),
        ("mischief", 0.8 + level(state.mischief) + (1.1 if state.time in {"dawn", "night"} else 0.3 if state.time == "evening" else 0.0) + (0.6 if state.room == "desk" else 0.0)),
        ("follow_up", 0.5 + inertia_bias + (1.1 if attention_bias >= ATTENTION_INDEX["tracking"] else 0.0)),
        ("observe", 0.5 + attention_bias + (0.9 if state.focus != "human" else 0.0) + (0.8 if state.time in {"morning", "night"} else 0.0)),
        ("ignore", 0.2 + (1.3 if response_bias <= RESPONSE_INDEX["selective"] else 0.0) + (0.9 if state.focus != "human" and attention_bias >= ATTENTION_INDEX["tracking"] else 0.0)),
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
    if choice == "observe":
        return observe_turn(state, rng)
    if choice == "ignore":
        return ignore_turn(state, rng)
    return follow_up_turn(state, rng)


def drift_state(state: EpisodeState, rng: random.Random) -> None:
    if rng.random() < 0.55:
        state.hunger = shift_level(state.hunger, 1)
    if state.last_action not in {"rest", "seek_affection", "dream"} and rng.random() < 0.45:
        state.energy = shift_level(state.energy, -1)
    if state.last_action in {"rest", "dream"} and rng.random() < 0.3:
        state.energy = shift_level(state.energy, 1)

    if state.bowl == "full" and rng.random() < 0.4:
        state.bowl = "half"
    elif state.bowl == "half" and rng.random() < 0.35:
        state.bowl = "empty"

    if state.vacuum == "hallway" and rng.random() < 0.35:
        state.vacuum = "closet"
    if state.focus == "toy" and rng.random() < 0.3:
        state.toy = rng.choice(("hallway", "under_couch", "sofa", "desk"))
    if state.focus == "box" and state.box != "claimed" and rng.random() < 0.2:
        state.box = "claimed"

    if state.attention == "locked" and rng.random() < 0.35:
        state.attention = "tracking"
    elif state.attention == "tracking" and rng.random() < 0.35:
        state.attention = "curious"
    elif state.attention == "curious" and rng.random() < 0.25:
        state.attention = "loose"

    if state.responsiveness == "eager" and rng.random() < 0.3:
        state.responsiveness = "present"
    elif state.responsiveness == "present" and rng.random() < 0.25:
        state.responsiveness = "selective"
    elif state.responsiveness == "selective" and rng.random() < 0.15:
        state.responsiveness = "ignoring"

    if state.last_action in {"play", "make_mischief", "hide", "observe", "ignore", "rest"}:
        if rng.random() < 0.45:
            state.inertia = shift_level(state.inertia, -1)
    elif rng.random() < 0.65:
        state.inertia = shift_level(state.inertia, -1)

    if state.last_action in {"ignore", "observe"} and state.focus != "human" and rng.random() < 0.25:
        state.memory = f"still_{state.focus}"

    if state.dream == "deep" and rng.random() < 0.32:
        state.dream = "drifting"
    elif state.dream == "drifting" and rng.random() < 0.4:
        state.dream = "awake"

    steps = 0
    if rng.random() < 0.24:
        steps = 1
    if state.last_action in {"rest", "dream"} and rng.random() < 0.15:
        steps += 1
    if steps:
        state.time = advance_time(state.time, steps)
        apply_circadian_context(state, rng)

    ensure_consistency(state, rng)


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
        ("k", state.inertia),
        ("c", state.time),
        ("r", state.room),
        ("z", state.dream),
        ("f", state.focus),
        ("i", state.attention),
        ("u", state.responsiveness),
        ("o", state.body),
        ("n", state.memory),
    ]

    if turn.action == "seek_food" or state.focus == "bowl":
        fields.append(("b", state.bowl))
    if turn.action == "play" or state.focus == "toy":
        fields.append(("y", state.toy))
    if turn.action == "hide" or state.focus == "vacuum":
        fields.append(("v", state.vacuum))
    if turn.action in {"rest", "dream"} or state.focus in {"sunbeam", "blanket"}:
        fields.append(("s", state.sunbeam))
    if turn.action == "make_mischief" or state.focus == "box":
        fields.append(("x", state.box))

    return fields


def reply_span(turn: Turn, lo: int, hi: int) -> tuple[int, int]:
    if turn.action == "ignore":
        return 1, min(2, hi)
    if turn.action == "observe":
        return 1, min(max(2, lo + 1), hi)
    if turn.action in {"rest", "dream"}:
        return 1, min(max(3, lo + 1), hi)
    return lo, hi


def reasoning_line(state: EpisodeState, rng: random.Random, lo: int, hi: int) -> str:
    turn = choose_turn(state, rng)
    think = " ".join(f"{key}={value}" for key, value in think_fields(state, turn))
    reply_lo, reply_hi = reply_span(turn, lo, hi)
    reply = utterance(rng, turn.mood, reply_lo, reply_hi)
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
