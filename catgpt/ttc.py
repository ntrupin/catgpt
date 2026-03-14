from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

import torch

MOODS = ("HUNGRY", "SLEEPY", "GRUMPY", "PLAYFUL")
REQUIRED_REASONING_CHARS = set("<>/USERTHINK")
MOOD_PRIOR = {
    "HUNGRY": {"HUNGRY": 0.6, "PLAYFUL": 0.2, "SLEEPY": 0.1, "GRUMPY": 0.1},
    "SLEEPY": {"SLEEPY": 0.62, "PLAYFUL": 0.22, "HUNGRY": 0.08, "GRUMPY": 0.08},
    "GRUMPY": {"GRUMPY": 0.58, "PLAYFUL": 0.2, "HUNGRY": 0.12, "SLEEPY": 0.1},
    "PLAYFUL": {"PLAYFUL": 0.58, "HUNGRY": 0.16, "SLEEPY": 0.16, "GRUMPY": 0.1},
}
PARSE_RE = re.compile(
    r"(?s)\s*(?P<think>.*?)</THINK>\s*<MOOD=(?P<mood>HUNGRY|SLEEPY|GRUMPY|PLAYFUL)>\s*(?P<reply>.*)"
)
MOOD_ONLY_RE = re.compile(r"(?s).*?<MOOD=(?P<mood>HUNGRY|SLEEPY|GRUMPY|PLAYFUL)>\s*(?P<reply>.*)")
ACTION_RE = re.compile(r"\b(?:action|a)=([a-z_]+)\b")
FIELD_RE = re.compile(r"\b([a-z_]+)=([A-Za-z0-9_>]+)")
THINK_TAG_RE = re.compile(r"<THINK>(.*?)</THINK>", re.DOTALL)
CAT_SOUND_RE = re.compile(r"meow|mew|mrr|mrrow|prr|hiss|nya|brr", re.IGNORECASE)
TAG_CUT_RE = re.compile(r"\s*<(?:USER|THINK|MOOD=|/USER|/THINK).*$", re.IGNORECASE)
LAST_MOOD_RE = re.compile(r"<MOOD=(HUNGRY|SLEEPY|GRUMPY|PLAYFUL)>")
WORD_RE = re.compile(r"[a-z_']+")
LEVELS = {"low": 0, "mid": 1, "high": 2, "peak": 3}
TIMES = ("dawn", "morning", "afternoon", "evening", "night")
TIME_INDEX = {name: i for i, name in enumerate(TIMES)}
ATTENTION_LEVELS = {"loose": 0, "curious": 1, "tracking": 2, "locked": 3}
RESPONSE_LEVELS = {"ignoring": 0, "selective": 1, "present": 2, "eager": 3}
DRIVE_KEYS = ("hunger", "energy", "trust", "mischief", "inertia")
WORLD_KEYS = (
    "time",
    "room",
    "dream",
    "focus",
    "memory",
    "body",
    "attention",
    "responsiveness",
    "bowl",
    "toy",
    "vacuum",
    "sunbeam",
    "box",
)
FIELD_ALIASES = {
    "q": "cue",
    "d": "drive",
    "a": "action",
    "p": "plan",
    "h": "hunger",
    "e": "energy",
    "t": "trust",
    "m": "mischief",
    "k": "inertia",
    "c": "time",
    "r": "room",
    "z": "dream",
    "f": "focus",
    "i": "attention",
    "u": "responsiveness",
    "o": "body",
    "n": "memory",
    "b": "bowl",
    "y": "toy",
    "v": "vacuum",
    "s": "sunbeam",
    "x": "box",
}
MOTION_STEPS = {
    "approach",
    "backstep",
    "bolt",
    "circle",
    "dart",
    "escort",
    "figure_eight",
    "launch",
    "lap_hop",
    "retreat",
    "spring",
    "trot",
    "vanish",
    "step_close",
    "kitchen_trot",
}
MISCHIEF_STEPS = {"creep", "paw_test", "push", "swat", "bolt", "launch", "scatter"}
DREAM_STEPS = {"doze", "drift", "moon_pounce", "cloud_chase", "twitch", "tail_flick", "snore", "paw_kick", "float"}
OBSERVE_STEPS = {"perch", "ear_flick", "stare", "listen", "track", "watch", "blink", "ear_turn", "still_tail"}
IGNORE_STEPS = {"ear_flick", "look_away", "stay_put", "keep_watch", "tail_flick", "turn_ear", "resume_stare"}
DREAM_HINTS = {"dream", "dreaming", "sleep", "sleeping", "nap", "asleep", "twitch", "twitching"}
OBJECT_HINTS = {
    "bowl": {"food", "treat", "dinner", "breakfast", "kibble", "bowl", "tuna", "eat"},
    "toy": {"toy", "laser", "string", "mouse", "play", "feather", "zoomies", "chase"},
    "vacuum": {"vacuum", "bath", "stranger", "noise", "scary", "door", "loud"},
    "sunbeam": {"nap", "sleep", "rest", "blanket", "bed", "sunbeam", "sleepy", "cozy"},
    "box": {"box", "bite", "cardboard"},
    "bird": {"bird", "window", "outside", "pigeon", "sparrow"},
    "hall_noise": {"hall", "hallway", "noise", "rustle", "sound", "outside"},
    "keyboard": {"keyboard", "desk"},
    "glass": {"glass", "desk"},
    "plant": {"plant", "leaves"},
    "counter": {"counter", "kitchen"},
    "human": {"lap", "pet", "pets", "cuddle", "gentle", "sweet", "good", "friend"},
}
GENERIC_FOLLOW_UPS = (
    "where is it now",
    "what are you staring at",
    "still thinking about that",
    "what did you do over there",
    "is it still there",
)
REST_BODIES = {"loaf", "curl", "sprawl", "sun_loaf", "lap_loaf"}
DREAM_BODIES = {"curl", "sprawl", "twitch"}
HIDE_BODIES = {"crouch", "under_bed", "flat_ears"}
PLAY_BODIES = {"crouch", "pounce_ready", "perch"}
MISCHIEF_BODIES = {"prowl", "perch", "tail_wrap"}
FOOD_BODIES = {"tail_up", "kitchen_orbit", "counter_peek"}
AFFECTION_BODIES = {"lap_loaf", "lean", "knead"}
OBSERVE_BODIES = {"perch", "desk_sphinx", "loaf"}


@dataclass
class Candidate:
    raw: str
    think: str
    mood: str | None
    reply: str
    action: str | None
    plan: tuple[str, ...]
    state: dict[str, str]
    avg_logprob: float
    score: float


@dataclass
class Decision:
    mood: str
    reply: str
    think: str
    action: str
    plan: tuple[str, ...]
    state: dict[str, str]
    history: str
    used_samples: int
    consensus: int
    gallery: list["Rollout"]


@dataclass
class Rollout:
    mood: str
    reply: str
    action: str
    plan: tuple[str, ...]
    state: dict[str, str]
    think: str
    score: float
    avg_logprob: float
    agreement: int
    winner: bool


def sanitize(text: str, tokenizer) -> str:
    if not text:
        return ""
    fallback = " " if " " in tokenizer.stoi else ""
    return "".join(ch if ch in tokenizer.stoi else fallback for ch in text)


def missing_reasoning_chars(tokenizer) -> list[str]:
    return sorted(ch for ch in REQUIRED_REASONING_CHARS if ch not in tokenizer.stoi)


def build_prompt(history: str, user_message: str, tokenizer) -> str:
    user = sanitize(user_message, tokenizer).strip()
    return f"{history}<USER>{user}</USER><THINK>"


def trim_history(history: str, tokenizer, block_size: int) -> str:
    lines = [line for line in history.splitlines() if line.strip()]
    if not lines:
        return ""
    while lines and len(tokenizer.encode("\n".join(lines) + "\n")) > max(16, block_size - 20):
        lines.pop(0)
    return ("\n".join(lines) + "\n") if lines else ""


def sample_continuation(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> tuple[str, float]:
    prompt_ids = tokenizer.encode(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    budget = max_new_tokens
    if prompt.endswith("<THINK>"):
        budget = max(max_new_tokens, 300)

    stop_token = tokenizer.stoi.get("\n")
    temp = max(temperature, 1e-5)
    logprob_sum = 0.0
    steps = 0

    with torch.no_grad():
        for _ in range(budget):
            logits, _ = model(idx[:, -model.config.block_size :])
            logits = logits[:, -1, :] / temp
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -torch.inf

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            token_prob = probs[0, next_id.item()].clamp_min(1e-12)
            logprob_sum += float(torch.log(token_prob).item())
            steps += 1

            idx = torch.cat((idx, next_id), dim=1)
            if stop_token is not None and int(next_id.item()) == int(stop_token):
                break

    text = tokenizer.decode(idx[0].tolist()[len(prompt_ids) :])
    return text, logprob_sum / max(1, steps)


def extract_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key, value in FIELD_RE.findall(text or ""):
        fields[FIELD_ALIASES.get(key, key)] = value
    return fields


def format_action(action: str | None) -> str | None:
    return action.replace("_", " ") if action else None


def split_plan(plan_text: str | None) -> tuple[str, ...]:
    if not plan_text:
        return ()
    return tuple(step for step in plan_text.split(">") if step)


def value_step(mapping: dict[str, int], prev: str, curr: str) -> int | None:
    if prev not in mapping or curr not in mapping:
        return None
    return mapping[curr] - mapping[prev]


def time_step(prev: str, curr: str) -> int | None:
    if prev not in TIME_INDEX or curr not in TIME_INDEX:
        return None
    return (TIME_INDEX[curr] - TIME_INDEX[prev]) % len(TIMES)


def room_matches_focus(room: str, focus: str) -> bool:
    if focus in {"bowl", "counter"}:
        return room == "kitchen"
    if focus in {"keyboard", "glass"}:
        return room == "desk"
    if focus in {"bird", "sunbeam"}:
        return room == "window"
    if focus in {"vacuum", "hall_noise"}:
        return room == "hallway"
    if focus == "human":
        return room in {"sofa", "bedroom"}
    if focus == "toy":
        return room in {"living_room", "hallway", "desk"}
    if focus == "plant":
        return room in {"living_room", "desk"}
    if focus == "box":
        return room in {"hallway", "sofa", "desk"}
    if focus in {"moon_moth", "ghost_mouse", "cloud_string", "starlight_bowl", "floating_box"}:
        return room in {"window", "sofa", "bedroom"}
    return True


def action_matches_focus(action: str | None, focus: str | None) -> bool:
    if not action or not focus:
        return False
    if action == "seek food":
        return focus == "bowl"
    if action == "play":
        return focus == "toy"
    if action == "seek affection":
        return focus == "human"
    if action == "hide":
        return focus == "vacuum"
    if action == "rest":
        return focus in {"sunbeam", "blanket"}
    if action == "dream":
        return focus in {"moon_moth", "ghost_mouse", "cloud_string", "starlight_bowl", "floating_box"}
    if action == "make mischief":
        return focus in {"keyboard", "glass", "plant", "box", "counter"}
    if action in {"observe", "ignore"}:
        return focus not in {None, "human"}
    return False


def body_matches_action(
    body: str | None,
    action: str | None,
    state: dict[str, str] | None = None,
    plan: tuple[str, ...] = (),
) -> bool:
    if not body or not action:
        return True
    state = state or {}
    if action == "seek food":
        return body in FOOD_BODIES
    if action == "play":
        return body in PLAY_BODIES
    if action == "seek affection":
        return body in AFFECTION_BODIES or (body in REST_BODIES and state.get("focus") == "human")
    if action == "rest":
        return body in REST_BODIES or body in DREAM_BODIES
    if action == "dream":
        return body in DREAM_BODIES or body in REST_BODIES
    if action == "hide":
        return body in HIDE_BODIES
    if action == "make mischief":
        return body in MISCHIEF_BODIES
    if action == "observe":
        return body in OBSERVE_BODIES or body in PLAY_BODIES
    if action == "ignore":
        return body in OBSERVE_BODIES or body in REST_BODIES or body in HIDE_BODIES
    if any(step in {"curl", "loaf", "doze"} for step in plan):
        return body in REST_BODIES
    return True


def mood_state_bonus(mood: str | None, state: dict[str, str]) -> float:
    if mood not in MOODS:
        return 0.0

    hunger = state.get("hunger")
    energy = state.get("energy")
    trust = state.get("trust")
    mischief = state.get("mischief")
    inertia = state.get("inertia")
    dream = state.get("dream")
    time_of_day = state.get("time")
    focus = state.get("focus")
    vacuum = state.get("vacuum")
    sunbeam = state.get("sunbeam")
    body = state.get("body")
    attention = state.get("attention")
    responsiveness = state.get("responsiveness")

    bonus = 0.0
    if mood == "HUNGRY" and hunger in {"high", "peak"}:
        bonus += 0.06
    if mood == "HUNGRY" and time_of_day in {"dawn", "morning", "evening"}:
        bonus += 0.02
    if mood == "SLEEPY" and energy in {"low", "mid"}:
        bonus += 0.05
    if mood == "SLEEPY" and dream in {"drifting", "deep"}:
        bonus += 0.06
    if mood == "SLEEPY" and sunbeam and sunbeam != "gone":
        bonus += 0.02
    if mood == "SLEEPY" and body in REST_BODIES | DREAM_BODIES:
        bonus += 0.03
    if mood == "GRUMPY" and (trust == "low" or vacuum == "hallway"):
        bonus += 0.06
    if mood == "GRUMPY" and body in HIDE_BODIES:
        bonus += 0.03
    if mood == "PLAYFUL" and (mischief in {"high", "peak"} or focus in {"toy", "keyboard", "glass", "plant", "box", "counter", "bird"}):
        bonus += 0.06
    if mood == "PLAYFUL" and time_of_day in {"dawn", "night"}:
        bonus += 0.02
    if attention in {"tracking", "locked"}:
        bonus += 0.01
    if responsiveness in {"ignoring", "selective"} and mood in {"SLEEPY", "GRUMPY"}:
        bonus += 0.02
    if inertia in {"high", "peak"} and focus:
        bonus += 0.01
    return bonus


def plan_quality_bonus(action: str | None, plan: tuple[str, ...]) -> float:
    if not plan:
        return 0.0

    bonus = 0.05
    if 2 <= len(plan) <= 4:
        bonus += 0.05
    if len(set(plan)) == len(plan):
        bonus += 0.03
    if any(step in MOTION_STEPS for step in plan):
        bonus += 0.02
    if action == "make mischief" and any(step in MISCHIEF_STEPS for step in plan):
        bonus += 0.05
    if action == "dream" and any(step in DREAM_STEPS for step in plan):
        bonus += 0.05
    if action == "hide" and any(step in {"freeze", "crouch", "retreat", "vanish", "backstep"} for step in plan):
        bonus += 0.03
    if action == "seek food" and any(step in {"trot", "escort", "munch", "check_bowl"} for step in plan):
        bonus += 0.03
    if action == "rest" and any(step in {"loaf", "curl", "doze", "purr"} for step in plan):
        bonus += 0.03
    if action == "observe" and any(step in OBSERVE_STEPS for step in plan):
        bonus += 0.04
    if action == "ignore" and any(step in IGNORE_STEPS for step in plan):
        bonus += 0.04
    return bonus


def behavior_fit_bonus(action: str | None, state: dict[str, str], plan: tuple[str, ...]) -> float:
    if not action:
        return 0.0

    bonus = 0.0
    body = state.get("body")
    time_of_day = state.get("time")
    focus = state.get("focus")
    attention = state.get("attention")
    responsiveness = state.get("responsiveness")
    inertia = state.get("inertia")

    if body_matches_action(body, action, state, plan):
        bonus += 0.05

    if action in {"observe", "ignore"} and focus and focus != "human" and attention in {"tracking", "locked"}:
        bonus += 0.05
    if action == "ignore" and responsiveness in {"ignoring", "selective"}:
        bonus += 0.05
    if action == "observe" and responsiveness in {"selective", "present"}:
        bonus += 0.03
    if action == "seek affection" and responsiveness in {"present", "eager"}:
        bonus += 0.04
    if action in {"play", "make mischief"} and time_of_day in {"dawn", "night"}:
        bonus += 0.02
    if action in {"rest", "dream"} and time_of_day in {"afternoon", "night"}:
        bonus += 0.02
    if action == "seek food" and time_of_day in {"dawn", "morning", "evening"}:
        bonus += 0.02
    if inertia in {"high", "peak"} and action in {"play", "make mischief", "rest", "hide", "observe", "ignore", "dream"}:
        bonus += 0.02
    return bonus


def parse_candidate(raw: str, avg_logprob: float) -> Candidate:
    line = raw.split("\n", 1)[0]
    m = PARSE_RE.match(line)
    if not m:
        mm = MOOD_ONLY_RE.match(line)
        if not mm:
            return Candidate(
                raw=line,
                think="",
                mood=None,
                reply="",
                action=None,
                plan=(),
                state={},
                avg_logprob=avg_logprob,
                score=avg_logprob - 0.4,
            )
        mood = mm.group("mood")
        reply = TAG_CUT_RE.sub("", mm.group("reply")).strip()
        score = avg_logprob + (0.12 if reply else -0.12) + (0.1 if CAT_SOUND_RE.search(reply) else 0.0)
        return Candidate(
            raw=line,
            think="",
            mood=mood,
            reply=reply,
            action=None,
            plan=(),
            state={},
            avg_logprob=avg_logprob,
            score=score,
        )

    think = m.group("think").strip()
    mood = m.group("mood")
    reply = TAG_CUT_RE.sub("", m.group("reply")).strip()
    fields = extract_fields(think)
    action = format_action(fields.get("action"))
    if action is None:
        action_match = ACTION_RE.search(think)
        action = format_action(action_match.group(1)) if action_match else None
    plan = split_plan(fields.get("plan"))
    state = {key: value for key, value in fields.items() if key in DRIVE_KEYS or key in WORLD_KEYS}

    score = avg_logprob
    if think:
        score += 0.07
    if reply:
        score += 0.1
    if CAT_SOUND_RE.search(reply):
        score += 0.14
    if mood in MOODS:
        score += 0.1
    if state:
        score += min(0.16, 0.02 * len(state))
    score += plan_quality_bonus(action, plan)
    score += mood_state_bonus(mood, state)
    score += behavior_fit_bonus(action, state, plan)

    return Candidate(
        raw=line,
        think=think,
        mood=mood,
        reply=reply,
        action=action,
        plan=plan,
        state=state,
        avg_logprob=avg_logprob,
        score=score,
    )


def last_mood_from_history(history: str) -> str | None:
    matches = LAST_MOOD_RE.findall(history)
    return matches[-1] if matches else None


def last_state_from_history(history: str) -> dict[str, str]:
    matches = THINK_TAG_RE.findall(history or "")
    if not matches:
        return {}
    return extract_fields(matches[-1])


def last_action_from_history(history: str) -> str | None:
    matches = THINK_TAG_RE.findall(history or "")
    if not matches:
        return None
    fields = extract_fields(matches[-1])
    action = format_action(fields.get("action"))
    if action:
        return action
    return None


def user_words(text: str) -> set[str]:
    return set(WORD_RE.findall(text.lower()))


def message_objects(words: set[str]) -> set[str]:
    matches = set()
    for name, hints in OBJECT_HINTS.items():
        if words & hints:
            matches.add(name)
    return matches


def is_generic_follow_up(text: str) -> bool:
    lowered = text.lower()
    if any(phrase in lowered for phrase in GENERIC_FOLLOW_UPS):
        return True
    words = user_words(lowered)
    return {"it", "there", "that"} & words and not message_objects(words)


def level_shift_is_reasonable(key: str, prev: str, curr: str, action: str | None, words: set[str]) -> bool:
    prev_i = LEVELS.get(prev)
    curr_i = LEVELS.get(curr)
    if prev_i is None or curr_i is None:
        return True
    diff = abs(curr_i - prev_i)
    if diff <= 1:
        return True
    if key == "hunger" and (action == "seek food" or words & OBJECT_HINTS["bowl"]):
        return curr_i <= prev_i
    if key == "energy" and action in {"rest", "seek affection"}:
        return curr_i >= prev_i
    if key == "energy" and action == "dream":
        return curr_i <= prev_i
    if key == "energy" and action in {"play", "make mischief"}:
        return curr_i <= prev_i
    if key == "trust" and (action == "seek affection" or words & OBJECT_HINTS["human"]):
        return curr_i >= prev_i
    if key == "trust" and (action == "hide" or words & OBJECT_HINTS["vacuum"]):
        return curr_i <= prev_i
    if key == "mischief" and action in {"play", "make mischief"}:
        return curr_i >= prev_i
    if key == "inertia" and action in {"play", "make mischief", "rest", "hide", "dream", "observe", "ignore"}:
        return curr_i >= prev_i
    return False


def time_shift_is_reasonable(prev: str, curr: str, action: str | None) -> bool:
    step = time_step(prev, curr)
    if step is None:
        return True
    if step in {0, 1}:
        return True
    if step == 2 and action in {"rest", "dream"}:
        return True
    return False


def attention_shift_is_reasonable(prev: str, curr: str, action: str | None) -> bool:
    step = value_step(ATTENTION_LEVELS, prev, curr)
    if step is None:
        return True
    if abs(step) <= 1:
        return True
    if action in {"play", "make mischief", "hide", "seek food", "observe", "ignore"}:
        return step >= 0
    if action in {"rest", "dream", "seek affection"}:
        return step <= 0
    return False


def response_shift_is_reasonable(prev: str, curr: str, action: str | None) -> bool:
    step = value_step(RESPONSE_LEVELS, prev, curr)
    if step is None:
        return True
    if abs(step) <= 1:
        return True
    if action in {"seek affection", "seek food"}:
        return step >= 0
    if action in {"ignore", "rest", "dream", "hide"}:
        return step <= 0
    return False


def world_shift_is_reasonable(
    key: str,
    prev: str,
    curr: str,
    candidate: Candidate,
    prev_state: dict[str, str],
    words: set[str],
) -> bool:
    action = candidate.action or infer_action(candidate.think, candidate.reply, candidate.mood or "PLAYFUL", candidate.state, candidate.plan)
    if prev == curr:
        return True
    if key == "time":
        return time_shift_is_reasonable(prev, curr, action)
    if key == "room":
        focus = candidate.state.get("focus")
        if focus and room_matches_focus(curr, focus):
            return True
        return bool(candidate.plan) and any(step in MOTION_STEPS for step in candidate.plan)
    if key == "dream":
        return action in {"rest", "dream"} or bool(words & DREAM_HINTS)
    if key == "focus":
        mentioned_objects = message_objects(words)
        prev_focus = prev_state.get("focus")
        if action_matches_focus(action, curr):
            return True
        if mentioned_objects and curr in mentioned_objects:
            return True
        if prev_focus and curr == prev_focus and prev_state.get("attention") in {"tracking", "locked"}:
            return True
        return False
    if key == "body":
        return body_matches_action(curr, action, candidate.state, candidate.plan)
    if key == "attention":
        return attention_shift_is_reasonable(prev, curr, action)
    if key == "responsiveness":
        return response_shift_is_reasonable(prev, curr, action)
    if key == "bowl":
        return action == "seek food" or bool(words & OBJECT_HINTS["bowl"])
    if key == "toy":
        return action in {"play", "make mischief"} or bool(words & OBJECT_HINTS["toy"])
    if key == "vacuum":
        return action == "hide" or bool(words & OBJECT_HINTS["vacuum"])
    if key == "sunbeam":
        return action in {"rest", "dream"} or bool(words & OBJECT_HINTS["sunbeam"]) or candidate.state.get("time") in {"dawn", "morning", "afternoon"}
    if key == "box":
        return action in {"make mischief", "rest"} or bool(words & OBJECT_HINTS["box"])
    if key == "memory":
        return True
    return False


def continuity_bonus(candidate: Candidate, prev_state: dict[str, str], user_message: str, prev_action: str | None) -> float:
    if not candidate.state:
        return 0.0

    words = user_words(user_message)
    mentioned_objects = message_objects(words)
    generic_follow_up = is_generic_follow_up(user_message)
    action = candidate.action or infer_action(candidate.think, candidate.reply, candidate.mood or "PLAYFUL", candidate.state, candidate.plan)
    bonus = 0.0

    for key in DRIVE_KEYS:
        prev = prev_state.get(key)
        curr = candidate.state.get(key)
        if not prev or not curr:
            continue
        if prev == curr:
            bonus += 0.02
        elif level_shift_is_reasonable(key, prev, curr, action, words):
            bonus += 0.005
        else:
            bonus -= 0.035

    for key in WORLD_KEYS:
        prev = prev_state.get(key)
        curr = candidate.state.get(key)
        if not prev or not curr:
            continue
        if prev == curr:
            bonus += 0.03
        elif world_shift_is_reasonable(key, prev, curr, candidate, prev_state, words):
            bonus += 0.006
        else:
            bonus -= 0.045

    prev_focus = prev_state.get("focus")
    curr_focus = candidate.state.get("focus")
    curr_memory = candidate.state.get("memory")
    if generic_follow_up:
        if prev_focus and (curr_focus == prev_focus or curr_memory == prev_focus or curr_memory == f"still_{prev_focus}"):
            bonus += 0.09
        elif prev_focus and curr_focus and curr_focus != prev_focus:
            bonus -= 0.06

    strong_lock = prev_state.get("inertia") in {"high", "peak"} or prev_state.get("attention") in {"tracking", "locked"}
    strong_redirect = bool(mentioned_objects and prev_focus and prev_focus not in mentioned_objects)
    if prev_action and strong_lock and not strong_redirect:
        if action == prev_action:
            bonus += 0.08
        else:
            bonus -= 0.05

    if prev_focus and curr_focus == prev_focus and strong_lock:
        bonus += 0.05
    elif prev_focus and curr_focus and curr_focus != prev_focus and strong_lock and not strong_redirect:
        bonus -= 0.04

    if prev_state.get("responsiveness") in {"ignoring", "selective"} and action == "ignore" and not mentioned_objects:
        bonus += 0.05
    if prev_state.get("responsiveness") == "eager" and action == "ignore" and mentioned_objects:
        bonus -= 0.03

    if words & DREAM_HINTS:
        if candidate.state.get("dream") in {"drifting", "deep"}:
            bonus += 0.05
        else:
            bonus -= 0.03
    if mentioned_objects:
        if curr_focus and curr_focus in mentioned_objects:
            bonus += 0.05
        elif curr_focus and curr_focus not in mentioned_objects and not generic_follow_up and not (strong_lock and curr_focus == prev_focus):
            bonus -= 0.03

    return bonus


def cluster_key(candidate: Candidate, prev_state: dict[str, str]) -> tuple[str, str, str]:
    action = candidate.action or infer_action(candidate.think, candidate.reply, candidate.mood or "PLAYFUL", candidate.state, candidate.plan)
    anchor = (
        candidate.state.get("focus")
        or candidate.state.get("room")
        or prev_state.get("focus")
        or prev_state.get("room", "")
    )
    return (candidate.mood or "PLAYFUL", action, anchor)


def select_candidate(
    candidates: list[Candidate],
    prev_mood: str | None,
    mood_inertia: float,
    prev_state: dict[str, str],
    prev_action: str | None,
    user_message: str,
) -> tuple[Candidate, int]:
    parsed = [c for c in candidates if c.mood is not None and c.reply]
    if not parsed:
        best = max(candidates, key=lambda c: c.score)
        return best, 1

    for c in parsed:
        c.score += continuity_bonus(c, prev_state=prev_state, user_message=user_message, prev_action=prev_action)

    if prev_mood in MOODS and mood_inertia > 0:
        weights = MOOD_PRIOR[prev_mood]
        for c in parsed:
            c.score += mood_inertia * weights.get(c.mood, 0.0)

    votes = Counter(cluster_key(c, prev_state) for c in parsed)
    weighted_votes = {key: float(count) for key, count in votes.items()}
    if prev_mood in MOODS and mood_inertia > 0:
        prior = MOOD_PRIOR[prev_mood]
        for key in weighted_votes:
            weighted_votes[key] += mood_inertia * prior.get(key[0], 0.0)

    cluster_best_scores = {
        key: max(c.score for c in parsed if cluster_key(c, prev_state) == key)
        for key in weighted_votes
    }
    cluster_avg_scores = {
        key: sum(c.score for c in parsed if cluster_key(c, prev_state) == key) / votes[key]
        for key in weighted_votes
    }
    top_key = max(
        weighted_votes,
        key=lambda key: (weighted_votes[key], cluster_best_scores[key], cluster_avg_scores[key], key),
    )
    finalists = [c for c in parsed if cluster_key(c, prev_state) == top_key]
    best = max(finalists, key=lambda c: c.score)
    return best, int(votes[top_key])


def infer_action(
    think: str,
    reply: str,
    mood: str,
    state: dict[str, str] | None = None,
    plan: tuple[str, ...] = (),
) -> str:
    m = ACTION_RE.search(think)
    if m:
        return m.group(1).replace("_", " ")

    state = state or {}
    if state.get("dream") in {"drifting", "deep"} or any(step in DREAM_STEPS for step in plan):
        return "dream"
    if any(step in IGNORE_STEPS for step in plan) and state.get("focus") != "human":
        return "ignore"
    if any(step in OBSERVE_STEPS for step in plan) and state.get("focus") not in {None, "human"}:
        return "observe"
    if plan and any(step in MISCHIEF_STEPS for step in plan):
        return "make mischief"
    if state.get("focus") == "vacuum" or state.get("body") in HIDE_BODIES:
        return "hide"
    if state.get("focus") == "human":
        return "seek affection"
    if state.get("focus") == "bowl" or mood == "HUNGRY":
        return "seek food"
    if state.get("focus") == "toy":
        return "play"
    if state.get("focus") in {"keyboard", "glass", "plant", "box", "counter"}:
        return "make mischief"
    if state.get("focus") not in {None, "human"} and state.get("attention") in {"tracking", "locked"}:
        if state.get("responsiveness") in {"ignoring", "selective"}:
            return "ignore"
        return "observe"
    if "hiss" in reply.lower() or (mood == "GRUMPY" and state.get("responsiveness") in {"ignoring", "selective"}):
        return "hide"
    if "prr" in reply.lower() or mood == "SLEEPY":
        return "rest"
    return "social play"


def fallback_reply(raw: str) -> str:
    words = re.findall(r"[A-Za-z!?\.]+", raw)
    cat_words = [w for w in words if CAT_SOUND_RE.search(w)]
    if cat_words:
        return " ".join(cat_words[:8])
    return "mrrp"


def build_rollout_gallery(candidates: list[Candidate], prev_state: dict[str, str], best: Candidate) -> list[Rollout]:
    if not candidates:
        return []

    votes = Counter(cluster_key(c, prev_state) for c in candidates)
    # Keep the gallery score-sorted so the TTC-selected sample does not always appear first.
    ranked = sorted(candidates, key=lambda c: (c.score, c.avg_logprob, c is best), reverse=True)
    gallery: list[Rollout] = []
    for c in ranked:
        action = c.action or infer_action(c.think, c.reply, c.mood or "PLAYFUL", c.state, c.plan)
        gallery.append(
            Rollout(
                mood=c.mood or "PLAYFUL",
                reply=c.reply or fallback_reply(c.raw),
                action=action,
                plan=c.plan,
                state=c.state,
                think=c.think,
                score=round(c.score, 3),
                avg_logprob=round(c.avg_logprob, 3),
                agreement=votes[cluster_key(c, prev_state)],
                winner=c is best,
            )
        )
    return gallery


def ttc_turn(
    model,
    tokenizer,
    device: str,
    history: str,
    user_message: str,
    rollouts: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    mood_inertia: float = 0.35,
) -> Decision:
    prompt = build_prompt(history, user_message, tokenizer)
    samples = max(1, int(rollouts))

    candidates: list[Candidate] = []
    for _ in range(samples):
        raw, avg_lp = sample_continuation(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        candidates.append(parse_candidate(raw, avg_lp))

    prev_mood = last_mood_from_history(history)
    prev_state = last_state_from_history(history)
    prev_action = last_action_from_history(history)
    best, consensus = select_candidate(
        candidates,
        prev_mood=prev_mood,
        mood_inertia=mood_inertia,
        prev_state=prev_state,
        prev_action=prev_action,
        user_message=user_message,
    )
    mood = best.mood or "PLAYFUL"
    reply = best.reply or fallback_reply(best.raw)
    think = best.think or "q=unknown d=watch a=observe p=blink>stare h=mid e=mid t=mid m=mid k=mid c=evening r=window z=awake f=bird i=tracking u=selective o=perch n=still_bird"
    state = best.state
    plan = best.plan
    action = best.action or infer_action(think, reply, mood, state=state, plan=plan)
    gallery_source = [c for c in candidates if c.mood is not None and c.reply] or candidates
    gallery = build_rollout_gallery(gallery_source, prev_state=prev_state, best=best)

    clean_user = sanitize(user_message, tokenizer).strip()
    turn = f"<USER>{clean_user}</USER><THINK>{think}</THINK><MOOD={mood}> {reply}"
    next_history = trim_history(history + turn + "\n", tokenizer, model.config.block_size)

    return Decision(
        mood=mood,
        reply=reply,
        think=think,
        action=action,
        plan=plan,
        state=state,
        history=next_history,
        used_samples=samples,
        consensus=consensus,
        gallery=gallery,
    )
