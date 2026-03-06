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
ACTION_RE = re.compile(r"\baction=([a-z_]+)\b")
CAT_SOUND_RE = re.compile(r"meow|mew|mrr|mrrow|prr|hiss|nya|brr", re.IGNORECASE)
TAG_CUT_RE = re.compile(r"\s*<(?:USER|THINK|MOOD=|/USER|/THINK).*$", re.IGNORECASE)
LAST_MOOD_RE = re.compile(r"<MOOD=(HUNGRY|SLEEPY|GRUMPY|PLAYFUL)>")


@dataclass
class Candidate:
    raw: str
    think: str
    mood: str | None
    reply: str
    action: str | None
    avg_logprob: float
    score: float


@dataclass
class Decision:
    mood: str
    reply: str
    think: str
    action: str
    history: str
    used_samples: int
    consensus: int


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

    stop_token = tokenizer.stoi.get("\n")
    temp = max(temperature, 1e-5)
    logprob_sum = 0.0
    steps = 0

    with torch.no_grad():
        for _ in range(max_new_tokens):
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


def parse_candidate(raw: str, avg_logprob: float) -> Candidate:
    line = raw.split("\n", 1)[0]
    m = PARSE_RE.match(line)
    if not m:
        mm = MOOD_ONLY_RE.match(line)
        if not mm:
            return Candidate(raw=line, think="", mood=None, reply="", action=None, avg_logprob=avg_logprob, score=avg_logprob - 0.4)
        mood = mm.group("mood")
        reply = TAG_CUT_RE.sub("", mm.group("reply")).strip()
        score = avg_logprob + (0.12 if reply else -0.12) + (0.1 if CAT_SOUND_RE.search(reply) else 0.0)
        return Candidate(raw=line, think="", mood=mood, reply=reply, action=None, avg_logprob=avg_logprob, score=score)

    think = m.group("think").strip()
    mood = m.group("mood")
    reply = TAG_CUT_RE.sub("", m.group("reply")).strip()

    action_match = ACTION_RE.search(think)
    action = action_match.group(1).replace("_", " ") if action_match else None

    score = avg_logprob
    if think:
        score += 0.07
    if reply:
        score += 0.1
    if CAT_SOUND_RE.search(reply):
        score += 0.14
    if mood in MOODS:
        score += 0.1

    return Candidate(
        raw=line,
        think=think,
        mood=mood,
        reply=reply,
        action=action,
        avg_logprob=avg_logprob,
        score=score,
    )


def last_mood_from_history(history: str) -> str | None:
    matches = LAST_MOOD_RE.findall(history)
    return matches[-1] if matches else None


def select_candidate(candidates: list[Candidate], prev_mood: str | None, mood_inertia: float) -> tuple[Candidate, int]:
    parsed = [c for c in candidates if c.mood is not None and c.reply]
    if not parsed:
        best = max(candidates, key=lambda c: c.score)
        return best, 1

    if prev_mood in MOODS and mood_inertia > 0:
        weights = MOOD_PRIOR[prev_mood]
        for c in parsed:
            c.score += mood_inertia * weights.get(c.mood, 0.0)

    mood_counts = Counter(c.mood for c in parsed)
    if prev_mood in MOODS and mood_inertia > 0:
        prior = MOOD_PRIOR[prev_mood]
        mood_votes = {
            m: mood_counts[m] + mood_inertia * prior.get(m, 0.0)
            for m in mood_counts
        }
        top_mood = max(mood_votes.items(), key=lambda kv: (kv[1], kv[0]))[0]
        top_count = int(mood_counts[top_mood])
    else:
        top_mood, top_count = max(mood_counts.items(), key=lambda kv: (kv[1], kv[0]))
    finalists = [c for c in parsed if c.mood == top_mood]
    best = max(finalists, key=lambda c: c.score)
    return best, int(top_count)


def infer_action(think: str, reply: str, mood: str) -> str:
    m = ACTION_RE.search(think)
    if m:
        return m.group(1).replace("_", " ")

    rep = reply.lower()
    if "hiss" in rep or mood == "GRUMPY":
        return "defend"
    if "prr" in rep or mood == "SLEEPY":
        return "rest"
    if mood == "HUNGRY":
        return "seek food"
    return "social/play"


def fallback_reply(raw: str) -> str:
    words = re.findall(r"[A-Za-z!?\.]+", raw)
    cat_words = [w for w in words if CAT_SOUND_RE.search(w)]
    if cat_words:
        return " ".join(cat_words[:8])
    return "mrrp"


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
    best, consensus = select_candidate(candidates, prev_mood=prev_mood, mood_inertia=mood_inertia)
    mood = best.mood or "PLAYFUL"
    reply = best.reply or fallback_reply(best.raw)
    think = best.think or "cue=unknown action=social_play mood=PLAYFUL"
    action = infer_action(think, reply, mood)

    clean_user = sanitize(user_message, tokenizer).strip()
    turn = f"<USER>{clean_user}</USER><THINK>{think}</THINK><MOOD={mood}> {reply}"
    next_history = trim_history(history + turn + "\n", tokenizer, model.config.block_size)

    return Decision(
        mood=mood,
        reply=reply,
        think=think,
        action=action,
        history=next_history,
        used_samples=samples,
        consensus=consensus,
    )
