from __future__ import annotations

import argparse
import random
import threading

import torch
from flask import Flask, jsonify, render_template, request, session

from .model import load_checkpoint
from .mood import initial_mood, next_mood
from .runtime import pick_device
from .ttc import missing_reasoning_chars, ttc_turn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CatGPT web UI")
    p.add_argument("--checkpoint", default="checkpoints/catgpt.pt")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--mode", default="reasoning", choices=["reasoning", "instant"])
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--reasoning-rollouts", type=int, default=8)
    p.add_argument("--mood-inertia", type=float, default=0.35)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def create_app(args: argparse.Namespace) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = "catgpt-local-dev"

    device = pick_device(args.device)
    model, tok, _ = load_checkpoint(args.checkpoint, device)

    if args.mode == "reasoning":
        missing = missing_reasoning_chars(tok)
        if missing:
            joined = "".join(missing)
            raise RuntimeError(
                f"Checkpoint tokenizer is missing reasoning chars: {joined!r}. "
                "Regenerate corpus with reasoning traces and retrain, or run --mode instant."
            )

    if args.seed is not None:
        torch.manual_seed(args.seed)

    rng = random.Random(args.seed)
    rng_lock = threading.Lock()
    model_lock = threading.Lock()

    def sample_instant_reply(mood: str) -> str:
        prefix = f"<MOOD={mood}> "
        encoded = tok.encode(prefix)
        x = torch.tensor([encoded], dtype=torch.long, device=device)
        with model_lock:
            y = model.generate(
                x,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                stop_token_id=tok.stoi.get("\n"),
            )
        return tok.decode(y[0].tolist())[len(prefix) :].split("\n", 1)[0].strip() or "mrrp"

    def ensure_session() -> tuple[str, str]:
        history = session.get("history")
        mood = session.get("mood")

        if history is None:
            history = ""
            session["history"] = history

        if mood is None:
            if args.mode == "instant":
                with rng_lock:
                    mood = initial_mood(rng)
            else:
                mood = "PLAYFUL"
            session["mood"] = mood

        return history, mood

    @app.get("/")
    def index():
        _, mood = ensure_session()
        return render_template(
            "index.html",
            mood=mood,
            rollouts=args.reasoning_rollouts,
            mode=args.mode,
        )

    @app.get("/api/state")
    def state():
        _, mood = ensure_session()
        return jsonify({"mood": mood, "rollouts": args.reasoning_rollouts, "mode": args.mode})

    @app.post("/api/reset")
    def reset():
        session["history"] = ""
        if args.mode == "instant":
            with rng_lock:
                mood = initial_mood(rng)
        else:
            mood = "PLAYFUL"
        session["mood"] = mood
        return jsonify({"mood": mood, "rollouts": args.reasoning_rollouts, "mode": args.mode})

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True) or {}
        message = str(payload.get("message", "")).strip()
        if not message:
            return jsonify({"error": "empty_message"}), 400

        history, mood = ensure_session()

        if args.mode == "instant":
            with rng_lock:
                mood = next_mood(mood, message, rng)
            reply = sample_instant_reply(mood)
            session["mood"] = mood
            return jsonify(
                {
                    "mode": "instant",
                    "mood": mood,
                    "reply": reply,
                    "action": "instant response",
                    "consensus": None,
                    "samples": None,
                }
            )

        with model_lock:
            d = ttc_turn(
                model=model,
                tokenizer=tok,
                device=device,
                history=history,
                user_message=message,
                rollouts=args.reasoning_rollouts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                mood_inertia=args.mood_inertia,
            )

        session["history"] = d.history
        session["mood"] = d.mood

        return jsonify(
            {
                "mode": "reasoning",
                "mood": d.mood,
                "reply": d.reply,
                "action": d.action,
                "think": d.think,
                "consensus": d.consensus,
                "samples": d.used_samples,
            }
        )

    return app


def main() -> None:
    args = parse_args()
    app = create_app(args)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
