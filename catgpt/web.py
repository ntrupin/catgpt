from __future__ import annotations

import argparse
import random
import threading

import torch
from flask import Flask, jsonify, render_template, request, session

from .model import load_checkpoint
from .mood import initial_mood, next_mood
from .runtime import pick_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CatGPT web UI")
    p.add_argument("--checkpoint", default="checkpoints/catgpt.pt")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--force-mood", default=None, choices=["HUNGRY", "SLEEPY", "GRUMPY", "PLAYFUL"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def create_app(args: argparse.Namespace) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = "catgpt-local-dev"

    device = pick_device(args.device)
    model, tok, _ = load_checkpoint(args.checkpoint, device)

    rng = random.Random(args.seed)
    rng_lock = threading.Lock()
    model_lock = threading.Lock()

    def rand_initial() -> str:
        with rng_lock:
            return initial_mood(rng)

    def rand_next(prev: str, msg: str) -> str:
        with rng_lock:
            return next_mood(prev, msg, rng)

    def ensure_mood() -> str:
        mood = session.get("mood")
        if mood is None:
            mood = args.force_mood or rand_initial()
            session["mood"] = mood
        return mood

    def sample_reply(mood: str) -> str:
        prefix = f"<MOOD={mood}> "
        x = torch.tensor([tok.encode(prefix)], dtype=torch.long, device=device)
        with model_lock:
            y = model.generate(
                x,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                stop_token_id=tok.stoi["\n"],
            )
        return tok.decode(y[0].tolist())[len(prefix) :].split("\n", 1)[0].strip() or "mrrp"

    @app.get("/")
    def index():
        return render_template("index.html", mood=ensure_mood())

    @app.get("/api/state")
    def state():
        return jsonify({"mood": ensure_mood()})

    @app.post("/api/reset")
    def reset():
        mood = args.force_mood or rand_initial()
        session["mood"] = mood
        return jsonify({"mood": mood})

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True) or {}
        message = str(payload.get("message", "")).strip()
        if not message:
            return jsonify({"error": "empty_message"}), 400

        mood = ensure_mood()
        if args.force_mood is None:
            mood = rand_next(mood, message)
            session["mood"] = mood

        return jsonify({"mood": mood, "reply": sample_reply(mood)})

    return app


def main() -> None:
    args = parse_args()
    app = create_app(args)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
