# CatGPT

Tiny GPT-style cat model with model-internal reasoning traces and test-time compute (self-consistency over sampled traces).

## Quickstart

```bash
# 1) Generate reasoning corpus
.venv/bin/python -m catgpt.generate_corpus --lines 1000000 --out data/cat_corpus.txt

# 2) Train (larger context is important for <USER>/<THINK> traces)
.venv/bin/python -m catgpt.train --data data/cat_corpus.txt --out checkpoints/catgpt.pt --block-size 160 --steps 3000

# 3) CLI chat with TTC
.venv/bin/python -m catgpt.chat --checkpoint checkpoints/catgpt.pt --reasoning-rollouts 8
# optional: disable TTC and use instant mood-transition mode
.venv/bin/python -m catgpt.chat --checkpoint checkpoints/catgpt.pt --mode instant
```

One-shot with reasoning debug:

```bash
.venv/bin/python -m catgpt.chat --checkpoint checkpoints/catgpt.pt --message "do you want food" --reasoning-rollouts 12 --show-reasoning
# optional: smoother turn-to-turn mood persistence
.venv/bin/python -m catgpt.chat --checkpoint checkpoints/catgpt.pt --reasoning-rollouts 12 --mood-inertia 0.55
```

Local web UI (Flask):

```bash
.venv/bin/python -m catgpt.web --checkpoint checkpoints/catgpt.pt --reasoning-rollouts 8 --port 8000
# open http://127.0.0.1:8000
# optional instant mode:
.venv/bin/python -m catgpt.web --checkpoint checkpoints/catgpt.pt --mode instant --port 8000
```

## Notes

- This TTC setup expects a checkpoint trained on the reasoning corpus format (`<USER>...<THINK>...</THINK><MOOD=...>...`).
- More `--reasoning-rollouts` means more inference compute and usually more stable mood/action outputs.
- `--mood-inertia` controls volatility (higher = stickier mood transitions).
- `--mode instant` disables TTC and uses lightweight stochastic mood transitions + direct `<MOOD=...>` generation.
