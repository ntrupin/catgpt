# CatGPT

Tiny GPT-style cat language model trained on synthetic cat-speech text.

## Quickstart

```bash
# 1) Generate corpus (change --lines to 1_000_000..5_000_000)
.venv/bin/python -m catgpt.generate_corpus --lines 1000000 --out data/cat_corpus.txt

# 2) Train tiny decoder-only model
.venv/bin/python -m catgpt.train --data data/cat_corpus.txt --out checkpoints/catgpt.pt --steps 3000

# 3) Chat
.venv/bin/python -m catgpt.chat --checkpoint checkpoints/catgpt.pt
```

One-shot reply:

```bash
.venv/bin/python -m catgpt.chat --checkpoint checkpoints/catgpt.pt --message "do you want food"
```

Local web UI (Flask):

```bash
.venv/bin/python -m catgpt.web --checkpoint checkpoints/catgpt.pt --port 8000
# open http://127.0.0.1:8000
```

## Defaults

- Model: `4` layers, `128` hidden, `4` heads, context `32`
- Tokenizer: character-level (learned from corpus)
- Mood control in corpus: `<MOOD=HUNGRY|SLEEPY|GRUMPY|PLAYFUL>`
