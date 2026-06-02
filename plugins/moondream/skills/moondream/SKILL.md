---
name: moondream
description: Use when a developer wants Moondream image inference or Lens finetuning with the PyPI moondream package, especially query, detect, or point finetunes; description/tagging query tasks; train/test split creation; Hugging Face or local dataset loading; metric logging; checkpointing; monitoring; Cloud API inference; or Photon local inference.
---

# Moondream

Use this skill to help a developer run Moondream inference or guide a Lens finetune end to end. Be hands-on: inspect their data shape, create a small safe first run, monitor metrics, and explain each next action plainly.

Terminology:

- Lens means the Moondream finetuning API exposed through `md.ft(...)`.
- Photon means running the model locally through the Python SDK with `md.vl(..., local=True)`.
- Cloud means hitting the hosted Moondream API through the Python SDK with `md.vl(api_key=...)`.

## Opening Flow

1. Check auth without printing secrets:
   - If the task uses Cloud, Photon local, or Lens, check whether `MOONDREAM_API_KEY` exists in the environment.
   - If it is missing, point the user to `references/signup.md` and ask them to set `export MOONDREAM_API_KEY=...`.
   - Never ask the user to paste the raw key in chat. Never echo, log, write, or hardcode it.
2. Classify the task:
   - Inference: ask for image path/source, capability (`caption`, `query`, `detect`, `point`), prompt/object/question, and backend (Cloud or Photon local).
   - Finetune: ask RL or SFT, defaulting to RL; ask Lens capability (`query`, `detect`, `point`), one-line use case, dataset location, expected label fields, metric, and whether they already have train/eval/test splits.
   - For tagging or description improvement, use a `query` finetune with an explicit question such as "Describe this image for product search" or "List concise searchable tags." Use exact/contains rewards for simple tags. For fuzzy descriptions, write a task-specific judge script and prompt in the user's project after confirming judge cost; do not run the bundled judge example unchanged.
3. For finetunes, require a held-out test set before training:
   - If the user has no split, create one deterministically from the dataset. Keep a final untouched test split separate from the eval split used during iteration.
   - Run a dry data validation before any Lens call.
4. Lens runs in Moondream Cloud and spends credits. Before starting or resuming training, ask for explicit confirmation.

## What To Load

- Inference details: read `references/inference.md`.
- Signup/API key setup: read `references/signup.md`.
- Finetune planning and best practices: read `references/lens-training.md`.
- SDK/HTTP method details: read `references/lens-api.md`.
- Object-detection reward details: read `references/reward-functions.md`.

## Inference Cheat Sheet

Use the PyPI package:

```python
import os
import moondream as md
from PIL import Image

model = md.vl(api_key=os.environ["MOONDREAM_API_KEY"])            # Cloud
model = md.vl(api_key=os.environ["MOONDREAM_API_KEY"], local=True) # Photon local

image = Image.open("image.jpg")
caption = model.caption(image, length="short")["caption"]
answer = model.query(image, "What is in this image?")["answer"]
objects = model.detect(image, "person")["objects"]
points = model.point(image, "person")["points"]
```

`detect` boxes and `point` coordinates are normalized `0..1`, not pixels.

## Finetune Loop

Use the scripts in `scripts/` when possible. Resolve script paths from the installed skill directory, not from the user's current project. In examples below, set `MOONDREAM_SKILL_DIR` to the absolute path of this skill directory.

1. Validate data and create splits without spending credits:
   ```bash
   python3 "$MOONDREAM_SKILL_DIR/scripts/train_loop.py" \
     --capability detect --mode rl --object "defect" \
     --local-jsonl data/examples.jsonl --dry-run
   ```
2. Start a small Lens smoke run only after cost confirmation:
   ```bash
   python3 "$MOONDREAM_SKILL_DIR/scripts/train_loop.py" \
     --capability detect --mode rl --object "defect" \
     --local-jsonl data/examples.jsonl --steps 10 --yes
   ```
3. Monitor while it runs:
   ```bash
   python3 "$MOONDREAM_SKILL_DIR/scripts/monitor.py" runs/latest/metrics.jsonl
   ```
4. Evaluate saved checkpoints on the untouched test split:
   ```bash
   python3 "$MOONDREAM_SKILL_DIR/scripts/eval_testset.py" \
     --capability detect --object "defect" \
     --local-jsonl runs/latest/test.jsonl \
     --model-id moondream3-preview/FINETUNE_ID@STEP \
     --limit 100
   ```

Default detection reward is mean IoU with explicit empty-image centering. Query rewards default to exact match or contains match; for fuzzy descriptions, `--reward judge` can call a user-provided `--judge-command`. The judge command should be custom code generated for the user's actual rubric and provider. The loop logs metrics every step to stdout and `metrics.jsonl`, saves checkpoints every 10 steps, and flags unhealthy reward trends.

## Defaults

- Minimum recommended `moondream` SDK: `1.2.2`. If the installed version is older, check docs and avoid relying on newer Lens helpers until upgraded.
- RL is the default for Lens because it is sample-efficient and lets the user encode "good" with a reward. Use SFT first when the base model cannot do the task at all or when exact labels are simple.
- Good starting hyperparameters: `rank=32`, `lr=1e-4`, and `num_rollouts=8` for RL. Use these unless the dataset is tiny, the run is only a dependency smoke test, or the user gives a task-specific reason to change them.
- Training temperature should be high enough for rollout diversity, usually `1.0`; evaluation should use `temperature=0.0`.
- LoRA rank must be one of `8`, `16`, `24`, or `32`. Start with `32` for real finetunes; lower it only for very cheap smoke tests or unusually simple tasks.
