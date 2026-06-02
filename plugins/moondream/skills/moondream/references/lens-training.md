# Lens Finetuning

Lens means the Moondream finetuning API exposed through `md.ft(...)`. It trains hosted LoRA adapters in Moondream Cloud. The user's code orchestrates data loading, rollout generation, rewards or SFT targets, train steps, evaluation, metrics, checkpoints, and monitoring.

## Default Flow

1. Confirm `MOONDREAM_API_KEY` is set without printing it.
2. Ask what the model should improve at:
   - `query`: answer questions, classify, tag images, or produce short descriptions.
   - `detect`: return object boxes.
   - `point`: return object centers.
3. Choose training mode:
   - Default to RL when the base model can partially do the task and a reward is easier than writing perfect targets.
   - For fuzzy tagging/description quality, use a `query` finetune and consider a custom judge reward only after checking a few examples manually.
   - Use SFT first when the base model cannot do the task or labels are exact and easy.
4. Inspect data:
   - Load 5-20 examples.
   - Confirm image field, label/answer field, boxes/points field, prompt/question/object text, and split names.
   - Confirm boxes and points are normalized `0..1`; convert pixel annotations before training.
5. Create splits if needed:
   - Train split for updates.
   - Eval split for iterative checkpoint selection.
   - Final test split kept untouched until the end.
6. Run a dry validation first. Do not spend Lens credits until data loads and metrics can be computed.
7. Confirm cost, then run a small smoke finetune.
8. Monitor metrics every step. Stop on collapse, plateau, bad data, or obvious reward bugs.
9. Save checkpoints every 10 steps and evaluate them on the held-out split.
10. Report the saved model ID and residual failure modes.

## Data Sources

The scripts support Hugging Face datasets and local JSONL.

Hugging Face example:

```bash
python3 "$MOONDREAM_SKILL_DIR/scripts/train_loop.py" \
  --capability detect --mode rl --object "vehicle" \
  --hf-dataset my-org/my-dataset \
  --train-split train --eval-split validation \
  --image-field image --boxes-field boxes \
  --dry-run
```

Tagging or description query example:

```bash
python3 "$MOONDREAM_SKILL_DIR/scripts/train_loop.py" \
  --capability query --mode sft \
  --local-jsonl data/images.jsonl \
  --question "List concise searchable tags for this image." \
  --answer-field tags \
  --dry-run
```

For fuzzy description RL, start with the same `query` framing. Before paid training, create a task-specific judge script in the user's project, test it on a few examples, and pass it through `--judge-command`. The bundled `scripts/query_judge_example.py` is only a stdin/stdout contract example, not the default judge to run.

```bash
python3 "$MOONDREAM_SKILL_DIR/scripts/train_loop.py" \
  --capability query --mode rl \
  --local-jsonl data/images.jsonl \
  --question "Describe this image for product search." \
  --answer-field description \
  --reward judge \
  --judge-rubric "Reward faithful, specific, concise descriptions. Penalize hallucinated attributes." \
  --judge-command "python3 scripts/judge_description.py" \
  --dry-run
```

Local JSONL example:

```json
{"image": "images/0001.jpg", "boxes": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.4, "y_max": 0.6}]}
{"image": "images/0002.jpg", "boxes": []}
```

```bash
python3 "$MOONDREAM_SKILL_DIR/scripts/train_loop.py" \
  --capability detect --mode rl --object "defect" \
  --local-jsonl data/examples.jsonl --dry-run
```

If no eval/test split exists, the training script creates deterministic `train.jsonl`, `eval.jsonl`, and `test.jsonl` files in the run directory.
If you provide an existing eval split but no test split, paid training stops unless you add `--allow-missing-test` after explicitly accepting that risk.

## Healthy Training Signals

Healthy:

- rewards are not all zero for many steps;
- reward variance exists during RL;
- eval metric improves or at least moves;
- train reward and eval metric are roughly consistent;
- checkpoint eval does not regress sharply.

Unhealthy:

- every rollout gets the same reward;
- reward is high but eval is low, often a reward bug;
- detection predicts no boxes on positive examples;
- detection predicts many boxes on empty examples;
- coordinates look like pixels instead of normalized values;
- eval split is accidentally part of training data;
- no checkpoints were saved, so there is no usable model.

## Suggested Defaults

- `rank=32` for real first finetunes. Lower to `8` or `16` only for cheap smoke tests, very simple tasks, or tight cost constraints.
- `lr=1e-4` as the default starting learning rate.
- `num_rollouts=8` for RL query, detect, and point finetunes.
- `temperature=1.0` for training rollouts.
- `temperature=0.0` for eval.
- Start with 10 steps for a smoke run, then scale.

## What The Agent Should Say During A Run

Keep the user oriented:

- "I loaded N rows and created train/eval/test splits."
- "The first few boxes are normalized and valid."
- "The run is logging to runs/.../metrics.jsonl."
- "Checkpoint step 10 is saved and can be evaluated as model ID ..."
- "The reward trend is flat; I am checking examples and reward parsing before continuing."
