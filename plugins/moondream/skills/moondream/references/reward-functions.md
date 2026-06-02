# Reward Functions

Use this reference when designing Lens RL rewards. The important built-in script is `scripts/reward.py`.

## Detection Reward

For object-detection finetunes, use mean IoU with explicit empty-image centering.

Inputs for one image:

```python
predictions = [[x_min, y_min, x_max, y_max], ...]
ground_truth = [[x_min, y_min, x_max, y_max], ...]
```

Coordinates must use the same normalized `0..1` convention that `detect` returns.

Rules:

- If ground truth is empty and prediction is empty, reward is `1.0`.
- If ground truth is empty and predictions are spurious:
  - `P=1 -> 0.5`
  - `P=2 -> 0.3`
  - `P=3 -> 0.1`
  - `P>=4 -> 0.0`
- If ground truth is non-empty, one-to-one match predictions to ground-truth boxes to maximize total IoU.
- Unmatched predictions and unmatched ground truth contribute `0`.
- Reward is `matched_iou_sum / max(G, P)`.

The empty-image branch is intentionally gentler than returning `0.0` for one false positive. It still penalizes spurious boxes, but it avoids teaching the model to collapse to "always predict nothing."

Anchor cases:

```python
detection_reward([], []) == 1.0
detection_reward([[0, 0, 1, 1]], []) == 0.5
detection_reward([[0, 0, 1, 1], [0, 0, 0.5, 0.5]], []) == 0.3
detection_reward([[0, 0, 1, 1]], [[0, 0, 1, 1]]) == 1.0
detection_reward([[0, 0, 1, 1], [0, 0, 0.5, 0.5]], [[0, 0, 1, 1]]) == 0.5
```

## Query Rewards

For query RL, the user must define the metric. Use query for classification, tagging, extraction, and short description tasks. Start simple:

- Classification: normalize punctuation/case and exact-match the label.
- Tagging: exact-match a canonical tag string, or use `contains` for simple required-tag checks.
- Extraction: exact-match after normalization, or use a strict parser.
- Free-form descriptions: ask the user to define a rubric before training.

Do not use `ground_truth` for query rollouts; compute query rewards in user code.

### Optional LLM Judge For Query

LLM-as-judge is only a suggestion for fuzzy description/tagging work. Prefer exact labels, strict parsers, or human-reviewed examples when those are available. If a judge is useful, generate a task-specific judge script and prompt in the user's project.

Before using a judge model, ask the user for:

- the desired answer style, length, and audience;
- what errors matter most, such as hallucination, missing details, OCR mistakes, tone, or verbosity;
- whether reference answers or human preference examples exist;
- judge model/provider and budget limits;
- whether the judge should compare only model outputs or also use a reference answer.

When `--reward judge` is used for a query finetune, the judge receives JSON on stdin with:

```json
{
  "image_data_url": "data:image/jpeg;base64,...",
  "question": "Describe this image for product search.",
  "candidates": ["candidate 0", "candidate 1"],
  "reference": "optional reference answer",
  "rubric": "score rubric"
}
```

It should return:

```json
{"rewards": [0.25, 1.0]}
```

Custom judges may also return a best-to-worst `ranking` for multi-rollout RL:

```json
{"ranking": [1, 0]}
```

The bundled `scripts/query_judge_example.py` is only a runnable contract example. Do not use it as the real judge for a finetune; write a local judge command with the user's chosen model/provider and rubric.

## Point Rewards

The bundled script rewards point tasks by one-to-one matching predicted points to target points and scoring each match by normalized distance. Perfectly matched multi-point examples score `1.0`; missing or extra points lower the score through the denominator.

Point-in-box rewards are a reasonable task-specific variant, but they are not built in. Add that locally only when the dataset labels boxes rather than target points.

Use normalized coordinates consistently.
