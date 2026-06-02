# Lens SDK And HTTP API

Prefer the Python SDK from the PyPI `moondream` package. Use the HTTP API only as a fallback when debugging wire format issues.

## Python SDK

```python
import os
import time
import moondream as md

ft = md.ft(
    api_key=os.environ["MOONDREAM_API_KEY"],
    name=f"my-finetune-{int(time.time())}",
    rank=32,
)
```

Main methods:

- `ft.rollouts(skill, image=..., question=..., object=..., num_rollouts=..., settings=...)`
- `ft.rollout_stream(requests)`: concurrent rollout generation. Each item is `(context, rollout_kwargs)`.
- `ft.train_step(groups, lr=1e-4)`
- `ft.log_metrics(step=step, metrics={...})`
- `ft.save_checkpoint()`
- `ft.list_checkpoints(limit=50, cursor=None)`
- `ft.delete_checkpoint(step=...)`
- `ft.model(step=...)`
- `ft.delete()`

Lens finetuning currently uses `query`, `detect`, and `point`. For tags or descriptions, use `query` with a task-specific question. `caption` remains useful for inference, but this skill does not treat it as a Lens training capability.

## RL Group

For RL, pass the rollout response metadata back unchanged:

```python
response = ft.rollouts(
    "query",
    image=image,
    question="What class is this?",
    num_rollouts=8,
    settings={"temperature": 1.0, "max_tokens": 8},
)

step = ft.train_step([{
    "mode": "rl",
    "request": response["request"],
    "rollouts": response["rollouts"],
    "rewards": rewards,
}], lr=1e-4)
```

Do not mutate `response["request"]` or `response["rollouts"]`. Rewards must match rollout order and length.

## SFT Group

For SFT, build supervised groups directly:

```python
step = ft.train_step([{
    "mode": "sft",
    "request": {
        "skill": "query",
        "image": image,
        "question": "What country is this?",
    },
    "target": {"answer": "United States"},
}], lr=1e-4)
```

Point targets can use `points`; the Lens API also accepts `boxes` for point SFT. Detect targets use `boxes`.

## Checkpoints And Inference

```python
checkpoint = ft.save_checkpoint()["checkpoint"]
model_id = ft.model(checkpoint["step"])
```

Only saved checkpoints are usable for inference. The model ID format is:

```text
moondream3-preview/{finetune_id}@{step}
```

## HTTP Fallback

Base URL:

```text
https://api.moondream.ai/v1/tuning/
```

Important endpoints:

- `POST /finetunes`
- `POST /rollouts`
- `POST /train_step`
- `POST /finetunes/:finetuneId/metrics`
- `GET /finetunes/:finetuneId/checkpoints`
- `POST /finetunes/:finetuneId/checkpoints/save`

Use header `X-Moondream-Auth: $MOONDREAM_API_KEY`. Keep HTTP examples in references only; scripts should use the SDK unless the SDK is unavailable.
