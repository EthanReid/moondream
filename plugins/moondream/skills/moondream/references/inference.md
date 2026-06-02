# Inference

Use the PyPI package:

```bash
pip install moondream pillow
```

If running commands from this repository root, the local `moondream/` source tree can shadow the PyPI SDK. Run SDK examples from another project directory, or remove the repo root from `PYTHONPATH` before importing `moondream`.

## Backends

```python
import os
import moondream as md

cloud = md.vl(api_key=os.environ["MOONDREAM_API_KEY"])
photon = md.vl(api_key=os.environ["MOONDREAM_API_KEY"], local=True)
finetune = md.vl(
    api_key=os.environ["MOONDREAM_API_KEY"],
    model="moondream3-preview/FINETUNE_ID@STEP",
)
```

- Cloud means hitting the hosted Moondream API through the Python SDK.
- Photon means running the model locally through the Python SDK with `local=True`.
- Lens means the Moondream finetuning API. A Lens checkpoint is used for inference by passing its saved model ID to `md.vl(...)`.
- A Lens checkpoint is usable only after it is saved; model ID format is `moondream3-preview/{finetune_id}@{step}`.

## Image Input

Pass a `PIL.Image.Image` to SDK calls:

```python
from PIL import Image

image = Image.open("image.jpg")
```

Use `model.encode_image(image)` when reusing the same image across many calls.

## Core Calls

```python
caption = model.caption(image, length="short")["caption"]
answer = model.query(image, "What is unusual here?")["answer"]
objects = model.detect(image, "person")["objects"]
points = model.point(image, "person")["points"]
```

Return shapes:

```python
{"caption": "..."}
{"answer": "..."}
{"objects": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.4, "y_max": 0.6}]}
{"points": [{"x": 0.52, "y": 0.31}]}
```

Coordinates are normalized `0..1`. Convert to pixels with image dimensions:

```python
px_box = {
    "x_min": int(obj["x_min"] * image.width),
    "y_min": int(obj["y_min"] * image.height),
    "x_max": int(obj["x_max"] * image.width),
    "y_max": int(obj["y_max"] * image.height),
}
```

## Streaming

`caption` and `query` can stream:

```python
for chunk in model.query(image, "Describe the scene.", stream=True)["answer"]:
    print(chunk, end="", flush=True)
```

`detect` and `point` are not streaming calls in the core docs. `segment` is also available in recent SDKs, but this skill focuses on `caption`, `query`, `detect`, and `point`.

## Common Mistakes

- Passing pixel boxes to APIs that expect normalized coordinates.
- Forgetting that finetuned model inference requires a saved checkpoint.
- Using high temperature for evaluation. Use `temperature=0.0` when measuring.
- Logging raw API keys in shell traces, notebooks, or JSON configs.
