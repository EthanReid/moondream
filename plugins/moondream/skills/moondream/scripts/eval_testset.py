"""Evaluate a saved Moondream model or Lens checkpoint on a held-out test set."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import time
from pathlib import Path
from typing import Any

from reward import detection_reward
from train_loop import (
    decode_image,
    import_moondream_sdk,
    load_hf_rows,
    normalize_boxes,
    normalize_points,
    point_reward,
    read_jsonl,
    score_query_rollouts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Moondream on a held-out split.")
    parser.add_argument("--capability", choices=["query", "detect", "point"], required=True)
    parser.add_argument("--model-id", required=True, help="Model ID, e.g. moondream3-preview/FT_ID@STEP.")
    parser.add_argument("--hf-dataset", help="Hugging Face dataset name.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--local-jsonl", help="Local JSONL split path.")
    parser.add_argument("--image-field", default="image")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--boxes-field", default="boxes")
    parser.add_argument("--points-field", default="points")
    parser.add_argument("--question", help="Question for query evaluation.")
    parser.add_argument("--object", dest="object_name", help="Object text for detect/point.")
    parser.add_argument("--reward", choices=["exact", "contains", "judge"], default="exact")
    parser.add_argument("--judge-rubric", default="Score answers by image faithfulness and usefulness for the task.")
    parser.add_argument(
        "--judge-command",
        help="Task-specific query reward command. Reads JSON on stdin and writes JSON rewards to stdout.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--yes", action="store_true", help="Confirm that a full Cloud eval may spend credits.")
    parser.add_argument("--output-jsonl", help="Optional per-example output file.")
    return parser.parse_args()


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if bool(args.hf_dataset) == bool(args.local_jsonl):
        raise SystemExit("Provide exactly one of --hf-dataset or --local-jsonl.")
    if args.hf_dataset:
        return load_hf_rows(args.hf_dataset, args.split, args.limit)
    rows = read_jsonl(Path(args.local_jsonl).resolve())
    return rows[: args.limit] if args.limit else rows


def predict(model: Any, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    image = decode_image(row, args.image_field)
    if args.capability == "query":
        if not args.question:
            raise SystemExit("--question is required for query evaluation.")
        return model.query(image, args.question, stream=False)
    if args.capability == "detect":
        if not args.object_name:
            raise SystemExit("--object is required for detect evaluation.")
        return model.detect(image, args.object_name)
    if not args.object_name:
        raise SystemExit("--object is required for point evaluation.")
    return model.point(image, args.object_name)


def score(result: dict[str, Any], row: dict[str, Any], args: argparse.Namespace) -> float:
    if args.capability == "detect":
        return float(detection_reward(result.get("objects", []), normalize_boxes(row.get(args.boxes_field))))
    if args.capability == "point":
        return float(point_reward(result.get("points", []), normalize_points(row.get(args.points_field))))

    fake_response = {"rollouts": [{"output": {"answer": result.get("answer", "")}}]}
    return score_query_rollouts(row, fake_response, args)[0]


def main() -> None:
    args = parse_args()
    if args.limit is None and not args.yes:
        raise SystemExit("Set --limit for a bounded eval or pass --yes to confirm full Cloud eval cost.")
    if args.reward == "judge" and args.capability != "query":
        raise SystemExit("--reward judge is only supported for query evaluation.")
    if args.reward == "judge":
        if not args.judge_command:
            raise SystemExit("--judge-command is required for query judge evaluation.")
        command = shlex.split(args.judge_command)
        if not command:
            raise SystemExit("--judge-command is empty.")
        executable = command[0]
        if not Path(executable).exists() and shutil.which(executable) is None:
            raise SystemExit(f"Judge command executable not found: {executable}")

    rows = load_rows(args)
    if not rows:
        raise SystemExit("No rows to evaluate.")

    if not os.environ.get("MOONDREAM_API_KEY"):
        raise SystemExit("MOONDREAM_API_KEY is not set. Export it in the shell; do not paste it in chat.")

    md = import_moondream_sdk()
    if not hasattr(md, "vl"):
        raise SystemExit("Imported moondream SDK does not expose md.vl. Run: pip install -U moondream")
    model = md.vl(api_key=os.environ["MOONDREAM_API_KEY"], model=args.model_id)

    output_handle = open(args.output_jsonl, "w") if args.output_jsonl else None
    scores = []
    try:
        for idx, row in enumerate(rows, start=1):
            result = predict(model, row, args)
            value = score(result, row, args)
            scores.append(value)
            if output_handle:
                output_handle.write(json.dumps({
                    "index": idx,
                    "score": value,
                    "result": result,
                }) + "\n")
            print(f"[{idx}/{len(rows)}] score={value:.3f}", flush=True)
    finally:
        if output_handle:
            output_handle.close()

    metric = sum(scores) / len(scores)
    summary = {
        "model_id": args.model_id,
        "capability": args.capability,
        "rows": len(rows),
        "metric": metric,
        "score_min": min(scores),
        "score_max": max(scores),
        "ts": time.time(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
