"""Moondream Lens training loop.

This script uses the PyPI moondream Python SDK (`md.ft`) for Lens training.
It deliberately avoids printing or storing MOONDREAM_API_KEY.
"""

from __future__ import annotations

import argparse
import base64
import io
import itertools
import json
import math
import os
import random
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from reward import detection_reward

CHECKPOINT_EVERY = 10
VALID_RANKS = {8, 16, 24, 32}
DEFAULT_RUN_ROOT = Path("runs")


def _repo_root_from_script() -> Path | None:
    path = Path(__file__).resolve()
    return path.parents[5] if len(path.parents) > 5 else None


def import_moondream_sdk():
    """Import the installed PyPI SDK, not this repo's local moondream package."""
    repo_root = _repo_root_from_script()
    if repo_root is not None and (repo_root / "moondream" / "__init__.py").exists():
        for entry in list(sys.path):
            resolved = Path(entry or os.getcwd()).resolve()
            if resolved == repo_root:
                sys.path.remove(entry)

    try:
        import moondream as md
    except ImportError as exc:
        raise SystemExit("Install the SDK first: pip install moondream pillow") from exc

    if not hasattr(md, "ft"):
        location = getattr(md, "__file__", "unknown")
        raise SystemExit(
            "Imported moondream does not expose md.ft. "
            f"Imported from {location}. Install/upgrade the PyPI SDK with: "
            "pip install -U moondream"
        )
    return md


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a guided Moondream Lens finetune.")
    parser.add_argument("--capability", choices=["query", "detect", "point"], required=True)
    parser.add_argument("--mode", choices=["rl", "sft"], default="rl")
    parser.add_argument("--hf-dataset", help="Hugging Face dataset name.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", help="Existing eval/validation split name.")
    parser.add_argument("--test-split", help="Existing untouched test split name.")
    parser.add_argument("--local-jsonl", help="Local JSONL dataset path.")
    parser.add_argument("--image-field", default="image")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--boxes-field", default="boxes")
    parser.add_argument("--points-field", default="points")
    parser.add_argument("--question", help="Question for query finetunes.")
    parser.add_argument("--object", dest="object_name", help="Object text for detect/point.")
    parser.add_argument("--reward", choices=["exact", "contains", "judge"], default="exact")
    parser.add_argument("--judge-rubric", default="Score answers by image faithfulness and usefulness for the task.")
    parser.add_argument(
        "--judge-command",
        help="Task-specific query reward command. Reads JSON on stdin and writes JSON rewards to stdout.",
    )
    parser.add_argument("--name", help="Finetune name. Defaults to a timestamped name.")
    parser.add_argument("--finetune-id", help="Resume an existing finetune ID.")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-limit", type=int, default=100)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--run-dir", help="Output directory. Defaults to runs/<name>-<timestamp>.")
    parser.add_argument(
        "--allow-missing-test",
        action="store_true",
        help="Allow paid training without a final untouched test split.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate data/splits without calling Lens.")
    parser.add_argument("--yes", action="store_true", help="Confirm that Lens Cloud training may spend credits.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_base_dir"] = str(path.parent.resolve())
            row["_row_id"] = idx
            rows.append(row)
    return rows


def load_hf_rows(dataset_name: str, split: str, limit: int | None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install datasets for Hugging Face loading: pip install datasets") from exc

    rows = load_dataset(dataset_name, split=split, token=os.environ.get("HF_TOKEN"))
    result = []
    for idx, row in enumerate(rows, start=1):
        row = dict(row)
        row["_row_id"] = idx
        result.append(row)
        if limit is not None and len(result) >= limit:
            break
    return result


def load_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, list[dict[str, Any]] | None]:
    if bool(args.hf_dataset) == bool(args.local_jsonl):
        raise SystemExit("Provide exactly one of --hf-dataset or --local-jsonl.")

    if args.hf_dataset:
        train = load_hf_rows(args.hf_dataset, args.train_split, args.train_limit)
        eval_rows = load_hf_rows(args.hf_dataset, args.eval_split, args.eval_limit) if args.eval_split else None
        test_rows = load_hf_rows(args.hf_dataset, args.test_split, args.eval_limit) if args.test_split else None
        return train, eval_rows, test_rows

    rows = read_jsonl(Path(args.local_jsonl).resolve())
    if args.train_limit is not None:
        rows = rows[: args.train_limit]
    return rows, None, None


def shuffled_split(
    rows: list[dict[str, Any]],
    eval_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 <= eval_ratio < 1 or not 0 <= test_ratio < 1 or eval_ratio + test_ratio >= 1:
        raise SystemExit("--eval-ratio and --test-ratio must be non-negative and sum to < 1.")

    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n = len(rows)
    if n < 3:
        raise SystemExit("Need at least 3 rows to create train/eval/test splits.")

    eval_count = max(1, int(round(n * eval_ratio)))
    test_count = max(1, int(round(n * test_ratio)))
    if eval_count + test_count >= n:
        eval_count = 1
        test_count = 1

    eval_rows = rows[:eval_count]
    test_rows = rows[eval_count : eval_count + test_count]
    train_rows = rows[eval_count + test_count :]
    return train_rows, eval_rows, test_rows


def decode_image(row: dict[str, Any], image_field: str):
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Install Pillow: pip install pillow") from exc

    value = row.get(image_field)
    if value is None:
        raise ValueError(f"missing image field {image_field!r}")

    if isinstance(value, Image.Image):
        return value.convert("RGB") if value.mode != "RGB" else value

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            import io

            with Image.open(io.BytesIO(value["bytes"])) as image:
                return image.convert("RGB")
        if value.get("path"):
            value = value["path"]

    if isinstance(value, str):
        image_path = Path(value)
        if not image_path.is_absolute() and row.get("_base_dir"):
            image_path = Path(row["_base_dir"]) / image_path
        with Image.open(image_path) as image:
            return image.convert("RGB")

    raise ValueError(f"unsupported image value for field {image_field!r}: {type(value).__name__}")


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped[0] in "[{":
            return json.loads(stripped)
    return value


def normalize_boxes(value: Any) -> list[dict[str, float]]:
    value = parse_jsonish(value)
    if value is None:
        return []
    boxes = value.get("boxes") if isinstance(value, dict) and "boxes" in value else value
    result = []
    for box in boxes or []:
        if isinstance(box, dict):
            coords = {
                "x_min": float(box["x_min"]),
                "y_min": float(box["y_min"]),
                "x_max": float(box["x_max"]),
                "y_max": float(box["y_max"]),
            }
        else:
            coords = {
                "x_min": float(box[0]),
                "y_min": float(box[1]),
                "x_max": float(box[2]),
                "y_max": float(box[3]),
            }
        if coords["x_min"] > coords["x_max"] or coords["y_min"] > coords["y_max"]:
            raise ValueError(f"invalid box coordinate order: {coords}")
        result.append(coords)
    return result


def normalize_points(value: Any) -> list[dict[str, float]]:
    value = parse_jsonish(value)
    if value is None:
        return []
    points = value.get("points") if isinstance(value, dict) and "points" in value else value
    result = []
    for point in points or []:
        if isinstance(point, dict):
            result.append({"x": float(point["x"]), "y": float(point["y"])})
        else:
            result.append({"x": float(point[0]), "y": float(point[1])})
    return result


def validate_normalized_boxes(boxes: list[dict[str, float]]) -> None:
    for box in boxes:
        for key, value in box.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"box coordinate {key}={value} is outside normalized 0..1 range")


def validate_normalized_points(points: list[dict[str, float]]) -> None:
    for point in points:
        for key, value in point.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"point coordinate {key}={value} is outside normalized 0..1 range")


def normalize_text(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def get_answer(row: dict[str, Any], field: str) -> str:
    if field not in row:
        raise ValueError(f"missing answer field {field!r}")
    return str(row[field])


def target_for_row(row: dict[str, Any], args: argparse.Namespace) -> Any:
    if args.capability == "detect":
        boxes = normalize_boxes(row.get(args.boxes_field))
        validate_normalized_boxes(boxes)
        return boxes
    if args.capability == "point":
        points = normalize_points(row.get(args.points_field))
        validate_normalized_points(points)
        return points
    return get_answer(row, args.answer_field)


def validate_rows(rows: list[dict[str, Any]], args: argparse.Namespace, label: str) -> None:
    if not rows:
        raise SystemExit(f"{label} split is empty.")
    bad = []
    positives = 0
    empty_targets = 0
    for idx, row in enumerate(rows[: min(len(rows), 50)], start=1):
        try:
            image = decode_image(row, args.image_field)
            if image.width <= 0 or image.height <= 0:
                raise ValueError("image has invalid dimensions")
            target = target_for_row(row, args)
            if args.capability in {"detect", "point"}:
                positives += int(bool(target))
                empty_targets += int(not target)
            else:
                positives += int(bool(normalize_text(target)))
                empty_targets += int(not normalize_text(target))
        except Exception as exc:  # noqa: BLE001 - show actionable row errors.
            bad.append(f"row {idx}: {exc}")
    if bad:
        raise SystemExit(f"{label} validation failed:\n" + "\n".join(bad[:10]))
    print(
        f"{label}: rows={len(rows)} checked={min(len(rows), 50)} "
        f"non_empty_targets={positives} empty_targets={empty_targets}",
        flush=True,
    )


def default_run_dir(args: argparse.Namespace) -> Path:
    name = args.name or f"moondream-{args.capability}-{args.mode}"
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name).strip("-") or "moondream"
    return DEFAULT_RUN_ROOT / f"{safe}-{int(time.time())}"


def update_latest(run_dir: Path) -> None:
    latest = run_dir.parent / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            if latest.is_dir() and not latest.is_symlink():
                shutil.rmtree(latest)
            else:
                latest.unlink()
        latest.symlink_to(run_dir.resolve(), target_is_directory=True)
    except OSError:
        pass


def materialize_row(row: dict[str, Any], args: argparse.Namespace, output_dir: Path, idx: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = decode_image(row, args.image_field)
    image_path = output_dir / f"{idx:06d}.jpg"
    image.save(image_path, quality=95)

    record: dict[str, Any] = {"image": str(image_path.resolve())}
    if args.capability == "detect":
        record["boxes"] = normalize_boxes(row.get(args.boxes_field))
    elif args.capability == "point":
        record["points"] = normalize_points(row.get(args.points_field))
    else:
        record["answer"] = get_answer(row, args.answer_field)
    return record


def write_split(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    image_dir = path.parent / "split_images" / path.stem
    with path.open("w") as handle:
        for idx, row in enumerate(rows, start=1):
            handle.write(json.dumps(materialize_row(row, args, image_dir, idx)) + "\n")


def prepare_data(args: argparse.Namespace, run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows, eval_rows, test_rows = load_rows(args)
    if eval_rows is None:
        train_rows, eval_rows, carved_test_rows = shuffled_split(
            train_rows,
            eval_ratio=args.eval_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        test_rows = test_rows or carved_test_rows
    elif test_rows is None:
        test_rows = []

    validate_rows(train_rows, args, "train")
    validate_rows(eval_rows, args, "eval")
    if test_rows:
        validate_rows(test_rows, args, "test")
    else:
        print("test: no separate untouched test split provided or carved", flush=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    write_split(run_dir / "train.jsonl", train_rows, args)
    write_split(run_dir / "eval.jsonl", eval_rows, args)
    if test_rows:
        write_split(run_dir / "test.jsonl", test_rows, args)

    manifest = {
        "capability": args.capability,
        "mode": args.mode,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "test_rows": len(test_rows),
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "test_ratio": args.test_ratio,
        "image_field": args.image_field,
        "answer_field": args.answer_field,
        "boxes_field": args.boxes_field,
        "points_field": args.points_field,
    }
    (run_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return train_rows, eval_rows, test_rows


def settings_for(args: argparse.Namespace, training: bool) -> dict[str, Any]:
    settings: dict[str, Any] = {"temperature": 1.0 if training else 0.0}
    if args.max_tokens is not None:
        settings["max_tokens"] = args.max_tokens
    elif args.capability == "detect":
        settings["max_tokens"] = 256
    else:
        settings["max_tokens"] = 128
    if args.capability == "detect" and args.max_objects is not None:
        settings["max_objects"] = args.max_objects
    return settings


def rollout_kwargs(row: dict[str, Any], args: argparse.Namespace, training: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "skill": args.capability,
        "image": decode_image(row, args.image_field),
        "settings": settings_for(args, training=training),
    }
    if args.capability == "query":
        if not args.question:
            raise SystemExit("--question is required for query finetunes.")
        kwargs["question"] = args.question
    else:
        if not args.object_name:
            raise SystemExit("--object is required for detect/point finetunes.")
        kwargs["object"] = args.object_name
    if training and args.mode == "rl":
        kwargs["num_rollouts"] = args.num_rollouts
    return kwargs


def train_request_stream(rows: list[dict[str, Any]], args: argparse.Namespace):
    while True:
        for row in rows:
            yield row, rollout_kwargs(row, args, training=True)


def score_query_rollouts(row: dict[str, Any], response: dict[str, Any], args: argparse.Namespace) -> list[float]:
    if args.reward == "judge":
        return score_query_with_judge(row, response, args)

    target = normalize_text(get_answer(row, args.answer_field))
    rewards = []
    for rollout in response["rollouts"]:
        answer = normalize_text(rollout.get("output", {}).get("answer", ""))
        rewards.append(float(bool(target) and target in answer) if args.reward == "contains" else float(answer == target))
    return rewards


def score_detection_rollouts(row: dict[str, Any], response: dict[str, Any], args: argparse.Namespace) -> list[float]:
    target = normalize_boxes(row.get(args.boxes_field))
    rewards = []
    for rollout in response["rollouts"]:
        predictions = rollout.get("output", {}).get("objects", [])
        rewards.append(float(detection_reward(predictions, target)))
    return rewards


def image_data_url(row: dict[str, Any], args: argparse.Namespace) -> str:
    image = decode_image(row, args.image_field)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def rewards_from_judge_output(output: str, count: int) -> list[float]:
    try:
        result = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge command did not emit JSON: {output[:500]}") from exc

    if isinstance(result, list):
        rewards = [float(value) for value in result]
    elif "rewards" in result:
        rewards = [float(value) for value in result["rewards"]]
    elif "ranking" in result:
        ranking = [int(value) for value in result["ranking"]]
        if sorted(ranking) != list(range(count)):
            raise ValueError(f"judge ranking must be a permutation of 0..{count - 1}")
        rewards = [0.0] * count
        if count == 1:
            raise ValueError("judge ranking is not meaningful for one rollout; return explicit rewards instead")
        else:
            for rank, index in enumerate(ranking):
                rewards[index] = 1.0 - rank / (count - 1)
    else:
        raise ValueError("judge JSON must contain rewards or ranking")

    if len(rewards) != count:
        raise ValueError(f"judge returned {len(rewards)} rewards for {count} rollouts")
    return [max(0.0, min(1.0, value)) for value in rewards]


def score_query_with_judge(row: dict[str, Any], response: dict[str, Any], args: argparse.Namespace) -> list[float]:
    if not args.judge_command:
        raise SystemExit(
            "Query judge rewards require --judge-command. Prefer exact/contains rewards when labels are simple."
        )

    answers = [rollout.get("output", {}).get("answer", "") for rollout in response["rollouts"]]
    payload = {
        "image_data_url": image_data_url(row, args),
        "question": args.question,
        "candidates": answers,
        "reference": row.get(args.answer_field, ""),
        "rubric": args.judge_rubric,
        "row_id": row.get("_row_id"),
    }
    completed = subprocess.run(
        shlex.split(args.judge_command),
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "query judge failed with exit code "
            f"{completed.returncode}: {completed.stderr.strip()[:500]}"
        )
    return rewards_from_judge_output(completed.stdout.strip(), len(answers))


def point_similarity(prediction: dict[str, float], target: dict[str, float]) -> float:
    distance = math.hypot(prediction["x"] - target["x"], prediction["y"] - target["y"])
    return max(0.0, 1.0 - distance / 0.25)


def point_reward(predictions: list[dict[str, float]], targets: list[dict[str, float]]) -> float:
    if not targets:
        return 1.0 if not predictions else max(0.0, 0.5 - 0.2 * (len(predictions) - 1))
    if not predictions:
        return 0.0

    pairs = sorted(
        (
            (point_similarity(prediction, target), pred_idx, target_idx)
            for pred_idx, prediction in enumerate(predictions)
            for target_idx, target in enumerate(targets)
        ),
        reverse=True,
    )
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    total = 0.0
    for value, pred_idx, target_idx in pairs:
        if pred_idx in used_predictions or target_idx in used_targets:
            continue
        used_predictions.add(pred_idx)
        used_targets.add(target_idx)
        total += value
    return total / max(len(predictions), len(targets))


def score_point_rollouts(row: dict[str, Any], response: dict[str, Any], args: argparse.Namespace) -> list[float]:
    target = normalize_points(row.get(args.points_field))
    rewards = []
    for rollout in response["rollouts"]:
        predictions = rollout.get("output", {}).get("points", [])
        rewards.append(float(point_reward(predictions, target)))
    return rewards


def score_rollouts(row: dict[str, Any], response: dict[str, Any], args: argparse.Namespace) -> list[float]:
    if args.capability == "detect":
        return score_detection_rollouts(row, response, args)
    if args.capability == "point":
        return score_point_rollouts(row, response, args)
    return score_query_rollouts(row, response, args)


def sft_group(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    request = rollout_kwargs(row, args, training=False)
    request.pop("num_rollouts", None)

    if args.capability == "detect":
        target = {"boxes": normalize_boxes(row.get(args.boxes_field))}
    elif args.capability == "point":
        target = {"points": normalize_points(row.get(args.points_field))}
    else:
        target = {"answer": get_answer(row, args.answer_field)}

    return {"mode": "sft", "request": request, "target": target}


def reward_stats(rewards: list[float]) -> dict[str, float | None]:
    if not rewards:
        return {"reward_mean": None, "reward_min": None, "reward_max": None}
    return {
        "reward_mean": sum(rewards) / len(rewards),
        "reward_min": min(rewards),
        "reward_max": max(rewards),
    }


def health_flags(history: list[dict[str, Any]], window: int = 5) -> list[str]:
    recent = [h for h in history[-window:] if h.get("reward_mean") is not None]
    if len(recent) < window:
        return []
    means = [float(h["reward_mean"]) for h in recent]
    flags = []
    if all(value <= 0.01 for value in means):
        flags.append("all_recent_rewards_near_zero")
    if statistics.pstdev(means) <= 1e-6:
        flags.append("recent_reward_flat")
    if means[-1] < means[0] - 0.2:
        flags.append("reward_drop")
    return flags


def log_record(metrics_path: Path, record: dict[str, Any]) -> None:
    with metrics_path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    reward_mean = record.get("reward_mean")
    reward_text = "reward_mean=n/a" if reward_mean is None else f"reward_mean={reward_mean:.3f}"
    print(
        f"[step {record['step']:4d}] {reward_text} "
        f"min={record.get('reward_min')} max={record.get('reward_max')} "
        f"lr={record['lr']} eval={record.get('eval_metric')} flags={','.join(record['health_flags']) or '-'}",
        flush=True,
    )
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(record)
    except ImportError:
        pass


def evaluate(ft: Any, rows: list[dict[str, Any]], args: argparse.Namespace) -> float:
    if args.eval_limit:
        rows = rows[: args.eval_limit]
    if not rows:
        return float("nan")

    scores = []
    for row in rows:
        response = ft.rollouts(**rollout_kwargs(row, args, training=False))
        fake_response = {"rollouts": response["rollouts"]}
        rewards = score_rollouts(row, fake_response, args)
        scores.append(rewards[0] if rewards else 0.0)
    return sum(scores) / len(scores)


def create_or_resume_ft(md: Any, args: argparse.Namespace):
    api_key = os.environ.get("MOONDREAM_API_KEY")
    if not api_key:
        raise SystemExit("MOONDREAM_API_KEY is not set. Export it in the shell; do not paste it in chat.")

    if args.finetune_id:
        return md.ft(api_key=api_key, finetune_id=args.finetune_id)

    if args.rank not in VALID_RANKS:
        raise SystemExit("--rank must be one of 8, 16, 24, or 32.")

    name = args.name or f"moondream-{args.capability}-{args.mode}-{int(time.time())}"
    return md.ft(api_key=api_key, name=name, rank=args.rank)


def save_checkpoint(ft: Any, step: int, run_dir: Path) -> int:
    checkpoint = ft.save_checkpoint()["checkpoint"]
    model_id = ft.model(checkpoint["step"])
    record = {
        "step": step,
        "checkpoint_step": checkpoint["step"],
        "checkpoint_id": checkpoint["checkpoint_id"],
        "model_id": model_id,
        "ts": time.time(),
    }
    with (run_dir / "checkpoints.jsonl").open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"saved checkpoint step={checkpoint['step']} model_id={model_id}", flush=True)
    return int(checkpoint["step"])


def preflight_before_paid_training(
    args: argparse.Namespace,
    test_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
) -> None:
    if not args.yes:
        raise SystemExit(
            "Lens training runs in Moondream Cloud and may spend credits. "
            "Rerun with --yes after the user confirms cost."
        )
    if not test_rows and not args.allow_missing_test:
        raise SystemExit(
            "No untouched test split is available. Provide --test-split, let the script "
            "carve one from unsplit data, or rerun with --allow-missing-test after "
            "explicitly accepting that risk."
        )
    if args.reward == "judge" and args.capability != "query":
        raise SystemExit("--reward judge is only supported for query finetunes.")
    if args.reward == "judge" and not args.judge_command:
        raise SystemExit(
            "Query judge rewards require --judge-command. Use --reward exact or --reward contains "
            "for simple tags/classes."
        )
    if args.judge_command:
        command = shlex.split(args.judge_command)
        if not command:
            raise SystemExit("--judge-command is empty.")
        executable = command[0]
        if not Path(executable).exists() and shutil.which(executable) is None:
            raise SystemExit(f"Judge command executable not found: {executable}")
    if args.reward == "judge":
        fake_response = {"rollouts": [{"output": {"answer": get_answer(train_rows[0], args.answer_field)}}]}
        score_query_rollouts(train_rows[0], fake_response, args)
        print("judge_preflight_ok=true", flush=True)


def run_training(args: argparse.Namespace, train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], run_dir: Path) -> None:
    md = import_moondream_sdk()
    ft = create_or_resume_ft(md, args)
    print(f"finetune_id={ft.finetune_id} name={getattr(ft, 'name', None)}", flush=True)

    metrics_path = run_dir / "metrics.jsonl"
    history: list[dict[str, Any]] = []
    best_eval = -float("inf")
    stale_evals = 0
    last_train_step: int | None = None
    last_checkpoint_step: int | None = None

    if args.mode == "rl":
        stream = ft.rollout_stream(train_request_stream(train_rows, args))
        batch = []
        batch_rewards: list[float] = []
        local_steps_done = 0
        while local_steps_done < args.steps:
            row, response = next(stream)
            rewards = score_rollouts(row, response, args)
            batch_rewards.extend(rewards)
            batch.append({
                "mode": "rl",
                "request": response["request"],
                "rollouts": response["rollouts"],
                "rewards": rewards,
            })
            if len(batch) < args.batch_size:
                continue

            step_response = ft.train_step(batch, lr=args.lr)
            step = int(step_response["step"])
            local_steps_done += 1
            last_train_step = step
            stats = reward_stats(batch_rewards)
            batch = []
            batch_rewards = []

            eval_metric = None
            if local_steps_done % args.eval_every == 0 or local_steps_done == args.steps:
                eval_metric = evaluate(ft, eval_rows, args)
                ft.log_metrics(step=step, metrics={"eval/metric": eval_metric})
                if eval_metric > best_eval:
                    best_eval = eval_metric
                    stale_evals = 0
                else:
                    stale_evals += 1

            record = {
                "step": step,
                "mode": args.mode,
                "capability": args.capability,
                "lr": args.lr,
                "num_rollouts": args.num_rollouts,
                "eval_metric": eval_metric,
                "ts": time.time(),
                "health_flags": [],
                **stats,
                "train_response": step_response,
            }
            history.append(record)
            record["health_flags"] = health_flags(history)
            log_record(metrics_path, record)

            if step % CHECKPOINT_EVERY == 0 or local_steps_done == args.steps:
                last_checkpoint_step = save_checkpoint(ft, step, run_dir)
            if stale_evals >= args.patience:
                print(f"early stop: eval metric failed to improve for {args.patience} evals", flush=True)
                break
    else:
        row_cycle = itertools.cycle(train_rows)
        for local_steps_done in range(1, args.steps + 1):
            groups = [sft_group(next(row_cycle), args) for _ in range(args.batch_size)]
            step_response = ft.train_step(groups, lr=args.lr)
            step = int(step_response["step"])
            last_train_step = step
            eval_metric = None
            if local_steps_done % args.eval_every == 0 or local_steps_done == args.steps:
                eval_metric = evaluate(ft, eval_rows, args)
                ft.log_metrics(step=step, metrics={"eval/metric": eval_metric})
                if eval_metric > best_eval:
                    best_eval = eval_metric
                    stale_evals = 0
                else:
                    stale_evals += 1

            record = {
                "step": step,
                "mode": args.mode,
                "capability": args.capability,
                "lr": args.lr,
                "num_rollouts": 0,
                "reward_mean": None,
                "reward_min": None,
                "reward_max": None,
                "eval_metric": eval_metric,
                "sft_loss": step_response.get("sft_loss"),
                "health_flags": [],
                "train_response": step_response,
                "ts": time.time(),
            }
            history.append(record)
            log_record(metrics_path, record)

            if step % CHECKPOINT_EVERY == 0 or local_steps_done == args.steps:
                last_checkpoint_step = save_checkpoint(ft, step, run_dir)
            if stale_evals >= args.patience:
                print(f"early stop: eval metric failed to improve for {args.patience} evals", flush=True)
                break

    if last_train_step is not None and last_checkpoint_step != last_train_step:
        save_checkpoint(ft, last_train_step, run_dir)


def main() -> None:
    args = parse_args()
    if args.capability == "query" and not args.question:
        raise SystemExit("--question is required for query finetunes.")
    if args.reward == "judge" and args.capability != "query":
        raise SystemExit("--reward judge is only supported for query finetunes.")
    if args.capability in {"detect", "point"} and not args.object_name:
        raise SystemExit("--object is required for detect/point finetunes.")
    if args.num_rollouts < 1 or args.num_rollouts > 16:
        raise SystemExit("--num-rollouts must be between 1 and 16.")

    run_dir = Path(args.run_dir).resolve() if args.run_dir else default_run_dir(args).resolve()
    train_rows, eval_rows, test_rows = prepare_data(args, run_dir)
    update_latest(run_dir)
    print(f"run_dir={run_dir}", flush=True)

    if args.dry_run:
        print("dry_run_ok=true", flush=True)
        return

    preflight_before_paid_training(args, test_rows, train_rows)
    run_training(args, train_rows, eval_rows, run_dir)


if __name__ == "__main__":
    main()
