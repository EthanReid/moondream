"""Inspect Moondream Lens metrics.jsonl health."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor a Moondream metrics.jsonl file.")
    parser.add_argument("metrics_jsonl")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--plot", help="Optional PNG output path if matplotlib is installed.")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    lines = path.read_text().splitlines()
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if idx == len(lines) - 1:
                break
            raise
    return records


def trend(values: list[float]) -> str:
    if len(values) < 2:
        return "insufficient"
    delta = values[-1] - values[0]
    if delta > 0.05:
        return "improving"
    if delta < -0.05:
        return "dropping"
    return "flat"


def health(records: list[dict[str, Any]], window: int) -> list[str]:
    recent = records[-window:]
    reward_values = [
        float(record["reward_mean"])
        for record in recent
        if record.get("reward_mean") is not None
    ]
    eval_values = [
        float(record["eval_metric"])
        for record in records
        if record.get("eval_metric") is not None
    ]
    flags = []
    if len(reward_values) >= window and all(value <= 0.01 for value in reward_values):
        flags.append("recent_rewards_near_zero")
    if len(reward_values) >= window and statistics.pstdev(reward_values) <= 1e-6:
        flags.append("recent_rewards_flat")
    if len(eval_values) >= 3 and eval_values[-1] < max(eval_values[:-1]) - 0.05:
        flags.append("eval_regressed_from_best")
    if not eval_values:
        flags.append("no_eval_metrics_logged")
    return flags


def maybe_plot(records: list[dict[str, Any]], output: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot")
        return

    steps = [record["step"] for record in records]
    rewards = [record.get("reward_mean") for record in records]
    eval_steps = [record["step"] for record in records if record.get("eval_metric") is not None]
    evals = [record["eval_metric"] for record in records if record.get("eval_metric") is not None]

    plt.figure(figsize=(8, 4))
    if any(value is not None for value in rewards):
        plt.plot(
            [step for step, value in zip(steps, rewards) if value is not None],
            [value for value in rewards if value is not None],
            label="train reward",
        )
    if evals:
        plt.plot(eval_steps, evals, label="eval metric", marker="o")
    plt.xlabel("step")
    plt.ylabel("score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    print(f"wrote plot: {output}")


def main() -> None:
    args = parse_args()
    path = Path(args.metrics_jsonl)
    records = load_records(path)
    if not records:
        raise SystemExit(f"No metrics found in {path}")

    latest = records[-1]
    reward_values = [
        float(record["reward_mean"])
        for record in records[-args.window :]
        if record.get("reward_mean") is not None
    ]
    eval_values = [
        float(record["eval_metric"])
        for record in records
        if record.get("eval_metric") is not None
    ]
    flags = health(records, args.window)

    print(f"records={len(records)} latest_step={latest['step']}")
    if reward_values:
        print(
            f"recent_reward_mean={sum(reward_values) / len(reward_values):.3f} "
            f"trend={trend(reward_values)}"
        )
    if eval_values:
        print(
            f"best_eval={max(eval_values):.3f} latest_eval={eval_values[-1]:.3f} "
            f"trend={trend(eval_values[-min(len(eval_values), args.window):])}"
        )
    print("health_flags=" + (",".join(flags) if flags else "-"))

    if args.plot:
        maybe_plot(records, args.plot)


if __name__ == "__main__":
    main()
