"""Reward helpers for Moondream/Lens finetuning.

Boxes are [x_min, y_min, x_max, y_max] in the normalized coordinate convention
returned by model.detect(...). Higher reward is better.
"""

from __future__ import annotations

from typing import Mapping, Sequence

Box = Sequence[float] | Mapping[str, float]


def _xyxy(box: Box) -> tuple[float, float, float, float]:
    if isinstance(box, Mapping):
        return (
            float(box["x_min"]),
            float(box["y_min"]),
            float(box["x_max"]),
            float(box["y_max"]),
        )
    return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))


def iou(a: Box, b: Box) -> float:
    ax0, ay0, ax1, ay1 = _xyxy(a)
    bx0, by0, bx1, by1 = _xyxy(b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _matched_iou_sum(predictions: list[Box], ground_truth: list[Box]) -> float:
    """Sum IoU over the one-to-one match that maximizes total IoU."""
    if not predictions or not ground_truth:
        return 0.0
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        scores = np.array([[iou(p, g) for g in ground_truth] for p in predictions])
        rows, cols = linear_sum_assignment(-scores)
        return float(scores[rows, cols].sum())
    except ImportError:
        pairs = sorted(
            (
                (iou(pred, gt), pred_idx, gt_idx)
                for pred_idx, pred in enumerate(predictions)
                for gt_idx, gt in enumerate(ground_truth)
            ),
            reverse=True,
        )
        used_predictions: set[int] = set()
        used_ground_truth: set[int] = set()
        total = 0.0
        for value, pred_idx, gt_idx in pairs:
            if pred_idx in used_predictions or gt_idx in used_ground_truth:
                continue
            used_predictions.add(pred_idx)
            used_ground_truth.add(gt_idx)
            total += value
        return total


def detection_reward(predictions: list[Box], ground_truth: list[Box]) -> float:
    """Mean-IoU detection reward with centered empty-image behavior."""
    ground_truth_count = len(ground_truth)
    prediction_count = len(predictions)

    if ground_truth_count == 0:
        if prediction_count == 0:
            return 1.0
        return max(0.0, 0.5 - 0.2 * (prediction_count - 1))

    return _matched_iou_sum(predictions, ground_truth) / max(
        ground_truth_count,
        prediction_count,
    )


def _assert_close(actual: float, expected: float) -> None:
    if abs(actual - expected) > 1e-9:
        raise AssertionError(f"expected {expected}, got {actual}")


def _run_anchor_tests() -> None:
    perfect = [0.0, 0.0, 1.0, 1.0]
    extra = [0.0, 0.0, 0.5, 0.5]

    _assert_close(detection_reward([], []), 1.0)
    _assert_close(detection_reward([perfect], []), 0.5)
    _assert_close(detection_reward([perfect, extra], []), 0.3)
    _assert_close(detection_reward([perfect], [perfect]), 1.0)
    _assert_close(detection_reward([perfect, extra], [perfect]), 0.5)

    dict_box = {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}
    _assert_close(detection_reward([dict_box], [dict_box]), 1.0)


if __name__ == "__main__":
    _run_anchor_tests()
    print("reward anchor tests passed")
