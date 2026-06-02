"""Example query-judge command contract.

This file is intentionally small. For a real finetune, write a task-specific
judge script in the user's project with a prompt, provider, and rubric matched
to their data. The training loop only requires this interface:

stdin:
{"question": "...", "candidates": ["..."], "reference": "...", "rubric": "..."}

stdout:
{"rewards": [0.0, 1.0]}
"""

from __future__ import annotations

import json
import sys


def score_candidate(candidate: str, reference: str) -> float:
    """Tiny placeholder scorer for contract testing, not a production judge."""
    candidate_terms = set(candidate.lower().split())
    reference_terms = set(reference.lower().split())
    if not reference_terms:
        return 0.0
    return len(candidate_terms & reference_terms) / len(reference_terms)


def main() -> None:
    payload = json.loads(sys.stdin.read())
    candidates = payload.get("candidates") or []
    reference = str(payload.get("reference", ""))
    if not candidates:
        raise SystemExit("payload must include candidates")
    rewards = [max(0.0, min(1.0, score_candidate(str(candidate), reference))) for candidate in candidates]
    print(json.dumps({"rewards": rewards}, sort_keys=True))


if __name__ == "__main__":
    main()
