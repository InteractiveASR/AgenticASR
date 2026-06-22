#!/usr/bin/env python3
"""Human-alignment helpers for S²ER analysis."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_human_alignment_csv(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_binary(value) -> int:
    if isinstance(value, bool):
        return int(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "correct", "semantic_correct"}:
        return 1
    if s in {"0", "false", "no", "wrong", "semantic_error"}:
        return 0
    raise ValueError(f"Unsupported binary label: {value}")


def accuracy(human: Iterable, judge: Iterable) -> float:
    pairs = list(zip(human, judge))
    if not pairs:
        return 0.0
    return sum(int(h == j) for h, j in pairs) / len(pairs)


def pearson(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or not x:
        return 0.0
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    num = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - x_mean) ** 2 for a in x))
    den_y = math.sqrt(sum((b - y_mean) ** 2 for b in y))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def cohen_kappa(human: List[int], judge: List[int]) -> float:
    if len(human) != len(judge) or not human:
        return 0.0
    po = accuracy(human, judge)
    human_pos = sum(human) / len(human)
    judge_pos = sum(judge) / len(judge)
    human_neg = 1 - human_pos
    judge_neg = 1 - judge_pos
    pe = human_pos * judge_pos + human_neg * judge_neg
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def evaluate_human_alignment(records: Iterable[Dict], human_key: str = "human_label", judge_key: str = "judge_label") -> Dict:
    records = list(records)
    human = [_to_binary(item[human_key]) for item in records]
    judge = [_to_binary(item[judge_key]) for item in records]
    return {
        "n": len(records),
        "accuracy": accuracy(human, judge) * 100,
        "pearson": pearson(human, judge),
        "cohen_kappa": cohen_kappa(human, judge),
    }
