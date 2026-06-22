#!/usr/bin/env python3
"""Formal S²ER metric utilities."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .common import get_current_prediction


def semantic_error_indicator(item: Dict) -> Optional[int]:
    if "is_semantic_correct" not in item:
        return None
    return 0 if item.get("is_semantic_correct") else 1


def exact_error_indicator(item: Dict) -> Optional[int]:
    if "is_correct" not in item:
        return None
    return 0 if item.get("is_correct") else 1


def compute_s2er(items: Iterable[Dict]) -> Dict:
    items = list(items)
    valid = [item for item in items if get_current_prediction(item) is not None]
    semantic_labeled = [item for item in valid if "is_semantic_correct" in item]
    exact_labeled = [item for item in valid if "is_correct" in item]

    if not valid:
        return {
            "total": 0,
            "semantic_labeled": 0,
            "s2er_count": 0,
            "s2er_rate": 0.0,
            "ser_count": 0,
            "ser_rate": 0.0,
        }

    s2er_count = sum(semantic_error_indicator(item) or 0 for item in semantic_labeled)
    ser_count = sum(exact_error_indicator(item) or 0 for item in exact_labeled)

    return {
        "total": len(valid),
        "semantic_labeled": len(semantic_labeled),
        "s2er_count": s2er_count,
        "s2er_rate": s2er_count / len(semantic_labeled) * 100 if semantic_labeled else 0.0,
        "ser_count": ser_count,
        "ser_rate": ser_count / len(exact_labeled) * 100 if exact_labeled else 0.0,
    }


def compute_loopwise_s2er(items: Iterable[Dict]) -> List[Dict]:
    items = list(items)
    max_loop = max((item.get("total_loop", 0) for item in items), default=0)
    reports = []
    for loop_id in range(max_loop + 1):
        loop_items = [item for item in items if item.get("total_loop", 0) >= loop_id]
        metric = compute_s2er(loop_items)
        metric["loop"] = loop_id
        reports.append(metric)
    return reports


def summarize_s2er_delta(before: Dict, after: Dict) -> Dict:
    return {
        "s2er_before": before.get("s2er_rate", 0.0),
        "s2er_after": after.get("s2er_rate", 0.0),
        "s2er_reduction": before.get("s2er_rate", 0.0) - after.get("s2er_rate", 0.0),
        "ser_before": before.get("ser_rate", 0.0),
        "ser_after": after.get("ser_rate", 0.0),
        "ser_reduction": before.get("ser_rate", 0.0) - after.get("ser_rate", 0.0),
    }
