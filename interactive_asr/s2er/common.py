#!/usr/bin/env python3
"""Shared helpers for S²ER modules."""

from __future__ import annotations

from typing import Dict, Optional


def get_current_prediction(item: Dict) -> Optional[str]:
    loop_keys = [k for k in item.keys() if k.startswith("loop_") and k.endswith("_pred")]
    if loop_keys:
        sorted_keys = sorted(loop_keys, key=lambda x: int(x.split("_")[1]))
        return item[sorted_keys[-1]]
    if "raw_pred" in item:
        return item["raw_pred"]
    return None
