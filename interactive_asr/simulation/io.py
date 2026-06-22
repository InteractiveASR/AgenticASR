#!/usr/bin/env python3
"""Shared IO helpers for simulation experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def save_jsonl(items: List[Dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_current_prediction(item: Dict) -> Optional[str]:
    loop_keys = [k for k in item.keys() if k.startswith("loop_") and k.endswith("_pred")]
    if loop_keys:
        sorted_keys = sorted(loop_keys, key=lambda x: int(x.split("_")[1]))
        return item[sorted_keys[-1]]
    if "raw_pred" in item:
        return item["raw_pred"]
    return None
