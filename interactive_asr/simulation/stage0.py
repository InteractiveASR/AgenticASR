#!/usr/bin/env python3
"""Stage-0 decoding for Interactive ASR benchmarks."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from interactive_asr.agentic_asr.api_clients import call_asr
from interactive_asr.simulation.io import load_jsonl

logger = logging.getLogger(__name__)


def process_single_item(item: Dict, index: int) -> Dict:
    audio_path = item.get("audio_path")
    result = {
        "id": item.get("id"),
        "gt": item.get("gt"),
        "audio_path": audio_path,
        "category": item.get("category"),
        "difficulty": item.get("difficulty"),
        "metadata": item.get("metadata", {}),
        "raw_pred": "",
        "latency_asr": 0.0,
        "error": None,
    }

    try:
        if not audio_path or not Path(audio_path).exists():
            result["error"] = f"audio file not found: {audio_path}"
            return result

        start_time = time.time()
        _, asr_text = call_asr(audio_path)
        result["raw_pred"] = asr_text
        result["latency_asr"] = time.time() - start_time
    except Exception as exc:
        result["error"] = str(exc)

    return result


def run_stage0(
    data_path: str,
    start_index: int = 0,
    max_samples: Optional[int] = None,
    concurrency: int = 8,
) -> List[Dict]:
    items = load_jsonl(data_path)
    if start_index >= len(items):
        return []

    if max_samples:
        items = items[start_index:start_index + max_samples]
    else:
        items = items[start_index:]

    results: List[Optional[Dict]] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_index = {
            executor.submit(process_single_item, item, i): i
            for i, item in enumerate(items)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                logger.error("Stage-0 failed at index %s: %s", index, exc, exc_info=True)
                results[index] = {
                    "id": items[index].get("id"),
                    "gt": items[index].get("gt"),
                    "audio_path": items[index].get("audio_path"),
                    "raw_pred": "",
                    "latency_asr": 0.0,
                    "error": str(exc),
                }

    return [r for r in results if r is not None]
