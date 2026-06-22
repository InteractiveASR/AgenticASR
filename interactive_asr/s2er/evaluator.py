#!/usr/bin/env python3
"""Sentence-level semantic evaluation utilities for Interactive ASR."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from interactive_asr.agentic_asr.api_clients import call_judge_with_consensus
from interactive_asr.agentic_asr.norm import norm

logger = logging.getLogger(__name__)


def load_prompts_from_file(prompts_path: str = "config/default_prompts.json") -> dict:
    prompts_file = Path(prompts_path)
    if not prompts_file.exists():
        raise FileNotFoundError(f"default_prompts.json file not found: {prompts_path}")

    with open(prompts_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_judge_prompt(prompts_path: Optional[str] = None) -> str:
    prompts_path = prompts_path or "config/default_prompts.json"
    prompts = load_prompts_from_file(prompts_path)
    if "judge" not in prompts or "system_prompt" not in prompts["judge"]:
        raise KeyError("default_prompts.json is missing judge.system_prompt")
    return prompts["judge"]["system_prompt"]


def get_current_prediction(item: Dict) -> Optional[str]:
    loop_keys = [k for k in item.keys() if k.startswith("loop_") and k.endswith("_pred")]
    if loop_keys:
        sorted_keys = sorted(loop_keys, key=lambda x: int(x.split("_")[1]))
        return item[sorted_keys[-1]]
    if "raw_pred" in item:
        return item["raw_pred"]
    return None


def calculate_is_correct(pred: str, gt: str) -> bool:
    return norm(pred) == norm(gt)


def evaluate_item(
    item: Dict,
    use_semantic_judge: bool = True,
    judge_prompt: Optional[str] = None,
    judge_k_rounds: int = 3,
) -> Dict:
    pred = get_current_prediction(item)
    if pred is None:
        logger.warning("Sample %s has no prediction", item.get("id"))
        return item

    gt = item.get("gt")
    if not gt:
        logger.warning("Sample %s has no ground truth", item.get("id"))
        return item

    is_correct = calculate_is_correct(pred, gt)
    result = {**item, "is_correct": is_correct}

    if is_correct:
        result["is_semantic_correct"] = True
    elif item.get("is_semantic_correct", False):
        result["is_semantic_correct"] = True
    elif use_semantic_judge:
        try:
            result["is_semantic_correct"] = call_judge_with_consensus(
                pred,
                gt,
                system_prompt=judge_prompt,
                k=judge_k_rounds,
                timeout=30,
            )
        except Exception as exc:
            logger.warning("Semantic judge failed for ID=%s: %s", item.get("id"), exc)
            result["is_semantic_correct"] = False

    if "total_loop" not in result:
        result["total_loop"] = 0

    return result


def evaluate_items_concurrent(
    items: List[Dict],
    use_semantic_judge: bool = True,
    judge_prompt: Optional[str] = None,
    judge_k_rounds: int = 3,
    concurrency: int = 256,
) -> List[Dict]:
    print(f"Concurrent evaluation (workers={concurrency})...")
    results: List[Optional[Dict]] = [None] * len(items)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_index = {
            executor.submit(evaluate_item, item, use_semantic_judge, judge_prompt, judge_k_rounds): i
            for i, item in enumerate(items)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                logger.error("Failed to evaluate index %s: %s", index, exc, exc_info=True)
                results[index] = items[index]

    return [r for r in results if r is not None]


def evaluate_items_serial(
    items: List[Dict],
    use_semantic_judge: bool = True,
    judge_prompt: Optional[str] = None,
    judge_k_rounds: int = 3,
) -> List[Dict]:
    results = []
    for item in items:
        results.append(evaluate_item(item, use_semantic_judge, judge_prompt, judge_k_rounds))
    return results


def calculate_metrics(results: List[Dict]) -> Dict:
    valid_results = [r for r in results if get_current_prediction(r) is not None]
    total = len(valid_results)

    if total == 0:
        return {
            "total": 0,
            "is_correct_count": 0,
            "is_correct_rate": 0.0,
            "ser": 0.0,
            "semantic_judged": 0,
            "semantic_correct": 0,
            "semantic_correct_rate": 0.0,
        }

    is_correct_count = sum(1 for r in valid_results if r.get("is_correct", False))
    semantic_results = [r for r in valid_results if "is_semantic_correct" in r]
    semantic_judged = len(semantic_results)
    semantic_correct = sum(1 for r in semantic_results if r.get("is_semantic_correct"))

    return {
        "total": total,
        "is_correct_count": is_correct_count,
        "is_correct_rate": is_correct_count / total * 100,
        "ser": (total - is_correct_count) / total * 100,
        "semantic_judged": semantic_judged,
        "semantic_correct": semantic_correct,
        "semantic_correct_rate": semantic_correct / semantic_judged * 100 if semantic_judged > 0 else 0.0,
    }


def print_report(results: List[Dict], metrics: Dict):
    print("\n" + "=" * 80)
    print("S²ER Evaluation Report")
    print("=" * 80 + "\n")
    print(f"Total samples: {len(results)}")
    print(f"Valid samples: {metrics['total']}")
    print(f"Exact-match accuracy: {metrics['is_correct_rate']:.2f}%")
    print(f"SER: {metrics['ser']:.2f}%")
    if metrics["semantic_judged"] > 0:
        print(f"Semantic correctness rate: {metrics['semantic_correct_rate']:.2f}%")
    print("=" * 80)
