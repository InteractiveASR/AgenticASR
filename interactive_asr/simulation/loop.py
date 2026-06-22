#!/usr/bin/env python3
"""Multi-turn interactive simulation loop."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from interactive_asr.agentic_asr import ASRAgent, HumanAgent
from interactive_asr.agentic_asr.api_clients import call_asr
from interactive_asr.simulation.io import get_current_prediction

logger = logging.getLogger(__name__)


def process_next_turn(
    item: Dict,
    human_agent: HumanAgent,
    asr_agent: ASRAgent,
    speaker_path: Optional[str],
    max_turns: int,
    audio_dir: str,
) -> Dict:
    is_correct = item.get("is_correct", False)
    is_semantic_correct = item.get("is_semantic_correct", False)
    current_loop = item.get("total_loop", 0)

    if is_correct or is_semantic_correct or current_loop >= max_turns:
        return item

    current_pred = get_current_prediction(item)
    gt = item.get("gt")
    if not current_pred or not gt:
        return item

    result = {**item}
    next_loop = current_loop + 1

    try:
        speaker_ref_path = item.get("audio_path") or speaker_path
        human_response = human_agent.process(
            gt=gt,
            on_screen_text=current_pred,
            speaker_path=speaker_ref_path,
            generate_audio=True,
        )

        Path(audio_dir).mkdir(parents=True, exist_ok=True)
        correction_audio_path = Path(audio_dir) / f"correction_{item.get('id', 'unknown')}_{next_loop}.wav"
        with open(correction_audio_path, "wb") as f:
            f.write(human_response.correction_audio)

        start_asr = time.time()
        language, asr_text = call_asr(str(correction_audio_path))
        correction_latency = time.time() - start_asr

        try:
            asr_agent_response = asr_agent.process(
                audio_path=str(correction_audio_path),
                on_screen_text=current_pred,
            )
            refined_text = asr_agent_response.on_screen_text
            result[f"loop_{next_loop}_asr_refine_think"] = asr_agent_response.thinking
            result[f"loop_{next_loop}_asr_refine_is_affirmation"] = asr_agent_response.is_affirmation
            result[f"loop_{next_loop}_asr_refine_latency_asr"] = asr_agent_response.latency_asr
            result[f"loop_{next_loop}_asr_refine_latency_llm"] = asr_agent_response.latency_llm
        except Exception as exc:
            logger.warning("ASR agent failed for ID=%s: %s", item.get("id"), exc)
            refined_text = asr_text
            result[f"loop_{next_loop}_asr_refine_think"] = None
            result[f"loop_{next_loop}_asr_refine_is_affirmation"] = None
            result[f"loop_{next_loop}_asr_refine_latency_asr"] = None
            result[f"loop_{next_loop}_asr_refine_latency_llm"] = None

        result[f"loop_{next_loop}_pred"] = refined_text
        result["total_loop"] = next_loop
        result[f"loop_{next_loop}_human_think"] = human_response.thinking
        result[f"loop_{next_loop}_human_correction"] = human_response.correction_text
        result[f"loop_{next_loop}_human_latency_llm"] = human_response.latency_llm
        result[f"loop_{next_loop}_human_latency_tts"] = human_response.latency_tts
        result[f"loop_{next_loop}_correction_audio"] = str(correction_audio_path)
        result[f"loop_{next_loop}_correction_asr"] = asr_text
        result[f"loop_{next_loop}_correction_asr_language"] = language
        result[f"loop_{next_loop}_correction_asr_latency"] = correction_latency
        result[f"loop_{next_loop}_asr_refine"] = refined_text
    except Exception as exc:
        logger.error("Loop processing failed for ID=%s: %s", item.get("id"), exc, exc_info=True)
        result[f"loop_{next_loop}_pred"] = current_pred
        result["total_loop"] = next_loop

    return result


def run_next_loop(
    items: List[Dict],
    max_turns: int = 3,
    concurrency: int = 256,
    speaker_path: Optional[str] = None,
    prompts_path: Optional[str] = None,
    audio_dir: str = "./temp_audio",
    enable_thinking: bool = True,
) -> List[Dict]:
    human_agent = HumanAgent(
        default_speaker_path=speaker_path,
        prompts_path=prompts_path,
        enable_thinking=enable_thinking,
    )
    asr_agent = ASRAgent(prompts_path=prompts_path, enable_thinking=enable_thinking)

    items_to_process = [
        item for item in items
        if not item.get("is_correct", False)
        and not item.get("is_semantic_correct", False)
        and item.get("total_loop", 0) < max_turns
    ]
    items_to_skip = [
        item for item in items
        if item.get("is_correct", False)
        or item.get("is_semantic_correct", False)
        or item.get("total_loop", 0) >= max_turns
    ]

    if not items_to_process:
        return items

    results: List[Optional[Dict]] = [None] * len(items_to_process)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_index = {
            executor.submit(
                process_next_turn,
                item,
                human_agent,
                asr_agent,
                speaker_path,
                max_turns,
                audio_dir,
            ): i
            for i, item in enumerate(items_to_process)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                logger.error("Loop worker failed at index %s: %s", index, exc, exc_info=True)
                results[index] = items_to_process[index]

    return items_to_skip + [r for r in results if r is not None]
