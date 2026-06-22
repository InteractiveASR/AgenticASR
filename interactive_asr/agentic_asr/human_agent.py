#!/usr/bin/env python3
"""
Human Agent类 - 负责生成自然语言纠错
"""

import re
import json
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass

from .api_clients import call_llm, call_tts, LLM_HUMAN_MODEL_NAME, LLM_HUMAN_BASE_URL, is_llm_response_failed

# ==================== Prompt加载函数 ====================
def load_prompts_from_file(prompts_path: str = "config/default_prompts.json") -> dict:
    """
    从JSON文件加载prompt配置

    Args:
        prompts_path: default_prompts.json 文件路径

    Returns:
        包含 asr_agent 和 human_agent system_prompt 的字典
    """
    prompts_file = Path(prompts_path)
    if not prompts_file.exists():
        return {}

    try:
        with open(prompts_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"警告: 加载 default_prompts.json 失败: {e}")
        return {}


def get_human_prompt(prompts_path: Optional[str] = None) -> str:
    """
    获取Human Agent的system prompt

    注意：不使用 fallback，如果 default_prompts.json 不存在或格式错误将抛出异常。

    Args:
        prompts_path: default_prompts.json 文件路径（默认"config/default_prompts.json"）

    Returns:
        system prompt字符串

    Raises:
        FileNotFoundError: 如果 default_prompts.json 文件不存在
        KeyError: 如果 default_prompts.json 中没有 human_agent.system_prompt
    """
    if prompts_path is None:
        prompts_path = "config/default_prompts.json"

    prompts = load_prompts_from_file(prompts_path)
    if "human_agent" not in prompts or "system_prompt" not in prompts["human_agent"]:
        raise KeyError("default_prompts.json 中未找到 human_agent.system_prompt")

    return prompts["human_agent"]["system_prompt"]


@dataclass
class HumanAgentResponse:
    """Human Agent响应"""
    correction_text: str       # 纠正文本（自然语言描述）
    correction_audio: bytes    # 纠正语音（TTS生成的音频）
    thinking: str              # 推理过程
    latency_llm: float         # LLM延迟
    latency_tts: float         # TTS延迟


class HumanAgent:
    """
    Human Agent - 模拟用户纠错行为

    功能:
    1. 比较屏幕文本与Ground Truth
    2. 生成自然语言纠错描述（因地制宜，可混合多种方式）
    3. 调用TTS将纠正文本合成为语音
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,  # API 端点 URL
        default_speaker_path: Optional[str] = None,
        prompts_path: Optional[str] = None,
        enable_thinking: bool = True,  # 控制thinking模式
    ):
        self.model = model or LLM_HUMAN_MODEL_NAME
        self.base_url = base_url or LLM_HUMAN_BASE_URL
        self.default_speaker_path = default_speaker_path
        self.system_prompt = get_human_prompt(prompts_path)
        self.enable_thinking = enable_thinking

    def _parse_llm_response(self, response: str, gt: str, on_screen_text: str) -> Tuple[str, str]:
        """
        解析LLM响应，带容错机制

        Args:
            response: LLM 原始响应
            gt: Ground Truth
            on_screen_text: 屏幕文本

        Returns:
            (correction, thinking)
        """
        thinking = ""
        correction = ""

        # 检查响应是否失败（标签缺失或格式错误）
        is_failed, fail_reason = is_llm_response_failed(
            response,
            required_tags=["correction_think", "correction"]
        )

        if is_failed:
            # 响应失败，使用默认策略：简单直接指出差异
            thinking = f"LLM响应格式异常: {fail_reason}，直接重复。"
            correction = f"不对，应该是：{gt}"
            return correction, thinking

        # 正常解析
        # 提取thinking
        thinking_match = re.search(r'<correction_think>(.*?)</correction_think>', response, re.DOTALL)
        if thinking_match:
            thinking = thinking_match.group(1).strip()

        # 提取correction
        correction_match = re.search(r'<correction>(.*?)</correction>', response, re.DOTALL)
        if correction_match:
            correction = correction_match.group(1).strip()
        else:
            # 理论上不会走到这里，因为前面已经检查了
            correction = f"不对，应该是：{gt}"

        return correction, thinking

    def process(
        self,
        gt: str,
        on_screen_text: str,
        speaker_path: Optional[str] = None,
        generate_audio: bool = True,
    ) -> HumanAgentResponse:
        """
        处理纠错任务，生成纠正文本和语音

        Args:
            gt: Ground Truth（标准答案）
            on_screen_text: 屏幕显示的文本
            speaker_path: 说话人参考音频路径（如果为None使用默认）
            generate_audio: 是否生成TTS音频

        Returns:
            HumanAgentResponse: 包含纠正文本、语音、推理过程等信息
        """
        import time
        import os

        speaker_path = speaker_path or self.default_speaker_path

        start_time = time.time()

        # 简单 prompt：直接给 LLM GT 和 on_screen_text，让它自己分析差异
        # 注意：判断逻辑由外部（evaluate.py + run_next_loop.py）完成
        prompt = f"""标准答案（GT）是：{gt}

当前系统显示的文本是：{on_screen_text}

请分析两者的差异，生成自然语言纠错文本。"""

        # 调用LLM
        messages = [{"role": "user", "content": prompt}]
        llm_response = call_llm(
            messages=messages,
            system_prompt=self.system_prompt,
            model=self.model,
            base_url=self.base_url,  # 使用 Human Agent 专用端口
            enable_thinking=self.enable_thinking,  # 只需传递这个标志
        )

        # 解析LLM响应
        correction_text, thinking = self._parse_llm_response(llm_response, gt, on_screen_text)

        latency_llm = time.time() - start_time
        latency_tts = 0

        # 生成TTS音频
        if generate_audio and speaker_path and os.path.exists(speaker_path):
            try:
                start_tts = time.time()
                correction_audio = call_tts(
                    text=correction_text,
                    prompt_audio_path=speaker_path,
                    timeout=60
                )
                latency_tts = time.time() - start_tts
            except Exception as e:
                correction_audio = b""
                print(f"TTS生成失败: {e}")
        else:
            correction_audio = b""

        return HumanAgentResponse(
            correction_text=correction_text,
            correction_audio=correction_audio,
            thinking=thinking,
            latency_llm=latency_llm,
            latency_tts=latency_tts,
        )


# 测试代码
if __name__ == "__main__":
    print("Human Agent 测试")

    agent = HumanAgent()

    # 测试1: 正确的情况
    print("\n=== 测试1: 正确 ===")
    response = agent.process("我要去云澜小区", "我要去云澜小区")
    print(f"纠正文本: {response.correction_text}")
    print(f"推理: {response.thinking}")

    # 测试2: 错误的情况
    print("\n=== 测试2: 错误（云南 vs 云澜）===")
    response = agent.process("我要去云澜小区", "我要去云南小区")
    print(f"纠正文本: {response.correction_text}")
    print(f"推理: {response.thinking}")

    # 测试3: 生成TTS
    print("\n=== 测试3: 生成TTS ===")
    response = agent.process("我要去云澜小区", "我要去云南小区", generate_audio=True)
    print(f"纠正文本: {response.correction_text}")
    print(f"音频大小: {len(response.correction_audio)} bytes")