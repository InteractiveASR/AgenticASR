#!/usr/bin/env python3
"""
ASR Agent类 - 负责ASR识别和LLM推理修正
"""

import re
import json
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass

from .api_clients import (
    call_asr,
    call_llm,
    LLM_ASR_MODEL_NAME,
    LLM_ASR_BASE_URL,
    is_llm_response_failed,
    is_text_change_too_large,
)


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


def get_asr_prompt(prompts_path: Optional[str] = None) -> str:
    """
    获取ASR Agent的system prompt

    注意：不使用 fallback，如果 default_prompts.json 不存在或格式错误将抛出异常。

    Args:
        prompts_path: default_prompts.json 文件路径（默认"config/default_prompts.json"）

    Returns:
        system prompt字符串

    Raises:
        FileNotFoundError: 如果 default_prompts.json 文件不存在
        KeyError: 如果 default_prompts.json 中没有 asr_agent.system_prompt
    """
    if prompts_path is None:
        prompts_path = "config/default_prompts.json"

    prompts = load_prompts_from_file(prompts_path)
    if "asr_agent" not in prompts or "system_prompt" not in prompts["asr_agent"]:
        raise KeyError("default_prompts.json 中未找到 asr_agent.system_prompt")

    return prompts["asr_agent"]["system_prompt"]


@dataclass
class ASRAgentResponse:
    """ASR Agent响应"""
    on_screen_text: str      # 最终屏幕文本
    thinking: str            # 推理过程
    is_affirmation: bool     # 是否为肯定词
    latency_asr: float       # ASR延迟
    latency_llm: float       # LLM延迟


class ASRAgent:
    """
    ASR Agent - 负责语音识别和智能纠错

    功能:
    1. 接收音频输入，调用ASR服务进行识别
    2. 使用LLM分析用户反馈（肯定/纠正）
    3. 如果是纠正，结合部首、组词、发音等信息修改屏幕文本
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,  # API端点URL
        prompts_path: Optional[str] = None,
        enable_thinking: bool = True,  # 控制thinking模式
    ):
        self.model = model or LLM_ASR_MODEL_NAME
        self.base_url = base_url or LLM_ASR_BASE_URL
        self.system_prompt = get_asr_prompt(prompts_path)
        self.enable_thinking = enable_thinking

    def _parse_llm_response(
        self,
        response: str,
        on_screen_text: str,
        cer_threshold: float = 1.0
    ) -> Tuple[str, str, bool]:
        """
        解析LLM响应，带容错机制

        Args:
            response: LLM 原始响应
            on_screen_text: 当前屏幕文本
            cer_threshold: CER 阈值，默认 1.0

        Returns:
            (answer, thinking, is_affirmation)
        """
        thinking = ""
        answer = ""
        is_affirmation = False

        # 检查响应是否失败（标签缺失或格式错误）
        is_failed, fail_reason = is_llm_response_failed(
            response,
            required_tags=["correction_think", "answer", "is_affirmation"]
        )

        if is_failed:
            # 响应失败，保持原文本不变
            thinking = f"LLM响应格式异常: {fail_reason}，保持原文本。"
            return on_screen_text, thinking, False

        # 正常解析
        # 提取thinking
        thinking_match = re.search(r'<correction_think>(.*?)</correction_think>', response, re.DOTALL)
        if thinking_match:
            thinking = thinking_match.group(1).strip()

        # 提取answer
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            # 理论上不会走到这里，因为前面已经检查了
            return on_screen_text, "未找到<answer>标签，保持原文本。", False

        # 提取is_affirmation
        affirmation_match = re.search(r'<is_affirmation>(.*?)</is_affirmation>', response, re.DOTALL)
        if affirmation_match:
            value = affirmation_match.group(1).strip().lower()
            is_affirmation = value == "true" or value == "是"
        else:
            is_affirmation = False

        # 检查文本变化是否过大
        if is_text_change_too_large(on_screen_text, answer, cer_threshold):
            thinking = f"文本变化过大（CER>{cer_threshold}），保持原文本。新文本: {answer}"
            return on_screen_text, thinking, is_affirmation

        return answer, thinking, is_affirmation

    def process(
        self,
        audio_path: str,
        on_screen_text: str,
        history: Optional[List[str]] = None,
    ) -> ASRAgentResponse:
        """
        处理音频输入，返回修正后的屏幕文本

        Args:
            audio_path: 输入音频路径
            on_screen_text: 当前屏幕显示的文本
            history: 历史推理记录（可选）

        Returns:
            ASRAgentResponse: 包含修正后文本、推理过程、是否肯定等信息
        """
        import time

        history = history or []
        start_time = time.time()

        # 1. ASR识别
        language, asr_text = call_asr(audio_path)
        latency_asr = time.time() - start_time
        start_time = time.time()

        # 2. 构建LLM输入
        if on_screen_text:
            # 有屏幕文本，分析用户反馈
            prompt = f"""当前屏幕显示的文本是：{on_screen_text}

用户最新的语音反馈（ASR识别结果）是：{asr_text}

请分析用户的意图：
- 如果是肯定词（如"对的"、"没错"、"好的"、"嗯"等），保持文本不变
- 如果是纠正词，结合用户描述的部首、组词、发音等信息，修正屏幕文本中的错误字

{f'历史推理记录：' + '\\n'.join(history[-3:]) if history else ''}

请按照指定格式回复。"""
        else:
            # 没有屏幕文本，直接使用ASR结果
            return ASRAgentResponse(
                on_screen_text=asr_text,
                thinking=f"初始识别，直接使用ASR结果: {asr_text}",
                is_affirmation=False,
                latency_asr=latency_asr,
                latency_llm=0.0,
            )

        # 3. 调用LLM
        messages = [{"role": "user", "content": prompt}]
        llm_response = call_llm(
            messages=messages,
            system_prompt=self.system_prompt,
            model=self.model,
            base_url=self.base_url,  # 使用 ASR Agent 专用端口
            enable_thinking=self.enable_thinking,  # 只需传递这个标志
        )
        latency_llm = time.time() - start_time

        # 4. 解析LLM响应
        answer, thinking, is_affirmation = self._parse_llm_response(llm_response, on_screen_text)

        return ASRAgentResponse(
            on_screen_text=answer if answer else on_screen_text,
            thinking=thinking,
            is_affirmation=is_affirmation,
            latency_asr=latency_asr,
            latency_llm=latency_llm,
        )

    def process_with_raw_asr(
        self,
        audio_path: str,
        on_screen_text: str,
        history: Optional[List[str]] = None,
    ) -> ASRAgentResponse:
        """
        处理音频输入，返回原始ASR结果和LLM修正

        这个方法会返回原始ASR识别结果，用于调试和日志记录。

        Args:
            audio_path: 输入音频路径
            on_screen_text: 当前屏幕显示的文本
            history: 历史推理记录（可选）

        Returns:
            ASRAgentResponse: 包含修正后文本、推理过程、是否肯定等信息
        """
        import time

        history = history or []
        start_time = time.time()

        # 1. ASR识别
        language, asr_raw = call_asr(audio_path)
        latency_asr = time.time() - start_time

        # 2. 解析ASR文本（如果需要）
        asr_text = asr_raw  # call_asr已经返回解析后的文本

        start_time = time.time()

        # 3. 构建LLM输入（包含原始ASR输出）
        if on_screen_text:
            prompt = f"""当前屏幕显示的文本是：{on_screen_text}

用户最新的语音反馈是：{asr_text}

ASR原始识别结果（包含语言信息）：language {language}{asr_raw}

请分析用户的意图：
- 如果是肯定词（如"对的"、"没错"、"好的"、"嗯"等），保持文本不变
- 如果是纠正词，结合用户描述的部首、组词、发音等信息，修正屏幕文本中的错误字

{f'历史推理记录：' + '\\n'.join(history[-3:]) if history else ''}

请按照指定格式回复。"""
        else:
            # 没有屏幕文本，直接使用ASR结果
            return ASRAgentResponse(
                on_screen_text=asr_text,
                thinking=f"初始识别，语言: {language}, 文本: {asr_text}",
                is_affirmation=False,
                latency_asr=latency_asr,
                latency_llm=0.0,
            )

        # 4. 调用LLM
        messages = [{"role": "user", "content": prompt}]
        llm_response = call_llm(
            messages=messages,
            system_prompt=self.system_prompt,
            model=self.model,
            base_url=self.base_url,  # 使用 ASR Agent 专用端口
            enable_thinking=self.enable_thinking,  # 只需传递这个标志
        )
        latency_llm = time.time() - start_time

        # 5. 解析LLM响应
        answer, thinking, is_affirmation = self._parse_llm_response(llm_response, on_screen_text)

        return ASRAgentResponse(
            on_screen_text=answer if answer else on_screen_text,
            thinking=thinking,
            is_affirmation=is_affirmation,
            latency_asr=latency_asr,
            latency_llm=latency_llm,
        )


# 测试代码
if __name__ == "__main__":
    print("ASR Agent 测试")

    # 模拟测试
    agent = ASRAgent()

    # 测试1: 初始识别
    print("\n=== 测试1: 初始识别 ===")
    response = agent.process("", "")
    print(f"屏幕文本: {response.on_screen_text}")
    print(f"推理: {response.thinking}")

    # 测试2: 肯定反馈
    print("\n=== 测试2: 肯定反馈 ===")
    # 使用在线音频作为模拟
    test_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
    try:
        response = agent.process(test_audio, "This is the text on screen.")
        print(f"屏幕文本: {response.on_screen_text}")
        print(f"推理: {response.thinking}")
        print(f"是否肯定: {response.is_affirmation}")
    except Exception as e:
        print(f"测试失败: {e}")