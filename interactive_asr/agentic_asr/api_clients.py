#!/usr/bin/env python3
"""
API客户端模块 - 封装ASR、TTS、LLM三个服务的调用
"""

import re
import os
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ==================== ASR 配置 ====================
# Whisper/Ray ASR (当前使用)
ASR_URL = os.environ.get("ASR_URL", "http://0.0.0.0:18080/v1/audio/transcriptions")
ASR_MODEL_NAME = os.environ.get("ASR_MODEL", "qwen3asr")

# Qwen3-ASR (备选配置)
# ASR_URL = "http://localhost:7879/v1/chat/completions"

# FireRedASR (备选配置)
# ASR_URL = "http://localhost:7880/v1/audio/transcriptions"

TTS_URL = os.environ.get("TTS_URL", "http://0.0.0.0:6006/tts_url")

# ==================== LLM 配置 ====================
# 三套独立配置：Human Agent / ASR Agent / Judge
# 每套都有独立的 BASE_URL 和 MODEL_NAME，通过环境变量控制

# Human Agent 配置
LLM_HUMAN_BASE_URL = os.environ.get("LLM_HUMAN_BASE_URL", "http://0.0.0.0:6790/v1")
LLM_HUMAN_MODEL_NAME = os.environ.get("LLM_HUMAN_MODEL", "qwen3.5-27b")

# ASR Agent 配置
LLM_ASR_BASE_URL = os.environ.get("LLM_ASR_BASE_URL", "http://0.0.0.0:6790/v1")
LLM_ASR_MODEL_NAME = os.environ.get("LLM_ASR_MODEL", "qwen3.5-27b")

# Judge 配置
LLM_JUDGE_BASE_URL = os.environ.get("LLM_JUDGE_BASE_URL", "http://0.0.0.0:6789/v1")
LLM_JUDGE_MODEL_NAME = os.environ.get("LLM_JUDGE_MODEL", "qwen3-32b")

# 兼容旧代码的默认配置
LLM_BASE_URL = LLM_JUDGE_BASE_URL
LLM_MODEL_NAME = LLM_JUDGE_MODEL_NAME


# ==================== ASR输出解析工具 ====================
_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "


def normalize_language_name(language: str) -> str:
    if language is None:
        raise ValueError("language is None")
    s = str(language).strip()
    if not s:
        raise ValueError("language is empty")
    return s[:1].upper() + s[1:].lower()


def parse_asr_output(raw: str, user_language: Optional[str] = None) -> Tuple[str, str]:
    """
    解析Qwen3-ASR原始输出为(语言, 文本)。

    Args:
        raw: ASR服务返回的原始字符串
        user_language: 如果提供，强制使用此语言

    Returns:
        Tuple[str, str]: (语言, 文本)
    """
    if raw is None:
        return "", ""
    s = str(raw).strip()
    if not s:
        return "", ""

    if user_language:
        return user_language, s

    meta_part = s
    text_part = ""
    has_tag = _ASR_TEXT_TAG in s
    if has_tag:
        meta_part, text_part = s.split(_ASR_TEXT_TAG, 1)
    else:
        return "", s.strip()

    meta_lower = meta_part.lower()

    if "language none" in meta_lower:
        t = text_part.strip()
        if not t:
            return "", ""
        return "", t

    lang = ""
    for line in meta_part.splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith(_LANG_PREFIX):
            val = line[len(_LANG_PREFIX):].strip()
            if val:
                lang = normalize_language_name(val)
            break

    return lang, text_part.strip()


# ==================== TTS客户端 ====================
def call_tts(
    text: str,
    prompt_audio_path: Optional[str] = None,
    output_path: Optional[str] = None,
    timeout: int = 60,
    **kwargs
) -> bytes:
    """
    调用 TTS 服务进行语音合成（IndexTTS 1.5 格式）。

    Args:
        text: 要合成的文本
        prompt_audio_path: 参考说话人音频路径（可选，为空则使用默认音色）
        output_path: 输出音频文件路径（可选）
        timeout: 超时时间（秒）
        **kwargs: 兼容性参数（当前版本不使用）

    Returns:
        bytes: WAV音频字节数据
    """
    def _normalize_audio_path(path: str) -> str:
        if path.startswith(("http://", "https://", "data:", "file://")):
            return path
        return str(Path(path).resolve())

    data = {"text": text}

    # 添加参考音频到请求中
    if prompt_audio_path is not None:
        if isinstance(prompt_audio_path, str):
            data["audio_paths"] = [_normalize_audio_path(prompt_audio_path)]
        else:  # 假设是列表
            data["audio_paths"] = [
                _normalize_audio_path(path) if isinstance(path, str) else path
                for path in prompt_audio_path
            ]

    response = requests.post(TTS_URL, json=data, timeout=timeout)
    response.raise_for_status()

    audio_bytes = response.content

    if output_path:
        with open(output_path, "wb") as f:
            f.write(audio_bytes)

    return audio_bytes


# ==================== ASR客户端 ====================
def call_asr(
    audio_path: str,
    timeout: int = 300
) -> Tuple[str, str]:
    """
    调用ASR服务进行语音识别。

    当前实现：Whisper/Ray 服务，接口为 /v1/audio/transcriptions。

    Args:
        audio_path: 音频路径（支持 http(s)/data/file:// URL 或本地路径）
        timeout: 超时时间（秒）

    Returns:
        Tuple[str, str]: (语言, 识别文本)

    Raises:
        requests.RequestException: 当请求失败时抛出
    """
    from pathlib import Path

    asr_url_lower = ASR_URL.lower()

    # 优先支持 transcription 端点（Qwen3ASR / Whisper 兼容）
    if asr_url_lower.endswith("/v1/audio/transcriptions"):
        # 先尝试 multipart（Qwen3ASR 常见格式）
        if not (
            audio_path.startswith("http://")
            or audio_path.startswith("https://")
            or audio_path.startswith("data:")
            or audio_path.startswith("file://")
        ):
            try:
                with open(audio_path, "rb") as f:
                    files = {"file": (Path(audio_path).name, f, "audio/wav")}
                    data = {"model": ASR_MODEL_NAME}
                    response = requests.post(ASR_URL, files=files, data=data, timeout=timeout)
                response.raise_for_status()
                result = response.json()
                text_raw = (result.get("text") or "").strip()
                language, text = parse_asr_output(text_raw)
                if text:
                    return language, text
                return language, text_raw
            except Exception:
                # 回退到 audio_url JSON 格式
                pass

        payload = {}
        if (
            audio_path.startswith("http://")
            or audio_path.startswith("https://")
            or audio_path.startswith("data:")
            or audio_path.startswith("file://")
        ):
            payload["audio_url"] = audio_path
        else:
            abs_path = str(Path(audio_path).resolve())
            payload["audio_url"] = f"file://{abs_path}"

        response = requests.post(ASR_URL, json=payload, timeout=timeout)
        response.raise_for_status()

        result = response.json()
        text_raw = (result.get("text") or "").strip()
        language = result.get("language") or ""
        if text_raw:
            parsed_language, parsed_text = parse_asr_output(text_raw)
            language = language or parsed_language
            text_raw = parsed_text or text_raw
        return language, text_raw

    # chat completions 兼容（兜底）
    if asr_url_lower.endswith("/v1/chat/completions"):
        if (
            audio_path.startswith("http://")
            or audio_path.startswith("https://")
            or audio_path.startswith("data:")
            or audio_path.startswith("file://")
        ):
            asr_input = audio_path
        else:
            abs_path = str(Path(audio_path).resolve())
            asr_input = f"file://{abs_path}"

        response = requests.post(
            ASR_URL,
            json={
                "model": ASR_MODEL_NAME,
                "messages": [{"role": "user", "content": asr_input}],
            },
            timeout=timeout
        )
        response.raise_for_status()
        result = response.json()
        content = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        language, text = parse_asr_output(content)
        return language, text or content.strip()

    raise ValueError(f"不支持的 ASR_URL: {ASR_URL}")


# ==================== LLM客户端 ====================
def call_llm(
    messages: List[dict],
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,  # 显式指定端口
    api_key: Optional[str] = None,  # 显式指定API Key（优先级最高）
    enable_thinking: bool = True,  # 唯一控制参数，默认True
    max_tokens: Optional[int] = None,  # 允许覆盖
    temperature: Optional[float] = None,  # 允许覆盖
    timeout: int = 60,
) -> str:
    """
    调用LLM服务进行推理，根据 enable_thinking 自动选择最优参数配置

    Args:
        messages: 消息列表
        system_prompt: 系统提示词
        model: 模型名称
        base_url: API 端点 URL（显式指定，如不指定则使用默认 LLM_BASE_URL）
        api_key: API Key（优先级：参数 > DASHSCOPE_API_KEY > OPENAI_API_KEY > dummy）
        enable_thinking: 是否启用thinking模式
        max_tokens: 最大生成token数（可选覆盖）
        temperature: 温度参数（可选覆盖）
        timeout: 超时时间
    """
    if model is None:
        model = LLM_MODEL_NAME

    if base_url is None:
        base_url = LLM_BASE_URL

    try:
        import openai
    except ImportError as exc:
        raise ImportError(
            "The `openai` package is required for LLM-backed evaluation and interaction. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    resolved_api_key = (
        api_key
        or os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "dummy"
    )

    client = openai.OpenAI(
        base_url=base_url,
        api_key=resolved_api_key
    )

    final_messages = []
    if system_prompt:
        final_messages.append({"role": "system", "content": system_prompt})
    final_messages.extend(messages)

    # ========== 根据 enable_thinking 选择参数配置 ==========
    if enable_thinking:
        # Thinking 模式配置（Qwen3 官方推荐）
        # 参考: https://huggingface.co/Qwen/Qwen3-8B#best-practices
        default_max_tokens = 8192
        default_temperature = 0.6  # 官方建议 thinking 模式用 0.6
        top_p = 0.95
        top_k = 20
        presence_penalty = 1.5  # 防止无限重复
    else:
        # Non-thinking 模式配置（官方推荐）
        default_max_tokens = 8192
        default_temperature = 0.7
        top_p = 0.8
        top_k = 20
        presence_penalty = 1.5

    # 允许外部覆盖
    actual_max_tokens = max_tokens if max_tokens is not None else default_max_tokens
    actual_temperature = temperature if temperature is not None else default_temperature

    is_dashscope = "dashscope.aliyuncs.com" in str(base_url).lower()

    extra_body = {
        "top_k": top_k,
        "presence_penalty": presence_penalty,
    }
    if is_dashscope:
        extra_body["enable_thinking"] = enable_thinking
    else:
        extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    # 调用 API
    response = client.chat.completions.create(
        model=model,
        messages=final_messages,
        max_tokens=actual_max_tokens,
        temperature=actual_temperature,
        top_p=top_p,
        extra_body=extra_body
    )

    return response.choices[0].message.content


# ==================== Judge客户端（语义判断） ====================
def call_judge(
    text1: str,
    text2: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = 30,
    enable_thinking: bool = False,  # 默认禁用thinking（性能优先）
) -> bool:
    """
    使用 LLM 判断两个文本在语义上是否等价

    注意：默认使用 non-thinking 模式以提高性能
    默认使用 LLM_JUDGE_MODEL_NAME 模型（可通过 model 参数覆盖）
    """
    import re

    if system_prompt is None:
        raise ValueError("call_judge: system_prompt 必须从 prompt 配置加载，不能为 None")

    if model is None:
        model = LLM_JUDGE_MODEL_NAME

    user_message = f"""文本1：{text1}
文本2：{text2}

请判断这两个文本在语义上是否等价。"""

    response = call_llm(
        messages=[{"role": "user", "content": user_message}],
        system_prompt=system_prompt,
        model=model,
        base_url=LLM_JUDGE_BASE_URL,  # 使用 Judge 专用端口
        enable_thinking=enable_thinking,
        timeout=timeout
    )

    # 解析结果
    # 支持两种格式: <is_equivalent>true</is_equivalent> 或 <is_equivalent>[true]</is_equivalent>
    match = re.search(r'<is_equivalent>\[?\s*(true|false)\s*\]?</is_equivalent>', response, re.IGNORECASE)
    if match:
        return match.group(1).lower() == 'true'

    # 响应格式错误
    raise ValueError(f"call_judge: 响应格式不符合预期，未找到 <is_equivalent> 标签。响应内容: {response[:100]}...")


# ==================== Judge客户端（k轮双向共识判断） ====================
def call_judge_with_consensus(
    text1: str,
    text2: str,
    system_prompt: str,
    k: int = 3,
    model: Optional[str] = None,
    timeout: int = 30,
    enable_thinking: bool = False,
) -> bool:
    """
    使用 k 轮双向判断语义等价性（带共识机制）

    判断逻辑：
    - 每轮：先判断 text1 vs text2，再判断 text2 vs text1
    - 只有双向都为 True，该轮才记为 True
    - k 轮中 True 数 > k//2 + 1 才认为语义一致

    Args:
        text1: 第一个文本（通常是 pred）
        text2: 第二个文本（通常是 gt）
        system_prompt: Judge 的 system prompt
        k: 判断轮数，默认 3
        model: 模型名称
        timeout: 单次调用超时时间
        enable_thinking: 是否启用 thinking 模式

    Returns:
        bool: 是否语义等价
    """
    if k < 1:
        raise ValueError(f"call_judge_with_consensus: k 必须 >= 1，当前 k={k}")

    if model is None:
        model = LLM_JUDGE_MODEL_NAME

    true_count = 0
    threshold = k // 2 + 1  # 3轮需要2票，5轮需要3票

    for round_idx in range(k):
        try:
            # 双向判断
            r1 = call_judge(text1, text2, system_prompt=system_prompt, model=model,
                           timeout=timeout, enable_thinking=enable_thinking)
            r2 = call_judge(text2, text1, system_prompt=system_prompt, model=model,
                           timeout=timeout, enable_thinking=enable_thinking)

            # 只有双向都 True，该轮才记为 True
            if r1 and r2:
                true_count += 1
                # 早停：已经达到阈值
                if true_count > threshold:
                    return True
        except Exception as e:
            # 单轮失败不影响整体，继续下一轮
            import logging
            logging.warning(f"call_judge_with_consensus 第 {round_idx + 1} 轮失败: {e}")

    return true_count > threshold


def call_judge_with_trace(
    text1: str,
    text2: str,
    system_prompt: str,
    k: int = 3,
    model: Optional[str] = None,
    timeout: int = 30,
    enable_thinking: bool = False,
) -> Dict:
    """
    Judge semantic equivalence with per-round tracing.

    Returns:
        A dictionary with round-level outcomes and the final consensus decision.
    """
    if k < 1:
        raise ValueError(f"call_judge_with_trace: k must be >= 1, got {k}")

    if model is None:
        model = LLM_JUDGE_MODEL_NAME

    threshold = k // 2 + 1
    true_count = 0
    rounds = []

    for round_idx in range(k):
        round_trace = {
            "round": round_idx + 1,
            "forward": None,
            "backward": None,
            "round_consensus": False,
            "error": None,
        }
        try:
            forward = call_judge(
                text1,
                text2,
                system_prompt=system_prompt,
                model=model,
                timeout=timeout,
                enable_thinking=enable_thinking,
            )
            backward = call_judge(
                text2,
                text1,
                system_prompt=system_prompt,
                model=model,
                timeout=timeout,
                enable_thinking=enable_thinking,
            )
            round_trace["forward"] = forward
            round_trace["backward"] = backward
            round_trace["round_consensus"] = bool(forward and backward)
            if round_trace["round_consensus"]:
                true_count += 1
        except Exception as e:
            round_trace["error"] = str(e)
        rounds.append(round_trace)

    return {
        "k": k,
        "threshold": threshold,
        "true_count": true_count,
        "semantic_equivalent": true_count >= threshold,
        "rounds": rounds,
        "model": model,
    }


# ==================== LLM 响应容错检测 ====================
def is_llm_response_failed(response: str, required_tags: List[str]) -> Tuple[bool, str]:
    """
    检查 LLM 响应是否失败（标签缺失或格式错误）。

    Args:
        response: LLM 原始响应
        required_tags: 必需的标签列表，如 ["correction_think", "correction"]

    Returns:
        Tuple[bool, str]: (是否失败, 失败原因)
    """
    if not response or not response.strip():
        return True, "空响应"

    for tag in required_tags:
        # 检查开始和结束标签都存在
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        if start_tag not in response or end_tag not in response:
            return True, f"缺少标签: <{tag}>"

        # 检查标签之间是否有内容
        pattern = rf'<{tag}>(.*?)</{tag}>'
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            return True, f"标签 <{tag}> 内容为空或格式错误"

        content = match.group(1).strip()
        if not content:
            return True, f"标签 <{tag}> 内容为空"

    return False, ""


def is_text_change_too_large(old_text: str, new_text: str, cer_threshold: float = 1.0) -> bool:
    """
    检查新文本相对于旧文本变化是否过大。

    与 evaluate.py 保持一致，使用 kaldialign 计算 CER。
    注意：不使用 fallback，如果 kaldialign 不可用将抛出异常。

    Args:
        old_text: 原始文本
        new_text: 新文本
        cer_threshold: CER 阈值，默认 1.0（0-1 范围）

    Returns:
        bool: True 表示变化过大

    Raises:
        ImportError: 如果 kaldialign 不可用
    """
    import kaldialign

    if not old_text or not new_text:
        return True

    # 简单的长度变化检查
    len_old = len(old_text)
    len_new = len(new_text)
    if len_old > 0:
        len_ratio = abs(len_new - len_old) / len_old
        if len_ratio > 0.8:  # 长度变化超过 80%
            return True

    # 使用 kaldialign 计算 CER（与 evaluate.py 保持一致）
    # 转换为字符列表
    old_chars = list(old_text)
    new_chars = list(new_text)
    # 计算编辑距离
    result = kaldialign.edit_distance(old_chars, new_chars)
    # err_rate 是 0-1 之间的值
    cer = result.get('err_rate', 0.0)
    return cer > cer_threshold


# ==================== 测试 ====================
if __name__ == "__main__":
    print("API客户端模块 - 测试各个服务")

    # 测试LLM服务
    print("\n=== 测试LLM服务 ===")
    try:
        result = call_llm(
            messages=[{"role": "user", "content": "你好，请用简短的话介绍一下自己。"}],
            max_tokens=256
        )
        print(f"LLM响应: {result}")
    except Exception as e:
        print(f"LLM测试失败: {e}")

    # 测试TTS服务（需要参考音频）
    print("\n=== 测试TTS服务 ===")
    spk_path = "data/example_audio/03155.wav"
    if os.path.exists(spk_path):
        try:
            audio = call_tts(
                text="你好，这是一个测试。",
                prompt_audio_path=spk_path,
                output_path="test_tts_output.wav"
            )
            print(f"TTS合成成功，音频大小: {len(audio)} bytes")
        except Exception as e:
            print(f"TTS测试失败: {e}")
    else:
        print(f"参考音频不存在: {spk_path}")

    # 测试ASR服务（需要音频文件）
    print("\n=== 测试ASR服务 ===")
    test_audio = "data/example_audio/03155.wav"
    try:
        asr_dict = call_asr(test_audio)
        print(f"ASR识别结果 - {asr_dict}")
    except Exception as e:
        print(f"ASR测试失败: {e}")
