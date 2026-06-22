"""
文本归一化模块

使用 cn_tn 进行中文文本归一化
"""

from .cn_tn import TextNorm

# 创建全局归一化器实例
# 配置选项：
# - to_banjiao: 全角转半角（如 ， -> , ）
# - to_upper/to_lower: 大小写转换（中文不需要）
# - remove_fillers: 移除"呃"、"啊"等填充词
# - remove_erhua: 去除儿化音
# - check_chars: 检查非法字符
# - remove_space: 移除多余空格（保留英文单词间空格）
_normalizer = TextNorm(
    to_banjiao=True,      # 转换全角字符为半角
    to_upper=True,        # 开个转大写
    to_lower=False,       # 不转小写
    remove_fillers=True,  # 移除填充词"呃"、"啊"
    remove_erhua=True,    # 去除儿化音
    check_chars=False,    # 不检查非法字符（宽松模式）
    remove_space=True,    # 移除多余空格
)


def norm(text: str) -> str:
    """
    中文文本归一化

    使用 cn_tn 的 TextNorm 进行归一化：
    1. 全角转半角（， -> , ）
    2. 移除填充词"呃"、"啊"等
    3. 去除儿化音（如"那边儿" -> "那边"）
    4. 规范化非标准词（日期、数字、电话号码等）
    5. 移除标点符号
    6. 移除多余空格

    Args:
        text: 原始文本

    Returns:
        归一化后的文本

    Example:
        >>> norm("三点水的澜，不是云南的云")
        '三点水的澜不是云南的云'
        >>> norm("我想自然风关了")
        '我想自然风关了'
        >>> norm("把灯调暗一点呃")
        '把灯调暗一点'
        >>> norm("2024年5月")
        '二零二四年五月'
    """
    if text is None:
        return ""
    return _normalizer(text)


def norm_for_cer(text: str) -> str:
    """
    用于 CER 计算的归一化

    与 norm() 相同，但在计算 CER 时使用更明确的函数名

    Args:
        text: 原始文本

    Returns:
        归一化后的文本
    """
    return norm(text)


def norm_for_ser(text: str) -> str:
    """
    用于 SER 计算的归一化

    SER（句子错误率）只需要简单的归一化来判断句子是否相等

    Args:
        text: 原始文本

    Returns:
        归一化后的文本
    """
    return norm(text).strip()