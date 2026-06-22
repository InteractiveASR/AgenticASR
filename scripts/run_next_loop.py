#!/usr/bin/env python3
"""CLI entrypoint for one additional interactive correction loop."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interactive_asr.simulation import run_next_loop
from interactive_asr.simulation.io import load_jsonl, save_jsonl


# ==================== 主函数 ====================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run one additional interactive correction loop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行第一轮纠错（调试信息默认嵌入到输出文件中）
  python run_next_loop.py --input logs/stage0_evaluated.jsonl --output logs/stage1.jsonl

  # 设置最大轮次
  python run_next_loop.py --input logs/stage0_evaluated.jsonl --output logs/stage1_evaluated.jsonl --max-turns 5

  # 指定并发数
  python run_next_loop.py --input logs/stage0_evaluated.jsonl --output logs/stage1_evaluated.jsonl --concurrency 8

输出文件字段说明（每轮纠错后新增）:
  - loop_1_pred
  - loop_1_human_think
  - loop_1_human_correction
  - loop_1_correction_asr
  - loop_1_asr_refine
  - loop_1_asr_refine_think
        """
    )
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum number of loops")
    parser.add_argument("--concurrency", type=int, default=256, help="Number of workers")
    parser.add_argument("--speaker", help="Fallback speaker reference audio")
    parser.add_argument("--prompts", default="config/default_prompts.json", help="Prompt config path")
    parser.add_argument("--audio-dir", default="./temp_audio", help="Temporary correction audio directory")

    # ========== 新增 thinking 模式选择 ==========
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument("--thinking", action="store_true", default=True,
                               help="Enable thinking mode")
    thinking_group.add_argument("--non-thinking", action="store_true", default=False,
                               help="Disable thinking mode")

    args = parser.parse_args()

    # 确定 thinking 模式
    enable_thinking = not args.non_thinking  # --non-thinking 会覆盖默认值

    # 检查输入文件
    if not Path(args.input).exists():
        print(f"Input file does not exist: {args.input}")
        return 1

    # 设置默认 speaker 路径
    speaker_path = args.speaker or None
    print(f"Loading data: {args.input}")
    items = load_jsonl(args.input)
    print(f"Samples: {len(items)}")

    # 运行下一轮纠错
    results = run_next_loop(
        items=items,
        max_turns=args.max_turns,
        concurrency=args.concurrency,
        speaker_path=speaker_path,
        prompts_path=args.prompts,
        audio_dir=args.audio_dir,
        enable_thinking=enable_thinking,
    )

    if not results:
        print("Loop execution failed")
        return 1

    save_jsonl(results, args.output)

    # 统计
    loop_counts = {}
    for item in results:
        loop = item.get("total_loop", 0)
        loop_counts[loop] = loop_counts.get(loop, 0) + 1

    print("\n" + "=" * 80)
    print("Interactive loop finished")
    print("=" * 80)
    for loop in sorted(loop_counts.keys()):
        print(f"  Loop {loop}: {loop_counts[loop]} samples")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
