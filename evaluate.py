#!/usr/bin/env python3
"""CLI entrypoint for S²ER evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from interactive_asr.s2er.evaluator import (
    calculate_metrics,
    evaluate_items_concurrent,
    evaluate_items_serial,
    get_judge_prompt,
    load_prompts_from_file,
    print_report,
)
from interactive_asr.s2er.metrics import compute_loopwise_s2er
from interactive_asr.simulation.io import load_jsonl, save_jsonl


# ==================== 主函数 ====================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate Interactive ASR outputs with exact-match and S²ER semantic judging.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 评估并输出到新文件
  python evaluate.py --input logs/stage0_raw_predictions.jsonl --output logs/stage0_evaluated.jsonl

  # 覆盖输入文件（需要 --overwrite）
  python evaluate.py --input logs/stage0_raw_predictions.jsonl --output logs/stage0_raw_predictions.jsonl --overwrite

  # 只生成报告，不修改文件
  python evaluate.py --input logs/stage1_evaluated.jsonl --report-only

  # 禁用语义判断
  python evaluate.py --input logs/stage0_raw_predictions.jsonl --output logs/stage0_evaluated.jsonl --no-semantic-judge

  # 并发评估（更快）
  python evaluate.py --input logs/stage0_raw_predictions.jsonl --output logs/stage0_evaluated.jsonl --concurrency 4

  # 指定 default_prompts.json 路径
  python evaluate.py --input logs/stage0_raw_predictions.jsonl --output logs/stage0_evaluated.jsonl --prompts config/default_prompts.json
        """
    )
    parser.add_argument("--input", required=True, help="输入文件路径")
    parser.add_argument("--output", help="输出文件路径（默认覆盖输入文件）")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖输入文件")
    parser.add_argument("--report-only", action="store_true", help="只生成报告，不修改文件")
    parser.add_argument("--no-semantic-judge", action="store_true", help="禁用 LLM judge")
    parser.add_argument("--concurrency", type=int, default=256, help="并发数（默认256）")
    parser.add_argument("--prompts", type=str, default="config/default_prompts.json", help="default_prompts.json 文件路径（默认 config/default_prompts.json）")
    parser.add_argument("--judge-k-rounds", type=int, default=3, help="语义判断轮数（默认3）")
    parser.add_argument("--save-judge-trace", action="store_true", help="Save round-level semantic judge traces")

    args = parser.parse_args()

    # 检查输入文件
    if not Path(args.input).exists():
        print(f"错误: 输入文件不存在: {args.input}")
        return 1

    # 确定输出文件
    if args.output is None:
        output_path = args.input
    else:
        output_path = args.output

    # 检查是否需要覆盖
    if output_path == args.input and not args.overwrite and not args.report_only:
        print("错误: 输出文件与输入文件相同，需要使用 --overwrite 参数")
        return 1

    # 加载数据
    print(f"Loading data: {args.input}")
    results = load_jsonl(args.input)
    print(f"Samples: {len(results)}")

    # 评估
    use_semantic_judge = not args.no_semantic_judge
    print(f"Semantic judging: {'enabled' if use_semantic_judge else 'disabled'}")

    # 加载 judge prompt（如果需要）
    judge_prompt = None
    if use_semantic_judge:
        try:
            judge_prompt = get_judge_prompt(args.prompts)
            load_prompts_from_file(args.prompts)
            print(f"Loaded judge prompt from: {args.prompts}")
        except Exception as e:
            print(f"Failed to load judge prompt: {e}")
            return 1

    if args.concurrency > 1:
        evaluated_results = evaluate_items_concurrent(
            results,
            use_semantic_judge,
            judge_prompt,
            args.judge_k_rounds,
            args.concurrency,
            args.save_judge_trace,
        )
    else:
        evaluated_results = evaluate_items_serial(
            results,
            use_semantic_judge,
            judge_prompt,
            args.judge_k_rounds,
            args.save_judge_trace,
        )

    # 计算指标
    metrics = calculate_metrics(evaluated_results)

    # 打印报告
    print_report(evaluated_results, metrics)
    loopwise = compute_loopwise_s2er(evaluated_results)
    if loopwise:
        print("\nLoop-wise S²ER")
        for row in loopwise:
            print(f"  loop={row['loop']}: S²ER={row['s2er_rate']:.2f}%  SER={row['ser_rate']:.2f}%")

    # 保存结果（如果不是 report-only）
    if not args.report_only:
        save_jsonl(evaluated_results, output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
