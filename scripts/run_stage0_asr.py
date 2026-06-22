#!/usr/bin/env python3
"""CLI entrypoint for stage-0 ASR decoding."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interactive_asr.simulation import run_stage0
from interactive_asr.simulation.io import save_jsonl


# ==================== 主函数 ====================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run stage-0 ASR decoding over a JSONL benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 完整评估
  python run_stage0_asr.py --data examples/example.jsonl

  # 部分评估（前100个样本）
  python run_stage0_asr.py --data examples/example.jsonl --max-samples 100

  # 从指定索引开始
  python run_stage0_asr.py --data examples/example.jsonl --start-index 100
        """
    )
    parser.add_argument("--data", default="examples/example.jsonl", help="Dataset path")
    parser.add_argument("--start-index", type=int, default=0, help="Start index")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples")
    parser.add_argument("--concurrency", type=int, default=8, help="Number of workers")
    parser.add_argument("--output", default="logs/stage0_raw_predictions.jsonl", help="Output JSONL path")

    args = parser.parse_args()

    # 检查数据文件
    if not Path(args.data).exists():
        print(f"Dataset file does not exist: {args.data}")
        return 1

    # 运行评估
    results = run_stage0(
        data_path=args.data,
        start_index=args.start_index,
        max_samples=args.max_samples,
        concurrency=args.concurrency,
    )

    if not results:
        print("Stage-0 decoding failed")
        return 1

    save_jsonl(results, args.output)

    # 统计
    success_count = len([r for r in results if not r.get("error")])
    error_count = len(results) - success_count

    print("\n" + "=" * 80)
    print("Stage 0 finished")
    print("=" * 80)
    print(f"Success: {success_count}/{len(results)}")
    print(f"Failure: {error_count}/{len(results)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
