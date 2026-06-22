#!/usr/bin/env python3
"""Evaluate agreement between human S²ER labels and judge labels."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interactive_asr.s2er.human_alignment import evaluate_human_alignment, load_human_alignment_csv


def main():
    parser = argparse.ArgumentParser(description="Evaluate human-vs-judge agreement for S²ER labels.")
    parser.add_argument("--input", required=True, help="CSV file with human and judge labels")
    parser.add_argument("--human-key", default="human_label", help="CSV column for human labels")
    parser.add_argument("--judge-key", default="judge_label", help="CSV column for judge labels")
    args = parser.parse_args()

    records = load_human_alignment_csv(args.input)
    metrics = evaluate_human_alignment(records, human_key=args.human_key, judge_key=args.judge_key)

    print("S²ER Human Alignment")
    print(f"n: {metrics['n']}")
    print(f"accuracy: {metrics['accuracy']:.2f}%")
    print(f"pearson: {metrics['pearson']:.4f}")
    print(f"cohen_kappa: {metrics['cohen_kappa']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
