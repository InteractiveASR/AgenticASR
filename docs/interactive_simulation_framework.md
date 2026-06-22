# Interactive Simulation Framework

The simulation framework turns interactive ASR into a scalable benchmark instead of a one-off demo.

## Inputs

Each example is a JSON line with at least:

- `id`
- `gt`
- `audio_path`

Optional metadata fields such as `category`, `difficulty`, and custom annotations are preserved in outputs.

## Pipeline

1. `run_stage0_asr.py`
   Produces the initial recognition file with `raw_pred`.
2. `evaluate.py`
   Adds `is_correct`, `is_semantic_correct`, and `total_loop`.
3. `run_next_loop.py`
   Applies one additional correction round to unresolved samples.
4. repeat evaluation and looping as needed

## Output Philosophy

Every output file is JSONL and keeps the original sample plus derived fields. This makes it easy to:

- resume experiments
- inspect bad cases
- compute loop-by-loop metrics
- visualize interaction traces

## Why It Matters

Without this framework, interactive ASR research is hard to compare fairly because every group would use a different human-in-the-loop protocol. The simulation layer provides a reproducible approximation of that protocol and ties the correction mechanism to explicit machine-readable traces.
