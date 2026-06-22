# Contributing

Thank you for your interest in Interactive ASR.

## Scope

This repository focuses on the public research artifact:

- Agentic ASR
- S²ER
- Interactive Simulation Framework

Please keep contributions aligned with reproducibility, clarity, and research usability.

## Recommended Contribution Types

- bug fixes in the public pipeline
- better documentation and reproducibility notes
- cleaner interfaces for swapping ASR, TTS, or LLM backends
- analysis and visualization utilities for loop-wise evaluation

## Before Opening a PR

1. keep changes scoped and reviewable
2. make sure Python files compile
3. update documentation if CLI behavior or file formats change
4. avoid committing large logs, generated audio, or private assets

## Local Check

```bash
python -m compileall interactive_asr evaluate.py scripts
```

## Notes

- external services are required for full end-to-end execution
- model weights and private deployment stacks are intentionally out of scope
