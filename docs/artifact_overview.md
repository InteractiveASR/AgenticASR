# Artifact Overview

This repository is organized as a research artifact rather than a production speech stack.

## Research Questions

1. Can ASR be improved through iterative, human-like correction loops instead of one-shot decoding?
2. Can semantic evaluation capture failures that token-level metrics miss?
3. Can interactive ASR be benchmarked reproducibly without collecting real humans for every experiment?

## Artifact-to-Paper Mapping

- `Agentic ASR`
  Implements the correction loop and the division of labor between the user-side correction agent and the ASR-side refinement agent.
- `S²ER`
  Implements semantic evaluation with exact-match filtering and LLM-based semantic consensus.
- `Interactive Simulation Framework`
  Implements the closed-loop experiment runner over JSONL datasets and structured loop traces.

## Design Principles

- Modular: ASR, TTS, and LLM backends are replaceable through environment variables.
- Auditable: every interaction loop can be saved with turn-level traces.
- Reproducible: the benchmark input format is plain JSONL and the outputs are append-only JSONL records.
- Public-safe: private training code and internal serving stacks are intentionally excluded.
