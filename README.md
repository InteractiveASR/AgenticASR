# Interactive ASR

[Project Page](https://interactiveasr.github.io/) | [GitHub](https://github.com/InteractiveASR/AgenticASR) | [Paper 1](https://arxiv.org/abs/2604.09121) | [Paper 2](https://arxiv.org/abs/2605.29430)

Official research repository for the **Interactive ASR** project family, including the public artifact for:

- **Agentic ASR**: multi-turn speech recognition with agentic correction
- **S²ER**: Sentence-level Semantic Error Rate for semantic ASR evaluation
- **Interactive Simulation Framework**: reproducible closed-loop benchmarking for interactive ASR

## Overview

Automatic speech recognition is typically treated as a one-pass transcription problem. This setup is misaligned with real human interaction, where recognition failures are often repaired through clarification, confirmation, and correction. Interactive ASR studies this missing loop directly.

<p align="center">
  <img src="docs/assets/teaser-1.png" alt="Interactive ASR overview" width="100%">
</p>

This repository consolidates the core executable components behind our Interactive ASR papers:

1. a user-side correction agent that produces human-like spoken feedback
2. an ASR-side refinement agent that edits the displayed hypothesis
3. a semantic evaluation layer that goes beyond token overlap
4. a simulation framework that scales the full interaction loop to benchmark settings

## Abstract

We formulate ASR as a multi-turn refinement problem rather than a single-shot transcription task. To support this setting, we introduce **Agentic ASR**, a closed-loop framework that combines base ASR decoding, user-like spoken correction, and reasoning-based transcript editing. We further introduce **S²ER**, a sentence-level semantic evaluation metric designed to capture meaning-critical ASR errors that token-level metrics often miss, and an **Interactive Simulation Framework** for scalable and reproducible benchmarking. Together, these components provide an executable research stack for studying interactive speech recognition under multilingual, named-entity-intensive, and code-switching conditions.

## What Is Included

This repository is organized around the three research contributions.

### Agentic ASR

Implements the correction loop:

- `HumanAgent`: generates natural spoken corrections from `GT` and the current ASR hypothesis
- `ASRAgent`: decides whether the correction confirms or edits the displayed transcript
- service clients for ASR, TTS, and LLM backends

Core code:

- [interactive_asr/agentic_asr/human_agent.py](/Users/zixuan/X-LANCE/AgenticASR/interactive_asr/agentic_asr/human_agent.py:1)
- [interactive_asr/agentic_asr/asr_agent.py](/Users/zixuan/X-LANCE/AgenticASR/interactive_asr/agentic_asr/asr_agent.py:1)
- [interactive_asr/agentic_asr/api_clients.py](/Users/zixuan/X-LANCE/AgenticASR/interactive_asr/agentic_asr/api_clients.py:1)

### S²ER

Implements semantic evaluation:

- exact-match evaluation after normalization
- LLM-based semantic equivalence judging
- multi-round consensus to improve judge stability

Core code:

- [interactive_asr/s2er/evaluator.py](/Users/zixuan/X-LANCE/AgenticASR/interactive_asr/s2er/evaluator.py:1)
- [evaluate.py](/Users/zixuan/X-LANCE/AgenticASR/evaluate.py:1)

### Interactive Simulation Framework

Implements the experiment pipeline:

- stage-0 decoding over JSONL datasets
- turn-by-turn correction scheduling
- append-only JSONL outputs with interaction traces
- reusable CLI scripts for iterative experiments

Core code:

- [interactive_asr/simulation/stage0.py](/Users/zixuan/X-LANCE/AgenticASR/interactive_asr/simulation/stage0.py:1)
- [interactive_asr/simulation/loop.py](/Users/zixuan/X-LANCE/AgenticASR/interactive_asr/simulation/loop.py:1)
- [scripts/run_stage0_asr.py](/Users/zixuan/X-LANCE/AgenticASR/scripts/run_stage0_asr.py:1)
- [scripts/run_next_loop.py](/Users/zixuan/X-LANCE/AgenticASR/scripts/run_next_loop.py:1)

## Repository Structure

```text
interactive_asr/
  agentic_asr/                  # correction agents, API clients, normalization
  s2er/                         # semantic evaluation and judge consensus
  simulation/                   # stage-0 decoding and interactive loop orchestration
scripts/
  run_stage0_asr.py             # initial ASR decoding
  run_next_loop.py              # one additional interaction loop
config/
  default_prompts.json          # prompts for ASR agent, human agent, judge
examples/
  example.jsonl                 # minimal benchmark file
  audio/                        # example audio samples
docs/
  artifact_overview.md
  agentic_asr.md
  s2er.md
  interactive_simulation_framework.md
evaluate.py                     # S²ER evaluation entrypoint
```

## Installation

### Environment

- Python `>= 3.10`
- one running ASR service
- one running TTS service
- one or more OpenAI-compatible LLM endpoints for `HumanAgent`, `ASRAgent`, and `Judge`

### Install dependencies

```bash
pip install -r requirements.txt
```

## Service Setup

This repository is the orchestration and evaluation layer. It does **not** include model training code or full serving stacks. Before running the pipeline, you must start external ASR, TTS, and LLM services and then point this repository to those endpoints.

### 1. ASR service

The public pipeline expects an OpenAI-compatible transcription endpoint:

```text
POST /v1/audio/transcriptions
```

The current client supports two ASR deployment styles:

- OpenAI-compatible transcription endpoint, such as a Whisper-style or vLLM-backed ASR server
- OpenAI-compatible chat completion endpoint for ASR models that return tagged text

Recommended environment variables:

```bash
export ASR_URL="http://0.0.0.0:18080/v1/audio/transcriptions"
export ASR_MODEL="qwen3asr"
```

If you are using a FireRedASR-style deployment, a typical launch looks like:

```bash
cd $FIRERED_ASR_DIR
export CUDA_VISIBLE_DEVICES=4,5
MODEL_PATH="$FIRERED_MODEL"
vllm serve "$MODEL_PATH" \
  -tp 2 \
  --dtype float32 \
  --gpu-memory-utilization 0.95 \
  --host 0.0.0.0 \
  --port 7880
```

Then point the repository to that endpoint:

```bash
export ASR_URL="http://0.0.0.0:7880/v1/audio/transcriptions"
```

### 2. TTS service

The TTS side is expected to expose a JSON HTTP endpoint compatible with the client in [interactive_asr/agentic_asr/api_clients.py](/Users/zixuan/X-LANCE/AgenticASR/interactive_asr/agentic_asr/api_clients.py:1):

```text
POST /tts_url
```

Expected request shape:

```json
{
  "text": "correction utterance",
  "audio_paths": ["/absolute/path/to/reference.wav"]
}
```

Recommended environment variable:

```bash
export TTS_URL="http://0.0.0.0:6006/tts_url"
```

In our internal experiments, we used an `IndexTTS-1.5` style service on port `6006`.

### 3. LLM services

The repository uses OpenAI-compatible chat completion APIs for three logical roles:

- `HumanAgent`: generates natural correction utterances
- `ASRAgent`: edits the current ASR hypothesis
- `Judge`: evaluates semantic equivalence for S²ER

You can run all three roles on one endpoint or split them across different endpoints.

Recommended environment variables:

```bash
export LLM_HUMAN_BASE_URL="http://0.0.0.0:6790/v1"
export LLM_ASR_BASE_URL="http://0.0.0.0:6790/v1"
export LLM_JUDGE_BASE_URL="http://0.0.0.0:6789/v1"

export LLM_HUMAN_MODEL="qwen3.5-27b"
export LLM_ASR_MODEL="qwen3.5-27b"
export LLM_JUDGE_MODEL="Gemma4-31B-it"
```

If your endpoint requires an API key:

```bash
export OPENAI_API_KEY="your-key"
```

### 4. Full environment summary

After all external services are running, a minimal local configuration looks like:

```bash
export ASR_URL="http://0.0.0.0:18080/v1/audio/transcriptions"
export TTS_URL="http://0.0.0.0:6006/tts_url"

export LLM_HUMAN_BASE_URL="http://0.0.0.0:6790/v1"
export LLM_ASR_BASE_URL="http://0.0.0.0:6790/v1"
export LLM_JUDGE_BASE_URL="http://0.0.0.0:6789/v1"

export LLM_HUMAN_MODEL="qwen3.5-27b"
export LLM_ASR_MODEL="qwen3.5-27b"
export LLM_JUDGE_MODEL="Gemma4-31B-it"
```

## Quick Start

### 1. Run stage-0 ASR

```bash
python scripts/run_stage0_asr.py \
  --data examples/example.jsonl \
  --output logs/example/loop_0.jsonl \
  --concurrency 4
```

### 2. Evaluate stage-0 outputs with S²ER

```bash
python evaluate.py \
  --input logs/example/loop_0.jsonl \
  --output logs/example/loop_0_eval.jsonl \
  --concurrency 4 \
  --prompts config/default_prompts.json
```

### 3. Run one interaction loop

```bash
python scripts/run_next_loop.py \
  --input logs/example/loop_0_eval.jsonl \
  --output logs/example/loop_1.jsonl \
  --concurrency 4 \
  --prompts config/default_prompts.json
```

### 4. Re-evaluate the updated hypotheses

```bash
python evaluate.py \
  --input logs/example/loop_1.jsonl \
  --output logs/example/loop_1_eval.jsonl \
  --concurrency 4 \
  --prompts config/default_prompts.json
```

## Data Format

Each benchmark sample is stored in JSONL format and must contain at least:

```json
{
  "id": "11226",
  "gt": "好久不见的酸奶燕麦 BOWL 啊还有雪梨",
  "audio_path": "examples/audio/11226.wav"
}
```

Optional metadata such as `category`, `difficulty`, and `metadata` will be preserved in outputs.

## Output Format

The pipeline incrementally appends fields to each JSONL record:

- `raw_pred`: initial stage-0 ASR hypothesis
- `is_correct`: exact-match correctness after normalization
- `is_semantic_correct`: semantic correctness judged by S²ER
- `total_loop`: number of completed correction rounds
- `loop_N_pred`: hypothesis after round `N`
- `loop_N_human_*`: user-side correction traces
- `loop_N_correction_asr_*`: ASR decoding traces on correction audio
- `loop_N_asr_refine_*`: ASR-agent refinement traces

This format is designed for:

- reproducible loop-by-loop experiments
- bad-case inspection
- semantic and token-level post hoc analysis
- visualization of interaction trajectories

## Documentation

- [docs/artifact_overview.md](/Users/zixuan/X-LANCE/AgenticASR/docs/artifact_overview.md:1)
- [docs/agentic_asr.md](/Users/zixuan/X-LANCE/AgenticASR/docs/agentic_asr.md:1)
- [docs/s2er.md](/Users/zixuan/X-LANCE/AgenticASR/docs/s2er.md:1)
- [docs/interactive_simulation_framework.md](/Users/zixuan/X-LANCE/AgenticASR/docs/interactive_simulation_framework.md:1)

## Roadmap

- release a more complete experiment configuration set for the public benchmarks used in the papers
- add standardized post-processing scripts for loop-wise summary tables and plots
- expand documentation for reproducing larger-scale studies beyond the toy example

## Limitations

- the repository depends on external ASR, TTS, and LLM services
- private training code, model weights, and internal deployment scripts are not included
- benchmark data used in the papers may have separate release constraints

## Citation

If you find this repository useful, please cite the corresponding paper(s).

```bibtex
@misc{interactiveasr2026_agentic,
  title={Interactive ASR: Towards Human-Like Interaction and Semantic Coherence Evaluation for Agentic Speech Recognition},
  author={Peng Wang and Yanqiao Zhu and Zixuan Jiang and Qinyuan Chen and Xingjian Zhao and Xipeng Qiu and Wupeng Wang and Zhifu Gao and Xiangang Li and Kai Yu and Xie Chen},
  year={2026},
  eprint={2604.09121},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@misc{interactiveasr2026_semantic,
  title={Towards Human-Like Interactive Speech Recognition With Agentic Correction and Semantic Evaluation},
  author={Zixuan Jiang and Yanqiao Zhu and Peng Wang and Qinyuan Chen and Xinjian Zhao and Xipeng Qiu and Wupeng Wang and Zhifu Gao and Xiangang Li and Kai Yu and Xie Chen},
  year={2026},
  eprint={2605.29430},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

## Acknowledgement

This public repository is the cleaned research artifact layer of the Interactive ASR project. It is intended to make the main ideas, protocols, and executable evaluation pipeline accessible to the community while keeping private training and serving infrastructure out of scope.
