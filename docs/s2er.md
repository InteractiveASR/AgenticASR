# S²ER

`S²ER` stands for `Sentence-level Semantic Error Rate`.

## Motivation

Character or word error rates are useful but incomplete. A transcript can look similar at the token level and still change the meaning of:

- a named entity
- a location
- a negation
- a time expression
- an English keyword in a code-switching utterance

S²ER is intended to capture that semantic layer.

## Formal Definition

For a sample with prediction `Pred` and ground truth `GT`, define a semantic error indicator:

- `E_sem = 0`, if `Pred` and `GT` are judged semantically equivalent
- `E_sem = 1`, otherwise

Then dataset-level `S²ER` is:

```text
S²ER = (1 / N) * Σ E_sem
```

where `N` is the number of valid evaluated samples.

In this repository:

- exact-match after normalization is treated as an immediate semantic match
- otherwise an LLM judge decides semantic equivalence
- the default judge protocol is bidirectional and multi-round

## Evaluation Protocol

1. normalize `Pred` and `GT`
2. if they match exactly, mark the sample as semantically correct
3. otherwise call an LLM judge
4. run multi-round consensus for stability
5. store `is_semantic_correct` next to the exact-match result

The public implementation now also supports:

- explicit `S²ER` computation at dataset level
- loop-wise `S²ER` reporting
- optional round-level judge traces through `--save-judge-trace`
- human-vs-judge agreement analysis

## Current Public Implementation

The public artifact exposes:

- exact-match evaluation after normalization
- LLM-based semantic judge
- bidirectional multi-round consensus through `call_judge_with_consensus`
- traced judge execution through `call_judge_with_trace`
- dataset-level metric utilities in `interactive_asr/s2er/metrics.py`
- human alignment utilities in `interactive_asr/s2er/human_alignment.py`

This repository therefore includes the executable evaluation layer that operationalizes the semantic metric used in the project.
