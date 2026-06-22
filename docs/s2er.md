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

## Evaluation Protocol

1. normalize `Pred` and `GT`
2. if they match exactly, mark the sample as semantically correct
3. otherwise call an LLM judge
4. run multi-round consensus for stability
5. store `is_semantic_correct` next to the exact-match result

## Current Public Implementation

The public artifact exposes:

- exact-match evaluation after normalization
- LLM-based semantic judge
- bidirectional multi-round consensus through `call_judge_with_consensus`

This repository therefore includes the executable evaluation layer that operationalizes the semantic metric used in the project.
