# Agentic ASR

`Agentic ASR` models speech recognition as a correction-driven interaction loop rather than a single decoding pass.

## Core Roles

- `HumanAgent`
  Simulates a user who reacts to an incorrect transcript with a concise spoken correction.
- `ASRAgent`
  Receives the correction audio and the current on-screen text, then decides whether the user is confirming or editing the transcript.

## Execution Path

1. base ASR decodes the original utterance into `raw_pred`
2. if the prediction is wrong, `HumanAgent` generates a natural-language correction
3. TTS synthesizes the correction with speaker conditioning
4. the correction audio is re-decoded by ASR
5. `ASRAgent` edits the on-screen hypothesis

## Why Two Agents

This split mirrors the actual interaction protocol:

- the user agent knows the intended text and produces a realistic correction signal
- the ASR agent does not see the ground truth and must recover the intended edit from the correction utterance

That separation is important. If the correction side directly rewrote the transcript, the task would collapse into text post-editing and lose the speech interaction constraint.

## Implementation Notes

- prompts are stored in `config/default_prompts.json`
- responses are constrained with XML-style tags for robust parsing
- the ASR-side edit is guarded by a text-change threshold to avoid catastrophic rewrites
