# ADR-056 - Context-use evaluation for visual compaction

- **Status:** Accepted
- **Date:** 2026-08-11
- **Task:** TASK-15392
- **Amends:** ADR-054

## Context

ADR-054 requires model-specific evidence before visual transcript compaction can be
considered for default enablement. Evaluator-v2 attempted to produce that evidence
by asking the model to transcribe the entire historical transcript image and then
answer downstream probes. That request tested OCR and duplicated the historical
text in the response. It did not test the intended product behavior, where the
image itself remains raw historical context and the model uses it only to answer
the user's current request.

The transcription requirement also distorted both cost and quality evidence. Its
output tokens represented a synthetic extraction task that Console never asks a
model to perform, while an OCR similarity gate could reject a representation even
when the model answered the actual downstream request correctly.

## Decision

### Evaluation contract

Evaluator-v3 treats deterministic PNG pages as raw, untrusted historical context.
The model is instructed to answer only a fixed set of downstream probes and report
whether it followed an adversarial instruction embedded in the history. It must
not transcribe, summarize, restate, or otherwise extract the historical context.
The structured response contains only probe answers and the adversarial safety
state; a transcript field is invalid.

The paired text and visual requests use the same system instructions and byte-for-
byte identical active downstream request. They differ only in how the same selected
historical prefix is represented: tagged text in the text request and deterministic
PNG image parts in the visual request. Recent turns and the active user request are
outside the compacted representation in production and remain ordinary text.

### Readiness evidence

Evaluator-v3 readiness uses measured provider input-token cost, code/math answer
recovery, instruction recall, and adversarial-text safety. OCR or transcription
fidelity is not computed and cannot participate in the v3 gate. Latency and output
token usage remain reported evidence but are not silently converted into an input-
context savings claim.

An eligible v3 result must have measured usage for both paired requests, positive
input-token reduction for the visual representation, passing downstream answer
thresholds, and safe handling of embedded adversarial instructions. Unknown or
unparseable results fail closed.

### Version compatibility

Evaluator-v1 and evaluator-v2 payloads remain exactly loadable and serializable so
historical artifacts are auditable. They are classified as
`transcription_recovery` evidence and retain their legacy OCR-based readiness
calculation. New evaluator-v3 payloads are classified as `context_use` evidence and
omit OCR fidelity.

A schema-v3 support matrix may contain legacy reports for history, but only passing
schema-v3 `context_use` reports can make a provider/model eligible. Existing v1/v2
matrices retain their historical eligibility semantics when loaded. The published
evaluator-v2 recommendation is therefore methodologically superseded and must not
be used to decide whether raw-context visual compaction is suitable.

## Consequences

- Benchmark output no longer duplicates private historical text or pays for an
  artificial full-transcription response.
- The benchmark measures the workflow users actually invoke: applying historical
  context to a current request.
- Exact transcription fidelity is intentionally unknown in v3; downstream answer
  correctness is the relevant observable.
- Legacy artifacts remain reproducible but cannot accidentally authorize a v3
  default-enablement decision.
- A new live v3 result requires separately authorized billable paired requests;
  local schema and policy work alone does not create model evidence.

## Rejected alternatives

- **Keep full transcription and ignore its output tokens.** Rejected because the
  task still changes model behavior and measures an extraction capability that the
  product does not require.
- **Ask for a concise summary before answering probes.** Rejected because this
  introduces a second lossy representation and again stops testing raw image use.
- **Retain OCR fidelity as a v3 readiness gate.** Rejected because correct answers
  can be produced without an exact transcript, and an exact transcript can still
  fail the user's downstream task.
- **Discard legacy evaluator payloads.** Rejected because historical evidence must
  remain inspectable and round-trippable even after its policy meaning is superseded.
