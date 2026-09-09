# Library reader: negative-evidence revision

Date: 2026-09-08 UTC. Task: [TASK-32029](../tasks/task-32029%20-%20Evaluate-question-directed-reading-of-selected-Library-sources.md).
Existing decision: [ADR-133](../decisions/133-question-directed-library-reading-experiment.md).
Baseline: [first DeepSeek development run](library-source-reader-deepseek-development-2026-09-08.md).

## Outcome

The revised reader retained the required negative/unspecified-information
evidence and delivered an answer for **both previously failing development
questions**, using four additional DeepSeek requests. The probe cost an
estimated **$0.001794**, bringing the cumulative total to **24 requests and
$0.005122**, within the existing 24-request/$1 authorization.

This is a focused regression probe, not a rerun of the full comparison. The
original results remain frozen and inconclusive. Positive control questions,
held-out evaluation, the retrieval baseline, and blind human grading still
need further qualification before any adoption or savings claim.

## Change and hypothesis

Only the reader's system instruction changed. It now treats explicit negative
facts, unspecified information, unresolved alternatives, and evidence rejecting
a question's premise as relevant findings. It requires uncertainty to remain
about what the source establishes: an unspecified fee must not become a claim
that no fee exists. Silence alone cannot support a positive or negative fact.

The baseline showed that empty findings originated in the model response, not
the quote validator. The hypothesis was that clarifying relevance would retain
the missing evidence without weakening validation or adding a fallback. The
JSON schema, source admission, quotation checks, answer instruction, models,
sampling settings, deadlines, and no-retry policy remain unchanged. No new ADR
is required for this prompt clarification under ADR-133.

## Focused results

| Development case | Previously, both repetitions | After revision, one probe |
| --- | --- | --- |
| Pet fee unspecified | Empty findings; no answer | Quoted “Pets are not discussed.”; answered that the source cannot establish a fee |
| No sign colour approved | Empty findings; no answer | Quoted “The committee did not vote. A proposal is not an approval.”; answered that no colour was approved |

Each revised finding passed host checks for exact source text, selected ID,
revision, and character span. No finding was rejected or omitted. Inspection
of the answers confirms the intended qualifications in these two cases; this
does not substitute for blind human grading or prove general semantic support.

The probe ran one worker and one main call per case. All four calls returned
usable normalized usage: 962 uncached input tokens, 256 cache-read tokens,
zero cache-write tokens, and 304 output tokens. The case latencies were 3.984
and 3.483 seconds. These calls were **peak-priced**, whereas the original run
was off-peak, so their dollar costs are not a controlled comparison with the
baseline. Dollar amounts are catalog estimates, not bill reconciliation.

## Verification and limits

Before changing the prompt, an offline check of the retained live responses
failed on `dev-03` with `no_evidence_found`. The check requires the specific
source quotations that were omitted and a nonempty final answer. The same
check passed for both revised probe attempts; no assertion merely checks that
the prompt contains the new wording.

The targeted reader, experiment, and real loopback-HTTP suites passed:
**47 tests**. Ruff lint and format checks passed for the changed module. The
environment still emits its existing Requests dependency warning. No full
suite was run.

The one-off probe reused the existing Library service, auxiliary gateway,
native DeepSeek adapter, session lifecycle, and evidence validator. Explicit
preflight and per-dispatch checks enforced the four remaining requests and
cumulative spending ceiling; no retries or fallback calls were introduced.
It preserved the original matrix and used no held-out question. The original
artifact hashes still match, and the answer-prompt hash and model resolutions
match the baseline. Only the reader-prompt hash changed.

A complete development rerun must include the original positive controls and
repeat the repaired cases before the held-out set is used. The current request
allowance is exhausted; no further provider call was dispatched. A general
no-relevant-evidence response policy remains outside this prompt-only revision.

## Frozen evidence

- [Probe configuration and budgets](library-source-reader-runs/2026-09-08-deepseek-negative-evidence-probe/probe_manifest.json)
- [Responses, exact findings and usage](library-source-reader-runs/2026-09-08-deepseek-negative-evidence-probe/attempts.json)
- [Exact auxiliary requests](library-source-reader-runs/2026-09-08-deepseek-negative-evidence-probe/requests.json)
- [Summary and limitations](library-source-reader-runs/2026-09-08-deepseek-negative-evidence-probe/summary.json)
- [Offline regression check](library-source-reader-runs/2026-09-08-deepseek-negative-evidence-probe/check_negative_evidence.py)
- [One-off probe source](library-source-reader-runs/2026-09-08-deepseek-negative-evidence-probe/probe_script.txt)
- [Artifact checksums](library-source-reader-runs/2026-09-08-deepseek-negative-evidence-probe/sha256.json)

These artifacts contain only nonsensitive fixtures and no credential or process
log. Request artifacts record the auxiliary boundary rather than final HTTP
bytes. The original blind grading packet and grading key remain unchanged.
