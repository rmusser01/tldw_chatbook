# Library source reader: DeepSeek development run

Date: 2026-09-08 UTC. Task: [TASK-32029](../tasks/task-32029%20-%20Evaluate-question-directed-reading-of-selected-Library-sources.md).
Governance: [ADR-133](../decisions/133-question-directed-library-reading-experiment.md)
and [approved design](../../Docs/superpowers/specs/2026-09-07-library-source-reader-design.md).

## Result

**Revise the reader before further evaluation; this run does not support adoption
or a savings claim.** The approved development run completed with 20 generation
requests and an estimated model cost of **$0.003328**, within the $1 and 24-request
ceilings. Four reader attempts returned no findings and consequently made no
answer request. All eight direct attempts returned answers. An answer being
present is not a human correctness grade.

The formal [comparison report](library-source-reader-runs/2026-09-08-deepseek-development/report.json)
is **inconclusive**: the retrieval arm is unavailable and blind human grades are
missing. It also retains all four reader failures. Held-out questions were not
submitted, and no product behavior changed.

| Development arm | Model requests | Answers returned | No-findings failures | Estimated total USD |
| --- | ---: | ---: | ---: | ---: |
| Direct Pro reading | 8 | 8/8 | 0 | $0.001765 |
| Flash reader, then Pro answer | 12 | 4/8 | 4 | $0.001563 |
| Selected-source retrieval | 0 | Unavailable | Unavailable | Unknown |

The reader's apparent aggregate discount comes from skipped answers. On each
attempt where it produced an answer, it cost **1.33–1.74 times** direct reading
and added **0.64–1.65 seconds**. These are diagnostics for the four completed
pairs, not a success-only replacement for the frozen eight-pair comparison.
The report's eight-pair median cost ratio is 0.752, but half the reader attempts
failed; the decision gate correctly refuses to interpret that as savings.

## Observed behavior

Each development question ran twice, with arm order fixed before generation.

| Question | Direct behavior, both repetitions | Reader behavior, both repetitions |
| --- | --- | --- |
| Wednesday garden opening | Answered 11:00 | Located the Wednesday exception and answered 11:00 |
| Tandem bicycles on the ferry | Answered yes, with advance booking | Retained the booking qualification and answered yes |
| Ferry pet fee | Said the source does not state a fee | Returned `{"findings":[]}`; no answer generated |
| Which sign colour was approved | Said no colour was approved because no vote occurred | Returned `{"findings":[]}`; no answer generated |

This is an inspection of development outputs, **not blind human grading**. The
frozen rubric also asks for ordinary 09:00 opening in the Wednesday question;
both arms omit that detail. Preserve the rubric and account for it in grading
rather than changing expected facts after seeing answers.

The empty findings expose two different cases. The pet fee is unspecified, and
the source explicitly says pets are not discussed. The sign source contains
positive evidence of a negative answer: the committee did not vote and a
proposal is not an approval. The reader discarded both kinds of relevant
evidence. Exact-quotation validation cannot detect this omission because it only
validates quotations the worker actually returns.

Four returned findings passed the host's exact-quote, selected-source, revision,
and span checks, with no rejected or omitted finding. This supports those
mechanical checks for this run; semantic support and coverage remain ungraded.

## Execution and accounting

- Models: `deepseek-v4-pro` for answers, `deepseek-v4-flash` for reading, through
  the official DeepSeek endpoint and existing sensitive auxiliary/native adapter.
- Both models used explicit non-thinking Chat Completions, temperature 0,
  top-p 1, no seed, no tools, and no automatic retries or fallback calls.
- Four synthetic development questions, two repetitions, and three frozen arm
  rows per pair produced 24 attempt rows. Eight retrieval rows remain
  `index_unavailable`; the 16 runnable rows dispatched 20 of 24 reserved calls.
- All 20 requests returned usable normalized usage with status `ok`. All
  intervals were off-peak. Totals: 2,436 uncached input tokens, 1,664 cache-read
  tokens, zero cache-write tokens, and 1,189 output tokens.
- Costs are catalog estimates computed from provider-reported usage and recorded
  pricing, not a reconciliation with the account's bill. Cache effects are
  observed provider behavior; this is not a controlled cold-cache comparison.
- Direct latency ranged up to 2.494 seconds; the slowest reader attempt was
  3.504 seconds. No request reached the result deadline.
- The executed manifest matches the reviewed preflight except `dry_run: false`.
  No follow-up or held-out request was dispatched.

The CLI exited 0 and offline report generation exited 0. Artifact verification
confirmed preflight equality, the 20-call/24-row counts, usage/cost completeness
for dispatched calls, preservation of all four empty-reader attempts, and
unknown retrieval costs. No runtime code was edited during this live run.

## Next experiment changes

1. Clarify the reader instructions to extract explicit negative facts and
   statements that information is unspecified. Keep the distinction between an
   exact source statement and a model's unsupported assertion of absence.
2. Define a useful no-evidence outcome without silently adding direct-reading
   fallbacks or counting a missing answer as success. Any fallback must be a
   separately declared, costed experiment condition.
3. Keep these tiny documents as failure probes. They do not establish whether
   reading longer sources can offset the extra call and evidence-envelope cost.
   Preserve the untouched held-out set until the development behavior is ready.
4. Connect the selected-source retrieval baseline, freeze its configuration and
   embedding/index accounting, and obtain blind human grades before an adoption
   decision. Do not ship automatic routing from these development observations.

## Frozen evidence

The nonsensitive run artifacts are preserved in
[the evidence directory](library-source-reader-runs/2026-09-08-deepseek-development/run_manifest.json):

- [Attempts, answers, usage and validated findings](library-source-reader-runs/2026-09-08-deepseek-development/attempts.json)
- [Exact auxiliary requests](library-source-reader-runs/2026-09-08-deepseek-development/requests.json)
- [Frozen fixture manifest](library-source-reader-runs/2026-09-08-deepseek-development/fixture_manifest.json)
- [Blind grading packet](library-source-reader-runs/2026-09-08-deepseek-development/blind_grading_packet.json)
- [Artifact SHA-256 checksums](library-source-reader-runs/2026-09-08-deepseek-development/sha256.json)

Give a human reviewer only the blind packet; the separate grading key belongs
to the operator. The request artifact captures the auxiliary boundary rather
than provider-final HTTP bytes, and served model identity remains the configured
gateway identity. The run used only disposable fixture data and existing
credentials; no credential or process log is included in the evidence directory.
