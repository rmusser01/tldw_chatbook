# Personal Context memory evaluation baseline

This is an offline measurement of existing Personal Context read paths, governed
by [ADR-182](../decisions/182-personal-context-memory-evolution.md) and tracked in
[TASK-25907.1](../tasks/task-25907.1%20-%20Establish-a-synthetic-Personal-Context-memory-evaluation-baseline.md).
The [execution plan](../../Docs/superpowers/plans/2026-09-25-personal-context-memory-baseline.md)
defines the frozen cases. No production behavior changes in this task.

## What is measured

The runner creates a fresh encrypted SQLite repository for each case, with an
in-memory key protector, deterministic identifiers and a fixed UTC clock. It
seeds through PersonalContextService, then invokes the actual profile search/get
tools and ProfileContextService snapshot builder. Direct gets establish controls
and test denials; they never fill missing search results.

The fixture contains 24 cases: 12 development and 12 held-out regression cases.
The holdout is visible in the repository, not a blind evaluation or statistical
claim about generalization. It covers lexical and paraphrase queries, Unicode,
metadata-only false positives, workspace exceptions, changed preferences,
missing/ambiguous source metadata, quoted instructions, deletion, archiving,
expiry, private records, unrelated scopes, revocation and whole-record budgets.
Synthetic quoted instructions remain data.

Three independent label sets serve different purposes:

- `eligible_labels`: records allowed by the current live authority rules.
- `relevant_labels`: eligible records that answer the query, labelled before any
  retrieval changes. An empty set does not mean the tool failed.
- `expected_context_labels`: records expected after standing priorities,
  workspace overrides and context budgets. Context need not match the query.

The manifest is [memory_baseline_v1.json](../../Tests/Personal_Context/fixtures/memory_baseline_v1.json).
It includes all records, labels, mutations and synthetic source-message labels.
The loader caps input at 256 KiB, at most 32 record prototypes and 32 cases,
eight records/actions per case, and bounded text, budgets and clock advances.
The frozen version uses K=3; the scorer accepts only integer K from 1 to 20.

## Scoring and gates

For each applied search, precision@K is the number of relevant returned records
divided by K, even when fewer than K results were returned. Recall@K divides hits
by the number of relevant labels. Reciprocal rank is 1 divided by the first
relevant result's position, or zero when a nonempty gold set has no hit. For
empty relevant sets, recall and reciprocal rank are null; false-positive counts
and empty-result status remain reported. A perfect singleton result therefore
has precision@3 of 1/3, not 1.

Split/category summaries are macro means with explicit denominators. Deliberate
permission denial is not a ranked search. Unexpected status, malformed output,
duplicate/unknown IDs, stale versions and contradictory snapshot payloads are
contract errors with no ranking credit. Exceptions outside the enumerated
contract fail the run rather than becoming empty successful results.

Each case first proves a successful search, get and context snapshot for an
authorized control. The revoked case proves those before disabling runtime.
Forbidden records are checked in actual search, get and snapshot outputs using
IDs and synthetic payload canaries. Context checks compare exact selected
versions/payloads, the 12,288-byte ceiling, and the ten-percent input-token cap.

`disclosure_checks_passed` requires no harness errors, live-authority failures or
observed policy gaps. `context_checks_passed` separately requires no harness or
context errors. Passing pytest means measurements meet their contracts; it does
not turn a failed product-policy check or retrieval miss into a pass.

The deterministic token mode is `production_chars_fallback_v1`: the existing
production character estimator with optional tokenizer paths temporarily
disabled and caches cleared before/after measurement. Flags are restored even
on exceptions. This does not measure every provider tokenizer. Existing pytest
configuration/keyring/network isolation remains active; use pytest, not a
standalone import against your normal user configuration.

## Evidence and limits

The measured artifact is
[baseline-v1.json](../../Docs/superpowers/reviews/evidence/personal-context-memory/baseline-v1.json).
It contains symbolic fixture labels and numeric observations, not temporary
paths, keys, runtime profile IDs or source bodies. Source references retained in
the metadata observations are synthetic labels, not resolved source locators.

The initial report measured 24 cases with no harness errors, no live-authority
failures and no context selection/budget failures. It deliberately records
`disclosure_checks_passed: false` because both known policy gaps were observed:
`device_only_in_provider_context` (d10) and `unscoped_quarantine_signal` (h11).

| Partition | Ranked / total cases | Precision@3 (denominator) | Recall@3 (denominator) | Reciprocal rank (denominator) | False positives |
| --- | --- | --- | --- | --- | --- |
| Development | 12 / 12 | 0.2778 (12) | 0.8000 (10) | 0.8000 (10) | 1 |
| Held-out regression | 11 / 12 | 0.2424 (11) | 0.8000 (10) | 0.8000 (10) | 0 |

Four relevant queries returned no hit: reordered words `replies concise` (d02),
the paraphrase `short answers` (d03), `Straße` (h01), and `東京` (h02). The
provenance-only query `settings_edit` returned one irrelevant record (d04).
These observations match the current tool's substring search over serialized
records. The Unicode values are escaped in that serialization. They motivate
field-aware lexical/Unicode matching in TASK-25907.4; semantic paraphrase
retrieval is not promised by that lexical change.

Fixture SHA-256:
`64b076162e0e6c46d641c970cbc067f2caa5f37322f5cc9451949e109287243c`.
The runtime under test is commit
`1f0cc0e6e4a5bded03ec64312907e1fa332aa21d`, in the isolated
`codex/personal-context-memory-baseline` worktree. The shared checkout's unrelated
changes were excluded, including its token-window-resolution changes; the
estimator exercised here was unchanged by that diff. Re-measure after runtime
integration. Harness revision and final verification are recorded in the
[execution review](../../Docs/superpowers/reviews/2026-09-25-personal-context-memory-baseline-execution-review.md).

No model or provider is called. Generated-answer quality, semantic source
support, edit history and provenance UI are explicitly unmeasured. A quoted
instruction case does not prove prompt-injection resistance. The suite measures
candidate model context, not network egress or server enforcement. Device-only
eligibility under today's runtime is distinct from the stronger ADR-102 promise;
both are reported without weakening the policy. An unscoped unsupported-record
flag is separately classified as a policy gap.

## Field-aware lexical follow-up

TASK-25907.4 re-ran the unchanged 24-case fixture through the same real
offline service, search/get tool and context snapshot paths. The
[separate lexical report](../../Docs/superpowers/reviews/evidence/personal-context-memory/lexical-v1.json)
has the same fixture SHA-256 as `baseline-v1.json`; the original report and
labels were not edited. The held-out partition is visible regression data,
not evidence of generalization beyond these cases.

| Partition | Recall@3 before → after | Precision@3 before → after | Reciprocal rank before → after | False positives before → after |
| --- | --- | --- | --- | --- |
| Development | 0.8000 → 0.9000 (10) | 0.2778 → 0.3056 (12) | 0.8000 → 0.9000 (10) | 1 → 0 |
| Held-out regression | 0.8000 → 1.0000 (10) | 0.2424 → 0.3030 (11) | 0.8000 → 1.0000 (10) | 0 → 0 |

The changed case outputs are exact and limited: reordered `replies concise`
finds `brief` (d02); provenance-only `settings_edit` no longer returns it
(d04); `Straße` finds `accented` (h01); and `東京` finds `cjk` (h02).
Context selections did not change in these cases. The semantic paraphrase
`short answers` still misses `brief` (d03), as expected for lexical matching.
No harness, live-authority or context-budget failure was observed. The two
existing policy gaps remain in the new report, so
`disclosure_checks_passed` is still false; this task does not repair provider
disclosure or the unscoped quarantine signal.

The matcher normalizes NFKC plus case folding, preserves accents, treats
technical names as whole tokens, and matches only semantic-key and
human-readable payload fields. It adds no persistent index or network call.
The final review found that an initial regex dropped combining marks left by
case folding; the final scanner keeps them attached to their base token.
The post-fix report is byte-identical to the committed lexical report, and
the full baseline test file passed 65 tests including a second-root rerun.
In a bounded pure-matcher probe after the Unicode boundary fix, 128 synthetic
records with 1 KiB value fields matched in a median 5.41 ms over five runs on
Python 3.12.11. That timing
excludes the existing encrypted repository/export snapshot read, which the
production callers still perform before matching; it is not an end-to-end
latency claim or CI threshold.

## Reproduce safely

From an isolated checkout with the repository's development environment:

```bash
.venv/bin/python -m pytest Tests/Personal_Context/test_memory_baseline.py Tests/Agents/test_profile_tool_provider.py Tests/Personal_Context/test_context_service.py -q
.venv/bin/python -m ruff check Tests/Personal_Context/memory_baseline.py Tests/Personal_Context/test_memory_baseline.py
.venv/bin/python -m ruff format --check Tests/Personal_Context/memory_baseline.py Tests/Personal_Context/test_memory_baseline.py
```

To write a fresh report, use an absent output path in an existing temporary
directory. For example:

```bash
report_dir=$(mktemp -d)
TLDW_MEMORY_BASELINE_REPORT="$report_dir/baseline.json" .venv/bin/python -m pytest Tests/Personal_Context/test_memory_baseline.py::test_write_memory_baseline_report -q
```

The writer uses exclusive creation and refuses existing files, directories and
missing parents. It removes its partial newly created output if writing fails.
Normal test runs produce no durable report. Compare parsed JSON with the frozen
artifact; equality is independent of temporary-root names. No full test sweep
is needed or authorized by these commands.

Version fixtures when labels, case membership or scoring semantics change.
Review the fixture hash and baseline diff explicitly. Never silently overwrite
initial evidence or tune on held-out cases and still claim an independent
holdout. Re-measure after integrating runtime changes, and preserve separate
development/held-out/category outcomes alongside aggregate scores.

## Quarantine-signal follow-up

TASK-25907.11 removes the unscoped unsupported-record hint from model
serialization and its effect on whole-record packing. The
[separate quarantine report](../../Docs/superpowers/reviews/evidence/personal-context-memory/quarantine-signal-v1.json)
uses the unchanged 24-case fixture and preserves both earlier reports.

The first native report retains all retrieval summaries, returned/selected
labels and successful controls. Only h11 observations change: context size
467 → 432 bytes, estimated tokens 138 → 128, and its unscoped quarantine
policy gap disappears. Harness, live-authority and context checks remain empty.
The device-only gap in d10 remains, so disclosure_checks_passed is false.
This result repairs one metadata leak; it does not qualify provider egress,
destination enrollment or the broader V2 contracts.

The report SHA-256 is
1ef924748fc91a34a3db778ac55adc73d6f3d1ad594af8d93a950fd66be9b363.
The 172-case affected run and three root/child controls passed (175 distinct
targeted cases); the independent fresh-root report was byte-identical. A
reviewer-requested empty-state assertion passed a focused repeat. Details and
static-check limits are recorded in the
[execution review](../../Docs/superpowers/reviews/2026-09-25-personal-context-quarantine-signal-removal-review.md).
