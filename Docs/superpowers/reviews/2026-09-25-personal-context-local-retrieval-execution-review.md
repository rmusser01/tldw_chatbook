# Personal Context local retrieval execution review

Date: 2026-09-25
Task: TASK-25907.4
Status: Complete after one reviewed Unicode boundary fix.

Plan: [Local retrieval implementation](../plans/2026-09-25-personal-context-local-retrieval.md)
Design: [Memory evolution, section D](../specs/2026-09-25-personal-context-memory-evolution-design.md#d-field-aware-lexical-recall)
ADR: [ADR-182](../../../backlog/decisions/182-personal-context-memory-evolution.md)
Tracker: [Memory roadmap](../../../backlog/docs/personal-context-memory-roadmap.md)

## Delivered behavior

`profile_search` now compiles a bounded query once and ranks only positive
matches from its existing eligible record view. A shared pure matcher inspects
the semantic key and human-readable payload text, rather than serialized model
JSON. Provenance, IDs, hashes, field names and controls cannot qualify a record.
Ranking uses distinct matched terms, subject hits, same-field phrase hits and
stable identity ties. The existing 1–20 result limit and response shape remain.

Automatic context uses the same positive-match predicate only for its existing
preference and working-context relevance group. Hard priorities, workspace
overrides, whole-record budgets, authorized candidate ownership and root-run
pinning remain with the context service. The change adds no storage, index,
embedding, provider call, source lookup, new dependency or extra repository
pass. The pre-existing export-snapshot read remains an end-to-end cost.

## Measured evidence

The unchanged 24-case fixture and initial `baseline-v1.json` remain frozen;
the [lexical report](evidence/personal-context-memory/lexical-v1.json) uses the
same fixture SHA-256,
`64b076162e0e6c46d641c970cbc067f2caa5f37322f5cc9451949e109287243c`.
The final post-fix report generated from a fresh temporary root was
byte-identical to the committed lexical report (SHA-256
`7e116019aa7b6dc7277e1c3fa0e302249ed7295a65dd1d2bef49c3a36f5e3486`).
The baseline file passed 65 tests, including a second-root reproducibility
run. The targeted matcher, profile-tool and context run passed 68 tests; the
related Chat and agent snapshot run passed 23. These are 156 distinct targeted
tests, not a full-suite claim.

| Partition | Recall@3 before → after | Precision@3 before → after | Reciprocal rank before → after | False positives before → after |
| --- | --- | --- | --- | --- |
| Development | 0.80 → 0.90 (10) | 0.2778 → 0.3056 (12) | 0.80 → 0.90 (10) | 1 → 0 |
| Held-out regression | 0.80 → 1.00 (10) | 0.2424 → 0.3030 (11) | 0.80 → 1.00 (10) | 0 → 0 |

Only four search outputs changed: reordered `replies concise` gained `brief`
(d02); metadata-only `settings_edit` lost that false hit (d04); `Straße`
gained `accented` (h01); and `東京` gained `cjk` (h02). Context selections did
not change in those cases. The semantic paraphrase `short answers` still
misses `brief` (d03). The reports have no harness, live-authority or context
budget failures. The observed `device_only_in_provider_context` (d10) and
`unscoped_quarantine_signal` (h11) policy gaps remain, so disclosure checks
still fail; this task does not claim provider-disclosure repair.

The bounded pure-matcher probe used 128 synthetic records with 1 KiB value
fields on Python 3.12.11. After the review fix its median of five runs was
5.41 ms, up from 1.56 ms before that fix. The test asserts the functional
count, not a machine-specific latency threshold. This excludes encrypted
repository reads, decryption, provider work and real-profile distributions.

## Independent review and resolution

One fresh read-only reviewer found no Critical issue. It found one Important
whole-token defect: the original regex dropped combining marks remaining after
NFKC and case folding. `i` could match `İstanbul`, `q` could match `q\u0301`,
and `café` could match `café\u0307`. Three negative regression cases failed
before the fix; a Unicode category scanner keeps marks attached to their base
token. All 24 matcher cases and the final affected runs passed afterward.
Technical-name behavior for `C`, `C++`, `C#` and `.NET` stayed covered.

One Minor coverage gap is deferred: the four-record ranking test has distinct
scores and does not directly prove equal-score identity ordering or exact
truncation of more than 20 matches. The implementation sorts by identity and
applies the existing limit after sorting; no user-visible defect was found.
The reviewer did not certify the final pytest receipts, independent timing,
full-suite or real-profile behavior, provider/network disclosure, UI,
semantic paraphrases or multilingual segmentation. The parent executor checked
the receipts and kept claims bounded to these synthetic offline paths.

The new matcher and its tests passed Ruff and formatting. Changed-line Ruff
was clean across the retrieval diff; whole-file Ruff on older tool files still
reports pre-existing findings. `git diff --check`, scoped task-ID validation,
local links and acceptance-text/dependency checks passed. The full suite was
not run under the repository's targeted-test rule.

## Acceptance and closeout

| Criterion | Evidence |
| --- | --- |
| 1: Meaningful-field positives and metadata exclusion | Pure matcher, production search and frozen d04 controls |
| 2: Positive any-term match and deterministic ranking | Caller scores, phrase/subject tests and stable sort/limit code |
| 3: Frozen Unicode/punctuation/bounds contract | Pre-code plan, technical-name/CJK/accent tests and reviewed mark fix |
| 4: Shared matcher with distinct context semantics | Production caller tests and unchanged baseline context selections |
| 5: Improved frozen lexical recall without authority regression | Separate development/held-out report, 65 baseline tests and honest d03 miss |
| 6: Local bounded path and cost disclosure | Code review, 128-record probe and explicit export-read limit |
| 7: Targeted production, privacy, international and limit checks | 156 targeted tests; exact equal-score/20 truncation pin deferred as Minor |

Existing ADR-182 governs the boundary; no new ADR was required. The specific
normalization trap and its measured cost are recorded in
[lessons-testing-evidence](../../../backlog/docs/lessons-testing-evidence.md#case-folding-can-leave-combining-marks-outside-a-unicode-regex-token).
This branch stays local. No push, PR or merge is part of this slice.
