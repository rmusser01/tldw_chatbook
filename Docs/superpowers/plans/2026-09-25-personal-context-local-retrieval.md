# Personal Context Local Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. The user retained native execution. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve ordinary agent-visible Personal Context recall through deterministic field-aware lexical matching without changing authority, storage or provider contracts.

**Architecture:** A small pure matcher compiles one bounded query and scores only semantic-key and human-readable payload fields. `profile_search` ranks positive matches from its existing eligible view; the automatic context selector uses the same positive-match predicate only for its existing relevance priority group. The frozen baseline is remeasured unchanged, with a separate post-change report.

**Tech Stack:** Python 3.12+, existing `tldw_profile_core` models, encrypted SQLite Personal Context service, offline pytest baseline.

**Spec:** [Approved memory evolution design, section D](../specs/2026-09-25-personal-context-memory-evolution-design.md#d-field-aware-lexical-recall)

**Backlog:** TASK-25907.4 — Done. Seven criteria checked against implementation evidence and review.

ADR required: yes — existing ADR-182 applies; no new ADR.
ADR path: backlog/decisions/182-personal-context-memory-evolution.md
Reason: ADR-182 already approves local field-aware lexical retrieval without schema, provider or storage changes.

## Frozen matching contract

- Normalize each bounded input with Unicode NFKC followed by `casefold()`. Preserve accents; composed and decomposed forms normalize together, but `cafe` does not become `café`, and no synonym or language-aware segmentation is implied.
- Scan at most 4,096 query code points, keep the first 32 distinct terms, and scan at most 16,384 code points per human-readable field. These are per-input work limits; the existing authorized candidate view still determines the record count. Do not add another repository/export read.
- A term is a Unicode letter/number run with combining marks attached to the preceding base character, optional internal `.`, `_` or `-`, an optional leading dot for `.NET`, and an optional `+`/`#` suffix for `C++`/`C#`. `/` and whitespace separate terms. Keep technical terms distinct: `C`, `C++`, `C#` are not aliases. A field also offers component aliases for `.`, `_` and `-` compounds (`response.detail` offers `detail`); a query's distinct-term count uses its original terms, not aliases. A single-character term is usable and matches a whole token only.
- Inspect `semantic_key.subject` and non-default `semantic_key.namespace` plus payload `subject`, `value`, `outcome`, or legacy `text` when present. Do not inspect schema keys, kind labels, polarity, record IDs, versions, provenance, source references/hashes, controls, timestamps or JSON field names. A namespace identical to the record kind is a technical default, not a content hit.
- Match any positive query term. Rank search by `(distinct matched terms, distinct subject-term matches, exact ordered query-token phrase in one field)`, descending, then by `record_id` and `version_id` ascending. Phrase tokens must be contiguous in one original field; joining fields never creates a phrase. Search with zero usable terms or zero positive matches returns an empty applied result. Keep the schema's existing 1–20 result limit.
- Automatic context uses only the positive-match predicate to place preferences and working context into its existing relevance group. Its hard correction/constraint priorities, workspace semantic-key override, whole-record byte/token budgets and root-run pinning remain unchanged; unmatched eligible records may still appear under its standing group.
- Freeze these cases before code: `C`, `C++`, `C#`, `.NET`, `response.detail`/`detail`, `café`/decomposed `café` versus plain `cafe`, exact `東京` and a no-segmentation `東京駅` boundary, one-character `C`, `Straße`/`strasse`, reordered `replies concise`, zero matches, provenance-only `settings_edit`, and same-field versus cross-field phrases.

## Considered approaches

1. **Pure shared matcher over authorized records (chosen).** The service and tool keep their own authority/result rules; a small module owns bounded normalization and scoring. It is testable without a database and adds no index or dependency.
2. **Independent regex changes in each caller.** Less initial plumbing but the two relevance definitions would drift and could count metadata differently. Rejected.
3. **SQLite FTS or embeddings.** They add indexing, plaintext/retention and schema costs unrelated to the measured lexical misses. Rejected by ADR-182 for this slice.

## Task 1: Freeze and implement the pure lexical matcher

**Files:** Create `tldw_chatbook/Personal_Context/lexical_match.py` and `Tests/Personal_Context/test_lexical_match.py`.

**Interfaces:** `compile_query(text: str) -> LexicalQuery` returns frozen original terms and phrase tokens; `match_record(record: ProfileRecord, query: LexicalQuery) -> LexicalMatch | None` returns a frozen score only after a positive content match. Neither function reads a repository or logs input.

- [x] **Step 1: Write failing table tests** for every frozen technical-name, punctuation, accent, CJK, single-character, compound-alias and metadata exclusion case. Construct canonical `ProfileRecord` fixtures with synthetic IDs/provenance, and assert score tuples plus no match for provenance-only inputs. Add a bounded 4,096/32 query and 16,384-per-field test.
- [x] **Step 2: Run `Tests/Personal_Context/test_lexical_match.py` RED**; require missing-interface failures, not malformed canonical fixtures.
- [x] **Step 3: Implement the pure matcher.** Tokenize normalized strings with bounded Unicode category scanning, retain original field tokens for phrase checking, add only compound aliases to the field's term set, and compare the query's distinct original terms with subject/all-field sets. Return `None` for no positive intersection; compute the three-part score otherwise. Do not serialize the record. The final review found that the initial regex dropped post-casefold combining marks; a category scanner preserves whole-token boundaries for dotted-I and non-composable accents.

```python
query = compile_query("replies concise")
match = match_record(record, query)
assert match is not None and match.distinct_terms == 1
```

- [x] **Step 4: Run the matcher tests GREEN**, Ruff/format/whitespace, and commit this independently testable unit.

## Task 2: Wire both production callers without changing their owners

**Files:** Modify `tldw_chatbook/Agents/profile_tool_provider.py`, `tldw_chatbook/Personal_Context/context_service.py`, `Tests/Agents/test_profile_tool_provider.py`, `Tests/Personal_Context/test_context_service.py`.

**Interfaces:** Search calls `compile_query` once, `match_record` for each record yielded by its existing `_eligible_records()` and sorts positive rows by score plus stable identity before applying `request.limit`. Context calls the same matcher after its existing eligibility/override checks and passes a boolean into `_priority_group`; no search-result shape enters snapshots.

- [x] **Step 1: Write failing production-entry tests**: reordered terms and Unicode positive controls; provenance/ID/hash/field-name-only zero matches; multi-term score, subject and same-field phrase tie-breakers; stable 1/20 limits; private, expired, conflicted and foreign-workspace denials; context relevance priority versus correction/constraint, overrides, whole-record budgets and root-run snapshot parity. Spy on the service's authorized-view read to prove no added repository pass.
- [x] **Step 2: Run focused caller tests RED** and inspect that failures are caused by current serialized substring and ASCII relevance behavior.
- [x] **Step 3: Replace only matching calls** with the Task 1 functions. Preserve `_eligible_records`, live scope validation, result serialization, existing context ordering and packer. Compile the query once per caller invocation; no provider/network call or persistent index.
- [x] **Step 4: Run the affected caller tests GREEN** plus the existing targeted profile-tool and context suites. Check changed-line Ruff, formatter and whitespace, then commit the caller unit.

## Task 3: Remeasure the frozen baseline and close the tracker

**Files:** Add `Docs/superpowers/reviews/evidence/personal-context-memory/lexical-v1.json` and an execution review; update `backlog/docs/personal-context-memory-evaluation.md`, `backlog/docs/personal-context-memory-roadmap.md`, TASK-25907.4 and this plan. Do not alter `memory_baseline_v1.json` or the initial `baseline-v1.json`.

- [x] **Step 1: Run the baseline through its real offline service/tool/context paths** with the unchanged fixture SHA and a fresh temporary root. Write a new deterministic report with `write_report`'s create-only behavior; compare development and held-out metrics and each declared case against the frozen baseline. Report paraphrase misses and existing disclosure gaps as measured, never as passes.
- [x] **Step 2: Measure bounded synthetic matching cost** over a fixed set of 128 authorized synthetic records and 1 KiB fields using `perf_counter`, recording Python version, median of five runs and candidate count. Assert only the functional result count in the test; do not turn one machine's timing into a flaky CI threshold. State the existing full-export read cost separately.
- [x] **Step 3: Run targeted baseline, matcher, tool, context and relevant agent tests**, static checks, local links, task-family IDs/59 unchanged child acceptance texts, backward dependencies and `git diff --check`. Request one fresh read-only whole-slice review focused on authority, metadata, Unicode/technical tokens, ranking determinism and cost. Reproduce and fix blocking findings with RED→GREEN tests; record exclusions and limits.
- [x] **Step 4: Check all seven acceptance criteria against evidence, add implementation notes and existing ADR-182 through Backlog CLI, mark Done only when DoD holds, update roadmap/evaluation docs and commit the review.** Keep the branch local; no push, PR or merge is part of this slice.
