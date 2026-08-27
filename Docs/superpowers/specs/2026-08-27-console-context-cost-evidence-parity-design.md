# Console Context-Cost Evidence Parity Design

**Task:** TASK-2525 — Console context-cost estimate has three small modelling gaps

## Goal

Make the Console context and price estimates model staged local evidence using the same eligibility, normalization, formatting, and 64-source limit as the send path. The estimate remains a zero-I/O preview; the send-time authority check can still remove sources whose authority changes after staging.

## Current Problem

The estimator independently filters staged references and joins raw snippets with blank lines. The send path instead normalizes references, adds source headers and separators, and limits the formatted prompt to 64 sources. Consequently, the estimate can undercount prompt characters and report more sources than the send path can include. The duplicated `source_owner == "local"` predicate can also drift.

## Design

Add two small pure helpers beside the existing local citation normalization and formatting functions:

- A Console evidence adapter will accept the bundle's available `EvidenceReference` values, retain only local references, translate them into `NormalizedLocalResult` values through the existing `normalize_local_result()` boundary, and preserve their current order.
- A prompt-formatting wrapper will compute the send path's existing full-candidate character allowance and call `format_local_evidence_context()`. This prevents either consumer from accidentally using the formatter's unrelated 90-character default or duplicating the allowance formula.

The adapter preserves the send path's current mapping contract: `source_type` supplies source identity, `source_id` supplies lineage, a non-empty string `metadata["chunk_id"]` becomes both the result ID and chunk lineage, a missing score becomes `0.0`, and candidate ranks are assigned sequentially after successful normalization. Empty snippet text remains valid and produces a counted header-only prompt entry; changing that send policy is outside this task.

Both consumers will use these helpers:

- The send path will normalize with the adapter, perform its existing asynchronous authority re-check, and pass the surviving candidates to the shared prompt-formatting wrapper.
- The estimator will normalize with the adapter and immediately pass those candidates to the same prompt-formatting wrapper without I/O.

The estimator's prompted-source count will be the number of entries returned by the formatter. Its prompted-evidence text will be the formatter's exact context string. This makes the two estimates derive from one canonical formatted result and automatically includes the existing `[S#]` headers, source labels, title text, `\n---\n` separators, and 64-entry cap.

The wrapper will keep the send path's existing character allowance calculation, `sum(len(title) + len(content) + 32 for candidate in candidates)`, so this change does not introduce a second truncation policy. Per-entry UTF-8 limits remain enforced by `format_local_evidence_context()`. No cache or new abstraction layer is needed.

The affected docstrings and call-site comments will describe this result as the formatted **pre-authority estimate**, not as an unconditional claim about what reaches the model. The sent-notice path continues to prefer the capture result's authoritative repair-contract ordinals; its launch-only fallback is explicitly best-effort because it cannot reproduce send-time authority checks without I/O.

## Data Flow

1. Parse the staged evidence bundle from the Console launch context.
2. Select its available references.
3. Normalize eligible local references with the shared pure adapter.
4. For estimates, use the shared full-candidate formatting wrapper immediately and derive both text and count from the result.
5. For sends, retain the existing authority review before formatting.

Invalid or non-normalizable references remain excluded, matching current send behavior. Empty snippets remain included as header-only entries. Missing or invalid bundles continue to produce empty estimate text and a zero count.

## Boundaries and Non-Goals

- Estimation performs no filesystem, database, server, or authority-check I/O.
- A later authority change may make the final sent evidence smaller than the estimate; this is intentional and fail-closed.
- `console_staged_source_count()` continues to report the true staged total for the UI. Only `console_prompted_source_count()` and prompted evidence text reflect normalization, formatting omissions, and the 64-entry prompt cap.
- No database schema, UI layout, dependency, provider contract, or persistence behavior changes.
- The dependency direction is intentionally `Chat.console_display_state` to the pure `RAG_Search.local_citation_capture` helpers. That module imports only lower-level Chat citation models/builders, not Console display state.
- `test_console_send_consumes_staging_and_shows_the_sent_transient` has a separate deterministic baseline fixture failure on unchanged `origin/dev`: its second durable turn is refused before evidence capture because the mounted harness's thread-local `:memory:` SQLite connection has no `conversations` or `world_books` schema. This does not invalidate the pure estimator baseline. Any harness repair remains a focused test-infrastructure change, not a TASK-2525 production change, and must be kept separate from the modelling diff.

## Verification

Use red/green focused tests to prove:

- Prompted evidence includes the same headers and separators as the send formatter for multiple sources whose combined text exceeds the formatter's 90-character default.
- Count and text exclude blocked, server-owned, and otherwise non-normalizable references consistently, while an empty local snippet remains a counted header-only entry.
- An invalid reference between two valid chunked references is omitted while the valid results retain successful-normalization order, sequential ranks, chunk lineage, and score fallback semantics.
- A 65-reference bundle reports and formats 64 sources, omitting the final source; the estimator's formatted context matches the shared formatter exactly.
- A two-source Console case estimates both sources without authority I/O, while a send-time authority rejection removes one and re-formats the survivor.
- A one-time fresh-interpreter import command verifies the dependency direction; no permanent subprocess regression test is added unless a real import cycle is observed.
- Updated docstrings and call-site comments consistently distinguish the formatted pre-authority estimate from authoritative send results.
- Existing Console estimate tests continue to pass.
- Existing local citation formatting and send-boundary tests remain green.

Run only the targeted suites required by the repository guidance; a full sweep requires explicit user approval.

## ADR Check

ADR required: no  
ADR path: N/A  
Reason: This is a routine parity bug fix that reuses existing normalization, authority, and prompt-formatting boundaries without changing architecture or policy.
