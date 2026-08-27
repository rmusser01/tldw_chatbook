# Console Context-Cost Evidence Parity Design

**Task:** TASK-2525 — Console context-cost estimate has three small modelling gaps

## Goal

Make the Console context and price estimates model staged local evidence using the same eligibility, normalization, formatting, and 64-source limit as the send path. The estimate remains a zero-I/O preview; the send-time authority check can still remove sources whose authority changes after staging.

## Current Problem

The estimator independently filters staged references and joins raw snippets with blank lines. The send path instead normalizes references, adds source headers and separators, and limits the formatted prompt to 64 sources. Consequently, the estimate can undercount prompt characters and report more sources than the send path can include. The duplicated `source_owner == "local"` predicate can also drift.

## Design

Add one pure helper beside the existing local citation normalization and formatting functions. It will accept staged `EvidenceReference` values, retain only available local references, translate them into `NormalizedLocalResult` values through the existing `normalize_local_result()` boundary, and preserve their current order.

Both consumers will use this helper:

- The send path will normalize with the helper, perform its existing asynchronous authority re-check, and pass the surviving candidates to `format_local_evidence_context()`.
- The estimator will normalize with the helper and immediately pass those candidates to `format_local_evidence_context()` without I/O.

The estimator's prompted-source count will be the number of entries returned by the formatter. Its prompted-evidence text will be the formatter's exact context string. This makes the two estimates derive from one canonical formatted result and automatically includes the existing `[S#]` headers, source labels, title text, `\n---\n` separators, and 64-entry cap.

The formatter will keep the send path's existing character allowance calculation so this change does not introduce a second truncation policy. No cache or new abstraction layer is needed.

## Data Flow

1. Parse the staged evidence bundle from the Console launch context.
2. Select its available references.
3. Normalize eligible local references with the shared pure helper.
4. For estimates, format immediately and derive both text and count from the result.
5. For sends, retain the existing authority review before formatting.

Invalid or non-normalizable references remain excluded, matching current send behavior. Missing or invalid bundles continue to produce empty estimate text and a zero count.

## Boundaries and Non-Goals

- Estimation performs no filesystem, database, server, or authority-check I/O.
- A later authority change may make the final sent evidence smaller than the estimate; this is intentional and fail-closed.
- No database schema, UI layout, dependency, provider contract, or persistence behavior changes.
- The pre-existing failing UI assertion in `test_console_send_consumes_staging_and_shows_the_sent_transient` is outside this modelling fix; it fails on an unchanged `origin/dev` checkout because the capture mock is invoked only for a send that has staged evidence.

## Verification

Use red/green focused tests to prove:

- Prompted evidence includes the same headers and separators as the send formatter.
- Count and text exclude blocked, server-owned, empty, and otherwise non-normalizable references consistently.
- A 65-reference bundle reports and formats 64 sources, omitting the final source.
- Existing Console estimate tests continue to pass.
- Existing local citation formatting and send-boundary tests remain green.

Run only the targeted suites required by the repository guidance; a full sweep requires explicit user approval.

## ADR Check

ADR required: no  
ADR path: N/A  
Reason: This is a routine parity bug fix that reuses existing normalization, authority, and prompt-formatting boundaries without changing architecture or policy.
