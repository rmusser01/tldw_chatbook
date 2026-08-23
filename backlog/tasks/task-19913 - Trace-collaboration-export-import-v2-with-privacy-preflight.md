---
id: TASK-19913
title: Trace collaboration export import v2 with privacy preflight
status: Done
assignee: []
created_date: '2026-08-22 18:31'
updated_date: '2026-08-23 17:09'
labels: []
dependencies:
  - TASK-19907
  - TASK-19910
  - TASK-19911
references:
  - >-
    Docs/superpowers/specs/2026-08-22-task-19907-trace-v2-exhaustive-collaboration-design.md
  - Docs/superpowers/plans/2026-08-22-task-19913-trace-v2-collaboration.md
  - >-
    backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make sharing a first-class, privacy-safe Trace workflow with a versioned causal event bundle, export preflight, and read-only collaborative import.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Export supports safe summary, redacted diagnostic, and explicit full-trace modes with a preflight inventory of sensitive, redacted, omitted, and truncated fields
- [x] #2 Trace format v2 preserves event identity, order, lineage, timing, schema version, redaction provenance, and missing-data reasons
- [x] #3 Imported bundles are visibly labeled read-only shared traces and never write conversation or trace data to local persistence
- [x] #4 Readers reject unsupported versions and integrity failures with actionable errors while retaining v1 import compatibility
- [x] #5 Round-trip, privacy, tamper, malformed input, and zero-database-write tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED pure bundle tests for safe-summary, default redacted-diagnostic, explicit full export, credential prohibition, one-pass privacy preflight inventory, canonical digest, normalized causal event round-trip, actionable v2 rejection, and retained v1 import.
2. Implement the smallest stdlib-only TraceExportProfile/TraceExportPreflight, canonical v2 builder/writer, and version-dispatched ImportedTrace reader using the existing atomic writer and pure snapshot boundary.
3. Mutation-check integrity verification, run the focused Chat export/import suite, then obtain independent specification review followed by code-quality/privacy review and resolve all findings.
4. Add RED production-CSS Pilot tests for the export preflight, default profile, explicit full confirmation, cancel/no-write, write failure, read-only shared title/metadata, error states, and zero persistence writes.
5. Implement the smallest responsive Textual dialog and screen integration, preserving x clear filters and adding w export trace plus o import trace under ADR-031; preserve ephemeral import and never attach database owners.
6. Run the focused UI/round-trip/privacy suites, one batched 60/80/100/120-column compositor pass with at most one correction, one post-UI Impeccable detector pass, then independent specification and Sr HCI/code-quality reviews until both are Ready Yes.
7. Refresh and integrate latest origin/dev, run full Trace collaboration verification and static checks, update docs/task notes, and close only when all five AC are proven.

ADR required: yes.
ADR path: backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md
Reason: ADR-080 already accepts the portable v2 data/security contract and explicit local-data egress workflow, so implementation follows it without a new ADR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a stdlib-only Trace v2 collaboration contract with safe-summary,
  redacted-diagnostic (default), and explicitly confirmed full-trace profiles.
  Every profile blocks credentials, records exact privacy field decisions and
  provenance, and writes atomically with a canonical SHA-256 content digest.
- Preserved normalized event identity, sequence, causal lineage, actor/run
  metadata, timing, status, and missing-data reasons. Import dispatch retains
  v1 compatibility while rejecting unsupported versions, malformed identity,
  prohibited credential content, and digest failures before projection.
- Added a responsive privacy-preflight export dialog and ephemeral read-only
  shared Trace import. The imported view surfaces profile, digest validity,
  unauthenticated-source status, privacy inventory, and the safe collaboration
  metadata in an inspectable import event; it receives no persistence owner.
- Preserved `x` as the established clear-filters recovery action and used `w`
  for export plus `o` for import under ADR-031. Export warns before full-trace
  egress and before replacing a file; import validation and export projection/
  writes run off the UI thread with visible progress and duplicate/stale guards.
- Sr HCI and specification reviews drove pinned privacy/inventory/status/profile
  information, honest integrity/authenticity language, visible compact keyboard
  focus cues, actionable failure recovery, and production-CSS compositor tests
  at 60/80/100/120 columns.
- Merged latest `origin/dev` and adopted its consolidated-CSS harness. Rebuilt
  the generated Trace/timeline widget-default sheets so full-detail, inspector
  scrolling, semantic brush theming, export, and filter overlays match the live
  app and satisfy the latest class-CSS allowlist ratchet.
- The final branch audit also restored best-effort terminal finalization when
  trace-step serialization fails, retained the one-retry recovery contract for
  database writes, and withheld reasoning fields that follow labeled steering
  prefixes in durable run logs.
- Verification: 444 Trace-focused tests passed; the compact-focus and
  consolidated-CSS checks passed 150 focused tests; and the final 35-file
  branch-affected sweep passed 1,697 tests with 2 loopback-listener tests
  skipped because the sandbox denied binding. Ruff and `git diff --check`
  passed. The repository's existing requests/pydub dependency warnings and
  pytest temporary-directory cleanup warnings remain.
- ADR required: yes. Existing ADR used:
  `backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md`.
<!-- SECTION:NOTES:END -->
