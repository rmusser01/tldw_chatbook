---
id: TASK-32663
title: Keep Library import recovery messages readable
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 01:14'
updated_date: '2026-09-16 01:29'
labels:
  - library
  - ingest
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the component review with Library Import media source entry and recovery. Ensure users can read why Start is blocked or needs confirmation and recover by keyboard at compact sizes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Import gate and consent messages remain complete above Start at compact and wide sizes in both themes.
- [x] #2 Keyboard entry, invalid-source recovery and Clear preserve visible focus and entered metadata without submitting work unintentionally.
- [x] #3 Changing the gate between empty, blocked and ready preserves mounted form fields and reserves at least one row without overlapping Start.
- [x] #4 Targeted tests and private native wide/compact evidence qualify the repaired flow, with no raised size budgets or provider/server requests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/014-library-ingest-service-authority-and-recovery.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md (existing)
Reason: Bounded layout and recovery repair within the existing Import media canvas; no authority, storage, submission or navigation boundary change.

1. Reproduce recovery/consent text clipping with the real state builder and production CSS. Cover missing paths, empty/unsupported selections and long option or tooling messages at compact and wide sizes in both themes.
2. Use content-driven gate height with the existing minimum-row token if the fixed height is confirmed as the cause; preserve the docked commit bar and mounted form controls. Rebuild generated CSS from source.
3. Exercise real local preflight and keyboard Clear/re-entry with metadata retained. Keep source inspection and form staging separate from submission; no extraction/provider/server request.
4. Run targeted new and neighboring ingest, token/bundle and applicable size checks. Compare static diagnostics against eda1d6bd65 without raising budgets. Review the diff.
5. Run a private native wide/compact check, inspect captures in one batch and at most one confirmation, verify persistence and normal exit, update QA/guide/audit/task and commit locally. No full suite, push or dev integration.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed the Library Import media gate and keyboard recovery review. Recovery/consent text now wraps above Start while an empty gate reserves one row. The docked commit bar and mounted form controls remain in place; metadata survives Clear and re-entry. Updated the widget, source/generated Library CSS, new targeted journeys, guide and workflow audit.

271 distinct targeted checks pass. Eight compact clipping reproductions now pass. One inherited Parakeet model-directory row overflow also fails with exact base eda1d6bd65 production files; it is recorded for the next options review. Zero new Ruff diagnostics, new Python formatting and changed-range formatting pass, budgets are unchanged, and independent read-only review found no actionable issue. Test fixture corrections (missing media DB and unsupported tuple CSS_PATH) are documented in QA.

Private native run-002 passed 170x48 dark and 80x24 light with six inspected captures. Real local preflight and keyboard Clear/re-entry retained metadata; Tab reached visibly focused Start without activation. Re-entry correctly preserves the staged source. Ten SQLite integrity checks pass, media/messages/ingest jobs remain zero, and the source is unchanged. Normal terminal Quit returned exit 0 after one repeated key; owned shell closed. No full suite, import/provider/server/extraction request, push or dev integration.

QA: Docs/superpowers/qa/2026-09-16-ingest-entry/README.md. Next: per-type ingest options, starting with the inherited Parakeet directory-row overflow, then queue activity and recovery.

ADR required: no. Existing backlog/decisions/014-library-ingest-service-authority-and-recovery.md, backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/161-component-pattern-library.md apply; no boundary or token-value change.
<!-- SECTION:NOTES:END -->
