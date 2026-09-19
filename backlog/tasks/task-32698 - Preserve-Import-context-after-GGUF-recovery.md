---
id: TASK-32698
title: Preserve Import context after GGUF recovery
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 05:53'
updated_date: '2026-09-16 06:14'
labels:
  - library
  - ui
  - focus
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provider recovery should preserve the next Import draft and the user’s keyboard context. Review actual picker cancellation and result delivery, explicit retry routing, and compact action visibility.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 GGUF picker cancellation, rejected selection and successful recovery retain the staged Import draft, selection and mounted controls while showing the correct provider status.
- [x] #2 Successful recovery targets only its failed job; newer focus or navigation survives late completion, and explicit faster-whisper retry retains its named-provider routing.
- [x] #3 Production-CSS journeys and isolated native evidence qualify the repair without real transcription, model installation or default-profile mutation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/041-direct-local-gguf-before-managed-acquisition.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: repair presentation continuity under existing provider/configuration and component contracts without changing admission, retry authority or persistence.
1. Reproduce the GGUF completion rebuild through the real mounted picker and worker, replacing only external model admission/persistence and retry execution in tests.
2. Reuse the existing in-place Import state synchronizer to update provider readiness and queue projection; retain current focus and navigation.
3. Cover cancel/reject/success, late completion and explicit faster-whisper actions at wide/compact sizes. Run targeted neighbors, static comparisons and independent review.
4. Verify actual TldwCli terminal journeys with a private profile and synthetic jobs, inspect captures, document evidence, close the Backlog task and commit locally.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Successful GGUF selection now updates the existing audio/video option group instead of rebuilding LibraryScreen. Draft editors, values and selection survive; provider status and chooser copy update in place, queue retries retain their existing listener, and newer focus or navigation wins over late worker completion. The production repair is one line in library_screen.py.

Added 16 production-CSS journeys covering real picker cancellation, rejection and success from both row and form entry points, delayed result delivery, and explicit faster-whisper routing with retry lineage. Updated the Import guide and recorded native runner, captures and receipts in Docs/superpowers/qa/2026-09-16-ingest-recovery/README.md. The baseline test reproduced replacement of the mounted canvas; 104 targeted checks now pass (16 new, 88 neighboring). Independent review found no actionable code issues.

Final isolated TldwCli/LinuxDriver run-004 passed ten journeys at 170x48 dark and 80x24 light and exited normally. The real picker, worker, app retry routing and registry execute; model admission/persistence and parse-pool dispatch are replaced. Ten private databases are healthy, media/messages/ingest jobs remain zero, fixture bytes and default-profile hashes are unchanged, and no ERROR/CRITICAL app log lines appeared. No actual transcription, model configuration, import or network operation ran.

Six final captures were inspected. A separate existing compact filtered-file-picker layout defect squeezes the filename field and hides Cancel; Escape and keyboard path submission still work. That shared layout is recorded for a separate repair and is not claimed fixed here. Earlier native attempts corrected stale fixture widget references after audio clearing, long Input visibility assumptions and transient Toast overlap; production remained unchanged.

Zero new Ruff diagnostics; new Python files formatted and clean, modified production line formatted, diff checks passed. The 205 inherited LibraryScreen Ruff diagnostics and its existing size-ceiling failure remain; line/function counts are unchanged, so the architecture ceiling was not rerun or claimed repaired. No budgets were raised and no full suite ran.

ADR required: no. Existing backlog/decisions/041-direct-local-gguf-before-managed-acquisition.md, 150-design-token-system-and-design-language.md and 161-component-pattern-library.md govern the repair and are linked from the plan and QA README. No admission, persistence, retry authority, execution or visual-token boundary changed. Plan followed; remaining shared picker layout work stays separate.
<!-- SECTION:NOTES:END -->
