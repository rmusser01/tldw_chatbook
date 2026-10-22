---
id: TASK-36100
title: Reject audiobook chapter results after their source text changes
status: To Do
created_date: 2026-10-22 01:25
labels:
- speech
- textual-patterns
- reliability
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-21-textual-book-after-pr3199.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
An audiobook chapter detector must not repopulate the current editor with results for source text the user has cleared or replaced. Current text must remain authoritative even before a replacement detection is dispatched.

### Evidence and scope

Medium priority, audit order 4 of 4. Relevant source blobs on fetched merged dev `2a5afa03b31f60e00bdcd10a161e3e351fad023b` are identical to diagnostic candidate `8ff5f0e2d4f3016445545e84f9e30a5233b8fddd`. A mounted AudioBookGenerationWidget used the actual ChapterDetector behind a threading.Event after it computed the old chapters. Actual F7/Backspace keys cleared the source, and both the TextArea and content_text were empty before release. Completion still displayed the old chapter in the editor and table; no replacement detection was scheduled for the empty source. The unchanged-source control passed. Wrong generated audio, provider spend or persisted-data harm was not exercised or claimed.

Source: [tldw_chatbook/UI/STTS_Window.py:872](https://github.com/rmusser01/tldw_chatbook/blob/2a5afa03b31f60e00bdcd10a161e3e351fad023b/tldw_chatbook/UI/STTS_Window.py#L872); [tldw_chatbook/UI/STTS_Window.py:999](https://github.com/rmusser01/tldw_chatbook/blob/2a5afa03b31f60e00bdcd10a161e3e351fad023b/tldw_chatbook/UI/STTS_Window.py#L999); [tldw_chatbook/UI/STTS_Window.py:1038](https://github.com/rmusser01/tldw_chatbook/blob/2a5afa03b31f60e00bdcd10a161e3e351fad023b/tldw_chatbook/UI/STTS_Window.py#L1038).

Book basis: Chapter 7 printed 149 / PDF 158 and chapter 17 printed 326-327 / PDF 335-336 explain off-loop work, cancellation checks and marshalled result publication. Current-source freshness is derived from the observed view-state bug, not a new cancellation or provider contract.

These are audit findings, not implemented fixes. The report distinguishes demonstrated cases from future regression controls.

### Existing ownership and decisions

TASK-15478 Done fences slow detector dispatch A against newer dispatch B; it does not invalidate authority on current-source edits or clear before another dispatch. TASK-15773/16849/16850 Done own chapter widget remount, table update and heading/body correctness. TASK-19576 Done owns import reachability, TASK-32800.4 Done owns the static worker/DOM contract, and TASK-2011 Done owns the analogous cleared Library preflight result. None covers this audiobook residual.

Related tasks: [TASK-15478](task-15478%20-%20STTS-audiobook-paste-box-queries-a-switch-that-is-never-composed.md); [TASK-15773](task-15773%20-%20ChapterEditorWidget-Select-mount-race-under-high-volume-DataTable-population.md); [TASK-16849](task-16849%20-%20Chapter-editor-in-place-edits-never-refresh-the-now-truthful-chapter-table.md); [TASK-16850](task-16850%20-%20ChapterDetector-emits-an-empty-placeholder-chapter-per-titled-heading-doubling-the-rows.md); [TASK-19576](task-19576%20-%20Two%20reachable%20user%20paths%20are%20broken%20by%20retired-surface%20drift%20-%20STTS%20AudioBook%20import%20crashes%20and%20Console%20file%20attach.md); [TASK-32800.4](task-32800.4%20-%20Add-a-guard-for-the-Textual-worker-and-post-await-DOM-contract.md); [TASK-2011](task-2011%20-%20Library-ingest-stale-preflight-result-resurrects-cleared-summary.md).

ADR required: no new ADR for this bounded repair.
ADR paths: [033-application-session-state-ownership](../decisions/033-application-session-state-ownership.md); [011-chatbook-workbench-ui-system](../decisions/011-chatbook-workbench-ui-system.md).
Reason: No new ADR is required for correcting source/chapter view-state freshness within its existing widget. Accepted ADR-033/011 apply. Provider synthesis, persistent state and global worker lifetime changes would exceed this bounded repair.

[Post-merge book audit and evidence](../../Docs/superpowers/qa/2026-10-21-textual-book-after-pr3199.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Clearing the source while detection is active prevents its eventual completion from repopulating the chapter editor or announcing chapter results for the cleared text.
- [ ] #2 Replacing source text during pending detection cannot publish chapters for the superseded text, including the interval before the replacement debounce dispatch.
- [ ] #3 Unchanged-source completion and a later valid source still populate the real chapter editor normally, while existing slow-dispatch/newer-dispatch ordering stays protected.
- [ ] #4 Bounded mounted regressions use actual TextArea change events and controlled detection for clear/replacement and ordinary completion, with no provider, synthesis or network effects.
<!-- AC:END -->
