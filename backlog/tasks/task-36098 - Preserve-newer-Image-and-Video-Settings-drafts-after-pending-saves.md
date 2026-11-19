---
id: TASK-36098
title: Preserve newer Image and Video Settings drafts after pending saves
status: Done
created_date: 2026-10-22 01:24
labels:
- settings
- textual-patterns
- drafts
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-21-textual-save-and-source-freshness.md
assignee:
- '@codex'
updated_date: 2026-10-22 05:52
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Saving Image Gen or Video Gen settings must not erase edits typed while the earlier Save is pending. Save completion must keep those later values visibly unsaved and preserve the active editing context.

### Evidence and scope

Medium priority, audit order 2 of 4. Relevant source blobs on fetched merged dev `2a5afa03b31f60e00bdcd10a161e3e351fad023b` are identical to diagnostic candidate `8ff5f0e2d4f3016445545e84f9e30a5233b8fddd`. For both Image and Video, mounted Settings keyboard witnesses held the real checked configuration write. The submitted `original-submitted` value reached TOML, but completion replaced the live `original-submitted-new` Input with a new Input containing the older saved value. Focus moved to the detail body and the category became clean. One normal model field per sibling is demonstrated; no generation, credentials or external HTTP was involved.

Source: [tldw_chatbook/UI/Screens/settings_screen.py:9247](https://github.com/rmusser01/tldw_chatbook/blob/2a5afa03b31f60e00bdcd10a161e3e351fad023b/tldw_chatbook/UI/Screens/settings_screen.py#L9247); [tldw_chatbook/UI/Screens/settings_screen.py:9937](https://github.com/rmusser01/tldw_chatbook/blob/2a5afa03b31f60e00bdcd10a161e3e351fad023b/tldw_chatbook/UI/Screens/settings_screen.py#L9937); [tldw_chatbook/Widgets/settings_image_gen_panel.py:186](https://github.com/rmusser01/tldw_chatbook/blob/2a5afa03b31f60e00bdcd10a161e3e351fad023b/tldw_chatbook/Widgets/settings_image_gen_panel.py#L186); [tldw_chatbook/Widgets/settings_video_gen_panel.py:148](https://github.com/rmusser01/tldw_chatbook/blob/2a5afa03b31f60e00bdcd10a161e3e351fad023b/tldw_chatbook/Widgets/settings_video_gen_panel.py#L148).

Book basis: Chapter 9 printed 170-171 / PDF 179-180 explains child recreation and Input-state reset under recomposition; printed 180 / PDF 189 covers focus loss. Chapter 7 printed 144-145 / PDF 153-154 distinguishes programmatic refresh events from edits.

These are audit findings, not implemented fixes. The report distinguishes demonstrated cases from future regression controls.

### Existing ownership and decisions

TASK-32926 Done owns off-thread media configuration reads; TASK-23191 Done suppresses Video mount/Revert echoes. TASK-33002.10/.11 Done concern draft help/counts. TASK-35680 and TASK-35867 Done are specifically Console Behavior event-origin work. TASK-32190 Done already provides submitted-only acknowledgement for Advanced Config/raw TOML, a useful precedent outside these panels. TASK-32810.3 Done is behavior-preserving inventory and explicitly declines a generic staging consolidation; TASK-3401.12 Done supplied Video settings. Same-file open PR changes were inspected and do not fix this outcome.

Related tasks: [TASK-32926](task-32926%20-%20Keyring-reads-on-a-cache-miss-still-run-on-the-UI-thread-in-Settings-and-Library.md); [TASK-23191](task-23191%20-%20Video-Generation-settings-show-State-Unsaved-changes-on-a-fresh-profile-with-no-edits.md); [TASK-33002.10](task-33002.10%20-%20Settings-says-in-the-app-that-switching-categories-keeps-a-draft.md); [TASK-33002.11](task-33002.11%20-%20Image-Gen-and-Video-Gen-unsaved-count-counts-fields-not-draft-keys.md); [TASK-35680](task-35680%20-%20Keep-parallel-run-form-refreshes-out-of-Settings-drafts.md); [TASK-35867](task-35867%20-%20Keep-remaining-Console-input-refreshes-out-of-user-drafts.md); [TASK-32190](task-32190%20-%20Preserve-Advanced-Config-drafts-and-guard-draft-replacement.md); [TASK-32810.3](task-32810.3%20-%20P3-consistency-bundle-Main-screens.md); [TASK-3401.12](task-3401.12%20-%20Video-generation-settings-page-templates-diagnostics-docs.md).

ADR required: no new ADR for this bounded repair.
ADR paths: [033-settings-commit-models-three-honestly-labeled](../decisions/033-settings-commit-models-three-honestly-labeled.md); [044-ephemeral-generated-video-storage-playback-and-streaming](../decisions/044-ephemeral-generated-video-storage-playback-and-streaming.md); [150-design-token-system-and-design-language](../decisions/150-design-token-system-and-design-language.md); [161-component-pattern-library](../decisions/161-component-pattern-library.md).
Reason: No new ADR is needed to honor the accepted staged Settings commit model within these existing panels. The Advanced Config amendment is a scoped precedent, not an unimplemented global programme. Preserve existing media/provider boundaries; ADR-044 is Accepted with decision 2 superseded by [ADR-176](../decisions/176-one-media-generation-core.md) (shared Media_Generation core); it is secondary media background, not a new settings contract. ADR-150/161 apply to incidental composition changes; no new settings subsystem or persistence contract is proposed.

[Post-merge book audit and evidence](../../Docs/superpowers/qa/2026-10-21-textual-book-after-pr3199.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A successful Image Gen Save persists its submitted settings while retaining later edits as visible unsaved values.
- [x] #2 A successful Video Gen Save retains later edits through its post-save configuration load while persisting only the submitted snapshot.
- [x] #3 Post-save refresh preserves the user's current editing context, including active Input focus, caret and selection, without stealing focus after the user moves elsewhere. Another explicit Save persists the remaining draft; no-edit success becomes clean and deliberate Revert restores persisted values.
- [x] #4 Mounted regressions with a controlled pending configuration write cover both panels, failed saves and saves without later edits, without staging effective defaults, borrowed credentials or incidental mount events as user changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/033-settings-commit-models-three-honestly-labeled.md; backlog/decisions/044-ephemeral-generated-video-storage-playback-and-streaming.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: Honor existing staged Settings acknowledgement in the two existing panels; ADR044 decision2 is superseded by ADR176, with no new media or persistence contract.
1. Read task/ADRs/design rules and checked-write audit witnesses; trace submitted category drafts, post-save loads/recomposition and current field/focus ownership.
2. Add Tests/UI/test_settings_media_pending_save_drafts.py with bounded mounted held-real-write Image/Video regressions. Verify later raw text/dirtiness, actual TOML snapshot, caret/selection/focus and moved-focus control, second Save, unchanged success, Revert and failed save. Demonstrate meaningful reds before changing product code.
3. Retain later drafts through existing successful completion/config refresh without staging defaults or borrowed credentials; preserve current editing context and avoid restoring obsolete focus. Use existing panel ownership rather than a new settings framework.
4. Run targeted new/adjacent media-settings regressions, scoped lint/format and relevant UI governance; record evidence/notes. Root handles shared census and independent Superpowers review before Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Image/Video Settings save completion acknowledges only the submitted draft, preserves later raw fields through config reload and retains current focus/selection, backend expansion and scroll. Explicit Clear/retype history survives incremental typing and unrelated equal-valued global events; initial mount/default values and borrowed credentials remain unstaged. Checked writes and screen-owned draft/runtime authority are unchanged. Narrow media_gen_save helper extraction preserves the existing screen ceiling. Original ten-case run passed; meaningful red review controls exposed expansion and Clear/retype defects; final-source five-case unchanged/keyboard-clear/history selection passes (144.37s), with earlier later-write/reload and nondefault sibling passes qualified. Final 92 targeted integration tests pass. Helpers/new test lint/format pass; 13 inherited screen diagnostics and existing formatter debt are unchanged. Media adjacent selection 124 pass/9 pre-existing native fixture/platform failures are source-qualified. Independent Superpowers review evaluated all corrections, replacing Qodo. Existing ADR-033 Settings, ADR-150/161 apply; no new ADR or framework. Modified settings_screen.py, focused media_gen_save.py and fifteen-case regression module. Evidence and limits: Docs/superpowers/qa/2026-10-21-textual-save-and-source-freshness.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
