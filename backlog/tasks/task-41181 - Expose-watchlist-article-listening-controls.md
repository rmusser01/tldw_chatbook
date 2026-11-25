---
id: TASK-41181
title: Expose watchlist article listening controls
status: Done
assignee:
  - '@codex'
created_date: '2026-11-15'
updated_date: '2026-11-25T04:54:00Z'
labels:
  - watchlists
  - tts
dependencies:
  - TASK-41180
references:
  - https://github.com/rmusser01/tldw_chatbook/issues/2864
documentation:
  - backlog/decisions/367-watchlist-article-listening-and-controlled-playback.md
  - Docs/superpowers/specs/2026-11-15-watchlist-article-listening-design.md
  - Docs/superpowers/plans/2026-11-15-watchlist-article-listening.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make saved-article and cited-queue playback discoverable and usable without losing the current reading or briefing context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reader Listen, selected-citation Listen and Play cited excerpts invoke listening without marking items read or leaving the briefing.
- [x] #2 Compact transport and global palette expose truthful preparation, timing, suspension, recovery and Pause, Resume, Stop, Skip, Clear and Play again actions.
- [x] #3 Unsupported bodies, server scope, missing capabilities and serve-mode host output have clear explanations before relevant actions.
- [x] #4 Mounted controls remain reachable and retain focus at 80x24 and wide sizes using existing tokens, patterns and keybinding rules.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/367-watchlist-article-listening-and-controlled-playback.md
Reason: project the existing accepted listening service into its approved reading, briefing, compact transport and palette surfaces. Reuse ADR-367 with ADR-150 tokens, ADR-161 component patterns and ADR-299 keybindings; no new storage/provider boundary or shell-wide structure.
1. Verify TASK-41180 reviewed public snapshot, opaque targets and brief action relation against the source map in the read-only UI preflight. Read design/component/keybinding guidance and product context before edits.
2. Add mounted failing tests for reader/citation Listen, full cited-count preview, Resume/Play again/replacement, truthful failure/host reasons, current scope and no read-state or view change. Cover reachable hit targets, focus and input at 80x24 and wide; observe RED.
3. Implement thin typed entry intents, one Watchlists-wide compact transport beside the workbench and global live palette commands. Use the app-owned late-bound service and captured opaque targets, preserve selection/scroll/focus and detach only UI observers on unmount. Reuse bounded output readiness and existing confirmations.
4. Edit source CSS using existing tokens/patterns, rebuild generated bundles, and run targeted mounted/action/palette, adjacent Watchlists, CSS/token/architecture and changed-Python static checks. Freeze source/tests during every guarded/native job; no full sweep.
5. Verify real native terminal and loopback serve geometry, focus, progress and pre-action host-output notice on the qualified Windows route. Use normal admitted profile/credential paths and checked synthetic fixtures; distinguish mounted, provider, process and acoustic evidence. Root coordinates exclusive native/provider/UI slot.
6. Complete independent spec/quality review, resolve blocking findings with regressions, update user/architecture docs and task evidence, then publish a separate draft child of PR3768. Preserve the ongoing plan workspace and reviewed draft parents.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Reader and citation Listen, cited-queue guidance/replacement, compact transport and global palette through the app-owned service. Opaque targets and publication tickets fence stale actions and off-loop eligibility. Ordinary Reader Open and read marking are preserved. Reused ADR-367 with ADR-150/161/299; updated user/architecture and qualification docs. Independent review and three scoped fix rounds approved source e74018a6e5 with no open Important findings. Mounted/token/CSS/palette batch: 105 passes; affected round-one 27 distinct nodes passed across recorded runs, round-two 11 passes, final round-three eight passes. Targeted format/syntax checks passed with the unchanged legacy lint baseline; no full sweep. Actual Windows 80x24 terminal controls and 151x43 authenticated served UI proved paused/manual restoration, host notice, progress, queue advance, citation/Reader Listen, suspended queue and active global Stop. Native receipts retain their exact source attribution and synthetic-tone limit; final same-row behavior has a discriminating mounted regression. All fixture app/player/server jobs and guard roots retired. The final end-to-end delivery owns provider-through-UI, recovery and acoustic qualification; issue #2864 remains open. The older pending-body/page-replacement interleaving is documented separately as mixed adjacent evidence.
<!-- SECTION:NOTES:END -->
