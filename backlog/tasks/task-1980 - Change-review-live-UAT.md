---
id: TASK-1980
title: 'Change review: live end-to-end verification (real app, real agent run)'
status: Done
assignee: []
created_date: '2026-08-02 21:00'
labels:
  - change-review
  - verification
dependencies:
  - TASK-1972
  - TASK-1973
  - TASK-1974
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The programme's standing lesson: green tests are not a usable feature. Drive the REAL app in tmux with an isolated TLDW_CONFIG_PATH profile: register a scratch root, run a real agent turn that creates+edits+deletes files (including one via a script side effect), read the summary row, open the Review screen, read each diff, revert one file, Undo-all a turn — at 80×24 and 212×64. File defects found as tasks before closing.

Spec: `Docs/superpowers/specs/2026-08-02-agent-change-review-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The full journey above completes on the live app at both sizes, evidenced by captured panes
- [x] #2 The script-side-effect file appears in the review (with its TASK-1978 badge if merged)
- [x] #3 Reverted files verified on DISK, not just in the UI
- [x] #4 Any defect found is filed as a backlog task and linked here
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Isolated profile (TLDW_CONFIG_PATH scratch config, users_name verify_1980) + seeded scratch root
2. Register workspace + writable folder binding through the Settings UI; agent turn via a deterministic local OpenAI-compatible stub on the Custom provider (config's Anthropic key is dead — 401 straight from the API)
3. Drive the full journey at 212×64 and 80×24; verify every revert on disk; file defects
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Full journey completed on the LIVE app, both sizes, panes captured throughout.

**What verified clean:** ✎ summary row appears with correct counts for agent
writes (+2 −1) AND for changes made outside file tools (+1 −1 from a mid-turn
external delete+create injected while the run was paused on approval — AC #2's
substance; no badge, TASK-1978 unmerged). Failed runs (provider 401) produce no
phantom rows; pre/post-turn external changes correctly excluded from B..E.
Review screen: turn selector with totals, A/M/D grouping, correct unified
diffs (create/modify/delete), j/k navigation, 80×24 layout with wrapped diff
and key footer. Reverts verified ON DISK three times: per-file un-create
(summary.md removed), Undo-all at 212×64 (notes.md restored to baseline),
Undo-all at 80×24 (sidecar.log restored byte-identical from B, script_output.txt
un-created). Approval card keys duplicate tools as write_file ×2.

**Defects filed (AC #4):** TASK-2030 (HIGH — the ✎ row's own `v`/Review action
always toasts "target no longer exists": recompute-synthesized marker rows are
not store messages; the inspector route works and is how the UAT proceeded),
TASK-2031 (chips stale after session-model Apply until session switch),
TASK-2032 (Review tree click doesn't load the diff; only j/k), TASK-2033
(Console boots on Default despite registry-active workspace — owner decision).

**Observations not filed:** workspace-create name input has no Enter-submit
(button only); turn selector lists a pickable "No turns" placeholder row;
Console at 80×24 shows only the context rail (pre-existing layout, not
change-review scope). The tool_sandbox is where RELATIVE write_file paths
land — invisible to change review by design (sandbox excluded from roots);
agents must use absolute paths under a bound root for tracked writes.

**Method note:** agent turns used a deterministic local OpenAI-compatible stub
(scripted write_file×2 → final text) behind the Custom provider — the config's
real Anthropic key is dead (401 from the API directly), and a scripted model
makes the tool sequence reproducible. Everything from the gateway inward
(tool dispatch, approvals, gate, tracking, DB, UI) was the real app.
<!-- SECTION:NOTES:END -->
