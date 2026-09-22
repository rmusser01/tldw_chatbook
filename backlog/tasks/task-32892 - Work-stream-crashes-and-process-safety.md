---
id: TASK-32892
title: "Work stream: crashes and process safety"
status: To Do
assignee: []
created_date: '2026-09-21 23:05'
labels:
  - tier2-review
  - review-crashes
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seven paths where the app dies, hangs, or kills itself. One is a P0: `SimpleAudioPlayer.stop()` sends
`SIGKILL` to `os.getpgid(...)` of a child spawned **without** `start_new_session`, so the child shares the
app's own process group and the app kills itself. The rest are Textual crashes -- a modal that cannot
compose, a non-ASCII chat topic that kills the Stats screen at mount, workers with the default
`exit_on_error=True`, and a DOM lookup in a `finally:` body.

All small diffs; all "the app dies". One PR.

Source: tier-2 code review 2026-09-21 -- `qa/tier2-code-review-2026-09-21/report.md` (26 slices, 890,356 lines: the surface tier 1 never reached). Per-slice evidence in `qa/tier2-code-review-2026-09-21/slices/`, reproductions in `phase4-verification.md`, and per-finding re-validation against `origin/dev d0face3ebe` in `validation/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The three P0s are closed and each has a test that is born red
- [ ] #2 No `Popen` in TTS spawns into the app's own process group
- [ ] #3 Every crash site in the children has a pinning test that fails before the fix
- [ ] #4 `./scripts/preflight.sh` is green and no size ratchet moves
<!-- AC:END -->
