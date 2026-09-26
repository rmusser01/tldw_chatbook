---
id: TASK-32904
title: Eliminate Console UI stall root causes (random lag investigation)
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-23 15:07'
updated_date: '2026-09-23 16:11'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fedora user reported random multi-second lag/unresponsiveness on latest dev. Investigation (2026-09-23) verified six root causes: (1) per-send context-window metadata probe blocking sends up to 1s with a 5s failure TTL and a guaranteed oversized-payload retry loop for OpenRouter; (2) synchronous SQLite write on the UI event loop from the 0.2s Console tick plus unguarded sync DB/preflight work in _run_agent_reply; (3) Linux SecretService/D-Bus keyring lookups on the UI thread (image/video gen settings panels, first skill-trust build, canvas policy, personal-context link); (4) quadratic per-delta thinking-envelope canonical JSON round-trips during reasoning streams; (5) redundant O(total-history) walks in the 0.2s Console tick; (6) readiness revision bumps every credential-TTL expiry triggering full summary rebuilds, and sync readiness work on the debounced keystroke estimate path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console sends never block on the context-window metadata probe; stale/cached windows are used and refreshed in the background
- [x] #2 Metadata probe does not re-download oversized payloads on a short TTL; usable-but-windowless responses are cached with the success TTL
- [x] #3 No synchronous SQLite write or project-instruction preflight runs on the UI event loop during agent sends or the 0.2s tick
- [x] #4 keyring lookups no longer run on the UI thread in image/video gen settings, skill trust first build, canvas policy resolution, or personal-context link
- [x] #5 Per-thinking-delta work no longer serializes the full accumulated envelope; serialization happens at structural boundaries and terminal persist only
- [x] #6 The 0.2s Console tick no longer re-tokenizes the growing streamed reply every tick (settings estimate cached with a 1s TTL; cost chip keeps its 10s TTL), bounding per-tick history walks to the transcript sync
- [x] #7 Subscription readiness revision only changes when the observed readiness value actually changes; keystroke estimate path avoids synchronous readiness resolution
- [x] #8 Targeted tests cover each fix; existing related suites pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Fix 1: probe off send path (cached window + background refresh task); TTL semantics split usable-response vs transport-failure; content-length pre-check. 2. Fix 2: fold-without-persist in read paths, deferred off-loop persistence for pending stream rows; to_thread wraps in _run_agent_reply. 3. Fix 3: move keyring-backed config reloads to thread workers (image/video gen panels + save/revert/test paths); async skill-trust first build; to_thread canvas policy + personal-context key custody. 4. Fix 4: skip full-envelope canonical JSON validation on text-append deltas; serialize at structural boundaries and terminal persist. 5. Fix 5: single messages_for_session materialization per tick (revision-keyed snapshot reuse). 6. Fix 6: value-gate subscription readiness revision; cached_context_window avoids sync readiness on warm cache. 7. Targeted tests per fix; ruff on touched files; update task notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All six verified root causes fixed with targeted regression tests; every touched suite verified at parity-or-better against a HEAD baseline worktree. The 8 remaining failures in test_console_session_settings/test_console_spend_projection were proven NOT task-32904's (they persist with all three behavioral changes neutralized) -- they belong to the uncommitted task-32708 WIP in console_session_settings.py. Lesson added to backlog/docs/lessons-textual.md (swallowed exit_on_error=False worker errors present as missing widgets).
<!-- SECTION:NOTES:END -->
