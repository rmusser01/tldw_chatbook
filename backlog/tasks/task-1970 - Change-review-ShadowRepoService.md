---
id: TASK-1970
title: 'Change review: ShadowRepoService — hardened per-root shadow git'
status: Done
assignee: []
created_date: '2026-08-02 21:00'
labels:
  - workspaces
  - change-review
  - agents
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Foundation for Agent Change Review. A service owning one shadow git repo per CANONICAL root path (symlinks resolved), GIT_DIR under the app data dir, `core.worktree` pointing at the root — the user's own `.git` is never touched. Every invocation passes explicit `--git-dir`/`--work-tree` with `GIT_*` env scrubbed. Init must pin the four configs that break on real machines: app-local `user.name`/`user.email` (commit fails without identity), `commit.gpgsign=false` (a global gpgsign signs or prompts on every snapshot), empty `core.hooksPath` (global husky-style hooks must not fire), `gc.auto=0`. Provides snapshot-commit, numstat, per-file diff/show, and a per-root lock (in-process + flock) with index.lock retry.

Spec: `Docs/superpowers/specs/2026-08-02-agent-change-review-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A snapshot commit of a scratch root succeeds in a scratch HOME configured with NO git identity, `commit.gpgsign=true`, and a global `core.hooksPath` whose hook would fail the run — all three hazards proven by test
- [x] #2 The same root reached through a symlink and directly resolves to ONE shadow repo
- [x] #3 No file or directory is ever created inside the tracked root (asserted by tree comparison)
- [x] #4 Two concurrent snapshot calls on one root serialize instead of failing on index.lock
- [x] #5 `shutil.which('git')` absent -> the service reports unavailable; nothing raises
- [x] #6 Forced excludes (.git/, node_modules/, .venv/, __pycache__/, build dirs) live in info/exclude and are honored
- [x] #7 All porcelain/diff parsing is `-z` NUL-delimited: a filename containing spaces AND a newline round-trips through snapshot, diff, and revert
- [x] #8 The cross-process lock is portable (atomic mkdir lockdir, no flock) and passes on the Windows CI lane
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TDD in `Tests/Workspaces/test_change_tracking.py` against real git: hostile-HOME snapshot (no identity + gpgsign=true + failing global hooksPath), symlink/direct one-repo, nothing-in-root tree comparison, concurrent serialize, git-absent typed unavailability, forced excludes, hostile-filename (spaces+newline) roundtrip through snapshot/numstat/restore, stale-lockdir recovery, changed-file classification (A/M/D/R/binary).
2. `tldw_chatbook/Workspaces/change_tracking.py`: `ShadowRepoService` (data-dir via `Utils.paths.get_user_data_dir`, canonical-root keying by sha256(resolved path)) and `ShadowRepo` (ensure_initialized pins identity/gpgsign=false/empty hooksPath/gc.auto=0/core.untrackedCache=true + info/exclude; `snapshot()` = add -A + commit-if-dirty with --no-verify, --allow-empty first commit; `changed_files()` = -z -M numstat+name-status merge; `diff_text`/`file_bytes`; low-level `restore_paths` checkout primitive — full revert semantics stay in TASK-1974).
3. Locking: per-repo in-process threading.Lock + portable atomic-mkdir lockdir with backoff and stale-age takeover; no fcntl/flock anywhere.
4. Every git call: explicit --git-dir/--work-tree, GIT_* env scrubbed, GIT_TERMINAL_PROMPT=0, subprocess timeout.
5. Sabotage pass on the guards that pass first try.

AC#5 interpretation (recorded): "nothing raises" = constructing the service and probing `.available` never raise; USING an unavailable service raises the typed `ChangeTrackingUnavailableError` — a silent no-op snapshot would be this programme's canonical false-pass bug.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`tldw_chatbook/Workspaces/change_tracking.py` + `Tests/Workspaces/test_change_tracking.py` (13 tests, all against real git in tmp dirs — zero mocks).

`ShadowRepoService.repo_for_root` canonicalizes (expanduser+resolve, sha256-keyed) so every spelling of a root shares one repo; `ShadowRepo` pins the four hostile-machine configs at every `ensure_initialized` (self-healing), snapshots with `add -A` + commit-if-dirty (`--no-verify`, `--allow-empty` only for the first tip), parses ONLY `-z` NUL streams (name-status + numstat merged, rename- and binary-aware), and takes an in-process lock + portable mkdir lockdir with logged stale takeover. `restore_paths` is deliberately a low-level checkout primitive — un-create and guards are TASK-1974's.

AC#5 interpretation recorded in the plan: constructing/probing never raises; USING an unavailable service raises the typed error (silent no-op snapshots are this programme's canonical false-pass).

**The sabotage pass earned its keep twice.** My first harness used an unquoted heredoc that collapsed `\0` — the parsing sabotage silently never applied, and "mutation survived" was actually "mutation never ran" (the harness now asserts replacement counts). And the env-scrub test originally asserted on `GIT_DIR`, which our explicit `--git-dir` flag already defeats — the LIVE hazard is `GIT_INDEX_FILE`, which flags do NOT override; the test now proves the index lands in the shadow repo, not the leaked path. Three sabotages (config pins removed / non-z parsing / scrub removed) each fail exactly one test.
<!-- SECTION:NOTES:END -->
