---
id: TASK-1343
title: 'Local agent tools phase 3b-ii: git tools (read-only port)'
status: Done
assignee: []
created_date: '2026-08-05 20:51'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md §2.5. Plan: Docs/superpowers/plans/2026-08-05-local-agent-tools-phase3b-ii.md. ADR-033 (process boundary). Port of tldw_server git_module.py @ 5605b9d9, async-to-sync.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 git subprocess wrapper enforces fixed-argv allowlist (subcommands + -C/--no-pager only), sanitized env, timeout, 1 MB output cap
- [x] #2 git_status/git_diff/git_log/git_blame/git_branches work against workspace-confined repos
- [x] #3 Non-repo paths, git-unavailable, and disallowed invocations return model-actionable errors (no raw exceptions)
- [x] #4 Injection attempts (flag smuggling via args, path escapes) are refused
- [x] #5 All new tests pass
<!-- AC:END -->


## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-08-05-local-agent-tools-phase3b-ii.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

Implemented on branch `feat/local-agent-tools-p2` (stacked on PRs #1352/#1358) via subagent-driven development with per-task spec + quality review.

- `Tools/git_tool_impls.py` (new, attribution header per re-plan §5): sync `run_git` — fixed-argv validation ported near-verbatim from the reference (8-subcommand allowlist, `-C`/`--no-pager` globals only), fully sanitized env (PATH + git safety vars; no HOME/GIT_DIR/credentials), stdin=DEVNULL, Popen with reader-thread bounded reads that kill the whole process group at the 1 MB cap or 30 s timeout. `prepare_repository`: git-availability check, workspace confinement, `rev-parse --show-toplevel` discovery, repo-root-must-be-inside-workspace rule (above-workspace repos refused). Five read-only cores ported from the reference's `_execute_*` with sync adaptation: `git_status` (porcelain v2), `git_branches`, `git_log` (count clamp 1..100, default 20), `git_diff` (`--no-ext-diff/--no-textconv/--no-color`; staged/commit_range/stat modes), `git_blame` (line-porcelain, optional -L range).
- `Agents/local_tool_provider.py`: five `git_*` specs with `tags == ()` — ADR-033's binding decision NOT to apply the `process` tag to the read-only allowlisted set, pinned by `test_git_specs_carry_no_risk_tags` (the tripwire: if a mutating subcommand is ever added, tags must change AND the ADR must be revisited).

Documented deviations from the reference (module header): (a) `commit_range`/`stat` diff modes added (regex-validated, leading-dash refused); (b) log default count=20; (c) blame header parse accepts ≥3-field lines — fixes a REAL latent bug in the reference (git's line-porcelain emits 3-field headers for non-group-leading lines; the reference silently drops them); (d) timeout raises LocalToolError instead of returning timed_out=True.

Review-driven hardening beyond the plan:
- **Critical:** closed a verified command-execution escape — `commit_range="--textconv"` (and any dash-leading value) smuggled a flag into argv's last position, re-enabling textconv drivers via last-occurrence-wins and executing repo-config shell commands. Dash-leading values now refused; the hostile-repo exploit is a regression test.
- Truncated (>1 MB) results are now delivered with the truncation marker instead of becoming a bogus error (SIGKILL-by-us is not a git failure).
- Kills are process-group-wide (`start_new_session` + `killpg`, POSIX with fallback) — grandchildren can't survive or stall the truncation fast-path.
- gpgsign disabled in git test fixtures (cross-machine safety).

Tests: 40+ new — allowlist/injection/env/cap/timeout/process-group tests against real tmp repos, core behavior tests per tool, provider schema/tag/smoke tests, find/load e2e for git_log. Final suite: 420 passed (Tests/Tools + Tests/Agents).

Final whole-phase review: Ready to merge; all 5 ACs verified; ADR-033 boundary integrity confirmed.
