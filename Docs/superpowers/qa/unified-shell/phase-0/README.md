# Phase 0 QA Evidence

Phase 0 validates the tracking system, not product UI behavior.

## Current Summary

- Date: 2026-05-03
- Branch: `codex/unified-shell-maturity-tracking`
- Evidence commit before closeout: `a050e818`
- Status: verified
- Scope: Backlog initialization, task seeding, roadmap creation, Backlog docs pointer, QA evidence structure, and current-state reconciliation.

## Verification Log

- `backlog init --defaults --agent-instructions agents --integration-mode cli --backlog-dir backlog`
  - Result: emitted an interactive project-name prompt and made no file changes.
  - Deviation: reran with explicit project name as `backlog init tldw_chatbook --defaults --agent-instructions agents --integration-mode cli --backlog-dir backlog`.
  - Result: initialized `backlog/` with folder config and CLI integration.
- `backlog config set remoteOperations false`
  - Result: disabled Backlog remote task fetches because the sandbox prevented Backlog's internal `git fetch origin --prune --quiet` from writing `FETCH_HEAD`.
- `backlog config set checkActiveBranches false`
  - Result: disabled active-branch checks for clean local Backlog smoke output in this sandbox.
- `backlog task list --plain`
  - Result before closeout: 11 tasks total; `TASK-1`, `TASK-1.1`, and `TASK-1.2` in progress; `TASK-2`, `TASK-2.1`, `TASK-2.2`, `TASK-3`, `TASK-4`, `TASK-5`, `TASK-6`, and `TASK-7` to do.
  - Result after closeout: `TASK-1`, `TASK-1.1`, and `TASK-1.2` done; Phase 1 through Phase 6 tasks remain to do.
- `backlog overview`
  - Result before closeout: 8 To Do tasks, 3 In Progress tasks, 0 Done tasks; `TASK-2.1` and `TASK-2.2` blocked by dependencies as expected.
- `find backlog -maxdepth 2 -type f`
  - Result: `backlog/config.yml`, 11 task files, and `backlog/docs/unified-shell-maturity-roadmap.md` exist.
- `find Docs/superpowers/qa/unified-shell -maxdepth 2 -type f`
  - Result: QA root README plus phase 0 through phase 6 README files exist.
- `grep -R "<phase-[0-9]" Docs/superpowers/trackers/unified-shell-maturity-roadmap.md`
  - Result: no unresolved phase ID placeholders.
- `git diff --check`
  - Result: passed with no whitespace errors.
- `backlog task view TASK-1 --plain`, `backlog task view TASK-1.1 --plain`, `backlog task view TASK-1.2 --plain`
  - Result after closeout: all Phase 0 acceptance criteria are checked, implementation plans are present, implementation notes are present, and status is `Done`.

## Backlog Task IDs

- Phase 0: `TASK-1`
- Phase 0.1: `TASK-1.1`
- Phase 0.2: `TASK-1.2`
- Phase 1: `TASK-2`
- Phase 1.1: `TASK-2.1`
- Phase 1.2: `TASK-2.2`
- Phase 2: `TASK-4`
- Phase 3: `TASK-3`
- Phase 4: `TASK-5`
- Phase 5: `TASK-6`
- Phase 6: `TASK-7`

## Reconciliation Summary

- Current merged shell foundation is captured in `Docs/Design/master-shell-route-inventory.md`, `Docs/Design/master-shell-design-system-contract.md`, and `Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md`.
- Known tracking gaps from the Phase 0 starting point are recorded in the roadmap, including stale local-only plans, the prior lack of Backlog initialization, and the lack of centralized QA evidence.
- Known product gaps are recorded in the roadmap: Home controls still need real adapters, Workflows service wiring is absent, ACP runtime launch is disabled, MCP management is not embedded in the top-level wrapper, Skills import/list/detail UX adoption is incomplete, and Library still relies on legacy sub-surfaces.
- Stale local-only plans are not treated as source of truth. The canonical roadmap is `Docs/superpowers/trackers/unified-shell-maturity-roadmap.md`.

## Product QA Boundary

No Textual UI code changes are made in Phase 0. Product workflows remain unverified until Phase 1 and later running-app walkthroughs.

## Residual Risk

- Phase 0 verifies tracking infrastructure only.
- Backlog remote branch/task scanning is disabled in repo config for deterministic local CLI behavior in this sandbox; future maintainers can re-enable it outside the sandbox if cross-branch task state is needed.
- Home, Console, destination wrappers, and live service flows remain unverified by this phase.
