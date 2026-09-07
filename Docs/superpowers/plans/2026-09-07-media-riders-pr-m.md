# Media riders PR M — failure surfaces and the rail (tasks 31943, 31948, 31944, 31949, 31960)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. The five backlog task files are the briefs; this plan carries the batching, constraints and verification.

**Goal:** close the wave-5 riders on the Media list's failure and receipt surfaces: the rail's `Media N` count stays stale after a bulk-delete Undo (31943); the browse-row error callout has no adjacent Retry (31948); a retry failure shows a bare exception class name (31944); the source-snapshot timeout branch leaves no log record (31949); "Review these" is the only list-wide action outside the failed-load gate — decide and record (31960).

**Spec:** the task files under `backlog/tasks/` (31943, 31948, 31944, 31949, 31960).

## Global Constraints
- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/media-riders-m`, branch `fix/media-riders-m` off dev. Every command `cd <worktree> && git branch --show-current`; python `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` with `PYTHONPATH=$PWD`; `-p no:cacheprovider`; UI test files one per process, SERIAL.
- Surfaces already in place: PR E's `_complete_library_media_mutation(focus_identity=…)` and bulk-delete receipt/Undo; PR G's `DestinationRecoveryState` callout (`load_failure_recovery_state`, `Horizontal#library-media-load-failure.ds-recovery-callout` with `Button#library-media-retry`), `_retry_failure_reason` with path redaction (privacy pin: non-OS exception text reduces to the class name — 31944's map must keep redaction and never leak exception text), the ONE shared logger call `_log_source_snapshot_failure` for the source snapshot (31949: reuse it with a timeout marker — no new `logger.*`; the diagnostic inventory must stay green); PR J's `_gate_failed_action` (Export only) and `_gate_stale_action` (Review these) on the Media canvas (31960).
- Painted pins for anything a user sees (the rail count, the callout + Retry, the reason copy); never region-only; no new toolbar buttons (a Retry inside the existing error callout is not a toolbar button); Find focus token untouched; do NOT edit `backlog/`.
- Evidence: failing NAME sets vs a detached copy of the merge-base; known dev reds per task-31249's census.
- Live: ONE app instance, tmux socket `wrm`; scratch profile via `TLDW_CONFIG_PATH` with `media_db_path` = a directory provokes a hard load failure without touching real data (PR G's recipe).
- Commit per batch with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Batches
### Batch 1: 31943 + 31948 (rail count after Undo; browse-row error callout Retry)
### Batch 2: 31944 + 31949 + 31960 (reason map; timeout log record; Review-these gate decision — document the exception where the gate is defined unless symmetry is one line)

## Land
Final whole-branch review → fix round → PR M.
