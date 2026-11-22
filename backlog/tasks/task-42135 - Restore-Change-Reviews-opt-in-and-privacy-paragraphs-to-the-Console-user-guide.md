---
id: TASK-42135
title: "Restore Change Review's opt-in and privacy paragraphs to the Console user guide"
status: Done
assignee:
- '@claude'
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 18:21
labels:
- follow-up
- docs
- change-review
priority: medium
references:
- Docs/User_Guide/console/agent-runs-and-tools.md
- tldw_chatbook/Workspaces/change_review_consent.py
- tldw_chatbook/Workspaces/change_retention.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the docs stream (round 18). Squash 6a6242ce3a (2026-09-12) removed the opening paragraphs of "Change review — reviewing a turn's file changes" in `Docs/User_Guide/console/agent-runs-and-tools.md` (the heading is now line 1046); 2e5d256a04 had added them. `git show 6a6242ce3a^:Docs/User_Guide/console/agent-runs-and-tools.md` has the text at lines 685-703. It said:
- Change Review is off for every new workspace until the user enables it in **Settings > Workspaces**; the global `[change_review] enabled` setting is only a capability switch;
- enabling prepares each bound folder in a bounded background queue (`preparing`, `ready` or `failed`, with a retry), and a turn that starts before a folder is ready continues with a warning;
- Change Review keeps shadow Git history, file contents included, in the application's data directory for 30 days by default (`[change_review] retention_days`), and disabling a workspace does not erase it.

No User Guide page states that retention of file contents now. Another session is changing Change Review code (Git LFS filters, TASK-42101, and `Workspaces/change_review_consent.py`), so each sentence must be checked against the code at the time this task is done, not at the filing base. `backlog/docs/lessons-backlog-hygiene.md` ("A 'page has no section for X' finding can be a section a squash deleted") describes the restore procedure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The "Change review" section again states the per-workspace opt-in, the preparation queue and its states, and the shadow Git retention of file contents with its default and its setting, each sentence checked against the code at the time of the change
- [x] #2 A sentence of the old text that no longer matches the code is corrected rather than restored as it was, and the notes list each correction
- [x] #3 The notes cite 6a6242ce3a as the source of the restored text
<!-- AC:END -->

## Implementation Plan

1. Recover the opening paragraphs from `git show 6a6242ce3a^:Docs/User_Guide/console/agent-runs-and-tools.md` (lines 685-703).
2. Check each sentence against the code at base dc371f0842: consent default (`registry_service.read_change_review_consent`), the global capability (`change_bounds.read_change_review_capability`), the Settings panel (`Widgets/Settings_Widgets/workspace_change_review.py`), the preparation queue and readiness states (`Workspaces/change_review_consent.py`, read only), the skipped-folder marker (`Chat/console_agent_bridge.py`), and retention (`Workspaces/change_retention.py`, `app_lifecycle.schedule_media_cleanup`).
3. Restore the paragraphs at the top of "Change review", correcting each sentence the code no longer supports; leave the "Project instructions before tools run" section alone.
4. Run Tests/Docs and preflight; write notes citing 6a6242ce3a, each correction, and which claims touch the open TASK-42101 work.

## Implementation Notes

Restored the three opening paragraphs of "Change review — reviewing a turn's file changes" in `Docs/User_Guide/console/agent-runs-and-tools.md`. Source: `git show 6a6242ce3a^:Docs/User_Guide/console/agent-runs-and-tools.md`, lines 685-703 (added by 2e5d256a04, removed by squash 6a6242ce3a). Each sentence was checked against the code at base dc371f0842. Citations are to that commit.

**Kept, confirmed by the code**
- Off for every new workspace: a workspace with no `workspace_change_review` row reads as disabled (`Workspaces/registry_service.py:3385-3389`). The only consent writer is the Settings panel's toggle (`UI/Screens/settings_screen.py:21970` calls `ChangeReviewConsentService.toggle`, `Workspaces/change_review_consent.py:373`).
- The global setting is only a capability switch: `_admit` returns no ready roots unless both the capability and the workspace's consent are enabled (`change_review_consent.py:300-307`).
- Chat and tools never wait: `admit_turn` schedules missing folders without waiting (`change_review_consent.py:281-285`). The pool is 2 daemon workers over a queue of 32 (`:205-233`).
- The skipped-folder warning: `format_change_review_skipped_marker` writes "⚠ change review skipped {alias}: {reason}" (`Chat/console_agent_bridge.py:1219-1221`), appended as a transcript note with no snapshot state (`:9212-9225`, from `Chat/console_agent_preparation.py:2738-2742`).
- Shadow Git history with file contents lives in the app data directory's `change_review` folder (`Workspaces/change_tracking.py:405-409`). The retention default is 30 (`Workspaces/change_bounds.py:36`), set by `[change_review] retention_days` (`Workspaces/change_retention.py:94-97`). Disabling changes consent only and deletes nothing: `toggle` discards in-memory readiness, and the Settings receipt is "Change Review disabled; existing history is retained." (`Widgets/Settings_Widgets/workspace_change_review.py:266-270`).

**Corrected (each differs from the old text)**
1. "If Git, the global capability, or the workspace registry is unavailable, the toggle is not offered" now says the toggle is also withheld when the global setting is *off*, and that Settings shows the reason in its place. The three rows are at `workspace_change_review.py:144-161`, and the setting defaults to true (`change_bounds.py:40-43`).
2. "Settings shows `preparing`, `ready`, or `failed` state" now says Settings counts the folders in each state (`workspace_change_review.py:177-201`).
3. Added the fourth state, `restart_required`: a folder whose earlier cleanup could not be proven waits for an app restart, and the retry does not clear it. The retry re-runs only FAILED folders (`change_review_consent.py:52-59`, `:313`, `:613-615`; Settings row `workspace_change_review.py:193-199`).
4. Added that a full queue marks a folder failed and retryable (`change_review_consent.py:616-617`, reason `:193`).
5. "an alias-only warning" now says what the alias is: the folder's binding id, not its path (`SkippedReviewRoot(binding.binding_id, …)`, `change_review_consent.py:350`).
6. "stores shadow Git history … for 30 days" overstated what is deleted, so it is now precise:
   - The pass runs about a minute after startup (5 s + 60 s), then daily (`app_lifecycle.py:1462-1468`).
   - It deletes snapshot rows older than the cutoff, except rows a phased workflow still refers to (`DB/AgentRuns_DB.py:1343-1348`).
   - Snapshots are linear `git commit`s (`change_tracking.py:1027`), so `gc` cannot drop older contents while a folder still has rows. The repo is removed only when no row for the folder remains, or when the folder is gone and the repo is older than retention (`change_retention.py:8-16`, `:149-163`).
   - `retention_days` of 0 or less skips the whole pass (`change_retention.py:98-99`).

**Open work this touches (described as the code is at the base)**
- The failed-state sentences touch TASK-42101 (branch `fu2/lfs-filters`). That branch changes `ChangeReviewConsentService._fail`/`_complete` to name an actionable Git refusal in the reason, and makes the Settings failed row append it. The restored text names no reason text, so it stays true after that lands.
- The skipped-folder warning sentence touches TASK-42111, filed on the same branch and not on dev. It reports that the first turn after one restart showed no skipped row; this was seen once and not reproduced. The sentence describes the design that task expects.
- No edit to `change_review_consent.py`, `native_git.py`, or the "Project instructions before tools run" section.

**Checks:** `Tests/Docs` 44 passed, `Tests/MCP/test_mcp_documentation_contract.py` 75 passed (-n 4, through the wrapper).
