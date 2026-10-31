# Console OS notification review repair implementation plan

> **For agentic workers:** Execute inline with superpowers:executing-plans. The parent supplies the final independent review.

**Goal:** Repair PR #3155's notifier lifetime, literal Linux text, and failure diagnostics without changing toast cadence or Settings ownership.

**Architecture:** Retain the fixed-argv dispatcher and existing once-guarded screen bridges. A non-daemon one-shot thread retains the subprocess timeout during orderly interpreter shutdown; Linux body text is XML escaped; failure logs carry fixed categories and status/type only.

**Tech Stack:** Python standard library, existing Textual/pytest/loguru stack; no new dependency.

**Spec:** [TASK-18925](../../../backlog/tasks/task-18925%20-%20Opt-in-OS-notifications-for-background-Console-events.md), plus the 2026-10-31 PR review findings.

ADR required: yes
ADR path: backlog/decisions/282-console-os-notification-dispatch.md
Reason: External platform dispatch, privacy boundary, and subprocess lifetime require a recorded decision.

## Global constraints

- Default OFF; preserve per-type gating and existing once-per-event cadence.
- Notify only conversation/workspace display names and fixed outcome categories.
- macOS osascript and Linux notify-send; unsupported platforms silently skip with debug metadata.
- Subprocess timeout remains 3 seconds; no blocking notifier work on the UI loop.
- User explicitly waived size limits: update touched rows to exact measured values, without reflow/extraction to fit.
- Targeted tests only. No push, auto-merge, release, or unrelated PR actions.

## Review focus

- Exit while the notifier is slow: real child must still time out and finish the sender's cleanup.
- Names containing XML tags, ampersands, or quotes: Linux renders literal identity, macOS keeps its existing literal form.
- Nonzero exits and exceptions containing private text: return False and log only category plus exit code/exception type.
- Thread identity: notifier work executes off the invoking thread.
- Settings and cadence regressions: rerun the existing mounted journeys and configuration pins.

## Task 1: Repair the three dispatch boundaries

**Files:** `Tests/Utils/test_os_notifications.py`, `tldw_chatbook/Utils/os_notifications.py`, `tldw_chatbook/UI/Screens/chat_screen.py`.

- [x] Add failing tests for XML escaping, nonzero/exception log privacy, and real slow-child interpreter shutdown.
- [x] Run those tests against the original implementation and record the expected failures.
- [x] Make the notifier thread non-daemon; escape only the Linux body with `html.escape(..., quote=False)`; inspect returncode and redact exception logging.
- [x] Rerun the dispatcher tests and existing Console/Settings/config tests.

## Task 2: Record the decision and verification

**Files:** ADR-282; task-18925; `Docs/Architecture/notifications.md`; both size-ratchet tests; diagnostic inventory; UI census and floor.

- [x] Link ADR-282 from task, plan, architecture, and ADR index; document supported platforms, privacy, separate inbox, and exit tradeoff.
- [x] Measure ChatScreen/Settings and update exact pins with the explicit waiver note.
- [x] Review changed diagnostic statements before regeneration; verify the inventory.
- [x] Append qualified UI coverage while preserving the existing 1,173-entry prefix; record any unqualified file explicitly.
- [x] Run targeted governance, lint, diff checks; document evidence and native-platform limitations; commit for independent review.

## Execution ledger

- Base for repairs: `3c011a3f0d5ea7c3e6aa7392545ab78767c749ab`; current-dev base `57e330e735fff25515778e839f31ed85602c34f2`.
- Existing isolated worktree: `pr3155-resync/tldw_tui`, branch `codex/backlog-pr3155-repair`.
- Parent's pre-repair run: 32 passed, 63 deselected, 2 warnings, 252.26 seconds; no claim of native desktop delivery.
- ADR allocation: all reachable refs/history and live worktree decision filenames scanned; highest canonical ID 281. ADR-282 is provisional until merge-time collision recheck.
- Ruling: retain the parent's explicit inline implementation and fresh final-review workflow; no extra planning approval or subagent dispatch.

- RED: the five new cases failed as intended (15.05 s): raw Linux markup, True on nonzero exit, two private exception logs, and missing shutdown completion. The real slow child exited naturally after its abandoned parent.

- Final receipts: dispatcher 28 passed; feature selection 35 passed / 2 private-child teardown failures; Settings grammar 8 passed; serial Console replay 4 passed / 2 different private-child teardown failures; 6 targeted size-pin checks passed. See the QA receipt for every failing node and the patched spawn/sender boundary.
- Ruling: only the verified Settings journey enters the verified-green census (1,174 total). Console remains in the repository but unqualified; no previous census entry was removed. No third suite or unrelated native-app cleanup is authorized in this repair.
- Ruling: config.py's 12 default-table lines also exceed its size pin; the same explicit user waiver pins its exact 12,245-line measurement. ChatScreen is 24,420/739; Settings is 35,758.
- Diagnostic regeneration and one-owner postcheck passed; new module/test Ruff checks and format checks passed, with no new findings in changed existing files. Native desktop display is unverified; CI was explicitly waived by the user. Task remains In Progress for fresh independent review.
