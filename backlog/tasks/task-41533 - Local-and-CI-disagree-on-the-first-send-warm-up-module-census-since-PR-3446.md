---
id: TASK-41533
title: "Local and CI disagree on the first-send warm-up module census since PR #3446"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-17 05:39
updated_date: 2026-11-18 07:12
labels:
- follow-up
- tests
- performance
priority: medium
references:
- Tests/Performance/test_screen_preimport_payload_budget.py
- tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py
dependencies:
- TASK-33620.15.53
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reported by the coordinator at the filing review (round 15). Since PR #3446 (merge 3917e2d0cb, TASK-33620.15.53) the warm-up test in `Tests/Performance/test_screen_preimport_payload_budget.py` fails locally and deterministically: 95 modules against a limit of 94. The new module is `tldw_chatbook.RAG_Search.scope_evidence`, imported lazily by `Event_Handlers/Chat_Events/chat_rag_events.py`. It passes at cc6006fcf5 and every merge before #3446. CI's boot budget step passes on dev, so the two disagree.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The reason local and CI differ is named (environment, import order, snapshot, or step coverage)
- [ ] #2 The snapshot and the census agree in both local and CI runs, on 3 of 3 runs each
- [ ] #3 If the lazy import is intended to stay, the pinned budget or allowlist records `scope_evidence` with its reason
<!-- AC:END -->

## Implementation Plan

1. Measure the warm-up census at the base and read the Perf Guard logs of dev pushes and of PR #3446's own runs since 2026-11-12.
2. Name why local and CI were thought to differ.
3. Name each module over the limit and its cause.
4. Fix only what fits the criteria without a budget raise; otherwise stop and report what needs a decision.

## Implementation Notes

**Stopped: NEEDS_CONTEXT.** AC#1 is answered below. AC#2 and AC#3 are not met. No change within the criteria brings the row back to 94 without a budget raise, and the brief rules out a raise. The options are at the end.

**AC#1: local and CI do not disagree. The cause is step coverage.** Only Perf Guard's "Run boot budget ratchets" step (`.github/workflows/perf-guard.yml`) runs this census, and Perf Guard is not the required check. For the same tree it measures what local measures:
- dev 5d1fc43130, just before #3446 (run 43861223546): 94/94, pass.
- PR #3446's own runs: a94c4aec67 (43331816906, 2026-11-12, before #3471 merged) 94/94, pass. e80393395e (43595456091, 2026-11-14, rebased onto a dev that had #3471) `assert 95 <= 94`, naming `RAG_Search.scope_evidence`. 090971dd76 (43861456650) failed the same way. #3446 merged red (3917e2d0cb).
- dev pushes after that: 3917e2d0cb (43869046143), 1a1f00c052 (43881752840) and 5e3c8b2bd4 (43917600520) each read `95 <= 94`, with `scope_evidence` NEW in both the listed entries and the modules, as local does. 97f8c58c0c (44226715284) reads `99 <= 94`, as this base does.
- The PR runs that pass are branches that change the warm-up list themselves: #3571 (db5397068b, 94/94) drops `scope_evidence`, and #3533's branch reads 93/94.
- The 95 comes from two PRs that each took the row's one free slot. #3471 (TASK-36947.3, bf9182a4f6, merged 2026-11-13 01:32 UTC) listed `scope_evidence` (93 -> 94). #3446 (TASK-33620.15.53) listed `agent_lesson_promotion` (93 -> 94) on a base without it, and re-pinned the snapshot there (1fabd2a7ff), so the snapshot's `listed` lacks `scope_evidence`.

**Measured at this base (67583dfa72): 99 against 94, the same on every run.** The census imports in list order in a fresh interpreter, so none of the extra modules depends on timing. Beyond the snapshot:
- `RAG_Search.scope_evidence` (+1) is a real dependency. The first Automatic send's evidence capture imports it on the UI loop (`chat_rag_events.capture_console_prepared_evidence_for_chat`). With it unlisted, `test_an_automatic_library_send_loads_only_the_two_library_modules_lazily` fails and names it as a third cold UI-thread load (checked). Importing it after the warm-up costs 1.1-1.7 ms (3 runs).
- `Chat.console_ordinary_tree`, `Agents.ordinary_tree`, `Agents.execution_capacity` and `Workspaces.worktree_activity` (+4) are all pulled in by the listed `Chat.console_agent_reply`, whose module-level import of `console_ordinary_tree` has not changed. They are new to this row because TASK-41319 (d2ea67fadc, #3583, in dev 97f8c58c0c) removed the boot modules' imports of `console_ordinary_tree`, so app + chat no longer loads them. The UI-ready census lost the first three at the same time (TASK-41512). The modules moved from boot to the post-ready warm-up; no new code was added. The agent reply awaits `run_ordinary_preparation` on the UI loop, so deferring that import would put it on the first agent send's loop.

**Why no change was made.**
- Unlisting `scope_evidence` (-1) breaks the owner's 2026-10-31 contract that a first Automatic send lazily loads exactly the two Library modules (TASK-33620.15.19). The other way is to move its import off the UI loop, and open PR #3571 (TASK-41037) already does that in the same lines.
- The +4 needs either an owner exception, or deferrals in other streams' modules for imports the live app already holds at UI-ready (lesson TASK-33620.15.53). Unlisting the two listed modules resident at UI-ready (`app_served`, `MCP.console_snapshot`) together with `scope_evidence` measures 97, not 96, because `MCP.console_tool_preparation` imports `console_snapshot` again at module scope. The other modules this census counts only because its start leaves out the mount are `worktree_activity` (via `console_ordinary_tree`), `Canvas.profiles` and `Canvas.runtime_assets` (via `Canvas.compiler`), and `Canvas.compilation` (via `Canvas.native_authority`). Each is a module-level import to defer in another stream's module.

**Options for the owner:**
- (a) An ADR-097 exception, 94 -> 99 (98 once #3571 lands), naming both causes.
- (b) Land #3571 (-1, to 98), then defer the imports above that the census alone counts (up to -6).
- (c) Start the census from app + chat + the UI-ready residents. This changes the method.

AC#2's CI half needs this branch's PR run, which the coordinator checks.

**Files.** None for this task. The lesson is in `backlog/docs/lessons-testing-evidence.md` ("Local fails, CI passes" was never checked against CI's log for the same tree).
