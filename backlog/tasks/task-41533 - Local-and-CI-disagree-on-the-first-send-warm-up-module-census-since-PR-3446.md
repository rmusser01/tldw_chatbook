---
id: TASK-41533
title: "Local and CI disagree on the first-send warm-up module census since PR #3446"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-17 05:39
updated_date: 2026-11-18 07:59
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
- [x] #3 If the lazy import is intended to stay, the pinned budget or allowlist records `scope_evidence` with its reason
- [x] #4 `import tldw_chatbook.app` is back under its ratchet (688), with the commit that added each new module named and each met by a deferral or an equal shed, not a raise (coordinator scope, 2026-11-18)
<!-- AC:END -->

## Implementation Plan

1. Measure the warm-up census at the base and read the Perf Guard logs of dev pushes and of PR #3446's own runs since 2026-11-12.
2. Name why local and CI were thought to differ.
3. Name each module over the limit and its cause.
4. Fix only what fits the criteria without a budget raise; otherwise stop and report what needs a decision.
5. (Coordinator scope, 2026-11-18) Bring both the warm-up census and the app-import census back under their ratchets, by deferring imports or shedding equal cost, naming the commit that added each module.

## Implementation Notes

**Status.** AC#1, AC#3 and AC#4 are met. AC#2 is met locally (3 of 3). Its CI half needs this branch's PR run, so it stays unticked and the task stays In Progress for the coordinator to close.

**AC#1: local and CI do not disagree. The cause is step coverage.** Only Perf Guard's "Run boot budget ratchets" step (`.github/workflows/perf-guard.yml`) runs this census. Perf Guard is not required: dev's only required check is "Derived artifacts reproduce from their sources", read from the branch protection API. For the same tree it measures what local measures:
- dev 5d1fc43130, just before #3446 (run 43861223546): 94/94, pass.
- PR #3446's own runs: a94c4aec67 (43331816906, 2026-11-12, before #3471 merged) 94/94, pass. e80393395e (43595456091, rebased onto a dev that had #3471) `assert 95 <= 94`, naming `RAG_Search.scope_evidence`. 090971dd76 (43861456650) failed the same way. #3446 merged red (3917e2d0cb).
- dev pushes since: 3917e2d0cb (43869046143), 1a1f00c052 (43881752840) and 5e3c8b2bd4 (43917600520) read `95 <= 94`, with the module named as local names it. 97f8c58c0c (44226715284) reads `99 <= 94`, and the same run also fails the app-import guard, `688 < 688`.
- The PR runs that pass are branches that change the list themselves: #3571 drops `scope_evidence`, and #3533 drops `served_relay_publisher`.
- The 95 came from two PRs that each took the row's one free slot. #3471 (TASK-36947.3, bf9182a4f6) listed `scope_evidence`. #3446 (TASK-33620.15.53) listed `agent_lesson_promotion` on a base without it, and re-pinned the snapshot there (1fabd2a7ff).

**How urgent.** Perf Guard has failed on every dev push since 2026-11-16 12:42 UTC and on every PR that touches the package. Since it is not required, nothing blocks a merge. Its red also hides the next breach: #3592's +1 on app import merged unseen.

**Warm-up census: 99 -> 94 at this base.** No raise. Each new module, the commit that added it, and how it is paid for:
- `RAG_Search.scope_evidence`: bf9182a4f6 (TASK-36947.3, #3471) listed it. It stays listed, because a first Automatic send's evidence capture imports it on the UI loop. Unlisted, `test_an_automatic_library_send_loads_only_the_two_library_modules_lazily` names it as a third cold UI-thread load (checked). The budget comment now records this (AC#3). Open PR #3571 moves this import off the UI loop and unlists it; it will free one slot when it lands.
- `Chat.console_ordinary_tree`, `Agents.ordinary_tree` and `Agents.execution_capacity`: d2ea67fadc (TASK-41319, #3583) removed `console_chat_store`'s module-level import of `console_ordinary_tree`. App + chat therefore no longer loads these three, and the listed `console_agent_reply` now pulls them in. They cannot be deferred without a behaviour change: the agent reply awaits `run_ordinary_preparation` on the UI loop, which calls `execution_capacity.current_ordinary_tree`, so a deferred import would land on the first agent send's loop. They are paid for by sheds.
- `Workspaces.worktree_activity`: the same commit. It is deferred. `console_ordinary_tree` now imports it inside `_capture_maximum`, its only user. The live app already holds it at UI-ready, so the deferral costs nothing.
- The sheds, four modules for the four above:
  - `served_relay_publisher` and its `Web_Server` package are unlisted. They are served-mode only, and the app imports them when it is built (`app.py` line 1137, `app_served.py` line 71). A native send never imports them, so the warm-up was loading them for nothing in native runs. #3533 makes the same one-line removal.
  - `app_served` is unlisted. It is resident at UI-ready (it is in the UI-ready census's `EXPECTED_AT_READY`), so its step did nothing in the live app.
  - `Canvas.compilation` moves under `TYPE_CHECKING` in `Canvas/native_authority.py`, which uses it only for annotations. The live app holds it at UI-ready.

  The row's unit is modules. In lines of code the warm-up still grew: the snapshot's informational LOC went from 58,480 to 66,203, and about 3,400 of that is the modules above. The rest is edits to listed modules since the last pin.

**App-import census: 688 -> 687 (limit `< 688`).** No raise. The snapshot showed +4/-3:
- `DB.chachanotes_sql_steps`: 3911d7df50 (TASK-41310 A, #3592) made `ChaChaNotes_DB` import it at module level. This is the +1 that took dev from 687 (CI 43917600520 at 5e3c8b2bd4: 687/688, drift +3/-3) to 688. It is deferred: the two upgrade steps that use it (v85->v86 and v86->v87) import it where they run. The module-size row for `ChaChaNotes_DB.py` rises by that fix's own line, 22544 -> 22545, with a dated comment, as the owner's size-ratchet rule for fixes asks.
- `Agents` and `Agents.agent_models`: 064d5cb014 added `console_chat_models`' re-export of `ConsoleSubmissionOrigin` (app.py -> console_raw_cli -> console_chat_models). `app_services`: 79a46b3b4e added `console_runtime`'s module-level import of it. Both commits are older than the snapshot's last pin (592425f087), which was regenerated before a rebase onto them. These three are already paid for, and have been counted inside the 687 since at least 5d1fc43130, by three modules that left the import path: `Character_Chat.local_chat_dictionary_service`, `Chat.console_preparation_reads` and `DB.private_sqlite_files`. The snapshot is refreshed, so the drift now reads zero.

**Evidence.**
- Red at this base: `assert 99 <= 94` and `assert 688 < 688`.
- Green after, 3 of 3 serial runs: `first-send-warmup: 94/94`, `boot-import-weight: 687/688`. The refreshed `first_send_warmup.json` matches the census's module set and listed entries exactly.
- No other boot budget moved: the preimport route walk stays 558/558, and the UI-ready census reads 1039/1040. That is one lower, because `chachanotes_sql_steps` is no longer resident on a warm boot.
- Covering tests, `-n 4`, load 15-35:
  - `test_console_first_send_preimport`, `test_console_first_send_import_cycles`, `test_console_first_send_review_imports`, `test_console_relay_approval_integration`, the boot budget files, `test_app_startup_delegation`, `test_console_approval_critical_gate`, `test_app_import_diet_closure` and the v87 migration test all passed. The only failures in that batch (132 passed, 8 failed) are the 8 cases of `test_first_send_warmup_gc_overlap`. That test calls `preimport_first_send_modules(app)`, but the function now also takes `on_complete`; the same 8 fail at the base. #3533 fixes that call.
  - The ordinary-tree, SSH-custody, archive-recovery-dispatch, workflow-scheduling and Canvas files gave 523 passed, 70 failed. The base (67583dfa72), run with the same command, fails the same 70 nodes. The Canvas startup cases among them fail on the known local `RecoveryRequired: raw_source_selection_changed`.
- `Tests/Architecture`, serial: 1471 passed, 5 failed, 5 skipped. All 5 also fail at the base: the app string-lookup census `[Widgets]`, the layer map (`config.py` -> `Chat.console_provider_endpoints`), the persistent diagnostic inventory, and the two timer-path inventory nodes. The module-size row for `ChaChaNotes_DB.py` passes at 22545.
- UI-ready census after these changes: 1039/1040 on 3 of 3 runs.
- `./scripts/preflight.sh` passes.

**Files.** `tldw_chatbook/Chat/console_first_send_modules.py`, `tldw_chatbook/Chat/console_ordinary_tree.py`, `tldw_chatbook/Canvas/native_authority.py`, `tldw_chatbook/DB/ChaChaNotes_DB.py`, `Tests/Performance/test_screen_preimport_payload_budget.py` (budget comment), `Tests/Performance/boot_budget_snapshots/first_send_warmup.json`, `Tests/Performance/boot_budget_snapshots/boot_import_modules.txt`, `Tests/Architecture/test_module_size_ratchet.py`. The lesson is in `backlog/docs/lessons-testing-evidence.md` ("Local fails, CI passes" was never checked against CI's log for the same tree).
