---
id: TASK-41533
title: "Local and CI disagree on the first-send warm-up module census since PR #3446"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-17 05:39
updated_date: 2026-11-18 19:27
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

**Rebased 2026-11-18 onto ffdf0c5472, after #3627 (TASK-41800).** The coordinator authorized the rebase. Dev's #3627 landed the same kind of fix as this task:
- 13e7d08740 defers `chachanotes_sql_steps`;
- e5f0484644 makes five census-only sheds from the warm-up.

Where this branch did the same thing, dev's code was kept and this branch's version dropped: the `chachanotes_sql_steps` deferral and its module-size row raise, the `app_served` unlisting, the `Canvas.compilation` `TYPE_CHECKING` move, and the `worktree_activity` deferral. This branch keeps:
- the real served-mode sheds (`served_relay_publisher`, `Web_Server`, and the third-party `textual_serve` step);
- the snapshots;
- the cause analysis, the accounting, and the owner ruling with its ADR-097 ledger row.

All the numbers below are measured on the rebased tree.

**Warm-up census: 99 -> 92.** No raise; the limit stays 94. The new modules, and the commit that added each:
- `RAG_Search.scope_evidence`: bf9182a4f6 (TASK-36947.3, #3471) listed it. It stays listed, because a first Automatic send's evidence capture imports it on the UI loop. Unlisted, `test_an_automatic_library_send_loads_only_the_two_library_modules_lazily` names it as a third cold UI-thread load (checked). The budget comment records this (AC#3). Open PR #3571 moves this import off the UI loop and unlists it.
- `Chat.console_ordinary_tree`, `Agents.ordinary_tree`, `Agents.execution_capacity` and `Workspaces.worktree_activity`: d2ea67fadc (TASK-41319, #3583) removed `console_chat_store`'s module-level import of `console_ordinary_tree`, so the listed `console_agent_reply` now pulls them in. The first three cannot be deferred without a behaviour change: the agent reply awaits `run_ordinary_preparation` on the UI loop, which calls `execution_capacity.current_ordinary_tree`.

How the census came back down:
- **#3627 (e5f0484644), census-only, 99 -> 94.** It removed `app_served`, `Canvas.compilation`, `Canvas.profiles`, `Canvas.runtime_assets` and `Workspaces.worktree_activity`. A booted Console already holds all five, so the live warm-up does not change.
- **This branch, real sheds, 94 -> 92.** `served_relay_publisher` and its `Web_Server` package are unlisted. They are served mode only, and the app imports them when it is built (`app.py` line 1137, `app_served.py` line 71). A native send never imports them, and a native warm-up no longer loads them. #3533 makes the same one-line removal. The `textual_serve` step, listed only as their dependency, goes too. It is third-party, so the census does not count it, but a native warm-up no longer imports it.
- **The limit stays 94** (headroom 2). ADR-097's tightening convention lowers a limit only when a reduction exceeds the guard's standard slack, and pre-import payload's slack is 20 modules.

**The live delta.** Method: probe `scratchpad/fu/r16cs/live_warmup_probe.py`. It warm-boots the app headless to `_ui_ready` the way the UI-ready census does, waits 3 s for the post-ready timers, runs the screen-route walk (where the real warm-up starts), then imports the list in order. It counts new `tldw_chatbook` modules and new third-party top-level packages.
- **Counts:**
  - this rebased head: 78 (2 runs);
  - ffdf0c5472, which is dev with #3627 but without this branch: 80 (2 runs);
  - the last in-budget trees, a94c4aec67 (#3446 before its rebase, CI run 43331816906) and 5d1fc43130 (dev before #3446, CI run 43861223546): 77 each.
- **The +1 against the in-budget trees still exists after #3627.** It is the collision of the two last-slot claims: 5d1fc43130 lists `scope_evidence` but not `agent_lesson_promotion`, a94c4aec67 the reverse, and this head lists both. It ends when #3571 unlists `scope_evidence`.
- **Otherwise the in-budget trees differ from this head by:**
  - the trio, +3;
  - this branch's served-mode sheds, -2;
  - `MCP.console_snapshot`, -1. TASK-41308 made it resident at UI-ready.
- **Lines** (for information only, at today's file sizes): about +1,530 against 5d1fc43130 and +2,250 against a94c4aec67.
- **Earlier comparisons:**
  - The round-1 note compared this head with 1fabd2a7ff, which was itself over budget (its census reads 95, the same collision).
  - The round-1 review's +2 (91 against 89) mixed trees: it applied this head's UI-ready set to the pin's census.
  - Counted right at `_ui_ready` with no settle, the count moves with timing: one tree read 93 and 80 in two runs. Only the settled count compares across trees.
- **Owner ruling 2026-11-18, "Accept until #3571".** The +1 live first-send module (the `scope_evidence`/`agent_lesson_promotion` collision) is accepted until PR #3571 unlists `scope_evidence`; re-measure when #3571 merges. It is recorded in ADR-097's exception ledger (row 2026-11-18, first-send warm-up). That row now describes the state after #3627.

**More real sheds were looked for and not found** (fix round 1). The search covered modules the live warm-up loads after a 3 s settle that are neither resident at UI-ready nor in the live first-send record (`LIVE_FIRST_SEND_IMPORTS` / `LIVE_LIBRARY_FIRST_SEND_IMPORTS`):
- the trio and `console_agent_reply`: agent sends need them on the UI loop;
- `scope_evidence`: needed by Automatic sends;
- `Chat.console_send_start`: TASK-36610 opens it at acceptance;
- `Chat.console_settings_persistence`: the store's `reconcile_durable_turn_settings` and the settings-persistence drain import it;
- the `hooks_v2` and `response_rules` package modules: parents of listed modules;
- seven `Tools.*` modules pulled in by `Agents.local_tool_provider`: the provider builds a `WorkspaceToolExecutor` when it is constructed, and `Tests/Agents/test_local_tool_provider.py` patches it.

None of these can leave without a behaviour change. That leaves the one live module (the collision) that the owner accepted until #3571. The line growth is informational.

**App-import census: 687/688 (limit `< 688`).** No raise.
- **`DB.chachanotes_sql_steps`**: 3911d7df50 (TASK-41310 A, #3592) made `ChaChaNotes_DB` import it at module level. That took dev from 687 to 688. Dev's 13e7d08740 (#3627) defers it to the two upgrade steps. This branch's identical deferral and its size-row raise were dropped in the rebase; dev's version keeps the file at its 22544-line row.
- **`Agents` and `Agents.agent_models`**: 064d5cb014 added `console_chat_models`' re-export of `ConsoleSubmissionOrigin` (app.py -> console_raw_cli -> console_chat_models).
- **`app_services`**: 79a46b3b4e added `console_runtime`'s module-level import of `tldw_chatbook.console_app_services`, and af3e54510f renamed that module to `tldw_chatbook.app_services`.
- Both of those commits are older than the snapshot's last pin (592425f087), which was regenerated before a rebase onto them. These three modules are paid for by three that left the import path: `Character_Chat.local_chat_dictionary_service`, `Chat.console_preparation_reads` and `DB.private_sqlite_files`.
- This branch refreshes the snapshot, so the drift reads zero.

**Evidence (rebased tree).**
- Before (ffdf0c5472, dev plus wave 2): the warm-up census reads 94/94.
- After: the warm-up census reads 92/94. Its refreshed snapshot matches the census.
- App import: 687/688, snapshot refreshed.
- UI-ready census: 1039/1040.
- Preimport route walk: 557/558.
- Live warm-up: 78 (80 at ffdf0c5472).
- 3 of 3 serial runs: `first-send-warmup: 92/94`, `boot-import-weight: 687/688`.
- Boot budget step, serial: 33 passed, 1 failed. The failure is `test_textual_css_fastpath.py::test_ancestor_scoped_bare_type_rule_count_is_a_ratchet` (`assert 270 <= 268`), which also fails at ffdf0c5472; this branch touches no CSS. In a run at load ~97, `test_app_import_stays_well_under_pre_fix_baseline` failed once on wall time (11.4 s against 8.0 s); it passed on the rerun.
- `Tests/CI` (`-n 4`): 1746 passed, 1 skipped.
- The first-send tests (`test_console_first_send_preimport`, `test_console_first_send_import_cycles`, `test_console_first_send_review_imports`) with `test_boot_budget_ratchet_messages`: 72 passed.
- `./scripts/preflight.sh` passes.

**Files (this branch, after the rebase).** `tldw_chatbook/Chat/console_first_send_modules.py` (the served-mode sheds), `Tests/UI/test_console_first_send_import_cycles.py`, `Tests/Performance/test_screen_preimport_payload_budget.py` (budget comment), `Tests/Performance/boot_budget_snapshots/first_send_warmup.json`, `Tests/Performance/boot_budget_snapshots/boot_import_modules.txt`, `backlog/decisions/097-boot-budget-ratchets.md` (ledger row). The lesson is in `backlog/docs/lessons-testing-evidence.md` ("Local fails, CI passes" was never checked against CI's log for the same tree), already on the base.
