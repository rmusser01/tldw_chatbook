# SDD ledger — plan: Docs/superpowers/plans/2026-08-26-console-full-semantic-capture.md

Workspace: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/full-semantic-capture`
Branch: `codex/full-semantic-capture`
Plan base: `de580f20ba9c3a521d4c336668c2c7946d67c614`

Baseline: `54 passed, 1 warning in 10.46s` for the four incumbent Task 1 seams. Pytest also emitted sandbox-only temporary-directory cleanup warnings after exit 0; no test failed.

## Preflight conflict scan

| Scope | Producer / code | Consumer / tests | Finding |
|---|---|---|---|
| Task 1 self-check | Safe/Full types, shared budget, migration, repository, matching provenance | Pure capture, migration, repository, store, and loader tests | Clean: tests exercise every produced interface and Full remains UI-inaccessible. |
| Task 2 self-check | Session state, admission resolver, frozen stream/call signals, config mutation | Admission matrix, retry/tool/fleet/provider/config tests | Clean: exact revision signatures and cancellation timing agree with examples. |
| Task 3 self-check | Queryable count/delete, staged cache swaps, controller quiescence, revision fences | DB rollback, live/ephemeral cache, writer-race, and stale Inspector tests | Clean: every fallible lookup/allocation is staged before durable commit. |
| Task 4 self-check | Shared policy bindings/modal, exchange exporter, Inspector/Trace/Settings wiring | Projection, confirmation, imported-read-only, 80x24, focus, and docs gates | Clean: projection fields match tests and `impeccable` is required before frontend edits. |
| Tasks 1 → 2 | `CaptureDetail`, resolver, budget, repository, persisted provenance | Store/controller/provider policy threading | Clean: Task 2 consumes only Task 1 public contracts. |
| Tasks 1 → 3 | Queryable non-null `capture_detail`, repository/storage seams | Conversation-wide Full count/delete and cache purge | Clean: Task 3 never decodes blobs to classify records. |
| Tasks 1 → 4 | Provenance-aware Inspector loader and historical Safe compatibility | Per-call labels and governed export | Clean: Task 4 reads stored provenance instead of current policy. |
| Tasks 2 → 3 | Frozen run/fleet ownership and process-local policy state | Quiescence blocker inventory and purge serialization | Clean: purge treats retained provider signals as writers. |
| Tasks 2 → 4 | Immutable snapshot/mutation/config-generation contract | Shared Inspector/live Trace/F9 controls | Clean: all surfaces share one owner and one stale-write fence. |
| Tasks 3 → 4 | Count/purge callbacks and capture revision | Purge UI, refresh, expansion/copy/save/export fences | Clean: successful deletion remains authoritative if repaint fails. |

Preflight result: no conflict with the spec or Global Constraints; no ruling required before Task 1.

Task 1: dispatched from base `de580f20ba9c3a521d4c336668c2c7946d67c614` to `/root/task1_capture_persistence` (`gpt-5.6-terra`, high).

Task 1: concern gate — focused implementation commit `311f8cff03`; controller full DB run completed with 8 failures and 1819 passes/1 skip implied by the 1828-item run. Exact failed node IDs are recorded in `.pytest_cache/v/cache/lastfailed`.

Task 1: Ruling: expand the brief's file ownership to `tldw_chatbook/DB/sql_validation.py`, `Tests/ChaChaNotesDB/test_index_census.py`, `Tests/DB/test_chachanotes_v49_messages_fts_update_scope.py`, and the existing allowlist guards — the mandatory complete DB gate proves the new table/index/current version must update those independently maintained schema contracts; if wrong, this adds four bookkeeping files to Gate 1 that the original file list omitted.

Task 1: concern resolved by `bd6fe35536`; controller rerun `Tests/ChaChaNotesDB Tests/DB -q` from fixed HEAD: `1827 passed, 1 skipped, 4 warnings in 204.31s`, exit 0.

Task 1: independent review dispatched to `/root/task1_review` over `de580f20..bd6fe355`.

Task 1: review found 3 Important privacy gaps (endpoint alias sanitization, JSON tool credential/binary sanitization, response-content stubbing); fix round 1/5 dispatched from `bd6fe35536`.

Task 1: fix round 1/5 (3 addressed, 0 open — endpoint alias, JSON tool credentials/binaries, response content; commit `16915283a4`).

Task 1: complete (commits `de580f20..707c8708`, review clean).

Task 2: dispatched from base `707c87084b6e934bf416369eb9c5db8dcb2d8769`; implementation owner `/root/task2_policy_threading` (`gpt-5.6-sol`, high).

Task 2: implementation commit `cf586ca823`; Gate 2 reported `541 passed, 2 skipped`, changed-file Ruff/py_compile/diff-check clean. Independent review `/root/task2_review` found 2 Critical and 3 Important issues: llama.cpp credential bypass, durable conversation write before revision ownership, unbounded/non-idempotent response capture, generic adapter-boundary distortion, and rejected ephemeral Full staging.

Task 2: Ruling: expand fix-round ownership to `tldw_chatbook/Chat/console_exchange_capture.py` and `Tests/Chat/test_console_exchange_capture.py` because the smallest safe llama.cpp correction requires one public shared arbitrary-value sanitizer, and the same review proved the existing short data-URI/base64 threshold violates the plan's absolute binary exclusion; if wrong, this adds one production seam and its focused tests to Task 2 instead of deferring a known privacy defect to final review.

Task 2: fix round 1/5 dispatched from `cf586ca823` for 2 Critical + 3 Important findings and the two directly related shared privacy regressions (short binary/data URI and exception `repr`).

Task 2: fix round 1/5 implemented in `86690344f4`; implementer reports `549 passed, 2 skipped`, Ruff/py_compile/diff-check clean. Scoped re-review dispatched over `cf586ca823..86690344f4`.

Task 2: fix round 1/5 re-review — 5 findings addressed; 1 Critical concurrency finding remains open (repeat cancellation reconciliation + admission/one-shot consumption during reservation), and 1 new Important issue found (session closure can strand the process-wide mutation token).

Task 2: fix round 2/5 dispatched from `86690344f4` for the remaining cancellation/admission divergence and terminal session-removal cleanup.

Task 2: fix round 2/5 implemented in `786f36133b`; implementer reports 4/4 exact regressions, 18/18 cumulative fix focus, Gate 2 `553 passed, 2 skipped`, Ruff/py_compile/diff-check clean. Scoped re-review dispatched over `86690344f4..786f36133b`.

Task 2: fix round 2/5 re-review — admission consumption and session-removal cleanup addressed; 1 cancellation-loop defect remains open because a cancelled durable task makes the shield loop spin forever.

Task 2: fix round 3/5 dispatched from `786f36133b` for cancelled durable-task termination/reconciliation.

Task 2: fix round 3/5 implemented in `69c26d8001`; implementer reports exact cancellation 2/2, cumulative fix focus 19/19, Gate 2 `554 passed, 2 skipped`, Ruff/py_compile/diff-check clean. Scoped re-review dispatched over `786f36133b..69c26d8001`.

Task 2: fix round 3/5 re-review — busy loop fixed, but reservation is still released before the uncancellable `to_thread` repository write settles; the new test encoded durable Full/runtime None and permits an older worker to overwrite a newer edit.

Task 2: fix round 4/5 dispatched from `69c26d8001` to retain ownership through repository settlement and runtime reconciliation.

Task 2: fix round 4/5 implemented in `510be33822`; implementer reports focused cancellation 2/2, cumulative fixes 14/14, sanitizer 3/3, Gate 2 `554 passed, 2 skipped`, Ruff/py_compile/diff-check clean. Scoped re-review dispatched over `69c26d8001..510be33822`.

Task 2: fix round 4/5 re-review — normal settlement is fenced, but cancellation loses precedence over repository exceptions and direct cancellation of the asyncio repository wrapper can still release ownership while the `to_thread` worker runs.

Task 2: fix round 5/5 dispatched from `510be33822`; final allowed round, requiring an independently retained worker-settlement handle plus explicit recorded-cancellation precedence.

Task 2: fix round 5/5 implemented in `e411781bd8`; implementer reports exact focus 4/4, cumulative cancellation/race 7/7, Task 2 focus 18/18, Gate 2 `558 passed, 2 skipped`, sanitizer 3/3, Ruff/py_compile/diff-check clean. Final scoped re-review dispatched over `510be33822..e411781bd8`.

Task 2: complete (implementation `cf586ca823..e411781bd8`, closeout `f9c1f66853`; final re-review clean, 5/5 bounded fix rounds used).

Task 3: dispatched from base `f9c1f66853eb009b0ef8324147595e41f438338a`; implementation owner `/root/task3_purge_revision` (`gpt-5.6-sol`, high).

Task 3: implementation commit `16cfc0991a`; implementer reports exact Gate 3 `116 passed`, Ruff/py_compile/diff-check clean. Independent review dispatched over `f9c1f66853..16cfc0991a`.

Task 3: independent review found 1 Critical + 4 Important issues: target purge overwrites unrelated global cache/tag/revision state and overlapping stages can roll revisions back; live swaps run on a worker thread; Inspector mounts can repopulate stale Full bodies; missing sessions raise instead of bounded results; durable sessions undercount live-only Full removals.

Task 3: fix round 1/5 dispatched from `16cfc0991a` for all review findings plus immutable staged nested collections and the directly related content-free exchange-flush log gap.

Task 3: fix round 1/5 implemented in `61439ade25`; implementer reports focused fixes 31/31, exact Gate 3 `125 passed`, Ruff/py_compile/diff-check clean. Scoped re-review dispatched over `16cfc0991a..61439ade25`.

Task 3: fix round 1/5 re-review — findings 1–6 addressed; 1 Important logging leak remains because `ChatPersistenceService.append_message_exchanges()` records `repr(exc)` and its test lacks a semantic canary.

Task 3: fix round 2/5 dispatched from `61439ade25` to `/root/task3_fix2_log` for the content-free persistence-wrapper log contract.

Task 3: fix round 2/5 implemented in `2f2be5d7ad`; exact wrapper RED 1 failed then GREEN 1 passed, exact Gate 3 `125 passed`, Ruff/py_compile/diff-check clean. Scoped re-review dispatched over `61439ade25..2f2be5d7ad`.

Task 3: complete (implementation `16cfc0991a..2f2be5d7ad`, closeout `1b50778714`; final re-review clean after 2/5 fix rounds).

Task 4: Impeccable context resolved to the incumbent Textual Operate surface; hardening/craft-floor and Textual layout/styling/worker/testing guidance applied. Dispatched from base `1b50778714c0d9183826d5180445f73acf50b091` to `/root/task4_shared_controls` (`gpt-5.6-sol`, high).

Task 4: Ruling: immutable Inspector/live-Trace bindings must refuse global apply when their opener session is no longer the controller's active owner, because the incumbent global mutation result snapshots `active_session_id`; if wrong, this adds one fail-closed guard instead of allowing a stale modal to report or retarget to another chat.

Task 4: Ruling: expand ownership to `tldw_chatbook/config.py` and `Tests/test_config_save_settings_semantics.py` to break the Task 2 top-level `config -> Chat.__init__ -> runtime_policy.bootstrap -> config` cycle with a deferred import of the single canonical `CaptureDetail`; if wrong, this adds one import-isolation regression and a local-import refactor rather than the broader/riskier alternative of redesigning eager `Chat.__init__` exports.

Task 4: Ruling: include the CSS builder's four additional generated outputs (`screen_css_self.tcss`, `screen_css_scoped.tcss`, `widget_defaults_self.tcss`, `widget_defaults_scoped.tcss`) because bundle sync is green only when all derived artifacts match `_agentic_terminal.tcss`; if wrong, this adds four mechanical files rather than shipping source/bundle drift.

Task 4: implementation commits `d685a90009` + `37b3041381`; implementer reports final matrix `861 passed, 2 skipped`, 80x24 gate `101 passed`, sentinel inspection green, Ruff/py_compile/CSS/docs/diff checks green.

Task 4: Impeccable detector run once over completed UI targets; two advisory literal-color findings are pre-existing (`#6f7782` commit `169a6ba040b`, `rgb(245,245,245)` commit `3263f846414`) and untouched by Task 4. No Task 4 detector correction required.

Task 4: independent review dispatched over `1b50778714..37b3041381`.

Task 4: independent review found 0 Critical + 8 Important issues: missing global-Full acknowledgement; modal selection/preview drift; F9 Settings bypassing the live coordinator; post-commit repaint masking successful purge; missing immediate pre-projection revision fence; raw atomic-write exception/path logging at the export boundary; a non-production sentinel test with false storage-owner claims; and screen-size-ratchet growth. One Minor issue retires callable raw Copy/Save disclosure paths.

Task 4: Ruling: expand fix-round ownership to `tldw_chatbook/Utils/atomic_file_ops.py` plus focused tests, and to a new `tldw_chatbook/UI/Console_Modules/capture_policy_bindings.py` seam plus architecture tests, because the selected export path crosses the shared helper's unsafe log boundary and the enforced one-way screen-size ratchet assigns the new binding/purge integration outside `chat_screen.py`; if wrong, this touches one shared utility option and one new Console module instead of either tolerating a privacy leak or raising the architecture budget.

Task 4: fix round 1/5 dispatched from `37b3041381` for all 8 Important findings and the directly related retired raw Copy/Save path cleanup.

Task 4: fix round 1/5 implemented in `478a9f1bae`; evidence/backlog/progress commit `cc0f5aab81`. Implementer reports exact matrix `867 passed, 2 skipped`, 80x24 gate `105 passed`, Settings/config/layout `379 passed`, focused fixes `77 passed`, exporter/privacy log sentinel `7 passed`, real gateway/controller/store/persistence/cache sentinel green, and Ruff/py_compile/CSS/docs/diff checks clean. `chat_screen.py` is `20,093` lines / `633` methods versus Task 4 base `20,099` / `633`; the explicit Task-4-delta node is green while the older repository-wide absolute ceiling remains independently stale and was not raised.

Task 4: fix round 1/5 scoped re-review dispatched over `37b3041381..cc0f5aab81`.

Task 4: fix round 1/5 re-review — 5 prior findings and the Minor raw-path cleanup are addressed; 3 Important findings remain open: scoped Global Full acknowledgement is bypassed when conversation Safe masks effective detail, Off-state prospective preview ignores the selected edit, and the sentinel still substitutes list persistence/checks compressed bytes instead of a real Anthropic-shaped SQLite/cache round trip.

Task 4: fix round 2/5 dispatched from `cc0f5aab81` for the three remaining review findings.

Task 4: fix round 1/5 implemented in `478a9f1bae`; exact matrix `867 passed, 2 skipped`, 80x24 gate `105 passed`, Settings/config/layout `379 passed`, exporter + log sentinel `7 passed`, focused fixes `77 passed`, Ruff/py_compile/CSS/docs/diff checks green. The explicit Task 4 screen delta is 20,093 lines/633 methods versus base `1b50778714` at 20,099/633; the older 17,727/593 repository ceiling remains independently stale and was not raised.

Task 4: fix round 2/5 implemented in `26218ae5aa`; scoped Global Full cancellation and Off-state prospective preview are GREEN, and the sentinel now uses Anthropic resolution through the real gateway/controller/store/`ChatPersistenceService`/ChaChaNotes SQLite seam with production queries and decoded cache/storage captures. Exact matrix `869 passed, 2 skipped`, 80x24 gate `107 passed`, Settings/config/layout `379 passed`, focused re-review set `99 passed`, real sentinel `1 passed`, and Ruff/py_compile/CSS/docs/diff checks green. Task 4 remains In Progress with ACs unchecked for scoped re-review.
