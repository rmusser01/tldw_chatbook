### Task 18: Repair stale interrupt fixtures and remove annotation-only host wiring

ADR required: no new ADR
ADR path: backlog/decisions/220-console-human-decision-coordination-ownership.md (dated annotation-only clarification); backlog/decisions/219-console-chat-destinations-and-bounded-starts.md.
Reason: reuse existing test binding and imported types while preserving all runtime live dependencies and authorization behavior.

A fresh sole source implementer follows approved Task17; no children. Read own-SDD task-17-interrupt-preflight.md and Ruling75 for complete constructor/type/caller census, plus exact Qodo feedback in task-17-initial-feedback.json. The recovery finding4179836584 is addressed separately by existing ADR126/219 policy and frozen10case evidence; no paired-store restore, new schema/migration or imported authority change is authorized.

- [ ] On the actual integrated BASE, run the two existing stale test owners Tests/Chat/test_console_decision_clock.py and Tests/UI/test_buddy_speech.py before repair to retain constructor RED (44 source-inferred cases). Preserve all actual collected IDs/failures/output, with shared Python3.12/WT PYTHONPATH/canonical private profiles/-p no:randomly/fresh basetemp/unchanged300s timeout.
- [ ] Replace only their actual InterruptRoundHost import with make_interrupt_host as InterruptRoundHost from Tests.Chat.console_interrupt_test_bindings. Keep separate KIND_SETTER_ATTRS/FakeSeamsFull imports in the clock owner. No fixture/helper attribute addition is needed. Preserve all six original positional call sites, function bodies/assertions/signatures/decorators/parameterization/waits/deadlines/events/cleanup.
- [ ] Add an accurate Google-style docstring to ConsoleChatStartCoordinator.authorizes(authorization, session_id)->bool, documenting nullable authorization and exact live coordinator/store/active-object/session-incarnation check. It establishes neither accepted state nor current primary visibility. Preserve executable method AST exactly; no docstring-only test.
- [ ] Replace exactly17 annotation references to self._read_global_Any()/self._read_global_Mapping() in InterruptRoundHost with existing imported Any/Mapping. Delete only their two keyword-only constructor parameters, two assignments, and matching controller/test-helper lambda arguments. Result120required keywords and33runtime global getters; retain all86controller readers and one write-through callback. Preserve non-annotation executable AST in every other method and exact runtime lookup/patch/nullability/order/alias/lock authority. Separate build_tool_review_hook Any interface and compaction accessors stay exact. No broad receiver/global bypass, dependency bag, proxy, import or new abstraction.
- [ ] Run the same two corrected test owners (44source-inferred cases) and only four binding selectors below (eight source-inferred cases), retaining all actual IDs/argv/source/exits/log/XML/warnings. Prove import reversal recovers both owner bodies and annotation removal/reversal preserves all nonconstructor executable methods. Verify unchanged unowned source/QA/ZIP; preserve old RED and all historical passes/warnings. Fatal Ruff/added-hunk formatter/whitespace plus actual host/controller size measurements; existing downward/slack rules apply only to a demonstrated shrink, never raise caps.
- [ ] Commit only authorized source/test overlays, freeze task-18-report.md and compact additive task-18-safe-evidence/source+AST manifest, return clean source/index/HEAD/closed processes. Root dispatches independent scoped spec/quality review. All four new Qodo threads remain open until reviewed repairs/rebuttal are published with evidence; one fresh current-head Qodo and normal checks/PerfGuard/ancestry precede merge.

**Selected binding controls:**

- Tests/Chat/test_console_owner_live_bindings.py::test_interrupt_remount_reads_replaced_sink_and_controller_state
- Tests/Chat/test_console_interrupt_host_wiring.py::test_legacy_registry_payload_and_lock_names_alias_the_host
- Tests/Chat/test_console_interrupt_host_wiring.py::test_approvals_register_the_permission_summary_as_the_after_remount_hook
- Tests/Chat/test_console_ask_user_round.py::test_timeout_reads_console_config_when_no_seam


### Task18 extension: Await actual mounted Stop control publication

ADR required: no
ADR path: N/A for a test-only synchronization repair; retain ADR094/219 native lifetime/custody.
Reason: current-head CI establishes STREAMING with an idle Stop projection; the awaited sync may coalesce, and pilot.pause is not a publication-completion barrier. No lost-request product defect is demonstrated.

- [ ] Read task-17-stop-ci-preflight.md and its exact frozen one-node receipts plus task-17-PR-fast-lane-failure-receipt.json. Preserve initial CI1fail/221pass/oldxfail/fivewarnings and unchanged06 local1PASS without treating the latter as a fix.
- [ ] In Tests/UI/test_console_runtime_ownership.py::test_accepted_agent_chat_start_has_visible_stop_in_mounted_target, add only await _wait_for_selector(chat, pilot, "#console-stop-generation.console-stop-active") after the existing sync/pilot.pause and before reading/asserting the Stop button. Reuse the already imported helper and its existing2second bound. Preserve every original statement, signature, marker, assertion, physical acceptance/provider entry,120second release guard,0.5second custody timeout, click/cancel/claim/drain/STOPPED/one-generation evidence. No production change, new helper, increased time, retry, skip/XFAIL or relaxed assertion.
- [ ] Run only this exact node once after repair with the same canonical private profile/Python/PYTHONPATH/-p no:randomly/fresh basetemp/300s bounds; retain original failure/localpass and complete new receipts. If it fails the unchanged bound, retain flags/owned workers at the deadline and report before another change/run. Include exact body reversal/source/QA/ZIP carry and fatal Ruff/added-hunk formatter/whitespace in Task18 report and independent scoped review.
