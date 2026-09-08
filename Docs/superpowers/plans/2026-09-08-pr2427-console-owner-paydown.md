# PR 2427 Console owner paydown implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Keep every move attributable and review each owner before the next.

**Goal:** Restore ChatScreen's existing 16,811-line / 505-method ceiling by completing approved existing-owner moves without changing behavior.

**Architecture:** Roleplay/settings persistence belongs to the existing settings durability controller, endpoint probing to settings navigation, handoff staging to the existing session controller, and message presentation to the existing message controller. Screen UI, lifecycle, event/action hooks and callback identity remain intact. No new controllers, mixins or generic dependency bags.

**Tech stack:** Python 3.12, Textual 8, asyncio, existing Console controllers and pytest.

**Approved design:** User approved additional existing-controller moves after checkpoint `c27723b623`; this is the exact scoped inventory for that proposal. Source baseline is 17,534 lines / 520 methods.

ADR required: no new ADR.
ADR path: N/A; `DESIGN.md` section 7 and `Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md` govern existing ownership and migration safety.
Reason: direct completion of established controller boundaries, no storage/runtime/service contract change.

## Task 1: Characterize and census

Files: `tldw_chatbook/UI/Screens/chat_screen.py`, existing controllers under `tldw_chatbook/UI/Console_Modules/`, `Tests/UI/test_console_controller_wiring.py`, affected roleplay/settings/handoff/presentation tests.

- [ ] Inventory every callsite, dynamic patch target and state read/write for the exact names below, including module-qualified imports and framework-name hooks. Distinguish screen identity from a callable seam; never send the controller where Textual expects the screen.
- [ ] Run unchanged `Tests/Architecture/test_screen_size_ratchet.py` and record the Console failure; add failing owner/wiring tests before moving methods. Add characterization only where behavior is untested. Verify real callback replacement after construction and long-lived callback identity.
- [ ] Record baseline complete roleplay-resume, settings durability/navigation, chat-handoff/live-work, presentation and wiring files selected by actual caller census. Preserve every behavioral assertion. Existing native-chat baseline:351 passed; rerun complete file after integration, not for each private method.

## Task 2: Existing settings durability owner

Files: `chat_screen.py`, `UI/Console_Modules/settings_durability.py`, `UI/Console_Modules/wiring.py`, exact private-callsite tests.

- [ ] Move `_global_chat_display_name`, `_apply_console_settings_result`, `_refresh_console_roleplay_projections`, `_drain_console_roleplay_persistence`, `_await_console_roleplay_persistence_task`, `_finish_console_roleplay_persistence_task`, `_console_roleplay_unmount_timeout_seconds`, `_publish_console_roleplay_repair_marker`, `_teardown_console_roleplay_persistence`, `_start_console_roleplay_persistence_drain`, `_dispatch_active_console_roleplay_refresh`, and `_consume_pending_console_roleplay_repair` to `ConsoleSettingsDurabilityController`.
- [ ] Move their three standalone completion/repair helpers: `_consume_console_roleplay_writer_completion`, `_release_console_roleplay_transition_after_writer`, `_consume_console_roleplay_repair_for_current_screen`. App-owned callbacks must still avoid retaining a departed screen/controller. Preserve logger binding and exact statements for later inventory comparison.
- [ ] Move only exclusively owned roleplay state initializers at the same construction phase. Preserve all default values and assignment compatibility. Keep shared screen state behind explicit late-bound ports; never hide dependency lookup behind a new generic object.
- [ ] Retain `_consume_pending_console_roleplay_repair` as a thin screen hook: the app-current-screen helper intentionally discovers it dynamically. Keep public identity/appearance refresh methods, UI sync, lifecycle teardown callers and worker groups on the correct existing owner.
- [ ] Route private internal callers to the owner and update exact test receivers/patch targets without changing assertions. Bind non-framework dependencies by named late-bound callables in `wiring.py`; keep framework services live. Reuse existing ports and remove superseded constructor arguments only after caller census.
- [ ] Run complete affected settings/roleplay/wiring files and scoped static checks. Obtain spec then correctness review before committing this owner change.

## Task 3: Existing settings navigation owner

Files: `chat_screen.py`, `UI/Console_Modules/settings_navigation.py`, `wiring.py`, exact probe tests.

- [x] Move static `_test_console_connection` to `ConsoleSettingsNavigationController`, adjusting its relative import to the same endpoint-probe module; keep exact purpose/identity/result behavior.
- [x] Retarget callsites and private tests, retaining bounded endpoint-probe/failure controls. Test a replaced callable where current wiring permits it.
- [x] Run affected complete probe/navigation files, review, and commit separately from Task 2. Complete four-file group:750 passed; private-delegate architecture:66 passed. Independent spec and correctness reviews pass. Evidence is recorded in the reconciliation report.

## Task 4: Existing session handoff owner

Files: `chat_screen.py`, `UI/Console_Modules/session.py`, `wiring.py`, `Tests/UI/test_console_chat_handoff_resume.py`, `test_console_live_work_handoffs.py`, and other exact caller tests.

- [ ] Move `_consume_pending_chat_handoff` and `_stage_handoff_as_console_live_work` into `ConsoleSessionController`; keep claim/release/acknowledgement ordering, cancellation, sanitization, evidence construction and repair behavior unchanged.
- [ ] Retain composer DOM access as one narrowly named screen hook with identical operations and ordering; wire it explicitly. Do not migrate widgets/IDs/nesting into the controller.
- [ ] Keep late-bound session/service ports and controller-owned consumption state. Character handoff stays on the existing session owner; no sibling back-door through screen attributes.
- [ ] Run complete handoff files and actual mounted staging/send controls. Preserve hit-test and geometry checks at160x45/235x52 where applicable. Review then commit.

## Task 5: Existing message presentation owner

Files: `chat_screen.py`, `UI/Console_Modules/message.py`, `wiring.py`, message/presentation tests.

- [ ] Move `_console_presentation_context` and `_console_message_presentation` into `ConsoleMessageController`. Keep screen transcript-style preference/UI refresh hooks, named current-session/global-display-name ports and active-session fallback unchanged.
- [ ] Retarget exact private receivers; no screen-wide fallback lookup or broad test-helper adaptation. Preserve previously repaired notes owner, Canvas callback identity and citation guards.
- [ ] Run complete message/presentation/wiring files, review and commit.

## Task 6: Combined qualification

- [ ] Re-measure source lines and direct class methods. Census predicts807 removable lines before retained hooks/ports and net15 methods removed; prove actual <=16811/505 without compression or raised limits. Keep the existing ceilings unless a legitimate lower exact pin is required by the guard.
- [ ] Run `.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_controller_wiring.py Tests/UI/test_console_message_controller.py Tests/UI/test_console_settings_failure_diagnostics.py --basetemp=<unique-temp-root> -q --tb=short --show-capture=no` plus complete roleplay/handoff files and all architecture owner/seam gates touched.
- [ ] Compare moved method bodies/statement inventories against baseline, including logger context and module aliases. Regenerate derived diagnostics only after reviewing each actual statement/owner delta; no privacy widening.
- [ ] Record measured counts and complete-file evidence in TASK-31932/reconciliation report. Root reviews/commits/pushes and performs final fresh-dev/review/CI qualification; no administrative merge bypass.

Work only in `.worktrees/pr2427-review-recovery`; use its `.venv/bin/python`. Each pytest invocation has a unique test-owned temporary root. Do not stage other workers' files, change the original checkout/environment, or run a full-repository sweep. If a moved helper genuinely requires a new owner or behavior policy, stop that cluster and report the exact design gap rather than guessing.
