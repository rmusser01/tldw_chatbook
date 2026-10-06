"""A real first persistence cannot hide live decisions behind cold config."""

import asyncio
import inspect
import json

import pytest
from textual.widgets import Static

from Tests.private_profile import private_profile_test
from Tests.UI import test_console_pending_interrupt_projection as original
from Tests.UI.app_factory import drain_active_service_patches, drain_created_dirs
from Tests.UI.test_console_pending_facts_cold_readiness import (
    _CheckedNavigationFactory,
)
from Tests.Performance._first_persistence_main_getter import (
    AttributedOriginalColdRead as _OriginalColdRead,
)
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import raw_participants, storage_admission
from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
from tldw_chatbook.UI.Screens import chat_screen
from tldw_chatbook.Widgets.Console.console_run_inspector import ConsoleRunInspector
from tldw_chatbook.Widgets.Console.console_send_authority_summary import (
    ConsoleSendAuthoritySummary,
)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@private_profile_test
async def test_first_persisted_conversation_keeps_pending_count_during_cold_config(
    request, tmp_path
):
    """Count2→1 after original first-persistence publication with the read held."""
    factory = _CheckedNavigationFactory()
    app = factory.build(tmp_path)
    gate = None
    workers, results, violations = [], [], []
    facts = {}
    projection = None
    controller = None
    try:
        async with app.run_test(size=(160, 48)) as pilot:
            try:
                await original._wait(pilot, lambda: app._initial_screen_pushed)
                console = app.screen
                assert type(console) is chat_screen.ChatScreen
                await original._wait(
                    pilot, lambda: bool(list(console.query("#console-native-composer")))
                )
                controller = console._ensure_console_chat_controller()
                store = console._console_chat_store
                # The ordinary new-chat API supplies a genuine unpersisted
                # Session; do not rewrite a seeded Session or fake a base key.
                session = controller.new_session(title="Pending first persistence")
                session_id = session.id
                assert store.active_session_id == session_id
                assert session.persisted_conversation_id is None
                # A real durable row is staged before the artificial hold. The
                # original store publisher performs its original first-binding
                # effect later; no session/base/cache/source flag is assigned.
                persistence = store.persistence
                assert persistence is not None
                conversation_id = persistence.create_conversation(
                    conversation_title="Pending first persistence control",
                    runtime_backend="local",
                    scope_type="global",
                )
                assert type(conversation_id) is str and conversation_id  # noqa: E721 -- real durable ID
                database = persistence.db
                assert database.get_conversation_by_id(conversation_id) is not None
                assert session.persisted_conversation_id is None
                for _ in range(2):
                    worker, result = original._arm(
                        controller, session_id, call=original._risk_row()
                    )
                    workers.append(worker)
                    results.append(result)
                await original._wait(
                    pilot, lambda: controller.pending_round_count(session_id) == 2
                )
                await original._wait(
                    pilot,
                    lambda: console.query_one("#chat-approval-card").display
                    and console.query_one("#chat-approval-card")._batch_round_id
                    in controller._pending_approval_rounds,
                )
                projection = spend.ConsoleReadinessConfigProjection.for_screen(console)
                assert await projection.warm()
                console._sync_console_rail_and_controls()
                await pilot.pause()
                inspector = console.query_one(
                    "#console-run-inspector-state", ConsoleRunInspector
                )
                authority = console.query_one(
                    "#console-send-authority-summary", ConsoleSendAuthoritySummary
                )
                assert inspector.state.pending_approval_count == 2
                base = console._console_pending_display_base
                before = chat_screen._console_pending_display_owner(console)
                assert base is not None and before is not None
                assert chat_screen._same_console_pending_owner(
                    base[0], before, prior_base=True
                )
                assert before[5] is session and before[11] is None
                gate = _OriginalColdRead(projection)
                publisher = inspect.getattr_static(
                    type(store), "publish_first_persisted_conversation"
                )
                creator = inspect.getattr_static(
                    type(persistence), "create_conversation"
                )
                gate._pin(publisher)
                gate._pin(creator)
                gate.slots.extend(
                    (
                        (
                            type(store),
                            "publish_first_persisted_conversation",
                            publisher,
                        ),
                        (type(persistence), "create_conversation", creator),
                    )
                )
                # The additive original publisher/creator pins introduce two
                # modules after the cold-read observer's initial file snapshot.
                # Bind every actual module file before its existing strict check.
                gate.sources = {
                    record[1]: record[-1] for record in gate.modules.values()
                }
                gate.module_files = {
                    name: record[0].__file__ for name, record in gate.modules.items()
                }
                gate.start()
                previous_config = config.current_config_identity()
                assert config.save_setting_to_cli_config(
                    "splash_screen", "enabled", False
                )
                gate.expected_source = config.current_config_identity()
                assert gate.expected_source != previous_config
                assert console._sync_console_rail_and_controls() is False
                await original._wait(pilot, gate.entered.is_set)
                assert projection.pending and gate.operation in raw_participants._states
                assert all(
                    lease in storage_admission._live_leases for lease in gate.leases
                )
                current = store.publish_first_persisted_conversation(
                    session_id, conversation_id
                )
                assert current is session and store.active_session_id == session_id
                after = chat_screen._console_pending_display_owner(console)
                assert after is not None
                assert any(base[0][index] != after[index] for index in (8, 11))
                assert before[11] is None and after[11] == conversation_id
                assert before[12] == after[12]
                assert all(
                    before[index] is after[index]
                    for index in (*range(7), 14, 15, 16, 17, 18, 20)
                )
                assert before[7] == after[7] and before[8][1] == after[8][1]
                assert before[9:11] == after[9:11] and before[12:14] == after[12:14]
                assert before[19] == after[19]
                assert not chat_screen._same_console_pending_owner(
                    base[0], after, prior_base=True
                )
                round_id = console.query_one("#chat-approval-card")._batch_round_id
                assert round_id in controller._pending_approval_rounds
                controller.resolve_pending_approval(
                    {"builtin__write_file": "deny"}, round_id=round_id
                )
                await original._wait(
                    pilot, lambda: controller.pending_round_count(session_id) == 1
                )
                assert controller.pending_round_count(session_id) == 1
                assert console._console_pending_approval_count() == 1
                assert console._sync_console_rail_and_controls() is False
                await pilot.pause()
                await original._wait(
                    pilot,
                    lambda: all(
                        any(row.is_mounted for row in inspector.query(f"#{row_id}"))
                        for row_id in (
                            "console-inspector-live-work",
                            "console-inspector-approvals",
                        )
                    ),
                )
                assert projection.pending and gate.operation in raw_participants._states
                assert all(
                    lease in storage_admission._live_leases for lease in gate.leases
                )
                facts.update(
                    original_durable_row_exists=True,
                    original_first_binding_same_Session_incarnation_host_and_all_other_fences=True,
                    actual_current_count=1,
                    held_original_raw_lease_count=len(gate.leases),
                    original_full_return_still_deferred=True,
                )
                rendered = "\n".join(
                    str(row.render()) for row in inspector.query(Static)
                )
                if inspector.state.pending_approval_count != 1:
                    violations.append("current count1 hidden after first persistence")
                if "Approvals: 1 pending" not in rendered:
                    violations.append(
                        "current approval row not painted after first persistence"
                    )
                if not inspector.state.has_pending_approval:
                    violations.append(
                        "current Review state unavailable after first persistence"
                    )
                if authority.last_state.pending_approval_count != 1:
                    violations.append(
                        "pinned current approval count stale after first persistence"
                    )
                assert controller.pending_round_count(session_id) == 1
                gate.release.set()
                await asyncio.wait_for(projection._settled.wait(), 10)
                await original._wait(
                    pilot,
                    lambda: console.query_one("#chat-approval-card").display
                    and console.query_one("#chat-approval-card")._batch_round_id
                    in controller._pending_approval_rounds,
                )
                remaining = console.query_one("#chat-approval-card")._batch_round_id
                controller.resolve_pending_approval(
                    {"builtin__write_file": "deny"}, round_id=remaining
                )
                for worker in workers:
                    await original._finish_worker(pilot, worker)
                assert all(
                    result["decisions"] == {"builtin__write_file": "deny"}
                    for result in results
                )
                assert controller.pending_round_count(session_id) == 0
                console._sync_console_rail_and_controls()
                await pilot.pause()
                assert gate.operation not in raw_participants._states
                assert all(
                    lease not in storage_admission._live_leases for lease in gate.leases
                )
                facts["all_original_denials_and_worker_native_retirement"] = True
            finally:
                if gate is not None:
                    gate.release.set()
                    if projection.pending:
                        await asyncio.wait_for(projection._settled.wait(), 10)
                if workers:
                    await original._stop_workers(controller, workers, pilot)
    finally:
        if gate is not None:
            gate.release.set()
        try:
            drain_created_dirs()
            drain_active_service_patches()
        finally:
            if gate is not None:
                gate.stop()
            (tmp_path / "pending-first-persistence.json").write_text(
                json.dumps(
                    {
                        "facts": facts,
                        "violations": violations,
                        "fixture": factory.facts,
                        "fixture_source_current": factory.current(),
                        "rows": gate.rows if gate is not None else [],
                        "invalid": gate.invalid
                        if gate is not None
                        else ["gate_not_reached"],
                        "monitor_retired": gate is not None and gate.retired,
                        "retirement": gate.retirement if gate is not None else {},
                        "held_raw_retired": gate is not None
                        and gate.operation not in raw_participants._states,
                        "held_leases_retired": gate is not None
                        and all(
                            lease not in storage_admission._live_leases
                            for lease in gate.leases
                        ),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    assert gate is not None and gate.retired and not gate.active
    assert not gate.invalid, gate.invalid
    assert facts["all_original_denials_and_worker_native_retirement"]
    assert not violations, violations
