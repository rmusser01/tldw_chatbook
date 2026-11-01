"""Ordinary owners consume the actual public consolidation's same target."""

import json

import pytest

from Tests.Backup_Recovery import mobile_consolidation_fixture
from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import _capture_fixture
from Tests.Backup_Recovery.test_mobile_owning_transactions import (
    _ordinary_imported_owner,
    _small_native_seed,
)


def test_public_target_retains_goal_and_exact_note_delivery_guards(
    tmp_path, monkeypatch
):
    from Tests.network_guard import blocked_attempts
    from Tests.real_profile_guard import take_violations
    from tldw_chatbook.Backup_Recovery import credentials, recovery_service
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        require_mobile_cutover,
    )
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.Chat.console_trace_read_budget import (
        _mobile_budget_pair,
        _TraceReadBudget,
    )
    from tldw_chatbook.DB.automatic_work import AutomaticWorkRefused
    from tldw_chatbook.DB.mobile_consolidation_receipts import _read_owner_receipt
    from tldw_chatbook.runtime_policy.server_credentials import (
        KeyringServerCredentialStore,
    )

    seeded = {}

    def seed(stores, **kwargs):
        value = _small_native_seed(stores, **kwargs, source_case="consumers_privacy")
        seeded["legacy" if kwargs["legacy"] else "canonical"] = value
        return value

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    with _capture_fixture(tmp_path, monkeypatch) as values:
        _, journal, _, stores, preview, _, _ = values

    from Tests.DB.test_mobile_imported_goal_privacy import _accounting
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.profile_paths import effective_config_path

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(preview.target_selector))
    assert effective_config_path() == preview.target_selector
    pending, profiles = bootstrap._records(bootstrap.default_bootstrap_root())
    assert not pending
    assert (
        bootstrap._binding(
            preview.target_selector,
            profiles,
            bootstrap._registry(bootstrap.default_bootstrap_root()),
        )
        is not None
    )

    class EmptyIsolatedKeyring:
        def get_password(self, service, username):
            return None

    credential_store = KeyringServerCredentialStore(
        keyring_backend=EmptyIsolatedKeyring()
    )
    monkeypatch.setattr(credentials, "_credential_store", lambda: credential_store)
    before = mobile_consolidation_fixture.fingerprints(preview.source_dir)
    issued, errors = [], []
    native_init, native_issue = _TraceReadBudget.__init__, recovery_service.issue_code

    def count(actual):
        native_init(actual)
        issued.append(actual)

    def issue(error, **kwargs):
        errors.append((type(error).__name__, str(error)))
        return native_issue(error, **kwargs)

    def pairs():
        return [
            {
                "coupled": _mobile_budget_pair(row._parent, row),
                "aggregate_remaining": row._parent._remaining,
                "native_remaining": row._remaining,
            }
            for row in issued
            if row._parent is not None
        ]

    monkeypatch.setattr(_TraceReadBudget, "__init__", count)
    monkeypatch.setattr(recovery_service, "issue_code", issue)
    service = RecoveryService(journal.root.parent)
    report = {"scope": "same public-produced target; ordinary owner API, not App UI"}
    try:
        preview = service.preview_mobile_consolidation(
            preview.source_selector, preview.target_selector
        )
        assert mobile_consolidation_fixture.fingerprints(preview.source_dir) == before
        operation = service.start_mobile_consolidation(
            preview,
            destination_confirmed=True,
            rollback_password=b"test-private-mobile-rollback",
        )
        state = service.wait(operation)
        report.update(
            operation=operation,
            state=state["state"],
            errors=errors,
            pairs_after_wait=pairs(),
        )
        assert state["state"] == "succeeded", report
        assert not service.pending_operations()
    finally:
        service.close()
        report["pairs_after_service_close"] = pairs()
        report["source_after_service_close"] = (
            mobile_consolidation_fixture.fingerprints(preview.source_dir)
        )
        (tmp_path / "public-consumers.json").write_text(json.dumps(report, indent=2))

    try:
        assert require_mobile_cutover(preview.source_dir, stores) is None
        with _ordinary_imported_owner(stores, "db.agent_runs") as runs:
            with runs.transaction() as connection:
                _, children = _read_owner_receipt(
                    connection,
                    operation,
                    preview.source_fingerprint,
                    "runs",
                    _TraceReadBudget(),
                )
            mapped = {
                (row["data"]["domain"], row["data"]["original"]): row["data"]["mapped"]
                for row in children
                if row["kind"] == "identity"
                and type(row["data"]["original"]) in (str, int)
            }
            original = seeded["legacy"]
            old_run = mapped["run", original["run"]]
            definition = mapped["agent_definition", original["definition"]]
            assert runs.get_agent_definition(definition)["enabled"] == 1
            enabled = {
                row["id"] for row in runs.list_agent_definitions(enabled_only=True)
            }
            assert (
                definition not in enabled
                and seeded["canonical"]["definition"] in enabled
            )
            goal = runs.goal_runs.get(mapped["goal", original["goal"]])
            assert goal.request.objective == "Repair fixture"
            with pytest.raises(AutomaticWorkRefused, match="import_review_required"):
                runs.goal_runs.set_provisioning(goal, status="ready")
            assert runs.goal_runs.get(goal.id) == goal
            conversation = runs.get_run(old_run)["conversation_id"]
            included = [
                row["id"] for row in runs.pending_notes_for_conversation(conversation)
            ]
            assert included == [mapped["change_note", original["note"]]]
            fresh = runs.create_run(conversation_id=conversation, agent_kind="primary")
            later = runs.add_change_note(
                run_id=fresh,
                root=str(tmp_path / "unused-project"),
                path="file.py",
                hunk_index=0,
                hunk_header="@@ -1 +1 @@",
                hunk_excerpt="Later ordinary annotation",
                note="Not included in prepared send",
                snapshot_id=43,
            )
            before_notes = runs.notes_for_run(old_run)
            with pytest.raises(AutomaticWorkRefused, match="import_review_required"):
                runs.mark_notes_delivered(included, delivered_by_run_id=old_run)
            assert runs.notes_for_run(old_run) == before_notes
            assert (
                runs.mark_notes_delivered(included, delivered_by_run_id=fresh)
                == included
            )
            assert [
                row["id"] for row in runs.pending_notes_for_conversation(conversation)
            ] == [later]
            assert runs.notes_for_run(old_run)[0]["delivered_by_run_id"] == fresh
            runs.set_status(fresh, "done", result="ordinary manual owner API")
            accounting = _accounting(runs)
            with runs.transaction() as connection:
                uncertain_before = tuple(
                    tuple(row)
                    for row in connection.execute(
                        "SELECT id,chain_id,state,actual_amount FROM automatic_work_reservations WHERE chain_id=?",
                        (goal.chain_id,),
                    )
                )
            assert len(uncertain_before) == 1 and uncertain_before[0][2:] == (
                "uncertain",
                None,
            )
            removed = runs.goal_runs.remove_payloads(goal.id)
            assert removed.status == "removed" and removed.request is None
            with runs.transaction() as connection:
                assert (
                    connection.execute(
                        "SELECT request_json FROM goal_runs WHERE id=?", (goal.id,)
                    ).fetchone()[0]
                    == ""
                )
            assert not removed.reports and not removed.checkpoints
            assert _accounting(runs) == accounting
            assert runs.goal_runs.remove_payloads(goal.id) == removed
            with pytest.raises(AutomaticWorkRefused, match="import_review_required"):
                runs.goal_runs.set_provisioning(removed, status="ready")
            durable_accounting = {
                key: value
                for key, value in accounting.items()
                if key != "automatic_work_runtime_owner"
            }
        assert require_mobile_cutover(preview.source_dir, stores) is None
        with _ordinary_imported_owner(stores, "db.agent_runs") as reopened:
            assert reopened.goal_runs.get(goal.id) == removed
            with reopened.transaction() as connection:
                assert (
                    connection.execute(
                        "SELECT request_json FROM goal_runs WHERE id=?", (goal.id,)
                    ).fetchone()[0]
                    == ""
                )
            assert {
                key: value
                for key, value in _accounting(reopened).items()
                if key != "automatic_work_runtime_owner"
            } == durable_accounting
            with reopened.transaction() as connection:
                assert (
                    tuple(
                        tuple(row)
                        for row in connection.execute(
                            "SELECT id,chain_id,state,actual_amount FROM automatic_work_reservations WHERE chain_id=?",
                            (goal.chain_id,),
                        )
                    )
                    == uncertain_before
                )
            assert reopened.goal_runs.remove_payloads(goal.id) == removed
            with pytest.raises(AutomaticWorkRefused, match="import_review_required"):
                reopened.goal_runs.set_provisioning(removed, status="ready")
        assert require_mobile_cutover(preview.source_dir, stores) is None
        report["goal_privacy"] = {
            "removed": True,
            "uncertainty_retained": True,
            "same_lease_accounting_equal": True,
            "reopened_durable_accounting_equal": True,
        }
        report["ordinary_consumers"] = True
    finally:
        report.update(
            source_before=before,
            source_after=mobile_consolidation_fixture.fingerprints(preview.source_dir),
            pairs_after_ordinary_close=pairs(),
            profile_guard=take_violations(),
            network_guard=blocked_attempts(),
        )
        (tmp_path / "public-consumers.json").write_text(json.dumps(report, indent=2))
    assert report["source_before"] == report["source_after"]
    assert not report["profile_guard"] and not report["network_guard"]
    assert len(report["pairs_after_ordinary_close"]) == 1
    assert report["pairs_after_ordinary_close"][0]["coupled"]


def test_public_target_ordinary_imported_history_and_trace(tmp_path, monkeypatch):
    from Tests.network_guard import blocked_attempts
    from Tests.real_profile_guard import take_violations
    from tldw_chatbook.Backup_Recovery import credentials, recovery_service
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        require_mobile_cutover,
    )
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
    from tldw_chatbook.Chat.console_trace_read_budget import (
        _mobile_budget_pair,
        _TraceReadBudget,
    )
    from tldw_chatbook.DB.mobile_consolidation_receipts import _read_owner_receipt
    from tldw_chatbook.runtime_policy.server_credentials import (
        KeyringServerCredentialStore,
    )

    seeded = {}

    def seed(stores, **kwargs):
        value = _small_native_seed(stores, **kwargs, source_case="history_consumers")
        seeded["legacy" if kwargs["legacy"] else "canonical"] = value
        return value

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    with _capture_fixture(tmp_path, monkeypatch) as values:
        _, journal, _, stores, preview, _, _ = values
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.profile_paths import effective_config_path

    # Select the reviewed profile before ordinary acquisition; binding already exists.
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(preview.target_selector))
    assert effective_config_path() == preview.target_selector
    pending, profiles = bootstrap._records(bootstrap.default_bootstrap_root())
    assert not pending
    assert (
        bootstrap._binding(
            preview.target_selector,
            profiles,
            bootstrap._registry(bootstrap.default_bootstrap_root()),
        )
        is not None
    )

    class EmptyIsolatedKeyring:
        def get_password(self, service, username):
            return None

    store = KeyringServerCredentialStore(keyring_backend=EmptyIsolatedKeyring())
    monkeypatch.setattr(credentials, "_credential_store", lambda: store)
    before = mobile_consolidation_fixture.fingerprints(preview.source_dir)
    budgets, errors = [], []
    original_init, original_issue = (
        _TraceReadBudget.__init__,
        recovery_service.issue_code,
    )

    def counted(actual):
        original_init(actual)
        budgets.append(actual)

    def issue(error, **kwargs):
        errors.append((type(error).__name__, str(error)))
        return original_issue(error, **kwargs)

    monkeypatch.setattr(_TraceReadBudget, "__init__", counted)
    monkeypatch.setattr(recovery_service, "issue_code", issue)
    service = RecoveryService(journal.root.parent)
    report = {
        "scope": "ordinary same-public-target RunLog/command/trace APIs, not App viewer"
    }
    try:
        try:
            preview = service.preview_mobile_consolidation(
                preview.source_selector, preview.target_selector
            )
            assert (
                mobile_consolidation_fixture.fingerprints(preview.source_dir) == before
            )
            operation = service.start_mobile_consolidation(
                preview,
                destination_confirmed=True,
                rollback_password=b"test-private-mobile-rollback",
            )
            state = service.wait(operation)
            report.update(operation=operation, state=state["state"], errors=errors)
            assert state["state"] == "succeeded", report
            assert not service.pending_operations()
        finally:
            service.close()
        assert require_mobile_cutover(preview.source_dir, stores) is None
        with _ordinary_imported_owner(stores, "db.agent_runs") as runs:
            with runs.transaction() as connection:
                _, children = _read_owner_receipt(
                    connection,
                    operation,
                    preview.source_fingerprint,
                    "runs",
                    _TraceReadBudget(),
                )
            mapping = {
                (row["data"]["domain"], row["data"]["original"]): row["data"]["mapped"]
                for row in children
                if row["kind"] == "identity"
                and type(row["data"]["original"]) in (str, int)
            }
            original = seeded["legacy"]
            mapped = mapping["run", original["run"]]
            page = runs.read_imported_run_log_page(
                mapped, operation, preview.source_fingerprint
            )
            assert page.next_cursor is None
            assert [part.record.run_id for part in page.slices] == [original["run"]]
            assert (
                "".join(part.record.content for part in page.slices)
                == "Original native frame Ω"
            )
            history = runs.command_receipts.list_imported_history(
                operation, preview.source_fingerprint
            )
            assert len(history.items) == 1 and history.next_cursor is None
            assert history.items[0].original_fields == original["command_fields"]
            assert history.items[0].run_associations == ((original["run"], mapped),)
            assert runs.command_receipts.get("retained-command") is None
            answer = runs.get_run(mapped)["assistant_message_id"]
        with _ordinary_imported_owner(stores, "db.chachanotes.primary") as chat:
            current = ConsoleTraceNativeReader(chat).read_calls(answer)
            expected = original["trace_original"]
            assert len(current) == len(expected) == 1
            actual, old = current[0], expected[0]
            assert (actual.verified, actual.uncertainty_codes) == (
                old.verified,
                old.uncertainty_codes,
            )
            for field in (
                "request",
                "response",
                "status",
                "omitted_keys",
                "provider",
                "model",
            ):
                assert getattr(actual.capture, field) == getattr(old.capture, field)
            assert actual.capture.run_tag == mapped
            report["trace"] = {
                "verified": actual.verified,
                "uncertainty": actual.uncertainty_codes,
                "omitted_keys": actual.capture.omitted_keys,
            }
        assert require_mobile_cutover(preview.source_dir, stores) is None
        report["ordinary_history"] = True
    finally:
        report.update(
            source_before=before,
            source_after=mobile_consolidation_fixture.fingerprints(preview.source_dir),
            profile_guard=take_violations(),
            network_guard=blocked_attempts(),
            pairs_after_close=[
                {
                    "coupled": _mobile_budget_pair(row._parent, row),
                    "aggregate_remaining": row._parent._remaining,
                    "native_remaining": row._remaining,
                }
                for row in budgets
                if row._parent is not None
            ],
        )
        (tmp_path / "public-history-consumers.json").write_text(
            json.dumps(report, indent=2)
        )
    assert report["source_before"] == report["source_after"]
    assert not report["profile_guard"] and not report["network_guard"]
    assert (
        len(report["pairs_after_close"]) == 1
        and report["pairs_after_close"][0]["coupled"]
    )


@pytest.mark.asyncio
async def test_public_target_native_controller_discards_imported_checkpoint(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from Tests.DB.test_mobile_imported_goal_privacy import _accounting
    from Tests.network_guard import blocked_attempts
    from Tests.real_profile_guard import take_violations
    from tldw_chatbook.Backup_Recovery import bootstrap, credentials, recovery_service
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        require_mobile_cutover,
    )
    from tldw_chatbook.Backup_Recovery.profile_paths import effective_config_path
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleDispatchRecoveryActionId,
        ConsoleDispatchRecoveryKind,
    )
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_conversation_hydration import (
        hydrate_console_session,
    )
    from tldw_chatbook.Chat.console_dispatch_repository import ConsoleDispatchRepository
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.Chat.console_trace_read_budget import (
        _mobile_budget_pair,
        _TraceReadBudget,
    )
    from tldw_chatbook.DB.mobile_consolidation_receipts import _read_owner_receipt
    from tldw_chatbook.runtime_policy.server_credentials import (
        KeyringServerCredentialStore,
    )

    seeded = {}

    def seed(stores, **kwargs):
        value = _small_native_seed(stores, **kwargs, source_case="passive")
        seeded["legacy" if kwargs["legacy"] else "canonical"] = value
        return value

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    with _capture_fixture(tmp_path, monkeypatch) as values:
        _, journal, _, stores, preview, _, _ = values
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(preview.target_selector))
    assert effective_config_path() == preview.target_selector
    pending, profiles = bootstrap._records(bootstrap.default_bootstrap_root())
    assert not pending
    assert (
        bootstrap._binding(
            preview.target_selector,
            profiles,
            bootstrap._registry(bootstrap.default_bootstrap_root()),
        )
        is not None
    )

    class EmptyIsolatedKeyring:
        def get_password(self, service, username):
            return None

    credential_store = KeyringServerCredentialStore(
        keyring_backend=EmptyIsolatedKeyring()
    )
    monkeypatch.setattr(credentials, "_credential_store", lambda: credential_store)
    before = mobile_consolidation_fixture.fingerprints(preview.source_dir)
    budgets, errors = [], []
    native_init, native_issue = _TraceReadBudget.__init__, recovery_service.issue_code

    def counted(actual):
        native_init(actual)
        budgets.append(actual)

    def issue(error, **kwargs):
        errors.append((type(error).__name__, str(error)))
        return native_issue(error, **kwargs)

    monkeypatch.setattr(_TraceReadBudget, "__init__", counted)
    monkeypatch.setattr(recovery_service, "issue_code", issue)
    report = {
        "scope": "same public-produced target native hydration/controller Discard; not full App/manual provider"
    }
    service = RecoveryService(journal.root.parent)
    try:
        try:
            preview = service.preview_mobile_consolidation(
                preview.source_selector, preview.target_selector
            )
            assert (
                mobile_consolidation_fixture.fingerprints(preview.source_dir) == before
            )
            operation = service.start_mobile_consolidation(
                preview,
                destination_confirmed=True,
                rollback_password=b"test-private-mobile-rollback",
            )
            state = service.wait(operation)
            report.update(operation=operation, state=state["state"], errors=errors)
            assert state["state"] == "succeeded", report
            assert not service.pending_operations()
        finally:
            service.close()
        assert require_mobile_cutover(preview.source_dir, stores) is None
        with _ordinary_imported_owner(stores, "db.agent_runs") as runs:
            durable_before = {
                key: value
                for key, value in _accounting(runs).items()
                if key != "automatic_work_runtime_owner"
            }
        with _ordinary_imported_owner(stores, "db.chachanotes.primary") as chat:
            with chat.transaction() as cursor:
                receipt_before = _read_owner_receipt(
                    cursor.connection,
                    operation,
                    preview.source_fingerprint,
                    "chat",
                    _TraceReadBudget(),
                )
            mapping = {
                (row["data"]["domain"], row["data"]["original"]): row["data"]["mapped"]
                for row in receipt_before[1]
                if row["kind"] == "identity"
                and type(row["data"]["original"]) in (str, int)
            }
            original = seeded["legacy"]
            conversation = mapping["conversation", original["conversation"]]
            assistant = mapping["message", original["dispatch"]]
            user = mapping["message", original["dispatch_user"]]
            user_before = chat.get_message_by_id(user)
            store = ConsoleChatStore(persistence=ChatPersistenceService(chat))
            session = await hydrate_console_session(
                app=SimpleNamespace(chachanotes_db=chat),
                store=store,
                conversation_id=conversation,
                tree=ChatConversationService(chat).get_conversation_tree(
                    conversation, root_limit=100, depth_cap=100
                ),
                settings=ConsoleSessionSettings(
                    provider="llama_cpp", model="test-model"
                ),
            )

            class NoProvider:
                def __init__(self):
                    self.calls = []

                async def resolve_for_send(self, *args, **kwargs):
                    self.calls.append("resolve")
                    raise AssertionError("Imported recovery must not resolve provider")

                async def stream_chat(self, *args, **kwargs):
                    self.calls.append("stream")
                    raise AssertionError("Imported recovery must not dispatch provider")
                    yield "unreachable"

            gateway = NoProvider()
            controller = ConsoleChatController(
                store=store, provider_gateway=gateway, agent_runtime_enabled=False
            )
            try:
                recovery = store.dispatch_recovery_for_session(session.id)
                assert recovery.kind is ConsoleDispatchRecoveryKind.QUARANTINED
                assert recovery.error_code == "imported_history"
                assert recovery.assistant_message_id == assistant
                assert [(row.action_id, row.enabled) for row in recovery.actions] == [
                    (ConsoleDispatchRecoveryActionId.DISCARD, True)
                ]
                assert recovery.queue_entry_id is recovery.preparation_id is None
                assert not controller._hydrate_dispatch_recovery_queue(
                    session.id, force=True
                )
                assert not (
                    await controller.retry_dispatch_recovery(session.id)
                ).accepted
                assert (await controller.discard_dispatch_recovery(session.id)).accepted
                assert not gateway.calls
                assert chat.get_message_by_id(user) == user_before
                assert (
                    chat.get_message_by_id(assistant)["assistant_generation_state"]
                    == "discarded"
                )
                assert store.dispatch_recovery_for_session(session.id) is None
                with chat.transaction() as cursor:
                    assert (
                        cursor.execute(
                            "SELECT count(*) FROM console_dispatch_checkpoints WHERE assistant_message_id=?",
                            (assistant,),
                        ).fetchone()[0]
                        == 0
                    )
                    assert (
                        _read_owner_receipt(
                            cursor.connection,
                            operation,
                            preview.source_fingerprint,
                            "chat",
                            _TraceReadBudget(),
                        )
                        == receipt_before
                    )
            finally:
                await controller.shutdown()
        assert require_mobile_cutover(preview.source_dir, stores) is None
        with _ordinary_imported_owner(stores, "db.chachanotes.primary") as chat:
            assert chat.get_message_by_id(user) == user_before
            assert (
                chat.get_message_by_id(assistant)["assistant_generation_state"]
                == "discarded"
            )
            assert (
                ConsoleDispatchRepository(chat).reconcile_for_session(conversation)
                is None
            )
            with chat.transaction() as cursor:
                assert (
                    _read_owner_receipt(
                        cursor.connection,
                        operation,
                        preview.source_fingerprint,
                        "chat",
                        _TraceReadBudget(),
                    )
                    == receipt_before
                )
        with _ordinary_imported_owner(stores, "db.agent_runs") as runs:
            assert {
                key: value
                for key, value in _accounting(runs).items()
                if key != "automatic_work_runtime_owner"
            } == durable_before
        assert require_mobile_cutover(preview.source_dir, stores) is None
        report["discard"] = {
            "provider_calls": gateway.calls,
            "mapped_user_retained": True,
            "checkpoint_deleted": True,
            "receipt_unchanged": True,
            "holds_accounting_unchanged": True,
        }
    finally:
        report.update(
            source_before=before,
            source_after=mobile_consolidation_fixture.fingerprints(preview.source_dir),
            profile_guard=take_violations(),
            network_guard=blocked_attempts(),
            pairs_after_close=[
                {
                    "coupled": _mobile_budget_pair(row._parent, row),
                    "aggregate_remaining": row._parent._remaining,
                    "native_remaining": row._remaining,
                }
                for row in budgets
                if row._parent is not None
            ],
        )
        (tmp_path / "public-discard-consumers.json").write_text(
            json.dumps(report, indent=2)
        )
    assert report["source_before"] == report["source_after"]
    assert not report["profile_guard"] and not report["network_guard"]
    assert (
        len(report["pairs_after_close"]) == 1
        and report["pairs_after_close"][0]["coupled"]
    )
