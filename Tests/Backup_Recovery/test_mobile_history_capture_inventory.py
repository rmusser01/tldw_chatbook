"""R177 inventory routing mechanics; native source authentication is tested separately."""

from dataclasses import replace
from threading import Event

import pytest

from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
from Tests.private_profile import private_profile_test
from tldw_chatbook.Backup_Recovery import replacement
from tldw_chatbook.Backup_Recovery.archive_models import SealedArchive
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.models import StorageItem
from tldw_chatbook.Backup_Recovery.restore_plan import LocalSnapshotSource
from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget


def test_capture_scope_yields_exact_native_checked_pair(tmp_path, monkeypatch):
    with replacement_case(tmp_path, monkeypatch) as case:
        _, plan, journal, session, _, _ = case
        stage = tmp_path / "native-scope-stage"
        stage.mkdir(mode=0o700)
        native = replacement._checked_originals
        pairs = []

        def checked(*args, **kwargs):
            pair = native(*args, **kwargs)
            pairs.append(pair)
            return pair

        monkeypatch.setattr(replacement, "_checked_originals", checked)
        limits = ArchiveLimits()
        with session._replacement_capture_scope(
            plan, journal, stage, limits=limits, byte_budget=limits.expanded_bytes
        ) as actual:
            assert len(pairs) == 1
            assert actual is pairs[0]
        assert all(
            not scope.active and not scope.resources for scope in session._scopes
        )


@pytest.mark.parametrize(
    "route",
    [
        "history",
        "history_descriptor",
        "history_config_drift",
        "package",
        "mixed",
        "not_preserved",
        "not_safety",
        "not_directory",
        "ordinary",
    ],
)
def test_capture_consumes_only_the_scheduled_native_inventory(
    tmp_path, monkeypatch, route
):
    """Controlled routing, with real ordinary inventory/capture; no mobile-proof claim."""

    class AtManifest(Exception):
        pass

    with replacement_case(tmp_path, monkeypatch) as case:
        candidate, original_plan, journal, session, _, selector = case
        fault = route if route.startswith("history_") else None
        route = "history" if fault else route
        root = tmp_path / "routing-history"
        history = StorageItem(
            owner="agents.history",
            logical_id="routing:history",
            path=root,
            status="included_directory",
            dependencies=(),
        )
        package = replace(
            history,
            logical_id="routing:package",
            owner="persona.visual_identity_builtin",
        )
        items = (
            ()
            if route == "ordinary"
            else (package,)
            if route == "package"
            else (history, package)
            if route == "mixed"
            else (history,)
        )
        if route == "not_directory":
            items = (replace(history, status="included"),)
        plan = replace(
            original_plan,
            target=replace(
                original_plan.target, items=(*original_plan.target.items, *items)
            ),
            preserve=original_plan.preserve
            + (
                ()
                if route == "not_preserved"
                else tuple((x.logical_id, x.path) for x in items)
            ),
            safety_scope=original_plan.safety_scope
            + (() if route == "not_safety" else tuple(x.logical_id for x in items)),
            local_snapshot=LocalSnapshotSource(
                journal.root.parent, journal.operation_id, "0" * 64
            ),
        )
        native = replacement._checked_originals
        pairs, manifested, attempts, workspace = [], [], [], []
        budget = _TraceReadBudget()
        archive = SealedArchive(tmp_path / "not-an-authority", "0" * 64, b"")
        options = (
            {}
            if route == "ordinary"
            else {
                "_source_archive": archive,
                "_source_read_budget": budget,
                "_source_journal": journal,
            }
        )
        expected_scope = route in {"history", "mixed"}

        def checked(actual_plan, actual_journal, actual_session, **kwargs):
            assert (
                actual_plan is plan
                and actual_journal is journal
                and actual_session is session
            )
            if route != "ordinary":
                assert kwargs["_source_read_budget"] is budget
                assert kwargs["_source_archive"] is archive
                assert kwargs["_source_journal"] is journal
            # The real ordinary fixture supplies local inventory. This wrapper
            # deliberately does not qualify a fabricated mobile source proof.
            attempts.append(actual_plan)
            prepared, inventory = native(original_plan, journal, session)
            if fault == "history_descriptor":
                prepared = prepared.model_copy(
                    update={
                        "publication": prepared.publication.model_copy(
                            update={
                                "descriptor": prepared.publication.descriptor.model_copy(
                                    update={"path": str(candidate / "wrong.json")}
                                )
                            }
                        )
                    }
                )
            label = "scope" if expected_scope or pairs else "entry"
            pair = prepared, replace(inventory, scope_digest=label)
            pairs.append(pair)
            return pair

        native_descriptor = replacement._descriptor

        def descriptor(actual_candidate, actual_plan):
            assert actual_candidate == candidate and actual_plan is plan
            return native_descriptor(candidate, original_plan)

        def manifest(inventory, *args, **kwargs):
            manifested.append(inventory)
            raise AtManifest

        monkeypatch.setattr(replacement, "_checked_originals", checked)
        monkeypatch.setattr(replacement, "_descriptor", descriptor)
        monkeypatch.setattr(replacement, "_manifest_for", manifest)
        native_create = replacement.create_private_directory

        def create(path):
            result = native_create(path)
            workspace.append(path)
            if fault == "history_config_drift" and path.name == "routing-work":
                selector.write_text(
                    selector.read_text() + "\n# changed after Prepared\n"
                )
            return result

        monkeypatch.setattr(replacement, "create_private_directory", create)
        if fault:
            from tldw_chatbook.Backup_Recovery import storage_admission as storage

            def no_payload(*args, **kwargs):
                pytest.fail("refused history source reached payload copy")

            monkeypatch.setattr(storage, "copy_capture_file", no_payload)
        error = ValueError if fault else AtManifest
        reason = (
            "^candidate_receipt_changed$"
            if fault == "history_descriptor"
            else "^target_changed$"
            if fault == "history_config_drift"
            else None
        )
        with pytest.raises(error, match=reason):
            replacement.capture_verify_rollback(
                candidate,
                plan,
                journal,
                tmp_path / "rollback.age",
                session=session,
                password=b"isolated-routing-test",
                work_root=tmp_path / "routing-work",
                cancel=Event(),
                acknowledged_credential_issues=("credential_format_unreadable",),
                **options,
            )
        if fault:
            assert len(attempts) == 1 and not manifested
            assert len(workspace) == 3 and all(path.is_dir() for path in workspace)
            assert all(
                not scope.active and not scope.resources for scope in session._scopes
            )
            return
        assert len(pairs) == (1 if expected_scope else 2)
        assert len(manifested) == 1
        assert manifested[0].scope_digest == ("scope" if expected_scope else "entry")
        assert all(
            not scope.active and not scope.resources for scope in session._scopes
        )
        assert (
            budget._remaining == 16 * 1024**2
            if route == "ordinary"
            else budget._remaining < 16 * 1024**2
        )


@pytest.mark.parametrize(
    "fault",
    [
        *(f"partial_{mask}" for mask in range(1, 7)),
        "archive",
        "budget",
        "journal",
        "session",
        "copied_session",
        "closed_session",
        "cancel_type",
        "cancelled",
        "plan",
        "mode",
        "target",
        "snapshot",
        "incomplete",
        "exhausted",
    ],
)
def test_capture_route_refuses_before_workspace_or_journal(
    tmp_path, monkeypatch, fault
):
    import copy

    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.limits import _TraceReadLimit

    with replacement_case(tmp_path, monkeypatch) as case:
        candidate, original, journal, session, _, _ = case
        history = StorageItem(
            "agents.history", "history", tmp_path / "history", "included_directory", ()
        )
        plan = replace(
            original,
            target=replace(original.target, items=(*original.target.items, history)),
            local_snapshot=LocalSnapshotSource(
                journal.root.parent, journal.operation_id, "0" * 64
            ),
            preserve=(*original.preserve, (history.logical_id, history.path)),
            safety_scope=(*original.safety_scope, history.logical_id),
        )
        budget, cancel = _TraceReadBudget(), Event()
        options = {
            "_source_archive": SealedArchive(tmp_path / "unread", "0" * 64, b""),
            "_source_read_budget": budget,
            "_source_journal": journal,
        }
        selected_session = session
        expected, reason = ValueError, "^mobile_safety_source_required$"
        if fault.startswith("partial_"):
            mask = int(fault.rsplit("_", 1)[1])
            options = {
                name: value if mask & (1 << bit) else None
                for bit, (name, value) in enumerate(options.items())
            }
        elif fault in {"archive", "budget", "journal"}:
            name = {
                "archive": "_source_archive",
                "budget": "_source_read_budget",
                "journal": "_source_journal",
            }[fault]
            options[name] = object()
        elif fault == "session":
            selected_session = object()
        elif fault in {"copied_session", "closed_session"}:
            selected_session = copy.copy(session)
            if fault == "closed_session":
                selected_session._active = False
            expected, reason = RecoveryRequired, "^maintenance_session_inactive$"
        elif fault == "cancel_type":
            cancel = object()
        elif fault == "cancelled":
            cancel.set()
            expected, reason = InterruptedError, None
        elif fault in {"plan", "mode", "target"}:
            plan = object() if fault == "plan" else replace(plan, **{fault: "invalid"})
            reason = "^rollback_plan_required$"
        elif fault == "snapshot":
            plan = replace(plan, local_snapshot=object())
        elif fault == "incomplete":
            plan = replace(plan, target=replace(plan.target, complete=False))
        elif fault == "exhausted":
            budget.reserve(budget._remaining - 1023)
            expected, reason = _TraceReadLimit, "^mobile_trace_input_limit$"
        before = budget._remaining

        def forbidden(*args, **kwargs):
            pytest.fail("invalid route reached workspace or source journal IO")

        monkeypatch.setattr(replacement, "_descriptor", forbidden)
        monkeypatch.setattr(replacement, "_checked_originals", forbidden)
        monkeypatch.setattr(replacement, "create_private_directory", forbidden)
        with pytest.raises(expected, match=reason):
            replacement.capture_verify_rollback(
                candidate,
                plan,
                journal,
                tmp_path / "never.age",
                session=selected_session,
                password=b"isolated-negative",
                work_root=tmp_path / "never-work",
                cancel=cancel,
                acknowledged_credential_issues=(),
                **options,
            )
        assert budget._remaining == before
        assert not (tmp_path / "never-work").exists()


@private_profile_test
def test_yielded_native_capture_scope_keeps_uncertain_close_quarantined(
    request, tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.admission import AdmissionError
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Backup_Recovery.native_files import create_private_file

    native_os = storage.os
    target_fd = None
    closes = []
    reached = {}
    holds = len(storage._failed_capture_holds)

    class NativeClose:
        def __getattr__(self, name):
            return getattr(native_os, name)

        def close(self, fd):
            native_os.close(fd)
            if fd == target_fd:
                closes.append(fd)
                raise OSError("native close completed with uncertain outcome")

    with (
        pytest.raises(
            (RecoveryRequired, AdmissionError), match="^capture_resources_not_retired$"
        ),
        replacement_case(tmp_path, monkeypatch) as case,
    ):
        _, plan, journal, session, _, _ = case
        stage = tmp_path / "uncertain-scope-stage"
        stage.mkdir(mode=0o700)
        limits = ArchiveLimits()
        monkeypatch.setattr(storage, "os", NativeClose())
        with session._replacement_capture_scope(
            plan, journal, stage, limits=limits, byte_budget=limits.expanded_bytes
        ) as pair:
            assert pair[1].items
            scope = storage._local.capture_scope
            custody = storage._CaptureFileDescriptors(scope)
            reached.update(session=session, scope=scope, custody=custody)
            with create_private_file(stage / "owned-close", _custody=custody) as fd:
                target_fd = fd
                native_os.write(fd, b"private native retirement evidence")
        pytest.fail("uncertain retirement returned to its capture consumer")
    assert closes == [target_fd]
    assert reached["custody"].close_failed
    assert reached["custody"] in reached["scope"].resources
    assert reached["scope"] in reached["session"]._scopes
    assert not reached["scope"].active and not reached["session"]._active
    assert len(storage._failed_capture_holds) > holds
