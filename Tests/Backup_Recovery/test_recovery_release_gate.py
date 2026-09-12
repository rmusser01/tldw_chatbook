"""Release availability is finite and enforced at every product boundary."""

import builtins
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

_HOST_IDENTITY = {
    "os": "Darwin",
    "release": "25.5.0",
    "arch": "arm64",
    "python": "3.12.11",
    "filesystem": "apfs",
    "flags": 76583040,
}
_IMAGE_IDENTITY = {**_HOST_IDENTITY, "flags": 76583448}
_PRODUCT_FACTS = frozenset({"owner_coverage", "archive", "restore", "product_flow"})


def _replacement_plan(*, local_snapshot=None):
    from tldw_chatbook.Backup_Recovery.restore_plan import RestorePlan

    return RestorePlan(
        archive_digest="reviewed-archive",
        mode="replace",
        restore=(),
        retire=(),
        preserve=(),
        target_fingerprint="reviewed-target",
        local_snapshot=local_snapshot,
    )


def test_unavailable_complete_backup_refuses_before_worker_allocation(
    tmp_path, monkeypatch
):
    """Removing the pre-start release gate would allocate a backup worker."""
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    service = RecoveryService(tmp_path / "control")

    def allocated(*_args, **_kwargs):
        pytest.fail("unavailable backup allocated a worker")

    monkeypatch.setattr(service, "_start", allocated)
    try:
        with pytest.raises(ValueError, match="release_capability_unavailable"):
            service.start_backup(
                (),
                "reviewed-scope",
                tmp_path / "backup.tldw-backup.zip",
                options={"allow_partial": True, "staging_parent": tmp_path},
                password=None,
            )
    finally:
        service.close()


def test_product_facts_are_operation_and_cell_specific(monkeypatch):
    """Broadening either declaration would authorize an unreviewed operation/cell."""
    from tldw_chatbook.Backup_Recovery import qualification

    assert (
        qualification._source_product_facts("complete_capture", _HOST_IDENTITY)
        == _PRODUCT_FACTS
    )
    assert (
        qualification._source_product_facts("new_replacement", _HOST_IDENTITY)
        == _PRODUCT_FACTS
    )
    assert not qualification._source_product_facts("complete_capture", _IMAGE_IDENTITY)
    assert (
        qualification._source_product_facts("new_replacement", _IMAGE_IDENTITY)
        == _PRODUCT_FACTS
    )
    assert not qualification._source_product_facts("unknown", _HOST_IDENTITY)

    for field, changed in (
        ("os", "Linux"),
        ("release", "25.5.1"),
        ("arch", "x86_64"),
        ("python", "3.12.12"),
        ("filesystem", "hfs"),
        ("flags", 76583041),
    ):
        identity = {**_HOST_IDENTITY, field: changed}
        assert not qualification._source_product_facts("complete_capture", identity), (
            field
        )

    monkeypatch.setattr(qualification, "_QUALIFICATION_PROTOCOL", 3)
    assert not qualification._source_product_facts("complete_capture", _HOST_IDENTITY)


def test_missing_product_fact_keeps_release_gate_closed(monkeypatch):
    """Dropping one reviewed fact must fail the unchanged seven-gate conjunction."""
    from tldw_chatbook.Backup_Recovery import qualification

    cell = (*_HOST_IDENTITY.values(), 2)
    monkeypatch.setattr(
        qualification,
        "_COMPLETE_CAPTURE_FACTS",
        {cell: _PRODUCT_FACTS - {"product_flow"}},
    )
    assert qualification._source_product_gates(
        "complete_capture", (_HOST_IDENTITY,)
    ) == (True, True, True, False)
    assert not qualification.release_capability(
        helper=True,
        owner_coverage=True,
        admission=True,
        archive=True,
        native_publish=True,
        restore=True,
        product_flow=False,
    )


def test_mixed_image_cell_refuses_complete_but_allows_replacement_facts():
    """Copying replacement image facts into Complete would broaden availability."""
    from tldw_chatbook.Backup_Recovery import qualification

    identities = (_HOST_IDENTITY, _IMAGE_IDENTITY)
    assert qualification._source_product_gates("complete_capture", identities) == (
        False,
        False,
        False,
        False,
    )
    assert qualification._source_product_gates("new_replacement", identities) == (
        True,
        True,
        True,
        True,
    )


@pytest.mark.parametrize("missing", ("helper", "admission", "publication"))
def test_release_evaluator_requires_independent_installed_gates(monkeypatch, missing):
    """Dropping any real installed gate must close otherwise-complete wiring."""
    from tldw_chatbook.Backup_Recovery import (
        crypto,
        native_files,
        qualification,
        restore_plan,
    )

    # This is a wiring test, not product qualification evidence. Each double keeps
    # one gate observable while avoiding native work.
    monkeypatch.setattr(
        qualification, "_source_product_gates", lambda *_a: (True, True, True, True)
    )
    monkeypatch.setattr(
        crypto,
        "helper_capability",
        lambda: (missing != "helper", "unit-helper"),
    )
    monkeypatch.setattr(restore_plan, "_ancestor", lambda path: path)

    @contextmanager
    def pinned(_path):
        yield 7

    monkeypatch.setattr(native_files, "pinned_directory", pinned)
    monkeypatch.setattr(qualification, "native_identity", lambda _fd: _HOST_IDENTITY)

    def qualified(operation, _root):
        if missing == "admission" and operation == "admission":
            return False, "unit-admission"
        if missing == "publication" and operation == "publish_directory":
            return False, "unit-publication"
        return True, "unit-qualified"

    monkeypatch.setattr(qualification, "qualified_for", qualified)
    assert qualification._release_for_roots(
        "complete_capture",
        product_roots=(Path("/product"),),
        admission_roots=(Path("/authority"),),
    ) == (False, "release_capability_unavailable")


def test_capability_wrappers_select_actual_write_and_authority_roots(
    tmp_path, monkeypatch
):
    """Omitting a participating root would leave that volume outside the decision."""
    from tldw_chatbook.Backup_Recovery import bootstrap, qualification
    from tldw_chatbook.Backup_Recovery.service_storage import work_root

    captured = []

    def evaluate(operation, *, product_roots, admission_roots):
        captured.append((operation, product_roots, admission_roots))
        return False, "release_capability_unavailable"

    monkeypatch.setattr(qualification, "_release_for_roots", evaluate)
    bootstrap_root = bootstrap.default_bootstrap_root()
    staging = tmp_path / "staging"
    destination = tmp_path / "output" / "backup.tldw-backup.zip"
    control = tmp_path / "service" / "control"
    assert not qualification.complete_capture_capability(
        staging_parent=staging,
        destination=destination,
        control_root=control,
    )[0]
    operation, product_roots, admission_roots = captured.pop(0)
    assert operation == "complete_capture"
    assert product_roots == (
        staging,
        destination,
        control,
        work_root(control),
        bootstrap_root.parent,
        bootstrap_root,
        bootstrap_root / "admission",
    )
    assert admission_roots == (
        bootstrap_root.parent,
        bootstrap_root,
        bootstrap_root / "admission",
    )

    plan = replace(
        _replacement_plan(),
        target=SimpleNamespace(),
        destinations=(("root", tmp_path / "image" / "destination"),),
        restore=(("file", tmp_path / "image" / "destination" / "file"),),
        retire=(("old", tmp_path / "host" / "old"),),
    )
    assert not qualification.replacement_capability(plan, control_root=control)[0]
    operation, product_roots, admission_roots = captured.pop(0)
    assert operation == "new_replacement"
    assert product_roots == (
        control,
        work_root(control),
        bootstrap_root.parent,
        bootstrap_root,
        bootstrap_root / "admission",
        tmp_path / "image" / "destination",
        tmp_path / "image" / "destination" / "file",
        tmp_path / "host" / "old",
    )
    assert admission_roots == (
        bootstrap_root.parent,
        bootstrap_root,
        bootstrap_root / "admission",
    )


def test_backup_rechecks_release_capability_inside_worker(tmp_path, monkeypatch):
    """Removing the worker-entry recheck would reach native capture."""
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    service = RecoveryService(tmp_path / "control")
    decisions = iter(
        (
            (True, "release_capability_available"),
            (False, "release_capability_unavailable"),
        )
    )
    monkeypatch.setattr(service, "backup_capability", lambda *_a, **_k: next(decisions))
    destination = tmp_path / "backup.tldw-backup.zip"
    try:
        operation = service.start_backup(
            (),
            "reviewed-scope",
            destination,
            options={"allow_partial": True, "staging_parent": tmp_path},
            password=None,
        )
        state = service.wait(operation, timeout=5)
        assert state["state"] == "failed"
        assert state["issues"] == ("release_capability_unavailable",)
        assert not destination.exists()
    finally:
        service.close()


def test_replacement_refuses_before_password_or_worker(tmp_path, monkeypatch):
    """Moving the gate after password validation would expose the wrong refusal."""
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    service = RecoveryService(tmp_path / "control")
    monkeypatch.setattr(
        service,
        "inspection",
        lambda _operation: SimpleNamespace(digest="reviewed-archive"),
    )
    monkeypatch.setattr(
        service,
        "replacement_capability",
        lambda _plan: (False, "release_capability_unavailable"),
        raising=False,
    )
    monkeypatch.setattr(
        service,
        "_start",
        lambda *_a, **_k: pytest.fail("unavailable replacement allocated a worker"),
    )
    try:
        with pytest.raises(ValueError, match="release_capability_unavailable"):
            service.start_restore(
                "inspection", _replacement_plan(), rollback_password=b""
            )
    finally:
        service.close()


def test_replacement_rechecks_release_capability_inside_worker(tmp_path, monkeypatch):
    """Removing the replacement worker recheck would reach native staging."""
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    service = RecoveryService(tmp_path / "control")
    monkeypatch.setattr(
        service,
        "inspection",
        lambda _operation: SimpleNamespace(digest="reviewed-archive"),
    )
    decisions = iter(
        (
            (True, "release_capability_available"),
            (False, "release_capability_unavailable"),
        )
    )
    monkeypatch.setattr(
        service, "replacement_capability", lambda _plan: next(decisions)
    )
    try:
        operation = service.start_restore(
            "inspection", _replacement_plan(), rollback_password=b"new-password"
        )
        state = service.wait(operation, timeout=5)
        assert state["state"] == "failed"
        assert state["issues"] == ("release_capability_unavailable",)
    finally:
        service.close()


def test_later_rollback_refuses_before_worker_allocation(tmp_path, monkeypatch):
    """Removing the later-operation gate would allocate a safety-copy worker."""
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.Backup_Recovery.restore_plan import LocalSnapshotSource

    service = RecoveryService(tmp_path / "control")
    plan = _replacement_plan(
        local_snapshot=LocalSnapshotSource(
            control_root=service.control_root,
            operation_id="copy-id",
            rollback_digest="reviewed-copy",
        )
    )
    monkeypatch.setattr(
        service,
        "replacement_capability",
        lambda _plan: (False, "release_capability_unavailable"),
        raising=False,
    )
    monkeypatch.setattr(
        service,
        "_start",
        lambda *_a, **_k: pytest.fail("unavailable rollback allocated a worker"),
    )
    try:
        with pytest.raises(ValueError, match="release_capability_unavailable"):
            service.start_rollback(
                "copy-id", plan, old_password=b"old", new_password=b"new"
            )
    finally:
        service.close()


def test_later_rollback_rechecks_release_capability_inside_worker(
    tmp_path, monkeypatch
):
    """Removing the later worker recheck would begin a new safety copy."""
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.Backup_Recovery.restore_plan import LocalSnapshotSource

    service = RecoveryService(tmp_path / "control")
    plan = _replacement_plan(
        local_snapshot=LocalSnapshotSource(
            control_root=service.control_root,
            operation_id="copy-id",
            rollback_digest="reviewed-copy",
        )
    )
    decisions = iter(
        (
            (True, "release_capability_available"),
            (False, "release_capability_unavailable"),
        )
    )
    monkeypatch.setattr(
        service, "replacement_capability", lambda _plan: next(decisions)
    )
    try:
        operation = service.start_rollback(
            "copy-id", plan, old_password=b"old", new_password=b"new"
        )
        state = service.wait(operation, timeout=5)
        assert state["state"] == "failed"
        assert state["issues"] == ("release_capability_unavailable",)
    finally:
        service.close()


def test_existing_pending_recovery_does_not_consult_new_operation_gate(
    tmp_path, monkeypatch
):
    """Adding a new-operation gate to recovery dispatch would strand pending work."""
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    service = RecoveryService(tmp_path / "control")
    monkeypatch.setattr(
        service,
        "status",
        lambda _operation: {"actions": ("abort",), "mode": "replace"},
    )
    monkeypatch.setattr(
        service,
        "replacement_capability",
        lambda _plan: pytest.fail("pending recovery consulted the new-operation gate"),
    )
    monkeypatch.setattr(
        service,
        "_start",
        lambda kind, _function: "pending-recovery" if kind == "recover" else None,
    )
    try:
        assert service.start_recovery("pending", action="abort") == "pending-recovery"
    finally:
        service.close()


def test_cli_replacement_refuses_before_confirmation_and_new_password(monkeypatch):
    """Removing the CLI decision would reach confirmation or a new secret prompt."""
    from tldw_chatbook.Backup_Recovery import launcher

    plan = _replacement_plan()

    class Service:
        def preview_backup(self, *_args, **_kwargs):
            return SimpleNamespace()

        def preview_restore(self, *_args, **_kwargs):
            return plan

        def replacement_capability(self, candidate):
            assert candidate is plan
            return False, "release_capability_unavailable"

    args = SimpleNamespace(
        archive=None,
        ask_password=False,
        mode="replace",
        target_config=Path("/target/config.toml"),
        destination=(),
        profile_name=(),
        acknowledge_credential_issue=(),
        safety_scope=(),
    )
    monkeypatch.setattr(launcher, "_inspect", lambda *_a, **_k: "inspection")
    monkeypatch.setattr(
        builtins,
        "input",
        lambda *_a, **_k: pytest.fail("unavailable replacement asked for confirmation"),
    )
    monkeypatch.setattr(
        launcher,
        "_password",
        lambda *_a, **_k: pytest.fail("unavailable replacement asked for password"),
    )
    assert launcher._restore(Service(), args) == 1


@pytest.mark.asyncio
async def test_ui_displays_backup_availability_separately_and_disables_create():
    """Ignoring availability would enable Create from coverage and capacity alone."""
    from textual.app import App
    from textual.widgets import Button, Static

    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    class Service:
        issue_code = staticmethod(lambda error: str(error))

        def current(self):
            return None

    screen = BackupRestoreScreen(Service(), config_paths=("/profile/config.toml",))

    class Harness(App):
        def on_mount(self):
            self.push_screen(screen)

    details = {
        "inventory": SimpleNamespace(
            complete=True, items=(), issues=(), scope_digest="reviewed-scope"
        ),
        "maintenance": "Writers pause during capture.",
        "capacity": (
            {
                "path": "/volume",
                "required_bytes": 1,
                "available_bytes": 2,
                "sufficient": True,
            },
        ),
        "credential_mode": "exclude",
        "availability": (False, "release_capability_unavailable"),
    }
    app = Harness()
    async with app.run_test(size=(90, 30)):
        screen._show_mode("create")
        reviewed = (
            ("/profile/config.toml",),
            SimpleNamespace(),
            {"allow_partial": True},
        )
        screen._show_preview(screen._revision, details, reviewed)
        coverage = str(screen.query_one("#backup-coverage", Static).render())
        assert "Availability: unavailable (release_capability_unavailable)" in coverage
        assert screen.query_one("#backup-create", Button).disabled


@pytest.mark.asyncio
async def test_ui_replacement_and_later_refuse_before_reading_new_passwords(
    monkeypatch,
):
    """Discarding stored decisions would let both handlers read new secrets."""
    from textual.app import App
    from textual.widgets import Static

    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    class Service:
        issue_code = staticmethod(lambda error: str(error))

        def current(self):
            return None

        def start_restore(self, *_args, **_kwargs):
            pytest.fail("unavailable replacement reached the service")

        def start_rollback(self, *_args, **_kwargs):
            pytest.fail("unavailable rollback reached the service")

    screen = BackupRestoreScreen(Service())

    class Harness(App):
        def on_mount(self):
            self.push_screen(screen)

    app = Harness()
    async with app.run_test(size=(90, 30)):
        monkeypatch.setattr(
            screen,
            "_input",
            lambda _name: pytest.fail("unavailable operation read a secret"),
        )
        plan = _replacement_plan()
        screen._restore_plan = plan
        screen._restore_availability = (False, "release_capability_unavailable")
        screen._start_restore()
        message = str(screen.query_one("#backup-message", Static).render())
        assert "release_capability_unavailable" in message

        screen._rollback_copy_id = "copy-id"
        screen._rollback_plan = plan
        screen._rollback_availability = (False, "release_capability_unavailable")
        screen._start_rollback()
        message = str(screen.query_one("#backup-message", Static).render())
        assert "release_capability_unavailable" in message


@pytest.mark.asyncio
async def test_ui_invalidates_stale_availability_decisions():
    """Retaining a decision across input changes would enable a stale operation."""
    from textual.app import App

    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    class Service:
        issue_code = staticmethod(lambda error: str(error))

        def current(self):
            return None

    screen = BackupRestoreScreen(Service())

    class Harness(App):
        def on_mount(self):
            self.push_screen(screen)

    app = Harness()
    async with app.run_test(size=(90, 30)):
        screen._backup_availability = (True, "release_capability_available")
        screen._restore_availability = (True, "release_capability_available")
        screen._rollback_availability = (True, "release_capability_available")
        screen._invalidate()
        assert screen._backup_availability is None
        assert screen._restore_availability is None
        assert screen._rollback_availability is None


@pytest.mark.asyncio
async def test_ui_starts_new_operations_off_the_ui_thread(monkeypatch):
    """Calling a service start directly would run the helper check on the UI thread."""
    import asyncio
    import threading

    from textual.app import App
    from textual.widgets import Checkbox

    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    calls = []
    reached = {name: threading.Event() for name in ("backup", "restore", "rollback")}

    class Service:
        issue_code = staticmethod(lambda error: str(error))

        def current(self):
            return None

        def start_backup(self, *_args, **_kwargs):
            calls.append(("backup", threading.get_ident()))
            reached["backup"].set()
            return "backup-operation"

        def start_restore(self, *_args, **_kwargs):
            calls.append(("restore", threading.get_ident()))
            reached["restore"].set()
            return "restore-operation"

        def start_rollback(self, *_args, **_kwargs):
            calls.append(("rollback", threading.get_ident()))
            reached["rollback"].set()
            return "rollback-operation"

    screen = BackupRestoreScreen(Service())

    class Harness(App):
        def on_mount(self):
            self.push_screen(screen)

    app = Harness()
    async with app.run_test(size=(90, 30)) as pilot:
        ui_thread = threading.get_ident()
        monkeypatch.setattr(screen, "_validate_password", lambda _options: None)
        monkeypatch.setattr(screen, "_input", lambda _name: "private-password")

        screen._preview = SimpleNamespace(scope_digest="reviewed-scope")
        screen._reviewed = (
            (),
            Path("/output/backup.tldw-backup.zip"),
            {"encrypted": False},
        )
        screen._backup_availability = (True, "release_capability_available")
        screen._create()
        assert await asyncio.to_thread(reached["backup"].wait, 5)

        plan = _replacement_plan()
        screen._inspection_id = "inspection"
        screen._restore_plan = plan
        screen._restore_availability = (True, "release_capability_available")
        screen._start_restore()
        assert await asyncio.to_thread(reached["restore"].wait, 5)

        screen._rollback_copy_id = "copy-id"
        screen._rollback_plan = plan
        screen._rollback_availability = (True, "release_capability_available")
        screen.query_one("#backup-later-confirm", Checkbox).value = True
        screen._start_rollback()
        assert await asyncio.to_thread(reached["rollback"].wait, 5)
        await pilot.pause()

    assert {name for name, _thread in calls} == {"backup", "restore", "rollback"}
    assert all(thread != ui_thread for _name, thread in calls)


@pytest.mark.parametrize("user_edits_form", (False, True))
@pytest.mark.asyncio
async def test_ui_password_clear_preserves_only_current_worker_association(
    user_edits_form,
):
    """Programmatic secret clearing is inert, while a real edit rejects stale work."""
    import asyncio
    import threading

    from textual.app import App
    from textual.widgets import Input

    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    release = threading.Event()
    returned = threading.Event()
    operation_id = "credential-review-operation"
    credential_issue = "credential_missing:config:test-slot"
    terminal = {
        "operation_id": operation_id,
        "kind": "backup",
        "state": "failed",
        "phase": "capturing",
        "issues": ("review_required",),
        "review_issues": (credential_issue,),
        "result": {},
    }

    class Service:
        issue_code = staticmethod(lambda error: str(error))

        def current(self):
            return terminal if returned.is_set() else None

        def start_backup(self, *_args, **_kwargs):
            assert release.wait(5)
            returned.set()
            return operation_id

    screen = BackupRestoreScreen(Service())

    class Harness(App):
        def on_mount(self):
            self.push_screen(screen)

    app = Harness()
    async with app.run_test(size=(90, 30)) as pilot:
        password = screen.query_one("#backup-password", Input)
        confirmation = screen.query_one("#backup-password-confirm", Input)
        password.value = confirmation.value = "private-password"
        await pilot.pause()
        screen._preview = SimpleNamespace(scope_digest="reviewed-scope")
        screen._reviewed = (
            (),
            Path("/output/backup.tldw-backup.zip"),
            {"encrypted": True},
        )
        screen._backup_availability = (True, "release_capability_available")
        screen._create()
        await pilot.pause()
        assert password.value == confirmation.value == ""

        if user_edits_form:
            screen.query_one("#backup-destination", Input).value = "/different/output.zip"
            await pilot.pause()

        release.set()
        assert await asyncio.to_thread(returned.wait, 5)
        for _ in range(20):
            await pilot.pause()
            if screen._requested_backup_operation is not None:
                break

        if user_edits_form:
            assert screen._requested_backup_operation is None
            assert not screen.query(".backup-acknowledge-credential")
        else:
            assert screen._requested_backup_operation == operation_id
            assert len(screen.query(".backup-acknowledge-credential")) == 1


@pytest.mark.parametrize("user_edits_form", (False, True))
@pytest.mark.asyncio
async def test_ui_replacement_password_clear_preserves_only_current_worker_association(
    user_edits_form,
):
    """Replacement retry association survives clearing, but not a later user edit."""
    import asyncio
    import threading

    from textual.app import App
    from textual.widgets import Input

    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    release = threading.Event()
    returned = threading.Event()
    operation_id = "replacement-credential-review"
    credential_issue = "credential_missing:config:test-slot"
    terminal = {
        "operation_id": operation_id,
        "kind": "restore",
        "state": "failed",
        "phase": "capturing_safety_copy",
        "issues": ("review_required",),
        "review_issues": (credential_issue,),
        "result": {},
    }

    class Service:
        issue_code = staticmethod(lambda error: str(error))

        def current(self):
            return terminal if returned.is_set() else None

        def start_restore(self, *_args, **_kwargs):
            assert release.wait(5)
            returned.set()
            return operation_id

    screen = BackupRestoreScreen(Service())

    class Harness(App):
        def on_mount(self):
            self.push_screen(screen)

    app = Harness()
    async with app.run_test(size=(90, 30)) as pilot:
        password = screen.query_one("#backup-rollback-password", Input)
        confirmation = screen.query_one("#backup-rollback-confirm", Input)
        password.value = confirmation.value = "private-password"
        await pilot.pause()
        screen._inspection_id = "inspection"
        screen._restore_plan = _replacement_plan()
        screen._restore_availability = (True, "release_capability_available")
        screen._start_restore()
        await pilot.pause()
        assert password.value == confirmation.value == ""

        if user_edits_form:
            screen.query_one("#backup-target-config", Input).value = "/different/config.toml"
            await pilot.pause()

        release.set()
        assert await asyncio.to_thread(returned.wait, 5)
        for _ in range(20):
            await pilot.pause()
            if screen._requested_restore_operation is not None:
                break

        if user_edits_form:
            assert screen._requested_restore_operation is None
            assert not screen.query(".backup-acknowledge-restore-credential")
        else:
            assert screen._requested_restore_operation == operation_id
            assert len(screen.query(".backup-acknowledge-restore-credential")) == 1


@pytest.mark.asyncio
async def test_ui_password_clear_preserves_current_worker_refusal_message():
    """A current refusal remains visible after clearing real password widgets."""
    import asyncio
    import threading

    from textual.app import App
    from textual.widgets import Input, Static

    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    release = threading.Event()
    refused = threading.Event()

    class Service:
        issue_code = staticmethod(lambda error: str(error))

        def current(self):
            return None

        def start_backup(self, *_args, **_kwargs):
            assert release.wait(5)
            refused.set()
            raise ValueError("release_capability_unavailable")

    screen = BackupRestoreScreen(Service())

    class Harness(App):
        def on_mount(self):
            self.push_screen(screen)

    app = Harness()
    async with app.run_test(size=(90, 30)) as pilot:
        screen.query_one("#backup-password", Input).value = "private-password"
        screen.query_one("#backup-password-confirm", Input).value = "private-password"
        await pilot.pause()
        screen._preview = SimpleNamespace(scope_digest="reviewed-scope")
        screen._reviewed = (
            (),
            Path("/output/backup.tldw-backup.zip"),
            {"encrypted": True},
        )
        screen._backup_availability = (True, "release_capability_available")
        screen._create()
        await pilot.pause()
        release.set()
        assert await asyncio.to_thread(refused.wait, 5)
        for _ in range(20):
            await pilot.pause()
            message = str(screen.query_one("#backup-message", Static).render())
            if "release_capability_unavailable" in message:
                break
        assert "release_capability_unavailable" in message
