"""Focused tests for the managed-model Installed view."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.css.query import NoMatches
from textual.widgets import Button

from tldw_chatbook.Model_Artifacts.service import ArtifactRef


class _InstalledApp(App):
    def __init__(self, view) -> None:
        self.view = view
        super().__init__()

    def compose(self) -> ComposeResult:
        yield self.view


@pytest.mark.asyncio
async def test_installed_view_performs_no_io_at_compose_time(tmp_path: Path) -> None:
    """Eagerly mounted model views stay idle until their rail row is selected."""
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    service_factory = MagicMock()
    view = InstalledView(service_factory=service_factory, legacy_dir=tmp_path)
    app = _InstalledApp(view)
    async with app.run_test() as pilot:
        await pilot.pause()

    service_factory.assert_not_called()


@pytest.mark.asyncio
async def test_models_host_lazily_wires_parakeet_activation_and_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Models host binds its app service on the Textual UI thread."""
    import threading

    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
    from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView
    from tldw_chatbook.config import get_cli_setting as real_get_cli_setting
    from Tests.UI.app_factory import _build_test_app

    root = ArtifactRef("parakeet-v2", "immutable-revision", "int8")
    vad = ArtifactRef("silero-vad", "immutable-revision", "f32")
    ui_thread = threading.get_ident()
    lifecycle_threads: list[tuple[str, int]] = []

    class _Source:
        def __init__(self) -> None:
            self.activated: list[ArtifactRef] = []
            self.deletion_checks: list[ArtifactRef] = []

        def release_scopes_except(self, scopes: set[str]) -> None:
            assert scopes == set()
            lifecycle_threads.append(("release", threading.get_ident()))

        def records(self) -> dict[object, object]:
            return {}

        def close(self) -> None:
            return None

        def on_root_activated(self, reference: ArtifactRef) -> None:
            self.activated.append(reference)

        def may_delete(self, reference: ArtifactRef) -> str:
            self.deletion_checks.append(reference)
            return "Managed dependency is in use."

    source = _Source()

    class _Core:
        def activate(self, reference: ArtifactRef) -> ArtifactRef:
            return reference

    def no_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr("tldw_chatbook.app.get_cli_setting", no_splash)
    monkeypatch.setattr(
        LLMManagementWindow,
        "_ollama_api_available",
        lambda _window: False,
    )
    app = _build_test_app()

    def create_source_service() -> _Source:
        lifecycle_threads.append(("construct", threading.get_ident()))
        return source

    def add_listener(_listener) -> None:
        lifecycle_threads.append(("listener", threading.get_ident()))

    def read_jobs() -> tuple[object, ...]:
        lifecycle_threads.append(("read", threading.get_ident()))
        return ()

    monkeypatch.setattr(app, "_create_parakeet_source_service", create_source_service)
    monkeypatch.setattr(app.library_ingest_jobs, "add_listener", add_listener)
    monkeypatch.setattr(app.library_ingest_jobs, "jobs", read_jobs)
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = LLMScreen(app)
        await app.push_screen(screen)
        for _ in range(6):
            await pilot.pause()

        screen.query_one(LLMManagementWindow)
        view = screen.query_one(InstalledView)
        assert [name for name, _thread in lifecycle_threads] == [
            "construct",
            "listener",
            "read",
            "release",
        ]
        assert all(thread == ui_thread for _name, thread in lifecycle_threads)

        ensure_after_mount = MagicMock(
            side_effect=AssertionError("activation must use the bound source service")
        )
        monkeypatch.setattr(
            app,
            "_ensure_parakeet_source_service",
            ensure_after_mount,
        )

        view._service_factory = _Core
        view._legacy_dir = tmp_path
        view._apply_lifecycle_result = MagicMock()
        await view._activate_model(root).wait()
        view._apply_lifecycle_result.assert_called_once_with("activate", None)
        assert source.activated == [root]
        ensure_after_mount.assert_not_called()

        assert view._may_delete(vad) == "Managed dependency is in use."
        assert source.deletion_checks == [vad]
        ensure_after_mount.assert_not_called()


def test_unmanaged_scan_is_bounded_and_labels_supported_model_files(
    tmp_path: Path,
) -> None:
    """Legacy GGUF/ONNX files stay visible without an unbounded result set."""
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    for index in range(3):
        (tmp_path / f"model-{index}.gguf").write_bytes(b"x" * (1024 * 1024 + 1))
    (tmp_path / "ignore.txt").write_text("not a model")

    rows = InstalledView.scan_unmanaged(tmp_path, limit=2)

    assert len(rows) == 2
    assert all(row.path.suffix == ".gguf" for row in rows)


def test_unmanaged_scan_validates_root_before_walking(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Configured legacy roots pass the shared path-safety boundary first."""
    from tldw_chatbook.UI.Screens import model_installed_view as module

    walk = MagicMock()
    monkeypatch.setattr(module.os, "walk", walk)

    with pytest.raises(ValueError, match="dangerous pattern"):
        module.InstalledView.scan_unmanaged(tmp_path / "../..")

    walk.assert_not_called()


@pytest.mark.asyncio
async def test_activation_controls_emit_intents_and_refuse_pending_reentry() -> None:
    """Controls post exact refs and disable both mutations while pending."""
    from tldw_chatbook.Widgets.ModelArtifacts.activation_controls import (
        ActivationRequested,
        DeletionRequested,
        ModelActivationControls,
    )

    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")

    class _ControlsApp(App):
        def __init__(self) -> None:
            self.messages = []
            super().__init__()

        def compose(self) -> ComposeResult:
            yield ModelActivationControls(reference, active=False, ready=True)

        def on_activation_requested(self, event: ActivationRequested) -> None:
            self.messages.append(event)

        def on_deletion_requested(self, event: DeletionRequested) -> None:
            self.messages.append(event)

    app = _ControlsApp()
    async with app.run_test() as pilot:
        controls = app.query_one(ModelActivationControls)
        await pilot.click(".model-activate")
        await pilot.pause()
        assert isinstance(app.messages[0], ActivationRequested)
        assert app.messages[0].reference == reference

        controls.set_pending(True)
        await pilot.pause()
        assert app.query_one(".model-activate", Button).disabled is True
        assert app.query_one(".model-delete", Button).disabled is True


@pytest.mark.asyncio
async def test_unassigned_controls_omit_activate_and_keep_delete_available() -> None:
    """An unassigned inventory row can be deleted but cannot request activation.

    This fails if disabling activation also removes Delete, or if controls
    render an activation affordance when policy disallows it.
    """
    from tldw_chatbook.Widgets.ModelArtifacts.activation_controls import (
        DeletionRequested,
        ModelActivationControls,
    )

    reference = ArtifactRef("remote-gguf", "immutable-revision", "q4_k_m")

    class _ControlsApp(App):
        def __init__(self) -> None:
            self.messages = []
            super().__init__()

        def compose(self) -> ComposeResult:
            yield ModelActivationControls(
                reference,
                active=False,
                ready=True,
                allow_activation=False,
            )

        def on_deletion_requested(self, event: DeletionRequested) -> None:
            self.messages.append(event)

    app = _ControlsApp()
    async with app.run_test() as pilot:
        controls = app.query_one(ModelActivationControls)
        assert len(app.query(".model-activate")) == 0
        assert app.query_one(".model-delete", Button).disabled is False

        controls.set_pending(True)
        await pilot.pause()
        assert app.query_one(".model-delete", Button).disabled is True

        controls.set_pending(False)
        await pilot.click(".model-delete")
        await pilot.pause()

    assert len(app.messages) == 1
    assert isinstance(app.messages[0], DeletionRequested)
    assert app.messages[0].reference == reference


@pytest.mark.asyncio
async def test_mounted_dependency_row_labels_ownership_and_omits_activate(
    tmp_path: Path,
) -> None:
    """The Installed surface exposes dependency ownership without root actions."""
    from dataclasses import replace

    from textual.widgets import Static

    from Tests.Model_Artifacts.test_acquisition_types import make_descriptor
    from tldw_chatbook.Model_Artifacts.service import ArtifactRole, InstalledArtifact
    from tldw_chatbook.UI.Screens.model_browser_state import inventory_rows
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    descriptor = replace(make_descriptor(), role=ArtifactRole.DEPENDENCY)
    rows = inventory_rows(
        (
            InstalledArtifact(
                path=tmp_path / "dependency",
                descriptor=descriptor,
                ready=False,
                active=False,
                error=None,
            ),
        ),
        None,
        (),
    )
    view = InstalledView(service_factory=MagicMock(), legacy_dir=tmp_path)
    app = _InstalledApp(view)

    async with app.run_test() as pilot:
        view._apply_inventory(rows, None, None)
        await pilot.pause()
        text = "\n".join(str(item.renderable) for item in view.query(Static))

        assert "Managed dependency" in text
        assert len(view.query(".model-activate")) == 0
        assert view.query_one(".model-delete", Button).disabled is False


def test_installed_view_refuses_a_second_lifecycle_operation() -> None:
    """Activation/deletion cannot re-enter while hashing or leasing is pending."""
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    view = InstalledView(service_factory=MagicMock(), legacy_dir=Path("/tmp/models"))
    view._operation_reference = ArtifactRef("parakeet-v2", "rev1", "int8")
    view._activate_model = MagicMock()

    view._request_activation(ArtifactRef("parakeet-v2", "rev2", "f32"))

    view._activate_model.assert_not_called()


def test_activation_changes_source_preference_only_after_core_success(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The preference callback follows the real activation boundary."""
    from tldw_chatbook.UI.Screens import model_installed_view as module

    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")
    events: list[str] = []

    class _Service:
        def activate(self, activated: ArtifactRef) -> ArtifactRef:
            assert activated == reference
            events.append("activate")
            return activated

    fake_app = MagicMock()
    monkeypatch.setattr(module.InstalledView, "app", property(lambda self: fake_app))
    view = module.InstalledView(
        service_factory=_Service,
        legacy_dir=tmp_path,
        on_root_activated=lambda activated: events.append(
            "prefer-managed" if activated == reference else "wrong-reference"
        ),
    )

    module.InstalledView._activate_model.__wrapped__(view, reference)

    assert events == ["activate", "prefer-managed"]
    fake_app.call_from_thread.assert_called_once_with(
        view._apply_lifecycle_result,
        "activate",
        None,
    )


def test_failed_activation_does_not_change_source_preference(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A failed core activation leaves the exact source preference untouched."""
    from tldw_chatbook.UI.Screens import model_installed_view as module

    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")
    preferred: list[ArtifactRef] = []
    service = MagicMock()
    service.activate.side_effect = RuntimeError("private activation detail")
    fake_app = MagicMock()
    monkeypatch.setattr(module.InstalledView, "app", property(lambda self: fake_app))
    view = module.InstalledView(
        service_factory=lambda: service,
        legacy_dir=tmp_path,
        on_root_activated=preferred.append,
    )

    module.InstalledView._activate_model.__wrapped__(view, reference)

    assert preferred == []
    fake_app.call_from_thread.assert_called_once()
    assert fake_app.call_from_thread.call_args.args[1] == "activate"
    assert fake_app.call_from_thread.call_args.args[2] is not None


def test_lease_blocked_deletion_message_is_specific_and_sanitized() -> None:
    """An active lease is named without surfacing raw internal exception text."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactInUseError
    from tldw_chatbook.UI.Screens.model_installed_view import lifecycle_failure_message

    marker = "RAW-LEASE-DETAIL"
    message = lifecycle_failure_message(
        ArtifactInUseError(marker),
        operation="delete",
    )

    assert "in use" in message
    assert marker not in message


def test_repair_summary_reports_every_reconciliation_outcome(tmp_path: Path) -> None:
    """Repair copy names state/staging cleanup and corruption without paths."""
    from tldw_chatbook.Model_Artifacts.service import ReconcileReport
    from tldw_chatbook.UI.Screens.model_installed_view import (
        reconcile_result_message,
    )

    marker = "PRIVATE-MODEL-PATH"
    report = ReconcileReport(
        readiness_created=2,
        state_removed=3,
        corrupt_artifacts=(tmp_path / marker,),
        staging_entries=(tmp_path / "staged-a", tmp_path / "staged-b"),
        staging_removed=(),
    )

    message = reconcile_result_message(report)

    assert "2 readiness" in message
    assert "3 stale state" in message
    assert "2 staging entries observed" in message
    assert "0 staging entries removed" in message
    assert "1 corrupt model" in message
    assert marker not in message


@pytest.mark.parametrize(
    ("worker_name", "service_method", "worker_args", "log_context"),
    (
        ("_load_inventory", "list_installed", (), ("legacy", "configured")),
        (
            "_activate_model",
            "activate",
            (ArtifactRef("parakeet-v2", "rev", "int8"),),
            ("parakeet-v2", "rev", "int8"),
        ),
        (
            "_delete_model",
            "delete",
            (ArtifactRef("parakeet-v2", "rev", "int8"),),
            ("parakeet-v2", "rev", "int8"),
        ),
        ("_repair_store", "reconcile", (), ("store", "shared")),
    ),
)
def test_installed_worker_failures_are_logged_and_sanitized(
    monkeypatch,
    tmp_path: Path,
    worker_name: str,
    service_method: str,
    worker_args: tuple,
    log_context: tuple[str, ...],
) -> None:
    """Every background failure retains diagnostics without exposing them in UI."""
    from tldw_chatbook.UI.Screens import model_installed_view as module

    marker = "PRIVATE-WORKER-DETAIL"
    service = MagicMock()
    getattr(service, service_method).side_effect = RuntimeError(marker)
    fake_app = MagicMock()
    fake_logger = MagicMock()
    fake_logger.opt.return_value = fake_logger
    monkeypatch.setattr(module.InstalledView, "app", property(lambda self: fake_app))
    monkeypatch.setattr(module, "logger", fake_logger)
    view = module.InstalledView(service_factory=lambda: service, legacy_dir=tmp_path)

    worker = getattr(module.InstalledView, worker_name).__wrapped__
    worker(view, *worker_args)

    fake_logger.opt.assert_called_once_with(exception=True)
    fake_logger.error.assert_called_once()
    logged = " ".join(str(value) for value in fake_logger.error.call_args.args).casefold()
    assert all(value in logged for value in log_context)
    assert marker not in str(fake_app.call_from_thread.call_args)


def test_deletion_requires_confirmation_before_starting_worker(monkeypatch) -> None:
    """The destructive service call starts only after explicit confirmation."""
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView
    from tldw_chatbook.Widgets.ModelArtifacts.activation_controls import (
        DeletionRequested,
    )
    from tldw_chatbook.Widgets.delete_confirmation_dialog import (
        DeleteConfirmationDialog,
    )

    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")
    fake_app = MagicMock()
    monkeypatch.setattr(InstalledView, "app", property(lambda self: fake_app))
    view = InstalledView(service_factory=MagicMock(), legacy_dir=Path("/tmp/models"))
    view.refresh = MagicMock()
    view._delete_model = MagicMock()

    view._deletion_requested(DeletionRequested(reference))

    view._delete_model.assert_not_called()
    dialog, callback = fake_app.push_screen.call_args[0]
    assert isinstance(dialog, DeleteConfirmationDialog)
    callback(True)
    view._delete_model.assert_called_once_with(reference)


def test_deletion_guard_blocks_before_and_after_confirmation(monkeypatch) -> None:
    """A dependency that becomes required cannot enter the delete worker."""
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView
    from tldw_chatbook.Widgets.ModelArtifacts.activation_controls import (
        DeletionRequested,
    )

    reference = ArtifactRef("silero-vad", "immutable-revision", "f32")
    decisions = iter((None, "Managed dependency is required by an external source."))
    fake_app = MagicMock()
    monkeypatch.setattr(InstalledView, "app", property(lambda self: fake_app))
    view = InstalledView(
        service_factory=MagicMock(),
        legacy_dir=Path("/tmp/models"),
        may_delete=lambda _reference: next(decisions),
    )
    view.refresh = MagicMock()
    view._delete_model = MagicMock()

    view._deletion_requested(DeletionRequested(reference))
    dialog, callback = fake_app.push_screen.call_args.args
    callback(True)

    view._delete_model.assert_not_called()
    fake_app.notify.assert_called_once()
    assert fake_app.notify.call_args.args == (
        "Managed dependency is required by an external source.",
    )
    assert fake_app.notify.call_args.kwargs["severity"] == "warning"


@pytest.mark.asyncio
async def test_empty_inventory_still_reports_managed_and_staging_space(
    tmp_path: Path,
) -> None:
    """Disk totals do not disappear merely because no manifest row exists."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactDiskUsage
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    view = InstalledView(service_factory=MagicMock(), legacy_dir=tmp_path)
    app = _InstalledApp(view)
    async with app.run_test() as pilot:
        view._apply_inventory(
            (),
            ArtifactDiskUsage(
                installed_bytes=0,
                staging_bytes=2 * 1024 * 1024,
                free_bytes=4 * 1024 * 1024,
            ),
            None,
        )
        await pilot.pause()
        text = "\n".join(str(item.renderable) for item in view.query("Static"))

    # All disk totals render through the single shared format_mib formatter
    # (TASK-596 delta port), which always renders MiB -- 2 MiB and 4 MiB
    # chosen so the expected strings are distinct and hand-verifiable.
    assert "2.0 MiB staging" in text
    assert "4.0 MiB free" in text


@pytest.mark.asyncio
async def test_mounted_install_progress_updates_without_recomposing_inventory(
    tmp_path: Path,
) -> None:
    """Frequent byte events mutate the progress widget, not every model row."""
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")
    progress = AcquisitionProgress(
        "fetch",
        reference,
        "encoder.onnx",
        512,
        1024,
    )
    view = InstalledView(service_factory=MagicMock(), legacy_dir=tmp_path)
    app = _InstalledApp(view)
    async with app.run_test() as pilot:
        view.set_install_state(None, active=True)
        await pilot.pause()
        view.refresh = MagicMock()
        view.set_install_state(progress, active=True)
        await pilot.pause()
        text = "\n".join(str(item.renderable) for item in view.query("Static"))

    view.refresh.assert_not_called()
    assert "Downloading" in text
    assert "encoder.onnx" in text


def test_forced_refresh_queues_behind_an_inflight_inventory_load(tmp_path: Path) -> None:
    """A lifecycle completion cannot lose its mandatory post-operation refresh."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactDiskUsage
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    view = InstalledView(service_factory=MagicMock(), legacy_dir=tmp_path)
    view._loading = True
    view._load_inventory = MagicMock()
    view.refresh = MagicMock()

    view.ensure_loaded(force=True)
    view._apply_inventory(
        (),
        ArtifactDiskUsage(installed_bytes=0, staging_bytes=0, free_bytes=4096),
        None,
    )

    view._load_inventory.assert_called_once_with()
    assert view._loading is True


@pytest.mark.asyncio
async def test_curated_view_performs_no_io_at_compose_time(tmp_path: Path) -> None:
    """Curated is also eagerly mounted but remains idle until selected."""
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView

    service_factory = MagicMock()
    registry_factory = MagicMock()
    view = CuratedView(
        service_factory=service_factory,
        registry_factory=registry_factory,
    )
    app = _InstalledApp(view)
    async with app.run_test() as pilot:
        await pilot.pause()

    service_factory.assert_not_called()
    registry_factory.assert_not_called()


def test_curated_progress_tolerates_recompose_gap() -> None:
    """A progress event is retained while its widget is temporarily absent.

    ``apply_progress`` (called only by the host screen, ``LLMScreen`` --
    see its own docstring for why ``CuratedView`` no longer renders
    itself in response to a bubbled ``InstallProgressed`` -- TASK-596
    delta port fix round 1) shares this exact tolerance with the
    self-listening handler it replaced.
    """
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView

    progress = AcquisitionProgress(
        "fetch",
        ArtifactRef("parakeet-v2", "immutable-revision", "int8"),
        "encoder.onnx",
        1,
        2,
    )
    view = CuratedView(service_factory=MagicMock(), registry_factory=MagicMock())
    view.query_one = MagicMock(side_effect=NoMatches)
    view.refresh = MagicMock()

    view.apply_progress(progress)

    assert view._progress is progress
    view.refresh.assert_called_once_with(recompose=True)


def test_models_rail_lists_curated_and_installed_and_drops_local_models() -> None:
    """Phase 1 replaces Local Models while preserving the legacy downloader."""
    from tldw_chatbook.UI.Screens.llm_screen import MODELS_RAIL_SECTIONS

    models_section = dict(MODELS_RAIL_SECTIONS)["Models"]
    keys = [key for key, _label in models_section]
    assert "curated" in keys
    assert "installed" in keys
    assert "local-models" not in keys
    assert "download-models" in keys
