"""Focused tests for the managed-model Installed view."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult
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


def test_installed_view_refuses_a_second_lifecycle_operation() -> None:
    """Activation/deletion cannot re-enter while hashing or leasing is pending."""
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    view = InstalledView(service_factory=MagicMock(), legacy_dir=Path("/tmp/models"))
    view._operation_reference = ArtifactRef("parakeet-v2", "rev1", "int8")
    view._activate_model = MagicMock()

    view._request_activation(ArtifactRef("parakeet-v2", "rev2", "f32"))

    view._activate_model.assert_not_called()


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
            ArtifactDiskUsage(installed_bytes=0, staging_bytes=2048, free_bytes=4096),
            None,
        )
        await pilot.pause()
        text = "\n".join(str(item.renderable) for item in view.query("Static"))

    assert "2.0 KiB staging" in text
    assert "4.0 KiB free" in text


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


def test_curated_preflight_result_opens_the_shared_modal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Selection shows the exact shared consent plan before acquisition."""
    from Tests.UI.test_model_artifact_widgets import _report
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView
    from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallModal

    fake_app = MagicMock()
    monkeypatch.setattr(CuratedView, "app", property(lambda self: fake_app))
    view = CuratedView(service_factory=MagicMock(), registry_factory=MagicMock())
    report = _report(tmp_path / "managed")
    view._operation_reference = report.root

    view._apply_preflight_result(report, None)

    assert view._pending_report is report
    modal, callback = fake_app.push_screen.call_args[0]
    assert isinstance(modal, ModelInstallModal)
    assert callback == view._confirm_install


def test_models_rail_lists_curated_and_installed_and_drops_local_models() -> None:
    """Phase 1 replaces Local Models while preserving the legacy downloader."""
    from tldw_chatbook.UI.Screens.llm_screen import MODELS_RAIL_SECTIONS

    models_section = dict(MODELS_RAIL_SECTIONS)["Models"]
    keys = [key for key, _label in models_section]
    assert "curated" in keys
    assert "installed" in keys
    assert "local-models" not in keys
    assert "download-models" in keys
