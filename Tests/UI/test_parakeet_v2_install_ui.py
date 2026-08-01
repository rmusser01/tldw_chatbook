"""Tests for the Library curated Parakeet install surface.

TASK-1696: the consent modal now renders from an immutable
``PreflightReport`` instead of hard-coded installer constants, and the
Library screen resolves that report in a background preflight worker
before ever showing the modal. The modal's control ids, the confirm-step
worker's zero-argument call contract, and the post-install batch-selection
side effect are all pinned here exactly as they were before this task --
only the modal's construction and the trigger that precedes it changed.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from textual.app import App
from textual.widgets import Button, Static

from tldw_chatbook.Library.library_ingest_state import LibraryIngestFormState
from tldw_chatbook.Model_Artifacts import ArtifactRef
from tldw_chatbook.Model_Artifacts.acquisition import ArtifactPreflightEntry, PreflightReport
from tldw_chatbook.UI.Screens.library_screen import (
    LibraryScreen,
    ParakeetV2InstallModal,
)


class _ModalApp(App):
    def compose(self):
        return []


def _report(
    *,
    destination: Path,
    download_bytes: int = 661_191_781,
    free_bytes: int = 10**12,
    required_bytes: int = 900_000_000,
    sufficient_space: bool = True,
    gating_errors: tuple[str, ...] = (),
    already_installed: bool = False,
) -> PreflightReport:
    """Build a real ``PreflightReport`` with the same shape ``run_parakeet_v2_preflight`` returns."""
    ref = ArtifactRef("parakeet-v2", "0bbb45a3365852604aef28b538a8f066f4ccaa85", "int8")
    entry = ArtifactPreflightEntry(
        ref=ref,
        source_url=(
            "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/"
            "0bbb45a3365852604aef28b538a8f066f4ccaa85/config.json"
        ),
        repository="istupakov/parakeet-tdt-0.6b-v2-onnx",
        revision="0bbb45a3365852604aef28b538a8f066f4ccaa85",
        license_id="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        precision="int8",
        total_bytes=download_bytes,
        file_count=4,
        already_installed=already_installed,
    )
    return PreflightReport(
        root=ref,
        closure_fingerprint="f" * 64,
        entries=(entry,),
        download_bytes=download_bytes,
        already_staged_bytes=0,
        staging_overhead_bytes=0,
        retained_bytes=0,
        destination=destination,
        free_bytes=free_bytes,
        required_bytes=required_bytes,
        sufficient_space=sufficient_space,
        gating_errors=gating_errors,
    )


# ---------------------------------------------------------------------------
# Modal: renders from the report, not hard-coded constants.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_install_modal_shows_consent_details_from_report_and_confirms(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "parakeet-v2"
    report = _report(destination=destination)
    app = _ModalApp()
    async with app.run_test() as pilot:
        captured: list[bool] = []
        await app.push_screen(
            ParakeetV2InstallModal(report),
            lambda result: captured.append(result),
        )
        await pilot.pause()

        text = "\n".join(
            str(static.renderable) for static in app.screen.query(Static)
        )
        assert "istupakov/parakeet-tdt-0.6b-v2-onnx" in text
        assert "0bbb45a3365852604aef28b538a8f066f4ccaa85" in text
        assert "CC-BY-4.0" in text
        assert "int8" in text
        assert str(destination) in text
        assert "Enough free space" in text
        assert app.screen.query_one("#parakeet-v2-install-confirm", Button).disabled is False

        await pilot.click(app.screen.query_one("#parakeet-v2-install-confirm", Button))
        await pilot.pause()
        assert captured == [True]


@pytest.mark.asyncio
async def test_install_modal_renders_values_from_the_injected_report(
    tmp_path: Path,
) -> None:
    """Proves the modal derives its content from the report: distinct
    destination and byte totals that differ from the legacy hard-coded
    ``PARAKEET_V2_TOTAL_BYTES``/``parakeet_v2_install_dir()`` constants must
    appear verbatim -- this could not pass if the modal still hard-coded
    those constants.
    """
    destination = tmp_path / "custom-report-destination"
    report = _report(
        destination=destination,
        download_bytes=123_456_789,
        free_bytes=999_999_999,
        required_bytes=200_000_000,
    )
    app = _ModalApp()
    async with app.run_test() as pilot:
        await app.push_screen(ParakeetV2InstallModal(report), lambda result: None)
        await pilot.pause()

        text = "\n".join(
            str(static.renderable) for static in app.screen.query(Static)
        )
        assert str(destination) in text
        assert f"{123_456_789 / (1024 * 1024):.1f}" in text
        assert f"{999_999_999 / (1024 * 1024):.1f}" in text


@pytest.mark.asyncio
async def test_install_modal_surfaces_gating_errors_and_disables_confirm(
    tmp_path: Path,
) -> None:
    report = _report(
        destination=tmp_path / "d",
        gating_errors=(
            "istupakov/parakeet-tdt-0.6b-v2-onnx requires a credential: set "
            "HUGGINGFACE_API_KEY (or HF_TOKEN)",
        ),
    )
    app = _ModalApp()
    async with app.run_test() as pilot:
        await app.push_screen(ParakeetV2InstallModal(report), lambda result: None)
        await pilot.pause()

        text = "\n".join(
            str(static.renderable) for static in app.screen.query(Static)
        )
        assert "requires a credential" in text
        # Confirming a plan that would immediately fail report.grant() is
        # pre-empted client-side rather than left to a caught background
        # error -- the button stays present (same id) but disabled.
        confirm = app.screen.query_one("#parakeet-v2-install-confirm", Button)
        assert confirm.disabled is True


@pytest.mark.asyncio
async def test_install_modal_shows_insufficient_space_verdict_and_disables_confirm(
    tmp_path: Path,
) -> None:
    report = _report(
        destination=tmp_path / "d",
        free_bytes=1_000,
        required_bytes=900_000_000,
        sufficient_space=False,
    )
    app = _ModalApp()
    async with app.run_test() as pilot:
        await app.push_screen(ParakeetV2InstallModal(report), lambda result: None)
        await pilot.pause()

        text = "\n".join(
            str(static.renderable) for static in app.screen.query(Static)
        )
        assert "Not enough free space" in text
        confirm = app.screen.query_one("#parakeet-v2-install-confirm", Button)
        assert confirm.disabled is True


# ---------------------------------------------------------------------------
# Trigger and worker wiring: preflight first, then the plan-driven modal.
# ---------------------------------------------------------------------------


def test_install_request_triggers_background_preflight_worker() -> None:
    screen = object.__new__(LibraryScreen)
    screen._parakeet_v2_install_worker = None
    screen.app_instance = MagicMock()
    expected_worker = MagicMock()
    screen._run_parakeet_v2_preflight = MagicMock(return_value=expected_worker)
    event = MagicMock()

    LibraryScreen.handle_parakeet_v2_install_requested(screen, event)

    event.stop.assert_called_once_with()
    screen._run_parakeet_v2_preflight.assert_called_once_with()
    assert screen._parakeet_v2_install_worker is expected_worker


def test_install_request_notifies_when_already_running() -> None:
    screen = object.__new__(LibraryScreen)
    busy_worker = MagicMock()
    busy_worker.is_finished = False
    screen._parakeet_v2_install_worker = busy_worker
    screen.app_instance = MagicMock()
    screen._run_parakeet_v2_preflight = MagicMock()
    event = MagicMock()

    LibraryScreen.handle_parakeet_v2_install_requested(screen, event)

    screen._run_parakeet_v2_preflight.assert_not_called()
    screen.app_instance.notify.assert_called_once()


def test_preflight_result_shows_modal_built_from_the_report(
    tmp_path: Path, monkeypatch
) -> None:
    fake_app = MagicMock()
    # ``Screen.app`` is a read-only Textual property; replace it on the
    # class for this test only (``monkeypatch`` reverts automatically).
    monkeypatch.setattr(LibraryScreen, "app", property(lambda self: fake_app))
    screen = object.__new__(LibraryScreen)
    screen._parakeet_v2_install_worker = MagicMock()
    report = _report(destination=tmp_path / "d")

    screen._apply_parakeet_v2_preflight_result(report, None)

    assert screen._parakeet_v2_install_worker is None
    assert screen._parakeet_v2_pending_report is report
    fake_app.push_screen.assert_called_once()
    call_args = fake_app.push_screen.call_args
    modal = call_args[0][0]
    assert isinstance(modal, ParakeetV2InstallModal)
    assert modal.report is report
    assert call_args[0][1] == screen._confirm_parakeet_v2_install


def test_preflight_failure_notifies_and_does_not_push_modal(monkeypatch) -> None:
    fake_app = MagicMock()
    monkeypatch.setattr(LibraryScreen, "app", property(lambda self: fake_app))
    screen = object.__new__(LibraryScreen)
    screen._parakeet_v2_install_worker = MagicMock()
    screen.app_instance = MagicMock()

    screen._apply_parakeet_v2_preflight_result(None, "boom")

    assert screen._parakeet_v2_install_worker is None
    screen.app_instance.notify.assert_called_once()
    fake_app.push_screen.assert_not_called()


def test_confirmation_starts_background_installer_worker() -> None:
    screen = object.__new__(LibraryScreen)
    screen._parakeet_v2_install_worker = None
    screen.app_instance = MagicMock()
    expected_worker = MagicMock()
    screen._run_parakeet_v2_install = MagicMock(return_value=expected_worker)

    screen._confirm_parakeet_v2_install(True)

    screen._run_parakeet_v2_install.assert_called_once_with()
    assert screen._parakeet_v2_install_worker is expected_worker


def test_declining_confirmation_does_not_start_installer_worker() -> None:
    screen = object.__new__(LibraryScreen)
    screen._parakeet_v2_install_worker = None
    screen._run_parakeet_v2_install = MagicMock()

    screen._confirm_parakeet_v2_install(False)

    screen._run_parakeet_v2_install.assert_not_called()


# ---------------------------------------------------------------------------
# Post-install batch selection: unchanged contract.
# ---------------------------------------------------------------------------


def test_successful_install_populates_current_batch_model_folder(
    tmp_path: Path,
) -> None:
    installed = tmp_path / "parakeet-v2"
    screen = object.__new__(LibraryScreen)
    screen._library_ingest_form = LibraryIngestFormState()
    screen._parakeet_v2_install_worker = MagicMock()
    screen.app_instance = MagicMock()
    screen.refresh = MagicMock()

    screen._apply_parakeet_v2_install_result(installed, None)

    options = screen._library_ingest_form.type_options["audio_video"]
    assert options["transcription_provider"] == "parakeet-onnx"
    assert options["transcription_model_dir"] == str(installed)
    screen.app_instance.notify.assert_called_once()
    screen.refresh.assert_called_once_with(recompose=True)
