"""Tests for the Library curated Parakeet install surface."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from textual.app import App
from textual.widgets import Button, Static

from tldw_chatbook.Library.library_ingest_state import LibraryIngestFormState
from tldw_chatbook.UI.Screens.library_screen import (
    LibraryScreen,
    ParakeetV2InstallModal,
)


class _ModalApp(App):
    def compose(self):
        return []


@pytest.mark.asyncio
async def test_install_modal_shows_consent_details_and_confirms(tmp_path: Path) -> None:
    destination = tmp_path / "parakeet-v2"
    app = _ModalApp()
    async with app.run_test() as pilot:
        captured: list[bool] = []
        await app.push_screen(
            ParakeetV2InstallModal(destination),
            lambda result: captured.append(result),
        )
        await pilot.pause()

        text = "\n".join(
            str(static.renderable) for static in app.screen.query(Static)
        )
        assert "istupakov/parakeet-tdt-0.6b-v2-onnx" in text
        assert "0bbb45a3365852604aef28b538a8f066f4ccaa85" in text
        assert "CC-BY-4.0" in text
        assert "630.6 MiB" in text
        assert str(destination) in text

        await pilot.click(app.screen.query_one("#parakeet-v2-install-confirm", Button))
        await pilot.pause()
        assert captured == [True]


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


def test_confirmation_starts_background_installer_worker() -> None:
    screen = object.__new__(LibraryScreen)
    screen._parakeet_v2_install_worker = None
    screen.app_instance = MagicMock()
    expected_worker = MagicMock()
    screen._run_parakeet_v2_install = MagicMock(return_value=expected_worker)

    screen._confirm_parakeet_v2_install(True)

    screen._run_parakeet_v2_install.assert_called_once_with()
    assert screen._parakeet_v2_install_worker is expected_worker
