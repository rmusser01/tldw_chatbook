"""Production-CSS folder-picker journeys; optional packages are UI-simulated."""

from copy import deepcopy
from dataclasses import replace

import pytest
from textual.widgets import Button, Collapsible, Input, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_ingest_entry_journeys import _painted
from Tests.UI.test_library_prompt_collection_journeys import _focus
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _seed_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Library.library_ingest_jobs import LibraryIngestJobRegistry
from tldw_chatbook.Third_Party.textual_fspicker import SelectDirectory
from tldw_chatbook.UI.Screens import library_screen as screen_module
from tldw_chatbook.Widgets.Library import library_ingest_canvas as canvas_module

DIRECTORY = "#opt-audio_video-transcription_model_dir"
BROWSE = f"{DIRECTORY}-browse"


async def _tab_to(screen, host, pilot, selector, label):
    for _ in range(40):
        if screen.focused is screen.query_one(selector):
            break
        await pilot.press("tab")
    await _focus(screen, host, pilot, selector, label)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("choose", ["select", "cancel", "same"])
async def test_model_directory_picker_preserves_draft_and_keyboard_context(
    tmp_path, monkeypatch, size, theme, choose
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    # Enable only the UI controls; no model, installer or ingestion is invoked.
    real_installed = canvas_module._is_installed
    monkeypatch.setattr(
        canvas_module,
        "_is_installed",
        lambda feature: (
            feature in {"audio_processing", "parakeet_onnx"} or real_installed(feature)
        ),
    )
    # Keep the real picker, directory listing and validation inside the fixture.
    monkeypatch.setattr(
        screen_module,
        "SelectDirectory",
        lambda _location, **kwargs: SelectDirectory(tmp_path, **kwargs),
    )
    selected = tmp_path / "model folder"
    selected.mkdir()
    source = tmp_path / "audio.wav"
    source.write_bytes(b"Synthetic preflight-only audio placeholder")
    app = _build_test_app()
    _seed_conversations(app, [], media=[])
    app.media_db = MediaDatabase(str(tmp_path / "media.db"), client_id="directory-ui")
    app.library_ingest_jobs = LibraryIngestJobRegistry()
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        screen = host.screen
        await _wait_for_library_shell(screen, pilot)
        await pilot.press("i")
        await _wait_for_condition(
            pilot,
            lambda: bool(screen.query("#library-ingest-path")),
            message="Import entry did not mount",
        )
        screen.query_one("#library-ingest-path", Input).value = str(source)
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._ingest_state.form.preflight is not None
                and "audio_video" in screen._ingest_state.form.preflight.type_groups
                and bool(screen.query("#type-group-audio_video"))
            ),
            message="Audio preflight did not settle",
        )
        screen.query_one("#type-group-audio_video", Collapsible).collapsed = False
        provider = await _wait_for_selector(
            screen, pilot, "#opt-audio_video-transcription_provider"
        )
        provider.value = "parakeet-onnx"
        await _wait_for_condition(
            pilot,
            lambda: not screen.query_one(BROWSE, Button).disabled,
            message="Parakeet controls did not enable",
        )
        title = screen.query_one("#library-ingest-title", Input)
        title.value = "Unsaved title"
        title.cursor_position = 4
        field = screen.query_one(DIRECTORY, Input)
        field.value = str(selected) if choose == "same" else "prior-model"
        initial_label = selected.name if choose == "same" else "prior-model"
        if choose == "same":
            await pilot.pause()
            # Deterministic tooling warning, independent of installed packages.
            screen._ingest_state.form.preflight = replace(
                screen._ingest_state.form.preflight,
                warnings=[
                    {
                        "feature": "audio_processing",
                        "label": "Audio processing",
                        "hint": "Missing fixture tooling",
                        "command": "",
                    }
                ],
            )
            screen.query_one("#library-ingest-start", Button).focus()
            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                lambda: screen._ingest_state.start_consent is not None,
                message="First Start press did not arm warning consent",
            )
            gate = screen.query_one("#library-ingest-start-quiet-line", Static)
            assert gate.has_class("-ingest-start-confirm")
        field.focus()
        await _focus(screen, host, pilot, DIRECTORY, initial_label)
        await pilot.press("tab")
        await _focus(screen, host, pilot, BROWSE, "Browse")
        browse = screen.query_one(BROWSE, Button)
        assert field.parent is browse.parent
        assert field.region.right <= browse.region.x
        assert browse.region.right <= field.parent.region.right <= size[0]
        await pilot.press("shift+tab")
        await _focus(screen, host, pilot, DIRECTORY, initial_label)
        await pilot.press("tab")
        draft = deepcopy(screen._ingest_state.form)
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: isinstance(host.screen, SelectDirectory),
            message="Browse did not open directory picker",
        )
        modal = host.screen
        modal.query_one("#path_input", Input).value = str(selected)
        if choose != "cancel":
            await _tab_to(modal, host, pilot, "#select", "Select")
            await pilot.press("enter")
            draft.type_options["audio_video"]["transcription_model_dir"] = str(selected)
        else:
            await pilot.press("escape")
        await _wait_for_condition(
            pilot,
            lambda: host.screen is screen,
            message="Picker did not return to import",
        )
        await _focus(screen, host, pilot, BROWSE, "Browse")
        assert screen._ingest_state.form == draft
        if choose == "same":
            assert screen._ingest_state.start_consent is None
            assert not gate.has_class("-ingest-start-confirm")
            assert "Press Start again" not in str(gate.renderable)
        assert screen.query_one("#library-ingest-title", Input).cursor_position == 4
        assert screen.query_one("#library-ingest-title") is title
        assert screen.query_one(DIRECTORY) is field
        assert title.value == "Unsaved title"
        assert field.value == (str(selected) if choose != "cancel" else "prior-model")
        await pilot.press("shift+tab")
        await _focus(
            screen,
            host,
            pilot,
            DIRECTORY,
            selected.name if choose != "cancel" else "prior-model",
        )
        await pilot.press("end", "x")
        assert screen._ingest_state.form.type_options["audio_video"][
            "transcription_model_dir"
        ].endswith("x")
        assert app.library_ingest_jobs.jobs() == ()
        assert source.read_bytes() == b"Synthetic preflight-only audio placeholder"
        assert list(selected.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_missing_parakeet_explains_disabled_directory_controls(
    monkeypatch, size, theme
):
    from Tests.UI.test_library_ingest_canvas import _default_form
    from Tests.UI.test_library_ingest_entry_journeys import _IngestHost
    from tldw_chatbook.Library.ingest_types import PreflightResult
    from tldw_chatbook.Library.library_ingest_state import build_library_ingest_state

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(canvas_module, "_is_installed", lambda _feature: False)
    form = _default_form()
    form.expanded_type_groups.add("audio_video")
    form.type_options = {"audio_video": {"transcription_provider": "parakeet-onnx"}}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=PreflightResult(
            {"audio_video": ["/fixture/audio.wav"]}, [], [], 1, False, 1
        ),
    )
    host = _IngestHost(state)
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        field = host.query_one(DIRECTORY, Input)
        browse = host.query_one(BROWSE, Button)
        siblings = list(field.parent.parent.children)
        label = siblings[siblings.index(field.parent) - 1]
        label.scroll_visible(animate=False)
        await _wait_for_condition(
            pilot,
            lambda: "installed" in _painted(host, label),
            message="Missing-package explanation was not painted",
        )
        assert "Local Parakeet model folder" in _painted(host, label)
        assert "Parakeet" in _painted(host, label)
        assert field.disabled and browse.disabled
        assert not field.focusable and not browse.focusable
