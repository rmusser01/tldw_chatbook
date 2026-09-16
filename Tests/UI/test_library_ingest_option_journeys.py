"""Import option changes retain editing context under production CSS."""

import pytest
from textual.widgets import Button, Checkbox, Input, Select, Static, TextArea

from Tests.UI.app_factory import _build_test_app
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
from tldw_chatbook.UI.Library_Modules.library_ingest_state import (
    LibraryIngestLastSubmission,
)
from tldw_chatbook.Widgets.Library import library_ingest_canvas as canvas_module


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("action", ["checkbox", "select", "reset", "text"])
async def test_option_actions_preserve_other_editors_and_visible_focus(
    tmp_path, monkeypatch, size, theme, action
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    # Availability is UI-only; these journeys never extract or install anything.
    monkeypatch.setattr(canvas_module, "_is_installed", lambda _feature: True)
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "notes.txt").write_text("Local preflight fixture")
    (sources / "document.pdf").write_bytes(b"%PDF-1.4\nPreflight only\n")
    app = _build_test_app()
    _seed_conversations(app, [], media=[])
    app.media_db = MediaDatabase(str(tmp_path / "media.db"), client_id="options-ui")
    app.library_ingest_jobs = LibraryIngestJobRegistry()
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        screen = host.screen
        await _wait_for_library_shell(screen, pilot)
        await pilot.press("i")
        path = await _wait_for_selector(screen, pilot, "#library-ingest-path")
        path.value = str(sources)
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._ingest_state.form.preflight is not None
                and set(screen._ingest_state.form.preflight.type_groups)
                == {"pdf", "generic"}
            ),
            message="Mixed-source preflight did not settle",
        )
        for group in ("pdf", "generic"):
            panel = await _wait_for_selector(screen, pilot, f"#type-group-{group}")
            panel.collapsed = False
        analyze = await _wait_for_selector(screen, pilot, "#opt-generic-analyze")
        analyze.value = True
        await _wait_for_condition(
            pilot,
            lambda: (
                not screen.query_one("#opt-generic-custom_prompt", TextArea).disabled
            ),
            message="Analysis prompt did not enable",
        )
        if action == "reset":
            screen.query_one("#opt-pdf-pdf_engine", Select).value = "docext"
            await pilot.pause()
            screen.query_one("#opt-pdf-ocr", Checkbox).value = True
            await pilot.pause()
            screen.query_one("#opt-pdf-ocr_language", Input).value = "fr"
            await pilot.pause()
        title = screen.query_one("#library-ingest-title", Input)
        title.value = "Unsaved metadata"
        title.selection = type(title.selection)(2, 7)
        prompt = screen.query_one("#opt-generic-custom_prompt", TextArea)
        prompt.load_text("First line\nSecond line")
        prompt.selection = type(prompt.selection)((0, 2), (1, 4))
        title_selection, prompt_selection = title.selection, prompt.selection
        await pilot.pause()
        screen._ingest_state.last_submission = LibraryIngestLastSubmission(
            source=str(sources / "notes.txt")
        )
        screen._update_library_ingest_dynamic_regions()
        await pilot.pause()
        retry = screen.query_one("#library-ingest-retry-last", Button)
        # Arm the real button directly: this test owns option edits, not queue traversal.
        retry.press()
        await pilot.pause()
        assert screen._ingest_state.retry_confirm_armed
        assert "replace" in str(retry.label)
        selectors = {
            "checkbox": ("#opt-generic-chunk", "Chunk content"),
            "select": ("#opt-generic-encoding", "UTF-8"),
            "reset": ("#opt-pdf-reset", "Reset to defaults"),
            "text": ("#opt-generic-chunk_size", "1000"),
        }
        selector, label = selectors[action]
        control = screen.query_one(selector)
        control.focus()
        await _focus(screen, host, pilot, selector, label)
        if action == "select":
            await pilot.press("enter", "home", "down", "enter")
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._ingest_state.form.type_options["generic"].get("encoding")
                    == "utf-8"
                ),
                message="Encoding selection did not reach form",
            )
        elif action == "text":
            await pilot.press("end", "x")
            await pilot.pause()
        else:
            await pilot.press("space" if action == "checkbox" else "enter")
            await pilot.pause()
        title_now = await _wait_for_selector(screen, pilot, "#library-ingest-title")
        prompt_now = await _wait_for_selector(
            screen, pilot, "#opt-generic-custom_prompt"
        )
        assert not screen._ingest_state.retry_confirm_armed
        assert str(retry.label) == "Retry this batch"
        assert title_now is title and prompt_now is prompt
        assert title_now.value == "Unsaved metadata"
        assert title_now.selection == title_selection
        assert prompt_now.text == "First line\nSecond line"
        assert prompt_now.selection == prompt_selection
        await _focus(screen, host, pilot, selector, label)
        assert screen._ingest_state.form.path == str(sources)
        if action == "checkbox":
            assert screen._ingest_state.form.chunk is False
            assert screen.query_one("#opt-generic-chunk_size", Input).disabled
            assert screen.query_one("#opt-generic-chunk_template", Select).disabled
            assert "needs Chunk content on" in str(
                screen.query_one("#opt-generic-chunk_size-label", Static).renderable
            )
        elif action == "reset":
            assert (
                screen.query_one("#opt-pdf-pdf_engine", Select).value == "pymupdf4llm"
            )
            assert screen.query_one("#opt-pdf-ocr_language", Input).value == "en"
            assert screen.query_one("#opt-pdf-ocr_language", Input).disabled
            assert screen._ingest_state.form.type_options["pdf"] == {}
        assert app.library_ingest_jobs.jobs() == ()


def _painted_content(host, widget):
    region = widget.content_region
    strips = list(host.screen._compositor.render_strips())
    return " ".join(
        " ".join(
            strips[y].crop(max(0, region.x), min(host.size.width, region.right)).text
            for y in range(max(0, region.y), min(len(strips), region.bottom))
        ).split()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("installed", [False, True], ids=["unavailable", "available"])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_option_labels_and_disabled_reasons_fit_compact_panel(
    monkeypatch, installed, theme
):
    from Tests.UI.test_library_ingest_entry_journeys import _IngestHost
    from tldw_chatbook.Library.ingest_types import PreflightResult
    from tldw_chatbook.Library.library_ingest_state import (
        LibraryIngestFormState,
        build_library_ingest_state,
    )

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(canvas_module, "_is_installed", lambda _feature: installed)
    groups = ["pdf", "document", "audio_video", "ebook", "image", "generic", "web"]
    form = LibraryIngestFormState(
        path="/private/fixtures", expanded_type_groups=set(groups)
    )
    # Render-only type inventory; web sources are never fetched.
    preflight = PreflightResult(
        {group: [f"/private/fixtures/{group}"] for group in groups},
        [],
        [],
        0,
        False,
        len(groups),
    )
    state = build_library_ingest_state((), form=form, preflight=preflight)
    host = _IngestHost(state)
    host.theme = theme
    async with host.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        controls = list(
            host.query(
                ".type-group-contents Checkbox, .type-group-contents Button, .type-group-field-label"
            )
        )
        assert all(host.query(f"#type-group-{group}") for group in groups)
        for widget in controls:
            if not widget.display:
                continue
            widget.scroll_visible(animate=False)
            await pilot.pause()
            label = (
                str(widget.label)
                if isinstance(widget, (Checkbox, Button))
                else str(widget.renderable)
            )
            painted = _painted_content(host, widget)
            assert label in painted, (widget.id, label, painted, widget.region)
            assert (
                widget.region.right
                <= widget.parent.content_region.right
                <= host.size.width
            )


@pytest.mark.asyncio
async def test_option_state_updates_do_not_emit_new_user_edits():
    from Tests.UI.test_library_ingest_canvas import _MessageRecordingHost
    from tldw_chatbook.Library.library_ingest_state import (
        LibraryIngestFormState,
        build_library_ingest_state,
    )

    state = build_library_ingest_state((), form=LibraryIngestFormState())
    host = _MessageRecordingHost(state)
    async with host.run_test() as pilot:
        await pilot.pause()
        canvas = host.query_one(canvas_module.LibraryIngestCanvas)
        # Two snapshots can settle before either programmatic Changed is handled.
        # Those echoes must not become a fresh pair of user edits indefinitely.
        for analyze in (True, False):
            form = LibraryIngestFormState(
                analyze=analyze, type_options={"generic": {"analyze": analyze}}
            )
            canvas.sync_option_group(
                "generic", build_library_ingest_state((), form=form)
            )
        await pilot.pause()
        assert host.option_changes == []
        assert not canvas.query_one("#opt-generic-analyze", Checkbox).value
        assert canvas.query_one("#opt-generic-custom_prompt", TextArea).disabled


@pytest.mark.asyncio
async def test_late_option_update_for_removed_group_keeps_current_form():
    from Tests.UI.test_library_ingest_entry_journeys import _IngestHost
    from tldw_chatbook.Library.library_ingest_state import (
        LibraryIngestFormState,
        build_library_ingest_state,
    )

    # Current source no longer has PDF options when an old option event arrives.
    state = build_library_ingest_state(
        (), form=LibraryIngestFormState(title="Current draft")
    )
    host = _IngestHost(state)
    async with host.run_test() as pilot:
        await pilot.pause()
        canvas = host.query_one(canvas_module.LibraryIngestCanvas)
        title = canvas.query_one("#library-ingest-title", Input)
        canvas.sync_option_group("pdf", state)
        await pilot.pause()
        assert canvas.query_one("#library-ingest-title") is title
        assert title.value == "Current draft"
        assert not canvas.query("#type-group-pdf")


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize(
    "group,values,dependent,disabled",
    [
        ("pdf", {"pdf_engine": "docext", "ocr": True}, "ocr_backend", False),
        ("document", {"processing_method": "native"}, "ocr", True),
        (
            "audio_video",
            {
                "transcription_provider": "parakeet-onnx",
                "transcription_model_dir": "retained-folder",
            },
            "transcription_model_dir",
            False,
        ),
        (
            "ebook",
            {"chunk_method": "words", "include_toc": False},
            "chunk_method",
            False,
        ),
        ("image", {"ocr": False, "ocr_language": "fr"}, "ocr_language", True),
        ("generic", {"chunk": False, "chunk_size": "abc"}, "chunk_size", True),
        ("web", {"scrape_method": "sitemap"}, "max_pages", False),
    ],
)
async def test_per_type_updates_keep_dependencies_and_editors_current(
    monkeypatch, theme, group, values, dependent, disabled
):
    from copy import deepcopy

    from Tests.UI.test_library_ingest_entry_journeys import _IngestHost
    from tldw_chatbook.Library.ingest_types import PreflightResult
    from tldw_chatbook.Library.library_ingest_state import (
        LibraryIngestFormState,
        build_library_ingest_state,
    )

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(canvas_module, "_is_installed", lambda _feature: True)
    form = LibraryIngestFormState(
        title="Retained metadata", expanded_type_groups={group}
    )
    preflight = PreflightResult({group: ["/private/source"]}, [], [], 0, False, 1)
    state = build_library_ingest_state((), form=form, preflight=preflight)
    host = _IngestHost(state)
    host.theme = theme
    async with host.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        canvas = host.query_one(canvas_module.LibraryIngestCanvas)
        title = host.query_one("#library-ingest-title", Input)
        title.selection = type(title.selection)(1, 5)
        old_selection = title.selection
        field = host.query_one(f"#opt-{group}-{dependent}")
        updated = deepcopy(form)
        updated.type_options[group] = deepcopy(values)
        if group == "generic":
            updated.chunk = False
            updated.chunk_size = "abc"
        canvas.sync_option_group(
            group, build_library_ingest_state((), form=updated, preflight=preflight)
        )
        await pilot.pause()
        assert field is host.query_one(f"#opt-{group}-{dependent}")
        assert field.disabled is disabled
        assert host.query_one("#library-ingest-title") is title
        assert title.selection == old_selection
        if group == "pdf":
            assert host.query_one("#opt-pdf-pdf_engine", Select).value == "docext"
            assert host.query_one("#opt-pdf-ocr", Checkbox).value
        elif group == "ebook":
            assert field.value == "words"
            assert not host.query_one("#opt-ebook-include_toc", Checkbox).value
        elif group == "image":
            assert field.value == "fr"
            assert host.query_one("#opt-image-ocr_backend", Select).disabled
        elif group == "generic":
            assert field.value == "abc"
            assert not host.query_one("#opt-generic-chunk_size-error").display
            assert not field.has_class("-ingest-option-invalid")
            assert host.query_one("#opt-generic-chunk_template", Select).disabled
        elif group == "web":
            assert host.query_one("#web-local-scope-note").display
        elif group == "audio_video":
            assert field.value == "retained-folder"
            assert not host.query_one(
                "#opt-audio_video-transcription_model_dir-browse", Button
            ).disabled
            assert not host.query_one(
                "#opt-audio_video-install-parakeet-v2", Button
            ).disabled
            chooser = host.query_one(
                "#opt-audio_video-choose-transcribe-cpp-gguf", Button
            )
            assert not chooser.display
            updated.type_options[group]["transcription_provider"] = "transcribe-cpp"
            canvas.sync_option_group(
                group, build_library_ingest_state((), form=updated, preflight=preflight)
            )
            await pilot.pause()
            assert chooser.display
            assert field.disabled
            assert host.query_one(
                "#opt-audio_video-install-parakeet-v2", Button
            ).disabled
            assert "GGUF" in str(chooser.label)
        await pilot.resize_terminal(170, 48)
        await pilot.pause()
        assert host.query_one("#library-ingest-title") is title
        assert title.selection == old_selection


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_compact_library_shell_keeps_full_install_reason(
    tmp_path, monkeypatch, theme
):
    """The compact shell's toolbar height must not clip this option explanation."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    source = tmp_path / "audio.wav"
    source.write_bytes(b"Audio preflight only")
    app = _build_test_app()
    _seed_conversations(app, [], media=[])
    app.media_db = MediaDatabase(str(tmp_path / "media.db"), client_id="install-copy")
    app.library_ingest_jobs = LibraryIngestJobRegistry()
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    async with host.run_test(size=(80, 24)) as pilot:
        screen = host.screen
        await _wait_for_library_shell(screen, pilot)
        await pilot.press("i")
        source_input = await _wait_for_selector(screen, pilot, "#library-ingest-path")
        source_input.value = str(source)
        install = await _wait_for_selector(
            screen, pilot, "#opt-audio_video-install-parakeet-v2"
        )
        screen.query_one("#type-group-audio_video").collapsed = False
        await pilot.pause()
        install.scroll_visible(animate=False)
        await pilot.pause()
        assert screen.query_one("#library-shell-grid").has_class(
            "library-notes-compact"
        )
        assert install.disabled
        assert "needs the parakeet-onnx provider" in str(install.label)
        assert str(install.label) in _painted_content(host, install)
