"""Provider recovery retains the Import draft through real picker/worker returns."""

from threading import Event
from unittest.mock import Mock

import pytest
from textual.widgets import Button, Input, Select, Static

from Tests.UI.test_library_ingest_queue_journeys import _queue_host, _stage_draft
from Tests.UI.test_library_ingest_resize_focus import _painted
from Tests.UI.test_library_shell import _wait_for_condition
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Third_Party.textual_fspicker import FileOpen
from tldw_chatbook.Third_Party.textual_fspicker.base_dialog import InputBar
from tldw_chatbook.UI.Screens import library_screen as library_module
from tldw_chatbook.Widgets.Library.library_ingest_canvas import LibraryIngestCanvas


def _failed_job(registry, tmp_path):
    job = registry.submit(source_path=str(tmp_path / "failed.wav"))
    registry.mark_failed(
        job.job_id,
        error="Local transcription model unavailable.",
        error_detail={
            "category": "stt_failure",
            "actions": ["choose_another_gguf", "retry_faster_whisper"],
        },
    )
    return job


async def _audio_draft(screen, pilot, source):
    audio = source.with_suffix(".wav")
    audio.write_bytes(b"Audio preflight fixture; never transcribed.")
    title = await _stage_draft(screen, pilot, audio)
    screen.query_one("#type-group-audio_video").collapsed = False
    screen.query_one(
        "#opt-audio_video-transcription_provider", Select
    ).value = "transcribe-cpp"
    await pilot.pause()
    title.selection = type(title.selection)(2, 7)
    return title, audio


async def _open_picker(screen, host, pilot, selector):
    for _ in range(64):
        if screen.focused is screen.query_one(selector):
            break
        await pilot.press("tab")
    button = screen.query_one(selector, Button)
    await _wait_for_condition(
        pilot,
        lambda: screen.focused is button and "GGUF" in _painted(host, button),
        message=lambda: f"GGUF focus={screen.focused!r}; region={button.region}",
    )
    # Tab legitimately selects an Input's text; establish the retained range
    # after traversing the form and before opening the modal.
    title = screen.query_one("#library-ingest-title", Input)
    title.selection = type(title.selection)(2, 7)
    await pilot.press("enter")
    await _wait_for_condition(
        pilot, lambda: isinstance(host.screen, FileOpen), message="GGUF picker missing"
    )
    return host.screen


async def _choose(picker, pilot, path):
    field = picker.query_one(InputBar).query_one(Input)
    field.focus()
    field.value = str(path)
    await pilot.pause()
    await pilot.press("enter")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "size,theme", [((170, 48), "textual-dark"), ((80, 24), "textual-light")]
)
@pytest.mark.parametrize("entry", ["row", "form"])
@pytest.mark.parametrize("outcome", ["success", "reject", "cancel"])
async def test_gguf_picker_result_preserves_import_context(
    tmp_path, monkeypatch, size, theme, entry, outcome
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    source, app, host = _queue_host(tmp_path, theme)
    registry = app.library_ingest_jobs
    failed = _failed_job(registry, tmp_path)
    chosen = tmp_path / "model.gguf"
    chosen.write_bytes(b"Admission is replaced at its external boundary.")
    configure = Mock(
        side_effect=ValueError("fixture rejection") if outcome == "reject" else None
    )
    monkeypatch.setattr(
        library_module, "configure_transcribe_cpp_model_path", configure
    )
    retry = Mock(side_effect=registry.requeue)
    app.retry_library_ingest_job = retry
    async with host.run_test(size=size) as pilot:
        screen = host.screen
        applied = Mock(wraps=screen._apply_transcribe_cpp_gguf_result)
        monkeypatch.setattr(screen, "_apply_transcribe_cpp_gguf_result", applied)
        title, audio = await _audio_draft(screen, pilot, source)
        canvas = screen.query_one(LibraryIngestCanvas)
        selection = title.selection
        selector = (
            f"#library-ingest-choose-gguf-{failed.job_id}"
            if entry == "row"
            else "#opt-audio_video-choose-transcribe-cpp-gguf"
        )
        picker = await _open_picker(screen, host, pilot, selector)
        if outcome == "cancel":
            await pilot.press("escape")
        else:
            await _choose(picker, pilot, chosen)
            await _wait_for_condition(
                pilot, lambda: applied.called, message="Admission result not delivered"
            )
        await pilot.pause()
        assert host.screen is screen
        assert screen.query_one(LibraryIngestCanvas) is canvas
        assert screen.query_one("#library-ingest-title") is title
        assert title.value == "Unsaved next import"
        assert title.selection == selection
        assert screen._ingest_state.form.path == str(audio)
        configured = outcome == "success"
        assert screen._transcribe_cpp_configured is configured
        status = screen.query_one("#opt-audio_video-transcribe-cpp-status", Static)
        assert str(status.render()) == (
            "Local GGUF configured." if configured else "No local GGUF configured."
        )
        if configured and entry == "row":
            retry.assert_called_once_with(failed.job_id)
            assert len(registry.jobs()) == 1
            assert registry.jobs()[0].job_id != failed.job_id
            target = screen.query_one("#library-ingest-path")
            label = _painted(host, target).strip()
            assert label and label in str(audio)
        else:
            retry.assert_not_called()
            target = screen.query_one(selector)
            label = "GGUF"
        assert screen.focused is target
        assert label in _painted(host, target)
        if outcome == "cancel":
            configure.assert_not_called()
        else:
            configure.assert_called_once_with(chosen)


@pytest.mark.asyncio
@pytest.mark.parametrize("destination", ["author", "hub"])
async def test_late_gguf_success_respects_newer_focus_and_navigation(
    tmp_path, monkeypatch, destination
):
    source, app, host = _queue_host(tmp_path, "textual-dark")
    failed = _failed_job(app.library_ingest_jobs, tmp_path)
    entered, release = Event(), Event()

    def configure(_path):
        entered.set()
        assert release.wait(10), "Test did not release admission"

    monkeypatch.setattr(
        library_module, "configure_transcribe_cpp_model_path", configure
    )
    app.retry_library_ingest_job = Mock(side_effect=app.library_ingest_jobs.requeue)
    chosen = tmp_path / "model.gguf"
    chosen.write_bytes(b"Fixture")
    async with host.run_test(size=(80, 24)) as pilot:
        screen = host.screen
        title, _ = await _audio_draft(screen, pilot, source)
        picker = await _open_picker(
            screen, host, pilot, f"#library-ingest-choose-gguf-{failed.job_id}"
        )
        try:
            await _choose(picker, pilot, chosen)
            await _wait_for_condition(
                pilot, entered.is_set, message="Admission not entered"
            )
            worker = next(
                w for w in screen.workers if w.group == "library_transcribe_cpp_gguf"
            )
            if destination == "hub":
                await pilot.press("escape")
                target = screen.query_one("#library-hub-action-import")
            else:
                target = screen.query_one("#library-ingest-author")
            target.focus()
            await pilot.pause()
            release.set()
            await worker.wait()
            await pilot.pause()
            assert target.is_attached
            assert screen.focused is target
            assert _painted(host, target).strip()
            if destination == "author":
                assert screen.query_one("#library-ingest-title") is title
                assert title.value == "Unsaved next import"
        finally:
            release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "size,theme", [((170, 48), "textual-dark"), ((80, 24), "textual-light")]
)
async def test_explicit_faster_whisper_recovery_keeps_draft_and_retry_lineage(
    tmp_path, monkeypatch, size, theme
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    source, app, host = _queue_host(tmp_path, theme)
    registry = app.library_ingest_jobs
    failed = _failed_job(registry, tmp_path)
    top_up = Mock()  # Only execution is replaced; routing and requeue are real.
    app._top_up_ingest_parse_pool = top_up
    app.retry_library_ingest_job = lambda *a, **kw: TldwCli.retry_library_ingest_job(
        app, *a, **kw
    )
    app.retry_library_ingest_job_with_provider = lambda *a, **kw: (
        TldwCli.retry_library_ingest_job_with_provider(app, *a, **kw)
    )
    async with host.run_test(size=size) as pilot:
        screen = host.screen
        title, audio = await _audio_draft(screen, pilot, source)
        selection = title.selection
        selector = f"#library-ingest-retry-faster-whisper-{failed.job_id}"
        for _ in range(64):
            if screen.focused is screen.query_one(selector):
                break
            await pilot.press("tab")
        button = screen.query_one(selector)
        assert screen.focused is button
        assert "Retry with faster-whisper" in _painted(host, button)
        title.selection = selection
        await pilot.press("enter")
        await pilot.pause()
        top_up.assert_called_once_with()
        jobs = registry.jobs()
        assert len(jobs) == 1 and jobs[0].job_id != failed.job_id
        assert jobs[0].source_path == failed.source_path
        assert (
            jobs[0].ingest_options["audio_video"]["transcription_provider"]
            == "faster-whisper"
        )
        assert screen.query_one("#library-ingest-title") is title
        assert title.value == "Unsaved next import" and title.selection == selection
        assert screen._ingest_state.form.path == str(audio)
        assert screen.focused is screen.query_one("#library-ingest-path")
        painted_path = _painted(host, screen.focused).strip()
        assert painted_path and painted_path in str(audio)
