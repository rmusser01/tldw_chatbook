"""Submission snapshots are checked before queue creation and fenced on edits."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

pytestmark = pytest.mark.bootstrap_profile


def _screen(backend="local"):
    queued, checks, notices = [], [], []
    form = SimpleNamespace(
        title="",
        author="",
        keywords="",
        analyze=False,
        chunk=False,
        chunk_size=1000,
        path="clip.mp4",
    )
    snapshot = {"audio_video": {"transcription_provider": "faster-whisper"}}
    screen = SimpleNamespace(
        app_instance=SimpleNamespace(
            submit_library_ingest_job=lambda **kwargs: queued.append(kwargs),
            _resolve_ingest_backend=lambda: backend,
            notify=lambda message, **kwargs: notices.append(message),
        ),
        _ingest_state=SimpleNamespace(form=form),
        _build_ingest_options_snapshot=lambda: snapshot,
        _safe_text=lambda value, **kwargs: value,
        _enqueue_library_ingest_snapshot=lambda kwargs: queued.append(kwargs),
        _notify_library_ingest_warning=lambda message: notices.append(message),
        _library_external_submit_scope_id=None,
        _library_external_submit_generation=0,
        _set_library_external_status=lambda *args, **kwargs: None,
        _check_library_transcription_submission=lambda *args: checks.append(args),
        _library_external_backend_is_current=lambda: backend == "local",
        _invalidate_library_external_submission=lambda: None,
    )
    screen._do_submit_ingest = lambda *args, **kwargs: LibraryScreen._do_submit_ingest(
        screen, *args, **kwargs
    )
    return screen, queued, checks, notices


def test_local_submission_waits_for_selected_backend_readiness():
    screen, queued, checks, _ = _screen()
    screen._do_submit_ingest("clip.mp4")
    assert not queued
    assert len(checks) == 1


def test_server_submission_does_not_probe_local_transcription():
    screen, queued, checks, _ = _screen("server")
    screen._do_submit_ingest("clip.mp4")
    assert len(queued) == 1
    assert not checks


@pytest.mark.parametrize(
    "failure", [None, "Install the missing transcription package."]
)
def test_readiness_result_enqueues_only_success_and_keeps_failure_actionable(failure):
    screen, queued, checks, notices = _screen()
    screen._do_submit_ingest("clip.mp4")
    assert not queued
    LibraryScreen._apply_library_transcription_readiness(screen, *checks[0], failure)
    assert len(queued) == (failure is None)
    if failure:
        assert failure in notices
        assert screen._ingest_state.form.path == "clip.mp4"


def test_old_readiness_result_cannot_queue_after_a_new_submission_generation():
    screen, queued, checks, notices = _screen()
    screen._do_submit_ingest("clip.mp4")
    screen._library_external_submit_generation += 1
    LibraryScreen._apply_library_transcription_readiness(screen, *checks[0], None)
    assert not queued
    assert not notices


def test_changed_snapshot_cannot_queue_after_readiness_finishes():
    screen, queued, checks, _ = _screen()
    screen._do_submit_ingest("clip.mp4")
    screen._build_ingest_options_snapshot = lambda: {
        "audio_video": {"transcription_provider": "parakeet-onnx"}
    }
    LibraryScreen._apply_library_transcription_readiness(screen, *checks[0], None)
    assert not queued


@pytest.mark.parametrize("field", ["title", "author", "keywords"])
def test_unchanged_metadata_mount_echo_does_not_cancel_readiness(field):
    from tldw_chatbook.UI.Library_Modules.library_ingest_controller import (
        LibraryIngestController,
    )

    cancelled = []
    screen = SimpleNamespace(
        _library_ingest_form=SimpleNamespace(**{field: "existing metadata"}),
        _invalidate_library_external_submission=lambda: cancelled.append(True),
        _library_ingest_start_consent=None,
    )
    event = SimpleNamespace(value="existing metadata", stop=lambda: None)
    handler = getattr(LibraryIngestController, f"handle_library_ingest_{field}_changed")
    handler(screen, event)
    assert not cancelled
    event.value = "edited metadata"
    handler(screen, event)
    assert cancelled == [True]


def test_readiness_worker_uses_admission_scan_limit_and_skips_non_audio(monkeypatch):
    from tldw_chatbook.UI.Screens import library_screen as module
    from tldw_chatbook.Library import transcription_readiness as readiness

    screen, _, checks, _ = _screen()
    screen._do_submit_ingest("folder")
    scans, applied = [], []
    screen.app = SimpleNamespace(
        call_from_thread=lambda callback, *args: applied.append(args)
    )
    screen._apply_library_transcription_readiness = lambda *args: None
    monkeypatch.setattr(
        module, "get_current_worker", lambda: SimpleNamespace(is_cancelled=False)
    )
    monkeypatch.setattr(module, "get_cli_setting", lambda key, *args: 1700)
    monkeypatch.setattr(
        module,
        "analyze_path",
        lambda source, **kwargs: (
            scans.append(kwargs) or SimpleNamespace(type_groups={"generic": [source]})
        ),
    )
    monkeypatch.setattr(
        readiness,
        "check_transcription_readiness",
        lambda *args, **kwargs: pytest.fail("text imports must not probe STT"),
    )
    LibraryScreen._check_library_transcription_submission.__wrapped__(
        screen, *checks[0]
    )
    assert scans == [{"scan_limit": 1700, "probe_url": False}]
    assert applied[0][-1] is None


@pytest.mark.parametrize(
    "reason,message",
    [
        (
            "package_missing",
            "Install the selected runtime in Chatbook's Python environment.",
        ),
        ("model_missing", "Download the selected model files before retrying."),
        ("vad_missing", "Install the managed Silero VAD dependency before retrying."),
    ],
)
def test_worker_failure_reaches_visible_status_without_queueing(
    monkeypatch, reason, message
):
    from tldw_chatbook.UI.Screens import library_screen as module
    from tldw_chatbook.Library import transcription_readiness as readiness

    screen, queued, checks, notices = _screen()
    statuses = []
    screen._set_library_external_status = lambda value, **kwargs: statuses.append(value)
    screen.app = SimpleNamespace(
        call_from_thread=lambda callback, *args: callback(*args)
    )
    screen._apply_library_transcription_readiness = (
        lambda *args: LibraryScreen._apply_library_transcription_readiness(
            screen, *args
        )
    )
    monkeypatch.setattr(
        module, "get_current_worker", lambda: SimpleNamespace(is_cancelled=False)
    )
    monkeypatch.setattr(module, "get_cli_setting", lambda key, default=None: default)
    monkeypatch.setattr(
        module,
        "analyze_path",
        lambda *args, **kwargs: SimpleNamespace(
            type_groups={"audio_video": ["clip.mp4"]}
        ),
    )

    def unavailable(*args, **kwargs):
        raise readiness.TranscriptionReadinessError(reason, message)

    monkeypatch.setattr(readiness, "check_transcription_readiness", unavailable)
    screen._do_submit_ingest("clip.mp4")
    LibraryScreen._check_library_transcription_submission.__wrapped__(
        screen, *checks[0]
    )
    assert queued == []
    assert notices == [message]
    assert statuses[-1] == message
    assert screen._ingest_state.form.path == "clip.mp4"


@pytest.mark.asyncio
async def test_metadata_recompose_keeps_pending_readiness_alive():
    from Tests.UI.test_library_ingest_canvas import _CanvasHost
    from tldw_chatbook.Library.library_ingest_state import (
        LibraryIngestFormState,
        build_library_ingest_state,
    )
    from tldw_chatbook.UI.Library_Modules.library_ingest_controller import (
        LibraryIngestController,
    )
    from textual.widgets import Input

    form = LibraryIngestFormState(title="A title", author="An author", keywords="a,b")
    cancelled = []
    controller = SimpleNamespace(
        _library_ingest_form=form,
        _library_ingest_start_consent=None,
        _invalidate_library_external_submission=lambda: cancelled.append(True),
    )

    class Host(_CanvasHost):
        def on_input_changed(self, event):
            field = (event.input.id or "").removeprefix("library-ingest-")
            if field in {"title", "author", "keywords"}:
                getattr(
                    LibraryIngestController, f"handle_library_ingest_{field}_changed"
                )(controller, event)

    host = Host(build_library_ingest_state((), form=form))
    async with host.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        host.refresh(recompose=True)
        await pilot.pause()
        assert not cancelled
        host.query_one("#library-ingest-title", Input).value = "Changed title"
        await pilot.pause()
        assert cancelled == [True]


def test_restaging_same_source_and_options_fences_pending_readiness():
    from tldw_chatbook.UI.Library_Modules.library_ingest_controller import (
        LibraryIngestController,
    )

    screen, queued, checks, _ = _screen()
    screen._do_submit_ingest("clip.mp4")
    screen._library_ingest_form = screen._ingest_state.form
    screen._library_ingest_last_submission = SimpleNamespace(
        source="clip.mp4",
        title="Previous title",
        author="",
        keywords="",
        analyze=False,
        chunk=False,
        chunk_size=1000,
        type_options=screen._build_ingest_options_snapshot(),
    )
    screen._library_ingest_restage_discards_work = lambda: False
    screen._disarm_library_ingest_retry_confirm = lambda: None
    screen._invalidate_library_external_submission = (
        lambda: LibraryScreen._invalidate_library_external_submission(screen)
    )
    screen._clear_library_external_vad_progress = lambda: None
    screen._invalidate_library_ingest_preflight = lambda: None
    screen._refresh_library_ingest_canvas_preserving_context = lambda: None
    screen._trigger_library_ingest_preflight = lambda source: None
    screen._focus_library_ingest_path = lambda: None
    screen.call_after_refresh = lambda callback: None

    LibraryIngestController._restage_library_ingest_last_submission(screen)
    assert screen._ingest_state.form.title == "Previous title"
    LibraryScreen._apply_library_transcription_readiness(screen, *checks[0], None)
    assert not queued
