"""Task 10: Meetings tab/destination/route/config/owner wiring."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def test_tab_constant_and_label():
    from tldw_chatbook.Constants import TAB_DISPLAY_LABELS, TAB_MEETINGS

    assert TAB_MEETINGS == "meetings" and TAB_DISPLAY_LABELS[TAB_MEETINGS] == "Meetings"


def test_shell_destination_registered_after_workflows():
    from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER, get_shell_destination

    ids = [d.destination_id for d in SHELL_DESTINATION_ORDER]
    assert ids.index("meetings") == ids.index("workflows") + 1
    dest = get_shell_destination("meetings")
    assert dest.primary_route == "meetings" and dest.label == "Meetings"


def test_screen_route_points_at_meetings_screen():
    from tldw_chatbook.UI.Navigation.screen_registry import registered_screen_routes

    route = next(r for r in registered_screen_routes() if r.screen_name == "meetings")
    assert (route.module_path, route.class_name) == (
        "tldw_chatbook.UI.Screens.meetings_screen", "MeetingsScreen",
    )


def test_app_help_text_covers_meetings():
    from tldw_chatbook.Constants import TAB_MEETINGS
    from tldw_chatbook.app import TabNavigationProvider

    assert "Meetings" in TabNavigationProvider.TAB_HELP_TEXT[TAB_MEETINGS]
    assert TAB_MEETINGS in TabNavigationProvider.NAVIGATION_TABS


def test_config_template_has_meetings_section():
    from tldw_chatbook.config import CONFIG_TOML_CONTENT

    block = CONFIG_TOML_CONTENT.split("[meetings]", 1)[1].split("\n[", 1)[0]
    for key in ("provider", "model", "system_source", "mic_device", "recordings_dir",
                "keep_raw_tracks", "post_transcribe", "post_diarize",
                "live_diarization", "diarizer_backend", "max_speakers"):
        assert f"\n{key} = " in block, key


def test_build_owner_marshals_submit_and_reads_job_state(tmp_path, monkeypatch):
    from tldw_chatbook.Audio import meeting_owner as mo

    monkeypatch.setattr(mo, "_config_accessors", lambda: (lambda s, k, d: d, lambda: tmp_path))
    jobs = {}

    class Registry:
        def submit(self, **kwargs):
            jobs["kw"] = kwargs
            return SimpleNamespace(job_id="ingest-job-9", state=SimpleNamespace(value="queued"))

        def get_job(self, job_id):
            return SimpleNamespace(job_id=job_id, state=SimpleNamespace(value="done")) if job_id == "ingest-job-9" else None

    marshalled = []

    class App:
        library_ingest_jobs = Registry()
        _thread_id = threading.get_ident() + 1   # pretend the UI thread is another thread

        def call_from_thread(self, fn, *args, **kwargs):
            marshalled.append(fn)
            return fn(*args, **kwargs)

    owner = mo.build_meeting_session_owner(App())
    assert owner.settings.recordings_dir == (tmp_path / "meetings").resolve()
    assert owner._submit_on_ui_thread(source_path="x") == "ingest-job-9" and marshalled
    assert owner._job_state("ingest-job-9") == "done" and owner._job_state("nope") is None


def test_build_owner_marshals_registry_listener_registration(tmp_path, monkeypatch):
    """Q12: the raw-track cleanup waits on the ingest registry, which is
    UI-thread-only -- add_listener/remove_listener go through the same
    marshalling as submit, not straight off the stopping thread."""
    from tldw_chatbook.Audio import meeting_owner as mo

    monkeypatch.setattr(mo, "_config_accessors", lambda: (lambda s, k, d: d, lambda: tmp_path))
    listeners = []

    class Registry:
        def add_listener(self, callback):
            listeners.append(callback)

        def remove_listener(self, callback):
            listeners.remove(callback)

    marshalled = []

    class App:
        library_ingest_jobs = Registry()
        _thread_id = threading.get_ident() + 1   # pretend the UI thread is another thread

        def call_from_thread(self, fn, *args, **kwargs):
            marshalled.append(fn)
            return fn(*args, **kwargs)

    owner = mo.build_meeting_session_owner(App())

    def callback():
        return None

    owner._subscribe_jobs(callback)
    assert listeners == [callback] and marshalled == [App.library_ingest_jobs.add_listener]
    owner._unsubscribe_jobs(callback)
    assert listeners == [] and marshalled[-1] == App.library_ingest_jobs.remove_listener


def test_build_owner_calls_directly_when_already_on_ui_thread(tmp_path, monkeypatch):
    from tldw_chatbook.Audio import meeting_owner as mo

    monkeypatch.setattr(mo, "_config_accessors", lambda: (lambda s, k, d: d, lambda: tmp_path))

    class App:
        library_ingest_jobs = SimpleNamespace(submit=lambda **kw: SimpleNamespace(job_id="j"), get_job=lambda j: None)
        _thread_id = threading.get_ident()

        def call_from_thread(self, fn, *args, **kwargs):
            raise RuntimeError("must not marshal from the UI thread")

    owner = mo.build_meeting_session_owner(App())
    assert owner._submit_on_ui_thread(source_path="x") == "j"


@pytest.mark.asyncio
async def test_real_app_owns_a_meeting_owner_and_shuts_it_down():
    import inspect

    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.Audio.meeting_owner import MeetingSessionOwner
    from tldw_chatbook.app import TldwCli

    app = _build_test_app()
    assert isinstance(app.meeting_session_owner, MeetingSessionOwner)
    # R2: `_shutdown_app_owned_lifecycles` also drains other app-owned
    # lifecycles that don't tolerate a factory-built app that never ran;
    # assert the shutdown call is present in source rather than awaiting it.
    assert "meeting_session_owner.shutdown" in inspect.getsource(TldwCli._shutdown_app_owned_lifecycles)
