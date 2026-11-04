"""The real restore child uses bounded diagnostics without a C watchdog."""

import faulthandler
import json
import threading
import time
from pathlib import Path

import pytest

from Tests import network_guard
from Tests.Backup_Recovery import thread_diagnostics
from Tests.Backup_Recovery.test_eval_retained_definitions import (
    _FULL_APP_CHILD_TIMEOUT,
    _RETAIN_RESTORE,
)
from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_temporary_media_capture import _PUBLIC


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_restore_diagnostics_preserve_failure_and_retire_observer(
    tmp_path, monkeypatch, cleanup_fails
):
    failure = RuntimeError("synthetic restore failure")
    path = tmp_path / "restore-stacks.json"
    original_observer = thread_diagnostics.observe_threads
    enabled_before = faulthandler.is_enabled()
    stopped = []

    def reject_c_watchdog(*args, **kwargs):
        raise AssertionError("restore must not start the racy C watchdog")

    def observe(selected, *, interval):
        assert selected == path and interval == 10
        stop = original_observer(selected, interval=0.01)

        def finish():
            stop()
            stopped.append(True)
            if cleanup_fails:
                raise OSError("synthetic diagnostic cleanup failure")

        return finish

    def enter_restore():
        deadline = time.monotonic() + 5
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert path.exists(), "restore must retain bounded stack diagnostics"
        raise failure

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(faulthandler, "dump_traceback_later", reject_c_watchdog)
    monkeypatch.setattr(thread_diagnostics, "observe_threads", observe)
    # Stop the actual child script at its first product entry, before file/network work.
    monkeypatch.setattr(network_guard, "install", enter_restore)
    with pytest.raises(RuntimeError) as caught:
        exec(compile(_RETAIN_RESTORE, "retained-eval-restore", "exec"), {})  # noqa: S102  # nosec B102
    assert caught.value is failure and stopped == [True]
    assert faulthandler.is_enabled() is enabled_before
    assert not any(t.name == "test-thread-diagnostics" for t in threading.enumerate())
    data = json.loads(path.read_text())
    assert 1 <= len(data) <= 4
    assert all(len(snapshot) <= 32 for snapshot in data)
    assert all(len(row["frames"]) <= 64 for snapshot in data for row in snapshot)
    assert "synthetic restore failure" not in path.read_text()
    assert (tmp_path / "restore-stacks-fatal.log").exists()
    assert bool(getattr(failure, "__notes__", ())) is cleanup_fails


def test_public_review_follows_successful_native_attention_read_retirement(tmp_path):
    anchor = "    preview = preview_capture((selector,), options=options)"
    probe = """    runtime = app.console_runtime
    task = runtime._attention_refresh_task
    try:
        assert task is not None and task.done(), 'constructor attention read is pending at review'
        assert not task.cancelled() and task.exception() is None
        assert attention_returns and all(off_thread and succeeded for off_thread, succeeded in attention_returns), 'native attention read did not complete successfully'
        assert not runtime._preparation_reads, 'attention worker still owns a preparation read'
        participant = app.chachanotes_db._maintenance_participant
        with storage._lock:
            assert not any(lease.resource_thread is not threading.current_thread()
                           for lease in participant.connections.values()), 'native worker DB handle is still live'
    except BaseException as primary:
        for cleanup in (app._shutdown_app_owned_lifecycles, app.tts_service.close, app.tts_service.wait_closed):
            try:
                await cleanup()
            except BaseException as error:
                primary.add_note('fixture_cleanup_error:' + type(error).__name__)
        raise
"""
    assert _PUBLIC.count(anchor) == 1
    observation = """from Tests.Backup_Recovery.thread_diagnostics import stop_observer
from tldw_chatbook.Chat.conversation_local_marks_service import ConversationLocalMarksService
attention_returns = []
main_thread = threading.get_ident()
def returned(code, offset, value):
    attention_returns.append((threading.get_ident() != main_thread, type(value) is tuple))
sys.monitoring.use_tool_id(4, 'capture-startup-read')
sys.monitoring.register_callback(4, sys.monitoring.events.PY_RETURN, returned)
attention_code = ConversationLocalMarksService.list_console_unseen_marks.__code__
sys.monitoring.set_local_events(4, attention_code, sys.monitoring.events.PY_RETURN)
def stop_monitor():
    try:
        sys.monitoring.set_local_events(4, attention_code, 0)
    finally:
        try:
            sys.monitoring.register_callback(4, sys.monitoring.events.PY_RETURN, None)
        finally:
            sys.monitoring.free_tool_id(4)
"""
    script = _PUBLIC.replace(anchor, probe + anchor)
    script = script.replace("async def main():", observation + "\nasync def main():", 1)
    script = script.replace(
        "asyncio.run(main())",
        "try:\n    asyncio.run(main())\nfinally:\n    stop_observer(stop_monitor)",
        1,
    )
    _run(
        tmp_path,
        "temporary",
        "complete",
        script=script,
        timeout=_FULL_APP_CHILD_TIMEOUT,
    )
