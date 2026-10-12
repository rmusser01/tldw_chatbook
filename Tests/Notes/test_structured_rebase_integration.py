"""Current-dev export and maintenance owners retain structured source authority."""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from Tests.Library.test_library_file_export import _answer, _FakeApp
from Tests.Notes.test_notes_sync_runtime import _Adapter, _owner
from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore


@pytest.mark.asyncio
@pytest.mark.parametrize("export_format", ["source", "portable"])
@pytest.mark.parametrize("source", ["\ufeffa\r\nb\r", "a\n", "a"])
async def test_source_export_preserves_bytes_and_refused_replacement(
    tmp_path, monkeypatch, source, export_format
):
    from tldw_chatbook.UI.Library_Modules import library_file_export as seam
    from tldw_chatbook.Utils import atomic_file_ops

    native_fdopen = atomic_file_ops.os.fdopen

    def windows_newline_default(fd, mode, **kwargs):
        if mode == "w" and kwargs.get("newline") is None:
            kwargs["newline"] = "\r\n"
        return native_fdopen(fd, mode, **kwargs)

    monkeypatch.setattr(atomic_file_ops.os, "fdopen", windows_newline_default)
    app = _FakeApp()
    finished = []
    screen = SimpleNamespace(
        app=app,
        app_instance=SimpleNamespace(notify=app.notify),
        _finish_library_notes_operation=lambda op, **kw: finished.append(kw),
    )
    target = tmp_path / "source.yaml"
    seam.export_library_note_file(
        screen,
        target,
        export_format,
        "Title",
        source,
        "",
        "note",
        "operation",
        content_language="yaml",
    )
    from tldw_chatbook.Library.library_notes_state import build_note_export_content

    expected = build_note_export_content(
        "Title", source, "", "note", export_format, content_language="yaml"
    )
    assert target.read_bytes() == expected.encode("utf-8")
    target.write_bytes(b"keep existing")
    seam.export_library_note_file(
        screen,
        target,
        export_format,
        "Title",
        source,
        "",
        "note",
        "operation",
        content_language="yaml",
    )
    assert target.read_bytes() == b"keep existing"
    await _answer(app, False)
    assert target.read_bytes() == b"keep existing"


@pytest.mark.asyncio
async def test_recovery_producer_drain_waits_for_accepted_write_and_fences_new_calls(
    tmp_path,
):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    store = NotesDeviceStateStore(tmp_path / "device.db")
    owner, _, _ = _owner(store=store, admitted=False, adapter=_Adapter([]))
    started, release = threading.Event(), threading.Event()

    def pending_write():
        started.set()
        assert release.wait(3)
        return "saved"

    task = asyncio.create_task(owner._run_draft_recovery(pending_write))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        owner._maintenance_close_admission()
        assert not await owner._producer_lifetime.drain(time.monotonic() + 0.03)
        with pytest.raises(RecoveryRequired, match="runtime_producer_paused"):
            await owner._run_draft_recovery(lambda: None)
    finally:
        release.set()
        assert await task == "saved"
        await owner.shutdown()
