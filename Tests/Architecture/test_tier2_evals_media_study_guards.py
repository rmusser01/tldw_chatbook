"""Evals / Media / Study fixes from the tier-2 review (S22 + S23 P2s).

Gate-free: one `run_test()` harness for the bounded read, and AST/source
checks for the other two. None of them boots the real app, so
`Backup_Recovery`'s ADR-126 recovery gate never runs.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

_PACKAGE = Path(__file__).resolve().parents[2] / "tldw_chatbook"


def _bare_snippet_editor(monkeypatch, notices: list[Any]) -> tuple[Any, dict[str, Any]]:
    """A `SnippetEditor` with no app behind it, and the two seams the import
    now crosses made observable: `run_worker` (which really does run the work
    on a thread) and `app.call_from_thread` (the way back)."""
    import threading
    from unittest.mock import MagicMock

    from tldw_chatbook.UI.Evals.snippet_editor import SnippetEditor

    record: dict[str, Any] = {"worker_kwargs": None, "marshalled": []}

    def _call_from_thread(callback, *args, **kwargs):
        record["marshalled"].append(callback.__name__)
        return callback(*args, **kwargs)

    app = MagicMock()
    app.call_from_thread = _call_from_thread
    # `app` is a read-only property on MessagePump, so shadow it on the class.
    monkeypatch.setattr(SnippetEditor, "app", app, raising=False)

    def _run_worker(work, **kwargs):
        record["worker_kwargs"] = kwargs
        thread = threading.Thread(target=work)
        thread.start()
        thread.join()
        return object()

    editor = SnippetEditor.__new__(SnippetEditor)
    editor._notify = lambda message, **kw: notices.append(  # type: ignore[method-assign]
        (message, kw.get("severity"))
    )

    class _ViewModel:
        db = None

    editor._view_model = _ViewModel()
    editor._dataset_id = "ds-1"
    editor._import_worker = None
    editor.run_worker = _run_worker  # type: ignore[method-assign]
    return editor, record


@pytest.mark.unit
def test_snippet_import_reads_the_picked_file_off_the_ui_thread(
    tmp_path, monkeypatch
) -> None:
    """Qodo #2813 finding 5: a `push_screen` callback IS the UI thread.

    S22 put a 2 MiB ceiling on this read, which bounds MEMORY but not
    LATENCY: the picked path can be on a slow or network-backed volume, and
    the whole read still happened inline in the selection callback, holding
    rendering and input behind it. CLAUDE.md's rule is "workers for
    operations >100ms", so the read belongs on a worker thread with only the
    parse, the write and the notifications marshalled back.
    """
    import threading

    from tldw_chatbook.UI.Evals import snippet_editor as module

    picked = tmp_path / "snippets.txt"
    picked.write_text("one snippet\n", encoding="utf-8")

    read_threads: list[int] = []
    real_read = module.read_bounded_import_text

    def _recording_read(file_path):
        read_threads.append(threading.get_ident())
        return real_read(file_path)

    monkeypatch.setattr(module, "read_bounded_import_text", _recording_read)

    notices: list[Any] = []
    editor, record = _bare_snippet_editor(monkeypatch, notices)
    ui_thread = threading.get_ident()

    editor._handle_import_file_selected(str(picked))

    assert read_threads, "the picked file was never read"
    assert ui_thread not in read_threads, (
        f"the snippet import read the picked file on the UI thread "
        f"({ui_thread}); a slow or network-backed picker path freezes "
        "rendering and input behind it"
    )
    kwargs = record["worker_kwargs"]
    assert kwargs is not None, "no worker was scheduled for the import read"
    assert kwargs.get("thread") is True, "the import read must be a THREAD worker"
    assert kwargs.get("group"), (
        "run_worker without group= -- this repo's rule (see task-222)"
    )
    assert "_apply_import_content" in record["marshalled"], (
        "the parse/write half must come back to the UI thread via "
        "call_from_thread, not run on the worker thread"
    )


@pytest.mark.unit
def test_snippet_import_refuses_a_file_over_the_cap(tmp_path, monkeypatch) -> None:
    """S22 P2: the snippet import read a user-picked file with no ceiling.

    `file_path.read_text()` ran inside a `push_screen` callback -- the UI
    thread -- so a multi-GB pick was read whole into memory with the app
    frozen behind it. The sibling importer
    `UI/Chunking_Lab_Modules/sample_region.read_sample_file` already caps at
    2 MiB and refuses with a message rather than reading.
    """
    from tldw_chatbook.UI.Chunking_Lab_Modules.sample_region import SAMPLE_BYTES
    from tldw_chatbook.UI.Evals.snippet_editor import _IMPORT_MAX_BYTES

    assert _IMPORT_MAX_BYTES == SAMPLE_BYTES, (
        "the two user-picked-text-file caps in this repo must agree"
    )

    oversized = tmp_path / "huge.txt"
    oversized.write_bytes(b"a\n" * (_IMPORT_MAX_BYTES // 2 + 16))
    assert oversized.stat().st_size > _IMPORT_MAX_BYTES

    small = tmp_path / "small.txt"
    small.write_text("one snippet\n", encoding="utf-8")

    notices: list[Any] = []
    editor, _record = _bare_snippet_editor(monkeypatch, notices)

    editor._handle_import_file_selected(str(oversized))
    assert notices, "an over-cap file must produce a message"
    assert "2 MiB" in notices[-1][0]
    assert notices[-1][1] == "error"
    assert len(notices) == 1, "it must refuse before parsing, not after"

    notices.clear()
    editor._handle_import_file_selected(str(small))
    assert notices, "an under-cap file must still be read and parsed"
    assert not any("2 MiB" in message for message, _ in notices), (
        f"an under-cap file must not hit the ceiling; got {notices}"
    )


@pytest.mark.unit
def test_media_viewer_delete_uses_the_shared_confirmation_dialog() -> None:
    """S23 P2: a `ModalScreen` subclass was defined inside the worker body.

    Every Delete press built a fresh type -- a new `_MessagePumpMeta` `@on`
    snapshot and a new `DEFAULT_CSS` registration for the session -- and that
    one-off dialog had no BINDINGS, no escape handling and no safe-dismiss,
    making it the only modal in its slice outside the repo's
    `Widgets/modal_dismissal` convention. It also read its answer off the
    instance instead of `dismiss(value)`.
    """
    from tldw_chatbook.Widgets.Media.media_viewer_panel import MediaViewerPanel

    import textwrap

    source = inspect.getsource(MediaViewerPanel._run_delete_confirmation.__wrapped__)
    tree = ast.parse(textwrap.dedent(source))

    nested_classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    assert not nested_classes, (
        f"a class is still defined inside the worker body: "
        f"{[n.name for n in nested_classes]}"
    )
    assert "DeleteConfirmationDialog" in source
    assert "dialog.result" not in source, (
        "read the answer from dismiss(value), not off the dialog instance"
    )


@pytest.mark.unit
def test_study_controllers_agree_on_the_workspace_scope_constant() -> None:
    """S23 P2: flashcards compared a string literal, quizzes the enum.

    Same runtime value today, with nothing enforcing that it stays that way.
    The day `StudyScopeType.WORKSPACE.value` changes, one controller silently
    starts treating every workspace as global.
    """
    for module in ("flashcards_handler.py", "quizzes_handler.py"):
        source = (_PACKAGE / "UI" / "Study_Modules" / module).read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        literals = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Compare)
            and any(
                isinstance(comparator, ast.Constant)
                and comparator.value == "workspace"
                for comparator in node.comparators
            )
        ]
        assert not literals, (
            f"{module} compares the scope against the bare string "
            f'"workspace" at line(s) {literals}; use '
            f"StudyScopeType.WORKSPACE.value like its sibling controller"
        )
