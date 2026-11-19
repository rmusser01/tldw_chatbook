"""Clipboard and link-open spawns report failure instead of a success toast."""

import threading
from types import SimpleNamespace

import pytest

import tldw_chatbook.Widgets.Console.console_conversation_inspector as inspector_module
import tldw_chatbook.Widgets.Console.console_transcript as transcript_module
from tldw_chatbook.Widgets.Console.console_conversation_inspector import (
    ConsoleConversationInspector,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleMarkdownMessage

pytestmark = pytest.mark.bootstrap_profile


class _InspectorShell(SimpleNamespace):
    """Pure namespace: the unbound handler runs against it, no Widget."""

    def notify(self, message, **kwargs):
        self.notices.append((message, kwargs.get("severity")))

    def run_worker(self, fn, **kwargs):
        assert kwargs.get("thread") is True
        self.workers.append(fn)


class _TranscriptShell(SimpleNamespace):
    def notify(self, message, **kwargs):
        self.notices.append((message, kwargs.get("severity")))

    def run_worker(self, fn, **kwargs):
        assert kwargs.get("thread") is True
        self.workers.append(fn)


def _run_threaded(fn):
    thread = threading.Thread(target=fn)
    thread.start()
    thread.join()


def _inspector_shell():
    shell = _InspectorShell(
        notices=[], workers=[], _target_invalid=False,
        _selection_generation=0, _snapshot_generation=0,
        _disclosure_generation=0, _snapshot_ready=True,
    )

    class _App:
        @staticmethod
        def call_from_thread(notify, *args, **kwargs):
            shell.notify(*args, **kwargs)

    shell.app = _App
    shell._target_authority_is_current = lambda: True
    shell._format_export_text = lambda: "{}"
    return shell


def _transcript_shell():
    shell = _TranscriptShell(notices=[], workers=[])

    class _App:
        @staticmethod
        def call_from_thread(notify, *args, **kwargs):
            shell.notify(*args, **kwargs)

    shell.app = _App
    return shell


def test_clipboard_failure_notifies_error(monkeypatch):
    import pyperclip

    shell = _inspector_shell()

    def boom(text):
        raise RuntimeError("no clipboard tool")

    monkeypatch.setattr(pyperclip, "copy", boom)
    ConsoleConversationInspector._copy_json(shell, SimpleNamespace(stop=lambda: None))
    assert len(shell.workers) == 1
    _run_threaded(shell.workers[0])
    assert shell.notices == [("Copy failed (RuntimeError).", "error")], (
        "a failed clipboard spawn must not show the success toast"
    )


def test_clipboard_success_notifies_once(monkeypatch):
    import pyperclip

    shell = _inspector_shell()
    monkeypatch.setattr(pyperclip, "copy", lambda text: None)
    ConsoleConversationInspector._copy_json(shell, SimpleNamespace(stop=lambda: None))
    _run_threaded(shell.workers[0])
    assert shell.notices == [("JSON copied to clipboard.", None)]


def test_link_open_failure_warns(monkeypatch):
    import webbrowser

    shell = _transcript_shell()
    monkeypatch.setattr(webbrowser, "open", lambda href: False)
    event = SimpleNamespace(stop=lambda: None, href="https://example.com/x")
    ConsoleMarkdownMessage._open_link(shell, event)
    assert len(shell.workers) == 1
    _run_threaded(shell.workers[0])
    assert shell.notices == [
        ("Could not open the browser for: https://example.com/x", "warning")
    ]


def test_link_open_success_toasts(monkeypatch):
    import webbrowser

    shell = _transcript_shell()
    monkeypatch.setattr(webbrowser, "open", lambda href: True)
    ConsoleMarkdownMessage._open_link(
        shell, SimpleNamespace(stop=lambda: None, href="https://example.com/")
    )
    _run_threaded(shell.workers[0])
    assert shell.notices == [("Opened in browser: https://example.com/", None)]
