"""Install-command actions must deliver bytes, not just emit a success toast."""

from __future__ import annotations

import asyncio

import pyperclip
import pytest
from textual.widgets import Button, Collapsible, Static

import tldw_chatbook.app  # noqa: F401 - bind config before fixtures rebind its profile.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Library.ingest_types import PreflightResult
from tldw_chatbook.Library.library_ingest_state import (
    LibraryIngestFormState,
    build_library_ingest_state,
)
from tldw_chatbook.Utils import install_clipboard
from tldw_chatbook.Utils.widget_helpers import FeatureNotAvailableDialog
from tldw_chatbook.Widgets.Library.library_ingest_canvas import LibraryIngestCanvas

COMMAND = 'pip install "tldw_chatbook[pdf]"'


@pytest.fixture(autouse=True)
def native_clipboard_boundary(monkeypatch):
    # Subprocess execution/readback has its own real-child tests. Here keep
    # delivery controllable while exercising both mounted actions.
    async def copy(command):
        try:
            pyperclip.copy(command)
            return pyperclip.paste() == command
        except pyperclip.PyperclipException:
            return False

    monkeypatch.setattr(install_clipboard, "_copy_native", copy)


class ClipboardHost(ConsolidatedCSSApp):
    @property
    def is_headless(self):
        # Simulate a native terminal; _copy_native is replaced by the fixture.
        return False

    def __init__(self, surface):
        super().__init__()
        self.surface = surface
        self.notices = []
        self.terminal_copies = []

    def compose(self):
        if self.surface == "feature":
            yield FeatureNotAvailableDialog("PDF", ["pymupdf"], COMMAND)
        else:
            form = LibraryIngestFormState(
                path="/tmp/example.pdf",
                preflight=PreflightResult(
                    type_groups={"pdf": ["/tmp/example.pdf"]},
                    total_files=1,
                    total_size=10,
                    truncated=False,
                    errors=[],
                    warnings=[
                        {
                            "feature": "pdf_processing",
                            "label": "PDF",
                            "command": COMMAND,
                        }
                    ],
                ),
            )
            yield LibraryIngestCanvas(build_library_ingest_state((), form=form))

    def notify(self, message, **kwargs):
        self.notices.append((str(message), kwargs.get("severity", "information")))

    def copy_to_clipboard(self, text):
        self.terminal_copies.append(text)
        super().copy_to_clipboard(text)


async def press_copy(host, pilot):
    selector = (
        "#copy-command"
        if host.surface == "feature"
        else "#ingest-preflight-copy-all-commands"
    )
    host.query_one(selector, Button).press()
    async with asyncio.timeout(5):
        while not host.notices:
            await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["feature", "ingest"])
async def test_install_command_reaches_native_clipboard(monkeypatch, surface):
    clipboard = []

    def copy(text):
        clipboard.append(text)

    for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(pyperclip, "copy", copy)
    monkeypatch.setattr(pyperclip, "paste", lambda: clipboard[-1])
    host = ClipboardHost(surface)
    async with host.run_test(size=(110, 42)) as pilot:
        await press_copy(host, pilot)
    assert clipboard == [COMMAND]
    assert any(
        "copied" in text.lower() and severity == "information"
        for text, severity in host.notices
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["feature", "ingest"])
@pytest.mark.parametrize("native_failure", ["unavailable", "silent"])
async def test_unconfirmed_copy_keeps_a_manual_command_and_honest_feedback(
    monkeypatch, surface, native_failure
):
    def copy(text):
        if native_failure == "unavailable":
            raise pyperclip.PyperclipException("No clipboard mechanism")

    monkeypatch.setattr(pyperclip, "copy", copy)
    monkeypatch.setattr(pyperclip, "paste", lambda: "old clipboard content")
    host = ClipboardHost(surface)
    async with host.run_test(size=(110, 42)) as pilot:
        await press_copy(host, pilot)
        assert host.terminal_copies == [COMMAND]
        assert not any("copied" in text.lower() for text, _ in host.notices)
        assert any(severity == "warning" for _, severity in host.notices)
        if surface == "ingest":
            assert not host.query_one(
                "#ingest-preflight-tooling-detail", Collapsible
            ).collapsed
        else:
            assert "[pdf]" in host.query_one(".install-command", Static).visual.plain


@pytest.mark.asyncio
async def test_remote_copy_never_writes_the_server_desktop_clipboard(monkeypatch):
    clipboard = []
    monkeypatch.setenv("SSH_CONNECTION", "remote session")
    monkeypatch.setattr(pyperclip, "copy", clipboard.append)
    monkeypatch.setattr(pyperclip, "paste", lambda: COMMAND)
    host = ClipboardHost("feature")
    async with host.run_test() as pilot:
        await press_copy(host, pilot)
    assert clipboard == []
    assert host.terminal_copies == [COMMAND]


@pytest.mark.asyncio
async def test_feature_alert_keeps_extras_in_both_displayed_commands(monkeypatch):
    from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE
    from tldw_chatbook.Utils.widget_helpers import show_feature_alert

    monkeypatch.setitem(DEPENDENCIES_AVAILABLE, "pdf_processing", False)
    host = ConsolidatedCSSApp()
    async with host.run_test(size=(110, 42)) as pilot:
        show_feature_alert(host, "PDF", "pdf_processing", "pdf")
        await pilot.pause()
        text = "\n".join(widget.visual.plain for widget in host.query(Static))
        assert 'pip install "tldw_chatbook[pdf]"' in text
        assert 'pip install -e ".[pdf]"' in text
