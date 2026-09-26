from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.widgets import Button, Collapsible, Input, Static

from Tests.UI.test_llm_gguf_source_modes import (
    _close_context,
    _deterministic_models_mount,  # noqa: F401 - registers the shared pytest fixture
    _mount_models,
)
from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
    release_server_claim,
    reserve_server_launch,
)
from tldw_chatbook.LLM_Management.llamacpp_connection import LlamaCppProbeResult
from tldw_chatbook.UI.LLM_Management.llamacpp_setup_view import LlamaCppSetupView

pytestmark = pytest.mark.usefixtures("_deterministic_models_mount")


def text(widget: Static) -> str:
    return str(widget.render())


async def preview_with_keyboard(view, pilot):
    view.query_one("#llamacpp-launch-preview", Collapsible).collapsed = False
    button = view.query_one("#llamacpp-preview-launch", Button)
    button.scroll_visible(animate=False)
    button.focus()
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 28), (160, 48)])
async def test_preview_copy_and_stale_draft_on_production_models_pane(
    monkeypatch, size
):
    app, pilot, context, _screen, window, _ = await _mount_models(
        monkeypatch, size=size, mount_llamafile=False
    )
    copied = []
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
    try:
        view = window.query_one(LlamaCppSetupView)
        assert view.query_one("#llamacpp-preview-launch", Button)
        window.query_one("#llamacpp-exec-path", Input).value = "/PRIVATE/server"
        window.query_one("#llamacpp-model-path", Input).value = "/PRIVATE/model.gguf"
        window.query_one("#llamacpp-port", Input).value = "8182"
        window.query_one(
            "#llamacpp-additional-args", Input
        ).value = '--api-key=SECRET --mmproj "/PRIVATE/vision.gguf" --unknown=HIDDEN'
        view.query_one("#llamacpp-context-size", Input).value = "4096"
        await pilot.pause()
        await preview_with_keyboard(view, pilot)
        preview = view.query_one("#llamacpp-preview-launch", Button)
        assert app.screen.focused is preview
        assert 0 <= preview.region.y < preview.region.bottom <= app.size.height
        rendered = text(view.query_one("#llamacpp-preview-command", Static))
        assert "--ctx-size 4096" in rendered
        assert "--port 8182" in rendered
        assert "chatbook-llamacpp" in rendered
        assert "PRIVATE" not in rendered and "SECRET" not in rendered
        assert "HIDDEN" not in rendered
        copy = view.query_one("#llamacpp-copy-launch", Button)
        assert not copy.disabled and copy.region.width > 0
        assert copy.region.right <= app.size.width
        await pilot.press("tab")
        await pilot.pause()
        assert app.screen.focused is copy
        assert 0 <= copy.region.y < copy.region.bottom <= app.size.height
        await pilot.press("enter")
        await pilot.pause()
        assert copied and "--ctx-size 4096" in copied[-1]
        assert not any(value in copied[-1] for value in ("PRIVATE", "SECRET", "HIDDEN"))
        view.query_one("#llamacpp-context-size", Input).value = "8192"
        await pilot.pause()
        assert copy.disabled
        assert "4096" not in text(view.query_one("#llamacpp-preview-command", Static))
        await preview_with_keyboard(view, pilot)
        assert "--ctx-size 8192" in text(
            view.query_one("#llamacpp-preview-command", Static)
        )
    finally:
        await _close_context(context)


@pytest.mark.asyncio
async def test_next_launch_edits_preserve_current_claim_and_verified_target(
    monkeypatch,
):
    app, pilot, context, _screen, window, _ = await _mount_models(
        monkeypatch, mount_llamafile=False
    )
    claim = None
    try:
        view = window.query_one(LlamaCppSetupView)
        claim = reserve_server_launch(app, "llamacpp")
        assert claim is not None
        claim._connection_url = "http://127.0.0.1:8181"
        claim._launch_preview = "<executable> --ctx-size 4096"
        app.llamacpp_server_process = SimpleNamespace(poll=lambda: None)
        view._local_claim = claim
        view._local_url = claim._connection_url
        request = view.owner.begin(
            claim._connection_url,
            runtime_owner="lab_process",
            live_check=lambda: True,
        )
        assert view.owner.accept(
            LlamaCppProbeResult(
                request, "ready", ("chatbook-llamacpp",), "chatbook-llamacpp"
            )
        )
        target = view.owner.snapshot().target
        window._sync_process_controls("llamacpp")
        view.refresh_state()
        field = view.query_one("#llamacpp-context-size", Input)
        assert not field.disabled
        assert not window.query_one("#llamacpp-additional-args", Input).disabled
        assert window.query_one("#llamacpp-model-path", Input).disabled
        field.value = "8192"
        window.query_one("#llamacpp-port", Input).value = "9191"
        await pilot.pause()
        await preview_with_keyboard(view, pilot)
        assert "Next launch" in text(view.query_one("#llamacpp-preview-title", Static))
        current = text(view.query_one("#llamacpp-current-launch", Static))
        assert "4096" in current and "8192" not in current
        assert "8192" in text(view.query_one("#llamacpp-preview-command", Static))
        assert view.owner.snapshot().target is target
        assert claim._launch_preview == "<executable> --ctx-size 4096"
    finally:
        app.llamacpp_server_process = None
        if claim is not None:
            release_server_claim(app, "llamacpp", claim)
        await _close_context(context)


@pytest.mark.asyncio
async def test_invalid_preview_cannot_copy_previous_command_or_error_payload(
    monkeypatch,
):
    app, pilot, context, _screen, window, _ = await _mount_models(
        monkeypatch, mount_llamafile=False
    )
    copied = []
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
    try:
        view = window.query_one(LlamaCppSetupView)
        await preview_with_keyboard(view, pilot)
        assert not view.query_one("#llamacpp-copy-launch", Button).disabled
        window.query_one("#llamacpp-additional-args", Input).value = '--model="PRIVATE'
        await pilot.pause()
        await preview_with_keyboard(view, pilot)
        assert view.query_one("#llamacpp-copy-launch", Button).disabled
        error = text(view.query_one("#llamacpp-preview-command", Static))
        assert "PRIVATE" not in error
        assert error.strip()
        assert not copied
    finally:
        await _close_context(context)
