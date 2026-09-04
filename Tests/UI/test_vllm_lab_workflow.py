from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Label

from tldw_chatbook.UI.LLM_Management.vllm_setup import (
    VllmLaunchDraft,
    VllmMode,
    VllmModelSource,
    VllmReadinessState,
)
from tldw_chatbook.UI.LLM_Management.vllm_setup_view import VllmSetupView


pytestmark = pytest.mark.asyncio


class _VllmHost(App[None]):
    def compose(self) -> ComposeResult:
        yield VllmSetupView(id="vllm-setup")


async def test_initial_vllm_setup_is_guided_and_blocks_start():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(VllmSetupView)
        copy = " ".join(str(label.renderable) for label in view.query(Label))
        buttons = " ".join(str(button.label) for button in view.query(Button))
        assert "Set up vLLM" in copy
        assert "Start on this computer" in buttons
        assert "Connect to existing server" in buttons
        assert app.query_one("#vllm-port", Input).value == "8000"
        assert app.query_one("#vllm-start-button", Button).disabled
        assert "GGUF" not in copy
        assert "checkpoint" not in copy.lower()


async def test_source_specific_controls_and_mode_drafts_are_preserved():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)) as pilot:
        hf_input = app.query_one("#vllm-hf-model", Input)
        local_input = app.query_one("#vllm-local-model-directory", Input)
        assert hf_input.display
        assert not local_input.display

        hf_input.value = "org/kept-model"
        await pilot.pause()
        await pilot.click("#vllm-connect-existing-button")
        assert app.query_one("#vllm-existing-server-url", Input).display
        await pilot.click("#vllm-start-local-button")
        assert app.query_one("#vllm-hf-model", Input).value == "org/kept-model"

        await pilot.click("#vllm-local-model-source-button")
        assert local_input.display
        assert not hf_input.display
        assert app.query_one("#vllm-browse-local-model-directory-button", Button).display


async def test_preflight_blocker_is_adjacent_and_start_enables_only_for_current_success():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(VllmSetupView)
        draft = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="org/model",
        )
        view.apply_state(
            draft=draft,
            state=VllmReadinessState.NEEDS_ATTENTION,
            preflight=None,
        )
        assert app.query_one("#vllm-start-button", Button).disabled
        assert "Check setup" in str(app.query_one("#vllm-start-blocker", Label).renderable)
