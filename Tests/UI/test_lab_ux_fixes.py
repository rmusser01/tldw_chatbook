"""Regression tests for the Lab UX critique fixes (UX-020..UX-031).

Covers the Evals inline-render fix (nested Screen -> widget), planned-card
honesty, the lab mode-chip CSS override, the grouped LLM sidebar, view
cycling, and truthful destination-header chips.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.UI.Evals.evals_window_v3 import EvalsWindowV3
from tldw_chatbook.UI.Evals.navigation import EvalNavigationScreen
from tldw_chatbook.UI.Evals.screens import EvaluationBrowserScreen, QuickTestScreen
from tldw_chatbook.UI.Screens.llm_screen import _SERVER_PROCESS_ATTRS
from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow

CSS_BUNDLE = Path("tldw_chatbook/css/tldw_cli_modular.tcss")


def test_eval_subscreens_are_widgets_not_screens() -> None:
    from textual.containers import Container
    from textual.screen import Screen

    for cls in (EvalNavigationScreen, QuickTestScreen, EvaluationBrowserScreen):
        assert issubclass(cls, Container)
        assert not issubclass(cls, Screen), (
            f"{cls.__name__} must not be a Screen: nested Screens get no layout"
        )


class _EvalsHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield EvalsWindowV3(None, id="evals-window")


@pytest.mark.asyncio
async def test_evals_hub_renders_with_nonzero_geometry() -> None:
    app = _EvalsHarness()
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()
        hub = app.query_one(EvalNavigationScreen)
        assert hub.region.width > 0 and hub.region.height > 0
        cards = list(app.query(".nav-card"))
        assert len(cards) == 6
        first = cards[0]
        assert first.region.width >= 30, "card grid must not collapse to zero width"


@pytest.mark.asyncio
async def test_planned_cards_are_disabled_and_labeled() -> None:
    app = _EvalsHarness()
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()
        for card_id in ("comparison", "batch_eval", "models"):
            card = app.query_one(f"#card-{card_id}")
            assert card.disabled, f"{card_id} should render disabled (Planned)"
            assert card.tooltip and "planned" in card.tooltip.lower()
        for card_id in ("quick_test", "results", "tasks"):
            assert not app.query_one(f"#card-{card_id}").disabled


def test_lab_mode_chip_has_active_override_in_bundle() -> None:
    bundle = CSS_BUNDLE.read_text()
    assert "#lab-mode-strip .lab-mode-chip.is-active" in bundle, (
        "app bundle needs an explicit .lab-mode-chip.is-active override so the "
        "generic .is-active border cannot clip the chip label"
    )
    assert "#lab-mode-strip Button.lab-mode-chip" in bundle


class _LLMHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield LLMManagementWindow(None)


@pytest.mark.asyncio
async def test_llm_sidebar_is_grouped_with_guidance(monkeypatch) -> None:
    # The library views need a full app instance (config, DBs); stub them so
    # this test can focus on the sidebar structure alone.
    from textual.containers import Container as _Container

    import tldw_chatbook.Widgets.HuggingFace as hf

    class _StubWidget(_Container):
        def __init__(self, *args, **kwargs):
            super().__init__(**{k: v for k, v in kwargs.items() if k == "id"})

    monkeypatch.setattr(hf, "LocalModelsWidget", _StubWidget)
    monkeypatch.setattr(hf, "HuggingFaceModelBrowser", _StubWidget)

    app = _LLMHarness()
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()
        sidebar = app.query_one("#llm-sidebar")
        statics = [str(w.render()) for w in sidebar.query(Static)]
        joined = " ".join(statics)
        assert "Serve a model" in joined
        assert "Model library" in joined
        assert "Ollama is the easiest" in joined
        buttons = [str(w.label) for w in sidebar.query(".llm-nav-button")]
        # Ollama leads the server group as the recommended first path, and
        # every button carries its jump digit (1-9 jump to view).
        assert buttons[0] == "1 Ollama"
        assert buttons[-2:] == ["8 Local Models", "9 Download Models"]


def test_llm_view_cycling_wraps() -> None:
    window = LLMManagementWindow(None)
    views = list(window.view_mapping)
    assert views, "view_mapping must not be empty"
    window.active_view = views[0]
    window._cycle_view(-1)
    assert window.active_view == views[-1]
    window._cycle_view(1)
    assert window.active_view == views[0]
    window._cycle_view(1)
    assert window.active_view == views[1]


def test_server_process_attrs_match_app_conventions() -> None:
    attrs = {attr for attr, _label in _SERVER_PROCESS_ATTRS}
    assert attrs == {
        "llamacpp_server_process",
        "llamafile_server_process",
        "ollama_server_process",
        "vllm_server_process",
        "onnx_server_process",
        "mlx_server_process",
    }
