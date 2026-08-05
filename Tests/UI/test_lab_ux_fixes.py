"""Regression tests for the Lab UX critique fixes (UX-020..UX-031).

Covers the lab mode-chip CSS override, view cycling, and the server-process
attribute contract. (The Evals card-hub tests were removed when dev retired
the hub, and the grouped-sidebar test when dev dropped the LLM sidebar —
commit 46b4c61b5 and dev's LLM_Management_Window redesign.)
"""

from __future__ import annotations

from pathlib import Path

from tldw_chatbook.UI.Screens.llm_screen import _SERVER_PROCESS_ATTRS
from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow

CSS_BUNDLE = Path("tldw_chatbook/css/tldw_cli_modular.tcss")


def test_lab_mode_chip_has_active_override_in_bundle() -> None:
    bundle = CSS_BUNDLE.read_text()
    assert "#lab-mode-strip .lab-mode-chip.is-active" in bundle, (
        "app bundle needs an explicit .lab-mode-chip.is-active override so the "
        "generic .is-active border cannot clip the chip label"
    )
    assert "#lab-mode-strip Button.lab-mode-chip" in bundle


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
