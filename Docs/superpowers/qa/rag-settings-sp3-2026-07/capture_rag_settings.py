"""Capture Settings > Domain Defaults > RAG QA screenshots with the REAL app
stylesheet + theme (task-5 visual-defect fix pass, SP3).

Mirrors ``Docs/superpowers/qa/personas-workbench/capture_personas_workbench.py``:
a bare Textual ``App`` whose ``CSS_PATH`` is the real generated bundle
(``tldw_chatbook/css/tldw_cli_modular.tcss``), the real ``agentic_terminal``
theme registered/activated, and the real ``SettingsScreen`` pushed onto it --
so the SVG/PNG exports show production styling, not a bare-harness fallback
(``DestinationHarness`` in ``Tests/UI/test_destination_shells.py`` does NOT
load the app stylesheet -- see the ``tldw-chatbook-dev-environment`` memory).

RAG profile data is wired the same way as
``Tests.UI.test_settings_configuration_hub._wire_rag_profile_adapter``: an
isolated ``ConfigProfileManager`` over a tmp dir, with the module-level
``_manager``/``_active_profile_id`` hooks in ``settings_rag_profile_adapter``
patched directly (no pytest ``monkeypatch`` fixture available outside a
test run, so this patches by hand -- each capture "session" below gets its
own explicit wiring call).

Run from the repo root with the cairo dylib on the search path (macOS
Homebrew build; see ``brew --prefix cairo``):

    DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \\
        .venv/bin/python3 Docs/superpowers/qa/rag-settings-sp3-2026-07/capture_rag_settings.py

Outputs SVGs directly into this directory; convert to PNG with
``svg_to_png.py`` in the same directory (cairosvg).
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

OUT = Path(__file__).resolve().parent

# Empirically calibrated against the original captures (2050x1240): Rich's
# SVG export viewBox is linear in terminal (cols, rows) --
# width = 18 + 12.2*cols, height = 50 + 24.4*rows -- so (167, 49) lands
# within a few px of the original viewport without distorting aspect ratio.
TERMINAL_SIZE = (167, 49)


def _wire_profiles(tmp_path: Path, *, active_id: str | None = None):
    """Point the Settings RAG adapter at an isolated profile store.

    Returns (manager, profile, state) where `profile` is a writable clone of
    the `hybrid_basic` builtin ("QA Demo Profile"). Mirrors
    ``Tests.UI.test_settings_configuration_hub._wire_rag_profile_adapter``
    without the pytest ``monkeypatch`` fixture (this is a standalone script,
    not a test run) -- patches module attributes directly.
    """
    from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager
    import tldw_chatbook.UI.Screens.settings_rag_profile_adapter as rag_adapter_module

    mgr = ConfigProfileManager(profiles_dir=tmp_path / "profiles")
    profile = mgr.clone_profile("hybrid_basic", "QA Demo Profile")
    mgr.save_profile(profile)
    state = {"active": active_id or profile.id}
    rag_adapter_module._manager = lambda: mgr
    rag_adapter_module._active_profile_id = lambda: state["active"]
    return mgr, profile, state


def _build_app_instance():
    """Real (unmounted) TldwCli instance -- reuses the established test
    helper so SettingsScreen's ``self.app_instance`` references resolve the
    same way they do under pytest."""
    from Tests.UI.test_screen_navigation import _build_test_app

    return _build_test_app()


async def _settle(pilot) -> None:
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


async def _open_rag_category(pilot) -> None:
    await _settle(pilot)
    await pilot.click("#settings-category-library-rag")
    await pilot.pause()
    await _settle(pilot)


def _make_qa_app_class(theme):
    from textual.app import App
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

    class QAApp(App):
        # The real generated bundle: gives the QA captures production styling.
        CSS_PATH = str(REPO_ROOT / "tldw_chatbook" / "css" / "tldw_cli_modular.tcss")

        def __init__(self, app_instance):
            super().__init__()
            self.app_instance = app_instance

        def on_mount(self) -> None:
            self.register_theme(theme)
            self.theme = theme.name
            self.push_screen(SettingsScreen(self.app_instance))

    return QAApp


async def main() -> None:
    from textual.widgets import Button, Collapsible, Select
    from textual.widgets._collapsible import CollapsibleTitle

    from tldw_chatbook.css.Themes.themes import agentic_terminal_theme

    tmp_dir = Path(tempfile.mkdtemp(prefix="rag-settings-qa-"))
    QAApp = _make_qa_app_class(agentic_terminal_theme)

    async def focus_collapsible_title(pilot, screen, collapsible_id: str) -> None:
        """Expand a Collapsible group and move focus onto its title bar --
        reproduces the original defect scenario (a focused Collapsible
        title) so the fix is visible in the capture.

        ``Widget.focus()`` auto-scrolls the real scrollable ancestor (the
        "Preference Detail" pane -- the card itself is an auto-height
        Vertical with no scroll of its own) via an ANIMATED
        ``scroll_visible()``. A single idle-wait ``pilot.pause()`` doesn't
        reliably outlast a multi-frame scroll animation for a target several
        rows below the fold (verified: worked for a 1-hop scroll, silently
        left an earlier target focused/on-screen for a longer one) -- use a
        wall-clock delay long enough for the animation to finish.
        """
        collapsible = screen.query_one(f"#{collapsible_id}", Collapsible)
        collapsible.collapsed = False
        collapsible.query_one(CollapsibleTitle).focus()
        await pilot.pause(0.6)

    # --- Captures 1 + 2: builtin "hybrid_basic" active (read-only). ---
    # Each capture "session" gets its own profiles_dir -- reusing one
    # directory across ConfigProfileManager instances let a later
    # clone_profile("hybrid_basic", "QA Demo Profile") collide with an
    # earlier session's file of the same name, which was silently
    # perturbing later widget state (observed: capture 4's "Set active"
    # focus not landing where the standalone repro said it would).
    _wire_profiles(tmp_dir / "session1", active_id="hybrid_basic")
    app1 = QAApp(_build_app_instance())
    async with app1.run_test(size=TERMINAL_SIZE) as pilot:
        screen = pilot.app.screen
        await _open_rag_category(pilot)

        # 1. Overview: Profiles block + Search group open (default state).
        svg = app1.export_screenshot(title="RAG category overview")
        (OUT / "01-rag-category-overview.svg").write_text(svg)

        # 2. Builtin read-only: DISTINCT from #1 (the original was a
        #    byte-duplicate) -- collapse Search to free vertical room, then
        #    expand only Embedding, so its disabled fields render together
        #    with the Profiles block + read-only banner, all in the initial
        #    (unscrolled) viewport.
        screen.query_one(
            "#settings-library-rag-search-group", Collapsible
        ).collapsed = True
        screen.query_one(
            "#settings-library-rag-embedding-group", Collapsible
        ).collapsed = False
        await pilot.pause(0.3)
        svg = app1.export_screenshot(title="RAG builtin read-only")
        (OUT / "02-builtin-readonly.svg").write_text(svg)

    # --- Captures 3 + 5: a writable (non-builtin) profile active. ---
    # Rerank ENABLED with top_k (25) > default results (15) so capture 5
    # demonstrates the display-only advisory Static under the Reranking group.
    _mgr2, _profile2, _state2 = _wire_profiles(tmp_dir / "session2", active_id=None)
    from tldw_chatbook.RAG_Search.reranker import RerankingConfig as _RerankingConfig
    _profile2.reranking_config = _RerankingConfig(top_k_to_rerank=25)
    _profile2.rag_config.search.enable_reranking = True
    _profile2.rag_config.search.default_top_k = 15
    _mgr2.save_profile(_profile2)
    app2 = QAApp(_build_app_instance())
    async with app2.run_test(size=TERMINAL_SIZE) as pilot:
        screen = pilot.app.screen
        await _open_rag_category(pilot)

        # 3. User profile editing: Embedding open (unfocused, for contrast)
        #    and Chunking open with its title FOCUSED -- reproduces the
        #    original defect scenario (a focused Collapsible title) so the
        #    fix is visible in the same shot composition.
        screen.query_one(
            "#settings-library-rag-embedding-group", Collapsible
        ).collapsed = False
        await pilot.pause()
        await focus_collapsible_title(
            pilot, screen, "settings-library-rag-chunking-group"
        )
        svg = app2.export_screenshot(title="RAG user profile editing")
        (OUT / "03-user-profile-editing.svg").write_text(svg)

        # 5. Reranking group: Chunking stays expanded (now unfocused, since
        #    focus moves to Reranking's title below) + Reranking open with
        #    its title FOCUSED.
        await focus_collapsible_title(
            pilot, screen, "settings-library-rag-reranking-group"
        )
        svg = app2.export_screenshot(title="RAG reranking group")
        (OUT / "05-reranking-group.svg").write_text(svg)

    # --- Capture 4: builtin active, a different (writable) profile
    #     selected in the dropdown, "Set active" focused. ---
    _mgr, other_profile, _state = _wire_profiles(
        tmp_dir / "session3", active_id="hybrid_basic"
    )
    app3 = QAApp(_build_app_instance())
    async with app3.run_test(size=TERMINAL_SIZE) as pilot:
        screen = pilot.app.screen
        await _open_rag_category(pilot)

        select = screen.query_one("#settings-library-rag-profile-select", Select)
        select.value = other_profile.id
        await pilot.pause()
        screen.query_one(
            "#settings-library-rag-profile-set-active", Button
        ).focus()
        # Note: the Button's focus style (bold + text-decoration:underline,
        # confirmed present in the exported SVG's <style> block) doesn't
        # render through cairosvg -- a converter limitation (cairosvg has no
        # text-decoration support), not a real rendering defect; the
        # Collapsible titles' focus cue is a drawn border-bottom instead,
        # which DOES render, hence it looking fine in captures 03/05.
        await pilot.pause(0.4)
        svg = app3.export_screenshot(title="RAG set-active flow")
        (OUT / "04-set-active-flow.svg").write_text(svg)

    print("captured:", sorted(p.name for p in OUT.glob("*.svg")))


if __name__ == "__main__":
    asyncio.run(main())
