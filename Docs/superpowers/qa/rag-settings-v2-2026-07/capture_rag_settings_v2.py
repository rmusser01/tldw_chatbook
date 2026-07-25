"""Capture the FIVE new Settings > Library/RAG UI states added by the v2 UX
program (task-541), for the user screen gate.

Derived from (not a mutation of)
``Docs/superpowers/qa/rag-settings-sp3-2026-07/capture_rag_settings.py`` --
same real-bundle-CSS + real-theme + real ``SettingsScreen`` approach (a bare
Textual ``App`` pushing the real screen, so exports show production
styling, not a bare-harness fallback), copied forward into this dated
directory so the SP3 originals keep reproducing the SP3-era states
unmodified.

New pieces this script needs that the SP3 one didn't:

- A "no user profiles" wiring variant (for the first-run starter panel --
  the predicate requires a genuinely empty user-profile list).
- A deterministic ``fetch_index_status()`` stub, patched onto
  ``settings_screen_module`` directly (module-level import binding, same
  seam ``Tests/UI/test_settings_rag_profile_region.py``'s
  ``_stub_index_status`` patches under pytest) -- the real function reads
  the ON-DISK Chroma collection for whatever profile ``resolve_active_rag_
  config()`` resolves, which is NOT the isolated tmp profiles_dir this
  script wires (a pre-existing quirk of the SP3 script too -- see its own
  docstring). Stubbing keeps every capture's index state exactly what the
  scenario needs regardless of this machine's real RAG state.

Run from the repo root with the cairo dylib on the search path (macOS
Homebrew build; see ``brew --prefix cairo``):

    DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \\
        .venv/bin/python3 Docs/superpowers/qa/rag-settings-v2-2026-07/capture_rag_settings_v2.py

Outputs SVGs directly into this directory; convert to PNG with this
directory's own copy of ``svg_to_png.py`` (cairosvg) -- it converts every
SVG under its own directory, so it's a plain copy, not a derivation.
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

# Same calibration as the SP3 script (linear in terminal cols/rows against
# Rich's SVG export viewBox) -- kept identical so the two capture sets are
# visually comparable side by side.
TERMINAL_SIZE = (167, 49)


def _wire_profiles(tmp_path: Path, *, active_id: str | None = None):
    """Point the Settings RAG adapter at an isolated profile store.

    Returns (manager, profile, state) where `profile` is a writable clone of
    the `hybrid_basic` builtin ("QA Demo Profile"). Mirrors
    ``Tests.UI.test_settings_configuration_hub._wire_rag_profile_adapter``
    without the pytest ``monkeypatch`` fixture (this is a standalone
    script) -- patches module attributes directly.
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


def _wire_profiles_no_user_profiles(tmp_path: Path, *, active_id: str = "hybrid_basic"):
    """Like ``_wire_profiles``, but WITHOUT the always-present "QA Demo
    Profile" user clone -- the first-run predicate (``is_first_run_state``)
    specifically requires a genuinely EMPTY user-profile list, which
    ``_wire_profiles`` can never produce (it registers a clone
    unconditionally). Mirrors
    ``Tests.UI.test_settings_rag_profile_region._wire_rag_profile_adapter_no_user_profiles``.
    """
    from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager
    import tldw_chatbook.UI.Screens.settings_rag_profile_adapter as rag_adapter_module

    mgr = ConfigProfileManager(profiles_dir=tmp_path / "profiles")
    state = {"active": active_id}
    rag_adapter_module._manager = lambda: mgr
    rag_adapter_module._active_profile_id = lambda: state["active"]
    return mgr, state


def _stub_index_status(state: str, *, count: int = 0, provenance: dict | None = None) -> None:
    """Deterministic ``fetch_index_status()`` -- patches the module-level
    binding INSIDE ``settings_screen`` (where it was imported by name),
    matching ``Tests/UI/test_settings_rag_profile_region.py``'s
    ``_stub_index_status`` seam. Must be called BEFORE the screen is
    mounted/the category is opened, since the category-show status fetch
    dispatches immediately."""
    import tldw_chatbook.UI.Screens.settings_screen as settings_screen_module

    settings_screen_module.fetch_index_status = lambda: {
        "state": state,
        "count": count,
        "provenance": provenance or {},
    }


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
    from textual.widgets import Button, Checkbox, Collapsible, Input, Select

    from tldw_chatbook.css.Themes.themes import agentic_terminal_theme

    tmp_dir = Path(tempfile.mkdtemp(prefix="rag-settings-v2-qa-"))
    QAApp = _make_qa_app_class(agentic_terminal_theme)

    # --- 1. First-run starter panel: builtin active, zero user profiles,
    #     index absent. AC5. ---
    _wire_profiles_no_user_profiles(tmp_dir / "session1", active_id="hybrid_basic")
    _stub_index_status("absent")
    app1 = QAApp(_build_app_instance())
    async with app1.run_test(size=TERMINAL_SIZE) as pilot:
        screen = pilot.app.screen
        await _open_rag_category(pilot)

        # The starter panel composes further down the "Preference Detail"
        # pane than the default 49-row viewport shows (the Profiles/Index
        # block above it, plus 5 collapsed-but-still-headered groups, push
        # it past the fold even though Search itself collapses on entry) --
        # scroll it into view, same technique the SP3 script uses for a
        # focused Collapsible title several rows below the fold.
        panel = screen.query_one("#settings-library-rag-starter-panel")
        assert panel.display is True
        panel.scroll_visible(animate=False)
        await pilot.pause()
        svg = app1.export_screenshot(title="RAG first-run starter panel")
        (OUT / "01-first-run-starter-panel.svg").write_text(svg)

    # --- 2. Preview-on-select: builtin active, a DIFFERENT writable
    #     profile browsed to in the picker (never clicking "Set active") --
    #     "Previewing: ..." banner + title + disabled fields. AC1. ---
    tmp2 = tmp_dir / "session2"
    from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager
    import tldw_chatbook.UI.Screens.settings_rag_profile_adapter as rag_adapter_module

    mgr2 = ConfigProfileManager(profiles_dir=tmp2 / "profiles")
    other_profile = mgr2.clone_profile("hybrid_basic", "Other RAG Profile")
    mgr2.save_profile(other_profile)
    state2 = {"active": "hybrid_basic"}
    rag_adapter_module._manager = lambda: mgr2
    rag_adapter_module._active_profile_id = lambda: state2["active"]
    _stub_index_status("absent")
    app2 = QAApp(_build_app_instance())
    async with app2.run_test(size=TERMINAL_SIZE) as pilot:
        screen = pilot.app.screen
        await _open_rag_category(pilot)

        select = screen.query_one("#settings-library-rag-profile-select", Select)
        select.value = other_profile.id
        await pilot.pause()
        assert screen._rag_preview_profile_id == other_profile.id

        # The Select itself sits near the top of the "Profiles" block and
        # is already in view -- but the "Previewing: ..." border title and
        # banner live on/inside `#settings-library-rag-editor-card`, a
        # SEPARATE, LATER sibling (composed after the Profiles/Index block
        # and the -- here hidden -- starter panel), well below the fold in
        # unscrolled document coordinates. Scroll the banner itself into
        # view (brings the editor card's top edge, and its border title,
        # into frame too).
        banner = screen.query_one("#settings-library-rag-preview-banner")
        assert banner.display is True
        banner.scroll_visible(animate=False, top=True)
        await pilot.pause()
        svg = app2.export_screenshot(title="RAG preview on select")
        (OUT / "02-preview-on-select.svg").write_text(svg)

    # --- 3. Pre-commit re-index confirm modal: writable active profile,
    #     index already BUILT, an index-determining field (embedding model)
    #     edited, Save clicked -> "Re-index required" confirm modal. AC2. ---
    _mgr3, profile3, _state3 = _wire_profiles(tmp_dir / "session3")
    _stub_index_status("built", count=1234, provenance={
        "embedding_model": profile3.rag_config.embedding.model,
        "chunk_size": profile3.rag_config.chunking.chunk_size,
        "chunk_overlap": profile3.rag_config.chunking.chunk_overlap,
    })
    app3 = QAApp(_build_app_instance())
    async with app3.run_test(size=TERMINAL_SIZE) as pilot:
        screen = pilot.app.screen
        await _open_rag_category(pilot)

        embedding_input = screen.query_one(
            "#settings-library-rag-embedding-model", Input
        )
        embedding_input.value = "a-brand-new-embedding-model-v2"
        screen.handle_library_rag_embedding_model_changed(
            Input.Changed(embedding_input, embedding_input.value)
        )
        await pilot.pause()

        await pilot.click("#settings-save-category")
        await pilot.pause()
        svg = app3.export_screenshot(title="RAG re-index confirm modal")
        (OUT / "03-reindex-confirm-modal.svg").write_text(svg)

    # --- 4. Checkbox toggles + dimmed rerank fields: Search group (default,
    #     open) shows the "Include citations" checkbox; Reranking group
    #     expanded with reranking OFF (the fresh clone's default) shows the
    #     "Enable reranking" checkbox plus the dimmed Reranker model / Rerank
    #     results fields with their "(enable reranking to edit)" suffix.
    #     AC4. ---
    _wire_profiles(tmp_dir / "session4")
    _stub_index_status("absent")
    app4 = QAApp(_build_app_instance())
    async with app4.run_test(size=TERMINAL_SIZE) as pilot:
        screen = pilot.app.screen
        await _open_rag_category(pilot)

        assert (
            screen.query_one(
                "#settings-library-rag-enable-reranking", Checkbox
            ).value
            is False
        )
        screen.query_one(
            "#settings-library-rag-reranking-group", Collapsible
        ).collapsed = False
        await pilot.pause(0.3)
        svg = app4.export_screenshot(title="RAG checkbox toggles + dimmed rerank fields")
        (OUT / "04-checkbox-toggles-dimmed-rerank.svg").write_text(svg)

    # --- 5. Context-sensitive inspector following a FOCUSED rerank field:
    #     reranking ON (so the field is genuinely interactive, not dimmed),
    #     Reranking group expanded, the Reranker model Input itself
    #     (not just the Collapsible title) focused -- the Scope Inspector
    #     rail shows "Focused group: Reranking" guidance. AC3. ---
    _mgr5, profile5, _state5 = _wire_profiles(tmp_dir / "session5")
    from tldw_chatbook.RAG_Search.reranker import RerankingConfig as _RerankingConfig

    profile5.reranking_config = _RerankingConfig()
    profile5.rag_config.search.enable_reranking = True
    _mgr5.save_profile(profile5)
    _stub_index_status("absent")
    app5 = QAApp(_build_app_instance())
    async with app5.run_test(size=TERMINAL_SIZE) as pilot:
        screen = pilot.app.screen
        await _open_rag_category(pilot)

        screen.query_one(
            "#settings-library-rag-reranking-group", Collapsible
        ).collapsed = False
        await pilot.pause()
        reranker_input = screen.query_one(
            "#settings-library-rag-reranker-model", Input
        )
        assert reranker_input.disabled is False
        reranker_input.focus()
        await pilot.pause(0.4)
        svg = app5.export_screenshot(title="RAG context-sensitive reranking inspector")
        (OUT / "05-context-inspector-reranking-focus.svg").write_text(svg)

    print("captured:", sorted(p.name for p in OUT.glob("*.svg")))


if __name__ == "__main__":
    asyncio.run(main())
