"""Product maturity Phase 1.6 empty/error/setup-state coverage contract."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button, Input, Static

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    RaisingLibraryNotesScopeService,
    RaisingSkillsScopeService,
    RaisingWatchlistsScopeService,
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticReadItLaterScopeService,
    _active_destination_screen,
    _visible_text,
    _wait_for_library_snapshot,
    _wait_for_personas_snapshot,
    _wait_for_skills_snapshot,
    _wait_for_wc_snapshot,
)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_personas_dictionaries import patch_character_paging
import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _wait_for_library_rag_query_ready,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook import config as app_config
from tldw_chatbook.Library import library_local_rag_search_service
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-6-empty-error-setup-states.md"
)
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_1_README = Path("Docs/superpowers/qa/product-maturity/phase-1/README.md")
TASK = Path(
    "backlog/tasks/task-8.6 - Product-Maturity-Phase-1.6-Empty-Error-Setup-State-Coverage.md"
)
LOCAL_PATH_PREFIXES = (
    "/Users/",
    "/home/",
    "/var/home/",
    "/private/var/folders/",
    "C:\\Users\\",
    "C:/Users/",
)


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _assert_no_local_path_prefixes(text: str) -> None:
    leaked_prefixes = [prefix for prefix in LOCAL_PATH_PREFIXES if prefix in text]
    assert not leaked_prefixes, (
        f"evidence contains local filesystem prefix(es): {leaked_prefixes}"
    )


def _test_cli_setting(section: str, key: str, default=None):
    if section == "splash_screen" and key == "enabled":
        return False
    return default


def _prepare_clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for env_var, path_name in (
        ("HOME", "home"),
        ("XDG_CONFIG_HOME", "xdg-config"),
        ("XDG_DATA_HOME", "xdg-data"),
        ("XDG_CACHE_HOME", "xdg-cache"),
    ):
        path = tmp_path / path_name
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(env_var, str(path))
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[chat_defaults]\nprovider = "OpenAI"\nmodel = "gpt-4o"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))


def _build_clean_setup_state_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _prepare_clean_environment(monkeypatch, tmp_path)
    app = _build_test_app()
    app.app_config["_first_run"] = True
    app._initial_tab_value = "chat"
    app.providers_models = {}
    return app


def _screen_text(app) -> str:
    pieces: list[str] = []
    for widget in app.screen.query(Static):
        pieces.append(str(widget.renderable))
    for widget in app.screen.query(Button):
        pieces.append(str(widget.label).strip())
    return "\n".join(piece for piece in pieces if piece.strip())


async def _wait_until(
    pilot,
    condition: Callable[[], bool],
    *,
    timeout_seconds: float = 10.0,
    interval_seconds: float = 0.05,
    context: str,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if condition():
            return
        await pilot.pause()
        await asyncio.sleep(interval_seconds)
    if condition():
        return
    raise AssertionError(
        f"condition was not met within {timeout_seconds:.1f}s for {context}"
    )


@pytest.mark.asyncio
async def test_clean_run_setup_and_runtime_blockers_expose_recovery_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _build_clean_setup_state_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_until(
                pilot,
                lambda: (
                    app.current_tab == "home"
                    and app.screen.__class__.__name__ == "HomeScreen"
                ),
                context="home initial setup state",
            )

            home_text = _screen_text(app)
            assert "Model: Blocked" in home_text
            assert "Set up Console model" in home_text
            assert "Console needs a working model before live AI tasks." in home_text

            await app.handle_screen_navigation(NavigateToScreen("chat"))
            await _wait_until(
                pilot,
                lambda: (
                    app.current_tab == "chat"
                    and app.screen.__class__.__name__ == "ChatScreen"
                ),
                context="console setup route",
            )
            # The clean-run fixture selects OpenAI/gpt-4o but provides no API
            # key, so the setup card resolves the provider-credential recovery
            # family rather than the provider/model pickers.
            await _wait_until(
                pilot,
                lambda: "Set up provider" in _screen_text(app),
                context="console provider setup controls",
            )
            assert (
                app.screen._console_provider_blocker_copy()
                == "Provider setup needed: API key missing for OpenAI"
            )
            # The shared Workbench recovery banner stays hidden — the setup
            # card's action button is the recovery/control surface now
            # (Phase 2 spec, section 2).
            recovery_callout = app.screen.query("#workbench-recovery-callout")
            assert recovery_callout and recovery_callout[0].display is False
            recovery_action = app.screen.query_one("#workbench-recovery-action", Button)
            assert recovery_action.display is False
            card_action = app.screen.query_one("#console-setup-modal-action", Button)
            assert card_action.display is True
            assert str(card_action.label) == "Set up provider"
            assert not list(app.screen.query("#console-open-provider-settings"))
            overflow_hint = app.screen.query_one("#nav-overflow-hint", Button)
            assert str(overflow_hint.label).strip() == "More ▾"

            await app.handle_screen_navigation(NavigateToScreen("acp"))
            await _wait_until(
                pilot,
                lambda: (
                    app.current_tab == "acp"
                    and app.screen.__class__.__name__ == "ACPScreen"
                ),
                context="acp runtime blocker",
            )
            acp_text = _screen_text(app)
            acp_launch = app.screen.query_one("#acp-launch-agent", Button)
            assert "Runtime not configured" in acp_text
            assert "Why: no ACP-compatible runtime is configured." in acp_text
            assert "Next: Configure ACP runtime setup in ACP before launch." in acp_text
            assert "Owner: ACP runtime." in acp_text
            assert acp_launch.disabled is True
            assert (
                "Configure an ACP-compatible runtime in ACP before launching an ACP agent."
                in str(acp_launch.tooltip)
            )


@pytest.mark.parametrize(
    ("route", "button_selector", "expected_copy", "setup"),
    [
        (
            "library",
            "#library-use-in-console",
            "Library source services unavailable; retry Library later.",
            "library-error",
        ),
        (
            "watchlists_collections",
            "#wc-attach-to-console",
            "Watchlists services unavailable; retry Watchlists later.",
            "wc-error",
        ),
        (
            "skills",
            "#skills-attach-to-console",
            "Skills service unavailable; retry Skills later.",
            "skills-error",
        ),
    ],
    ids=("library", "watchlists", "skills"),
)
@pytest.mark.asyncio
async def test_service_unavailable_states_disable_false_console_handoffs(
    route: str,
    button_selector: str,
    expected_copy: str,
    setup: str,
) -> None:
    app = _build_test_app()
    if setup == "library-error":
        app.notes_scope_service = RaisingLibraryNotesScopeService()
        app.media_reading_scope_service = StaticLibraryMediaScopeService([])
        app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
        wait_for_snapshot = _wait_for_library_snapshot
    elif setup == "wc-error":
        app.watchlist_scope_service = RaisingWatchlistsScopeService()
        app.collections_feeds_scope_service = StaticReadItLaterScopeService([])
        wait_for_snapshot = _wait_for_wc_snapshot
    elif setup == "skills-error":
        app.skills_scope_service = RaisingSkillsScopeService()
        wait_for_snapshot = _wait_for_skills_snapshot
    else:
        raise AssertionError(f"unexpected setup: {setup}")

    host = DestinationHarness(app, route)

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await wait_for_snapshot(screen, pilot)
        button = screen.query_one(button_selector, Button)

        assert expected_copy in _visible_text(screen)
        if setup == "library-error":
            assert button.disabled is False
            assert button.has_class("library-source-action-blocked")
        else:
            assert button.disabled is True
        assert "unavailable" in str(button.tooltip).lower()


@pytest.mark.asyncio
async def test_personas_default_state_disables_false_console_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Personas starts local-first; Console attach stays blocked until selection.

    F-031 auto-selects the first library row on first paint when rows exist,
    so the no-selection contract this test pins needs an empty library
    (stubbed deterministically; the harness otherwise reads the ambient
    character DB).
    """
    monkeypatch.setattr(character_handler_module, "fetch_all_characters", lambda: [])
    patch_character_paging(monkeypatch, records=[])
    app = _build_test_app()
    host = DestinationHarness(app, "personas")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_personas_snapshot(screen, pilot)
        button = screen.query_one("#personas-attach-to-console", Button)

        visible_text = _visible_text(screen)
        assert "Pick a character or persona to start chatting." in visible_text
        assert button.disabled is True


@pytest.mark.asyncio
async def test_library_rag_mode_dependency_missing_state_names_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase-1.6 re-add (task-14, RAG UX v2 PR-2): the SearchRAGWindow
    dependency-missing exemplar this file used to carry.

    PR #1258 (task 7b470c344) retired the standalone Search screen and
    deleted that exemplar outright rather than mechanically re-pointing it
    -- SearchRAGWindow's harness (a bare widget mount) has no equivalent
    against the Library RAG panel (a full rail-shell screen). The
    retirement commit flagged re-adding equivalent coverage against
    Library's RAG panel as backlog; this is that re-add, against the sole
    surviving Search/RAG surface. Uses ``LibraryHarness`` (the Library
    screen's own dedicated harness from ``test_library_shell``), not the
    generic ``DestinationHarness`` this file otherwise uses, so the panel
    mounts exactly as production wires it and the flow reaches
    ``LibraryScreen._start_library_rag_query`` the same way a live rail
    press -> mode toggle -> Run press would.

    Task 13 (625a20a16) taught the RAG-unavailable recovery copy to name
    the pip extra to install; this test pins that exact copy so a future
    revert of Task 13 fails here.

    Two assertions, not one: the failure path (RAG mode renders the
    recovery copy, install hint included) AND the positive control
    (keyword "search" mode -- the default canvas mode -- still returns
    real results with the same deps missing). The recovery copy's own
    "or switch mode to Search" escape clause only means something if
    keyword mode genuinely keeps working when the embeddings deps are
    gone; a test that only checked the failure path would let a
    regression that broke that escape hatch pass silently.

    Also needs a resolvable provider credential: PR-T2 Task 7 made
    `library_rag_answer_provider_ready` check `Chat/provider_readiness.
    get_provider_readiness`, not just an endpoint NAME, so `rag` mode's Run
    gate is genuinely blocked without one -- see `test_product_maturity_
    gate16_library_search_rag.py`'s `_ready_library_rag_provider` for the
    same pattern applied at file scope there.
    """
    monkeypatch.setattr(
        library_local_rag_search_service,
        "embeddings_rag_deps_installed",
        lambda: False,
    )
    monkeypatch.setattr(app_config, "default_api_endpoint", "openai", raising=False)
    real_load_settings = app_config.load_settings

    def _load_settings_with_ready_openai_key(*args, **kwargs):
        settings = dict(real_load_settings(*args, **kwargs))
        api_settings = dict(settings.get("api_settings") or {})
        openai_settings = dict(api_settings.get("openai") or {})
        openai_settings["api_key"] = "sk-test-phase1-ready-key"
        api_settings["openai"] = openai_settings
        settings["api_settings"] = api_settings
        return settings

    monkeypatch.setattr(
        app_config, "load_settings", _load_settings_with_ready_openai_key
    )

    app = _build_test_app()
    assert getattr(app, "_rag_service", None) is None
    _seed_conversations(
        app,
        [
            {
                "title": "Planning Chat",
                "conversation_id": "chat-1",
                "message_count": 2,
                "updated_at": "2026-06-01T10:00:00Z",
            }
        ],
        notes=[
            {
                "title": "Quarterly Retention Policy",
                "id": "note-1",
                "content": "Applies to all local archives.",
            }
        ],
    )
    host = LibraryHarness(app)
    query = "Quarterly"

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        # --- Positive control: keyword ("search") mode, the default
        # canvas mode, must keep returning real results -- it never
        # touches the RAG runtime the deps gate blocks.
        assert str(screen.query_one("#library-rag-mode-toggle", Button).label) == (
            "mode: ✓ Search ⇄ RAG Answer"
        )
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_library_rag_query_ready(screen, pilot, query)
        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")

        assert not screen.query("#library-rag-service-error")
        assert "Quarterly Retention Policy" in _visible_text(screen)

        # --- Failure path: cycle to RAG mode and run the same query.
        screen.query_one("#library-rag-mode-toggle", Button).press()
        for _ in range(150):
            toggles = list(screen.query("#library-rag-mode-toggle"))
            if toggles and str(toggles[0].label) == "mode: Search ⇄ ✓ RAG Answer":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Mode toggle never switched to RAG Answer.")

        await _wait_for_library_rag_query_ready(screen, pilot, query)
        run_button = screen.query_one("#library-rag-run-query", Button)
        assert run_button.disabled is False

        run_button.press()
        await _wait_for_selector(screen, pilot, "#library-rag-service-error")

        visible_text = _visible_text(screen)
        assert "RAG unavailable" in visible_text
        # The display sanitizer's ``escape_markup`` pass backslash-escapes
        # the opening "[" (same reason ``library-rag-history-*`` labels
        # escape entries -- unescaped, Rich would try to parse
        # "[embeddings_rag]" as a style tag); the backslash resolves away
        # at real paint time and is not visible to the user, but
        # ``_visible_text`` reads ``.renderable`` directly, before that
        # resolution. Mirrors
        # ``test_product_maturity_gate16_library_search_rag.py``'s Task 13
        # assertion for the same copy.
        assert (
            'Install RAG support: pip install "tldw_chatbook\\[embeddings_rag]", '
            "then restart, or switch mode to Search." in visible_text
        )
        # (2026-08-03 task-15 finding-1 fix) The display sanitizer no longer
        # HTML-entity-escapes plain text for display -- a Rich `Static`
        # never decodes "&gt;" back to ">", so re-encoding here was itself
        # the over-escaping bug finding 1 fixed (for "&" in evidence
        # snippets; ">" in this recovery copy is the same class of bug).
        # The recovery route's ">" now renders as the literal character.
        assert "Recovery: Settings > RAG." in visible_text
