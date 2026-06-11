"""Product maturity Phase 1.7 narrow core-loop proof contract."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Static, TextArea

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Widgets.Chat_Widgets.chat_handoff_card import ChatHandoffCard


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path("Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-7-core-loop-proof.md")
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_1_README = Path("Docs/superpowers/qa/product-maturity/phase-1/README.md")
TASK = Path("backlog/tasks/task-8.7 - Product-Maturity-Phase-1.7-Narrow-Core-Loop-Proof.md")


def _text(path: Path) -> str:
    """Read a repository-relative text fixture.

    Args:
        path: Repository-relative path to read.

    Returns:
        UTF-8 decoded file contents.
    """
    return (REPO_ROOT / path).read_text(encoding="utf-8")


async def _wait_until(
    pilot,
    condition: Callable[[], bool],
    *,
    timeout_seconds: float = 10.0,
    interval_seconds: float = 0.05,
    context: str,
) -> None:
    """Wait for an async Textual pilot condition to become true.

    Args:
        pilot: Textual test pilot driving the running app.
        condition: Zero-argument predicate that returns true when ready.
        timeout_seconds: Maximum time to wait before failing.
        interval_seconds: Delay between event-loop polls.
        context: Human-readable state being awaited for failure messages.

    Raises:
        AssertionError: If the condition is still false after the timeout.
    """
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if condition():
            return
        await pilot.pause()
        await asyncio.sleep(interval_seconds)
    if condition():
        return
    raise AssertionError(f"condition was not met within {timeout_seconds:.1f}s for {context}")


def _core_loop_payload() -> ChatHandoffPayload:
    """Build the deterministic Search/RAG handoff payload used by Phase 1.7.

    Returns:
        Handoff payload that stages a local RAG result into Console.
    """
    return ChatHandoffPayload(
        source="search-rag",
        item_type="rag-result",
        title="Transcript chunk: Agentic terminal design",
        body="The source states that Console is the live agentic control surface.",
        source_id="rag-chunk-1",
        content_ref="rag:rag-chunk-1",
        display_summary="Console is the live agentic control surface.",
        suggested_prompt="Use this RAG evidence to summarize the Console role.",
        runtime_backend="local",
        source_owner="local",
        source_selector_state="local",
        discovery_owner="rag_search",
        discovery_entity_id="rag-chunk-1",
        metadata={"score": 0.91, "source": "notes"},
    )


def _test_cli_setting(section: str, key: str, default=None):
    """Return deterministic settings for the running-app core-loop test.

    Args:
        section: Config section name requested by the app.
        key: Config key requested by the app.
        default: Caller-provided fallback value.

    Returns:
        Test-pinned setting value for relevant chat/splash keys, otherwise
        the caller default.
    """
    if section == "splash_screen" and key == "enabled":
        return False
    if section == "chat_defaults" and key == "enable_tabs":
        return True
    if section == "chat_defaults" and key == "max_tabs":
        return 10
    return default


@pytest.mark.skip(reason="Stale release-era snapshot (copy/evidence drifted); re-pin or retire via backlog task-98")
@pytest.mark.asyncio
async def test_search_rag_result_stages_context_into_console_core_loop() -> None:
    app = _build_test_app()
    app.app_config["_first_run"] = False
    app._initial_tab_value = "home"
    payload = _core_loop_payload()

    with (
        patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting),
        patch("tldw_chatbook.UI.Chat_Window_Enhanced.get_cli_setting", side_effect=_test_cli_setting),
        patch(
            "tldw_chatbook.Widgets.Chat_Widgets.chat_tab_container.get_cli_setting",
            side_effect=_test_cli_setting,
        ),
    ):
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_until(
                pilot,
                lambda: app.current_tab == "home",
                context="home initial route",
            )

            app.open_chat_with_handoff(payload)

            await _wait_until(
                pilot,
                lambda: app.current_tab == "chat" and app.screen.__class__.__name__ == "ChatScreen",
                context="console route after RAG handoff",
            )
            await _wait_until(
                pilot,
                lambda: bool(app.screen.query(ChatHandoffCard)),
                context="staged RAG context card",
            )

            card_text = "\n".join(
                str(card.query_one(".chat-handoff-card-body", Static).renderable)
                for card in app.screen.query(ChatHandoffCard)
            )
            draft_text = "\n".join(widget.text for widget in app.screen.query(TextArea))

            assert "Context staged from RAG Search" in card_text
            assert "Title: Transcript chunk: Agentic terminal design" in card_text
            assert "Type: rag-result" in card_text
            assert "Backend: local" in card_text
            assert "Source: Local source" in card_text
            assert "score: 0.91" in card_text
            assert payload.suggested_prompt in draft_text
            assert app.pending_chat_handoff is None


