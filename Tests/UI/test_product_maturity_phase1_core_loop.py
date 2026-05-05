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
    return (REPO_ROOT / path).read_text(encoding="utf-8")


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
    raise AssertionError(f"condition was not met within {timeout_seconds:.1f}s for {context}")


def _core_loop_payload() -> ChatHandoffPayload:
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
    if section == "splash_screen" and key == "enabled":
        return False
    if section == "chat_defaults" and key == "enable_tabs":
        return True
    return default


@pytest.mark.asyncio
async def test_search_rag_result_stages_context_into_console_core_loop() -> None:
    app = _build_test_app()
    app.app_config["_first_run"] = False
    app._initial_tab_value = "home"
    payload = _core_loop_payload()

    with (
        patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting),
        patch("tldw_chatbook.UI.Chat_Window_Enhanced.get_cli_setting", side_effect=_test_cli_setting),
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
                str(widget.renderable)
                for card in app.screen.query(ChatHandoffCard)
                for widget in card.query(Static)
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


def test_phase_1_7_core_loop_evidence_is_tracked() -> None:
    evidence = _text(EVIDENCE)
    tracker = _text(TRACKER)
    readme = _text(PHASE_1_README)
    task = _text(TASK)

    assert "Phase 1.7 narrow core-loop proof" in evidence
    assert "Search/RAG result -> Console staged context" in evidence
    assert "Source authority: local" in evidence
    assert "No P0/P1 defects found" in evidence
    assert "Phase 1.7 verifies the remaining narrow core-loop proof gate" in tracker
    assert (
        "Phase 1: QA Baseline And Usability Guardrails | "
        "Establish clean-run usability guardrails before feature depth. | verified"
    ) in tracker
    assert "2026-05-05-phase-1-7-core-loop-proof.md" in readme
    assert "status: Done" in task
    assert "- [x] #1" in task
    assert "- [x] #4" in task
