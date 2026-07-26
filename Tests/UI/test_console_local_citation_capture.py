from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from Tests.UI.test_console_dictionary_send_integration import (
    _CapturingGateway,
    _final_user_content,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.citation_evidence_models import EvidenceBundle
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
    LocalRagContextResult,
)
from tldw_chatbook.Library.library_rag_service import LibraryRagSearchRequest
from tldw_chatbook.Library.library_rag_state import LibraryRagResultRow
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module


@pytest.mark.asyncio
async def test_console_controller_wires_current_staged_rag_capture(monkeypatch):
    app = _build_test_app()
    launch = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Source",
        payload={"evidence_bundle": {"bundle_id": "unused"}},
        status="staged",
    )
    capture = AsyncMock(
        return_value=LocalRagContextResult(
            context="[S1] MEDIA — Source\nexact body",
            citation_builder=object(),
        )
    )
    monkeypatch.setattr(
        chat_screen_module,
        "capture_console_staged_evidence_for_chat",
        capture,
    )

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        screen._pending_console_launch_context = launch
        controller = screen._ensure_console_chat_controller()
        gateway = _CapturingGateway()
        controller.provider_gateway = gateway
        controller._agent_runtime_enabled = False

        result = await controller.submit_draft("question")

    assert result.accepted is True
    capture.assert_awaited_once_with(
        app,
        launch,
        user_message="question",
    )
    assert _final_user_content(gateway.captured) == (
        "Evidence: [S1] MEDIA — Source\nexact body\n\n---\n\nquestion"
    )


@pytest.mark.asyncio
async def test_console_library_rag_stages_all_retrieved_evidence():
    rows = tuple(
        LibraryRagResultRow.from_result(
            {
                "source_id": f"media-{index}",
                "chunk_id": f"chunk-{index}",
                "title": f"Source {index}",
                "content": f"Body {index}",
                "score": 1.0 - index / 10,
                "runtime_backend": "local",
                "source_type": "media",
            }
        )
        for index in (1, 2)
    )
    staged = []
    screen = SimpleNamespace(
        is_mounted=True,
        _stage_console_library_rag_launch=staged.append,
    )
    request = LibraryRagSearchRequest(
        query="question",
        source_types=("media",),
        mode="rag",
        top_k=5,
    )
    outcome = SimpleNamespace(results=rows)

    await chat_screen_module.ChatScreen._apply_console_library_rag_search_outcome(
        screen,
        request,
        outcome,
    )

    assert len(staged) == 1
    payload = staged[0].payload
    bundle = EvidenceBundle.from_payload(payload["evidence_bundle"])
    assert [reference.source_id for reference in bundle.references] == [
        "media-1",
        "media-2",
    ]
    assert payload["requested_top_k"] == 5
    assert payload["search_mode"] == "rag"
