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
from tldw_chatbook.Chat.citation_repair import CitationRepairContract
from tldw_chatbook.Chat.citation_trace_builder import CitationTraceBuilder
from tldw_chatbook.Chat.citation_trace_identity import (
    CitationFingerprintCodec,
    LocalCitationIdentityContext,
)
from tldw_chatbook.Chat.citation_trace_models import MarkerNamespace, PolicyCapability
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
    LocalRagContextResult,
)
from tldw_chatbook.Library.library_rag_service import LibraryRagSearchRequest
from tldw_chatbook.Library.library_rag_state import LibraryRagResultRow
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module


def _repair_contract(context: str) -> CitationRepairContract:
    return CitationRepairContract(
        schema_version=1,
        marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
        allowed_ordinals=(1,),
        evidence_context=context,
    )


@pytest.mark.asyncio
async def test_controller_returns_same_repair_contract_object_at_capture_boundary():
    context = "[S1] MEDIA — Source\nexact body"
    contract = _repair_contract(context)
    builder = CitationTraceBuilder.local(
        request_id="request-controller-boundary",
        generation_id="generation-controller-boundary",
        identity_context=LocalCitationIdentityContext(
            profile_id="profile-controller-boundary",
            local_authority_id="authority-controller-boundary",
            fingerprint_key_id="key-controller-boundary",
        ),
        fingerprint_codec=CitationFingerprintCodec(b"k" * 32),
        policy_version="controller-boundary-v1",
        policy_capabilities=(PolicyCapability.VIEW_SNAPSHOT,),
    )
    capture = AsyncMock(
        return_value=LocalRagContextResult(
            context=context,
            citation_builder=builder,
            prompt_evidence_set_id="prompt-set-independent",
            citation_repair_contract=contract,
        )
    )
    controller = ConsoleChatController(
        store=ConsoleChatStore(),
        provider_gateway=_CapturingGateway(),
        rag_capture_provider=capture,
    )

    captured = await controller._capture_rag_context("question")

    assert captured == (context, builder, "prompt-set-independent", contract)
    assert captured[1] is builder
    assert captured[3] is contract


@pytest.mark.asyncio
async def test_controller_rejects_duck_typed_repair_contract():
    context = "[S1] MEDIA — Source\nexact body"
    duck_contract = SimpleNamespace(
        schema_version=1,
        marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
        allowed_ordinals=(1,),
        evidence_context=context,
    )
    capture = AsyncMock(
        return_value=SimpleNamespace(
            context=context,
            citation_builder=None,
            prompt_evidence_set_id=None,
            citation_repair_contract=duck_contract,
        )
    )
    controller = ConsoleChatController(
        store=ConsoleChatStore(),
        provider_gateway=_CapturingGateway(),
        rag_capture_provider=capture,
    )

    captured = await controller._capture_rag_context("question")

    assert captured == (context, None, None, None)


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
