from dataclasses import replace
from hashlib import sha256

import pytest

from tldw_chatbook.Research_Workspace.chat_models import (
    MAX_CHAT_APPEND_RECEIPTS,
    ResearchChatAppendReceipt,
    ResearchChatTurnRequest,
    ResearchRetrievalSettings,
    ResearchSourceVersionRef,
)
from tldw_chatbook.Research_Workspace.contracts import (
    ProcessingRoute,
    QualifiedWorkspaceRef,
    WorkspaceDataSource,
)
from tldw_chatbook.Research_Workspace.egress_consent import (
    MAX_EGRESS_CONSENTS,
    ResearchEgressConsentService,
)
from tldw_chatbook.Research_Workspace.layout_state import ResearchPanePreferences
from tldw_chatbook.Research_Workspace.overlay_store import (
    OverlayValidationError,
    ResearchPresentationOverlayStore,
)


def remote_turn(workspace_id="workspace-a", **route_changes):
    ref = QualifiedWorkspaceRef("local", workspace_id)
    return ResearchChatTurnRequest(
        workspace=ref,
        conversation_id="conversation-a",
        user_text="Private unsent question",
        mode="rag",
        source_snapshot=(
            ResearchSourceVersionRef(
                ref,
                "association-a",
                "42",
                "article",
                catalog_item_version=1,
                ready_modes=("fts",),
            ),
        ),
        retrieval=ResearchRetrievalSettings(),
        processing_route=replace(
            ProcessingRoute(
                WorkspaceDataSource.LOCAL,
                "local-chat",
                provider="openai-compatible",
                model="model-a",
                remote=True,
                endpoint_class="public_network",
                endpoint_fingerprint=sha256(b"https://first.example").hexdigest(),
            ),
            **route_changes,
        ),
        context_revision=1,
    )


def test_local_source_approval_is_exact_to_workspace_and_route():
    service = ResearchEgressConsentService()
    request = remote_turn()
    preflight = service.preflight(request)
    assert preflight.source_count == 1
    assert preflight.source_types == ("article",)
    assert preflight.endpoint_class == "public_network"
    assert "Private unsent question" not in repr(preflight)
    approved = service.approve(preflight, ())
    assert service.is_approved(request, approved)
    assert service.preflight(request, approved) is None
    for changed in (
        remote_turn(workspace_id="workspace-b"),
        remote_turn(provider="another-provider"),
        remote_turn(endpoint_class="private_network"),
        remote_turn(endpoint_fingerprint=sha256(b"https://second.example").hexdigest()),
        remote_turn(redaction_policy="strip-identifiers"),
        remote_turn(source_body_mode="full"),
    ):
        assert not service.is_approved(changed, approved)
        assert service.preflight(changed, approved) is not None
    assert service.preflight(replace(request, mode="general")) is None
    assert (
        service.preflight(remote_turn(remote=False, endpoint_class="on_device")) is None
    )
    with pytest.raises(ValueError, match="destination"):
        service.preflight(
            remote_turn(endpoint_class="unknown", endpoint_fingerprint="")
        )


def test_consent_eviction_is_bounded_and_keeps_the_newest_route():
    service = ResearchEgressConsentService()
    approved = ()
    for index in range(MAX_EGRESS_CONSENTS + 1):
        request = remote_turn(provider=f"provider-{index}")
        approved = service.approve(service.preflight(request), approved)
    assert len(approved) == MAX_EGRESS_CONSENTS
    assert service.is_approved(request, approved)
    assert not service.is_approved(remote_turn(provider="provider-0"), approved)


def test_chat_overlay_only_stores_private_draft_and_bounded_recovery_metadata(tmp_path):
    store = ResearchPresentationOverlayStore(tmp_path / "research" / "overlay.json")
    ref = remote_turn().workspace
    prefs = ResearchPanePreferences()
    receipt = ResearchChatAppendReceipt(
        "turn-a",
        "conversation-a",
        "user_committed",
        user_message_id="user-a",
    )
    consent = ResearchEgressConsentService()
    fingerprints = consent.approve(consent.preflight(remote_turn()), ())
    saved = store.save(
        ref,
        prefs,
        expected_revision=0,
        chat_draft="Unsent question",
        chat_conversation_id="conversation-a",
        chat_egress_consents=fingerprints,
        chat_append_receipts=(receipt,),
    )
    assert store.load(ref) == saved
    assert store.load(remote_turn(workspace_id="other").workspace) is None
    raw = store.path.read_text()
    assert "Unsent question" in raw
    assert "https://first.example" not in raw
    assert "Private unsent question" not in raw

    with pytest.raises(OverlayValidationError, match="fingerprint"):
        store.save(
            ref, prefs, expected_revision=1, chat_egress_consents=("token=secret",)
        )
    with pytest.raises(OverlayValidationError, match="receipt"):
        store.save(
            ref,
            prefs,
            expected_revision=1,
            chat_append_receipts=(receipt,) * (MAX_CHAT_APPEND_RECEIPTS + 1),
        )
    updated = store.save(ref, prefs, expected_revision=1, chat_draft="")
    assert updated.chat_append_receipts == (receipt,)
    assert updated.chat_egress_consents == fingerprints
    assert "Unsent question" not in store.path.read_text()
