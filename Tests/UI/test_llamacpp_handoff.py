from __future__ import annotations

import pytest


def test_exact_ready_generation_required_and_store_copies_the_intent():
    from tldw_chatbook.LLM_Management.llamacpp_connection import (
        LlamaCppConnectionOwner,
        LlamaCppProbeResult,
    )
    from tldw_chatbook.UI.Navigation.llamacpp_handoff import (
        LlamaCppConsoleIntent,
        LlamaCppDefaultIntent,
        owner_has_current_intent,
    )
    from tldw_chatbook.UI.Navigation.pending_handoff_store import (
        HandoffChannel,
        PendingHandoffStore,
    )

    owner = LlamaCppConnectionOwner()
    request = owner.begin("http://localhost:8080", runtime_owner="external_server")
    owner.accept(LlamaCppProbeResult(request, "ready", ("org/model",), "org/model"))
    target = owner.snapshot().target
    store = PendingHandoffStore()
    for intent_type, channel in [
        (LlamaCppConsoleIntent, HandoffChannel.LLAMACPP_CONSOLE),
        (LlamaCppDefaultIntent, HandoffChannel.LLAMACPP_DEFAULT),
    ]:
        intent = intent_type.from_target(target)
        assert owner_has_current_intent(owner, intent)
        store.stage(channel, intent)
        claim = store.claim(channel)
        assert claim.value == intent
        assert claim.value is not intent
        assert store.acknowledge(claim)
    owner.invalidate()
    assert not owner_has_current_intent(owner, intent)


@pytest.mark.parametrize(
    "model", ["/private/model.gguf", "bad\u202eidentity", "", "name\nvalue"]
)
def test_handoff_rejects_private_or_malformed_identity(model):
    from tldw_chatbook.UI.Navigation.llamacpp_handoff import LlamaCppConsoleIntent

    with pytest.raises(ValueError):
        LlamaCppConsoleIntent("http://localhost:8080", model, 1, "external_server")
