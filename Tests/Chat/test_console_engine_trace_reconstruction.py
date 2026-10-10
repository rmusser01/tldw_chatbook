"""Registry-driven gateway values must remain independently trace-verifiable."""

from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_trace_final_values import _continuation_checkpoint
from tldw_chatbook.Chat.Chat_Functions import project_chat_handler_kwargs
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.Chat.console_trace_final_values import (
    reconstruct_provider_gateway_kwargs,
    verify_provider_request_shadow,
)
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_provenance import (
    ProviderArtifactTraceProvenance,
    ProviderRequestProvenance,
    SavedRevisionTraceProvenance,
    TraceOmissionReason,
    TraceProvenanceSource,
)
from tldw_chatbook.provider_registry import ENGINE_RECORDS

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize("provider", [record.key for record in ENGINE_RECORDS])
@pytest.mark.parametrize(
    "continuation", [False, True], ids=["empty", "owned-continuation"]
)
@pytest.mark.parametrize(
    "change", [None, "endpoint", "provider", "default", "continuation"]
)
def test_every_engine_preserves_selected_gateway_values(provider, continuation, change):
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    checkpoint = _continuation_checkpoint(provider="zai", model="glm-4.5")
    resolution = ConsoleProviderResolution(
        ready=True,
        provider=provider,
        execution_key=provider,
        model="fixture-model",
        api_key="fixture-key",
        streaming=False,
        base_url="https://selected.example/v1",
        temperature=0.5,
    )
    request = SimpleNamespace(
        system_message=None,
        messages_payload=({"role": "user", "content": "hello"},),
        tools=(),
        response_format={"type": "json_object"},
        continuation_groups=(SimpleNamespace(checkpoint=checkpoint),)
        if continuation
        else (),
    )
    saved = SavedRevisionTraceProvenance(new_opaque_id())
    provenance = ProviderRequestProvenance(
        messages=(saved,),
        messages_payload=(saved,),
        continuations=(
            ProviderArtifactTraceProvenance(TraceProvenanceSource.CONTINUATION, policy),
        )
        if continuation
        else (),
    )
    actual = ConsoleProviderGateway._chat_api_kwargs_from_prepared(resolution, request)
    expected = reconstruct_provider_gateway_kwargs(resolution, request)
    assert actual["api_base_url"] == "https://selected.example/v1"
    assert expected.get("api_base_url") == "https://selected.example/v1"
    if continuation:
        assert actual["provider_continuations"][0] is checkpoint
        assert expected.get("provider_continuations") == [checkpoint]
    else:
        assert "provider_continuations" not in expected
    if change == "endpoint":
        actual["api_base_url"] = "https://changed.example/v1"
    elif change == "provider":
        actual["api_endpoint"] = "foreign-provider"
    elif change == "default":
        actual["temp"] = 0.25
    elif change == "continuation":
        actual["provider_continuations"] = [
            _continuation_checkpoint(provider="zai", model="glm-4.5", result="changed")
        ]

    def project(values):
        endpoint = values.pop("api_endpoint")
        return project_chat_handler_kwargs(endpoint, values)

    bundle = verify_provider_request_shadow(
        actual_kwargs=actual,
        expected_kwargs=expected,
        provenance=provenance,
        project_handler_kwargs=project,
        known_credentials=(resolution.api_key,),
        endpoint_identity=resolution.base_url,
    )
    if change is None:
        assert bundle.available, bundle.omission_reason
        assert bundle.boundary_kwargs["api_base_url"] == "https://selected.example/v1"
    else:
        assert not bundle.available
        assert bundle.omission_reason is TraceOmissionReason.ALIGNMENT_MISMATCH
