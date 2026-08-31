"""Final provider-bound semantic value verification for normalized traces."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from tldw_chatbook.Chat.Chat_Functions import (
    API_CALL_HANDLERS,
    PROVIDER_PARAM_MAP,
    SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS,
    project_chat_handler_kwargs,
)
from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY
from tldw_chatbook.Chat.console_session_settings import (
    CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
)

from tldw_chatbook.Chat.console_trace_final_values import (
    FinalValueIntent,
    ProviderCredentialSource,
    verify_provider_request_shadow,
)
from tldw_chatbook.Chat.console_trace_models import new_opaque_id
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy
from tldw_chatbook.Chat.console_trace_provenance import (
    ProviderArtifactTraceProvenance,
    ProviderRequestProvenance,
    SavedRevisionTraceProvenance,
    TraceOmissionReason,
    TraceProvenanceSource,
)
from tldw_chatbook.Chat.provider_continuation import (
    ProviderContinuationCheckpoint,
    parse_provider_continuation_json,
)


def _provenance() -> ProviderRequestProvenance:
    saved = SavedRevisionTraceProvenance(new_opaque_id())
    return ProviderRequestProvenance(messages=(saved,), messages_payload=(saved,))


def _artifact(source: TraceProvenanceSource) -> ProviderArtifactTraceProvenance:
    return ProviderArtifactTraceProvenance(
        source,
        FrozenTracePolicy(
            policy_id=new_opaque_id(),
            credential_filter_version="credentials-v1",
            pii_redaction_enabled=False,
            pii_ruleset_revision_id=None,
        ),
    )


def _project(values: dict[str, object]) -> dict[str, object]:
    return {
        "input_data": values["messages_payload"],
        "model": values["model"],
    }


def _chatbook_project(values: dict[str, object]) -> dict[str, object]:
    endpoint = values.pop("api_endpoint")
    assert isinstance(endpoint, str)
    return project_chat_handler_kwargs(endpoint, values)


def _continuation_checkpoint(
    *,
    provider: str = "moonshot",
    model: str = "kimi-k2",
    result: str = "done",
) -> ProviderContinuationCheckpoint:
    return parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 2,
            "provider": provider,
            "protocol": "chat_completions",
            "model": model,
            "api_base_url": f"https://{provider}.invalid/v1",
            "state": "complete",
            "rounds": [
                {
                    "assistant_content": "answer",
                    "reasoning_blocks": ["private reasoning"],
                    "calls": [
                        {
                            "call_id": "call_1",
                            "name": "lookup",
                            "arguments": "{}",
                            "state": "completed",
                            "result": result,
                        }
                    ],
                }
            ],
        }
    )


def test_verified_shadow_binds_revisions_and_structural_values() -> None:
    secret = "resolved-credential-without-a-pattern"
    actual = {
        "api_endpoint": "openai",
        "messages_payload": [{"role": "user", "content": "hello"}],
        "api_key": secret,
        "model": "gpt-4.1",
    }

    result = verify_provider_request_shadow(
        actual_kwargs=actual,
        expected_kwargs=dict(actual),
        provenance=_provenance(),
        project_handler_kwargs=_project,
        known_credentials=(secret,),
    )

    assert result.available is True
    assert result.redacted is True
    assert result.credential_source is ProviderCredentialSource.RESOLVED_PRESENT
    assert result.boundary_kwargs == {
        "api_endpoint": "openai",
        "messages_payload": [{"role": "user", "content": "hello"}],
        "model": "gpt-4.1",
    }
    assert result.handler_kwargs == {
        "input_data": [{"role": "user", "content": "hello"}],
        "model": "gpt-4.1",
    }
    message = next(
        item for item in result.components if item.name == "messages_payload"
    )
    assert message.intents == (FinalValueIntent.REVISION_REFERENCE,)
    assert secret not in repr(result)


def test_mismatch_is_content_free_and_never_projects_actual_values() -> None:
    secret = "mismatch-secret"
    projected = False

    def project(_values: dict[str, object]) -> dict[str, object]:
        nonlocal projected
        projected = True
        return {}

    result = verify_provider_request_shadow(
        actual_kwargs={
            "api_endpoint": "openai",
            "messages_payload": [{"role": "user", "content": secret}],
        },
        expected_kwargs={
            "api_endpoint": "openai",
            "messages_payload": [{"role": "user", "content": "different"}],
        },
        provenance=_provenance(),
        project_handler_kwargs=project,
        known_credentials=(secret,),
    )

    assert result.available is False
    assert result.omission_reason is TraceOmissionReason.ALIGNMENT_MISMATCH
    assert result.boundary_kwargs == {}
    assert result.handler_kwargs == {}
    assert projected is False
    assert secret not in repr(result)
    assert secret not in json.dumps(result.as_content_free_record())


def test_raw_semantic_mismatch_cannot_collapse_through_credential_redaction() -> None:
    first = "resolved-content-secret-one"
    second = "resolved-content-secret-two"

    result = verify_provider_request_shadow(
        actual_kwargs={
            "api_endpoint": "openai",
            "messages_payload": [{"role": "user", "content": first}],
        },
        expected_kwargs={
            "api_endpoint": "openai",
            "messages_payload": [{"role": "user", "content": second}],
        },
        provenance=_provenance(),
        project_handler_kwargs=_project,
        known_credentials=(first, second),
    )

    assert result.available is False
    assert result.omission_reason is TraceOmissionReason.ALIGNMENT_MISMATCH
    assert first not in repr(result)
    assert second not in repr(result)


def test_resolved_credential_values_and_category_must_match_ephemerally() -> None:
    actual = {
        "api_endpoint": "openai",
        "messages_payload": [{"role": "user", "content": "safe"}],
        "api_key": "resolved-key-one",
        "model": "gpt-4.1",
    }
    expected = {**actual, "api_key": "resolved-key-two"}

    result = verify_provider_request_shadow(
        actual_kwargs=actual,
        expected_kwargs=expected,
        provenance=_provenance(),
        project_handler_kwargs=_project,
        known_credentials=("resolved-key-one", "resolved-key-two"),
    )

    assert result.available is False
    assert result.credential_source is ProviderCredentialSource.RESOLVED_PRESENT
    assert result.omission_reason is TraceOmissionReason.ALIGNMENT_MISMATCH
    assert "resolved-key-one" not in repr(result)
    assert "resolved-key-two" not in repr(result)


def test_successful_shadow_redaction_has_bounded_reason_metadata() -> None:
    values = {
        "api_endpoint": "openai",
        "messages_payload": [{"role": "user", "content": "safe"}],
        "api_key": "resolved-key-one",
        "model": "gpt-4.1",
    }

    result = verify_provider_request_shadow(
        actual_kwargs=values,
        expected_kwargs=dict(values),
        provenance=_provenance(),
        project_handler_kwargs=_project,
        known_credentials=("resolved-key-one",),
    )

    assert result.available is True
    assert result.redacted is True
    assert ("credential_redaction", "mandatory_filter") in {
        (overlay.kind, overlay.source) for overlay in result.overlays
    }
    assert "resolved-key-one" not in repr(result)


def test_endpoint_identity_is_canonical_or_content_free_unavailable() -> None:
    values = {
        "api_endpoint": "openai",
        "messages_payload": [{"role": "user", "content": "safe"}],
        "model": "gpt-4.1",
    }
    equivalent = verify_provider_request_shadow(
        actual_kwargs=values,
        expected_kwargs=dict(values),
        provenance=_provenance(),
        project_handler_kwargs=_project,
        endpoint_identity="HTTPS://Example.Invalid:443/v1/",
    )
    invalid = verify_provider_request_shadow(
        actual_kwargs=values,
        expected_kwargs=dict(values),
        provenance=_provenance(),
        project_handler_kwargs=_project,
        endpoint_identity="not a valid endpoint token",
    )

    assert equivalent.available is True
    assert equivalent.endpoint_identity == "https://example.invalid/v1"
    assert invalid.available is False
    assert invalid.as_content_free_record() == {
        "available": False,
        "omission_reason": "sanitizer_failed",
    }


def test_sanitizer_or_projection_failure_returns_same_content_free_shape() -> None:
    recursive: list[object] = []
    recursive.append(recursive)

    sanitize_failure = verify_provider_request_shadow(
        actual_kwargs={"api_endpoint": "openai", "messages_payload": recursive},
        expected_kwargs={"api_endpoint": "openai", "messages_payload": recursive},
        provenance=_provenance(),
        project_handler_kwargs=_project,
    )
    projection_failure = verify_provider_request_shadow(
        actual_kwargs={"api_endpoint": "openai", "messages_payload": []},
        expected_kwargs={"api_endpoint": "openai", "messages_payload": []},
        provenance=ProviderRequestProvenance(),
        project_handler_kwargs=lambda _values: (_ for _ in ()).throw(
            RuntimeError("unsafe secret context")
        ),
    )

    assert sanitize_failure.available is False
    assert sanitize_failure.omission_reason is TraceOmissionReason.SANITIZER_FAILED
    assert projection_failure.available is False
    assert projection_failure.omission_reason is TraceOmissionReason.SANITIZER_FAILED
    assert "unsafe secret context" not in repr(projection_failure)


def test_provider_overlay_provenance_is_bounded_and_structural() -> None:
    actual = {
        "api_endpoint": "moonshot",
        "messages_payload": [],
        "model": "kimi-k2",
        "request_retries": 2,
        "request_retry_delay": 0.5,
        "provider_continuations": [{"kind": "opaque"}],
    }

    result = verify_provider_request_shadow(
        actual_kwargs=actual,
        expected_kwargs=dict(actual),
        provenance=ProviderRequestProvenance(
            continuations=(_artifact(TraceProvenanceSource.CONTINUATION),)
        ),
        project_handler_kwargs=lambda values: values,
    )

    assert result.available is True
    assert {item.kind for item in result.overlays} >= {
        "transport_retry_policy",
        "provider_continuation",
    }
    assert all(len(item.kind) <= 64 for item in result.overlays)


def test_exact_provider_continuation_checkpoint_is_normalized_then_sanitized() -> None:
    secret = "checkpoint-secret"
    checkpoint = _continuation_checkpoint(result=f"prefix{secret}suffix")
    actual = {
        "api_endpoint": "moonshot",
        "messages_payload": [],
        "model": "kimi-k2",
        "provider_continuations": [checkpoint],
    }

    result = verify_provider_request_shadow(
        actual_kwargs=actual,
        expected_kwargs=dict(actual),
        provenance=ProviderRequestProvenance(
            continuations=(_artifact(TraceProvenanceSource.CONTINUATION),)
        ),
        project_handler_kwargs=lambda values: values,
        known_credentials=(secret,),
    )

    assert result.available is True
    assert result.redacted is True
    normalized = result.boundary_kwargs["provider_continuations"]
    assert normalized == [
        {
            "schema_version": 1,
            "checkpoint_revision": 2,
            "provider": "moonshot",
            "protocol": "chat_completions",
            "model": "kimi-k2",
            "api_base_url": "https://moonshot.invalid/v1",
            "state": "complete",
            "rounds": [
                {
                    "assistant_content": "answer",
                    "reasoning_blocks": ["private reasoning"],
                    "calls": [
                        {
                            "call_id": "call_1",
                            "name": "lookup",
                            "arguments": "{}",
                            "state": "completed",
                            "result": "prefix[credential omitted]suffix",
                        }
                    ],
                }
            ],
        }
    ]
    assert "provider_continuation" in {item.kind for item in result.overlays}
    assert secret not in repr(result)
    assert secret not in json.dumps(result.boundary_kwargs)


def test_provider_continuation_mismatch_fails_before_lossy_sanitization() -> None:
    first = "checkpoint-secret-one"
    second = "checkpoint-secret-two"
    actual = {
        "api_endpoint": "zai",
        "messages_payload": [],
        "model": "glm-4.5",
        "provider_continuations": [
            _continuation_checkpoint(provider="zai", model="glm-4.5", result=first)
        ],
    }
    expected = {
        **actual,
        "provider_continuations": [
            _continuation_checkpoint(provider="zai", model="glm-4.5", result=second)
        ],
    }

    result = verify_provider_request_shadow(
        actual_kwargs=actual,
        expected_kwargs=expected,
        provenance=ProviderRequestProvenance(
            continuations=(_artifact(TraceProvenanceSource.CONTINUATION),)
        ),
        project_handler_kwargs=lambda values: values,
        known_credentials=(first, second),
    )

    assert result.available is False
    assert result.omission_reason is TraceOmissionReason.ALIGNMENT_MISMATCH
    assert result.boundary_kwargs == {}
    assert first not in repr(result)
    assert second not in repr(result)


def test_arbitrary_dataclass_is_not_normalized_at_final_value_boundary() -> None:
    @dataclass(frozen=True)
    class UnknownPrivateValue:
        value: str

    result = verify_provider_request_shadow(
        actual_kwargs={
            "api_endpoint": "moonshot",
            "messages_payload": [],
            "model": "kimi-k2",
            "unknown": UnknownPrivateValue("private"),
        },
        expected_kwargs={
            "api_endpoint": "moonshot",
            "messages_payload": [],
            "model": "kimi-k2",
            "unknown": UnknownPrivateValue("private"),
        },
        provenance=ProviderRequestProvenance(),
        project_handler_kwargs=lambda values: values,
    )

    assert result.available is False
    assert result.omission_reason is TraceOmissionReason.SANITIZER_FAILED


def test_checkpoint_normalization_does_not_construct_arbitrary_sequences() -> None:
    class UnknownSequence(list[object]):
        pass

    values = {
        "api_endpoint": "moonshot",
        "messages_payload": [],
        "model": "kimi-k2",
        "provider_continuations": UnknownSequence([_continuation_checkpoint()]),
    }

    result = verify_provider_request_shadow(
        actual_kwargs=values,
        expected_kwargs=values,
        provenance=ProviderRequestProvenance(
            continuations=(_artifact(TraceProvenanceSource.CONTINUATION),)
        ),
        project_handler_kwargs=lambda projected: projected,
    )

    assert result.available is False
    assert result.omission_reason is TraceOmissionReason.SANITIZER_FAILED


@pytest.mark.parametrize(
    ("endpoint", "extra", "expected_handler", "expected_overlay"),
    [
        (
            "openai",
            {},
            {"input_data": [{"role": "user", "content": "hello"}]},
            None,
        ),
        (
            "anthropic",
            {"system_message": "system", "prompt_caching": True},
            {"system_prompt": "system", "prompt_caching": True},
            "anthropic_cache_overlay",
        ),
        (
            "google",
            {},
            {
                "input_data": [
                    {
                        "role": "user",
                        "content": "hello",
                        EPHEMERAL_ORIGIN_KEY: "project_instructions",
                    }
                ]
            },
            None,
        ),
        (
            "qwencloud",
            {"api_mode": "responses"},
            {"api_mode": "responses"},
            "qwen_api_mode",
        ),
        (
            "zai",
            {
                "reasoning_effort": "high",
                "provider_continuations": [{"id": "opaque"}],
                "request_retries": 2,
            },
            {
                "reasoning_effort": "high",
                "provider_continuations": [{"id": "opaque"}],
                "request_retries": 2,
            },
            "provider_continuation",
        ),
        (
            "moonshot",
            {
                "provider_continuations": [{"id": "opaque"}],
                "request_retry_delay": 0.25,
            },
            {
                "provider_continuations": [{"id": "opaque"}],
                "request_retry_delay": 0.25,
            },
            "transport_retry_policy",
        ),
        (
            "custom-openai-api",
            {"api_key_resolved": True},
            {"api_key_resolved": True},
            "credential_decision",
        ),
    ],
)
def test_provider_shape_matrix_is_sanitized_and_projected(
    endpoint: str,
    extra: dict[str, object],
    expected_handler: dict[str, object],
    expected_overlay: str | None,
) -> None:
    message = {"role": "user", "content": "hello"}
    if endpoint == "google":
        message[EPHEMERAL_ORIGIN_KEY] = "project_instructions"
    actual = {
        "api_endpoint": endpoint,
        "messages_payload": [message],
        "model": "model",
        **extra,
    }

    provenance = _provenance()
    provenance = ProviderRequestProvenance(
        system_message=(provenance.messages[0] if "system_message" in actual else None),
        messages=provenance.messages,
        messages_payload=provenance.messages_payload,
        continuations=(
            (_artifact(TraceProvenanceSource.CONTINUATION),)
            if "provider_continuations" in actual
            else ()
        ),
    )
    result = verify_provider_request_shadow(
        actual_kwargs=actual,
        expected_kwargs=dict(actual),
        provenance=provenance,
        project_handler_kwargs=_chatbook_project,
    )

    assert result.available is True
    for name, value in expected_handler.items():
        assert result.handler_kwargs[name] == value
    if expected_overlay is not None:
        assert expected_overlay in {item.kind for item in result.overlays}
    if endpoint == "custom-openai-api":
        assert result.credential_source is ProviderCredentialSource.EXPLICIT_KEYLESS


def test_verified_bundle_values_are_recursively_immutable() -> None:
    actual = {
        "api_endpoint": "openai",
        "messages_payload": [{"role": "user", "content": "hello"}],
        "model": "gpt-4.1",
    }
    result = verify_provider_request_shadow(
        actual_kwargs=actual,
        expected_kwargs=dict(actual),
        provenance=_provenance(),
        project_handler_kwargs=_project,
        literal_payload={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert result.available is True
    with pytest.raises(TypeError):
        result.literal_payload["messages"] = []  # type: ignore[index]
    messages = next(
        item.value for item in result.components if item.name == "messages_payload"
    )
    with pytest.raises(TypeError):
        messages[0]["content"] = "changed"  # type: ignore[index]


def test_provider_mapping_and_sensitive_route_censuses_are_bidirectional() -> None:
    handler_keys = frozenset(API_CALL_HANDLERS)

    assert handler_keys == frozenset(PROVIDER_PARAM_MAP)
    assert handler_keys == SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS
    assert handler_keys == CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS
    assert {"llama_cpp", "local_llamacpp"} < handler_keys


def test_credential_category_or_descriptor_mismatch_is_content_free() -> None:
    actual = {
        "api_endpoint": "openai",
        "messages_payload": [{"role": "user", "content": "hello"}],
        "api_key": "resolved-secret",
        "model": "gpt-4.1",
    }
    missing_credential = dict(actual)
    missing_credential.pop("api_key")

    credential_mismatch = verify_provider_request_shadow(
        actual_kwargs=actual,
        expected_kwargs=missing_credential,
        provenance=_provenance(),
        project_handler_kwargs=_project,
        known_credentials=("resolved-secret",),
    )
    descriptor_mismatch = verify_provider_request_shadow(
        actual_kwargs=actual,
        expected_kwargs=dict(actual),
        provenance=ProviderRequestProvenance(),
        project_handler_kwargs=_project,
        known_credentials=("resolved-secret",),
    )

    assert credential_mismatch.available is False
    assert credential_mismatch.omission_reason is TraceOmissionReason.ALIGNMENT_MISMATCH
    assert descriptor_mismatch.available is False
    assert descriptor_mismatch.omission_reason is TraceOmissionReason.ALIGNMENT_MISMATCH
    assert credential_mismatch.as_content_free_record() == {
        "available": False,
        "omission_reason": "alignment_mismatch",
    }


def test_tool_schema_is_provider_artifact_bound_after_sanitization() -> None:
    provenance = _provenance()
    provenance = ProviderRequestProvenance(
        messages=provenance.messages,
        messages_payload=provenance.messages_payload,
        tools=(_artifact(TraceProvenanceSource.TOOL_DEFINITION),),
    )
    actual = {
        "api_endpoint": "openai",
        "messages_payload": [{"role": "user", "content": "hello"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"},
                    "api_key": "TOOL-SECRET",
                },
            }
        ],
        "model": "gpt-4.1",
    }

    result = verify_provider_request_shadow(
        actual_kwargs=actual,
        expected_kwargs=dict(actual),
        provenance=provenance,
        project_handler_kwargs=_chatbook_project,
        handler_source_names={"tools": "tools"},
    )

    assert result.available is True
    tool_binding = next(item for item in result.components if item.name == "tools")
    assert tool_binding.intents == (FinalValueIntent.PROVIDER_ARTIFACT,)
    assert tool_binding.redacted is True
    handler_tool_binding = next(
        item for item in result.handler_components if item.name == "tools"
    )
    assert handler_tool_binding.redacted is True
    assert result.redacted is True
    assert "TOOL-SECRET" not in json.dumps(result.boundary_kwargs)
