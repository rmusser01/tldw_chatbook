"""Request-scoped logging policy tests for sensitive auxiliary completions."""

from __future__ import annotations

import ast
import asyncio
import inspect
import logging
import os
import subprocess
import sys
import textwrap
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path

import httpx
import pytest
import requests
from loguru import logger

import tldw_chatbook.Chat.Chat_Functions as chat_functions
import tldw_chatbook.LLM_Calls.LLM_API_Calls as cloud_adapters
import tldw_chatbook.LLM_Calls.LLM_API_Calls_Local as local_adapters
import tldw_chatbook.LLM_Calls.hosted_chat as hosted_chat
from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
from tldw_chatbook.Chat.Chat_Functions import SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_provider_gateway import (
    AuxiliaryCompletionRequest,
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationValidationError,
    parse_provider_continuation_json,
    read_provider_continuation_json,
)
from tldw_chatbook.config import RuntimeConfigSnapshot
from tldw_chatbook.Utils.sensitive_llm_logging import (
    is_sensitive_llm_request,
    llm_content_byte_count,
    llm_retry_count,
    safe_llm_error_detail,
    safe_llm_log_value,
    safe_llm_request_payload_summary,
    safe_llm_url_host,
    sensitive_llm_request,
)


CANARIES = (
    "OPTIMIZER-CANARY",
    "SYSTEM-CANARY",
    "USER-CANARY",
    "BLOCK-CANARY",
    "OPAQUE-CANARY",
    "RESPONSE-CANARY",
    "ERROR-BODY-CANARY",
    "EXCEPTION-CANARY",
    "ENDPOINT-QUERY-CANARY",
    "ENDPOINT-PATH-CANARY",
    "ENDPOINT-USER-CANARY",
    "ENDPOINT-PASSWORD-CANARY",
    # task-2117 Qodo round: prompt-bearing fields the old messages/contents
    # denylist never accounted for.
    "OPENAI-RESPONSES-INPUT-CANARY",
    "ANTHROPIC-SYSTEM-FIELD-CANARY",
    "GOOGLE-SYSTEM-INSTRUCTION-CANARY",
    "UNKNOWN-PAYLOAD-FIELD-CANARY",
    "TOOL-SCHEMA-DESCRIPTION-CANARY",
    "TOOL-SCHEMA-ENUM-CANARY",
    "HUGGINGFACE-USER-CANARY",
    "CONTINUATION-CREDENTIAL-CANARY",
    "CONTINUATION-RAW-BODY-CANARY",
)


def test_safe_llm_url_host_public_docstring_is_google_style() -> None:
    docstring = inspect.getdoc(safe_llm_url_host) or ""

    assert "Args:" in docstring
    assert "Returns:" in docstring


class _ListHandler(logging.Handler):
    def __init__(self, messages: list[str]) -> None:
        super().__init__(level=logging.DEBUG)
        self._messages = messages

    def emit(self, record: logging.LogRecord) -> None:
        self._messages.append(record.getMessage())


@contextmanager
def _captured_logs() -> Iterator[list[str]]:
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    root = logging.getLogger()
    handler = _ListHandler(messages)
    old_level = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    try:
        yield messages
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
        logger.remove(sink_id)


def _assert_canaries_absent(*values: object) -> None:
    rendered = "\n".join(str(value) for value in values)
    for canary in CANARIES:
        assert canary not in rendered


def test_sensitive_continuation_validation_never_logs_or_chains_private_data() -> None:
    private_value = {
        "schema_version": 1,
        "checkpoint_revision": 1,
        "provider": "deepseek",
        "protocol": "responses",
        "model": "deepseek-test",
        "api_base_url": "https://api.deepseek.example.test/v1",
        "state": "active",
        "rounds": [],
        "credential": "CONTINUATION-CREDENTIAL-CANARY",
        "raw_provider_body": "CONTINUATION-RAW-BODY-CANARY",
    }

    with _captured_logs() as logs, pytest.raises(ContinuationValidationError) as caught:
        parse_provider_continuation_json(private_value)

    tolerant = read_provider_continuation_json(private_value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert tolerant.checkpoint is None
    assert tolerant.warning == "Exact tool continuation was discarded."
    _assert_canaries_absent(caught.value, repr(caught.value), logs, tolerant)


def _transport_logger_state() -> dict[str, dict[str, object]]:
    state: dict[str, dict[str, object]] = {}
    for name in ("httpx", "httpcore", "urllib3"):
        target = logging.getLogger(name)
        state[name] = {
            "filters": (
                len(target.filters),
                tuple(id(item) for item in target.filters),
            ),
            "handlers": (
                len(target.handlers),
                tuple(id(item) for item in target.handlers),
            ),
            "level": target.level,
            "propagate": target.propagate,
        }
    return state


def test_sensitive_context_never_mutates_shared_transport_logger_configuration() -> (
    None
):
    before = _transport_logger_state()

    with sensitive_llm_request():
        during = _transport_logger_state()

    after = _transport_logger_state()
    assert during == before
    assert after == before


class _FakeResponse:
    def __init__(
        self,
        data: object,
        *,
        status_code: int = 200,
        text: str = "",
    ) -> None:
        self._data = data
        self.status_code = status_code
        self.text = text
        self.connection = None

    def json(self) -> object:
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def iter_lines(self, decode_unicode: bool = False):
        del decode_unicode
        return iter(())

    def close(self) -> None:
        return None


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.posts: list[dict[str, object]] = []

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def mount(self, *_args: object, **_kwargs: object) -> None:
        return None

    def post(self, url: str, **kwargs: object) -> _FakeResponse:
        self.posts.append({"url": url, **kwargs})
        return self.response

    def close(self) -> None:
        return None


def _runtime_config(provider: str, values: dict[str, object]) -> RuntimeConfigSnapshot:
    return RuntimeConfigSnapshot(
        generation=0,
        values={"api_settings": {provider: values}},
    )


def _sensitive_messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "USER-CANARY BLOCK-CANARY OPAQUE-CANARY"}]


def test_sensitive_context_is_nested_and_resets_after_exception() -> None:
    assert is_sensitive_llm_request() is False
    with pytest.raises(RuntimeError):
        with sensitive_llm_request():
            assert is_sensitive_llm_request() is True
            with sensitive_llm_request():
                assert is_sensitive_llm_request() is True
            assert is_sensitive_llm_request() is True
            raise RuntimeError("EXCEPTION-CANARY")
    assert is_sensitive_llm_request() is False


def test_sensitive_helpers_redact_before_preview_serialization_or_url_rendering() -> (
    None
):
    body = {"content": "USER-CANARY", "nested": ["RESPONSE-CANARY"]}
    with sensitive_llm_request():
        assert safe_llm_log_value(body) == "<sensitive-content-redacted>"
        assert safe_llm_error_detail("ERROR-BODY-CANARY") == (
            "<sensitive-error-detail-redacted>"
        )
        assert (
            safe_llm_url_host(
                "https://user:pass@example.test/path?token=ENDPOINT-QUERY-CANARY"
            )
            == "example.test"
        )
        assert llm_content_byte_count(body) > 0


def test_ordinary_helpers_preserve_existing_diagnostics() -> None:
    assert safe_llm_log_value("ordinary-diagnostic") == "ordinary-diagnostic"
    assert safe_llm_error_detail("ordinary-error") == "ordinary-error"
    assert llm_retry_count(3) == 3


def test_sensitive_auxiliary_request_disables_provider_http_retries() -> None:
    with sensitive_llm_request():
        assert llm_retry_count(3) == 0


@pytest.mark.asyncio
async def test_sensitive_worker_and_simultaneous_ordinary_thread_do_not_bleed() -> None:
    sensitive_started = threading.Event()
    release_sensitive = threading.Event()
    observations: list[tuple[str, bool]] = []

    def adapter(**_kwargs: object) -> str:
        observations.append(("sensitive", is_sensitive_llm_request()))
        sensitive_started.set()
        release_sensitive.wait(timeout=2)
        return "ok"

    gateway = ConsoleProviderGateway(chat_api_call_fn=adapter)
    request = AuxiliaryCompletionRequest(
        resolution=ConsoleProviderResolution(
            provider="openai",
            base_url="",
            model="gpt-test",
            ready=True,
            readiness_key="openai",
            execution_key="openai",
        ),
        messages=({"role": "user", "content": "USER-CANARY"},),
        response_format=None,
        max_output_tokens=8,
    )
    sensitive_task = asyncio.create_task(gateway.complete_auxiliary(request))
    await asyncio.to_thread(sensitive_started.wait, 1)

    def ordinary() -> None:
        observations.append(("ordinary", is_sensitive_llm_request()))
        logging.debug("ordinary diagnostic remains visible")

    with _captured_logs() as logs:
        await asyncio.to_thread(ordinary)
    release_sensitive.set()
    await sensitive_task

    assert observations == [("sensitive", True), ("ordinary", False)]
    assert "ordinary diagnostic remains visible" in "\n".join(logs)
    assert is_sensitive_llm_request() is False


def test_sensitive_generic_dispatch_redacts_raw_response_and_exception_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeResponse(
        {}, status_code=500, text="ERROR-BODY-CANARY EXCEPTION-CANARY"
    )

    def failing_handler(**_kwargs: object) -> None:
        raise requests.exceptions.HTTPError(response=response)

    failing_handler.__name__ = "failing_handler"
    monkeypatch.setitem(chat_functions.API_CALL_HANDLERS, "openai", failing_handler)

    with _captured_logs() as logs, sensitive_llm_request():
        with pytest.raises(ChatProviderError) as exc_info:
            chat_functions.chat_api_call(
                api_endpoint="openai",
                messages_payload=_sensitive_messages(),
                model="gpt-test",
                streaming=False,
            )

    _assert_canaries_absent(logs, exc_info.value)
    assert "status=500" in "\n".join(logs)


def test_sensitive_audit_registry_covers_every_registered_chat_handler() -> None:
    assert SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS == frozenset(
        chat_functions.API_CALL_HANDLERS
    )


def test_registered_chat_handlers_accept_pinned_endpoint_override() -> None:
    missing = {
        endpoint
        for endpoint, handler in chat_functions.API_CALL_HANDLERS.items()
        if "api_base_url" not in inspect.signature(handler).parameters
    }

    assert missing == set()


@pytest.mark.parametrize("endpoint", sorted(chat_functions.API_CALL_HANDLERS))
def test_sensitive_dispatcher_redacts_http_bodies_for_every_registered_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
) -> None:
    response = _FakeResponse({}, status_code=500, text="ERROR-BODY-CANARY")

    def failing_handler(**_kwargs: object) -> None:
        raise requests.exceptions.HTTPError(response=response)

    failing_handler.__name__ = "failing_handler"
    monkeypatch.setitem(chat_functions.API_CALL_HANDLERS, endpoint, failing_handler)

    with _captured_logs() as logs, sensitive_llm_request():
        with pytest.raises(Exception) as exc_info:
            chat_functions.chat_api_call(
                api_endpoint=endpoint,
                messages_payload=_sensitive_messages(),
                model="model-test",
                streaming=False,
            )

    _assert_canaries_absent(logs, exc_info.value)


def test_sensitive_anthropic_error_body_and_exception_are_not_exposed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(_FakeResponse({}, status_code=500, text="ERROR-BODY-CANARY"))
    monkeypatch.setattr(
        cloud_adapters,
        "load_settings",
        lambda: {
            "anthropic_api": {
                "api_key": "key",
                "api_base_url": "https://anthropic.test/v1",
                "api_retries": 0,
            }
        },
    )
    monkeypatch.setattr(cloud_adapters, "create_default_session", lambda: session)

    with _captured_logs() as logs, sensitive_llm_request():
        with pytest.raises(Exception) as exc_info:
            cloud_adapters.chat_with_anthropic(
                input_data=_sensitive_messages(),
                api_key="key",
                system_prompt="SYSTEM-CANARY OPTIMIZER-CANARY",
                model="claude-test",
                streaming=False,
            )

    _assert_canaries_absent(logs, exc_info.value)


def test_sensitive_shared_local_transport_redacts_endpoint_body_and_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(_FakeResponse({}, status_code=500, text="ERROR-BODY-CANARY"))
    monkeypatch.setattr(local_adapters, "create_default_session", lambda: session)

    with _captured_logs() as logs, sensitive_llm_request():
        with pytest.raises(Exception) as exc_info:
            local_adapters._chat_with_openai_compatible_local_server(
                api_base_url=("https://local.test/v1?token=ENDPOINT-QUERY-CANARY"),
                model_name="local-test",
                input_data=_sensitive_messages(),
                api_key="key",
                system_message="SYSTEM-CANARY OPTIMIZER-CANARY",
                streaming=False,
                api_retries=0,
            )

    _assert_canaries_absent(logs, exc_info.value)


def test_sensitive_openai_http_error_log_and_exception_are_metadata_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(_FakeResponse({}, status_code=500, text="ERROR-BODY-CANARY"))
    monkeypatch.setattr(
        cloud_adapters,
        "load_settings",
        lambda: {
            "openai_api": {
                "api_base_url": "https://api.openai.test/v1",
                "api_retries": 0,
            }
        },
    )
    monkeypatch.setattr(cloud_adapters, "create_default_session", lambda: session)

    with _captured_logs() as logs, sensitive_llm_request():
        with pytest.raises(requests.exceptions.HTTPError) as exc_info:
            cloud_adapters.chat_with_openai(
                input_data=_sensitive_messages(),
                api_key="key",
                system_message="SYSTEM-CANARY OPTIMIZER-CANARY",
                model="gpt-test",
                streaming=False,
            )

    _assert_canaries_absent(logs)
    assert "ERROR-BODY-CANARY" in str(exc_info.value.response.text)


def test_sensitive_cohere_request_and_response_logs_are_metadata_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(
        _FakeResponse(
            {
                "id": "id",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "RESPONSE-CANARY"}],
                },
                "finish_reason": "COMPLETE",
            }
        )
    )
    monkeypatch.setattr(
        cloud_adapters,
        "get_runtime_config_snapshot",
        lambda: _runtime_config(
            "cohere",
            {
                "api_key": "key",
                "api_base_url": "https://api.cohere.test",
                "api_retries": 0,
            },
        ),
    )
    monkeypatch.setattr(cloud_adapters, "create_default_session", lambda: session)

    with _captured_logs() as logs, sensitive_llm_request():
        result = cloud_adapters.chat_with_cohere(
            input_data=_sensitive_messages(),
            api_key="key",
            system_prompt="SYSTEM-CANARY OPTIMIZER-CANARY",
            model="command-test",
            streaming=False,
        )

    assert result["choices"][0]["message"]["content"] == "RESPONSE-CANARY"
    _assert_canaries_absent(logs)
    rendered = "\n".join(logs)
    assert "model" in rendered.lower()
    assert "message" in rendered.lower()


def test_sensitive_google_request_content_and_error_body_are_not_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(_FakeResponse({}, status_code=500, text="ERROR-BODY-CANARY"))
    monkeypatch.setattr(
        cloud_adapters,
        "get_runtime_config_snapshot",
        lambda: _runtime_config("google", {"api_key": "key", "api_retries": 0}),
    )
    monkeypatch.setattr(cloud_adapters, "create_default_session", lambda: session)

    with _captured_logs() as logs, sensitive_llm_request():
        with pytest.raises(Exception) as exc_info:
            cloud_adapters.chat_with_google(
                input_data=_sensitive_messages(),
                api_key="key",
                system_message="SYSTEM-CANARY OPTIMIZER-CANARY",
                model="gemini-test",
                streaming=False,
            )

    _assert_canaries_absent(logs, exc_info.value)


# ---- task-2117 Qodo round: allowlist, not denylist -------------------------
#
# TASK-2116 made the "Request Payload" debug logs above actually
# interpolate. The redaction added alongside them only stripped
# `messages`/`contents` -- a denylist that has now failed twice: it never
# accounted for OTHER prompt-bearing fields providers carry outside those
# two keys (OpenAI Responses API `input`, Anthropic `system`, Google
# `system_instruction`). These tests pin the allowlist that replaced it:
# only known-safe scalar metadata is logged, everything else -- including a
# payload key nobody has seen yet -- is dropped by default.


@pytest.mark.parametrize("sensitive", [False, True])
def test_openai_responses_api_input_field_is_never_logged(
    monkeypatch: pytest.MonkeyPatch,
    sensitive: bool,
) -> None:
    """Confirm the Responses API's input field never reaches debug logs.

    The Responses API (used whenever ``reasoning_effort`` is set) carries
    the whole conversation -- system message included -- under ``input``,
    not ``messages``. The old messages-only denylist never accounted for
    this field.

    Args:
        monkeypatch: Pytest fixture used to stub config loading and the
            outgoing HTTP session.
        sensitive: Whether to also exercise the sensitive-auxiliary logging
            path alongside the ordinary path.
    """
    # The response body deliberately avoids any CANARIES entry: this test's
    # concern is the request-payload allowlist, not error-detail redaction
    # (already covered elsewhere) -- an ordinary, non-sensitive request is
    # expected to surface a real HTTP error detail.
    session = _FakeSession(
        _FakeResponse({}, status_code=500, text="non-sensitive-path-http-failure")
    )
    monkeypatch.setattr(
        cloud_adapters,
        "load_settings",
        lambda: {
            "openai_api": {
                "api_base_url": "https://api.openai.test/v1",
                "api_retries": 0,
            }
        },
    )
    monkeypatch.setattr(cloud_adapters, "create_default_session", lambda: session)

    def _invoke() -> None:
        with pytest.raises(Exception):
            cloud_adapters.chat_with_openai(
                input_data=[
                    {"role": "user", "content": "OPENAI-RESPONSES-INPUT-CANARY"}
                ],
                api_key="key",
                system_message="SYSTEM-CANARY OPTIMIZER-CANARY",
                model="gpt-test",
                streaming=False,
                reasoning_effort="medium",
            )

    context = sensitive_llm_request() if sensitive else nullcontext()
    with _captured_logs() as logs, context:
        _invoke()

    _assert_canaries_absent(logs)


@pytest.mark.parametrize("sensitive", [False, True])
def test_anthropic_system_field_is_never_logged(
    monkeypatch: pytest.MonkeyPatch,
    sensitive: bool,
) -> None:
    """Confirm Anthropic's top-level system field never reaches debug logs.

    Args:
        monkeypatch: Pytest fixture used to stub config loading and the
            outgoing HTTP session.
        sensitive: Whether to also exercise the sensitive-auxiliary logging
            path alongside the ordinary path.
    """
    # See the OpenAI test above for why the response body avoids CANARIES.
    session = _FakeSession(
        _FakeResponse({}, status_code=500, text="non-sensitive-path-http-failure")
    )
    monkeypatch.setattr(
        cloud_adapters,
        "load_settings",
        lambda: {
            "anthropic_api": {
                "api_key": "key",
                "api_base_url": "https://anthropic.test/v1",
                "api_retries": 0,
            }
        },
    )
    monkeypatch.setattr(cloud_adapters, "create_default_session", lambda: session)

    def _invoke() -> None:
        with pytest.raises(Exception):
            cloud_adapters.chat_with_anthropic(
                input_data=[{"role": "user", "content": "USER-CANARY"}],
                api_key="key",
                system_prompt="ANTHROPIC-SYSTEM-FIELD-CANARY",
                model="claude-test",
                streaming=False,
            )

    context = sensitive_llm_request() if sensitive else nullcontext()
    with _captured_logs() as logs, context:
        _invoke()

    _assert_canaries_absent(logs)


@pytest.mark.parametrize("sensitive", [False, True])
def test_google_system_instruction_field_is_never_logged(
    monkeypatch: pytest.MonkeyPatch,
    sensitive: bool,
) -> None:
    """Confirm Google's system_instruction field never reaches debug logs.

    Args:
        monkeypatch: Pytest fixture used to stub config loading and the
            outgoing HTTP session.
        sensitive: Whether to also exercise the sensitive-auxiliary logging
            path alongside the ordinary path.
    """
    # See the OpenAI test above for why the response body avoids CANARIES.
    session = _FakeSession(
        _FakeResponse({}, status_code=500, text="non-sensitive-path-http-failure")
    )
    monkeypatch.setattr(
        cloud_adapters,
        "get_runtime_config_snapshot",
        lambda: _runtime_config("google", {"api_key": "key", "api_retries": 0}),
    )
    monkeypatch.setattr(cloud_adapters, "create_default_session", lambda: session)

    def _invoke() -> None:
        with pytest.raises(Exception):
            cloud_adapters.chat_with_google(
                input_data=[{"role": "user", "content": "USER-CANARY"}],
                api_key="key",
                system_message="GOOGLE-SYSTEM-INSTRUCTION-CANARY",
                model="gemini-test",
                streaming=False,
            )

    context = sensitive_llm_request() if sensitive else nullcontext()
    with _captured_logs() as logs, context:
        _invoke()

    _assert_canaries_absent(logs)


def test_safe_llm_request_payload_summary_drops_unrecognized_payload_keys() -> None:
    """Confirm the allowlist drops any payload key it does not explicitly recognize.

    This is the property that stops a third recurrence of this bug class: a
    provider payload growing a brand-new field must be safe by default, not
    exposed by default, even before anyone updates the allowlist.
    """
    payload = {
        "model": "gpt-test",
        "stream": False,
        "messages": [{"role": "user", "content": "hi"}],
        "a_field_no_provider_has_shipped_yet": "UNKNOWN-PAYLOAD-FIELD-CANARY",
    }

    summary = safe_llm_request_payload_summary(payload)

    assert "a_field_no_provider_has_shipped_yet" not in summary
    assert "UNKNOWN-PAYLOAD-FIELD-CANARY" not in str(summary)
    assert summary["model"] == "gpt-test"
    assert summary["message_count"] == 1


def test_tool_definitions_log_names_only_never_schema_or_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirm tool debug logs carry only names, never descriptions or JSON-schema parameters.

    Args:
        monkeypatch: Pytest fixture used to stub config loading and the
            outgoing HTTP session.
    """
    # See test_openai_responses_api_input_field_is_never_logged for why the
    # response body avoids CANARIES.
    session = _FakeSession(
        _FakeResponse({}, status_code=500, text="non-sensitive-path-http-failure")
    )
    monkeypatch.setattr(
        cloud_adapters,
        "load_settings",
        lambda: {
            "openai_api": {
                "api_base_url": "https://api.openai.test/v1",
                "api_retries": 0,
            }
        },
    )
    monkeypatch.setattr(cloud_adapters, "create_default_session", lambda: session)

    with _captured_logs() as logs:
        with pytest.raises(Exception):
            cloud_adapters.chat_with_openai(
                input_data=[{"role": "user", "content": "hello"}],
                api_key="key",
                model="gpt-test",
                streaming=False,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_weather",
                            "description": "TOOL-SCHEMA-DESCRIPTION-CANARY",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )

    rendered = "\n".join(logs)
    _assert_canaries_absent(logs)
    assert "lookup_weather" in rendered


@pytest.mark.parametrize("sensitive", [False, True])
def test_huggingface_tool_logs_are_names_only(
    monkeypatch: pytest.MonkeyPatch,
    sensitive: bool,
) -> None:
    """Confirm HuggingFace logs only tool names outside sensitive requests.

    Args:
        monkeypatch: Pytest fixture used to stub config loading and the
            outgoing HTTP session.
        sensitive: Whether to exercise the sensitive-request logging policy.
    """
    response_data = {"id": "hf-test", "choices": [{"message": {"content": "ok"}}]}
    session = _FakeSession(_FakeResponse(response_data))
    monkeypatch.setattr(
        cloud_adapters,
        "load_settings",
        lambda: {
            "huggingface_api": {
                "api_base_url": "https://hf.test/v1",
                "api_chat_path": "chat/completions",
                "api_retries": 0,
            }
        },
    )
    monkeypatch.setattr(cloud_adapters, "create_default_session", lambda: session)

    context = sensitive_llm_request() if sensitive else nullcontext()
    with _captured_logs() as logs, context:
        cloud_adapters.chat_with_huggingface(
            input_data=[{"role": "user", "content": "hello"}],
            api_key="key",
            model="org/model",
            streaming=False,
            user="HUGGINGFACE-USER-CANARY",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_hf_weather",
                        "description": "TOOL-SCHEMA-DESCRIPTION-CANARY",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "unit": {
                                    "type": "string",
                                    "enum": ["TOOL-SCHEMA-ENUM-CANARY"],
                                }
                            },
                        },
                    },
                }
            ],
        )

    _assert_canaries_absent(logs)
    final_payload_label = "HuggingFace Final Payload (safe fields only):"
    final_payload_logs = [entry for entry in logs if final_payload_label in entry]
    tool_logs = [entry for entry in logs if "HuggingFace Tools:" in entry]
    if sensitive:
        assert final_payload_logs == []
        assert tool_logs == []
    else:
        assert len(final_payload_logs) == 1
        final_payload_summary = ast.literal_eval(
            final_payload_logs[0].split(final_payload_label, 1)[1].strip()
        )
        assert final_payload_summary["model"] == "org/model"
        assert final_payload_summary["message_count"] == 1
        assert len(tool_logs) == 1
        tools_summary = tool_logs[0].split("HuggingFace Tools: ", 1)[1].strip()
        assert tools_summary == "{'tool_names': ['lookup_hf_weather']}"


def test_sensitive_huggingface_error_body_endpoint_and_exception_are_not_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(_FakeResponse({}, status_code=500, text="ERROR-BODY-CANARY"))
    endpoint = "https://hf.test/v1?token=ENDPOINT-QUERY-CANARY"
    monkeypatch.setattr(
        cloud_adapters,
        "load_settings",
        lambda: {
            "huggingface_api": {
                "api_base_url": endpoint,
                "api_chat_path": "chat/completions",
                "api_retries": 0,
            }
        },
    )
    monkeypatch.setattr(cloud_adapters, "create_default_session", lambda: session)

    with _captured_logs() as logs, sensitive_llm_request():
        with pytest.raises(Exception) as exc_info:
            cloud_adapters.chat_with_huggingface(
                input_data=_sensitive_messages(),
                api_key="key",
                system_message="SYSTEM-CANARY OPTIMIZER-CANARY",
                model="org/model",
                streaming=False,
            )

    _assert_canaries_absent(logs, exc_info.value)


@pytest.mark.parametrize("provider", ["moonshot", "zai"])
def test_sensitive_openai_compatible_error_bodies_are_not_logged(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    session = _FakeSession(_FakeResponse({}, status_code=500, text="ERROR-BODY-CANARY"))
    monkeypatch.setattr(hosted_chat, "create_default_session", lambda: session)
    if provider == "moonshot":
        call: Callable[..., object] = cloud_adapters.chat_with_moonshot
    else:
        call = cloud_adapters.chat_with_zai

    with _captured_logs() as logs, sensitive_llm_request():
        with pytest.raises(Exception) as exc_info:
            call(
                input_data=_sensitive_messages(),
                api_key="key",
                system_message="SYSTEM-CANARY OPTIMIZER-CANARY",
                model="model-test",
                streaming=False,
            )

    _assert_canaries_absent(logs, exc_info.value)


def test_sensitive_native_kobold_prompt_response_and_errors_are_not_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(
        _FakeResponse({"unexpected": "RESPONSE-CANARY ERROR-BODY-CANARY"})
    )
    monkeypatch.setattr(
        local_adapters,
        "get_runtime_config_snapshot",
        lambda: _runtime_config(
            "koboldcpp",
            {
                "api_url": ("https://kobold.test/generate?token=ENDPOINT-QUERY-CANARY"),
                "model": "kobold-test",
                "api_retries": 0,
            },
        ),
    )
    monkeypatch.setattr(local_adapters, "create_default_session", lambda: session)

    with _captured_logs() as logs, sensitive_llm_request():
        with pytest.raises(ChatProviderError) as exc_info:
            local_adapters.chat_with_kobold(
                input_data=_sensitive_messages(),
                api_key="key",
                system_message="SYSTEM-CANARY OPTIMIZER-CANARY",
                model="kobold-test",
                streaming=False,
            )

    _assert_canaries_absent(logs, exc_info.value)


@pytest.mark.asyncio
async def test_sensitive_direct_llama_logs_no_request_response_or_endpoint_secrets() -> (
    None
):
    sensitive_started = asyncio.Event()
    release_sensitive = asyncio.Event()

    async def handler(_request: httpx.Request) -> httpx.Response:
        sensitive_started.set()
        await release_sensitive.wait()
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "RESPONSE-CANARY"}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway = ConsoleProviderGateway(http_client=client)
    request = AuxiliaryCompletionRequest(
        resolution=ConsoleProviderResolution(
            provider="llama_cpp",
            base_url=(
                "http://127.0.0.1:9099/ENDPOINT-PATH-CANARY/"
                "userinfo-ENDPOINT-USER-CANARY/credential-ENDPOINT-PASSWORD-CANARY"
                "?token=ENDPOINT-QUERY-CANARY"
            ),
            model="llama-test",
            ready=True,
            readiness_key="llama_cpp",
            execution_key="llama_cpp",
        ),
        messages=(
            {"role": "system", "content": "SYSTEM-CANARY OPTIMIZER-CANARY"},
            {"role": "user", "content": "USER-CANARY OPAQUE-CANARY"},
        ),
        response_format=None,
        max_output_tokens=8,
    )

    with _captured_logs() as logs:
        sensitive_task = asyncio.create_task(gateway.complete_auxiliary(request))
        try:
            await asyncio.wait_for(sensitive_started.wait(), timeout=1)
            logging.getLogger("httpx").info("ordinary downstream HTTP diagnostic")
        finally:
            release_sensitive.set()
        result = await sensitive_task

    assert result.text == "RESPONSE-CANARY"
    _assert_canaries_absent(logs)
    assert "ordinary downstream HTTP diagnostic" in "\n".join(logs)
    await client.aclose()


@pytest.mark.asyncio
async def test_sensitive_direct_llama_rejects_embedded_userinfo_without_logging_it() -> (
    None
):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200))
    )
    gateway = ConsoleProviderGateway(http_client=client)
    request = AuxiliaryCompletionRequest(
        resolution=ConsoleProviderResolution(
            provider="llama_cpp",
            base_url=(
                "http://ENDPOINT-USER-CANARY:ENDPOINT-PASSWORD-CANARY@"
                "127.0.0.1:9099/v1?token=ENDPOINT-QUERY-CANARY"
            ),
            model="llama-test",
            ready=True,
            readiness_key="llama_cpp",
            execution_key="llama_cpp",
        ),
        messages=({"role": "user", "content": "USER-CANARY"},),
        response_format=None,
        max_output_tokens=8,
    )

    with _captured_logs() as logs, pytest.raises(ChatProviderError):
        await gateway.complete_auxiliary(request)

    _assert_canaries_absent(logs)
    await client.aclose()


@pytest.mark.asyncio
async def test_auxiliary_pins_configured_endpoint_when_selection_url_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pinned_endpoint = "https://pinned.example.test/v1"
    changed_endpoint = "https://changed.example.test/v1"
    gateway_config = {
        "api_settings": {
            "openai": {
                "api_key": "key",
                "api_base_url": pinned_endpoint,
                "model": "gpt-test",
            }
        }
    }
    adapter_config = {
        "openai_api": {
            "api_base_url": pinned_endpoint,
            "api_retries": 3,
        }
    }
    session = _FakeSession(_FakeResponse({"choices": [{"message": {"content": "ok"}}]}))
    monkeypatch.setattr(cloud_adapters, "load_settings", lambda: adapter_config)
    monkeypatch.setattr(cloud_adapters, "create_default_session", lambda: session)
    gateway = ConsoleProviderGateway(config_provider=lambda: gateway_config, environ={})
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="openai",
            explicit_model="gpt-test",
        )
    )
    assert resolution.ready is True
    assert resolution.base_url == pinned_endpoint

    request = AuxiliaryCompletionRequest(
        resolution=resolution,
        messages=({"role": "user", "content": "USER-CANARY"},),
        response_format=None,
        max_output_tokens=8,
    )
    gateway_config["api_settings"]["openai"]["api_base_url"] = changed_endpoint
    adapter_config["openai_api"]["api_base_url"] = changed_endpoint

    result = await gateway.complete_auxiliary(request)

    assert result.text == "ok"
    assert session.posts[0]["url"] == f"{pinned_endpoint}/chat/completions"
    assert changed_endpoint not in str(session.posts)


@pytest.mark.asyncio
async def test_auxiliary_pins_default_openai_endpoint_before_config_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_endpoint = "https://api.openai.com/v1"
    changed_endpoint = "https://changed-openai.example.test/v1"
    gateway_config = {
        "api_settings": {"openai": {"api_key": "key", "model": "gpt-test"}}
    }
    adapter_config = {"openai_api": {"api_retries": 3}}
    session = _FakeSession(_FakeResponse({"choices": [{"message": {"content": "ok"}}]}))
    monkeypatch.setattr(cloud_adapters, "load_settings", lambda: adapter_config)
    monkeypatch.setattr(cloud_adapters, "create_default_session", lambda: session)
    gateway = ConsoleProviderGateway(config_provider=lambda: gateway_config, environ={})
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-test")
    )
    assert resolution.base_url == default_endpoint
    request = AuxiliaryCompletionRequest(
        resolution=resolution,
        messages=({"role": "user", "content": "USER-CANARY"},),
        response_format=None,
        max_output_tokens=8,
    )
    gateway_config["api_settings"]["openai"]["api_base_url"] = changed_endpoint
    adapter_config["openai_api"]["api_base_url"] = changed_endpoint

    result = await gateway.complete_auxiliary(request)

    assert result.text == "ok"
    assert session.posts[0]["url"] == f"{default_endpoint}/chat/completions"
    assert changed_endpoint not in str(session.posts)


@pytest.mark.asyncio
async def test_auxiliary_pins_default_anthropic_endpoint_before_config_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_endpoint = "https://api.anthropic.com/v1"
    changed_endpoint = "https://changed-anthropic.example.test/v1"
    gateway_config = {
        "api_settings": {"anthropic": {"api_key": "key", "model": "claude-test"}}
    }
    adapter_config = {"anthropic_api": {"api_retries": 3}}
    session = _FakeSession(
        _FakeResponse(
            {
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "model": "claude-test",
            }
        )
    )
    monkeypatch.setattr(cloud_adapters, "load_settings", lambda: adapter_config)
    monkeypatch.setattr(cloud_adapters, "create_default_session", lambda: session)
    gateway = ConsoleProviderGateway(config_provider=lambda: gateway_config, environ={})
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="anthropic", explicit_model="claude-test")
    )
    assert resolution.base_url == default_endpoint
    request = AuxiliaryCompletionRequest(
        resolution=resolution,
        messages=({"role": "user", "content": "USER-CANARY"},),
        response_format=None,
        max_output_tokens=8,
    )
    gateway_config["api_settings"]["anthropic"]["api_base_url"] = changed_endpoint
    adapter_config["anthropic_api"]["api_base_url"] = changed_endpoint

    result = await gateway.complete_auxiliary(request)

    assert result.text == "ok"
    assert session.posts[0]["url"] == f"{default_endpoint}/messages"
    assert changed_endpoint not in str(session.posts)


@pytest.mark.asyncio
async def test_auxiliary_huggingface_router_url_matches_ordinary_adapter_after_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router_base_url = "https://router.example.test/hf-inference"
    api_base_url = "https://api-base.example.test/v1"
    expected_url = f"{router_base_url}/models/org/model/v1/chat/completions"
    gateway_config = {
        "api_settings": {
            "huggingface": {
                "api_key": "key",
                "model": "org/model",
                "use_router_url_format": True,
                "router_base_url": router_base_url,
                "api_base_url": api_base_url,
            }
        }
    }
    adapter_config = {
        "huggingface_api": {
            "use_router_url_format": True,
            "router_base_url": router_base_url,
            "api_base_url": api_base_url,
            "api_retries": 3,
        }
    }
    session = _FakeSession(_FakeResponse({"choices": [{"message": {"content": "ok"}}]}))
    monkeypatch.setattr(cloud_adapters, "load_settings", lambda: adapter_config)
    monkeypatch.setattr(cloud_adapters, "create_default_session", lambda: session)

    ordinary = cloud_adapters.chat_with_huggingface(
        input_data=[{"role": "user", "content": "ordinary"}],
        model="org/model",
        api_key="key",
        streaming=False,
    )
    gateway = ConsoleProviderGateway(config_provider=lambda: gateway_config, environ={})
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="huggingface", explicit_model="org/model")
    )
    assert resolution.base_url == router_base_url
    request = AuxiliaryCompletionRequest(
        resolution=resolution,
        messages=({"role": "user", "content": "USER-CANARY"},),
        response_format=None,
        max_output_tokens=8,
    )
    gateway_config["api_settings"]["huggingface"]["router_base_url"] = (
        "https://changed-router.example.test/hf-inference"
    )
    gateway_config["api_settings"]["huggingface"]["api_base_url"] = (
        "https://changed-api.example.test/v1"
    )
    adapter_config["huggingface_api"]["router_base_url"] = (
        "https://changed-router.example.test/hf-inference"
    )
    adapter_config["huggingface_api"]["api_base_url"] = (
        "https://changed-api.example.test/v1"
    )

    auxiliary = await gateway.complete_auxiliary(request)

    assert ordinary["choices"][0]["message"]["content"] == "ok"
    assert auxiliary.text == "ok"
    assert [post["url"] for post in session.posts] == [expected_url, expected_url]


# ---- task-2119: sink-level diagnose=False must stop credential leaks ------
#
# `logger.opt(exception=True)` in the provider handlers attaches the live
# exception to the log record. Loguru's own `diagnose` option (default True)
# then dumps every stack frame's LOCAL VARIABLES alongside the traceback --
# for these handlers that includes `headers` (the raw
# `Authorization`/`x-api-key` value) and `final_api_key`. A real Moonshot key
# was disclosed this way via an ordinary HTTP 429 on 2026-08-03 (see
# .superpowers/sdd/multi-provider-usage-verification-2026-08-03.md). Payload
# redaction (the allowlist tests above) cannot touch this -- the secret
# arrives via frame locals, not a logged dict. The fix is sink-level
# (`Logging_Config.py`, `Metrics/logger_config.py`, and
# `tldw_chatbook/__init__.py`'s package-import default), not per call site;
# these tests pin that sink-level contract directly with a distinctive,
# obviously-fake sentinel standing in for a live credential.

SENTINEL_API_KEY = "sk-SENTINEL-DO-NOT-LOG-1234567890"

# task-2119: deliberately NOT a fully-mocked `requests.Session`. A fake
# `.post()` that raises immediately has no intermediate frames, and this
# app's own call site (`chat_with_openai`'s `response = session.post(`) is a
# multi-line call -- `headers=` lands on a continuation line, past the
# single source line loguru's diagnose renderer annotates for that specific
# frame, so a fully-mocked session does not actually reproduce a leak and
# would make the positive-control test below pass for the wrong reason
# (verified empirically while writing this test). The real 2026-08-03 leak
# happened a few frames deeper, inside `requests` itself
# (`Session.request`'s `**kwargs`, which contains `headers`) -- reachable
# only by letting a REAL `ConnectionError` propagate through `requests`'
# actual internals. Port 1 on loopback is always refused immediately (no
# real network access, no listener ever binds a privileged port as a normal
# user), so this is fast and deterministic while still exercising the real
# code path end-to-end.
_UNREACHABLE_LOCAL_URL = "http://127.0.0.1:1"


@contextmanager
def _loguru_sink_with_diagnose(diagnose: bool) -> Iterator[list[str]]:
    """Capture loguru's fully-formatted sink text under a pinned ``diagnose``.

    Mirrors what a real stream/file sink receives: ``str(message)`` includes
    loguru's own traceback rendering, with per-frame locals attached when
    ``diagnose=True``. Unlike ``_captured_logs()`` above, this pins
    ``diagnose`` explicitly instead of inheriting the ambient process
    default -- the ambient default is not a reliable proxy for production
    behavior in this suite: ``Tests/conftest.py`` imports loguru ahead of
    ``tldw_chatbook``, so this package's own ``LOGURU_DIAGNOSE``-setting
    import (see ``tldw_chatbook/__init__.py``) never gets to influence the
    bound default for the whole test session.
    """
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message)),
        level="DEBUG",
        diagnose=diagnose,
        backtrace=True,
    )
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def _plant_sentinel_openai_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cloud_adapters,
        "load_settings",
        lambda: {
            "openai_api": {
                "api_key": SENTINEL_API_KEY,
                "api_base_url": _UNREACHABLE_LOCAL_URL,
                "api_retries": 0,
                "api_timeout": 2,
            }
        },
    )


@pytest.mark.parametrize("sensitive", [False, True])
@pytest.mark.allow_network
def test_sentinel_api_key_never_reaches_diagnose_false_sink_on_request_exception(
    monkeypatch: pytest.MonkeyPatch,
    sensitive: bool,
) -> None:
    """task-2119 AC#1/#3: a real credential-shaped local must never leak.

    Forces the OpenAI adapter's ``RequestException`` handler (the branch
    that calls ``logger.opt(exception=True)`` on the non-sensitive path --
    LLM_API_Calls.py's ``chat_with_openai``, ``except
    requests.exceptions.RequestException`` block) with a distinctive
    sentinel standing in for the live API key, on BOTH the sensitive and
    non-sensitive request paths, and asserts the sentinel never reaches a
    sink configured the way this app's real sinks now are
    (``diagnose=False``).

    Args:
        monkeypatch: Pytest fixture used to plant the sentinel-bearing
            OpenAI config pointing at an unreachable local endpoint.
        sensitive: Whether to wrap the request in the
            ``sensitive_llm_request`` context (both paths must be safe).
    """
    _plant_sentinel_openai_config(monkeypatch)

    context = sensitive_llm_request() if sensitive else nullcontext()
    with _loguru_sink_with_diagnose(False) as logs, context:
        with pytest.raises(requests.exceptions.ConnectionError):
            cloud_adapters.chat_with_openai(
                input_data=[{"role": "user", "content": "hi"}],
                api_key=SENTINEL_API_KEY,
                model="gpt-test",
                streaming=False,
            )

    rendered = "\n".join(logs)
    assert SENTINEL_API_KEY not in rendered


@pytest.mark.allow_network
def test_sentinel_api_key_leaks_via_diagnose_true_sink_confirming_mechanism_is_real(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control for the regression test above.

    Without a ``diagnose=False`` sink, the SAME non-sensitive
    ``RequestException`` DOES attach ``headers``/``final_api_key``
    frame-local values to the sink text -- this is the exact mechanism that
    disclosed a real Moonshot key on 2026-08-03. If this test ever stops
    leaking, the regression test above has stopped proving anything (it
    would be passing vacuously, e.g. because the sentinel was never actually
    exercised).

    Args:
        monkeypatch: Pytest fixture used to plant the sentinel-bearing
            OpenAI config pointing at an unreachable local endpoint.
    """
    _plant_sentinel_openai_config(monkeypatch)

    with _loguru_sink_with_diagnose(True) as logs:
        with pytest.raises(requests.exceptions.ConnectionError):
            cloud_adapters.chat_with_openai(
                input_data=[{"role": "user", "content": "hi"}],
                api_key=SENTINEL_API_KEY,
                model="gpt-test",
                streaming=False,
            )

    rendered = "\n".join(logs)
    assert SENTINEL_API_KEY in rendered


def test_persistent_and_legacy_sink_configuration_pin_diagnose_false() -> None:
    """task-2119 AC#2/#4: pin the sink-level contract itself.

    A future contributor flipping ``diagnose=False`` back to unset (or to
    ``True``) on either of this app's real loguru sinks must fail a test
    rather than silently reopening the credential-leak hole.
    ``backtrace=True`` is pinned alongside it: traceback/diagnostic value
    (exception type, message, stack of source lines) must stay intact --
    only frame-local dumping goes away.

    Checked by parsing the call's actual keyword arguments, NOT by substring
    search over the source text. The first version of this test did the
    latter and was vacuous: the security rationale comment sitting directly
    above the kwarg contains the literal string ``diagnose=False``, so
    deleting the real argument still left the assertion satisfied by the
    comment. Confirmed by mutation -- removing the kwarg now fails here.
    """
    from tldw_chatbook import Logging_Config
    from tldw_chatbook.Metrics import logger_config as metrics_logger_config

    def _sink_kwargs(function: object, receiver: str) -> dict[str, ast.expr]:
        """Return the keyword arguments of the ``<receiver>.add(...)`` call."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == receiver
            ):
                return {
                    keyword.arg: keyword.value
                    for keyword in node.keywords
                    if keyword.arg is not None
                }
        raise AssertionError(f"no {receiver}.add(...) call found")

    for function, receiver in (
        (Logging_Config.configure_application_logging, "loguru_logger"),
        (metrics_logger_config.setup_logger, "logger"),
    ):
        kwargs = _sink_kwargs(function, receiver)
        for name, expected in (("diagnose", False), ("backtrace", True)):
            node = kwargs.get(name)
            assert node is not None, (
                f"{function.__qualname__}: sink is missing an explicit "
                f"{name}= argument; it must not inherit loguru's default"
            )
            assert isinstance(node, ast.Constant) and node.value is expected, (
                f"{function.__qualname__}: sink must pass {name}={expected}"
            )


# The child script deliberately mirrors the provider-handler shape from the
# 2026-08-03 live incident: an ``Authorization`` header dict built from the
# key sits in a frame whose FAILING LINE references it (``_post(headers)``),
# which is exactly what loguru's ``diagnose`` annotates with the variable's
# live value. The sentinel arrives via the environment, never as a literal
# in the script, so a plain source-line backtrace (``diagnose=False``,
# ``backtrace=True``) cannot reveal it -- only frame-local dumping can.
_INCIDENT_SHAPE_CHILD_SCRIPT = """\
import os

{package_import}
from loguru import logger

secret = os.environ["LEAK_SENTINEL"]


def _post(headers):
    raise ConnectionError("simulated transient provider error")


def _request(final_api_key):
    headers = {{"Authorization": "Bearer " + final_api_key}}
    return _post(headers)


try:
    _request(secret)
except ConnectionError:
    logger.opt(exception=True).error("Request failed")

print("LOGURU_DIAGNOSE=" + repr(os.environ.get("LOGURU_DIAGNOSE")))
print("CHILD-DONE")
"""


@pytest.mark.parametrize(
    ("import_package", "expect_leak"),
    [
        pytest.param(True, False, id="package-import-neutralizes-default-sink"),
        pytest.param(False, True, id="bare-loguru-default-leaks-positive-control"),
    ],
)
def test_fresh_process_incident_shape_leaks_only_without_package_import(
    tmp_path: Path,
    import_package: bool,
    expect_leak: bool,
) -> None:
    """task-2119: reproduce the exact shape of the 2026-08-03 live incident.

    A fresh process logs a caught exception whose frames hold a
    credential-shaped local, through whatever loguru sink is active after
    (a) importing only ``tldw_chatbook`` first, or (b) importing nothing --
    loguru's own auto-init default sink, the pre-fix world. The sentinel
    must stay out of the output in (a) and MUST appear in (b): the positive
    control proves this script shape genuinely leaks under loguru's
    defaults, so (a) cannot pass vacuously. Behavioral on purpose -- an
    earlier version introspected ``logger._core.handlers`` private
    internals, which loguru upgrades could rename without any real
    regression. Out-of-process on purpose: the live incident happened in a
    script that imported the provider adapters directly and never called
    ``Logging_Config.configure_application_logging``, and this test
    session's own ``Tests/conftest.py`` imports loguru before
    ``tldw_chatbook`` ever could. The child env is scrubbed of ``LOGURU_*``
    so an ambient ``LOGURU_DIAGNOSE`` export can neither mask the leak in
    (b) nor fail (a) for reasons unrelated to the code. The script runs
    from a real file, not ``-c``, because diagnose resolves the variables
    it dumps from traceback source lines, which ``<string>`` frames do not
    have.

    Args:
        tmp_path: Pytest fixture; the child script is written here so its
            traceback frames carry real source lines.
        import_package: Whether the child imports ``tldw_chatbook`` before
            logging (the fix under test) or exercises bare loguru.
        expect_leak: Whether the sentinel must appear in the child's
            stderr (the positive control) or must be absent (the fix).
    """
    script_path = tmp_path / "incident_shape_child.py"
    script_path.write_text(
        _INCIDENT_SHAPE_CHILD_SCRIPT.format(
            package_import="import tldw_chatbook" if import_package else "",
        ),
        encoding="utf-8",
    )

    project_root = Path(__file__).resolve().parents[2]
    child_env = {
        key: value for key, value in os.environ.items() if not key.startswith("LOGURU_")
    }
    child_env["LEAK_SENTINEL"] = SENTINEL_API_KEY
    child_env["PYTHONPATH"] = str(project_root) + (
        os.pathsep + child_env["PYTHONPATH"] if child_env.get("PYTHONPATH") else ""
    )

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(project_root),
        env=child_env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "CHILD-DONE" in result.stdout
    # The exception must actually have been rendered (backtrace stays on);
    # an empty stderr would make the no-leak assertion below vacuous.
    assert "ConnectionError" in result.stderr
    if expect_leak:
        assert SENTINEL_API_KEY in result.stderr
    else:
        assert SENTINEL_API_KEY not in result.stderr
        assert "LOGURU_DIAGNOSE='0'" in result.stdout


def test_package_import_preserves_host_configured_loguru_sinks(
    tmp_path: Path,
) -> None:
    """Package import must only replace loguru's auto-init default sink.

    A host application that configured loguru BEFORE importing
    ``tldw_chatbook`` (removed the default sink, installed its own) must
    keep its sinks working and must not gain a duplicate stderr sink from
    the package init. Pins the ``remove(0)``-not-``remove()`` narrowing:
    reverting to a bare ``remove()`` wipes the host's sink and fails here.

    Args:
        tmp_path: Pytest fixture; the child script is written here.
    """
    script_path = tmp_path / "host_configured_child.py"
    script_path.write_text(
        textwrap.dedent(
            """\
            from loguru import logger

            captured = []
            logger.remove()
            logger.add(captured.append, format="{message}")

            import tldw_chatbook

            logger.info("host-sink-message")
            assert any(
                "host-sink-message" in str(message) for message in captured
            ), "host-configured sink was clobbered by package import"
            print("CHILD-DONE")
            """
        ),
        encoding="utf-8",
    )

    project_root = Path(__file__).resolve().parents[2]
    child_env = {
        key: value for key, value in os.environ.items() if not key.startswith("LOGURU_")
    }
    child_env["PYTHONPATH"] = str(project_root) + (
        os.pathsep + child_env["PYTHONPATH"] if child_env.get("PYTHONPATH") else ""
    )

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(project_root),
        env=child_env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "CHILD-DONE" in result.stdout
    # No duplicate package stderr sink: the message reached ONLY the host sink.
    assert "host-sink-message" not in result.stderr
