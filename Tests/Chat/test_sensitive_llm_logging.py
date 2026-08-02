"""Request-scoped logging policy tests for sensitive auxiliary completions."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import httpx
import pytest
import requests
from loguru import logger

import tldw_chatbook.Chat.Chat_Functions as chat_functions
import tldw_chatbook.LLM_Calls.LLM_API_Calls as cloud_adapters
import tldw_chatbook.LLM_Calls.LLM_API_Calls_Local as local_adapters
from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
from tldw_chatbook.Chat.Chat_Functions import SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS
from tldw_chatbook.Chat.console_provider_gateway import (
    AuxiliaryCompletionRequest,
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.config import RuntimeConfigSnapshot
from tldw_chatbook.Utils.sensitive_llm_logging import (
    is_sensitive_llm_request,
    llm_content_byte_count,
    llm_retry_count,
    safe_llm_error_detail,
    safe_llm_log_value,
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
)


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
    monkeypatch.setattr(cloud_adapters.requests, "Session", lambda: session)

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
    monkeypatch.setattr(local_adapters.requests, "Session", lambda: session)

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
    monkeypatch.setattr(cloud_adapters.requests, "Session", lambda: session)

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
    monkeypatch.setattr(cloud_adapters.requests, "Session", lambda: session)

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
    monkeypatch.setattr(cloud_adapters.requests, "Session", lambda: session)

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
    monkeypatch.setattr(cloud_adapters.requests, "Session", lambda: session)

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
    monkeypatch.setattr(cloud_adapters.requests, "Session", lambda: session)
    if provider == "moonshot":
        monkeypatch.setattr(
            cloud_adapters,
            "load_settings",
            lambda: {
                "moonshot_api": {
                    "api_key": "key",
                    "api_base_url": "https://moonshot.test",
                }
            },
        )
        call: Callable[..., object] = cloud_adapters.chat_with_moonshot
    else:
        monkeypatch.setattr(
            cloud_adapters,
            "get_runtime_config_snapshot",
            lambda: _runtime_config(
                "zai", {"api_key": "key", "api_base_url": "https://zai.test"}
            ),
        )
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
    monkeypatch.setattr(local_adapters.requests, "Session", lambda: session)

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
async def test_sensitive_direct_llama_logs_no_request_response_or_endpoint_query() -> (
    None
):
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "RESPONSE-CANARY"}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway = ConsoleProviderGateway(http_client=client)
    request = AuxiliaryCompletionRequest(
        resolution=ConsoleProviderResolution(
            provider="llama_cpp",
            base_url="http://127.0.0.1:9099/v1?token=ENDPOINT-QUERY-CANARY",
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
        result = await gateway.complete_auxiliary(request)

    assert result.text == "RESPONSE-CANARY"
    _assert_canaries_absent(logs)
    await client.aclose()
