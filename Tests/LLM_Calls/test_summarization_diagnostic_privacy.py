"""Stable, exhaustive guard for summarization diagnostic privacy review."""

from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import importlib
import inspect
import json
import logging as stdlib_logging
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator

import pytest
from loguru import logger as loguru_logger

from scripts import check_persistent_diagnostic_inventory as diagnostic_inventory
from tldw_chatbook.LLM_Calls import Local_Summarization_Lib as local_summarization
from tldw_chatbook.LLM_Calls import (
    Summarization_General_Lib as general_summarization,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "Tests/fixtures/summarization_diagnostic_review.json"
INVENTORY_PATH = REPO_ROOT / "Docs/security/production-diagnostic-inventory.json"
STARTING_PROJECTION_SHA256 = (
    "85a5c6b74f0cd4eb15f8ca0f8abfa5e18ca7f26f749d97fc7b781090cabd7733"
)
MODULE_COUNTS = {
    "tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py": 242,
    "tldw_chatbook/LLM_Calls/Summarization_General_Lib.py": 281,
}
PRIVATE_GROUP_COUNTS = {
    "local_core": 24,
    "local_adapters": 23,
    "local_vllm_ollama": 22,
    "local_custom": 31,
    "general_core": 36,
    "general_mid": 24,
    "general_streaming": 20,
    "general_tail": 20,
}
PRIVATE_CATEGORY_COUNTS = {
    "response/output content": 72,
    "exception/error detail": 58,
    "raw/processed/extracted input": 21,
    "credential fragment": 21,
    "prompt content": 17,
    "private endpoint/path": 11,
}
PRIVATE_CATEGORY_COUNTS_BY_MODULE = {
    "tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py": {
        "response/output content": 29,
        "exception/error detail": 36,
        "raw/processed/extracted input": 13,
        "credential fragment": 8,
        "prompt content": 8,
        "private endpoint/path": 6,
    },
    "tldw_chatbook/LLM_Calls/Summarization_General_Lib.py": {
        "response/output content": 43,
        "exception/error detail": 22,
        "raw/processed/extracted input": 8,
        "credential fragment": 13,
        "prompt content": 9,
        "private endpoint/path": 5,
    },
}


@dataclass(frozen=True)
class _CapturedDiagnostics:
    caplog: pytest.LogCaptureFixture
    loguru_messages: list[str]

    @property
    def text(self) -> str:
        return "\n".join([*self.caplog.messages, *self.loguru_messages])


@contextmanager
def _capture_stdlib_and_loguru(
    caplog: pytest.LogCaptureFixture,
) -> Iterator[_CapturedDiagnostics]:
    caplog.clear()
    caplog.set_level(stdlib_logging.DEBUG)
    loguru_messages: list[str] = []
    sink_id = loguru_logger.add(
        loguru_messages.append,
        level="DEBUG",
        format="{message}",
    )
    try:
        yield _CapturedDiagnostics(caplog, loguru_messages)
    finally:
        loguru_logger.remove(sink_id)


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        json_data: object | None = None,
        lines: tuple[bytes, ...] = (),
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self._lines = lines
        self.text = text
        self.iter_lines_started = False
        self.closed = False

    def json(self) -> object:
        if isinstance(self._json_data, BaseException):
            raise self._json_data
        return self._json_data

    def iter_lines(self) -> Iterator[bytes]:
        self.iter_lines_started = True
        yield from self._lines

    def raise_for_status(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response

    def mount(self, prefix: str, adapter: object) -> None:
        del prefix, adapter

    def post(self, *args: object, **kwargs: object) -> _FakeResponse:
        del args, kwargs
        return self.response

    # task-19830: `summarize_with_local_llm` now opens its session with
    # `with create_default_session() as session:` -- this fake stands in for
    # that factory's return value, so it needs the context-manager protocol
    # too. `requests.Session.__exit__` just calls `self.close()`; nothing
    # here needs cleanup, so both dunders are no-ops.
    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, *exc_info: object) -> None:
        del exc_info
        return None


def _local_settings(*, llama_endpoint: str = "http://llama.invalid") -> dict[str, Any]:
    # task-17382: this fixture used to key the llama section as "llama_api",
    # which is the name the SUMMARIZER invented -- the loader builds
    # "llama_cpp_api". These tests therefore passed by feeding the code its own
    # mistake, while every real llama.cpp summarization failed with
    # KeyError('llama_api') before reaching a server. Keyed to the loader now.
    # The "api_keys" and "local_api_ip" sections below are still names nothing
    # produces; the Kobold/TabbyAPI summarizers that read them fail the same
    # way in production, tracked as task-17383.
    return {
        "llama_cpp_api": {
            "api_key": "fixed-llama-key",
            "api_ip": llama_endpoint,
            "temperature": 0.7,
            "max_tokens": 64,
            "streaming": False,
            "api_retries": 0,
            "api_retry_delay": 0,
        },
        # Same for Kobold (task-17383): its own section is what exists.
        "kobold_api": {
            "api_key": "fixed-kobold-key",
            "api_ip": "http://kobold.invalid/generate",
            "api_streaming_ip": "http://kobold.invalid/chat",
            "api_retries": 0,
            "api_retry_delay": 0,
        },
    }


def _local_adapter_settings(
    *,
    oobabooga_endpoint: str = "http://oobabooga.invalid/v1/chat/completions",
    tabby_key: str | None = "fixed-tabby-key",
) -> dict[str, Any]:
    return {
        "ooba_api": {
            "api_key": "fixed-oobabooga-key",
            "api_ip": oobabooga_endpoint,
            "temperature": 0.7,
            "api_retries": 0,
            "api_retry_delay": 0,
        },
        # task-17383: `api_keys`, `local_api_ip` and `models` are names the
        # loader has never built -- keying the fixture to them is what let the
        # TabbyAPI summarizer pass here while failing on every real config.
        "tabby_api": {
            "api_key": tabby_key,
            "api_ip": "http://tabby.invalid/v1/chat/completions",
            "model": "fixed-tabby-model",
            "api_retries": 0,
            "api_retry_delay": 0,
        },
    }


def _vllm_settings(*, api_key: str = "fixed-vllm-key") -> dict[str, Any]:
    return {
        "vllm_api": {
            "api_key": api_key,
            "api_ip": "http://vllm.invalid/v1/chat/completions",
            "model": "fixed-vllm-model",
            "temperature": 0.2,
            "max_tokens": 64,
            "api_retries": 0,
            "api_retry_delay": 0,
        }
    }


def _ollama_settings() -> dict[str, Any]:
    return {
        "ollama_api": {
            "api_key": "fixed-ollama-key",
            "api_url": "http://ollama.invalid/api/chat",
            "model": "fixed-ollama-model",
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 64,
            "api_timeout": 5,
            "api_retries": 0,
            "api_retry_delay": 0,
        }
    }


def _custom_openai_settings(
    *,
    provider_1_key: str | None = "fixed-custom-openai-1-key",
    provider_1_url: str = "http://custom-openai-1.invalid/v1/chat/completions",
    provider_1_model: str = "fixed-custom-openai-1-model",
    provider_2_key: str | None = "fixed-custom-openai-2-key",
    provider_2_url: str = "http://custom-openai-2.invalid/v1/chat/completions",
    provider_2_model: str = "fixed-custom-openai-2-model",
) -> dict[str, Any]:
    return {
        "custom_openai_api": {
            "api_key": provider_1_key,
            "api_ip": provider_1_url,
            "model": provider_1_model,
            "temperature": 0.2,
            "max_tokens": 64,
            "streaming": False,
            "api_retries": 0,
            "api_retry_delay": 0,
        },
        "custom_openai_api_2": {
            "api_key": provider_2_key,
            "api_ip": provider_2_url,
            "model": provider_2_model,
            "temperature": 0.2,
            "max_tokens": 64,
            "streaming": False,
            "api_retries": 0,
            "api_retry_delay": 0,
        },
    }


def _install_signature_bound_settings(
    monkeypatch: pytest.MonkeyPatch,
    result: dict[str, Any] | BaseException,
) -> list[tuple[tuple[object, ...], dict[str, object]]]:
    real_load_settings = local_summarization.load_settings
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_load_settings(*args: object, **kwargs: object) -> dict[str, Any]:
        inspect.signature(real_load_settings).bind(*args, **kwargs)
        calls.append((args, kwargs))
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(local_summarization, "load_settings", fake_load_settings)
    return calls


def _install_signature_bound_session_post(
    monkeypatch: pytest.MonkeyPatch,
    result: _FakeResponse | BaseException,
) -> list[tuple[tuple[object, ...], dict[str, object]]]:
    real_post = local_summarization.requests.Session.post
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_post(session: object, *args: object, **kwargs: object) -> _FakeResponse:
        inspect.signature(real_post).bind(session, *args, **kwargs)
        calls.append((args, kwargs))
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(local_summarization.requests.Session, "post", fake_post)
    return calls


def _consume_generator(generator: Iterator[str]) -> tuple[list[str], object]:
    chunks: list[str] = []
    while True:
        try:
            chunks.append(next(generator))
        except StopIteration as stop:
            return chunks, stop.value


LOCAL_INPUT_CANARY = "LOCAL_INPUT_CANARY_3796"
LOCAL_PROMPT_CANARY = "LOCAL_PROMPT_CANARY_3796"
LOCAL_CREDENTIAL_CANARY = "K3YQZ"
LOCAL_PATH_CANARY = "http://LOCAL_PATH_CANARY_3796.invalid"
LOCAL_RESPONSE_CANARY = "LOCAL_RESPONSE_CANARY_3796"
LOCAL_EXCEPTION_CANARY = "LOCAL_EXCEPTION_CANARY_3796"
OOBABOOGA_PROMPT_CANARY = "OOBABOOGA_PROMPT_CANARY_3796"
OOBABOOGA_CREDENTIAL_CANARY = "O0BAZ"
OOBABOOGA_ENDPOINT_CANARY = (
    "http://OOBABOOGA_ENDPOINT_CANARY_3796.invalid/v1/chat/completions"
)
OOBABOOGA_RESPONSE_CANARY = "OOBABOOGA_RESPONSE_CANARY_3796"
OOBABOOGA_EXCEPTION_CANARY = "OOBABOOGA_EXCEPTION_CANARY_3796"
TABBY_INPUT_CANARY = "TABBY_INPUT_CANARY_3796"
TABBY_CREDENTIAL_CANARY = "T4BBY"
TABBY_STREAM_CANARY = "TABBY_STREAM_CANARY_3796"
TABBY_EXCEPTION_CANARY = "TABBY_EXCEPTION_CANARY_3796"
VLLM_INPUT_CANARY = "VLLM_INPUT_CANARY_3796"
VLLM_PROMPT_CANARY = "VLLM_PROMPT_CANARY_3796"
VLLM_CREDENTIAL_CANARY = "V11MK3YQZ"
VLLM_RESPONSE_CANARY = "VLLM_RESPONSE_CANARY_3796"
VLLM_STREAM_CANARY = "VLLM_STREAM_CANARY_3796"
VLLM_EXCEPTION_CANARY = "VLLM_EXCEPTION_CANARY_3796"
OLLAMA_PROMPT_CANARY = "OLLAMA_PROMPT_CANARY_3796"
OLLAMA_RESPONSE_CANARY = "OLLAMA_RESPONSE_CANARY_3796"
OLLAMA_STREAM_CANARY = "OLLAMA_STREAM_CANARY_3796"
OLLAMA_EXCEPTION_CANARY = "OLLAMA_EXCEPTION_CANARY_3796"
CUSTOM_OPENAI_INPUT_CANARY = "CUSTOM_OPENAI_INPUT_CANARY_3796"
CUSTOM_OPENAI_PROMPT_CANARY = "CUSTOM_OPENAI_PROMPT_CANARY_3796"
CUSTOM_OPENAI_1_CREDENTIAL_CANARY = "CUST0M_1_K3Y_MIDDLE_Z1X1W"
CUSTOM_OPENAI_2_CREDENTIAL_CANARY = "CUST0M_2_K3Y_MIDDLE_Z2X2W"
CUSTOM_OPENAI_1_ENDPOINT_CANARY = (
    "http://CUSTOM_OPENAI_1_ENDPOINT_CANARY_3796.invalid/v1/chat/completions"
)
CUSTOM_OPENAI_2_ENDPOINT_CANARY = (
    "http://CUSTOM_OPENAI_2_ENDPOINT_CANARY_3796.invalid/v1/chat/completions"
)
CUSTOM_OPENAI_1_MODEL = "fixed-custom-openai-1-model"
CUSTOM_OPENAI_2_MODEL = "fixed-custom-openai-2-model"
CUSTOM_OPENAI_RESPONSE_CANARY = "CUSTOM_OPENAI_RESPONSE_CANARY_3796"
CUSTOM_OPENAI_STREAM_CANARY = "CUSTOM_OPENAI_STREAM_CANARY_3796"
CUSTOM_OPENAI_EXCEPTION_CANARY = "CUSTOM_OPENAI_EXCEPTION_CANARY_3796"
CUSTOM_OPENAI_FILE_PATH_CANARY = "CUSTOM_OPENAI_FILE_PATH_CANARY_3796"
GENERAL_INPUT_CANARY = "GENERAL_INPUT_CANARY_3796"
GENERAL_PROMPT_CANARY = "GENERAL_PROMPT_CANARY_3796"
GENERAL_CREDENTIAL_CANARY = "G3N3R"
GENERAL_PATH_CANARY = "GENERAL_PATH_CANARY_3796"
GENERAL_RESPONSE_CANARY = "GENERAL_RESPONSE_CANARY_3796"
GENERAL_EXCEPTION_CANARY = "GENERAL_EXCEPTION_CANARY_3796"
GENERAL_ANALYZE_STREAM_EXCEPTION_CANARY = "GENERAL_ANALYZE_STREAM_EXCEPTION_CANARY_3796"
GENERAL_OPENAI_ENDPOINT_CANARY = "http://GENERAL_OPENAI_ENDPOINT_CANARY_3796.invalid/v1"
GENERAL_OPENAI_STREAM_CANARY = "GENERAL_OPENAI_STREAM_CANARY_3796"
GENERAL_OPENAI_STREAM_EXCEPTION_CANARY = "GENERAL_OPENAI_STREAM_EXCEPTION_CANARY_3796"
GENERAL_OPENAI_PRIVATE_STREAMING_VALUE = "PRIVATE_STREAMING_VALUE_3796"
GENERAL_OPENAI_EXCEPTION_CANARY = "GENERAL_OPENAI_EXCEPTION_CANARY_3796"
GENERAL_ANTHROPIC_RESPONSE_CANARY = "GENERAL_ANTHROPIC_RESPONSE_CANARY_3796"
GENERAL_ANTHROPIC_STREAM_CANARY = "GENERAL_ANTHROPIC_STREAM_CANARY_3796"
GENERAL_ANTHROPIC_EXCEPTION_CANARY = "GENERAL_ANTHROPIC_EXCEPTION_CANARY_3796"
COHERE_CREDENTIAL_CANARY = "C0H3R3_K3Y_3796"
COHERE_PROMPT_CANARY = "COHERE_PROMPT_CANARY_3796"
COHERE_RESPONSE_CANARY = "COHERE_RESPONSE_CANARY_3796"
COHERE_STREAM_CANARY = "COHERE_STREAM_CANARY_3796"
COHERE_EVENT_TYPE_CANARY = "COHERE_EVENT_TYPE_CANARY_3796"
COHERE_EXCEPTION_CANARY = "COHERE_EXCEPTION_CANARY_3796"
COHERE_PRIVATE_STREAMING_VALUE = "COHERE_PRIVATE_STREAMING_VALUE_3796"
GROQ_CREDENTIAL_CANARY = "GR0Q_K3Y_3796"
GROQ_INPUT_CANARY = "GROQ_INPUT_CANARY_3796"
GROQ_PROMPT_CANARY = "GROQ_PROMPT_CANARY_3796"
GROQ_RESPONSE_CANARY = "GROQ_RESPONSE_CANARY_3796"
GROQ_STREAM_CANARY = "GROQ_STREAM_CANARY_3796"
GROQ_EXCEPTION_CANARY = "GROQ_EXCEPTION_CANARY_3796"
GROQ_PRIVATE_STREAMING_VALUE = "GROQ_PRIVATE_STREAMING_VALUE_3796"
OPENROUTER_CREDENTIAL_CANARY = "0P3NR0UT3R_K3Y_3796"
OPENROUTER_INPUT_CANARY = "OPENROUTER_INPUT_CANARY_3796"
OPENROUTER_PROMPT_CANARY = "OPENROUTER_PROMPT_CANARY_3796"
OPENROUTER_RESPONSE_CANARY = "OPENROUTER_RESPONSE_CANARY_3796"
OPENROUTER_STREAM_CANARY = "OPENROUTER_STREAM_CANARY_3796"
OPENROUTER_EXCEPTION_CANARY = "OPENROUTER_EXCEPTION_CANARY_3796"
OPENROUTER_PRIVATE_STREAMING_VALUE = "OPENROUTER_PRIVATE_STREAMING_VALUE_3796"
HUGGINGFACE_CREDENTIAL_CANARY = "HUGG1NGFACE_K3Y_HFEND"
HUGGINGFACE_PROMPT_CANARY = "HUGGINGFACE_PROMPT_CANARY_3796"
HUGGINGFACE_RESPONSE_CANARY = "HUGGINGFACE_RESPONSE_CANARY_3796"
HUGGINGFACE_STREAM_CANARY = "HUGGINGFACE_STREAM_CANARY_3796"
HUGGINGFACE_EXCEPTION_CANARY = "HUGGINGFACE_EXCEPTION_CANARY_3796"
HUGGINGFACE_PRIVATE_STREAMING_VALUE = "HUGGINGFACE_PRIVATE_STREAMING_VALUE_3796"
DEEPSEEK_CREDENTIAL_CANARY = "D33PSEEK_K3Y_DSEND"
DEEPSEEK_RESPONSE_CANARY = "DEEPSEEK_RESPONSE_CANARY_3796"
DEEPSEEK_STREAM_CANARY = "DEEPSEEK_STREAM_CANARY_3796"
DEEPSEEK_EXCEPTION_CANARY = "DEEPSEEK_EXCEPTION_CANARY_3796"
DEEPSEEK_PRIVATE_STREAMING_VALUE = "DEEPSEEK_PRIVATE_STREAMING_VALUE_3796"
MISTRAL_CREDENTIAL_CANARY = "M1STRAL_K3Y_MSEND"
MISTRAL_RESPONSE_CANARY = "MISTRAL_RESPONSE_CANARY_3796"
MISTRAL_STREAM_CANARY = "MISTRAL_STREAM_CANARY_3796"
MISTRAL_EXCEPTION_CANARY = "MISTRAL_EXCEPTION_CANARY_3796"
MISTRAL_PRIVATE_STREAMING_VALUE = "MISTRAL_PRIVATE_STREAMING_VALUE_3796"
GOOGLE_CREDENTIAL_CANARY = "G00GL3_K3Y_GGEND"
GOOGLE_INPUT_CANARY = "GOOGLE_INPUT_CANARY_3796"
GOOGLE_PROMPT_CANARY = "GOOGLE_PROMPT_CANARY_3796"
GOOGLE_RESPONSE_CANARY = "GOOGLE_RESPONSE_CANARY_3796"
GOOGLE_STREAM_CANARY = "GOOGLE_STREAM_CANARY_3796"
GOOGLE_EXCEPTION_CANARY = "GOOGLE_EXCEPTION_CANARY_3796"
GOOGLE_PRIVATE_STREAMING_VALUE = "GOOGLE_PRIVATE_STREAMING_VALUE_3796"
MOCK_PROMPT_CANARY = "MOCK_PROMPT_CANARY_3796"
MOCK_SYSTEM_CANARY = "MOCK_SYSTEM_CANARY_3796"
MOCK_PRIVATE_STREAMING_VALUE = "MOCK_PRIVATE_STREAMING_VALUE_3796"
MOCK_EXCEPTION_CANARY = "MOCK_EXCEPTION_CANARY_3796"
CHUNK_RESPONSE_CANARY = "CHUNK_RESPONSE_CANARY_3796"
CHUNK_EXCEPTION_CANARY = "CHUNK_EXCEPTION_CANARY_3796"


@dataclass(frozen=True)
class _CustomOpenAIVariant:
    id: str
    summarizer: Callable[..., object]
    event_prefix: str
    settings_section: str
    api_key: str
    endpoint: str
    model: str


CUSTOM_OPENAI_VARIANTS = (
    _CustomOpenAIVariant(
        "local-custom-openai-1",
        local_summarization.summarize_with_custom_openai,
        "Custom OpenAI API",
        "custom_openai_api",
        CUSTOM_OPENAI_1_CREDENTIAL_CANARY,
        CUSTOM_OPENAI_1_ENDPOINT_CANARY,
        CUSTOM_OPENAI_1_MODEL,
    ),
    _CustomOpenAIVariant(
        "local-custom-openai-2",
        local_summarization.summarize_with_custom_openai_2,
        "Custom OpenAI API-2",
        "custom_openai_api_2",
        CUSTOM_OPENAI_2_CREDENTIAL_CANARY,
        CUSTOM_OPENAI_2_ENDPOINT_CANARY,
        CUSTOM_OPENAI_2_MODEL,
    ),
)


def _invoke_local_input(monkeypatch: pytest.MonkeyPatch) -> object:
    response = _FakeResponse(
        json_data={"choices": [{"message": {"content": "  fixed summary  "}}]}
    )
    # task-19830: `summarize_with_local_llm` gets its session from
    # `create_default_session()` now, not a bare `requests.post` call -- the
    # factory (imported into this module's namespace) is the seam to fake.
    monkeypatch.setattr(
        local_summarization, "create_default_session", lambda: _FakeSession(response)
    )
    return local_summarization.summarize_with_local_llm(
        LOCAL_INPUT_CANARY,
        "fixed prompt",
        0.2,
    )


def _invoke_local_prompt(monkeypatch: pytest.MonkeyPatch) -> object:
    response = _FakeResponse(json_data={"content": "  fixed llama summary  "})
    monkeypatch.setattr(local_summarization, "load_settings", _local_settings)
    monkeypatch.setattr(
        local_summarization,
        "create_default_session",
        lambda: _FakeSession(response),
    )
    return local_summarization.summarize_with_llama(
        "fixed input",
        LOCAL_PROMPT_CANARY,
        api_key="fixed-llama-key",
        system_message="fixed system message",
    )


def _invoke_local_credential(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setattr(local_summarization, "load_settings", _local_settings)
    generator = local_summarization.summarize_with_kobold(
        {"summary": "existing summary"},
        f"{LOCAL_CREDENTIAL_CANARY}-middle-{LOCAL_CREDENTIAL_CANARY}",
        "fixed prompt",
    )
    return _consume_generator(generator)


def _invoke_local_path(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setattr(
        local_summarization,
        "load_settings",
        lambda: _local_settings(llama_endpoint=LOCAL_PATH_CANARY),
    )
    return local_summarization.summarize_with_llama(
        {"summary": "existing summary"},
        "fixed prompt",
        api_key="fixed-llama-key",
    )


def _invoke_local_response(monkeypatch: pytest.MonkeyPatch) -> object:
    response = _FakeResponse(
        lines=(f"data: {{{LOCAL_RESPONSE_CANARY}".encode(), b"data: [DONE]")
    )
    monkeypatch.setattr(
        local_summarization, "create_default_session", lambda: _FakeSession(response)
    )
    generator = local_summarization.summarize_with_local_llm(
        "fixed input",
        "fixed prompt",
        0.2,
        streaming=True,
    )
    return list(generator)


def _invoke_local_exception(monkeypatch: pytest.MonkeyPatch) -> object:
    def raise_private_exception(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError(LOCAL_EXCEPTION_CANARY)

    # `summarize_with_local_llm` wraps its ENTIRE body (session creation
    # included) in one `try/except Exception`, so raising here -- at
    # `create_default_session()` -- reaches the exact same handler a raise
    # from `.post()` would, with the same observable (hidden-exception)
    # outcome.
    monkeypatch.setattr(
        local_summarization, "create_default_session", raise_private_exception
    )
    return local_summarization.summarize_with_local_llm(
        "fixed input",
        "fixed prompt",
        0.2,
    )


def _assert_fixed_summary(result: object) -> None:
    assert result == "fixed summary"


def _assert_fixed_llama_summary(result: object) -> None:
    assert result == "fixed llama summary"


def _assert_existing_kobold_summary(result: object) -> None:
    assert result == ([], "existing summary")


def _assert_existing_llama_summary(result: object) -> None:
    assert result == "existing summary"


def _assert_empty_stream(result: object) -> None:
    assert result == []


def _assert_local_exception_contract(result: object) -> None:
    assert result == (
        f"Local LLM: Error occurred while processing summary: {LOCAL_EXCEPTION_CANARY}"
    )


@dataclass(frozen=True)
class RuntimeSentinelCase:
    module: str
    category: str
    canary: str
    invoke: Callable[[pytest.MonkeyPatch, Path], object]
    assert_contract: Callable[[object], None]
    expected_event: str


def _ignore_runtime_tmp_path(
    invoke: Callable[[pytest.MonkeyPatch], object],
) -> Callable[[pytest.MonkeyPatch, Path], object]:
    def wrapped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> object:
        del tmp_path
        return invoke(monkeypatch)

    return wrapped


LOCAL_RUNTIME_SENTINEL_CASES = (
    RuntimeSentinelCase(
        "local",
        "input",
        LOCAL_INPUT_CANARY,
        _ignore_runtime_tmp_path(_invoke_local_input),
        _assert_fixed_summary,
        "Local LLM: Type of data:",
    ),
    RuntimeSentinelCase(
        "local",
        "prompt",
        LOCAL_PROMPT_CANARY,
        _ignore_runtime_tmp_path(_invoke_local_prompt),
        _assert_fixed_llama_summary,
        "Llama Summarize: Prompt prepared; character_count=",
    ),
    RuntimeSentinelCase(
        "local",
        "credential",
        LOCAL_CREDENTIAL_CANARY,
        _ignore_runtime_tmp_path(_invoke_local_credential),
        _assert_existing_kobold_summary,
        "Kobold: Credential state resolved",
    ),
    RuntimeSentinelCase(
        "local",
        "path",
        LOCAL_PATH_CANARY,
        _ignore_runtime_tmp_path(_invoke_local_path),
        _assert_existing_llama_summary,
        "Llama: API endpoint configured",
    ),
    RuntimeSentinelCase(
        "local",
        "response",
        LOCAL_RESPONSE_CANARY,
        _ignore_runtime_tmp_path(_invoke_local_response),
        _assert_empty_stream,
        "Local LLM: Failed to decode streamed JSON; line_length=",
    ),
    RuntimeSentinelCase(
        "local",
        "exception",
        LOCAL_EXCEPTION_CANARY,
        _ignore_runtime_tmp_path(_invoke_local_exception),
        _assert_local_exception_contract,
        "Local LLM: Processing failed; exception_type=RuntimeError",
    ),
)


def _install_signature_bound_general_settings(
    monkeypatch: pytest.MonkeyPatch,
    values: dict[tuple[str, str], object],
) -> list[tuple[tuple[object, ...], dict[str, object]]]:
    real_get_cli_setting = general_summarization.get_cli_setting
    signature = inspect.signature(real_get_cli_setting)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_get_cli_setting(*args: object, **kwargs: object) -> object:
        bound = signature.bind(*args, **kwargs)
        calls.append((args, kwargs))
        key = (bound.arguments["section"], bound.arguments.get("key"))
        if key in values:
            return values[key]
        return bound.arguments.get("default")

    monkeypatch.setattr(
        general_summarization,
        "get_cli_setting",
        fake_get_cli_setting,
    )
    return calls


def _install_signature_bound_general_session_post(
    monkeypatch: pytest.MonkeyPatch,
    result: _FakeResponse | BaseException,
) -> list[tuple[tuple[object, ...], dict[str, object]]]:
    real_post = general_summarization.requests.Session.post
    signature = inspect.signature(real_post)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_post(session: object, *args: object, **kwargs: object) -> _FakeResponse:
        signature.bind(session, *args, **kwargs)
        calls.append((args, kwargs))
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(general_summarization.requests.Session, "post", fake_post)
    return calls


def _install_signature_bound_general_requests_post(
    monkeypatch: pytest.MonkeyPatch,
    result: _FakeResponse | BaseException,
) -> list[tuple[tuple[object, ...], dict[str, object]]]:
    real_post = general_summarization.requests.post
    signature = inspect.signature(real_post)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_post(*args: object, **kwargs: object) -> _FakeResponse:
        signature.bind(*args, **kwargs)
        calls.append((args, kwargs))
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(general_summarization.requests, "post", fake_post)
    return calls


def _general_provider_settings(
    *,
    openai_key: str = "fixed-general-openai-key",
    openai_endpoint: str = "http://openai.invalid/v1",
) -> dict[tuple[str, str], object]:
    return {
        ("openai_api", "api_key"): openai_key,
        ("openai_api", "model"): "fixed-general-openai-model",
        ("openai_api", "api_retries"): 0,
        ("openai_api", "api_retry_delay"): 0,
        ("openai_api", "api_timeout"): 5,
        ("openai_api", "api_base_url"): openai_endpoint,
        ("anthropic_api", "api_key"): "fixed-general-anthropic-key",
        ("anthropic_api", "model"): "fixed-general-anthropic-model",
        ("anthropic_api", "api_retries"): 0,
        ("anthropic_api", "api_retry_delay"): 0,
    }


def _general_mid_provider_settings() -> dict[tuple[str, str], object]:
    return {
        ("cohere_api", "api_key"): "fixed-cohere-key",
        ("cohere_api", "model"): "fixed-cohere-model",
        ("cohere_api", "api_retries"): 0,
        ("cohere_api", "api_retry_delay"): 0,
        ("groq_api", "api_key"): "fixed-groq-key",
        ("groq_api", "model"): "fixed-groq-model",
        ("groq_api", "api_retries"): 0,
        ("groq_api", "api_retry_delay"): 0,
        ("openrouter_api", "api_key"): "fixed-openrouter-key",
        ("openrouter_api", "model"): "fixed-openrouter-model",
        ("openrouter_api", "api_retries"): 0,
        ("openrouter_api", "api_retry_delay"): 0,
    }


def _general_streaming_provider_settings() -> dict[tuple[str, str], object]:
    return {
        ("huggingface_api", "api_key"): "fixed-huggingface-key",
        ("huggingface_api", "model"): "fixed-huggingface-model",
        ("huggingface_api", "api_retries"): 0,
        ("huggingface_api", "api_retry_delay"): 0,
        ("deepseek_api", "api_key"): "fixed-deepseek-key",
        ("deepseek_api", "model"): "fixed-deepseek-model",
        ("deepseek_api", "api_retries"): 0,
        ("deepseek_api", "api_retry_delay"): 0,
        ("mistral_api", "api_key"): "fixed-mistral-key",
        ("mistral_api", "model"): "fixed-mistral-model",
        ("mistral_api", "api_retries"): 0,
        ("mistral_api", "api_retry_delay"): 0,
    }


def _google_provider_settings() -> dict[tuple[str, str], object]:
    return {
        ("google_api", "api_key"): "fixed-google-key",
        ("google_api", "model"): "fixed-google-model",
        ("google_api", "api_retries"): 0,
        ("google_api", "api_retry_delay"): 0,
    }


def _install_signature_bound_general_config_loader(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException | None = None,
) -> list[tuple[tuple[object, ...], dict[str, object]]]:
    real_loader = general_summarization.load_and_log_configs
    signature = inspect.signature(real_loader)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_loader(*args: object, **kwargs: object) -> None:
        signature.bind(*args, **kwargs)
        calls.append((args, kwargs))
        if failure is not None:
            raise failure

    monkeypatch.setattr(general_summarization, "load_and_log_configs", fake_loader)
    return calls


def _invoke_general_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> object:
    del monkeypatch, tmp_path
    return general_summarization.extract_text_from_segments(
        [{"Text": GENERAL_INPUT_CANARY}, {"missing": GENERAL_INPUT_CANARY}]
    )


def _invoke_general_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> object:
    del tmp_path
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_provider_settings(),
    )
    _install_signature_bound_general_session_post(
        monkeypatch,
        _FakeResponse(
            json_data={"choices": [{"message": {"content": "  fixed openai  "}}]}
        ),
    )
    return general_summarization.summarize_with_openai(
        "fixed-general-openai-key",
        "fixed input",
        GENERAL_PROMPT_CANARY,
        system_message=f"system-{GENERAL_PROMPT_CANARY}",
    )


def _invoke_general_credential(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> object:
    del monkeypatch, tmp_path
    return general_summarization.summarize_with_anthropic(
        GENERAL_CREDENTIAL_CANARY,
        {"summary": "fixed existing anthropic summary"},
        "fixed prompt",
    )


def _invoke_general_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> object:
    del monkeypatch
    private_path = tmp_path / f"{GENERAL_PATH_CANARY}.txt"
    private_path.write_text("  fixed file contents  ", encoding="utf-8")
    return general_summarization.extract_text_from_input(str(private_path))


def _invoke_general_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> object:
    del tmp_path
    real_dispatch = general_summarization._dispatch_to_api
    signature = inspect.signature(real_dispatch)

    def fake_dispatch(*args: object, **kwargs: object) -> Iterator[str]:
        signature.bind(*args, **kwargs)

        def response_stream() -> Iterator[str]:
            yield GENERAL_RESPONSE_CANARY

        return response_stream()

    monkeypatch.setattr(general_summarization, "_dispatch_to_api", fake_dispatch)
    return general_summarization.analyze(
        "openai",
        "fixed input",
        "fixed prompt",
    )


def _invoke_general_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> object:
    del monkeypatch, tmp_path

    def raise_private_exception(text: str) -> str:
        assert text == "fixed chunk"
        raise RuntimeError(GENERAL_EXCEPTION_CANARY)

    return general_summarization.recursive_summarize_chunks(
        ["fixed chunk"],
        raise_private_exception,
    )


def _assert_general_input_contract(result: object) -> None:
    assert result == GENERAL_INPUT_CANARY


def _assert_general_prompt_contract(result: object) -> None:
    assert result == "fixed openai"


def _assert_general_credential_contract(result: object) -> None:
    assert result == "fixed existing anthropic summary"


def _assert_general_path_contract(result: object) -> None:
    assert result == "fixed file contents"


def _assert_general_response_contract(result: object) -> None:
    assert result == GENERAL_RESPONSE_CANARY


def _assert_general_exception_contract(result: object) -> None:
    assert result == (
        f"Error: Unexpected failure during recursive step 1: {GENERAL_EXCEPTION_CANARY}"
    )


GENERAL_RUNTIME_SENTINEL_CASES = (
    RuntimeSentinelCase(
        "general",
        "input",
        GENERAL_INPUT_CANARY,
        _invoke_general_input,
        _assert_general_input_contract,
        "Skipping segment due to missing text key or wrong type",
    ),
    RuntimeSentinelCase(
        "general",
        "prompt",
        GENERAL_PROMPT_CANARY,
        _invoke_general_prompt,
        _assert_general_prompt_contract,
        "OpenAI: Request options prepared",
    ),
    RuntimeSentinelCase(
        "general",
        "credential",
        GENERAL_CREDENTIAL_CANARY,
        _invoke_general_credential,
        _assert_general_credential_contract,
        "Anthropic: Using API key provided as parameter",
    ),
    RuntimeSentinelCase(
        "general",
        "path",
        GENERAL_PATH_CANARY,
        _invoke_general_path,
        _assert_general_path_contract,
        "Input resolved as file path",
    ),
    RuntimeSentinelCase(
        "general",
        "response",
        GENERAL_RESPONSE_CANARY,
        _invoke_general_response,
        _assert_general_response_contract,
        "Summarization completed successfully. Final Length:",
    ),
    RuntimeSentinelCase(
        "general",
        "exception",
        GENERAL_EXCEPTION_CANARY,
        _invoke_general_exception,
        _assert_general_exception_contract,
        "Unexpected error calling summarize_func; step=1 exception_type=RuntimeError",
    ),
)

RUNTIME_SENTINEL_CASES = (
    *LOCAL_RUNTIME_SENTINEL_CASES,
    *GENERAL_RUNTIME_SENTINEL_CASES,
)


def _guard() -> ModuleType:
    try:
        return importlib.import_module("Tests.LLM_Calls.summarization_diagnostic_guard")
    except ModuleNotFoundError:
        pytest.fail("summarization diagnostic guard is not implemented")


def _message_shape(expression: str) -> str:
    node = ast.parse(f"logger.info({expression})").body[0]
    assert isinstance(node, ast.Expr)
    assert isinstance(node.value, ast.Call)
    return ast.dump(node.value.args[0], include_attributes=False)


def _single_call(source: str):
    calls = _guard().discover_diagnostic_calls(source, module="synthetic.py")
    assert len(calls) == 1
    return calls[0]


def _review_fixture() -> dict[str, object]:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    assert ledger["schema_version"] == 1
    return ledger


def _ledger_sites() -> list[dict[str, object]]:
    ledger = _review_fixture()
    return ledger["sites"]


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_inventory_projection(
    inventory: dict[str, object], owned_paths: set[str]
) -> dict[str, object]:
    normalized = copy.deepcopy(inventory)
    summary = normalized["summary"]
    assert isinstance(summary, dict)
    owners = normalized["owners"]
    assert isinstance(owners, list)
    for owner in owners:
        assert isinstance(owner, dict)
        if owner.get("path") in owned_paths:
            owner["call_count"] = "<task-3796-owned-call-count>"
            owner["diagnostic_digest"] = "<task-3796-owned-diagnostic-digest>"
    summary["task_492_calls"] = "<derived-task-492-call-count>"
    return normalized


def _assert_task_492_summary(
    inventory: dict[str, object], *, inventory_name: str
) -> None:
    summary = inventory["summary"]
    assert isinstance(summary, dict)
    owners = inventory["owners"]
    assert isinstance(owners, list)
    task_492_call_counts = []
    for owner in owners:
        assert isinstance(owner, dict)
        if owner.get("owner") != "TASK-492":
            continue
        call_count = owner.get("call_count")
        assert type(call_count) is int
        task_492_call_counts.append(call_count)
    task_492_calls = summary.get("task_492_calls")
    assert type(task_492_calls) is int
    assert task_492_calls == sum(task_492_call_counts), (
        f"{inventory_name} TASK-492 summary does not equal its owner call counts"
    )


def _starting_projection(
    sites: list[dict[str, object]],
) -> list[dict[str, object]]:
    projection = []
    for site in sorted(sites, key=lambda item: item["site_id"]):
        record = {
            key: site[key]
            for key in (
                "site_id",
                "module",
                "qualname",
                "group",
                "starting_classification",
            )
        }
        detail = (
            "category"
            if site["starting_classification"] == "private"
            else "safe_reason"
        )
        record[detail] = site[detail]
        record["starting"] = site["starting"]
        projection.append(record)
    return projection


def _call_from_record(site: dict[str, object], record_name: str):
    record = site[record_name]
    assert isinstance(record, dict)
    return _guard().DiagnosticCall(
        module=site["module"],
        qualname=site["qualname"],
        method=record["method"],
        event=record["event"],
        occurrence=record["occurrence"],
        message_shape=record["message_shape"],
        expressions=tuple(record["expressions"]),
        captures_exception=record["captures_exception"],
        level_expression=record.get("level_expression"),
    )


def _assert_ledger_lifecycle(sites: list[dict[str, object]]) -> None:
    for site in sites:
        classification = site["starting_classification"]
        outcome = site["outcome"]
        if classification == "reviewed_safe":
            assert outcome == "frozen", (
                f"reviewed_safe diagnostic must remain frozen: {site['site_id']}"
            )
        elif classification == "private":
            assert outcome in {"pending", "metadata", "deleted"}, (
                "private diagnostic has unapproved lifecycle outcome: "
                f"{site['site_id']}={outcome}"
            )
        else:
            raise AssertionError(
                "unknown starting diagnostic classification: "
                f"{site['site_id']}={classification}"
            )

        if outcome == "deleted":
            deletion_reason = site.get("deletion_reason")
            assert isinstance(deletion_reason, str) and deletion_reason.strip(), (
                "deleted diagnostic requires a non-empty deletion_reason: "
                f"{site['site_id']}"
            )
        else:
            assert "deletion_reason" not in site, (
                f"only deleted diagnostics may have deletion_reason: {site['site_id']}"
            )


def _assert_private_category_matrix(sites: list[dict[str, object]]) -> None:
    private_sites = [
        site for site in sites if site["starting_classification"] == "private"
    ]
    assert Counter(site["category"] for site in private_sites) == (
        PRIVATE_CATEGORY_COUNTS
    )
    for module, expected in PRIVATE_CATEGORY_COUNTS_BY_MODULE.items():
        assert (
            Counter(
                site["category"] for site in private_sites if site["module"] == module
            )
            == expected
        )


def test_guard_finds_stdlib_loguru_nested_and_bound_calls() -> None:
    source = """
import logging
from loguru import logger as loguru_logger

audit_logger = logging.getLogger(__name__)

audit_logger.error("stdlib event", account_id)

def outer():
    logger.info("duplicate label", first)
    logger.info("duplicate label", second)

    def stream_generator():
        loguru_logger.bind(session=session_id).opt(colors=True).warning(
            f"stream chunk {chunk.index}", extra_field
        )

    return stream_generator
"""

    calls = _guard().discover_diagnostic_calls(source, module="synthetic.py")

    assert [(call.qualname, call.method, call.event) for call in calls] == [
        ("<module>", "error", "stdlib event"),
        ("outer", "info", "duplicate label"),
        ("outer", "info", "duplicate label"),
        ("outer.stream_generator", "warning", "stream chunk "),
    ]
    assert [call.occurrence for call in calls] == [1, 1, 2, 1]
    assert calls[3].expressions == (
        "chunk.index",
        "extra_field",
        "session=session_id",
        "colors=True",
    )


def test_guard_finds_imported_logging_method_callables() -> None:
    source = """
from logging import error, warning as warn

error(private_value)
warn(msg=other_private)
"""

    calls = _guard().discover_diagnostic_calls(source, module="synthetic.py")

    assert [(call.method, call.event) for call in calls] == [
        ("error", ""),
        ("warning", ""),
    ]
    assert [call.expressions for call in calls] == [
        ("private_value",),
        ("other_private",),
    ]
    for call in calls:
        with pytest.raises(AssertionError, match="constant string first argument"):
            _guard().assert_review_outcome(call, call, outcome="metadata")


def test_guard_imported_exception_callable_captures_exception() -> None:
    call = _single_call(
        'from logging import exception as log_exception\nlog_exception("fixed")\n'
    )

    assert call.method == "exception"
    assert call.captures_exception is True
    with pytest.raises(AssertionError, match="must not capture exception"):
        _guard().assert_review_outcome(call, call, outcome="metadata")


def test_guard_finds_imported_getlogger_factory_results() -> None:
    source = """
from logging import getLogger, getLogger as factory

direct = getLogger(__name__)
aliased = factory(__name__)
direct.warning("direct event", direct_private)
aliased.error("aliased event", aliased_private)
"""

    calls = _guard().discover_diagnostic_calls(source, module="synthetic.py")

    assert [(call.method, call.event, call.expressions) for call in calls] == [
        ("warning", "direct event", ("direct_private",)),
        ("error", "aliased event", ("aliased_private",)),
    ]


def test_guard_does_not_follow_arbitrary_factory_results() -> None:
    source = """
def factory(name):
    return object()

audit = factory(__name__)
audit.error("fixed", private)
"""

    assert _guard().discover_diagnostic_calls(source, module="synthetic.py") == []


def test_guard_alias_state_clears_on_later_getlogger_assignment() -> None:
    source = """
from logging import getLogger
from loguru import logger

audit = logger.bind(secret=private_value).opt(exception=private_exc)
audit.error("first")
audit = getLogger(__name__)
audit.error("second")
"""

    calls = _guard().discover_diagnostic_calls(source, module="synthetic.py")

    assert [call.expressions for call in calls] == [
        ("secret=private_value", "exception=private_exc"),
        (),
    ]
    assert [call.captures_exception for call in calls] == [True, False]


def test_guard_alias_state_uses_later_bound_assignment() -> None:
    source = """
from logging import getLogger
from loguru import logger

audit = getLogger(__name__)
audit.error("first")
audit = logger.bind(secret=private_value).opt(exception=private_exc)
audit.error("second")
"""

    calls = _guard().discover_diagnostic_calls(source, module="synthetic.py")

    assert [call.expressions for call in calls] == [
        (),
        ("secret=private_value", "exception=private_exc"),
    ]
    assert [call.captures_exception for call in calls] == [False, True]


def test_guard_follows_direct_logger_alias() -> None:
    call = _single_call("audit = logger\naudit.info(private_value)\n")

    assert call.method == "info"
    assert call.expressions == ("private_value",)


def test_guard_follows_getlogger_factory_alias_chain() -> None:
    source = """
from logging import getLogger

factory = getLogger
audit = factory(__name__)
audit.error(private_value)
"""

    call = _single_call(source)

    assert call.method == "error"
    assert call.expressions == ("private_value",)


def test_guard_follows_imported_severity_callable_alias() -> None:
    source = """
from logging import error

emit = error
emit(private_value)
"""

    call = _single_call(source)

    assert call.method == "error"
    assert call.expressions == ("private_value",)


def test_guard_alias_state_is_lexically_scoped() -> None:
    source = """
from logging import getLogger
from loguru import logger

def first():
    audit = logger.bind(secret=first_private)
    audit.info("first")

def second():
    audit = getLogger(__name__)
    audit.info("second")
"""

    calls = _guard().discover_diagnostic_calls(source, module="synthetic.py")

    assert [(call.qualname, call.expressions) for call in calls] == [
        ("first", ("secret=first_private",)),
        ("second", ()),
    ]


def test_guard_branch_alias_state_unions_fields_and_exception_capture() -> None:
    source = """
from logging import getLogger
from loguru import logger

audit = logger.bind(shared=shared_value)
if use_private_backend:
    audit = logger.bind(
        shared=shared_value,
        branch_secret=private_value,
    ).opt(exception=private_exc)
audit.error("after branch")
"""

    call = _single_call(source)

    assert call.expressions == (
        "shared=shared_value",
        "branch_secret=private_value",
        "exception=private_exc",
    )
    assert call.captures_exception is True


def test_guard_branch_alias_state_retains_prebranch_possible_value() -> None:
    source = """
from logging import getLogger
from loguru import logger

audit = logger.bind(secret=private_value).opt(exception=private_exc)
if use_standard_logger:
    audit = getLogger(__name__)
audit.error("after branch")
"""

    call = _single_call(source)

    assert call.expressions == (
        "secret=private_value",
        "exception=private_exc",
    )
    assert call.captures_exception is True


def test_guard_branch_callable_method_union_is_discovered() -> None:
    source = """
import logging

if use_error:
    emit = logging.error
else:
    emit = logging.warning
emit(private_value)
"""

    call = _single_call(source)

    assert call.method == "error|warning"
    assert call.expressions == ("private_value",)
    assert call.captures_exception is False
    _guard().assert_review_outcome(call, call, outcome="pending")
    _guard().assert_review_outcome(call, call, outcome="frozen")


@pytest.mark.parametrize(
    ("body_method", "else_method"),
    [("log_error", "log_warning"), ("log_warning", "log_error")],
)
def test_guard_imported_callable_method_union_is_sorted(
    body_method: str, else_method: str
) -> None:
    source = f"""
from logging import error as log_error, warning as log_warning

if choose_body:
    emit = {body_method}
else:
    emit = {else_method}
emit(private_value)
"""

    call = _single_call(source)

    assert call.method == "error|warning"
    assert call.expressions == ("private_value",)


def test_guard_branch_callable_method_union_captures_any_exception() -> None:
    source = """
import logging

if include_traceback:
    emit = logging.exception
else:
    emit = logging.error
emit("fixed")
"""

    call = _single_call(source)

    assert call.method == "error|exception"
    assert call.captures_exception is True


def test_guard_same_severity_method_union_can_migrate_to_metadata() -> None:
    starting = _single_call(
        """
import logging

if include_traceback:
    emit = logging.exception
else:
    emit = logging.error
emit(private_value)
"""
    )
    current = _single_call('logging.error("fixed")\n')

    _guard().assert_review_outcome(starting, current, outcome="metadata")


def test_guard_ambiguous_severity_method_union_rejects_metadata() -> None:
    starting = _single_call(
        """
import logging

if use_error:
    emit = logging.error
else:
    emit = logging.warning
emit(private_value)
"""
    )
    current = _single_call('logging.error("fixed")\n')

    with pytest.raises(AssertionError, match="unambiguous diagnostic severity"):
        _guard().assert_review_outcome(starting, current, outcome="metadata")


def test_guard_closure_inherits_later_enclosing_bound_logger_state() -> None:
    source = """
from logging import getLogger
from loguru import logger

def outer():
    audit = getLogger(__name__)

    def inner():
        audit.error("inside closure")

    audit = logger.bind(secret=private_value).opt(exception=private_exc)
    inner()
"""

    call = _single_call(source)

    assert call.qualname == "outer.inner"
    assert call.expressions == (
        "secret=private_value",
        "exception=private_exc",
    )
    assert call.captures_exception is True


def test_guard_closure_conservatively_retains_definition_state_after_clear() -> None:
    source = """
from logging import getLogger
from loguru import logger

def outer():
    audit = logger.bind(secret=private_value).opt(exception=private_exc)

    def inner():
        audit.error("inside closure")

    audit = getLogger(__name__)
    inner()
"""

    call = _single_call(source)

    assert call.qualname == "outer.inner"
    assert call.expressions == (
        "secret=private_value",
        "exception=private_exc",
    )
    assert call.captures_exception is True


def test_guard_closure_parameter_shadows_enclosing_logger_alias() -> None:
    source = """
from loguru import logger

def outer():
    audit = logger.bind(secret=private_value).opt(exception=private_exc)

    def inner(audit):
        audit.error("parameter-owned")
"""

    assert _guard().discover_diagnostic_calls(source, module="synthetic.py") == []


def test_guard_closure_local_logger_aliases_replace_enclosing_state() -> None:
    source = """
from logging import getLogger
from loguru import logger

def outer():
    audit = logger.bind(secret=outer_private).opt(exception=outer_exc)

    def standard():
        audit = getLogger(__name__)
        audit.error("standard")

    def bound():
        audit = logger.bind(secret=inner_private).opt(exception=inner_exc)
        audit.error("bound")
"""

    calls = _guard().discover_diagnostic_calls(source, module="synthetic.py")

    assert [(call.qualname, call.expressions) for call in calls] == [
        ("outer.standard", ()),
        (
            "outer.bound",
            ("secret=inner_private", "exception=inner_exc"),
        ),
    ]
    assert [call.captures_exception for call in calls] == [False, True]


def test_guard_closure_keeps_sibling_and_nested_generator_scopes_distinct() -> None:
    source = """
from logging import getLogger
from loguru import logger

def outer():
    audit = logger.bind(secret=outer_private)

    def first():
        audit.info("first")

    def second():
        audit = getLogger(__name__)

        def stream_generator():
            audit.warning("stream")

        return stream_generator
"""

    calls = _guard().discover_diagnostic_calls(source, module="synthetic.py")

    assert [(call.qualname, call.expressions) for call in calls] == [
        ("outer.first", ("secret=outer_private",)),
        ("outer.second.stream_generator", ()),
    ]


def test_guard_class_method_inherits_later_module_logger_state() -> None:
    source = """
from logging import getLogger

class C:
    def method(self):
        audit.error(private_value)

audit = getLogger(__name__)
"""

    call = _single_call(source)

    assert call.qualname == "C.method"
    assert call.method == "error"
    assert call.expressions == ("private_value",)


def test_guard_class_attribute_does_not_shadow_method_free_logger() -> None:
    source = """
from loguru import logger

audit = logger.bind(secret=module_private).opt(exception=module_exc)

class C:
    audit = object()

    def method(self):
        audit.error(private_value)
"""

    call = _single_call(source)

    assert call.qualname == "C.method"
    assert call.expressions == (
        "private_value",
        "secret=module_private",
        "exception=module_exc",
    )
    assert call.captures_exception is True


def test_guard_nested_class_method_inherits_enclosing_function_logger() -> None:
    source = """
from loguru import logger

def outer():
    audit = logger.bind(secret=outer_private).opt(exception=outer_exc)

    class C:
        audit = object()

        def method(self):
            audit.warning(private_value)
"""

    call = _single_call(source)

    assert call.qualname == "outer.C.method"
    assert call.expressions == (
        "private_value",
        "secret=outer_private",
        "exception=outer_exc",
    )
    assert call.captures_exception is True


def test_guard_class_body_diagnostics_use_class_body_state() -> None:
    source = """
from loguru import logger

audit = logger.bind(secret=module_private)

class C:
    audit.info("before class assignment")
    audit = logger.bind(secret=class_private)
    audit.info("after class assignment")
"""

    calls = _guard().discover_diagnostic_calls(source, module="synthetic.py")

    assert [(call.qualname, call.expressions) for call in calls] == [
        ("C", ("secret=module_private",)),
        ("C", ("secret=class_private",)),
    ]


def test_guard_method_parameter_and_local_assignment_shadow_free_logger() -> None:
    source = """
from logging import getLogger
from loguru import logger

audit = logger.bind(secret=module_private).opt(exception=module_exc)

class C:
    def parameter(self, audit):
        audit.error("parameter")

    def local(self):
        audit = getLogger(__name__)
        audit.error(private_value)
"""

    calls = _guard().discover_diagnostic_calls(source, module="synthetic.py")

    assert [(call.qualname, call.expressions) for call in calls] == [
        ("C.local", ("private_value",)),
    ]
    assert calls[0].captures_exception is False


def test_guard_nested_class_body_uses_module_bound_logger_not_outer_attribute() -> None:
    source = """
from logging import getLogger
from loguru import logger

audit = logger.bind(secret=module_private).opt(exception=module_exc)

class Outer:
    audit = getLogger(__name__)

    class Inner:
        audit.error(private_value)
"""

    call = _single_call(source)

    assert call.qualname == "Outer.Inner"
    assert call.expressions == (
        "private_value",
        "secret=module_private",
        "exception=module_exc",
    )
    assert call.captures_exception is True


def test_guard_nested_class_body_ignores_outer_bound_logger_attribute() -> None:
    source = """
from logging import getLogger
from loguru import logger

audit = getLogger(__name__)

class Outer:
    audit = logger.bind(secret=outer_private).opt(exception=outer_exc)

    class Inner:
        audit.error(private_value)
"""

    call = _single_call(source)

    assert call.qualname == "Outer.Inner"
    assert call.expressions == ("private_value",)
    assert call.captures_exception is False


def test_guard_finds_bind_and_opt_derived_logger_aliases() -> None:
    source = """
from loguru import logger

bound = logger.bind(context=private_value)
configured = logger.opt(exception=private_exc)
bound.info("bound event")
configured.warning("configured event")
"""

    calls = _guard().discover_diagnostic_calls(source, module="synthetic.py")

    assert [(call.method, call.event) for call in calls] == [
        ("info", "bound event"),
        ("warning", "configured event"),
    ]
    assert calls[0].expressions == ("context=private_value",)
    assert calls[1].expressions == ("exception=private_exc",)
    assert calls[0].captures_exception is False
    assert calls[1].captures_exception is True
    with pytest.raises(AssertionError, match="approved metadata expression"):
        _guard().assert_review_outcome(calls[0], calls[0], outcome="metadata")
    with pytest.raises(AssertionError, match="must not capture exception"):
        _guard().assert_review_outcome(calls[1], calls[1], outcome="frozen")


def test_guard_preserves_bind_and_opt_keyword_names() -> None:
    safe = _single_call(
        'logger.bind(safe_count=value).opt(colors=True).info("event")\n'
    )
    private = _single_call(
        'logger.bind(private_summary=value).opt(colors=True).info("event")\n'
    )

    assert safe.expressions == ("safe_count=value", "colors=True")
    assert private.expressions == ("private_summary=value", "colors=True")
    assert safe.expressions != private.expressions


@pytest.mark.timeout(1)
def test_guard_handles_logger_rebound_to_derived_alias() -> None:
    source = """
from loguru import logger

logger = logger.bind(context=private_value)
logger.info("event")
"""

    call = _single_call(source)

    assert call.expressions == ("context=private_value",)


def test_guard_identity_ignores_line_movement() -> None:
    compact = """
def summarize():
    logger.info("summary ready", len(summary))
"""
    moved = """


# unrelated navigation-only movement
def summarize():

    logger.info("summary ready", len(summary))
"""

    before = _single_call(compact)
    after = _single_call(moved)

    assert before.identity == after.identity
    assert before == after


def test_guard_records_keyword_only_message_and_detects_addition() -> None:
    starting_source = 'logger.info("existing event")\n'
    added_source = starting_source + "logger.info(msg=private_summary)\n"

    starting = _guard().discover_diagnostic_calls(
        starting_source, module="synthetic.py"
    )
    added = _guard().discover_diagnostic_calls(added_source, module="synthetic.py")

    assert len(added) == len(starting) + 1
    assert {call.identity for call in added} != {call.identity for call in starting}
    keyword_call = added[-1]
    assert keyword_call.event == ""
    assert keyword_call.message_shape == _message_shape("private_summary")
    assert keyword_call.expressions == ("private_summary",)
    with pytest.raises(AssertionError, match="constant string first argument"):
        _guard().assert_review_outcome(keyword_call, keyword_call, outcome="metadata")


def test_guard_records_recognized_call_without_message() -> None:
    call = _single_call("logger.info(extra=private_value)\n")

    assert call.event == ""
    assert call.message_shape == "<missing>"
    assert call.expressions == ("private_value",)
    with pytest.raises(AssertionError, match="constant string first argument"):
        _guard().assert_review_outcome(call, call, outcome="metadata")


def test_guard_log_calls_use_second_positional_message() -> None:
    source = """
import logging
from loguru import logger

logging.log(logging.WARNING, "stdlib event: %s", stdlib_field)
logger.log("INFO", f"loguru event: {private_value}", loguru_field)
"""

    calls = _guard().discover_diagnostic_calls(source, module="synthetic.py")

    assert [call.event for call in calls] == [
        "stdlib event: %s",
        "loguru event: ",
    ]
    assert [call.message_shape for call in calls] == [
        _message_shape('"stdlib event: %s"'),
        _message_shape('f"loguru event: {private_value}"'),
    ]
    assert [call.expressions for call in calls] == [
        ("stdlib_field",),
        ("private_value", "loguru_field"),
    ]
    assert [getattr(call, "level_expression", None) for call in calls] == [
        "logging.WARNING",
        "'INFO'",
    ]


def test_guard_ledger_record_round_trips_log_level_expression() -> None:
    site = {
        "module": "synthetic.py",
        "qualname": "summarize",
        "current": {
            "method": "log",
            "event": "fixed event",
            "occurrence": 1,
            "message_shape": _message_shape('"fixed event"'),
            "expressions": [],
            "captures_exception": False,
            "level_expression": "logging.ERROR",
        },
    }

    call = _call_from_record(site, "current")

    assert call.level_expression == "logging.ERROR"


def test_guard_records_dynamic_format_receiver() -> None:
    call = _single_call("logger.error(template.format(secret))\n")

    assert call.event == ""
    assert call.message_shape == _message_shape("template.format(secret)")
    assert call.expressions == ("template", "secret")
    with pytest.raises(AssertionError, match="constant string first argument"):
        _guard().assert_review_outcome(call, call, outcome="metadata")


def test_guard_records_nested_fstring_format_spec_expressions() -> None:
    call = _single_call('logger.info(f"secret={secret:{width}.{precision}f}")\n')

    assert call.event == "secret="
    assert call.message_shape == _message_shape(
        'f"secret={secret:{width}.{precision}f}"'
    )
    assert call.expressions == ("secret", "width", "precision")
    with pytest.raises(AssertionError, match="constant string first argument"):
        _guard().assert_review_outcome(call, call, outcome="metadata")


def test_guard_percent_event_excludes_literal_rhs() -> None:
    call = _single_call('logger.warning("private marker: %s" % "literal-private")\n')

    assert call.event == "private marker: %s"
    assert call.message_shape == _message_shape(
        '"private marker: %s" % "literal-private"'
    )
    assert call.expressions == ("'literal-private'",)
    with pytest.raises(AssertionError, match="constant string first argument"):
        _guard().assert_review_outcome(call, call, outcome="metadata")


def test_guard_rejects_changed_reviewed_safe_expression() -> None:
    starting = _single_call('logger.info("Retry count: {}", retry_count)\n')
    changed = _single_call('logger.info("Retry count: {}", retry_total)\n')

    assert starting.identity == changed.identity
    with pytest.raises(AssertionError, match="frozen diagnostic changed"):
        _guard().assert_review_outcome(starting, changed, outcome="frozen")


def test_guard_accepts_metadata_replacement_with_new_fixed_event() -> None:
    starting = _single_call('logger.error(f"Request failed: {error}")\n')
    repaired = _single_call(
        'logger.error("Request failed; exception_type=%s", '
        "safe_metadata_token(type(exc).__name__))\n"
    )

    _guard().assert_review_outcome(starting, repaired, outcome="metadata")


@pytest.mark.parametrize(
    "expression",
    [
        "response.text",
        "str(exc)",
        "exc",
        "api_key[:5]",
        "custom_prompt_arg",
        "input_data",
        "summary",
        "response_data",
        "type(exc).__name__",
        "response.headers",
        "response.status",
        "http_response.status_code",
        "safe_metadata_token(event_type, provider_name)",
        "safe_metadata_token(value=event_type)",
        "utils.safe_metadata_token(event_type)",
    ],
)
def test_guard_metadata_rejects_private_lazy_expression(
    expression: str,
) -> None:
    starting = _single_call('logger.error(f"Legacy failure: {private}")\n')
    current = _single_call(f'logger.error("Fixed failure; field=%s", {expression})\n')

    with pytest.raises(AssertionError, match="approved metadata expression"):
        _guard().assert_review_outcome(starting, current, outcome="metadata")


def test_guard_metadata_rejection_names_rejected_source_expression() -> None:
    starting = _single_call('logger.error(f"Legacy failure: {private}")\n')
    current = _single_call('logger.error("Fixed failure; field=%s", response.text)\n')

    with pytest.raises(AssertionError, match=r"response\.text"):
        _guard().assert_review_outcome(starting, current, outcome="metadata")


@pytest.mark.parametrize(
    "expression",
    [
        "7",
        "True",
        "len(prompt)",
        "len(response.content)",
        "response.status_code",
        "character_count",
        "payload_length",
        "retry_count",
        "attempt + 1",
        "streaming",
        "safe_metadata_token(type(exc).__name__)",
    ],
)
def test_guard_metadata_accepts_approved_lazy_expression(
    expression: str,
) -> None:
    starting = _single_call('logger.error(f"Legacy failure: {private}")\n')
    current = _single_call(f'logger.error("Fixed failure; field=%s", {expression})\n')

    _guard().assert_review_outcome(starting, current, outcome="metadata")


@pytest.mark.parametrize(
    "expression",
    [
        "i + 1",
        "safe_metadata_token(event_type)",
        "safe_metadata_token(provider_name)",
    ],
)
def test_guard_metadata_accepts_index_arithmetic_and_sanitized_tokens(
    expression: str,
) -> None:
    starting = _single_call('logger.error(f"Legacy failure: {private}")\n')
    current = _single_call(f'logger.error("Fixed failure; field=%s", {expression})\n')

    _guard().assert_review_outcome(starting, current, outcome="metadata")


@pytest.mark.parametrize(
    "expression",
    [
        "safe_metadata_token(*private_values)",
        "safe_metadata_token(*(event_type, provider_name))",
    ],
)
def test_guard_metadata_rejects_starred_sanitizer_arguments(
    expression: str,
) -> None:
    starting = _single_call('logger.error(f"Legacy failure: {private}")\n')
    current = _single_call(f'logger.error("Fixed failure; field=%s", {expression})\n')

    with pytest.raises(AssertionError, match="approved metadata expression"):
        _guard().assert_review_outcome(starting, current, outcome="metadata")


@pytest.mark.parametrize("expression", ["i", "index", "idx"])
def test_guard_metadata_rejects_bare_index_names(expression: str) -> None:
    starting = _single_call('logger.error(f"Legacy failure: {private}")\n')
    current = _single_call(f'logger.error("Fixed failure; field=%s", {expression})\n')

    with pytest.raises(AssertionError, match="approved metadata expression"):
        _guard().assert_review_outcome(starting, current, outcome="metadata")


@pytest.mark.parametrize(
    "expression",
    [
        "i + safe_metadata_token(event_type)",
        "idx + streaming",
        "index + True",
    ],
)
def test_guard_metadata_rejects_non_numeric_index_compositions(
    expression: str,
) -> None:
    starting = _single_call('logger.error(f"Legacy failure: {private}")\n')
    current = _single_call(f'logger.error("Fixed failure; field=%s", {expression})\n')

    with pytest.raises(AssertionError, match="approved metadata expression"):
        _guard().assert_review_outcome(starting, current, outcome="metadata")


def test_guard_metadata_accepts_fixed_event_without_fields() -> None:
    starting = _single_call('logger.error(f"Legacy failure: {private}")\n')
    current = _single_call('logger.error("Fixed failure")\n')

    _guard().assert_review_outcome(starting, current, outcome="metadata")


def test_guard_metadata_allows_exception_to_error_same_severity() -> None:
    starting = _single_call('logging.exception("Legacy failure: %s", private)\n')
    current = _single_call('logging.error("Fixed failure")\n')

    _guard().assert_review_outcome(starting, current, outcome="metadata")


@pytest.mark.parametrize(
    ("starting_method", "current_method"),
    [
        ("error", "warning"),
        ("error", "debug"),
        ("warning", "error"),
        ("info", "debug"),
        ("critical", "error"),
        ("success", "info"),
        ("trace", "debug"),
    ],
)
def test_guard_metadata_rejects_severity_change(
    starting_method: str, current_method: str
) -> None:
    starting = _single_call(
        f'logger.{starting_method}(f"Legacy failure: {{private}}")\n'
    )
    current = _single_call(f'logger.{current_method}("Fixed failure")\n')

    with pytest.raises(AssertionError, match="preserve diagnostic severity"):
        _guard().assert_review_outcome(starting, current, outcome="metadata")


def test_guard_metadata_rejects_log_level_change() -> None:
    starting = _single_call('logger.log("ERROR", f"Legacy failure: {private}")\n')
    current = _single_call('logger.log("WARNING", "Fixed failure")\n')

    with pytest.raises(AssertionError, match="preserve log level"):
        _guard().assert_review_outcome(starting, current, outcome="metadata")


def test_guard_records_and_rejects_bare_name_message() -> None:
    call = _single_call("logger.info(message)\n")

    assert call.event == ""
    assert call.message_shape == _message_shape("message")
    assert call.expressions == ("message",)
    with pytest.raises(AssertionError, match="constant string first argument"):
        _guard().assert_review_outcome(call, call, outcome="metadata")


def test_guard_records_and_rejects_percent_formatted_message() -> None:
    call = _single_call('logger.warning("failed: %s" % error)\n')

    assert call.event == "failed: %s"
    assert call.message_shape == _message_shape('"failed: %s" % error')
    assert call.expressions == ("error",)
    with pytest.raises(AssertionError, match="constant string first argument"):
        _guard().assert_review_outcome(call, call, outcome="metadata")


def test_guard_records_and_rejects_dot_format_message() -> None:
    call = _single_call('logger.error("failed: {}".format(error_detail))\n')

    assert call.event == "failed: {}"
    assert call.message_shape == _message_shape('"failed: {}".format(error_detail)')
    assert call.expressions == ("error_detail",)
    with pytest.raises(AssertionError, match="constant string first argument"):
        _guard().assert_review_outcome(call, call, outcome="metadata")


def test_guard_records_and_rejects_concatenated_message() -> None:
    call = _single_call('logger.debug("result: " + result_text)\n')

    assert call.event == "result: "
    assert call.message_shape == _message_shape('"result: " + result_text')
    assert call.expressions == ("result_text",)
    with pytest.raises(AssertionError, match="constant string first argument"):
        _guard().assert_review_outcome(call, call, outcome="metadata")


def test_guard_rejects_exception_and_traceback_capture() -> None:
    source = """
logger.exception("operation failed")
logger.error("operation failed", exc_info=True)
logger.warning("operation failed", stack_info=True)
logger.opt(exception=error).warning("operation failed")
logger.opt(exception=False).warning("ordinary failure")
logger.error("ordinary failure", exc_info=False, stack_info=None)
"""
    calls = _guard().discover_diagnostic_calls(source, module="synthetic.py")

    assert [call.captures_exception for call in calls] == [
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    _guard().assert_review_outcome(calls[0], calls[0], outcome="pending")
    for outcome in ("frozen", "metadata"):
        with pytest.raises(
            AssertionError, match="must not capture exception or traceback"
        ):
            _guard().assert_review_outcome(calls[0], calls[0], outcome=outcome)


def test_manifest_boundary_tracks_reconciled_checked_and_generated_baselines() -> None:
    boundary = _review_fixture()["manifest_boundary"]
    checked_sha256 = boundary["checked_normalized_inventory_sha256"]
    generated_sha256 = boundary["origin_dev_generated_normalized_inventory_sha256"]

    assert checked_sha256 == generated_sha256
    for digest in (checked_sha256, generated_sha256):
        assert isinstance(digest, str)
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)


def test_manifest_boundary_changes_only_summarization_owner_diagnostics() -> None:
    fixture = _review_fixture()
    boundary = fixture["manifest_boundary"]
    assert isinstance(boundary, dict)
    expected_owned_entries = boundary["owned_entries"]
    assert isinstance(expected_owned_entries, list)
    owned_paths = {
        entry["path"] for entry in expected_owned_entries if isinstance(entry, dict)
    }
    assert owned_paths == set(MODULE_COUNTS)

    checked_inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    generated_inventory = diagnostic_inventory.build_inventory()
    inventories = {
        "checked": (
            checked_inventory,
            boundary["checked_normalized_inventory_sha256"],
        ),
        "generated": (
            generated_inventory,
            boundary["origin_dev_generated_normalized_inventory_sha256"],
        ),
    }
    owner_maps: dict[str, dict[str, dict[str, object]]] = {}
    for name, (inventory, expected_sha256) in inventories.items():
        _assert_task_492_summary(inventory, inventory_name=name)
        assert (
            _canonical_sha256(_normalized_inventory_projection(inventory, owned_paths))
            == expected_sha256
        ), f"{name} inventory changed outside the two summarization owners"

        owners = inventory["owners"]
        assert isinstance(owners, list)
        owner_map: dict[str, dict[str, object]] = {}
        for owner in owners:
            assert isinstance(owner, dict)
            path = owner.get("path")
            assert isinstance(path, str)
            assert path not in owner_map, f"duplicate diagnostic owner path: {path}"
            owner_map[path] = owner
        owner_maps[name] = owner_map
        owned_entries = [
            {key: owner_map[path][key] for key in ("path", "owner", "reason")}
            for path in sorted(owned_paths)
        ]
        assert owned_entries == expected_owned_entries
        for path in owned_paths:
            assert set(owner_map[path]) == {
                "path",
                "owner",
                "reason",
                "call_count",
                "diagnostic_digest",
            }
            call_count = owner_map[path]["call_count"]
            assert type(call_count) is int and call_count >= 0
            diagnostic_digest = owner_map[path]["diagnostic_digest"]
            assert (
                isinstance(diagnostic_digest, str)
                and len(diagnostic_digest) == 20
                and all(
                    character in "0123456789abcdef" for character in diagnostic_digest
                )
            ), f"{name} owned diagnostic digest has invalid schema"

    mutable_fields = {"call_count", "diagnostic_digest"}
    for path in owned_paths:
        checked_owner = owner_maps["checked"][path]
        generated_owner = owner_maps["generated"][path]
        changed_fields = {
            key
            for key in checked_owner.keys() | generated_owner.keys()
            if checked_owner.get(key) != generated_owner.get(key)
        }
        assert changed_fields <= mutable_fields
        assert not changed_fields, (
            f"checked summarization owner must match generated inventory: {path}"
        )

    sites = _ledger_sites()
    deleted_by_module = Counter(
        site["module"] for site in sites if site["outcome"] == "deleted"
    )
    assert deleted_by_module == {
        "tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py": 13,
        "tldw_chatbook/LLM_Calls/Summarization_General_Lib.py": 10,
    }
    for path, starting_count in MODULE_COUNTS.items():
        assert owner_maps["generated"][path]["call_count"] == (
            starting_count - deleted_by_module[path]
        )


def _run_manifest_boundary_mutant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    inventory: dict[str, object],
) -> None:
    inventory_path = tmp_path / "production-diagnostic-inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "INVENTORY_PATH", inventory_path)
    monkeypatch.setattr(
        diagnostic_inventory,
        "build_inventory",
        lambda: copy.deepcopy(inventory),
    )
    test_manifest_boundary_changes_only_summarization_owner_diagnostics()


def test_manifest_boundary_rejects_unknown_top_level_sections(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    inventory["unreviewed_section"] = {"private_value": "must-not-be-ignored"}

    with pytest.raises(AssertionError, match="outside the two summarization owners"):
        _run_manifest_boundary_mutant(monkeypatch, tmp_path, inventory)


def test_manifest_boundary_rejects_new_generated_origin_dev_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checked_inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    generated_inventory = diagnostic_inventory.build_inventory()
    generated_inventory["summary"]["owner_files"] += 1
    inventory_path = tmp_path / "production-diagnostic-inventory.json"
    inventory_path.write_text(json.dumps(checked_inventory), encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "INVENTORY_PATH", inventory_path)
    monkeypatch.setattr(
        diagnostic_inventory,
        "build_inventory",
        lambda: copy.deepcopy(generated_inventory),
    )

    with pytest.raises(AssertionError, match="outside the two summarization owners"):
        test_manifest_boundary_changes_only_summarization_owner_diagnostics()


def test_manifest_boundary_rejects_forged_task_492_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    inventory["summary"]["task_492_calls"] = 999_999

    with pytest.raises(AssertionError, match="TASK-492 summary"):
        _run_manifest_boundary_mutant(monkeypatch, tmp_path, inventory)


def test_manifest_boundary_rejects_owned_digest_schema_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    owned_entry = next(
        owner
        for owner in inventory["owners"]
        if owner["path"] == "tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py"
    )
    owned_entry["diagnostic_digest"] = 123

    with pytest.raises(AssertionError, match="diagnostic digest"):
        _run_manifest_boundary_mutant(monkeypatch, tmp_path, inventory)


def test_manifest_boundary_rejects_unreconciled_owned_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checked_inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    owned_entry = next(
        owner
        for owner in checked_inventory["owners"]
        if owner["path"] == "tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py"
    )
    owned_entry["diagnostic_digest"] = "0" * 20
    inventory_path = tmp_path / "production-diagnostic-inventory.json"
    inventory_path.write_text(json.dumps(checked_inventory), encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "INVENTORY_PATH", inventory_path)

    with pytest.raises(AssertionError, match="must match generated inventory"):
        test_manifest_boundary_changes_only_summarization_owner_diagnostics()


def test_ledger_retains_all_523_starting_sites() -> None:
    sites = _ledger_sites()

    assert len(sites) == 523
    assert len({site["site_id"] for site in sites}) == 523
    assert Counter(site["starting_classification"] for site in sites) == {
        "private": 200,
        "reviewed_safe": 323,
    }
    assert Counter(site["module"] for site in sites) == MODULE_COUNTS
    assert (
        Counter(
            site["group"]
            for site in sites
            if site["starting_classification"] == "private"
        )
        == PRIVATE_GROUP_COUNTS
    )

    identities = [_call_from_record(site, "starting").identity for site in sites]
    assert len(set(identities)) == 523
    encoded = json.dumps(
        _starting_projection(sites),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == STARTING_PROJECTION_SHA256


def test_ledger_private_categories_match_authoritative_inventory() -> None:
    _assert_private_category_matrix(_ledger_sites())


def test_ledger_current_state_matches_sources() -> None:
    sites = _ledger_sites()
    _assert_ledger_lifecycle(sites)

    discovered = []
    for module in MODULE_COUNTS:
        source = (REPO_ROOT / module).read_text(encoding="utf-8")
        discovered.extend(_guard().discover_diagnostic_calls(source, module=module))
    discovered_by_identity = {call.identity: call for call in discovered}
    assert len(discovered_by_identity) == len(discovered), (
        "source contains duplicate diagnostic identities"
    )

    declared_by_identity = {}
    for site in sites:
        starting = _call_from_record(site, "starting")
        if site["outcome"] == "deleted":
            assert site["current"] is None
            assert starting.identity not in discovered_by_identity, (
                f"deleted diagnostic still exists: {site['site_id']}"
            )
            continue

        current = _call_from_record(site, "current")
        assert current.identity not in declared_by_identity, (
            f"duplicate ledger identity: {site['site_id']}"
        )
        declared_by_identity[current.identity] = (site, starting, current)

    assert discovered_by_identity.keys() == declared_by_identity.keys(), (
        "summarization diagnostic calls were added, deleted, or changed "
        "without ledger review"
    )
    for identity, actual in discovered_by_identity.items():
        site, starting, current = declared_by_identity[identity]
        assert actual == current, (
            f"current diagnostic record changed: {site['site_id']}"
        )
        _guard().assert_review_outcome(starting, actual, outcome=site["outcome"])


def test_ledger_rejects_reviewed_safe_outcome_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sites = copy.deepcopy(_ledger_sites())
    reviewed_safe = next(
        site
        for site in sites
        if site["starting_classification"] == "reviewed_safe"
        and site["starting"]["message_shape"].startswith("Constant(value=")
    )
    reviewed_safe["outcome"] = "metadata"
    monkeypatch.setattr(sys.modules[__name__], "_ledger_sites", lambda: sites)

    with pytest.raises(
        AssertionError, match="reviewed_safe diagnostic must remain frozen"
    ):
        test_ledger_current_state_matches_sources()


def test_ledger_rejects_private_outcome_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sites = copy.deepcopy(_ledger_sites())
    private = next(
        site
        for site in sites
        if site["starting_classification"] == "private"
        and not site["starting"]["captures_exception"]
    )
    private["outcome"] = "frozen"
    monkeypatch.setattr(sys.modules[__name__], "_ledger_sites", lambda: sites)

    with pytest.raises(
        AssertionError,
        match="private diagnostic has unapproved lifecycle outcome",
    ):
        test_ledger_current_state_matches_sources()


def test_ledger_lifecycle_uses_exact_approved_outcomes() -> None:
    approved = {
        "reviewed_safe": {"frozen"},
        "private": {"pending", "metadata", "deleted"},
    }
    all_outcomes = {"pending", "frozen", "metadata", "deleted"}

    for classification, allowed in approved.items():
        for outcome in allowed:
            site = {
                "site_id": "synthetic-lifecycle-site",
                "starting_classification": classification,
                "outcome": outcome,
            }
            if outcome == "deleted":
                site["deletion_reason"] = "redundant legacy diagnostic removed"
            _assert_ledger_lifecycle([site])
        for outcome in all_outcomes - allowed:
            with pytest.raises(AssertionError):
                _assert_ledger_lifecycle(
                    [
                        {
                            "site_id": "synthetic-lifecycle-site",
                            "starting_classification": classification,
                            "outcome": outcome,
                        }
                    ]
                )


def test_ledger_deletion_reason_contract() -> None:
    deleted = {
        "site_id": "synthetic-deleted-site",
        "starting_classification": "private",
        "outcome": "deleted",
    }
    with pytest.raises(AssertionError, match="non-empty deletion_reason"):
        _assert_ledger_lifecycle([deleted])
    with pytest.raises(AssertionError, match="non-empty deletion_reason"):
        _assert_ledger_lifecycle([{**deleted, "deletion_reason": "   "}])

    _assert_ledger_lifecycle(
        [{**deleted, "deletion_reason": "redundant legacy diagnostic removed"}]
    )

    pending_with_reason = {
        "site_id": "synthetic-pending-site",
        "starting_classification": "private",
        "outcome": "pending",
        "deletion_reason": "not actually deleted",
    }
    with pytest.raises(AssertionError, match="only deleted diagnostics"):
        _assert_ledger_lifecycle([pending_with_reason])


def test_no_pending_local_core_sites() -> None:
    pending = [
        site
        for site in _ledger_sites()
        if site["group"] == "local_core" and site["outcome"] == "pending"
    ]

    assert not pending, (
        f"local_core has {len(pending)} pending private diagnostics: "
        f"{[site['site_id'] for site in pending]}"
    )


def test_no_pending_local_adapters_sites() -> None:
    pending = [
        site
        for site in _ledger_sites()
        if site["group"] == "local_adapters" and site["outcome"] == "pending"
    ]

    assert not pending, (
        f"local_adapters has {len(pending)} pending private diagnostics: "
        f"{[site['site_id'] for site in pending]}"
    )


def test_no_pending_local_vllm_ollama_sites() -> None:
    pending = [
        site
        for site in _ledger_sites()
        if site["group"] == "local_vllm_ollama" and site["outcome"] == "pending"
    ]

    assert not pending, (
        f"local_vllm_ollama has {len(pending)} pending private diagnostics: "
        f"{[site['site_id'] for site in pending]}"
    )


def test_no_pending_local_custom_sites() -> None:
    pending = [
        site
        for site in _ledger_sites()
        if site["group"] == "local_custom"
        and site["starting_classification"] == "private"
        and site["outcome"] == "pending"
    ]

    assert not pending, (
        f"local_custom has {len(pending)} pending private diagnostics: "
        f"{[site['site_id'] for site in pending]}"
    )


def test_local_custom_module_has_no_pending_private_sites() -> None:
    local = [
        site
        for site in _ledger_sites()
        if site["module"].endswith("Local_Summarization_Lib.py")
    ]

    assert len(local) == 242
    assert (
        sum(site["starting_classification"] == "reviewed_safe" for site in local) == 142
    )
    assert sum(site["outcome"] == "frozen" for site in local) == 142
    assert not [site for site in local if site["outcome"] == "pending"]
    assert sum(site["outcome"] in {"metadata", "deleted"} for site in local) == 100


@pytest.mark.parametrize(
    "case",
    RUNTIME_SENTINEL_CASES,
    ids=lambda case: f"{case.module}-{case.category}",
)
def test_runtime_sentinel_hides_private_value(
    case: RuntimeSentinelCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _capture_stdlib_and_loguru(caplog) as captured:
        result = case.invoke(monkeypatch, tmp_path)
        case.assert_contract(result)

    assert case.canary not in captured.text
    assert case.expected_event in captured.text


def test_local_llm_success_contract_hides_input_canary(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _capture_stdlib_and_loguru(caplog) as captured:
        result = _invoke_local_input(monkeypatch)

    _assert_fixed_summary(result)
    assert LOCAL_INPUT_CANARY not in captured.text


def test_local_llm_malformed_stream_contract_hides_response_canary(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _capture_stdlib_and_loguru(caplog) as captured:
        result = _invoke_local_response(monkeypatch)

    _assert_empty_stream(result)
    assert LOCAL_RESPONSE_CANARY not in captured.text
    assert "Local LLM: Failed to decode streamed JSON; line_length=" in captured.text


def test_local_core_llama_malformed_stream_logs_only_safe_length(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = _FakeResponse(
        lines=(f"data: {{{LOCAL_RESPONSE_CANARY}".encode(), b"data: [DONE]")
    )
    monkeypatch.setattr(local_summarization, "load_settings", _local_settings)
    monkeypatch.setattr(
        local_summarization,
        "create_default_session",
        lambda: _FakeSession(response),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        generator = local_summarization.summarize_with_llama(
            "fixed input",
            "fixed prompt",
            api_key="fixed-llama-key",
            streaming=True,
        )
        result = list(generator)

    assert result == []
    assert response.iter_lines_started is True
    assert LOCAL_RESPONSE_CANARY not in captured.text
    assert "Llama: Failed to decode streamed JSON; line_length=" in captured.text


def test_local_llm_exception_contract_hides_exception_canary(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _capture_stdlib_and_loguru(caplog) as captured:
        result = _invoke_local_exception(monkeypatch)

    _assert_local_exception_contract(result)
    assert LOCAL_EXCEPTION_CANARY not in captured.text


def test_local_core_llama_success_hides_prompt_and_endpoint_canaries(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = _FakeResponse(json_data={"content": "  fixed llama summary  "})
    monkeypatch.setattr(
        local_summarization,
        "load_settings",
        lambda: _local_settings(llama_endpoint=LOCAL_PATH_CANARY),
    )
    monkeypatch.setattr(
        local_summarization,
        "create_default_session",
        lambda: _FakeSession(response),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = local_summarization.summarize_with_llama(
            "fixed input",
            LOCAL_PROMPT_CANARY,
            api_key="fixed-llama-key",
            system_message="fixed system message",
        )

    _assert_fixed_llama_summary(result)
    assert LOCAL_PROMPT_CANARY not in captured.text
    assert LOCAL_PATH_CANARY not in captured.text


def test_local_core_llama_accepts_non_string_system_message_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeResponse(json_data={"content": "  fixed llama summary  "})
    monkeypatch.setattr(local_summarization, "load_settings", _local_settings)
    monkeypatch.setattr(
        local_summarization,
        "create_default_session",
        lambda: _FakeSession(response),
    )

    result = local_summarization.summarize_with_llama(
        "fixed input",
        "fixed prompt",
        api_key="fixed-llama-key",
        system_message=object(),
    )

    _assert_fixed_llama_summary(result)


def test_local_core_kobold_missing_key_error_contract_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def transport_must_not_run() -> _FakeSession:
        raise AssertionError("transport invoked before missing-key failure")

    settings = _local_settings()
    # task-17383: the credential lives in the section the loader builds now;
    # the intent is unchanged -- an ABSENT key still hits the historical
    # TypeError contract, distinct from a configured-but-blank one.
    settings["kobold_api"]["api_key"] = None
    monkeypatch.setattr(local_summarization, "load_settings", lambda: settings)
    monkeypatch.setattr(
        local_summarization,
        "create_default_session",
        transport_must_not_run,
    )

    generator = local_summarization.summarize_with_kobold(
        "fixed input",
        None,
        "fixed prompt",
    )
    result = _consume_generator(generator)

    assert result == (
        [],
        "Kobold: Error occurred while processing summary with Kobold: "
        "'NoneType' object is not subscriptable",
    )


def test_local_core_kobold_stream_fully_consumed_without_private_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = _FakeResponse(
        lines=(f"data: {{{LOCAL_RESPONSE_CANARY}".encode(), b"data: [DONE]")
    )
    monkeypatch.setattr(local_summarization, "load_settings", _local_settings)
    monkeypatch.setattr(
        local_summarization,
        "create_default_session",
        lambda: _FakeSession(response),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        generator = local_summarization.summarize_with_kobold(
            "fixed input",
            "fixed-kobold-key",
            "fixed prompt",
            streaming=True,
        )
        result = _consume_generator(generator)

    assert result == ([], None)
    assert LOCAL_RESPONSE_CANARY not in captured.text
    assert (
        "Kobold: Failed to decode streamed JSON; exception_type=JSONDecodeError"
        in captured.text
    )


def test_oobabooga_success_contract_hides_prompt_credential_endpoint_and_response(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = _FakeResponse(
        json_data={
            "choices": [{"message": {"content": f"  {OOBABOOGA_RESPONSE_CANARY}  "}}]
        }
    )
    monkeypatch.setattr(
        local_summarization,
        "load_settings",
        lambda: _local_adapter_settings(oobabooga_endpoint=OOBABOOGA_ENDPOINT_CANARY),
    )
    calls = _install_signature_bound_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = local_summarization.summarize_with_oobabooga(
            "fixed input",
            OOBABOOGA_CREDENTIAL_CANARY,
            OOBABOOGA_PROMPT_CANARY,
            system_message="fixed system message",
        )

    assert result == OOBABOOGA_RESPONSE_CANARY
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (OOBABOOGA_ENDPOINT_CANARY,)
    assert kwargs["headers"] == {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {OOBABOOGA_CREDENTIAL_CANARY}",
    }
    assert kwargs["json"]["messages"][1]["content"].endswith(OOBABOOGA_PROMPT_CANARY)
    assert OOBABOOGA_PROMPT_CANARY not in captured.text
    assert OOBABOOGA_CREDENTIAL_CANARY not in captured.text
    assert OOBABOOGA_ENDPOINT_CANARY not in captured.text
    assert OOBABOOGA_RESPONSE_CANARY not in captured.text
    assert "Oobabooga: Credential configured" in captured.text
    assert "Oobabooga: API endpoint configured" in captured.text
    assert "Oobabooga: Prompt prepared; character_count=" in captured.text
    assert "Ooba API: Summarization successful" in captured.text


def test_oobabooga_error_response_contract_hides_response_body(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = _FakeResponse(status_code=503, text=OOBABOOGA_RESPONSE_CANARY)
    monkeypatch.setattr(local_summarization, "load_settings", _local_adapter_settings)
    _install_signature_bound_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = local_summarization.summarize_with_oobabooga(
            "fixed input",
            "fixed-oobabooga-key",
            "fixed prompt",
            api_url="http://fixed-oobabooga.invalid/v1/chat/completions",
        )

    assert result == "Ooba API: Failed to process summary. Status code: 503"
    assert OOBABOOGA_RESPONSE_CANARY not in captured.text
    assert "Ooba API: Error response received; status_code=503" in captured.text


def test_oobabooga_request_exception_contract_hides_message_and_traceback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(local_summarization, "load_settings", _local_adapter_settings)
    _install_signature_bound_session_post(
        monkeypatch,
        local_summarization.requests.RequestException(OOBABOOGA_EXCEPTION_CANARY),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = local_summarization.summarize_with_oobabooga(
            "fixed input",
            "fixed-oobabooga-key",
            "fixed prompt",
            api_url="http://fixed-oobabooga.invalid/v1/chat/completions",
        )

    assert result == (
        "Ooba API: Error making API request: " + OOBABOOGA_EXCEPTION_CANARY
    )
    assert OOBABOOGA_EXCEPTION_CANARY not in captured.text
    assert (
        "Ooba API: API request failed; exception_type=RequestException" in captured.text
    )
    assert all(record.exc_info is None for record in caplog.records)


def test_oobabooga_malformed_stream_contract_fully_consumes_and_hides_line(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    private_line = f'data: {{"content":"{OOBABOOGA_RESPONSE_CANARY}"'.encode()
    response = _FakeResponse(lines=(private_line, b"data: [DONE]"))
    monkeypatch.setattr(local_summarization, "load_settings", _local_adapter_settings)
    calls = _install_signature_bound_session_post(monkeypatch, response)

    generator = local_summarization.summarize_with_oobabooga(
        "fixed input",
        "fixed-oobabooga-key",
        "fixed prompt",
        api_url="http://fixed-oobabooga.invalid/v1/chat/completions",
        streaming=True,
    )
    assert len(calls) == 1

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = _consume_generator(generator)

    assert result == ([], None)
    assert OOBABOOGA_RESPONSE_CANARY not in captured.text
    assert "Oobabooga: Failed to decode streamed JSON; exception_type=" in (
        captured.text
    )
    assert f"line_length={len(private_line) - len(b'data: ')}" in captured.text


def test_tabby_malformed_stream_contract_hides_input_credential_and_lines(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    malformed_line = f'data: {{"content":"{TABBY_STREAM_CANARY}"'.encode()
    non_data_line = f"event: {TABBY_STREAM_CANARY}".encode()
    response = _FakeResponse(lines=(malformed_line, non_data_line, b"data: [DONE]"))
    monkeypatch.setattr(local_summarization, "load_settings", _local_adapter_settings)
    calls = _install_signature_bound_session_post(monkeypatch, response)

    generator = local_summarization.summarize_with_tabbyapi(
        TABBY_INPUT_CANARY,
        "fixed prompt",
        api_key=TABBY_CREDENTIAL_CANARY,
        streaming=True,
    )
    assert calls == []

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = _consume_generator(generator)

    assert result == ([], None)
    assert len(calls) == 1
    assert TABBY_INPUT_CANARY not in captured.text
    assert TABBY_CREDENTIAL_CANARY not in captured.text
    assert TABBY_STREAM_CANARY not in captured.text
    assert "TabbyAPI: Credential state resolved" in captured.text
    assert "TabbyAPI: Input received" in captured.text
    assert "TabbyAPI: Failed to parse streamed JSON; exception_type=" in (captured.text)
    assert "TabbyAPI: Ignored non-data stream line; line_length=" in captured.text


def test_tabby_request_exception_contract_hides_message(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(local_summarization, "load_settings", _local_adapter_settings)
    _install_signature_bound_session_post(
        monkeypatch,
        local_summarization.requests.RequestException(TABBY_EXCEPTION_CANARY),
    )

    generator = local_summarization.summarize_with_tabbyapi(
        "fixed input",
        "fixed prompt",
        api_key="fixed-tabby-key",
        streaming=True,
    )
    with _capture_stdlib_and_loguru(caplog) as captured:
        result = _consume_generator(generator)

    assert result == (
        [f"Error summarizing with TabbyAPI: {TABBY_EXCEPTION_CANARY}"],
        None,
    )
    assert TABBY_EXCEPTION_CANARY not in captured.text
    assert (
        "TabbyAPI: Streaming request failed; exception_type=RequestException"
        in captured.text
    )


def test_tabby_missing_credential_error_contract_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _local_adapter_settings(tabby_key=None)
    monkeypatch.setattr(local_summarization, "load_settings", lambda: settings)

    def transport_must_not_run() -> object:
        raise AssertionError("transport invoked before missing-key failure")

    monkeypatch.setattr(
        local_summarization,
        "create_default_session",
        transport_must_not_run,
    )

    generator = local_summarization.summarize_with_tabbyapi(
        "fixed input",
        "fixed prompt",
    )
    result = _consume_generator(generator)

    assert result == (
        [],
        "TabbyAPI: Unexpected error in summarization process: "
        "'NoneType' object is not subscriptable",
    )


def test_tabby_empty_configured_credential_reports_resolved_state_truthfully(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = _FakeResponse(
        json_data={
            "id": "fixed-id",
            "choices": [{"message": {"content": "fixed tabby summary"}}],
            "created": 1,
            "model": "fixed-tabby-model",
            "object": "chat.completion",
            "usage": {},
        }
    )
    monkeypatch.setattr(
        local_summarization,
        "load_settings",
        lambda: _local_adapter_settings(tabby_key=""),
    )
    calls = _install_signature_bound_session_post(monkeypatch, response)

    generator = local_summarization.summarize_with_tabbyapi(
        "fixed input",
        "fixed prompt",
    )
    assert calls == []

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = _consume_generator(generator)

    assert result == ([], "fixed tabby summary")
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("http://tabby.invalid/v1/chat/completions",)
    assert kwargs["headers"] == {"Content-Type": "application/json"}
    assert "Authorization" not in kwargs["headers"]
    assert "TabbyAPI: No API key found in config file" in captured.text
    assert "TabbyAPI: Credential state resolved" in captured.text
    assert "TabbyAPI: Credential configured" not in captured.text


def test_vllm_success_hides_input_prompt_and_response(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = _FakeResponse(
        json_data={"choices": [{"message": {"content": f"  {VLLM_RESPONSE_CANARY}  "}}]}
    )
    settings_calls = _install_signature_bound_settings(
        monkeypatch,
        _vllm_settings(),
    )
    transport_calls = _install_signature_bound_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = local_summarization.summarize_with_vllm(
            None,
            VLLM_INPUT_CANARY,
            VLLM_PROMPT_CANARY,
            system_message="fixed system message",
        )

    assert result == VLLM_RESPONSE_CANARY
    assert settings_calls
    assert len(transport_calls) == 1
    args, kwargs = transport_calls[0]
    assert args == ("http://vllm.invalid/v1/chat/completions",)
    assert kwargs["headers"]["Authorization"] == "Bearer fixed-vllm-key"
    assert kwargs["json"]["messages"][1]["content"] == (
        f"{VLLM_INPUT_CANARY} \n\n\n\n{VLLM_PROMPT_CANARY}"
    )
    for canary in (
        VLLM_INPUT_CANARY,
        VLLM_PROMPT_CANARY,
        VLLM_RESPONSE_CANARY,
    ):
        assert canary not in captured.text
    assert "vLLM Summarize: Credential config lookup completed" in captured.text
    assert "vLLM Summarize: Raw input received" in captured.text
    assert "vLLM Summarize: Input processing completed" in captured.text
    assert "vLLM Summarize: Text extraction completed" in captured.text
    assert "vLLM Summarize: Custom prompt received" in captured.text
    assert "vLLM Summarization: Summary produced; character_count=" in captured.text


def test_vllm_credential_fragments_are_not_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = _FakeResponse(
        json_data={"choices": [{"message": {"content": "fixed summary"}}]}
    )
    settings_calls = _install_signature_bound_settings(
        monkeypatch,
        _vllm_settings(api_key=VLLM_CREDENTIAL_CANARY),
    )
    transport_calls = _install_signature_bound_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = local_summarization.summarize_with_vllm(
            None,
            "fixed input",
            "fixed prompt",
            system_message="fixed system message",
        )

    assert result == "fixed summary"
    assert settings_calls
    assert len(transport_calls) == 1
    args, kwargs = transport_calls[0]
    assert args == ("http://vllm.invalid/v1/chat/completions",)
    assert kwargs["headers"]["Authorization"] == f"Bearer {VLLM_CREDENTIAL_CANARY}"
    assert kwargs["json"]["messages"][1]["content"] == (
        "fixed input \n\n\n\nfixed prompt"
    )
    assert VLLM_CREDENTIAL_CANARY[:5] not in captured.text
    assert VLLM_CREDENTIAL_CANARY[-5:] not in captured.text
    assert "vLLM Summarize: Credential config lookup completed" in captured.text


def test_vllm_empty_configured_credential_reports_state_neutrally(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = _FakeResponse(
        json_data={"choices": [{"message": {"content": "fixed empty-key summary"}}]}
    )
    settings_calls = _install_signature_bound_settings(
        monkeypatch,
        _vllm_settings(api_key=""),
    )
    transport_calls = _install_signature_bound_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = local_summarization.summarize_with_vllm(
            "",
            "fixed input",
            "fixed prompt",
            system_message="fixed system message",
        )

    assert result == "fixed empty-key summary"
    assert settings_calls
    assert len(transport_calls) == 1
    args, kwargs = transport_calls[0]
    assert args == ("http://vllm.invalid/v1/chat/completions",)
    assert kwargs["headers"]["Authorization"] == "Bearer "
    assert kwargs["json"]["messages"][1]["content"] == (
        "fixed input \n\n\n\nfixed prompt"
    )
    misleading_events = {
        event
        for event in (
            "vLLM Summarize: Credential loaded from config",
            "vLLM Summarize: Credential applied to request",
        )
        if event in captured.text
    }
    assert not misleading_events, (
        f"empty credential emitted misleading state: {sorted(misleading_events)}"
    )
    assert "vLLM Summarize: Credential config lookup completed" in captured.text
    assert "vLLM Summarize: Authorization header prepared" in captured.text


def test_vllm_error_response_hides_body_and_preserves_status_contract(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = _FakeResponse(status_code=503, text=VLLM_RESPONSE_CANARY)
    _install_signature_bound_settings(monkeypatch, _vllm_settings())
    _install_signature_bound_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = local_summarization.summarize_with_vllm(
            None,
            "fixed input",
            "fixed prompt",
        )

    assert result == ("vLLM Summarization: Failed to process summary. Status code: 503")
    assert VLLM_RESPONSE_CANARY not in captured.text
    assert (
        "vLLM Summarization: Summarization failed with status code 503" in captured.text
    )


def test_vllm_malformed_stream_is_lazy_fully_consumed_and_hides_line(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = _FakeResponse(
        lines=(
            b'data: {"choices":[{"delta":{"content":"fixed chunk"}}]}',
            f'data: {{"content":"{VLLM_STREAM_CANARY}"'.encode(),
            b"data: [DONE]",
        )
    )
    _install_signature_bound_settings(monkeypatch, _vllm_settings())
    transport_calls = _install_signature_bound_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        generator = local_summarization.summarize_with_vllm(
            None,
            "fixed input",
            "fixed prompt",
            streaming=True,
        )
        assert len(transport_calls) == 1
        assert response.iter_lines_started is False
        result = _consume_generator(generator)

    assert result == (["fixed chunk"], None)
    assert response.iter_lines_started is True
    assert VLLM_STREAM_CANARY not in captured.text
    assert "vLLM Summarize: Failed to decode streamed JSON; line_length=" in (
        captured.text
    )


def test_vllm_request_exception_hides_message_and_traceback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_settings(monkeypatch, _vllm_settings())
    _install_signature_bound_session_post(
        monkeypatch,
        local_summarization.requests.RequestException(VLLM_EXCEPTION_CANARY),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = local_summarization.summarize_with_vllm(
            None,
            "fixed input",
            "fixed prompt",
        )

    assert result == (
        "vLLM Summarization: Error making API request: " + VLLM_EXCEPTION_CANARY
    )
    assert VLLM_EXCEPTION_CANARY not in captured.text
    assert (
        "vLLM Summarization: API request failed; exception_type=RequestException"
        in captured.text
    )
    assert all(record.exc_info is None for record in caplog.records)


def test_ollama_success_hides_prompt_and_response(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = _FakeResponse(json_data={"response": f"  {OLLAMA_RESPONSE_CANARY}  "})
    settings_calls = _install_signature_bound_settings(monkeypatch, _ollama_settings())
    transport_calls = _install_signature_bound_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = local_summarization.summarize_with_ollama(
            "fixed input",
            OLLAMA_PROMPT_CANARY,
        )

    assert result == OLLAMA_RESPONSE_CANARY
    assert settings_calls == [((), {})]
    assert len(transport_calls) == 1
    args, kwargs = transport_calls[0]
    assert args == ("http://ollama.invalid/api/chat",)
    assert kwargs["json"]["messages"][1]["content"] == (
        f"{OLLAMA_PROMPT_CANARY}\n\nfixed input"
    )
    assert OLLAMA_PROMPT_CANARY not in captured.text
    assert OLLAMA_RESPONSE_CANARY not in captured.text
    assert "Ollama: Summarization prompt prepared; character_count=" in captured.text
    assert "Ollama: Response parsed" in captured.text


def test_ollama_malformed_stream_is_lazy_fully_consumed_and_hides_line(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = _FakeResponse(
        lines=(
            b'{"response":"fixed chunk","done":false}',
            f'{{"response":"{OLLAMA_STREAM_CANARY}"'.encode(),
            b'{"done":true}',
        )
    )
    _install_signature_bound_settings(monkeypatch, _ollama_settings())
    transport_calls = _install_signature_bound_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        generator = local_summarization.summarize_with_ollama(
            "fixed input",
            "fixed prompt",
            streaming=True,
        )
        assert len(transport_calls) == 1
        assert response.iter_lines_started is False
        result = _consume_generator(generator)

    assert result == (["fixed chunk"], None)
    assert response.iter_lines_started is True
    assert OLLAMA_STREAM_CANARY not in captured.text
    assert "Ollama: Failed to decode streamed JSON; line_length=" in captured.text


def test_ollama_config_exception_hides_message_and_preserves_error_contract(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings_calls = _install_signature_bound_settings(
        monkeypatch,
        RuntimeError(OLLAMA_EXCEPTION_CANARY),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = local_summarization.summarize_with_ollama(
            "fixed input",
            "fixed prompt",
        )

    assert settings_calls == [((), {})]
    assert result == f"Ollama: Error loading config: {OLLAMA_EXCEPTION_CANARY}"
    assert OLLAMA_EXCEPTION_CANARY not in captured.text
    assert (
        "summarize_with_ollama: Config loading failed; exception_type=RuntimeError"
        in captured.text
    )
    assert all(record.exc_info is None for record in caplog.records)


def test_ollama_http_exception_hides_message_and_preserves_error_contract(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_settings(monkeypatch, _ollama_settings())
    _install_signature_bound_session_post(
        monkeypatch,
        local_summarization.requests.exceptions.HTTPError(OLLAMA_EXCEPTION_CANARY),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = local_summarization.summarize_with_ollama(
            "fixed input",
            "fixed prompt",
        )

    assert result == f"Ollama: HTTP error: {OLLAMA_EXCEPTION_CANARY}"
    assert OLLAMA_EXCEPTION_CANARY not in captured.text
    assert "Ollama: HTTP request failed; exception_type=HTTPError" in captured.text
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.parametrize(
    "variant",
    CUSTOM_OPENAI_VARIANTS,
    ids=lambda variant: variant.id,
)
def test_local_custom_openai_success_hides_input_prompt_key_endpoint_and_response(
    variant: _CustomOpenAIVariant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = _FakeResponse(
        json_data={
            "choices": [
                {"message": {"content": f"  {CUSTOM_OPENAI_RESPONSE_CANARY}  "}}
            ]
        }
    )
    settings_calls = _install_signature_bound_settings(
        monkeypatch,
        _custom_openai_settings(
            provider_1_key=CUSTOM_OPENAI_1_CREDENTIAL_CANARY,
            provider_1_url=CUSTOM_OPENAI_1_ENDPOINT_CANARY,
            provider_1_model=CUSTOM_OPENAI_1_MODEL,
            provider_2_key=CUSTOM_OPENAI_2_CREDENTIAL_CANARY,
            provider_2_url=CUSTOM_OPENAI_2_ENDPOINT_CANARY,
            provider_2_model=CUSTOM_OPENAI_2_MODEL,
        ),
    )
    transport_calls = _install_signature_bound_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = variant.summarizer(
            None,
            CUSTOM_OPENAI_INPUT_CANARY,
            CUSTOM_OPENAI_PROMPT_CANARY,
            temp=0.2,
            system_message="fixed system message",
        )

    assert result == CUSTOM_OPENAI_RESPONSE_CANARY
    assert settings_calls == [((), {})]
    assert transport_calls == [
        (
            (variant.endpoint,),
            {
                "headers": {
                    "Authorization": f"Bearer {variant.api_key}",
                    "Content-Type": "application/json",
                },
                "json": {
                    "model": variant.model,
                    "messages": [
                        {"role": "system", "content": "fixed system message"},
                        {
                            "role": "user",
                            "content": (
                                f"{CUSTOM_OPENAI_INPUT_CANARY} "
                                f"\n\n\n\n{CUSTOM_OPENAI_PROMPT_CANARY}"
                            ),
                        },
                    ],
                    "max_tokens": 64,
                    "temperature": 0.2,
                    "stream": False,
                },
            },
        )
    ]
    for canary in (
        CUSTOM_OPENAI_INPUT_CANARY,
        CUSTOM_OPENAI_PROMPT_CANARY,
        variant.api_key[:5],
        variant.api_key[-5:],
        variant.endpoint,
        CUSTOM_OPENAI_RESPONSE_CANARY,
    ):
        assert canary not in captured.text
    assert f"{variant.event_prefix}: Credential configured" in captured.text
    assert f"{variant.event_prefix}: Input received" in captured.text
    assert f"{variant.event_prefix}: Input processing completed" in captured.text
    assert f"{variant.event_prefix}: Text extraction completed" in captured.text
    assert f"{variant.event_prefix}: Prompt prepared; character_count=" in captured.text
    assert f"{variant.event_prefix}: API endpoint configured" in captured.text
    assert (
        f"{variant.event_prefix}: Chat response received; character_count="
        in captured.text
    )


@pytest.mark.parametrize(
    "variant",
    CUSTOM_OPENAI_VARIANTS,
    ids=lambda variant: variant.id,
)
def test_local_custom_openai_non_success_hides_response_body_and_preserves_contract(
    variant: _CustomOpenAIVariant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = _FakeResponse(status_code=503, text=CUSTOM_OPENAI_RESPONSE_CANARY)
    _install_signature_bound_settings(monkeypatch, _custom_openai_settings())
    transport_calls = _install_signature_bound_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = variant.summarizer(
            "fixed-custom-openai-key",
            "fixed input",
            "fixed prompt",
            temp=0.2,
            system_message="fixed system message",
        )

    assert result == "OpenAI: Failed to process chat response. Status code: 503"
    assert len(transport_calls) == 1
    assert CUSTOM_OPENAI_RESPONSE_CANARY not in captured.text
    assert (
        f"{variant.event_prefix}: Chat request failed with status code 503"
        in captured.text
    )


@pytest.mark.parametrize(
    "variant",
    CUSTOM_OPENAI_VARIANTS,
    ids=lambda variant: variant.id,
)
def test_local_custom_openai_request_exception_hides_message_and_traceback(
    variant: _CustomOpenAIVariant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_settings(monkeypatch, _custom_openai_settings())
    transport_calls = _install_signature_bound_session_post(
        monkeypatch,
        local_summarization.requests.RequestException(CUSTOM_OPENAI_EXCEPTION_CANARY),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = variant.summarizer(
            "fixed-custom-openai-key",
            "fixed input",
            "fixed prompt",
            temp=0.2,
            system_message="fixed system message",
        )

    assert result == (
        f"{variant.event_prefix}: Error making API request: "
        f"{CUSTOM_OPENAI_EXCEPTION_CANARY}"
    )
    assert len(transport_calls) == 1
    assert CUSTOM_OPENAI_EXCEPTION_CANARY not in captured.text
    assert (
        f"{variant.event_prefix}: API request failed; exception_type=RequestException"
        in captured.text
    )
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.parametrize(
    "variant",
    CUSTOM_OPENAI_VARIANTS,
    ids=lambda variant: variant.id,
)
def test_local_custom_openai_unexpected_exception_hides_message_and_traceback(
    variant: _CustomOpenAIVariant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _custom_openai_settings()
    settings["custom_openai_api"]["max_tokens"] = CUSTOM_OPENAI_EXCEPTION_CANARY
    settings["custom_openai_api_2"]["max_tokens"] = CUSTOM_OPENAI_EXCEPTION_CANARY
    _install_signature_bound_settings(monkeypatch, settings)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = variant.summarizer(
            "fixed-custom-openai-key",
            "fixed input",
            "fixed prompt",
            temp=0.2,
            system_message="fixed system message",
        )

    assert result.startswith(f"{variant.event_prefix}: Unexpected error occurred: ")
    assert CUSTOM_OPENAI_EXCEPTION_CANARY in result
    assert CUSTOM_OPENAI_EXCEPTION_CANARY not in captured.text
    assert (
        f"{variant.event_prefix}: Unexpected failure; exception_type=ValueError"
        in captured.text
    )
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.parametrize(
    "variant",
    CUSTOM_OPENAI_VARIANTS,
    ids=lambda variant: variant.id,
)
def test_local_custom_openai_malformed_stream_is_lazy_consumed_and_hides_line(
    variant: _CustomOpenAIVariant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    private_line = f'data: {{"content":"{CUSTOM_OPENAI_STREAM_CANARY}"'.encode()
    response = _FakeResponse(
        lines=(
            b'data: {"choices":[{"delta":{"content":"fixed chunk"}}]}',
            private_line,
            b"data: [DONE]",
        )
    )
    _install_signature_bound_settings(monkeypatch, _custom_openai_settings())
    transport_calls = _install_signature_bound_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        generator = variant.summarizer(
            "fixed-custom-openai-key",
            "fixed input",
            "fixed prompt",
            temp=0.2,
            system_message="fixed system message",
            streaming=True,
        )
        assert len(transport_calls) == 1
        assert response.iter_lines_started is False
        result = _consume_generator(generator)

    assert result == (["fixed chunk", "fixed chunk"], None)
    assert response.iter_lines_started is True
    assert CUSTOM_OPENAI_STREAM_CANARY not in captured.text
    assert (
        f"{variant.event_prefix}: Failed to decode streamed JSON; line_length="
        in captured.text
    )
    assert f"line_length={len(private_line) - len(b'data: ')}" in captured.text


@pytest.mark.parametrize(
    "variant",
    CUSTOM_OPENAI_VARIANTS,
    ids=lambda variant: variant.id,
)
def test_local_custom_openai_response_json_error_hides_detail_and_preserves_contract(
    variant: _CustomOpenAIVariant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    decode_error = json.JSONDecodeError(
        CUSTOM_OPENAI_EXCEPTION_CANARY,
        "private response",
        0,
    )
    response = _FakeResponse(json_data=decode_error)
    _install_signature_bound_settings(monkeypatch, _custom_openai_settings())
    _install_signature_bound_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = variant.summarizer(
            "fixed-custom-openai-key",
            "fixed input",
            "fixed prompt",
            temp=0.2,
            system_message="fixed system message",
        )

    assert result == (
        f"{variant.event_prefix}: Error decoding JSON input: {decode_error}"
    )
    assert CUSTOM_OPENAI_EXCEPTION_CANARY not in captured.text
    assert (
        f"{variant.event_prefix}: Response JSON decode failed; "
        "exception_type=JSONDecodeError" in captured.text
    )
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.parametrize(
    "variant",
    CUSTOM_OPENAI_VARIANTS,
    ids=lambda variant: variant.id,
)
def test_local_custom_openai_malformed_input_hides_parse_detail_and_still_submits(
    variant: _CustomOpenAIVariant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    malformed_input = '{"private":"' + CUSTOM_OPENAI_INPUT_CANARY
    response = _FakeResponse(
        json_data={"choices": [{"message": {"content": "fixed summary"}}]}
    )
    _install_signature_bound_settings(monkeypatch, _custom_openai_settings())
    transport_calls = _install_signature_bound_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = variant.summarizer(
            "fixed-custom-openai-key",
            malformed_input,
            "fixed prompt",
            temp=0.2,
            system_message="fixed system message",
        )

    assert result == "fixed summary"
    assert len(transport_calls) == 1
    assert transport_calls[0][1]["json"]["messages"][1]["content"].startswith(
        malformed_input
    )
    assert CUSTOM_OPENAI_INPUT_CANARY not in captured.text
    assert (
        f"{variant.event_prefix}: Input JSON parse failed; exception_type="
        in captured.text
    )


@pytest.mark.parametrize(
    "variant",
    CUSTOM_OPENAI_VARIANTS,
    ids=lambda variant: variant.id,
)
def test_local_custom_openai_missing_config_credential_contract_is_unchanged(
    variant: _CustomOpenAIVariant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _custom_openai_settings(
        provider_1_key=None,
        provider_2_key=None,
    )
    settings_calls = _install_signature_bound_settings(monkeypatch, settings)

    def transport_must_not_run() -> object:
        raise AssertionError("transport invoked before missing-key return")

    monkeypatch.setattr(
        local_summarization,
        "create_default_session",
        transport_must_not_run,
    )

    result = variant.summarizer(
        None,
        "fixed input",
        "fixed prompt",
        temp=0.2,
        system_message="fixed system message",
    )

    assert settings_calls == [((), {})]
    assert result == (
        f"{variant.event_prefix}: API Key Not Provided/Found in Config file or is empty"
    )


@pytest.mark.parametrize(
    "variant",
    CUSTOM_OPENAI_VARIANTS,
    ids=lambda variant: variant.id,
)
def test_local_custom_openai_missing_provider_section_preserves_error_contract(
    variant: _CustomOpenAIVariant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _custom_openai_settings()
    del settings[variant.settings_section]
    settings_calls = _install_signature_bound_settings(monkeypatch, settings)
    transport_calls = _install_signature_bound_session_post(
        monkeypatch,
        AssertionError("transport invoked before missing-section failure"),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = variant.summarizer(
            None,
            "fixed input",
            "fixed prompt",
            temp=0.2,
            system_message="fixed system message",
        )

    assert settings_calls == [((), {})]
    assert transport_calls == []
    assert result == (
        f"{variant.event_prefix}: Unexpected error occurred: "
        f"'{variant.settings_section}'"
    )
    assert variant.settings_section not in captured.text
    assert (
        f"{variant.event_prefix}: Unexpected failure; exception_type=KeyError"
        in captured.text
    )


@pytest.mark.parametrize(
    "variant",
    CUSTOM_OPENAI_VARIANTS,
    ids=lambda variant: variant.id,
)
def test_local_custom_openai_missing_model_preserves_error_contract(
    variant: _CustomOpenAIVariant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _custom_openai_settings()
    del settings[variant.settings_section]["model"]
    settings_calls = _install_signature_bound_settings(monkeypatch, settings)
    transport_calls = _install_signature_bound_session_post(
        monkeypatch,
        AssertionError("transport invoked before missing-model failure"),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = variant.summarizer(
            "fixed-explicit-key",
            "fixed input",
            "fixed prompt",
            temp=0.2,
            system_message="fixed system message",
        )

    assert settings_calls == [((), {})]
    assert transport_calls == []
    assert result == f"{variant.event_prefix}: Unexpected error occurred: 'model'"
    assert "'model'" not in captured.text
    assert (
        f"{variant.event_prefix}: Unexpected failure; exception_type=KeyError"
        in captured.text
    )


@pytest.mark.parametrize(
    "variant",
    CUSTOM_OPENAI_VARIANTS,
    ids=lambda variant: variant.id,
)
def test_local_custom_openai_non_string_prompt_preserves_stringification_contract(
    variant: _CustomOpenAIVariant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FixedPrompt:
        def __init__(self) -> None:
            self.calls = 0

        def __str__(self) -> str:
            self.calls += 1
            return "fixed-object-prompt"

    prompt = FixedPrompt()
    response = _FakeResponse(
        json_data={"choices": [{"message": {"content": "fixed summary"}}]}
    )
    _install_signature_bound_settings(monkeypatch, _custom_openai_settings())
    transport_calls = _install_signature_bound_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = variant.summarizer(
            "fixed-custom-openai-key",
            "fixed input",
            prompt,
            temp=0.2,
            system_message="fixed system message",
        )

    assert result == "fixed summary"
    assert prompt.calls == 2
    assert len(transport_calls) == 1
    assert transport_calls[0][1]["json"]["messages"][1]["content"].endswith(
        "fixed-object-prompt"
    )
    assert "fixed-object-prompt" not in captured.text
    assert (
        f"{variant.event_prefix}: Prompt prepared; character_count=19" in captured.text
    )


def test_local_save_summary_to_file_hides_path_and_preserves_write_contract(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source_path = tmp_path / f"{CUSTOM_OPENAI_FILE_PATH_CANARY}-segments.json"
    expected_path = tmp_path / f"{CUSTOM_OPENAI_FILE_PATH_CANARY}-segments_summary.txt"

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = local_summarization.save_summary_to_file(
            "fixed private summary",
            str(source_path),
        )

    assert result is None
    assert expected_path.read_text() == "fixed private summary"
    assert CUSTOM_OPENAI_FILE_PATH_CANARY not in captured.text
    assert "Summary saved to file" in captured.text


def test_local_save_summary_to_file_does_not_report_success_before_open_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source_path = tmp_path / "nested" / "fixed-segments.json"
    real_open = builtins.open
    open_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def failing_open(*args: object, **kwargs: object) -> object:
        inspect.signature(real_open).bind(*args, **kwargs)
        open_calls.append((args, kwargs))
        raise OSError(CUSTOM_OPENAI_EXCEPTION_CANARY)

    monkeypatch.setattr(builtins, "open", failing_open)

    with _capture_stdlib_and_loguru(caplog) as captured:
        with pytest.raises(OSError, match=CUSTOM_OPENAI_EXCEPTION_CANARY):
            local_summarization.save_summary_to_file(
                "fixed private summary",
                str(source_path),
            )

    assert source_path.parent.is_dir()
    assert open_calls == (
        [
            (
                (str(source_path.with_name("fixed-segments_summary.txt")), "w"),
                {},
            )
        ]
    )
    assert CUSTOM_OPENAI_EXCEPTION_CANARY not in captured.text
    assert "Summary saved to file" not in captured.text


def test_local_save_summary_to_file_does_not_report_success_before_write_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source_path = (
        tmp_path / "nested" / f"{CUSTOM_OPENAI_FILE_PATH_CANARY}-segments.json"
    )
    real_open = builtins.open
    open_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class WriteFailingFile:
        entered = False
        exit_type: type[BaseException] | None = None

        def __enter__(self) -> WriteFailingFile:
            self.entered = True
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: object,
        ) -> bool:
            del exc_value, traceback
            self.exit_type = exc_type
            return False

        def write(self, summary: str) -> None:
            assert summary == "fixed private summary"
            raise OSError(CUSTOM_OPENAI_EXCEPTION_CANARY)

    fake_file = WriteFailingFile()

    def fake_open(*args: object, **kwargs: object) -> WriteFailingFile:
        inspect.signature(real_open).bind(*args, **kwargs)
        open_calls.append((args, kwargs))
        return fake_file

    monkeypatch.setattr(builtins, "open", fake_open)

    with _capture_stdlib_and_loguru(caplog) as captured:
        with pytest.raises(OSError, match=CUSTOM_OPENAI_EXCEPTION_CANARY):
            local_summarization.save_summary_to_file(
                "fixed private summary",
                str(source_path),
            )

    assert source_path.parent.is_dir()
    assert open_calls == [
        (
            (
                str(
                    source_path.with_name(
                        f"{CUSTOM_OPENAI_FILE_PATH_CANARY}-segments_summary.txt"
                    )
                ),
                "w",
            ),
            {},
        )
    ]
    assert fake_file.entered is True
    assert fake_file.exit_type is OSError
    assert CUSTOM_OPENAI_EXCEPTION_CANARY not in captured.text
    assert CUSTOM_OPENAI_FILE_PATH_CANARY not in captured.text
    assert "Summary saved to file" not in captured.text


def test_no_pending_general_core_sites() -> None:
    pending = [
        site
        for site in _ledger_sites()
        if site["group"] == "general_core"
        and site["starting_classification"] == "private"
        and site["outcome"] == "pending"
    ]

    assert not pending, (
        f"general_core has {len(pending)} pending private diagnostics: "
        f"{[site['site_id'] for site in pending]}"
    )


def test_analyze_nested_generator_exception_is_consumed_without_private_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    real_dispatch = general_summarization._dispatch_to_api
    signature = inspect.signature(real_dispatch)
    stream_finished = False

    def fake_dispatch(*args: object, **kwargs: object) -> Iterator[str]:
        signature.bind(*args, **kwargs)

        def failing_stream() -> Iterator[str]:
            nonlocal stream_finished
            yield "fixed partial"
            stream_finished = True
            raise RuntimeError(GENERAL_ANALYZE_STREAM_EXCEPTION_CANARY)

        return failing_stream()

    monkeypatch.setattr(general_summarization, "_dispatch_to_api", fake_dispatch)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.analyze(
            "openai",
            "fixed input",
            "fixed prompt",
        )

    assert result == (
        f"Error consuming stream: {GENERAL_ANALYZE_STREAM_EXCEPTION_CANARY}"
    )
    assert stream_finished is True
    assert GENERAL_ANALYZE_STREAM_EXCEPTION_CANARY not in captured.text
    assert "Error consuming generator; exception_type=RuntimeError" in captured.text
    assert not [record for record in captured.caplog.records if record.exc_info]


def test_analyze_error_result_logs_safe_provider_context(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    real_dispatch = general_summarization._dispatch_to_api
    signature = inspect.signature(real_dispatch)

    def fake_dispatch(*args: object, **kwargs: object) -> str:
        signature.bind(*args, **kwargs)
        return "Error: fixed provider failure"

    monkeypatch.setattr(general_summarization, "_dispatch_to_api", fake_dispatch)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.analyze(
            "fixed-provider",
            "fixed input",
            "fixed prompt",
        )

    assert result == "Error: fixed provider failure"
    assert "Summarization failed; provider=fixed-provider" in captured.text


def test_analyze_chunk_failure_preserves_placeholder_without_logging_private_output(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    real_chunker = general_summarization.improved_chunking_process
    chunker_signature = inspect.signature(real_chunker)
    real_dispatch = general_summarization._dispatch_to_api
    dispatch_signature = inspect.signature(real_dispatch)

    def fake_chunker(*args: object, **kwargs: object) -> list[dict[str, str]]:
        chunker_signature.bind(*args, **kwargs)
        return [{"text": "fixed chunk"}]

    def fake_dispatch(*args: object, **kwargs: object) -> str:
        dispatch_signature.bind(*args, **kwargs)
        return f"Error: {GENERAL_RESPONSE_CANARY}"

    monkeypatch.setattr(general_summarization, "CHUNKER_AVAILABLE", True)
    monkeypatch.setattr(
        general_summarization,
        "improved_chunking_process",
        fake_chunker,
    )
    monkeypatch.setattr(general_summarization, "_dispatch_to_api", fake_dispatch)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.analyze(
            "openai",
            "fixed input",
            "fixed prompt",
            chunked_summarization=True,
        )

    assert result == (f"[Error summarizing chunk 1: Error: {GENERAL_RESPONSE_CANARY}]")
    assert GENERAL_RESPONSE_CANARY not in captured.text
    assert "Failed to summarize chunk; chunk=1" in captured.text


def test_analyze_critical_exception_preserves_error_contract_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    real_extract = general_summarization.extract_text_from_input
    signature = inspect.signature(real_extract)

    def fake_extract(*args: object, **kwargs: object) -> str:
        signature.bind(*args, **kwargs)
        raise RuntimeError(GENERAL_EXCEPTION_CANARY)

    monkeypatch.setattr(general_summarization, "extract_text_from_input", fake_extract)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.analyze(
            "openai",
            "fixed input",
            "fixed prompt",
        )

    assert result == (
        "Error: An unexpected error occurred during summarization: "
        f"{GENERAL_EXCEPTION_CANARY}"
    )
    assert GENERAL_EXCEPTION_CANARY not in captured.text
    assert "Critical error in summarize function; exception_type=RuntimeError" in (
        captured.text
    )
    assert not [record for record in captured.caplog.records if record.exc_info]


def test_general_core_dispatch_exception_preserves_in_band_error_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    real_openai = general_summarization.summarize_with_openai
    signature = inspect.signature(real_openai)

    def failing_openai(*args: object, **kwargs: object) -> str:
        signature.bind(*args, **kwargs)
        raise RuntimeError(GENERAL_EXCEPTION_CANARY)

    monkeypatch.setattr(
        general_summarization,
        "summarize_with_openai",
        failing_openai,
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization._dispatch_to_api(
            "fixed input",
            "fixed prompt",
            "openai",
            "fixed key",
            0.2,
            "fixed system",
            streaming=False,
        )

    assert result == f"Error calling API openai: {GENERAL_EXCEPTION_CANARY}"
    assert GENERAL_EXCEPTION_CANARY not in captured.text
    assert (
        "Error during dispatch to API; provider=openai exception_type=RuntimeError"
        in captured.text
    )
    assert not [record for record in captured.caplog.records if record.exc_info]


def test_general_openai_config_credential_and_endpoint_are_not_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings_calls = _install_signature_bound_general_settings(
        monkeypatch,
        _general_provider_settings(
            openai_key=GENERAL_CREDENTIAL_CANARY,
            openai_endpoint=GENERAL_OPENAI_ENDPOINT_CANARY,
        ),
    )
    post_calls = _install_signature_bound_general_session_post(
        monkeypatch,
        _FakeResponse(
            json_data={"choices": [{"message": {"content": "fixed summary"}}]}
        ),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_openai(
            "",
            "fixed input",
            "fixed prompt",
        )

    assert result == "fixed summary"
    assert settings_calls
    assert post_calls[0][0][0] == (f"{GENERAL_OPENAI_ENDPOINT_CANARY}/chat/completions")
    assert GENERAL_CREDENTIAL_CANARY not in captured.text
    assert GENERAL_OPENAI_ENDPOINT_CANARY not in captured.text
    assert "OpenAI Summarize: Config credential lookup completed" in captured.text
    assert "OpenAI: Endpoint configured" in captured.text


def test_general_openai_missing_config_credential_preserves_error_contract(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_provider_settings(openai_key=""),
    )
    post_calls = _install_signature_bound_general_session_post(
        monkeypatch,
        AssertionError("transport must not run without a credential"),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_openai(
            "",
            "fixed input",
            "fixed prompt",
        )

    assert result == "Error: OpenAI API Key Not Provided/Found or is empty."
    assert post_calls == []
    assert "OpenAI Summarize: Config credential lookup completed" in captured.text
    assert "OpenAI: Credential configured" not in captured.text


def test_general_openai_truthy_non_boolean_streaming_value_is_not_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_provider_settings(),
    )
    response = _FakeResponse(
        lines=(
            b'data: {"choices":[{"delta":{"content":"fixed streamed chunk"}}]}',
            b"data: [DONE]",
        )
    )
    post_calls = _install_signature_bound_general_session_post(
        monkeypatch,
        response,
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        stream = general_summarization.summarize_with_openai(
            "fixed-general-openai-key",
            "fixed input",
            "fixed prompt",
            streaming=GENERAL_OPENAI_PRIVATE_STREAMING_VALUE,
        )
        assert inspect.isgenerator(stream)
        assert response.iter_lines_started is False
        assert response.closed is False
        chunks = list(stream)

    assert chunks == ["fixed streamed chunk"]
    assert len(post_calls) == 1
    post_args, post_kwargs = post_calls[0]
    assert post_args == ("http://openai.invalid/v1/chat/completions",)
    assert post_kwargs["stream"] == GENERAL_OPENAI_PRIVATE_STREAMING_VALUE
    assert post_kwargs["json"]["stream"] == GENERAL_OPENAI_PRIVATE_STREAMING_VALUE
    assert response.iter_lines_started is True
    assert response.closed is True
    assert GENERAL_OPENAI_PRIVATE_STREAMING_VALUE not in captured.text
    assert "OpenAI: Request options prepared" in captured.text


@pytest.mark.parametrize(
    "line",
    [
        f"data: {{{GENERAL_OPENAI_STREAM_CANARY}".encode(),
        f"data: {json.dumps({'choices': [], 'private': GENERAL_OPENAI_STREAM_CANARY})}".encode(),
    ],
    ids=["invalid-json", "unexpected-shape"],
)
def test_general_openai_malformed_stream_is_fully_consumed_without_private_diagnostic(
    line: bytes,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_provider_settings(),
    )
    response = _FakeResponse(lines=(line, b"data: [DONE]"))
    _install_signature_bound_general_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        stream = general_summarization.summarize_with_openai(
            "fixed-general-openai-key",
            "fixed input",
            "fixed prompt",
            streaming=True,
        )
        chunks = list(stream)

    assert chunks == []
    assert response.iter_lines_started is True
    assert response.closed is True
    assert GENERAL_OPENAI_STREAM_CANARY not in captured.text
    assert "OpenAI Stream: Response event rejected" in captured.text


def test_general_openai_stream_iterator_exception_preserves_lazy_error_contract(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class IteratorFailingResponse(_FakeResponse):
        def iter_lines(self) -> Iterator[bytes]:
            self.iter_lines_started = True
            raise RuntimeError(GENERAL_OPENAI_STREAM_EXCEPTION_CANARY)
            yield b"unreachable"

    _install_signature_bound_general_settings(
        monkeypatch,
        _general_provider_settings(),
    )
    response = IteratorFailingResponse()
    post_calls = _install_signature_bound_general_session_post(
        monkeypatch,
        response,
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        stream = general_summarization.summarize_with_openai(
            "fixed-general-openai-key",
            "fixed input",
            "fixed prompt",
            streaming=True,
        )
        assert inspect.isgenerator(stream)
        assert response.iter_lines_started is False
        assert response.closed is False
        chunks = list(stream)

    assert chunks == [
        f"Error during streaming: {GENERAL_OPENAI_STREAM_EXCEPTION_CANARY}"
    ]
    assert len(post_calls) == 1
    post_args, post_kwargs = post_calls[0]
    assert post_args == ("http://openai.invalid/v1/chat/completions",)
    assert post_kwargs["stream"] is True
    assert post_kwargs["timeout"] == 5
    assert post_kwargs["json"]["stream"] is True
    assert post_kwargs["json"]["messages"] == [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "fixed input \n\n\n\nfixed prompt"},
    ]
    assert response.iter_lines_started is True
    assert response.closed is True
    assert GENERAL_OPENAI_STREAM_EXCEPTION_CANARY not in captured.text
    assert (
        "OpenAI Stream: Streaming failed; exception_type=RuntimeError" in captured.text
    )
    assert not [record for record in captured.caplog.records if record.exc_info]


def test_general_openai_missing_summary_shape_hides_response_body(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_provider_settings(),
    )
    _install_signature_bound_general_session_post(
        monkeypatch,
        _FakeResponse(json_data={"private": GENERAL_RESPONSE_CANARY}),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_openai(
            "fixed-general-openai-key",
            "fixed input",
            "fixed prompt",
        )

    assert result == "Error: OpenAI Summary not found in response."
    assert GENERAL_RESPONSE_CANARY not in captured.text
    assert "OpenAI: Summary not found in response" in captured.text


def test_general_openai_request_exception_hides_message_and_traceback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_provider_settings(),
    )
    _install_signature_bound_general_session_post(
        monkeypatch,
        general_summarization.requests.exceptions.RequestException(
            GENERAL_OPENAI_EXCEPTION_CANARY
        ),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_openai(
            "fixed-general-openai-key",
            "fixed input",
            "fixed prompt",
        )

    assert result == (
        f"Error: OpenAI API request failed: {GENERAL_OPENAI_EXCEPTION_CANARY}"
    )
    assert GENERAL_OPENAI_EXCEPTION_CANARY not in captured.text
    assert "OpenAI: API request failed; exception_type=RequestException" in (
        captured.text
    )
    assert not [record for record in captured.caplog.records if record.exc_info]


def test_anthropic_success_hides_prompt_credential_and_response(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_provider_settings(),
    )
    post_calls = _install_signature_bound_general_requests_post(
        monkeypatch,
        _FakeResponse(
            json_data={
                "content": [{"type": "text", "text": GENERAL_ANTHROPIC_RESPONSE_CANARY}]
            }
        ),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_anthropic(
            GENERAL_CREDENTIAL_CANARY,
            "fixed input",
            GENERAL_PROMPT_CANARY,
            max_retries=1,
            retry_delay=0,
        )

    assert result == GENERAL_ANTHROPIC_RESPONSE_CANARY
    assert len(post_calls) == 1
    assert GENERAL_CREDENTIAL_CANARY not in captured.text
    assert GENERAL_PROMPT_CANARY not in captured.text
    assert GENERAL_ANTHROPIC_RESPONSE_CANARY not in captured.text
    assert "Anthropic: Prompt prepared" in captured.text
    assert "Anthropic: Summarization successful" in captured.text


def test_anthropic_malformed_stream_is_fully_consumed_without_private_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_provider_settings(),
    )
    response = _FakeResponse(
        lines=(
            b"event: content_block_delta",
            f"data: {{{GENERAL_ANTHROPIC_STREAM_CANARY}".encode(),
        )
    )
    _install_signature_bound_general_requests_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        stream = general_summarization.summarize_with_anthropic(
            "fixed-general-anthropic-key",
            "fixed input",
            "fixed prompt",
            streaming=True,
            max_retries=1,
            retry_delay=0,
        )
        chunks = list(stream)

    assert chunks == []
    assert response.iter_lines_started is True
    assert GENERAL_ANTHROPIC_STREAM_CANARY not in captured.text
    assert "Anthropic: Stream JSON decode failed" in captured.text


def test_anthropic_unexpected_response_shape_hides_response_text(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_provider_settings(),
    )
    _install_signature_bound_general_requests_post(
        monkeypatch,
        _FakeResponse(json_data=[], text=GENERAL_ANTHROPIC_RESPONSE_CANARY),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_anthropic(
            "fixed-general-anthropic-key",
            "fixed input",
            "fixed prompt",
            max_retries=1,
            retry_delay=0,
        )

    assert result is None
    assert GENERAL_ANTHROPIC_RESPONSE_CANARY not in captured.text
    assert "Unexpected response format from Anthropic API" in captured.text


def test_anthropic_non_success_hides_response_body_and_preserves_status_contract(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_provider_settings(),
    )
    _install_signature_bound_general_requests_post(
        monkeypatch,
        _FakeResponse(
            status_code=429,
            text=GENERAL_ANTHROPIC_RESPONSE_CANARY,
        ),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_anthropic(
            "fixed-general-anthropic-key",
            "fixed input",
            "fixed prompt",
            max_retries=1,
            retry_delay=0,
        )

    assert result is None
    assert GENERAL_ANTHROPIC_RESPONSE_CANARY not in captured.text
    assert "Failed to process summary; status_code=429" in captured.text


def test_anthropic_request_exception_hides_message_and_preserves_retry_contract(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_provider_settings(),
    )
    post_calls = _install_signature_bound_general_requests_post(
        monkeypatch,
        general_summarization.requests.RequestException(
            GENERAL_ANTHROPIC_EXCEPTION_CANARY
        ),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_anthropic(
            "fixed-general-anthropic-key",
            "fixed input",
            "fixed prompt",
            max_retries=1,
            retry_delay=0,
        )

    assert result == (f"Anthropic: Network error: {GENERAL_ANTHROPIC_EXCEPTION_CANARY}")
    assert len(post_calls) == 1
    assert GENERAL_ANTHROPIC_EXCEPTION_CANARY not in captured.text
    assert (
        "Anthropic: Network error during attempt; attempt=1 retry_count=1 "
        "exception_type=RequestException"
    ) in captured.text


def test_anthropic_file_error_hides_path_and_preserves_in_band_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    real_log_debug_data = general_summarization.log_debug_data
    signature = inspect.signature(real_log_debug_data)

    def failing_log_debug_data(*args: object, **kwargs: object) -> None:
        signature.bind(*args, **kwargs)
        raise FileNotFoundError

    monkeypatch.setattr(
        general_summarization,
        "log_debug_data",
        failing_log_debug_data,
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_anthropic(
            "fixed-general-anthropic-key",
            GENERAL_PATH_CANARY,
            "fixed prompt",
        )

    assert result == f"Anthropic: File not found: {GENERAL_PATH_CANARY}"
    assert GENERAL_PATH_CANARY not in captured.text
    assert "Anthropic: File not found" in captured.text


def test_no_pending_general_mid_sites() -> None:
    pending = [
        site
        for site in _ledger_sites()
        if site["group"] == "general_mid"
        and site["starting_classification"] == "private"
        and site["outcome"] == "pending"
    ]

    assert len(pending) == 0, (
        f"general_mid has {len(pending)} pending private diagnostics: "
        f"{[site['site_id'] for site in pending]}"
    )


def test_cohere_success_hides_credential_prompt_and_response(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings_calls = _install_signature_bound_general_settings(
        monkeypatch,
        _general_mid_provider_settings(),
    )
    response = _FakeResponse(
        json_data={
            "message": {"content": [{"type": "text", "text": COHERE_RESPONSE_CANARY}]}
        }
    )
    post_calls = _install_signature_bound_general_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_cohere(
            COHERE_CREDENTIAL_CANARY,
            "fixed input",
            COHERE_PROMPT_CANARY,
            system_message="fixed system",
        )

    assert result == COHERE_RESPONSE_CANARY
    assert settings_calls == [
        (("cohere_api", "model", "command-a-03-2025"), {}),
        (("cohere_api", "api_retries", 3), {}),
        (("cohere_api", "api_retry_delay", 5), {}),
    ]
    assert len(post_calls) == 1
    post_args, post_kwargs = post_calls[0]
    assert post_args == ("https://api.cohere.com/v2/chat",)
    assert post_kwargs["headers"]["Authorization"] == (
        f"Bearer {COHERE_CREDENTIAL_CANARY}"
    )
    assert post_kwargs["json"]["messages"] == [
        {"role": "system", "content": "fixed system"},
        {
            "role": "user",
            "content": f"fixed input \n\n\n\n{COHERE_PROMPT_CANARY}",
        },
    ]
    for canary in (
        COHERE_CREDENTIAL_CANARY,
        COHERE_PROMPT_CANARY,
        COHERE_RESPONSE_CANARY,
    ):
        assert canary not in captured.text
    assert "Cohere: Credential configured" in captured.text
    assert "Cohere: Prompt prepared; character_count=" in captured.text
    assert "Cohere: API response received" in captured.text


def test_cohere_missing_config_credential_does_not_claim_configuration(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _general_mid_provider_settings()
    settings[("cohere_api", "api_key")] = ""
    settings_calls = _install_signature_bound_general_settings(monkeypatch, settings)
    post_calls = _install_signature_bound_general_session_post(
        monkeypatch,
        AssertionError("transport must not run without a credential"),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_cohere(
            "",
            "fixed input",
            "fixed prompt",
        )

    assert result == "Cohere: API Key Not Provided/Found in Config file or is empty"
    assert settings_calls == [(("cohere_api", "api_key"), {})]
    assert post_calls == []
    assert "Cohere: Credential configured" not in captured.text


def test_cohere_stream_is_lazy_and_hides_rejected_lines(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_mid_provider_settings(),
    )
    response = _FakeResponse(
        lines=(
            f"retry: {COHERE_STREAM_CANARY}".encode(),
            f"data: {{{COHERE_STREAM_CANARY}".encode(),
            f"data: {json.dumps([COHERE_STREAM_CANARY])}".encode(),
            json.dumps(
                {
                    "type": "content-delta",
                    "delta": {"message": {"content": {"text": "fixed cohere chunk"}}},
                }
            ).encode(),
            b"data: [DONE]",
        )
    )
    post_calls = _install_signature_bound_general_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        stream = general_summarization.summarize_with_cohere(
            "fixed-cohere-key",
            "fixed input",
            "fixed prompt",
            streaming=COHERE_PRIVATE_STREAMING_VALUE,
        )
        assert inspect.isgenerator(stream)
        assert response.iter_lines_started is False
        assert response.closed is False
        chunks = list(stream)

    assert chunks == ["fixed cohere chunk"]
    assert response.iter_lines_started is True
    assert response.closed is True
    assert len(post_calls) == 1
    assert post_calls[0][1]["json"]["stream"] == COHERE_PRIVATE_STREAMING_VALUE
    assert post_calls[0][1]["stream"] is True
    assert COHERE_STREAM_CANARY not in captured.text
    assert COHERE_PRIVATE_STREAMING_VALUE not in captured.text
    assert "Cohere Stream: Non-JSON line skipped" in captured.text
    assert "Cohere Stream: Response event rejected" in captured.text
    assert "Cohere Stream: Non-object event skipped" in captured.text


def test_cohere_unknown_stream_event_hides_provider_controlled_type(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_mid_provider_settings(),
    )
    response = _FakeResponse(
        lines=(
            json.dumps({"type": COHERE_EVENT_TYPE_CANARY}).encode(),
            json.dumps(
                {
                    "type": "content-delta",
                    "delta": {"message": {"content": {"text": "fixed chunk"}}},
                }
            ).encode(),
            b"data: [DONE]",
        )
    )
    post_calls = _install_signature_bound_general_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        stream = general_summarization.summarize_with_cohere(
            "fixed-cohere-key",
            "fixed input",
            "fixed prompt",
            streaming=True,
        )
        assert inspect.isgenerator(stream)
        assert response.iter_lines_started is False
        assert response.closed is False
        chunks = list(stream)

    assert chunks == ["fixed chunk"]
    assert response.iter_lines_started is True
    assert response.closed is True
    assert len(post_calls) == 1
    assert post_calls[0][1]["json"]["stream"] is True
    assert post_calls[0][1]["stream"] is True
    assert COHERE_EVENT_TYPE_CANARY not in captured.text
    assert "Cohere: Unhandled streaming event" in captured.text


@pytest.mark.parametrize("streaming", [False, True], ids=["nonstream", "stream"])
def test_cohere_status_failure_hides_response_body_and_preserves_return(
    streaming: bool,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_mid_provider_settings(),
    )
    response = _FakeResponse(status_code=429, text=COHERE_RESPONSE_CANARY)
    _install_signature_bound_general_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_cohere(
            "fixed-cohere-key",
            "fixed input",
            "fixed prompt",
            streaming=streaming,
        )

    assert result == f"Cohere: API request failed: {COHERE_RESPONSE_CANARY}"
    assert response.iter_lines_started is False
    assert COHERE_RESPONSE_CANARY not in captured.text
    assert "Cohere: API request failed; status_code=429" in captured.text


def test_cohere_transport_exception_hides_message_and_traceback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_mid_provider_settings(),
    )
    _install_signature_bound_general_session_post(
        monkeypatch,
        RuntimeError(COHERE_EXCEPTION_CANARY),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_cohere(
            "fixed-cohere-key",
            "fixed input",
            "fixed prompt",
        )

    assert result == (
        "Cohere: Error occurred while processing summary with Cohere: "
        f"{COHERE_EXCEPTION_CANARY}"
    )
    assert COHERE_EXCEPTION_CANARY not in captured.text
    assert "Cohere: Processing failed; exception_type=RuntimeError" in captured.text
    assert not [record for record in captured.caplog.records if record.exc_info]


def test_groq_success_hides_input_prompt_credential_and_response(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings_calls = _install_signature_bound_general_settings(
        monkeypatch,
        _general_mid_provider_settings(),
    )
    response = _FakeResponse(
        json_data={"choices": [{"message": {"content": GROQ_RESPONSE_CANARY}}]}
    )
    post_calls = _install_signature_bound_general_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_groq(
            GROQ_CREDENTIAL_CANARY,
            GROQ_INPUT_CANARY,
            GROQ_PROMPT_CANARY,
            system_message="fixed system",
        )

    assert result == GROQ_RESPONSE_CANARY
    assert settings_calls == [
        (("groq_api", "model", "llama3-70b-8192"), {}),
        (("groq_api", "api_retries", 3), {}),
        (("groq_api", "api_retry_delay", 5), {}),
    ]
    assert len(post_calls) == 1
    post_args, post_kwargs = post_calls[0]
    assert post_args == ("https://api.groq.com/openai/v1/chat/completions",)
    assert post_kwargs["headers"]["Authorization"] == (
        f"Bearer {GROQ_CREDENTIAL_CANARY}"
    )
    assert post_kwargs["json"]["messages"] == [
        {"role": "system", "content": "fixed system"},
        {
            "role": "user",
            "content": f"{GROQ_INPUT_CANARY} \n\n\n\n{GROQ_PROMPT_CANARY}",
        },
    ]
    for canary in (
        GROQ_CREDENTIAL_CANARY,
        GROQ_INPUT_CANARY,
        GROQ_PROMPT_CANARY,
        GROQ_RESPONSE_CANARY,
    ):
        assert canary not in captured.text
    assert "Groq: Credential configured" in captured.text
    assert "Groq: Input prepared; character_count=" in captured.text
    assert "Groq: Prompt prepared; character_count=" in captured.text
    assert "Groq: API response received" in captured.text


def test_groq_missing_config_credential_does_not_claim_configuration(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _general_mid_provider_settings()
    settings[("groq_api", "api_key")] = ""
    _install_signature_bound_general_settings(monkeypatch, settings)
    post_calls = _install_signature_bound_general_session_post(
        monkeypatch,
        AssertionError("transport must not run without a credential"),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_groq(
            "",
            "fixed input",
            "fixed prompt",
        )

    assert result == "Groq: API Key Not Provided/Found in Config file or is empty"
    assert post_calls == []
    assert "Groq: Credential configured" not in captured.text


def test_groq_stream_preserves_raw_flag_and_hides_malformed_line(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_mid_provider_settings(),
    )
    response = _FakeResponse(
        lines=(
            f"data: {{{GROQ_STREAM_CANARY}".encode(),
            json.dumps({"choices": [{"delta": {"content": "fixed groq chunk"}}]})
            .join(["data: ", ""])
            .encode(),
            b"data: [DONE]",
        )
    )
    post_calls = _install_signature_bound_general_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        stream = general_summarization.summarize_with_groq(
            "fixed-groq-key",
            "fixed input",
            "fixed prompt",
            streaming=GROQ_PRIVATE_STREAMING_VALUE,
        )
        assert inspect.isgenerator(stream)
        assert response.iter_lines_started is False
        chunks = list(stream)

    assert chunks == ["fixed groq chunk"]
    assert response.iter_lines_started is True
    assert len(post_calls) == 1
    assert post_calls[0][1]["json"]["stream"] == GROQ_PRIVATE_STREAMING_VALUE
    assert post_calls[0][1]["stream"] is True
    assert GROQ_STREAM_CANARY not in captured.text
    assert GROQ_PRIVATE_STREAMING_VALUE not in captured.text
    assert "Groq Stream: Response event rejected" in captured.text


def test_groq_status_failure_hides_response_body_and_preserves_return(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_mid_provider_settings(),
    )
    response = _FakeResponse(
        status_code=503,
        json_data={"error": GROQ_RESPONSE_CANARY},
        text=GROQ_RESPONSE_CANARY,
    )
    _install_signature_bound_general_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_groq(
            "fixed-groq-key",
            "fixed input",
            "fixed prompt",
        )

    assert result == f"Groq: API request failed: {GROQ_RESPONSE_CANARY}"
    assert GROQ_RESPONSE_CANARY not in captured.text
    assert "Groq: API request failed; status_code=503" in captured.text


def test_groq_transport_exception_hides_message_and_traceback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_mid_provider_settings(),
    )
    _install_signature_bound_general_session_post(
        monkeypatch,
        RuntimeError(GROQ_EXCEPTION_CANARY),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_groq(
            "fixed-groq-key",
            "fixed input",
            "fixed prompt",
        )

    assert result == (
        "Groq: Error occurred while processing summary with Groq: "
        f"{GROQ_EXCEPTION_CANARY}"
    )
    assert GROQ_EXCEPTION_CANARY not in captured.text
    assert "Groq: Processing failed; exception_type=RuntimeError" in captured.text
    assert not [record for record in captured.caplog.records if record.exc_info]


def test_groq_input_conversion_remains_eager_without_logging_converted_value(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class UnsupportedInput:
        calls = 0

        def __str__(self) -> str:
            self.calls += 1
            return GROQ_INPUT_CANARY

    input_value = UnsupportedInput()
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_mid_provider_settings(),
    )
    post_calls = _install_signature_bound_general_session_post(
        monkeypatch,
        AssertionError("transport must not run for unsupported input"),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_groq(
            "fixed-groq-key",
            input_value,
            "fixed prompt",
        )

    assert result == (
        "Groq: Error occurred while processing summary with Groq: "
        "Groq: Invalid input data format"
    )
    assert input_value.calls == 1
    assert post_calls == []
    assert GROQ_INPUT_CANARY not in captured.text
    assert "Groq: Input prepared; character_count=" in captured.text


def test_openrouter_success_hides_credential_input_prompt_and_response(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings_calls = _install_signature_bound_general_settings(
        monkeypatch,
        _general_mid_provider_settings(),
    )
    config_calls = _install_signature_bound_general_config_loader(monkeypatch)
    response = _FakeResponse(
        json_data={"choices": [{"message": {"content": OPENROUTER_RESPONSE_CANARY}}]}
    )
    post_calls = _install_signature_bound_general_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_openrouter(
            OPENROUTER_CREDENTIAL_CANARY,
            OPENROUTER_INPUT_CANARY,
            OPENROUTER_PROMPT_CANARY,
            system_message="fixed system",
        )

    assert result == OPENROUTER_RESPONSE_CANARY
    assert config_calls == [((), {})]
    assert settings_calls == [
        (("openrouter_api", "model", "mistralai/mistral-7b-instruct"), {}),
        (("openrouter_api", "api_retries", 3), {}),
        (("openrouter_api", "api_retry_delay", 5), {}),
    ]
    assert len(post_calls) == 1
    post_args, post_kwargs = post_calls[0]
    assert post_args == ()
    assert post_kwargs["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert post_kwargs["headers"]["Authorization"] == (
        f"Bearer {OPENROUTER_CREDENTIAL_CANARY}"
    )
    payload = json.loads(post_kwargs["data"])
    assert payload["messages"] == [
        {"role": "system", "content": "fixed system"},
        {
            "role": "user",
            "content": f"{OPENROUTER_INPUT_CANARY} \n\n\n\n{OPENROUTER_PROMPT_CANARY}",
        },
    ]
    for canary in (
        OPENROUTER_CREDENTIAL_CANARY,
        OPENROUTER_INPUT_CANARY,
        OPENROUTER_PROMPT_CANARY,
        OPENROUTER_RESPONSE_CANARY,
    ):
        assert canary not in captured.text
    assert "OpenRouter: Credential configured" in captured.text
    assert "OpenRouter: API response received" in captured.text


def test_openrouter_stream_hides_returned_content_and_consumes_lines(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_mid_provider_settings(),
    )
    _install_signature_bound_general_config_loader(monkeypatch)
    response = _FakeResponse(
        lines=(
            f"data: {json.dumps({'choices': [{'delta': {'content': OPENROUTER_STREAM_CANARY}}]})}".encode(),
            b"data: [DONE]",
        )
    )
    post_calls = _install_signature_bound_general_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_openrouter(
            "fixed-openrouter-key",
            "fixed input",
            "fixed prompt",
            streaming=OPENROUTER_PRIVATE_STREAMING_VALUE,
        )

    assert result == OPENROUTER_STREAM_CANARY
    assert response.iter_lines_started is True
    assert len(post_calls) == 1
    assert json.loads(post_calls[0][1]["data"])["stream"] is True
    assert post_calls[0][1]["stream"] is True
    assert OPENROUTER_STREAM_CANARY not in captured.text
    assert OPENROUTER_PRIVATE_STREAMING_VALUE not in captured.text
    assert "OpenRouter Stream: Content received" in captured.text


def test_openrouter_stream_non_string_content_preserves_historical_error_contract(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_mid_provider_settings(),
    )
    _install_signature_bound_general_config_loader(monkeypatch)
    response = _FakeResponse(
        lines=(
            f"data: {json.dumps({'choices': [{'delta': {'content': 7}}]})}".encode(),
        )
    )
    post_calls = _install_signature_bound_general_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_openrouter(
            "fixed-openrouter-key",
            "fixed input",
            "fixed prompt",
            streaming=True,
        )

    assert result == (
        "openrouter: Error occurred while processing stream: can only concatenate str "
        '(not "int") to str'
    )
    assert response.iter_lines_started is True
    assert response.closed is False
    assert len(post_calls) == 1
    assert post_calls[0][0] == ()
    assert post_calls[0][1]["url"] == ("https://openrouter.ai/api/v1/chat/completions")
    assert post_calls[0][1]["stream"] is True
    assert json.loads(post_calls[0][1]["data"])["stream"] is True
    assert "OpenRouter Stream: Content received" in captured.text
    assert "OpenRouter Stream: Processing failed; exception_type=TypeError" in (
        captured.text
    )


def test_openrouter_stream_status_failure_hides_body_and_preserves_return(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_mid_provider_settings(),
    )
    _install_signature_bound_general_config_loader(monkeypatch)
    response = _FakeResponse(status_code=429, text=OPENROUTER_RESPONSE_CANARY)
    _install_signature_bound_general_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_openrouter(
            "fixed-openrouter-key",
            "fixed input",
            "fixed prompt",
            streaming=True,
        )

    assert result == (
        "openrouter: Streaming API request failed with status code 429: "
        f"{OPENROUTER_RESPONSE_CANARY}"
    )
    assert OPENROUTER_RESPONSE_CANARY not in captured.text
    assert "OpenRouter Stream: API request failed; status_code=429" in captured.text


def test_openrouter_stream_exception_hides_message_and_preserves_return(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_mid_provider_settings(),
    )
    _install_signature_bound_general_config_loader(monkeypatch)
    _install_signature_bound_general_session_post(
        monkeypatch,
        RuntimeError(OPENROUTER_EXCEPTION_CANARY),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_openrouter(
            "fixed-openrouter-key",
            "fixed input",
            "fixed prompt",
            streaming=True,
        )

    assert result == (
        "openrouter: Error occurred while processing stream: "
        f"{OPENROUTER_EXCEPTION_CANARY}"
    )
    assert OPENROUTER_EXCEPTION_CANARY not in captured.text
    assert "OpenRouter Stream: Processing failed; exception_type=RuntimeError" in (
        captured.text
    )


def test_openrouter_nonstream_status_failure_hides_body_and_preserves_return(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_mid_provider_settings(),
    )
    _install_signature_bound_general_config_loader(monkeypatch)
    response = _FakeResponse(
        status_code=503,
        json_data={"error": OPENROUTER_RESPONSE_CANARY},
        text=OPENROUTER_RESPONSE_CANARY,
    )
    _install_signature_bound_general_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_openrouter(
            "fixed-openrouter-key",
            "fixed input",
            "fixed prompt",
        )

    assert result == f"openrouter: API request failed: {OPENROUTER_RESPONSE_CANARY}"
    assert OPENROUTER_RESPONSE_CANARY not in captured.text
    assert "OpenRouter: API request failed; status_code=503" in captured.text


def test_openrouter_config_exception_is_hidden_and_transport_is_not_started(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_mid_provider_settings(),
    )
    config_calls = _install_signature_bound_general_config_loader(
        monkeypatch,
        RuntimeError(OPENROUTER_EXCEPTION_CANARY),
    )
    post_calls = _install_signature_bound_general_session_post(
        monkeypatch,
        AssertionError("transport must not run after config failure"),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_openrouter(
            "fixed-openrouter-key",
            "fixed input",
            "fixed prompt",
        )

    assert result == (
        "OpenRouter: Error occurred while processing config file with OpenRouter: "
        f"{OPENROUTER_EXCEPTION_CANARY}"
    )
    assert config_calls == [((), {})]
    assert post_calls == []
    assert OPENROUTER_EXCEPTION_CANARY not in captured.text


def test_openrouter_missing_config_credential_does_not_claim_configuration(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _general_mid_provider_settings()
    settings[("openrouter_api", "api_key")] = ""
    settings_calls = _install_signature_bound_general_settings(monkeypatch, settings)
    config_calls = _install_signature_bound_general_config_loader(monkeypatch)
    post_calls = _install_signature_bound_general_session_post(
        monkeypatch,
        AssertionError("transport must not run without a credential"),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_openrouter(
            "",
            "fixed input",
            "fixed prompt",
        )

    assert result == (
        "OpenRouter: Error occurred while processing config file with OpenRouter: "
        "No valid Anthropic API key available"
    )
    assert settings_calls == [
        (("openrouter_api", "api_key"), {}),
        (("openrouter_api", "model", "mistralai/mistral-7b-instruct"), {}),
    ]
    assert config_calls == [((), {})]
    assert post_calls == []
    assert "OpenRouter: Credential configured" not in captured.text


def test_no_pending_general_streaming_sites() -> None:
    pending = [
        site
        for site in _ledger_sites()
        if site["group"] == "general_streaming"
        and site["starting_classification"] == "private"
        and site["outcome"] == "pending"
    ]

    assert len(pending) == 0, (
        f"general_streaming has {len(pending)} pending private diagnostics: "
        f"{[site['site_id'] for site in pending]}"
    )


def test_huggingface_credential_fragments_do_not_reach_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_config_loader(monkeypatch)
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_streaming_provider_settings(),
    )
    response = _FakeResponse(json_data={"generated_text": "fixed summary"})
    post_calls = _install_signature_bound_general_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_huggingface(
            HUGGINGFACE_CREDENTIAL_CANARY,
            "fixed input",
            "fixed prompt",
        )

    assert result == "fixed summary"
    assert len(post_calls) == 1
    assert post_calls[0][1]["headers"]["Authorization"] == (
        f"Bearer {HUGGINGFACE_CREDENTIAL_CANARY}"
    )
    assert HUGGINGFACE_CREDENTIAL_CANARY[:5] not in captured.text
    assert HUGGINGFACE_CREDENTIAL_CANARY[-5:] not in captured.text
    assert "HuggingFace: Credential configured" in captured.text


def test_deepseek_credential_fragments_do_not_reach_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_streaming_provider_settings(),
    )
    response = _FakeResponse(
        json_data={"choices": [{"message": {"content": "fixed summary"}}]}
    )
    post_calls = _install_signature_bound_general_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_deepseek(
            DEEPSEEK_CREDENTIAL_CANARY,
            "fixed input",
            "fixed prompt",
        )

    assert result == "fixed summary"
    assert len(post_calls) == 1
    assert post_calls[0][1]["headers"]["Authorization"] == (
        f"Bearer {DEEPSEEK_CREDENTIAL_CANARY}"
    )
    assert DEEPSEEK_CREDENTIAL_CANARY[:5] not in captured.text
    assert DEEPSEEK_CREDENTIAL_CANARY[-5:] not in captured.text
    assert "DeepSeek: Credential configured" in captured.text


def test_mistral_credential_fragments_do_not_reach_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_streaming_provider_settings(),
    )
    response = _FakeResponse(
        json_data={"choices": [{"message": {"content": "fixed summary"}}]}
    )
    post_calls = _install_signature_bound_general_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_mistral(
            MISTRAL_CREDENTIAL_CANARY,
            "fixed input",
            "fixed prompt",
        )

    assert result == "fixed summary"
    assert len(post_calls) == 1
    assert post_calls[0][1]["headers"]["Authorization"] == (
        f"Bearer {MISTRAL_CREDENTIAL_CANARY}"
    )
    assert MISTRAL_CREDENTIAL_CANARY[:5] not in captured.text
    assert MISTRAL_CREDENTIAL_CANARY[-5:] not in captured.text
    assert "Mistral: Credential configured" in captured.text


def test_huggingface_success_hides_credential_prompt_and_response(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class PromptValue:
        calls = 0

        def __str__(self) -> str:
            self.calls += 1
            return HUGGINGFACE_PROMPT_CANARY

    prompt = PromptValue()
    config_calls = _install_signature_bound_general_config_loader(monkeypatch)
    settings_calls = _install_signature_bound_general_settings(
        monkeypatch,
        _general_streaming_provider_settings(),
    )
    response = _FakeResponse(json_data={"generated_text": HUGGINGFACE_RESPONSE_CANARY})
    post_calls = _install_signature_bound_general_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_huggingface(
            HUGGINGFACE_CREDENTIAL_CANARY,
            "fixed input",
            prompt,
        )

    assert result == HUGGINGFACE_RESPONSE_CANARY
    assert prompt.calls == 1
    assert config_calls == [((), {})]
    assert settings_calls == [
        (("huggingface_api", "model", "mistralai/Mistral-7B-Instruct-v0.2"), {}),
        (("huggingface_api", "api_retries", 3), {}),
        (("huggingface_api", "api_retry_delay", 5), {}),
    ]
    assert len(post_calls) == 1
    post_args, post_kwargs = post_calls[0]
    assert post_args == (
        "https://api-inference.huggingface.co/models/fixed-huggingface-model",
    )
    assert post_kwargs["headers"]["Authorization"] == (
        f"Bearer {HUGGINGFACE_CREDENTIAL_CANARY}"
    )
    assert post_kwargs["json"]["inputs"] == (
        f"{HUGGINGFACE_PROMPT_CANARY}\n\n\nfixed input"
    )
    for canary in (
        HUGGINGFACE_CREDENTIAL_CANARY,
        HUGGINGFACE_PROMPT_CANARY,
        HUGGINGFACE_RESPONSE_CANARY,
    ):
        assert canary not in captured.text
    assert "HuggingFace: Credential configured" in captured.text
    assert "HuggingFace: Prompt prepared; character_count=" in captured.text
    assert "HuggingFace: API response received" in captured.text


def test_huggingface_stream_is_lazy_and_hides_rejected_events(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_config_loader(monkeypatch)
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_streaming_provider_settings(),
    )
    response = _FakeResponse(
        lines=(
            f"data: {json.dumps({'private': HUGGINGFACE_STREAM_CANARY})}".encode(),
            f"data: {{{HUGGINGFACE_STREAM_CANARY}".encode(),
            b'data: {"token":{"text":"fixed huggingface token"}}',
            b'data: {"generated_text":"fixed huggingface generated"}',
            b"data: [DONE]",
        )
    )
    post_calls = _install_signature_bound_general_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        stream = general_summarization.summarize_with_huggingface(
            "fixed-huggingface-key",
            "fixed input",
            "fixed prompt",
            streaming=HUGGINGFACE_PRIVATE_STREAMING_VALUE,
        )
        assert inspect.isgenerator(stream)
        assert response.iter_lines_started is False
        assert response.closed is False
        chunks = list(stream)

    assert chunks == ["fixed huggingface token", "fixed huggingface generated"]
    assert response.iter_lines_started is True
    assert response.closed is False
    assert len(post_calls) == 1
    assert post_calls[0][1]["json"]["stream"] == (HUGGINGFACE_PRIVATE_STREAMING_VALUE)
    assert post_calls[0][1]["stream"] is True
    assert HUGGINGFACE_STREAM_CANARY not in captured.text
    assert HUGGINGFACE_PRIVATE_STREAMING_VALUE not in captured.text
    assert "HuggingFace Stream: Response event rejected" in captured.text
    assert "HuggingFace Stream: JSON decode failed" in captured.text


def test_deepseek_stream_preserves_yields_and_hides_decode_and_key_failures(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_streaming_provider_settings(),
    )
    response = _FakeResponse(
        lines=(
            f"data: {{{DEEPSEEK_STREAM_CANARY}".encode(),
            f"data: {json.dumps({'choices': [{}], 'private': DEEPSEEK_STREAM_CANARY})}".encode(),
            b'data: {"choices":[{"delta":{"content":"fixed deepseek chunk"}}]}',
            b"data: [DONE]",
        )
    )
    post_calls = _install_signature_bound_general_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        stream = general_summarization.summarize_with_deepseek(
            DEEPSEEK_CREDENTIAL_CANARY,
            "fixed input",
            "fixed prompt",
            streaming=DEEPSEEK_PRIVATE_STREAMING_VALUE,
        )
        assert inspect.isgenerator(stream)
        assert response.iter_lines_started is False
        assert response.closed is False
        chunks = list(stream)

    assert chunks == ["fixed deepseek chunk", "fixed deepseek chunk"]
    assert response.iter_lines_started is True
    assert response.closed is False
    assert len(post_calls) == 1
    assert post_calls[0][1]["json"]["stream"] == DEEPSEEK_PRIVATE_STREAMING_VALUE
    assert post_calls[0][1]["stream"] is True
    for canary in (
        DEEPSEEK_CREDENTIAL_CANARY,
        DEEPSEEK_STREAM_CANARY,
        DEEPSEEK_PRIVATE_STREAMING_VALUE,
    ):
        assert canary not in captured.text
    assert "DeepSeek: Credential configured" in captured.text
    assert "DeepSeek Stream: JSON decode failed" in captured.text
    assert "DeepSeek Stream: Response event missing required field" in captured.text


def test_mistral_stream_preserves_yields_and_hides_rejected_events(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_streaming_provider_settings(),
    )
    response = _FakeResponse(
        lines=(
            f"data: {json.dumps({'private': MISTRAL_STREAM_CANARY})}".encode(),
            f"data: {{{MISTRAL_STREAM_CANARY}".encode(),
            f"data: {json.dumps({'choices': [{}], 'private': MISTRAL_STREAM_CANARY})}".encode(),
            b'data: {"choices":[{"delta":{"content":"fixed mistral chunk"}}]}',
            b"data: [DONE]",
        )
    )
    post_calls = _install_signature_bound_general_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        stream = general_summarization.summarize_with_mistral(
            MISTRAL_CREDENTIAL_CANARY,
            "fixed input",
            "fixed prompt",
            streaming=MISTRAL_PRIVATE_STREAMING_VALUE,
        )
        assert inspect.isgenerator(stream)
        assert response.iter_lines_started is False
        assert response.closed is False
        chunks = list(stream)

    assert chunks == ["fixed mistral chunk"]
    assert response.iter_lines_started is True
    assert response.closed is False
    assert len(post_calls) == 1
    assert post_calls[0][1]["json"]["stream"] == MISTRAL_PRIVATE_STREAMING_VALUE
    assert post_calls[0][1]["stream"] is True
    for canary in (
        MISTRAL_CREDENTIAL_CANARY,
        MISTRAL_STREAM_CANARY,
        MISTRAL_PRIVATE_STREAMING_VALUE,
    ):
        assert canary not in captured.text
    assert "Mistral: Credential configured" in captured.text
    assert "Mistral Stream: Response event rejected" in captured.text
    assert "Mistral Stream: JSON decode failed" in captured.text
    assert "Mistral Stream: Response event missing required field" in captured.text


@pytest.mark.parametrize(
    ("provider_name", "summarizer", "api_key", "response_canary", "expected"),
    [
        (
            "HuggingFace",
            general_summarization.summarize_with_huggingface,
            "fixed-huggingface-key",
            HUGGINGFACE_RESPONSE_CANARY,
            "HuggingFace: Failed to process summary. Status code: 503",
        ),
        (
            "DeepSeek",
            general_summarization.summarize_with_deepseek,
            "fixed-deepseek-key",
            DEEPSEEK_RESPONSE_CANARY,
            "DeepSeek: Failed to process summary. Status code: 503",
        ),
        (
            "Mistral",
            general_summarization.summarize_with_mistral,
            "fixed-mistral-key",
            MISTRAL_RESPONSE_CANARY,
            "Mistral: Failed to process summary. Status code: 503",
        ),
    ],
    ids=["huggingface", "deepseek", "mistral"],
)
def test_general_streaming_provider_status_failure_hides_response_body(
    provider_name: str,
    summarizer: Callable[..., object],
    api_key: str,
    response_canary: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_config_loader(monkeypatch)
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_streaming_provider_settings(),
    )
    response = _FakeResponse(status_code=503, text=response_canary)
    _install_signature_bound_general_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = summarizer(api_key, "fixed input", "fixed prompt")

    assert result == expected
    assert response_canary not in captured.text
    if provider_name == "HuggingFace":
        assert "HuggingFace: Summarization failed; status_code=503" in captured.text
    else:
        assert (
            f"{provider_name}: Summarization failed with status code 503"
            in captured.text
        )


@pytest.mark.parametrize(
    ("provider_name", "summarizer", "api_key", "exception_canary", "expected"),
    [
        (
            "HuggingFace",
            general_summarization.summarize_with_huggingface,
            "fixed-huggingface-key",
            HUGGINGFACE_EXCEPTION_CANARY,
            "HuggingFace: Error occurred while processing summary with HuggingFace: "
            f"{HUGGINGFACE_EXCEPTION_CANARY}",
        ),
        (
            "DeepSeek",
            general_summarization.summarize_with_deepseek,
            "fixed-deepseek-key",
            DEEPSEEK_EXCEPTION_CANARY,
            "DeepSeek: Error occurred while processing summary: "
            f"{DEEPSEEK_EXCEPTION_CANARY}",
        ),
        (
            "Mistral",
            general_summarization.summarize_with_mistral,
            "fixed-mistral-key",
            MISTRAL_EXCEPTION_CANARY,
            "Mistral: Error occurred while processing summary: "
            f"{MISTRAL_EXCEPTION_CANARY}",
        ),
    ],
    ids=["huggingface", "deepseek", "mistral"],
)
def test_general_streaming_provider_exception_hides_message_and_traceback(
    provider_name: str,
    summarizer: Callable[..., object],
    api_key: str,
    exception_canary: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_config_loader(monkeypatch)
    _install_signature_bound_general_settings(
        monkeypatch,
        _general_streaming_provider_settings(),
    )
    _install_signature_bound_general_session_post(
        monkeypatch,
        RuntimeError(exception_canary),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = summarizer(api_key, "fixed input", "fixed prompt")

    assert result == expected
    assert exception_canary not in captured.text
    assert (
        f"{provider_name}: Processing failed; exception_type=RuntimeError"
        in captured.text
    )
    assert not [record for record in captured.caplog.records if record.exc_info]


@pytest.mark.parametrize(
    ("provider_name", "summarizer", "settings_key", "expected"),
    [
        (
            "HuggingFace",
            general_summarization.summarize_with_huggingface,
            ("huggingface_api", "api_key"),
            "HuggingFace: Error occurred while processing summary with HuggingFace: "
            "No valid Anthropic API key available",
        ),
        (
            "DeepSeek",
            general_summarization.summarize_with_deepseek,
            ("deepseek_api", "api_key"),
            "DeepSeek: API Key Not Provided/Found in Config file or is empty",
        ),
        (
            "Mistral",
            general_summarization.summarize_with_mistral,
            ("mistral_api", "api_key"),
            "Mistral: API Key Not Provided/Found in Config file or is empty",
        ),
    ],
    ids=["huggingface", "deepseek", "mistral"],
)
def test_general_streaming_provider_missing_config_credential_is_truthful(
    provider_name: str,
    summarizer: Callable[..., object],
    settings_key: tuple[str, str],
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _general_streaming_provider_settings()
    settings[settings_key] = ""
    _install_signature_bound_general_config_loader(monkeypatch)
    settings_calls = _install_signature_bound_general_settings(monkeypatch, settings)
    post_calls = _install_signature_bound_general_session_post(
        monkeypatch,
        AssertionError("transport must not run without a credential"),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = summarizer("", "fixed input", "fixed prompt")

    assert result == expected
    assert settings_calls == [((settings_key[0], settings_key[1]), {})]
    assert post_calls == []
    assert f"{provider_name}: Credential configured" not in captured.text


def test_no_pending_general_tail_sites() -> None:
    private = [
        site
        for site in _ledger_sites()
        if site["group"] == "general_tail"
        and site["starting_classification"] == "private"
    ]

    assert len(private) == 20
    assert not [site for site in private if site["outcome"] == "pending"], (
        "general_tail has pending private diagnostics: "
        f"{[site['site_id'] for site in private if site['outcome'] == 'pending']}"
    )
    assert sum(site["outcome"] in {"metadata", "deleted"} for site in private) == 20


def test_google_success_hides_credential_input_prompt_and_response(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class PromptValue:
        calls = 0

        def __str__(self) -> str:
            self.calls += 1
            return GOOGLE_PROMPT_CANARY

    prompt = PromptValue()
    config_calls = _install_signature_bound_general_config_loader(monkeypatch)
    settings_calls = _install_signature_bound_general_settings(
        monkeypatch,
        _google_provider_settings(),
    )
    response = _FakeResponse(
        json_data={"choices": [{"message": {"content": GOOGLE_RESPONSE_CANARY}}]}
    )
    post_calls = _install_signature_bound_general_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_google(
            GOOGLE_CREDENTIAL_CANARY,
            GOOGLE_INPUT_CANARY,
            prompt,
            system_message="fixed system",
        )

    assert result == GOOGLE_RESPONSE_CANARY
    assert prompt.calls == 1
    assert config_calls == [((), {})]
    assert settings_calls == [
        (("google_api", "model", "gemini-1.5-pro"), {}),
        (("google_api", "api_retries", 3), {}),
        (("google_api", "api_retry_delay", 5), {}),
    ]
    assert len(post_calls) == 1
    post_args, post_kwargs = post_calls[0]
    assert post_args == ("https://generativelanguage.googleapis.com/v1beta/openai/",)
    assert post_kwargs["headers"]["Authorization"] == (
        f"Bearer {GOOGLE_CREDENTIAL_CANARY}"
    )
    assert post_kwargs["json"]["messages"][1]["content"] == (
        f"{GOOGLE_INPUT_CANARY} \n\n\n\n{GOOGLE_PROMPT_CANARY}"
    )
    for canary in (
        GOOGLE_CREDENTIAL_CANARY,
        GOOGLE_CREDENTIAL_CANARY[:5],
        GOOGLE_CREDENTIAL_CANARY[-5:],
        GOOGLE_INPUT_CANARY,
        GOOGLE_PROMPT_CANARY,
        GOOGLE_RESPONSE_CANARY,
    ):
        assert canary not in captured.text
    assert "Google: Credential configured" in captured.text
    assert "Google: Input prepared; character_count=" in captured.text
    assert "Google: Prompt prepared; character_count=" in captured.text
    assert "Google: Summary generated; character_count=" in captured.text


def test_google_model_lookup_precedes_prompt_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str] | str] = []

    class PromptValue:
        def __str__(self) -> str:
            events.append("prompt")
            return "fixed prompt"

    real_get_cli_setting = general_summarization.get_cli_setting
    signature = inspect.signature(real_get_cli_setting)
    settings = _google_provider_settings()

    def fake_get_cli_setting(*args: object, **kwargs: object) -> object:
        bound = signature.bind(*args, **kwargs)
        key = (bound.arguments["section"], bound.arguments.get("key"))
        events.append(key)
        if key in settings:
            return settings[key]
        return bound.arguments.get("default")

    monkeypatch.setattr(
        general_summarization,
        "get_cli_setting",
        fake_get_cli_setting,
    )
    _install_signature_bound_general_config_loader(monkeypatch)
    _install_signature_bound_general_session_post(
        monkeypatch,
        _FakeResponse(json_data={"choices": [{"message": {"content": "fixed"}}]}),
    )

    result = general_summarization.summarize_with_google(
        GOOGLE_CREDENTIAL_CANARY,
        "fixed input",
        PromptValue(),
        system_message="fixed system",
    )

    assert result == "fixed"
    assert events.index(("google_api", "model")) < events.index("prompt")


def test_google_missing_config_credential_does_not_claim_configuration(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _google_provider_settings()
    settings[("google_api", "api_key")] = ""
    _install_signature_bound_general_config_loader(monkeypatch)
    settings_calls = _install_signature_bound_general_settings(monkeypatch, settings)
    post_calls = _install_signature_bound_general_session_post(
        monkeypatch,
        AssertionError("transport must not run without a credential"),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_google(
            "",
            "fixed input",
            "fixed prompt",
        )

    assert result == "Google: API Key Not Provided/Found in Config file or is empty"
    assert settings_calls == [(("google_api", "api_key"), {})]
    assert post_calls == []
    assert "Google: Credential configured" not in captured.text


def test_google_stream_preserves_yields_and_hides_rejected_lines(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_config_loader(monkeypatch)
    _install_signature_bound_general_settings(
        monkeypatch,
        _google_provider_settings(),
    )
    response = _FakeResponse(
        lines=(
            f"data: {{{GOOGLE_STREAM_CANARY}".encode(),
            f"data: {json.dumps({'private': GOOGLE_STREAM_CANARY})}".encode(),
            b'data: {"choices":[{"delta":{"content":"fixed google chunk"}}]}',
            b"data: [DONE]",
        )
    )
    post_calls = _install_signature_bound_general_session_post(monkeypatch, response)

    with _capture_stdlib_and_loguru(caplog) as captured:
        stream = general_summarization.summarize_with_google(
            GOOGLE_CREDENTIAL_CANARY,
            "fixed input",
            "fixed prompt",
            streaming=GOOGLE_PRIVATE_STREAMING_VALUE,
        )
        assert inspect.isgenerator(stream)
        assert response.iter_lines_started is False
        chunks = list(stream)

    assert chunks == ["fixed google chunk"]
    assert response.iter_lines_started is True
    assert len(post_calls) == 1
    assert post_calls[0][1]["json"]["stream"] == GOOGLE_PRIVATE_STREAMING_VALUE
    assert post_calls[0][1]["stream"] is True
    for canary in (
        GOOGLE_CREDENTIAL_CANARY,
        GOOGLE_STREAM_CANARY,
        GOOGLE_PRIVATE_STREAMING_VALUE,
    ):
        assert canary not in captured.text
    assert "Google Stream: JSON decode failed" in captured.text
    assert "Google Stream: Response event missing required field" in captured.text


def test_google_status_failure_hides_response_body_and_preserves_return(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_config_loader(monkeypatch)
    _install_signature_bound_general_settings(
        monkeypatch,
        _google_provider_settings(),
    )
    _install_signature_bound_general_session_post(
        monkeypatch,
        _FakeResponse(status_code=503, text=GOOGLE_RESPONSE_CANARY),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_google(
            GOOGLE_CREDENTIAL_CANARY,
            "fixed input",
            "fixed prompt",
        )

    assert result == "Google: Failed to process summary. Status code: 503"
    assert GOOGLE_RESPONSE_CANARY not in captured.text
    assert "Google: Summarization failed with status code 503" in captured.text


def test_google_input_json_error_hides_detail_and_preserves_return(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    real_loads = general_summarization.json.loads
    signature = inspect.signature(real_loads)
    error = json.JSONDecodeError(GOOGLE_EXCEPTION_CANARY, "x", 0)

    def failing_loads(*args: object, **kwargs: object) -> object:
        signature.bind(*args, **kwargs)
        raise error

    _install_signature_bound_general_config_loader(monkeypatch)
    monkeypatch.setattr(general_summarization.json, "loads", failing_loads)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_google(
            GOOGLE_CREDENTIAL_CANARY,
            "{fixed invalid json",
            "fixed prompt",
        )

    assert result == f"Google: Error parsing JSON input: {error}"
    assert GOOGLE_EXCEPTION_CANARY not in captured.text
    assert "Google: JSON input parsing failed; exception_type=JSONDecodeError" in (
        captured.text
    )


def test_google_response_json_error_hides_detail_and_traceback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_config_loader(monkeypatch)
    _install_signature_bound_general_settings(
        monkeypatch,
        _google_provider_settings(),
    )
    error = json.JSONDecodeError(GOOGLE_EXCEPTION_CANARY, "x", 0)
    _install_signature_bound_general_session_post(
        monkeypatch,
        _FakeResponse(json_data=error),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_google(
            GOOGLE_CREDENTIAL_CANARY,
            "fixed input",
            "fixed prompt",
        )

    assert result == f"Google: Error decoding JSON input: {error}"
    assert GOOGLE_EXCEPTION_CANARY not in captured.text
    assert (
        "Google: JSON decoding failed; exception_type=JSONDecodeError" in captured.text
    )
    assert not [record for record in captured.caplog.records if record.exc_info]


def test_google_request_exception_hides_detail_and_traceback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_signature_bound_general_config_loader(monkeypatch)
    _install_signature_bound_general_settings(
        monkeypatch,
        _google_provider_settings(),
    )
    _install_signature_bound_general_session_post(
        monkeypatch,
        general_summarization.requests.RequestException(GOOGLE_EXCEPTION_CANARY),
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_google(
            GOOGLE_CREDENTIAL_CANARY,
            "fixed input",
            "fixed prompt",
        )

    assert result == f"Google: Error making API request: {GOOGLE_EXCEPTION_CANARY}"
    assert GOOGLE_EXCEPTION_CANARY not in captured.text
    assert (
        "Google: API request failed; exception_type=RequestException" in captured.text
    )
    assert not [record for record in captured.caplog.records if record.exc_info]


def test_google_unexpected_exception_hides_detail_and_traceback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    real_get_cli_setting = general_summarization.get_cli_setting
    signature = inspect.signature(real_get_cli_setting)

    def failing_setting(*args: object, **kwargs: object) -> object:
        bound = signature.bind(*args, **kwargs)
        if (
            bound.arguments["section"],
            bound.arguments.get("key"),
        ) == ("google_api", "model"):
            raise RuntimeError(GOOGLE_EXCEPTION_CANARY)
        return bound.arguments.get("default")

    _install_signature_bound_general_config_loader(monkeypatch)
    monkeypatch.setattr(
        general_summarization,
        "get_cli_setting",
        failing_setting,
    )

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_google(
            GOOGLE_CREDENTIAL_CANARY,
            "fixed input",
            "fixed prompt",
        )

    assert result == f"Google: Unexpected error occurred: {GOOGLE_EXCEPTION_CANARY}"
    assert GOOGLE_EXCEPTION_CANARY not in captured.text
    assert "Google: Processing failed; exception_type=RuntimeError" in captured.text
    assert not [record for record in captured.caplog.records if record.exc_info]


@pytest.mark.parametrize("streaming", [False, MOCK_PRIVATE_STREAMING_VALUE])
def test_mock_llm_hides_prompt_system_and_arbitrary_streaming_value(
    streaming: bool | str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(general_summarization.time, "sleep", lambda seconds: None)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_mock_llm(
            "fixed input",
            MOCK_PROMPT_CANARY,
            temp=0.2,
            system_message=MOCK_SYSTEM_CANARY,
            streaming=streaming,
        )
        if streaming:
            assert inspect.isgenerator(result)
            result = list(result)

    if streaming:
        assert result == ["Mocked summary for: fixed input..."]
    else:
        assert isinstance(result, str)
        assert f"Custom prompt: '{MOCK_PROMPT_CANARY}'" in result
        assert f"System message: '{MOCK_SYSTEM_CANARY}'" in result
    for canary in (
        MOCK_PROMPT_CANARY,
        MOCK_SYSTEM_CANARY,
        MOCK_PRIVATE_STREAMING_VALUE,
    ):
        assert canary not in captured.text
    assert "MOCK-LLM (MOCK): Request options prepared" in captured.text


def test_mock_llm_exception_hides_detail_and_traceback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def failing_sleep(seconds: float) -> None:
        assert seconds == 0.5
        raise RuntimeError(MOCK_EXCEPTION_CANARY)

    monkeypatch.setattr(general_summarization.time, "sleep", failing_sleep)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_with_mock_llm(
            "fixed input",
            "fixed prompt",
        )

    assert result == (
        f"Error: OpenAI mock function unexpected error: {MOCK_EXCEPTION_CANARY}"
    )
    assert MOCK_EXCEPTION_CANARY not in captured.text
    assert "OpenAI (MOCK): Processing failed; exception_type=RuntimeError" in (
        captured.text
    )
    assert not [record for record in captured.caplog.records if record.exc_info]


@pytest.mark.parametrize(
    ("analyze_result", "expected"),
    [
        (f"Error: {CHUNK_RESPONSE_CANARY}", None),
        ("fixed summary", "fixed summary"),
    ],
    ids=["error", "success"],
)
def test_summarize_chunk_string_result_hides_private_output(
    analyze_result: str,
    expected: str | None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    real_analyze = general_summarization.analyze
    signature = inspect.signature(real_analyze)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_analyze(*args: object, **kwargs: object) -> str:
        signature.bind(*args, **kwargs)
        calls.append((args, kwargs))
        return analyze_result

    monkeypatch.setattr(general_summarization, "analyze", fake_analyze)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_chunk(
            "fixed-provider",
            "fixed input",
            "fixed prompt",
            "fixed key",
            temp=0.2,
            system_message="fixed system",
        )

    assert result == expected
    assert len(calls) == 1
    assert CHUNK_RESPONSE_CANARY not in captured.text
    if analyze_result.startswith("Error:"):
        assert "Summarization failed; provider=fixed-provider" in captured.text


def test_summarize_chunk_stream_preserves_result_and_hides_error_chunk(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    real_analyze = general_summarization.analyze
    signature = inspect.signature(real_analyze)

    def fake_analyze(*args: object, **kwargs: object) -> Iterator[str]:
        signature.bind(*args, **kwargs)

        def stream() -> Iterator[str]:
            yield "fixed prefix"
            yield f"Error: {CHUNK_RESPONSE_CANARY}"

        return stream()

    monkeypatch.setattr(general_summarization, "analyze", fake_analyze)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_chunk(
            "fixed-provider",
            "fixed input",
            "fixed prompt",
            "fixed key",
        )

    assert result == f"Error: {CHUNK_RESPONSE_CANARY}"
    assert CHUNK_RESPONSE_CANARY not in captured.text
    assert "Streaming summarization failed; provider=fixed-provider" in captured.text


def test_summarize_chunk_exception_hides_detail_and_traceback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    real_analyze = general_summarization.analyze
    signature = inspect.signature(real_analyze)

    def failing_analyze(*args: object, **kwargs: object) -> object:
        signature.bind(*args, **kwargs)
        raise RuntimeError(CHUNK_EXCEPTION_CANARY)

    monkeypatch.setattr(general_summarization, "analyze", failing_analyze)

    with _capture_stdlib_and_loguru(caplog) as captured:
        result = general_summarization.summarize_chunk(
            "fixed-provider",
            "fixed input",
            "fixed prompt",
            "fixed key",
        )

    assert result is None
    assert CHUNK_EXCEPTION_CANARY not in captured.text
    assert (
        "Error in summarize_chunk; provider=fixed-provider exception_type=RuntimeError"
        in captured.text
    )
    assert not [record for record in captured.caplog.records if record.exc_info]


def test_general_tail_completes_general_module_private_inventory() -> None:
    general = [
        site
        for site in _ledger_sites()
        if site["module"].endswith("Summarization_General_Lib.py")
    ]

    assert len(general) == 281
    assert (
        sum(site["starting_classification"] == "reviewed_safe" for site in general)
        == 181
    )
    assert sum(site["outcome"] == "frozen" for site in general) == 181
    assert not [site for site in general if site["outcome"] == "pending"]
    assert sum(site["outcome"] in {"metadata", "deleted"} for site in general) == 100


def test_general_tail_complete_ledger_reconciles_without_private_sites() -> None:
    sites = _ledger_sites()

    assert len(sites) == 523
    assert (
        sum(site["starting_classification"] == "reviewed_safe" for site in sites) == 323
    )
    assert sum(site["outcome"] == "frozen" for site in sites) == 323
    assert sum(site["outcome"] in {"metadata", "deleted"} for site in sites) == 200
    assert not [site for site in sites if site["outcome"] == "pending"]

    deleted_count = sum(site["outcome"] == "deleted" for site in sites)
    discovered = []
    for module in MODULE_COUNTS:
        source = (REPO_ROOT / module).read_text(encoding="utf-8")
        discovered.extend(_guard().discover_diagnostic_calls(source, module=module))
    assert len(discovered) == 523 - deleted_count
    test_ledger_current_state_matches_sources()
