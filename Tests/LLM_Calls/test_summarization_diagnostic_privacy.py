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

from Tests.ast_shape import stable_dump
from scripts import check_persistent_diagnostic_inventory as diagnostic_inventory

import tldw_chatbook.LLM_Calls.recovery_review as recovery_review


def _is_lazy_provider_stream(value: object) -> bool:
    """Lazy-stream predicate for summarize_with_* streaming returns.

    TASK-32628's recovery admission retains every Iterator a provider call
    returns behind recovery_review._OpenAIStream, so the laziness contracts
    these tests pin must accept both the raw generator and the wrapper --
    the wrapper defers all iteration and close() to the wrapped stream.
    """
    return inspect.isgenerator(value) or isinstance(
        value, recovery_review._OpenAIStream
    )


from tldw_chatbook.LLM_Calls import (
    Summarization_General_Lib as general_summarization,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "Tests/fixtures/summarization_diagnostic_review.json"
INVENTORY_PATH = REPO_ROOT / "Docs/security/production-diagnostic-inventory.json"
# Ledger correction (task-3796 precedent) for dev's task-32805.3: the
# kobold/tabby streaming statements moved into the nested
# _kobold_stream/_tabby_stream generators; the nine affected sites'
# qualnames and starting occurrences follow the source (each move documented
# in the correction PR), so the canonical starting-projection digest changes
# from the task-3796-corrected 85a5c6b74f0c... value to the one below.
STARTING_PROJECTION_SHA256 = (
    "01aaf9ccf57420b45ad3bfc16104657d7544685a2e5aef75e9af2feee2ae2ea8"
)
MODULE_COUNTS = {
    "tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py": 242,
    "tldw_chatbook/LLM_Calls/Summarization_General_Lib.py": 281,
}
# TASK-34895: MODULE_COUNTS is the historical starting snapshot (the 523
# sites keep their module attribution). Only Summarization_General_Lib is
# live source now; Local_Summarization_Lib was deleted and its sites are
# recorded with outcome="deleted" + deletion_reason per the task-3796
# correction procedure.
LIVE_SCAN_MODULES = ["tldw_chatbook/LLM_Calls/Summarization_General_Lib.py"]
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
        self.iter_content_started = False
        self.closed = False

    def json(self) -> object:
        if isinstance(self._json_data, BaseException):
            raise self._json_data
        return self._json_data

    def iter_lines(self) -> Iterator[bytes]:
        self.iter_lines_started = True
        yield from self._lines

    def iter_content(self, chunk_size: int = 8192) -> Iterator[bytes]:
        # TASK-32853: the hosted_chat engine owns the response and decodes
        # SSE from iter_content chunks; the migrated summarize paths read
        # this seam instead of iter_lines.
        del chunk_size
        self.iter_content_started = True
        if self._lines:
            yield b"\n\n".join(self._lines) + b"\n\n"

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


def _invoke_general_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> object:
    del monkeypatch
    private_path = tmp_path / f"{GENERAL_PATH_CANARY}.txt"
    private_path.write_text("  fixed file contents  ", encoding="utf-8")
    return general_summarization.extract_text_from_input(str(private_path))


def _assert_general_input_contract(result: object) -> None:
    assert result == GENERAL_INPUT_CANARY


def _assert_general_path_contract(result: object) -> None:
    assert result == "fixed file contents"


@dataclass(frozen=True)
class RuntimeSentinelCase:
    module: str
    category: str
    canary: str
    invoke: Callable[[pytest.MonkeyPatch, Path], object]
    assert_contract: Callable[[object], None]
    expected_event: str


GENERAL_RUNTIME_SENTINEL_CASES = (
    # TASK-34895: the prompt/credential/response/exception cases drove the
    # deleted per-provider functions' transports; the input and path cases
    # pin surviving pure functions.
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
        "path",
        GENERAL_PATH_CANARY,
        _invoke_general_path,
        _assert_general_path_contract,
        "Input resolved as file path",
    ),
)

# TASK-34895: every LOCAL sentinel case drove a deleted
# Local_Summarization_Lib function; the general pure-function cases
# survive.
RUNTIME_SENTINEL_CASES = (*GENERAL_RUNTIME_SENTINEL_CASES,)


def _guard() -> ModuleType:
    try:
        return importlib.import_module("Tests.LLM_Calls.summarization_diagnostic_guard")
    except ModuleNotFoundError:
        pytest.fail("summarization diagnostic guard is not implemented")


def _message_shape(expression: str) -> str:
    node = ast.parse(f"logger.info({expression})").body[0]
    assert isinstance(node, ast.Expr)
    assert isinstance(node.value, ast.Call)
    return stable_dump(node.value.args[0], include_attributes=False)


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
    owners = normalized["owners"]
    assert isinstance(owners, list)
    for owner in owners:
        assert isinstance(owner, dict)
        if owner.get("path") in owned_paths:
            owner["call_count"] = "<task-3796-owned-call-count>"
            owner["diagnostic_digest"] = "<task-3796-owned-diagnostic-digest>"
    return normalized


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
        # TASK-32853 groq + anthropic migrations: the non-200 status log
        # site and the per-attempt network log site joined the deleted
        # ledger (10 -> 12). deepseek/mistral keep their frozen status
        # logs verbatim (bound to the typed error) per the freeze.
        # TASK-34650 review-B B26 removed the per-chunk "Content received"
        # INFO log (12 -> 13); stream completion stays frozen.
        "tldw_chatbook/LLM_Calls/Summarization_General_Lib.py": 13,
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
    generated_inventory["owners"].append(
        {
            "path": "tldw_chatbook/zz_unreviewed_drift.py",
            "owner": "TASK-494",
            "reason": "remaining Chatbook production diagnostic owner",
            "call_count": 1,
            "diagnostic_digest": "0" * 20,
        }
    )
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

    # TASK-34895 refreeze (task-3796 correction procedure): the deleted
    # per-provider code took its reviewed_safe sites with it (a frozen
    # diagnostic cannot carry outcome="deleted"), while every deleted-code
    # PRIVATE site stays recorded with outcome="deleted" + deletion_reason.
    assert len(sites) == 258
    assert len({site["site_id"] for site in sites}) == 258
    assert Counter(site["starting_classification"] for site in sites) == {
        "private": 200,
        "reviewed_safe": 58,
    }
    assert Counter(site["module"] for site in sites) == {
        "tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py": 100,
        "tldw_chatbook/LLM_Calls/Summarization_General_Lib.py": 158,
    }
    assert (
        Counter(
            site["group"]
            for site in sites
            if site["starting_classification"] == "private"
        )
        == PRIVATE_GROUP_COUNTS
    )

    identities = [_call_from_record(site, "starting").identity for site in sites]
    assert len(set(identities)) == 258
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
    for module in LIVE_SCAN_MODULES:
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

    # TASK-34895 refreeze: the remaining Local sites are the recorded
    # private deletions (outcome="deleted"); the module itself is gone.
    assert len(local) == 100
    assert (
        sum(site["starting_classification"] == "reviewed_safe" for site in local) == 0
    )
    assert sum(site["outcome"] == "frozen" for site in local) == 0
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
            assert _is_lazy_provider_stream(result)
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


def test_general_tail_completes_general_module_private_inventory() -> None:
    general = [
        site
        for site in _ledger_sites()
        if site["module"].endswith("Summarization_General_Lib.py")
    ]

    # TASK-34895 refreeze: 73 live diagnostics + recorded private
    # deletions from the removed per-provider functions.
    assert len(general) == 158
    assert (
        sum(site["starting_classification"] == "reviewed_safe" for site in general)
        == 58
    )
    assert sum(site["outcome"] == "frozen" for site in general) == 58
    assert not [site for site in general if site["outcome"] == "pending"]
    assert sum(site["outcome"] in {"metadata", "deleted"} for site in general) == 100


def test_general_tail_complete_ledger_reconciles_without_private_sites() -> None:
    sites = _ledger_sites()

    # TASK-34895 refreeze (258 retained sites; see the retains test).
    assert len(sites) == 258
    assert (
        sum(site["starting_classification"] == "reviewed_safe" for site in sites) == 58
    )
    assert sum(site["outcome"] == "frozen" for site in sites) == 58
    assert sum(site["outcome"] in {"metadata", "deleted"} for site in sites) == 200
    assert not [site for site in sites if site["outcome"] == "pending"]

    deleted_count = sum(site["outcome"] == "deleted" for site in sites)
    discovered = []
    for module in LIVE_SCAN_MODULES:
        source = (REPO_ROOT / module).read_text(encoding="utf-8")
        discovered.extend(_guard().discover_diagnostic_calls(source, module=module))
    assert len(discovered) == 73
    test_ledger_current_state_matches_sources()
