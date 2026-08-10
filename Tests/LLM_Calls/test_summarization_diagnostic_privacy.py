"""Stable, exhaustive guard for summarization diagnostic privacy review."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
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

from tldw_chatbook.LLM_Calls import Local_Summarization_Lib as local_summarization


REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "Tests/fixtures/summarization_diagnostic_review.json"
STARTING_PROJECTION_SHA256 = (
    "a4c9ba5f999199f02fd1c6186d1d88120f6d5f696071127ee192dff2c3503047"
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
    "general_mid": 23,
    "general_streaming": 20,
    "general_tail": 20,
}
PRIVATE_CATEGORY_COUNTS = {
    "response/output content": 71,
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
        "response/output content": 42,
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

    def json(self) -> object:
        if isinstance(self._json_data, BaseException):
            raise self._json_data
        return self._json_data

    def iter_lines(self) -> Iterator[bytes]:
        yield from self._lines


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response

    def mount(self, prefix: str, adapter: object) -> None:
        del prefix, adapter

    def post(self, *args: object, **kwargs: object) -> _FakeResponse:
        del args, kwargs
        return self.response


def _local_settings(*, llama_endpoint: str = "http://llama.invalid") -> dict[str, Any]:
    return {
        "llama_api": {
            "api_key": "fixed-llama-key",
            "api_ip": llama_endpoint,
            "temperature": 0.7,
            "max_tokens": 64,
            "streaming": False,
            "api_retries": 0,
            "api_retry_delay": 0,
        },
        "api_keys": {"kobold": "fixed-kobold-key"},
        "local_api_ip": {
            "kobold": "http://kobold.invalid/generate",
            "kobold_openai": "http://kobold.invalid/chat",
        },
        "kobold_api": {"api_retries": 0, "api_retry_delay": 0},
    }


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


def _invoke_local_input(monkeypatch: pytest.MonkeyPatch) -> object:
    response = _FakeResponse(
        json_data={"choices": [{"message": {"content": "  fixed summary  "}}]}
    )
    monkeypatch.setattr(
        local_summarization.requests, "post", lambda *args, **kwargs: response
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
        local_summarization.requests,
        "Session",
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
        local_summarization.requests, "post", lambda *args, **kwargs: response
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

    monkeypatch.setattr(local_summarization.requests, "post", raise_private_exception)
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
    invoke: Callable[[pytest.MonkeyPatch], object]
    assert_contract: Callable[[object], None]
    expected_event: str


RUNTIME_SENTINEL_CASES = (
    RuntimeSentinelCase(
        "local",
        "input",
        LOCAL_INPUT_CANARY,
        _invoke_local_input,
        _assert_fixed_summary,
        "Local LLM: Type of data:",
    ),
    RuntimeSentinelCase(
        "local",
        "prompt",
        LOCAL_PROMPT_CANARY,
        _invoke_local_prompt,
        _assert_fixed_llama_summary,
        "Llama Summarize: Prompt prepared; character_count=",
    ),
    RuntimeSentinelCase(
        "local",
        "credential",
        LOCAL_CREDENTIAL_CANARY,
        _invoke_local_credential,
        _assert_existing_kobold_summary,
        "Kobold: Credential state resolved",
    ),
    RuntimeSentinelCase(
        "local",
        "path",
        LOCAL_PATH_CANARY,
        _invoke_local_path,
        _assert_existing_llama_summary,
        "Llama: API endpoint configured",
    ),
    RuntimeSentinelCase(
        "local",
        "response",
        LOCAL_RESPONSE_CANARY,
        _invoke_local_response,
        _assert_empty_stream,
        "Local LLM: Failed to decode streamed JSON",
    ),
    RuntimeSentinelCase(
        "local",
        "exception",
        LOCAL_EXCEPTION_CANARY,
        _invoke_local_exception,
        _assert_local_exception_contract,
        "Local LLM: Processing failed; exception_type=RuntimeError",
    ),
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


def _ledger_sites() -> list[dict[str, object]]:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    assert ledger["schema_version"] == 1
    return ledger["sites"]


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


def test_ledger_retains_all_523_starting_sites() -> None:
    sites = _ledger_sites()

    assert len(sites) == 523
    assert len({site["site_id"] for site in sites}) == 523
    assert Counter(site["starting_classification"] for site in sites) == {
        "private": 199,
        "reviewed_safe": 324,
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


@pytest.mark.parametrize(
    "case",
    RUNTIME_SENTINEL_CASES,
    ids=lambda case: f"{case.module}-{case.category}",
)
def test_runtime_sentinel_hides_private_value(
    case: RuntimeSentinelCase,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _capture_stdlib_and_loguru(caplog) as captured:
        result = case.invoke(monkeypatch)
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
        local_summarization.requests,
        "Session",
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
        local_summarization.requests,
        "Session",
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
    settings["api_keys"]["kobold"] = None
    monkeypatch.setattr(local_summarization, "load_settings", lambda: settings)
    monkeypatch.setattr(
        local_summarization.requests,
        "Session",
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
        local_summarization.requests,
        "Session",
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
