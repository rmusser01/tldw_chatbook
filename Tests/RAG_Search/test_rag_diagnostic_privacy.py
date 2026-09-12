"""No RAG_Search diagnostic carries the user's words (TASK-21700).

## What was found, and what the sink actually does

`rag_service.py` logged `f"[{correlation_id}] Cache hit for query:
'{query[:50]}...'"` at INFO. Search queries are user content, and the
persistent diagnostic inventory exists to keep exactly that class of value out
of persistent sinks.

Measured on this branch before deciding anything, because the severity turns on
it: **that record never reached the persistent file sink.** The shipped
`PrivateRotatingFileHandler` carries `PersistentDiagnosticFilter`, which is
default-DENY -- it admits a Chatbook record only when it carries the
`_tldw_metadata_only_record` marker, and only `log_persistent_metadata` sets
that. A loguru record crossing `_forward_loguru_to_standard` cannot carry it
(that function rebuilds `extra` from scratch, deliberately: if the marker
crossed the boundary, any code could `logger.bind(...)` its way past the
schema). A probe that installed the real sink, emitted this exact statement
from a real `RAG_Search.simplified` module through the real loguru bridge, and
read the file back found the query absent and a control `persist_event` record
present. The Logs screen's "Copy all"/"Copy visible" export reuses the same
`PersistentDiagnosticFilter` object and replaces every non-metadata message
body with `***REDACTED***`, so the clipboard route is closed too.

So the fix-at-the-sink prior art this repo already has (ADR-029's admission
boundary; `redact_log_line`'s home-path collapse for the in-app view) applies
here and was *already applied*. The residual exposure was the live terminal and
the in-app Logs view -- real, but narrower than "written to a file a user
attaches to a bug report". These call sites were changed anyway, for a reason
that stands on its own: the query text bought nothing. See
`content_fingerprint`'s docstring -- a 50-character prefix loses the identity a
maintainer reads the line for while keeping the words they must not read.

## What this module pins

* the two cache paths and the LLM reranker, exercised for real, emit no query
  or response text -- and still emit a usable handle;
* a census over the whole `RAG_Search` tree, so a future refactor cannot
  reintroduce the shape anywhere in it.

Deliberately NOT re-pinned here: the sink boundary itself, which
`Tests/test_remaining_diagnostic_sentinel_matrix.py` already covers for the
`rag_search` domain owner.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path
from typing import Callable, Iterator, List

import pytest
from loguru import logger as loguru_logger

from tldw_chatbook.Chat.Chat_Functions import chat_api_call as real_chat_api_call
from tldw_chatbook.RAG_Search import reranker as reranker_module
from tldw_chatbook.RAG_Search.reranker import PointwiseReranker, RerankingConfig
from tldw_chatbook.RAG_Search.simplified.simple_cache import SimpleRAGCache
from tldw_chatbook.RAG_Search.simplified.vector_store import SearchResult
from tldw_chatbook.Utils.log_sanitizer import (
    EMPTY_FINGERPRINT,
    content_fingerprint,
)


#: A query no other code in the suite can produce, so its presence in a captured
#: line is unambiguous. Long enough that a `[:50]` truncation would still leave
#: a recognisable fragment -- the head is checked separately for that reason.
SENTINEL_QUERY = (
    "ZZ-T21700-PRIVATE-QUERY my divorce settlement and psychiatric records "
    "for the deposition on the fourteenth"
)
SENTINEL_HEAD = SENTINEL_QUERY[:50]

#: The per-transition cache lines. Deliberately NOT a bare "Cache" prefix:
#: "Cache initialized" names no query and would drag an unrelated line into
#: the debuggability assertion below.
CACHE_EVENTS = (
    "Cache miss",
    "Cache hit",
    "Cache expired",
    "Cache entry expired",
    "Cached results",
)


@pytest.fixture
def captured_lines() -> Iterator[List[str]]:
    """Collect every loguru message emitted during the test.

    Adds a sink and removes only that sink id: a bare `logger.remove()` would
    tear down the sink `tldw_chatbook/__init__.py` installs and leak that
    teardown into unrelated tests.
    """
    lines: List[str] = []
    sink_id = loguru_logger.add(
        lambda message: lines.append(message.record["message"]),
        level="TRACE",
        format="{message}",
        diagnose=False,
    )
    try:
        yield lines
    finally:
        loguru_logger.remove(sink_id)


def _assert_no_query_text(lines: List[str]) -> None:
    joined = "\n".join(lines)
    assert SENTINEL_QUERY not in joined, (
        "a diagnostic carried the whole query:\n" + joined
    )
    assert SENTINEL_HEAD not in joined, (
        "a diagnostic carried a truncated prefix of the query -- truncation is "
        "not redaction:\n" + joined
    )
    # Belt and braces: any distinctive word from the query is a leak, even if
    # some future rewording no longer contains the exact prefix.
    for word in ("divorce", "psychiatric", "deposition"):
        assert word not in joined, (
            f"a diagnostic carried the query word {word!r}:\n" + joined
        )


# ---------------------------------------------------------------------------
# The cache paths, run for real
# ---------------------------------------------------------------------------


def test_sync_cache_diagnostics_carry_a_handle_but_no_query_text(captured_lines):
    """Miss, store, hit and expiry through the real synchronous cache."""
    cache = SimpleRAGCache(max_size=4, ttl_seconds=3600, enabled=True)

    assert cache.get(SENTINEL_QUERY, "semantic", 5) is None  # miss
    cache.put(SENTINEL_QUERY, "semantic", 5, [], "context")  # store
    assert cache.get(SENTINEL_QUERY, "semantic", 5) is not None  # hit

    expired = SimpleRAGCache(max_size=4, ttl_seconds=0, enabled=True)
    expired.put(SENTINEL_QUERY, "semantic", 5, [], "context")
    assert expired.get(SENTINEL_QUERY, "semantic", 5) is None  # expired

    _assert_no_query_text(captured_lines)
    fingerprint = content_fingerprint(SENTINEL_QUERY)
    # Debuggability is the other half of the contract: a line with the query
    # stripped and nothing put in its place is a worse diagnostic, not a safer
    # one. Every cache line that used to name the query must still identify it.
    named = [line for line in captured_lines if line.startswith(CACHE_EVENTS)]
    assert named, f"no cache diagnostics were emitted at all: {captured_lines}"
    for line in named:
        assert fingerprint in line, f"cache line lost its query handle: {line}"
        assert "key=" in line, f"cache line lost its cache-key handle: {line}"


def test_async_cache_diagnostics_carry_a_handle_but_no_query_text(captured_lines):
    """The same four transitions through `get_async`/`put_async`."""

    async def exercise() -> None:
        cache = SimpleRAGCache(max_size=4, ttl_seconds=3600, enabled=True)
        assert await cache.get_async(SENTINEL_QUERY, "hybrid", 5) is None
        await cache.put_async(SENTINEL_QUERY, "hybrid", 5, [], "context")
        assert await cache.get_async(SENTINEL_QUERY, "hybrid", 5) is not None

        expired = SimpleRAGCache(max_size=4, ttl_seconds=0, enabled=True)
        await expired.put_async(SENTINEL_QUERY, "hybrid", 5, [], "context")
        assert await expired.get_async(SENTINEL_QUERY, "hybrid", 5) is None

    asyncio.run(exercise())

    _assert_no_query_text(captured_lines)
    fingerprint = content_fingerprint(SENTINEL_QUERY)
    named = [line for line in captured_lines if line.startswith(CACHE_EVENTS)]
    assert named, f"no cache diagnostics were emitted at all: {captured_lines}"
    for line in named:
        assert fingerprint in line, f"cache line lost its query handle: {line}"


# ---------------------------------------------------------------------------
# The LLM reranker's parse-failure path, run for real
# ---------------------------------------------------------------------------


SENTINEL_RESPONSE = (
    "I cannot score this. ZZ-T21700-MODEL-REPLY The document describes the "
    "patient's diagnosis in detail, and the query asks about the settlement, "
    "so scoring would require disclosing both."
)


def _install_fake_provider(monkeypatch, responder: Callable[[list], str]) -> None:
    """Fake the reranker's single provider seam, BOUND to the real signature.

    Copied in spirit from `test_reranker_degraded_paths.py`: binding against
    `inspect.signature(chat_api_call)` and refusing positionals is what stops a
    fake from agreeing with a mis-ordered call (TASK-17065).
    """
    signature = inspect.signature(real_chat_api_call)

    def fake_chat_api_call(*args, **kwargs):
        assert not args, f"reranker must call chat_api_call by keyword: {args!r}"
        bound = signature.bind(*args, **kwargs)
        return responder(bound.arguments["messages_payload"])

    monkeypatch.setattr(reranker_module, "chat_api_call", fake_chat_api_call)


@pytest.mark.asyncio
async def test_reranker_parse_failure_reports_shape_not_model_reply(
    monkeypatch, captured_lines
):
    """A malformed reranker reply must be diagnosed without being quoted.

    The reply is user content by derivation: the scoring prompt hands the model
    the user's query, the document title, and 500 characters of the document
    body, so a non-JSON reply is usually prose about all three.
    """
    _install_fake_provider(monkeypatch, lambda _messages: SENTINEL_RESPONSE)
    reranker = PointwiseReranker(
        RerankingConfig(
            strategy="pointwise", top_k_to_rerank=2, retry_on_failure=False
        )
    )
    results = [
        SearchResult(
            id=f"doc-{i}",
            score=0.9 - 0.1 * i,
            document=f"body of document number {i}",
            metadata={"doc_title": f"doc-{i}", "source_type": "media"},
        )
        for i in range(2)
    ]

    await reranker.rerank(SENTINEL_QUERY, results)

    joined = "\n".join(captured_lines)
    assert "ZZ-T21700-MODEL-REPLY" not in joined, (
        "the model's reply was quoted into a diagnostic:\n" + joined
    )
    assert SENTINEL_RESPONSE[:200] not in joined
    for word in ("diagnosis", "settlement", "patient"):
        assert word not in joined, f"reply word {word!r} reached a diagnostic"

    failures = [line for line in captured_lines if "Failed to parse" in line]
    assert failures, f"the parse failure was not diagnosed at all: {captured_lines}"
    for line in failures:
        # The structural facts a maintainer reads this line for must survive.
        assert "JSONDecodeError" in line, f"lost the failure type: {line}"
        assert f"chars={len(SENTINEL_RESPONSE)}" in line, f"lost the length: {line}"
        assert content_fingerprint(SENTINEL_RESPONSE) in line, (
            f"lost the reply handle, so 'always the same bad body?' is now "
            f"unanswerable: {line}"
        )


# ---------------------------------------------------------------------------
# The census: the shape must not come back anywhere in RAG_Search
# ---------------------------------------------------------------------------


RAG_SEARCH_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "RAG_Search"

LOG_METHODS = frozenset(
    {
        "critical",
        "debug",
        "error",
        "exception",
        "info",
        "log",
        "success",
        "trace",
        "warning",
    }
)

#: Local names whose VALUE is user content wherever they appear in this tree.
#: Deliberately names only bare locals: `entry.document['id']` and
#: `doc.get('id')` are identifiers, not bodies, and flagging them would make
#: this census noise -- and a noisy census gets suppressed rather than read.
CONTENT_NAMES = frozenset(
    {
        "query",
        "response",
        "text",
        "content",
        "document",
        "chunk",
        "chunk_text",
        "prompt",
        "body",
        "snippet",
    }
)

#: Calls that reduce user content to a scalar safe to log. A risky name is
#: allowed only as a DIRECT argument of one of these.
#:
#: An ATTRIBUTE of a risky name still counts as a finding -- `response.text`,
#: `response.content` and `choices[0].message.content` are precisely how model
#: output escapes, and no attribute allowlist survives contact with the next
#: provider SDK. The cost is a false positive on genuinely scalar attributes
#: such as `response.status_code`; the escape hatch is to bind the scalar to a
#: differently-named local first (`status = response.status_code`), which is
#: what the line means anyway. Widening this set is not the fix.
SAFE_WRAPPERS = frozenset({"len", "content_fingerprint", "type", "id", "bool"})


def _logger_symbols(tree: ast.AST) -> set[str]:
    symbols = {"logger", "logging", "loguru_logger"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {"logging", "loguru"}:
            for alias in node.names:
                symbols.add(alias.asname or alias.name)
    return symbols


def _is_diagnostic(node: ast.Call, symbols: set[str]) -> bool:
    if not isinstance(node.func, ast.Attribute) or node.func.attr not in LOG_METHODS:
        return False
    receiver = node.func.value
    while isinstance(receiver, (ast.Call, ast.Attribute)):
        receiver = receiver.func if isinstance(receiver, ast.Call) else receiver.value
    return isinstance(receiver, ast.Name) and (
        receiver.id in symbols or receiver.id.casefold().endswith("logger")
    )


def _interpolated_expressions(node: ast.Call) -> list[ast.AST]:
    """Every expression whose VALUE is rendered into the emitted message."""
    expressions: list[ast.AST] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.FormattedValue):
            expressions.append(sub.value)
    # The FIRST argument is the message body, not just a format string, so it
    # is scanned like any other. `logger.error(query)` and `logger.info(
    # query[:50])` render the whole value -- the second is precisely the
    # truncation shape TASK-21700 was filed for, merely passed positionally
    # instead of interpolated, and an `args[1:]` scan let both through. A
    # literal format string is harmless here because it contains no `ast.Name`
    # at all; verified on the whole tree, this scans 0 -> 0 findings, so the
    # widening costs no false positive.
    #
    # loguru brace style and %-style put the remaining values in trailing args,
    # and `extra=` keywords are rendered by some handlers too.
    expressions.extend(node.args)
    expressions.extend(keyword.value for keyword in node.keywords)
    return expressions


def _unsafe_content_names(expression: ast.AST) -> set[str]:
    """Risky bare names in `expression` that are not inside a safe wrapper."""
    safe: set[int] = set()
    for sub in ast.walk(expression):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
            if sub.func.id in SAFE_WRAPPERS:
                safe.update(id(arg) for arg in sub.args)
    return {
        sub.id
        for sub in ast.walk(expression)
        if isinstance(sub, ast.Name)
        and sub.id in CONTENT_NAMES
        and id(sub) not in safe
    }


def rag_search_content_interpolations() -> list[tuple[str, int, str, list[str]]]:
    """Every RAG_Search diagnostic that renders user content verbatim.

    Exposed as a function rather than inlined so a mutation check can call it
    against a modified tree.
    """
    findings: list[tuple[str, int, str, list[str]]] = []
    for path in sorted(RAG_SEARCH_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        symbols = _logger_symbols(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_diagnostic(node, symbols):
                continue
            names: set[str] = set()
            for expression in _interpolated_expressions(node):
                names |= _unsafe_content_names(expression)
            if names:
                findings.append(
                    (
                        path.as_posix(),
                        node.lineno,
                        " ".join((ast.get_source_segment(source, node) or "").split()),
                        sorted(names),
                    )
                )
    return findings


def test_census_actually_scans_the_tree() -> None:
    """A census that scanned nothing would pass vacuously.

    `pytest "no tests ran"` has an exact analogue here: an empty finding list
    is the PASS condition, so the scanner has to be shown to have work to do.
    """
    total = 0
    for path in sorted(RAG_SEARCH_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        symbols = _logger_symbols(tree)
        total += sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _is_diagnostic(node, symbols)
        )
    assert total > 300, f"the census found only {total} diagnostics; it is not scanning"


def test_no_rag_search_diagnostic_interpolates_user_content() -> None:
    """The regression guard. Every hit is a query/body/reply rendered verbatim.

    To fix a failure: keep the diagnostic, drop the words. `len(value)` and
    `content_fingerprint(value)` are both accepted, and between them they carry
    what these lines are actually read for -- how big, and is it the same one
    as last time.
    """
    findings = rag_search_content_interpolations()
    assert not findings, "user content rendered into RAG_Search diagnostics:\n" + "\n".join(
        f"  {path}:{line} {names}\n      {text}"
        for path, line, text, names in findings
    )


def test_census_detects_the_shape_it_exists_to_catch() -> None:
    """The census must fail on the exact statement TASK-21700 was filed for.

    Without this, a scanner bug that matched nothing would leave the guard
    green forever -- the failure mode of every census test.
    """
    module = ast.parse(
        "from loguru import logger\n"
        "def f(query, correlation_id):\n"
        "    logger.info(f\"[{correlation_id}] Cache hit for query: "
        "'{query[:50]}...'\")\n"
    )
    symbols = _logger_symbols(module)
    calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and _is_diagnostic(node, symbols)
    ]
    assert len(calls) == 1
    names: set[str] = set()
    for expression in _interpolated_expressions(calls[0]):
        names |= _unsafe_content_names(expression)
    assert names == {"query"}


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # The accepted replacements must NOT be flagged, or the guard would
        # push people back towards truncation.
        ("logger.info(f'{content_fingerprint(query)}')", set()),
        ("logger.info(f'{len(query)}')", set()),
        ("logger.info(f'{len(query)} {content_fingerprint(query)}')", set()),
        # ...and every way of rendering the value itself must be.
        ("logger.info(f'{query}')", {"query"}),
        ("logger.info(f'{query[:50]}')", {"query"}),
        ("logger.info(f'{query!r}')", {"query"}),
        ("logger.info('q=%s', query)", {"query"}),
        ("logger.info('q={}', response)", {"response"}),
        ("logger.debug(f'{query.strip()}')", {"query"}),
        # The message argument itself. loguru renders `args[0]` as the body,
        # so these leak exactly as hard as the f-string forms above -- and an
        # `args[1:]` scan passed every one of them.
        ("logger.error(query)", {"query"}),
        ("logger.warning(response)", {"response"}),
        ("logger.exception(text)", {"text"}),
        # ...including the precise truncation shape TASK-21700 was filed for,
        # merely passed positionally instead of interpolated.
        ("logger.info(query[:50])", {"query"}),
        ("logger.debug('prefix ' + query)", {"query"}),
        # `logger.log` takes the level first, so the body is args[1]; both
        # positions must be scanned for this to be caught.
        ("logger.log('INFO', query)", {"query"}),
        # A literal format string stays harmless: it contains no name at all.
        ("logger.info('Cache hit for query')", set()),
        ("logger.info('q=%s', len(query))", set()),
    ],
)
def test_census_classifier_boundaries(source: str, expected: set[str]) -> None:
    module = ast.parse("from loguru import logger\n" + source)
    symbols = _logger_symbols(module)
    call = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and _is_diagnostic(node, symbols)
    )
    names: set[str] = set()
    for expression in _interpolated_expressions(call):
        names |= _unsafe_content_names(expression)
    assert names == expected


# ---------------------------------------------------------------------------
# The replacement handle itself
# ---------------------------------------------------------------------------


def test_content_fingerprint_carries_no_plaintext_and_is_stable() -> None:
    fingerprint = content_fingerprint(SENTINEL_QUERY)
    assert fingerprint == content_fingerprint(SENTINEL_QUERY), "not stable"
    assert len(fingerprint) == 12
    assert all(char in "0123456789abcdef" for char in fingerprint)
    for word in ("divorce", "psychiatric", "ZZ-T21700"):
        assert word not in fingerprint
    assert content_fingerprint("a") != content_fingerprint("b")
    # A prefix collision is exactly the case truncation could not distinguish.
    long_a = SENTINEL_HEAD + "alpha"
    long_b = SENTINEL_HEAD + "bravo"
    assert content_fingerprint(long_a) != content_fingerprint(long_b)


def test_content_fingerprint_distinguishes_empty_from_a_real_value() -> None:
    assert content_fingerprint("") == EMPTY_FINGERPRINT
    assert content_fingerprint(None) == EMPTY_FINGERPRINT
    assert content_fingerprint("x") != EMPTY_FINGERPRINT
    # Non-strings must not raise: a diagnostic that crashes is worse than one
    # that leaks, and these sites run inside `except` blocks.
    assert content_fingerprint(12345) == content_fingerprint("12345")
