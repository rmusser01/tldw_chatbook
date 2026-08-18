"""TASK-17165: the endpoint value is caller-supplied text, so it must never
be echoed verbatim into logs, metrics labels, or error strings.

The incident: `BaseReranker._call_llm_impl` passes its arguments to
`chat_api_call` POSITIONALLY in the wrong order (TASK-3502's fix wave,
TASK-17065), so an API KEY arrives as `api_endpoint`. Today that value is
echoed at INFO on EVERY call, stamped into a metrics LABEL, logged at
ERROR, and interpolated into two exception messages -- five sites, two of
which fire whether or not the call fails.

The defence is at the SINK (the loguru-diagnose lesson: fix where the
value is rendered, not where the bad caller lives), and it is an
ALLOWLIST: a recognised endpoint is safe by definition and prints
verbatim; anything else is unknown-provenance text and is redacted. A
blocklist of key-shaped patterns would miss any credential that does not
look like one.
"""

import logging

import pytest
from loguru import logger as loguru_logger

from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.Logging_Config import _forward_loguru_to_standard

SECRET = "REDACTED-abc123def456-not-a-real-key-value"

# CASE-INSENSITIVE on purpose: the sink lowercases the endpoint before
# logging it (`endpoint_lower`), so a case-sensitive substring check passes
# VACUOUSLY against every lowercased site -- which is exactly how the first
# draft of the metrics-label test in this file "passed" while the label
# still carried the whole value.


def _leaks(haystack: str) -> bool:
    return SECRET.lower() in str(haystack).lower()


@pytest.fixture
def loguru_caplog(caplog):
    """Bridge loguru records into caplog (the repo's standard pattern)."""
    sink_id = loguru_logger.add(_forward_loguru_to_standard, level="DEBUG")
    caplog.set_level(logging.DEBUG)
    try:
        yield caplog
    finally:
        loguru_logger.remove(sink_id)


def test_an_unrecognised_endpoint_never_reaches_a_log_record(loguru_caplog):
    with pytest.raises(Exception):
        chat_api_call(SECRET, [{"role": "user", "content": "q"}])

    for record in loguru_caplog.records:
        assert not _leaks(record.getMessage()), record.getMessage()


def test_an_unrecognised_endpoint_never_reaches_the_exception_text():
    with pytest.raises(Exception) as excinfo:
        chat_api_call(SECRET, [{"role": "user", "content": "q"}])

    assert not _leaks(str(excinfo.value))


def test_a_recognised_endpoint_still_prints_verbatim(loguru_caplog):
    """The allowlist must not make real diagnostics useless: a registered
    provider name is safe by definition and stays readable."""
    with pytest.raises(Exception):
        # No credentials configured in the test env -> it fails later, but
        # the ROUTING log has already happened by then.
        chat_api_call("openai", [{"role": "user", "content": "q"}])

    routing = [
        r.getMessage()
        for r in loguru_caplog.records
        if "Routing to endpoint" in r.getMessage()
    ]
    assert routing, "expected a routing log line"
    assert any("openai" in line for line in routing)


def test_the_metrics_label_carries_no_unrecognised_endpoint(monkeypatch):
    """The label is unbounded-cardinality AND a leak channel: a key-shaped
    endpoint would be stamped into exported metrics."""
    seen: list[dict] = []

    import tldw_chatbook.Chat.Chat_Functions as cf

    def _capture(name, value=1, labels=None, **kwargs):
        seen.append(dict(labels or {}))

    monkeypatch.setattr(cf, "log_counter", _capture)
    with pytest.raises(Exception):
        chat_api_call(SECRET, [{"role": "user", "content": "q"}])

    for labels in seen:
        for label_value in labels.values():
            assert not _leaks(label_value), labels


def test_the_metrics_label_is_bounded_not_merely_redacted(monkeypatch):
    """Qodo PR-1759: redacting the VALUE did not bound the CARDINALITY.

    The first fix reused one marker for logs and metrics, and that marker
    embeds `{len(text)} chars` -- useful in a log line, but it means N
    unrecognised endpoints of N different lengths still mint N distinct
    metric series. That is the exact hazard the fix's own comment named,
    reintroduced by the marker it introduced.

    The label must therefore come from a BOUNDED set: a registered endpoint
    name, or one constant. Logs keep the length.
    """
    seen: list[str] = []

    import tldw_chatbook.Chat.Chat_Functions as cf

    def _capture(name, value=1, labels=None, **kwargs):
        if labels and "api_endpoint" in labels:
            seen.append(str(labels["api_endpoint"]))

    monkeypatch.setattr(cf, "log_counter", _capture)

    # Twelve unrecognised endpoints, every one a different length.
    for n in range(3, 15):
        with pytest.raises(Exception):
            chat_api_call("x" * n, [{"role": "user", "content": "q"}])

    assert len(seen) == 12, f"expected one label per call, got {len(seen)}"
    assert len(set(seen)) == 1, (
        f"unbounded metrics cardinality: {len(set(seen))} distinct labels "
        f"for 12 unknown endpoints -- {sorted(set(seen))[:4]}"
    )
    assert not any(ch.isdigit() for ch in seen[0]), (
        f"the label still carries a caller-controlled number: {seen[0]!r}"
    )


def test_the_log_line_keeps_the_length_for_debugging(loguru_caplog):
    """Bounding the LABEL must not blind the LOG: the length is the one
    detail that tells an operator what shape of value arrived."""
    with pytest.raises(Exception):
        chat_api_call("abcdefghij", [{"role": "user", "content": "q"}])

    routing = [r.getMessage() for r in loguru_caplog.records
               if "Routing to endpoint" in r.getMessage()]
    assert routing, "expected a routing log line"
    assert any("10 chars" in line for line in routing), routing[:2]
