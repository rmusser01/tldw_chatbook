"""task-2116: a debug log missing its `f` prefix logs literal template text
instead of the values it was written to show.

Found in ``chat_with_anthropic`` (LLM_API_Calls.py): the log statement meant
to show the outgoing request payload (minus ``messages``) was a plain string
containing an unevaluated dict-comprehension -- ``logger.debug("... {k: v
for k, v in data.items() ...}")`` instead of an f-string. It printed the
literal source text on every call, never the actual payload, which cost real
diagnostic time during the cost-ticker live-provider verification (see the
task file). Sweeping the module found the SAME copy-pasted bug repeated for
every other cloud provider's "Request Payload" debug log (OpenAI, DeepSeek,
Google, Groq, Mistral, OpenRouter, Moonshot, Z.AI) -- nine hits total, all
fixed together.

Two tests:
    ``test_no_logging_call_has_an_unevaluated_brace_placeholder``: a static
        AST sweep of every module under ``tldw_chatbook/LLM_Calls/`` that
        would have caught all nine original hits (verified against the
        pre-fix source during development) and stays as a permanent
        regression guard against the same class of bug creeping back in via
        copy-paste.
    ``test_anthropic_debug_log_interpolates_payload_values``: a runtime
        check that ``chat_with_anthropic``'s specific log line now emits
        real payload values (AC#1), not the dict-comprehension source text,
        and that ``messages`` content -- the one thing the log's own name
        promises to exclude -- never leaks into it.
"""

from __future__ import annotations

import ast
import pathlib

LLM_CALLS_ROOT = (
    pathlib.Path(__file__).resolve().parents[2] / "tldw_chatbook" / "LLM_Calls"
)

_LOG_METHODS = {"debug", "info", "warning", "error", "critical", "trace", "success"}


def _logger_call_method(node: ast.Call) -> str | None:
    """Return the log-level method name for a ``logger.<method>(...)`` or
    ``logger.opt(...).<method>(...)`` call node, else ``None``."""
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr not in _LOG_METHODS:
        return None
    base = func.value
    if isinstance(base, ast.Name) and base.id == "logger":
        return func.attr
    if (
        isinstance(base, ast.Call)
        and isinstance(base.func, ast.Attribute)
        and isinstance(base.func.value, ast.Name)
        and base.func.value.id == "logger"
    ):
        return func.attr
    return None


def _brace_placeholder_hits(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return ``(lineno, text)`` for every logger call whose first argument
    is a plain (non-f) string literal containing a ``{...}`` placeholder --
    the exact shape of the missing-`f`-prefix bug this task fixes."""
    tree = ast.parse(path.read_text(), filename=str(path))
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if _logger_call_method(node) is None:
            continue
        first = node.args[0]
        # A real f-string parses as ast.JoinedStr, not ast.Constant -- only
        # a plain (mistakenly non-f) string literal reaches this branch.
        if (
            isinstance(first, ast.Constant)
            and isinstance(first.value, str)
            and "{" in first.value
            and "}" in first.value
            and len(node.args) == 1
            and not node.keywords
        ):
            hits.append((node.lineno, first.value[:120]))
    return hits


def test_brace_placeholder_guard_distinguishes_loguru_format_values(
    tmp_path: pathlib.Path,
) -> None:
    broken = tmp_path / "broken.py"
    broken.write_text('logger.error("status={status}")\n', encoding="utf-8")
    positional = tmp_path / "positional.py"
    positional.write_text('logger.error("status={}", status)\n', encoding="utf-8")
    keyword = tmp_path / "keyword.py"
    keyword.write_text(
        'logger.error("status={status}", status=status)\n', encoding="utf-8"
    )

    assert _brace_placeholder_hits(broken) == [(1, "status={status}")]
    assert _brace_placeholder_hits(positional) == []
    assert _brace_placeholder_hits(keyword) == []


def test_no_logging_call_has_an_unevaluated_brace_placeholder():
    all_hits: list[str] = []
    for path in sorted(LLM_CALLS_ROOT.rglob("*.py")):
        for lineno, text in _brace_placeholder_hits(path):
            all_hits.append(f"{path.relative_to(LLM_CALLS_ROOT)}:{lineno}: {text!r}")
    assert not all_hits, (
        "Logging call(s) with a '{...}' placeholder but no f-prefix -- these "
        "log literal template text instead of interpolated values:\n"
        + "\n".join(all_hits)
    )


def test_anthropic_debug_log_interpolates_payload_values(monkeypatch):
    """Confirm the request-payload log shows real values, not comprehension source text or message content.

    Args:
        monkeypatch: Pytest fixture used to stub config loading, the debug
            logger, and the outgoing HTTP session.
    """
    from tldw_chatbook.LLM_Calls import LLM_API_Calls

    debug_messages: list[str] = []
    monkeypatch.setattr(
        LLM_API_Calls,
        "load_settings",
        lambda: {"anthropic_api": {"api_base_url": "https://api.anthropic.test/v1"}},
    )
    monkeypatch.setattr(
        LLM_API_Calls.logger,
        "debug",
        lambda message, *args, **kwargs: debug_messages.append(str(message)),
    )

    class _CapturedSession:
        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return False

        def mount(self, *_args, **_kwargs):
            return None

        def post(
            self,
            url,
            *,
            headers=None,
            json=None,
            stream=False,
            timeout=None,
            allow_redirects=None,
        ):
            return _FakeResponse()

    class _FakeResponse:
        status_code = 200
        text = "{}"

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "id": "msg_test",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "hi back"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 4, "output_tokens": 5},
            }

    monkeypatch.setattr(LLM_API_Calls.requests, "Session", lambda: _CapturedSession())

    secret_user_text = "MY-VERY-PRIVATE-MESSAGE-CONTENT"
    LLM_API_Calls.chat_with_anthropic(
        input_data=[{"role": "user", "content": secret_user_text}],
        api_key="test-anthropic-key",
        model="claude-sonnet-4-6",
        streaming=False,
        max_tokens=64,
    )

    payload_lines = [
        message for message in debug_messages if "Request Payload" in message
    ]
    assert payload_lines, "expected the Anthropic request-payload debug log to fire"
    payload_line = payload_lines[0]

    # Real values, not the old literal template text.
    assert "claude-sonnet-4-6" in payload_line
    assert "for k, v in data.items()" not in payload_line
    assert "{k: v" not in payload_line

    # The log's own name promises "(excluding messages)" -- still true now
    # that it actually interpolates the dict.
    assert secret_user_text not in payload_line
