"""The named summarization POSTs must be bounded (TASK-19560).

Both sites issued `requests.post(...)` with no timeout, so a server that
accepted the connection and then stalled hung the summarization forever --
no error, no cancellation, nothing in the log.

Scope note recorded deliberately: a static audit of these two modules found
**29** timeout-less `post`/`get` calls. This task's AC names two of them
(`Summarization_General_Lib.py`'s Anthropic post and
`Local_Summarization_Lib.py`'s local-LLM post) and those are what these tests
pin. The remaining 27 are reported in the task notes rather than fixed here:
bounding them one at a time invites an arbitrary partial job, and the right
answer is a session-level default, which is a design change deserving its own
review.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
GENERAL = REPO_ROOT / "tldw_chatbook" / "LLM_Calls" / "Summarization_General_Lib.py"
LOCAL = REPO_ROOT / "tldw_chatbook" / "LLM_Calls" / "Local_Summarization_Lib.py"


def _call_at(path: pathlib.Path, line: int) -> ast.Call:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.lineno == line:
            return node
    raise AssertionError(f"no call found at {path.name}:{line}")


def _find_post(path: pathlib.Path, url_fragment: str) -> ast.Call:
    """Locate a post by the URL it targets, not by line number.

    Line numbers drift; the endpoint is the stable identity.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "post"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if url_fragment in arg.value:
                    return node
    raise AssertionError(f"no post to {url_fragment!r} found in {path.name}")


@pytest.mark.parametrize(
    "path, fragment",
    [
        (GENERAL, "api.anthropic.com/v1/messages"),
        (LOCAL, "127.0.0.1:8080/v1/chat/completions"),
    ],
    ids=["anthropic-summarization", "local-llm-summarization"],
)
def test_named_summarization_posts_carry_a_timeout(path, fragment):
    call = _find_post(path, fragment)
    kwargs = {keyword.arg for keyword in call.keywords}
    assert "timeout" in kwargs, (
        f"{path.name}: the POST to {fragment} has no timeout; a stalled "
        "server hangs the summarization forever"
    )


def test_local_summarization_can_resolve_its_timeout_setting():
    """The timeout is config-driven, so the accessor must actually be imported.

    Caught during implementation: the setting was read via `get_cli_setting`
    in a module that never imported it, which would have raised NameError on
    the first real call -- a runtime failure no AST check would notice.
    """
    import tldw_chatbook.LLM_Calls.Local_Summarization_Lib as module

    assert callable(getattr(module, "get_cli_setting", None)), (
        "Local_Summarization_Lib reads get_cli_setting but does not import it"
    )
    assert int(module.get_cli_setting("local_llm", "api_timeout", 120)) > 0
