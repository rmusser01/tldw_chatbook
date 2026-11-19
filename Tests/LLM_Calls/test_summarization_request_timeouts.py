"""Summarization request timeouts (TASK-19560, retargeted by TASK-34895).

The original pins watched two summarizer-owned POSTs (SG's Anthropic post
and Local_Summarization_Lib's local-LLM post) for a `timeout=`. TASK-34895
deleted both modules' per-provider HTTP code: summarization now rides
chat_api_call, so the bounding contract lives where the requests are
actually made. Two invariants replace the old pins:

1. Summarization_General_Lib performs NO direct HTTP posts at all (the
   transport, retries, and timeouts are entirely the chat layer's
   contract, pinned by the chat-side suites) -- a timeout-less
   summarization post can no longer be *written* there.
2. The shared local chat sender (`_chat_with_openai_compatible_local_server`,
   which every local handler -- local-LLM included -- posts through) still
   bounds every post with the config-driven timeout.
"""

from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
GENERAL = REPO_ROOT / "tldw_chatbook" / "LLM_Calls" / "Summarization_General_Lib.py"
LOCAL_CHAT = REPO_ROOT / "tldw_chatbook" / "LLM_Calls" / "LLM_API_Calls_Local.py"


def test_summarization_module_makes_no_direct_http_posts():
    """Transport is fully delegated to the chat dispatch (TASK-34895)."""
    tree = ast.parse(GENERAL.read_text(encoding="utf-8"))
    offenders = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"post", "put", "request"}
    ]
    assert offenders == [], (
        f"Summarization_General_Lib performs direct HTTP calls ({offenders}); "
        "transport and timeouts belong to the chat layer"
    )


def test_shared_local_sender_posts_carry_a_timeout():
    """Every local chat post is bounded (local_llm.api_timeout et al)."""
    tree = ast.parse(LOCAL_CHAT.read_text(encoding="utf-8"))
    handler = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_chat_with_openai_compatible_local_server"
    )
    posts = [
        node
        for node in ast.walk(handler)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "post"
    ]
    assert posts, (
        "the shared local sender no longer posts directly -- repin at "
        "whichever seam now carries the URL"
    )
    for call in posts:
        kwargs = {keyword.arg for keyword in call.keywords}
        assert "timeout" in kwargs, (
            "a local chat post has no timeout; a stalled server hangs the "
            "summarization forever"
        )
