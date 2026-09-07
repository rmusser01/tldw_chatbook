"""The reranker's system prompt must reach the model on EVERY provider
(TASK-17265).

The reranker used to build its instruction as an in-band
``{"role": "system"}`` entry inside ``messages_payload`` and never pass
``chat_api_call``'s own ``system_message=``. Two mainstream providers throw
an in-band system turn away:

* ``chat_with_anthropic`` builds its wire ``messages`` from user/assistant
  turns only and fills the top-level ``system`` field from its
  ``system_prompt`` parameter alone.
* ``chat_with_google`` maps ``user->user``/``assistant->model`` and
  ``continue``s past every other role, setting ``system_instruction`` only
  from its ``system_message`` parameter.

The lost text is the JSON contract ("return only a JSON object with a
'score' field"), so its absence turns billed calls into unscored rows.

**Where these tests observe.** AC#1 requires the ASSEMBLED PAYLOAD, "not
just the reranker's own call site", so the fake goes at the TRANSPORT seam
-- ``requests.Session.post``, the single call every one of these handlers
makes to put bytes on the wire. Everything above it is real: the reranker's
own assembly, ``chat_api_call``'s ``PROVIDER_PARAM_MAP`` translation, and
the provider handler's payload construction all execute, and what is
asserted is the exact JSON body that would have been sent. Faking
``chat_api_call`` (what the sibling seam guard in
``test_reranker_degraded_paths.py`` does, for its own different purpose)
could not see this bug at all: the dispatcher and the handler are precisely
where an in-band system turn goes missing. AC#4: no live call is made --
``Session.post`` never runs.

**Why these four providers.** They cover both mapping targets and both wire
shapes, because the MAPPING is what must be exercised, not a parameter name:

===========  ==========================  ==============================
provider     ``system_message`` maps to  lands on the wire as
===========  ==========================  ==============================
anthropic    ``system_prompt``           top-level ``system``
google       ``system_message``          ``system_instruction.parts``
openai       ``system_message``          an in-band ``role: system``
llama_cpp    ``system_prompt``           an in-band ``role: system``
===========  ==========================  ==============================

``llama_cpp`` additionally covers the keyless local family (it resolves no
credential and reaches the shared OpenAI-compatible local sender).
"""

import json
from typing import Any, Dict, List, Optional

import pytest
import requests

from tldw_chatbook.RAG_Search.reranker import PointwiseReranker, RerankingConfig
from tldw_chatbook.RAG_Search.simplified.vector_store import SearchResult

# The JSON contract the system prompt carries; if this sentence does not
# reach the model, the model has not been told to answer in JSON.
CONTRACT_SENTENCE = "Return only a JSON object with a 'score' field"

MODEL_REPLY = '{"score": 0.75}'


class _FakeResponse:
    """Just enough of ``requests.Response`` for the non-streaming paths."""

    def __init__(self, body: Dict[str, Any]):
        self._body = body
        self.status_code = 200
        self.headers: Dict[str, str] = {}
        self.text = json.dumps(body)
        self.content = self.text.encode()
        # Some handlers touch these in their `finally` cleanup.
        self.connection = None
        self.raw = None

    def json(self) -> Dict[str, Any]:
        return self._body

    def raise_for_status(self) -> None:
        return None

    def close(self) -> None:
        return None

    def iter_lines(self, *_args, **_kwargs):  # pragma: no cover - non-streaming
        return iter(())


def _openai_body() -> Dict[str, Any]:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "model": "fake-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": MODEL_REPLY},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _anthropic_body() -> Dict[str, Any]:
    return {
        "id": "msg_fake",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [{"type": "text", "text": MODEL_REPLY}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _google_body() -> Dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": MODEL_REPLY}], "role": "model"},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
    }


class _Snapshot:
    """Stand-in for ``get_runtime_config_snapshot()``."""

    def __init__(self, values: Dict[str, Any]):
        self.values = values


# A credential per keyed provider and a base URL for the keyless local one.
# These exist ONLY so the real handler runs far enough to assemble a payload;
# nothing is ever sent (``Session.post`` is faked).
_FAKE_SETTINGS: Dict[str, Any] = {
    "anthropic_api": {
        "api_key": "test-anthropic-key",
        "model": "claude-3-5-sonnet-20241022",
    },
    "openai_api": {"api_key": "test-openai-key", "model": "gpt-4o-mini"},
    "api_settings": {
        "anthropic": {"api_key": "test-anthropic-key"},
        "openai": {"api_key": "test-openai-key"},
        "google": {"api_key": "test-google-key", "model": "gemini-1.5-flash-latest"},
        "llama_cpp": {"api_url": "http://127.0.0.1:8080", "model": "local-model"},
    },
}


@pytest.fixture
def wire(monkeypatch) -> List[Dict[str, Any]]:
    """Fake the transport and the config lookups; return the posted bodies.

    Only ``requests.Session.post`` is replaced, so the reranker, the
    dispatcher and the provider handler all execute for real and the
    captured ``json=`` body is the literal request body.
    """
    from tldw_chatbook.LLM_Calls import LLM_API_Calls, LLM_API_Calls_Local

    snapshot = _Snapshot(_FAKE_SETTINGS)
    monkeypatch.setattr(LLM_API_Calls, "load_settings", lambda *a, **k: _FAKE_SETTINGS)
    monkeypatch.setattr(
        LLM_API_Calls, "get_runtime_config_snapshot", lambda *a, **k: snapshot
    )
    monkeypatch.setattr(
        LLM_API_Calls_Local, "load_settings", lambda *a, **k: _FAKE_SETTINGS
    )
    monkeypatch.setattr(
        LLM_API_Calls_Local, "get_runtime_config_snapshot", lambda *a, **k: snapshot
    )

    posts: List[Dict[str, Any]] = []

    def fake_post(_self, url, **kwargs):
        body = kwargs.get("json")
        posts.append({"url": url, "json": body})
        if "anthropic" in str(url):
            return _FakeResponse(_anthropic_body())
        if "generativelanguage" in str(url) or "googleapis" in str(url):
            return _FakeResponse(_google_body())
        return _FakeResponse(_openai_body())

    monkeypatch.setattr(requests.Session, "post", fake_post)
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(_openai_body()))
    return posts


def _rerank_once(
    provider: str, model: str, posts: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Run ONE real pointwise rerank against `provider` and return the body
    that would have gone on the wire."""
    import asyncio

    reranker = PointwiseReranker(
        RerankingConfig(
            model_provider=provider,
            model_name=model,
            top_k_to_rerank=1,
            retry_on_failure=False,
            cache_results=False,
        )
    )
    # The prompt under test is the SHIPPED registry default, populated by
    # PointwiseReranker.__init__ -- not a value this test invented.
    assert CONTRACT_SENTENCE in (reranker.config.system_prompt or "")

    results = [
        SearchResult(
            id="doc-0",
            score=0.5,
            document="quokkas are small macropods native to Western Australia",
            metadata={"doc_title": "quokka", "source_type": "media"},
        )
    ]
    outcome = asyncio.run(reranker.rerank("quokka habitat", results))

    assert len(posts) == 1, (
        f"expected exactly one request to {provider}; got {len(posts)} "
        f"(rerank outcome: failed={outcome.failed}/{outcome.total})"
    )
    assert outcome.failed == 0, (
        f"the {provider} round trip did not complete: "
        f"{outcome.failed}/{outcome.total} scorings failed"
    )
    return posts[0]["json"]


def _anthropic_system_text(body: Dict[str, Any]) -> Optional[str]:
    """Anthropic's top-level ``system`` is a string, or a list of text blocks
    when prompt caching adds a ``cache_control`` breakpoint."""
    system = body.get("system")
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "".join(
            block.get("text", "") for block in system if isinstance(block, dict)
        )
    return None


def _in_band_system_messages(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [m for m in body.get("messages", []) if m.get("role") == "system"]


# --------------------------------------------------------------------------
# AC#1 -- it ARRIVES on the two providers that drop an in-band system turn
# --------------------------------------------------------------------------


def test_anthropic_receives_the_system_prompt(wire):
    """Anthropic reads system text ONLY from the top-level ``system`` field."""
    body = _rerank_once("anthropic", "claude-3-5-sonnet-20241022", wire)

    system_text = _anthropic_system_text(body)
    assert system_text, (
        "the reranker's system prompt never reached Anthropic: the request "
        f"has no top-level 'system' field (payload keys: {sorted(body)})"
    )
    assert CONTRACT_SENTENCE in system_text
    # And it did not ALSO try to smuggle one in-band, where Anthropic
    # silently discards it.
    assert _in_band_system_messages(body) == []


def test_google_receives_the_system_prompt(wire):
    """Gemini reads system text ONLY from ``system_instruction``."""
    body = _rerank_once("google", "gemini-1.5-flash-latest", wire)

    instruction = body.get("system_instruction")
    assert instruction, (
        "the reranker's system prompt never reached Google: the request has "
        f"no 'system_instruction' (payload keys: {sorted(body)})"
    )
    text = "".join(part.get("text", "") for part in instruction.get("parts", []))
    assert CONTRACT_SENTENCE in text
    # Gemini's `contents` carries turns only; nothing system-shaped there.
    roles = [entry.get("role") for entry in body.get("contents", [])]
    assert "system" not in roles


# --------------------------------------------------------------------------
# AC#2 -- exactly one system instruction where a system turn IS accepted
# --------------------------------------------------------------------------


def test_openai_receives_exactly_one_system_instruction(wire):
    """OpenAI accepts an in-band system turn AND a ``system_message``; sending
    both is how a duplicate reaches the wire. Exactly one must arrive."""
    body = _rerank_once("openai", "gpt-4o-mini", wire)

    systems = _in_band_system_messages(body)
    assert len(systems) == 1, (
        f"OpenAI received {len(systems)} system messages, expected exactly 1: "
        f"{[m.get('content') for m in systems]}"
    )
    assert CONTRACT_SENTENCE in systems[0]["content"]
    assert [m.get("role") for m in body["messages"]] == ["system", "user"]


def test_a_local_provider_receives_exactly_one_system_instruction(wire):
    """The keyless local family (llama.cpp here) resolves no credential and
    reaches the shared OpenAI-compatible sender, which PREPENDS
    ``system_message`` to the messages it is given -- so a caller that also
    kept an in-band copy would put two on the wire."""
    body = _rerank_once("llama_cpp", "local-model", wire)

    systems = _in_band_system_messages(body)
    assert len(systems) == 1, (
        f"llama_cpp received {len(systems)} system messages, expected exactly "
        f"1: {[m.get('content') for m in systems]}"
    )
    assert CONTRACT_SENTENCE in systems[0]["content"]
    assert [m.get("role") for m in body["messages"]] == ["system", "user"]


# --------------------------------------------------------------------------
# AC#3 -- the dispatch itself carries the prompt in the transportable form
# --------------------------------------------------------------------------


def test_no_in_band_system_role_is_sent(wire, monkeypatch):
    """Observed at the dispatcher's own argument boundary, because "what the
    reranker SENDS" is not recoverable from the wire once a handler has
    normalised it: anthropic and google drop an in-band turn, and openai and
    the local family merge the two forms into one indistinguishable list.

    The spy DELEGATES to the real ``chat_api_call``, so the payload-boundary
    assertions above still hold on the same run: the messages list carries
    the user turn only, and the prompt travels as ``system_message=``, the
    one argument all 29 provider maps translate.
    """
    from tldw_chatbook.RAG_Search import reranker as reranker_module
    from tldw_chatbook.Chat.Chat_Functions import chat_api_call as real_call

    seen: List[Dict[str, Any]] = []

    def spy(*args, **kwargs):
        assert not args, f"chat_api_call must be called by keyword; got {args!r}"
        seen.append(dict(kwargs))
        return real_call(**kwargs)

    monkeypatch.setattr(reranker_module, "chat_api_call", spy)

    body = _rerank_once("openai", "gpt-4o-mini", wire)

    assert len(seen) == 1
    call = seen[0]
    roles = [m.get("role") for m in call["messages_payload"]]
    assert roles == ["user"], (
        f"the reranker still sends an in-band system turn: roles={roles}"
    )
    assert CONTRACT_SENTENCE in (call.get("system_message") or ""), (
        "the reranker did not pass system_message= (the only form every "
        f"provider map translates); kwargs sent: {sorted(call)}"
    )
    # ...and that single argument is what produced the wire's one system turn.
    assert len(_in_band_system_messages(body)) == 1
