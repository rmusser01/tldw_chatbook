"""Regression tests for ``Summarization_General_Lib.analyze`` dispatch (task-3301 xhigh review round, F1).

The no-chunking direct dispatch used to live in the ``else`` of
``if CHUNKER_AVAILABLE:`` -- with the chunk lib importable (every normal
install) a plain ``analyze(api_name=..., input_data=...)`` call left
``final_result`` unassigned and returned
``'Error: Summarization failed unexpectedly.'`` WITHOUT making any API
call. That dead zone also swallowed ``process_pdf``'s per-chunk analysis
pass and ``process_document``'s ``auto_summarize``.

These tests pin the repaired structure: with the chunker available and no
chunking strategy requested, ``analyze`` must actually dispatch.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

import pytest

from tldw_chatbook.LLM_Calls import Summarization_General_Lib as sgl


@pytest.fixture(autouse=True)
def _require_chunker_available():
    """The bug only exists when the chunk lib imports -- pin the precondition.

    If this venv ever loses the chunking stack these tests would silently
    exercise the (previously working) fallback branch instead of the fixed
    one, so fail loudly rather than pass vacuously.
    """
    assert sgl.CHUNKER_AVAILABLE, (
        "chunking stack not importable in this venv; the F1 regression "
        "tests need CHUNKER_AVAILABLE=True to exercise the repaired branch"
    )


def _stub_summarizer(monkeypatch, response: str = "PROVIDER SUMMARY."):
    """Replace the openai chat-dispatch core with a signature-checked recorder.

    TASK-34895: openai routes through _summarize_via_chat_api, so that is
    the seam analyze()'s dispatch actually resolves; stubbing it stands in
    for the provider exactly the way the old per-provider function did.
    """
    real = sgl._summarize_via_chat_api
    sig = inspect.signature(real)
    calls: list[Dict[str, Any]] = []

    def fake_core(*args, **kwargs):
        # Bind against the real signature so the dispatch call shape can
        # never drift from the seam it stands in for.
        sig.bind(*args, **kwargs)
        calls.append({"args": args, "kwargs": kwargs})
        return response

    monkeypatch.setattr(sgl, "_summarize_via_chat_api", fake_core)
    return calls


def test_analyze_direct_path_dispatches_with_chunker_available(monkeypatch):
    """No chunking strategy + chunker importable must still dispatch."""
    calls = _stub_summarizer(monkeypatch)

    result = sgl.analyze(
        api_name="openai",
        input_data="Direct dispatch body text.",
        custom_prompt_arg="Summarize this.",
        api_key="sk-test-not-real",
    )

    assert result == "PROVIDER SUMMARY."
    assert calls, (
        "analyze() never reached the provider dispatch -- the direct path "
        "is dead again (F1 regression)"
    )


def test_analyze_direct_path_returns_provider_error_string(monkeypatch):
    """In-band provider errors surface verbatim (callers gate on 'Error:')."""
    _stub_summarizer(monkeypatch, response="Error: upstream said no")

    result = sgl.analyze(
        api_name="openai",
        input_data="Body.",
        custom_prompt_arg=None,
        api_key="sk-test-not-real",
    )

    assert result == "Error: upstream said no"


def test_analyze_mock_llm_end_to_end_no_mocks():
    """A fully unmocked pass through _dispatch_to_api's mock provider."""
    result = sgl.analyze(
        api_name="mock-llm",
        input_data="End to end body text for the mock provider.",
        custom_prompt_arg="Summarize.",
    )

    assert isinstance(result, str)
    assert result
    assert not result.startswith("Error:")


class TestDispatchNameAliases:
    """(F5 adjunct) `_dispatch_to_api` accepts the chat dispatcher's
    provider spellings for providers it implements, so the pdf/ebook/audio
    processors' analyze() calls work with the ingest seam's normalized
    names instead of failing with "Invalid API Name"."""

    @pytest.mark.parametrize(
        "chat_name, spec_key",
        [
            # TASK-34895: every implemented spelling now resolves to a
            # _SummarizerChatSpec key routed through the chat-dispatch
            # core (the per-provider functions are retired as dispatch
            # seams); the alias table's job is unchanged -- normalize the
            # chat dispatcher's spellings to the spec keys.
            ("koboldcpp", "kobold"),
            ("oobabooga", "ooba"),
            ("mistralai", "mistral"),
            ("llama_cpp", "llama.cpp"),
            ("local_ollama", "ollama"),
            ("local_vllm", "vllm"),
            ("local_llamacpp", "llama.cpp"),
            ("local_llamafile", "llama.cpp"),
            ("local_llm", "local-llm"),
        ],
    )
    def test_chat_dispatch_spellings_route_to_a_provider(
        self, monkeypatch, chat_name: str, spec_key: str
    ):
        real = sgl._summarize_via_chat_api
        sig = inspect.signature(real)
        routed: list[str] = []

        def fake_core(*args, **kwargs):
            # Bind against the real signature so the dispatch call shape
            # can never drift from the seam it stands in for.
            sig.bind(*args, **kwargs)
            routed.append(args[0])
            return "ALIASED SUMMARY."

        monkeypatch.setattr(sgl, "_summarize_via_chat_api", fake_core)

        result = sgl._dispatch_to_api(
            "Body text.",
            "Summarize.",
            chat_name,
            None,
            0.7,
            None,
            streaming=False,
        )

        assert result == "ALIASED SUMMARY."
        assert routed == [spec_key]

    def test_unknown_name_still_errors_in_band(self):
        result = sgl._dispatch_to_api(
            "Body text.",
            "Summarize.",
            "definitely-not-a-provider",
            None,
            0.7,
            None,
            streaming=False,
        )

        assert isinstance(result, str)
        assert result.startswith("Error: Invalid API Name")
