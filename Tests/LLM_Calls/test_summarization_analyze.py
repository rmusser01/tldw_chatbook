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
    """Replace the openai provider function with a signature-checked recorder."""
    real = sgl.summarize_with_openai
    sig = inspect.signature(real)
    calls: list[Dict[str, Any]] = []

    def fake_summarize_with_openai(*args, **kwargs):
        # Bind against the real signature so the dispatch call shape can
        # never drift from the seam it stands in for.
        sig.bind(*args, **kwargs)
        calls.append({"args": args, "kwargs": kwargs})
        return response

    monkeypatch.setattr(sgl, "summarize_with_openai", fake_summarize_with_openai)
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
        "chat_name, provider_func_name",
        [
            ("koboldcpp", "summarize_with_kobold"),
            ("oobabooga", "summarize_with_oobabooga"),
            ("mistralai", "summarize_with_mistral"),
            ("llama_cpp", "summarize_with_llama"),
            ("local_ollama", "summarize_with_ollama"),
            ("local_vllm", "summarize_with_vllm"),
            ("local_llamacpp", "summarize_with_llama"),
            ("local_llamafile", "summarize_with_llama"),
            ("local_llm", "summarize_with_local_llm"),
        ],
    )
    def test_chat_dispatch_spellings_route_to_a_provider(
        self, monkeypatch, chat_name: str, provider_func_name: str
    ):
        real = getattr(sgl, provider_func_name)
        sig = inspect.signature(real)
        calls: list[Dict[str, Any]] = []

        def fake_provider(*args, **kwargs):
            sig.bind(*args, **kwargs)
            calls.append({"args": args, "kwargs": kwargs})
            return "ALIASED SUMMARY."

        monkeypatch.setattr(sgl, provider_func_name, fake_provider)

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
        assert calls

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
