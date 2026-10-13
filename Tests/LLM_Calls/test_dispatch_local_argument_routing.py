"""TASK-34812: local summarizer dispatch routes arguments correctly.

Pins the exact (args, kwargs) each local provider function receives through
``_dispatch_to_api`` — the three routes here previously bound values to the
wrong parameters positionally (kobold: streaming onto ``kobold_api_ip``;
tabbyapi: ``api_key`` receiving the system message; ollama: the streaming
flag onto ``model``).
"""

from __future__ import annotations

import pytest

import tldw_chatbook.LLM_Calls.Summarization_General_Lib as sgl


def _recorder(name: str, calls: list) -> object:
    def fake(*args, **kwargs):
        calls.append((name, args, kwargs))
        return f"{name}-stub"

    return fake


@pytest.mark.parametrize(
    ("provider", "fn_name", "expected_args", "expected_kwargs"),
    [
        pytest.param(
            "kobold",
            "summarize_with_kobold",
            ("TEXT", "KEY", "PROMPT", "SYSTEM", 0.7),
            {"streaming": True},
            id="kobold",
        ),
        pytest.param(
            "tabbyapi",
            "summarize_with_tabbyapi",
            ("TEXT", "PROMPT", "SYSTEM", "KEY", 0.7),
            {"streaming": True},
            id="tabbyapi",
        ),
        pytest.param(
            "ollama",
            "summarize_with_ollama",
            ("TEXT", "PROMPT"),
            {
                "api_key": "KEY",
                "temp": 0.7,
                "system_message": "SYSTEM",
                "streaming": True,
            },
            id="ollama",
        ),
    ],
)
def test_local_dispatch_argument_routing(
    provider, fn_name, expected_args, expected_kwargs, monkeypatch
) -> None:
    calls: list = []
    monkeypatch.setattr(sgl, fn_name, _recorder(provider, calls))

    result = sgl._dispatch_to_api(
        "TEXT", "PROMPT", provider, "KEY", 0.7, "SYSTEM", True
    )

    assert result == f"{provider}-stub"
    assert calls == [(provider, expected_args, expected_kwargs)]
