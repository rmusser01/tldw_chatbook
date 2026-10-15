"""TASK-34845: summarize_chunk passes arguments into analyze correctly.

The call previously rotated the first three arguments (text landing on
api_name) and swapped temp/system_message. Callers audited: production
paths dispatch through _dispatch_to_api directly; summarize_chunk is
exercised by the diagnostic-privacy suites, which bind by signature only.
"""

from __future__ import annotations

import tldw_chatbook.LLM_Calls.Summarization_General_Lib as sgl


def test_summarize_chunk_routes_arguments_correctly(monkeypatch) -> None:
    calls: list = []

    def fake_analyze(*args, **kwargs):
        calls.append((args, kwargs))
        return "fixed summary"

    monkeypatch.setattr(sgl, "analyze", fake_analyze)

    result = sgl.summarize_chunk(
        "fixed-provider",
        "fixed input",
        "fixed prompt",
        "fixed key",
        temp=0.4,
        system_message="fixed system",
    )

    assert result == "fixed summary"
    assert calls == [
        (
            ("fixed-provider", "fixed input", "fixed prompt"),
            {
                "api_key": "fixed key",
                "system_message": "fixed system",
                "temp": 0.4,
            },
        )
    ]
