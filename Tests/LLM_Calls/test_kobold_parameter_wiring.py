"""TASK-34844: kobold honors temp and system_message.

The parameters were accepted but never read (temperature hardcoded to 0.7,
system message silently dropped). Wiring pins: defaults preserve the
historical payload byte-for-byte; provided values are honored.
"""

from __future__ import annotations

import pytest

import tldw_chatbook.LLM_Calls.Local_Summarization_Lib as lsl
import tldw_chatbook.LLM_Calls.Summarization_General_Lib as sgl


class _FakeResponse:
    status_code = 200
    text = ""

    def json(self):
        return {"results": [{"text": " fixed summary "}]}


def _run_kobold(monkeypatch, *, temp=None, system_message=None):
    captured: dict = {}

    def fake_post(url, headers, payload, streaming, **kwargs):
        captured["payload"] = payload
        return _FakeResponse()

    monkeypatch.setattr(sgl, "_post_with_retry", fake_post)
    monkeypatch.setattr(
        lsl,
        "load_settings",
        lambda: {"api_settings": {"koboldcpp": {}}, "kobold_api": {}},
    )
    monkeypatch.setattr(
        lsl,
        "_resolve_local_provider_config",
        lambda cfg, modern, legacy: (
            {"api_url": "http://127.0.0.1:5001/api/v1/generate"},
            {"api_retries": 0, "api_retry_delay": 0},
        ),
    )
    result = lsl.summarize_with_kobold(
        "fixed input text",
        "fixed-key",
        "fixed instruction",
        system_message=system_message,
        temp=temp,
        streaming=False,
    )
    return result, captured["payload"]


def test_defaults_preserve_historical_payload(monkeypatch) -> None:
    result, payload = _run_kobold(monkeypatch)
    assert result == "fixed summary"
    assert payload["temperature"] == 0.7
    assert payload["prompt"] == "fixed instruction\n\n\n\nfixed input text"


def test_provided_temp_is_honored(monkeypatch) -> None:
    _, payload = _run_kobold(monkeypatch, temp=0.3)
    assert payload["temperature"] == 0.3


def test_provided_system_message_is_prepended(monkeypatch) -> None:
    _, payload = _run_kobold(monkeypatch, system_message="fixed system")
    assert payload["prompt"].startswith("fixed system\n\nfixed instruction")
    assert payload["prompt"].endswith("fixed input text")
