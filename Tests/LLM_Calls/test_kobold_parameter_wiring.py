"""TASK-34844: kobold honors temp and system_message (rewired, TASK-34895).

The parameters were accepted but never read (temperature hardcoded to 0.7,
system message silently dropped). The original pins drove the deleted
``Local_Summarization_Lib.summarize_with_kobold``; summarization now routes
through the chat-dispatch core, so the wiring is pinned at the new seam:
the kobold spec's prompt order and the default/provided temperature and
system message reaching the chat handler (which prepends the system
message to kobold's native prompt, preserving the TASK-34844 composition).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

import tldw_chatbook.LLM_Calls.Summarization_General_Lib as sgl


class _KoboldCapture:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, endpoint, messages, **kwargs):
        self.calls.append({"endpoint": endpoint, "messages": messages, **kwargs})
        return {"choices": [{"message": {"content": " fixed summary "}}]}


@pytest.fixture
def kobold_capture(monkeypatch):
    capture = _KoboldCapture()
    monkeypatch.setattr("tldw_chatbook.Chat.Chat_Functions.chat_api_call", capture)

    def fake_setting(section, key, *args, **kwargs):
        if key == "api_key":
            return "cfg-key"
        return args[0] if args else kwargs.get("default")

    monkeypatch.setattr(sgl, "get_cli_setting", fake_setting)
    return capture


def _run_kobold(capture, *, temp=None, system_message=None):
    result = sgl._summarize_via_chat_api(
        "kobold",
        "fixed-key",
        "fixed input text",
        "fixed instruction",
        temp,
        system_message,
        False,
    )
    assert result == "fixed summary"
    assert len(capture.calls) == 1
    return capture.calls[0]


def test_defaults_preserve_historical_wiring(kobold_capture) -> None:
    call = _run_kobold(kobold_capture)
    spec = sgl._SUMMARIZER_CHAT_SPECS["kobold"]
    assert call["endpoint"] == "koboldcpp"
    assert call["temp"] == spec.default_temp
    # Default prompt order preserved: instruction-first with the \n\n\n\n glue.
    assert call["messages"][0]["content"] == (
        "fixed instruction\n\n\n\nfixed input text"
    )
    # Default system message reaches the handler (prepended to the native
    # prompt there, exactly as TASK-34844 established).
    assert call["system_message"] == spec.system_default


def test_provided_temp_is_honored(kobold_capture) -> None:
    call = _run_kobold(kobold_capture, temp=0.3)
    assert call["temp"] == 0.3


def test_provided_system_message_is_passed_through(kobold_capture) -> None:
    call = _run_kobold(kobold_capture, system_message="fixed system")
    assert call["system_message"] == "fixed system"


def test_analyze_default_system_message_reaches_the_seam(
    kobold_capture, monkeypatch
) -> None:
    """Composition pin (original review round 1): analyze()'s substituted
    default system message reaches the chat-dispatch seam, which the
    handler prepends to the kobold prompt."""
    from tldw_chatbook.Backup_Recovery import bootstrap

    monkeypatch.setattr(sgl, "CHUNKER_AVAILABLE", False)
    try:
        sgl.analyze(
            api_name="kobold",
            input_data="fixed input text",
            custom_prompt_arg="fixed instruction",
            api_key="fixed-key",
            temp=None,
            streaming=False,
        )
    except bootstrap.RecoveryRequired:
        # The registry default resolves through the config bootstrap, which
        # this machine's pre-existing recovery state refuses (the original
        # TASK-34844 pin had the same guard); the seam-level pins above
        # cover the composition without it.
        pytest.skip("config bootstrap in recovery state (known env)")
    assert kobold_capture.calls, "analyze never reached the chat seam"
    assert kobold_capture.calls[0]["system_message"]
