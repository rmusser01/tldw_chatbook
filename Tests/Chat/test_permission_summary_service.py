"""ADR-080: permission-summary config resolution."""

from types import SimpleNamespace

from tldw_chatbook.Chat import permission_summary_service as svc


def _config(section=None):
    return {"permission_summary": section if section is not None else {}}


def _ready(monkeypatch, ready=True, api_key="k"):
    monkeypatch.setattr(
        svc,
        "get_provider_readiness",
        lambda provider, config, environ=None: SimpleNamespace(
            ready=ready, api_key=api_key if ready else None
        ),
    )


def test_off_is_default_and_inactive():
    assert svc.resolve_permission_summary(_config()).mode == "off"
    assert svc.resolve_permission_summary(_config()).active is False


def test_invalid_mode_degrades_to_off():
    out = svc.resolve_permission_summary(_config({"mode": "sometimes"}))
    assert out.mode == "off" and out.active is False


def test_active_when_mode_provider_and_readiness_align(monkeypatch):
    _ready(monkeypatch)
    out = svc.resolve_permission_summary(
        _config({"mode": "fallback", "provider": "OpenAI", "model": "gpt-4o-mini"})
    )
    assert out.active is True
    assert out.mode == "fallback"
    assert out.dispatch_name == "openai"
    assert out.api_key == "k"
    assert out.model == "gpt-4o-mini"
    assert out.timeout_seconds == 4.0 and out.max_tokens == 120
    assert out.tail_max_chars == 4000


def test_missing_provider_or_unready_key_keeps_inactive(monkeypatch):
    _ready(monkeypatch, ready=False)
    assert (
        svc.resolve_permission_summary(_config({"mode": "always"})).active is False
    )
    _ready(monkeypatch)
    # no dispatchable handler for this spelling -> inactive
    out = svc.resolve_permission_summary(
        _config({"mode": "always", "provider": "not-a-chat-provider"})
    )
    assert out.active is False


def test_never_raises_on_junk_config():
    out = svc.resolve_permission_summary({"permission_summary": "junk"})
    assert out.active is False and out.mode == "off"


def test_explicit_api_key_and_system_prompt_override(monkeypatch):
    _ready(monkeypatch)
    out = svc.resolve_permission_summary(
        _config(
            {
                "mode": "always",
                "provider": "OpenAI",
                "api_key": "explicit",
                "system_prompt": "custom",
            }
        )
    )
    assert out.api_key == "explicit"
    assert out.system_prompt == "custom"


# ---------------------------------------------------------------------------
# tail / prompt / call (ADR-080 §4)
# ---------------------------------------------------------------------------

import json

from tldw_chatbook.Chat.permission_summary_service import (
    PermissionSummaryResolution as _Res,
    build_messages_tail,
    build_summary_messages,
    pending_calls_info_from_payload,
    summarize_pending_round,
)


_ACTIVE = _Res(mode="fallback", active=True, dispatch_name="openai",
               api_key="k", model="m")


def test_tail_keeps_user_assistant_text_only_and_budgeted():
    messages = [
        {"role": "system", "content": "secret system prompt"},
        {"role": "user", "content": "oldest " * 400},  # 2400 chars
        {"role": "assistant", "content": "middle"},
        {"role": "tool", "content": "TOOL RESULT FILE CONTENTS"},
        {"role": "user", "content": "newest"},
    ]
    tail = build_messages_tail(messages, 100)
    assert [m["role"] for m in tail] == ["assistant", "user"]
    assert tail[-1]["content"] == "newest"
    assert sum(len(m["content"]) for m in tail) <= 100 + len("middle")


def test_pending_calls_info_redacts_arguments():
    rows = [{
        "tool_name": "fs_write", "llm_name": "fs_write",
        "server_label": "Local", "description": "Writes files",
        "arguments": {"path": "a.txt", "api_key": "supersecret"},
    }]
    info = pending_calls_info_from_payload(rows)
    blob = json.dumps(info)
    assert "supersecret" not in blob
    assert info[0]["tool_name"] == "fs_write"
    assert "Writes files" in blob


def test_prompt_is_neutral_and_carries_context():
    msgs = build_summary_messages(
        [{"role": "user", "content": "please fix the config"}],
        [{"tool_name": "fs_write", "server_label": "Local",
          "description": "Writes files", "arguments_summary": '{"path":"a"}'}],
        "SYS",
    )
    assert msgs[0] == {"role": "system", "content": "SYS"}
    body = msgs[1]["content"]
    assert "please fix the config" in body
    assert "fs_write" in body and "Writes files" in body


def test_summarize_success_and_output_cap():
    calls = []

    def _call_fn(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "B" * 900}}]}

    out = summarize_pending_round(_ACTIVE, [{"role": "user", "content": "u"}],
                                  [{"tool_name": "t"}], call_fn=_call_fn)
    assert out is not None and len(out) == 240 and out.endswith("B")
    assert calls[0]["api_endpoint"] == "openai"
    assert calls[0]["streaming"] is False and calls[0]["request_retries"] == 0


def test_summarize_fails_open():
    def _boom(**kwargs):
        raise RuntimeError("provider down")

    assert summarize_pending_round(_ACTIVE, [], [{"tool_name": "t"}],
                                   call_fn=_boom) is None
    assert summarize_pending_round(_ACTIVE, [], [{"tool_name": "t"}],
                                   call_fn=lambda **k: {}) is None
    inactive = _Res(mode="off", active=False)
    assert summarize_pending_round(inactive, [], [{"tool_name": "t"}],
                                   call_fn=lambda **k: (_ for _ in ()).throw(
                                       AssertionError("must not call"))) is None


def test_settings_payload_validates_mode():
    from tldw_chatbook.Chat.permission_summary_service import (
        permission_summary_settings_payload,
    )

    out = permission_summary_settings_payload("fallback", " OpenAI ", "gpt-4o-mini")
    assert out == {
        "mode": "fallback", "provider": "OpenAI", "model": "gpt-4o-mini"
    }
    assert permission_summary_settings_payload("nonsense", "", "")["mode"] == "off"
