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
