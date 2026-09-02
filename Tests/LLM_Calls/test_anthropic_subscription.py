"""TASK-26022: read-only Claude subscription credential borrow for Anthropic."""

from __future__ import annotations

import json
import time

import pytest

from tldw_chatbook.LLM_Calls.anthropic_subscription import (
    SubscriptionCredential,
    anthropic_auth_source,
    read_claude_code_credential,
    subscription_headers,
)


def _write_cred(path, token="sk-ant-oat01-" + "a" * 40, expires_in_ms=3_600_000):
    path.write_text(json.dumps({
        "claudeAiOauth": {
            "accessToken": token,
            "refreshToken": "sk-ant-ort01-" + "b" * 40,
            "expiresAt": int(time.time() * 1000) + expires_in_ms,
            "scopes": ["user:inference"],
            "subscriptionType": "max",
        }
    }))
    return path


# --- reading (read-only, never raises outward) ---

def test_missing_file_is_none(tmp_path):
    assert read_claude_code_credential(tmp_path / "nope.json") is None


def test_malformed_file_is_none(tmp_path):
    p = tmp_path / "creds.json"
    p.write_text("{not json")
    assert read_claude_code_credential(p) is None
    p.write_text(json.dumps({"claudeAiOauth": {"accessToken": ""}}))
    assert read_claude_code_credential(p) is None


def test_good_credential_parses(tmp_path):
    cred = read_claude_code_credential(_write_cred(tmp_path / "c.json"))
    assert cred is not None
    assert cred.access_token.startswith("sk-ant-oat01-")
    assert cred.expired is False
    assert cred.subscription_type == "max"


def test_expired_credential_flagged_not_dropped(tmp_path):
    cred = read_claude_code_credential(_write_cred(tmp_path / "c.json", expires_in_ms=-60_000))
    assert cred is not None and cred.expired is True


# --- the token never leaks through repr/str (AC#3) ---

def test_credential_repr_masks_token(tmp_path):
    cred = read_claude_code_credential(_write_cred(tmp_path / "c.json"))
    for rendered in (repr(cred), str(cred)):
        assert "sk-ant-oat01-" not in rendered
        assert "a" * 20 not in rendered


# --- headers (AC#7's shape; live-verified by the owner before close) ---

def test_subscription_headers_shape(tmp_path):
    cred = read_claude_code_credential(_write_cred(tmp_path / "c.json"))
    headers = subscription_headers(cred)
    assert headers["authorization"] == f"Bearer {cred.access_token}"
    assert "oauth" in headers["anthropic-beta"]
    assert "x-api-key" not in headers


# --- explicit opt-in (AC#4/#6) ---

def test_auth_source_defaults_to_api_key():
    assert anthropic_auth_source({}) == "api_key"
    assert anthropic_auth_source({"auth_source": "api_key"}) == "api_key"
    assert anthropic_auth_source({"auth_source": "claude_subscription"}) == "claude_subscription"
    # junk falls back to the safe default
    assert anthropic_auth_source({"auth_source": "yolo"}) == "api_key"


# --- log sanitizer covers the OAuth token shape (AC#3) ---

def test_log_sanitizer_redacts_oauth_tokens():
    from tldw_chatbook.Utils.log_sanitizer import sanitize_string
    tok = "sk-ant-oat01-" + "Q" * 40
    assert tok not in sanitize_string(f"header was Bearer {tok} oops")


# --- chat_with_anthropic wiring (AC#1/#2/#4/#6) ---

class _FakeResp:
    status_code = 200
    def json(self):
        return {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1}, "model": "m"}
    def raise_for_status(self):
        return None
    text = "{}"


def _call_anthropic(monkeypatch, tmp_path, auth_source, cred_kwargs=None, api_key=None):
    """Invoke chat_with_anthropic with a captured transport; return headers."""
    from tldw_chatbook.LLM_Calls import LLM_API_Calls as mod

    captured = {}
    cred_path = tmp_path / "creds.json"
    if cred_kwargs is not None:
        _write_cred(cred_path, **cred_kwargs)

    # point the reader at the tmp credential + count reads (AC#6)
    reads = {"n": 0}
    real_read = mod.read_claude_code_credential
    def counting_read(path=None):
        reads["n"] += 1
        return real_read(cred_path)
    monkeypatch.setattr(mod, "read_claude_code_credential", counting_read)

    monkeypatch.setattr(
        mod, "load_settings",
        lambda *a, **k: {"anthropic_api": {"auth_source": auth_source, "model": "claude-sonnet-5"}},
    )

    class _Session:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def mount(self, *a, **k): pass
        def post(self, url, headers=None, json=None, data=None, stream=False, timeout=None, **kw):
            captured["headers"] = headers
            captured["url"] = url
            return _FakeResp()
    monkeypatch.setattr(mod, "create_default_session", lambda: _Session())

    result = mod.chat_with_anthropic(
        input_data=[{"role": "user", "content": "hi"}],
        model="claude-sonnet-5",
        api_key=api_key,
        streaming=False,
    )
    return captured, reads


def test_subscription_mode_sends_bearer_not_api_key(monkeypatch, tmp_path):
    """AC#1 + header shape (AC#7's static half)."""
    captured, _ = _call_anthropic(monkeypatch, tmp_path, "claude_subscription", cred_kwargs={})
    h = captured["headers"]
    assert h["authorization"].startswith("Bearer sk-ant-oat01-")
    assert "x-api-key" not in h
    assert "oauth" in h["anthropic-beta"]


def test_subscription_mode_expired_fails_with_refresh_message(monkeypatch, tmp_path):
    """AC#2: clear message, no request, no silent API-key fallback."""
    from tldw_chatbook.Chat.Chat_Deps import ChatConfigurationError
    with pytest.raises(ChatConfigurationError) as exc:
        _call_anthropic(monkeypatch, tmp_path, "claude_subscription",
                        cred_kwargs={"expires_in_ms": -1000}, api_key="sk-ant-api03-x")
    assert "Claude Code" in str(exc.value)


def test_subscription_mode_missing_credential_fails_clearly(monkeypatch, tmp_path):
    from tldw_chatbook.Chat.Chat_Deps import ChatConfigurationError
    with pytest.raises(ChatConfigurationError) as exc:
        _call_anthropic(monkeypatch, tmp_path, "claude_subscription", cred_kwargs=None)
    assert "Claude Code" in str(exc.value)


def test_default_mode_never_reads_credential_and_uses_api_key(monkeypatch, tmp_path):
    """AC#4/#6: a credential on disk changes nothing unless opted in."""
    captured, reads = _call_anthropic(
        monkeypatch, tmp_path, "api_key", cred_kwargs={}, api_key="sk-ant-api03-realkey"
    )
    assert captured["headers"]["x-api-key"] == "sk-ant-api03-realkey"
    assert "authorization" not in captured["headers"]
    assert reads["n"] == 0, "default mode must not even read the credential file"


# --- readiness reports the credential source (AC#5) ---

def _readiness(monkeypatch, tmp_path, auth_source, cred_kwargs):
    from tldw_chatbook.Chat import provider_readiness as pr

    cred_path = tmp_path / "creds.json"
    if cred_kwargs is not None:
        _write_cred(cred_path, **cred_kwargs)
    monkeypatch.setattr(
        pr, "read_claude_code_credential", lambda path=None: read_claude_code_credential(cred_path)
    )
    app_config = {"api_settings": {"anthropic": {"auth_source": auth_source}}}
    return pr.get_provider_readiness("Anthropic", app_config, environ={})


def test_readiness_subscription_ready_and_labeled(monkeypatch, tmp_path):
    r = _readiness(monkeypatch, tmp_path, "claude_subscription", {})
    assert r.ready is True
    assert r.api_key is None, "readiness must never carry the borrowed token"
    assert r.api_key_source == "subscription:claude_code"
    assert "subscription" in r.reason.lower()


def test_readiness_subscription_expired_blocked_with_refresh_copy(monkeypatch, tmp_path):
    r = _readiness(monkeypatch, tmp_path, "claude_subscription", {"expires_in_ms": -1000})
    assert r.ready is False
    assert "Claude Code" in (r.recovery or r.reason)


def test_readiness_subscription_missing_blocked(monkeypatch, tmp_path):
    r = _readiness(monkeypatch, tmp_path, "claude_subscription", None)
    assert r.ready is False
    assert "Claude Code" in (r.recovery or r.reason)


def test_readiness_api_key_mode_unchanged(monkeypatch, tmp_path):
    from tldw_chatbook.Chat import provider_readiness as pr
    app_config = {"api_settings": {"anthropic": {"api_key": "sk-ant-api03-" + "k" * 30}}}
    r = pr.get_provider_readiness("Anthropic", app_config, environ={})
    assert r.ready is True
    assert r.api_key_source == "config:api_settings.anthropic.api_key"
