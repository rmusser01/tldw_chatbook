"""TASK-26022: read-only Claude subscription credential borrow for Anthropic."""

from __future__ import annotations

import json
import time

import pytest

from tldw_chatbook.LLM_Calls.anthropic_subscription import (
    CLAUDE_CODE_IDENTITY,
    SubscriptionCredential,
    anthropic_auth_source,
    read_claude_code_credential,
    subscription_headers,
    with_claude_code_identity,
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


def _call_anthropic(monkeypatch, tmp_path, auth_source, cred_kwargs=None, api_key=None, system_prompt=None):
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
        # Qodo #5 (PR #2313): mirror the REAL load_settings shape -- auth_source
        # lives in api_settings.anthropic; the legacy anthropic_api mapping
        # never carries it.
        lambda *a, **k: {
            "anthropic_api": {"model": "claude-sonnet-5"},
            "api_settings": {"anthropic": {"auth_source": auth_source}},
        },
    )

    class _Session:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def mount(self, *a, **k): pass
        def post(self, url, headers=None, json=None, data=None, stream=False, timeout=None, **kw):
            captured["headers"] = headers
            captured["url"] = url
            captured["json"] = json
            return _FakeResp()
    monkeypatch.setattr(mod, "create_default_session", lambda: _Session())

    result = mod.chat_with_anthropic(
        input_data=[{"role": "user", "content": "hi"}],
        model="claude-sonnet-5",
        api_key=api_key,
        streaming=False,
        system_prompt=system_prompt,
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


# --- TASK-26022 live-verify follow-ups: Keychain source (macOS) ---

def test_keychain_fallback_used_when_file_absent(monkeypatch, tmp_path):
    """AC#1 on macOS: the credential Claude Code stored in the Keychain is found."""
    import tldw_chatbook.LLM_Calls.anthropic_subscription as sub
    payload = json.dumps({"claudeAiOauth": {
        "accessToken": "sk-ant-oat01-" + "k" * 40,
        "expiresAt": int(time.time() * 1000) + 3_600_000,
        "subscriptionType": "max",
    }})
    monkeypatch.setattr(sub, "_keychain_credential_raw", lambda: payload)
    cred = read_claude_code_credential(tmp_path / "absent.json")
    assert cred is not None
    assert cred.access_token.startswith("sk-ant-oat01-")
    assert cred.subscription_type == "max"


def test_file_wins_over_keychain(monkeypatch, tmp_path):
    """A present file is authoritative; the Keychain is only a fallback."""
    import tldw_chatbook.LLM_Calls.anthropic_subscription as sub
    p = _write_cred(tmp_path / "creds.json", token="sk-ant-oat01-" + "f" * 40)
    called = {"n": 0}
    def _kc():
        called["n"] += 1
        return json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-" + "k" * 40}})
    monkeypatch.setattr(sub, "_keychain_credential_raw", _kc)
    cred = read_claude_code_credential(p)
    assert cred.access_token == "sk-ant-oat01-" + "f" * 40
    assert called["n"] == 0, "keychain must not be consulted when the file has a credential"


def test_keychain_malformed_is_none(monkeypatch, tmp_path):
    import tldw_chatbook.LLM_Calls.anthropic_subscription as sub
    monkeypatch.setattr(sub, "_keychain_credential_raw", lambda: "{not json")
    assert read_claude_code_credential(tmp_path / "absent.json") is None


def test_keychain_source_path_has_no_token(monkeypatch, tmp_path):
    """AC#3: the Keychain-sourced credential never leaks the token in repr/source."""
    import tldw_chatbook.LLM_Calls.anthropic_subscription as sub
    tok = "sk-ant-oat01-" + "s" * 40
    monkeypatch.setattr(
        sub, "_keychain_credential_raw",
        lambda: json.dumps({"claudeAiOauth": {"accessToken": tok}}),
    )
    cred = read_claude_code_credential(tmp_path / "absent.json")
    assert tok not in repr(cred)
    assert tok not in cred.source_path


def test_keychain_reader_returns_none_off_darwin(monkeypatch):
    """AC#6: on non-macOS the Keychain reader is inert, so file-only behavior holds."""
    import tldw_chatbook.LLM_Calls.anthropic_subscription as sub
    monkeypatch.setattr(sub.sys, "platform", "linux")
    assert sub._keychain_credential_raw() is None


# --- TASK-26022 live-verify follow-ups: Claude Code identity (OAuth gate) ---

def test_identity_from_none_is_single_block():
    out = with_claude_code_identity(None)
    assert out == [{"type": "text", "text": CLAUDE_CODE_IDENTITY}]


def test_identity_prepends_and_preserves_string_prompt():
    out = with_claude_code_identity("Answer in French.")
    assert out[0] == {"type": "text", "text": CLAUDE_CODE_IDENTITY}
    assert out[1]["text"] == "Answer in French."


def test_identity_prepends_and_preserves_block_list():
    blocks = [{"type": "text", "text": "Be terse.", "cache_control": {"type": "ephemeral"}}]
    out = with_claude_code_identity(blocks)
    assert out[0]["text"] == CLAUDE_CODE_IDENTITY
    assert out[1] == blocks[0]


def test_identity_is_idempotent():
    once = with_claude_code_identity("hello")
    twice = with_claude_code_identity(once)
    assert twice == once


def test_subscription_send_injects_claude_code_identity(monkeypatch, tmp_path):
    """AC#1/#7: the real send path leads its system with the Claude Code identity
    (the OAuth token 429s otherwise) while preserving the user's system prompt."""
    captured, _ = _call_anthropic(
        monkeypatch, tmp_path, "claude_subscription", cred_kwargs={},
        system_prompt="You are a helpful research assistant.",
    )
    system = captured["json"]["system"]
    assert isinstance(system, list)
    assert system[0]["text"] == CLAUDE_CODE_IDENTITY
    assert any("research assistant" in b.get("text", "") for b in system[1:])


def test_api_key_send_does_not_inject_identity(monkeypatch, tmp_path):
    """AC#6: the identity spoof only applies to the subscription path."""
    captured, _ = _call_anthropic(
        monkeypatch, tmp_path, "api_key", api_key="sk-ant-api03-realkey",
        system_prompt="You are a helpful research assistant.",
    )
    system = captured["json"].get("system")
    # system may be a str or a cached block list depending on the model; the
    # point is that the Claude Code identity is never injected in api_key mode.
    flat = json.dumps(system)
    assert CLAUDE_CODE_IDENTITY not in flat
    assert "research assistant" in flat
