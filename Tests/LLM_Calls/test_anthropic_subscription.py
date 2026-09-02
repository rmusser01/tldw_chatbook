"""TASK-26022: read-only Claude subscription credential borrow for Anthropic."""

from __future__ import annotations

import json
import time

import pytest

from tldw_chatbook.LLM_Calls.anthropic_subscription import (
    CLAUDE_CODE_IDENTITY,
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


def test_missing_credential_message_mentions_keychain_on_macos():
    """M3: the refresh message points at the Keychain too, not only the file."""
    from tldw_chatbook.LLM_Calls.anthropic_subscription import MISSING_CREDENTIAL_MESSAGE
    assert "Keychain" in MISSING_CREDENTIAL_MESSAGE


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

def _stub_default_file(monkeypatch, sub, tmp_path, contents=None):
    """Point DEFAULT_CREDENTIALS_PATH at a tmp file (absent unless contents given)."""
    default = tmp_path / "default_creds.json"
    if contents is not None:
        default.write_text(contents)
    monkeypatch.setattr(sub, "DEFAULT_CREDENTIALS_PATH", default)
    monkeypatch.setattr(sub, "_KEYCHAIN_CACHE", None, raising=False)
    return default


def test_keychain_fallback_used_when_default_file_absent(monkeypatch, tmp_path):
    """AC#1 on macOS: the credential Claude Code stored in the Keychain is found."""
    import tldw_chatbook.LLM_Calls.anthropic_subscription as sub
    _stub_default_file(monkeypatch, sub, tmp_path)
    payload = json.dumps({"claudeAiOauth": {
        "accessToken": "sk-ant-oat01-" + "k" * 40,
        "expiresAt": int(time.time() * 1000) + 3_600_000,
        "subscriptionType": "max",
    }})
    monkeypatch.setattr(sub, "_keychain_credential_raw", lambda: payload)
    cred = read_claude_code_credential()
    assert cred is not None
    assert cred.access_token.startswith("sk-ant-oat01-")
    assert cred.subscription_type == "max"
    assert cred.source_path.startswith("keychain:")


def test_present_but_tokenless_default_file_falls_through_to_keychain(monkeypatch, tmp_path):
    """AC#1: a file that parses but carries no token still falls through to the Keychain."""
    import tldw_chatbook.LLM_Calls.anthropic_subscription as sub
    _stub_default_file(monkeypatch, sub, tmp_path, contents=json.dumps({"claudeAiOauth": {}}))
    payload = json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-" + "k" * 40}})
    monkeypatch.setattr(sub, "_keychain_credential_raw", lambda: payload)
    cred = read_claude_code_credential()
    assert cred is not None and cred.access_token.startswith("sk-ant-oat01-")


def test_explicit_path_never_consults_keychain(monkeypatch, tmp_path):
    """Hermeticity: an explicit path means read that file only -- no Keychain reach."""
    import tldw_chatbook.LLM_Calls.anthropic_subscription as sub
    called = {"n": 0}
    def _kc():
        called["n"] += 1
        return json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-" + "k" * 40}})
    monkeypatch.setattr(sub, "_keychain_credential_raw", _kc)
    assert read_claude_code_credential(tmp_path / "absent.json") is None
    assert called["n"] == 0


def test_default_file_wins_over_keychain(monkeypatch, tmp_path):
    """A present default file is authoritative; the Keychain is only a fallback."""
    import tldw_chatbook.LLM_Calls.anthropic_subscription as sub
    _stub_default_file(monkeypatch, sub, tmp_path,
                       contents=json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-" + "f" * 40}}))
    called = {"n": 0}
    def _kc():
        called["n"] += 1
        return json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-" + "k" * 40}})
    monkeypatch.setattr(sub, "_keychain_credential_raw", _kc)
    cred = read_claude_code_credential()
    assert cred.access_token == "sk-ant-oat01-" + "f" * 40
    assert called["n"] == 0


def test_non_utf8_file_is_none_not_raised(monkeypatch, tmp_path):
    """I1 regression: a non-UTF-8 credential file returns None, never raises out."""
    p = tmp_path / "creds.json"
    p.write_bytes(b"\xff\xfe\x00\x01not utf8")
    # explicit path -> no keychain; must still be a clean None
    assert read_claude_code_credential(p) is None


def test_keychain_malformed_is_none(monkeypatch, tmp_path):
    import tldw_chatbook.LLM_Calls.anthropic_subscription as sub
    _stub_default_file(monkeypatch, sub, tmp_path)
    monkeypatch.setattr(sub, "_keychain_credential_raw", lambda: "{not json")
    assert read_claude_code_credential() is None


def test_keychain_source_path_has_no_token(monkeypatch, tmp_path):
    """AC#3: the Keychain-sourced credential never leaks the token in repr/source."""
    import tldw_chatbook.LLM_Calls.anthropic_subscription as sub
    _stub_default_file(monkeypatch, sub, tmp_path)
    tok = "sk-ant-oat01-" + "s" * 40
    monkeypatch.setattr(sub, "_keychain_credential_raw",
                        lambda: json.dumps({"claudeAiOauth": {"accessToken": tok}}))
    cred = read_claude_code_credential()
    assert tok not in repr(cred)
    assert tok not in cred.source_path


def test_keychain_reader_returns_none_off_darwin(monkeypatch):
    """AC#6: on non-macOS the Keychain reader is inert, so file-only behavior holds."""
    import tldw_chatbook.LLM_Calls.anthropic_subscription as sub
    monkeypatch.setattr(sub, "_KEYCHAIN_CACHE", None, raising=False)
    monkeypatch.setattr(sub.sys, "platform", "linux")
    assert sub._keychain_credential_raw() is None


def test_keychain_reader_nonzero_returncode_is_none(monkeypatch):
    """I4: security exit!=0 (item absent) -> None."""
    import tldw_chatbook.LLM_Calls.anthropic_subscription as sub
    monkeypatch.setattr(sub, "_KEYCHAIN_CACHE", None, raising=False)
    monkeypatch.setattr(sub.sys, "platform", "darwin")
    class _P:
        returncode = 44
        stdout = ""
    monkeypatch.setattr(sub.subprocess, "run", lambda *a, **k: _P())
    assert sub._keychain_credential_raw() is None


def test_keychain_reader_is_memoized(monkeypatch):
    """I3: repeated reads within the TTL do not re-spawn `security`."""
    import tldw_chatbook.LLM_Calls.anthropic_subscription as sub
    monkeypatch.setattr(sub, "_KEYCHAIN_CACHE", None, raising=False)
    monkeypatch.setattr(sub.sys, "platform", "darwin")
    calls = {"n": 0}
    class _P:
        returncode = 0
        stdout = '{"claudeAiOauth": {"accessToken": "sk-ant-oat01-x"}}'
    def _run(*a, **k):
        calls["n"] += 1
        return _P()
    monkeypatch.setattr(sub.subprocess, "run", _run)
    first = sub._keychain_credential_raw()
    second = sub._keychain_credential_raw()
    assert first == second
    assert calls["n"] == 1, "second read within TTL must be served from the memo"


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


def test_identity_string_already_led_is_not_doubled():
    """M1: a string that already leads with the identity is not prepended twice."""
    led = CLAUDE_CODE_IDENTITY + "\n\nBe terse."
    out = with_claude_code_identity(led)
    assert sum(1 for b in out if b["text"].startswith(CLAUDE_CODE_IDENTITY)) == 1


def test_identity_unknown_shape_preserves_content():
    """M2: an unexpected system shape is preserved as text, not dropped."""
    out = with_claude_code_identity({"weird": "shape"})
    assert out[0]["text"] == CLAUDE_CODE_IDENTITY
    assert len(out) == 2 and "weird" in out[1]["text"]


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


# --- M4: subscription + caching degrade-retry must keep the oauth beta ---

def test_cache_degrade_retry_preserves_oauth_beta(monkeypatch, tmp_path):
    """A 400 cache_control degrade-retry on the subscription path must keep the
    oauth beta header (required for the bearer) while dropping only the ttl beta."""
    from tldw_chatbook.LLM_Calls import LLM_API_Calls as mod

    default = tmp_path / "creds.json"
    _write_cred(default)
    import tldw_chatbook.LLM_Calls.anthropic_subscription as sub
    monkeypatch.setattr(sub, "DEFAULT_CREDENTIALS_PATH", default)
    monkeypatch.setattr(sub, "_KEYCHAIN_CACHE", None, raising=False)
    monkeypatch.setattr(mod, "read_claude_code_credential", lambda path=None: sub.read_claude_code_credential())
    monkeypatch.setattr(mod, "load_settings", lambda *a, **k: {
        "anthropic_api": {"model": "claude-sonnet-5"},
        "api_settings": {"anthropic": {"auth_source": "claude_subscription"}},
    })
    # force caching on so the request carries cache_control + the ttl beta
    monkeypatch.setattr(mod, "_anthropic_supports_caching", lambda *a, **k: True)
    monkeypatch.setattr(mod, "_anthropic_caching_enabled", lambda *a, **k: True)

    posts = []
    class _Resp400:
        status_code = 400
        text = "cache_control not supported"
        def json(self): return {}
        def raise_for_status(self): return None
    class _Resp200:
        status_code = 200
        text = "{}"
        def json(self): return {"content": [{"type": "text", "text": "ok"}],
                                "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}, "model": "m"}
        def raise_for_status(self): return None
    class _Session:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def mount(self, *a, **k): pass
        def post(self, url, headers=None, json=None, data=None, stream=False, timeout=None, **kw):
            posts.append(headers)
            return _Resp400() if len(posts) == 1 else _Resp200()
    monkeypatch.setattr(mod, "create_default_session", lambda: _Session())

    mod.chat_with_anthropic(
        input_data=[{"role": "user", "content": "hi"}],
        model="claude-sonnet-5", api_key=None, streaming=False,
        system_prompt="You are a terse assistant.",
    )
    assert len(posts) == 2, "expected a degrade-retry"
    retry_beta = posts[1].get("anthropic-beta", "")
    assert "oauth-2025-04-20" in retry_beta, "oauth beta must survive the degrade-retry"
    assert "extended-cache-ttl" not in retry_beta, "ttl beta should be stripped"
