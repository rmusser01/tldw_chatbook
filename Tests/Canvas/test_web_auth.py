from __future__ import annotations

from dataclasses import replace

import pytest

from tldw_chatbook.Canvas.web_auth import (
    AuthenticationError,
    BindPolicyError,
    RequestFacts,
    WebAuthManager,
    build_web_auth_policy,
    is_loopback_host,
    resolve_web_access_token,
)


@pytest.mark.parametrize(
    ("host", "want"),
    [
        ("127.0.0.1", True),
        ("127.99.10.2", True),
        ("::1", True),
        ("[::1]", True),
        ("localhost", True),
        ("0.0.0.0", False),
        ("::", False),
        ("192.168.1.25", False),
        ("10.0.0.8", False),
        ("203.0.113.8", False),
        ("chatbook.example", False),
    ],
)
def test_loopback_classification_never_treats_wildcard_or_private_as_local(
    host: str, want: bool
) -> None:
    assert is_loopback_host(host) is want


def test_localhost_fails_closed_when_resolution_contains_non_loopback() -> None:
    def resolver(*_args, **_kwargs):
        return [
            (2, 1, 6, "", ("127.0.0.1", 0)),
            (2, 1, 6, "", ("192.168.1.9", 0)),
        ]

    assert is_loopback_host("localhost", resolver=resolver) is False


@pytest.mark.parametrize(
    ("environment", "configured", "keyring_value", "want_value", "want_source"),
    [
        (
            {"TLDW_CHATBOOK_WEB_ACCESS_TOKEN": "env-token"},
            "cfg-token",
            "kr-token",
            "env-token",
            "environment",
        ),
        ({}, "cfg-token", "kr-token", "cfg-token", "config"),
        ({}, "", "kr-token", "kr-token", "keyring"),
        ({}, "", None, None, "missing"),
    ],
)
def test_access_token_precedence_is_dedicated_and_explicit(
    environment, configured, keyring_value, want_value, want_source
) -> None:
    resolved = resolve_web_access_token(
        configured,
        environ=environment,
        keyring_get=lambda _service, _account: keyring_value,
    )

    assert resolved.reveal() == want_value
    assert resolved.source == want_source
    assert "token" not in repr(resolved).lower()
    for secret in ("env-token", "cfg-token", "kr-token"):
        assert secret not in repr(resolved)


def test_shipped_web_server_defaults_fail_closed_for_remote_access() -> None:
    from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML

    web = DEFAULT_CONFIG_FROM_TOML["web_server"]
    assert web["access_token"] == ""
    assert web["public_url"] == ""
    assert web["trusted_proxy_addresses"] == []
    assert web["tls_certificate"] == ""
    assert web["tls_private_key"] == ""
    assert web["allow_insecure_remote_http"] is False


def test_remote_bind_requires_dedicated_access_token() -> None:
    with pytest.raises(BindPolicyError, match="access token"):
        build_web_auth_policy(host="0.0.0.0", port=8000, access_token=None)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20", "203.0.113.4"])
def test_remote_plaintext_bind_fails_closed_by_default(host: str) -> None:
    with pytest.raises(BindPolicyError, match="HTTPS"):
        build_web_auth_policy(
            host=host,
            port=8000,
            access_token="correct horse",
            public_url="http://chatbook.example",
        )


def test_remote_plaintext_requires_explicit_insecure_development_override() -> None:
    policy = build_web_auth_policy(
        host="192.168.1.20",
        port=8000,
        access_token="correct horse",
        allow_insecure_remote_http=True,
    )

    assert policy.insecure_remote_http is True


def test_https_public_origin_allows_remote_bind_without_plaintext_override() -> None:
    policy = build_web_auth_policy(
        host="0.0.0.0",
        port=8000,
        access_token="correct horse",
        public_url="https://chatbook.example",
        direct_tls=True,
    )

    assert policy.allowed_hosts == frozenset({"chatbook.example"})
    assert policy.secure_cookies is True


def test_https_proxy_origin_requires_an_explicit_trusted_proxy() -> None:
    policy = build_web_auth_policy(
        host="127.0.0.1",
        port=8000,
        access_token="correct horse",
        public_url="https://chatbook.example",
        trusted_proxy_addresses=["127.0.0.1"],
    )

    assert policy.allowed_hosts == frozenset({"chatbook.example"})
    assert policy.is_trusted_proxy("127.0.0.1") is True


def test_https_public_origin_cannot_be_claimed_without_tls_or_proxy() -> None:
    with pytest.raises(BindPolicyError, match="TLS"):
        build_web_auth_policy(
            host="192.168.1.20",
            port=8000,
            access_token="correct horse",
            public_url="https://chatbook.example",
            allow_insecure_remote_http=True,
        )


def test_wildcard_remote_bind_requires_a_public_origin() -> None:
    with pytest.raises(BindPolicyError, match="public_url"):
        build_web_auth_policy(
            host="0.0.0.0",
            port=8000,
            access_token="correct horse",
            allow_insecure_remote_http=True,
        )


@pytest.mark.parametrize(
    ("proxy", "want"),
    [
        ("127.0.0.1", True),
        ("::1", True),
        ("10.0.0.3", True),
        ("10.0.0.4", False),
        ("203.0.113.2", False),
    ],
)
def test_forwarded_headers_are_trusted_only_from_exact_proxy_allowlist(
    proxy: str, want: bool
) -> None:
    policy = build_web_auth_policy(
        host="127.0.0.1",
        port=8000,
        access_token=None,
        trusted_proxy_addresses=["127.0.0.1", "::1", "10.0.0.3"],
    )
    assert policy.is_trusted_proxy(proxy) is want


def _manager(*, now: list[float] | None = None, max_attempts: int = 5):
    clock = (lambda: now[0]) if now is not None else (lambda: 100.0)
    policy = build_web_auth_policy(
        host="0.0.0.0",
        port=8000,
        access_token="correct horse battery staple",
        public_url="https://chatbook.example",
        direct_tls=True,
    )
    return WebAuthManager(
        policy,
        clock=clock,
        idle_timeout_seconds=30,
        absolute_timeout_seconds=120,
        login_attempts_per_minute=max_attempts,
    )


def test_one_time_bootstrap_nonce_is_consumed_and_never_contains_access_token() -> None:
    manager = _manager()
    nonce = manager.issue_bootstrap()
    assert "correct horse battery staple" not in nonce

    grant = manager.exchange_bootstrap(nonce, client_ip="203.0.113.9")
    assert grant.cookie_value
    assert grant.csrf_token
    with pytest.raises(AuthenticationError):
        manager.exchange_bootstrap(nonce, client_ip="203.0.113.9")


def test_access_token_login_uses_constant_time_comparison(monkeypatch) -> None:
    calls = []

    def compared(left: bytes, right: bytes) -> bool:
        calls.append((left, right))
        return left == right

    monkeypatch.setattr("tldw_chatbook.Canvas.web_auth.hmac.compare_digest", compared)
    manager = _manager()

    grant = manager.login_with_access_token(
        "correct horse battery staple", client_ip="203.0.113.9"
    )

    assert grant.cookie_value
    assert calls == [(b"correct horse battery staple", b"correct horse battery staple")]


def test_unicode_access_token_authenticates_without_a_comparison_type_error() -> None:
    policy = build_web_auth_policy(
        host="0.0.0.0",
        port=8000,
        access_token="pässword-🔐",
        public_url="https://chatbook.example",
        direct_tls=True,
    )
    manager = WebAuthManager(policy)

    assert manager.login_with_access_token(
        "pässword-🔐", client_ip="203.0.113.9"
    ).cookie_value


def test_comparison_failure_is_converted_to_a_content_free_denial(monkeypatch) -> None:
    def fail_comparison(_left: bytes, _right: bytes) -> bool:
        raise RuntimeError("forced comparator failure")

    monkeypatch.setattr(
        "tldw_chatbook.Canvas.web_auth.hmac.compare_digest", fail_comparison
    )
    manager = _manager()

    with pytest.raises(AuthenticationError, match="denied") as raised:
        manager.login_with_access_token(
            "correct horse battery staple", client_ip="203.0.113.9"
        )

    assert raised.value.__cause__ is None
    assert "correct horse battery staple" not in repr(raised.value)


def test_login_rate_limit_is_bounded_per_client() -> None:
    manager = _manager(max_attempts=2)

    for _ in range(2):
        with pytest.raises(AuthenticationError, match="denied"):
            manager.login_with_access_token("wrong", client_ip="203.0.113.9")
    with pytest.raises(AuthenticationError, match="temporarily unavailable"):
        manager.login_with_access_token(
            "correct horse battery staple", client_ip="203.0.113.9"
        )
    assert manager.rate_limit_subject_count == 1


def test_successful_logins_do_not_reset_the_admission_rate_limit() -> None:
    manager = _manager(max_attempts=2)

    for _ in range(2):
        manager.login_with_access_token(
            "correct horse battery staple", client_ip="203.0.113.9"
        )
    with pytest.raises(AuthenticationError, match="temporarily unavailable"):
        manager.login_with_access_token(
            "correct horse battery staple", client_ip="203.0.113.9"
        )


def _facts(grant, **overrides) -> RequestFacts:
    base = RequestFacts(
        method="POST",
        path="/canvas/submit",
        peer_ip="203.0.113.9",
        scheme="https",
        host="chatbook.example",
        origin="https://chatbook.example",
        cookie_value=grant.cookie_value,
        csrf_token=grant.csrf_token,
    )
    return replace(base, **overrides)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"host": "evil.example"}, "host"),
        (
            {
                "host": "chatbook.example:8443",
                "origin": "https://chatbook.example:8443",
            },
            "host",
        ),
        ({"origin": "https://evil.example"}, "origin"),
        ({"origin": "http://chatbook.example"}, "origin"),
        ({"origin": "https://chatbook.example:8443"}, "origin"),
        ({"csrf_token": "wrong"}, "CSRF"),
        ({"csrf_token": "é"}, "CSRF"),
        ({"cookie_value": "wrong"}, "session"),
    ],
)
def test_state_changing_request_rejects_host_origin_session_and_csrf_mismatch(
    overrides, match
) -> None:
    manager = _manager()
    grant = manager.login_with_access_token(
        "correct horse battery staple", client_ip="203.0.113.9"
    )

    with pytest.raises(AuthenticationError, match=match):
        manager.authenticate_request(_facts(grant, **overrides), require_csrf=True)


@pytest.mark.parametrize(
    "malformed_host",
    [
        "chatbook.example#evil",
        "chatbook.example?evil",
        "chatbook.example/path",
        "chatbook.example@evil",
        "chatbook.example:",
    ],
)
def test_malformed_host_authorities_fail_closed(malformed_host: str) -> None:
    manager = _manager()

    with pytest.raises(AuthenticationError, match="host"):
        manager.validate_public_request(
            RequestFacts(
                method="GET",
                path="/auth/login",
                peer_ip="203.0.113.9",
                scheme="https",
                host=malformed_host,
            ),
            require_origin=False,
        )


def test_trusted_proxy_values_replace_transport_values() -> None:
    manager = _manager()
    grant = manager.login_with_access_token(
        "correct horse battery staple", client_ip="203.0.113.9"
    )
    facts = _facts(
        grant,
        peer_ip="127.0.0.1",
        scheme="http",
        host="127.0.0.1:8000",
        forwarded_for="203.0.113.9",
        forwarded_proto="https",
        forwarded_host="chatbook.example",
    )
    manager.policy = replace(
        manager.policy, trusted_proxy_addresses=frozenset({"127.0.0.1"})
    )

    session = manager.authenticate_request(facts, require_csrf=True)

    assert session.client_ip == "203.0.113.9"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("forwarded_for", "203.0.113.9, 10.0.0.1"),
        ("forwarded_for", "not-an-ip"),
        ("forwarded_proto", "https,http"),
        ("forwarded_proto", "javascript"),
        ("forwarded_host", "chatbook.example,evil.example"),
        ("forwarded_host", "https://chatbook.example"),
        ("forwarded_host", "chatbook.example#evil"),
        ("forwarded_host", "chatbook.example?evil"),
        ("forwarded_host", "chatbook.example:"),
    ],
)
def test_malformed_forwarded_headers_from_trusted_proxy_fail_closed(
    field, value
) -> None:
    manager = _manager()
    manager.policy = replace(
        manager.policy, trusted_proxy_addresses=frozenset({"127.0.0.1"})
    )
    grant = manager.login_with_access_token(
        "correct horse battery staple", client_ip="203.0.113.9"
    )
    facts = _facts(
        grant,
        peer_ip="127.0.0.1",
        scheme="http",
        host="127.0.0.1:8000",
        forwarded_for="203.0.113.9",
        forwarded_proto="https",
        forwarded_host="chatbook.example",
    )

    with pytest.raises(AuthenticationError, match="forwarded"):
        manager.authenticate_request(
            replace(facts, **{field: value}), require_csrf=True
        )


def test_untrusted_forwarded_headers_are_ignored() -> None:
    manager = _manager()
    grant = manager.login_with_access_token(
        "correct horse battery staple", client_ip="203.0.113.9"
    )
    facts = _facts(
        grant,
        forwarded_for="not-an-ip",
        forwarded_proto="javascript",
        forwarded_host="evil.example",
    )

    assert manager.authenticate_request(facts, require_csrf=True).session_id


def test_https_policy_rejects_direct_plaintext_bypass_around_trusted_proxy() -> None:
    manager = _manager()
    manager.policy = replace(
        manager.policy, trusted_proxy_addresses=frozenset({"127.0.0.1"})
    )
    facts = RequestFacts(
        method="GET",
        path="/auth/login",
        peer_ip="203.0.113.9",
        scheme="http",
        host="chatbook.example",
    )

    with pytest.raises(AuthenticationError, match="transport"):
        manager.validate_public_request(facts, require_origin=False)


def test_websocket_requires_upgrade_origin_session_and_csrf_subprotocol() -> None:
    manager = _manager()
    grant = manager.login_with_access_token(
        "correct horse battery staple", client_ip="203.0.113.9"
    )
    facts = _facts(
        grant,
        method="GET",
        path="/ws",
        upgrade="websocket",
        connection="keep-alive, Upgrade",
        websocket_protocols=("chatbook-v1", f"csrf.{grant.csrf_token}"),
        csrf_token=None,
    )

    assert manager.authenticate_request(facts, websocket=True).session_id
    with pytest.raises(AuthenticationError, match="websocket"):
        manager.authenticate_request(replace(facts, upgrade=""), websocket=True)
    with pytest.raises(AuthenticationError, match="CSRF"):
        manager.authenticate_request(
            replace(facts, websocket_protocols=("chatbook-v1", "csrf.wrong")),
            websocket=True,
        )
    with pytest.raises(AuthenticationError, match="CSRF"):
        manager.authenticate_request(
            replace(facts, websocket_protocols=("chatbook-v1", "csrf.é")),
            websocket=True,
        )


def test_request_repr_redacts_all_cookie_header_and_websocket_credentials() -> None:
    facts = RequestFacts(
        method="GET",
        path="/ws",
        peer_ip="127.0.0.1",
        scheme="http",
        host="127.0.0.1:8000",
        cookie_value="cookie-secret",
        csrf_token="header-secret",
        websocket_protocols=("chatbook-v1", "csrf.websocket-secret"),
    )

    rendered = repr(facts)
    assert "cookie-secret" not in rendered
    assert "header-secret" not in rendered
    assert "websocket-secret" not in rendered


def test_session_idle_and_absolute_expiry_are_enforced() -> None:
    now = [100.0]
    manager = _manager(now=now)
    first = manager.login_with_access_token(
        "correct horse battery staple", client_ip="203.0.113.9"
    )
    now[0] = 131.0
    with pytest.raises(AuthenticationError, match="expired"):
        manager.authenticate_request(_facts(first), require_csrf=True)

    now[0] = 200.0
    second = manager.login_with_access_token(
        "correct horse battery staple", client_ip="203.0.113.9"
    )
    for current in (225.0, 250.0, 275.0, 300.0):
        now[0] = current
        manager.authenticate_request(_facts(second), require_csrf=True)
    now[0] = 321.0
    with pytest.raises(AuthenticationError, match="expired"):
        manager.authenticate_request(_facts(second), require_csrf=True)


def test_revocation_invalidates_session_and_allows_bounded_global_shutdown() -> None:
    manager = _manager()
    grant = manager.login_with_access_token(
        "correct horse battery staple", client_ip="203.0.113.9"
    )
    manager.revoke(grant.cookie_value)
    with pytest.raises(AuthenticationError, match="session"):
        manager.authenticate_request(_facts(grant), require_csrf=True)

    other = manager.login_with_access_token(
        "correct horse battery staple", client_ip="203.0.113.10"
    )
    manager.revoke_all()
    with pytest.raises(AuthenticationError, match="session"):
        manager.authenticate_request(_facts(other), require_csrf=True)


def test_revocation_closes_registered_live_channels_exactly_once() -> None:
    manager = _manager()
    grant = manager.login_with_access_token(
        "correct horse battery staple", client_ip="203.0.113.9"
    )
    session = manager.authenticate_request(_facts(grant), require_csrf=True)
    closed = []
    unregister = manager.register_channel(session, lambda: closed.append("closed"))

    manager.revoke(grant.cookie_value)
    manager.revoke(grant.cookie_value)
    with pytest.raises(AuthenticationError, match="session"):
        manager.touch_session(session)
    with pytest.raises(AuthenticationError, match="session"):
        manager.register_channel(session, lambda: None)
    unregister()

    assert closed == ["closed"]


def test_idle_expiry_closes_a_registered_channel_without_another_http_request() -> None:
    now = [100.0]
    manager = _manager(now=now)
    grant = manager.login_with_access_token(
        "correct horse battery staple", client_ip="203.0.113.9"
    )
    session = manager.authenticate_request(_facts(grant), require_csrf=True)
    closed = []
    manager.register_channel(session, lambda: closed.append("closed"))

    now[0] = 131.0
    assert manager.expire_session_if_due(session) is True
    assert closed == ["closed"]


def test_session_and_bootstrap_retention_are_bounded_and_evict_oldest() -> None:
    policy = build_web_auth_policy(
        host="0.0.0.0",
        port=8000,
        access_token="correct horse battery staple",
        public_url="https://chatbook.example",
        direct_tls=True,
    )
    manager = WebAuthManager(
        policy,
        clock=lambda: 100.0,
        max_sessions=2,
        max_bootstraps=2,
    )
    first = manager.login_with_access_token(
        "correct horse battery staple", client_ip="203.0.113.1"
    )
    manager.login_with_access_token(
        "correct horse battery staple", client_ip="203.0.113.2"
    )
    manager.login_with_access_token(
        "correct horse battery staple", client_ip="203.0.113.3"
    )

    assert manager.session_count == 2
    with pytest.raises(AuthenticationError, match="session"):
        manager.authenticate_request(_facts(first), require_csrf=True)

    bootstrap_one = manager.issue_bootstrap()
    manager.issue_bootstrap()
    manager.issue_bootstrap()
    assert manager.bootstrap_count == 2
    with pytest.raises(AuthenticationError, match="denied"):
        manager.exchange_bootstrap(bootstrap_one, client_ip="203.0.113.1")


def test_session_churn_never_evicts_registered_live_channels() -> None:
    policy = build_web_auth_policy(
        host="127.0.0.1",
        port=8000,
        access_token=None,
    )
    manager = WebAuthManager(policy, max_sessions=2)
    first = manager.authenticate_local(client_ip="127.0.0.1")
    second = manager.authenticate_local(client_ip="127.0.0.1")
    second_session = manager.authenticate_request(
        RequestFacts(
            method="GET",
            path="/",
            peer_ip="127.0.0.1",
            scheme="http",
            host="127.0.0.1:8000",
            cookie_value=second.cookie_value,
        )
    )
    first_session = manager.authenticate_request(
        RequestFacts(
            method="GET",
            path="/",
            peer_ip="127.0.0.1",
            scheme="http",
            host="127.0.0.1:8000",
            cookie_value=first.cookie_value,
        )
    )
    first_closed = []
    manager.register_channel(first_session, lambda: first_closed.append(True))

    third = manager.authenticate_local(client_ip="127.0.0.1")

    assert first_closed == []
    assert (
        manager.authenticate_request(
            RequestFacts(
                method="GET",
                path="/",
                peer_ip="127.0.0.1",
                scheme="http",
                host="127.0.0.1:8000",
                cookie_value=first.cookie_value,
            )
        )
        is first_session
    )
    with pytest.raises(AuthenticationError, match="session"):
        manager.authenticate_request(
            RequestFacts(
                method="GET",
                path="/",
                peer_ip="127.0.0.1",
                scheme="http",
                host="127.0.0.1:8000",
                cookie_value=second.cookie_value,
            )
        )
    with pytest.raises(AuthenticationError, match="session"):
        manager.register_channel(second_session, lambda: None)

    third_session = manager.authenticate_request(
        RequestFacts(
            method="GET",
            path="/",
            peer_ip="127.0.0.1",
            scheme="http",
            host="127.0.0.1:8000",
            cookie_value=third.cookie_value,
        )
    )
    manager.register_channel(third_session, lambda: None)
    with pytest.raises(AuthenticationError, match="capacity"):
        manager.authenticate_local(client_ip="127.0.0.1")
    assert first_closed == []
