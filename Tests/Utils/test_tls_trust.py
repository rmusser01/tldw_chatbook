"""Tests for the app-wide TLS trust policy (Utils/tls_trust.py) + config template."""
import ssl as _ssl
import tomllib
from pathlib import Path

import certifi
import pytest
from loguru import logger

import tldw_chatbook.Utils.tls_trust as tls_trust
import tldw_chatbook.config as config_module


def test_default_config_template_has_network_ssl_verify():
    parsed = tomllib.loads(config_module.CONFIG_TOML_CONTENT)
    assert parsed["network"]["ssl_verify"] is True


@pytest.fixture(autouse=True)
def _clean_warn_state():
    tls_trust._warned_modes.clear()
    yield
    tls_trust._warned_modes.clear()


@pytest.fixture
def _set_ssl_config(monkeypatch):
    def _install(value):
        monkeypatch.setattr(
            tls_trust,
            "get_cli_setting",
            lambda section, key=None, default=None: (
                value if (section, key) == ("network", "ssl_verify") else default
            ),
        )

    return _install


@pytest.mark.parametrize(
    "raw,expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("1", True),
        ("ON", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("OFF", False),
        ("", True),
        ("   ", True),
        (5, True),          # unsupported type -> fail safe
        (None, True),       # unsupported type -> fail safe
        (["x"], True),      # unsupported type -> fail safe
    ],
)
def test_tls_verify_setting_coercion(_set_ssl_config, raw, expected):
    _set_ssl_config(raw)
    assert tls_trust.tls_verify_setting() is expected


def test_tls_verify_setting_existing_path_string(tmp_path, _set_ssl_config):
    ca = tmp_path / "corp.pem"
    ca.write_text("# corp")
    _set_ssl_config(str(ca))
    assert tls_trust.tls_verify_setting() == str(ca)


def test_tls_verify_setting_missing_path_fails_safe(tmp_path, _set_ssl_config):
    _set_ssl_config(str(tmp_path / "missing.pem"))
    assert tls_trust.tls_verify_setting() is True


def test_tls_verify_setting_missing_path_logs_error(tmp_path, _set_ssl_config, capsys):
    _set_ssl_config(str(tmp_path / "missing.pem"))
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="ERROR")
    try:
        tls_trust.tls_verify_setting()
    finally:
        logger.remove(sink_id)
    assert any("ssl_verify" in m and "existing file" in m for m in messages)


def test_warn_tls_policy_once_per_mode(_set_ssl_config):
    _set_ssl_config(False)
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING")
    try:
        tls_trust.warn_tls_policy()
        tls_trust.warn_tls_policy()
    finally:
        logger.remove(sink_id)
    warnings = [m for m in messages if "DISABLED" in m]
    assert len(warnings) == 1
    assert "API keys" in warnings[0]


def _context_certs(ctx: "_ssl.SSLContext") -> set[bytes]:
    return {bytes(der) for der in ctx.get_ca_certs(binary_form=True)}


_CUSTOM_PEM = (
    "-----BEGIN CERTIFICATE-----\n"
    "MIICIDCCAcYCCQDceGLIPeXd0zAKBggqhkjOPQQDAjAeMRwwGgYDVQQDDBN0bHMt\n"
    "dHJ1c3QtcGxhbi10ZXN0MB4XDTI2MDgyOTIyMTQ1M1oXDTM2MDgyNjIyMTQ1M1ow\n"
    "HjEcMBoGA1UEAwwTdGxzLXRydXN0LXBsYW4tdGVzdDCCAUswggEDBgcqhkjOPQIB\n"
    "MIH3AgEBMCwGByqGSM49AQECIQD/////AAAAAQAAAAAAAAAAAAAAAP//////////\n"
    "/////zBbBCD/////AAAAAQAAAAAAAAAAAAAAAP///////////////AQgWsY12Ko6\n"
    "k+ez671VdpiGvGUdBrDMU7D2O848PifSYEsDFQDEnTYIhucEk2pmeOETnSa3gZ9+\n"
    "kARBBGsX0fLhLEJH+Lzm5WOkQPJ3A32BLeszoPShOUXYmMKWT+NC4v4af5uO5+tK\n"
    "fA+eFivOM1drMV7Oy7ZAaDe/UfUCIQD/////AAAAAP//////////vOb6racXnoTz\n"
    "ucrC/GMlUQIBAQNCAARJY3gkP7zefsi/pnJW3KSsqc5nUiDQaLk/pB+yUHyazyqn\n"
    "S8AbLvsD1yhRO0B1rWN4VE4ghed8tZcclprS9j38MAoGCCqGSM49BAMCA0gAMEUC\n"
    "ICWp+dTRy9tkb1JSpx3yInFXId3QEjaL3DBQ9yI+/RFAAiEA+PfkQVSpmC0qJ80f\n"
    "SU8n1MnQXxWjOLJNSSPjSCbZBe4=\n"
    "-----END CERTIFICATE-----\n"
)
# A real (throwaway-key) self-signed certificate generated during planning:
# loading requires a parseable PEM body — a fake base64 blob would raise, and
# a file with no PEM block at all also raises SSLError ("no certificate or
# crl found"), which the helper's (OSError, ssl.SSLError) catch converts to
# the fail-safe verify-on path.


def test_ssl_context_default_returns_none(_set_ssl_config):
    _set_ssl_config(True)
    assert tls_trust.ssl_context_for_transport() is None


def test_ssl_context_off_returns_unverified_context(_set_ssl_config):
    _set_ssl_config(False)
    ctx = tls_trust.ssl_context_for_transport()
    assert isinstance(ctx, _ssl.SSLContext)
    assert ctx.check_hostname is False
    assert ctx.verify_mode == _ssl.CERT_NONE


def test_ssl_context_additive_contains_certifi_plus_custom(
    tmp_path, _set_ssl_config
):
    ca = tmp_path / "corp.pem"
    ca.write_text(_CUSTOM_PEM)
    _set_ssl_config(str(ca))
    ctx = tls_trust.ssl_context_for_transport()
    assert isinstance(ctx, _ssl.SSLContext)
    certifi_only = _context_certs(
        _ssl.create_default_context(cafile=certifi.where())
    )
    merged = _context_certs(ctx)
    assert certifi_only < merged  # strictly more certs than certifi alone


def test_ssl_context_corrupt_pem_fails_safe(tmp_path, _set_ssl_config):
    ca = tmp_path / "corp.pem"
    ca.write_text(
        "-----BEGIN CERTIFICATE-----\ngarbage body\n-----END CERTIFICATE-----\n"
    )
    _set_ssl_config(str(ca))
    assert tls_trust.ssl_context_for_transport() is None


def test_requests_verify_bool_passthrough(_set_ssl_config):
    _set_ssl_config(False)
    assert tls_trust.requests_verify() is False
    _set_ssl_config(True)
    assert tls_trust.requests_verify() is True


def test_requests_verify_custom_ca_yields_merged_bundle(
    tmp_path, monkeypatch, _set_ssl_config
):
    ca = tmp_path / "corp.pem"
    ca.write_text(_CUSTOM_PEM)
    _set_ssl_config(str(ca))
    monkeypatch.setattr(
        tls_trust, "get_user_data_dir", lambda: tmp_path / "user_data"
    )
    merged_path = tls_trust.requests_verify()
    assert isinstance(merged_path, str)
    body = Path(merged_path).read_text()
    assert "BEGIN CERTIFICATE" in body
    # merged bundle loads cleanly as a CA store (comment header tolerated)
    ctx = _ssl.create_default_context(cafile=merged_path)
    assert _context_certs(ctx)


def test_merged_bundle_regenerates_when_custom_changes(
    tmp_path, monkeypatch, _set_ssl_config
):
    data_dir = tmp_path / "user_data"
    ca = tmp_path / "corp.pem"
    ca.write_text(_CUSTOM_PEM)
    _set_ssl_config(str(ca))
    monkeypatch.setattr(tls_trust, "get_user_data_dir", lambda: data_dir)
    first = tls_trust.requests_verify()
    first_body = Path(first).read_text()
    ca.write_text(_CUSTOM_PEM + _CUSTOM_PEM)  # content (and mtime) change
    second = tls_trust.requests_verify()
    assert Path(second).read_text() != first_body


def test_merged_bundle_reused_when_sources_unchanged(
    tmp_path, monkeypatch, _set_ssl_config
):
    ca = tmp_path / "corp.pem"
    ca.write_text(_CUSTOM_PEM)
    _set_ssl_config(str(ca))
    monkeypatch.setattr(
        tls_trust, "get_user_data_dir", lambda: tmp_path / "user_data"
    )
    first = Path(tls_trust.requests_verify())
    first_mtime = first.stat().st_mtime_ns
    second = Path(tls_trust.requests_verify())
    assert second == first
    assert second.stat().st_mtime_ns == first_mtime  # not rewritten


def _ssl_context_of(client) -> "_ssl.SSLContext":
    return client._transport._pool._ssl_context  # httpx 0.28 / httpcore layout


def test_httpx_verify_never_returns_bare_path(tmp_path, _set_ssl_config):
    ca = tmp_path / "corp.pem"
    ca.write_text(_CUSTOM_PEM)
    _set_ssl_config(str(ca))
    value = tls_trust.httpx_verify()
    assert isinstance(value, _ssl.SSLContext)  # never the bare path


def test_build_httpx_client_injects_policy(_set_ssl_config):
    _set_ssl_config(False)
    client = tls_trust.build_httpx_client()
    try:
        assert _ssl_context_of(client).verify_mode == _ssl.CERT_NONE
    finally:
        client.close()


def test_build_httpx_client_explicit_verify_wins(_set_ssl_config):
    _set_ssl_config(False)
    client = tls_trust.build_httpx_client(verify=True)
    try:
        assert _ssl_context_of(client).verify_mode != _ssl.CERT_NONE
    finally:
        client.close()


def test_build_httpx_client_default_is_verification(_set_ssl_config):
    _set_ssl_config(True)
    client = tls_trust.build_httpx_client()
    try:
        assert _ssl_context_of(client).verify_mode == _ssl.CERT_REQUIRED
    finally:
        client.close()


async def test_build_httpx_async_client_injects_policy(_set_ssl_config):
    _set_ssl_config(False)
    client = tls_trust.build_httpx_async_client()
    try:
        assert _ssl_context_of(client).verify_mode == _ssl.CERT_NONE
    finally:
        await client.aclose()


def test_build_requests_session_injects_policy(_set_ssl_config):
    _set_ssl_config(False)
    session = tls_trust.build_requests_session()
    assert session.verify is False


def test_build_requests_session_explicit_verify_wins(_set_ssl_config):
    _set_ssl_config(False)
    session = tls_trust.build_requests_session(verify=True)
    assert session.verify is True  # explicit verify beats the disabled policy


def test_create_default_session_carries_tls_policy(_set_ssl_config):
    """The shared session factory carries the TLS policy on session.verify.

    dev's task-19830 centralized session construction in
    ``Utils.egress.create_default_session``; the policy rides that seam so
    every LLM-call/summarization session inherits it (per-request
    ``verify=`` kwargs still win, preserving Subscriptions' per-feed flag).
    """
    from tldw_chatbook.Utils.egress import create_default_session

    _set_ssl_config(False)
    assert create_default_session().verify is False
    _set_ssl_config(True)
    assert create_default_session().verify is True


def test_get_openai_embeddings_passes_tls_policy(_set_ssl_config, monkeypatch):
    """Representative LLM call routes its request through the policy seam."""
    import tldw_chatbook.LLM_Calls.LLM_API_Calls as llm_calls

    captured: dict = {}

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {"data": [{"embedding": [0.0, 1.0]}]}

    class _FakeSession:
        def post(self, url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return _FakeResponse()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        llm_calls, "create_default_session", lambda **kw: _FakeSession()
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _set_ssl_config(False)
    llm_calls.get_openai_embeddings("hello", "text-embedding-3-small")
    assert captured.get("url") == "https://api.openai.com/v1/embeddings"
