"""TASK-34100.6: certificate and plain-HTTP failures are named, everywhere.

Review findings coverage-19 (a check behind a TLS-inspecting proxy, or against
a self-signed server, read 'could not reach the server' with no pointer to
the setting that fixes it) and provider-15 (https against a plain-HTTP server
read the same, with no 'try http://').

The classification lives in the shared probe/discovery modules, so the
Settings check, the wizard's Provider and Model steps and the Console all
inherit it. Exception shapes are the live ones recorded in
``test_local_server_discovery_engines.py``.
"""

from __future__ import annotations

import asyncio
import ssl

import httpx
import pytest

from tldw_chatbook.Chat import local_server_discovery
from tldw_chatbook.Chat.provider_test_evidence import (
    ProviderReadinessSnapshot,
    provider_readiness_verdict,
)
from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
    discover_openai_compatible_models,
)
from tldw_chatbook.UI.Screens import settings_endpoint_probe
from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
    probe_settings_endpoint,
    provider_probe_result_from_settings_outcome,
)

pytestmark = pytest.mark.local_server_probe


@pytest.fixture(autouse=True)
def _egress_allows_the_test_origin(monkeypatch):
    """The egress policy is not under test here, and it reads config.

    Under the per-test config sandbox that read can trip ADR-126's
    ``RecoveryRequired``, which the probe folds into 'blocked by network
    policy' before any request -- hiding the transport failure these tests
    are about.
    """

    async def _allowed(_url, *, trusted_origins=frozenset()):
        del trusted_origins

    monkeypatch.setattr(settings_endpoint_probe, "check_url_or_raise_async", _allowed)


def _untrusted() -> BaseException:
    return ssl.SSLCertVerificationError(1, "certificate verify failed: self-signed")


def _wrong_version() -> BaseException:
    error = ssl.SSLError(1, "[SSL: WRONG_VERSION_NUMBER] wrong version number")
    error.reason = "WRONG_VERSION_NUMBER"
    return error


def _client(outer: type[httpx.HTTPError], inner_factory) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        error = outer("handshake failed", request=request)
        raise error from inner_factory()

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _stalled_handshake() -> BaseException:
    timeout = TimeoutError()
    timeout.__cause__ = ssl.SSLWantReadError(2, "The operation did not complete")
    return timeout


def _verdict(category: str) -> str:
    return provider_readiness_verdict(
        ProviderReadinessSnapshot(
            configuration="configured",
            endpoint="unreachable",
            model="unconfirmed",
            category=category,
        )
    ).detail


@pytest.mark.asyncio
async def test_a_self_signed_server_reads_as_an_untrusted_certificate():
    outcome = await probe_settings_endpoint(
        "https://127.0.0.1:8443",
        provider="custom",
        http_client=_client(httpx.ConnectError, _untrusted),
    )
    assert outcome.state == "unreachable"
    assert outcome.category == "tls_untrusted"
    assert provider_probe_result_from_settings_outcome(outcome).category == "tls_untrusted"
    detail = _verdict("tls_untrusted")
    assert "answered" in detail and "certificate isn't trusted" in detail
    assert "Settings ▸ Network" in detail
    assert "could not reach" not in detail


def _plain_http_answers(monkeypatch, answers: bool) -> list[str]:
    """Stub the plain-HTTP confirmation: the loopback port answers or stays silent."""
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(str(request.url))
        if answers:
            return httpx.Response(400, text="Bad Request")
        raise httpx.ConnectTimeout("silent", request=request)

    monkeypatch.setattr(
        local_server_discovery,
        "_plain_http_confirm_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return asked


@pytest.mark.asyncio
async def test_https_against_a_plain_http_server_that_answers_at_once_suggests_http():
    outcome = await probe_settings_endpoint(
        "https://127.0.0.1:8181",
        provider="custom",
        http_client=_client(httpx.ConnectError, _wrong_version),
    )
    assert outcome.category == "tls_plain_http"
    assert "try http:// instead of https://" in _verdict("tls_plain_http")


@pytest.mark.asyncio
async def test_a_stalled_loopback_handshake_is_plain_http_only_when_plain_http_answers(
    monkeypatch,
):
    """llama-server waits for a request line, so https stalls its handshake."""
    asked = _plain_http_answers(monkeypatch, answers=True)
    outcome = await probe_settings_endpoint(
        "https://127.0.0.1:8181",
        provider="custom",
        http_client=_client(httpx.ConnectTimeout, _stalled_handshake),
    )
    assert outcome.category == "tls_plain_http"
    assert asked == ["http://127.0.0.1:8181/"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "outer", "asks"),
    [
        ("https://127.0.0.1:8181", httpx.ConnectTimeout, True),  # nothing answers plain
        ("https://127.0.0.1:8181", httpx.ReadTimeout, False),  # TLS was up: a timeout
        ("https://api.openai.com", httpx.ConnectTimeout, False),  # never probes off-box
    ],
    ids=["loopback-silent", "read-timeout", "cloud"],
)
async def test_a_tls_timeout_stays_a_timeout(monkeypatch, endpoint, outer, asks):
    """Review round 1, F2: every TLS timeout carries SSLWantReadError.

    A slow cloud API or a stalled TLS server is a timeout, never 'speaks plain
    HTTP — use http://', which would point the user at plaintext.
    """
    asked = _plain_http_answers(monkeypatch, answers=False)
    outcome = await probe_settings_endpoint(
        endpoint, provider="custom", http_client=_client(outer, _stalled_handshake)
    )
    assert outcome.category == "timeout"
    assert bool(asked) is asks


@pytest.mark.asyncio
async def test_model_discovery_confirms_a_stalled_handshake_the_same_way(monkeypatch):
    _plain_http_answers(monkeypatch, answers=False)
    result = await discover_openai_compatible_models(
        provider="custom",
        provider_list_key="custom",
        endpoint="https://127.0.0.1:8181/v1",
        api_key=None,
        client=_client(httpx.ConnectTimeout, _stalled_handshake),
    )
    assert result.error is not None and result.error.category == "timeout"
    _plain_http_answers(monkeypatch, answers=True)
    result = await discover_openai_compatible_models(
        provider="custom",
        provider_list_key="custom",
        endpoint="https://127.0.0.1:8181/v1",
        api_key=None,
        client=_client(httpx.ConnectTimeout, _stalled_handshake),
    )
    assert result.error is not None and result.error.category == "tls_plain_http"


def _self_signed_context(tmp_path) -> ssl.SSLContext:
    """A server context with a throwaway self-signed certificate for 127.0.0.1."""
    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    return context


async def _serve(handler, *, tls: ssl.SSLContext | None = None):
    server = await asyncio.start_server(handler, "127.0.0.1", 0, ssl=tls)
    return server, server.sockets[0].getsockname()[1]


async def _never_answer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        await reader.read(65536)
        await asyncio.sleep(30)
    except (asyncio.CancelledError, ConnectionError):
        pass
    finally:
        writer.close()


async def _answer_plain_http_after_a_request_line(reader, writer) -> None:
    """llama-server's shape: nothing is sent until a full request arrives."""
    try:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
        await writer.drain()
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
        pass
    finally:
        writer.close()


@pytest.mark.loopback_network
@pytest.mark.asyncio
@pytest.mark.parametrize("tls", [False, True], ids=["handshake-stall", "tls-read-timeout"])
async def test_real_tls_stalls_are_timeouts_not_plain_http(tmp_path, monkeypatch, tls):
    """Negative controls on real sockets (review round 1, F2's two probes).

    A listener that accepts and stays silent stalls the handshake; a real TLS
    server that completes it and never answers times out on the read. Both
    are timeouts.
    """
    monkeypatch.setattr(local_server_discovery, "PLAIN_HTTP_CONFIRM_TIMEOUT_SECONDS", 0.3)
    server, port = await _serve(_never_answer, tls=_self_signed_context(tmp_path) if tls else None)
    async with server:
        outcome = await probe_settings_endpoint(
            f"https://127.0.0.1:{port}",
            provider="custom",
            http_client=httpx.AsyncClient(timeout=0.5, verify=False),
        )
    assert outcome.category == "timeout"


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_real_https_against_a_server_that_waits_for_a_request_is_plain_http():
    """The llama-server case, on a real socket: confirmed, then named."""
    server, port = await _serve(_answer_plain_http_after_a_request_line)
    async with server:
        outcome = await probe_settings_endpoint(
            f"https://127.0.0.1:{port}",
            provider="custom",
            http_client=httpx.AsyncClient(timeout=0.5),
        )
    assert outcome.category == "tls_plain_http"


@pytest.mark.asyncio
async def test_model_discovery_classifies_the_certificate_failure_too():
    """The Model step and the Provider step's own listing read discovery."""
    result = await discover_openai_compatible_models(
        provider="custom",
        provider_list_key="custom",
        endpoint="https://127.0.0.1:8443/v1",
        api_key=None,
        client=_client(httpx.ConnectError, _untrusted),
    )
    assert result.status == "error"
    assert result.error is not None and result.error.category == "tls_untrusted"


@pytest.mark.asyncio
async def test_the_probe_honours_the_apps_tls_trust_policy(monkeypatch):
    """Settings ▸ Network's CA bundle reaches the check it is pointed at."""
    seen: dict[str, object] = {}

    def recording_client(**kwargs):
        seen.update(kwargs)
        return httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"data": [{"id": "m"}]})
            )
        )

    monkeypatch.setattr(settings_endpoint_probe, "build_httpx_async_client", recording_client)
    outcome = await probe_settings_endpoint("http://127.0.0.1:9099", provider="llama_cpp")
    assert outcome.state == "reachable"
    assert "timeout" in seen


async def _answer_models(reader, writer) -> None:
    """A minimal OpenAI-compatible /v1/models answer."""
    try:
        await reader.readuntil(b"\r\n\r\n")
        body = b'{"data": [{"id": "m"}]}'
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        await writer.drain()
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
        pass
    finally:
        writer.close()


@pytest.mark.loopback_network
@pytest.mark.asyncio
async def test_the_apps_ca_bundle_decides_the_probe_and_a_bad_policy_never_costs_it(
    tmp_path, monkeypatch
):
    """coverage-19 / AC#7; review round 1, TE-11(a): the policy test asserted
    only that a timeout was passed. On a real self-signed loopback server:
    with the app's trust policy holding that certificate (what Settings ▸
    Network's CA bundle does) the probe is reachable; without it, the
    certificate is untrusted; and when the policy can't be read the check
    still runs with default verification instead of failing."""
    from tldw_chatbook.Utils import tls_trust

    server_tls = _self_signed_context(tmp_path)
    trusting = ssl.create_default_context(cafile=str(tmp_path / "cert.pem"))
    server, port = await _serve(_answer_models, tls=server_tls)
    url = f"https://127.0.0.1:{port}"
    async with server:
        monkeypatch.setattr(tls_trust, "httpx_verify", lambda: trusting)
        trusted = await probe_settings_endpoint(url, provider="custom")

        monkeypatch.setattr(tls_trust, "httpx_verify", lambda: ssl.create_default_context())
        untrusted = await probe_settings_endpoint(url, provider="custom")

        def unreadable_policy(**_kwargs):
            raise RuntimeError("the TLS policy could not be read")

        monkeypatch.setattr(
            settings_endpoint_probe, "build_httpx_async_client", unreadable_policy
        )
        fallback = await probe_settings_endpoint(url, provider="custom")
    assert trusted.state == "reachable", trusted
    assert untrusted.category == "tls_untrusted", untrusted
    assert fallback.category == "tls_untrusted", "the check ran (default verification)"
