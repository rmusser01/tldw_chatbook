"""Exercise the real HTTP parser with fake socket/TLS boundaries, never real peers."""

import gzip
import io
import socket
import ssl

import pytest

from tldw_chatbook.Petdex import network


class FakeSocket:
    def __init__(self, response, peer="93.184.216.34"):
        self.data = io.BytesIO(response)
        self.peer = peer
        self.sent = b""
        self.destination = None
        self.closed = False

    def settimeout(self, timeout):
        assert 0 < timeout <= 5

    def connect(self, destination):
        self.destination = destination

    def getpeername(self):
        return (self.peer, 443)

    def sendall(self, data):
        self.sent += data

    def recv(self, size):
        return self.data.read(min(size, 37))

    def close(self):
        self.closed = True


def response(body=b"ok", headers=b"", status=b"200 OK"):
    return (
        b"HTTP/1.1 " + status + b"\r\n" + headers + b"Connection: close\r\n\r\n" + body
    )


@pytest.fixture
def transport(monkeypatch):
    sockets = []
    resolutions = []
    sni = []
    answers = ["93.184.216.34"]
    replies = [response()]
    peer = ["93.184.216.34"]

    def resolve(host, port, **kwargs):
        resolutions.append((host, port))
        return [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, 443))
            for ip in answers
        ]

    def new_socket(*args):
        sock = FakeSocket(replies.pop(0), peer[0])
        sockets.append(sock)
        return sock

    class Context:
        check_hostname = True
        verify_mode = ssl.CERT_REQUIRED

        def wrap_socket(self, sock, *, server_hostname):
            sni.append(server_hostname)
            return sock

    monkeypatch.setattr(network.socket, "getaddrinfo", resolve)
    monkeypatch.setattr(network.socket, "socket", new_socket)
    monkeypatch.setattr(network.ssl, "create_default_context", Context)
    return sockets, resolutions, sni, answers, replies, peer


@pytest.mark.parametrize(
    "url",
    [
        "http://petdex.dev/a",
        "https://evil.test/a",
        "https://petdex.dev.evil.test/a",
        "https://user:secret@petdex.dev/a",
        "https://petdex.dev:444/a",
        "https://petdex.dev/a#fragment",
        "https://127.0.0.1/a",
        "https://petdex.dev./a",
        "https://petdex.dev/\r\nX: y",
        "https://petdex.dev/a\\b",
    ],
)
def test_untrusted_url_never_resolves_or_connects(url, transport):
    with pytest.raises(network.PetdexNetworkError):
        network.fetch_bytes(url, max_bytes=100)
    assert not transport[0] and not transport[1]


@pytest.mark.parametrize(
    "ips",
    [
        ["127.0.0.1"],
        ["93.184.216.34", "10.0.0.1"],
        ["169.254.169.254"],
        ["::ffff:127.0.0.1"],
        ["224.0.0.1"],
        [],
    ],
)
def test_all_dns_answers_must_be_public(ips, transport):
    transport[3][:] = ips
    with pytest.raises(network.PetdexNetworkError):
        network.fetch_bytes("https://petdex.dev/api/manifest", max_bytes=100)
    assert not transport[0]


def test_numeric_connection_original_sni_and_host_ignores_proxies(
    transport, monkeypatch
):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8080")
    assert (
        network.fetch_bytes("https://petdex.dev/api/manifest?a=1", max_bytes=100)
        == b"ok"
    )
    sock = transport[0][0]
    assert sock.destination == ("93.184.216.34", 443)
    assert transport[1] == [("petdex.dev", 443)]
    assert transport[2] == ["petdex.dev"]
    assert b"GET /api/manifest?a=1 HTTP/1.1\r\nHost: petdex.dev\r\n" in sock.sent
    assert sock.closed


def test_peer_mismatch_rejected_before_request(transport):
    transport[5][0] = "93.184.216.35"
    with pytest.raises(network.PetdexNetworkError, match="peer"):
        network.fetch_bytes("https://petdex.dev/a", max_bytes=100)
    assert transport[0][0].sent == b""
    assert transport[0][0].closed


def test_redirect_gets_fresh_dns_and_correct_sni(transport):
    transport[4][:] = [
        response(
            headers=b"Location: https://assets.petdex.dev/pet.json\r\n",
            status=b"302 Found",
        ),
        response(b"{}"),
    ]
    assert network.fetch_bytes("https://petdex.dev/a", max_bytes=100) == b"{}"
    assert transport[1] == [("petdex.dev", 443), ("assets.petdex.dev", 443)]
    assert transport[2] == ["petdex.dev", "assets.petdex.dev"]
    assert all(sock.closed for sock in transport[0])


@pytest.mark.parametrize(
    "location",
    [b"http://petdex.dev/a", b"https://127.0.0.1/a", b"https://attacker.test/a"],
)
def test_redirect_cannot_escape_allowlist(location, transport):
    transport[4][:] = [
        response(headers=b"Location: " + location + b"\r\n", status=b"302 Found")
    ]
    with pytest.raises(network.PetdexNetworkError):
        network.fetch_bytes("https://petdex.dev/a", max_bytes=100)
    assert len(transport[0]) == 1


def test_streamed_body_limit_without_content_length(transport):
    transport[4][:] = [response(b"x" * 101)]
    with pytest.raises(network.PetdexNetworkError, match="size"):
        network.fetch_bytes("https://petdex.dev/a", max_bytes=100)


def test_gzip_decoded_limit(transport):
    transport[4][:] = [
        response(gzip.compress(b"x" * 10001), b"Content-Encoding: gzip\r\n")
    ]
    with pytest.raises(network.PetdexNetworkError, match="size"):
        network.fetch_bytes("https://petdex.dev/a", max_bytes=100)


def test_gzip_decodes_within_limit(transport):
    transport[4][:] = [response(gzip.compress(b"hello"), b"Content-Encoding: gzip\r\n")]
    assert network.fetch_bytes("https://petdex.dev/a", max_bytes=100) == b"hello"


def test_cancel_before_dns(transport):
    with pytest.raises(network.PetdexNetworkError, match="cancel"):
        network.fetch_bytes(
            "https://petdex.dev/a", max_bytes=100, cancel_requested=lambda: True
        )
    assert not transport[0] and not transport[1]


def test_cancel_during_body_closes_socket(transport):
    transport[4][:] = [response(b"x" * 90)]

    def cancelled():
        return bool(transport[0] and transport[0][0].data.tell() > 40)

    with pytest.raises(network.PetdexNetworkError, match="cancel"):
        network.fetch_bytes(
            "https://petdex.dev/a", max_bytes=100, cancel_requested=cancelled
        )
    assert transport[0][0].closed


def test_status_has_safe_category_without_response_secrets(transport):
    transport[4][:] = [response(b"token=secret", status=b"429 secret-token")]
    with pytest.raises(network.PetdexNetworkError) as caught:
        network.fetch_bytes("https://petdex.dev/a?secret=value", max_bytes=100)
    assert caught.value.status_code == 429
    assert "secret" not in str(caught.value)


def test_redirect_private_reresolution_does_not_connect(monkeypatch, transport):
    transport[4][:] = [response(headers=b"Location: /next\r\n", status=b"302 Found")]
    calls = []

    def rebinding(host, port, **kwargs):
        calls.append(host)
        ip = "93.184.216.34" if len(calls) == 1 else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, 443))]

    monkeypatch.setattr(network.socket, "getaddrinfo", rebinding)
    with pytest.raises(network.PetdexNetworkError, match="DNS"):
        network.fetch_bytes("https://petdex.dev/a", max_bytes=100)
    assert len(transport[0]) == 1
    assert len(calls) == 2


def test_tls_certificate_failure_closes_raw_socket(monkeypatch, transport):
    class Context:
        def wrap_socket(self, sock, *, server_hostname):
            assert self.check_hostname
            assert self.verify_mode == ssl.CERT_REQUIRED
            raise ssl.SSLCertVerificationError("secret server details")

    monkeypatch.setattr(network.ssl, "create_default_context", Context)
    with pytest.raises(network.PetdexNetworkError, match="TLS") as caught:
        network.fetch_bytes("https://petdex.dev/a", max_bytes=100)
    assert "secret" not in str(caught.value)
    assert transport[0][0].closed
    assert not transport[0][0].sent


def test_slow_response_hits_total_deadline(monkeypatch, transport):
    now = [100.0]
    original_recv = FakeSocket.recv

    def slow_recv(self, size):
        now[0] += 4
        return original_recv(self, min(size, 1))

    monkeypatch.setattr(network.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(FakeSocket, "recv", slow_recv)
    with pytest.raises(network.PetdexNetworkError, match="timeout"):
        network.fetch_bytes("https://petdex.dev/a", max_bytes=100)
    assert transport[0][0].closed


def test_cancel_during_dns_never_connects(monkeypatch, transport):
    import threading

    started = threading.Event()
    release = threading.Event()

    def slow_dns(*args, **kwargs):
        started.set()
        release.wait(1)
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("93.184.216.34", 443),
            )
        ]

    monkeypatch.setattr(network.socket, "getaddrinfo", slow_dns)
    try:
        with pytest.raises(network.PetdexNetworkError, match="cancel"):
            network.fetch_bytes(
                "https://petdex.dev/a", max_bytes=100, cancel_requested=started.is_set
            )
        assert not transport[0]
    finally:
        release.set()


def test_redirect_loop_is_bounded(transport):
    transport[4][:] = [response(headers=b"Location: /a\r\n", status=b"302 Found")] * 5
    with pytest.raises(network.PetdexNetworkError, match="redirect"):
        network.fetch_bytes("https://petdex.dev/a", max_bytes=100)
    assert len(transport[0]) == 4


def test_gzip_wire_body_limit_even_when_decoded_body_fits(transport):
    # gzip's header/trailer exceed the decoded empty body.
    transport[4][:] = [response(gzip.compress(b""), b"Content-Encoding: gzip\r\n")]
    with pytest.raises(network.PetdexNetworkError, match="size"):
        network.fetch_bytes("https://petdex.dev/a", max_bytes=10)


def test_valid_chunked_gzip_is_decoded(transport):
    body = gzip.compress(b"hello")
    chunks = f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n"
    transport[4][:] = [
        response(chunks, b"Transfer-Encoding: chunked\r\nContent-Encoding: gzip\r\n")
    ]
    assert network.fetch_bytes("https://petdex.dev/a", max_bytes=100) == b"hello"


@pytest.mark.parametrize(
    "headers",
    [
        b"Content-Length: 2\r\nContent-Length: 3\r\n",
        b"Transfer-Encoding: chunked\r\nContent-Length: 2\r\n",
        b"Transfer-Encoding: compress\r\n",
    ],
)
def test_ambiguous_or_unsupported_body_framing_rejected(headers, transport):
    transport[4][:] = [response(b"2\r\nok\r\n0\r\n\r\n", headers)]
    with pytest.raises(network.PetdexNetworkError, match="framing"):
        network.fetch_bytes("https://petdex.dev/a", max_bytes=100)


@pytest.mark.parametrize(
    "body", [gzip.compress(b"hi")[:-2], gzip.compress(b"hi") + gzip.compress(b"extra")]
)
def test_truncated_or_concatenated_gzip_rejected(body, transport):
    transport[4][:] = [response(body, b"Content-Encoding: gzip\r\n")]
    with pytest.raises(network.PetdexNetworkError, match="compressed"):
        network.fetch_bytes("https://petdex.dev/a", max_bytes=100)


def test_truncated_content_length_is_not_a_success(transport):
    transport[4][:] = [response(b"{}", b"Content-Length: 10\r\n")]
    with pytest.raises(network.PetdexNetworkError, match="truncated"):
        network.fetch_bytes("https://petdex.dev/a", max_bytes=100)
