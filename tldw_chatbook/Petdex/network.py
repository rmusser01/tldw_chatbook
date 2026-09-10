"""Bounded public Petdex HTTPS reads with DNS-to-connection pinning."""

from __future__ import annotations

import http.client
import io
import ipaddress
import queue
import socket
import ssl
import threading
import time
import zlib
from collections.abc import Callable
from urllib.parse import SplitResult, urljoin, urlsplit

from tldw_chatbook.Utils.egress import _classify_ip

ALLOWED_HOSTS = frozenset({"petdex.dev", "assets.petdex.dev"})
TOTAL_TIMEOUT = 30.0
IO_TIMEOUT = 5.0
MAX_REDIRECTS = 3
_CHUNK = 64 * 1024


class PetdexNetworkError(ValueError):
    """Safe transport category, deliberately excluding URLs and response text."""

    def __init__(self, category: str, *, status_code: int | None = None):
        self.category = category
        self.status_code = status_code
        super().__init__(f"Petdex request failed: {category}")


def validate_url(url: str) -> SplitResult:
    """Require credential-free HTTPS on the fixed Petdex hosts and port."""
    try:
        if not isinstance(url, str) or len(url) > 4096:
            raise ValueError
        if any(ord(char) <= 32 or ord(char) >= 127 for char in url) or "\\" in url:
            raise ValueError
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname not in ALLOWED_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in (None, 443)
            or "#" in url
        ):
            raise ValueError
        return parsed
    except (ValueError, TypeError):
        raise PetdexNetworkError("untrusted URL") from None


class _Budget:
    def __init__(self, cancel_requested: Callable[[], bool]):
        self.deadline = time.monotonic() + TOTAL_TIMEOUT
        self.cancel_requested = cancel_requested

    def check(self) -> float:
        if self.cancel_requested():
            raise PetdexNetworkError("cancelled")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise PetdexNetworkError("timeout")
        return min(IO_TIMEOUT, remaining)


def _resolve(host: str, budget: _Budget) -> tuple[int, str]:
    # getaddrinfo has no timeout API. The daemon only resolves: after cancellation
    # it cannot open a connection or publish data. The caller's wait is bounded.
    result: queue.Queue = queue.Queue(maxsize=1)

    def resolve() -> None:
        try:
            result.put(
                socket.getaddrinfo(
                    host, 443, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
                )
            )
        except (OSError, UnicodeError):
            result.put(None)

    budget.check()
    threading.Thread(target=resolve, daemon=True, name="petdex-dns").start()
    dns_deadline = time.monotonic() + IO_TIMEOUT
    while True:
        budget.check()
        if time.monotonic() >= dns_deadline:
            raise PetdexNetworkError("DNS timeout")
        try:
            answers = result.get(timeout=0.05)
            break
        except queue.Empty:
            continue
    budget.check()
    if not answers:
        raise PetdexNetworkError("DNS unavailable")
    approved = []
    for family, _, _, _, address in answers:
        ip = address[0]
        try:
            if (
                family not in (socket.AF_INET, socket.AF_INET6)
                or "%" in ip
                or _classify_ip(ip) != "public"
            ):
                raise ValueError
            approved.append((family, str(ipaddress.ip_address(ip))))
        except ValueError:
            raise PetdexNetworkError("non-public DNS answer") from None
    return approved[0]


class _SocketReader(io.RawIOBase):
    """Apply the deadline/cancellation and raw-wire cap to every socket read."""

    def __init__(self, sock: ssl.SSLSocket, budget: _Budget, wire_limit: int):
        self.sock = sock
        self.budget = budget
        self.remaining = wire_limit
        super().__init__()

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        self.sock.settimeout(self.budget.check())
        data = self.sock.recv(min(len(buffer), self.remaining + 1))
        self.budget.check()
        self.remaining -= len(data)
        if self.remaining < 0:
            raise PetdexNetworkError("wire size limit")
        buffer[: len(data)] = data
        return len(data)


class _ResponseSocket:
    def __init__(self, sock: ssl.SSLSocket, budget: _Budget, max_bytes: int):
        self.sock = sock
        self.budget = budget
        self.max_bytes = max_bytes

    def makefile(self, mode: str) -> io.BufferedReader:
        # Allow bounded header/chunk framing overhead in addition to body bytes.
        return io.BufferedReader(
            _SocketReader(self.sock, self.budget, self.max_bytes + _CHUNK)
        )


def _read_body(
    response: http.client.HTTPResponse, max_bytes: int, budget: _Budget
) -> bytes:
    lengths = response.headers.get_all("Content-Length", [])
    transfers = response.headers.get_all("Transfer-Encoding", [])
    if len(lengths) > 1 or (
        transfers
        and (lengths or len(transfers) != 1 or transfers[0].lower() != "chunked")
    ):
        raise PetdexNetworkError("ambiguous or unsupported response framing")
    encoding = response.getheader("Content-Encoding", "identity").strip().lower()
    if encoding not in ("identity", "gzip"):
        raise PetdexNetworkError("unsupported content encoding")
    length = response.getheader("Content-Length")
    if length is not None:
        try:
            if int(length) < 0 or int(length) > max_bytes:
                raise ValueError
        except ValueError:
            raise PetdexNetworkError("body size limit") from None
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS) if encoding == "gzip" else None
    output = bytearray()
    wire_bytes = 0
    while True:
        budget.check()
        block = response.read1(min(_CHUNK, max_bytes - wire_bytes + 1))
        budget.check()
        if not block:
            break
        wire_bytes += len(block)
        if wire_bytes > max_bytes:
            raise PetdexNetworkError("wire size limit")
        decoded = (
            decoder.decompress(block, max_bytes - len(output) + 1) if decoder else block
        )
        output.extend(decoded)
        if len(output) > max_bytes or (decoder and decoder.unconsumed_tail):
            raise PetdexNetworkError("decoded size limit")
    if length is not None and wire_bytes != int(length):
        raise PetdexNetworkError("truncated response")
    if decoder and (not decoder.eof or decoder.unused_data):
        raise PetdexNetworkError("invalid compressed response")
    return bytes(output)


def fetch_bytes(
    url: str, *, max_bytes: int, cancel_requested: Callable[[], bool] = lambda: False
) -> bytes:
    """Fetch bounded bytes without proxies, pinning DNS and verified TLS peers.

    Args:
        url: Public Petdex registry or asset URL.
        max_bytes: Both wire-body and decoded-body byte limit.
        cancel_requested: Cooperative cancellation, checked at every read boundary.

    Raises:
        PetdexNetworkError: Safe validation, HTTP, size, timeout or cancellation error.
    """
    if type(max_bytes) is not int or max_bytes <= 0:
        raise PetdexNetworkError("invalid size limit")
    budget = _Budget(cancel_requested)
    try:
        for redirects in range(MAX_REDIRECTS + 1):
            budget.check()
            parsed = validate_url(url)
            host = parsed.hostname
            family, ip = _resolve(host, budget)
            sock = socket.socket(family, socket.SOCK_STREAM)
            try:
                sock.settimeout(budget.check())
                sock.connect((ip, 443))
                budget.check()
                context = ssl.create_default_context()
                context.check_hostname = True
                context.verify_mode = ssl.CERT_REQUIRED
                sock.settimeout(budget.check())
                sock = context.wrap_socket(sock, server_hostname=host)
                budget.check()
                peer = sock.getpeername()[0]
                if _classify_ip(peer) != "public" or ipaddress.ip_address(
                    peer
                ) != ipaddress.ip_address(ip):
                    raise PetdexNetworkError("connection peer mismatch")
                target = parsed.path or "/"
                if parsed.query:
                    target += "?" + parsed.query
                request = f"GET {target} HTTP/1.1\r\nHost: {host}\r\nAccept-Encoding: gzip, identity\r\nConnection: close\r\nUser-Agent: tldw-chatbook-petdex\r\n\r\n"
                sock.settimeout(budget.check())
                sock.sendall(request.encode("ascii"))
                budget.check()
                with http.client.HTTPResponse(
                    _ResponseSocket(sock, budget, max_bytes)
                ) as response:
                    response.begin()
                    budget.check()
                    if response.status in (301, 302, 303, 307, 308):
                        location = response.getheader("Location")
                        if not location or redirects == MAX_REDIRECTS:
                            raise PetdexNetworkError(
                                "redirect limit or missing location"
                            )
                        # Validate the raw reference before urljoin can strip controls.
                        if (
                            any(
                                ord(char) <= 32 or ord(char) >= 127 for char in location
                            )
                            or "\\" in location
                        ):
                            raise PetdexNetworkError("untrusted redirect")
                        url = urljoin(url, location)
                        validate_url(url)
                        continue
                    if response.status != 200:
                        category = {
                            401: "authentication required",
                            403: "access denied",
                            429: "rate limited",
                        }.get(response.status, "HTTP error")
                        raise PetdexNetworkError(category, status_code=response.status)
                    return _read_body(response, max_bytes, budget)
            finally:
                sock.close()
    except PetdexNetworkError:
        raise
    except TimeoutError:
        raise PetdexNetworkError("timeout") from None
    except ssl.SSLError:
        raise PetdexNetworkError("TLS verification or transport error") from None
    except (OSError, http.client.HTTPException, zlib.error, ValueError):
        raise PetdexNetworkError("invalid or unavailable response") from None
    raise PetdexNetworkError("redirect limit")
