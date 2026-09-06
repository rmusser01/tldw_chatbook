"""Origin-wide browser authentication for Chatbook's served mode.

The configured credential is used only to admit a browser.  Successful login
creates an opaque, in-memory session; the long-lived credential is never put in
a URL, cookie, response, exception, or representation.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
import secrets
import socket
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from urllib.parse import urlsplit

WEB_ACCESS_TOKEN_ENV = "TLDW_CHATBOOK_WEB_ACCESS_TOKEN"
WEB_ACCESS_TOKEN_KEYRING_SERVICE = "tldw_chatbook_web"
WEB_ACCESS_TOKEN_KEYRING_ACCOUNT = "access_token"
SESSION_COOKIE_NAME = "chatbook_session"
CSRF_HEADER_NAME = "X-Chatbook-CSRF"
WEBSOCKET_PROTOCOL = "chatbook-v1"

_DEFAULT_BOOTSTRAP_TTL_SECONDS = 60.0
_DEFAULT_IDLE_TIMEOUT_SECONDS = 30 * 60.0
_DEFAULT_ABSOLUTE_TIMEOUT_SECONDS = 8 * 60 * 60.0
_DEFAULT_LOGIN_ATTEMPTS_PER_MINUTE = 8
_DEFAULT_MAX_SESSIONS = 512
_DEFAULT_MAX_BOOTSTRAPS = 64
_MAX_RATE_LIMIT_SUBJECTS = 2048


class BindPolicyError(ValueError):
    """Raised when a requested served-mode exposure is unsafe."""


class AuthenticationError(PermissionError):
    """A content-free browser authentication refusal."""


@dataclass(frozen=True)
class ResolvedCredential:
    """A resolved admission credential whose representation is always redacted."""

    _value: str | None = field(repr=False)
    source: str

    def reveal(self) -> str | None:
        """Return the value only to the authentication boundary."""

        return self._value

    def __repr__(self) -> str:
        return f"ResolvedCredential(source={self.source!r}, value=<redacted>)"


@dataclass(frozen=True)
class WebAuthPolicy:
    """Validated network exposure and origin policy."""

    bind_host: str
    port: int
    local_only: bool
    allowed_hosts: frozenset[str]
    allowed_authorities: frozenset[tuple[str, int]]
    external_scheme: str
    access_credential: ResolvedCredential = field(repr=False)
    trusted_proxy_addresses: frozenset[str] = frozenset()
    insecure_remote_http: bool = False
    direct_tls: bool = False

    @property
    def secure_cookies(self) -> bool:
        return self.external_scheme == "https"

    @property
    def automatic_local_login(self) -> bool:
        """Whether direct loopback browsers may enter without token ceremony."""

        return self.local_only and all(
            is_loopback_host(host) for host in self.allowed_hosts
        )

    def is_trusted_proxy(self, address: str) -> bool:
        normalized = _normalize_ip(address)
        return normalized is not None and normalized in self.trusted_proxy_addresses


@dataclass(frozen=True)
class SessionGrant:
    """Values needed to establish an authenticated browser session."""

    cookie_value: str = field(repr=False)
    csrf_token: str = field(repr=False)
    expires_at: float


@dataclass
class BrowserSession:
    """Private in-memory browser session state."""

    session_id: str
    cookie_digest: bytes = field(repr=False)
    csrf_token: str = field(repr=False)
    client_ip: str
    created_at: float
    last_seen_at: float
    absolute_expires_at: float
    revoked: bool = False


@dataclass(frozen=True)
class RequestFacts:
    """Framework-neutral request facts consumed by the auth boundary."""

    method: str
    path: str
    peer_ip: str
    scheme: str
    host: str
    origin: str | None = None
    cookie_value: str | None = field(default=None, repr=False)
    csrf_token: str | None = field(default=None, repr=False)
    upgrade: str = ""
    connection: str = ""
    websocket_protocols: tuple[str, ...] = field(default=(), repr=False)
    forwarded_for: str | None = None
    forwarded_proto: str | None = None
    forwarded_host: str | None = None
    fetch_site: str | None = None


def _nonempty(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def resolve_web_access_token(
    configured_value: object,
    *,
    environ: Mapping[str, str] | None = None,
    keyring_get: Callable[[str, str], str | None] | None = None,
) -> ResolvedCredential:
    """Resolve only the dedicated web admission credential.

    Precedence is environment, config, then the dedicated OS-keyring entry.
    Keyring failures are treated as absence and are never logged with values.
    """

    environment = os.environ if environ is None else environ
    from_environment = _nonempty(environment.get(WEB_ACCESS_TOKEN_ENV))
    if from_environment is not None:
        return ResolvedCredential(from_environment, "environment")

    from_config = _nonempty(configured_value)
    if from_config is not None:
        return ResolvedCredential(from_config, "config")

    if keyring_get is None:
        try:
            import keyring

            keyring_get = keyring.get_password
        except Exception:  # noqa: BLE001  # pragma: no cover - plugin import boundary
            keyring_get = None
    if keyring_get is not None:
        try:
            from_keyring = _nonempty(
                keyring_get(
                    WEB_ACCESS_TOKEN_KEYRING_SERVICE,
                    WEB_ACCESS_TOKEN_KEYRING_ACCOUNT,
                )
            )
        except Exception:  # noqa: BLE001 - OS keyring backends raise vendor errors
            from_keyring = None
        if from_keyring is not None:
            return ResolvedCredential(from_keyring, "keyring")
    return ResolvedCredential(None, "missing")


def _strip_brackets(host: str) -> str:
    value = host.strip()
    if value.startswith("[") and value.endswith("]"):
        return value[1:-1]
    return value


def _normalize_ip(value: str) -> str | None:
    try:
        return str(ipaddress.ip_address(_strip_brackets(value).split("%", 1)[0]))
    except ValueError:
        return None


def is_loopback_host(
    host: str,
    *,
    resolver: Callable[..., Sequence[tuple]] = socket.getaddrinfo,
) -> bool:
    """Return whether a bind target can resolve only to loopback addresses."""

    normalized_host = _strip_brackets(host).rstrip(".").lower()
    parsed = _normalize_ip(normalized_host)
    if parsed is not None:
        return ipaddress.ip_address(parsed).is_loopback
    if normalized_host != "localhost":
        return False
    try:
        answers = resolver(normalized_host, None, type=socket.SOCK_STREAM)
    except (OSError, ValueError):
        return False
    addresses = {
        _normalize_ip(str(answer[4][0]))
        for answer in answers
        if len(answer) >= 5 and answer[4]
    }
    return (
        bool(addresses)
        and None not in addresses
        and all(
            ipaddress.ip_address(address).is_loopback
            for address in addresses
            if address
        )
    )


def _parse_public_origin(public_url: str) -> tuple[str, str, int | None]:
    parsed = urlsplit(public_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise BindPolicyError("web_server.public_url must be an HTTP(S) origin")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise BindPolicyError("web_server.public_url must contain only an origin")
    if parsed.path not in {"", "/"}:
        raise BindPolicyError("web_server.public_url must contain only an origin")
    try:
        parsed_port = parsed.port
    except ValueError as exc:
        raise BindPolicyError("web_server.public_url has an invalid port") from exc
    return parsed.scheme, parsed.hostname.rstrip(".").lower(), parsed_port


def _validated_proxy_addresses(values: Sequence[str]) -> frozenset[str]:
    normalized: set[str] = set()
    for value in values:
        parsed = _normalize_ip(value)
        if parsed is None:
            raise BindPolicyError(
                "trusted proxy addresses must be literal IP addresses"
            )
        normalized.add(parsed)
    return frozenset(normalized)


def build_web_auth_policy(
    *,
    host: str,
    port: int,
    access_token: str | ResolvedCredential | None,
    public_url: str | None = None,
    allow_insecure_remote_http: bool = False,
    trusted_proxy_addresses: Sequence[str] = (),
    direct_tls: bool = False,
) -> WebAuthPolicy:
    """Validate bind, TLS, proxy, and credential policy before listening."""

    credential = (
        access_token
        if isinstance(access_token, ResolvedCredential)
        else ResolvedCredential(_nonempty(access_token), "explicit")
    )
    local_only = is_loopback_host(host)
    proxies = _validated_proxy_addresses(trusted_proxy_addresses)

    if not local_only and credential.reveal() is None:
        raise BindPolicyError("non-loopback bind requires a Chatbook web access token")

    wildcard = _strip_brackets(host) in {"0.0.0.0", "::"}
    if public_url is None:
        if wildcard and not local_only:
            raise BindPolicyError("a wildcard bind requires web_server.public_url")
        scheme = "https" if direct_tls else "http"
        allowed_host = _strip_brackets(host).rstrip(".").lower()
    else:
        scheme, allowed_host, public_port = _parse_public_origin(public_url)

    externally_remote = not is_loopback_host(allowed_host)
    remote_exposure = not local_only or externally_remote
    if remote_exposure and credential.reveal() is None:
        raise BindPolicyError("remote served mode requires a Chatbook web access token")

    proxy_tls = scheme == "https" and bool(proxies)
    secure_transport = direct_tls or proxy_tls
    if scheme == "https" and not secure_transport:
        raise BindPolicyError(
            "an https public origin requires direct TLS or a trusted TLS proxy"
        )
    insecure_remote = remote_exposure and not secure_transport
    if insecure_remote and not allow_insecure_remote_http:
        raise BindPolicyError(
            "non-loopback served mode requires HTTPS or a trusted HTTPS reverse proxy"
        )
    if direct_tls and scheme != "https":
        raise BindPolicyError("direct TLS requires an https public origin")

    if local_only and public_url is None:
        allowed_hosts = {allowed_host, "localhost", "127.0.0.1", "::1"}
        allowed_authorities = {(candidate, port) for candidate in allowed_hosts}
    else:
        allowed_hosts = {allowed_host}
        allowed_authorities = {
            (allowed_host, _origin_port(scheme, public_port if public_url else port))
        }
    return WebAuthPolicy(
        bind_host=host,
        port=port,
        local_only=local_only,
        allowed_hosts=frozenset(allowed_hosts),
        allowed_authorities=frozenset(allowed_authorities),
        external_scheme=scheme,
        access_credential=credential,
        trusted_proxy_addresses=proxies,
        insecure_remote_http=insecure_remote,
        direct_tls=direct_tls,
    )


def _cookie_digest(value: str) -> bytes:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).digest()


def _constant_time_secret_matches(supplied: str, expected: str) -> bool:
    """Compare UTF-8 secrets without letting secret-bearing errors escape."""

    try:
        supplied_bytes = str(supplied).encode("utf-8")
        expected_bytes = expected.encode("utf-8")
        return hmac.compare_digest(supplied_bytes, expected_bytes)
    except Exception:  # noqa: BLE001 - credential locals must never reach traceback logs
        return False


def _parse_host_header(value: str) -> tuple[str, int | None]:
    if (
        not value
        or value.endswith(":")
        or any(character in value for character in "\r\n,/@\\")
    ):
        raise AuthenticationError("invalid host")
    try:
        parsed = urlsplit(f"//{value}")
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise AuthenticationError("invalid host") from exc
    if (
        not hostname
        or parsed.username
        or parsed.password
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise AuthenticationError("invalid host")
    return hostname.rstrip(".").lower(), parsed.port


def _parse_origin(value: str) -> tuple[str, str, int | None]:
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError as exc:
        raise AuthenticationError("invalid origin") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise AuthenticationError("invalid origin")
    return parsed.scheme, parsed.hostname.rstrip(".").lower(), parsed.port


def _origin_port(scheme: str, explicit_port: int | None) -> int:
    return (
        explicit_port
        if explicit_port is not None
        else (443 if scheme == "https" else 80)
    )


class WebAuthManager:
    """Own one process's bootstrap nonces, sessions, and admission rate limits."""

    def __init__(
        self,
        policy: WebAuthPolicy,
        *,
        clock: Callable[[], float] = time.monotonic,
        token_factory: Callable[[int], str] = secrets.token_urlsafe,
        bootstrap_ttl_seconds: float = _DEFAULT_BOOTSTRAP_TTL_SECONDS,
        idle_timeout_seconds: float = _DEFAULT_IDLE_TIMEOUT_SECONDS,
        absolute_timeout_seconds: float = _DEFAULT_ABSOLUTE_TIMEOUT_SECONDS,
        login_attempts_per_minute: int = _DEFAULT_LOGIN_ATTEMPTS_PER_MINUTE,
        max_sessions: int = _DEFAULT_MAX_SESSIONS,
        max_bootstraps: int = _DEFAULT_MAX_BOOTSTRAPS,
    ) -> None:
        self.policy = policy
        self._clock = clock
        self._token_factory = token_factory
        self._bootstrap_ttl = max(1.0, float(bootstrap_ttl_seconds))
        self._idle_timeout = max(1.0, float(idle_timeout_seconds))
        self._absolute_timeout = max(
            self._idle_timeout, float(absolute_timeout_seconds)
        )
        self._login_limit = max(1, int(login_attempts_per_minute))
        self._max_sessions = max(1, int(max_sessions))
        self._max_bootstraps = max(1, int(max_bootstraps))
        self._bootstraps: OrderedDict[bytes, float] = OrderedDict()
        self._sessions: OrderedDict[bytes, BrowserSession] = OrderedDict()
        self._attempts: OrderedDict[str, deque[float]] = OrderedDict()
        self._revocation_callbacks: dict[str, set[Callable[[], None]]] = {}

    @property
    def rate_limit_subject_count(self) -> int:
        return len(self._attempts)

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    @property
    def bootstrap_count(self) -> int:
        return len(self._bootstraps)

    def _prune_bootstraps(self, now: float) -> None:
        expired = [
            digest for digest, expiry in self._bootstraps.items() if now >= expiry
        ]
        for digest in expired:
            self._bootstraps.pop(digest, None)

    def issue_bootstrap(self) -> str:
        now = self._clock()
        self._prune_bootstraps(now)
        while len(self._bootstraps) >= self._max_bootstraps:
            self._bootstraps.popitem(last=False)
        nonce = self._token_factory(32)
        self._bootstraps[_cookie_digest(nonce)] = now + self._bootstrap_ttl
        return nonce

    def exchange_bootstrap(self, nonce: str, *, client_ip: str) -> SessionGrant:
        now = self._clock()
        self._prune_bootstraps(now)
        expiry = self._bootstraps.pop(_cookie_digest(nonce), None)
        if expiry is None or now >= expiry:
            raise AuthenticationError("authentication denied")
        return self._new_session(client_ip, now)

    def _record_attempt(self, client_ip: str) -> None:
        now = self._clock()
        attempts = self._attempts.pop(client_ip, deque())
        cutoff = now - 60.0
        while attempts and attempts[0] <= cutoff:
            attempts.popleft()
        if len(attempts) >= self._login_limit:
            self._attempts[client_ip] = attempts
            raise AuthenticationError("authentication temporarily unavailable")
        attempts.append(now)
        self._attempts[client_ip] = attempts
        while len(self._attempts) > _MAX_RATE_LIMIT_SUBJECTS:
            self._attempts.popitem(last=False)

    def login_with_access_token(self, supplied: str, *, client_ip: str) -> SessionGrant:
        self._record_attempt(client_ip)
        expected = self.policy.access_credential.reveal()
        if expected is None or not _constant_time_secret_matches(supplied, expected):
            raise AuthenticationError("authentication denied")
        return self._new_session(client_ip, self._clock())

    def authenticate_local(self, *, client_ip: str) -> SessionGrant:
        if not self.policy.local_only or not is_loopback_host(client_ip):
            raise AuthenticationError("authentication denied")
        return self._new_session(client_ip, self._clock())

    def _new_session(self, client_ip: str, now: float) -> SessionGrant:
        self._prune_sessions(now)
        while len(self._sessions) >= self._max_sessions:
            eviction_candidate = next(
                (
                    session
                    for session in self._sessions.values()
                    if session.session_id not in self._revocation_callbacks
                ),
                None,
            )
            if eviction_candidate is None:
                raise AuthenticationError("session capacity unavailable")
            self._revoke_session(eviction_candidate)
        cookie = self._token_factory(32)
        csrf = self._token_factory(24)
        absolute_expiry = now + self._absolute_timeout
        digest = _cookie_digest(cookie)
        self._sessions[digest] = BrowserSession(
            session_id=self._token_factory(18),
            cookie_digest=digest,
            csrf_token=csrf,
            client_ip=client_ip,
            created_at=now,
            last_seen_at=now,
            absolute_expires_at=absolute_expiry,
        )
        return SessionGrant(cookie, csrf, absolute_expiry)

    def _prune_sessions(self, now: float) -> None:
        expired = [
            session
            for session in self._sessions.values()
            if now
            >= min(
                session.last_seen_at + self._idle_timeout,
                session.absolute_expires_at,
            )
        ]
        for session in expired:
            self._revoke_session(session)

    def _effective_request(self, facts: RequestFacts) -> tuple[str, str, str]:
        client_ip = facts.peer_ip
        scheme = facts.scheme.lower()
        host = facts.host
        if self.policy.is_trusted_proxy(facts.peer_ip):
            if not facts.forwarded_for or _normalize_ip(facts.forwarded_for) is None:
                raise AuthenticationError("invalid forwarded headers")
            if facts.forwarded_proto not in {"http", "https"}:
                raise AuthenticationError("invalid forwarded headers")
            if not facts.forwarded_host:
                raise AuthenticationError("invalid forwarded headers")
            try:
                _parse_host_header(facts.forwarded_host)
            except AuthenticationError as exc:
                raise AuthenticationError("invalid forwarded headers") from exc
            client_ip = str(ipaddress.ip_address(facts.forwarded_for))
            scheme = facts.forwarded_proto
            host = facts.forwarded_host
        host_name, host_port = _parse_host_header(host)
        return client_ip, scheme, f"{host_name}:{_origin_port(scheme, host_port)}"

    def validate_public_request(
        self, facts: RequestFacts, *, require_origin: bool
    ) -> tuple[str, str, str]:
        """Validate routing headers before a public login/bootstrap action."""

        client_ip, scheme, authority = self._effective_request(facts)
        host_name, _, host_port_text = authority.rpartition(":")
        host_port = int(host_port_text)
        if (
            not self.policy.insecure_remote_http
            and scheme != self.policy.external_scheme
        ):
            raise AuthenticationError("invalid transport")
        if (host_name, host_port) not in self.policy.allowed_authorities:
            raise AuthenticationError("invalid host")
        if require_origin:
            if facts.origin is None:
                raise AuthenticationError("invalid origin")
            origin_scheme, origin_host, origin_explicit_port = _parse_origin(
                facts.origin
            )
            if (
                origin_scheme != scheme
                or origin_host != host_name
                or _origin_port(origin_scheme, origin_explicit_port) != host_port
            ):
                raise AuthenticationError("invalid origin")
        return client_ip, scheme, host_name

    def authenticate_request(
        self,
        facts: RequestFacts,
        *,
        require_csrf: bool = False,
        websocket: bool = False,
    ) -> BrowserSession:
        client_ip, scheme, host_name = self.validate_public_request(
            facts, require_origin=False
        )

        if facts.origin is None:
            if require_csrf or websocket:
                raise AuthenticationError("invalid origin")
        else:
            origin_scheme, origin_host, origin_explicit_port = _parse_origin(
                facts.origin
            )
            _, host_explicit_port = _parse_host_header(
                facts.forwarded_host
                if self.policy.is_trusted_proxy(facts.peer_ip)
                and facts.forwarded_host is not None
                else facts.host
            )
            if (
                origin_scheme != scheme
                or origin_host != host_name
                or _origin_port(origin_scheme, origin_explicit_port)
                != _origin_port(scheme, host_explicit_port)
            ):
                raise AuthenticationError("invalid origin")

        if not facts.cookie_value:
            raise AuthenticationError("invalid session")
        session = self._sessions.get(_cookie_digest(facts.cookie_value))
        if session is None or session.revoked:
            raise AuthenticationError("invalid session")
        self.touch_session(session)

        if websocket:
            if facts.method.upper() != "GET":
                raise AuthenticationError("invalid websocket upgrade")
            if facts.upgrade.lower() != "websocket" or "upgrade" not in {
                value.strip().lower() for value in facts.connection.split(",")
            }:
                raise AuthenticationError("invalid websocket upgrade")
            csrf_values = [
                protocol[5:]
                for protocol in facts.websocket_protocols
                if protocol.startswith("csrf.")
            ]
            if len(csrf_values) != 1 or not _constant_time_secret_matches(
                csrf_values[0], session.csrf_token
            ):
                raise AuthenticationError("invalid CSRF proof")
            if WEBSOCKET_PROTOCOL not in facts.websocket_protocols:
                raise AuthenticationError("invalid websocket protocol")
        elif require_csrf and (
            facts.csrf_token is None
            or not _constant_time_secret_matches(facts.csrf_token, session.csrf_token)
        ):
            raise AuthenticationError("invalid CSRF proof")

        session.client_ip = client_ip
        return session

    def revoke(self, cookie_value: str) -> None:
        session = self._sessions.get(_cookie_digest(cookie_value))
        if session is not None:
            self._revoke_session(session)

    def register_channel(
        self, session: BrowserSession, close_callback: Callable[[], None]
    ) -> Callable[[], None]:
        """Close a live browser channel immediately if its session is revoked."""

        self.touch_session(session)
        callbacks = self._revocation_callbacks.setdefault(session.session_id, set())
        callbacks.add(close_callback)

        def unregister() -> None:
            registered = self._revocation_callbacks.get(session.session_id)
            if registered is None:
                return
            registered.discard(close_callback)
            if not registered:
                self._revocation_callbacks.pop(session.session_id, None)

        return unregister

    def seconds_until_expiry(self, session: BrowserSession) -> float:
        """Return the bounded delay until a session's next expiry boundary."""

        deadline = min(
            session.last_seen_at + self._idle_timeout,
            session.absolute_expires_at,
        )
        return max(0.0, deadline - self._clock())

    def expire_session_if_due(self, session: BrowserSession) -> bool:
        """Revoke a session once either idle or absolute lifetime is exhausted."""

        if self.seconds_until_expiry(session) > 0:
            return False
        current = self._sessions.get(session.cookie_digest)
        if current is not session:
            return True
        self._revoke_session(session)
        return True

    def touch_session(self, session: BrowserSession) -> None:
        """Record authenticated browser activity or fail when already expired."""

        if session.revoked or self._sessions.get(session.cookie_digest) is not session:
            raise AuthenticationError("invalid session")
        if self.expire_session_if_due(session):
            raise AuthenticationError("session expired")
        session.last_seen_at = self._clock()
        self._sessions.move_to_end(session.cookie_digest)

    def _revoke_session(self, session: BrowserSession) -> None:
        self._sessions.pop(session.cookie_digest, None)
        session.revoked = True
        callbacks = self._revocation_callbacks.pop(session.session_id, set())
        for callback in callbacks:
            callback()

    def revoke_all(self) -> None:
        for session in self._sessions.values():
            session.revoked = True
        callbacks = [
            callback
            for registered in self._revocation_callbacks.values()
            for callback in registered
        ]
        self._revocation_callbacks.clear()
        self._sessions.clear()
        self._bootstraps.clear()
        for callback in callbacks:
            callback()
