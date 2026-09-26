"""TASK-26022: read-only Claude subscription credential borrow.

A Claude Pro/Max subscriber otherwise pays API rates on top of a subscription
they already hold. This module reads the OAuth credential that Claude Code
itself minted (``~/.claude/.credentials.json``) so Anthropic requests can carry
the subscription authorization header instead of an API key.

Deliberately the NARROW slice, per the task and its ADR note:
- **Read-only.** The credential is never written, refreshed, or rotated here.
  A stale credential produces a clear "refresh it in Claude Code" message
  (AC#2). If this ever grows toward minting/refreshing tokens, stop and raise
  an ADR first — that crosses into owning a credential lifecycle.
- **Explicit opt-in** (AC#4): only ``[api_settings.anthropic]
  auth_source = "claude_subscription"`` activates it; a credential discovered
  on disk never silently changes how requests are billed.
- **Never leaks** (AC#3): the dataclass masks the token in ``repr``/``str``,
  and the log sanitizer covers the ``sk-ant-oat01-`` shape.

Owner decision 2026-09-02: proceed; the ToS/account-risk call is the owner's,
and AC#7 (live verification against a real subscription) is owner-driven
before the task closes.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Literal

#: Claude Code's credential file. Chatbook only ever reads it.
DEFAULT_CREDENTIALS_PATH = Path.home() / ".claude" / ".credentials.json"

#: Beta flag the subscription authorization path requires.
OAUTH_BETA = "oauth-2025-04-20"

#: macOS Keychain service Claude Code stores its credential under. On macOS the
#: credential lives here, NOT in the file (found by AC#7 live verify 2026-09-02).
KEYCHAIN_SERVICE = "Claude Code-credentials"

#: Short TTL for the Keychain read memo. Readiness runs on UI redraw paths and
#: would otherwise spawn `security` on every evaluation (and stall up to the
#: subprocess timeout if the Keychain is locked). The memo caches success AND a
#: None result. The lock also coalesces concurrent send/background reads.
_KEYCHAIN_TTL_S = 5.0
_KEYCHAIN_CACHE: tuple[float, str | None] | None = None
_KEYCHAIN_LOCK = Lock()

#: The subscription OAuth token is gated to the Claude Code identity: Anthropic
#: rejects (as a misleading 429) any request whose ``system`` does not lead with
#: this line. Verified against a real Max account 2026-09-02. Borrowing the
#: credential therefore means presenting as Claude Code, which is exactly the
#: credential's own scope (``user:sessions:claude_code``).
CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."

AUTH_SOURCE_API_KEY = "api_key"
AUTH_SOURCE_SUBSCRIPTION = "claude_subscription"

#: Copy for AC#2 — chatbook never refreshes; the owning tool does.
STALE_CREDENTIAL_MESSAGE = (
    "The Claude subscription credential is expired. Chatbook only reads it — "
    "refresh it in the tool that owns it (run Claude Code and log in again), "
    'or set [api_settings.anthropic] auth_source back to "api_key".'
)
MISSING_CREDENTIAL_MESSAGE = (
    'auth_source is "claude_subscription" but no Claude Code credential was '
    "found (checked ~/.claude/.credentials.json and, on macOS, the login "
    'Keychain item "Claude Code-credentials"). Log in with Claude Code first '
    "(and unlock your Keychain if prompted), or set [api_settings.anthropic] "
    'auth_source back to "api_key".'
)


@dataclass(frozen=True)
class SubscriptionCredential:
    """One borrowed, read-only subscription credential.

    ``repr``/``str`` never include the token (AC#3).
    """

    access_token: str = field(repr=False)
    expires_at_ms: int = field(default=0)
    subscription_type: str = ""
    source_path: str = ""

    @property
    def expired(self) -> bool:
        if self.expires_at_ms <= 0:
            return False  # no expiry recorded -> let the API be the judge
        return time.time() * 1000 >= self.expires_at_ms

    def __repr__(self) -> str:  # defensive: no token, ever
        return (
            f"SubscriptionCredential(source={self.source_path!r}, "
            f"type={self.subscription_type!r}, expired={self.expired})"
        )

    __str__ = __repr__


SubscriptionStatus = Literal["pending", "ready", "expired", "missing"]


class _SubscriptionReadinessCache:
    """One bounded, secret-free snapshot and at most one credential reader.

    UI callers never join a worker or acquire the Keychain lock. Synchronous
    send callers continue to read the actual credential independently.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._worker: Thread | None = None
        self._refreshing = False
        self._path: Path | None = None
        self._completed_at: float | None = None
        self._present = False
        self._expires_at_ms = 0
        self._observed_once = False
        self.revision = 0

    def status(self) -> SubscriptionStatus:
        with self._lock:
            if (
                self._path == DEFAULT_CREDENTIALS_PATH
                and self._completed_at is not None
                and time.monotonic() - self._completed_at < _KEYCHAIN_TTL_S
            ):
                if not self._present:
                    return "missing"
                return (
                    "expired"
                    if self._expires_at_ms > 0
                    and time.time() * 1000 >= self._expires_at_ms
                    else "ready"
                )
            if not self._refreshing:
                self._refreshing = True
                self._worker = Thread(
                    target=self._refresh,
                    args=(DEFAULT_CREDENTIALS_PATH,),
                    name="claude-credential-readiness",
                    daemon=True,
                )
                self._worker.start()
            return "pending"

    def _refresh(self, path: Path) -> None:
        present = False
        expires_at_ms = 0
        try:
            credential = read_claude_code_credential()
            if credential is not None:
                present = True
                expires_at_ms = credential.expires_at_ms
        except Exception:  # noqa: BLE001 - credential errors must remain secret-free
            # Never publish exception text, file content, paths, or tokens.
            # A failed background read gets the same bounded TTL as a miss.
            present = False
        with self._lock:
            # task-32904: value-gate the revision. A TTL refresh that
            # observed the exact same state must not wake the 4Hz UI
            # pollers into full summary rebuilds every few seconds while
            # chatting; only a real state change (or the first completed
            # observation) publishes a new revision.
            changed = (
                not self._observed_once
                or self._present != present
                or self._expires_at_ms != expires_at_ms
            )
            self._path = path
            self._present = present
            self._expires_at_ms = expires_at_ms
            self._completed_at = time.monotonic()
            self._refreshing = False
            self._observed_once = True
            if changed:
                self.revision += 1


_SUBSCRIPTION_READINESS_CACHE = _SubscriptionReadinessCache()


def subscription_credential_status(*, background: bool = False) -> SubscriptionStatus:
    """Resolve credential presence without exposing its token to readiness.

    Args:
        background: Return a cached UI snapshot immediately, starting one
            background read when stale. The default resolves synchronously for
            send-time checks and never mistakes unfinished UI work for absence.

    Returns:
        A bounded state; ``pending`` occurs only for a background lookup.
    """
    if background:
        return _SUBSCRIPTION_READINESS_CACHE.status()
    credential = read_claude_code_credential()
    if credential is None:
        return "missing"
    return "expired" if credential.expired else "ready"


def subscription_readiness_revision() -> int:
    """Return the completion revision for mounted UI refresh polling, without I/O."""
    return _SUBSCRIPTION_READINESS_CACHE.revision


def read_claude_code_credential(
    path: Path | str | None = None,
) -> SubscriptionCredential | None:
    """Read Claude Code's credential file. Read-only; never raises outward.

    Args:
        path: Credential file to read; ``None`` uses
            ``DEFAULT_CREDENTIALS_PATH`` (``~/.claude/.credentials.json``).

    Returns:
        ``None`` when the file is missing or malformed (AC#6: an absent
        credential leaves behavior exactly as today). An EXPIRED credential is
        returned with ``expired=True`` so the caller can show the AC#2
        refresh message instead of a generic missing-credential one.
    """
    explicit = path is not None
    target = Path(path) if explicit else DEFAULT_CREDENTIALS_PATH
    try:
        file_text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # I1: a non-UTF-8 file must fall through, never raise into callers
        # (readiness runs on UI paths). UnicodeDecodeError is a ValueError,
        # not an OSError, so it must be caught explicitly.
        file_text = None
    if file_text is not None:
        cred = _parse_oauth_json(file_text, source_label=str(target))
        if cred is not None:
            return cred
    # An EXPLICIT path means "read that file, period" -- never reach for the
    # Keychain (I2: keeps callers that pass a path hermetic). The Keychain
    # fallback is only for the default-location read the real callers use.
    if explicit:
        return None
    # Fallback: on macOS Claude Code stores the credential in the Keychain, not
    # the file (AC#7 live verify). Read-only, like the file path.
    keychain_text = _keychain_credential_raw()
    if keychain_text is not None:
        return _parse_oauth_json(
            keychain_text, source_label=f"keychain:{KEYCHAIN_SERVICE}"
        )
    return None


def _parse_oauth_json(
    raw_text: str, *, source_label: str
) -> SubscriptionCredential | None:
    """Parse a Claude Code credential JSON blob into a credential.

    Args:
        raw_text: The raw JSON text from the file or Keychain.
        source_label: A non-secret label describing where it came from; stored
            in ``source_path`` (never the token).

    Returns:
        The parsed credential, or ``None`` when the blob is missing the OAuth
        section or a usable access token.
    """
    try:
        raw = json.loads(raw_text)
    except (ValueError, UnicodeDecodeError):
        return None
    oauth = raw.get("claudeAiOauth") if isinstance(raw, dict) else None
    if not isinstance(oauth, dict):
        return None
    token = oauth.get("accessToken")
    if not isinstance(token, str) or not token.strip():
        return None
    token = token.strip()
    try:
        expires_at_ms = int(oauth.get("expiresAt") or 0)
    except (TypeError, ValueError, OverflowError):
        return None
    return SubscriptionCredential(
        access_token=token,
        expires_at_ms=expires_at_ms,
        subscription_type=str(oauth.get("subscriptionType") or ""),
        source_path=source_label,
    )


def _keychain_credential_raw() -> str | None:
    """Return Claude Code's Keychain credential JSON on macOS, else ``None``.

    Read-only: shells out to ``security find-generic-password -w``. Any failure
    (non-macOS, item absent, ``security`` unavailable) yields ``None`` so the
    caller falls through to today's no-credential behavior (AC#6).

    Returns:
        The raw JSON string stored under ``KEYCHAIN_SERVICE``, or ``None``.
    """
    global _KEYCHAIN_CACHE
    if sys.platform != "darwin":
        return None
    with _KEYCHAIN_LOCK:
        cached = _KEYCHAIN_CACHE
        if cached is not None and time.monotonic() - cached[0] < _KEYCHAIN_TTL_S:
            return cached[1]
        result: str | None = None
        try:
            proc = subprocess.run(
                [
                    "/usr/bin/security",
                    "find-generic-password",
                    "-s",
                    KEYCHAIN_SERVICE,
                    "-w",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            # ValueError covers a non-decodable stdout under text=True.
            proc = None
        if proc is not None and proc.returncode == 0:
            result = (proc.stdout or "").strip() or None
        _KEYCHAIN_CACHE = (time.monotonic(), result)
        return result


def anthropic_auth_source(anthropic_config: Mapping[str, Any] | None) -> str:
    """The configured auth source; anything unrecognized is the safe default.

    Args:
        anthropic_config: The ``[api_settings.anthropic]`` mapping (or the
            legacy mapping); may be ``None``.

    Returns:
        ``"claude_subscription"`` only for that exact configured value;
        otherwise ``"api_key"``.
    """
    raw = str((anthropic_config or {}).get("auth_source") or "").strip().lower()
    if raw == AUTH_SOURCE_SUBSCRIPTION:
        return AUTH_SOURCE_SUBSCRIPTION
    return AUTH_SOURCE_API_KEY


def subscription_headers(credential: SubscriptionCredential) -> dict[str, str]:
    """The auth headers for the subscription path (AC#7's shape).

    Replaces ``x-api-key`` entirely — the caller must not send both.

    Args:
        credential: A parsed, non-expired subscription credential.

    Returns:
        The ``authorization`` bearer header plus the OAuth beta flag.
    """
    return {
        "authorization": f"Bearer {credential.access_token}",
        "anthropic-beta": OAUTH_BETA,
    }


def subscription_headers_for_token(access_token: str) -> dict[str, str]:
    """`subscription_headers` for an already-extracted token.

    Args:
        access_token: The credential's access token.

    Returns:
        The ``authorization`` bearer header plus the OAuth beta flag.
    """
    return {
        "authorization": f"Bearer {access_token}",
        "anthropic-beta": OAUTH_BETA,
    }


def with_claude_code_identity(system: Any) -> list[dict[str, Any]]:
    """Lead an Anthropic ``system`` value with the Claude Code identity block.

    The subscription OAuth token is rejected unless the request's ``system``
    begins with :data:`CLAUDE_CODE_IDENTITY`, so the subscription send path runs
    the caller's system prompt through this. The caller's own prompt is
    preserved as following block(s); already-led inputs are returned unchanged
    (idempotent).

    Args:
        system: The ``system`` value the caller assembled: ``None``, a string,
            or a list of Anthropic text blocks.

    Returns:
        A list of Anthropic text blocks whose first block is the identity.
    """
    identity = {"type": "text", "text": CLAUDE_CODE_IDENTITY}
    if system is None or (isinstance(system, str) and not system.strip()):
        return [identity]
    if isinstance(system, str):
        if system.startswith(CLAUDE_CODE_IDENTITY):  # M1: don't double-prepend
            return [{"type": "text", "text": system}]
        return [identity, {"type": "text", "text": system}]
    if isinstance(system, list):
        blocks = list(system)
        first = blocks[0] if blocks else None
        if isinstance(first, dict) and str(first.get("text", "")).startswith(
            CLAUDE_CODE_IDENTITY
        ):
            return blocks
        return [identity, *blocks]
    # M2: an unexpected shape is preserved as text, never silently dropped.
    return [identity, {"type": "text", "text": str(system)}]
