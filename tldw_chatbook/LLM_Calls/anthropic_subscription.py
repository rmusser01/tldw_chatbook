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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

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
#: None result so a locked/denied Keychain stalls at most once per window.
_KEYCHAIN_TTL_S = 5.0
_KEYCHAIN_CACHE: Optional[tuple[float, Optional[str]]] = None

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
    "or set [api_settings.anthropic] auth_source back to \"api_key\"."
)
MISSING_CREDENTIAL_MESSAGE = (
    "auth_source is \"claude_subscription\" but no Claude Code credential was "
    "found (checked ~/.claude/.credentials.json and, on macOS, the login "
    "Keychain item \"Claude Code-credentials\"). Log in with Claude Code first "
    "(and unlock your Keychain if prompted), or set [api_settings.anthropic] "
    "auth_source back to \"api_key\"."
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


def read_claude_code_credential(
    path: Path | str | None = None,
) -> Optional[SubscriptionCredential]:
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
) -> Optional[SubscriptionCredential]:
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
    token = str(oauth.get("accessToken") or "").strip()
    if not token:
        return None
    try:
        expires_at_ms = int(oauth.get("expiresAt") or 0)
    except (TypeError, ValueError):
        expires_at_ms = 0
    return SubscriptionCredential(
        access_token=token,
        expires_at_ms=expires_at_ms,
        subscription_type=str(oauth.get("subscriptionType") or ""),
        source_path=source_label,
    )


def _keychain_credential_raw() -> Optional[str]:
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
    now = time.time()
    cached = _KEYCHAIN_CACHE
    if cached is not None and now - cached[0] < _KEYCHAIN_TTL_S:
        return cached[1]
    result: Optional[str] = None
    try:
        proc = subprocess.run(  # noqa: S603 - fixed constant command, no user input
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
    _KEYCHAIN_CACHE = (now, result)
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
        if (
            isinstance(first, dict)
            and str(first.get("text", "")).startswith(CLAUDE_CODE_IDENTITY)
        ):
            return blocks
        return [identity, *blocks]
    # M2: an unexpected shape is preserved as text, never silently dropped.
    return [identity, {"type": "text", "text": str(system)}]
