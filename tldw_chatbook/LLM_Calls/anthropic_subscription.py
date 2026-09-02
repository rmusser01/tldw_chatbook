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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

#: Claude Code's credential file. Chatbook only ever reads it.
DEFAULT_CREDENTIALS_PATH = Path.home() / ".claude" / ".credentials.json"

#: Beta flag the subscription authorization path requires.
OAUTH_BETA = "oauth-2025-04-20"

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
    "found at ~/.claude/.credentials.json. Log in with Claude Code first, or "
    "set [api_settings.anthropic] auth_source back to \"api_key\"."
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

    Returns ``None`` when the file is missing or malformed (AC#6: absent
    credential must leave behavior exactly as today). An EXPIRED credential is
    returned with ``expired=True`` so the caller can show the AC#2 message
    instead of a generic missing-credential one.
    """
    target = Path(path) if path is not None else DEFAULT_CREDENTIALS_PATH
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
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
        source_path=str(target),
    )


def anthropic_auth_source(anthropic_config: Mapping[str, Any] | None) -> str:
    """The configured auth source; anything unrecognized is the safe default."""
    raw = str((anthropic_config or {}).get("auth_source") or "").strip().lower()
    if raw == AUTH_SOURCE_SUBSCRIPTION:
        return AUTH_SOURCE_SUBSCRIPTION
    return AUTH_SOURCE_API_KEY


def subscription_headers(credential: SubscriptionCredential) -> dict[str, str]:
    """The auth headers for the subscription path (AC#7's shape).

    Replaces ``x-api-key`` entirely — the caller must not send both.
    """
    return {
        "authorization": f"Bearer {credential.access_token}",
        "anthropic-beta": OAUTH_BETA,
    }


def subscription_headers_for_token(access_token: str) -> dict[str, str]:
    """`subscription_headers` for an already-extracted token."""
    return {
        "authorization": f"Bearer {access_token}",
        "anthropic-beta": OAUTH_BETA,
    }
