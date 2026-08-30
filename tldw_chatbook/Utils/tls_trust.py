"""App-wide TLS trust policy for outbound HTTP/HTTPS/WebSocket clients.

One config knob, ``[network] ssl_verify``:

- ``true`` (default) -> verify against the default bundle (certifi).
- ``false``          -> verification DISABLED (insecure escape hatch for
                        TLS-inspecting corporate networks).
- ``"/path/ca.pem"`` -> ALSO trust this CA bundle (corporate root CA) —
                        ADDITIVE: certifi + custom, never replace.

Fail-safe direction is ALWAYS verification-on: any invalid value (bad type,
missing/unreadable file, unparseable PEM, bundle-write failure) logs an error
with the remedy and yields default verification.

Governance: backlog/decisions/079-network-tls-trust-policy.md and
Docs/superpowers/specs/2026-08-29-network-tls-trust-policy-design.md.
"""
from __future__ import annotations

import os
import ssl
import tempfile
from pathlib import Path
from typing import Any

import httpx
import requests as _requests
from loguru import logger

from ..Metrics.metrics_logger import log_counter
from ..config import get_cli_setting
from .paths import get_user_data_dir

_TRUE_STRINGS = frozenset({"true", "1", "on"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "off"})

_warned_modes: set[str] = set()


def tls_verify_setting() -> bool | str:
    """Normalized ``[network] ssl_verify``.

    Returns:
        ``True`` (verify on), ``False`` (verification off), or the string
        path of an EXISTING CA-bundle file. Never raises.
    """
    value = get_cli_setting("network", "ssl_verify", True)
    if isinstance(value, bool):
        result: bool | str = value
    elif isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_STRINGS or not lowered:
            result = True
        elif lowered in _FALSE_STRINGS:
            result = False
        else:
            path = Path(value.strip()).expanduser()
            if path.is_file():
                result = str(path)
            else:
                logger.error(
                    f"[network] ssl_verify path {str(path)!r} is not an existing"
                    " file; falling back to default certificate verification."
                    " Remedy: point ssl_verify at an existing CA bundle (PEM)"
                    " file."
                )
                result = True
    else:
        logger.error(
            "[network] ssl_verify has unsupported type"
            f" {type(value).__name__}; falling back to default certificate"
            " verification."
        )
        result = True
    _maybe_warn(result)
    return result


def warn_tls_policy() -> None:
    """Warn (once per process per mode) + metric when verification is relaxed."""
    _maybe_warn(tls_verify_setting())


def _maybe_warn(setting: bool | str) -> None:
    if setting is True:
        return
    if setting is False:
        mode, message = "off", (
            "TLS certificate verification is DISABLED"
            " ([network] ssl_verify = false). API keys and conversation"
            " content can be intercepted by anyone on the network path."
            " Restore ssl_verify = true unless this is required by a"
            " TLS-inspecting corporate network."
        )
    else:
        mode, message = "custom_ca", (
            "TLS verification additionally trusts custom CA bundle"
            f" {setting!r} ([network] ssl_verify). Ensure this is your"
            " organisation's root CA."
        )
    if mode in _warned_modes:
        return
    _warned_modes.add(mode)
    log_counter(f"network_tls_verify_{mode}")
    logger.warning(message)


_MERGED_BUNDLE_NAME = "merged-ca-bundle.pem"


def _additive_context(custom_ca: str) -> ssl.SSLContext:
    """Context trusting certifi's bundle PLUS ``custom_ca`` (never replace)."""
    import certifi

    context = ssl.create_default_context(cafile=certifi.where())
    context.load_verify_locations(cafile=custom_ca)
    return context


def ssl_context_for_transport() -> None | ssl.SSLContext:
    """Trust value for aiohttp ``TCPConnector(ssl=...)`` / websockets ``connect(ssl=...)``.

    Returns:
        ``None`` for default verification, an UNVERIFIED ``ssl.SSLContext``
        (CERT_NONE) when verification is disabled, or an ADDITIVE
        ``ssl.SSLContext`` for a custom CA. Never raises; load failures
        fail safe to ``None``.
    """
    setting = tls_verify_setting()
    if setting is True:
        return None
    if setting is False:
        # websockets >= 14 rejects a bare ``False`` when it also sets
        # ``server_hostname`` for wss:// URIs (ValueError before the dial),
        # and aiohttp treats an unverified context exactly like
        # ``ssl=False`` — so the portable "verification disabled" spelling
        # is a CERT_NONE context.
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
    try:
        return _additive_context(setting)
    except (OSError, ssl.SSLError) as exc:
        logger.error(
            f"[network] ssl_verify bundle {setting!r} could not be loaded"
            f" ({exc}); falling back to default certificate verification."
        )
        return None


def _merged_bundle_path() -> str:
    """Path to a cached PEM containing certifi + the custom CA.

    Regenerated (atomic tmp + ``os.replace``) whenever either source's
    ``(mtime_ns, size)`` changes — a comment header records the fingerprint,
    and OpenSSL's PEM reader ignores non-PEM lines.
    """
    import certifi

    setting = tls_verify_setting()
    assert isinstance(setting, str)
    cache_dir = Path(get_user_data_dir()) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    merged = cache_dir / _MERGED_BUNDLE_NAME
    sources = (Path(certifi.where()), Path(setting))
    fingerprint = ";".join(
        f"{p}|{p.stat().st_mtime_ns}|{p.stat().st_size}" for p in sources
    )
    header = f"# tls-trust-sources: {fingerprint}\n"
    if merged.is_file() and merged.read_text(errors="replace").startswith(header):
        return str(merged)
    body = header + "".join(p.read_text() + "\n" for p in sources)
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
        os.replace(tmp, merged)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return str(merged)


def requests_verify() -> bool | str:
    """``verify=`` value for requests sessions/requests (bool or merged-bundle path)."""
    setting = tls_verify_setting()
    if setting is True or setting is False:
        return setting
    try:
        return _merged_bundle_path()
    except (OSError, UnicodeDecodeError) as exc:
        logger.error(
            f"[network] merged CA bundle could not be written ({exc});"
            " falling back to default certificate verification."
        )
        return True


def httpx_verify() -> bool | ssl.SSLContext:
    """``verify=`` value for httpx clients.

    ``bool`` or the additive ``SSLContext`` — NEVER a bare custom-CA path,
    which httpx would load as the only trusted bundle (replace semantics).
    """
    setting = tls_verify_setting()
    if setting is True or setting is False:
        return setting
    context = ssl_context_for_transport()
    return context if isinstance(context, ssl.SSLContext) else True


def build_httpx_async_client(**kwargs: Any) -> httpx.AsyncClient:
    """``httpx.AsyncClient`` with the app TLS trust policy applied by default.

    Callers may override with an explicit ``verify=`` (it wins).
    """
    kwargs.setdefault("verify", httpx_verify())
    return httpx.AsyncClient(**kwargs)


def build_httpx_client(**kwargs: Any) -> httpx.Client:
    """``httpx.Client`` with the app TLS trust policy applied by default."""
    kwargs.setdefault("verify", httpx_verify())
    return httpx.Client(**kwargs)


def build_requests_session(*, verify: bool | str | None = None) -> _requests.Session:
    """``requests.Session`` with the app TLS trust policy applied by default.

    An explicit ``verify`` (bool or CA-bundle path) wins over the policy,
    mirroring the httpx factories' setdefault semantics. ``requests.Session``
    accepts no constructor kwargs, so there is intentionally no kwargs
    forwarding here.
    """
    session = _requests.Session()
    session.verify = requests_verify() if verify is None else verify
    return session
