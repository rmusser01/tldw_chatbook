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

from pathlib import Path

from loguru import logger

from ..Metrics.metrics_logger import log_counter
from ..config import get_cli_setting

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
            f"[network] ssl_verify has unsupported type"
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
            f"TLS verification additionally trusts custom CA bundle"
            f" {setting!r} ([network] ssl_verify). Ensure this is your"
            " organisation's root CA."
        )
    log_counter(f"network_tls_verify_{mode}")
    if mode in _warned_modes:
        return
    _warned_modes.add(mode)
    logger.warning(message)
