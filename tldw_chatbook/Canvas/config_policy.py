"""Canvas execution and delivery policy resolved from configuration.

Moved out of ``tldw_chatbook.config`` (TASK-36945.2). Stateful config reads go
through the ``config`` module at call time so patches on it still apply.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from tldw_chatbook import config as _config
from tldw_chatbook.Canvas.limits import CanvasLimits

if TYPE_CHECKING:
    from tldw_chatbook.Canvas.web_auth import WebAuthPolicy


CanvasRemoteAccessStatus = Literal[
    "loopback",
    "authenticated_tls",
    "refused",
    "misconfigured",
    "insecure_development",
]


@dataclass(frozen=True, slots=True)
class CanvasConfigPolicy:
    """Effective Canvas execution and delivery policy without credentials."""

    enabled: bool
    auto_open_on_create: bool
    limits: CanvasLimits
    remote_access_status: CanvasRemoteAccessStatus
    remote_access_summary: str
    diagnostics: tuple[str, ...] = ()


_CANVAS_CONFIG_KEYS = frozenset({"enabled", "auto_open_on_create"})
_CANVAS_ENABLED_ENV = "TLDW_CANVAS_ENABLED"
_CANVAS_AUTO_OPEN_ENV = "TLDW_CANVAS_AUTO_OPEN_ON_CREATE"


def _strict_canvas_bool(
    section: Mapping[str, Any],
    key: str,
    *,
    default: bool,
    invalid_diagnostic: str,
    diagnostics: list[str],
) -> bool:
    """Read one exact Canvas boolean, failing closed on malformed values."""

    if key not in section:
        return default
    value = section.get(key)
    if type(value) is bool:
        return value
    diagnostics.append(invalid_diagnostic)
    return False


def _resolve_canvas_bool(
    section: Mapping[str, Any],
    key: str,
    *,
    environment_name: str,
    environ: Mapping[str, str],
    default: bool,
    invalid_config_diagnostic: str,
    invalid_environment_diagnostic: str,
    diagnostics: list[str],
) -> bool:
    """Resolve one strict environment-over-config Canvas preference."""

    raw_environment = environ.get(environment_name)
    if raw_environment is not None:
        if type(raw_environment) is not str:
            diagnostics.append(invalid_environment_diagnostic)
            return False
        normalized = raw_environment.strip().lower()
        if normalized:
            if normalized == "true":
                return True
            if normalized == "false":
                return False
            diagnostics.append(invalid_environment_diagnostic)
            return False
    return _strict_canvas_bool(
        section,
        key,
        default=default,
        invalid_diagnostic=invalid_config_diagnostic,
        diagnostics=diagnostics,
    )


def _normalize_canvas_execution(
    config: Mapping[str, Any] | None,
    *,
    environ: Mapping[str, str],
    diagnostics: list[str],
) -> tuple[Mapping[str, Any], bool, bool]:
    """Return the Canvas section, strict execution gate, and table validity."""

    values = config if isinstance(config, Mapping) else {}
    raw_canvas = values.get("canvas")
    if raw_canvas is None:
        canvas: Mapping[str, Any] = {}
        valid_table = True
    elif isinstance(raw_canvas, Mapping):
        canvas = raw_canvas
        valid_table = True
    else:
        canvas = {}
        valid_table = False
        diagnostics.append("canvas must be a table; Canvas is disabled")

    if not valid_table:
        return canvas, False, False
    enabled = _resolve_canvas_bool(
        canvas,
        "enabled",
        environment_name=_CANVAS_ENABLED_ENV,
        environ=environ,
        default=True,
        invalid_config_diagnostic=(
            "canvas.enabled must be a boolean; Canvas is disabled"
        ),
        invalid_environment_diagnostic=(
            "TLDW_CANVAS_ENABLED must be true or false; Canvas is disabled"
        ),
        diagnostics=diagnostics,
    )
    return canvas, enabled, True


def _summarize_canvas_web_auth_policy(
    policy: "WebAuthPolicy",
) -> tuple[CanvasRemoteAccessStatus, str]:
    """Return credential-free copy for one already-validated web policy."""

    if policy.automatic_local_login:
        return (
            "loopback",
            "Loopback only — browsers on this Chatbook host enter locally.",
        )
    if policy.insecure_remote_http:
        return (
            "insecure_development",
            "Insecure development mode — authenticated remote HTTP can be observed on the network.",
        )
    return (
        "authenticated_tls",
        "Authenticated TLS/proxy — dedicated browser admission is configured.",
    )


def _canvas_remote_access_status(
    web: Mapping[str, Any],
    *,
    environ: Mapping[str, str],
    keyring_get: Callable[[str, str], str | None] | None,
    web_auth_policy: "WebAuthPolicy | None",
) -> tuple[CanvasRemoteAccessStatus, str]:
    """Validate configured or effective served auth and return redacted copy."""

    from tldw_chatbook.Canvas.web_auth import (
        BindPolicyError,
        ResolvedCredential,
        build_web_auth_policy,
        is_loopback_host,
        resolve_web_access_token,
    )

    if web_auth_policy is not None:
        return _summarize_canvas_web_auth_policy(web_auth_policy)

    host = str(web.get("host") or "localhost").strip()
    public_url = str(web.get("public_url") or "").strip() or None
    port = web.get("port", 8000)
    if type(port) is not int or not 0 <= port <= 65535:
        return (
            "misconfigured",
            "Remote access misconfigured — review the configured served bind, origin, TLS, and proxy settings.",
        )
    certificate = web.get("tls_certificate")
    private_key = web.get("tls_private_key")
    if bool(certificate) != bool(private_key):
        return (
            "misconfigured",
            "Remote access misconfigured — review the configured served bind, origin, TLS, and proxy settings.",
        )

    remote_candidate = not is_loopback_host(host) or public_url is not None
    credential = (
        resolve_web_access_token(
            web.get("access_token"),
            environ=environ,
            keyring_get=keyring_get,
        )
        if remote_candidate
        else ResolvedCredential(None, "not_required")
    )
    proxies = web.get("trusted_proxy_addresses", ())
    if not isinstance(proxies, (list, tuple)):
        proxies = ("invalid-proxy-configuration",)
    try:
        policy = build_web_auth_policy(
            host=host,
            port=port,
            access_token=credential,
            public_url=public_url,
            allow_insecure_remote_http=(web.get("allow_insecure_remote_http") is True),
            trusted_proxy_addresses=tuple(str(value) for value in proxies),
            direct_tls=bool(certificate and private_key),
        )
    except BindPolicyError as exc:
        if credential.reveal() is None and "token" in str(exc).lower():
            return (
                "refused",
                "Remote access refused — configure the dedicated Chatbook web access token.",
            )
        return (
            "misconfigured",
            "Remote access misconfigured — review the configured served bind, origin, TLS, and proxy settings.",
        )
    return _summarize_canvas_web_auth_policy(policy)


def build_canvas_config_policy(
    config: Mapping[str, Any] | None,
    *,
    environ: Mapping[str, str] | None = None,
    keyring_get: Callable[[str, str], str | None] | None = None,
    web_auth_policy: "WebAuthPolicy | None" = None,
) -> CanvasConfigPolicy:
    """Build the sole normalized Canvas policy from untrusted configuration."""

    values = config if isinstance(config, Mapping) else {}
    diagnostics: list[str] = []
    effective_environment = os.environ if environ is None else environ
    canvas, enabled, valid_canvas_table = _normalize_canvas_execution(
        values,
        environ=effective_environment,
        diagnostics=diagnostics,
    )
    auto_open = (
        _resolve_canvas_bool(
            canvas,
            "auto_open_on_create",
            environment_name=_CANVAS_AUTO_OPEN_ENV,
            environ=effective_environment,
            default=True,
            invalid_config_diagnostic=(
                "canvas.auto_open_on_create must be a boolean; auto-open is disabled"
            ),
            invalid_environment_diagnostic=(
                "TLDW_CANVAS_AUTO_OPEN_ON_CREATE must be true or false; auto-open is disabled"
            ),
            diagnostics=diagnostics,
        )
        if valid_canvas_table
        else False
    )
    if any(key not in _CANVAS_CONFIG_KEYS for key in canvas):
        diagnostics.append(
            "Canvas quota overrides are unsupported; hard limits remain fixed"
        )

    raw_web = values.get("web_server")
    web = raw_web if isinstance(raw_web, Mapping) else {}
    remote_status, remote_summary = _canvas_remote_access_status(
        web,
        environ=effective_environment,
        keyring_get=keyring_get,
        web_auth_policy=web_auth_policy,
    )
    return CanvasConfigPolicy(
        enabled=enabled,
        auto_open_on_create=auto_open,
        limits=CanvasLimits(),
        remote_access_status=remote_status,
        remote_access_summary=remote_summary,
        diagnostics=tuple(diagnostics),
    )


def get_canvas_config_policy(
    *, web_auth_policy: "WebAuthPolicy | None" = None
) -> CanvasConfigPolicy:
    """Return the current effective Canvas policy from the shared config cache."""

    return build_canvas_config_policy(
        _config.load_cli_config_and_ensure_existence(),
        web_auth_policy=web_auth_policy,
    )


def get_canvas_execution_enabled() -> bool:
    """Read only the global execution gate without resolving web credentials."""

    config = _config.load_cli_config_and_ensure_existence()
    _canvas, enabled, _valid_table = _normalize_canvas_execution(
        config,
        environ=os.environ,
        diagnostics=[],
    )
    return enabled
