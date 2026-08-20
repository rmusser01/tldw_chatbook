"""Pure control-state and identity contracts for Console project instructions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit


PROJECT_CONTEXT_VERSION = 1
EPHEMERAL_ORIGIN_KEY = "_chatbook_ephemeral_origin"

LOCATOR_FINGERPRINT_DOMAIN = b"tldw_chatbook.console.project-instructions.locator.v1\0"
PROVIDER_DESTINATION_FINGERPRINT_DOMAIN = (
    b"tldw_chatbook.console.project-instructions.provider-destination.v1\0"
)
NOTICE_KEY_FINGERPRINT_DOMAIN = (
    b"tldw_chatbook.console.project-instructions.notice-key.v1\0"
)

_CONTROL_KEYS = (
    "project_instructions_enabled",
    "working_folder_binding_id",
    "working_folder_locator_fingerprint",
    "project_instruction_notice_key",
)
_PROJECT_CONTEXT_KEYS = frozenset(("version", *_CONTROL_KEYS))
_INVALID_ENDPOINT_IDENTITY = "invalid-endpoint"


@dataclass(frozen=True, slots=True)
class ProjectInstructionControlState:
    """Local-only Console session controls for automatic project guidance."""

    project_instructions_enabled: bool
    working_folder_binding_id: str | None = None
    working_folder_locator_fingerprint: str | None = None
    project_instruction_notice_key: str | None = None

    @classmethod
    def new_session(cls) -> ProjectInstructionControlState:
        """Return the explicit opt-in default for a newly created session."""
        return cls(project_instructions_enabled=True)

    @classmethod
    def legacy_disabled(cls) -> ProjectInstructionControlState:
        """Return the fail-closed default for absent or untrusted state."""
        return cls(project_instructions_enabled=False)


def encode_project_context_json(state: ProjectInstructionControlState) -> str:
    """Encode only the versioned four-field control contract."""
    payload = {
        "version": PROJECT_CONTEXT_VERSION,
        "project_instructions_enabled": state.project_instructions_enabled,
        "working_folder_binding_id": state.working_folder_binding_id,
        "working_folder_locator_fingerprint": (
            state.working_folder_locator_fingerprint
        ),
        "project_instruction_notice_key": state.project_instruction_notice_key,
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def decode_project_context_json(
    raw_state: str | None,
) -> ProjectInstructionControlState:
    """Decode a proven v1 control envelope, otherwise fail closed."""
    if raw_state is None:
        return ProjectInstructionControlState.legacy_disabled()
    try:
        payload = json.loads(raw_state)
    except (json.JSONDecodeError, TypeError):
        return ProjectInstructionControlState.legacy_disabled()
    if not isinstance(payload, dict):
        return ProjectInstructionControlState.legacy_disabled()
    if set(payload) != _PROJECT_CONTEXT_KEYS:
        return ProjectInstructionControlState.legacy_disabled()
    version = payload.get("version")
    if type(version) is not int or version != PROJECT_CONTEXT_VERSION:
        return ProjectInstructionControlState.legacy_disabled()
    if type(payload["project_instructions_enabled"]) is not bool:
        return ProjectInstructionControlState.legacy_disabled()
    optional_values = tuple(payload[key] for key in _CONTROL_KEYS[1:])
    if any(
        value is not None and not isinstance(value, str) for value in optional_values
    ):
        return ProjectInstructionControlState.legacy_disabled()
    return ProjectInstructionControlState(
        project_instructions_enabled=payload["project_instructions_enabled"],
        working_folder_binding_id=payload["working_folder_binding_id"],
        working_folder_locator_fingerprint=payload[
            "working_folder_locator_fingerprint"
        ],
        project_instruction_notice_key=payload["project_instruction_notice_key"],
    )


def fingerprint_canonical_locator(canonical_locator_identity: str) -> str:
    """Return an opaque domain-separated fingerprint of a canonical locator."""
    return hashlib.sha256(
        LOCATOR_FINGERPRINT_DOMAIN + canonical_locator_identity.encode("utf-8")
    ).hexdigest()


def fingerprint_provider_destination(provider: str, endpoint: str | None) -> str:
    """Fingerprint a provider adapter and credential-free endpoint identity."""
    identity = (
        _canonical_provider_key(provider)
        + "\0"
        + _canonical_endpoint_identity(endpoint)
    )
    return hashlib.sha256(
        PROVIDER_DESTINATION_FINGERPRINT_DOMAIN + identity.encode("utf-8")
    ).hexdigest()


def project_instruction_notice_key(
    locator_fingerprint: str,
    provider: str,
    endpoint: str | None,
) -> str:
    """Return the consent key for one locator and resolved destination."""
    destination_fingerprint = fingerprint_provider_destination(provider, endpoint)
    framed_identity = (
        locator_fingerprint.encode("ascii")
        + b"\0"
        + destination_fingerprint.encode("ascii")
    )
    return hashlib.sha256(NOTICE_KEY_FINGERPRINT_DOMAIN + framed_identity).hexdigest()


def sanitized_destination_label(
    provider_label: str, custom_endpoint: str | None
) -> str:
    """Show a provider and, when custom, only its credential-free URL origin."""
    label = provider_label.strip() or "Provider"
    if not custom_endpoint:
        return label
    origin = _endpoint_origin(custom_endpoint)
    return f"{label} ({origin or 'invalid endpoint'})"


def _canonical_provider_key(provider: str) -> str:
    return provider.strip().lower()


def _canonical_endpoint_identity(endpoint: str | None) -> str:
    parsed = _parse_endpoint(endpoint)
    if parsed is None:
        return "" if not str(endpoint or "").strip() else _INVALID_ENDPOINT_IDENTITY
    scheme, hostname, port, path = parsed
    return urlunsplit((scheme, _netloc(scheme, hostname, port), path, "", ""))


def _endpoint_origin(endpoint: str) -> str:
    parsed = _parse_endpoint(endpoint)
    if parsed is None:
        return ""
    scheme, hostname, port, _path = parsed
    return urlunsplit((scheme, _netloc(scheme, hostname, port), "", "", ""))


def _parse_endpoint(endpoint: str | None) -> tuple[str, str, int | None, str] | None:
    raw_endpoint = str(endpoint or "").strip()
    if not raw_endpoint or any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in raw_endpoint
    ):
        return None
    candidate = raw_endpoint if "://" in raw_endpoint else f"http://{raw_endpoint}"
    try:
        parsed = urlsplit(candidate)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if scheme not in {"http", "https"} or not hostname:
        return None
    return scheme, hostname.lower(), port, parsed.path.rstrip("/")


def _netloc(scheme: str, hostname: str, port: int | None) -> str:
    host = f"[{hostname}]" if ":" in hostname else hostname
    default_port = (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    )
    return host if port is None or default_port else f"{host}:{port}"
