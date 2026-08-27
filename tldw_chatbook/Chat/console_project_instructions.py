"""Pure control-state and identity contracts for Console project instructions."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass, replace
from ipaddress import ip_address
from urllib.parse import urlsplit, urlunsplit

from httpx import InvalidURL, URL


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


class _DuplicateJSONKeyError(ValueError):
    """Raised when an untrusted JSON object repeats a key."""


@dataclass(frozen=True, slots=True)
class ProjectInstructionControlState:
    """Local-only Console session controls for automatic project guidance."""

    project_instructions_enabled: bool
    working_folder_binding_id: str | None = None
    working_folder_locator_fingerprint: str | None = None
    project_instruction_notice_key: str | None = None

    @classmethod
    def new_session(cls) -> ProjectInstructionControlState:
        """Return the explicit opt-in default for a newly created session.

        Returns:
            Enabled control state with no selected binding or notice key.
        """
        return cls(project_instructions_enabled=True)

    @classmethod
    def legacy_disabled(cls) -> ProjectInstructionControlState:
        """Return the fail-closed default for absent or untrusted state.

        Returns:
            Disabled control state with no selected binding or notice key.
        """
        return cls(project_instructions_enabled=False)


def sanitize_fork_project_instruction_state(
    source: ProjectInstructionControlState,
) -> ProjectInstructionControlState:
    """Retain declarative project controls without copying source consent.

    Args:
        source: Validated controls captured from the source session.

    Returns:
        The same declarative selection with a fresh notice boundary.
    """

    return replace(source, project_instruction_notice_key=None)


def encode_project_context_json(state: ProjectInstructionControlState) -> str:
    """Encode only the versioned four-field control contract.

    Args:
        state: Validated local Console project-instruction controls.

    Returns:
        Compact v1 JSON containing only the version and four control fields.
    """
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
    """Decode a proven v1 control envelope, otherwise fail closed.

    Args:
        raw_state: Untrusted local JSON, or ``None`` for missing legacy state.

    Returns:
        Parsed v1 controls, or legacy-disabled controls for every invalid shape,
        including duplicate object keys.
    """
    if raw_state is None:
        return ProjectInstructionControlState.legacy_disabled()
    try:
        payload = json.loads(raw_state, object_pairs_hook=_strict_json_object)
    except (json.JSONDecodeError, TypeError, _DuplicateJSONKeyError):
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
    """Return an opaque domain-separated fingerprint of a canonical locator.

    Args:
        canonical_locator_identity: Already-canonical workspace locator identity.

    Returns:
        Lowercase SHA-256 hex digest without retaining the raw locator.
    """
    return hashlib.sha256(
        LOCATOR_FINGERPRINT_DOMAIN + canonical_locator_identity.encode("utf-8")
    ).hexdigest()


def fingerprint_provider_destination(provider: str, endpoint: str | None) -> str:
    """Fingerprint a provider adapter and credential-free endpoint identity.

    Args:
        provider: Resolved provider adapter key.
        endpoint: Resolved provider endpoint, or blank for the adapter default.

    Returns:
        Lowercase domain-separated SHA-256 hex digest.

    Raises:
        ValueError: If a nonempty endpoint cannot be canonicalized safely.
    """
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
    """Return the consent key for one locator and resolved destination.

    Args:
        locator_fingerprint: Opaque canonical-locator fingerprint.
        provider: Resolved provider adapter key.
        endpoint: Resolved provider endpoint, or blank for the adapter default.

    Returns:
        Lowercase domain-separated SHA-256 consent-key digest.

    Raises:
        ValueError: If a nonempty endpoint cannot be canonicalized safely.
    """
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
    """Show a provider and, when custom, only its credential-free URL origin.

    Args:
        provider_label: User-visible provider name.
        custom_endpoint: Custom endpoint to sanitize, or blank for the provider
            default.

    Returns:
        Provider label alone for a default endpoint, provider plus canonical
        origin for a valid custom endpoint, or a content-free invalid label.
    """
    label = provider_label.strip() or "Provider"
    if custom_endpoint is None or not custom_endpoint.strip():
        return label
    origin = _endpoint_origin(custom_endpoint)
    return f"{label} ({origin or 'invalid endpoint'})"


def _canonical_provider_key(provider: str) -> str:
    return provider.strip().lower()


def _canonical_endpoint_identity(endpoint: str | None) -> str:
    parsed = _parse_endpoint(endpoint)
    if parsed is None:
        if not str(endpoint or "").strip():
            return ""
        raise ValueError("invalid provider endpoint")
    scheme, hostname, port, path = parsed
    return urlunsplit((scheme, _netloc(scheme, hostname, port), path, "", ""))


def canonical_provider_endpoint_identity(endpoint: str | None) -> str:
    """Return a credential-free canonical provider endpoint identity."""
    return _canonical_endpoint_identity(endpoint)


def _endpoint_origin(endpoint: str) -> str:
    parsed = _parse_endpoint(endpoint)
    if parsed is None:
        return ""
    scheme, hostname, port, _path = parsed
    return urlunsplit((scheme, _netloc(scheme, hostname, port), "", "", ""))


def _parse_endpoint(endpoint: str | None) -> tuple[str, str, int | None, str] | None:
    raw_value = str(endpoint or "")
    raw_endpoint = raw_value.strip()
    if not raw_endpoint or any(
        character == "\\"
        or character.isspace()
        or unicodedata.category(character).startswith("C")
        for character in raw_value
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
    normalized_hostname = _canonical_hostname(hostname)
    if normalized_hostname is None:
        return None
    return scheme, normalized_hostname, port, parsed.path.rstrip("/")


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKeyError
        result[key] = value
    return result


def _canonical_hostname(hostname: str) -> str | None:
    try:
        return str(ip_address(hostname))
    except ValueError:
        pass
    try:
        normalized = (
            URL(f"http://{hostname.rstrip('.')}").raw_host.decode("ascii").lower()
        )
    except (InvalidURL, UnicodeDecodeError):
        return None
    labels = normalized.split(".")
    if (
        not normalized
        or len(normalized) > 253
        or any(not _valid_dns_label(label) for label in labels)
    ):
        return None
    return normalized


def _valid_dns_label(label: str) -> bool:
    return (
        1 <= len(label) <= 63
        and label[0].isalnum()
        and label[-1].isalnum()
        and all(character.isalnum() or character == "-" for character in label)
    )


def _netloc(scheme: str, hostname: str, port: int | None) -> str:
    host = f"[{hostname}]" if ":" in hostname else hostname
    default_port = (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    )
    return host if port is None or default_port else f"{host}:{port}"
