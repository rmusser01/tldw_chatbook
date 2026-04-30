from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote


SERVER_CREDENTIAL_ACCESS_TOKEN = "access_token"
SERVER_CREDENTIAL_REFRESH_TOKEN = "refresh_token"
SERVER_CREDENTIAL_API_KEY = "api_key"
SERVER_CREDENTIAL_BEARER_TOKEN = "bearer_token"
DEFAULT_KEYRING_SERVICE_NAME = "tldw_chatbook.server_credentials"
_KEYRING_INDEX_USERNAME = "__credential_refs__"

_KNOWN_SERVER_CREDENTIAL_PURPOSES = (
    SERVER_CREDENTIAL_ACCESS_TOKEN,
    SERVER_CREDENTIAL_REFRESH_TOKEN,
    SERVER_CREDENTIAL_API_KEY,
    SERVER_CREDENTIAL_BEARER_TOKEN,
)
_SECURE_KEYRING_MODULE_PARTS = ("macos", "windows", "secretservice")
_INSECURE_KEYRING_MODULE_PARTS = ("fail", "null", "plaintext", "file")


class ServerCredentialStore(Protocol):
    def set_secret(self, server_id: str, purpose: str, secret: str) -> None: ...

    def get_secret(self, server_id: str, purpose: str) -> str | None: ...

    def delete_secret(self, server_id: str, purpose: str) -> None: ...

    def clear_server(self, server_id: str) -> None: ...

    def clear_all(self) -> None: ...


class CredentialStoreUnavailable(RuntimeError):
    reason_code = "credential_store_unavailable"


@dataclass(frozen=True)
class ServerCredentialRef:
    server_id: str
    purpose: str

    @property
    def username(self) -> str:
        return f"{self.server_id}:{self.purpose}"


@dataclass(frozen=True)
class ServerCredentialScope:
    server_profile_id: str
    normalized_origin: str
    credential_type: str
    principal_id: str | None = None

    @classmethod
    def legacy(cls, server_id: str, purpose: str) -> "ServerCredentialScope":
        normalized = _normalize_non_empty(server_id, "server_id")
        return cls(
            server_profile_id=normalized,
            normalized_origin=normalized,
            credential_type=_normalize_purpose(purpose),
            principal_id=None,
        )


def redact_secret(secret: str | None) -> str:
    if not secret:
        return "<unset>"
    if len(secret) <= 8:
        return "<redacted>"
    candidate = f"{secret[:2]}...<redacted>...{secret[-4:]}"
    if candidate == secret:
        candidate = f"{secret[:2]}...<redacted:x>...{secret[-4:]}"
    return candidate


def _normalize_non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _normalize_purpose(purpose: str) -> str:
    normalized = _normalize_non_empty(purpose, "purpose")
    if ":" in normalized:
        raise ValueError("purpose must not contain ':'")
    return normalized


def _credential_ref(server_id: str, purpose: str) -> ServerCredentialRef:
    return ServerCredentialRef(
        server_id=_normalize_non_empty(server_id, "server_id"),
        purpose=_normalize_purpose(purpose),
    )


def _username_for_scope(scope: ServerCredentialScope) -> str:
    principal_kind = "none" if scope.principal_id is None else "value"
    principal_value = "" if scope.principal_id is None else scope.principal_id
    return (
        f"{DEFAULT_KEYRING_SERVICE_NAME}:v1|"
        f"profile={quote(scope.server_profile_id, safe='')}|"
        f"origin={quote(scope.normalized_origin, safe='')}|"
        f"principal_kind={principal_kind}|"
        f"principal={quote(principal_value, safe='')}|"
        f"type={quote(scope.credential_type, safe='')}"
    )


def _legacy_username_for_scope(scope: ServerCredentialScope) -> str:
    return f"{scope.server_profile_id}:{scope.credential_type}"


def _is_legacy_scope(scope: ServerCredentialScope) -> bool:
    return (
        scope.normalized_origin == scope.server_profile_id
        and scope.principal_id is None
    )


def _normalize_scope(scope: ServerCredentialScope) -> ServerCredentialScope:
    principal_id = scope.principal_id
    if principal_id is not None:
        principal_id = _normalize_non_empty(principal_id, "principal_id")
    return ServerCredentialScope(
        server_profile_id=_normalize_non_empty(scope.server_profile_id, "server_profile_id"),
        normalized_origin=_normalize_non_empty(scope.normalized_origin, "normalized_origin"),
        credential_type=_normalize_purpose(scope.credential_type),
        principal_id=principal_id,
    )


def _scope_from_index_entry(entry: Any) -> ServerCredentialScope | None:
    if isinstance(entry, (list, tuple)) and len(entry) == 2:
        server_id, purpose = entry
        if not isinstance(server_id, str) or not isinstance(purpose, str):
            return None
        try:
            return ServerCredentialScope.legacy(server_id, purpose)
        except ValueError:
            return None

    if not isinstance(entry, dict):
        return None
    if entry.get("version") != 1:
        return None

    server_profile_id = entry.get("server_profile_id")
    normalized_origin = entry.get("normalized_origin")
    credential_type = entry.get("credential_type")
    principal_id = entry.get("principal_id")
    if not all(isinstance(value, str) for value in [server_profile_id, normalized_origin, credential_type]):
        return None
    if principal_id is not None and not isinstance(principal_id, str):
        return None

    try:
        return ServerCredentialScope(
            server_profile_id=_normalize_non_empty(server_profile_id, "server_profile_id"),
            normalized_origin=_normalize_non_empty(normalized_origin, "normalized_origin"),
            credential_type=_normalize_purpose(credential_type),
            principal_id=principal_id,
        )
    except ValueError:
        return None


def _index_entry_for_scope(scope: ServerCredentialScope) -> dict[str, Any]:
    return {
        "version": 1,
        "server_profile_id": scope.server_profile_id,
        "normalized_origin": scope.normalized_origin,
        "principal_id": scope.principal_id,
        "credential_type": scope.credential_type,
        "username": _username_for_scope(scope),
    }


class InMemoryServerCredentialStore:
    def __init__(self) -> None:
        self._secrets: dict[tuple[str, str], str] = {}

    def set_secret(self, server_id: str, purpose: str, secret: str) -> None:
        ref = _credential_ref(server_id, purpose)
        self._secrets[(ref.server_id, ref.purpose)] = secret

    def get_secret(self, server_id: str, purpose: str) -> str | None:
        ref = _credential_ref(server_id, purpose)
        return self._secrets.get((ref.server_id, ref.purpose))

    def delete_secret(self, server_id: str, purpose: str) -> None:
        ref = _credential_ref(server_id, purpose)
        self._secrets.pop((ref.server_id, ref.purpose), None)

    def clear_server(self, server_id: str) -> None:
        normalized_server_id = _normalize_non_empty(server_id, "server_id")
        for key in list(self._secrets):
            if key[0] == normalized_server_id:
                self._secrets.pop(key, None)

    def clear_all(self) -> None:
        self._secrets.clear()


class UnavailableServerCredentialStore:
    def __init__(self, message: str) -> None:
        self.message = message

    def _raise_unavailable(self) -> None:
        raise CredentialStoreUnavailable(self.message)

    def set_secret(self, server_id: str, purpose: str, secret: str) -> None:
        self._raise_unavailable()

    def get_secret(self, server_id: str, purpose: str) -> str | None:
        self._raise_unavailable()

    def delete_secret(self, server_id: str, purpose: str) -> None:
        self._raise_unavailable()

    def clear_server(self, server_id: str) -> None:
        self._raise_unavailable()

    def clear_all(self) -> None:
        self._raise_unavailable()


def _keyring_backend_children(keyring_backend: Any) -> list[Any]:
    children: list[Any] = []
    for attr_name in ("backends", "_backends", "keyrings", "backend"):
        child = getattr(keyring_backend, attr_name, None)
        if child is None:
            continue
        if isinstance(child, (list, tuple, set)):
            children.extend(child)
        else:
            children.append(child)
    return children


def is_secure_keyring_backend(keyring_backend: Any) -> bool:
    return _is_secure_keyring_backend(keyring_backend, seen=set())


def _is_secure_keyring_backend(keyring_backend: Any, *, seen: set[int]) -> bool:
    backend_id = id(keyring_backend)
    if backend_id in seen:
        return False
    seen.add(backend_id)

    children = _keyring_backend_children(keyring_backend)
    if children:
        return any(_is_secure_keyring_backend(child, seen=seen) for child in children)

    module_name = str(getattr(keyring_backend.__class__, "__module__", "")).lower()
    priority = getattr(keyring_backend, "priority", None)
    if isinstance(priority, (int, float)) and priority <= 0:
        return False
    if any(part in module_name for part in _INSECURE_KEYRING_MODULE_PARTS):
        return False
    return any(part in module_name for part in _SECURE_KEYRING_MODULE_PARTS)


def build_default_server_credential_store(keyring_backend: Any | None = None) -> ServerCredentialStore:
    if keyring_backend is None:
        import keyring

        keyring_backend = keyring.get_keyring()
    get_keyring = getattr(keyring_backend, "get_keyring", None)
    if callable(get_keyring):
        keyring_backend = get_keyring()
    if not is_secure_keyring_backend(keyring_backend):
        raise CredentialStoreUnavailable("No secure OS-backed credential store is available.")
    return KeyringServerCredentialStore(keyring_backend=keyring_backend)


class KeyringServerCredentialStore:
    def __init__(
        self,
        service_name: str = DEFAULT_KEYRING_SERVICE_NAME,
        keyring_backend: Any | None = None,
    ) -> None:
        self.service_name = _normalize_non_empty(service_name, "service_name")
        if keyring_backend is None:
            import keyring

            keyring_backend = keyring
        self._keyring = keyring_backend

    def _load_index(self) -> list[ServerCredentialScope]:
        payload = self._keyring.get_password(self.service_name, _KEYRING_INDEX_USERNAME)
        if not payload:
            return []

        try:
            entries = json.loads(payload)
        except (TypeError, ValueError):
            return []

        scopes: list[ServerCredentialScope] = []
        for entry in entries:
            scope = _scope_from_index_entry(entry)
            if scope is not None:
                scopes.append(scope)
        return scopes

    def _save_index(self, scopes: list[ServerCredentialScope]) -> None:
        if not scopes:
            self._delete_index_record()
            return

        unique_scopes = sorted(
            {
                (
                    scope.server_profile_id,
                    scope.normalized_origin,
                    scope.principal_id,
                    scope.credential_type,
                )
                for scope in scopes
            },
            key=lambda scope_parts: (
                scope_parts[0],
                scope_parts[1],
                scope_parts[2] or "",
                scope_parts[3],
            ),
        )
        payload = json.dumps(
            [
                _index_entry_for_scope(
                    ServerCredentialScope(
                        server_profile_id=server_profile_id,
                        normalized_origin=normalized_origin,
                        principal_id=principal_id,
                        credential_type=credential_type,
                    )
                )
                for server_profile_id, normalized_origin, principal_id, credential_type in unique_scopes
            ]
        )
        self._keyring.set_password(self.service_name, _KEYRING_INDEX_USERNAME, payload)

    def _add_scope_to_index(self, scope: ServerCredentialScope) -> None:
        scopes = self._load_index()
        scopes.append(scope)
        self._save_index(scopes)

    def _delete_index_record(self) -> None:
        if self._keyring.get_password(self.service_name, _KEYRING_INDEX_USERNAME) is None:
            return

        values = getattr(self._keyring, "values", None)
        if isinstance(values, dict):
            values.pop((self.service_name, _KEYRING_INDEX_USERNAME), None)
            return

        self._keyring.delete_password(self.service_name, _KEYRING_INDEX_USERNAME)

    def _remove_scope_from_index(self, scope: ServerCredentialScope) -> None:
        scopes = [
            indexed_scope
            for indexed_scope in self._load_index()
            if (
                indexed_scope.server_profile_id,
                indexed_scope.normalized_origin,
                indexed_scope.principal_id,
                indexed_scope.credential_type,
            )
            != (
                scope.server_profile_id,
                scope.normalized_origin,
                scope.principal_id,
                scope.credential_type,
            )
        ]
        self._save_index(scopes)

    def set_secret(self, server_id: str, purpose: str, secret: str) -> None:
        self.set_scoped_secret(ServerCredentialScope.legacy(server_id, purpose), secret)

    def get_secret(self, server_id: str, purpose: str) -> str | None:
        return self.get_scoped_secret(ServerCredentialScope.legacy(server_id, purpose))

    def delete_secret(self, server_id: str, purpose: str) -> None:
        self.delete_scoped_secret(ServerCredentialScope.legacy(server_id, purpose))

    def set_scoped_secret(self, scope: ServerCredentialScope, secret: str) -> None:
        scope = _normalize_scope(scope)
        self._keyring.set_password(self.service_name, _username_for_scope(scope), secret)
        self._add_scope_to_index(scope)

    def get_scoped_secret(self, scope: ServerCredentialScope) -> str | None:
        scope = _normalize_scope(scope)
        secret = self._keyring.get_password(self.service_name, _username_for_scope(scope))
        if secret is not None:
            return secret
        if not _is_legacy_scope(scope):
            return None
        return self._keyring.get_password(self.service_name, _legacy_username_for_scope(scope))

    def delete_scoped_secret(self, scope: ServerCredentialScope) -> None:
        scope = _normalize_scope(scope)
        username = _username_for_scope(scope)
        legacy_username = _legacy_username_for_scope(scope)

        if self._keyring.get_password(self.service_name, username) is None:
            if not _is_legacy_scope(scope):
                self._remove_scope_from_index(scope)
                return
            if self._keyring.get_password(self.service_name, legacy_username) is None:
                self._remove_scope_from_index(scope)
                return
            self._keyring.delete_password(self.service_name, legacy_username)
            self._remove_scope_from_index(scope)
            return

        self._keyring.delete_password(self.service_name, username)
        legacy_secret_exists = self._keyring.get_password(self.service_name, legacy_username) is not None
        if _is_legacy_scope(scope) and legacy_secret_exists:
            self._keyring.delete_password(self.service_name, legacy_username)
        self._remove_scope_from_index(scope)

    def clear_server(self, server_id: str) -> None:
        normalized_server_id = _normalize_non_empty(server_id, "server_id")
        for scope in list(self._load_index()):
            if scope.server_profile_id == normalized_server_id:
                self.delete_scoped_secret(scope)
        for purpose in _KNOWN_SERVER_CREDENTIAL_PURPOSES:
            self.delete_scoped_secret(ServerCredentialScope.legacy(normalized_server_id, purpose))

    def clear_all(self) -> None:
        for scope in list(self._load_index()):
            self.delete_scoped_secret(scope)
        self._save_index([])


__all__ = [
    "CredentialStoreUnavailable",
    "DEFAULT_KEYRING_SERVICE_NAME",
    "SERVER_CREDENTIAL_ACCESS_TOKEN",
    "SERVER_CREDENTIAL_API_KEY",
    "SERVER_CREDENTIAL_BEARER_TOKEN",
    "SERVER_CREDENTIAL_REFRESH_TOKEN",
    "InMemoryServerCredentialStore",
    "KeyringServerCredentialStore",
    "ServerCredentialRef",
    "ServerCredentialScope",
    "ServerCredentialStore",
    "UnavailableServerCredentialStore",
    "build_default_server_credential_store",
    "is_secure_keyring_backend",
    "redact_secret",
]
