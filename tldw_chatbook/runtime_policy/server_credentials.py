from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol


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


class ServerCredentialStore(Protocol):
    def set_secret(self, server_id: str, purpose: str, secret: str) -> None: ...

    def get_secret(self, server_id: str, purpose: str) -> str | None: ...

    def delete_secret(self, server_id: str, purpose: str) -> None: ...

    def clear_server(self, server_id: str) -> None: ...

    def clear_all(self) -> None: ...


@dataclass(frozen=True)
class ServerCredentialRef:
    server_id: str
    purpose: str

    @property
    def username(self) -> str:
        return f"{self.server_id}:{self.purpose}"


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

    def _load_index(self) -> list[ServerCredentialRef]:
        payload = self._keyring.get_password(self.service_name, _KEYRING_INDEX_USERNAME)
        if not payload:
            return []

        try:
            entries = json.loads(payload)
        except (TypeError, ValueError):
            return []

        refs: list[ServerCredentialRef] = []
        for entry in entries:
            if not isinstance(entry, list | tuple) or len(entry) != 2:
                continue
            server_id, purpose = entry
            if not isinstance(server_id, str) or not isinstance(purpose, str):
                continue
            try:
                refs.append(_credential_ref(server_id, purpose))
            except ValueError:
                continue
        return refs

    def _save_index(self, refs: list[ServerCredentialRef]) -> None:
        if not refs:
            self._delete_index_record()
            return

        unique_refs = sorted({(ref.server_id, ref.purpose) for ref in refs})
        payload = json.dumps([[server_id, purpose] for server_id, purpose in unique_refs])
        self._keyring.set_password(self.service_name, _KEYRING_INDEX_USERNAME, payload)

    def _add_ref_to_index(self, ref: ServerCredentialRef) -> None:
        refs = self._load_index()
        refs.append(ref)
        self._save_index(refs)

    def _delete_index_record(self) -> None:
        if self._keyring.get_password(self.service_name, _KEYRING_INDEX_USERNAME) is None:
            return

        values = getattr(self._keyring, "values", None)
        if isinstance(values, dict):
            values.pop((self.service_name, _KEYRING_INDEX_USERNAME), None)
            return

        self._keyring.delete_password(self.service_name, _KEYRING_INDEX_USERNAME)

    def _remove_ref_from_index(self, ref: ServerCredentialRef) -> None:
        refs = [
            indexed_ref
            for indexed_ref in self._load_index()
            if (indexed_ref.server_id, indexed_ref.purpose) != (ref.server_id, ref.purpose)
        ]
        self._save_index(refs)

    def set_secret(self, server_id: str, purpose: str, secret: str) -> None:
        ref = _credential_ref(server_id, purpose)
        self._keyring.set_password(self.service_name, ref.username, secret)
        self._add_ref_to_index(ref)

    def get_secret(self, server_id: str, purpose: str) -> str | None:
        ref = _credential_ref(server_id, purpose)
        return self._keyring.get_password(self.service_name, ref.username)

    def delete_secret(self, server_id: str, purpose: str) -> None:
        ref = _credential_ref(server_id, purpose)
        if self._keyring.get_password(self.service_name, ref.username) is None:
            self._remove_ref_from_index(ref)
            return
        self._keyring.delete_password(self.service_name, ref.username)
        self._remove_ref_from_index(ref)

    def clear_server(self, server_id: str) -> None:
        normalized_server_id = _normalize_non_empty(server_id, "server_id")
        for purpose in _KNOWN_SERVER_CREDENTIAL_PURPOSES:
            self.delete_secret(normalized_server_id, purpose)

    def clear_all(self) -> None:
        for ref in list(self._load_index()):
            self.delete_secret(ref.server_id, ref.purpose)
        self._save_index([])


__all__ = [
    "DEFAULT_KEYRING_SERVICE_NAME",
    "SERVER_CREDENTIAL_ACCESS_TOKEN",
    "SERVER_CREDENTIAL_API_KEY",
    "SERVER_CREDENTIAL_BEARER_TOKEN",
    "SERVER_CREDENTIAL_REFRESH_TOKEN",
    "InMemoryServerCredentialStore",
    "KeyringServerCredentialStore",
    "ServerCredentialRef",
    "ServerCredentialStore",
    "redact_secret",
]
