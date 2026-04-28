from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


SERVER_CREDENTIAL_ACCESS_TOKEN = "access_token"
SERVER_CREDENTIAL_REFRESH_TOKEN = "refresh_token"
SERVER_CREDENTIAL_API_KEY = "api_key"
SERVER_CREDENTIAL_BEARER_TOKEN = "bearer_token"
DEFAULT_KEYRING_SERVICE_NAME = "tldw_chatbook.server_credentials"

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


def _credential_ref(server_id: str, purpose: str) -> ServerCredentialRef:
    return ServerCredentialRef(
        server_id=_normalize_non_empty(server_id, "server_id"),
        purpose=_normalize_non_empty(purpose, "purpose"),
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

    def set_secret(self, server_id: str, purpose: str, secret: str) -> None:
        ref = _credential_ref(server_id, purpose)
        self._keyring.set_password(self.service_name, ref.username, secret)

    def get_secret(self, server_id: str, purpose: str) -> str | None:
        ref = _credential_ref(server_id, purpose)
        return self._keyring.get_password(self.service_name, ref.username)

    def delete_secret(self, server_id: str, purpose: str) -> None:
        ref = _credential_ref(server_id, purpose)
        try:
            self._keyring.delete_password(self.service_name, ref.username)
        except Exception:
            return

    def clear_server(self, server_id: str) -> None:
        normalized_server_id = _normalize_non_empty(server_id, "server_id")
        for purpose in _KNOWN_SERVER_CREDENTIAL_PURPOSES:
            self.delete_secret(normalized_server_id, purpose)


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
