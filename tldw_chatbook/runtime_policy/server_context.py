from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping
from uuid import UUID

from loguru import logger

from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
from tldw_chatbook.MCP.unified_control_models import ConfiguredServerTarget

if TYPE_CHECKING:
    from tldw_chatbook.tldw_api import TLDWAPIClient

from .bootstrap import (
    RuntimePolicyContext,
    build_runtime_api_client,
    derive_configured_server_binding,
)
from .server_credentials import (
    CredentialStoreUnavailable,
    SERVER_CREDENTIAL_ACCESS_TOKEN,
    SERVER_CREDENTIAL_API_KEY,
    SERVER_CREDENTIAL_BEARER_TOKEN,
    SERVER_CREDENTIAL_REFRESH_TOKEN,
    ServerCredentialStore,
)
from .types import (
    RuntimeSourceState,
    ServerContextFailure,
    ServerContextFailureReason,
)

_CHARACTER_AUTHORITY_DOMAIN = b"tldw-chatbook.character-authority"
_CHARACTER_AUTHORITY_VERSION = b"1"
_MAX_SERVER_USER_ID = 2**63 - 1


def encode_server_user_character_authority(
    authority_scope_id: str,
    user_id: int,
) -> str:
    """Encode one configured-target scope and authenticated user as authority.

    Args:
        authority_scope_id: Exact canonical lowercase hyphenated UUIDv4 scope.
        user_id: Exact positive integer server user identifier.

    Returns:
        The fixed-width version-one server-user authority identifier.

    Raises:
        ValueError: If either identity component is outside the canonical
            encoding contract.
    """
    if type(authority_scope_id) is not str:
        raise ValueError("authority_scope_id must be a canonical UUIDv4")
    try:
        scope_ascii = authority_scope_id.encode("ascii")
        parsed_scope = UUID(authority_scope_id)
    except (UnicodeEncodeError, ValueError):
        raise ValueError("authority_scope_id must be a canonical UUIDv4") from None
    if parsed_scope.version != 4 or str(parsed_scope) != authority_scope_id:
        raise ValueError("authority_scope_id must be a canonical UUIDv4")

    if type(user_id) is not int or not 1 <= user_id <= _MAX_SERVER_USER_ID:
        raise ValueError("user_id must be a positive signed 64-bit integer")
    user_id_ascii = str(user_id).encode("ascii")

    def _length_prefix(value: bytes) -> bytes:
        return len(value).to_bytes(4, byteorder="big", signed=False) + value

    frame = b"".join(
        (
            _length_prefix(_CHARACTER_AUTHORITY_DOMAIN),
            _length_prefix(_CHARACTER_AUTHORITY_VERSION),
            _length_prefix(scope_ascii),
            _length_prefix(user_id_ascii),
        )
    )
    return f"server-user-v1:{hashlib.sha256(frame).hexdigest()}"


@dataclass(frozen=True, slots=True)
class ActiveServerContext:
    active_server_id: str
    label: str | None
    base_url: str
    auth_method: str
    auth_token: str | None
    credential_source: str
    target: ConfiguredServerTarget
    server_headers: Mapping[str, str]
    capabilities: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _CachedClientKey:
    active_server_id: str
    base_url: str
    auth_method: str
    credential_source: str
    token_fingerprint: str | None


@dataclass(frozen=True, slots=True)
class _CharacterAuthorityContextCapture:
    runtime_state: RuntimeSourceState = field(repr=False)
    runtime_revision: int
    expected_server_id: str = field(repr=False)
    authority_scope_id: str = field(repr=False)
    active_context: ActiveServerContext = field(repr=False)
    client_cache_key: _CachedClientKey = field(repr=False)
    client: TLDWAPIClient = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _CachedCharacterAuthority:
    capture: _CharacterAuthorityContextCapture = field(repr=False)
    authority_id: str = field(repr=False)


class ServerContextError(RuntimeError):
    reason_code = "server_context_unavailable"

    def __init__(
        self,
        message: str,
        *,
        reason_code: ServerContextFailureReason | str | None = None,
        recoverable: bool = True,
        active_server_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code or self.reason_code
        self.recoverable = recoverable
        self.active_server_id = active_server_id

    def to_contract(self) -> dict[str, object]:
        return ServerContextFailure(
            reason_code=self.reason_code,
            message=str(self),
            recoverable=self.recoverable,
            active_server_id=self.active_server_id,
        ).to_contract()


class ServerContextUnavailable(ServerContextError):
    reason_code = "server_not_configured"


class ServerCredentialsUnavailable(ServerContextError):
    reason_code = "server_credentials_unavailable"


class ServerIdentityUnavailable(ServerContextError):
    reason_code = "server_identity_unavailable"

    def __init__(self) -> None:
        super().__init__("Server identity unavailable.")


class RuntimeServerContextProvider:
    def __init__(
        self,
        *,
        runtime_context: RuntimePolicyContext,
        target_store: ConfiguredServerTargetStore,
        credential_store: ServerCredentialStore,
        app_config: Mapping[str, Any] | None,
    ) -> None:
        self.runtime_context = runtime_context
        self.target_store = target_store
        self.credential_store = credential_store
        self.app_config = app_config or {}
        self._legacy_cleared_server_ids: set[str] = set()
        self._cached_client_key: _CachedClientKey | None = None
        self._cached_client: TLDWAPIClient | None = None
        self._cached_character_authority: _CachedCharacterAuthority | None = None
        self._pending_client_close_tasks: set[asyncio.Task[None]] = set()

    def get_active_context(self) -> ActiveServerContext:
        active_server_id = self._require_active_server_id()
        target = self.resolve_target()
        using_legacy_fallback_target = False
        if target is None:
            target = self._legacy_target_for_active_server(active_server_id)
            using_legacy_fallback_target = target is not None
        if target is None:
            raise ServerContextUnavailable(
                "Active server profile is unavailable.",
                reason_code="server_profile_missing",
                active_server_id=active_server_id,
            )

        auth_token, credential_source = self._resolve_auth_token(
            active_server_id,
            target,
            allow_legacy_config=self._should_allow_legacy_config(
                active_server_id,
                target,
                using_legacy_fallback_target=using_legacy_fallback_target,
            ),
        )
        return ActiveServerContext(
            active_server_id=active_server_id,
            label=target.label or None,
            base_url=target.base_url,
            auth_method=target.auth_mode,
            auth_token=auth_token,
            credential_source=credential_source,
            target=target,
            server_headers=self._build_server_headers(target.auth_mode, auth_token),
            capabilities=self._build_capabilities(target),
        )

    def build_client(self) -> TLDWAPIClient:
        context = self.get_active_context()
        self._ensure_client_context_usable(context)
        cache_key = self._client_cache_key(context)
        if self._cached_client is not None and self._cached_client_key == cache_key:
            return self._cached_client

        self._invalidate_cached_client()
        self._cached_client_key = cache_key
        self._cached_client = build_runtime_api_client(
            endpoint_url=context.base_url,
            auth_method=context.auth_method,
            auth_token=context.auth_token,
        )
        return self._cached_client

    async def resolve_character_authority_id(
        self,
        *,
        expected_server_id: str,
        context_capture: object | None = None,
    ) -> str:
        """Resolve authority for the expected target's authenticated user.

        Args:
            expected_server_id: Configured target identifier already carried
                by the caller's character context.
            context_capture: Optional opaque capture returned by
                :meth:`capture_character_authority_context`. Supplying it
                binds identity resolution to the caller's exact authenticated
                context across later operations.

        Returns:
            The versioned server-user character authority identifier.

        Raises:
            ServerIdentityUnavailable: If exact stable target and account
                authority cannot be proven.
            asyncio.CancelledError: If the caller cancels the identity lookup.
        """
        try:
            if context_capture is None:
                capture = self._capture_character_authority_context(
                    expected_server_id
                )
            elif (
                type(context_capture) is _CharacterAuthorityContextCapture
                and context_capture.expected_server_id == expected_server_id
                and self._character_authority_capture_matches(context_capture)
            ):
                capture = context_capture
            else:
                raise ServerIdentityUnavailable()
        except Exception:
            raise ServerIdentityUnavailable() from None

        cached = self._cached_character_authority
        if cached is not None and self._same_authority_capture(
            cached.capture,
            capture,
        ):
            return cached.authority_id

        try:
            response = await capture.client.get_current_user_profile(
                sections="identity"
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            raise ServerIdentityUnavailable() from None

        if not self._character_authority_capture_matches(capture):
            raise ServerIdentityUnavailable() from None

        try:
            user_id = self._extract_identity_user_id(response)
            authority_id = encode_server_user_character_authority(
                capture.authority_scope_id,
                user_id,
            )
        except Exception:
            raise ServerIdentityUnavailable() from None

        cached = self._cached_character_authority
        if cached is not None and self._same_authority_capture(
            cached.capture,
            capture,
        ):
            if cached.authority_id != authority_id:
                raise ServerIdentityUnavailable() from None
            return cached.authority_id

        self._cached_character_authority = _CachedCharacterAuthority(
            capture=capture,
            authority_id=authority_id,
        )
        return authority_id

    def capture_character_authority_context(
        self,
        *,
        expected_server_id: str,
    ) -> object:
        """Return an opaque proof of the current authenticated character context.

        The returned object intentionally exposes no public credential or
        client fields. Callers may only pass it back to the provider's
        character-authority APIs.
        """
        try:
            return self._capture_character_authority_context(expected_server_id)
        except Exception:
            raise ServerIdentityUnavailable() from None

    def is_character_authority_context_current(self, capture: object) -> bool:
        """Return whether an opaque capture still owns the authenticated client."""
        return (
            type(capture) is _CharacterAuthorityContextCapture
            and self._character_authority_capture_matches(capture)
        )

    async def close_cached_client(self) -> None:
        cached_client = self._cached_client
        self._cached_client = None
        self._cached_client_key = None
        self._cached_character_authority = None
        if cached_client is not None:
            await cached_client.close()
        pending_tasks = list(self._pending_client_close_tasks)
        if pending_tasks:
            await asyncio.gather(*pending_tasks)

    def clear_active_server_credentials(self) -> None:
        active_server_id = self._require_active_server_id()
        self.credential_store.clear_server(active_server_id)
        self._mark_legacy_server_id_cleared(active_server_id)
        self._invalidate_cached_client()

    def clear_server_credentials(self, server_id: str) -> None:
        self.credential_store.clear_server(server_id)
        self._mark_legacy_server_id_cleared(server_id)
        self._invalidate_cached_client()

    def clear_all_credentials(self) -> None:
        self.credential_store.clear_all()
        self._legacy_cleared_server_ids.update(self._legacy_server_ids_for_signout())
        self._invalidate_cached_client()

    def invalidate_for_server_switch(
        self,
        previous_server_id: str | None,
        next_server_id: str | None,
    ) -> None:
        previous_normalized = self._normalize_optional_server_id(previous_server_id)
        next_normalized = self._normalize_optional_server_id(next_server_id)
        if previous_normalized == next_normalized:
            return

        self._invalidate_cached_client()
        self._invalidate_event_handles_for_server_switch(
            previous_normalized, next_normalized
        )
        self._invalidate_sync_handles_for_server_switch(
            previous_normalized, next_normalized
        )

    def rebind_app_config(
        self,
        app_config: Mapping[str, Any] | None,
        *,
        previous_server_id: str | None,
        next_server_id: str | None,
    ) -> None:
        self.app_config = app_config or {}
        self._invalidate_cached_client()

        previous = self._normalize_optional_server_id(previous_server_id)
        next_id = self._normalize_optional_server_id(next_server_id)
        if previous != next_id:
            self._invalidate_event_handles_for_server_switch(previous, next_id)
            self._invalidate_sync_handles_for_server_switch(previous, next_id)

        try:
            self.target_store.upsert_legacy_config_target(self.app_config)
        except Exception as exc:
            logger.warning(
                "Legacy server target refresh failed after runtime commit "
                "(exception_category={}).",
                type(exc).__name__,
            )

    def clear_active_server_auth_tokens(self) -> None:
        active_server_id = self._require_active_server_id()
        self.credential_store.delete_secret(
            active_server_id, SERVER_CREDENTIAL_ACCESS_TOKEN
        )
        self.credential_store.delete_secret(
            active_server_id, SERVER_CREDENTIAL_REFRESH_TOKEN
        )
        self._invalidate_cached_client()

    def store_auth_tokens(
        self,
        *,
        access_token: str | None = None,
        refresh_token: str | None = None,
    ) -> None:
        context = self.get_active_context()
        if access_token:
            self.credential_store.set_secret(
                context.active_server_id,
                SERVER_CREDENTIAL_ACCESS_TOKEN,
                access_token,
            )
        if refresh_token:
            self.credential_store.set_secret(
                context.active_server_id,
                SERVER_CREDENTIAL_REFRESH_TOKEN,
                refresh_token,
            )
        if access_token or refresh_token:
            self._legacy_cleared_server_ids.discard(context.active_server_id)
            self._invalidate_cached_client()

    def resolve_target(self) -> ConfiguredServerTarget | None:
        active_server_id = self._require_active_server_id()
        target = self.target_store.get_target(active_server_id)
        if target is not None:
            return target

        resolved_target = self.target_store.resolve_active_target()
        if (
            resolved_target is not None
            and resolved_target.server_id == active_server_id
        ):
            return resolved_target
        return None

    def _capture_character_authority_context(
        self,
        expected_server_id: str,
    ) -> _CharacterAuthorityContextCapture:
        if type(expected_server_id) is not str:
            raise ServerIdentityUnavailable()
        if not expected_server_id or expected_server_id != expected_server_id.strip():
            raise ServerIdentityUnavailable()

        runtime_state, runtime_revision = self.runtime_context.snapshot()
        if not self._runtime_state_matches_expected_target(
            runtime_state,
            expected_server_id,
        ):
            raise ServerIdentityUnavailable()

        authority_scope_id = self.target_store.ensure_authority_scope_id(
            expected_server_id
        )
        active_context = self.get_active_context()
        if (
            active_context.active_server_id != expected_server_id
            or active_context.target.server_id != expected_server_id
            or active_context.target.authority_scope_id != authority_scope_id
        ):
            raise ServerIdentityUnavailable()

        client_cache_key = self._client_cache_key(active_context)
        client = self.build_client()
        capture = _CharacterAuthorityContextCapture(
            runtime_state=runtime_state,
            runtime_revision=runtime_revision,
            expected_server_id=expected_server_id,
            authority_scope_id=authority_scope_id,
            active_context=active_context,
            client_cache_key=client_cache_key,
            client=client,
        )
        if not self._character_authority_capture_matches(capture):
            raise ServerIdentityUnavailable()
        return capture

    def _character_authority_capture_matches(
        self,
        capture: _CharacterAuthorityContextCapture,
    ) -> bool:
        runtime_state, runtime_revision = self.runtime_context.snapshot()
        if (
            runtime_revision != capture.runtime_revision
            or runtime_state != capture.runtime_state
            or not self._runtime_state_matches_expected_target(
                runtime_state,
                capture.expected_server_id,
            )
        ):
            return False

        try:
            authority_scope_id = self.target_store.ensure_authority_scope_id(
                capture.expected_server_id
            )
            active_context = self.get_active_context()
            client_cache_key = self._client_cache_key(active_context)
        except Exception:
            return False

        return (
            authority_scope_id == capture.authority_scope_id
            and active_context == capture.active_context
            and client_cache_key == capture.client_cache_key
            and self._cached_client_key == capture.client_cache_key
            and self._cached_client is capture.client
        )

    @staticmethod
    def _same_authority_capture(
        left: _CharacterAuthorityContextCapture,
        right: _CharacterAuthorityContextCapture,
    ) -> bool:
        return left == right and left.client is right.client

    @staticmethod
    def _runtime_state_matches_expected_target(
        runtime_state: RuntimeSourceState,
        expected_server_id: str,
    ) -> bool:
        active_server_id = runtime_state.active_server_id
        return (
            runtime_state.active_source == "server"
            and runtime_state.server_configured
            and active_server_id == expected_server_id
        )

    @staticmethod
    def _extract_identity_user_id(response: Any) -> int:
        missing = object()
        if isinstance(response, Mapping):
            mapped_user = response.get("user", missing)
            attribute_user = getattr(response, "user", missing)
            if attribute_user is not missing and attribute_user != mapped_user:
                raise ValueError("ambiguous user identity response")
            user = mapped_user
        else:
            user = getattr(response, "user", missing)

        if not isinstance(user, Mapping):
            raise ValueError("missing user identity")
        user_id = user.get("id", missing)
        if user_id is missing:
            raise ValueError("missing user identifier")
        if type(user_id) is not int or not 1 <= user_id <= _MAX_SERVER_USER_ID:
            raise ValueError("invalid user identifier")
        return user_id

    def _require_active_server_id(self) -> str:
        state = self.runtime_context.state
        active_server_id = str(state.active_server_id or "").strip()
        if (
            state.active_source != "server"
            or not state.server_configured
            or not active_server_id
        ):
            raise ServerContextUnavailable(
                "Runtime policy does not have an active configured server.",
                reason_code="server_not_configured",
            )
        return active_server_id

    def _legacy_target_for_active_server(
        self, active_server_id: str
    ) -> ConfiguredServerTarget | None:
        legacy_binding = derive_configured_server_binding(self.app_config)
        if (
            not legacy_binding.server_configured
            or legacy_binding.active_server_id != active_server_id
        ):
            return None
        return ConfiguredServerTarget.from_legacy_tldw_api_config(self.app_config)

    def _resolve_auth_token(
        self,
        server_id: str,
        target: ConfiguredServerTarget,
        *,
        allow_legacy_config: bool,
    ) -> tuple[str | None, str]:
        purpose = self._purpose_from_auth_reference(target.auth_reference)
        if purpose is not None:
            secret = self._get_credential_secret(server_id, purpose)
            if secret is not None:
                return secret, f"credential_store:{purpose}"
            return None, "none"

        credential_error: ServerCredentialsUnavailable | None = None
        for candidate_purpose in self._purposes_for_auth_mode(target.auth_mode):
            try:
                secret = self._get_credential_secret(server_id, candidate_purpose)
            except ServerCredentialsUnavailable as exc:
                credential_error = exc
                break
            if secret is not None:
                return secret, f"credential_store:{candidate_purpose}"

        if allow_legacy_config:
            if server_id in self._legacy_cleared_server_ids:
                if credential_error is not None:
                    raise credential_error
                raise ServerCredentialsUnavailable(
                    "The active server profile is no longer authorized.",
                    reason_code="profile_no_longer_authorized",
                    active_server_id=server_id,
                )
            legacy_token = self._legacy_config_token()
            if legacy_token is not None:
                imported_purpose = self._import_legacy_token(
                    server_id, target.auth_mode, legacy_token
                )
                if imported_purpose is not None:
                    return legacy_token, f"credential_store:{imported_purpose}"
                return legacy_token, "legacy:tldw_api"
        if credential_error is not None:
            raise credential_error
        return None, "none"

    def _ensure_client_context_usable(self, context: ActiveServerContext) -> None:
        state = self.runtime_context.state
        if state.server_reachability == "unreachable":
            raise ServerContextUnavailable(
                "Active server is currently unavailable.",
                reason_code="server_unavailable",
                active_server_id=context.active_server_id,
            )
        if state.server_auth_state == "session_invalid":
            raise ServerCredentialsUnavailable(
                "Active server authorization is stale.",
                reason_code="stale_authorization",
                active_server_id=context.active_server_id,
            )
        if state.server_auth_state == "auth_required":
            raise ServerCredentialsUnavailable(
                "Active server authentication is required.",
                reason_code="auth_required",
                active_server_id=context.active_server_id,
            )
        if context.auth_token is None and self._purposes_for_auth_mode(
            context.auth_method
        ):
            raise ServerCredentialsUnavailable(
                "Active server authentication is required.",
                reason_code="auth_required",
                active_server_id=context.active_server_id,
            )

    def _get_credential_secret(self, server_id: str, purpose: str) -> str | None:
        try:
            return self.credential_store.get_secret(server_id, purpose)
        except CredentialStoreUnavailable as exc:
            raise ServerCredentialsUnavailable(
                "Credential store is unavailable for the active server.",
                reason_code=exc.reason_code,
                active_server_id=server_id,
            ) from exc
        except Exception as exc:
            raise ServerCredentialsUnavailable(
                "Server credentials are unavailable for the active server.",
                reason_code="server_credentials_unavailable",
                active_server_id=server_id,
            ) from exc

    def _legacy_config_token(self) -> str | None:
        api_config = self._legacy_api_config()
        token = (
            api_config.get("auth_token")
            or api_config.get("api_key")
            or api_config.get("bearer_token")
        )
        if token is None:
            return None
        normalized = str(token).strip()
        return normalized or None

    def _legacy_api_config(self) -> Mapping[str, Any]:
        from tldw_chatbook.config import resolve_tldw_api_config

        return resolve_tldw_api_config(self.app_config)

    def _should_allow_legacy_config(
        self,
        active_server_id: str,
        target: ConfiguredServerTarget,
        *,
        using_legacy_fallback_target: bool,
    ) -> bool:
        if (
            not using_legacy_fallback_target
            and target.auth_reference != "legacy:tldw_api"
        ):
            return False

        legacy_binding = derive_configured_server_binding(self.app_config)
        return (
            legacy_binding.server_configured
            and legacy_binding.active_server_id == active_server_id
        )

    def _mark_legacy_server_id_cleared(self, server_id: str) -> None:
        normalized_server_id = str(server_id or "").strip()
        if not normalized_server_id:
            return
        if normalized_server_id in self._legacy_server_ids_for_signout():
            self._legacy_cleared_server_ids.add(normalized_server_id)

    def _legacy_server_ids_for_signout(self) -> set[str]:
        server_ids = {
            target.server_id
            for target in self.target_store.list_targets()
            if target.auth_reference == "legacy:tldw_api"
        }
        legacy_binding = derive_configured_server_binding(self.app_config)
        if legacy_binding.server_configured and legacy_binding.active_server_id:
            server_ids.add(legacy_binding.active_server_id)
        return server_ids

    def _import_legacy_token(
        self, server_id: str, auth_mode: str, token: str
    ) -> str | None:
        purposes = self._purposes_for_auth_mode(auth_mode)
        if not purposes:
            return None

        purpose = purposes[0]
        try:
            self.credential_store.set_secret(server_id, purpose, token)
        except Exception:
            return None
        return purpose

    def _invalidate_cached_client(self) -> None:
        cached_client = self._cached_client
        self._cached_client = None
        self._cached_client_key = None
        self._cached_character_authority = None
        if cached_client is not None:
            self._close_client_sync_safe(cached_client)

    def _close_client_sync_safe(self, client: TLDWAPIClient) -> None:
        async def _close() -> None:
            try:
                await client.close()
            except Exception as exc:
                logger.warning(
                    "Runtime API client cleanup failed (exception_category={}).",
                    type(exc).__name__,
                )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_close())
            return

        task = loop.create_task(_close())
        self._pending_client_close_tasks.add(task)
        task.add_done_callback(self._pending_client_close_tasks.discard)

    def _invalidate_event_handles_for_server_switch(
        self,
        previous_server_id: str | None,
        next_server_id: str | None,
    ) -> None:
        return None

    def _invalidate_sync_handles_for_server_switch(
        self,
        previous_server_id: str | None,
        next_server_id: str | None,
    ) -> None:
        return None

    @classmethod
    def _client_cache_key(cls, context: ActiveServerContext) -> _CachedClientKey:
        return _CachedClientKey(
            active_server_id=context.active_server_id,
            base_url=context.base_url,
            auth_method=context.auth_method,
            credential_source=context.credential_source,
            token_fingerprint=cls._token_fingerprint(context.auth_token),
        )

    @staticmethod
    def _token_fingerprint(auth_token: str | None) -> str | None:
        if not auth_token:
            return None
        return hashlib.sha256(auth_token.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_optional_server_id(server_id: str | None) -> str | None:
        normalized = str(server_id or "").strip()
        return normalized or None

    def _build_capabilities(self, target: ConfiguredServerTarget) -> dict[str, Any]:
        state = self.runtime_context.state
        return {
            "server_configured": state.server_configured,
            "reachability": state.server_reachability,
            "auth_state": state.server_auth_state,
            "last_known_server_label": state.last_known_server_label,
            "target_last_known_reachability": target.last_known_reachability,
            "target_last_known_auth_state": target.last_known_auth_state,
            "target_last_known_server_label": target.last_known_server_label,
        }

    @staticmethod
    def _build_server_headers(
        auth_method: str, auth_token: str | None
    ) -> dict[str, str]:
        if not auth_token:
            return {}
        if auth_method in {"bearer", "custom_token"}:
            return {"Authorization": f"Bearer {auth_token}"}
        if auth_method == "api_key":
            return {"X-API-KEY": auth_token}
        return {}

    @staticmethod
    def _purpose_from_auth_reference(auth_reference: str | None) -> str | None:
        if not auth_reference:
            return None
        prefix = "keyring:"
        if not auth_reference.startswith(prefix):
            return None
        purpose = auth_reference[len(prefix) :].strip()
        return purpose or None

    @staticmethod
    def _purposes_for_auth_mode(auth_mode: str) -> tuple[str, ...]:
        if auth_mode == "bearer":
            return (SERVER_CREDENTIAL_BEARER_TOKEN, SERVER_CREDENTIAL_ACCESS_TOKEN)
        if auth_mode == "api_key":
            return (SERVER_CREDENTIAL_API_KEY,)
        if auth_mode == "custom_token":
            return (SERVER_CREDENTIAL_BEARER_TOKEN, SERVER_CREDENTIAL_ACCESS_TOKEN)
        return ()


__all__ = [
    "ActiveServerContext",
    "RuntimeServerContextProvider",
    "ServerContextError",
    "ServerContextUnavailable",
    "ServerCredentialsUnavailable",
    "ServerIdentityUnavailable",
    "encode_server_user_character_authority",
]
