"""Mode-aware routing for local and server-backed chat dictionaries."""

from __future__ import annotations

import inspect
from typing import Any


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "chat.dictionary.activity.local",
        "source": "local",
        "supported": False,
        "reason_code": "local_scope_missing",
        "user_message": "Local chat dictionary activity history is not available.",
        "affected_action_ids": ["chat.dictionary.activity.list.local"],
    },
    {
        "operation_id": "chat.dictionary.versions.local",
        "source": "local",
        "supported": False,
        "reason_code": "local_scope_missing",
        "user_message": "Local chat dictionary version history is not available.",
        "affected_action_ids": [
            "chat.dictionary.versions.detail.local",
            "chat.dictionary.versions.list.local",
            "chat.dictionary.versions.restore.local",
        ],
    },
]

_SERVER_UNSUPPORTED_CAPABILITIES: list[dict[str, Any]] = []


class ChatDictionaryScopeService:
    """Route chat dictionary operations to the selected backend."""

    def __init__(self, *, local_service: Any, server_service: Any, policy_enforcer: Any = None):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: str | None) -> str:
        normalized_mode = "local" if mode is None else str(mode).strip().lower()
        if normalized_mode not in {"local", "server"}:
            raise ValueError(f"Invalid chat dictionary mode: {mode!r}. Expected 'local' or 'server'.")
        return normalized_mode

    def _backend(self, mode: str | None) -> Any:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == "server":
            if self.server_service is None:
                raise ValueError("Server chat dictionary backend is unavailable.")
            return self.server_service
        if self.local_service is None:
            raise ValueError("Local chat dictionary backend is unavailable.")
        return self.local_service

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    def _dictionary_action(mode: str, action: str) -> str:
        return f"chat.dictionaries.{action}.{mode}"

    @staticmethod
    def _entry_action(mode: str, action: str) -> str:
        return f"chat.dictionary.entries.{action}.{mode}"

    @staticmethod
    def _activity_action(mode: str, action: str) -> str:
        return f"chat.dictionary.activity.{action}.{mode}"

    @staticmethod
    def _version_action(mode: str, action: str) -> str:
        return f"chat.dictionary.versions.{action}.{mode}"

    @staticmethod
    def _statistics_action(mode: str, action: str) -> str:
        return f"chat.dictionary.statistics.{action}.{mode}"

    def list_unsupported_capabilities(self, *, mode: str | None = None) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == "local":
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]

    async def _invoke(
        self,
        mode: str | None,
        action_id: str,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(action_id)
        backend = self._backend(normalized_mode)
        method = getattr(backend, method_name, None)
        if not callable(method):
            raise ValueError(f"Chat dictionary backend does not provide {method_name}().")
        return await self._maybe_await(method(*args, **kwargs))

    def _raise_local_activity_unsupported(self) -> None:
        raise ValueError(_LOCAL_UNSUPPORTED_CAPABILITIES[0]["user_message"])

    def _raise_local_versions_unsupported(self) -> None:
        raise ValueError(_LOCAL_UNSUPPORTED_CAPABILITIES[1]["user_message"])

    async def list_dictionaries(self, mode: str = "local", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        return await self._invoke(
            normalized_mode,
            self._dictionary_action(normalized_mode, "list"),
            "list_dictionaries",
            **kwargs,
        )

    async def create_dictionary(self, request_data: Any, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        return await self._invoke(
            normalized_mode,
            self._dictionary_action(normalized_mode, "create"),
            "create_dictionary",
            request_data,
        )

    async def get_dictionary(self, dictionary_id: int, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        return await self._invoke(
            normalized_mode,
            self._dictionary_action(normalized_mode, "detail"),
            "get_dictionary",
            dictionary_id,
        )

    async def update_dictionary(
        self,
        dictionary_id: int,
        request_data: Any,
        mode: str = "local",
        **kwargs: Any,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        return await self._invoke(
            normalized_mode,
            self._dictionary_action(normalized_mode, "update"),
            "update_dictionary",
            dictionary_id,
            request_data,
            **kwargs,
        )

    async def delete_dictionary(self, dictionary_id: int, mode: str = "local", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        return await self._invoke(
            normalized_mode,
            self._dictionary_action(normalized_mode, "delete"),
            "delete_dictionary",
            dictionary_id,
            **kwargs,
        )

    async def add_entry(self, dictionary_id: int, request_data: Any, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        return await self._invoke(
            normalized_mode,
            self._entry_action(normalized_mode, "create"),
            "add_entry",
            dictionary_id,
            request_data,
        )

    async def list_entries(self, dictionary_id: int, mode: str = "local", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        return await self._invoke(
            normalized_mode,
            self._entry_action(normalized_mode, "list"),
            "list_entries",
            dictionary_id,
            **kwargs,
        )

    async def update_entry(self, entry_id: int | str, request_data: Any, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        return await self._invoke(
            normalized_mode,
            self._entry_action(normalized_mode, "update"),
            "update_entry",
            entry_id,
            request_data,
        )

    async def delete_entry(self, entry_id: int | str, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        return await self._invoke(
            normalized_mode,
            self._entry_action(normalized_mode, "delete"),
            "delete_entry",
            entry_id,
        )

    async def reorder_entries(self, dictionary_id: int, request_data: Any, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        return await self._invoke(
            normalized_mode,
            self._entry_action(normalized_mode, "reorder"),
            "reorder_entries",
            dictionary_id,
            request_data,
        )

    async def process_text(self, request_data: Any, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        return await self._invoke(
            normalized_mode,
            self._dictionary_action(normalized_mode, "process"),
            "process_text",
            request_data,
        )

    async def import_markdown(self, request_data: Any, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        return await self._invoke(
            normalized_mode,
            self._dictionary_action(normalized_mode, "import"),
            "import_markdown",
            request_data,
        )

    async def export_markdown(self, dictionary_id: int, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        return await self._invoke(
            normalized_mode,
            self._dictionary_action(normalized_mode, "export"),
            "export_markdown",
            dictionary_id,
        )

    async def import_json(self, request_data: Any, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        return await self._invoke(
            normalized_mode,
            self._dictionary_action(normalized_mode, "import"),
            "import_json",
            request_data,
        )

    async def export_json(self, dictionary_id: int, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        return await self._invoke(
            normalized_mode,
            self._dictionary_action(normalized_mode, "export"),
            "export_json",
            dictionary_id,
        )

    async def list_activity(self, dictionary_id: int, mode: str = "local", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._activity_action(normalized_mode, "list"))
        if normalized_mode == "local":
            self._raise_local_activity_unsupported()
        return await self._maybe_await(self._backend(normalized_mode).list_activity(dictionary_id, **kwargs))

    async def list_versions(self, dictionary_id: int, mode: str = "local", **kwargs: Any) -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._version_action(normalized_mode, "list"))
        if normalized_mode == "local":
            self._raise_local_versions_unsupported()
        return await self._maybe_await(self._backend(normalized_mode).list_versions(dictionary_id, **kwargs))

    async def get_version(self, dictionary_id: int, revision: int, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._version_action(normalized_mode, "detail"))
        if normalized_mode == "local":
            self._raise_local_versions_unsupported()
        return await self._maybe_await(self._backend(normalized_mode).get_version(dictionary_id, revision))

    async def revert_version(self, dictionary_id: int, revision: int, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._version_action(normalized_mode, "restore"))
        if normalized_mode == "local":
            self._raise_local_versions_unsupported()
        return await self._maybe_await(self._backend(normalized_mode).revert_version(dictionary_id, revision))

    async def get_statistics(self, dictionary_id: int, mode: str = "local") -> Any:
        normalized_mode = self._normalize_mode(mode)
        return await self._invoke(
            normalized_mode,
            self._statistics_action(normalized_mode, "detail"),
            "get_statistics",
            dictionary_id,
        )


__all__ = ["ChatDictionaryScopeService"]
