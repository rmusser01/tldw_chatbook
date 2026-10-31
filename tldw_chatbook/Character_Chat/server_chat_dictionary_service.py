"""Policy-gated server chat dictionary adapter."""

from __future__ import annotations

from typing import Any

from tldw_chatbook.runtime_policy.bootstrap import (
    build_runtime_api_client_provider_from_config,
)
from tldw_chatbook.runtime_policy.types import PolicyDeniedError


class ServerChatDictionaryService:
    """Delegate chat dictionary/world-book operations to tldw_server."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        policy_enforcer: Any | None = None,
        client_provider: Any | None = None,
    ):
        self.client = client
        self.policy_enforcer = policy_enforcer
        self.client_provider = client_provider
        # ADR-221 dictionary half, interface parity with the local service:
        # a monotonic generation bumped by every successful remote mutation.
        # Server-backed dictionaries never enter the local send-path resolve
        # (which reads the local CharactersRAGDB only), so nothing keys a
        # cache on this today; it exists so consumers of the service seam can
        # observe "the store changed" uniformly across backends. Remote
        # mutations performed outside this process are not observable.
        self._generation = 0

    @property
    def generation(self) -> int:
        """Monotonic counter of successful remote mutations through this
        service; 0 until the first one. Reads and failed calls never bump."""
        return self._generation

    def _bump_generation(self) -> None:
        self._generation += 1

    @classmethod
    def from_config(
        cls,
        app_config: dict[str, Any] | None,
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerChatDictionaryService":
        return cls(
            client_provider=build_runtime_api_client_provider_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    @classmethod
    def from_server_context_provider(
        cls,
        provider: Any,
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerChatDictionaryService":
        return cls(client_provider=provider, policy_enforcer=policy_enforcer)

    def _require_client(self) -> Any:
        if self.client is not None:
            return self.client
        if self.client_provider is not None:
            return self.client_provider.build_client()
        raise ValueError("Server chat dictionary client is unavailable.")

    @staticmethod
    def _payload(request_data: Any) -> Any:
        if hasattr(request_data, "model_dump"):
            return request_data.model_dump(
                exclude_none=True, exclude_unset=True, mode="json"
            )
        if isinstance(request_data, dict):
            return dict(request_data)
        return request_data

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        require_ui_action_allowed = getattr(
            self.policy_enforcer, "require_ui_action_allowed", None
        )
        if callable(require_allowed):
            require_allowed(action_id=action_id)
            return
        if callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(action_id=action_id)
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id=action_id,
                    reason_code=getattr(decision, "reason_code", None)
                    or "authority_denied",
                    user_message=getattr(decision, "user_message", None)
                    or "Server chat dictionary action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None)
                    or "server",
                    authority_owner=getattr(decision, "authority_owner", None)
                    or "server",
                )

    @staticmethod
    def _dictionary_action(action: str) -> str:
        return f"chat.dictionaries.{action}.server"

    @staticmethod
    def _entry_action(action: str) -> str:
        return f"chat.dictionary.entries.{action}.server"

    @staticmethod
    def _version_action(action: str) -> str:
        return f"chat.dictionary.versions.{action}.server"

    @staticmethod
    def _activity_action(action: str) -> str:
        return f"chat.dictionary.activity.{action}.server"

    @staticmethod
    def _statistics_action(action: str) -> str:
        return f"chat.dictionary.statistics.{action}.server"

    async def list_dictionaries(self, **kwargs: Any) -> Any:
        self._enforce(self._dictionary_action("list"))
        return await self._require_client().list_chat_dictionaries(**kwargs)

    async def create_dictionary(self, request_data: Any) -> dict[str, Any]:
        self._enforce(self._dictionary_action("create"))
        result = await self._require_client().create_chat_dictionary(
            self._payload(request_data)
        )
        self._bump_generation()
        return result

    async def get_dictionary(self, dictionary_id: int) -> dict[str, Any]:
        self._enforce(self._dictionary_action("detail"))
        return await self._require_client().get_chat_dictionary(dictionary_id)

    async def update_dictionary(
        self, dictionary_id: int, request_data: Any
    ) -> dict[str, Any]:
        self._enforce(self._dictionary_action("update"))
        result = await self._require_client().update_chat_dictionary(
            dictionary_id, self._payload(request_data)
        )
        self._bump_generation()
        return result

    async def delete_dictionary(self, dictionary_id: int, **kwargs: Any) -> bool:
        self._enforce(self._dictionary_action("delete"))
        await self._require_client().delete_chat_dictionary(dictionary_id, **kwargs)
        self._bump_generation()
        return True

    async def add_entry(self, dictionary_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce(self._entry_action("create"))
        result = await self._require_client().add_chat_dictionary_entry(
            dictionary_id, self._payload(request_data)
        )
        self._bump_generation()
        return result

    async def list_entries(self, dictionary_id: int, **kwargs: Any) -> Any:
        self._enforce(self._entry_action("list"))
        return await self._require_client().list_chat_dictionary_entries(
            dictionary_id, **kwargs
        )

    async def update_entry(self, entry_id: int, request_data: Any) -> dict[str, Any]:
        self._enforce(self._entry_action("update"))
        result = await self._require_client().update_chat_dictionary_entry(
            entry_id, self._payload(request_data)
        )
        self._bump_generation()
        return result

    async def delete_entry(self, entry_id: int) -> bool:
        self._enforce(self._entry_action("delete"))
        await self._require_client().delete_chat_dictionary_entry(entry_id)
        self._bump_generation()
        return True

    async def bulk_entries(self, request_data: Any) -> dict[str, Any]:
        self._enforce(self._entry_action("update"))
        result = await self._require_client().bulk_chat_dictionary_entry_operations(
            self._payload(request_data)
        )
        self._bump_generation()
        return result

    async def reorder_entries(
        self, dictionary_id: int, request_data: Any
    ) -> dict[str, Any]:
        self._enforce(self._entry_action("reorder"))
        result = await self._require_client().reorder_chat_dictionary_entries(
            dictionary_id, self._payload(request_data)
        )
        self._bump_generation()
        return result

    async def process_text(self, request_data: Any) -> dict[str, Any]:
        self._enforce(self._dictionary_action("process"))
        return await self._require_client().process_chat_dictionaries(
            self._payload(request_data)
        )

    async def import_markdown(self, request_data: Any) -> dict[str, Any]:
        self._enforce(self._dictionary_action("import"))
        result = await self._require_client().import_chat_dictionary_markdown(
            self._payload(request_data)
        )
        self._bump_generation()
        return result

    async def export_markdown(self, dictionary_id: int) -> dict[str, Any]:
        self._enforce(self._dictionary_action("export"))
        return await self._require_client().export_chat_dictionary_markdown(
            dictionary_id
        )

    async def export_json(self, dictionary_id: int) -> dict[str, Any]:
        self._enforce(self._dictionary_action("export"))
        return await self._require_client().export_chat_dictionary_json(dictionary_id)

    async def import_json(self, request_data: Any) -> dict[str, Any]:
        self._enforce(self._dictionary_action("import"))
        result = await self._require_client().import_chat_dictionary_json(
            self._payload(request_data)
        )
        self._bump_generation()
        return result

    async def list_activity(self, dictionary_id: int, **kwargs: Any) -> Any:
        self._enforce(self._activity_action("list"))
        return await self._require_client().list_chat_dictionary_activity(
            dictionary_id, **kwargs
        )

    async def list_versions(self, dictionary_id: int, **kwargs: Any) -> Any:
        self._enforce(self._version_action("list"))
        return await self._require_client().list_chat_dictionary_versions(
            dictionary_id, **kwargs
        )

    async def get_version(self, dictionary_id: int, revision: int) -> dict[str, Any]:
        self._enforce(self._version_action("detail"))
        return await self._require_client().get_chat_dictionary_version(
            dictionary_id, revision
        )

    async def revert_version(self, dictionary_id: int, revision: int) -> dict[str, Any]:
        self._enforce(self._version_action("restore"))
        result = await self._require_client().revert_chat_dictionary_version(
            dictionary_id, revision
        )
        self._bump_generation()
        return result

    async def get_statistics(self, dictionary_id: int) -> dict[str, Any]:
        self._enforce(self._statistics_action("detail"))
        return await self._require_client().get_chat_dictionary_statistics(
            dictionary_id
        )
