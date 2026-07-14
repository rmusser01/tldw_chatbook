"""Source-aware routing for local and server skills."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any


class SkillsBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_BACKEND_UNAVAILABLE_CAPABILITY = [
    {
        "operation_id": "skills.local_backend_unavailable",
        "source": "local",
        "supported": False,
        "reason_code": "local_backend_unavailable",
        "user_message": "Local skills backend is unavailable.",
        "affected_action_ids": [],
    }
]


class SkillsScopeService:
    """Route source-aware SKILL.md actions across local and server backends."""

    def __init__(self, *, local_service: Any = None, server_service: Any = None, policy_enforcer: Any = None):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: SkillsBackend | str | None) -> SkillsBackend:
        if mode is None:
            return SkillsBackend.SERVER
        if isinstance(mode, SkillsBackend):
            return mode
        try:
            return SkillsBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid skills backend: {mode}") from exc

    def _require_service(self, mode: SkillsBackend) -> Any:
        if mode == SkillsBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local skills backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server skills backend is unavailable.")
        return self.server_service

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    def _source_action_id(action_id: str, mode: SkillsBackend) -> str:
        if mode == SkillsBackend.LOCAL and action_id.endswith(".server"):
            return f"{action_id[:-len('.server')]}.local"
        return action_id

    @staticmethod
    def _with_record_id(mode: SkillsBackend, kind: str, item: dict[str, Any]) -> dict[str, Any]:
        record = dict(item or {})
        record.setdefault("backend", mode.value)
        source_id = record.get("name") or record.get("skill_name") or record.get("id")
        if source_id is not None:
            record.setdefault("record_id", f"{mode.value}:{kind}:{source_id}")
        return record

    def _normalize_response(self, mode: SkillsBackend, result: Any) -> Any:
        if isinstance(result, list):
            return [self._normalize_item(mode, item) for item in result]
        if not isinstance(result, dict):
            return result
        payload = dict(result)
        payload.setdefault("backend", mode.value)
        normalized_collection = False
        if isinstance(payload.get("skills"), list):
            payload["skills"] = [self._with_record_id(mode, "skill", item) for item in payload["skills"]]
            normalized_collection = True
        if isinstance(payload.get("available_skills"), list):
            payload["available_skills"] = [
                self._with_record_id(mode, "skill", item) for item in payload["available_skills"]
            ]
            normalized_collection = True
        if isinstance(payload.get("blocked_skills"), list):
            payload["blocked_skills"] = [
                self._with_record_id(mode, "skill", item) for item in payload["blocked_skills"]
            ]
            normalized_collection = True
        if normalized_collection:
            return payload
        return self._normalize_item(mode, payload)

    def _normalize_item(self, mode: SkillsBackend, item: Any) -> Any:
        if not isinstance(item, dict):
            return item
        if "rendered_prompt" in item and "skill_name" in item:
            return self._with_record_id(mode, "skill_execution", item)
        if "seeded" in item:
            record = dict(item)
            record.setdefault("backend", mode.value)
            return record
        return self._with_record_id(mode, "skill", item)

    def list_unsupported_capabilities(
        self,
        *,
        mode: SkillsBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == SkillsBackend.LOCAL and self.local_service is None:
            return [dict(item) for item in _LOCAL_BACKEND_UNAVAILABLE_CAPABILITY]
        return []

    async def _call(
        self,
        *,
        mode: SkillsBackend | str | None,
        action_id: str,
        method_name: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_service(normalized_mode)
        self._enforce_policy(self._source_action_id(action_id, normalized_mode))
        result = await self._maybe_await(getattr(service, method_name)(*args, **(kwargs or {})))
        return self._normalize_response(normalized_mode, result)

    async def list_skills(self, *, mode: SkillsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="skills.list.server",
            method_name="list_skills",
            kwargs=kwargs,
        )

    async def get_context(self, *, mode: SkillsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="skills.context.list.server",
            method_name="get_context",
        )

    async def count_skills(self, *, mode: SkillsBackend | str | None = None, **kwargs: Any) -> int:
        """Return the total managed skills count for one backend.

        Unlike ``list_skills``/``get_context``, the result is a bare ``int``
        rather than a dict/list envelope, so this bypasses ``_call``'s
        ``_normalize_response`` step (which only mutates dict/list
        payloads) and routes directly -- mirroring ``delete_skill``'s
        bespoke dispatch.

        Args:
            mode: Which backend to query (``local`` or ``server``);
                defaults to ``server``.
            **kwargs: Forwarded to the backend service's ``count_skills``.

        Returns:
            The total managed skills count (trusted plus needs-review) for
            the selected backend.
        """
        normalized_mode = self._normalize_mode(mode)
        service = self._require_service(normalized_mode)
        self._enforce_policy(self._source_action_id("skills.context.list.server", normalized_mode))
        result = await self._maybe_await(service.count_skills(**kwargs))
        return int(result)

    async def get_skill(self, skill_name: str, *, mode: SkillsBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="skills.detail.server",
            method_name="get_skill",
            args=(skill_name,),
        )

    async def create_skill(self, *, mode: SkillsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="skills.create.server",
            method_name="create_skill",
            kwargs=kwargs,
        )

    async def update_skill(
        self,
        skill_name: str,
        *,
        mode: SkillsBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="skills.update.server",
            method_name="update_skill",
            args=(skill_name,),
            kwargs=kwargs,
        )

    async def delete_skill(
        self,
        skill_name: str,
        *,
        mode: SkillsBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_service(normalized_mode)
        self._enforce_policy(self._source_action_id("skills.delete.server", normalized_mode))
        result = await self._maybe_await(service.delete_skill(skill_name, **kwargs))
        if not isinstance(result, dict):
            result = {"name": skill_name, "deleted": bool(result)}
        return self._normalize_response(normalized_mode, result)

    async def import_skill(self, *, mode: SkillsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="skills.import.launch.server",
            method_name="import_skill",
            kwargs=kwargs,
        )

    async def import_skill_file(
        self,
        file_content: bytes,
        *,
        mode: SkillsBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="skills.import.launch.server",
            method_name="import_skill_file",
            args=(file_content,),
            kwargs=kwargs,
        )

    async def export_skill(self, skill_name: str, *, mode: SkillsBackend | str | None = None) -> Any:
        return await self._call(
            mode=mode,
            action_id="skills.export.launch.server",
            method_name="export_skill",
            args=(skill_name,),
        )

    async def execute_skill(
        self,
        skill_name: str,
        *,
        mode: SkillsBackend | str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="skills.execute.launch.server",
            method_name="execute_skill",
            args=(skill_name,),
            kwargs=kwargs,
        )

    async def seed_builtin_skills(self, *, mode: SkillsBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="skills.seed.launch.server",
            method_name="seed_builtin_skills",
            kwargs=kwargs,
        )
