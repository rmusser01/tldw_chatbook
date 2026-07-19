"""Local prompt CRUD adapter for source-aware prompt parity."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import Prompts_Interop as prompts_interop


class LocalPromptService:
    """Expose local prompt interop operations through the source-scope contract."""

    def __init__(self, interop_module: Any = prompts_interop):
        self.interop = interop_module

    def _resolve_prompt(self, prompt_identifier: int | str, *, include_deleted: bool = True) -> dict[str, Any] | None:
        return self.interop.fetch_prompt_details(prompt_identifier, include_deleted=include_deleted)

    def _prompt_version_snapshots(self, prompt_identifier: int | str) -> list[dict[str, Any]]:
        prompt = self._resolve_prompt(prompt_identifier, include_deleted=True)
        if not prompt:
            raise ValueError(f"Prompt '{prompt_identifier}' not found.")

        prompt_uuid = prompt.get("uuid")
        if not prompt_uuid:
            return []

        db = self.interop.get_db_instance()
        snapshots_by_version: dict[int, dict[str, Any]] = {}
        for entry in db.get_sync_log_entries(since_change_id=0):
            if entry.get("entity") != "Prompts":
                continue
            if entry.get("entity_uuid") != prompt_uuid:
                continue
            if entry.get("operation") not in {"create", "update"}:
                continue

            payload = entry.get("payload")
            if not isinstance(payload, Mapping):
                continue

            raw_version = payload.get("version", entry.get("version"))
            try:
                version = int(raw_version)
            except (TypeError, ValueError):
                continue

            snapshots_by_version[version] = {
                "version": version,
                "prompt_uuid": prompt_uuid,
                "operation": entry.get("operation"),
                "change_id": entry.get("change_id"),
                "created_at": entry.get("timestamp"),
                "updated_at": payload.get("last_modified") or entry.get("timestamp"),
                "name": payload.get("name"),
                "author": payload.get("author"),
                "details": payload.get("details"),
                "system_prompt": payload.get("system_prompt"),
                "user_prompt": payload.get("user_prompt"),
                "prompt_format": payload.get("prompt_format"),
                "prompt_schema_version": payload.get("prompt_schema_version"),
                "prompt_definition": payload.get("prompt_definition"),
            }

        return sorted(
            snapshots_by_version.values(),
            key=lambda snapshot: snapshot["version"],
            reverse=True,
        )

    @staticmethod
    def _prompt_update_from_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
        fields = (
            "name",
            "author",
            "details",
            "system_prompt",
            "user_prompt",
            "prompt_format",
            "prompt_schema_version",
            "prompt_definition",
        )
        return {field: snapshot.get(field) for field in fields if field in snapshot}

    async def list_prompts(
        self,
        *,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        per_page = max(1, int(limit or 100))
        page = max(1, int(offset // per_page) + 1)
        prompts, _total_pages, _current_page, _total_count = self.interop.list_prompts(
            page=page,
            per_page=per_page,
            include_deleted=include_deleted,
        )
        return prompts

    async def count_prompts(self) -> int:
        """Count all non-deleted local prompts.

        Mirrors ``list_prompts`` above: a direct, un-offloaded call into
        the interop layer (this service's methods are all thin synchronous
        wrappers, not backed by a thread pool), fetching a single row
        (``per_page=1``) purely to read the paginated response's exact
        total rather than materializing a full page.

        Returns:
            The exact number of non-deleted prompts, taken from
            ``PromptsDatabase.list_prompts``'s ``total_items`` (the fourth
            element of its return tuple).
        """
        _prompts, _total_pages, _current_page, total_items = self.interop.list_prompts(
            page=1,
            per_page=1,
            include_deleted=False,
        )
        return int(total_items)

    async def create_prompt(
        self,
        *,
        name: str,
        author: str | None = None,
        details: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        keywords: list[str] | None = None,
        overwrite: bool = False,
        prompt_format: str | None = "legacy",
        prompt_schema_version: int | None = None,
        prompt_definition: Any = None,
    ) -> dict[str, Any]:
        prompt_id, prompt_uuid, message = self.interop.add_prompt(
            name=name,
            author=author,
            details=details,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            keywords=keywords,
            overwrite=overwrite,
            prompt_format=prompt_format,
            prompt_schema_version=prompt_schema_version,
            prompt_definition=prompt_definition,
        )
        created = self._resolve_prompt(prompt_uuid or prompt_id, include_deleted=True) if (prompt_uuid or prompt_id) else None
        return created or {"id": prompt_id, "uuid": prompt_uuid, "name": name, "message": message}

    async def preview_prompt(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "name": kwargs.get("name"),
            "author": kwargs.get("author"),
            "details": kwargs.get("details"),
            "system_prompt": kwargs.get("system_prompt"),
            "user_prompt": kwargs.get("user_prompt"),
            "prompt_format": kwargs.get("prompt_format") or "legacy",
            "prompt_schema_version": kwargs.get("prompt_schema_version"),
            "prompt_definition": kwargs.get("prompt_definition"),
        }

    async def update_prompt(self, prompt_id: int | str, **kwargs: Any) -> dict[str, Any]:
        existing = self._resolve_prompt(prompt_id, include_deleted=True)
        if not existing:
            raise ValueError(f"Prompt '{prompt_id}' not found.")

        update_payload = dict(kwargs)
        db = self.interop.get_db_instance()
        prompt_uuid, message = db.update_prompt_by_id(existing["id"], update_payload)
        updated = self._resolve_prompt(prompt_uuid or existing["id"], include_deleted=True)
        return updated or {"id": existing["id"], "uuid": prompt_uuid, "message": message}

    async def delete_prompt(self, prompt_id: int | str) -> bool:
        return bool(self.interop.soft_delete_prompt(prompt_id))

    async def list_prompt_versions(self, prompt_id: int | str) -> list[dict[str, Any]]:
        return self._prompt_version_snapshots(prompt_id)

    async def restore_prompt_version(self, prompt_id: int | str, version: int) -> dict[str, Any]:
        prompt = self._resolve_prompt(prompt_id, include_deleted=True)
        if not prompt:
            raise ValueError(f"Prompt '{prompt_id}' not found.")

        for snapshot in self._prompt_version_snapshots(prompt_id):
            if snapshot.get("version") != int(version):
                continue
            db = self.interop.get_db_instance()
            prompt_uuid, message = db.update_prompt_by_id(
                int(prompt["id"]),
                self._prompt_update_from_snapshot(snapshot),
            )
            restored = self._resolve_prompt(prompt_uuid or prompt_id, include_deleted=True)
            return restored or {
                "id": prompt.get("id"),
                "uuid": prompt_uuid or prompt.get("uuid"),
                "name": snapshot.get("name") or prompt.get("name"),
                "message": message,
            }

        raise ValueError(f"Local prompt version {version} was not found for prompt '{prompt_id}'.")
