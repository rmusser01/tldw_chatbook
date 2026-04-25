"""Local prompt CRUD adapter for source-aware prompt parity."""

from __future__ import annotations

from typing import Any

from . import Prompts_Interop as prompts_interop


class LocalPromptService:
    """Expose local prompt interop operations through the source-scope contract."""

    def __init__(self, interop_module: Any = prompts_interop):
        self.interop = interop_module

    def _resolve_prompt(self, prompt_identifier: int | str, *, include_deleted: bool = True) -> dict[str, Any] | None:
        return self.interop.fetch_prompt_details(prompt_identifier, include_deleted=include_deleted)

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
