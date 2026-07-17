"""Local/offline SKILL.md management service."""

from __future__ import annotations

import asyncio
import io
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from ..runtime_policy.types import PolicyDeniedError
from ..Utils.input_validation import sanitize_string, validate_text_input
from ..Utils.path_validation import get_safe_relative_path, validate_path_simple
from .skill_trust_models import SkillTrustBlockedError


_INDEX_FILENAME = "tldw_chatbook_skills.json"
_SKILLS_DIRNAME = "skills"
_SKILL_FILENAME = "SKILL.md"
_FRONT_MATTER_PATTERN = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_METADATA_FIELDS = {
    "name",
    "description",
    "argument_hint",
    "allowed_tools",
    "allowed-tools",
    "license",
    "compatibility",
    "metadata",
    "model",
    "context",
    "user_invocable",
    "disable_model_invocation",
}
_AGENT_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
_FRONT_MATTER_MAX_LENGTH = 500000
_AGENT_SKILL_DESCRIPTION_MAX = 1000
_TEXT_FIELD_LIMITS = {
    "name": 64,
    "description": _AGENT_SKILL_DESCRIPTION_MAX,
    "argument_hint": 100,
    "license": 100,
    "compatibility": 500,
    "model": 128,
    "metadata_key": 128,
    "metadata_value": 1000,
    "allowed_tool": 128,
}
_TRUST_STATUS_SERVICE_UNAVAILABLE = "trust_locked"
_TRUST_REASON_SERVICE_UNAVAILABLE = "trust_service_unavailable"


class LocalSkillsService:
    """Chatbook-owned local skill library.

    This service intentionally stores only Chatbook local skills under the caller
    supplied ``store_dir``. It does not read or mutate Codex runtime skills.
    """

    def __init__(
        self,
        *,
        store_dir: str | Path,
        policy_enforcer: Any | None = None,
        trust_service: Any | None = None,
        allow_untrusted_without_trust_service: bool = False,
    ) -> None:
        self.store_dir = Path(store_dir)
        self.skills_dir = self.store_dir / _SKILLS_DIRNAME
        self.index_path = self.store_dir / _INDEX_FILENAME
        self.policy_enforcer = policy_enforcer
        self.trust_service = trust_service
        self.allow_untrusted_without_trust_service = allow_untrusted_without_trust_service
        self._lock = asyncio.Lock()

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        require_ui_action_allowed = getattr(self.policy_enforcer, "require_ui_action_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)
            return
        if callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(action_id=action_id)
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id=action_id,
                    reason_code=getattr(decision, "reason_code", None) or "authority_denied",
                    user_message=getattr(decision, "user_message", None) or "Local skill action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "local",
                    authority_owner=getattr(decision, "authority_owner", None) or "local",
                )

    @staticmethod
    def _dump(response: Any) -> Any:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        if isinstance(response, list):
            return [LocalSkillsService._dump(item) for item in response]
        if isinstance(response, (dict, bool)):
            return response
        return dict(response or {})

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load_index(self) -> dict[str, dict[str, Any]]:
        if not self.index_path.exists():
            return {}
        with self.index_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        skills = payload.get("skills", {})
        if not isinstance(skills, dict):
            return {}
        return {str(name): dict(record) for name, record in skills.items()}

    def _save_index(self, records: dict[str, dict[str, Any]]) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "skills": records}
        temp_path = self.index_path.with_suffix(".json.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temp_path.replace(self.index_path)

    def _skill_dir(self, skill_name: str) -> Path:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.skills_schemas import _normalize_skill_name

        return self.skills_dir / _normalize_skill_name(skill_name)

    @staticmethod
    def _write_text_atomic(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
        temp_path.replace(path)

    @staticmethod
    def _parse_front_matter(content: str) -> tuple[dict[str, Any], str]:
        match = _FRONT_MATTER_PATTERN.match(content)
        if match is None:
            return {}, content
        yaml_text = sanitize_string(match.group(1), max_length=_FRONT_MATTER_MAX_LENGTH)
        try:
            raw_metadata = yaml.safe_load(yaml_text) or {}
        except yaml.YAMLError:
            raw_metadata = {}
        if not isinstance(raw_metadata, dict):
            raw_metadata = {}
        metadata = {str(key): value for key, value in raw_metadata.items() if str(key) in _METADATA_FIELDS}
        return metadata, content[match.end() :]

    @classmethod
    def _body_description(cls, content: str) -> str | None:
        _, body = cls._parse_front_matter(content)
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            return stripped[:1000]
        return None

    @staticmethod
    def _safe_front_matter_text(
        value: Any,
        *,
        max_length: int,
        allow_html: bool = False,
    ) -> str | None:
        if not isinstance(value, str):
            return None
        text = sanitize_string(value, max_length=max_length).strip()
        if not text:
            return None
        if not validate_text_input(text, max_length=max_length, allow_html=allow_html):
            return None
        return text

    @classmethod
    def _sanitize_metadata_map(cls, value: Any) -> dict[str, str] | None:
        if not isinstance(value, dict):
            return None
        sanitized: dict[str, str] = {}
        for key, item in value.items():
            if not isinstance(item, (str, int, float, bool)):
                continue
            safe_key = cls._safe_front_matter_text(
                str(key),
                max_length=_TEXT_FIELD_LIMITS["metadata_key"],
            )
            safe_value = cls._safe_front_matter_text(
                str(item),
                max_length=_TEXT_FIELD_LIMITS["metadata_value"],
            )
            if safe_key and safe_value:
                sanitized[safe_key] = safe_value
        return sanitized or None

    @classmethod
    def _normalize_allowed_tools(cls, value: Any) -> list[str] | None:
        if value in (None, ""):
            return None
        if isinstance(value, str):
            tools = [
                tool
                for raw_tool in value.split()
                if (
                    tool := cls._safe_front_matter_text(
                        raw_tool,
                        max_length=_TEXT_FIELD_LIMITS["allowed_tool"],
                    )
                )
            ]
            return tools or None
        if isinstance(value, list):
            tools = [
                tool
                for raw_tool in value
                if (
                    tool := cls._safe_front_matter_text(
                        raw_tool,
                        max_length=_TEXT_FIELD_LIMITS["allowed_tool"],
                    )
                )
            ]
            return tools or None
        return None

    @classmethod
    def _agent_skill_validation(cls, *, directory_name: str, front_matter: dict[str, Any]) -> dict[str, Any]:
        errors: list[str] = []
        agent_skill_name = front_matter.get("name")
        description = front_matter.get("description")

        if not isinstance(agent_skill_name, str) or not agent_skill_name.strip():
            errors.append("name is required")
            normalized_agent_name = None
        else:
            normalized_agent_name = agent_skill_name.strip()
            if (
                not _AGENT_SKILL_NAME_PATTERN.match(normalized_agent_name)
                or "--" in normalized_agent_name
            ):
                errors.append("name must use lowercase letters, numbers, and hyphens")
            if normalized_agent_name != directory_name:
                errors.append("name must match the parent directory name")

        if not isinstance(description, str) or not description.strip():
            errors.append("description is required")
        elif len(description) > _AGENT_SKILL_DESCRIPTION_MAX:
            errors.append("description must be 1000 characters or fewer")

        return {
            "agent_skill_name": normalized_agent_name,
            "validation_status": "invalid" if errors else "valid",
            "validation_errors": errors,
        }

    @classmethod
    def _metadata_from_content(
        cls,
        *,
        name: str,
        content: str,
        skill_dir: Path,
        existing: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.skills_schemas import _normalize_skill_name

        from ..tldw_api import SkillSummary

        front_matter, _ = cls._parse_front_matter(content)
        now = cls._now_iso()
        base = {
            "id": f"local-skill-{name}",
            "name": name,
            "description": None,
            "argument_hint": None,
            "disable_model_invocation": False,
            "user_invocable": True,
            "allowed_tools": None,
            "model": None,
            "context": "inline",
            "directory_path": str(skill_dir),
            "created_at": now,
            "last_modified": now,
            "version": 1,
        }
        if existing is not None:
            for field in base:
                if field in existing:
                    base[field] = existing[field]
            base["last_modified"] = now
            base["directory_path"] = str(skill_dir)
        if front_matter:
            for field, value in front_matter.items():
                if field == "name":
                    safe_value = cls._safe_front_matter_text(
                        value,
                        max_length=_TEXT_FIELD_LIMITS["name"],
                    )
                    if safe_value is not None:
                        base["agent_skill_name"] = safe_value
                elif field == "description":
                    base["description"] = cls._safe_front_matter_text(
                        value,
                        max_length=_TEXT_FIELD_LIMITS["description"],
                    )
                elif field == "argument_hint":
                    base["argument_hint"] = cls._safe_front_matter_text(
                        value,
                        max_length=_TEXT_FIELD_LIMITS["argument_hint"],
                    )
                elif field == "allowed-tools":
                    base["allowed_tools"] = cls._normalize_allowed_tools(value)
                elif field == "allowed_tools":
                    base["allowed_tools"] = cls._normalize_allowed_tools(value)
                elif field == "license":
                    license_value = cls._safe_front_matter_text(
                        value,
                        max_length=_TEXT_FIELD_LIMITS["license"],
                    )
                    if license_value is not None:
                        base["license"] = license_value
                elif field == "compatibility":
                    compatibility_value = cls._safe_front_matter_text(
                        value,
                        max_length=_TEXT_FIELD_LIMITS["compatibility"],
                    )
                    if compatibility_value is not None:
                        base["compatibility"] = compatibility_value
                elif field == "metadata":
                    metadata_value = cls._sanitize_metadata_map(value)
                    if metadata_value is not None:
                        base["metadata"] = metadata_value
                elif field == "model":
                    base["model"] = cls._safe_front_matter_text(
                        value,
                        max_length=_TEXT_FIELD_LIMITS["model"],
                    )
                elif field == "context":
                    if value in {"inline", "fork"}:
                        base["context"] = value
                elif field in {"user_invocable", "disable_model_invocation"}:
                    if isinstance(value, bool):
                        base[field] = value
                else:
                    base[field] = value
        if base["description"] is None:
            base["description"] = cls._body_description(content)
        base.update(
            cls._agent_skill_validation(
                directory_name=_normalize_skill_name(name),
                front_matter=front_matter,
            )
        )
        SkillSummary(
            name=base["name"],
            description=base["description"],
            argument_hint=base["argument_hint"],
            user_invocable=base["user_invocable"],
            disable_model_invocation=base["disable_model_invocation"],
            context=base["context"],
        )
        return base

    @staticmethod
    def _read_supporting_files(skill_dir: Path) -> dict[str, str] | None:
        if not skill_dir.exists():
            return None
        supporting_files: dict[str, str] = {}
        for path in sorted(skill_dir.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.name == _SKILL_FILENAME:
                continue
            supporting_files[path.name] = LocalSkillsService._read_text_preserving_newlines(
                path,
                base_dir=skill_dir,
            )
        return supporting_files or None

    @staticmethod
    def _read_text_preserving_newlines(path: Path, *, base_dir: Path | None = None) -> str:
        base_dir = validate_path_simple(base_dir or path.parent)
        if base_dir.is_symlink():
            raise ValueError("unsafe local skill path")
        safe_path = validate_path_simple(path)
        if safe_path.is_symlink():
            raise ValueError("unsafe local skill path")
        if get_safe_relative_path(safe_path, base_dir) is None:
            raise ValueError("unsafe local skill path")
        return safe_path.read_bytes().decode("utf-8")

    def _response_for_record(self, record: dict[str, Any]) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import SkillResponse

        skill_name = str(record["name"])
        skill_dir = self._skill_dir(skill_name)
        content = self._read_text_preserving_newlines(
            skill_dir / _SKILL_FILENAME,
            base_dir=skill_dir,
        )
        response = SkillResponse(
            **record,
            content=content,
            supporting_files=self._read_supporting_files(skill_dir),
        )
        payload = self._dump(response)
        payload.update(self._trust_fields_for_record(record))
        return payload

    def _summary_for_record(self, record: dict[str, Any]) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import SkillSummary

        summary = LocalSkillsService._dump(
            SkillSummary(
                name=record["name"],
                description=record.get("description"),
                argument_hint=record.get("argument_hint"),
                user_invocable=record.get("user_invocable", True),
                disable_model_invocation=record.get("disable_model_invocation", False),
                context=record.get("context", "inline"),
            )
        )
        for field in ("agent_skill_name", "validation_status", "validation_errors", "record_id", "backend"):
            if field in record:
                summary[field] = record[field]
        summary.update(self._trust_fields_for_record(record))
        return summary

    def _trust_fields_for_record(self, record: dict[str, Any]) -> dict[str, Any]:
        if self.trust_service is None:
            if not self.allow_untrusted_without_trust_service:
                return {
                    "trust_status": _TRUST_STATUS_SERVICE_UNAVAILABLE,
                    "trust_reason_code": _TRUST_REASON_SERVICE_UNAVAILABLE,
                    "trust_blocked": True,
                    "trust_changed_files": [],
                    "trust_manifest_generation": None,
                    "trust_last_verified_at": None,
                }
            return {
                "trust_status": "trusted",
                "trust_reason_code": None,
                "trust_blocked": False,
                "trust_changed_files": [],
                "trust_manifest_generation": None,
                "trust_last_verified_at": None,
            }
        return self.trust_service.status_for_skill(str(record["name"])).response_fields()

    def _require_trusted_skill(self, skill_name: str) -> None:
        if self.trust_service is None:
            if self.allow_untrusted_without_trust_service:
                return
            raise SkillTrustBlockedError(
                skill_name=skill_name,
                reason_code=_TRUST_REASON_SERVICE_UNAVAILABLE,
                trust_status=_TRUST_STATUS_SERVICE_UNAVAILABLE,
            )
        self.trust_service.ensure_skill_trusted(skill_name)

    def _trust_after_approved_mutation(self, skill_name: str, *, trust_approved: bool) -> None:
        if not trust_approved:
            return
        # Writes and index updates intentionally happen before re-trust. If this
        # fails, later list/execute paths remain blocked until review or retry.
        if self.trust_service is None:
            if self.allow_untrusted_without_trust_service:
                return
            raise SkillTrustBlockedError(
                skill_name=skill_name,
                reason_code=_TRUST_REASON_SERVICE_UNAVAILABLE,
                trust_status=_TRUST_STATUS_SERVICE_UNAVAILABLE,
            )
        self.trust_service.trust_current_skill(
            skill_name,
            audit_event="trust_chatbook_mutation",
        )

    def _verify_exact_skill_content(self, skill: dict[str, Any]) -> None:
        if self.trust_service is None:
            self._require_trusted_skill(str(skill["name"]))
            return
        verifier = getattr(self.trust_service, "verify_skill_content", None)
        if not callable(verifier):
            raise SkillTrustBlockedError(
                skill_name=str(skill["name"]),
                reason_code="trust_verifier_unavailable",
                trust_status="trust_locked",
            )
        verifier(
            str(skill["name"]),
            skill_content=str(skill["content"]),
            supporting_files=skill.get("supporting_files"),
        )

    def _require_record(self, skill_name: str, records: dict[str, dict[str, Any]]) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.skills_schemas import _normalize_skill_name

        normalized_name = _normalize_skill_name(skill_name)
        record = records.get(normalized_name)
        if record is None:
            raise ValueError(f"local_skill_not_found:{normalized_name}")
        return record

    @staticmethod
    def _check_expected_version(skill_name: str, record: dict[str, Any], expected_version: int | None) -> None:
        if expected_version is not None and int(record.get("version", 0)) != expected_version:
            raise ValueError(f"local_skill_version_conflict:{skill_name}")

    @staticmethod
    def _apply_supporting_files(skill_dir: Path, supporting_files: dict[str, str | None] | None) -> None:
        if supporting_files is None:
            return
        for filename, content in supporting_files.items():
            path = skill_dir / filename
            if content is None:
                if path.exists():
                    path.unlink()
                continue
            LocalSkillsService._write_text_atomic(path, content)

    @staticmethod
    def _derive_name_from_filename(filename: str) -> str:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.skills_schemas import _normalize_skill_name

        candidate = PurePosixPath(filename).name
        if candidate.lower().endswith(".zip"):
            candidate = candidate[:-4]
        elif "." in candidate:
            candidate = candidate.rsplit(".", 1)[0]
        candidate = re.sub(r"[^a-z0-9-]+", "-", candidate.strip().lower()).strip("-")
        if not candidate:
            candidate = "skill-import"
        return _normalize_skill_name(candidate[:64].rstrip("-") or "skill-import")

    @staticmethod
    def _validate_archive_member(name: str) -> str:
        posix_path = PurePosixPath(name)
        if posix_path.is_absolute() or ".." in posix_path.parts or len(posix_path.parts) != 1:
            raise ValueError(f"local_skill_invalid_archive:{name}")
        filename = posix_path.name
        if not filename:
            raise ValueError(f"local_skill_invalid_archive:{name}")
        return filename

    async def list_skills(
        self,
        *,
        include_hidden: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import SkillsListResponse

        self._enforce("skills.list.local")
        records = self._load_index()
        summaries = [self._summary_for_record(record) for _, record in sorted(records.items())]
        page = summaries[offset : offset + limit]
        return self._dump(SkillsListResponse(skills=page, count=len(page), total=len(summaries), limit=limit, offset=offset))

    async def get_context(self) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import SkillContextPayload

        self._enforce("skills.context.list.local")
        records = self._load_index()
        available: list[dict[str, Any]] = []
        blocked: list[dict[str, Any]] = []
        for _, record in sorted(records.items()):
            summary = self._summary_for_record(record)
            if summary.get("trust_blocked"):
                blocked.append(summary)
                continue
            available.append(summary)
        context_lines = []
        for summary in available:
            description = f": {summary['description']}" if summary.get("description") else ""
            argument_hint = f" (args: {summary['argument_hint']})" if summary.get("argument_hint") else ""
            context_lines.append(f"- {summary['name']}{description}{argument_hint}")
        payload = self._dump(
            SkillContextPayload(
                available_skills=available,
                context_text="\n".join(context_lines),
            )
        )
        payload["blocked_skills"] = blocked
        return payload

    async def count_skills(self) -> int:
        """Return the total managed skills count, trusted plus needs-review.

        Reuses ``get_context`` so the count always matches what it would
        enumerate: both the trusted ``available_skills`` population and the
        ``blocked_skills`` (trust needs-review) population, per the Skills
        spec's blocked-skills visibility rule -- a skill pending trust
        review is still a managed skill even though it can't be invoked
        yet.

        Returns:
            ``len(available_skills) + len(blocked_skills)``.
        """
        ctx = await self.get_context()
        return len(ctx.get("available_skills") or []) + len(ctx.get("blocked_skills") or [])

    async def get_skill(self, skill_name: str) -> dict[str, Any]:
        self._enforce("skills.detail.local")
        records = self._load_index()
        return self._response_for_record(self._require_record(skill_name, records))

    async def create_skill(
        self,
        *,
        name: str,
        content: str,
        supporting_files: dict[str, str] | None = None,
        trust_approved: bool = False,
    ) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import SkillCreate

        self._enforce("skills.create.local")
        request = SkillCreate(name=name, content=content, supporting_files=supporting_files)
        async with self._lock:
            records = self._load_index()
            skill_name = request.name
            if skill_name in records:
                raise ValueError(f"local_skill_exists:{skill_name}")
            skill_dir = self._skill_dir(skill_name)
            skill_dir.mkdir(parents=True, exist_ok=True)
            self._write_text_atomic(skill_dir / _SKILL_FILENAME, request.content)
            self._apply_supporting_files(skill_dir, request.supporting_files)
            records[skill_name] = self._metadata_from_content(
                name=skill_name,
                content=request.content,
                skill_dir=skill_dir,
            )
            self._save_index(records)
            self._trust_after_approved_mutation(skill_name, trust_approved=trust_approved)
            return self._response_for_record(records[skill_name])

    async def update_skill(
        self,
        skill_name: str,
        *,
        content: str | None = None,
        supporting_files: dict[str, str | None] | None = None,
        expected_version: int | None = None,
        trust_approved: bool = False,
    ) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.skills_schemas import _normalize_skill_name

        from ..tldw_api import SkillUpdate

        self._enforce("skills.update.local")
        request = SkillUpdate(content=content, supporting_files=supporting_files)
        async with self._lock:
            records = self._load_index()
            normalized_name = _normalize_skill_name(skill_name)
            record = self._require_record(normalized_name, records)
            self._check_expected_version(normalized_name, record, expected_version)
            skill_dir = self._skill_dir(normalized_name)
            skill_content_path = skill_dir / _SKILL_FILENAME
            next_content = request.content
            if next_content is not None:
                self._write_text_atomic(skill_content_path, next_content)
            else:
                next_content = self._read_text_preserving_newlines(skill_content_path)
            self._apply_supporting_files(skill_dir, request.supporting_files)
            next_record = self._metadata_from_content(
                name=normalized_name,
                content=next_content,
                skill_dir=skill_dir,
                existing=record,
            )
            next_record["version"] = int(record.get("version", 0)) + 1
            records[normalized_name] = next_record
            self._save_index(records)
            self._trust_after_approved_mutation(normalized_name, trust_approved=trust_approved)
            return self._response_for_record(next_record)

    async def delete_skill(self, skill_name: str, *, expected_version: int | None = None) -> bool:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api.skills_schemas import _normalize_skill_name

        self._enforce("skills.delete.local")
        async with self._lock:
            records = self._load_index()
            normalized_name = _normalize_skill_name(skill_name)
            record = self._require_record(normalized_name, records)
            self._check_expected_version(normalized_name, record, expected_version)
            records.pop(normalized_name, None)
            shutil.rmtree(self._skill_dir(normalized_name), ignore_errors=True)
            self._save_index(records)
            return True

    async def import_skill(
        self,
        *,
        content: str,
        name: str | None = None,
        supporting_files: dict[str, str] | None = None,
        overwrite: bool = False,
        trust_approved: bool = False,
    ) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import SkillImportRequest

        self._enforce("skills.import.launch.local")
        request = SkillImportRequest(
            name=name,
            content=content,
            supporting_files=supporting_files,
            overwrite=overwrite,
        )
        skill_name = request.name or self._derive_name_from_filename("imported-skill.md")
        async with self._lock:
            records = self._load_index()
            if skill_name in records and not request.overwrite:
                raise ValueError(f"local_skill_exists:{skill_name}")
            skill_dir = self._skill_dir(skill_name)
            if request.overwrite and skill_dir.exists():
                shutil.rmtree(skill_dir)
            skill_dir.mkdir(parents=True, exist_ok=True)
            existing = records.get(skill_name) if request.overwrite else None
            self._write_text_atomic(skill_dir / _SKILL_FILENAME, request.content)
            self._apply_supporting_files(skill_dir, request.supporting_files)
            record = self._metadata_from_content(
                name=skill_name,
                content=request.content,
                skill_dir=skill_dir,
                existing=existing,
            )
            if existing is not None:
                record["version"] = int(existing.get("version", 0)) + 1
            records[skill_name] = record
            self._save_index(records)
            self._trust_after_approved_mutation(skill_name, trust_approved=trust_approved)
            return self._response_for_record(record)

    async def import_skill_file(
        self,
        file_content: bytes,
        *,
        filename: str = _SKILL_FILENAME,
        content_type: str = "text/markdown",
        overwrite: bool = False,
        trust_approved: bool = False,
    ) -> dict[str, Any]:
        self._enforce("skills.import.launch.local")
        is_zip = content_type in {"application/zip", "application/x-zip-compressed"} or filename.lower().endswith(".zip")
        if not is_zip:
            return await self.import_skill(
                name=self._derive_name_from_filename(filename),
                content=file_content.decode("utf-8"),
                overwrite=overwrite,
                trust_approved=trust_approved,
            )

        supporting_files: dict[str, str] = {}
        skill_content: str | None = None
        with zipfile.ZipFile(io.BytesIO(file_content), "r") as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                member_name = self._validate_archive_member(member.filename)
                data = archive.read(member).decode("utf-8")
                if member_name == _SKILL_FILENAME:
                    skill_content = data
                else:
                    supporting_files[member_name] = data
        if skill_content is None:
            raise ValueError("local_skill_invalid_archive:missing_skill_md")
        return await self.import_skill(
            name=self._derive_name_from_filename(filename),
            content=skill_content,
            supporting_files=supporting_files or None,
            overwrite=overwrite,
            trust_approved=trust_approved,
        )

    async def export_skill(self, skill_name: str) -> Any:
        self._enforce("skills.export.launch.local")
        skill = await self.get_skill(skill_name)
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(_SKILL_FILENAME, skill["content"])
            for filename, content in sorted((skill.get("supporting_files") or {}).items()):
                archive.writestr(filename, content)
        return {
            "content": archive_buffer.getvalue(),
            "filename": f"{skill['name']}.zip",
            "content_type": "application/zip",
        }

    async def execute_skill(self, skill_name: str, *, args: str | None = None) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import SkillExecuteRequest, SkillExecutionResult

        self._enforce("skills.execute.launch.local")
        self._require_trusted_skill(skill_name)
        request = SkillExecuteRequest(args=args)
        skill = await self.get_skill(skill_name)
        self._verify_exact_skill_content(skill)
        _, body = self._parse_front_matter(skill["content"])
        rendered_prompt = body.strip().replace("{{args}}", request.args or "")
        return self._dump(
            SkillExecutionResult(
                skill_name=skill["name"],
                rendered_prompt=rendered_prompt,
                allowed_tools=skill.get("allowed_tools"),
                model_override=skill.get("model"),
                execution_mode=skill.get("context") or "inline",
                fork_output=None,
            )
        )

    async def seed_builtin_skills(self, *, overwrite: bool = False) -> dict[str, Any]:
        self._enforce("skills.seed.launch.local")
        return {"seeded": [], "count": 0}
