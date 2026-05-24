"""Serializable payloads for staging source context into Chat."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .answer_citations import format_evidence_for_cited_answer


SECRET_CONTRACT_KEYS = frozenset({"credential_source", "token", "secret", "api_key", "password"})
HANDOFF_BODY_CHAR_LIMIT = 80_000


@dataclass
class ChatHandoffPayload:
    source: str
    item_type: str
    title: str
    body: str
    body_truncated: bool = False
    content_ref: Optional[str] = None
    source_id: Optional[str] = None
    display_summary: str = ""
    suggested_prompt: str = ""
    runtime_backend: str = "local"
    source_owner: str = "local"
    source_selector_state: str = "local"
    active_server_profile_id: Optional[str] = None
    discovery_owner: str = "general_chat"
    discovery_entity_id: Optional[str] = None
    scope_type: Optional[str] = None
    workspace_id: Optional[str] = None
    backend_contracts: Dict[str, Any] = field(default_factory=dict)
    unsupported_reports: list[Dict[str, Any]] = field(default_factory=list)
    sync_dry_run_report: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "staged"

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "source": self.source,
            "item_type": self.item_type,
            "title": self.title,
            "body": self.body,
            "body_truncated": self.body_truncated,
            "content_ref": self.content_ref,
            "source_id": self.source_id,
            "display_summary": self.display_summary,
            "suggested_prompt": self.suggested_prompt,
            "runtime_backend": self.runtime_backend,
            "source_owner": self.source_owner,
            "source_selector_state": self.source_selector_state,
            "active_server_profile_id": self.active_server_profile_id,
            "discovery_owner": self.discovery_owner,
            "discovery_entity_id": self.discovery_entity_id,
            "scope_type": self.scope_type,
            "workspace_id": self.workspace_id,
            "backend_contracts": _json_safe_contract_snapshot(self.backend_contracts or {}),
            "unsupported_reports": _json_safe_contract_snapshot(self.unsupported_reports or []),
            "sync_dry_run_report": (
                _json_safe_contract_snapshot(self.sync_dry_run_report)
                if self.sync_dry_run_report
                else None
            ),
            "metadata": _json_safe_contract_snapshot(self.metadata or {}),
            "status": self.status,
        }
        return {key: value for key, value in data.items() if value is not None}

    @classmethod
    def from_source_content(
        cls,
        *,
        body: str,
        content_ref: Optional[str] = None,
        **kwargs: Any,
    ) -> "ChatHandoffPayload":
        body_text = str(body or "")
        upstream_truncated = bool(kwargs.pop("body_truncated", False))
        body_truncated = upstream_truncated or len(body_text) > HANDOFF_BODY_CHAR_LIMIT
        if body_truncated:
            body_text = body_text[:HANDOFF_BODY_CHAR_LIMIT]
        return cls(
            body=body_text,
            body_truncated=body_truncated,
            content_ref=content_ref,
            **kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | "ChatHandoffPayload" | None) -> Optional["ChatHandoffPayload"]:
        if data is None:
            return None
        if isinstance(data, cls):
            return cls(**data.to_dict())
        if not isinstance(data, Mapping):
            return None
        return cls(
            source=str(data.get("source") or "unknown"),
            item_type=str(data.get("item_type") or "item"),
            title=str(data.get("title") or "Untitled"),
            body=str(data.get("body") or ""),
            body_truncated=bool(data.get("body_truncated", False)),
            content_ref=data.get("content_ref"),
            source_id=data.get("source_id"),
            display_summary=str(data.get("display_summary") or ""),
            suggested_prompt=str(data.get("suggested_prompt") or ""),
            runtime_backend=str(data.get("runtime_backend") or "local"),
            source_owner=str(data.get("source_owner") or "local"),
            source_selector_state=str(
                data.get("source_selector_state") or data.get("runtime_backend") or "local"
            ),
            active_server_profile_id=data.get("active_server_profile_id"),
            discovery_owner=str(data.get("discovery_owner") or "general_chat"),
            discovery_entity_id=data.get("discovery_entity_id"),
            scope_type=data.get("scope_type"),
            workspace_id=data.get("workspace_id"),
            backend_contracts=_json_safe_contract_snapshot(data.get("backend_contracts") or {}),
            unsupported_reports=_json_safe_contract_snapshot(data.get("unsupported_reports") or []),
            sync_dry_run_report=(
                _json_safe_contract_snapshot(data["sync_dry_run_report"])
                if data.get("sync_dry_run_report")
                else None
            ),
            metadata=_json_safe_contract_snapshot(data.get("metadata") or {}),
            status=str(data.get("status") or "staged"),
        )

    def default_prompt(self) -> str:
        return self.suggested_prompt.strip() or "Help me use this context."

    def model_context_block(self) -> str:
        metadata_lines = []
        metadata_snapshot = _json_safe_contract_snapshot(self.metadata or {})
        for key, value in sorted(metadata_snapshot.items()):
            if key == "evidence_bundle":
                continue
            if value not in (None, ""):
                metadata_lines.append(f"- {key}: {value}")
        metadata = "\n".join(metadata_lines)
        evidence = format_evidence_for_cited_answer(metadata_snapshot.get("evidence_bundle"))
        return (
            "[Staged context]\n"
            f"Source: {self.source}\n"
            f"Item type: {self.item_type}\n"
            f"Title: {self.title}\n"
            f"Source ID: {self.source_id or 'unknown'}\n"
            f"Content ref: {self.content_ref or 'none'}\n"
            f"Body truncated: {self.body_truncated}\n"
            f"Source owner: {self.source_owner}\n"
            f"Source selector: {self.source_selector_state}\n"
            f"Active server: {self.active_server_profile_id or 'none'}\n"
            f"Workspace: {self.workspace_id or 'none'}\n"
            f"Sync dry-run only: {bool(self.sync_dry_run_report)}\n"
            f"Summary: {self.display_summary or 'none'}\n"
            f"Metadata:\n{metadata or '- none'}\n\n"
            f"{evidence}"
            f"Content:\n{self.body}"
        )

    def format_for_model(self, user_prompt: str) -> str:
        return f"{self.model_context_block()}\n\n[User prompt]\n{user_prompt.strip()}"


def _json_safe_contract_snapshot(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe_contract_snapshot(item)
            for key, item in value.items()
            if item is not None and not _is_secret_contract_key(str(key))
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_contract_snapshot(item) for item in value if item is not None]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_payload"):
        return _json_safe_contract_snapshot(value.to_payload())
    return str(value)


def _is_secret_contract_key(key: str) -> bool:
    normalized = key.lower()
    return any(secret_key in normalized for secret_key in SECRET_CONTRACT_KEYS)
