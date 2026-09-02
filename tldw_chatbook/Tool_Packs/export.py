"""Side-effect-free capture and deterministic serialization of Tool Packs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from io import BytesIO
import re
import unicodedata
import zipfile
from typing import BinaryIO

from tldw_chatbook.MCP.permission_store import (
    BUILTIN_TOOL_SERVER_KEY,
    DEFAULT_GLOBAL,
    GatedToolRef,
    PermissionStoreSnapshot,
    profile_lifecycle_disposition,
    resolve_builtin_state,
    resolve_effective_state,
    resolve_effective_state_by_key,
)
from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.Tool_Packs.catalog_snapshot import (
    PermissionInventoryRegistry,
    PermissionInventorySnapshot,
    PermissionInventoryTool,
    thaw_hub_tool,
)
from tldw_chatbook.Tool_Packs.contracts import (
    PROFILE_PATH,
    TOOL_PACK_SCHEMA,
    TOOL_PROFILE_SCHEMA,
    PortableFallback,
    PortableToolRule,
    ToolPackError,
    ToolPackDocument,
    ToolPackManifest,
    ToolProfilePayload,
    canonical_json_bytes,
)


_RAW_SHELL_NAME = "shell_exec"
_RAW_SHELL_SERVER = "local:__local__"
_SUGGESTED_ID_WORDS = re.compile(r"[^a-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class ToolPackExportSnapshot:
    """The fixed portable document sent to a later publication operation."""

    manifest: ToolPackManifest
    payload: ToolProfilePayload


@dataclass(frozen=True, slots=True)
class ToolPackExportReview:
    """Review data that never includes local configuration or runtime gates."""

    snapshot: ToolPackExportSnapshot
    inventory_digest: str
    excluded_counts: tuple[tuple[str, int], ...]
    omitted_allow_ask: tuple[tuple[str, str], ...]
    pending_denies: tuple[tuple[str, str], ...]

    @property
    def payload(self) -> ToolProfilePayload:
        return self.snapshot.payload


def _safe_suggested_id(source_id: str, profile_id: str) -> str:
    """Slug a source label while ensuring reserved profile names never escape."""
    if profile_id == "default" or profile_id.startswith("ws-"):
        return "tool-profile"
    if type(source_id) is not str:
        raise ToolPackError("export", "profile_invalid")
    normalized = unicodedata.normalize("NFC", source_id).casefold().strip()
    normalized = _SUGGESTED_ID_WORDS.sub("-", normalized).strip(".-_")[:128]
    if not normalized or normalized == "default" or normalized.startswith("ws-"):
        return "tool-profile"
    return normalized


def _state_for_tool(
    payload: Mapping[str, object], item: PermissionInventoryTool, profile_id: str
) -> str:
    tool = thaw_hub_tool(item.tool)
    if item.authority == "builtin":
        state = resolve_builtin_state(
            payload,  # type: ignore[arg-type]
            GatedToolRef(
                server_key=BUILTIN_TOOL_SERVER_KEY,
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema,
                tags=tool.tags,
            ),
            profile_id=profile_id,
        ).state
    else:
        state = resolve_effective_state(
            payload, tool, profile_id=profile_id  # type: ignore[arg-type]
        ).state
    if tool.server_key == _RAW_SHELL_SERVER and tool.name == _RAW_SHELL_NAME:
        return "deny" if state == "deny" else "ask"
    return state


def _unseen_name(payload: Mapping[str, object], server_key: str) -> str:
    candidate = "__tool_pack_unseen__"
    profiles = payload.get("profiles")
    names: set[str] = set()
    if isinstance(profiles, Mapping):
        for profile in profiles.values():
            if isinstance(profile, Mapping):
                servers = profile.get("servers")
                entry = servers.get(server_key) if isinstance(servers, Mapping) else None
                tools = entry.get("tools") if isinstance(entry, Mapping) else None
                if isinstance(tools, Mapping):
                    names.update(name for name in tools if isinstance(name, str))
    suffix = 0
    while candidate in names:
        suffix += 1
        candidate = f"__tool_pack_unseen__{suffix}"
    return candidate


def _fallback_state(
    payload: Mapping[str, object], authority: str, server_key: str, profile_id: str
) -> str:
    if authority == "mcp" and server_key == "*":
        profiles = payload.get("profiles")
        profile_ids = (
            [profile_id] if profile_id == "default" else [profile_id, "default"]
        )
        state = None
        if isinstance(profiles, Mapping):
            for candidate_id in profile_ids:
                profile = profiles.get(candidate_id)
                if isinstance(profile, Mapping) and profile.get("global_default") in {
                    "allow",
                    "ask",
                    "deny",
                }:
                    state = profile["global_default"]
                    break
        state = state or DEFAULT_GLOBAL
    elif authority == "builtin":
        state = resolve_builtin_state(
            payload,  # type: ignore[arg-type]
            GatedToolRef(server_key, _unseen_name(payload, server_key), "", None, ()),
            profile_id=profile_id,
        ).state
    else:
        state = resolve_effective_state(
            payload,  # type: ignore[arg-type]
            _unseen_hub_tool(server_key, _unseen_name(payload, server_key)),
            profile_id=profile_id,
        ).state
    return "ask" if state == "allow" else state


def _unseen_hub_tool(server_key: str, name: str) -> HubTool:
    """Create a neutral unseen-tool definition for a resolver fallback probe."""
    return HubTool(
        server_key=server_key,
        server_label="",
        source="local",
        name=name,
        description="",
        input_schema=None,
        tags=(),
        stale=False,
        executable=False,
    )


def _missing_rules(
    payload: Mapping[str, object],
    profile_id: str,
    live: set[tuple[str, str]],
) -> tuple[list[PortableToolRule], list[tuple[str, str]], list[tuple[str, str]]]:
    profiles = payload.get("profiles")
    if not isinstance(profiles, Mapping):
        return [], [], []
    profile_ids = [profile_id] if profile_id == "default" else [profile_id, "default"]
    pending: list[PortableToolRule] = []
    omitted: list[tuple[str, str]] = []
    pending_ids: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for candidate_id in profile_ids:
        profile = profiles.get(candidate_id)
        if not isinstance(profile, Mapping):
            continue
        servers = profile.get("servers")
        if not isinstance(servers, Mapping):
            continue
        for server_key, entry in servers.items():
            if not isinstance(server_key, str) or not isinstance(entry, Mapping):
                continue
            tools = entry.get("tools")
            if not isinstance(tools, Mapping):
                continue
            for tool_name, rule in tools.items():
                identity = (server_key, tool_name)
                if (
                    not isinstance(tool_name, str)
                    or identity in live
                    or identity in seen
                    or not isinstance(rule, Mapping)
                ):
                    continue
                seen.add(identity)
                state = rule.get("state")
                if state == "deny":
                    authority = "builtin" if server_key == BUILTIN_TOOL_SERVER_KEY else "mcp"
                    if authority == "builtin":
                        effective = resolve_builtin_state(
                            payload,  # type: ignore[arg-type]
                            GatedToolRef(server_key, tool_name, "", None, ()),
                            profile_id=profile_id,
                        ).state
                    else:
                        effective = resolve_effective_state_by_key(
                            payload, server_key, tool_name, profile_id=profile_id  # type: ignore[arg-type]
                        ).state
                    if effective == "deny":
                        pending.append(PortableToolRule(authority, server_key, tool_name, "deny", None))
                        pending_ids.append(identity)
                elif state in {"allow", "ask"}:
                    omitted.append(identity)
    return pending, omitted, pending_ids


def _manifest_for(
    payload: ToolProfilePayload, *, suggested_id: str, display_name: str, producer_name: str, producer_version: str
) -> ToolPackManifest:
    payload_bytes = canonical_json_bytes(payload.to_dict(), operation="export")
    manifest_without_digest: dict[str, object] = {
        "schema": TOOL_PACK_SCHEMA,
        "producer": {"name": producer_name, "version": producer_version},
        "required_features": [],
        "profile": {
            "suggested_id": suggested_id,
            "display_name": display_name,
            "payload": PROFILE_PATH,
        },
        "files": [
            {
                "path": PROFILE_PATH,
                "size": len(payload_bytes),
                "sha256": hashlib.sha256(payload_bytes).hexdigest(),
            }
        ],
    }
    digest = hashlib.sha256(
        TOOL_PACK_SCHEMA.encode("ascii")
        + b"\0"
        + canonical_json_bytes(manifest_without_digest, operation="export")
        + b"\0"
        + payload_bytes
    ).hexdigest()
    manifest = dict(manifest_without_digest)
    manifest["content_digest"] = digest
    return ToolPackManifest.from_dict(manifest, operation="export")


class ToolPackExportService:
    """Captures one strict store and one complete provider inventory per export."""

    def __init__(
        self, permission_store: object, inventory: PermissionInventoryRegistry,
        *, producer_name: str = "tldw_chatbook", producer_version: str = "1"
    ) -> None:
        self._permission_store = permission_store
        self._inventory = inventory
        self._producer_name = producer_name
        self._producer_version = producer_version

    def capture(
        self, *, profile_id: str, display_name: str, suggested_id: str
    ) -> ToolPackExportReview:
        if (
            type(profile_id) is not str
            or not profile_id
            or profile_id != profile_id.strip()
        ):
            raise ToolPackError("export", "profile_invalid")
        try:
            store = self._permission_store.read_snapshot_strict()
        except Exception:
            raise ToolPackError("export", "store_invalid") from None
        if type(store) is not PermissionStoreSnapshot:
            raise ToolPackError("export", "store_invalid")
        payload = store.payload
        profiles = payload.get("profiles")
        profile = profiles.get(profile_id) if isinstance(profiles, Mapping) else None
        if not isinstance(profile, Mapping):
            raise ToolPackError("export", "profile_unavailable")
        if profile_lifecycle_disposition(profile) in {"invalid", "tombstone"}:
            raise ToolPackError("export", "profile_invalid")
        try:
            inventory = self._inventory.capture_for_export()
        except ToolPackError:
            raise
        except Exception:
            raise ToolPackError("export", "inventory_incomplete") from None
        try:
            return self._flatten(
                payload,
                inventory,
                profile_id,
                display_name,
                _safe_suggested_id(suggested_id, profile_id),
            )
        except ToolPackError as error:
            if error.operation == "export":
                raise
            category = "too_large" if error.category == "too_large" else "profile_invalid"
            raise ToolPackError("export", category) from None

    def _flatten(
        self,
        payload: Mapping[str, object],
        inventory: PermissionInventorySnapshot,
        profile_id: str,
        display_name: str,
        suggested_id: str,
    ) -> ToolPackExportReview:
        rules = [
            PortableToolRule(
                item.authority,
                item.tool.server_key,
                item.tool.name,
                _state_for_tool(payload, item, profile_id),
                item.contract_sha256,
            )
            for item in inventory.tools
        ]
        live = {(item.tool.server_key, item.tool.name) for item in inventory.tools}
        pending, omitted, pending_ids = _missing_rules(payload, profile_id, live)
        rules.extend(pending)
        rules.sort(key=lambda item: (item.authority, item.server_key, item.tool_name))
        fallback_ids = {("mcp", "*"), ("builtin", BUILTIN_TOOL_SERVER_KEY)}
        fallback_ids.update((item.authority, item.tool.server_key) for item in inventory.tools)
        fallback_ids.update((item.authority, item.server_key) for item in rules)
        fallbacks = [
            PortableFallback(
                authority, server_key,
                _fallback_state(payload, authority, server_key, profile_id),
            )
            for authority, server_key in sorted(fallback_ids)
        ]
        portable_payload = ToolProfilePayload(TOOL_PROFILE_SCHEMA, tuple(fallbacks), tuple(rules))
        manifest = _manifest_for(
            portable_payload,
            suggested_id=suggested_id,
            display_name=display_name,
            producer_name=self._producer_name,
            producer_version=self._producer_version,
        )
        return ToolPackExportReview(
            ToolPackExportSnapshot(manifest, portable_payload),
            inventory.digest,
            inventory.excluded_counts,
            tuple(sorted(omitted)),
            tuple(sorted(pending_ids)),
        )


def write_tool_pack_archive(snapshot: ToolPackExportSnapshot, sink: BinaryIO) -> str:
    """Write exactly two deterministic ZIP_STORED members and return their digest."""
    if type(snapshot) is not ToolPackExportSnapshot or not hasattr(sink, "write"):
        raise ToolPackError("export", "profile_invalid")
    profile_bytes = canonical_json_bytes(snapshot.payload.to_dict(), operation="export")
    ToolPackDocument.from_dicts(
        snapshot.manifest.to_dict(), snapshot.payload.to_dict(), profile_bytes=profile_bytes,
        operation="export",
    )
    manifest_bytes = canonical_json_bytes(snapshot.manifest.to_dict(), operation="export")
    archive = BytesIO()
    with zipfile.ZipFile(
        archive,
        mode="w",
        compression=zipfile.ZIP_STORED,
        strict_timestamps=True,
    ) as output:
        for name, content in (("tool-pack.json", manifest_bytes), (PROFILE_PATH, profile_bytes)):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.create_version = 20
            info.extract_version = 20
            info.flag_bits = 0
            info.external_attr = 0o100644 << 16
            info.internal_attr = 0
            info.extra = b""
            info.comment = b""
            output.writestr(info, content)
    raw = archive.getvalue()
    offset = 0
    while offset < len(raw):
        try:
            written = sink.write(raw[offset:])
        except Exception:
            raise ToolPackError("export", "publication_failed") from None
        if type(written) is not int or written <= 0 or written > len(raw) - offset:
            raise ToolPackError("export", "publication_failed")
        offset += written
    return hashlib.sha256(raw).hexdigest()

__all__ = [
    "ToolPackExportReview",
    "ToolPackExportService",
    "ToolPackExportSnapshot",
    "write_tool_pack_archive",
]
