"""Pure Workspace transformations from the fixed authenticated native captures."""

import json
from dataclasses import dataclass

from .mobile_identity_plan import _IdentityPlan, _rows

_DEFAULT = "workspace-default"
_TABLES = (
    "workspace_records",
    "workspace_runtime_bindings",
    "workspace_memberships",
    "workspace_rag_scopes",
    "workspace_change_review",
    "workspace_handoff_audit",
    "workspace_agent_backfill",
    "research_source_operations",
    "research_quick_note_receipts",
)


@dataclass(frozen=True)
class _WorkspaceRow:
    table: str
    values: tuple[tuple[str, str | int | float | bytes | None], ...]


@dataclass(frozen=True)
class _WorkspaceBlocker:
    table: str
    identity: str
    workspace_id: str | None
    reason: str
    dependency_domain: str | None = None
    dependency_id: str | int | None = None


class _DependencyUnproven(ValueError):
    def __init__(self, domain, identity):
        super().__init__("workspace_dependency_unproven")
        self.domain = domain
        self.identity = identity


@dataclass(frozen=True)
class _WorkspacePlan:
    identities: _IdentityPlan
    rows: tuple[_WorkspaceRow, ...]
    source_rows: tuple[_WorkspaceRow, ...]
    names: tuple[tuple[str, str, str], ...]
    binding_holds: tuple[str, ...]
    authority_references: tuple
    blockers: tuple[_WorkspaceBlocker, ...]


def _object(value):
    from tldw_chatbook.MCP.permission_store import (
        _reject_duplicate_keys,
        _reject_json_constant,
    )

    result = json.loads(
        value,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_json_constant,
    )
    if type(result) is not dict:
        raise ValueError("object_required")
    return result


def _defaults(reads, value):
    from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults

    if value is None:
        return ()
    payload = _object(value)
    known = {
        "assistant_kind",
        "assistant_id",
        "persona_memory_mode",
        "voice",
        "style",
        "tool_policy_profile_id",
    }
    if set(payload) - known:
        raise ValueError("assistant_defaults_unknown")
    defaults = WorkspaceAssistantDefaults(**payload)
    proofs = []
    for kind, identity in (
        ("persona", defaults.assistant_id),
        ("permission_profile", defaults.tool_policy_profile_id),
    ):
        if identity is not None:
            try:
                proofs.append(reads.authority_reference(kind, identity))
            except ValueError:
                raise _DependencyUnproven(kind, identity) from None
    return tuple(proofs)


def _scope(payload, identities):
    from tldw_chatbook.Chat.rag_scope import parse_scope

    value = _object(payload)
    allowed = {"version", "updated_at", "items", "empty_is_scoped"}
    if (
        set(value) - allowed
        or type(value.get("version")) is not int
        or value["version"] not in (1, 2)
    ):
        raise ValueError("scope_grammar_unproven")
    if parse_scope(value) is None:
        raise ValueError("scope_grammar_unproven")
    items = value["items"]
    for entry in items:
        if (
            type(entry) is not dict
            or set(entry) != {"source_type", "source_id"}
            or type(entry["source_id"]) is not str
        ):
            raise ValueError("scope_grammar_unproven")
        kind, identity = entry["source_type"], entry["source_id"]
        if kind not in {"note", "media"}:
            raise ValueError("scope_grammar_unproven")
        if kind == "media":
            raise _DependencyUnproven(kind, identity)
        try:
            entry["source_id"] = identities.mapped("note", identity)
        except ValueError:
            raise _DependencyUnproven(kind, identity) from None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _plan_workspaces(reads):
    from tldw_chatbook.Workspaces.registry_service import (
        _membership_from_row,
        _runtime_binding_from_row,
        _workspace_from_row,
    )

    from .mobile_captured_sqlite import _CapturedSQLite

    if type(reads) is not _CapturedSQLite:
        raise ValueError("mobile_workspace_native_snapshot_required")
    reads.check()
    identities = reads.identity_plan()
    source = {table: _rows(reads, "legacy", "workspaces", table) for table in _TABLES}
    canonical = {
        table: _rows(reads, "canonical", "workspaces", table) for table in _TABLES
    }
    output, blockers, names, holds, proofs = [], [], [], [], []

    def block(table, identity, workspace, reason, dependency=None):
        blockers.append(
            _WorkspaceBlocker(
                table,
                identity,
                workspace,
                reason,
                dependency.domain
                if isinstance(dependency, _DependencyUnproven)
                else None,
                dependency.identity
                if isinstance(dependency, _DependencyUnproven)
                else None,
            )
        )

    def emit(table, row):
        output.append(_WorkspaceRow(table, tuple(row.items())))

    for row in source["workspace_agent_backfill"]:
        if row["key"] != "assistant_defaults":
            block("workspace_agent_backfill", row["key"], None, "backfill_key_unknown")
    source_complete = any(
        row["key"] == "assistant_defaults" for row in source["workspace_agent_backfill"]
    )
    target_complete = any(
        row["key"] == "assistant_defaults"
        for row in canonical["workspace_agent_backfill"]
    )
    existing = {row["workspace_id"]: row for row in canonical["workspace_records"]}
    occupied_names = {
        row["name"].strip().casefold()
        for row in canonical["workspace_records"]
        if not row["archived"]
    }
    reserved_names = occupied_names | {
        row["name"].strip().casefold()
        for row in source["workspace_records"]
        if not row["archived"]
    }
    for original in sorted(
        source["workspace_records"],
        key=lambda row: (row["created_at"], row["workspace_id"]),
    ):
        row = dict(original)
        identity = row["workspace_id"]
        try:
            model = _workspace_from_row(row)
            proofs.extend(_defaults(reads, row["assistant_defaults"]))
        except (TypeError, ValueError) as error:
            block(
                "workspace_records",
                identity,
                identity,
                "assistant_or_workspace_dependency_unproven",
                error,
            )
            continue
        if identity == _DEFAULT and identity in existing:
            # Desktop timestamps and active selection remain authoritative; every other
            # stored source field has to mean exactly the same reserved Default.
            semantic = set(row) - {"active", "created_at", "updated_at"}
            if any(row[key] != existing[identity][key] for key in semantic):
                block(
                    "workspace_records", identity, identity, "default_content_conflict"
                )
            continue
        if identity == _DEFAULT and (
            model.assistant_defaults is not None or model.archived
        ):
            block("workspace_records", identity, identity, "default_content_conflict")
            continue
        if (
            identity != _DEFAULT
            and not model.archived
            and model.assistant_defaults is None
            and not model.assistant_defaults_explicit_none
            and source_complete != target_complete
        ):
            block("workspace_records", identity, identity, "backfill_frontier_conflict")
        row["workspace_id"] = identities.mapped("workspace", identity)
        row["active"] = 0
        old_name = row["name"]
        if not model.archived:
            name = old_name.strip().casefold()
            if name in occupied_names:
                suffix = 2
                while f"{old_name.strip()} ({suffix})".casefold() in reserved_names:
                    suffix += 1
                row["name"] = f"{old_name.strip()} ({suffix})"
                reserved_names.add(row["name"].casefold())
            occupied_names.add(row["name"].strip().casefold())
        names.append((identity, old_name, row["name"]))
        emit("workspace_records", row)

    for original in source["workspace_runtime_bindings"]:
        row = dict(original)
        identity, workspace = row["binding_id"], row["workspace_id"]
        if workspace == _DEFAULT:
            block(
                "workspace_runtime_bindings",
                identity,
                workspace,
                "default_binding_conflict",
            )
            continue
        try:
            _object(row["metadata_json"])
            _runtime_binding_from_row(row)
            row["binding_id"] = identities.mapped("binding", identity)
            row["workspace_id"] = identities.mapped("workspace", workspace)
        except (TypeError, ValueError):
            block(
                "workspace_runtime_bindings",
                identity,
                workspace,
                "binding_unrepresentable",
            )
            continue
        # Exact locator/metadata/status survive; later owner writes persist the hold
        # in the same transaction. No projected ready binding is activated here.
        holds.append(row["binding_id"])
        emit("workspace_runtime_bindings", row)

    member_keys = {
        (row["workspace_id"], row["item_type"], row["item_id"], row["role"])
        for row in canonical["workspace_memberships"]
    }
    for original in source["workspace_memberships"]:
        row = dict(original)
        identity, workspace = row["membership_id"], row["workspace_id"]
        try:
            _membership_from_row(row)
            domain = {"note": "note", "conversation": "conversation"}.get(
                row["item_type"]
            )
            if domain is None:
                raise _DependencyUnproven(row["item_type"], row["item_id"])
            row["workspace_id"] = identities.mapped("workspace", workspace)
            try:
                row["item_id"] = identities.mapped(domain, row["item_id"])
            except ValueError:
                raise _DependencyUnproven(domain, row["item_id"]) from None
            row["membership_id"] = identities.mapped("membership", identity)
            key = (row["workspace_id"], row["item_type"], row["item_id"], row["role"])
            if key in member_keys:
                raise ValueError("membership_collision")
            member_keys.add(key)
        except (TypeError, ValueError) as error:
            block(
                "workspace_memberships",
                identity,
                workspace,
                "membership_dependency_unproven",
                error,
            )
            continue
        emit("workspace_memberships", row)

    for table in ("workspace_rag_scopes", "workspace_change_review"):
        target_by_workspace = {row["workspace_id"]: row for row in canonical[table]}
        for original in source[table]:
            row = dict(original)
            workspace = row["workspace_id"]
            try:
                row["workspace_id"] = identities.mapped("workspace", workspace)
                if table == "workspace_rag_scopes":
                    row["payload"] = _scope(row["payload"], identities)
                if workspace == _DEFAULT and workspace in target_by_workspace:
                    other = target_by_workspace[workspace]
                    if table == "workspace_rag_scopes":
                        equal = _object(row["payload"]) == _object(other["payload"])
                    else:
                        equal = row["enabled"] == other["enabled"]
                    if not equal:
                        raise ValueError("default_setting_conflict")
                    continue
            except (TypeError, ValueError) as error:
                block(
                    table,
                    workspace,
                    workspace,
                    "scope_or_default_dependency_unproven",
                    error,
                )
                continue
            emit(table, row)

    for original in source["workspace_handoff_audit"]:
        row = dict(original)
        identity, workspace = row["audit_id"], row["workspace_id"]
        try:
            if _object(row["manifest_json"]):
                raise ValueError("audit_manifest_unsupported")
            row["audit_id"] = identities.mapped("workspace_audit", identity)
            row["workspace_id"] = identities.mapped("workspace", workspace)
        except (TypeError, ValueError):
            block(
                "workspace_handoff_audit",
                identity,
                workspace,
                "audit_manifest_unsupported",
            )
            continue
        emit("workspace_handoff_audit", row)
    for table, key in (
        ("research_source_operations", "operation_id"),
        ("research_quick_note_receipts", "receipt_id"),
    ):
        for row in source[table]:
            block(
                table,
                row[key],
                row["workspace_id"],
                "external_protocol_obligation_unsupported",
            )
    reads.check()
    return _WorkspacePlan(
        identities,
        tuple(output),
        tuple(
            _WorkspaceRow(table, tuple(row.items()))
            for table in _TABLES
            for row in source[table]
        ),
        tuple(names),
        tuple(holds),
        tuple(proofs),
        tuple(blockers),
    )
