from __future__ import annotations

import json
from typing import Any, Mapping


def render_unified_mcp_section(section: str, payload: Mapping[str, Any] | None) -> str:
    if section == "inventory":
        return render_inventory_section(payload)
    if section == "catalogs":
        return render_catalogs_section(payload)
    if section == "external_servers":
        return render_external_servers_section(payload)
    if section == "governance":
        return render_governance_section(payload)
    if section == "advanced":
        return render_advanced_section(payload)
    return render_overview_section(payload)


def render_overview_section(payload: Mapping[str, Any] | None) -> str:
    payload = dict(payload or {})
    return _render_json_block("Unified MCP Overview", payload)


def render_inventory_section(payload: Mapping[str, Any] | None) -> str:
    payload = dict(payload or {})
    tools = list(payload.get("tools") or [])
    resources = list(payload.get("resources") or [])
    prompts = list(payload.get("prompts") or [])
    lines = [
        "Unified MCP Inventory",
        "",
        f"Server: {payload.get('server_id') or 'local'}",
        f"Scope: {payload.get('selected_scope') or payload.get('scope') or 'personal'}",
        f"Scope Ref: {payload.get('selected_scope_ref') or payload.get('scope_ref') or '-'}",
        "",
        f"Tools ({len(tools)}):",
    ]
    lines.extend(_render_named_items(tools, key="name"))
    lines.append("")
    lines.append(f"Resources ({len(resources)}):")
    lines.extend(_render_named_items(resources, key="uri"))
    lines.append("")
    lines.append(f"Prompts ({len(prompts)}):")
    lines.extend(_render_named_items(prompts, key="name"))
    return "\n".join(lines)


def render_catalogs_section(payload: Mapping[str, Any] | None) -> str:
    payload = dict(payload or {})
    catalogs = list(payload.get("catalogs") or [])
    lines = [
        "Unified MCP Catalogs",
        "",
        f"Server: {payload.get('server_id') or '-'}",
        f"Scope: {payload.get('selected_scope') or payload.get('scope') or 'personal'}",
        f"Scope Ref: {payload.get('selected_scope_ref') or payload.get('scope_ref') or '-'}",
        f"Cache Mode: {payload.get('cache_mode') or 'unknown'}",
        "",
        f"Catalogs ({len(catalogs)}):",
    ]
    lines.extend(_render_named_items(catalogs, key="name"))
    return "\n".join(lines)


def render_external_servers_section(payload: Mapping[str, Any] | None) -> str:
    payload = dict(payload or {})
    profiles = list(payload.get("profiles") or [])
    external_servers = list(payload.get("external_servers") or [])
    items = external_servers or profiles
    key = "name" if external_servers else "profile_id"
    lines = [
        "Unified MCP External Servers",
        "",
        f"Server: {payload.get('server_id') or 'local'}",
        f"Scope: {payload.get('selected_scope') or payload.get('scope') or 'personal'}",
        f"Scope Ref: {payload.get('selected_scope_ref') or payload.get('scope_ref') or '-'}",
        f"Cache Mode: {payload.get('cache_mode') or 'live'}",
        "",
        f"Entries ({len(items)}):",
    ]
    lines.extend(_render_named_items(items, key=key))
    return "\n".join(lines)


def render_governance_section(payload: Mapping[str, Any] | None) -> str:
    payload = dict(payload or {})
    rules = list(payload.get("rules") or [])
    if rules and not any(
        payload.get(key)
        for key in ("permission_profiles", "policy_assignments", "approval_policies", "acp_profiles")
    ):
        lines = [
            "Unified MCP Governance",
            "",
            "Server: local",
            "Scope: personal",
            "Scope Ref: -",
            "Cache Mode: live",
            "",
            f"Rules ({len(rules)}):",
        ]
        if not rules:
            lines.append("  - none")
        else:
            for rule in rules[:10]:
                if isinstance(rule, Mapping):
                    lines.append(
                        f"  - {rule.get('rule_id')}: {rule.get('capability_id')} => {rule.get('decision')}"
                    )
                else:
                    lines.append(f"  - {rule}")
            if len(rules) > 10:
                lines.append(f"  - ... {len(rules) - 10} more")
        return "\n".join(lines)

    permission_profiles = list(payload.get("permission_profiles") or [])
    policy_assignments = list(payload.get("policy_assignments") or [])
    approval_policies = list(payload.get("approval_policies") or [])
    acp_profiles = list(payload.get("acp_profiles") or [])
    effective_policy = dict(payload.get("effective_policy") or {})
    lines = [
        "Unified MCP Governance",
        "",
        f"Server: {payload.get('server_id') or '-'}",
        f"Scope: {payload.get('selected_scope') or payload.get('scope') or 'personal'}",
        f"Scope Ref: {payload.get('selected_scope_ref') or payload.get('scope_ref') or '-'}",
        f"Cache Mode: {payload.get('cache_mode') or 'live'}",
        "",
        f"Effective Policy Enabled: {effective_policy.get('enabled')}",
        f"Effective Approval Mode: {effective_policy.get('approval_mode') or '-'}",
        "",
        f"Permission Profiles ({len(permission_profiles)}):",
    ]
    lines.extend(_render_named_items(permission_profiles, key="name"))
    lines.append("")
    lines.append(f"Policy Assignments ({len(policy_assignments)}):")
    lines.extend(_render_named_items(policy_assignments, key="target_id"))
    lines.append("")
    lines.append(f"Approval Policies ({len(approval_policies)}):")
    lines.extend(_render_named_items(approval_policies, key="name"))
    lines.append("")
    lines.append(f"ACP Profiles ({len(acp_profiles)}):")
    lines.extend(_render_named_items(acp_profiles, key="name"))
    return "\n".join(lines)


def render_advanced_section(payload: Mapping[str, Any] | None) -> str:
    payload = dict(payload or {})
    runtime_status = dict(payload.get("runtime_status") or {})
    if runtime_status:
        protocol = dict(payload.get("protocol") or {})
        request_methods = list(protocol.get("request_methods") or [])
        lines = [
            "Unified MCP Advanced",
            "",
            f"Server: {runtime_status.get('server_id') or 'local'}",
            f"Label: {runtime_status.get('server_label') or 'tldw_chatbook local MCP'}",
            f"MCP SDK Available: {runtime_status.get('mcp_sdk_available')}",
            f"Tools: {runtime_status.get('tool_count')}",
            f"Resources: {runtime_status.get('resource_count')}",
            f"Prompts: {runtime_status.get('prompt_count')}",
            "",
            f"Adapter: {protocol.get('adapter') or 'direct_in_process'}",
            f"Supports Batch: {protocol.get('supports_batch')}",
            "",
            f"Request Methods ({len(request_methods)}):",
        ]
        lines.extend(_render_named_items(request_methods, key="name"))
        return "\n".join(lines)

    tool_registry_summary = dict(payload.get("tool_registry_summary") or {})
    modules = list(tool_registry_summary.get("modules") or [])
    entries = list(tool_registry_summary.get("entries") or [])
    governance_packs = list(payload.get("governance_packs") or [])
    capability_mappings = list(payload.get("capability_mappings") or [])
    path_scope_objects = list(payload.get("path_scope_objects") or [])
    workspace_set_objects = list(payload.get("workspace_set_objects") or [])
    shared_workspaces = list(payload.get("shared_workspaces") or [])
    audit_findings = list((payload.get("governance_audit_findings") or {}).get("items") or [])
    lines = [
        "Unified MCP Advanced",
        "",
        f"Server: {payload.get('server_id') or '-'}",
        f"Scope: {payload.get('selected_scope') or payload.get('scope') or 'personal'}",
        f"Scope Ref: {payload.get('selected_scope_ref') or payload.get('scope_ref') or '-'}",
        f"Cache Mode: {payload.get('cache_mode') or 'stale_allowed'}",
        "",
        f"Registry Modules ({len(modules)}):",
    ]
    lines.extend(_render_named_items(modules, key="module"))
    lines.append("")
    lines.append(f"Registry Entries ({len(entries)}):")
    lines.extend(_render_named_items(entries, key="tool_name"))
    lines.append("")
    lines.append(f"Governance Packs ({len(governance_packs)}):")
    lines.extend(_render_named_items(governance_packs, key="name"))
    lines.append("")
    lines.append(f"Capability Mappings ({len(capability_mappings)}):")
    lines.extend(_render_named_items(capability_mappings, key="mapping_id"))
    lines.append("")
    lines.append(f"Path Scope Objects ({len(path_scope_objects)}):")
    lines.extend(_render_named_items(path_scope_objects, key="name"))
    lines.append("")
    lines.append(f"Workspace Sets ({len(workspace_set_objects)}):")
    lines.extend(_render_named_items(workspace_set_objects, key="name"))
    lines.append("")
    lines.append(f"Shared Workspaces ({len(shared_workspaces)}):")
    lines.extend(_render_named_items(shared_workspaces, key="display_name"))
    lines.append("")
    lines.append(f"Audit Findings ({len(audit_findings)}):")
    lines.extend(_render_named_items(audit_findings, key="message"))
    return "\n".join(lines)


def _render_named_items(items: list[Any], *, key: str) -> list[str]:
    if not items:
        return ["  - none"]
    rendered: list[str] = []
    for item in items[:10]:
        if isinstance(item, Mapping):
            rendered.append(f"  - {item.get(key) or item}")
        else:
            rendered.append(f"  - {getattr(item, key, item)}")
    if len(items) > 10:
        rendered.append(f"  - ... {len(items) - 10} more")
    return rendered


def _render_json_block(title: str, payload: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            title,
            "",
            json.dumps(payload, indent=2, sort_keys=True, default=str),
        ]
    )
