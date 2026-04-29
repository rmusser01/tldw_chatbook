"""Read-only mirror report generation for sync dry runs."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from tldw_chatbook.runtime_policy.server_parity_models import SyncIdentityMapEntry


def _record_by_id(records: Iterable[Mapping[str, Any]], key: str = "id") -> dict[str, Mapping[str, Any]]:
    return {str(record[key]): record for record in records if key in record}


def _identity_to_dict(entry: SyncIdentityMapEntry) -> dict[str, Any]:
    return {
        "domain": entry.domain,
        "source_authority": entry.source_authority,
        "source_scope": entry.source_scope,
        "local_entity_id": entry.local_entity_id,
        "remote_entity_id": entry.remote_entity_id,
        "server_profile_id": entry.server_profile_id,
        "workspace_id": entry.workspace_id,
        "remote_version": entry.remote_version,
        "last_observed_remote_at": entry.last_observed_remote_at,
        "last_local_dirty_at": entry.last_local_dirty_at,
    }


def build_sync_mirror_report(
    *,
    domain: str,
    server_profile_id: str,
    workspace_id: str | None,
    identity_map: Iterable[SyncIdentityMapEntry] = (),
    local_records: Iterable[Mapping[str, Any]] = (),
    remote_records: Iterable[Mapping[str, Any]] = (),
    remote_service: Any = None,
) -> dict[str, Any]:
    """Return a read-only mirror plan; never calls remote mutation methods."""

    local_by_id = _record_by_id(local_records)
    remote_by_id = _record_by_id(remote_records)
    scoped_entries = [
        entry
        for entry in identity_map
        if entry.domain == domain
        and entry.server_profile_id == server_profile_id
        and entry.workspace_id == workspace_id
    ]

    actions: list[dict[str, Any]] = []
    for entry in scoped_entries:
        local_record = local_by_id.get(entry.local_entity_id)
        remote_record = remote_by_id.get(str(entry.remote_entity_id)) if entry.remote_entity_id else None
        actions.append(
            {
                "action": "would_compare",
                "mutation_allowed": False,
                "identity": _identity_to_dict(entry),
                "local_present": local_record is not None,
                "remote_present": remote_record is not None,
                "local_version": local_record.get("version") if local_record else None,
                "remote_version": remote_record.get("version") if remote_record else entry.remote_version,
            }
        )

    return {
        "domain": domain,
        "server_profile_id": server_profile_id,
        "workspace_id": workspace_id,
        "dry_run": True,
        "write_enabled": False,
        "mapped_count": len(scoped_entries),
        "actions": actions,
    }

