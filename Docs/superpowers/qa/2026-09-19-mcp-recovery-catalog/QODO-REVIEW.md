# Qodo review dispositions

Qodo reviewed `e57b3f6dc731cb4b2b2070d9089bdfb42e72f0c6`: zero bugs,
two rule findings. This follow-up changes tests and documentation only.
The product and native runner bytes remain those qualified by the current gallery.

## Isolated unit coverage — addressed

`Tests/Backup_Recovery/test_mcp_recovery_catalog_unit.py` calls the actual
`MCPWorkbench._record_mcp_recovery_review` method with controlled service and
inspector doubles, without mounting a Textual app. Cases cover empty/successful
conversion, copied records, built-in readiness, malformed service returns,
reader exceptions, retry privacy, and successful/failed reads completing after
their token was replaced. The stale cases require later catalog/snapshot objects
and the successor token to remain intact, with no inspector render or notification.
The existing actual-owner mounted journeys remain the integration coverage.

## Additional Pydantic schema — not needed for the reported input

The reported example was a non-mapping `env_placeholders` value causing the UI
to discard valid sibling servers. The production path already normalizes that
field before this handler:

1. `LocalMCPStore.get_external_catalog()` loads `LocalMCPStoreState.from_dict()`.
2. `LocalExternalMCPProfile.from_storage_dict()` passes the field through
   `_coerce_mapping`, turning non-mappings into `{}`. The dataclass's
   `__post_init__` sanitizes its contents; `to_dict()` returns a dictionary.
3. `LocalMCPControlService.get_external_servers()` emits those profile dictionaries.
   `UnifiedMCPControlPlaneService.local_external_catalog()` adds runtime state.
4. The UI consumes that existing passive service result.

Four real store/service cases write the reported malformed field (string, list,
integer and null) beside a valid sibling. Both profiles remain in the actual
catalog; the malformed mapping becomes `{}`, the sibling's `$TOKEN` placeholder
survives, readiness construction succeeds, and no connection/discovery/execution
operation occurs. Independent review confirmed this interpretation.

This is a narrow finding disposition, not a claim that every nested catalog
field has strict validation. Invalid placeholder *contents* can still fail at
the existing storage boundary; another UI schema would not repair that upstream
failure. A deliberately broken service double is separately covered by the safe
refresh-failure tests. Duplicating the established internal profile schema in the
UI would add a new validation policy without fixing the reported scenario.

[Focused test receipt](integration/tests/qodo-001.txt) records 20 passing cases:
14 new isolated/storage cases and the six existing actual-owner catalog journeys.
Together with the previous unchanged-source qualification, 182 distinct cases pass.
No new ADR or product change is required; existing ADR-126 applies.
