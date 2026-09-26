# ADR-183: MCP character card authoring

Status: Accepted
Date: 2026-09-25
Related: TASK-32952; GitHub #2827; ADR-053

## Decision

Expose `create_character(name, fields)` and
`update_character(character_id, expected_version, fields)` through the existing
standalone MCP server and in-process runtime. Both call
`LocalCharacterPersonaService`, retaining its request validation, transactions,
optimistic locking, FTS, and synchronization semantics. Return bounded receipts
with id, name, and version; include version in the existing character roster.

The first release accepts the service's text and JSON card fields. Reject images,
unknown fields, blank names, empty updates, and invalid ids or versions explicitly.
Omitted fields remain unchanged. Duplicate names and stale versions produce
structured conflict errors. Callers must reread and explicitly choose their next
update; tools never substitute the current version for the caller's version.

These two tools carry code-owned `mutates` risk tags. Both the catalog-backed
permission resolver and the catalog-free by-key resolver apply the existing ask
floor to inherited allow. Explicit tool-level grants remain authoritative.
Standalone calls consult the same permission store and kill switch on every call;
ask refuses with instructions to obtain an operator grant. In-process calls retain
the existing approval and runtime-policy gates. Inventory-supplied risk tags remain
untrusted and cannot change built-in policy.

Character writes use the existing MCP worker-admission helper. It retains the
actual admitted source set independently of the waiting task, so cancellation
and timeout cannot release recovery protection before persistence finishes.
Standalone writes also require current native recovery review for their installed
MCP sources, in addition to the tool permission grant.

## Alternatives

- Automatically reading the latest version discards optimistic conflict detection.
- Direct database writes duplicate the established service boundary.
- Inheriting unrestricted built-in permissions allows unreviewed persistent card
  replacement. The existing write-tag approval mechanism already handles this.
- Image authoring adds large payload and encoding concerns to this text authoring
  request; reject it explicitly instead of silently dropping it.

No schema, migration, new dependency, or delete/restore tool is introduced.
