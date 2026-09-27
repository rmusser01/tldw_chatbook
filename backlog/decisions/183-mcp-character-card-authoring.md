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

## External character reads, long-field guard, and server mode amendment (2026-09-27)

TASK-32955 publishes the Console's `character_search` and `character_get` to
external clients behind `[mcp] expose_character_tools`, default off and
independent of both `[mcp] expose_local_tools` and the Console's `[tools]
character_tools_enabled`. The standalone server builds them over the real
`LocalCharacterPersonaService` and re-marks only those two specs as external
when composing its provider, so the Hub-local composition and its Console-only
permission rows are unchanged. The reads use the ordinary external local-tool
gate: an explicit Allow runs them and ask is refused. `character_save` stays
Console-only; these two ADR-183 tools remain the only external write path, and
their standing-grant rule above is unchanged.

Each standalone server process owns one `CharacterReadGuard` (stdio serves one
client per process), shared by the external `character_get` and
`update_character`. `update_character` refuses with `read_full_field_first`
when a field it would replace is longer than one `character_get` card page and
that field has not been read contiguously in full at the card's current
version, using the Console's thresholds and message. With the reads unexposed,
long fields therefore cannot be updated externally; short fields are
unaffected. The in-process runtime applies no guard: it has no read tool that
could satisfy one, so a guard there would block every long-field edit from the
Hub. Its calls keep the gates above (the Hub's gated action, or the Console's
approval card, which an explicit tool-level Allow skips), and the Console's own
`character_save` keeps its per-session guard.

In both runtimes `create_character` and `update_character` return
`error_code: "unsupported"` with the Console's local-only message while the
default profile's runtime source is `server`; the external reads return the
same refusal. A failing runtime-source read fails the write closed.

## Alternatives

- Automatically reading the latest version discards optimistic conflict detection.
- Direct database writes duplicate the established service boundary.
- Inheriting unrestricted built-in permissions allows unreviewed persistent card
  replacement. The existing write-tag approval mechanism already handles this.
- Image authoring adds large payload and encoding concerns to this text authoring
  request; reject it explicitly instead of silently dropping it.

No schema, migration, new dependency, or delete/restore tool is introduced.
