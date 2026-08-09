# 43. OPML ↔ watchlist folder mapping

Date: 2026-08-08

## Status

Accepted. Implements the phase-4 slice of the reader-first design
(`Docs/superpowers/specs/2026-08-05-watchlists-reader-first-design.md`);
complements ADR-042 (the re-IA), which deliberately did not cover data
interchange.

## Context

Watchlists OPML import/export (`Subscriptions/watchlist_opml_service.py`)
was flat both ways: the parser discarded folder outlines, and the exporter
serialized one flat outline per source. Every import landed all feeds in
Unassigned, and every export lost watchlist membership — so moving a
subscription list between this app and another reader (or restoring a
backup) silently destroyed the user's structure. Phase 4 maps folders to
watchlists in both directions; the mapping rules are a policy future
contributors will ask about, so they are recorded here rather than in
code comments alone.

## Decision

1. **Folder**: an `<outline>` with no `xmlUrl`/`htmlUrl` that contains feed
   outlines. Its nearest feed descendants map to a watchlist of its name.
2. **Nested folders flatten to the innermost** — the folder directly
   containing the feed wins (single-level readers are the norm; the closest
   ancestor is the most specific user intent).
3. **A feed outline with children is a feed**; its children are evaluated
   under its folder context, never as a folder of the feed's own.
4. **Folder names match watchlists case-insensitively**; a match reuses the
   existing watchlist, otherwise the import creates it. Top-level feeds (no
   folder ancestor) land **Unassigned** — the pre-mapping behavior for every
   feed, preserved for the folderless case.
5. **Export nests deterministically**: one folder outline per watchlist
   (ordered by name) containing member feeds (ordered by name), then
   Unassigned feeds top-level. A source in several watchlists appears under
   each — membership is many-to-many and the document says so faithfully.
6. **Import is additive only**: it never removes sources or memberships.
   Re-importing an exported document is a structural no-op (URL dedupe at
   source creation + name reuse), which is what the round-trip test pins.

## Consequences

- Export → import round-trips watchlist structure losslessly (test-pinned
  as the phase's done-when).
- A hostile folder/watchlist name is inert: ElementTree escapes on
  serialize, the parser surfaces the literal string, and the rail already
  renders names through `escape_markup`.
- Folderless OPML is byte-for-byte the old behavior: flat in, Unassigned
  out, flat export when no watchlists exist.
- Rejected alternatives: outermost-folder wins (less specific intent);
  case-sensitive matching (creates "AI" beside "ai" — a duplicate the user
  cannot tell apart in the rail); destructive sync (import removing
  memberships not present in the document — OPML is an interchange
  document, not a replica; silent deletion on import is the failure mode
  this programme exists to avoid).
