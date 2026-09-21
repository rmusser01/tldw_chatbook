# ADR-172: Browse Artifacts within Library while retaining source owners

Status: Accepted; implementation pending
Date: 2026-09-19
Related Tasks: TASK-32869

## Decision

Library gains an Artifacts section with All artifacts, Chatbooks, and Reports.
It reuses the Library adaptive reader structure, visual tokens, and focus/pane
contracts. After feature parity, the permanent top-level Artifacts destination
is removed while its route and Ctrl+6 shortcut continue into Library. The
`chatbooks` route remains the existing ZIP-pack manager, linked from Library.
The user explicitly chose this staged manager boundary rather than inventory
or workflow unification now.

Reports defaults to All reports and offers a Kept filter. Browsing never changes
the current manual or scheduled retention policy. Durable kept reports are read
independently of their live Watchlist sources, so deleting the source cannot
remove access to the saved content.

Registry Chatbooks, live reports, and kept copies keep their existing service
and storage owners. A Library-local artifact coordinator composes read results;
it owns no persisted aggregate database, source mutations, or generic Library
data framework. No schema migration or dependency is introduced.

The registry and ZIP manager inventories are intentionally distinct. Library
lists registered Chatbooks and saved responses; the manager discovers ZIPs and
retains pack creation/import/templates/export/delete. Actions depend on actual
capabilities and validated files, not the broad type label.

## Copy identity and retention

Live and kept reports have separate namespaced identities and explicit copy
labels. All reports includes both; its totals count report copies. A kept body
is the saved snapshot, never substituted by the latest live body. Scripts kept
with it remain readable; audio is not promised as part of retention.

Do not deduplicate by `source_briefing_id`. Chatbook import preserves that
device-local integer and the original `origin` field; neither proves linkage
to a local live report. The existing importer itself documents this collision.
Keeping distinct copy rows avoids hiding content or falsely asserting that a
different live report is durable. Keep selects the returned durable copy but
must report a differing pre-existing snapshot honestly. Source navigation from
ambiguous imported copies remains unavailable.

Global cross-device report identity and automatic grouping of saved/live copies
are separate storage/provenance decisions and are not required for this move.

## Bounded composite reads

ADR-067 remains authoritative for existing Library sources. Artifacts extends
it narrowly for this fixed composite: first/previous/next/last keyset pages of
at most 20 summaries, exact total/range for the participating read snapshots,
and bounded stable-ID location. No generic pager/controller replaces existing
source implementations.

Each DB owner keeps count and boundary/row queries in one read transaction for
the request; registry queries use one immutable in-memory parse of the existing
JSON registry. The coordinator merges at most one 20-row candidate batch from
each source per page direction. Counts sum safely because copy identities are
distinct. Search and ordering use the same normalized metadata keys in every
source. Ties include source kind and native ID.

An exact-item locator reads the target key, sums source rank counts, and merges
bounded preceding/following candidates to place it on its containing page. It
does not walk earlier pages, load every body, or inject a row into page 1.
Opening transactions is ordered consistently and read-only; the source
snapshot tuple is not a promise of cross-database atomicity. Reads end with the
worker. Pages may move on a later refresh as sources change.

Configured-source errors suppress composite exact totals and preserve a stale
last-good page with Retry. Healthy type/filter routes remain usable. Stale rows
cannot authorize destructive or publish actions. Unknown onboarding evidence
is not empty evidence. All application and profile lifecycle fences continue
to apply.

## Authority and compatibility

Artifacts remains local even when other Library sources use a server. Moving
the browse surface grants no RAG indexing, tool access, or publication authority.
Keep, export, audio, source handoffs, and ZIP operations retain existing owners.
Artifact sharing remains app-owned under ADR-123, with multi-item review and a
Library-wide active-share strip exposing Manage and Stop across navigation.

The old Artifacts screen remains reachable during staged delivery. Remove it
from the permanent shell only after Reports, registered Chatbooks, source
handoffs, share lifetime, empty/error recovery, and global shortcuts pass the
capability inventory in the design. Never alias the ZIP manager back to itself.

## Alternatives considered

| Option | Reason not chosen |
| --- | --- |
| Restyle a permanent standalone Artifacts screen | Duplicates Library browse/reader navigation and leaves related saved material in separate top-level destinations. |
| Merge the ZIP manager and registry immediately | Expands into inventory reconciliation and pack workflow migration; user chose a linked manager first. |
| Show only kept reports | User chose all reports with a Kept filter; hides generated working reports. |
| Automatically keep everything listed | Changes retention and storage without a user action; not implied by Library placement. |
| Deduplicate copies on numeric source ID | Imported IDs can identify different local reports; hides or mislabels content. |
| Persist a unified artifact catalog | Adds ownership, invalidation, and migration work without needing a new source of truth. |
| Page capped recent lists | Makes older items unreachable and counts misleading. |
| Materialize every DB body for search/sort | Violates bounded reading and moves expensive work into a browse refresh. |
| Add an application-wide generic data controller | These three artifact sources do not justify replacing existing Library owners. |

## Consequences

Library becomes the coherent place to browse saved and generated outputs while
Console, Watchlists, and the pack manager retain authoring/management authority.
Some reports appear twice with explicit Watchlist/Kept copy labels; this is an
honest version distinction until stronger provenance supports grouping. The
JSON registry still requires a complete parse under its current storage format;
only its returned summary candidates and mounted UI rows are bounded. SQLite
sources use bounded row queries and body-on-selection reads.

Navigation migration is the last delivery stage. Production-CSS and live TUI
evidence are required; the reviewed browser mockup is visual evidence only.

## Links

- [Design](../../Docs/superpowers/specs/2026-09-19-library-artifacts-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-09-19-library-artifacts.md)
- [ADR-086: adaptive reader](086-library-adaptive-reader-shell.md)
- [ADR-067: source pagination](067-library-top-level-pagination-contracts.md)
- [ADR-123: artifact sharing](123-artifact-share-web-export.md)
- [ADR-031: keys and footer](031-tui-keybinding-and-footer-hint-conventions.md)
