# Research Workspace

Research is Chatbook's authority-explicit workspace for collecting sources and
taking canonical Quick Notes. It is a separate screen from the durable Research
Runs operator.

## Open Research

- Press **F10** from anywhere to open Research Workspace.
- Choose **F10 Research** in the top navigation bar.
- Open **Ctrl+P** and choose **Tab Navigation: Switch to Research**.

Research has two real screen modes:

- **Workspace** opens the source workbench.
- **Runs** opens the existing Research Runs screen for run lifecycle,
  checkpoints, events, bundles, and artifacts.

Changing modes navigates between screens; it does not embed one in the other.
Saved or programmatic links to the existing `research` route continue to open
Runs.

## Choose the data authority

The pinned **Workspace data: Local | Server** selector chooses the complete
owner used by the workspace catalog, sources, notes, and every mutation on this
screen.

- **Local** uses this installation's Local Library, Local Notes, and Local
  workspace memberships.
- **Server** uses one qualified server profile, principal, and server workspace.

The two authorities are never blended. Selecting Server does not copy Server
Media or notes into Local databases. If the selected server is missing,
unreachable, unauthorized, or unsupported, Server remains selected and the
screen shows recovery. It does not fall back to Local.

The **Processing** status is separate. Local data ownership does not imply that
future inference will run on-device.

## Collapse Sources and Studio

Sources and Studio are independently collapsible around the central Grounded
Chat region. The visible labels are exact ASCII:

| Pane | Collapse | Reveal |
|---|---|---|
| Sources | **`<---`** | **`--->`** |
| Studio | **`--->`** | **`<---`** |

Each control has a full accessible action name. Focus moves to a surviving
reveal control after collapse and into the revealed pane after expansion.
Responsive collapse does not overwrite the stored wide-layout preference.

At medium widths, **Sources**, **Chat**, and **Studio** choose the active
companion layout. At narrow widths they switch the one visible pane.

## Sources

### Add sources

Choose **Add Sources** or use **Quick add URL**. The Add dialog always displays
the captured authority and offers five paths:

- **Import Files** in Local or **Upload** in Server: choose one file for one
  durable receipt.
- **Local Library** or **My Media**: search, page, select, and associate an
  existing canonical catalog item.
- **URL**: add one URL or a batch with one URL per line.
- **Paste**: add titled text through private managed staging.
- **Search Local** or **Search Server**: Local searches the Local Library. The
  Server web-search variant is visibly **Unavailable** because this client has
  no configured canonical web-search result owner; use **My Media** or add a
  result URL through URL intake.

Every intake first creates or reuses an item in the selected authority's
general catalog:

- Local: a normal Library Media item, followed by
  `WorkspaceMembership(role="source")`.
- Server: a normal Server Media item, followed by a server workspace-source
  row.

A completed Server receipt uses the media ID returned by that canonical My
Media result. Chatbook does not mirror the Server ID into Local Library media
or create a Local fallback record.

The qualified workspace target is saved before intake begins. Navigating to a
different workspace while work finishes cannot retarget it. A
`workspace:<name>` keyword may be added for search display, but that tag is not
membership and cannot grant ownership.

### Browse and organize

The Sources pane includes:

- Refresh and current-page text filtering.
- Advanced readiness, type, date, and direct-selection filters.
- Manual, title, and updated-time sorting.
- **Select all**, **Select visible**, **Clear**, and an exact selected count.
- **Preview visible selected** and **Remove visible selected** batch actions.
- Per-source Select/Deselect, details, device folders, Preview/annotate,
  reorder, and Remove controls.
- Previous/Next pagination.

**Move / Copy** remains visible but unavailable until a canonical transfer
owner is implemented. Unsupported owner fields such as URL, file size,
duration, or page count are not guessed.

Remove means **unlink this workspace association**. It also removes that
source from this workspace's selected retrieval scope, but it does not delete
the canonical Local Library or Server Media item. Catalog deletion remains a
separate owner-routed action.

### Selection and readiness

Desired source selection and effective retrieval readiness are different:

- Selection records what should ground future requests and persists even while
  a source is parsing or indexing.
- Readiness reports attached, parsing, indexing, FTS-ready, vector-ready,
  failed, unavailable, or stale.
- Hybrid is effective only when both FTS and vector paths are ready. Missing
  embeddings are labeled FTS-only, never Hybrid/vector ready.

Use **Refresh** to re-read owner state. Failed receipt stages show a
stage-specific action such as **Retry Local/Server ingest**, **Retry workspace
link**, or **Retry readiness**. A catalog success is never rolled back because
association or readiness later fails.

### Receipts, folders, and annotations

Recent receipts independently show catalog, workspace-link, and readiness
outcomes. They remain available after the Add dialog closes and survive
restart. The list is bounded; the general Library operation history remains
the full catalog history.

Nested source folders and source annotations are **Device-only organization**.
They are stored in the private Research overlay keyed by the complete qualified
workspace identity. They are not filesystem roots, canonical memberships,
uploaded server records, shared content, or cross-device state.

## Quick Notes

Quick Notes lives in Studio and supports:

- Load, New, search, Previous/Next, and a canonical note selector.
- Title, comma-separated tags, Markdown Edit/Preview, Save, and Delete.
- Capture selected-source provenance.
- Download as Markdown, Clear, and one-level Undo.

Local notes are normal Local Notes records plus a
`WorkspaceMembership(role="note")`. Server notes use the canonical server
workspace-notes API. Titles, bodies, tags, and provenance never enter the
device overlay or a parallel Research note store.

Updates use the owner's expected version. A stale save never overwrites
silently; choose **Reload**, **Copy as new**, or **Cancel**. Before authority,
workspace, or editor navigation, a non-empty dirty draft is saved to its exact
captured owner. If saving fails, navigation pauses with **Retry**, **Discard
editor changes**, and **Cancel**.

Server Quick Note Delete is visible but disabled because the current server
endpoint does not enforce the supplied expected version. Use a server client
that provides versioned deletion. **Capture message** is also disabled until
Grounded Chat exposes a canonical message owner.

## Current limitations

- Grounded Chat is mounted as a separate work area, but persistent grounded
  messaging belongs to the next implementation phase and Send is not offered.
- Studio output generators are not yet available. Summary, Flashcards, Quiz,
  Report, Compare Sources, and the extended output set remain future phases;
  this screen does not advertise working generation buttons.
- Cross-authority Move/Copy is not implemented in this phase.
- Server web search and version-safe Server Quick Note deletion remain
  unavailable for the reasons shown beside their controls.

## Related settings and screens

- **Settings ▸ Workspaces** remains the full Local workspace manager.
- [Library](library.md) owns global Local catalog browsing, ingestion, and
  editing.
- [Console](console.md) remains the live agent and tool surface.
- [ADR-078](../../backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md)
  records the authority, canonical-owner, overlay, and unlink boundaries.

—
*Verified against `codex/research-workspace` — 2026-08-24. Targeted Research
Workspace verification was run; the full pytest suite was not run.*
