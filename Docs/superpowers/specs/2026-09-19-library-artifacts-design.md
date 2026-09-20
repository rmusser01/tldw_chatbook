# Library Artifacts design

Status: Approved direction; implementation pending
Date: 2026-09-19
Design task: TASK-32869
Baseline: `origin/dev` at `ad0f76e23b8737904f24eb34760bbee9ac01a04c`

## Decision and user choices

Integrate artifact browsing into Library as a collapsible **Artifacts** rail
section, using Library's existing list/reader geometry and interaction language.
Reports open with **All reports** selected and offer a **Kept** filter. Showing a
report in Library does not keep it. Existing manual Keep and scheduled auto-keep
policies retain their current behavior.

Keep the existing Chatbook ZIP-pack manager available through **Manage Chatbook
packs…** in Library. Its inventory and workflows remain independent for this
delivery. Library's Chatbooks list contains registered artifacts, including
Console saved responses; it does not claim to list every ZIP on disk.

These choices were confirmed after reviewing the visual proposal. The review
also found retention, source identity, capability, sharing, and keyboard details
that must be resolved before replacing the old screen.

## Global constraints

- Python ≥3.12; Textual ≥8.0.0,<9; no new dependencies.
- Compose the existing Library adaptive reader and design-token/component patterns.
- UI pages contain at most 20 artifact summaries; bodies load only on selection.
- Artifact browsing is local to the active profile, including when other Library sources use a server.
- Browsing does not change retention, RAG indexing, assistant-tool authority, or publication consent.
- Keep existing global shortcuts, including Ctrl+6, F6, and F4; do not renumber destinations.
- Preserve existing source services, database owners, share controller, and ZIP manager.
- No persisted aggregate artifact database and no schema migration in this delivery.
- Verify with targeted tests and mounted production CSS; do not run a full test sweep without user approval.

## Layout and visual language

The shell is **Library rail → Items → reader**. Use the existing two collapse
grips, pane sizing policy, focus regions, and narrow-terminal behavior from
[ADR-086](../../../backlog/decisions/086-library-adaptive-reader-shell.md).
Artifacts becomes a concrete consumer of `LibraryAdaptiveReaderShell`; the
Media/Notes-specific `LibraryBrowseReaderShell` is not assumed to support it.

The rail keeps existing Browse and other sections. The Artifacts section contains
**All artifacts**, **Chatbooks**, and **Reports**. Its open state persists through
the same rail preference owner as neighboring sections. Do not add empty type
placeholders. Draft is a status and Export is an action. Canvas remains with its
Console conversation/branch owner.

Items has a type title, local-scope label, metadata search, and sort control.
Reports adds **All reports / Kept**. Rows use a title, type/copy label, source,
date, and a small factual status. Reuse Library row density and selection colors.
Counts describe the active filter and admitted inventory; no recent-item caps
masquerade as complete collections. Report totals count copies, explicitly
labeled, because a saved snapshot and a Watchlist copy may both be present.

The reader starts with title, type, retention state, and applicable actions,
followed by **Preview / Details**. Preview is the default. Details contains
provenance, timestamps, generation information, and format metadata; it does not
become a permanent fourth pane. Retention consequences and active sharing stay
visible outside Details.

Use `$ds-*` tokens, component classes, and existing Library typography. Change
source CSS modules and rebuild the bundle; do not hand-edit generated CSS.
No global token-value change is required for this design.

## Reports: retention and identity

All reports includes live Watchlist report copies and durable kept copies.
**Kept** reads `kept_briefings` and `kept_scripts` independently of Subscriptions.
It must work after source deletion and when the Subscriptions service is absent.

| Copy | Primary label and behavior |
| --- | --- |
| Live Watchlist report | **Watchlist copy**; show **Keep in Library** when complete and nonempty. Explain near Keep: deleting the Watchlist removes this copy. |
| Durable snapshot | **Kept in Library**; render its stored body and scripts, even if the original no longer exists. Show the keep date separately from the original date. |
| In-progress, empty, or failed report | Show its actual status; explain why Keep is unavailable. Preserve Watchlists navigation and existing demo recovery where applicable. |

Retain separate identities: `report:live:<id>` and `report:kept:<id>` within the
current profile. Imported `source_briefing_id` values are device-local integers;
matching one does not prove common provenance. The importer preserves original
origin fields, so an `origin=manual` value is not proof either. Do not collapse,
hide, or mark a live report kept based solely on that ID. Display live and kept
copies as distinct rows in this delivery, with clear copy labels. This replaces
the review's initial suggestion to deduplicate by source ID.

A successful Keep selects the returned kept identity and displays the durable
body. The Keep service must check a pre-existing parent's content compatibility
**before any parent or script mutation**, including after a concurrent-create
conflict. If the snapshot differs, refuse the operation and explain that a saved
copy already exists and differs; leave its body and entire script set unchanged.
A UI comparison after calling the current service is too late: the service can
already have attached scripts to that conflicting parent. Preserve additive
script keeping for compatible snapshots and the existing saved origin/time.
Keep this guard with the existing service so every caller receives it; no import
identity or storage migration is required.

Kept bodies never silently become live bodies. Kept scripts remain readable
without the source. Keeping a report does not preserve audio files; playback is
offered only for a live source whose existing path and file checks succeed.
Source navigation is enabled only for a source whose identity is verified; an
imported kept copy with ambiguous origin still opens and exports normally.

## Chatbooks: truthful inventory and actions

The registry includes both Console saved responses and exported bundles. The
existing ZIP manager scans an export directory; these are different inventories.
Show a quiet explanation near **Manage Chatbook packs…**: **Other ZIP packs,
imports, templates, and pack creation are available in the manager.** The link is
available from the Chatbooks empty state and toolbar, including at narrow widths.

| Item | Available behavior |
| --- | --- |
| Console saved response | Preview stored response; show source conversation if resolvable. Honor `content_truncated`; do not present the current 1,000-character preview excerpt as the complete saved body. |
| Registered Chatbook with a valid exported ZIP | Show metadata/manifest preview and existing source/manager actions. Allow the existing ZIP-based share workflow. |
| Registered item without a usable ZIP | Keep metadata readable; explain unavailable sharing and offer the manager. Do not invent a generic Export action for a workflow that does not exist. |

The complete stored Console response is already bounded by its existing 20,000
character save limit. A saved excerpt must be labeled as such. Reading the
registry never imports or executes its content. Rendering continues to use the
existing safe Markdown/content boundaries.

Preserve Create, Import, Templates, pack export/delete, and multi-selection in
their existing manager. Moving those workflows into Library is outside this
delivery, as the user requested.

## Sharing

Keep `ArtifactShareController` application-owned under
[ADR-123](../../../backlog/decisions/123-artifact-share-web-export.md). Preserve the
multi-item share dialog, its existing consent/review controls, path validation,
and publication lifetime. A selected row may seed a compatible bundle; it must
not replace multi-selection.

An active-share strip in Library reads **Sharing N Chatbooks · Manage · Stop**.
It remains reachable across Library sections, filters, selections, and narrow
pane collapse. Re-entering Library reconstructs it from the controller; unmount
does not stop the server. App shutdown and Stop retain their existing behavior.
The strip reports controller status, not selected-item status. Sharing continues
to serve the staged ZIP snapshot, and the UI distinguishes that snapshot from
live registry content. Reports do not gain web publication through this redesign.

## Catalog, paging, and failures

Use one artifact-specific read coordinator over the three existing owners:
registry Chatbooks, live reports, and kept reports. It owns no durable content,
mutations, generic Library sources, or service startup. Source methods apply
metadata filters and deterministic ordering before returning bounded summaries.

Search covers displayed title/source text and existing Chatbook description,
tags, and categories. Label it **Search artifact names and sources**; full report
body search is outside scope. Default sort is **Newest** by original report
creation or Chatbook creation, falling back to keep time only when the imported
original time is unavailable. **Title A–Z** uses a shared normalized title key.
All orderings end with source kind and stable native ID; equal timestamps/titles
must not drop or repeat records.

Use first/previous/next/last keyset pages with exact result ranges, rather than
adding a cross-store offset table. For one operation, each DB owner keeps its
count, boundary probes, and rows in one read transaction; the JSON registry is
read once. The coordinator merges at most 20 candidates per participating
source. It reports an exact total for that tuple of source snapshots, never
claims a globally atomic instant, and releases all reads when the worker ends.
No snapshot survives user idle time. Cross-store writes may move later pages.
The Subscriptions owner explicitly starts a deferred read transaction before its
first target, rank, count, or row query when no transaction is already active.
Its default `transaction()` context does not itself begin a read snapshot.
Follow its existing Watchlists reader pattern; borrow an active transaction
without beginning or committing a nested one, and take no write lock for browse.

The source contracts include count-before-key and bounded rows-before/after-key
reads. Deep links fetch the target's source key, calculate its rank using each
participating source, and fetch at most 19 preceding candidates per source plus
20 following candidates per source. They do not walk every preceding page.
This Artifacts-specific composite cursor contract is recorded in
[ADR-172](../../../backlog/decisions/172-library-artifacts-browse-and-navigation.md)
as a narrow extension to ADR-067. Existing Library source pagers do not change.

When a participating source fails, keep its failure explicit. All artifacts
does not publish a misleading exact total or silently turn the failure into an
empty source. Retain the last good composite page as stale, disable unsafe stale
actions, and offer Retry and working type/filter routes. Chatbooks can work when
reports fail, and Kept can work when live reports fail. A known unconfigured
source is distinct from a failed configured source. Loading is distinct from
empty, no matches, missing item, missing file, and source unavailable.

## Interaction, restoration, and navigation

Reuse Library's bindings and focus system. With the list focused, arrows change
the selected row; Enter opens/focuses the reader. While the reader is focused,
arrows belong to the reader and do not move the list selection. Back, or Escape
when no field/modal/route handler consumes it, returns to the selected list row
and scroll position; arrows then navigate the list again. Escape gives the
focused field, active editor/modal, and route-specific recovery first refusal
before a pane return. Clearing search updates both its visible field and applied
query. Selection must not destroy the focused list or lose its keyboard handlers.

Maintain independent applied query, sort, Kept filter, selected identity, pane
mode, and list scroll for each artifact view. Store lightweight state, not
private records or transient worker failures. Requested and applied query state
remain distinct. Explicit incoming navigation has precedence over restored
state. Separate selected from loaded identity; fence each load by profile,
artifact view, scope, item identity, and generation. Unmount invalidates results.

Library is reusable: leaving it normally suspends it, rather than unmounting it.
Separate retained data reads from presentation effects. Compatible in-flight
reads may finish into retained state while hidden; do not cancel them all or
strand the screen in Loading. Stop visit-owned debounce timers on suspend and
reconcile visible state on resume through Library's existing lifecycle hooks.
Delayed modal/focus publication requires a still-current presentation request,
the same profile and visit, and an active Library screen. Invalidate pending
presentation requests on suspend, canvas navigation, replacement, and teardown;
check again on the UI thread immediately before presenting. Returning to Library
must not resurrect a request abandoned on the previous visit.

Opening Library's own share dialog also suspends Library. Once that dialog is
presented, its explicit result belongs to that exact dialog/profile and is not
invalidated merely by covering its parent. Keep the pre-publication visit guard
separate from accepted dialog results and app-owned share operations. An existing
share and an explicitly approved operation retain their established lifetime.

After Library reaches capability parity, remove Artifacts from the permanent
top-level destination list. Keep `artifacts` as a compatibility route into
Library → Artifacts, and keep **Ctrl+6** as the direct shortcut. Existing exact
Chatbook handoffs still land on the requested item, including after re-creation.
The `chatbooks` route continues to open the actual ZIP manager and highlights
Library as its parent. Do not create a redirect loop or renumber other globals.

Profiles containing only reports or registered Chatbooks must reach the expanded
Library rail. Add an explicit Artifacts evidence source to the existing
six-source onboarding contract and update all callers/tests together. Partial
source failure is unknown evidence, never proof of an empty profile. Do not
create services or seed content merely to answer onboarding evidence.

## Review corrections and verification

The HTML proposal demonstrated the layout only. It is not proof of Textual
behavior. Its review found lost focus after list activation, search clearing
that left the query applied, and a weekly report labeled Daily in Details.
Implementation must test those cases in the real Library shell. Display report
type from available cadence data; use generic **Report** if cadence is unknown.

Required evidence covers: source deletion, absent Subscriptions, imported ID
collision, differing kept/live bodies, script preservation, missing audio/ZIP,
more than 20 rows per source, ties across sources, late workers, filter switch
during load, active-share navigation, old routes/shortcuts, artifact-only
onboarding, no matches, retry, and narrow/wide pane restoration.
Also verify that a conflicting Keep leaves the parent and scripts unchanged in
both ordinary and concurrent-create paths; a writer committing between metadata
and row probes cannot split one browse snapshot; and a delayed dialog cannot
appear after a Library suspend/resume or canvas change. Exercise the explicit
Enter → reader → Back/Escape → list sequence, including consumed field Escape.

Use a harness with production `APP_STYLESHEETS` and a real mounted app for live
checks. Exercise 160×50, 100×40, 64×30, and 50×25 terminals in light/dark themes.
Check painted controls and focus, not just widget existence. New query tests use
real SQLite; worker tests need isolated file-backed DBs because `:memory:` is
per connection. Record commands, outcomes, and screenshots before cutover.

## Delivery and references

Deliver Reports first, then registered Chatbooks and sharing, then the combined
All artifacts view and top-level route cutover. Keep the old Artifacts screen
until the final stage demonstrates parity. The detailed plan is
[here](../plans/2026-09-19-library-artifacts.md).

- [ADR-172](../../../backlog/decisions/172-library-artifacts-browse-and-navigation.md)
- [Design language](../../../backlog/docs/design-language.md)
- [ADR-067: paging](../../../backlog/decisions/067-library-top-level-pagination-contracts.md)
- [ADR-031: keys and footer](../../../backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md)
- [Live verification lessons](../../../backlog/docs/lessons-live-verification.md)
- [Testing evidence lessons](../../../backlog/docs/lessons-testing-evidence.md)
