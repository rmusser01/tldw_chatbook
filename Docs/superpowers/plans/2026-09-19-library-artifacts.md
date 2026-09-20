# Library Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Follow the current session's delegation authorization; inline execution is sufficient.

**Goal:** Browse reports and registered Chatbooks in Library's established reader layout, with an All reports default, Kept filter, and linked existing ZIP-pack manager.

**Architecture:** Add an artifact-specific read coordinator and concrete Library reader consumer over existing storage/service owners. Keep mutation, sharing, and manager lifetimes intact. Ship Reports, then Chatbooks/sharing, then the composite view and route cutover; do not introduce a persisted aggregate catalog.

**Tech Stack:** Python ≥3.12, Textual 8.x, SQLite, the existing JSON Chatbook registry, pytest, existing TCSS build/governance tools.

**Spec:** [Approved design](../specs/2026-09-19-library-artifacts-design.md)

ADR required: yes
ADR path: backlog/decisions/172-library-artifacts-browse-and-navigation.md
Reason: The navigation move and bounded composite read contract cross long-lived UX and service boundaries. [ADR-172](../../../backlog/decisions/172-library-artifacts-browse-and-navigation.md) records the decision before implementation.

Final-review corrections implement ADR-172's existing content integrity, coherent
read, source ownership, and focus requirements. They require no new ADR and do
not amend the accepted decision. Application implementation is still pending.

Design baseline: `ad0f76e23b8737904f24eb34760bbee9ac01a04c` (`origin/dev`, 2026-09-19).
Design task: TASK-32869. Implementation stages: TASK-32870, TASK-32871, TASK-32872.
Read each Backlog task before starting it; set it In Progress and add its
Implementation Plan before changing code. Record targeted evidence and notes
before checking its ACs and marking it Done. Each stage is independently reviewable.

## Global Constraints

- Python ≥3.12; Textual ≥8.0.0,<9; no new dependencies.
- Compose the existing Library adaptive reader and design-token/component patterns.
- UI pages contain at most 20 artifact summaries; bodies load only on selection.
- Artifact browsing is local to the active profile, including when other Library sources use a server.
- Browsing does not change retention, RAG indexing, assistant-tool authority, or publication consent.
- Keep existing global shortcuts, including Ctrl+6, F6, and F4; do not renumber destinations.
- Preserve existing source services, database owners, share controller, and ZIP manager.
- No persisted aggregate artifact database and no schema migration in this delivery.
- Verify with targeted tests and mounted production CSS; do not run a full test sweep without user approval.

## File boundaries

| File | Responsibility |
| --- | --- |
| Create `tldw_chatbook/Library/library_artifacts_state.py` | Immutable keys, scope, page/detail envelopes, pure copy/action labels, applied/selected/loaded state. |
| Create `tldw_chatbook/Library/library_artifacts_catalog.py` | Artifact-only orchestration of source reads; merge, exact range, bounded target location, source failure. No mutations or widgets. |
| Modify `tldw_chatbook/DB/Subscriptions_DB.py` | Live report metadata count/boundary/window and exact-ID reads. |
| Modify `tldw_chatbook/DB/ChaChaNotes_DB.py` | Kept report metadata count/boundary/window and exact-ID reads independent of live reports. |
| Modify `tldw_chatbook/Subscriptions/briefing_keep.py` | Refuse incompatible existing kept parents before any script mutation, including the concurrent-create fallback; preserve compatible additive keeps. |
| Modify `tldw_chatbook/Chatbooks/local_chatbook_service.py` | One-parse registry snapshot, complete metadata filtering/sorting, bounded candidates; no ZIP inventory merge. |
| Create `tldw_chatbook/UI/Library_Modules/library_artifacts_controller.py` | Workers, scope transitions, selection/detail fences, existing action delegation, return state. |
| Create `tldw_chatbook/Widgets/Library/library_artifacts_reader_shell.py` | Concrete `LibraryAdaptiveReaderShell` consumer with artifact Items and Preview/Details slots. |
| Create `tldw_chatbook/Widgets/Library/library_artifacts_widgets.py` | Artifact rows, toolbar, detail presentation, capability/recovery copy. No DB calls. |
| Modify `tldw_chatbook/UI/Screens/library_screen.py` | Wiring/controller construction and route dispatch only. Keep new domain logic out of this large screen. |
| Modify `tldw_chatbook/Library/library_shell_state.py`, `library_rail_state.py`, `library_content_evidence.py` | Rail section/routes and explicit artifact onboarding evidence. |
| Modify `tldw_chatbook/Widgets/Library/library_rail.py` | Render new section using existing group and open-state patterns. |
| Modify `tldw_chatbook/UI/Library_Modules/library_navigation_controller.py` | Admission and explicit-context precedence for artifact routes. |
| Modify `tldw_chatbook/config.py`, `tldw_chatbook/Utils/adaptive_reader_state.py` | Normalize `[library.artifacts_reader]` via the existing destination preference mechanism. |
| Modify `tldw_chatbook/css/features/_library.tcss`, `_library_panels.tcss` | Token-backed styles; rebuild generated CSS with `python tldw_chatbook/css/build_css.py`. |
| Stage 2: create `tldw_chatbook/UI/Library_Modules/library_artifacts_share_controller.py` | Library presentation/dialog adapter to app-owned sharing; no server ownership. |
| Stage 3: modify `tldw_chatbook/UI/Navigation/shell_destinations.py`, `screen_registry.py`, `pending_handoff_store.py`, `tldw_chatbook/app.py`, `Constants.py` | Compatibility routing and direct shortcut without renumbering globals. |

Existing `artifacts_screen.py`, `artifact_share_dialog.py`, `chatbooks_screen.py`,
`Chatbooks_Window_Improved.py`,
`Subscriptions/briefing_export.py`, and
`UI/Watchlists_Modules/kept_briefings_modal.py` are
capability references. Avoid moving their domain logic into Library. Reuse
existing functions; extract a small shared helper only where both real callers
need it. The ZIP manager implementation is not redesigned.

## Common artifact contracts

Define these in `library_artifacts_state.py`. The names below are the interfaces
used by all three stages, not a generic framework for other Library sources.

```python
from dataclasses import dataclass
from typing import Literal

ArtifactSource = Literal["chatbook", "live_report", "kept_report"]
ArtifactView = Literal["all", "chatbooks", "reports"]
ArtifactSort = Literal["newest", "title"]
ReadDirection = Literal["after", "before"]
# Within one sort, the first field has one type: int for newest, str for title.
ArtifactOrderKey = tuple[int | str, ArtifactSource, int]

@dataclass(frozen=True)
class ArtifactKey:
    source: ArtifactSource
    native_id: int

@dataclass(frozen=True)
class ArtifactScope:
    view: ArtifactView = "reports"
    query: str = ""
    sort: ArtifactSort = "newest"
    kept_only: bool = False

@dataclass(frozen=True)
class ArtifactSummary:
    key: ArtifactKey
    order_key: ArtifactOrderKey
    title: str
    source_label: str
    copy_label: str
    status: str
    type_label: str  # Report unless known cadence justifies a more specific label.
    revision: str  # Owner-derived revision/fingerprint, not body text.

@dataclass(frozen=True)
class ArtifactSourceWindow:
    items: tuple[ArtifactSummary, ...]
    total: int
    before_boundary: int
    equal_boundary: int

@dataclass(frozen=True)
class ArtifactPage:
    scope: ArtifactScope
    items: tuple[ArtifactSummary, ...]
    total: int
    start: int  # Zero-based live rank of the first returned row.

@dataclass(frozen=True)
class ArtifactDetail:
    key: ArtifactKey
    revision: str
    body: str
    truncated: bool
    can_keep: bool
    can_export: bool
    can_play: bool
    can_share: bool
    source_available: bool
    details: tuple[tuple[str, str], ...]  # Safe provenance/format fields for Details.
```

Kept script content is not packed into the page or fetched wholesale into
`ArtifactDetail`. Its reader delegates to the existing kept-script browser and
export path. View-specific pure state adds requested/applied scope, selected key,
loaded key/revision, preview/details mode, scroll, freshness, and generation.

The coordinator constructor receives already-owned dependencies:

```text
LibraryArtifactsCatalog(*, subscriptions_db, chachanotes_db, chatbook_service=None)
```

It exposes:

```text
read_page(scope: ArtifactScope, *, boundary: ArtifactOrderKey | None = None,
          direction: ReadDirection = "after") -> ArtifactPage
locate(scope: ArtifactScope, key: ArtifactKey) -> ArtifactPage | None
read_detail(key: ArtifactKey) -> ArtifactDetail | None
```

These are synchronous read operations executed as exclusive thread workers by
the Library controller. The existing asynchronous registry APIs remain intact;
add a synchronous, read-only snapshot seam for the coordinator rather than
running event loops inside worker threads. The coordinator has no write API.

Each DB owner adds `read_artifact_window(scope, *, boundary, direction, limit, inclusive=False)`
and `get_artifact_summary(scope, key)`. Inputs/outputs use the types above;
`limit` is an integer in 1..20 (not bool). `inclusive` only affects row candidates;
boundary counts always retain their strict-before/equal meanings. With a null
boundary, after uses before/equal counts of zero; before uses before=total and
equal=0. Every call honors an already-open
owner transaction. The registry returns the equivalent results from one parsed
snapshot captured for the coordinator operation. Count and rows use identical
predicates. Reject malformed/duplicate identities and inconsistent envelopes;
do not discard rows silently.

Normalize `scope.query` by trimming; use the same SQLite-compatible ASCII
case-insensitive metadata comparison across all owners. Title sort uses that
same normalization, not Python-only Unicode casefold in one source. Newest
uses negative UTC whole epoch seconds, followed by source and native ID; naive
SQLite timestamps are UTC. Missing imported creation time falls back to kept
time; a missing/unparseable final timestamp sorts last. Test Unicode, timestamp
ties, and missing metadata explicitly. Source kind/ID resolves all final ties.

For every composite request, enter only the participating sources' snapshots in
the order Subscriptions then ChaChaNotes, read one registry snapshot if admitted,
perform all probes, then close before publishing. Kept-only reads never enter
Subscriptions, even if its handle exists. No direct SQL against another owner's
tables, ATTACH, persistent read cache, or long-lived transaction is needed.
Count/row coherence describes these snapshots; it does not imply cross-store
atomicity.

Add the following source-owned `SubscriptionsDB.artifact_read_snapshot()` context
manager. `transaction()` alone only tracks Python nesting on its ordinary read
path; it does not issue SQLite BEGIN. The new boundary must wrap the first
`get_artifact_summary`/rank/count read through the final page query, not just the
last row query. Follow the existing `get_reader_items_page` guarded deferred-BEGIN
pattern, retaining source access/lifecycle guards through `transaction()`.

```python
@_core_transaction
@contextmanager
def artifact_read_snapshot(self) -> Iterator[sqlite3.Connection]:
    connection = self.conn
    if connection.in_transaction:
        depth = getattr(self._local, "transaction_depth", 0)
        self._local.transaction_depth = depth + 1
        try:
            yield connection
        finally:
            self._local.transaction_depth = depth
        return
    with self.transaction() as connection:
        if not connection.in_transaction:
            connection.execute("BEGIN DEFERRED")
        yield connection
```

The active-transaction branch borrows without committing or rolling back the
caller's transaction. Its temporary nesting depth prevents existing source
methods that enter `transaction()` from committing a borrowed native transaction;
restore the original depth on success and exceptions. The inactive branch uses
the existing transaction owner
to finish its deferred snapshot. Both retain the core access/lifecycle guard;
the helper never requests a write lock for browse. ChaChaNotes keeps its existing
explicitly begun transaction boundary.
Add a coordinated WAL writer test following
`Tests/DB/test_subscriptions_db_watchlists_reader_snapshot.py` so a commit
between the first metadata probe and row query cannot change one response's
counts, ranks, or rows. Also assert nested owner transactions remain active and
their owner can still roll back.

## Stage 1 — Reports reader and retention (TASK-32870)

**Deliverable:** Library → Artifacts → Reports works while the old Artifacts
destination remains available. No empty Chatbooks/All placeholders ship yet.

**Files:** Create the state/catalog/controller/reader/widgets modules above and
`Tests/Library/test_library_artifacts_state.py`,
`Tests/Library/test_library_artifacts_catalog.py`,
`Tests/UI/test_library_artifacts_canvas.py`. Modify the two DB owners, rail,
navigation/controller wiring, preference normalization, CSS, and relevant
onboarding tests. Existing report action tests remain as regression coverage.
Also modify `Subscriptions/briefing_keep.py` and extend
`Tests/Subscriptions/test_briefing_keep.py` for the pre-write conflict guard.
The new catalog tests cover the explicit read-snapshot boundary using the
existing Watchlists WAL regression as a fixture pattern.

**Consumes:** Existing `keep_briefing`, live/kept DB reads, report export,
Watchlists navigation, audio validation, and shared adaptive reader.
**Produces:** The common contracts above and a Reports reader accepting
`ArtifactScope(view="reports", kept_only=False)` by default.

- [x] **1. Add the first real-storage regression and run it red.** Use isolated
  file-backed SQLite for worker compatibility. Put this test in the new catalog
  test module (imports are complete below); the new catalog import must fail
  before implementation.

```python
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.briefing_keep import keep_briefing
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService
from tldw_chatbook.Library.library_artifacts_catalog import LibraryArtifactsCatalog
from tldw_chatbook.Library.library_artifacts_state import ArtifactKey, ArtifactScope

def test_kept_is_readable_without_its_source(tmp_path):
    subs = SubscriptionsDB(tmp_path / "subs.db", "artifact-test")
    kept = CharactersRAGDB(tmp_path / "kept.db", client_id="artifact-test")
    try:
        manager = WatchlistBundleService(subs)
        watch = int(manager.create("Weekly digest")["id"])
        live = subs.insert_briefing(watch)
        subs.update_briefing(live, status="complete", body_markdown="# Saved body")
        saved = keep_briefing(subs, kept, live, origin="manual")
        manager.delete(watch)
        assert subs.get_briefing(live) is None
        catalog = LibraryArtifactsCatalog(subscriptions_db=None, chachanotes_db=kept)
        page = catalog.read_page(ArtifactScope(kept_only=True))
        key = ArtifactKey("kept_report", saved["kept_id"])
        assert page.total == 1
        assert [row.key for row in page.items] == [key]
        assert catalog.read_detail(key).body == "# Saved body"
    finally:
        kept.close_connection()
        subs.close()
```

Run: `python -m pytest Tests/Library/test_library_artifacts_catalog.py::test_kept_is_readable_without_its_source -q`.
Expected initially: import failure for the new catalog, then PASS after steps 2–3.

- [x] **2. Implement immutable contracts and source-owned metadata reads.**
  Add parameterized SQL projections that omit body/script/audio-path blobs from
  list rows. Metadata search uses watchlist/snapshot name; count and window share
  predicates. Read kept rows directly, never by enumerating live reports first.
  Full live/kept body reads happen only through `read_detail` on selection.
  Keep all existing DB/service APIs compatible.

- [x] **2a. Guard Keep conflicts in the service before copying any scripts.**
  Keep the `keep_briefing` signature/result shape and existing `KeepRefused`
  error type. Add a private compatibility guard in `briefing_keep.py` and call
  it for both an initially existing parent and the parent returned after a
  concurrent-create `ConflictError`. A mismatch raises `KeepRefused` with a
  conflict reason before `_copy_missing_scripts` or any other mutation. Do not
  rely on a Library-only precheck or a comparison after the service returns.

  Compare the body and stored content fields: normalized original creation and
  coverage timestamps, coverage item ID, selection mode, model, and normalized
  item/featured/overflow counts. Preserve the create path's None/zero handling.
  Do not compare keep time, saved origin, or the mutable Watchlist display name:
  a renamed Watchlist or manual-to-scheduled re-keep must remain compatible.
  Normalize naive SQLite timestamps as UTC and compare instants with the
  timezone-aware kept values. This compatibility check permits additive keeping
  of matching content; it does not establish global report identity or authorize
  browse deduplication. Do not call the importer's private comparator unchanged,
  since it deliberately compares origin and Watchlist name for import semantics.

  Hold one source read snapshot across the live parent/script reads and one
  ChaChaNotes write transaction across parent lookup/validation and script copy,
  in the established Subscriptions → ChaChaNotes order. Use the existing
  `transaction(immediate=True)` write boundary and preserve nested/borrowed owner
  semantics. Recheck compatibility after the create-race fallback; any refusal
  must leave the existing parent and its full script set unchanged. Existing
  compatible re-keep and script-race behavior stays covered by its current tests.

  Add this real-storage regression to `Tests/Subscriptions/test_briefing_keep.py`,
  using that module's existing helpers/imports. It must fail against the current
  service before the guard is implemented.

```python
def test_keep_conflict_does_not_attach_scripts_to_imported_parent(tmp_path):
    subs = _subs_db(tmp_path)
    kept = _chacha_db(tmp_path)
    try:
        live = _complete_briefing(subs, _watchlist(subs), body="# Local report")
        _script(subs, live, preset_name="Local cast")
        saved_id = kept.create_kept_briefing(
            source_briefing_id=live,
            watchlist_name="Imported Watchlist",
            body_markdown="# Different imported report",
            origin="manual",
        )
        kept.create_kept_script(
            saved_id, source_script_id=None, preset_name="Imported cast",
            roster_snapshot_json="[]", turns_json="[]",
        )
        before = (kept.get_kept_briefing(saved_id), kept.list_kept_scripts(saved_id))
        with pytest.raises(KeepRefused, match="conflict"):
            keep_briefing(subs, kept, live, origin="manual")
        after = (kept.get_kept_briefing(saved_id), kept.list_kept_scripts(saved_id))
        assert after == before
    finally:
        kept.close_connection()
        subs.close()
```

  Run: `python -m pytest Tests/Subscriptions/test_briefing_keep.py -q`.
  Extend the existing raced-create fixture so its winning parent has different
  content; assert the same refusal and unchanged parent/scripts on that branch.
  Add compatibility cases for a renamed Watchlist, manual versus scheduled
  origin, and equal instants with different timestamp representations. Existing
  additive-script and complete-script-only tests must still pass.

- [x] **3. Implement bounded merging and direct location.** Both report owners
  participate in All reports; only ChaChaNotes participates in Kept. For a
  boundary, request up to 20 candidates per source, then merge the first 20
  (after) or last 20 (before) in ascending order. Sum exact totals. For after,
  `start = sum(before_boundary + equal_boundary)`; for before,
  `start = max(0, sum(before_boundary) - len(items))`. A null after boundary is
  first; a null before boundary is last. No deduplication of source IDs occurs.

```python
def merge_candidates(windows, *, direction):
    candidates = sorted(
        (row for window in windows for row in window.items),
        key=lambda row: row.order_key,
    )
    return tuple(candidates[:20] if direction == "after" else candidates[-20:])
```

  `locate` obtains the target summary under the active filter, computes
  `rank = sum(count_before(target.order_key))`, and takes `rank % 20` nearest
  predecessors from up to 19 per source. Use the earliest predecessor (or target)
  as an inclusive boundary, then merge up to 20 forward candidates per source.
  Validate the target is present at `rank % 20` and returned `start` is aligned.
  Use `read_artifact_window(..., inclusive=True)` for the forward locate read;
  inclusive location must not alter ordinary strict cursors. Rank counts remain
  strict and the locator validates them before application.

- [x] **4. Add boundary, identity, and failure tests before wiring the UI.**
  Extend the real-DB setup above to 45 live and 45 kept copies, deliberately
  overlapping source IDs. Traverse first/next/previous/last; expect 90 distinct
  namespaced identities, no page over 20, and correct ranges. Locate the final
  kept identity without page-walking; instrument candidate counts, not elapsed
  timing. An imported kept row with the same source ID and different body must
  remain a second row. Filter Kept must make zero Subscriptions calls. Simulate
  live read failure: All reports is unavailable/stale but Kept still works.
  Test tie timestamps, missing original time, Unicode title order, zero matches,
  and a deleted cursor anchor. Empty/out-of-range pages get one first/last
  recovery; a second moving-boundary failure stays stale with Retry.
  Coordinate a separate WAL writer to insert/delete after the first metadata
  read but before the window query. Assert the same response retains its original
  total/rank/rows, the writer commits without waiting for a browse write lock,
  and the next request sees the change. Cover the exact-item locator as well as
  ordinary pages. Verify borrowed native and nested owner transactions remain
  active after both successful reads and exceptions, including nested calls to
  existing `get_briefing`/script methods; only their owner may end them. Release
  coordination events and close each thread's own DB handle in
  test cleanup so a failing assertion cannot leave a writer waiting.

- [x] **5. Build the concrete reader and controller.** Construct
  `LibraryArtifactsReaderShell` from `LibraryAdaptiveReaderShell` with concrete
  list/work builders. Add toolbar/reader IDs prefixed `library-artifacts-`.
  Controller methods are `request_scope(scope)`, `select(key)`,
  `request_page(direction)`, `open_target(key)`, `suspend()`, `resume()`, and
  `dispose()`; all return None
  and schedule/apply existing Textual worker/navigation mechanisms. `dispose`
  invalidates generations. Store separate applied state by view; never persist
  result bodies. Use the current profile identity in every apply fence.

```python
# Apply in the controller after a worker returns; capture this tuple before I/O.
request_identity = (profile_id, scope, selected_key, selected_revision, detail_generation)
# Discard the reply unless the same five values still describe the screen.
# Commit loaded_key/revision only with that verified detail response.
```

  Wire `suspend()`/`resume()` to Library's existing `on_screen_suspend` and
  `on_screen_resume` hooks. Stop owned debounce timers and invalidate pending
  modal/focus presentation requests on suspend. Do not bump data-read generations
  merely because Library is covered: matching reads may finish into retained
  state, consistent with the current reusable-screen contract. Resume reconciles
  the visible surface without replaying abandoned presentation effects; actual
  unmount still invalidates reads through `dispose()`. Keep a separate
  presentation generation and recheck it, the profile, and active screen on the
  UI thread before moving focus or pushing a delayed dialog. Also invalidate
  presentation requests on canvas navigation or a replacing user request.

  Keep invokes the guarded `keep_briefing` service from step 2a in a worker.
  On `KeepRefused` conflict, retain the selected live item and explain the
  refusal without claiming a save or selecting a different body. On success,
  read and select the returned durable identity. A successful write followed by
  a failed refresh remains a
  successful write with stale results and Retry. Live playback stays behind
  the existing path/existence checks. Use the kept-script browser for saved
  scripts; retain report export and Watchlists/demo actions with their owners.

- [x] **6. Integrate rail, preferences, and onboarding atomically.** Add Reports
  to a collapsible Artifacts group. Extend the six-source evidence tuple to an
  explicitly defined seven-source contract, with the seventh value aggregating
  available local artifact owners: any user content wins, all known-empty is
  empty, and incomplete/error reads remain unknown. Update every caller and
  fixed-length assertion together. Empty registry evidence may use an existing
  service handle without constructing one. Add `[library.artifacts_reader]`
  through existing normalized preference code; share Library visibility and use
  destination-specific Items visibility/width. Resize must not reread data.

- [x] **7. Verify interaction with production CSS.** In
  `test_library_artifacts_canvas.py`, use the existing
  `_CssTrueDestinationHarness` pattern from `Tests/UI/test_destination_shells.py`.
  Test list arrows change selection; Enter focuses the reader; reader arrows do
  not change list selection; Back/Escape returns to the same row/scroll; then
  Down advances the list selection. Separately test a focused field consumes
  Escape before pane return. Search `zzzz` then clear it with Escape and verify
  both the field and applied query reset. Hold a detail worker, change
  selection/filter/profile, then release
  it and assert its body/actions cannot apply. Validate buttons paint at 160×50,
  100×40, 64×30, 50×25. Retention is visible without opening Details. Unknown
  cadence is Report, not Daily report. Input handling precedes pane Escape.
  Suspend Library during a delayed data read, let it finish, and resume: retained
  content must settle without stealing focus or staying in Loading. Verify
  stopped debounce timers and stale focus callbacks cannot run against another
  foreground screen. The explicit dialog-publication cases are in stage 2.

- [x] **8. Run targeted checks, review, and commit the releasable stage.**

```bash
python tldw_chatbook/css/build_css.py
python -m pytest Tests/Library/test_library_artifacts_state.py Tests/Library/test_library_artifacts_catalog.py Tests/UI/test_library_artifacts_canvas.py Tests/Library/test_library_content_evidence.py Tests/Widgets/Library/test_library_rail.py Tests/UI/test_artifacts_screen_reports.py Tests/Subscriptions/test_briefing_keep.py Tests/Subscriptions/test_briefing_export_markdown.py Tests/Watchlists/test_kept_briefings_modal.py -q
python -m pytest Tests/UI/test_design_token_governance.py Tests/UI/test_component_pattern_governance.py Tests/UI/test_css_bundle_sync_guard.py Tests/Architecture/test_screen_size_ratchet.py -q
git diff --check
```

  Run the repository-configured formatter/linter on changed Python files, using
  their configured targets. Record a mounted-app check in an isolated profile:
  keep a report, delete its Watchlist, navigate back to Library, filter Kept, and
  read/export the saved body. Record evidence in the task; do not infer native
  UI correctness from headless widget existence. Commit only this stage's files
  as `feat: browse live and kept reports in Library` after review.

## Stage 2 — Registered Chatbooks, manager link, and sharing (TASK-32871)

**Deliverable:** Chatbooks joins the Artifacts rail; the existing manager remains
linked and an active share can be managed/stopped throughout Library.

**Files:** Modify `local_chatbook_service.py`, the artifact modules, rail wiring,
and `library_screen.py`; create the share presentation adapter above and
`Tests/UI/test_library_artifacts_sharing.py`. Extend
`Tests/Chatbooks/test_local_chatbook_service.py`, the new catalog/canvas tests,
and existing share dialog/screen tests where shared helpers change.

**Consumes:** Stage 1 contracts and concrete reader; `ArtifactShareController`,
`ArtifactShareDialog`, `chatbooks` route, existing registry saved-response payload.
**Produces:** `ArtifactScope(view="chatbooks")` reads, capability-derived actions,
and a Library-wide share-status projection independent of selection/view.

- [x] **1. Add pure action-policy regressions before presentation changes.**
  Define `chatbook_actions(*, is_saved_response: bool, usable_zip: bool) ->
  frozenset[str]` in `library_artifacts_state.py` with this complete behavior:

```python
def test_saved_response_does_not_pretend_to_be_an_exported_bundle():
    from tldw_chatbook.Library.library_artifacts_state import chatbook_actions
    assert chatbook_actions(is_saved_response=True, usable_zip=False) == frozenset(
        {"preview", "manage_packs"}
    )
    assert chatbook_actions(is_saved_response=False, usable_zip=True) == frozenset(
        {"preview", "manage_packs", "share"}
    )
```

  Run: `python -m pytest Tests/Library/test_library_artifacts_state.py -q` and
  confirm the new test fails before implementing the function. Preview is
  metadata or saved body; source navigation is resolved separately, not assumed
  by this policy.

- [x] **2. Extend registry reads without changing its inventory.** Add a
  read-only snapshot operation to `LocalChatbookService` that loads the registry
  once and returns a request-owned view exposing window and exact-ID queries.
  Filter/sort all registered records before slicing candidates; do not use
  `list_chatbooks(limit=25)` or `limit=1000` as a complete collection. Preserve
  existing schema/record copies and import/export APIs. Only a selected record
  resolves its full saved content and validates a ZIP for share. Use the stored
  response payload and truncation flag; do not reuse the old 1,000-character
  preview as a complete body. Existing source conversation handoff stays exact.

- [x] **3. Wire the Chatbooks reader and manager link.** Add the Chatbooks rail
  row only once the view works. Mount **Manage Chatbook packs…** in its toolbar
  and empty state; dispatch the existing `chatbooks` route. Display the inventory
  explanation from the spec and validate the manager remains its actual screen.
  Render capability failures inline: missing export, missing file, excerpt,
  unavailable source. Do not add a new pack export action to saved responses.

- [x] **4. Add a Library share adapter, preserving application ownership.**
  `LibraryArtifactsShareController` exposes `open_dialog(preselected_key=None)`,
  `refresh_status()`, `stop_share()`, `suspend()`, `resume()`,
  `invalidate_pending_presentation()`, and `dispose()` (all return None). It uses
  the app's existing controller and share dialog. A pending listing captures the
  profile and presentation generation. Suspend, canvas navigation, a newer open
  request, or disposal invalidates it. Immediately before `push_screen`, check
  that token and `screen.app.screen is screen` on the UI thread; a check in the
  background worker alone leaves a navigation race. A late request must not
  show a dialog or focus/toast effect on another screen, even if Library later
  resumes. None of these presentation guards stop an existing share.

  Record ownership of the exact dialog/profile before pushing it. Pushing that
  modal itself suspends Library; this invalidates pending presentation, not the
  already-presented dialog's explicit result. Consume an accepted result once
  using its dialog identity, captured profile, and existing action authority,
  separately from the now-expired pre-publication visit token. Do not reject an
  ordinary Share confirmation merely because its modal covered Library. Keep
  already-approved share work application-owned and reconcile status on resume.
  The adapter does not replay old dialog opens when returning to Library.

  Enumerate the complete eligible
  registry under the dialog's existing selection contract; eliminate the old
  arbitrary 1,000-record ceiling with owner paging, without changing consent.
  The Library-wide strip is outside the route-owned reader so filters/pane
  collapse cannot hide Stop. Refresh it on mount/resume and controller events;
  do not poll all artifact data to update status.

- [x] **5. Exercise source/capability and sharing regressions.** Seed more than
  25 registry records including a saved response with >1,000 stored characters,
  a response marked truncated, a valid ZIP, and a missing ZIP. Every matching
  registry record must be reachable in deterministic pages; a newer record at
  the old slice tail cannot disappear. Use existing registry test constructors
  and export fixtures rather than hand-writing an invalid registry shape.
  The active-share test selects multiple real exported bundles through
  `ArtifactShareDialog`, navigates Chatbooks → Reports → another Library section,
  collapses panes, and verifies Manage/Stop still act on the same controller.
  Leave/recreate Library and confirm the staged share is still active. Hold the
  dialog-listing worker, navigate to another screen without unmounting Library,
  then release it: no modal or focus change may appear. Repeat with a Library
  canvas change and with leave → return before releasing the old result. Use a
  fresh request after resume to prove cancellation has not stranded the adapter.
  Also verify the normal share modal suspends its parent and still accepts one
  explicit confirmation, while Cancel starts nothing. Late listing replies
  after real unmount remain rejected. Active/approved sharing survives all the
  presentation invalidations above.

- [x] **6. Run targeted checks, live manager/share navigation, and commit.**

```bash
python -m pytest Tests/Chatbooks/test_local_chatbook_service.py Tests/Chatbooks/test_local_chatbook_service_export.py Tests/Library/test_library_artifacts_state.py Tests/Library/test_library_artifacts_catalog.py Tests/UI/test_library_artifacts_canvas.py Tests/UI/test_library_artifacts_sharing.py Tests/UI/test_artifacts_screen_share.py Tests/UI/test_artifact_share_dialog.py -q
git diff --check
```

  Run changed-file lint/format checks. If CSS changed, rebuild and rerun the
  three UI governance tests listed in stage 1. In an isolated running app,
  verify manager Create/Import/Templates remain reachable, preview a saved
  response, start an explicitly reviewed local test share, navigate away/back,
  then stop it. Record the inventory limitation and evidence in TASK-32871.
  Commit as `feat: browse registered Chatbooks and manage sharing in Library`.

## Stage 3 — All artifacts and compatibility cutover (TASK-32872)

**Deliverable:** All artifacts composes the admitted inventory; Library becomes
the permanent browse destination while old routes and Ctrl+6 continue to work.

**Files:** Extend the artifact coordinator/state/reader and tests. Modify
`shell_destinations.py`, `screen_registry.py`, `pending_handoff_store.py`,
`library_navigation_controller.py`, `app.py`, `Constants.py`, and
`Tests/UI/test_shell_destinations.py`, `Tests/UI/test_destination_shells.py`,
`Tests/State/test_pending_handoff_store.py`. Retire old browse composition from
`artifacts_screen.py` only after callers and regression expectations move.

**Consumes:** Stages 1–2 source contracts, working views, capability actions, and
share adapter. Existing `ARTIFACT_CHATBOOK_TARGET` pending handoff is retained.
**Produces:** `ArtifactScope(view="all")`, compatibility route dispatch into
Library, and the complete visual/behavioral design.

- [x] **1. Extend exact composite tests first.** The existing read coordinator
  now admits registry Chatbooks alongside the two report sources. Reuse the
  bounded candidate algorithm; no fourth storage layer. Build a real-source
  fixture with >20 records in each source and timestamps interleaved across all
  three. Assert the union order and total, distinct copy identities, scope-local
  restoration, and exact target location near the end. Search and title sort
  must match the same metadata rules as individual views.

```python
# In the real-source test, `expected` is independently assembled from seed IDs
# and timestamps; it is not obtained by calling the catalog under test.
page = catalog.read_page(ArtifactScope(view="all"))
seen = list(page.items)
while page.start + len(page.items) < page.total:
    page = catalog.read_page(
        page.scope, boundary=page.items[-1].order_key, direction="after"
    )
    assert 0 < len(page.items) <= 20
    seen.extend(page.items)
assert [row.key for row in seen] == expected
```

- [x] **2. Add All artifacts and honest failure recovery.** Its source scope
  is registered Chatbooks plus live and kept report copies. Hide no failed
  source behind a smaller total. A configured-source failure retains the last
  good composite page as stale, hides exact counts and disables stale actions;
  Retry and independent healthy type routes stay available. Distinguish source
  unconfigured from failure. Verify artifact-only profiles reach the expanded
  rail without remote fetches, auto-indexing, or creating source services.

- [x] **3. Write route/shortcut/handoff regressions before changing routes.**
  Keep `artifacts` accepted in saved preferences, command palette, default
  destination, and direct navigation. It resolves once into Library Artifacts;
  Ctrl+6 targets the same view without changing Ctrl+1…other destinations or F4.
  Explicit `local:chatbook:<id>` context must override restored filters and land
  on its containing Chatbooks page. Missing target shows recovery instead of a
  different selected item. Keep pending-handoff single-claim and generation
  semantics. `chatbooks` still opens `ChatbooksScreen` and has Library parent
  navigation; it must not redirect back into the registry view.

- [x] **4. Cut over permanent shell navigation.** Remove only the permanent
  Artifacts destination entry after route parity tests pass. Keep compatibility
  registration/dispatch as a thin adapter, not a second browse implementation.
  Migrate existing report/share tests to assert equivalent Library behavior and
  retain a small legacy-route suite. Footer/F1 hints advertise only implemented
  actions in the actual focus context. Source-specific aliases and manager
  routes must remain distinct.

- [x] **5. Run targeted integration/governance checks and native verification.**

```bash
python tldw_chatbook/css/build_css.py
python -m pytest Tests/Library/test_library_artifacts_state.py Tests/Library/test_library_artifacts_catalog.py Tests/UI/test_library_artifacts_canvas.py Tests/UI/test_library_artifacts_sharing.py Tests/UI/test_shell_destinations.py Tests/UI/test_destination_shells.py Tests/State/test_pending_handoff_store.py Tests/Library/test_library_content_evidence.py Tests/UI/test_library_adaptive_reader_shell.py -q
python -m pytest Tests/UI/test_design_token_governance.py Tests/UI/test_component_pattern_governance.py Tests/UI/test_css_bundle_sync_guard.py Tests/Architecture/test_screen_size_ratchet.py -q
git diff --check
```

  Follow `backlog/docs/lessons-live-verification.md` for an isolated profile and
  real app launch. Inspect light/dark at 160×50, 100×40, 64×30, 50×25; do not alter
  the user's normal config to stage fixtures. Verify Enter/Down, field-first
  Escape, query reset, type switch/restoration, grips, focus on resize, retained
  report after deletion, missing ZIP/audio, retry, manager launch, active-share
  Stop, Ctrl+6, old route, and exact Console-save handoff. Include pending-dialog
  navigation without unmount, resume before a stale reply, and Enter → reader →
  Back/Escape → list. Record images plus
  observed behavior; browser prototype screenshots cannot substitute.

- [x] **6. Finish task documentation and commit.** Confirm the capability table
  below has evidence for every row. Run changed-file lint/format checks, record
  test outcomes/remaining limitations, link ADR-172 in final task notes, and
  update docs that advertise a separate top-level Artifacts destination. Commit
  as `feat: integrate artifact navigation into Library`. Do not mark the task
  Done or remove fallback browse paths while any parity check remains open.

## Capability and review coverage gate

| Requirement | Stage and evidence |
| --- | --- |
| All reports default; Kept independent of source | 1: real storage deletion/absent-source test and mounted filter check |
| Copy identity; imported ID collision; different saved body | 1: service pre-write compatibility guard; ordinary/raced conflict tests assert unchanged parent and scripts |
| Complete scripts, export, live audio, Watchlists/demo handoff | 1: existing domain regressions plus reader action tests |
| At most 20 summaries; complete reachability; exact target | 1 source tests; 3 composite test and target-at-end locator |
| Field Escape, focus after Enter, weekly/generic type | 1: Enter → reader → Back/Escape → list tests; 3: native keyboard check |
| Adaptive pane geometry and lightweight restoration | 1: preferences/canvas; 3: four terminal widths |
| Registry vs ZIP manager; excerpts and capability truth | 2: registry fixtures, manager route, missing ZIP |
| Multi-item share, app lifetime, Manage/Stop reachable | 2: dialog/navigation/late-callback tests and local share check |
| Generation/profile fences; stale counts/actions; Retry | 1 controller; 3 composite source-failure tests |
| Coherent DB read snapshots and borrowed ownership | 1: explicit deferred snapshot and coordinated WAL writer/transaction-ownership tests |
| Suspended Library cannot publish stale dialogs or focus | 1: retained-read lifecycle tests; 2–3: delayed-dialog navigation and ordinary modal-result tests |
| Artifact-only onboarding and local scope | 1 explicit evidence contract; 3 no remote/service-creation test |
| Ctrl+6, old route, exact handoff, manager without alias loop | 3 navigation/pending-handoff regressions |
| No new indexing/tool/publication authority | 1–3 existing owner delegation plus call assertions in integration tests |

Plan self-review: all spec sections map to a stage above. The separate-copy
policy deliberately supersedes the initial review's unsafe source-ID dedup idea.
The final four review corrections are incorporated above: service checks before
mutation, an explicit source-owned read snapshot, presentation guards across
suspend/resume, and one consistent reader/list keyboard sequence. ADR-172's
accepted architecture remains unchanged.
The plan creates no production-code changes by itself. Execute and collect the
listed evidence before claiming the new TUI is implemented or verified.


## Completion evidence — 2026-09-20

All three stages are implemented under ADR-172. See
[the verification record](2026-09-19-library-artifacts-verification.md) for
focused results, native captures, review fixes, sequencing adjustments, and
explicit baseline test limitations. Full-suite execution was not requested.
