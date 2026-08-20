# TASK-16483 Library Media Pagination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Drive every production slice through focused RED/GREEN tests and preserve unrelated worktree changes.

**Goal:** Make every active local Media item reachable through truthful 20-item Library pages, expose the complete Media type set in one bounded keyboard-accessible chooser, and keep paging, mutations, focus, restoration, and diagnostics safe under failures and races.

**Architecture:** Keep Media ownership source-specific. Extend the existing Media DB and reading services with true limit/offset paging, a coherent exact count, and a thin adapter over the existing complete distinct-type query. Add a small Media-specific browse controller beside the existing Media state so requested scope, last applied page, retained stale rows, request generation, and pager truth have one owner. `LibraryScreen` continues to own navigation, detail, selection, mutation, and focus; `LibraryMediaCanvas` continues to own Media rendering and receives the source-owned pager/facet state. Do not create a polymorphic Library pager controller, add Media search/sort controls, or paginate Media Trash.

**Tech Stack:** Python 3.12, Textual 8.x, SQLite, immutable dataclasses, pytest/pytest-asyncio, Ruff, existing TCSS bundle tooling.

**ADR required:** no new ADR  
**ADR path:** `backlog/decisions/067-library-top-level-pagination-contracts.md`  
**Reason:** This task directly implements ADR-067's approved Media paging, complete-facet, stale-recovery, and privacy contracts without changing storage schema, runtime, or cross-source ownership.

---

## Scope and ownership

**Production owners:**

- `tldw_chatbook/DB/Client_Media_DB_v2.py` — coherent exact count/page read, true offset execution, complete distinct types, stable ordering, and metadata-only diagnostics.
- `tldw_chatbook/Media/local_media_reading_service.py` — explicit local limit/offset and complete type-facet adapter.
- `tldw_chatbook/Media/media_reading_scope_service.py` — existing mode boundary; normalize the local page/type results without changing unrelated server contracts.
- `tldw_chatbook/Library/library_media_state.py` — exact Media page scope/result/identity validation and pure canvas projection.
- `tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py` — new Media-specific requested/applied/retained/freshness/generation owner; no generic controller.
- `tldw_chatbook/UI/Screens/library_screen.py` — existing Media navigation, restore, selection, detail, focus, request, and mutation owner.
- `tldw_chatbook/Widgets/Library/library_media_canvas.py` — Media pager, complete type `OptionList`, rows, Retry, disabled reasons, and selection notice.
- `tldw_chatbook/css/components/_agentic_terminal.tcss` and generated bundle only if production-hierarchy geometry proves a CSS change is needed.

**Focused test owners:**

- `Tests/DB/test_client_media_pagination.py` (new focused DB contract tests) and `Tests/DB/test_client_media_debug_logging.py`.
- `Tests/Media/test_local_media_reading_service.py` and `Tests/Media/test_media_reading_scope_service.py`.
- `Tests/Library/test_library_media_state.py` and `Tests/UI/test_library_media_browse_controller.py` (new controller tests).
- `Tests/UI/test_library_shell.py`, `Tests/UI/test_library_multiselect_media.py`, `Tests/UI/test_library_media_side_by_side.py`, and the existing Media choice/canvas tests that are changed by the `OptionList` contract.

**Out of scope:** Media Trash paging (TASK-16487), Collection member paging, new Media search or sort controls, server API redesign, a generic Library controller/widget, persistent row caches, or unrelated Media Window paging. The pure/service scope still carries the established query/sort fields so existing callers and recovery remain correct; only page and type have mounted top-level controls today.

**Verification boundary:** Per explicit user direction, do not run the repository-wide test suite. Final automated evidence is limited to the modified Media DB/service/state/controller/canvas/screen components and their direct owner/regression tests, plus Ruff, bundle parity when touched, diff checks, and isolated live verification.

---

### Task 0: Freeze the task boundary and focused baseline

**Files:**

- Modify: `backlog/tasks/task-16483 - Add-authoritative-paging-and-complete-facets-to-Library-Media.md`
- Add: `Docs/superpowers/plans/2026-08-16-task-16483-library-media-pagination.md`

- [ ] Record the current `origin/dev` base SHA and confirm the worktree contains only the task/plan changes.
- [ ] Link ADR-067 and this plan from TASK-16483. Record that no new ADR is required.
- [ ] Run only the current direct Media owners to establish a focused baseline; save exact node names for any pre-existing failures.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/DB/test_pagination.py \
  Tests/DB/test_client_media_debug_logging.py \
  Tests/Media/test_local_media_reading_service.py \
  Tests/Media/test_media_reading_scope_service.py \
  Tests/Library/test_library_media_state.py

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_multiselect_media.py \
  Tests/UI/test_library_media_side_by_side.py \
  Tests/UI/test_library_shell.py -k 'library and media'
```

- [ ] Commit the approved plan/task metadata before production work.

```bash
git add "backlog/tasks/task-16483 - Add-authoritative-paging-and-complete-facets-to-Library-Media.md" \
  Docs/superpowers/plans/2026-08-16-task-16483-library-media-pagination.md
git commit -m "docs(library): plan media pagination"
```

---

### Task 1: Page Media coherently at the database boundary

**Files:**

- Modify: `tldw_chatbook/DB/Client_Media_DB_v2.py`
- Add: `Tests/DB/test_client_media_pagination.py`
- Modify: `Tests/DB/test_client_media_debug_logging.py`

- [ ] **Write RED database tests** using a real temporary SQLite database with at least 45 active items, repeated primary sort values, multiple types, deleted/trashed decoys, and unique privacy sentinels. Prove:
  - pages at offsets `0`, `20`, and `40` return `20`, `20`, and `5` rows with exact total `45`;
  - query/type filters apply before count and paging;
  - every supported Library sort ends with stable Media ID ordering;
  - Library DB page rows contain exactly `id`, `title`, `type`, and `last_modified`, and never content, embeddings, binary/blob fields, filesystem paths, provenance payloads, client identifiers, or other detail-only fields;
  - only the requested page is read (SQLite trace/progress evidence rejects prefix fetching);
  - count and rows share one read transaction under coordinated WAL insert/delete;
  - non-bool integer `limit`/`offset` validation rejects negatives, zero limit, and values beyond SQLite's signed 64-bit boundary before SQL;
  - each DB/local summary row contains exactly `id: int`, where the value is positive, non-bool, and page-unique; scope/Library tests separately require normalized `id: "local:media:<n>"` plus `backing_media_id: int` and reject missing/`None`/bool/non-integer/mismatched/duplicate identities;
  - `get_distinct_media_types` returns the complete active filtered set, not only page-visible types.

- [ ] **Run RED** on the new DB tests and touched logging tests.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/DB/test_client_media_pagination.py \
  Tests/DB/test_client_media_debug_logging.py
```

Expected RED: the current page-only API cannot express an exact offset independently, count/page can observe different snapshots, and diagnostics expose raw query/title/ID/path values.

- [ ] **Implement the minimum compatible DB seam.** Preserve existing callers by adding optional keywords `offset: int | None = None` and `library_summary: bool = False`: omitted offset retains legacy page coordinates, supplied offset uses the exact arbitrary coordinate, and generic callers keep their broad row shape. With `library_summary=True`, the DB SELECT allowlist is exactly `id`, `title`, `type`, and `last_modified`; `id` remains the raw positive integer. Never read detail-only/private columns and strip them later. Build the WHERE clause and parameters once; execute count and page rows through the same read transaction; keep current selected sort plus `m.id` as the final tie-break. Do not fetch a prefix or hold a browse-session snapshot. Direct transaction-owned `conn.execute` also avoids the shared parameter-preview logger. The existing transaction context logs exception strings, so catch raw SQLite exceptions *inside* its body and convert them to fixed metadata-only project errors (without cause traceback) before they reach that logger, or use an equally small read-transaction helper that never logs raw exception text. If FTS execution fails, either preserve a privacy-safe fallback with the identical complete filter predicate or fail closed; do not retain the current fallback that silently drops scope filters.
- [ ] **Make touched diagnostics metadata-only.** Capture both Loguru and stdlib handlers on success and forced error. Remove query text, title/body, stable private ID, SQL parameters, database path, credentials, raw exception/path text, cause tracebacks, and `exception=True` from the touched path. Keep bounded metadata such as operation, mode, limit, offset, result count, total, sort key, and exception class.
- [ ] **Run GREEN and inverse mutations.** Rerun the two focused files. Separately restore split count/page reads and remove the stable-ID tie-break; require the coordinated snapshot and deterministic-order tests to fail, then restore.
- [ ] Commit this slice.

```bash
git add tldw_chatbook/DB/Client_Media_DB_v2.py \
  Tests/DB/test_client_media_pagination.py \
  Tests/DB/test_client_media_debug_logging.py
git commit -m "fix(library): page media at the database"
```

---

### Task 2: Propagate true offsets and complete type facets through Media services

**Files:**

- Modify: `tldw_chatbook/Media/local_media_reading_service.py`
- Modify: `tldw_chatbook/Media/media_reading_scope_service.py`
- Test: `Tests/Media/test_local_media_reading_service.py`
- Test: `Tests/Media/test_media_reading_scope_service.py`
- Test: `Tests/Media/test_media_reading_scope_service_off_loop.py` only if the new local call affects its thread-boundary contract.

- [ ] **Write RED service tests** proving:
  - `limit=20, offset=40` reaches the DB as an exact 20-row read at offset 40, never `results_per_page=60` from page 1;
  - with `library_summary=True`, local and scope envelopes preserve the presence and exact values of `items`, `total`, `offset`, and `limit` without missing-key defaults, coercion, or repair; removing each field or changing an echo remains observable for Task 3 to reject;
  - the local Library row is exactly `id`, `title`, `type`, `last_modified`; the normalized scope/Library row is exactly canonical `id`, positive integer `backing_media_id`, `title`, `media_type`, and `updated_at`, with no content, path/provenance, blob, embedding, client, or other detail-only fields;
  - malformed request coordinates fail at the service boundary, while malformed response envelopes pass through unchanged to the Library result validator;
  - a complete, sorted distinct-type result is returned through the local and scope service seams, including 60+ types whose last option is absent from the current page;
  - no type values or private item fields are logged;
  - local SQLite work remains off the Textual/event loop according to the existing service contract.

- [ ] **Run RED.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Media/test_local_media_reading_service.py \
  Tests/Media/test_media_reading_scope_service.py \
  Tests/Media/test_media_reading_scope_service_off_loop.py \
  -k 'library_media or search_media or media_type or offset'
```

- [ ] **Replace prefix fetching with exact storage coordinates.** Propagate `library_summary: bool = False` through DB/local/scope search. When true and local, use the exact summary projection and a dedicated pass-through branch that never supplies fallback envelope keys; when false, preserve generic/server behavior. Validate only request primitives at the shared service boundary; do not impose the Library result/cardinality/identity contract on generic/server callers. The local service passes the requested limit/offset directly to the DB and returns its coherent exact total without falsy/default repair. Normalize Library rows once into the exact five-key canonical shape; never feed a canonical ID back through `normalize_local_media_row`. Add only thin local/scope methods over the DB's existing active `get_distinct_media_types` query. Exact response cardinality/identity validation belongs to Task 3's Library-only `MediaBrowseResult` boundary.
- [ ] **Run GREEN and the required prefix-fetch inverse.** Temporarily restore `limit + offset` page-1 fetching; the deep-page propagation/query-bound test must fail, then restore.
- [ ] Commit this slice.

```bash
git add tldw_chatbook/Media/local_media_reading_service.py \
  tldw_chatbook/Media/media_reading_scope_service.py \
  Tests/Media/test_local_media_reading_service.py \
  Tests/Media/test_media_reading_scope_service.py \
  Tests/Media/test_media_reading_scope_service_off_loop.py
git commit -m "fix(library): expose complete media type facets"
```

---

### Task 3: Add exact Media page state and a source-owned controller

**Files:**

- Modify: `tldw_chatbook/Library/library_media_state.py`
- Add: `tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py`
- Test: `Tests/Library/test_library_media_state.py`
- Add: `Tests/UI/test_library_media_browse_controller.py`

- [ ] **Write RED pure tests** for `MediaBrowseScope` and `MediaBrowseResult`:
  - fixed Library page size 20, non-bool positive page, checked `(page - 1) * 20 <= 2**63 - 1`;
  - unfiltered type scope uses unambiguous internal `None`; blank normalizes to `None`, while every nonblank stored type remains literal, including valid source values `All`, `all`, and `ALL`;
  - exact request coordinate echoes and exact cardinality `min(20, max(total-offset, 0))`;
  - every normalized item is exactly the five-key Library summary mapping; `backing_media_id` is a non-bool positive integer, canonical `id` equals `local:media:<backing_media_id>`, and raw/canonical identities are page-unique; missing/`None`/bool/non-integer/mismatched IDs fail closed before projection;
  - malformed/duplicate identities fail closed instead of being silently dropped/deduplicated;
  - query/type/sort/page are immutable and normalized only at the source boundary.
- [ ] **Write RED controller tests** for requested/applied separation, retained last-good rows during loading/failure, source-specific pager display, exact page-only vs scope-failure copy, Retry using requested scope, navigation/mutation refresh from the entire applied scope, one automatic clamp for a limit/offset out-of-range response, double-shrink stale fallback, page-generation rejection, and separately fenced complete-facet generations/fingerprints.
- [ ] **Run RED.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Library/test_library_media_state.py \
  Tests/UI/test_library_media_browse_controller.py
```

- [ ] **Implement the smallest source-specific model.** Reuse `build_library_pager_display`; do not copy page-title/range/button calculations. Keep latest requested scope separate from last applied exact result. Keep `retained_items` independent of the exact envelope so a known mutation can be represented while stale without forging totals. Facet options are a separate complete-source tuple, not derived from page rows.
- [ ] **Run GREEN and validation mutations.** Remove duplicate-ID validation and allow a second automatic clamp; require exact tests to fail, then restore.
- [ ] Commit this slice.

```bash
git add tldw_chatbook/Library/library_media_state.py \
  tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py \
  Tests/Library/test_library_media_state.py \
  Tests/UI/test_library_media_browse_controller.py
git commit -m "fix(library): retain applied media pages"
```

---

### Task 4: Give Media dedicated screen authority and lifecycle safety

**Files:**

- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Test: `Tests/UI/test_library_shell.py`
- Test: `Tests/UI/test_library_multiselect_media.py`
- Test: `Tests/UI/test_library_media_side_by_side.py`

- [ ] **Write mounted RED tests** using the real `LibraryScreen`/`TldwCli` route and production-faithful services. Cover:
  - default Media entry starts one dedicated page request and shows loading without fabricated totals;
  - the legacy broad Library snapshot cannot populate or overwrite Media canvas rows;
  - a newer generation wins over a gated older page/scope response;
  - a newer complete-facet generation wins over a gated older facet response, and facets cannot apply after navigation/unmount;
  - unmount/navigation invalidates Media authority before the first await, and a fresh screen has distinct controller identity;
  - restore persists only the last applied scope, normalizes invalid/overflow page values, discards rows/transient loading/error, and refetches;
  - Previous/Next and type changes use applied-vs-requested rules from the controller; pure/controller coverage preserves those same rules for existing query/sort scope fields without adding mounted controls;
  - mounted page/type changes exit Select mode, clear current-page selection, and show `Selection cleared.`; pure/controller tests cover the same contract for programmatic query/sort scope changes;
  - detail/back returns to the applied page/scope after a failed draft request;
  - an external shrink triggers one guarded final-page reload; a coordinated second shrink enters stale recovery instead of looping or applying an empty false page;
  - all service calls, workers, and gates drain with bounded cleanup and no app-wide wait that can mask a pre-admission failure.

- [ ] **Run RED** only on Media lifecycle selectors.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_shell.py \
  Tests/UI/test_library_multiselect_media.py \
  Tests/UI/test_library_media_side_by_side.py \
  -k 'library and media and (page or paging or filter or type or restore or retry or stale or unmount or snapshot or selection or focus or detail)'
```

- [ ] **Wire the controller as the only Media canvas page authority.** Broad snapshots remain rail/landing/RAG summaries only. Capture generation/navigation authority before every await. Bind the separate complete-facet read to its own current generation/fingerprint (or the same guarded request lifetime) so an old response cannot overwrite newer options or apply after unmount. Restore applied scope only. Dispatch page/type requests through existing worker ownership; do not introduce a second generic worker layer or persist page records.
- [ ] **Run GREEN and the required generation inverse.** Remove the generation fence; the gated stale-result test must fail, then restore.
- [ ] Commit this slice.

```bash
git add tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_shell.py \
  Tests/UI/test_library_multiselect_media.py \
  Tests/UI/test_library_media_side_by_side.py
git commit -m "fix(library): harden media page lifecycle"
```

---

### Task 5: Render the resilient pager and one bounded complete-type chooser

**Files:**

- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` only if required by measured geometry.
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss` only through the repository bundle tool if component CSS changes.
- Test: existing Media canvas/choice tests, `Tests/UI/test_library_shell.py`, `Tests/UI/test_library_multiselect_media.py`, `Tests/UI/test_library_media_side_by_side.py`.
- Test: `Tests/UI/test_library_choice_strips.py`, `Tests/UI/test_library_canvas_sync_defects.py`, `Tests/UI/test_library_canvas_scoped_sync.py`, and `Tests/UI/test_library_entry_compose_once.py` where their Media fixtures/contracts change.

- [ ] **Write mounted RED tests** at both `100x30` and `170x48` with the production Library hierarchy and exact `TldwCli.CSS_PATH`. Require:
  - title `Media (45)`, ranges `1–20`, `21–40`, `41–45`, page numbers, and visible disabled reasons;
  - unknown initial/error states never display `(0)` or an exact range;
  - retained rows remain visible during loading/failure while unsafe controls follow pager freshness;
  - the wide/compact hierarchy is explicitly `list pane Vertical -> row VerticalScroll + pager/status`, paired with the preview; row viewport scrolls independently to row 20 while the pager/status x/y region remains inside the list-pane region and visible;
  - focus returns to invoking pager button, then the enabled opposite button, then the filter/control when both are disabled after clamp;
  - Retry is unique and focus recovery is bounded;
  - one `OptionList`-style widget exposes all 60+ complete types with constant mounted widget count, `✓` active marker, keyboard reach/commit of the final option, Escape/cancel returning focus to the opener, and no requested/applied scope change before commit.

- [ ] **Run RED** on the exact new Media canvas/geometry/type selectors.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_shell.py \
  Tests/UI/test_library_multiselect_media.py \
  Tests/UI/test_library_media_side_by_side.py \
  Tests/UI/test_library_choice_strips.py \
  Tests/UI/test_library_canvas_sync_defects.py \
  Tests/UI/test_library_canvas_scoped_sync.py \
  Tests/UI/test_library_entry_compose_once.py \
  -k 'media and (pager or page or geometry or type or choice or retry or loading or error or focus or selection)'
```
- [ ] **Implement declarative rendering.** Pass the controller's pager and complete facet tuple into constructor and `sync_state`. Replace the per-type Button strip with one bounded `OptionList`; retain one opener button. Render disabled labels/reasons declaratively so recomposition cannot re-enable stale actions. Keep the pager outside the row scroll viewport.
- [ ] Change CSS only if the real geometry test demonstrates a failure. If changed, rebuild the modular bundle and assert component/bundle parity; never hand-edit generated semantics.
- [ ] **Run GREEN and required facet inverse.** Derive type options from current-page rows; the 60-type final-option test must fail, then restore. Also remove the row `1fr/min-height:0` containment rule if one was added; its narrow geometry test must fail, then restore.
- [ ] Commit this slice.

```bash
git add tldw_chatbook/Widgets/Library/library_media_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_shell.py \
  Tests/UI/test_library_multiselect_media.py \
  Tests/UI/test_library_media_side_by_side.py \
  Tests/UI/test_library_choice_strips.py \
  Tests/UI/test_library_canvas_sync_defects.py \
  Tests/UI/test_library_canvas_scoped_sync.py \
  Tests/UI/test_library_entry_compose_once.py
# If changed, also stage tldw_chatbook/css/components/_agentic_terminal.tcss
# and tldw_chatbook/css/tldw_cli_modular.tcss after bundle parity passes.
git commit -m "feat(library): render resilient media pager"
```

---

### Task 6: Preserve applied Media scope through delete/undo and stale recovery

**Files:**

- Modify: `tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py` only if declarative mutation freshness needs an additional input.
- Test: `Tests/UI/test_library_media_browse_controller.py`
- Test: `Tests/UI/test_library_multiselect_media.py`
- Test: `Tests/UI/test_library_media_trash.py`
- Test: `Tests/UI/test_library_shell.py`

- [ ] **Write RED mutation tests** for single delete, bulk delete, receipt undo, metadata edit whose title/last-modified value can move ordering, and top-level Media reappearance after Trash restore. Media Trash paging remains out of scope; its mutation side effect on the authoritative top-level page is in scope. Require:
  - read generation invalidates before the durable write;
  - from pre-write invalidation until the durable write settles, row/bulk/pager/type controls cannot dispatch or alter requested/applied scope;
  - the known deleted/restored row is reconciled locally without forging a new exact envelope;
  - follow-up refresh starts from the full applied scope, not page 1 or a failed draft;
  - a one-page shrink accepts one coherent clamp and performs one post-write authoritative read sequence;
  - a successful mutation plus failed refresh remains committed, enters stale presentation, suppresses total/range, keeps Retry/scope controls enabled, and disables every row/bulk/pager action with non-colour marker and exact reason across any recompose;
  - Retry or another successful authoritative scope request clears stale and restores valid actions;
  - after a committed refresh failure, type/scope recovery is enabled while row/bulk/pager actions remain disabled;
  - facet options refresh authoritatively when a mutation removes/restores the final item of a type;
  - every local update/delete/restore call receives the validated positive integer `backing_media_id`, never the canonical canvas `id`; focused spies fail on canonical/double-prefixed mutation IDs.
- [ ] **Run RED** on the exact Media mutation selectors.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_browse_controller.py \
  Tests/UI/test_library_multiselect_media.py \
  Tests/UI/test_library_media_trash.py \
  Tests/UI/test_library_shell.py \
  -k 'media and (delete or bulk or undo or edit or restore or mutation or stale or retry)'
```
- [ ] **Implement pre-write invalidation and retained reconciliation** through one Media mutation-completion choke point shared by delete, bulk delete, undo, metadata edit, and Trash restore. Preserve their existing detail/receipt/partial-failure ownership. Do not mutate a broad snapshot as canvas authority or report a committed mutation as failed merely because its page refresh failed.
- [ ] **Run GREEN and the required stale-action inverse.** Leave one stale row or bulk action enabled; the mounted accessibility/safety test must fail, then restore. Force mutation refresh to page 1; the applied-scope test must fail, then restore.
- [ ] Commit this slice.

```bash
git add tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/Widgets/Library/library_media_canvas.py \
  Tests/UI/test_library_media_browse_controller.py \
  Tests/UI/test_library_multiselect_media.py \
  Tests/UI/test_library_media_trash.py \
  Tests/UI/test_library_shell.py
git commit -m "fix(library): retain media mutation scope"
```

---

### Task 7: Run touched-component verification, live proof, review, and closeout

**Files:**

- Modify: `Docs/User_Guide/library/media-and-conversations.md` or the canonical Media Library guide identified during implementation.
- Modify: `backlog/tasks/task-16483 - Add-authoritative-paging-and-complete-facets-to-Library-Media.md`
- Modify: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md` only if this task produces a reusable incident-backed lesson.

- [ ] **Run the authoritative touched-component automated gate only. Do not run `pytest` without explicit paths/selectors.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/DB/test_client_media_pagination.py \
  Tests/DB/test_client_media_debug_logging.py \
  Tests/Media/test_local_media_reading_service.py \
  Tests/Media/test_media_reading_scope_service.py \
  Tests/Media/test_media_reading_scope_service_off_loop.py \
  Tests/Library/test_library_media_state.py \
  Tests/UI/test_library_media_browse_controller.py

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_shell.py \
  Tests/UI/test_library_multiselect_media.py \
  Tests/UI/test_library_media_side_by_side.py \
  Tests/UI/test_library_choice_strips.py \
  Tests/UI/test_library_canvas_sync_defects.py \
  Tests/UI/test_library_canvas_scoped_sync.py \
  Tests/UI/test_library_entry_compose_once.py \
  Tests/UI/test_library_media_trash.py \
  -k 'library and media'
```

If Task 5 changes another existing Media canvas/choice test file, add that exact file to this gate. Do not expand to unrelated Library, Notes, Prompt, Conversation, Skill, Collection, or full-repository tests.

- [ ] Run Ruff only on changed Python files, `git diff --check`, and CSS build/parity only if CSS changed.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/DB/Client_Media_DB_v2.py \
  tldw_chatbook/Media/local_media_reading_service.py \
  tldw_chatbook/Media/media_reading_scope_service.py \
  tldw_chatbook/Library/library_media_state.py \
  tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/Widgets/Library/library_media_canvas.py \
  Tests/DB/test_client_media_pagination.py \
  Tests/DB/test_client_media_debug_logging.py \
  Tests/Media/test_local_media_reading_service.py \
  Tests/Media/test_media_reading_scope_service.py \
  Tests/Media/test_media_reading_scope_service_off_loop.py \
  Tests/Library/test_library_media_state.py \
  Tests/UI/test_library_media_browse_controller.py \
  Tests/UI/test_library_shell.py \
  Tests/UI/test_library_multiselect_media.py \
  Tests/UI/test_library_media_side_by_side.py \
  Tests/UI/test_library_choice_strips.py \
  Tests/UI/test_library_canvas_sync_defects.py \
  Tests/UI/test_library_canvas_scoped_sync.py \
  Tests/UI/test_library_entry_compose_once.py \
  Tests/UI/test_library_media_trash.py
git diff --check

# Only if component CSS changed:
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
```
- [ ] Re-run the four required inverses one at a time and restore each immediately:
  1. remove Media generation guard;
  2. restore prefix fetching;
  3. derive type facets from current page;
  4. enable a stale row/bulk action.
- [ ] **Perform isolated live verification** at real `100x30` and `170x48` geometries with fresh `env -i` profiles, real production `TldwCli`, real local Media DB/service, at least 45 items, and at least 60 distinct types. Verify page 1/2/final, row-20 scroll with fixed pager, type chooser final-option keyboard commit/cancel, filter/type exact totals, selection clearing, controlled page failure/Retry, mutation-refresh stale recovery, detail/back, and clean exit.
- [ ] Capture success and controlled-error logs and assert unique query, title/body, stable ID, DB-path, and credential sentinels never appear. Capture PID/open-handle/profile geometry evidence and prove the real user profile is byte-identical before/after.
- [ ] Request independent spec and quality/minimality reviews. Resolve every Critical/Important finding with focused RED/GREEN evidence.
- [ ] Update the Media user guide, check every TASK-16483 acceptance criterion, add concise implementation notes naming automated/mutation/geometry/live evidence and the user-directed no-full-suite deviation, set the task to Done through Backlog CLI, and commit closeout docs.

```bash
git add "backlog/tasks/task-16483 - Add-authoritative-paging-and-complete-facets-to-Library-Media.md" \
  Docs/User_Guide/library/media-and-conversations.md
git commit -m "docs(library): close media pagination task"
```

---

## Completion criteria

TASK-16483 is complete only when all seven acceptance criteria are checked, every required inverse mutation has turned its focused test RED and been restored GREEN, both live geometries pass against isolated real profiles, touched Media diagnostics contain no private sentinels, focused changed-component tests/lint/diff checks are green, independent reviews report no Critical/Important findings, documentation is current, and the Backlog task is Done. A repository-wide pytest run is explicitly excluded by user direction.
