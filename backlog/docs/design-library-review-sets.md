# Design — Library review sets

**Task:** TASK-28024 · **Status:** DRAFT, awaiting user approval (AC#2) · **Date:** 2026-09-02

Reviewing a *set* of media items — a conference's talks, a tag/keyword-filtered
browse result, or a hand-picked selection — should be a first-class object, not
a memory exercise. This spec defines that object, grounded in the seams that
already ship on dev.

---

## 1. What a review set is

A **review set** is:

- an **ordered list of media item ids**, *pinned at creation time* (it does not
  re-query or re-sort afterward — it is a snapshot of "these items, in this
  order");
- a **cursor** (which item you're on);
- a **per-item done mark** (reviewed / not yet);
- an explicit **completion state** (all-reviewed, with a timestamp).

While a set is *active*, the media Reader walks it with Next/Prev instead of
walking the current browse page; it shows progress ("12 of 40"), resumes at the
cursor across app restarts, and ends with an explicit all-reviewed state.

### Hard constraints inherited from the codebase

These are not choices — the existing code fixes them:

1. **Local-only.** The browse result and `begin_selection` both hard-require
   `local:media:{int}` ids (`library_media_state.py:174`,
   `library_media_reader_state.py:219`). Server items are read-only external
   detail and can't be walked. **Review sets contain only local media ids.**
2. **The Reader session state is per-item and transient** — it holds one
   selected + one loaded identity and *no* list position
   (`LibraryMediaReaderSessionState`). The set's cursor/done/completion must
   live in their own persistent structure, not in that state.
3. **Per-item scroll resume already exists and is a different concern.**
   `ReadingProgress` (Client_Media_DB_v2) stores `{scroll_x, scroll_y}` per
   media id — resume *within* an item. The review-set cursor is resume *across*
   items. They compose (walking to an item still restores its scroll) but must
   not be folded together.

---

## 2. Data model

Neither existing collection family in `Library_Collections_DB.py` fits: the
legacy `library_collections` tables are **frozen read-only** (every write raises
`LegacyCollectionsReadOnlyError`) and carry no order/cursor/done; the
`collection_capture_*` tables are a separate web-clip domain. So review sets get
their **own new tables** in that DB, added with its established migration
pattern (bump `_CURRENT_SCHEMA_VERSION` 3 → 4, append an idempotent DDL tuple in
`_initialize_schema`, gate on `schema_version`).

```
review_sets(
    set_id        TEXT PRIMARY KEY,      -- uuid
    name          TEXT NOT NULL,         -- e.g. "Tag: conference-2026" / "8 selected items"
    origin        TEXT NOT NULL,         -- 'browse' | 'selection' | 'read_later' (provenance)
    cursor        INTEGER NOT NULL DEFAULT 0,   -- absolute position of the current item (0-based)
    active        INTEGER NOT NULL DEFAULT 0,    -- exactly one row may be 1 (see partial unique index)
    completed_at  DATETIME,              -- NULL until every live item is done
    created_at    DATETIME NOT NULL,
    updated_at    DATETIME NOT NULL,
    deleted_at    DATETIME               -- soft delete, matching the DB's conventions
)
-- CREATE UNIQUE INDEX review_sets_one_active ON review_sets(active) WHERE active = 1 AND deleted_at IS NULL;
--   ^ enforces "at most one active set" in the schema, and makes it durable across
--     restarts without stashing runtime state in config.toml.

review_set_items(
    set_id            TEXT NOT NULL REFERENCES review_sets(set_id),
    position          INTEGER NOT NULL,  -- pinned order, 0-based, dense at creation, NEVER renumbered
    backing_media_id  INTEGER NOT NULL,  -- the local Media(id); canonical id derived as local:media:{n}
    title_snapshot    TEXT NOT NULL,     -- title at pin time, for the list UI when an item is gone
    done              INTEGER NOT NULL DEFAULT 0,
    done_at           DATETIME,
    PRIMARY KEY (set_id, position)
)
-- index (set_id, backing_media_id) for "is this item in the active set / mark it done"
```

Notes:
- **`active` lives in the DB, not config.** The active-set pointer is durable
  runtime state, not a user setting; a partial unique index guarantees the
  "one active set" invariant (§3) and re-activation on launch is a single
  `WHERE active = 1` read.
- **Position is pinned; it never renumbers** — deletions leave a tombstone
  (see §7) so the cursor and "X of M" stay stable and resumable.
- `backing_media_id` (int) is stored, not the canonical string — one derive
  (`f"local:media:{n}"`) at read time, matching how the browse result already
  carries both.
- `title_snapshot` is **load-bearing**, not a convenience: see the next note.

### Cross-database reality (important)

`Library_Collections_DB` and `Client_Media_DB_v2` are **separate SQLite files**
(each is constructed with its own `db_path`). Therefore `backing_media_id`
**cannot** carry a foreign key to `Media(id)` — there is no cross-file
referential integrity. Two consequences the implementation must own:

- **No cascade.** Deleting a media item does *not* touch the review-set rows.
  That is exactly what we want for tombstones — the set is a snapshot — but it
  means membership can silently point at deleted items.
- **Tombstone detection is a runtime resolve.** "Is this item still live?" is a
  lookup against the Media DB at walk/render time (does `local:media:{n}`
  resolve), not a JOIN. `title_snapshot` is the only title available once an
  item is gone, so the list UI reads it directly rather than joining.

---

## 3. Set lifecycle

```
        create                     walk                       complete
  (pin ordered ids)  ──▶  active: cursor advances,   ──▶  every live item done
                          items marked done                 → completed_at set
        │                        │                                 │
        │                        ▼                                 ▼
        │                  pause/switch                     reopen (clear
        │                 (set persists,                     completed_at) or
        └───────────────  cursor saved)                      dismiss (soft-delete)
```

- **Create** pins the ordered ids *now* and opens the set as **active**, cursor
  at 0 (the first item), loading it in the Reader.
- **Active** is a single-set property: **at most one set is active at a time**
  (the one the Reader is walking). Other sets persist and can be resumed. The
  active set id is a small piece of app/session state (a config key or a
  one-row pointer), so a restart re-activates it and jumps to its cursor.
- **Complete** is derived: when every *live* (non-tombstoned) item is `done`,
  `completed_at` is stamped and the Reader shows an explicit all-reviewed state.
- **Reopen / dismiss**: a completed set can be reopened (clear `completed_at`,
  keep done marks) or dismissed (soft-delete). Dismiss just deactivates + soft
  deletes; the pinned rows stay for audit until purged.

---

## 4. Entry points

Three ways to create a set, all producing the same object:

1. **"Review these"** on the **browse result** — a button on the media list
   toolbar. Pins the ordered ids of the current view. "Search result" is not a
   separate surface: filtering the media list is just a `query` on
   `MediaBrowseScope`, so a filtered list *is* a browse result and the same
   button covers it. (The RAG semantic search at TAB_SEARCH is a different thing
   — ranked chunks, not an ordered media-id list — and is **not** a review-set
   source.)
2. **"Review selected"** — a third **Select-mode bulk action**, slotting next to
   "Export selected" / "Delete selected" in the select-mode toolbar
   (`library_media_canvas.py`). Pins the selected ids.

Plus a natural third, cheap because the data already exists:

3. **"Review read-later"** — build a set from `list_read_it_later_media_ids()`
   (already an ordered id list, `saved_at DESC`). Optional; good first
   real-data test.

**Ordering rule for "Review selected" (revised after review).** `RowSelection`
is an *unordered* `set[str]` with **no insertion order**, and a selection can
include ids from pages the user paged past — so re-projecting against the
*mounted* rows is undefined for off-page ids. Instead, **order the selected ids
by a deterministic query**: one `SELECT` over the selected backing ids in the
list's active sort order (default `updated_at DESC` — the order the user was
seeing), with `backing_media_id` as the tiebreak. This is stable regardless of
which page is mounted and does not depend on the DOM.

**Scope — pin the whole filtered result, with a cap (revised after review).**
The browse controller holds only the **current page** (`applied_result.items`),
the same page-boundary limit that constrains 28005. A conference or a tag is
usually more than one page, so "Review these" pins the **whole filtered result**
by paging through at creation (loop `with_page` to `last_page`). Three
constraints the build must honor:
- **Cap the size** (proposed **500** items; beyond it, warn and offer to pin the
  first 500 or narrow the filter). A 3,000-item "review set" is not a review
  set; the cap keeps the pinned list and the walk sane.
- **Off the event loop.** The multi-page collect runs in a worker
  (`run_worker(thread=True)`), not on the UI thread.
- **Not perfectly atomic.** An item ingested *during* the multi-page collect
  could be missed or seen twice; de-duplicate by id at build and accept minor
  drift — a set is a snapshot "as of creation," not a live query.

---

## 5. Viewer behavior

When a set is active and the Reader is in its plain item view:

- **Next/Prev walk the set, not the page.** The `]` / `[` actions
  (`action_library_media_next_item` / `_prev_item`) and the escape/traversal
  gating stay, but the neighbour source swaps: instead of
  `_library_media_adjacent_row` (which reads mounted `.library-media-row`
  widgets, current page only), an active set advances the **cursor over the
  pinned id list** and resolves the id at the new cursor. The per-step actuator
  is unchanged — `_select_library_media_reader_row(media_id, title, …)` — so
  fenced loading, scroll capture/restore, and mode-persistence all keep working.
- **Progress readout, defined over LIVE items.** A compact indicator —
  `12 of 40 · 7 reviewed` — in the Reader chrome (the honest-footer / Reader
  header idiom already used for `]`/`[`). Precise meaning, because tombstones
  make "position" and "count" diverge: the **cursor is an absolute position**
  (stable, never renumbers), but the readout is computed over **live**
  (non-tombstoned) items — `M` = live-item count, `X` = the cursor item's ordinal
  *among live items*, `reviewed` = live items marked done. If the cursor lands on
  a tombstone (an item deleted since it was pinned), the walker advances to the
  next live position. Advancing off the last live item (or marking the last one
  done) surfaces the **all-reviewed** state.
- **Mark-done semantics (needs your call, §9-B).** Recommended default:
  **advancing forward (`]`) marks the item you're leaving as done** (this
  matches "read through each analysis, don't miss anything"), with an explicit
  key to toggle a mark. The alternative is fully-explicit marking (a key per
  item, no auto). Auto-on-advance is less bookkeeping; the toggle covers "I
  skimmed, not done." The edges, spelled out so implementation isn't guesswork:
  - **Prev (`[`)** moves the cursor back **without un-marking** anything (you
    can re-read a done item; use the toggle to un-mark deliberately).
  - **The last item** has no "advance past," so it is marked done by the
    explicit toggle or by the completion gesture — not automatically.
  - **Jumping via the picker** (§6) moves the cursor without auto-marking the
    item left behind (a jump is not a linear read).
  - Auto-mark applies **only** to the linear `]` step.
- **Resume.** On launch, if an active set exists, its cursor item loads and the
  progress readout appears — resuming exactly where the user left off (the
  cursor is persisted on every advance; the within-item scroll is restored by
  the existing `ReadingProgress` seam).
- **Reader mode carries across the set**, because `begin_selection` already
  preserves mode — so a user reviewing on the **Analysis** tab reads every
  item's analysis in turn (and the Analysis-tab search from TASK-28026 works
  throughout). This is the concrete payoff of the whole review program.

---

## 6. Where it lives in the UI (surfaces)

- **Create**: toolbar buttons (browse / search) + the select-mode bulk action.
- **Active indicator + progress**: Reader chrome, with an **Exit review** action.
  Note the two are distinct: pressing **Escape** leaves the Reader for the list
  but **keeps the set active** (re-entering resumes at the cursor); **Exit
  review** *deactivates* the set (clears `active`) without deleting it. Only Exit
  review stops the `]`/`[` keys from walking the set.
- **Set list / resume**: a lightweight picker to resume, switch, or dismiss
  saved sets. Cheapest home is a small modal/picker opened from the media list
  (reusing the choice-strip / picker idioms already in the Library); a rail row
  is possible later but not required for v1.
- **(Improvement, v2 — not v1)** When a set is active, the media **list rows**
  could mark which items belong to it and their done state. This is the same
  surface as the has-analysis / read markers of task-28008 / task-28009, so it's
  best built *with* those rather than bolted on here.

---

## 7. Edge cases

| Case | Behavior |
|---|---|
| **Item deleted mid-set** | The position becomes a **tombstone**, detected by a **runtime resolve** against the Media DB (no FK exists — §2). The walker **skips tombstones**; the list UI shows the `title_snapshot` greyed as "removed". "X of M" counts **live** items; completion ignores tombstones. Positions never renumber, so the cursor stays stable. |
| **Every item tombstoned** | A set with **zero live items** is not "complete" — it is **empty**. Show an empty state ("all items in this set were removed") and offer **dismiss**; never stamp `completed_at` on an empty set. |
| **Re-ingest / dedup** | The set pins `backing_media_id`. If a deleted item is re-ingested as a **new** Media row (new id), it is a *different* item and is **not** auto-added — the old position stays a tombstone. If ingest dedup **reuses** the same Media id, the item simply resolves again and the set is intact. (No silent membership mutation — a pinned set is a snapshot.) |
| **Multiple concurrent sets** | Allowed to **exist**; exactly **one active** at a time. Creating a new set deactivates the previous (its cursor is saved). The picker (§6) switches between them. |
| **Item filtered out of the current browse view** | Irrelevant — the set walks its **own pinned ids** by loading each directly, independent of the current browse filter/sort/page. This is the point of pinning. |
| **Empty / single-item set** | An empty selection offers no "Review" action. A single-item set is legal (walk is a no-op; completes on marking the one item). |
| **Set spans more than one page at creation** | Handled by paging through at build time (§4-A). After creation the set never pages — it holds all ids. |
| **Duplicate ids in a selection** | De-duplicated at build (a set is a set of positions over *distinct* items). |

---

## 8. Relationship to the foundations (AC#3)

- **TASK-28005 (viewer Prev/Next over the current browse result) — SHIPPED, and
  this design *supersedes its traversal source when a set is active*, without
  removing it.** With no active set, `]` / `[` keep walking the mounted browse
  rows (28005's behavior). With an active set, the same keys walk the pinned
  cursor instead. 28005 is the default; the review set is the opt-in overlay.
  Concretely: keep `action_library_media_next_item` and the actuator; branch the
  *neighbour lookup* on "is a set active."
- **TASK-28009 (read markers) — DOES NOT EXIST yet.** There is no global
  read/unread marker for media today. A review set's **done marks are
  set-local** (per `(set, item)`), which is the right scope and does not depend
  on 28009. If 28009 later adds a *global* per-item read flag (shaped like the
  existing read-later state), the bridge is a one-line choice: *marking an item
  done in a set may also flip its global read marker.* This spec keeps done
  marks set-local for now and names that future bridge rather than blocking on
  it.
- **read-later (TASK-28027) — SHIPPED, and is a *source*, not the same thing.**
  `MediaReadItLaterState` is a global per-item queue; `list_read_it_later_media_ids`
  gives an ordered id list that "Review read-later" (§4-4) can pin. A set's done
  marks are distinct from the read-later boolean (reviewing an item in a set
  does not remove it from read-later unless we choose to — another §9 knob if
  wanted).

---

## 9. Decisions that need your explicit call

These change behavior enough that I want your sign-off before filing
implementation tasks (AC#2):

- **A. Set-size cap** — the design pins the **whole filtered result** (settled),
  capped at **500** items. Is 500 the right ceiling, and on overflow do you want
  *pin-first-500* or *refuse-and-narrow*?
- **B. Mark-done semantics** — **auto-mark on forward advance** + explicit toggle
  (recommended) or **fully explicit** marking only?
- **C. One active set vs several active** — recommendation is **one active,
  others saved** (now enforced by a partial unique index). Confirm, or do you
  want several live at once?
- **D. Persistence home** — new **v4 tables in `Library_Collections_DB.py`**
  (recommended, since a set *is* a collection-shaped thing and that DB has the
  infra) or a **separate DB module**? (I lean Collections DB.)
- **E. Scope of v1** — is the **read-later source (§4-3)** in or out for the
  first implementation? It's the cheapest way to test against real ordered data.
- **F. Membership mutation** — v1 pins at creation and never adds/removes
  (snapshot semantics). Do you want a **"remove from set"** prune (cheap:
  position → tombstone) in v1, or is snapshot-only fine to start?

---

## 10. Proposed phasing (for after approval — not filed yet)

1. **Persistence + pure model** — v4 tables in Library_Collections_DB, a
   `ReviewSet` service (create/get/advance-cursor/mark-done/complete/dismiss),
   pure cursor logic with tombstone skipping. Unit-tested with in-memory SQLite.
2. **Walker integration** — branch the `]`/`[` neighbour lookup on an active
   set; Reader progress readout + all-reviewed state; resume on launch.
3. **Entry points** — "Review these" (browse + search, whole-result build) and
   "Review selected" (re-projected order).
4. **Set picker** — resume / switch / dismiss saved sets.
5. **(Optional) read-later source** and any 28009 bridge once that marker exists.

Each phase ships independently and is separately testable. Implementation tasks
get filed **only after this design is approved.**

---

## 11. Review pass (2026-09-02) — issues caught and resolved

A critical read of the first draft, re-verified against the code, surfaced these
and folded fixes into the spec above:

1. **Cross-DB, no FK (correctness).** The set tables and the Media table are in
   *separate SQLite files*, so `backing_media_id` can't have a foreign key.
   Tombstone detection is a runtime resolve, not a cascade; `title_snapshot` is
   load-bearing. — §2.
2. **"Review selected" ordering was undefined.** `RowSelection` is an unordered
   `set` with no insertion order and can span pages, so re-projecting against the
   mounted rows breaks. Replaced with a deterministic sort-order query. — §4.
3. **Whole-result build needs a cap + worker + a non-atomic caveat.** Added a
   500-item cap, off-loop paging, and de-dupe-by-id. — §4.
4. **Active-set pointer belongs in the DB, not config.** Added an `active` column
   + partial unique index enforcing "one active." — §2/§3.
5. **Cursor vs progress under tombstones was ambiguous.** Defined: cursor is an
   absolute position; `X of M` is computed over live items; cursor-on-tombstone
   advances. — §5.
6. **Mark-done edges unspecified.** Defined Prev (no un-mark), last item
   (explicit), picker jumps (no auto-mark). — §5.
7. **All-items-tombstoned set** is *empty*, not *complete*. — §7.

Clarifications: "search result" is just the browse `query` filter (same surface),
not the RAG search; a v2 list-membership marker belongs with task-28008/28009,
not here. New open question **F** (membership prune in v1?) added to §9.
