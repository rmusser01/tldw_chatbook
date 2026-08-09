# Watchlists reader-first, phase 4: OPML folder round-trip

**Spec**: `Docs/superpowers/specs/2026-08-05-watchlists-reader-first-design.md` §Phasing ("Phase 4 — round-trip and polish", :334-335).
**ADR**: ADR-042 covers the re-IA; the OPML folder↔watchlist mapping is an **interchange-policy** decision (see Task 1).
**Phases 1–3 shipped**: PRs #1383 / #1430 / #1443 (TASK-2513 line, TASK-3072, TASK-3791). This plan assumes that tree.

**Scope finding (recorded here so the phase boundary is honest):** the spec names polish tasks 2308/2310/2312/2313 for this phase — all four are already **Done** (2308 was additionally absorbed into `Subscriptions/item_dates.py` per the spec's own note). What remains of phase 4 is the OPML round-trip: **watchlists map to OPML folders on BOTH import and export**, so moving between this app and another reader never flattens the user's structure.

## Current state (what phases 1–3 left)

- `Subscriptions/watchlist_opml_service.py` (39 lines) is FLAT both ways:
  - `parse` (:10) iterates every `<outline>`, keeps those with `xmlUrl`/`htmlUrl`, and **discards folder outlines entirely** — folder membership never reaches the caller.
  - `export` (:28) serializes one flat `<outline>` per source under `<body>` — watchlist membership is lost on the way out.
- `watchlist_scope_service.import_opml` (:897) creates one source per payload and stops: everything lands **Unassigned**. The UI (`opml_dialogs.py:23`, `OpmlImportDialog`) is a paste-XML modal returning raw text — there is no target-watchlist picker, so folder mapping conflicts with no existing UI contract. `export_opml` (:927) lists flat and serializes flat.
- Membership writes exist and are UI-proven: `WatchlistBundleService.create` / `add_source` (the rail's New/Add-existing verbs, task-895), and `list_watchlists` / `list_source_rows` give export the structure. `create_source` dedupes by URL (existing behavior — the round-trip pin relies on it for idempotent re-import).
- The rail renders escaped names (`escape_markup` at `watchlist_tree.py`), so a hostile folder/watchlist name is inert on that surface already; ElementTree escapes attributes on serialize. The hostile test still pins both ends.

## Mapping rules (the policy Task 1's ADR records)

1. An `<outline>` with **no `xmlUrl`/`htmlUrl` that contains feed outlines is a FOLDER**. Its nearest feed descendants map to a watchlist of its name.
2. **Nested folders flatten to the innermost folder** — the one directly containing the feed (most readers are single-level; the closest ancestor is the user's most specific intent).
3. An outline with a feed URL **and** children is a feed; its children are evaluated under its folder context, not as a folder of its own.
4. **Folder names match watchlists case-insensitively** (`"AI"` reuses `AI`); no match creates the watchlist. Top-level feeds (no folder ancestor) land **Unassigned** — today's behavior for every feed, preserved deliberately for the folderless case.
5. Export nests: one folder outline per watchlist (ordered by name) containing its member feeds (by name), then top-level Unassigned feeds. Deterministic ordering keeps exports diff-stable.
6. Import is additive only: it never removes memberships or sources (a re-import after export is a structural no-op, which is what the round-trip pin proves).

## Tasks

### Task 1: Task bookkeeping + ADR + docs commit

- [ ] New short ADR `backlog/decisions/043-opml-watchlist-folder-mapping.md` recording the six mapping rules above (interchange policy: naming, nesting, reuse, additive-only) — required here because this is an interchange/conflict policy future contributors will ask about; ADR-042 (IA) does not cover it.
- [ ] Create the backlog task (ACs below), In Progress, plan + ADR linked. Commit message: `docs(watchlists): phase 4 plan — OPML folder round-trip (task-3604)`

### Task 2: `parse` returns folder structure

**Files:**
- Modify: `tldw_chatbook/Subscriptions/watchlist_opml_service.py` (`parse` :10)
- Test: `Tests/Subscriptions/` — find the OPML tests with `grep -rln "WatchlistOpmlService\|import_opml" Tests/Subscriptions/`

- [ ] **Step 1: failing tests.** A folder outline groups its feeds (`folder` key on each payload); nested folders resolve to the innermost; a feed-with-children is a feed and its children inherit its folder context; top-level feeds carry `folder=None`; two folders whose names differ only by case produce feeds that name the SAME logical folder (the reuse rule's input — normalized at assignment, Task 3, but parse must surface the raw name faithfully); a folder named `<script>alert(1)</script>` parses to that literal string; malformed XML still raises into the dialog's existing handler (no behavior change on the error path).
- [ ] **Step 2: implement.** Recursive walk carrying the current folder context; the feed payload gains `"folder": str | None`. No schema change.
- [ ] **Step 3: run + commit** `feat(watchlists): OPML parse preserves folder structure (task-3604)`

### Task 3: Import assigns folder membership

**Files:**
- Modify: `tldw_chatbook/Subscriptions/watchlist_scope_service.py` (`import_opml` :897)
- Modify: `tldw_chatbook/Subscriptions/watchlist_bundle_service.py` (a find-watchlist-by-name helper if none exists — check `list_watchlists` first)
- Test: scope-service level + a DB-backed membership test

- [ ] **Step 1: failing tests.** Importing an OPML with folders creates watchlists for new folder names, reuses existing ones case-insensitively, assigns member sources, leaves top-level feeds Unassigned, and returns an honest summary (`created_sources`, `watchlists_created`, `watchlists_reused`, `assignments`); re-importing the same document is a structural no-op (dedupe by URL + reuse by name); the server-backend refusal and policy enforcement are unchanged.
- [ ] **Step 2: implement.** Group payloads by normalized folder name; resolve-or-create per group via the bundle service; `add_source` per member; collect the summary. Still additive only.
- [ ] **Step 3: run + commit** `feat(watchlists): OPML import maps folders to watchlists (task-3604)`

### Task 4: Export nests watchlists as folders

**Files:**
- Modify: `tldw_chatbook/Subscriptions/watchlist_opml_service.py` (`export` :28 — new shape taking structured input), `tldw_chatbook/Subscriptions/watchlist_scope_service.py` (`export_opml` :927 — supply watchlists + membership, not a flat list)
- Test: service-level + hostile-name serialization

- [ ] **Step 1: failing tests.** A profile with two watchlists (one sharing a source between them), plus unassigned feeds, exports as: folder per watchlist (name-ordered) with member feeds nested (name-ordered), unassigned feeds top-level; a watchlist named with markup/metacharacters serializes escaped and re-parses to the literal name; a source in two watchlists appears under both (membership is many-to-many; the document says so twice — that is the faithful representation).
- [ ] **Step 2: implement.** `export` takes `(watchlists, membership_rows, unassigned_sources)`; deterministic ordering; ElementTree escaping does the safety work.
- [ ] **Step 3: run + commit** `feat(watchlists): OPML export nests watchlists as folders (task-3604)`

### Task 5: The round-trip pin + docs

- [ ] **The pin**: export a structured profile → import the document into a FRESH database → assert identical structure (same watchlist-name set, same membership sets per name, same unassigned set). This is the phase's done-when, machine-checked.
- [ ] The import dialog's confirmation/toast reads the new summary honestly ("12 sources into 3 watchlists (2 new) + 2 unassigned"), not just a source count.
- [ ] Full suite: `Tests/Subscriptions/` + `Tests/Watchlists/` + coupled `Tests/UI/` green; ruff clean.
- [ ] Task Implementation Notes + backlog `-s Done` via CLI; record the additive-only choice and the innermost-folder rule's rationale.
- [ ] Commit `feat(watchlists): OPML round-trip pin + honest import summary (task-3604)`

## Definition of done (phase 4 — and the spec's last phase)

- Export → import round-trips watchlist structure losslessly (pinned).
- Folderless OPML behaves exactly as before (top-level → Unassigned, flat export unchanged when there are no watchlists).
- Hostile names are inert on parse, serialize, and rail render.
- All ACs checked; suites green; ruff clean; ADR-043 linked from the task.
