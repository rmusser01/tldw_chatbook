# Library L2b.2 — Notes sync/import/count + tab deprecation — QA evidence (2026-07-08)

Branch: `claude/library-l2b2` (worktree, off origin/dev @ dd76a03e, incl. #586).
Captured from textual-serve (real app CSS + worktree code via PYTHONPATH),
headless chromium, viewport 2050×1240, isolated HOME `/private/tmp/tldw-l2-qa2`
with 4 REAL notes seeded via `CharactersRAGDB.add_note` and a real sync folder
`~/Documents/Notes` holding a `from-disk.md`.

- `navbar-no-notes-tab-2026-07-08.png` — THE DEPRECATION: the top nav bar
  (Home · Console · Library · Artifacts · Personas · Watchlists · Schedules ·
  Workflows · MCP · ACP · Skills · Settings) no longer has a **Notes** tab.
  Notes now lives entirely inside the Library.
- `notes-list-exact-count-2026-07-08.png` — Browse ▸ Notes: rail badge and
  canvas both read **`Notes (4)`** (EXACT — the new `count_notes` seam; was
  `(4+)` before), and the list header carries **Sync** + **Import note**.
- `sync-panel-idle-2026-07-08.png` — Notes sync panel AFTER the approved
  sr-UX/HCI fix wave (487d512f): purpose line ("Mirror notes between a folder
  on disk and the Library."), labeled `folder` input + `Browse…`, grouped
  settings (`direction: Bidirectional ▸` / `conflicts: Newer wins ▸` /
  `auto-sync: every 5m ○` — no `Select`/`Switch`), and `Sync now` in the
  primary accent, status `idle`. All render-safe.
- `sync-direction-label-2026-07-08.png` — direction cycled once: reads
  `Disk → Library` (jargon fix — no "DB" anywhere on the panel; conflicts
  offers Newer/Disk/Library wins, the dead "Ask" mode was removed).
- `sync-run-complete-2026-07-08.png` — a REAL sync through the live
  `NotesSyncService`/`NotesSyncEngine` after the fix wave: status
  `done · 1 file` (singular), activity `Sync complete: 1 note created` /
  `Starting sync: Notes` rendered dimmer than the status headline
  (`from-disk-2.md` was ingested). Thread-marshaled progress, no crash;
  `Sync now` renders disabled ("Syncing…") while running.
- `create-templates-dated-2026-07-08.png` — Create view after the sync:
  count is now `Notes (5)` (the count seam tracked the synced note), and the
  template rows show clean dated secondaries incl. **`Project Plan - 2026-07-08`**
  and **`Research Notes - 2026-07-08`** (Task 4's dangling-separator fix).

## Verification

- All 5 build tasks task-reviewed (2 on Opus: sync + deprecation); in-loop
  findings fixed + re-reviewed. Notable: the flush-vs-autosave self-conflict,
  a latent `sync_engine.py` NEWER_WINS datetime crash, and the deprecation's
  deep-link re-points (proven red-catchers).
- Deprecation completeness independently verified: `grep TAB_NOTES` → only a
  docstring; `python -c "import tldw_chatbook.app"` clean; old `tab="notes"`
  config resolves via a `"notes"→"library"` compat alias. The 10 full-suite
  failures were independently reproduced at the base commit (pre-existing).
- Real-backend discipline throughout: `count_notes` on a real DB; a real
  tmp-dir sync integration test (db_to_disk / disk_to_db / newer_wins).

## Notes to weigh at approval

- The standalone `notes_screen.py` + `Widgets/Note_Widgets/*` stay in-tree
  (now unreachable) for the tracked dead-code sweep — not deleted this phase.
- Auto-sync's 300s timer runs across other rail rows once enabled (documented,
  matches the old pane's scope) — a future refinement.
- Rail count badge refreshes on next navigation, not at sync completion (a
  sync that creates notes shows the new count after the next rail press) —
  pre-existing snapshot behavior, follow-up polish.
- 2026-07-08: a 10-finding sr-UX/HCI review of the sync panel was approved and
  applied (487d512f): dead "Ask" mode removed + stale-config coercion, honest
  conflict copy, Syncing…/disabled run button, pluralization, Library (not
  "DB") labels, folder label, grouping + primary Sync now, purpose line,
  auto-sync cadence label, status/activity/failed styling.
