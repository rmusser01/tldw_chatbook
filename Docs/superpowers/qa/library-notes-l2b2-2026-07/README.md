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
- `sync-panel-idle-2026-07-08.png` — Notes sync panel: folder input +
  `Browse…`, cycling `direction: Bidirectional ▸` / `conflicts: Newer wins ▸`
  (no `Select`), `auto-sync: ○` toggle (no `Switch`), `Sync now`, status
  `idle`. All render-safe.
- `sync-run-complete-2026-07-08.png` — a REAL sync executed through the live
  `NotesSyncService`/`NotesSyncEngine`: status `done · 1 files`, activity log
  `Sync complete: 1 notes created` / `Starting sync: Notes` (the `from-disk.md`
  was ingested). Status+activity updated via thread-marshaled progress, no
  crash, no mid-run recompose.
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
