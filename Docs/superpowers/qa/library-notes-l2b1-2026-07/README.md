# Library Browse ▸ Notes core canvas (L2b.1) — QA evidence (2026-07-08)

Branch: `claude/library-l2b` (worktree, off origin/dev @ post-L2a merge).
Captured from textual-serve (real app CSS + worktree code via PYTHONPATH),
headless chromium, viewport 2050×1240, isolated HOME `/private/tmp/tldw-l2-qa2`
with 4 REAL notes seeded via `CharactersRAGDB.add_note` (no fixtures).

- `list-populated-2026-07-08.png` — Browse ▸ Notes renders IN-Library:
  `Notes (4)` header, filter input, `sort: Newest ▸` cycling button, four
  title+age rows. Rail row reads "in Library" (was "opens screen").
- `filter-active-2026-07-08.png` — filter `retro` submitted: status
  `filter: retro · 1 result`, list narrowed via the real notes search seam.
- `editor-2026-07-08.png` — in-canvas editor: Back, title Input, bounded body
  TextArea, keywords Input, muted meta (`Created 16h · Modified 16h · v1`),
  toolbar Save/Preview/Use in Console/Export .md/.txt/Copy with Delete
  de-emphasized at the far end. Autosave always-on (2s debounce, never
  recomposes; conflict policy Overwrite/Reload; flush on every exit).
- `preview-2026-07-08.png` — Preview ON: TextArea swaps to a Markdown render,
  button reads `Edit`. Unsaved edits survive the round-trip (pilot-covered).
- `create-templates-2026-07-08.png` — Create ▸ New note canvas: `Blank note`
  + 9 template rows ({date}/{time}/{datetime} substituted on create, matching
  the standalone screen's semantics).
- `delete-confirm-2026-07-08.png` — inline confirm: quiet copy + swapped
  Delete/Cancel toolbar; flush-before-confirm; version-checked delete.
- `filtered-delete-refreshed-2026-07-08.png` — the final-review fix proven
  live: filter → open → delete → confirm lands on `Notes (3)` with the filter
  cleared and NO ghost row; rail badge decremented.

## Verification

- All 8 build tasks task-reviewed; in-loop findings fixed + re-reviewed:
  2 Criticals (save-race version corruption; preview-save silent data loss),
  4 Importants (template {date} substitution parity, delete-confirm autosave
  leak, conflict-preview stale flag, filtered-delete ghost row) — every fix
  RED-proven by regression test.
- Final whole-branch review (opus, 15686049..60e0d512): READY FOR QA GATE.
  433 passed, 2 pre-existing skips across Tests/Library/ + test_library_shell
  + test_destination_shells + Tests/Notes/.
- Real-backend discipline: every mutation (save/create/delete/keywords,
  conflict + stale paths) pinned against a real ChaChaNotes DB; test fakes
  mirror the real seam shapes (ConflictError, bare-string create id).

## Notes to weigh at approval

- Rail badge `Notes (4+)` vs canvas `Notes (4)`: plain-list responses mark the
  total unknown — pre-existing `(N+)` honesty grammar, not new.
- The standalone Notes screen is UNCHANGED this phase (deprecation is L2b.2);
  discovered along the way: its Export has a latent runtime TypeError
  (FileSave kwargs) — tracked follow-up, moot once L2b.2 deprecates it.
- Sync + Templates management + Import Note arrive in L2b.2 per the spec's
  parity split.
