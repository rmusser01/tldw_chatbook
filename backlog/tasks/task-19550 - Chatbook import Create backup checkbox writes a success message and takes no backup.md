---
id: TASK-19550
title: >-
  Chatbook import "Create backup" checkbox writes a success message and takes no
  backup
status: Done
assignee:
  - '@claude'
created_date: '2026-08-21 20:00'
updated_date: '2026-08-21 22:30'
labels:
  - chatbooks
  - data-loss
  - honesty
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 6 (UX coherence / error handling /
honesty) — its **P0**. CONFIRMED, and re-verified at this branch base.

`tldw_chatbook/UI/Wizards/ChatbookImportWizard.py` composes a
`Checkbox("Create backup", value=True)` (line 553) under a heading offering to
back the database up before importing. The value **is** read at import time,
and the import path then writes a completed status row:

```
751:            # Create backup if requested
755:                # TODO: Implement actual backup functionality
757:                self._update_status("status-backup", "completed", "✓ Created backup")
```

No backup is taken. This is a default-ON safety control that displays a
spinner, then asserts an outcome it did not produce, immediately before a
**database-mutating import that has no rollback**. A user who loses data during
or after an import will look for the backup the wizard told them it made.

This is the sharpest instance of the theme Lane 6 identified as its most
actionable output: *the app asserts outcomes it did not produce* — a backup
that was not taken, an export that dropped items and said "successfully", a
sync that discarded an edit and said "no changes" (see also TASK-19554, the
Notes-sync `DISK_WINS` silent no-op, which converges with this).

**Disposition (the lane's own, and it matters):** REMOVE the checkbox and the
fake status row. Do not merely disable it — a greyed-out "Create backup" box
reads as "backup already handled". Per the owner's standing ruling
(durable/pragmatic over clever/unstable): removing a lying control is
strictly better than shipping a hurried backup implementation behind it. If a
real backup is wanted, it is a separate, deliberately scoped piece of work.

## Acceptance Criteria

- [x] The import wizard no longer presents any control that offers to back up
      the database unless a backup is actually performed
- [x] The wizard never emits "✓ Created backup" (or any equivalent success
      claim) for work it did not do
- [x] The remaining status rows in the import flow are audited: every one that
      reports "completed" corresponds to work that actually ran
- [x] A test fails if a status row can report success on a code path whose
      implementation is a `TODO`/no-op
- [x] The user-facing risk is stated honestly somewhere the user sees it
      before confirming: this import mutates the database and cannot be
      rolled back
- [x] `Docs/User_Guide/` is updated wherever it describes the import wizard's
      backup behaviour

## Implementation Plan

1. Decide the disposition on evidence, not preference: search the repo for a
   backup primitive the wizard could actually *call* (DB `backup_database`,
   `private_sqlite.copy_private_sqlite`, the Settings backup workers, any
   `.bak` convention) and cost out what a real, non-lying backup here would
   need. Record the finding either way.
2. Write the born-red tests first, using the repo's wizard idioms
   (`Tests/Wizards/test_first_run_setup_wizard.py`'s single-step host app;
   `Tests/Chatbooks/test_chatbook_import_wizard_validation.py`'s no-mount
   style for pure logic):
   - options step composes no backup control and `get_step_data()` carries
     no `create_backup` key;
   - options step states the irreversibility risk before the user confirms;
   - driving `ImportProgressStep._import_chatbook` end-to-end never emits a
     backup status row, even when a stale `create_backup=True` option is
     present;
   - a static guard (AC#4): no function in the module that emits a
     `"completed"` status row may contain a `TODO`/"For now"/"not
     implemented" marker, and the set of status ids that can be marked
     completed is pinned to an audited allowlist.
3. Show the tests red at the branch base.
4. Apply the disposition in `ChatbookImportWizard.py`.
5. Audit every other `_update_status(..., "completed", ...)` in the file
   against the work that actually runs; fix what is trivially in scope,
   report the rest for filing.
6. Green the tests; run `Tests/Chatbooks/`, `Tests/Wizards/`, the UI tests
   that touch this wizard, and a repo-wide `--collect-only -q`.
7. Check `Docs/User_Guide/` for any page describing this wizard's backup
   behaviour; update it if one exists.

## Implementation Notes

Removed the lying control (the lane's filed disposition), and replaced it with
the fact it was hiding: the import writes straight into the live databases and
cannot be undone. Gone from `ChatbookImportWizard.py`: the `Create backup`
checkbox and its "Backup current database before importing" caption, the
`create_backup` options key, the `○ Creating backup` progress row, and the
`TODO`-backed block that painted `✓ Created backup`. Added an
`#import-irreversible-warning` line at the top of the Import Options step (the
last screen the user can still cancel from — the next step starts writing on
`on_show`).

### Why remove rather than implement

The controller's bar for implementing was "only if the repo already has a
safe, reachable primitive you can call". It does not, quite:

- The primitives are owner-gated. `ChaChaNotes_DB.backup_database`,
  `Client_Media_DB_v2.backup_database` and `Prompts_DB.backup_database` all
  route through `DB/private_sqlite.py`, whose `SQLITE_OWNER_REGISTRY` keys each
  backup to a named owning module, pinned by
  `Tests/DB/test_private_sqlite_inventory.py:1009` against
  `backlog/docs/sqlite-private-owner-inventory.md`. Reusing
  `settings.bulk_backup` (owned by `UI/Tools_Settings_Window`) from the wizard
  would be registry misuse.
  **Correction (review finding — this bullet originally overstated the case):**
  a new registry row is NOT required. `db.chachanotes.backup`, `db.media.backup`
  and `db.prompts.backup` already exist with `centralized_backup_allowed=True`,
  already back one-line public `backup_database(path)` methods that carry their
  own tests, and the app already holds live instances of all three reachable
  from the wizard. A minimal three-call backup was therefore available. The
  REMOVE disposition does not rest on this bullet — it rests on the two below
  (a naive three-call backup has no staging or atomicity, unlike the Settings
  implementation) and, decisively, on the absence of any user-reachable
  RESTORE. Recorded so a future task does not inherit the wrong premise.
- A correct backup here is three databases, not one. `get_chatbook_database_paths()`
  hands the importer ChaChaNotes, Prompts **and** Media; backing up a subset
  and saying "✓ Created backup" is the same defect with a smaller blast
  radius. The Media DB is also the one that can be multi-GB, inside a worker
  with no cancel and no disk-space check.
- The only correct implementation in-tree is ~120 lines. `Tools_Settings_Window`'s
  `_backup_*_worker` shows what it actually takes: profile-scoped root,
  `mkstemp` staging, atomic `os.replace`, cancellation checks between every
  file, unlink of published artifacts if any later file fails, a manifest, and
  a loud failure when a source path will not resolve. Extracting that out of a
  retired 7.7k-line screen, or reimplementing a subset in a wizard, is exactly
  the "hurried backup implementation" the lane warned about.
- There is nowhere to restore it from. The DB backup/restore UI lives in
  `ToolsSettingsWindow`, which is retired and nav-unreachable (its route
  resolves to the MCP screen). Shipping a backup with no in-app restore would
  need its own UX work to be worth the checkbox.

That is a deliberately scoped piece of work, which is what the task says. Per
the owner's standing stability-over-quick-wins ruling, removing the lying
control and stating the risk is the durable answer today.

### Sibling status-row audit (AC#3) — every `"completed"` row in the file

Verified true, keeping their rows:

- `status-prepare` → "✓ Prepared import" / "✓ Submitted server import": the
  importer really is constructed / the server really returned a `job_id`
  (missing id raises).
- `status-indexes` → "✓ Server import completed": gated on a terminal
  `completed|success` status; anything else raises.
- `status-indexes` → "✓ Updated indexes" (local): the wizard does no index
  work itself, but the claim is true — ChaChaNotes carries 100 `CREATE TRIGGER`
  statements maintaining FTS5 inside the import's own writes.
- `status-finalize` → "✓ Import completed" / "✓ Import finalized": reached only
  on the importer's success return.

**Found, not fixed here — recommend filing (same "asserts outcomes it did not
produce" family):**

1. **"✓ Imported conversations/notes/characters/media" can be false.**
   `ChatbookImporter.import_chatbook` returns `success=True` when everything
   was *skipped* (`chatbook_importer.py:399` — `success = (successful_items +
   skipped_items) > 0 or total_items == 0`), and its own honest message
   ("Skipped N/N items due to conflicts") is discarded by the wizard on the
   success branch. The four per-type rows are gated on **manifest counts**, not
   on results, so re-importing an already-imported chatbook under the default
   `Skip existing items` strategy paints four "✓ Imported …" rows and
   "✅ Import Completed Successfully!" over `Imported: 0 / Skipped: N`. Not
   fixed here because `ImportStatus` has no per-type counters — an honest row
   needs per-type results plumbed through the importer, and a whole-import-
   granularity patch would just move the overstatement to the partial case
   (3 of 5 skipped still reads "✓ Imported"). Half-measures were explicitly
   out of scope.
2. **Two dead controls.** `preserve_timestamps` and `import_tags` are read into
   the options dict and consumed by nothing anywhere in the repo (grep:
   `ChatbookImportWizard.py` is the only hit for either).
3. **One mislabelled default-ON control.** "Merge with existing tags"
   ("Combine imported tags with any existing tags") is passed as
   `prefix_imported`, whose only effect is to rename every imported item
   `[Imported] <name>` (`chatbook_importer.py:507,1180,1294,1447`). It does not
   touch tags at all.

### Evidence

- Born red at base, all four for the defect itself: the options step listed
  `'create backup'`; no irreversibility copy was rendered; the driven import
  emitted `status-backup`; and the AST guard reported
  `_import_chatbook: TODO, For now`.
- Green after: 4/4. `Tests/Chatbooks/` 253 passed 1 skipped;
  `Tests/Wizards/` + the three UI files touching this wizard 900 passed;
  repo-wide `--collect-only -q` 53,633 collected, exit 0. `ruff check` clean
  (the file's one pre-existing `ruff format` deviation at line ~286 was left
  alone rather than reformatting unrelated lines).
- AC#6 is vacuous, checked rather than assumed: no `Docs/User_Guide/` page
  documents this wizard or its backup behaviour (the Chatbooks screen has no
  guide page; `library/import-and-export.md` covers the Library's own
  import/export, and `artifacts.md` is a stub). Nothing to update or restamp.

### Files

- `tldw_chatbook/UI/Wizards/ChatbookImportWizard.py`
- `Tests/Chatbooks/test_chatbook_import_wizard_backup_honesty.py` (new)
