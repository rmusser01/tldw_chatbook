# Fresh-install data root recovery (TASK-32011)

The authorized outcome is an application fix for the reported fresh-install
failure, requiring no reporter response or manual permission repair.

ADR required: yes
ADR path: backlog/decisions/127-fresh-install-private-data-root-recovery.md
Reason: Durable selection of an alternate default storage location under ADR-029.

## Design

Add `_secure_default_data_dir()` beside the existing conventional-path function
in `config.py`. Keep `_default_base_data_dir()` pure and compatible with existing
callers. `get_user_data_dir()` delegates only its default branch to the new
resolver; configured roots and profile child hardening are unchanged.
Share a read-only `_selected_default_base_data_dir()` helper with Settings storage
diagnostics, so displaying the active path honors the durable fallback without
creating directories. This integration requirement was found during review and
added to TASK-32011 acceptance criteria before changing the Settings implementation.

1. Derive the conventional root from the existing HOME-aware function and the
   fallback `~/.tldw_cli-data` from that same home.
2. If the fallback entry exists, refuse if the conventional entry also exists;
   otherwise secure/reuse the fallback. Presence pins selection across processes.
3. Otherwise secure/create the conventional root normally.
4. Only on `UNSAFE_PARENT/shared_writable_parent`, probe the conventional entry.
   If absent, secure/create the fallback with the unchanged private-path helper.
   Existing entries and probe errors cannot trigger a fresh empty profile.
5. Append and secure the same user-profile child as before.

No changes to the directory guard, config-file selection, account/group trust,
database formats, profile naming, explicit storage settings, or unrelated files.
Two existing default roots are an explicit conflict, never an implicit migration.

## Implementation and verification plan

1. Write regression tests in `Tests/test_database_path_privacy.py`: conventional
   fresh bootstrap under permissive umask; writable `.local`/`share`; fallback
   private modes and persistence; existing conventional and ambiguous roots;
   unsafe home, links, explicit overrides, and unrelated failures.
2. Add a fresh-process config-import test using an isolated HOME/config. Persist
   a SQLite row through the private SQLite owner, restart, repair ancestor modes,
   and prove the same data/root is retained.
3. Run the new tests red against the unchanged resolver.
4. Implement the helper and default-branch delegation, then run focused tests.
5. Run existing config/private-path/private-SQLite/profile-isolation checks and
   relevant architecture checks. Run Ruff and formatting checks on changed code;
   preserve unrelated formatting in the large config module.
6. Document the fallback and conflicts in README, self-review the diff, and record
   exact test evidence in TASK-32011. No full-suite sweep is authorized.

## Qodo review follow-up

1. Rebase on latest dev, then add a deterministic real two-process regression for
   a permission repair between conventional-root refusal and fallback creation.
2. Hold a private HOME-level interprocess lock across selection, root creation,
   and profile-child creation. Preserve the read-only Settings helper.
3. Validate probe paths without resolving, share the fallback directory name,
   and use the SQLite connection transaction context in the subprocess test.
4. Route explicitly named prompt exports through the selected runtime base,
   update the compatibility-consumer architecture assertion, and cover custom and
   fallback roots with actual prompt database files.
5. Run the label-gated workflow on synchronize events too, retaining exact-head
   checkout. Include the new race and export regressions in the Linux matrix.
6. Reply to all six Qodo threads with evidence, rerun required checks on the
   rebased head, and merge only when review findings and required gates are clear.
