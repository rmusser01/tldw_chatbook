# TASK-19873 Dead CCP Handler Retirement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Completely retire two unconstructed CCP handlers and five broken,
unrouted Tools Settings operation families while preserving every reachable CCP,
Chatbooks, bulk-maintenance, and shared private-SQLite behavior.

**Architecture:** Treat absence as the new contract. Add positive retirement
guards before deleting modules, exports, UI controls, dispatch branches, worker
families, and orphan owner policies. Keep the live CCP character/persona path,
canonical Chatbooks import path, bulk database maintenance, and the generic
private-SQLite seams; retarget shared seam tests to retained owner policies.

**Tech Stack:** Python 3.11, Textual 8.x, pytest, Ruff, Backlog.md CLI, repository
diagnostic-inventory and boot-budget scripts.

**Design:**
`Docs/superpowers/specs/2026-08-29-task-19873-dead-ccp-handlers-retirement-design.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This is dead-code removal that enforces existing navigation and
ownership boundaries. It introduces no storage, service, security, runtime, or
cross-module decision.

---

## Preconditions and verification policy

- Work only in
  `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-19873-retire-dead-ccp`
  on `codex/task-19873-retire-dead-ccp`.
- Use the shared environment at
  `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`.
- Do not run the full repository test suite; TASK-19873 has a focused gate.
- The untouched `origin/dev` baseline has one known failure in
  `test_restore_refuses_a_dangerous_backup_path_via_path_validation`: mounting
  the app creates the ChaChaNotes database before the test's precondition. The
  test belongs solely to the single-restore family being deleted. Do not repair
  or preserve that retired contract.
- After every production edit, run the smallest named test first, then the
  relevant focused module.
- Preserve historical task files and superseded plans/specs. Update only current
  architecture documents and generated inventories.

## Task 1: Make CCP handler retirement an explicit tested contract

**Files:**

- Modify: `Tests/UI/test_legacy_entrypoints_retired.py`
- Modify: `Tests/UI/test_ccp_handlers.py`
- Modify: `Tests/UI/test_file_picker_filters_callable.py`
- Modify: `tldw_chatbook/UI/CCP_Modules/__init__.py`
- Delete: `tldw_chatbook/UI/CCP_Modules/ccp_conversation_handler.py`
- Delete: `tldw_chatbook/UI/CCP_Modules/ccp_dictionary_handler.py`

- [ ] **Step 1: Add failing positive retirement guards**

  In `test_legacy_entrypoints_retired.py`:

  - add `tldw_chatbook.UI.CCP_Modules.ccp_conversation_handler` and
    `tldw_chatbook.UI.CCP_Modules.ccp_dictionary_handler` to `RETIRED_MODULES`;
  - add both source paths to `RETIRED_FILES`;
  - remove both from the tuple that describes live CCP handler files;
  - add `ccp_persona_handler.py` to that live tuple so its inventory exactly
    matches the character/persona production pair rather than merely shrinking.

- [ ] **Step 2: Prove the old tree violates the new contract**

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
    Tests/UI/test_legacy_entrypoints_retired.py -q
  ```

  Expected: FAIL because both modules and files still exist.

- [ ] **Step 3: Delete the dead modules, exports, and handler-only tests**

  - remove both imports and `__all__` entries from `CCP_Modules/__init__.py`;
  - delete both handler modules;
  - remove the conversation-handler import, its dedicated test class, and its
    stale-result test from `test_ccp_handlers.py`;
  - remove the dictionary-handler case from the callable file-picker
    parameterization;
  - retain live character/persona tests and standalone dictionary-library tests.

- [ ] **Step 4: Verify retirement and live CCP behavior**

  First rerun the exact RED node:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
    Tests/UI/test_legacy_entrypoints_retired.py::test_retired_legacy_entrypoint_modules_are_not_importable \
    Tests/UI/test_legacy_entrypoints_retired.py::test_retired_legacy_entrypoint_files_are_removed -q
  ```

  Expected: PASS.

  Then run the broader CCP slice:

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
    Tests/UI/test_legacy_entrypoints_retired.py \
    Tests/UI/test_ccp_handlers.py \
    Tests/UI/test_file_picker_filters_callable.py -q
  ```

  Expected: PASS.

- [ ] **Step 5: Commit the CCP deletion**

  ```bash
  git add Tests/UI/test_legacy_entrypoints_retired.py \
    Tests/UI/test_ccp_handlers.py \
    Tests/UI/test_file_picker_filters_callable.py \
    tldw_chatbook/UI/CCP_Modules
  git commit -m "refactor(ccp): retire unused handlers"
  ```

## Task 2: Retire the five broken Tools Settings operation families

**Files:**

- Modify: `Tests/UI/test_tools_settings_window.py`
- Modify: `tldw_chatbook/UI/Tools_Settings_Window.py`

- [ ] **Step 1: Rewrite the composition contract before production deletion**

  Update `test_database_tools_composition` so it asserts:

  - individual vacuum, backup, restore, integrity-check, and legacy Chatbook
    import controls are absent;
  - all per-database `Last Backup` labels are absent;
  - retained bulk backup/vacuum/integrity controls and retained advanced/status
    controls are still present.

  Fold `test_import_chatbook_button` into the absence contract. Delete tests
  whose sole subject is a retired individual/import family, including the known
  baseline-failing dangerous-restore-path test. Keep `_get_database_path`,
  configuration, bulk backup/vacuum/integrity, export, and generic window tests.

  Add a separate
  `test_retired_database_tool_operations_are_absent` source/attribute guard that:

  - asserts `ToolsSettingsWindow` has no `_vacuum_single_database`,
    `_vacuum_single_worker`, `_backup_single_database`,
    `_backup_single_worker`, `_restore_single_database`,
    `_restore_single_worker`, `_check_single_database`,
    `_check_single_worker`, `_import_chatbook`, `_import_chatbook_worker`,
    `_get_chatbook_import_database_paths`, `_validate_maintenance_path`,
    `_get_schema_version`, or `_update_last_backup_status` attributes;
  - inspects `on_button_pressed` and asserts it contains no retired individual
    button IDs and no calls to those operation wrappers.

- [ ] **Step 2: Prove the old UI violates the new contract**

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
    Tests/UI/test_tools_settings_window.py::test_database_tools_composition -q
  ```

  Expected: FAIL because the retired controls and labels still compose.

  Run the new method/dispatcher guard as a second RED node:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
    Tests/UI/test_tools_settings_window.py::test_retired_database_tool_operations_are_absent -q
  ```

  Expected: FAIL because the methods and dispatch branches still exist.

- [ ] **Step 3: Remove the UI and dispatch surface**

  In `Tools_Settings_Window.py`:

  - update database-tools copy so it no longer promises individual maintenance;
  - remove the five retired control families and their `Last Backup` labels;
  - remove their button-dispatch branches;
  - remove legacy Chatbook import picker/worker/helpers;
  - remove individual vacuum, backup, restore, and integrity picker/wrapper/worker
    methods;
  - remove orphan helpers `_validate_maintenance_path`, `_get_schema_version`,
    and `_update_last_backup_status`;
  - remove imports used only by deleted code, including
    `restore_private_sqlite` and `validate_path_simple` if reference search
    confirms no retained caller;
  - update the bulk-backup comment that names deleted single-operation methods.

  Do not change `_get_database_path`, bulk backup/vacuum/integrity workers, or
  unrelated advanced utilities.

- [ ] **Step 4: Run the smallest retained behavior checks**

  First rerun both exact RED nodes:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
    Tests/UI/test_tools_settings_window.py::test_database_tools_composition \
    Tests/UI/test_tools_settings_window.py::test_retired_database_tool_operations_are_absent -q
  ```

  Expected: PASS.

  Then run the smallest retained bulk behavior checks:

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
    Tests/UI/test_tools_settings_window.py::test_database_tools_composition \
    Tests/UI/test_tools_settings_window.py::test_vacuum_worker_operates_on_resolved_paths_not_literals \
    Tests/UI/test_tools_settings_window.py::test_vacuum_all_fails_loudly_for_an_unresolvable_database \
    Tests/UI/test_tools_settings_window.py::test_integrity_all_fails_loudly_for_an_unresolvable_database \
    Tests/UI/test_tools_settings_window.py::test_backup_all_fails_loudly_for_an_unresolvable_database -q
  ```

  Expected: PASS.

- [ ] **Step 5: Run the full focused Tools Settings module**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
    Tests/UI/test_tools_settings_window.py -q
  ```

  Expected: PASS with the invalid retired-path test absent.

- [ ] **Step 6: Commit the deprecated-operation deletion**

  ```bash
  git add tldw_chatbook/UI/Tools_Settings_Window.py \
    Tests/UI/test_tools_settings_window.py
  git commit -m "refactor(settings): retire dead database operations"
  ```

## Task 3: Remove orphan SQLite owner policies without weakening shared seam tests

**Files:**

- Modify: `Tests/DB/test_private_sqlite.py`
- Modify: `Tests/DB/test_private_sqlite_inventory.py`
- Modify: `Tests/DB/test_pragma_settings.py`
- Modify: `tldw_chatbook/DB/private_sqlite.py`
- Modify: `backlog/docs/sqlite-private-owner-inventory.md`

- [ ] **Step 1: Add the policy-retirement assertion first**

  Add
  `test_retired_settings_owner_policies_are_absent` asserting that these owner
  IDs are absent:

  - `settings.schema`
  - `settings.single_backup`
  - `settings.pre_restore_backup`
  - `settings.restore`

  Assert in the same test that `settings.bulk_backup`, `settings.vacuum`, and
  `settings.integrity` remain registered.

- [ ] **Step 2: Prove the registry still violates the contract**

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
    Tests/DB/test_private_sqlite.py::test_retired_settings_owner_policies_are_absent -q
  ```

  Expected: FAIL because the four retired policies remain registered.

- [ ] **Step 3: Retarget generic backup/restore tests to retained owners**

  - remove `settings.single_backup` from `COPY_BACKUP_OWNER_IDS` and replace
    hard-coded copies with `settings.bulk_backup` where the behavior is generic;
  - run generic `restore_private_sqlite` tests with
    `tts.profile_restore_stage` for both the restore and safety-snapshot owner
    arguments. That retained policy has the required private/read-only kinds and
    centralized-backup permission;
  - use `tts.profile_restore_stage` for setup and readback connections in the
    behavioral restore-owner test as well;
  - remove `RESTORE_BACKUP_OWNER_IDS` from backup-policy coverage;
  - in the close-failure instrumentation test, identify the safety connection
    with `Path(database) == pre_restore` rather than by owner ID, because both
    restore arguments now deliberately share the same retained owner.

  Preserve the generic seam's transactional, race, mode, rollback, cleanup, and
  alias-safety tests; only their owner labels change.

- [ ] **Step 4: Delete registry policies and update the executable inventory**

  - remove the four orphan policies from `SQLITE_OWNER_REGISTRY`;
  - delete matching C10, B10-B12, B16, P21, and P22 inventory rows;
  - change C08/C09 evidence from `_vacuum_single_worker`/
    `_check_single_worker` to retained `_vacuum_worker`/`_integrity_worker`;
  - mirror those exact changes in `test_private_sqlite_inventory.py`;
  - update the current pragma-test narrative so it no longer names retired
    policies.

- [ ] **Step 5: Verify registry and shared storage behavior**

  First rerun the exact RED node:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
    Tests/DB/test_private_sqlite.py::test_retired_settings_owner_policies_are_absent -q
  ```

  Expected: PASS.

  Then run the broader storage slice:

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
    Tests/DB/test_private_sqlite.py \
    Tests/DB/test_private_sqlite_inventory.py \
    Tests/DB/test_pragma_settings.py -q
  ```

  Expected: PASS.

- [ ] **Step 6: Commit the owner-policy cleanup**

  ```bash
  git add tldw_chatbook/DB/private_sqlite.py \
    Tests/DB/test_private_sqlite.py \
    Tests/DB/test_private_sqlite_inventory.py \
    Tests/DB/test_pragma_settings.py \
    backlog/docs/sqlite-private-owner-inventory.md
  git commit -m "refactor(sqlite): remove retired settings owners"
  ```

## Task 4: Correct current architecture documentation

**Files:**

- Modify: `Docs/Development/Chatbook/Chatbooks-DatabaseTools-Implementation.md`
- Modify: `Docs/Development/ccp-refactoring-complete.md`
- Modify: `Docs/Parity/2026-04-19-data-compatibility-map.md`
- Modify: `Tests/Chatbooks/test_chatbook_importer.py`

- [ ] **Step 1: Update only current-state claims**

  - document bulk/advanced database tools as the retained deprecated-window
    behavior and the canonical Chatbooks screen as the import owner;
  - remove conversation/dictionary handlers from current CCP architecture;
  - remove the conversation handler from the live data-compatibility map;
  - correct the Chatbook importer test comment that names the retired legacy
    picker.

  Do not rewrite historical task files or superseded plans/specs.

- [ ] **Step 2: Commit documentation corrections**

  ```bash
  git add Docs/Development/Chatbook/Chatbooks-DatabaseTools-Implementation.md \
    Docs/Development/ccp-refactoring-complete.md \
    Docs/Parity/2026-04-19-data-compatibility-map.md \
    Tests/Chatbooks/test_chatbook_importer.py
  git commit -m "docs(task-19873): record retired legacy paths"
  ```

## Task 5: Regenerate only affected generated artifacts

**Files:**

- Modify: `Docs/security/production-diagnostic-inventory.json`
- Modify: `Tests/Performance/boot_budget_snapshots/preimport_payload.json`

- [ ] **Step 1: Run Ruff before regenerating the indentation-sensitive inventory**

  Run the exact modified-Python-file gate:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
    tldw_chatbook/UI/CCP_Modules/__init__.py \
    tldw_chatbook/UI/Tools_Settings_Window.py \
    tldw_chatbook/DB/private_sqlite.py \
    Tests/UI/test_legacy_entrypoints_retired.py \
    Tests/UI/test_ccp_handlers.py \
    Tests/UI/test_file_picker_filters_callable.py \
    Tests/UI/test_tools_settings_window.py \
    Tests/DB/test_private_sqlite.py \
    Tests/DB/test_private_sqlite_inventory.py \
    Tests/DB/test_pragma_settings.py \
    Tests/Chatbooks/test_chatbook_importer.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
    tldw_chatbook/UI/CCP_Modules/__init__.py \
    tldw_chatbook/UI/Tools_Settings_Window.py \
    tldw_chatbook/DB/private_sqlite.py \
    Tests/UI/test_legacy_entrypoints_retired.py \
    Tests/UI/test_ccp_handlers.py \
    Tests/UI/test_file_picker_filters_callable.py \
    Tests/UI/test_tools_settings_window.py \
    Tests/DB/test_private_sqlite.py \
    Tests/DB/test_private_sqlite_inventory.py \
    Tests/DB/test_pragma_settings.py \
    Tests/Chatbooks/test_chatbook_importer.py
  ```

  Expected: PASS before inventory regeneration.

- [ ] **Step 2: Confirm diagnostic drift before blessing it**

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
    scripts/check_persistent_diagnostic_inventory.py
  ```

  Expected: FAIL with only deleted-handler/deleted-Tools-Settings statement
  drift. Inspect the reported statements before proceeding.

- [ ] **Step 3: Regenerate and verify the diagnostic inventory**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
    scripts/check_persistent_diagnostic_inventory.py --write
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
    scripts/check_persistent_diagnostic_inventory.py
  ```

  Expected: PASS after regeneration.

- [ ] **Step 4: Run narrow post-regeneration retirement audits**

  Each no-output search below is restricted to an owner file or current-state
  document where the retired name has no legitimate meaning. It deliberately
  does not search the canonical Chatbooks implementation, unrelated schema
  helpers, historical records, or the positive absence tests.

  ```bash
  rg -n "CCPConversationHandler|CCPDictionaryHandler|ccp_(conversation|dictionary)_handler" \
    tldw_chatbook/UI/CCP_Modules/__init__.py \
    Docs/Development/ccp-refactoring-complete.md \
    Docs/Parity/2026-04-19-data-compatibility-map.md
  rg -n "settings\.(schema|single_backup|pre_restore_backup|restore)" \
    tldw_chatbook/DB/private_sqlite.py \
    Tests/DB/test_private_sqlite_inventory.py \
    backlog/docs/sqlite-private-owner-inventory.md \
    Docs/Development/Chatbook/Chatbooks-DatabaseTools-Implementation.md
  rg -n "_(vacuum|backup|restore|check)_single_(database|worker)|_import_chatbook(_worker)?|_get_chatbook_import_database_paths|_validate_maintenance_path|_get_schema_version|_update_last_backup_status|db-last-backup-" \
    tldw_chatbook/UI/Tools_Settings_Window.py
  rg -n "CCPConversationHandler|CCPDictionaryHandler|ccp_(conversation|dictionary)_handler|_(vacuum|backup|restore|check)_single_(database|worker)|ToolsSettingsWindow\._import_chatbook(_worker)?|_get_chatbook_import_database_paths|_validate_maintenance_path|_get_schema_version|_update_last_backup_status" \
    Docs/security/production-diagnostic-inventory.json
  ```

  Expected: every command produces no output. Separately inspect the focused
  retirement tests and confirm deleted names occur only in their positive
  absence lists/assertions.

- [ ] **Step 5: Re-measure only the CCP pre-import budget**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
    scripts/update_boot_budget_snapshots.py --only preimport
  git diff -- Tests/Performance/boot_budget_snapshots
  ```

  Confirm that only `preimport_payload.json` changed and that the removed CCP
  modules account for the intended census reduction. Do not run the updater for
  import-weight, CSS, or UI-ready snapshots.

- [ ] **Step 6: Verify the generated-artifact guards**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
    Tests/Performance/test_screen_preimport_payload_budget.py -q
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
    scripts/check_persistent_diagnostic_inventory.py
  ```

  Expected: PASS.

- [ ] **Step 7: Commit generated artifacts**

  ```bash
  git add Docs/security/production-diagnostic-inventory.json \
    Tests/Performance/boot_budget_snapshots/preimport_payload.json
  git commit -m "chore(task-19873): refresh affected inventories"
  ```

## Task 6: Complete task evidence and run the focused final gate

**Files:**

- Modify: `backlog/tasks/task-19873 - Decide-the-fate-of-two-CCP-handlers-that-have-never-been-able-to-run.md`

- [ ] **Step 1: Run the complete focused test gate**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
    Tests/UI/test_legacy_entrypoints_retired.py \
    Tests/UI/test_ccp_handlers.py \
    Tests/UI/test_file_picker_filters_callable.py \
    Tests/UI/test_tools_settings_window.py \
    Tests/DB/test_private_sqlite.py \
    Tests/DB/test_private_sqlite_inventory.py \
    Tests/DB/test_pragma_settings.py \
    Tests/Chatbooks/test_chatbook_importer.py \
    Tests/Performance/test_screen_preimport_payload_budget.py -q
  ```

  Expected: PASS. Do not run the full suite unless the user separately opts in.

- [ ] **Step 2: Re-run static, diagnostic, and diff checks**

  Re-run the exact Ruff file list from Task 5, Step 1, then run the diagnostic
  checker after Ruff so line/indentation-sensitive inventory evidence is final:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
    tldw_chatbook/UI/CCP_Modules/__init__.py tldw_chatbook/UI/Tools_Settings_Window.py \
    tldw_chatbook/DB/private_sqlite.py Tests/UI/test_legacy_entrypoints_retired.py \
    Tests/UI/test_ccp_handlers.py Tests/UI/test_file_picker_filters_callable.py \
    Tests/UI/test_tools_settings_window.py Tests/DB/test_private_sqlite.py \
    Tests/DB/test_private_sqlite_inventory.py Tests/DB/test_pragma_settings.py \
    Tests/Chatbooks/test_chatbook_importer.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
    tldw_chatbook/UI/CCP_Modules/__init__.py tldw_chatbook/UI/Tools_Settings_Window.py \
    tldw_chatbook/DB/private_sqlite.py Tests/UI/test_legacy_entrypoints_retired.py \
    Tests/UI/test_ccp_handlers.py Tests/UI/test_file_picker_filters_callable.py \
    Tests/UI/test_tools_settings_window.py Tests/DB/test_private_sqlite.py \
    Tests/DB/test_private_sqlite_inventory.py Tests/DB/test_pragma_settings.py \
    Tests/Chatbooks/test_chatbook_importer.py
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
    scripts/check_persistent_diagnostic_inventory.py
  git diff --check
  git diff origin/dev...HEAD --check
  ```

  Expected: all PASS.

- [ ] **Step 3: Self-review the deletion boundary**

  Inspect `git diff --stat origin/dev...HEAD` and
  `git diff origin/dev...HEAD`. Confirm:

  - no reachable CCP, canonical Chatbooks, bulk database-maintenance, or shared
    SQLite behavior was removed;
  - no compatibility export, broken dispatch, writerless label, or orphan owner
    policy remains;
  - generated diffs contain only reviewed consequences of the deletions;
  - no unrelated formatting or historical-document churn entered the branch.

- [ ] **Step 4: Complete TASK-19873 hygiene**

  - check all five acceptance criteria;
  - add concise implementation notes preserving the evidence that the paths had
    no production construction/routing path, that TASK-19563 repaired only dead
    code, and that the invalid baseline restore test was removed with its retired
    contract;
  - record focused test, Ruff, diagnostic, boot-budget, and diff-check evidence;
  - record `ADR required: no`, `ADR path: N/A`, and the reason;
  - directly edit the normalized TASK-19873 Markdown source to set
    `status: Done`; do not run `backlog task edit` for this five-digit ID.
    `backlog/docs/lessons-backlog-hygiene.md` records that some CLI builds
    silently create `task-task- - .md` for five-digit task edits.
  - before staging closeout, verify no stray file exists:

    ```bash
    test -z "$(find backlog/tasks -maxdepth 1 -name 'task-task- - .md' -print -quit)"
    ```

- [ ] **Step 5: Commit task closeout**

  ```bash
  git add backlog/tasks/task-19873*
  git commit -m "docs(task-19873): close dead-code decision"
  ```

## Task 7: Rebase, reverify, and prepare the pull request

- [ ] **Step 1: Re-fetch and compare `origin/dev`**

  ```bash
  git fetch origin dev
  git merge-base --is-ancestor origin/dev HEAD
  ```

  If `origin/dev` advanced, rebase onto it, push with an exact
  `--force-with-lease=<branch>:<observed-remote-sha>`, and rerun Tasks 5-6's
  focused generated-artifact, test, Ruff, and diff checks.

- [ ] **Step 2: Push and open a PR**

  Push `codex/task-19873-retire-dead-ccp`, open a PR against `dev`, and use the
  TASK-19873 implementation notes plus verification evidence as the PR body.

- [ ] **Step 3: Review the published diff and checks**

  Reinspect the PR diff and every current-head check. Validate review findings
  against the code before changing anything; fix only branch-caused issues with
  targeted regression tests.
