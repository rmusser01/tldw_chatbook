---
id: TASK-928
title: ChatbookImporter key casing does not match what _import_chatbook passes
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 09:00'
updated_date: '2026-07-27 18:08'
labels:
  - settings
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while fixing TASK-899.

`ChatbookImporter` expects capitalized database keys (`"ChaChaNotes"`, `"Prompts"`, `"Media"`), but `Tools_Settings_Window._import_chatbook` builds its dictionary with lowercase keys (`"chachanotes"`, `"prompts"`, `"media"`).

Pre-existing and independent of the path-resolution work, so it was deliberately left alone there. The effect is that the importer does not receive the database paths under the names it looks for; confirm whether it silently falls back, imports nothing, or raises, and fix the mismatch at whichever end is the real contract.

Worth pinning with a test once resolved, since a casing mismatch between two modules is invisible to type checking and to any test that stubs one side.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The key casing agrees between `_import_chatbook` and `ChatbookImporter`
- [x] #2 The real behaviour of the current mismatch is established and recorded in the task notes
- [x] #3 A test fails if the two sides disagree again
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read ChatbookImporter's db_paths.get(...) call sites and _import_chatbook's db_paths construction to establish current casing on both sides.
2. If they already agree (via a shared canonical helper), verify no other call site still builds a mismatched dict, and reproduce live what a lowercase mismatch would do to record real behavior in the notes.
3. Fix the mismatch at whichever end is the real contract, if a mismatch is found.
4. Add an AST-based contract test (not a text/comment-matching scan) tying ChatbookImporter's actual lookups to get_chatbook_database_paths()'s actual output, plus a behavioral test documenting the mismatch failure mode.
5. Revert-check the contract test from both sides (importer key rename, canonical-helper key rename), confirm it fails for the right reason, then restore.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
No mismatch exists today; already fixed by a prior dev-reconciliation commit before this task was picked up. Every real UI call site (Tools_Settings_Window._get_chatbook_import_database_paths, ChatbookExportManagementWindow, ChatbookImportWizard x2, local_chatbook_service) builds db_paths via the single shared Chatbooks/database_paths.py::get_chatbook_database_paths() helper, which returns capitalized keys {"ChaChaNotes", "Prompts", "Media"} -- exactly matching ChatbookImporter's five self.db_paths.get(...) lookups in chatbook_importer.py. (There is a separate, intentionally-lowercase _DB_PATH_RESOLVERS map in Tools_Settings_Window.py for the unrelated DB-maintenance backup/vacuum/restore panel -- not the same contract, not in scope.)

Established real behaviour of a casing mismatch by live reproduction (not speculation): constructed a real ChatbookImporter with lowercase keys ("chachanotes"/"prompts"/"media") and ran import_chatbook() against a real sample chatbook zip with mocked CharactersRAGDB. Result: no crash, no silent success. Every content-type import method's `self.db_paths.get("ChaChaNotes")` (etc.) returns None, so each records a clean "<Name> database path not configured" error and skips that type; import_chatbook() then returns (False, "Failed to import any items from chatbook") with status.errors = ["ChaChaNotes database path not configured"]. The UI's _import_chatbook_worker surfaces this as an error toast ("Import failed: ChaChaNotes database path not configured"). This is now pinned as a permanent test (TestChatbookImporterKeyCasingMismatch.test_mismatched_lowercase_keys_fail_cleanly_not_silently_not_crashing), alongside a control test proving the correctly-cased keys import successfully.

Added test_chatbook_importer_key_lookups_match_get_chatbook_database_paths (Tests/Chatbooks/test_chatbook_importer.py): an AST-based contract test (visits actual `self.db_paths.get(<literal>)` Call nodes in chatbook_importer.py, not a source-text/comment substring match -- the codebase already has one of those for a related but different check, test_import_chatbook_paths_reuse_the_single_source_of_truth, and it would pass even if the string only appeared in a comment) asserting every key the importer actually looks up is produced by get_chatbook_database_paths(). Revert-checked from both directions: renamed a key inside get_chatbook_database_paths() -> test failed with a precise diff; separately renamed one of chatbook_importer.py's own .get("Prompts") calls to lowercase -> test failed identically. Restored both; full file green (17 passed) afterward.

No product code was changed. Modified: Tests/Chatbooks/test_chatbook_importer.py only.
<!-- SECTION:NOTES:END -->
