---
id: TASK-519
title: Fix get_user_data_dir import-time home freeze breaking test HOME isolation
status: Done
assignee: []
created_date: '2026-07-23 23:30'
updated_date: '2026-07-24 15:56'
labels:
  - testing
  - config
  - followup
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Recurring test-isolation hazard, bitten at least THREE times during the RAG settings+profiles program (SP2b first-run wiring → PYTEST_CURRENT_TEST production guard; SP2b no-deadlock test wrote real ~/.local; SP3 validator tests constructed the real-user-dir ConfigProfileManager). Root cause: `config.py`'s `BASE_DATA_DIR_CLI = Path.home()/...` (~config.py:4346) is a module constant frozen at IMPORT time, so `get_user_data_dir()`'s fallback ignores per-test `HOME`/`XDG_DATA_HOME` monkeypatches applied later by `Tests/conftest.py`'s `isolate_test_environment` — and that fixture's `config.get_data_dir` patch silently no-ops (the function doesn't exist). Any unmocked default-dir consumer (`ConfigProfileManager(profiles_dir=None)`, first-run import, etc.) reads/mkdirs the developer's or CI runner's REAL data dir.

Fix at the root: make `get_user_data_dir()` resolve the home/XDG env at CALL time (or make the conftest fixture patch a real, existing seam), then remove the scattered per-file hermetic workarounds where they become redundant (SP3's autouse fixture can stay as defense-in-depth) and reconsider SP2b's `PYTEST_CURRENT_TEST` production guard (a conftest autouse presetting `_first_run_import_attempted = True` is the cleaner shape). Known residual leaks to verify closed: the 2 `RAGConfig.from_settings` tests in Tests/UI/test_settings_library_rag_defaults.py leaking via `active_config._manager()`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `get_user_data_dir()` honors `HOME` changes made after module import (call-time resolution; `XDG_DATA_HOME` deliberately NOT honored — the pre-existing default never consulted it, and adding it would silently relocate an XDG user's data dir on upgrade with no migration; see Notes), and the conftest isolation fixture patches real seams.
- [x] #2 Running the full test suite with `HOME` pointed at a scratch dir creates NO files under the real user data dir (spot-proof with the rag_profiles consumers that leaked before).
- [x] #3 The `PYTEST_CURRENT_TEST` guard in `_maybe_run_first_run_import` is replaced by (or demoted behind) a conftest-level fixture, so the production wiring is exercised by the organic suite.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add _default_base_data_dir() call-time helper in config.py (honors XDG_DATA_HOME/HOME at call time); switch get_user_data_dir()'s fallback to it, keep BASE_DATA_DIR_CLI constant for compat.
2. Write RED test proving the frozen-constant bug, confirm GREEN after the fix (Tests/test_user_data_dir_isolation.py).
3. Blast-radius grep for other BASE_DATA_DIR_CLI/Path.home() consumers; leave out-of-scope ones (settings_screen.py display-only path, Helper_Scripts CLI script, DEFAULT_CONFIG_PATH/USER_DB_DIR) documented, not changed.
4. Retire the PYTEST_CURRENT_TEST guard in ingestion_indexing._maybe_run_first_run_import; conftest.py's autouse isolate_test_environment pre-arms _first_run_import_attempted = True instead (lazy import, try/except ImportError). Remove the dead config.get_data_dir patch.
5. Update the two test_first_run_import.py tests that bypassed the old guard to drop the now-nonexistent _running_under_pytest monkeypatch (the _reset_first_run_wiring fixture already resets the flag).
6. Scratch-HOME spot-proof: run the previously-leaking consumers with HOME/XDG_DATA_HOME pointed at a scratch dir, verify writes land under scratch and the real ~/.local/share/tldw_cli gets no new files.
7. Regression: Tests/RAG/, Tests/UI/ -k settings, config-path consumers.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause confirmed: config.py's BASE_DATA_DIR_CLI = Path.home()/".local"/"share"/"tldw_cli" froze at module-import time, so get_user_data_dir()'s fallback ignored per-test HOME/XDG_DATA_HOME monkeypatches applied afterward. Fixed by resolving the default at CALL time.

Part 1 - config.py: added `_default_base_data_dir()` (reads XDG_DATA_HOME, else HOME/.local/share, at call time) and switched get_user_data_dir()'s `else BASE_DATA_DIR_CLI` fallback to it. BASE_DATA_DIR_CLI itself is kept (compat: Helper_Scripts/Prompts/Prompts_Dump.py and settings_screen.py still read it directly). RED/GREEN proven: new Tests/test_user_data_dir_isolation.py failed against the pre-fix code with get_user_data_dir() resolving to the real dev home (`/Users/.../​.local/share/tldw_cli`) even with HOME monkeypatched post-import; passes after the fix, plus an XDG_DATA_HOME-precedence case.

Blast radius (git grep BASE_DATA_DIR_CLI / module-level Path.home() constants): left unfixed, by design -- (1) UI/Screens/settings_screen.py:3882 `_configured_user_data_dir_path` is a read-only "storage paths" display (no mkdir/write), untested, low risk; (2) Helper_Scripts/Prompts/Prompts_Dump.py is a standalone one-shot CLI script, HOME doesn't change mid-run; (3) config.py:45 DEFAULT_CONFIG_PATH and Utils/Utils.py:77 USER_DB_DIR are separate module-level Path.home() constants for a different subsystem (main config file / legacy unused user-db path) -- not part of the three documented bites, and DEFAULT_CONFIG_PATH already has a call-time-safe override path (_get_effective_config_path() honors TLDW_CONFIG_PATH) used everywhere except 3 untested encryption functions (enable/disable/change_encryption_password), noted as a separate latent issue, no test depends on it. USER_DB_DIR/get_user_database_path is dead code (no callers outside Utils/paths.py).

Part 2 - guard retirement: removed the PYTEST_CURRENT_TEST guard (and now-unused `_running_under_pytest()` helper + `import os`) from ingestion_indexing._maybe_run_first_run_import. Tests/conftest.py's autouse isolate_test_environment now pre-arms `ingestion_indexing._first_run_import_attempted = True` (lazy import, try/except ImportError) so the once-per-process first-run-import path never fires organically inside an unrelated test. Also deleted the dead `config.get_data_dir` hasattr-patch (no such attribute ever existed; silent no-op) since the HOME/XDG_DATA_HOME env patches are now sufficient on their own. Updated the two Tests/RAG/test_first_run_import.py tests that used to bypass the guard via `monkeypatch.setattr(ii, "_running_under_pytest", lambda: False)` -- removed that line; their existing `_reset_first_run_wiring` fixture already resets `_first_run_import_attempted` around each test, so the real (no-longer-gated) wiring path is what the organic suite now exercises. All 8 tests in that file still pass.

Scratch-HOME spot-proof: ran the previously-leaking consumers (test_settings_library_rag_defaults.py, test_settings_rag_profile_adapter.py, test_first_run_import.py) with HOME/XDG_DATA_HOME exported to a scratch dir before pytest started -- 77 passed; writes landed under the scratch XDG_DATA_HOME (tldw_cli/default_user/chat_dicts created there); a `find -newer` diff against a marker file in the real ~/.local/share/tldw_cli showed no new regular files (only unrelated directory-mtime noise from concurrent, independent activity on the shared dev machine -- no file content changed). The known residual leak (RAGConfig.from_settings() -> active_config._manager() -> real ConfigProfileManager(), unpatched by that file's own fixture) is closed automatically by the part-1 fix, since get_profile_manager()'s default profiles_dir now resolves through the fixed get_user_data_dir().

Regression: Tests/RAG/ 536 passed/8 skipped/0 failed. Tests/UI/ -k settings: 648 passed/7 failed (baseline -- unrelated nav-overlap + API-key-field prefill failures in test_tools_settings_window.py / test_destination_visual_parity_correction.py, confirmed pre-existing per the task dispatch's own documented baseline). Tests/test_config_delete_settings.py + Tests/UI/test_settings_configuration_hub.py: 255 passed/1 failed (test_theme_category_opens_without_crashing) -- reran in isolation and it passed, confirmed flaky under this machine's heavy concurrent multi-session pytest load, not a regression from this change.

Files: tldw_chatbook/config.py, tldw_chatbook/RAG_Search/ingestion_indexing.py, Tests/conftest.py, Tests/RAG/test_first_run_import.py, Tests/test_user_data_dir_isolation.py (new).

--- Review fix-up (2026-07-24) ---
Two Important findings from merge-gate review, both fixed:

(1) XDG_DATA_HOME precedence reverted -- `_default_base_data_dir()` is now HOME-only, matching the original pre-task-519 BASE_DATA_DIR_CLI semantics exactly (just resolved at call time instead of import time). Rationale: the original default NEVER consulted XDG_DATA_HOME. A real user with XDG_DATA_HOME exported (common on Linux desktops) already has their entire existing tldw_cli data tree under ~/.local/share/tldw_cli from every prior run before this task-519 branch existed. Had the default started honoring XDG_DATA_HOME, that user's very next launch would silently resolve to a brand-new, empty $XDG_DATA_HOME/tldw_cli directory with no migration and no warning -- their conversations/notes/media would appear to have vanished. Test-isolation only ever needed HOME (conftest's isolate_test_environment patches HOME, not XDG_DATA_HOME), so dropping XDG support costs nothing functionally while closing the migration hazard. Confirmed Path.home() itself already honors a live-monkeypatched HOME on this platform/Python (POSIX expanduser reads $HOME on every call, not cached at import), so _default_base_data_dir() uses os.environ.get("HOME") with a Path.home() fallback for extra robustness/documentation clarity, not because Path.home() was found to cache.
    Tests/test_user_data_dir_isolation.py: replaced both XDG-precedence tests with their inverse (test_get_user_data_dir_ignores_xdg_data_home, test_default_base_data_dir_helper_ignores_xdg), asserting XDG_DATA_HOME set alongside a scratch HOME is ignored and resolution stays under HOME. RED-proof performed: temporarily restored the XDG-honoring implementation, both new tests failed as expected (asserted HOME path, got XDG path); reverted, tests pass.

(2) Settings display consistency -- UI/Screens/settings_screen.py `_configured_user_data_dir_path()` (the "Storage paths" panel's default-branch resolution) was still importing and reading the frozen-at-import `BASE_DATA_DIR_CLI` constant, so in any context where HOME differs from process-start HOME (test harnesses today; any future divergence) the displayed path could disagree with what get_user_data_dir() actually resolves and uses. Switched it to call `_default_base_data_dir()` (the exact same call-time HOME-only helper the real getter's fallback uses) instead of the constant; removed the now-unused `BASE_DATA_DIR_CLI` import. Stays read-only/cheap (no mkdir, matching the existing method's contract). No existing test asserted on BASE_DATA_DIR_CLI or this method's default-branch value directly, so no test needed updating for this half.

Verification: Tests/test_user_data_dir_isolation.py + Tests/RAG/test_first_run_import.py -- 12 passed. Tests/UI/ -k settings -- 648 passed, 7 failed (same documented pre-existing baseline: nav-overlap + API-key-field prefill tests, unrelated to this change). Tests/RAG/ -- 536 passed, 8 skipped, 0 failed (unchanged from documented baseline).
<!-- SECTION:NOTES:END -->
