---
id: TASK-519
title: >-
  Fix get_user_data_dir import-time home freeze breaking test HOME isolation
status: To Do
assignee: []
created_date: '2026-07-23 23:30'
updated_date: '2026-07-23 23:30'
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
- [ ] `get_user_data_dir()` (and any sibling data-dir accessor) honors `HOME`/`XDG_DATA_HOME` changes made after module import (call-time resolution), or the conftest isolation fixture patches a real seam that achieves the same.
- [ ] Running the full test suite with `HOME` pointed at a scratch dir creates NO files under the real user data dir (spot-proof with the rag_profiles consumers that leaked before).
- [ ] The `PYTEST_CURRENT_TEST` guard in `_maybe_run_first_run_import` is replaced by (or demoted behind) a conftest-level fixture, so the production wiring is exercised by the organic suite.
<!-- AC:END -->
