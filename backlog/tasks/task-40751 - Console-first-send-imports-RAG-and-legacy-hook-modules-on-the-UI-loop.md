---
id: TASK-40751
title: Console first send imports RAG and legacy hook modules on the UI loop
status: To Do
type: bug
created_date: '2026-11-13 08:05'
labels:
- console
- performance
- ci
priority: medium
references:
- https://github.com/rmusser01/tldw_chatbook/actions/runs/41289667906
- Tests/UI/test_console_first_send_preimport.py
- tldw_chatbook/Library/library_rag_state.py
- tldw_chatbook/Chat/console_runtime.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The mounted llama.cpp send harness reports three imports on the UI loop outside its warm set. Trace each import to its caller and restore the first-send guard while preserving RAG profile and hook behavior.

### Reproduce

Use a clean Windows checkout of dev at `beab5c1e4e1128086ac518e525a9274dff4baf5f` with the repository's dev dependencies installed. The audit used Python 3.12.10, Textual 8.2.8 and pytest 8.3.5. From the repository root in PowerShell, activate the dev virtual environment and use a new private test root under a private parent such as `$env:USERPROFILE/.codex`.

```powershell
$testRoot = Join-Path "$env:USERPROFILE/.codex" ('ci-repro-' + [guid]::NewGuid().ToString('N'))
python -c "import sys; from tldw_chatbook.Utils.platform_files import os; os.mkdir(sys.argv[1], 0o700)" "$testRoot"
$env:TLDW_TEST_CONFIG_ROOT = $testRoot
$env:TMP = $testRoot
$env:TEMP = $testRoot
python -m pytest 'Tests/UI/test_console_first_send_preimport.py::test_a_harness_send_imports_nothing_lazy_beyond_the_list' -q --tb=short --no-cov
```

### Findings

The guard reported these module/caller pairs:

| Module | Caller |
| --- | --- |
| `tldw_chatbook.RAG_Search.simplified.active_config` | `library_rag_profile_top_k` |
| `tldw_chatbook.RAG_Search.config_profiles` | `_manager` |
| `tldw_chatbook.Agents.legacy_hook_ownership` | `ConsoleRuntime.ensure_run_hooks` |

The November 1 CI run reported the same guard with a different import set, `console_plugin_context` and `hook_permissions`. The current task covers the three modules reproduced on dev. Related warm-up work includes TASK-33620.15.5, TASK-33620.15.15 and the review-import work in TASK-33620.15.53.

The audit also tested a detached checkout of the same dev commit after reverse-applying the first-parent diff of merge commit `cf5fda70d3` for `tldw_chatbook/UI/Console_Modules/session.py` and `tldw_chatbook/UI/Console_Modules/wiring.py`, the two production files changed by #3306. The guard reported the same three module/caller pairs listed above.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The mounted first-send import guard passes and still detects a deliberately unwarmed send-path import.
- [ ] #2 The existing first-send warm-up and UI-ready import-budget checks pass with RAG profile and hook behavior preserved.
<!-- AC:END -->
