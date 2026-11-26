---
id: TASK-42551
title: "Two received-dispatch modules load on the app import path since #3763"
status: Done
assignee:
- '@claude'
created_date: 2026-11-25 22:10
labels:
- console
- performance
- dev-red
priority: medium
references:
- tldw_chatbook/Chat/console_runtime.py
- tldw_chatbook/Chat/console_received_dispatch.py
- Tests/Performance/test_app_import_weight.py
- Tests/Architecture/test_module_size_ratchet.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by round 26's final gate (2026-11-25). `Tests/Performance/test_app_import_weight.py::test_app_import_own_module_count_stays_at_the_post_diet_size` is red on dev 692acb43dc: `import tldw_chatbook.app loaded 689 tldw_chatbook modules (ratchet limit 688)`. Dev 4bf5706204 passes (687); #3763's merge 0704967ec9 is the first red commit.

**Cause.** #3763 replaced `ConsoleRuntime._require_received_custody_current`, a method that imported `console_received_dispatch` when called, with a class-body `from .console_received_dispatch import (...)` that also binds the new `owns_session`. A class-body import runs when the class is created, so `import tldw_chatbook.app` now loads `Chat.console_received_dispatch` and, through it, `Chat.console_preparation_reads`: two more modules on the boot path, and the ADR-097 module-count guard fails.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `import tldw_chatbook.app` loads no more `tldw_chatbook` modules than on dev 4bf5706204, and the module-count guard passes without a limit raise
- [x] #2 `ConsoleRuntime.owns_session` and `_require_received_custody_current` behave as before for every caller
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`ConsoleRuntime` gets the two methods back as wrappers that import `console_received_dispatch` when called, the pattern #3763 removed. The functions themselves stay in `console_received_dispatch.py`. That module is still resident by UI ready through `console_spend_projection`, as `Tests/UI/test_console_first_send_preimport.py` expects; only the bare app import is lighter.

Evidence:
- **Module diff.** On a scratch probe that reuses the guard's isolated-import helper, dev 4bf5706204 loaded 687 modules and 0704967ec9 loaded 689. The two added modules are exactly `Chat.console_received_dispatch` and `Chat.console_preparation_reads`. An import trace shows `console_runtime.py:3043`, the class-body import, pulling both.
- **Guard.** The guard is red on dev 692acb43dc and green with this change. The rest of the boot budget set passes (34 passed).
- **Callers.** The tests that reach `owns_session` or `_require_received_custody_current` (18 files) give 340 passed, 21 failed and 1 error with this change. Run serially on dev 692acb43dc, the same 21 fail, so none of them comes from this change. They are in `test_native_plugin_hook_preparation.py` (8), `test_native_v2_mcp_events.py` (7), `test_console_poll_reconciliation.py` (3) and `test_console_staged_evidence_sessions.py` (3).
- **Size row.** `console_runtime.py` measures 7,103 on dev. It is +5 here, so the row goes 7,104 -> 7,108, an exact raise with a dated comment.
<!-- SECTION:NOTES:END -->
