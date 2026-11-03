# PR3346 UI16 import inventory and UI10/UI12 controls

Product HEAD79e659dcde6f5360edb0070f9d184713bd8584d1 with approved uncommitted memory-receipt fix. Parent authorized exact investigation, then named lazy inventory correction if resident intent was confirmed. No commits, pushes, manifests or unrelated file edits.

## Cause and classification

`UI/Console_Modules/console_spend_projection.py:29` imports `stock_native_methods` from `Chat.console_received_dispatch` at module scope. That helper checks exact controller methods for spend dependency ownership. The file is unchanged versus pinned dev62a169f906508ed02bd7546ab278d916638d1607; the eager import is from upstream6357516f55. It intentionally makes dispatch resident with the initial route. Three UI16 checks still treated dispatch as lazy.

The fourth UI16 check initially timed out in CI because the real memory DB operational receipt could not save; the separate memory fix makes it reach its original first-send import assertion. That assertion then exposed five exact lazy modules absent from the paced post-ready list:

- Chat.console_plugin_context
- Agents.hook_permissions
- Chat.console_prompt_mixin_service
- DB.console_prompt_mixins
- Chat.console_pre_provider_stop

Exact four-node comparison on the existing pristine dev62a archive reproduced all four failures in16.48s: same dispatch residency and only the three non-mixin lazy omissions. The branch introduces the two mixin omissions. The initial corrected-memory branch run reproduced the four import failures and passed the UI10/UI12 controls (4 failed,2 passed in32.20s).

## Fix

Remove the one already-resident dispatch entry from FIRST_SEND_MODULES and its stale lazy test record. Explicitly assert that it is resident in the fresh interpreter and accepted by the resident guard. Add the five measured lazy imports to the existing paced list and fresh-process cold assertions. Clarify in the test that the historical live record is supplemented by later mounted observations; no new live-provider claim.

This changes only the named import inventory, not startup eager imports, readiness census, warm-up order, pacing, permission authority, error handling, or blanket exclusions. The actual first-send guard still rejects every unknown lazy import.

## Evidence

- Current pre-fix exact UI16 four nodes plus UI10/UI12 controls:4 failed,2 passed in32.20s. UI10 held run passed14.08s; UI12 Discard send passed6.24s. Both controls were unchanged; they confirm the memory-receipt fix covers their send path.
- Pinned dev62a exact four UI16 nodes:4 failed in16.48s, proving inherited residency and three lazy omissions.
- Final entire `Tests/UI/test_console_first_send_preimport.py` plus `Tests/UI/test_console_first_send_import_cycles.py`:13 passed in15.41s. Includes fresh-interpreter cold/warm assertions, actual mounted send, pacing, shutdown, failed imports, manifest parse caching, module cycle and dynamic-import guards.
- Both changed files Ruff clean, formatter check clean, scoped git diff --check clean.

No full suite, increased timeout, weakened import guard, new before-ready import, or census repin.

## Explicit owned artifacts

Additional repository files (beyond the two in ui6-fix-report.md):
- `tldw_chatbook/Chat/console_first_send_modules.py` (+5/-1)
- `Tests/UI/test_console_first_send_preimport.py` (+11/-1)

Private report: this file.

Private evidence:
- `/private/tmp/mixins-ui16-cause.log`
- `/private/tmp/mixins-ui16-dev.log`
- `/private/tmp/mixins-ui16-green.log`

Owned pytest basetemp directories:
- `/private/tmp/mixins-ui16-cause`
- `/private/tmp/mixins-ui16-dev`
- `/private/tmp/mixins-ui16-green`

The dev archive under `mixins-ci3346-cizd3tw9/baseline` and input CI logs are parent-owned. No archive source files were edited. See ui6-fix-report.md for earlier logs/directories and the two memory-fix source/test files. These four repository files are my full owned implementation set; all other working changes belong to the parent or other agents.
