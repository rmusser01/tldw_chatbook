# Task 2 — shared Direct Library factory correction

Status: **DONE_WITH_CONCERNS for scoped review**. Base: `75456b7a2d3f8ea4f711f41c061d212abdd03add`. Worktree: `/Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook`. Root owns modified task/plan/ledger and live evidence.

The shared `_library_provider_for_app` no longer supplies unsupported `collections_service` to the real `LocalLibraryToolService` constructor. The production diff is exactly that one-line deletion. Supported services/enforcer and frozen Direct/RAG selection remain intact; no service API, collection feature, trace guard, authority, UI/config/schema or dependency changes. ADR required: no, routine call-site repair; ADR097/166 context remains unchanged.

Read the complete shared factory, its runtime partial binding, both named controller consumers, supported service constructor, UI/MCP construction references, Backlog and brief. The disposable Direct app traceback names this exact unsupported argument. Earlier Task1/review phases and source hashes remain preserved; this correction does not reopen them or claim fresh live qualification.

Added three cases in the existing runtime-policy file using the real provider/service constructors via the controller's frozen authority selection and production partial binding. Both Direct cases carry a collections attribute; optional backends/enforcer are absent or distinct sentinels, with exact forwarding and real media-chunk construction verified. RAG retains its exact backend and one-tool catalog under the captured selector (opposite compatibility configuration), and absent context stays inert. No arbitrary-kwargs constructor stub was used.

All commands ran in this worktree. Pytest prefix: `PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -p no:cacheprovider`. Ruff prefix: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff`. Outputs redirect to matching `/private/tmp/archive-capture-task2-*.log`.

| Phase / command | Raw exit | Result |
| --- | --- | --- |
| RED: `Tests/Chat/test_console_library_runtime_policy.py -q -k 'shared_direct_factory or shared_factory_preserves' --basetemp=/private/tmp/archive-capture-task2-red-stores` after pytest prefix | 1 | 2 failed, 1 passed, 10 deselected, 1.55s. Both real Direct constructors reject `collections_service`; RAG/backend control passes. |
| GREEN: `Tests/Chat/test_console_library_runtime_policy.py Tests/Chat/test_console_archive_recovery_followup.py::test_captured_native_turn_restores_original_second_row -q --basetemp=/private/tmp/archive-capture-task2-green-stores` after pytest prefix | 0 | **15 passed, 14.92s**, no teardown/network-guard errors. 13 policy/factory checks plus two real captured Direct/RAG positives with documented fake transport. |
| `ruff check --no-cache Tests/Chat/test_console_library_runtime_policy.py tldw_chatbook/Chat/console_runtime.py` | 1 | Existing runtime debt plus test import formatting/order; raw output preserved. |
| `ruff format --no-cache --check Tests/Chat/test_console_library_runtime_policy.py tldw_chatbook/Chat/console_runtime.py` | 1 | New long import plus existing whole-runtime formatting debt; raw output preserved. |
| `ruff check --no-cache Tests/Chat/test_console_library_runtime_policy.py` after import wrap | 1 | Existing import order flagged in the touched import block; preserved as `ruff-test-final.log`. |
| `ruff check --no-cache --select I --diff Tests/Chat/test_console_library_runtime_policy.py` | 1 | Printed the two-name ordering correction only; no source write. |
| Final `ruff check --no-cache Tests/Chat/test_console_library_runtime_policy.py` | 0 | Pass (`ruff-test-clean.log`). |
| `ruff format --no-cache --check Tests/Chat/test_console_library_runtime_policy.py` after wrap and final order | 0 each | Pass; both outputs retained. |
| Python baseline/current static comparison | 0 | Baseline/current Ruff each exit1, 34 diagnostic signatures; exact `Counter((code,message))` equality. Both whole runtime sources remain unformatted. Exact production bytes equal baseline with only the unsupported line removed. |
| Final hash/phase JSON generation | 0 | Hashes below; only test import wrapping/order followed GREEN, with no executable behavior change. |
| `git diff --check -- tldw_chatbook/Chat/console_runtime.py Tests/Chat/test_console_library_runtime_policy.py` | 0 | Source/test whitespace clean. |
| `git diff --cached --check` after raw evidence staging | 2 | The byte-preserved `Docs/superpowers/qa/console/2026-09-28-console-direct-library-factory/red.log` has 14 original pytest traceback trailing-space lines. No whole-artifact clean claim. |
| `git diff --cached --check -- . ':!Docs/superpowers/qa/console/2026-09-28-console-direct-library-factory/red.log'` | 0 | Source/tests/authored QA and every other newly staged evidence path clean. |

[Manifest/raw phases](2026-09-28-console-direct-library-factory/manifest.json) preserve the new evidence byte-for-byte; [static details](2026-09-28-console-direct-library-factory/static-final.json) retain final hashes and baseline comparison. The comparison uses `git show 75456b7a2d3f8ea4f711f41c061d212abdd03add:tldw_chatbook/Chat/console_runtime.py`, feeds baseline/current text to `ruff check --no-cache --stdin-filename tldw_chatbook/Chat/console_runtime.py --output-format=json -`, and compares formatter stdout from `ruff format --no-cache --stdin-filename tldw_chatbook/Chat/console_runtime.py -` without writing production source.

Controller, trace runtime/factory/service, existing capture tests and service API remain unchanged. Source SHA256: `3cd6e642f5b8e9bec25329af6d7cac2d9d0d2fc26a2d02be2efce2beb9ec8cf2`. Test SHA256: `01add45624951c6aad3d37114115da33a61210759942db2fbcd3984391eb1266`.

Self-review confirmed the exact constructor contract, all shared callers/references, optional backend/enforcer/media-chunk forwarding, frozen RAG branch, controller Allowed gating and source-only deletion. No source apply rejection or unresolved defect was found. Existing runtime lint/format debt is a limitation; automated fake-boundary checks are not live qualification. No app/model/network request, normal-profile mutation, download, full sweep, subagent, push, PR, merge or cleanup was performed. Root's independent scoped review and fresh Direct live qualification remain pending.

## Scoped review and live boundary

Independent review of75456b7→e4a2717 approved the source correction with0actionable findings; [review report](2026-09-28-console-direct-library-factory/review.md). Root subsequently mounted the real corrected Direct app: provider construction/tool discovery/archive search succeed, but the shared10-request test ceiling is reached before a restore follow-up. [Live results](2026-09-28-agent-archive-recovery-capture-live.md) preserve this incomplete boundary and the successful RAG-only qualification.
