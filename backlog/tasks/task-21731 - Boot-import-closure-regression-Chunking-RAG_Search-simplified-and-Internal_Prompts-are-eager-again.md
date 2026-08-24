---
id: TASK-21731
title: >-
  Boot import-closure regression - Chunking, the RAG_Search.simplified stack and Internal_Prompts are eager again
status: Done
assignee: []
created_date: '2026-08-24'
labels:
  - performance
  - startup
  - regression
priority: high
dependencies: []
---

## Description

`import tldw_chatbook.app` executes 703 of this repo's own modules, up from 636. The 67
added are the whole `Chunking` engine (33), the `RAG_Search.simplified` service stack (24)
and `Internal_Prompts` (10) — none of which any user needs before first paint, including
one who never opens a RAG surface.

This undoes the guarantee TASK-21102 shipped ("app boot no longer executes the ~15k-LOC
chunking engine") and it is invisible: the two guards that exist to catch exactly this —
`Tests/Performance/test_app_import_weight.py::test_app_import_own_module_count_stays_at_the_post_diet_size`
and `Tests/Packaging/test_chunking_import_closure.py` — are both red on dev, so every
branch inherits a failing guard and the next boot-weight regression would land unseen.

Context: this belongs to the holistic performance review recorded in
`Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21102 is the guarantee being
restored).

## Acceptance Criteria

- [x] `import tldw_chatbook.app` resolves zero `tldw_chatbook.Chunking*`, zero
      `tldw_chatbook.Internal_Prompts*` and zero `tldw_chatbook.RAG_Search.simplified*`
      modules.
- [x] `Tests/Performance/test_app_import_weight.py` passes **without** relaxing
      `MAX_TLDW_MODULE_COUNT` (it stays at 660).
- [x] `Tests/Packaging/test_chunking_import_closure.py` passes again.
- [x] The removed work is gone, not relocated: boot-to-first-screen-usable is not worse,
      and no deferred import is moved into `on_mount` or a post-paint task.
- [x] The MCP profile-driven RAG search shipped by PR #2049 still resolves and behaves
      identically — the deferred seam is exercised by a test that calls it, not merely
      asserted absent.
- [x] A failure at first use of a deferred import surfaces through the same path it did
      when the import was eager (documented walk, plus a test for the optional-dependency
      case).
- [x] Every new test fails against a deliberately broken implementation (mutation result
      recorded per test).

## Implementation Plan

1. Reproduce the closure and time it in an isolated-config subprocess; bisect the chain
   with an import tracer rather than trusting a reported chain.
2. Break every eager edge into `RAG_Search.simplified` from the boot path, following the
   house pattern established by TASK-21102 (pure constants/helpers move to a stdlib-only
   module outside the heavy package; everything else defers to its use site).
3. Re-measure the closure and wall time, arms interleaved, and measure time-to-interactive
   to prove the work is not merely relocated.
4. Walk the first-use failure modes of every deferred import and cover the reachable ones.
5. Add closure guards; mutation-test each one.

## Implementation Notes

**Two edges, not one.** The reported chain named three MCP files; an import tracer
recording `(importer, imported)` edges over the whole boot disagreed with all three
(`MCP/server.py` is in the closure but imports no RAG; `MCP/tools.py` does eagerly
import `simplified.search_service` but is **not** in the boot closure;
`UI/MCP_Modules/mcp_inspector.py` imports no RAG at all). The real boot edge was a
single line PR #2049 added to `Library/library_local_rag_search_service.py`:
`from ...simplified.active_config import normalize_rag_search_mode`. That module is on
the app's import path, and `active_config` pulls `simplified.rag_service` ->
`chunking_service` -> the Chunking engine -> `Internal_Prompts`.

Deferring that alone took the closure from **703 to 637** and turned both red guards
green — and bought the user **nothing**. A time-to-interactive probe (import ->
`TldwCli()` -> `run_test()` -> `_ui_ready`, censusing `sys.modules` at readiness)
showed Chunking 33 / simplified 19 / Internal_Prompts 10 still resident when the app
became usable: `Event_Handlers/Chat_Events/chat_rag_events.py`, imported during the
**initial Chat screen mount** via `UI/Console_Modules/retrieval.py`, ran a module-scope
`try: from ...RAG_Search.simplified import ...` availability probe — **50 ms** on the
event loop, timed with the same tracer. The eager boot import had merely been paying it
early.

**What shipped.**

1. `tldw_chatbook/RAG_Search/search_modes.py` — new, stdlib-only: `RAG_SEARCH_MODES` +
   `normalize_rag_search_mode`, lifted out of `active_config`. Same shape as
   `chunking_engine_version.py` (TASK-21102): `active_config` re-imports both names, so
   there is one object in the process and no copy that can drift from the vocabulary
   `SearchConfig.default_search_mode` validates against. The Library service imports it
   at module scope, which keeps the existing monkeypatch seam
   (`test_resolve_profile_search_mode_delegates_normalization`) working.
2. `chat_rag_events` resolves its availability probe on first ask
   (`_rag_services_available()`, cached), with a PEP 562 module `__getattr__` keeping
   `chat_rag_events.RAG_SERVICES_AVAILABLE` readable for external callers. The module's
   own consumer had to switch to the function — a bare global lookup does not consult a
   module `__getattr__`.

**First-use failure walk (all three deferrals).** The `search_modes` import cannot fail
(stdlib only), and it strictly *removes* a boot failure mode: an `active_config` that
failed to import used to take the app's import down with it. The `chat_rag_events`
probe keeps its `except ImportError` verbatim, so a plain install without the
`embeddings_rag` extra gets the identical observable outcome — one warning, a `False`
flag, `None` from `get_or_initialize_rag_service` — just at first ask
(pinned by `test_missing_rag_extras_degrade_at_first_use...`). A NON-`ImportError`
raised by `simplified` used to abort the Chat-screen module import (a dead app); it now
propagates out of `get_or_initialize_rag_service`, which has **zero production callers**
— every live RAG path resolves through `ingestion_indexing.get_shared_rag_service`,
whose own error handling is untouched.

**Deliberately left eager, with reasons.** `Internal_Prompts` still loads at Chat-screen
mount (10 modules, 2.6 ms) via `Agents/agent_service`, which reads
`CATALOG["agents.subagent_system"].default` into a module constant; it leaves the boot
closure but deferring the mount leg means restructuring that constant, for 2.6 ms.
`UI/Screens/settings_rag_profile_adapter` imports `active_config`,
`collection_fingerprint` and `collection_indexes` at module scope as documented
monkeypatch seams, so the background screen pre-importer still warms the RAG stack
~0.2 s after first paint on a daemon thread (pre-existing, both arms, finding 21113's
territory). Neither is part of this regression.

**Measured, arms interleaved twice (5 runs/arm/pass for import, 3 for TTI; machine under
concurrent load, load average 8–16).**

| | base `65386b917` | fixed |
|---|---|---|
| `import tldw_chatbook.app` own modules | 703 | **637** |
| total `sys.modules` | 2123 | **1960** |
| import wall, pass A / pass B (median) | 876.0 / 885.6 ms | **834.3 / 854.9 ms** |
| time-to-interactive, pass A / pass B (median) | 2699.9 / 2778.1 ms | **2478.5 / 2650.9 ms** |
| resident at `_ui_ready`: Chunking / simplified / Internal_Prompts | 33 / 19 / 10 | **0 / 0 / 10** |
| total `tldw_chatbook.*` at `_ui_ready` | 984 | **928** |

The last two rows are the point: the work is absent when the app becomes usable, not moved
somewhere later in the boot.

**Test A/B (identical command, both arms).** `Tests/RAG Tests/RAG_Search Tests/MCP
Tests/Packaging Tests/Performance`: base **16 failed / 2527 passed / 18 skipped**, fixed
**9 failed / 2534 passed / 18 skipped**. The 9 are byte-identical on both arms
(`test_fts5_match_forms_shared`, `test_gateway_runtime_prompts`,
`test_installed_distribution::…migrates_v35…[source|sdist]`, and five
`test_rag_citation_provenance_benchmark` tests) — all pre-existing. The 7-failure
difference is exactly this task's guards: the two that were red on dev
(`test_app_import_own_module_count_stays_at_the_post_diet_size`,
`test_chunking_import_closure::test_app_import_does_not_execute_chunking`) plus the 5 new
ones, all red on base and green here.

**Pre-existing preflight red, not mine.** `./scripts/preflight.sh` reports 4/5 green and
one failure — the production diagnostic inventory, naming
`RAG_Search/simplified/{enhanced_rag_service_v2,rag_service,search_service}.py`. The same
three rows drift byte-identically on pristine `65386b917` (verified by restoring the base
tree and re-running the check), so the pin went stale at PR #2049's merge. None of the
three is touched here, so `--write` was deliberately NOT run.

**Modified/added files.** `tldw_chatbook/RAG_Search/search_modes.py` (new),
`tldw_chatbook/RAG_Search/simplified/active_config.py`,
`tldw_chatbook/Library/library_local_rag_search_service.py`,
`tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py`,
`Tests/Packaging/test_rag_boot_import_closure.py` (new, 5 tests),
`Tests/Performance/test_app_import_weight.py` (budget docstring only — the limit is
unchanged at 660), `backlog/docs/lessons-testing-evidence.md`.
