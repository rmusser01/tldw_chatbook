---
id: TASK-285
title: Lazy tldw_api package imports (~469ms of startup)
status: In Progress
assignee: []
created_date: '2026-07-16 14:30'
labels: [performance, startup]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tldw_api/__init__.py eagerly imports 54 schema files = 1,313 Pydantic models, ~469ms (31% of the ~1.5-2.0s app import), forced by app.py:353 plus 69 files importing names from the package. Fix: PEP 562 lazy __getattr__ re-exports (working, documented pattern in Local_Ingestion/__init__.py) and/or Server*Service modules importing their own schema submodules. Longer-term note: TldwCli.__init__ constructs ~30 Server*Service objects even in local-only mode. Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P2 C1).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 import tldw_chatbook.app no longer pays the full tldw_api package cost (measured importtime delta reported, expected >300ms). PHASE-1 RESULT (controller-verified A/B, scratch HOME, same protocol both sides): app-import wall time UNCHANGED (~1.47s branch vs ~1.49s dev) — the lazy layer + client deferral are correct (bare package import cost ~430-580ms → ~0; client.py 15k lines no longer loads on app import, regression-net-tested; 44 on-path TLDWAPIClient importers deferred) but ~47 schema submodules still load because Server*Service modules name-import their schemas at MODULE scope, and the lazy layer faithfully serves them. PHASE 2 (required for this AC): TYPE_CHECKING/method-local schema imports across the on-path service modules, or stop constructing all Server*Services in TldwCli.__init__ (the audit's longer-term note)
- [x] #2 All existing consumers keep working (full test suite green; no import cycles)
- [x] #3 Remote-server mode still functions (schema names resolve on demand)
<!-- AC:END -->

## Implementation Plan

1. Read `Local_Ingestion/__init__.py` (existing PEP 562 lazy `__getattr__` pattern in this repo) and mirror its structure/docstring style.
2. Write a throwaway AST-based generator script that parses the current eager `tldw_api/__init__.py`'s `from .x import (...)` statements and emits: (a) a `name -> submodule` mapping (last-import-wins, matching current runtime semantics), (b) the exact `__all__` list unchanged, (c) any module-level aliases defined directly in `__init__.py` (found: `WebProcessResponse = IngestWebContentResponse`).
3. Generate and install the new lazy `__init__.py`; verify the generated `__all__` is byte-identical to the original and the full old/new name-set (1,271 names) matches exactly.
4. Verify correctness: bare-import laziness, cross-submodule name resolution + `__module__` identity, the `WebProcessResponse` alias, `AttributeError` shape, `dir()`, and a full sweep resolving every one of the 1,271 mapped names (superset of the 950-name `__all__`) with zero failures.
5. Measure `import tldw_chatbook.app` before/after with `-X importtime` (5 runs each) under a scratch HOME, plus an isolated bare-package-import measurement and a `TldwCli()` construction probe, to honestly determine whether the win reaches `import tldw_chatbook.app` or only benefits isolated package/submodule access.
6. Write `Tests/Utils/test_tldw_api_lazy.py` covering laziness, cross-submodule resolution, alias identity, error shape, `dir()`, and the full `__all__`/mapping resolution sweep.
7. Run `Tests/tldw_api/`, `Tests/Chat/`, `Tests/Utils/` plus a full `Tests/` collect-only pass; confirm no existing test files were modified; update this task's AC/notes with the measured deltas.

## Implementation Notes

**Approach.** Rewrote `tldw_chatbook/tldw_api/__init__.py` as a lazy PEP 562 re-export layer, generated (not hand-transcribed) from the previous eager file via a throwaway AST-parsing script so no name could be silently dropped. The new file keeps `__all__` byte-identical (950 names, same order) and adds `_SUBMODULE_BY_NAME` (1,271 names — a strict superset of `__all__`, since the old file also let callers `from tldw_chatbook.tldw_api import X` for names never listed in `__all__`), `_ATTR_ALIASES` (one entry: `WebProcessResponse` -> `IngestWebContentResponse`, the only module-level alias the old file defined directly rather than importing), `__getattr__`, and `__dir__`. No aliased (`import X as Y`) statements and no name shadowing (two submodules exporting the same name) existed in the old file, so there was nothing to disambiguate beyond the one alias.

**Correctness verification** (all passed): bare `import tldw_chatbook.tldw_api` in a fresh subprocess loads none of the heavy schema submodules; `from tldw_chatbook.tldw_api import X` for names from six different submodules resolves each to a class whose `__module__` is the correct defining submodule; `WebProcessResponse is IngestWebContentResponse` (identity preserved); unknown attribute access raises `AttributeError` naming the package; `dir()` includes lazy names; a full sweep of all 1,271 mapped names (not just the 950 in `__all__`) resolves via `getattr` with zero failures; the old-file/new-file name sets are set-equal (1,271 == 1,271, no drops, no additions).

**Measurement — the honest finding.** `-X importtime -c "import tldw_chatbook.app"` under a scratch HOME/XDG_CONFIG_HOME/TLDW_CONFIG_PATH, 5 runs each:
- Baseline (eager, mean of 5): **1,556,828 µs** (~1557 ms)
- After (lazy, mean of 5): **1,559,242 µs** (~1559 ms)
- Delta: **+2,415 µs** (i.e., no measurable improvement — within run-to-run noise, nowhere near the AC's expected >300ms reduction)

Root cause: `tldw_chatbook/runtime_policy/bootstrap.py:9` does a module-level `from tldw_chatbook.tldw_api import TLDWAPIClient`, and that module is reached almost immediately in `app.py`'s own import chain (via `Chat/server_chat_conversation_service.py`, imported by `app.py` around line 101 — long before `app.py:353`'s `MCPUnifiedClient` import). Resolving `TLDWAPIClient` triggers `tldw_api/client.py` (15,111 lines), which **independently and eagerly re-imports virtually the same ~54-submodule / 1,270-name schema surface at its own module scope**, because its method signatures reference nearly every request/response type. This is a separate, pre-existing eager-import site — not something the `__init__.py` lazy layer can fix on its own, and it sits inside `client.py`/consumer files that are explicitly out of scope here (task shape says "nothing else in the package changes"; app.py's `Server*Service` construction is explicitly out of scope too). Confirmed with a `TldwCli()` construction probe: `tldw_api.client` is already in `sys.modules` *before* `TldwCli()` is even constructed — the cost is paid during plain `import tldw_chatbook.app`, not deferred to app construction as the task's framing anticipated it might be.

**Where the win genuinely lands.** Isolated `import tldw_chatbook.tldw_api` (bare, no `TLDWAPIClient`/`MCPUnifiedClient` touch), 3 runs each, cumulative cost attributable to the package's own submodule imports (excluding shared parent-package cost): baseline ~430,000-580,000 µs of child-schema-import cost vs. after ~0 (no schema children touched at all). This benefits any process that imports `tldw_chatbook.tldw_api` or a specific schema submodule without needing `TLDWAPIClient` — confirmed via `Tests/tldw_api/` (402 tests, many doing direct submodule imports) staying green and fast, and is a prerequisite for any future fix that defers `TLDWAPIClient` resolution in `runtime_policy/bootstrap.py` and the ~44 `Server*Service` modules that import it at module scope.

**AC #1 status: not met as measured.** The package is genuinely lazy and correct, but `import tldw_chatbook.app` does not show the expected >300ms drop because of the `client.py` re-import chain described above. Closing AC #1 for real would require a follow-up (out of scope here) to defer `TLDWAPIClient`'s eager import in `runtime_policy/bootstrap.py` and/or the `Server*Service` modules — and possibly making `client.py` itself use `TYPE_CHECKING`-guarded imports for its ~50 schema type hints, since it is the actual 449-492ms cost center. Left AC #1 unchecked and task status at **In Progress** pending a scope decision on that follow-up.

**Tests.** New file `Tests/Utils/test_tldw_api_lazy.py` (7 tests, all passing): subprocess laziness check, cross-submodule resolution + `__module__` identity, alias identity, `AttributeError` shape, `dir()`, full `__all__` sweep, full `_SUBMODULE_BY_NAME` sweep (superset of `__all__`).

**Verification run:** `Tests/tldw_api/` 402 passed; `Tests/Chat/` 905 passed, 69 skipped; `Tests/Utils/` 187 passed (incl. the 7 new); full `Tests/` `--collect-only` (10,756 tests) collects cleanly with zero errors, confirming no import cycles introduced anywhere in the suite. `python -c "import tldw_chatbook.app"` succeeds under a scratch HOME. `git diff origin/dev --diff-filter=M --name-only -- Tests/` is empty (no existing test file modified).

**Files changed:** `tldw_chatbook/tldw_api/__init__.py` (rewritten, 1,681 -> 2,296 lines: the `__all__`/mapping literals are larger than the old import statements they replace, but there is now exactly one function body instead of 54 import statements); `Tests/Utils/test_tldw_api_lazy.py` (new).

## Implementation Notes — continuation (client.py off the app path)

**Scope extension executed:** the coordinator ruled the `TLDWAPIClient` eager-import chain in scope for AC #1. Inventoried every module-scope importer of `TLDWAPIClient` that loads during `import tldw_chatbook.app` (sys.modules diff against the 47 importer files): **44 on-path modules** (42 cookie-cutter `Server*Service`/scope-service files + `runtime_policy/bootstrap.py` + `runtime_policy/server_context.py`); 3 off-path files (`Outputs/server_outputs_service.py`, `Sharing/server_sharing_service.py`, `WebClipper/server_web_clipper_service.py`) left untouched per instruction. All 44 converted:
- 42 files (all with `from __future__ import annotations`, `TLDWAPIClient` annotation-only — AST-verified zero runtime uses per file): mechanical conversion — name removed from the module-scope `tldw_api` import, `if TYPE_CHECKING:` guarded import added, `TYPE_CHECKING` added to the existing `from typing import` line.
- `runtime_policy/bootstrap.py`: import moved under `TYPE_CHECKING` + a function-local `from tldw_chatbook.tldw_api import TLDWAPIClient` at the top of `build_runtime_api_client()` — the single construction site (`Tests/RuntimePolicy/test_boundary_guards.py` statically confirms construction stays confined to this file).
- `Chatbooks/server_chatbook_service.py` (had no future-annotations import): added `from __future__ import annotations` + `TYPE_CHECKING` guard. A first attempt quoted the three annotations instead, which broke `Tests/RuntimePolicy/test_server_client_provider_migration_audit.py` (exact-text signature matching against `Docs/Development/server-client-provider-migration-audit.md`); the future-import approach restores the original signature text so both the audit doc and the test stay untouched.

**Verified after conversion:** bare `import tldw_chatbook.app` no longer loads `tldw_chatbook.tldw_api.client` (sys.modules assertion, now a permanent subprocess regression test); `TldwCli()` construction in local mode (57 ms) still does not load it — the `LegacyConfigServerClientProvider` fallbacks defer `build_client()` until the first actual server call; the remote-mode on-demand path verified end-to-end (provider construction stays lazy; `provider.build_client()` triggers the deferred import and returns a working `TLDWAPIClient`; bearer and api_key auth modes both exercised).

**Measured headline (drift-controlled interleaved A/B, scratch HOME, 6 rounds alternating a baseline worktree at `origin/dev`@1bf8c0d3 vs this tree, `-X importtime` cumulative for `tldw_chatbook.app`):** baseline mean 1,513,473 µs / median 1,515,983 µs; after mean 1,510,294 µs / median 1,490,478 µs (one after-round was a 1,627 ms outlier; trimmed means 1,508,871 vs 1,486,864 µs). **Genuine improvement ≈ 22–26 ms (median −25.5 ms) — real, but not the >300 ms AC #1 expects.**

**Why the big number doesn't materialize (measured root cause #2):** the ~438–450 ms of tldw_api cost in the baseline trace is almost entirely the **schema submodules**, not `client.py`'s own exec (~3–9 ms). With `client.py` off the path, **46 schema submodules still load eagerly**, because the `Server*Service` modules import their schema names at module scope and genuinely **use them at runtime** — AST audit across the 40 schema-importing on-path services: **290 runtime-use sites (request-object construction in method bodies) vs 56 annotation-only names**. app.py imports those service modules at its own module scope, so the schema surface is forced independently of `client.py`. The importtime trace confirms the attribution: `Media/server_media_reading_service` shows 52 ms self-time and `Kanban_Interop/server_kanban_service` 19.7 ms — that is their schema submodules loading through the lazy package layer (`-X importtime` books `__getattr__`-triggered imports into the *importing* module's self-time rather than as separate module lines; sys.modules is the ground truth for what loaded). Deferring those 290 runtime-use sites would be a sprawling rewrite across 40 files (or requires the explicitly out-of-scope Server*Service-construction restructure) — **stopped and reported** per instruction rather than improvised.

**AC #1 final status: still unchecked.** The structural fix is fully in place and regression-netted (package lazy; `client.py` off the app-import path and off the local-mode construction path), but the app-import headline improves only ~25 ms because the schema surface is independently forced by the services' own runtime-used schema imports. The remaining ~400 ms prize needs a follow-up scope decision: function-local schema imports across ~290 sites, or lazy `Server*Service` module loading (the audit's "longer-term" item).

**Additional test:** `test_app_import_does_not_load_tldw_api_client` (fresh-subprocess assertion that `import tldw_chatbook.app` leaves `tldw_chatbook.tldw_api.client` out of sys.modules) added to `Tests/Utils/test_tldw_api_lazy.py` (now 8 tests).

**Verification (continuation):** Tests/Utils/ 188 passed (incl. the 8 lazy tests); Tests/tldw_api/ 402 passed; Tests/Chat/ 905 passed, 69 skipped; Tests/RuntimePolicy/ + Tests/Chatbooks/ 386 passed, 1 skipped (includes the boundary-guard and migration-audit static guards); full `Tests/` collect-only: 10,757 tests, zero collection errors; `git diff origin/dev --diff-filter=M --name-only -- Tests/` empty.

**Files changed (continuation):** 44 consumer modules (TYPE_CHECKING conversion; function-local construction import in `runtime_policy/bootstrap.py`), `Tests/Utils/test_tldw_api_lazy.py` (one added test).
