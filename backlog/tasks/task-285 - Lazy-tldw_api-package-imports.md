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
- [ ] #1 import tldw_chatbook.app no longer pays the full tldw_api package cost (measured importtime delta reported, expected >300ms)
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
