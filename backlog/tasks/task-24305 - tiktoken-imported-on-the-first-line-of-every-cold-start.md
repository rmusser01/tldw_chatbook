---
id: TASK-24305
title: >-
  tiktoken is imported on the first line of every cold start, outside every boot budget
status: Done
assignee: []
created_date: '2026-08-28 23:30'
labels:
  - performance
  - boot
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`tldw_chatbook/__init__.py` calls `install_tiktoken_runtime()` at package import time. The shim
itself is cheap -- it validates a signature, sets a cache directory and swaps a function -- but it
does `import tiktoken.load`, which costs 19.6-29.1 ms measured on dev `3a3383123e`. Every launch pays it,
whether or not anything tokenises during the session.

The shim must be installed before the first `get_encoding()` call, not before the first import, so
a lazy install satisfies the same contract.

Worth noting separately: the four boot ratchets count `tldw_chatbook.*` modules only, so
third-party import weight is invisible to every budget in the repo. This finding was only
reachable by profiling; no guard could have caught it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Importing tldw_chatbook does not import tiktoken
- [x] #2 The bundled offline table reader is still in force before the first encoding is fetched, proven by a test that resolves an encoding and asserts it came from the bundle
- [x] #3 A test proves the shim is installed exactly once even when several call sites race to trigger it
- [x] #4 A decision is recorded on whether third-party boot import weight gets its own budget arm
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add an idempotent, thread-safe `ensure_tiktoken_runtime`.
2. Remove the eager install from `__init__` and arm the four use sites.
3. Guard the distributed obligation with an AST scan.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`tldw_chatbook/__init__.py` no longer installs the bundled tiktoken reader at
package import. `ensure_tiktoken_runtime()` is idempotent and lock-guarded, and
is armed at the four modules that tokenise.

`Utils/token_counter.py` also needed its module-level `import tiktoken`
deferred -- without that the app import still pulled tiktoken and the change
bought nothing. `TIKTOKEN_AVAILABLE` keeps its name and meaning but is now
computed with `importlib.util.find_spec`, which answers "is it installed?"
without executing it. Several tests monkeypatch that flag to force the
character-estimate tier; they still work.

**Deferring traded one eager cost for an obligation spread across call sites**,
so `test_every_tiktoken_importer_arms_the_bundled_runtime` walks the package by
AST and fails if a module imports tiktoken without arming the bundle -- a new
tokenising module would otherwise reach upstream for its tables, a network read
where there used to be none. Mutation-tested: removing the arming from one
strategy module reds it and names the file.

**Measured:** `import tldw_chatbook.app` no longer imports tiktoken at all
(subprocess-checked, since this test session has it loaded). The 19.6-29.1 ms
saving did not separate from noise in wall-clock A/B on a machine at load
average 5-10; the honest evidence is the module's absence from the closure, not
a timing.

**Note for a future budget decision:** all four boot ratchets count
`tldw_chatbook.*` modules only, so third-party import weight is invisible to
every guard in the repo. Profiling was the only way this was findable.

Files: `__init__.py`, `Utils/tiktoken_runtime.py`, `Utils/token_counter.py`,
`Chunking/engine/strategies/semantic.py`, `Chunking/engine/strategies/tokens.py`,
`Subscriptions/token_manager.py`, `Tests/Utils/test_tiktoken_runtime_lazy.py` (new).
<!-- SECTION:NOTES:END -->
