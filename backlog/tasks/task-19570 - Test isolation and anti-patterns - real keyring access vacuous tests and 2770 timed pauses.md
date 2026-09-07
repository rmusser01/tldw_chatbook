---
id: TASK-19570
title: >-
  Test isolation and anti-patterns — mounted-app tests touch the real keyring,
  vacuous tests that pass if deleted, and 2,770 timed pauses
status: To Do
assignee: []
created_date: '2026-08-21 20:20'
labels:
  - testing
  - tech-debt
priority: medium
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 5 (test-suite health & guard efficacy)
— its **(e) isolation** and **(d) anti-patterns**. All numbers re-measured at
this branch base.

Calibration, so this is scoped honestly: the lane found the sandbox is
**strong** — two-layer, with **zero leaks in the classic categories** — and the
vacuous-fixture category is **essentially empty** (zero `MagicMock` fixture
bodies tree-wide; autouse fixtures patch supporting seams only). These are the
specific exceptions.

**A — every mounted-app test touches the OS credential subsystem.** CONFIRMED.
`PYTHON_KEYRING_BACKEND` appears in **zero `conftest.py` files anywhere under
`Tests/`**. Its only two occurrences are subprocess env dicts in
`Tests/Packaging/` — which isolate a *spawned* process, not the test session.
`Tests/UI/app_factory.py` contains **no `keyring` string at all**; its patch
stack ends at line 341 and it constructs `TldwCli()` at **line 342**. The
resolution path is unconditional on every construction:
`app.py:5549` `_wire_server_context_provider()` → `app.py:5878`
`build_default_server_credential_store()` →
`runtime_policy/server_credentials.py:296-298` `import keyring;
keyring.get_keyring()`.
On macOS a first read can raise a Keychain consent dialog or block on a locked
keychain. The risk was clearly *known* — `Tests/Packaging/` sets a null backend
for its subprocesses — but the awareness never propagated to the shared
app-construction seam.

**B — tests that would pass if deleted.** All confirmed by reading:
- `Tests/unit/test_core_imports_unit.py` — the entire file is one test whose
  body is **five comment lines and `assert True`**. There is not a single
  import statement inside the function; the module it claims to test is never
  imported.
- **Two** committed debug probes, not one: `Tests/Chat/test__zz_probe.py`
  (timing loop, `print("PROBE", …)`, ends `assert True`) and
  `Tests/Chat/test__zz_probe2.py` (threading probe, **zero assertions of any
  kind**). Both committed in `bd69f4a5e`; both pass today.
- `Tests/RAG/simplified/test_simple_cache_concurrent.py:94
  test_concurrent_clear` and `:276 test_concurrent_expiry_and_access` — each
  runs a `ThreadPoolExecutor` race and then asserts that a **brand-new,
  unrelated key** round-trips (`"final_query"`, `"post_race"`), touched by no
  thread. A completely non-thread-safe cache passes both. A third instance of
  the same pattern sits at `:232 test_thread_interrupt_handling`.
  An exhaustive sweep of every other `ThreadPoolExecutor` test body in `Tests/`
  found no further matches — the other concurrency tests assert on state the
  concurrency actually produced.

**C — the `pilot.pause` family, at scale.** **2,770 numeric `pilot.pause(<n>)`
calls across 229 files.** Worst file: `Tests/Wizards/test_first_run_setup_wizard.py`
— **279 timed pauses over 253 test functions**. Next:
`Tests/UI/test_library_shell.py` (142), `test_library_prompts_canvas.py` (117),
`test_watchlists_destination_shell.py` (117),
`Tests/Watchlists/test_watchlists_collections_screen.py` (114). One test sleeps
0.7 s to outwait a **named production constant** rather than driving a clock,
and a magic `time.sleep(1.6)` comment is copy-pasted across three unrelated
files.

**This is a strategy task, not a mass edit.** 2,770 sites cannot be
hand-converted, and a sweeping rewrite of the UI suite is exactly the
clever-and-unstable option the owner's standing ruling rejects. What is wanted
is a decided direction plus a guard that stops the number growing.

**Do not cite "456 attempt-count loops across 93 files."** This filing tried to
reproduce that figure and could not recover the original pattern; every
plausible definition lands in the hundreds across ~95-205 files. The finding is
directionally sound — if anything understated — but that specific pair of
numbers is not a measured figure.

Also recorded from the same lane: `Tests/Architecture` is 290 tests in 220 s,
and the lane established this is **accidental, not inherent** — each test
re-walks the whole package and the headline test subprocesses the checker for a
**third** full walk. The fix pattern already exists in-repo (`@lru_cache` on
`_measure`).

## Acceptance Criteria

- [ ] `PYTHON_KEYRING_BACKEND` is set to a null backend for the whole test
      session in `Tests/conftest.py`, so no test can reach the OS credential
      store
- [ ] The shared app-construction seam (`Tests/UI/app_factory.py`) patches the
      keyring resolution before `TldwCli()` is constructed
- [ ] A test asserts the real keyring is never resolved during a mounted-app
      test — mutation-checked
- [ ] `Tests/unit/test_core_imports_unit.py` either performs the imports it
      claims to test or is deleted; `test__zz_probe.py` and
      `test__zz_probe2.py` are removed from the suite
- [ ] The three `test_simple_cache_concurrent.py` tests assert on the state
      their concurrency actually produced, and fail against a non-thread-safe
      cache — or are deleted as untestable-as-written
- [ ] A decided direction on timed pauses is recorded (clock-driven waits,
      a condition-based helper, or an accepted-and-bounded status quo) with the
      reasoning, and the highest-value file is converted as a worked example
- [ ] A ratchet prevents the `pilot.pause(<number>)` count from growing beyond
      its current measured value
- [ ] `Tests/Architecture` runtime is reduced by caching the package walk
      (`@lru_cache` on `_measure`, per the in-repo pattern), with before/after
      timings recorded
