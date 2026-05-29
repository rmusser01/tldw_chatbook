# Performance Lag Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce perceived lag in tldw_chatbook by removing eager startup work, preserving mounted UI state where safe, and replacing full transcript remounts with incremental updates.

**Architecture:** Keep the existing Textual screen architecture, but move heavyweight work behind lazy boundaries and add narrow performance guardrails before each change. Treat startup, navigation, and streaming transcript updates as separate surfaces with separate regression tests so each fix can land independently. Avoid assuming Textual `Screen` instances can be safely reused; prove lifecycle behavior first and prefer data/state caching when screen reuse is unsafe.

**Tech Stack:** Python 3.11+, Textual, pytest, Loguru, SQLite, existing tldw_chatbook UI and service modules.

## Closeout Status

Implemented in `codex/performance-lag-remediation` and closed under `TASK-72`. The work covered the planned guardrails, lazy optional dependency checks, lazy app/splash imports, cacheable destination navigation, incremental native console transcript updates, deferred nonessential startup services, metric log gating, Notes duplicate-load removal, and final measurement/verification documentation.

Conditional audit items that did not require extra code changes are recorded in the task notes and closeout artifact: Prometheus registry instrumentation was not gated because the observed high-volume issue was Loguru metric emission, and Search/RAG/Media duplicate-load work did not show the same duplicate owner problem found in Notes during the focused destination pass.

---

## Baseline Evidence

Current local measurements from the performance review:

- `import tldw_chatbook.app`: about 4.5 seconds.
- `TldwCli()` construction after import: about 0.04 seconds.
- Headless startup to `_ui_ready`: about 6.5 seconds.
- Route switches on small local data: about 0.26 to 0.61 seconds.
- Local DBs are small, so the current lag is not primarily local database volume.

The remediation target is not one giant refactor. Each task below should be a separate PR-sized slice with its own tests and before/after measurements.

## File Map

- `tldw_chatbook/Utils/optional_deps.py`: make embeddings/RAG dependency checks truly lazy.
- `tldw_chatbook/Event_Handlers/embeddings_events.py`: preserve explicit embeddings checks on real embeddings actions.
- `tldw_chatbook/app.py`: reduce top-level imports, introduce lazy screen resolution, reduce destination remount cost, defer nonessential startup services, reduce noisy global select logging.
- `tldw_chatbook/UI/Navigation/screen_registry.py`: new lazy route-to-screen registry, if needed.
- `tldw_chatbook/UI/Navigation/main_navigation.py`: keep route message contract unchanged.
- `tldw_chatbook/Widgets/Console/console_transcript.py`: replace full transcript remount refreshes with keyed incremental updates.
- `tldw_chatbook/UI/Screens/chat_screen.py`: avoid unconditional transcript refresh when native console state has not changed.
- `tldw_chatbook/Metrics/metrics_logger.py`: gate normal Loguru metric log emission behind config/env/debug settings.
- `tldw_chatbook/Metrics/metrics.py`: evaluate Prometheus metrics registry overhead separately from log-volume reduction.
- `tldw_chatbook/Utils/Splash_Screens/__init__.py`: stop importing every splash effect on package import, if import profiling still shows this as material after the first two tasks.
- `Tests/Performance/test_app_startup_performance.py`: new structural and optional timed startup/import guardrails.
- `Tests/UI/test_screen_navigation.py`: navigation behavior and cache reuse coverage.
- `Tests/UI/test_console_native_transcript.py`: transcript incremental update and large-transcript behavior coverage.
- `Tests/Utils/test_optional_deps.py`: lazy dependency behavior coverage.

## Definition Of Done

- Performance work is represented by Backlog.md task files before implementation starts, and every task moved to Done has completed acceptance criteria, implementation notes, and verification evidence.
- Import, startup, navigation, and transcript-refresh measurements are captured before and after the work.
- No embeddings/RAG heavyweight dependency check runs during `import tldw_chatbook.app` unless explicitly requested by env/config.
- Common destination navigation avoids unnecessary reload/remount work while preserving explicit cache invalidation on profile, runtime-policy, and context changes.
- Native console transcript updates do not remount the entire message list for appends, streaming content updates, or selection changes.
- Initial `_ui_ready` is not blocked by TTS/STTS, media cleanup, DB-size polling, or other nonessential startup work.
- Normal user interactions do not emit high-volume INFO/METRIC logs unless debug/performance logging is enabled.
- Existing UI smoke tests and focused performance regression tests pass.

---

## Task A: Create Backlog Tracking Tasks

**Files:**

- Created: `backlog/tasks/task-72 - Performance-lag-remediation.md`
- Optional create: child tasks in `backlog/tasks/` if this plan is split across several PRs.

- [x] **Step 1: Create the parent task**

  Created `TASK-72` in `To Do` status:

  Equivalent CLI command:

  ```bash
  backlog task create "Performance lag remediation" -d "Reduce startup, navigation, and console transcript lag by removing eager work and adding performance guardrails." --ac "Startup and navigation measurements are captured before and after remediation,Embeddings and heavy optional dependencies are not checked during plain app import,Console transcript updates avoid full transcript remounts during streaming,Nonessential startup services do not block UI readiness,Focused UI and performance tests pass"
  ```

- [ ] **Step 2: Move the task to In Progress before code changes**

  Run:

  ```bash
  backlog task edit <id> -s "In Progress"
  ```

- [ ] **Step 3: Add this plan to the task**

  Add a concise implementation plan that points to this document and lists the first implementation slice. Do not paste the whole plan into the task; use the task file for current-slice status and closeout notes.

- [ ] **Step 4: Split child tasks only when a slice is too large for one PR**

  If needed, create child tasks in dependency order:

  - performance guardrails and lazy optional deps
  - lazy app import graph
  - transcript incremental rendering
  - deferred startup work
  - destination load deduplication

  Keep each child task independently testable. Do not reference future child task ids until they exist.

---

## Task 0: Add Performance Guardrails

**Files:**

- Create: `Tests/Performance/test_app_startup_performance.py`
- Modify: `pyproject.toml` only if the existing `performance` marker needs adjustment.

- [ ] **Step 1: Add an isolated subprocess helper**

  Add a helper that runs Python snippets in a subprocess with isolated config and data paths. The helper must set:

  ```python
  env = {
      **os.environ,
      "TLDW_TEST_MODE": "1",
      "XDG_DATA_HOME": str(tmp_path / "data"),
      "XDG_CONFIG_HOME": str(tmp_path / "config"),
      "HOME": str(tmp_path / "home"),
  }
  ```

  This is required because pytest's autouse fixture only patches the parent process. Subprocess probes must not read or write the real user config or DBs.

- [ ] **Step 2: Add phase-specific structural import guards**

  Add separate subprocess guards instead of one broad guard:

  1. Optional dependency guard for Task 1:

  ```python
  OPTIONAL_DEPS_IMPORT_GUARDS = (
      "torch",
      "transformers",
      "chromadb",
      "sentence_transformers",
  )
  ```

  This guard imports `tldw_chatbook.Utils.optional_deps`.

  2. App import graph guard for Task 2:

  ```python
  APP_IMPORT_GUARDS = (
      "tldw_chatbook.UI.Evals.evals_window_v3",
      "tldw_chatbook.UI.STTS_Window",
      "tldw_chatbook.UI.SearchWindow",
      "tldw_chatbook.UI.MediaWindow_v2",
  )
  ```

  This guard imports `tldw_chatbook.app`.

  Avoid asserting `scipy` is absent in the Task 1 guard unless the failing import path proves it comes from embeddings checks. `scipy` may also arrive through other eager feature imports and belongs in the Task 2 analysis.

- [ ] **Step 3: Mark guards honestly until their slice lands**

  Do not land permanently failing tests. Use `pytest.mark.xfail(strict=True, reason="...")` for a guard until the corresponding remediation task is implemented, or land the guard in the same PR as the fix.

- [ ] **Step 4: Add optional timed smoke probe**

  Add a `pytest.mark.performance` test that records:

  - import duration
  - `TldwCli()` init duration
  - `run_test()` entry duration
  - time until `_ui_ready`
  - route switch timings for `library`, `notes`, `media`, `search`, `settings`, and `chat`

  Keep hard assertions conservative at first. The structural import guard should be the required CI signal; exact timing can be informational or marked performance until the fixes stabilize.

- [ ] **Step 5: Run the new tests and record the expected failures**

  Run:

  ```bash
  .venv/bin/python -m pytest Tests/Performance/test_app_startup_performance.py -q
  ```

  Expected before implementation: phase-specific guards are either xfailed or fail for the known eager import paths. Timed output should be recorded as baseline evidence, not treated as a stable CI budget yet.

- [ ] **Step 6: Commit only after at least one remediation task makes each guard meaningful**

  Do not land a permanently failing guard. Either mark expected-fail with a clear reason in the first PR or land it together with Task 1.

---

## Task 1: Make Embeddings And Optional Dependency Checks Truly Lazy

**Files:**

- Modify: `tldw_chatbook/Utils/optional_deps.py`
- Modify: `tldw_chatbook/Event_Handlers/embeddings_events.py`
- Test: `Tests/Utils/test_optional_deps.py`
- Test: `Tests/Performance/test_app_startup_performance.py`

- [ ] **Step 1: Write the failing lazy-import test**

  Add a test that imports `tldw_chatbook.Utils.optional_deps` in the isolated subprocess helper from Task 0 and asserts the subprocess output/logs do not include:

  - `Checking embeddings dependencies early`
  - imports of `torch`, `transformers`, `chromadb`, or `sentence_transformers`

- [ ] **Step 2: Keep explicit eager mode test coverage**

  Add or update coverage so `TLDW_EAGER_DEPENDENCY_CHECK=true` still calls `initialize_dependency_checks()` and checks configured dependencies. Use isolated subprocess envs for eager-mode coverage too, because import-time behavior is the behavior under test.

- [ ] **Step 3: Remove the unconditional embeddings check**

  In `optional_deps.py`, remove this import-time block:

  ```python
  if 'PYTEST_CURRENT_TEST' not in os.environ:
      logger.info("Checking embeddings dependencies early to ensure UI loads correctly...")
      check_embeddings_rag_deps()
  ```

  Keep `check_embeddings_rag_deps()`, `force_recheck_embeddings()`, and explicit event-handler calls intact.

- [ ] **Step 4: Represent unknown dependency state explicitly**

  If any UI needs to distinguish "not checked yet" from "checked and missing", add a small status helper rather than reading `DEPENDENCIES_AVAILABLE["embeddings_rag"]` as final truth before a check has run.

- [ ] **Step 5: Verify explicit embeddings actions still check dependencies**

  Run:

  ```bash
  .venv/bin/python -m pytest Tests/Utils/test_optional_deps.py Tests/RAG/simplified/test_embeddings_wrapper.py -q
  ```

- [ ] **Step 6: Re-run the relevant performance guards**

  Run:

  ```bash
  .venv/bin/python -m pytest Tests/Performance/test_app_startup_performance.py -q
  ```

  Expected after this task: the optional-deps guard passes. The app import graph guard may still xfail until Task 2 because `app.py` still eagerly imports screen and feature modules.

---

## Task 2: Lazy-Load Application Screens And Heavy Feature Windows

**Files:**

- Create: `tldw_chatbook/UI/Navigation/screen_registry.py`
- Modify: `tldw_chatbook/app.py`
- Test: `Tests/UI/test_screen_navigation.py`
- Test: `Tests/Performance/test_app_startup_performance.py`

- [ ] **Step 1: Inventory app import ownership before moving imports**

  Classify top-level imports in `tldw_chatbook/app.py` into:

  - needed for `TldwCli` class definition or compose-time shell chrome
  - needed only when resolving a destination screen
  - needed only when handling a specific event
  - unused legacy window imports

  Do not move imports blindly. Some services are used during `TldwCli.__init__`; those should be delayed only when the owning initialization path is also delayed and tested.

- [ ] **Step 2: Add a lazy screen registry test**

  In `Tests/UI/test_screen_navigation.py`, add a test that resolves every visible shell destination route and verifies it returns the same screen class name as today.

  The test should not require importing all screen modules during `import tldw_chatbook.app`.

- [ ] **Step 3: Create a lazy screen target data structure**

  In `screen_registry.py`, define route entries that store:

  - screen name
  - canonical tab id
  - module path
  - class name

  Add a loader that imports the module only when the route is requested.

- [ ] **Step 4: Move destination screen imports out of app module import time**

  In `app.py`, remove top-level imports for heavyweight screen modules where possible and replace `_resolve_screen_navigation_target()` internals with the lazy registry.

  Keep lightweight constants and message classes at module import time.

- [ ] **Step 5: Move legacy feature-window imports behind their use sites**

  `app.py` also imports legacy windows and feature windows that are not required for a plain app import, including Evals, STTS, Search, Media, and Chatbooks windows. Move these behind their use sites or remove them if they are no longer used.

- [ ] **Step 6: Preserve current route aliases**

  Ensure existing route aliases still work:

  - `chat`
  - `library`
  - `notes`
  - `media`
  - `search`
  - `settings`
  - all routes covered by `SHELL_DESTINATION_ORDER`

- [ ] **Step 7: Add structural import assertions**

  Update the performance guard to assert that these are not loaded after plain app import:

  - `tldw_chatbook.UI.Evals.evals_window_v3`
  - `tldw_chatbook.UI.STTS_Window`
  - media/search/admin windows not needed for the startup route

- [ ] **Step 8: Verify navigation and import behavior**

  Run:

  ```bash
  .venv/bin/python -m pytest Tests/UI/test_screen_navigation.py Tests/UI/test_master_shell_navigation.py Tests/Performance/test_app_startup_performance.py -q
  ```

  Expected after this task: plain import gets materially faster, and route resolution still reaches every destination.

---

## Task 3: Reduce Destination Remount Cost Safely

**Files:**

- Modify: `tldw_chatbook/app.py`
- Potentially modify: destination-specific services or state holders after measurement.
- Test: `Tests/UI/test_screen_navigation.py`
- Test: `Tests/UI/test_product_maturity_phase1_navigation_smoke.py`

- [ ] **Step 1: Run a Textual lifecycle spike**

  Before implementing caching, write a small local experiment or focused test that answers whether a previously switched-away `Screen` instance can be safely reattached with current Textual behavior. Record the result in the task implementation notes.

- [ ] **Step 2: Write the remount-cost regression test**

  Add coverage that navigates `chat -> library -> chat` and verifies the second `chat` navigation avoids the expensive work identified by measurement. Prefer asserting against load counters, preserved state snapshots, or cached service data instead of asserting raw `Screen` object identity.

- [ ] **Step 3: Choose the lowest-risk caching strategy**

  Choose in this order:

  1. Destination data/service cache with explicit invalidation.
  2. Persistent destination content mounted inside a stable shell, if Textual supports it cleanly.
  3. Reuse `Screen` instances only if the lifecycle spike proves this is safe.

  This repository guidance says caches should be cleared on context switch, so every cache must define invalidation for profile changes, runtime-policy source changes, workspace/context changes, and explicit refresh actions.

- [ ] **Step 4: Add a cache policy only for the chosen strategy**

  Add a small allowlist for destinations where cache reuse is expected:

  ```python
  CACHEABLE_SCREEN_ROUTES = {
      TAB_CHAT,
      TAB_LIBRARY,
      TAB_NOTES,
      TAB_MEDIA,
      TAB_SEARCH,
      TAB_SETTINGS,
  }
  ```

  Do not cache modal/transient screens.

- [ ] **Step 5: Implement the selected reuse path**

  If the lifecycle spike proves screen reuse is safe, add app helpers that:

  - return cached screens for cacheable routes
  - create fresh screens for non-cacheable routes
  - invalidate cached screens on runtime-policy or profile changes
  - preserve current `save_state()` / `restore_state()` compatibility

  If screen reuse is not safe, implement only destination data/state caching in this task and leave screen reuse out of scope.

- [ ] **Step 6: Measure route switch deltas**

  Run the performance probe from Task 0 and compare the route timings with the baseline.

  Target after this task on small local data:

  - repeat navigation does not rerun the expensive data loads discovered by the probe
  - no visible duplicate screen mount/load work in logs
  - route timings improve or remain stable without stale data

- [ ] **Step 7: Verify smoke navigation**

  Run:

  ```bash
  .venv/bin/python -m pytest Tests/UI/test_screen_navigation.py Tests/UI/test_product_maturity_phase1_navigation_smoke.py -q
  ```

---

## Task 4: Incrementalize Native Console Transcript Rendering

**Files:**

- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_native_transcript.py`
- Test: `Tests/UI/test_console_native_chat_flow.py`
- Test: `Tests/UI/test_console_internals_decomposition.py`

- [ ] **Step 1: Introduce row-level test seams**

  Add or expose a test-friendly way to observe transcript reconciliation, such as stable row ids, render signatures, or an internal update counter. Avoid tests that depend on arbitrary Textual child widget object identity.

- [ ] **Step 2: Write append preservation test**

  In `test_console_native_transcript.py`, mount a transcript with many messages, append one message, refresh, and assert existing message rows were not rebuilt according to the row-level seam from Step 1.

- [ ] **Step 3: Write streaming update test**

  Mount a transcript with an assistant message, update only that message content/status, refresh, and assert unrelated rows were not remounted.

- [ ] **Step 4: Write selection update test**

  Select one message, then another, and assert only the previous and new selected rows/action rows changed.

- [ ] **Step 5: Add row identity and signatures**

  In `ConsoleTranscript`, track row objects by message id and a render signature such as:

  - role
  - content
  - status
  - selected state
  - variant selection state
  - available actions state, if needed

- [ ] **Step 6: Replace whole-list `remove_children()` refresh**

  Replace `refresh_messages()` with an incremental reconciler:

  - remove rows for deleted message ids
  - mount rows for new message ids
  - update or replace only rows whose signature changed
  - update the empty state only when the transcript transitions between empty and non-empty

- [ ] **Step 7: Skip transcript refresh when nothing changed**

  In `chat_screen.py`, store a lightweight native transcript fingerprint before calling `refresh_messages()`. If the active session id and message signatures are unchanged, skip the refresh.

- [ ] **Step 8: Keep the legacy fallback behavior intact**

  The legacy chat-log fallback can remain full-rendered for now because the native transcript is the main path. Do not expand this task into a full legacy chat rewrite.

- [ ] **Step 9: Verify transcript behavior**

  Run:

  ```bash
  .venv/bin/python -m pytest Tests/UI/test_console_native_transcript.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_internals_decomposition.py -q
  ```

  Expected after this task: streaming and long-chat updates no longer scale with total transcript length on every poll.

---

## Task 5: Move Nonessential Startup Work To Idle Or First Use

**Files:**

- Modify: `tldw_chatbook/app.py`
- Potentially create: `tldw_chatbook/Services/startup_tasks.py`
- Test: `Tests/Performance/test_app_startup_performance.py`
- Test: focused UI tests for TTS/STTS controls if present.

- [ ] **Step 1: Write readiness ordering test**

  Add a test that patches TTS/STTS initialization to sleep or record calls and asserts `_ui_ready` can become true before those optional services finish.

- [ ] **Step 2: Defer TTS/STTS initialization**

  Replace awaited startup initialization with an idle/background initializer:

  - create safe handler placeholders synchronously so existing controls can query service state without attribute errors
  - create real handler objects lazily
  - initialize on first use or after `_ui_ready`
  - expose a "not ready yet" state to controls if needed
  - do not block app readiness on failures

- [ ] **Step 3: Add service-state tests for deferred TTS/STTS**

  Add focused tests that assert TTS/STTS controls, events, or command paths handle the placeholder state correctly before the real service has initialized.

- [ ] **Step 4: Defer DB-size initial update**

  Keep the footer widget, but schedule the first `db_status_manager.update_db_sizes()` after readiness or on a short idle timer. The footer can show a pending/unknown value until then.

- [ ] **Step 5: Defer media cleanup**

  Keep periodic cleanup, but avoid `call_later(self.perform_media_cleanup)` immediately after startup. Use one of:

  - a delayed timer after `_ui_ready`
  - first visit to media/settings cleanup UI
  - manual cleanup only when `cleanup_on_startup` is false

  Preserve the user-configured cleanup policy, but stop running it in the initial responsiveness window.

- [ ] **Step 6: Move splash completion before optional work**

  Ensure splash removal and `_ui_ready = True` are based on essential UI mount/binding only, not optional services.

- [ ] **Step 7: Verify startup readiness**

  Run:

  ```bash
  .venv/bin/python -m pytest Tests/Performance/test_app_startup_performance.py Tests/UI/test_product_maturity_phase1_first_run.py -q
  ```

  Expected after this task: `total_to_ready_s` falls, even if deferred tasks continue after readiness.

---

## Task 6: Reduce Normal Logging And Metric Overhead

**Files:**

- Modify: `tldw_chatbook/Metrics/metrics_logger.py`
- Inspect, then modify only if measured: `tldw_chatbook/Metrics/metrics.py`
- Modify: `tldw_chatbook/app.py`
- Modify: high-volume call sites only if metrics remain noisy after central gating.
- Test: new focused tests in `Tests/Utils/` or `Tests/Performance/test_app_startup_performance.py`

- [ ] **Step 1: Add metric gating tests**

  Verify metric logging can be disabled for normal runs and enabled explicitly through env/config.

  Suggested env name:

  ```text
  TLDW_METRICS_LOGGING=1
  ```

- [ ] **Step 2: Gate Loguru metric emission**

  In `metrics_logger.py`, make `_log_metric()` return early unless metrics logging is enabled. Keep counters/histograms callable so instrumentation sites do not need broad changes.

- [ ] **Step 3: Measure Prometheus registry overhead separately**

  Startup code also calls `tldw_chatbook.Metrics.metrics` functions. Do not disable Prometheus metrics as part of Loguru log-volume cleanup unless profiling shows registry/label overhead is material. If it is material, add a separate config/env gate with tests.

- [ ] **Step 4: Lower select-change logging**

  In `app.py:on_select_changed`, change the always-on INFO log to DEBUG. Remove the duplicate `conv-char-character-select` branch while staying behaviorally equivalent.

- [ ] **Step 5: Avoid token-counter work from synthetic mount select events**

  If initial widget population fires `Select.Changed` events, guard token-counter updates so they run only after `_ui_ready` and only for real Chat provider/model changes.

- [ ] **Step 6: Verify log volume reduction**

  Run the startup/navigation probe and compare stderr/log size with the baseline. Normal route switching should not produce a flood of path-validation and metric lines.

- [ ] **Step 7: Run focused tests**

  Run:

  ```bash
  .venv/bin/python -m pytest Tests/UI/test_screen_navigation.py Tests/Performance/test_app_startup_performance.py -q
  ```

---

## Task 7: Audit And Deduplicate Destination Mount Data Loading

**Files:**

- Modify after measurement, likely:
  - `tldw_chatbook/UI/Notes_Window.py`
  - `tldw_chatbook/Notes/Notes_Library.py`
  - Search/RAG destination screen modules
  - Media destination screen modules
- Test:
  - `Tests/UI/test_destination_shells.py`
  - `Tests/UI/test_product_maturity_gate16_library_search_rag.py`
  - screen-specific Notes/Media/Search tests.

- [ ] **Step 1: Instrument destination mount load counts**

  Add test-only counters, monkeypatch spies, or scoped instrumentation around notes list loads, RAG profile loads, media select initialization, and search history loads. Avoid leaving permanent noisy instrumentation in normal UI paths.

- [ ] **Step 2: Write duplicate-load regression tests**

  For each heavy destination, navigate to the route once and assert initial data population happens once unless explicitly refreshed by the user. Use the isolated `run_test()` environment from the existing UI harness so tests do not depend on the developer's real local DB size.

- [ ] **Step 3: Fix Notes duplicate list loading**

  Remove duplicated Notes population paths between screen `on_mount` and `NotesTabInitializer.on_tab_shown()`. Keep one owner for initial load and one explicit refresh path.

- [ ] **Step 4: Fix Search/RAG initial load duplication**

  Load RAG profiles/search history once per mounted destination or cache them behind a short-lived service/state object. Do not check embeddings dependencies as part of rendering static Search UI.

- [ ] **Step 5: Fix Media mount churn**

  Avoid repeated config/path validation and synthetic select-change side effects on media route mount. Cache stable config-derived options for the mounted screen lifetime.

- [ ] **Step 6: Verify each destination remains refreshable**

  Manual or automated explicit refresh actions should still reload data. The fix is to remove accidental duplicate initial loads, not to make data stale.

- [ ] **Step 7: Run focused destination tests**

  Run:

  ```bash
  .venv/bin/python -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_product_maturity_gate16_library_search_rag.py -q
  ```

---

## Task 8: Closeout Measurement And Release Notes

**Files:**

- Modify: `Docs/superpowers/qa/performance/2026-05-28-lag-remediation-closeout.md`
- Potentially modify: `README.md` or relevant docs if startup/config behavior changes.

- [ ] **Step 1: Capture final performance measurements**

  Run the same probe used in the initial review:

  ```bash
  .venv/bin/python -X importtime -c "import tldw_chatbook.app"
  .venv/bin/python -m pytest Tests/Performance/test_app_startup_performance.py -q
  ```

- [ ] **Step 2: Compare before/after**

  Record:

  - import duration
  - time to `_ui_ready`
  - route switch timings
  - long transcript append/update timing
  - log volume during startup/navigation

- [ ] **Step 3: Run final focused suite**

  Run:

  ```bash
  .venv/bin/python -m pytest Tests/Utils/test_optional_deps.py Tests/UI/test_screen_navigation.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_internals_decomposition.py Tests/Performance/test_app_startup_performance.py -q
  ```

- [ ] **Step 4: Run broader suite or agreed CI subset**

  If full local pytest is too slow, run the largest relevant UI subset and document any skipped optional-dependency suites.

- [ ] **Step 5: Document known residual risks**

  Include:

  - Textual screen caching limitations, if any.
  - Any remaining screen-specific mount cost.
  - Whether splash-screen auto-discovery still contributes material startup time.
  - Whether representative test-user data still shows lag after architectural fixes.

- [ ] **Step 6: Close Backlog task hygiene**

  Before marking the Backlog task Done:

  - check every acceptance criterion in the task file
  - add implementation notes with modified files, measurement deltas, and known residual risks
  - run the focused verification suite or document why a broader suite could not be run
  - update task status to Done with the Backlog CLI
