# 2026-05-28 Performance Lag Remediation Closeout

## Scope

This closeout covers TASK-72 remediation for perceived lag during import, startup readiness, route switching, console transcript updates, and high-volume metric logging.

## Baseline

Measurements from the approved plan:

- Plain `import tldw_chatbook.app`: about 4.5 seconds.
- `TldwCli()` construction after import: about 0.04 seconds.
- Headless startup to `_ui_ready`: about 6.5 seconds.
- Route switches on small local data: about 0.26 to 0.61 seconds.

## After Measurements

Fresh isolated local probe with `TLDW_TEST_MODE=1`, temporary config/data paths, splash disabled for the UI run, and startup media cleanup disabled for the UI run:

- Plain `import tldw_chatbook.app`: 4.365 seconds.
- Patched test-app construction: 0.057 seconds.
- `run_test()` to `_ui_ready`: 0.492 seconds.
- Route switches:
  - library: 0.286 seconds.
  - notes: 0.274 seconds.
  - media: 0.547 seconds.
  - search: 0.373 seconds.
  - settings: 0.360 seconds.
  - chat: 0.341 seconds.
- Captured stderr log volume:
  - import probe: 93 lines, 0 `METRIC` lines.
  - UI startup/navigation probe: 153 lines, 0 `METRIC` lines.

## Changes Verified

- Optional dependency checks no longer run heavyweight embeddings/RAG checks during plain optional-deps import unless `TLDW_EAGER_DEPENDENCY_CHECK=true`.
- Plain app import no longer loads guarded legacy feature windows or splash effect modules.
- Common destination screens are lazily resolved and allowlisted routes reuse cached screen instances.
- TTS/STTS initialization, DB-size initial update, and startup media cleanup no longer block `_ui_ready`.
- Native Console transcript updates reconcile rows incrementally and skip unchanged native transcript refreshes.
- Normal metric logging is disabled unless `TLDW_METRICS_LOGGING=1`.
- Notes screen navigation refreshes scope once per entry instead of through duplicate screen and tab-initializer owners.

## Verification

- `Tests/Performance/test_app_startup_performance.py Tests/UI/test_product_maturity_phase1_first_run.py`: 18 passed.
- `Tests/UI/test_screen_navigation.py Tests/Performance/test_app_startup_performance.py Tests/Utils/test_metrics_logger.py`: 40 passed.
- `Tests/UI/test_destination_shells.py Tests/UI/test_product_maturity_gate16_library_search_rag.py`: 119 passed.
- Final focused suite: `Tests/Utils/test_optional_deps.py Tests/Utils/test_metrics_logger.py Tests/Utils/test_startup_polish_regressions.py Tests/UI/test_screen_navigation.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_internals_decomposition.py Tests/Performance/test_app_startup_performance.py`: 212 passed.
- Static checks: targeted `py_compile` passed and `git diff --check` reported no whitespace errors.

## Residual Risks

- Plain app import is only modestly improved. It still imports broad non-splash subsystems, including chunking/ingestion paths, during `tldw_chatbook.app` import.
- Splash effect discovery is now lazy, but splash card definitions and the SplashScreen widget still remain in the app import graph.
- Startup measurements use the existing test harness with patched local services, not a real large user database.
- Long transcript behavior is verified structurally through row reconciliation and refresh-skip tests; no stable wall-clock transcript budget was added in this slice.
