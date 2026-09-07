---
id: TASK-497
title: Image-generation Phase-1 polish follow-ups
status: Done
assignee: []
created_date: '2026-07-23 12:59'
updated_date: '2026-07-25 06:55'
labels:
  - image-generation
  - followup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Non-blocking polish items surfaced by the whole-branch review of the image-generation multi-provider foundation (Phase 1, PR #800). None block that PR; group them into one cleanup pass. Separate from [[task-498]] (real egress/SSRF hardening) and the deferred Phase-2/3 feature work. See the Phase-1 design spec `Docs/superpowers/specs/2026-07-22-image-generation-multiprovider-foundation-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Demo panel: the `_generate` error branch no longer calls `query_one(...)` on the worker thread — resolve the status widget on the UI thread (or pass it in) so a mid-generation screen pop can't raise uncaught under `@work(exit_on_error=True)`.
- [x] #2 `http_client.py`: add Google-style docstrings to the public functions; `create_client` treats an explicit `timeout=0` correctly instead of falling back to the default; `DEFAULT_MAX_REDIRECTS` tolerates a malformed `HTTP_MAX_REDIRECTS` env value without raising at import.
- [x] #3 `worker.py`: add a test covering the adapter-load-failure raise branch (`get_adapter` returns None); `test_worker.py` resets the `get_registry` singleton (autouse fixture) to match the sibling test convention.
- [x] #4 `Image_Generation/__init__.py`: `__getattr__` raises a descriptive `AttributeError` (module + attribute name) per the PEP 562 idiom.
- [x] #5 `test_cold_start.py`: restore `sys.modules` after the purge (try/finally or `monkeypatch`) so it doesn't permanently mutate the process-wide module cache for later tests.
- [x] #6 Demo panel: add the size and steps inputs and the distinct "enabled-but-not-configured" inline message described in design spec §7 (or explicitly record them as intentionally omitted for the throwaway panel).
- [x] #7 Add opt-in live integration tests (spec §8): one per backend, `@pytest.mark.optional`, skipped unless creds/servers/binary are present.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
0. Verify-first pass against current code for all 7 ACs before writing any fix:
   - AC1 (demo error branch thread-safety): already fixed -- _generate()'s except branch
     already calls self.app.call_from_thread(self._set_status, ...), no query_one on the
     worker thread. No code change.
   - AC2 docstrings + malformed-env handling: already present (_int_env try/except,
     Google-style docstrings on all public fns). GENUINE GAP: create_client(timeout=0)
     falls back to _DEFAULT_TIMEOUT via `timeout or _DEFAULT_TIMEOUT` (0 is falsy) --
     needs an `is None` check + a regression test.
   - AC3: no test exists for get_adapter()->None (adapter-load-failure) raise branch in
     run_generation; test_worker.py has no autouse get_registry-singleton reset fixture
     (sibling convention in test_adapter_registry.py / test_listing.py). Both missing --
     implement both, TDD.
   - AC4: __getattr__ raises bare AttributeError(name); PEP 562 idiom wants
     "module {name!r} has no attribute {attr!r}". Missing -- fix + test.
   - AC5: test_cold_start.py deletes sys.modules entries and never restores them --
     permanent process-wide mutation for later tests. Missing -- wrap in try/finally.
   - AC6: demo panel has no size/steps inputs and no distinct
     enabled-but-not-configured inline message (only a generic Error: ... on failure).
     Given the panel is explicitly throwaway and P2a already shipped the real Console
     chat card (console_generation_card.py) as the production proof surface, record as
     intentionally omitted per the AC's own escape hatch, with rationale in the notes.
   - AC7: no live/optional backend tests exist. Missing -- implement one
     @pytest.mark.optional test per backend (stable_diffusion_cpp, swarmui, openrouter,
     novita, together, modelstudio), mirroring Tests/Chat/test_live_thinking_provider_apis.py's
     pytestmark + _required_env-skip convention.
1. AC2 fix: http_client.create_client -- replace `timeout or _DEFAULT_TIMEOUT` with an
   explicit `is None` check; add a regression test asserting an explicit timeout=0 is
   preserved on the built httpx.Client.
2. AC3 fix: add test_worker.py::test_run_generation_adapter_load_failure_raises (patches
   get_registry to return a resolver where get_adapter() -> None) asserting
   ImageGenerationError; add the autouse _reset fixture (get_registry/reset_registry)
   matching the sibling tests.
3. AC4 fix: Image_Generation/__init__.py __getattr__ raises
   f"module {__name__!r} has no attribute {name!r}"; add a regression test.
4. AC5 fix: test_cold_start.py wraps the sys.modules purge/import in try/finally,
   restoring the pre-purge modules afterward so later tests are unaffected.
5. AC6: no code change; document the omission rationale in Implementation Notes.
6. AC7: new Tests/Image_Generation/test_live_backends.py, pytestmark
   [integration, optional, slow], one test per backend gated on env vars
   (OPENROUTER_API_KEY / NOVITA_API_KEY / TOGETHER_API_KEY / DASHSCOPE_API_KEY /
   TLDW_LIVE_SWARMUI_BASE_URL / TLDW_LIVE_SD_CPP_BINARY+MODEL), each building a real
   request via worker.build_request/run_generation end to end.
7. Run the Image_Generation test suite (foreground, chunked), ruff on changed files,
   `python -c "import tldw_chatbook.app"`; update AC checkboxes + Implementation Notes;
   set status Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
A verify-first pass found 4 of 7 items already satisfied by earlier review-fix waves; only the genuine gaps got code/test changes (TDD: failing test added first, then the fix, confirmed green).

- **AC1 (already done)** — `UI/Screens/image_gen_demo_screen.py:117-121`: the `_generate` except branch already runs `self.app.call_from_thread(self._set_status, f"Error: {exc}")`; no `query_one` on the worker thread. No change.
- **AC2** — docstrings and the `HTTP_MAX_REDIRECTS` malformed-env guard (`_int_env` try/except) were already in place (already done). Genuine bug found and fixed: `create_client(timeout=0)` used `timeout or _DEFAULT_TIMEOUT`, so an explicit `0` (falsy) silently became 120s. Changed to `_DEFAULT_TIMEOUT if timeout is None else timeout` (`Image_Generation/http_client.py:101-105`). Added `test_create_client_respects_explicit_zero_timeout` + `test_create_client_defaults_when_timeout_omitted` (`Tests/Image_Generation/test_http_client.py`); the first reproduced the bug (asserted `Timeout(120.0) == Timeout(0)` failed) before the fix.
- **AC3** — no test existed for the `get_adapter() -> None` raise branch, and `test_worker.py` had no autouse registry-reset fixture (sibling convention in `test_adapter_registry.py`/`test_listing.py`). Added both: `test_run_generation_adapter_load_failure_raises` (fake registry whose `get_adapter` returns `None`, asserts `ImageGenerationError` matching "failed to load") and an autouse `_reset` fixture calling `reset_registry()` before/after each test (`Tests/Image_Generation/test_worker.py`).
- **AC4** — `__getattr__` raised bare `AttributeError(name)`. Changed to `AttributeError(f"module {__name__!r} has no attribute {name!r}")` (`Image_Generation/__init__.py:26`), the standard PEP 562 idiom. Added `test_getattr_raises_descriptive_attribute_error` (`Tests/Image_Generation/test_package_skeleton.py`), which failed against the old bare message before the fix.
- **AC5** — `test_cold_start.py` deleted `sys.modules` entries and never restored them, permanently losing the fully-loaded modules for every later test in the process. Wrapped the purge/import in try/finally: snapshot the purged entries first, restore them afterward, and also drop any newly-imported entries that weren't in the original snapshot so the cache ends up byte-for-byte where it started (`Tests/Image_Generation/test_cold_start.py`). Verified manually that a sentinel attribute set on the pre-test module survives the round trip via `sys.modules` (the one edge case *not* covered — and not covered by a bare `monkeypatch.delitem` either — is the parent package's `Image_Generation` attribute binding, which Python's `import a.b` machinery updates independently of `sys.modules`; out of scope for what the AC asks).
- **AC6 (recorded as intentionally omitted)** — no size/steps inputs or a distinct enabled-but-not-configured message were added to the throwaway demo panel. The AC's own text allows this: "or explicitly record them as intentionally omitted for the throwaway panel." Rationale: per the design spec (§9 Phase roadmap) the demo panel was explicitly a Phase-1 proof surface to be replaced in Phase 2, and Phase 2's real Console chat card (`Widgets/Console/console_generation_card.py`, shipped in P2a / PR #832) is now the production proof surface. Investing further UI polish in a screen slated for deletion isn't worth it; the panel already surfaces the one condition needed to unblock a demo run (no backends enabled at all, via the Select's sentinel `"none"` value) via `_set_status`.
- **AC7** — no live/optional backend tests existed. Added `Tests/Image_Generation/test_live_backends.py`: one test per backend (`stable_diffusion_cpp`, `swarmui`, `openrouter`, `novita`, `together`, `modelstudio`), `pytestmark = [integration, optional, slow]` mirroring `Tests/Chat/test_live_thinking_provider_apis.py`'s `_required_env`-skip convention. Each test drives the real public entry point (`worker.build_request` + `worker.run_generation`) end to end. Gating: API-key backends skip unless their real key env var is set (`OPENROUTER_API_KEY`/`NOVITA_API_KEY`/`TOGETHER_API_KEY`/`DASHSCOPE_API_KEY`, read directly by `config._resolve_secret`, no monkeypatching needed); `swarmui` skips unless `TLDW_LIVE_SWARMUI_BASE_URL` points at a reachable server; `stable_diffusion_cpp` skips unless `TLDW_LIVE_SD_CPP_BINARY` + `TLDW_LIVE_SD_CPP_MODEL_PATH` resolve to real files on disk. An autouse fixture resets the config cache and registry singleton around each test. Verified all 6 skip cleanly by default and, with `--run-slow`, skip with the correct per-backend reason (no live creds available in this environment) rather than erroring.

Files changed: `tldw_chatbook/Image_Generation/http_client.py`, `tldw_chatbook/Image_Generation/__init__.py`, `Tests/Image_Generation/test_http_client.py`, `Tests/Image_Generation/test_worker.py`, `Tests/Image_Generation/test_package_skeleton.py`, `Tests/Image_Generation/test_cold_start.py`, `Tests/Image_Generation/test_live_backends.py` (new). No production changes to `worker.py` or `image_gen_demo_screen.py` were needed — only test coverage.

Verification: `pytest Tests/Image_Generation/ -q` → 51 passed, 6 skipped (0 failed); `ruff check` clean on all changed files; `python -c "import tldw_chatbook.app"` clean.
<!-- SECTION:NOTES:END -->
