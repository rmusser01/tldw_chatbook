---
id: TASK-497
title: >-
  Image-generation Phase-1 polish follow-ups
status: To Do
assignee: []
created_date: '2026-07-23 12:59'
updated_date: '2026-07-23 12:59'
labels:
  - image-generation
  - followup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Non-blocking polish items surfaced by the whole-branch review of the image-generation multi-provider foundation (Phase 1, PR #800). None block that PR; group them into one cleanup pass. Separate from [[task-485]] (real egress/SSRF hardening) and the deferred Phase-2/3 feature work. See the Phase-1 design spec `Docs/superpowers/specs/2026-07-22-image-generation-multiprovider-foundation-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Demo panel: the `_generate` error branch no longer calls `query_one(...)` on the worker thread — resolve the status widget on the UI thread (or pass it in) so a mid-generation screen pop can't raise uncaught under `@work(exit_on_error=True)`.
- [ ] `http_client.py`: add Google-style docstrings to the public functions; `create_client` treats an explicit `timeout=0` correctly instead of falling back to the default; `DEFAULT_MAX_REDIRECTS` tolerates a malformed `HTTP_MAX_REDIRECTS` env value without raising at import.
- [ ] `worker.py`: add a test covering the adapter-load-failure raise branch (`get_adapter` returns None); `test_worker.py` resets the `get_registry` singleton (autouse fixture) to match the sibling test convention.
- [ ] `Image_Generation/__init__.py`: `__getattr__` raises a descriptive `AttributeError` (module + attribute name) per the PEP 562 idiom.
- [ ] `test_cold_start.py`: restore `sys.modules` after the purge (try/finally or `monkeypatch`) so it doesn't permanently mutate the process-wide module cache for later tests.
- [ ] Demo panel: add the size and steps inputs and the distinct "enabled-but-not-configured" inline message described in design spec §7 (or explicitly record them as intentionally omitted for the throwaway panel).
- [ ] Add opt-in live integration tests (spec §8): one per backend, `@pytest.mark.optional`, skipped unless creds/servers/binary are present.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:NOTES:END -->
