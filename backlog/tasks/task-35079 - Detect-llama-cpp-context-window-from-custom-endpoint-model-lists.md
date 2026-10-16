---
id: TASK-35079
title: Detect llama.cpp context window from custom-endpoint model lists
status: Done
assignee:
- '@claude'
created_date: 2026-10-15 18:55
labels:
- bug
- llama.cpp
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A user reported that the context size is not detected automatically when using llama.cpp. Live verification against llama-server 0.5.0 showed the dedicated llama_cpp family works (its /props probe parses default_generation_settings.n_ctx correctly), but configuring the same server under a generic OpenAI-compatible family (custom, custom_2, or an openai_compatible registry entry) never detects: the probe's /v1/models record scan only checks four flat keys (max_model_len, context_length, context_window, max_context_length), while llama.cpp reports the window under a nested meta object (meta.n_ctx serving window, meta.n_ctx_train training bound). Every such user sees the unverified 32,000-token application fallback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A llama.cpp server configured under a generic OpenAI-compatible family has its /v1/models meta.n_ctx resolved as the context window and shown as verified server metadata
- [x] #2 When a model record exposes only the training bound (meta.n_ctx_train, e.g. router mode with the model not yet loaded), that value is used instead of the 32,000-token application fallback
- [x] #3 A supported-family probe that completes without a usable window logs one credential-free debug line identifying family and endpoint, so "not detected" reports are diagnosable
- [x] #4 Existing /props detection for the llama_cpp and local_llamacpp families and all existing context-window tests still pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing tests to Tests/Chat/test_console_context_window.py reproducing llama.cpp's real /v1/models record shape: meta.n_ctx present (must win over n_ctx_train), and only n_ctx_train present
2. Watch them fail against the current flat-key record scan
3. Extend _probe's generic record scan to read a nested meta dict: serving n_ctx first, generic aliases, n_ctx_train last
4. Add one debug log when a supported-family probe completes without a usable window
5. Re-run the module tests; live-verify against a local llama-server with the custom family

ADR required: no -- routine bug fix inside the existing metadata-probe seam; no storage, schema, or provider-boundary change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause (verified live against llama-server 0.5.0 before and after): the dedicated llama_cpp family was never broken (its /props probe parsed `default_generation_settings.n_ctx` correctly); the gap was the generic family record scan. llama.cpp reports the window in /v1/models under a nested `meta` object (`meta.n_ctx` = per-slot serving window, `meta.n_ctx_train` = training bound), and `_probe` only checked four flat record keys, so custom/custom_2/openai_compatible configurations always fell back to the unverified 32,000-token estimate.

Changes in `Chat/console_context_window.py`:

- The model-record scan now also reads a nested `meta` dict: `n_ctx` first (serving value wins over `n_ctx_train`), then the generic aliases, then `n_ctx_train` as the last resort (router mode with the model registered but not yet loaded serves only the training bound; a 0 slot value is already rejected by `positive_window`).
- The probe's supported-family set moved to a module constant `_SUPPORTED_PROBE_FAMILIES` (same membership), and `_probe` is now a thin wrapper: unprobed families return None silently as before, while a supported-family probe that completes without a usable window logs one credential-free debug line ("Context-window metadata probe found no usable window (family=… endpoint=…)") via `log_origin` — TTL-throttled by the cache, so at most one line per 60 s per target. Previously every miss was silent, which is why "context size not detected" reports were undiagnosable.

Tests in `Tests/Chat/test_console_context_window.py` (TDD: watched all three fail first — meta cases returned (32000, 'application fallback', False), log case found 0 lines):

- `test_llama_cpp_custom_endpoint_model_list_reports_nested_meta` — llama.cpp's real /v1/models record shape; serving `meta.n_ctx` wins over `meta.n_ctx_train`, and a train-bound-only record (8192/32768 respectively) resolves as verified server metadata.
- `test_windowless_probe_logs_one_credential_free_debug_line` — exactly one debug line on a windowless supported-family probe, carrying family and endpoint, and no line for an unprobed (openai) family.

Evidence: module file 39/39 passed; `Tests/UI/test_console_context_window_modal.py` 3/3; `Tests/Performance/test_ui_ready_module_census.py` 4/4 (the module already imported httpx/egress at call time; the added `loguru` import is boot-resident everywhere). `Tests/UI/test_console_endpoint_discovery.py::test_shared_evidence_change_refreshes_the_console_once` fails, and fails identically with the UNMODIFIED module (A/B by copy-aside, no stash): pre-existing dev drift in the private-profile harness (`Tests/private_profile.py:382`), not this change; it needs its own task. Live verification with the homebrew llama-server 0.5.0 on 127.0.0.1:8931 (-c 8192): custom family now returns (8192, 'server metadata', verified=True), previously (32000, 'application fallback', False); llama_cpp family unchanged at (8192, verified=True). Repro script kept at /tmp/repro_real_llama.py.

Scope note: this fix does not make the model picker popover probe the server (it stays catalog-only by design for keystroke latency), and router-mode detection still requires the model to be loaded or at minimum a train-bound-only record — both are pre-existing design boundaries, now debuggable via the new log line.

ADR check: no ADR — routine bug fix inside the existing metadata-probe seam; no storage, schema, or provider-boundary change.

Files: `tldw_chatbook/Chat/console_context_window.py`, `Tests/Chat/test_console_context_window.py`, `backlog/tasks/task-35079 - Detect-llama-cpp-context-window-from-custom-endpoint-model-lists.md`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
