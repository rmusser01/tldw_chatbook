# Guided audio.cpp POSIX Launch and Supervision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn an accepted guided audio.cpp package snapshot into one private generated configuration and run it through Chatbook's existing managed lifecycle on macOS and Linux.

**Architecture:** Preserve the complete guided settings mapping at the provider boundary, then derive an immutable launch snapshot only inside deliberate adapter readiness. A small POSIX materializer performs final recipe/file/backend validation, selects a loopback port, and owns the generated artifact; the existing app-scoped `AudioCppSupervisor` remains the sole child/process authority and retires the artifact with its exact generation. The adapter cross-checks the returned catalog against the accepted recipe projection and publishes typed `tts`/`clone` capabilities.

**Tech Stack:** Python 3.11+, asyncio, Pydantic, stdlib filesystem/socket/process APIs, httpx, pytest.

**ADR required:** no new ADR.

**ADR path:** `backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md` (extends ADR-023).

**Reason:** TASK-13201 directly implements the already-approved generated-configuration and one-supervisor boundary without changing ownership or adding another durable/runtime authority.

---

### Task 1: Preserve guided configuration and typed speech catalog evidence

**Files:**
- Modify: `tldw_chatbook/TTS/audio_cpp_guided_config.py`
- Modify: `tldw_chatbook/TTS/adapter_bootstrap.py`
- Modify: `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py`
- Modify: `tldw_chatbook/TTS/audio_cpp_contract.py`
- Modify: `tldw_chatbook/TTS/adapter_types.py`
- Modify: `tldw_chatbook/TTS/adapters/audio_cpp.py`
- Test: `Tests/TTS/test_audio_cpp_guided_config.py`
- Test: `Tests/TTS/test_audio_cpp_contract.py`
- Test: `Tests/TTS/test_audio_cpp_adapter.py`
- Test: `Tests/TTS/test_stts_settings_reconfiguration.py`

- [x] Write regressions proving provider publication retains the complete guided mapping without process/socket/HTTP/artifact work.
- [x] Run the focused tests and confirm they fail because the current provider projection discards guided fields.
- [x] Add a pure full-settings projector and use it only at the provider factory/publication boundary; retain `AudioCppConfig` for the active transport limits.
- [x] Write exact task-filter tests for lowercase `tts` and `clone`, rejection of other/case-variant tasks, and typed model capability publication.
- [x] Run the focused suites and confirm they pass without changing External or user-JSON behavior.

### Task 2: Materialize one safe generation-local launch artifact

**Files:**
- Create: `tldw_chatbook/TTS/audio_cpp_guided_launch.py`
- Modify: `tldw_chatbook/TTS/audio_cpp_managed_config.py`
- Test: `Tests/TTS/test_audio_cpp_guided_launch.py`

- [x] Write golden tests for the exact top-level/model JSON allowlist, absolute package paths, deterministic option projection, lazy loading, loopback-only host, omitted CORS, and disabled body logging.
- [x] Run the tests and confirm the materializer API is absent.
- [x] Implement final accepted-package revalidation through the sealed recipe registry and bounded scanner on a worker thread.
- [x] Implement POSIX platform/backend intersection, selecting only `expected` or `verified` recipe tuples; resolve Auto from the bounded platform order and reject unsupported explicit overrides.
- [x] Implement a bounded private-port selector and inject it in tests.
- [x] Atomically create an owner-only attempt directory and read-only `server.json` using no-follow descriptors, retaining exact directory/file ownership for cleanup.
- [x] Add stable path-independent validation/materialization/cleanup errors and exception-graph privacy assertions.
- [x] Run the new suite and existing recipe/scanner/config suites until green.

### Task 3: Reuse the existing supervisor for generated launches

**Files:**
- Modify: `tldw_chatbook/TTS/adapters/audio_cpp.py`
- Modify: `tldw_chatbook/TTS/audio_cpp_supervisor.py`
- Modify: `tldw_chatbook/TTS/audio_cpp_managed_config.py`
- Test: `Tests/TTS/test_audio_cpp_managed_integration.py`
- Test: `Tests/TTS/test_audio_cpp_supervisor.py`

- [x] Write a concurrent-first-use regression proving one materialization and one child.
- [x] Run it and confirm current Managed readiness only understands user-provided JSON.
- [x] Add a guided-only preparation lock and lazy launch factory to the adapter while leaving user-JSON validation unchanged.
- [x] Attach generated-artifact ownership and expected-model evidence to `AudioCppManagedLaunchConfig`.
- [x] Make the supervisor validate the immutable artifact before spawn and retire it after generation clients/tasks/drains on every pre-spawn, rollback, exit, stop, restart, and close path.
- [x] Add source-change tests proving a Running generation is reused without reopening source files and a later deliberate replacement revalidates and fails safely.
- [x] Add failure-injection tests for materialization, spawn, cleanup, crash, cancellation, and shutdown with zero child/task/client/artifact leaks.
- [x] Run the managed adapter/supervisor/integration suites until green.

### Task 4: Fence generated catalogs and backend recovery

**Files:**
- Modify: `tldw_chatbook/TTS/adapters/audio_cpp.py`
- Modify: `tldw_chatbook/TTS/audio_cpp_supervisor.py`
- Test: `Tests/TTS/test_audio_cpp_adapter.py`
- Test: `Tests/TTS/test_audio_cpp_managed_integration.py`

- [x] Write regressions for missing, extra, wrong-family, wrong-task, and wrong-mode generated catalog entries.
- [x] Require the exact expected generated model set before publishing capability/catalog evidence.
- [x] Preserve recipe-declared `tts`/`clone` capabilities, including a clone-only catalog model, while excluding every other upstream task.
- [x] Keep automatic backend fallback disabled unless a stable allowlisted backend-unavailable classification is present; prove generic spawn/contract/synthesis failures never fall through to another backend.
- [x] Run catalog, capability, privacy, and managed lifecycle suites until green.

### Task 5: POSIX process evidence and closeout

**Files:**
- Modify: `Tests/TTS/fixtures/fake_audiocpp_server.py`
- Modify: `Tests/TTS/test_audio_cpp_managed_integration.py`
- Modify: `Docs/Development/TTS/TTS_MODULE_GUIDE.md`
- Modify: `backlog/tasks/task-13201 - Generate-and-supervise-guided-audio.cpp-configurations-on-POSIX.md`

- [x] Extend the stdlib real-child fixture to expose the safe generated model list without test-only generated JSON fields.
- [x] Prove direct argv/no shell, generated endpoint, multi-model lazy registration, first synthesis, staged replacement, crash recovery, exact shutdown, and zero artifact/orphan leakage on POSIX.
- [x] Run the complete affected TTS surface outside the sandbox for real loopback/process fixtures.
- [x] Run Ruff, format check, compileall, scoped mypy, privacy/boundary gates, and `git diff --check`.
- [x] Update the module guide, task acceptance checklist, implementation notes, ADR reference, and status only after verification is green.
