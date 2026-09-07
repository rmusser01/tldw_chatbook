# Managed audio.cpp Runtime Core Implementation Plan

> **For the implementing agent:** REQUIRED SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Use subagent-driven development only if the user explicitly requests delegation. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the dormant managed audio.cpp runtime core so one application-owned supervisor can validate, lazily launch, supervise, restart, and stop one user-provided loopback `audiocpp_server` while preserving the existing External adapter and complete-WAV contract.

**Architecture:** Extend the existing active-mode audio.cpp configuration projection, add one provider-specific `AudioCppSupervisor`, and inject that single supervisor into every audio.cpp adapter materialized by the app-owned registry. The registry remains lease and exclusive-transition authority; the service stages managed configuration while it is unapplied and prepares the latest eligible generation only at a deliberate operation. Slice 4 exposes typed service APIs and deterministic tests but no Managed selector, lifecycle button, diagnostic panel, or other Slice 5 UI.

**Tech Stack:** Python 3.11+, asyncio subprocesses, httpx, Pydantic v2, pytest/pytest-asyncio, Ruff, mypy, Textual application bootstrap, existing `TTSAdapterRegistry` and `TTSService`.

## Global Constraints

- Missing `[app_tts.audio_cpp].mode` means `external`; existing external mappings and requests remain unchanged.
- Managed launch is loopback-only and accepts `host` exactly `127.0.0.1` plus an integer `port` from 1 through 65,535; booleans are invalid.
- The only launch vector is `[managed_binary_path, "--config", managed_server_json_path]`; never use a shell or expose arbitrary arguments.
- Startup timeout defaults to 30 seconds and accepts 1–300 seconds.
- Health interval defaults to 10 seconds and accepts 2–300 seconds.
- Termination grace defaults to 5 seconds and accepts 0.1–60 seconds.
- Read at most 1 MiB from `server.json`; reject larger input before JSON parsing.
- Strict JSON rejects duplicate keys and the non-JSON constants `NaN`, `Infinity`, and `-Infinity` at every depth.
- Diagnostics retain at most 200 lines, 64 KiB of sanitized UTF-8 display text, and 4 KiB per display line.
- One exit monitor is the sole reaper for each spawned child. Startup, stop, rollback, and terminal close signal the child and await that monitor; they never race it with a second `wait()`.
- Parent-side stdout/stderr cleanup is bounded after child exit even when an unowned descendant inherited a pipe descriptor. Chatbook never signals or adopts that descendant.
- Periodic and request-triggered health checks share at most one generation-bound in-flight task scheduled by the supervisor through an adapter-owned probe.
- Application shutdown establishes one TTS deadline and drains registry leases before managed-child termination. After registry cleanup completes, fails, or times out, `TTSService` always performs the sole terminal `AudioCppSupervisor.close()` plus `wait_closed()` under that deadline.
- Managed adapter retirement is generation-local: after its leases drain, it performs an idempotent nonterminal stop of only its bound process generation, joins that generation's health/startup/exit/output tasks, and only then closes remaining HTTP/cache resources. It never terminal-closes the app-scoped supervisor or stops a replacement generation.
- Complete PCM16 WAV buffering, structural validation, one-item async delivery, request limits, and no-fallback behavior remain unchanged.
- Normal CI uses fakes and a controlled local subprocess fixture; it requires no audio.cpp binary, model, download, audio hardware, or external network.
- Slice 4 adds no managed UI, no process controls in Settings, no persistent diagnostics, no automatic restart, no process adoption, and no generic process-supervisor framework.

ADR required: yes

ADR paths:

- `backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md`
- `backlog/decisions/039-global-and-studio-tts-settings-ownership.md`

Reason: TASK-3792 implements the accepted provider-runtime, process-ownership, staged-configuration, shutdown, security, and cross-module service boundaries. Both ADRs were amended and landed before this plan; no third ADR is required.

---

## File and Interface Map

### New production modules

- `tldw_chatbook/TTS/audio_cpp_managed_config.py` — validates the user-selected executable and bounded `server.json`, derives one immutable loopback launch snapshot, and projects the sanitized child environment.
- `tldw_chatbook/TTS/audio_cpp_supervisor.py` — owns the exact child, process generation, startup/health/exit/drain tasks, state snapshots, and bounded in-memory diagnostics.

### Existing production modules to modify

- `tldw_chatbook/TTS/audio_cpp_config.py` — add active-mode `external|managed` projection and managed timing/path fields while keeping the default external mapping byte-for-byte equivalent.
- `tldw_chatbook/TTS/adapter_types.py` — extend the bounded operation-code vocabulary for managed failures; do not add process methods to the provider-neutral adapter protocol.
- `tldw_chatbook/TTS/adapters/audio_cpp.py` — preserve the HTTP contract while binding managed HTTP clients and caches to one supervisor process generation.
- `tldw_chatbook/TTS/adapter_bootstrap.py` — construct exactly one supervisor and inject it into the lazy audio.cpp factory plus `TTSService`.
- `tldw_chatbook/TTS/adapter_registry.py` — retain latest staged provider configuration and add one exclusive transition that drains leases before a supplied provider lifecycle action.
- `tldw_chatbook/TTS/TTS_Generation.py` — stage managed saves, prepare staged configuration on deliberate operations, expose process lifecycle APIs, and close the supervisor at terminal service shutdown.
- `tldw_chatbook/TTS/request_admission.py` — prepare the selected provider before entering the read-side resolution/admission gate and use already-prepared internal catalog readers while that gate is held.
- `tldw_chatbook/TTS/effective_settings.py` — expose one provider-only projection used by both pre-resolution and full effective resolution so precedence cannot diverge.
- `tldw_chatbook/TTS/__init__.py` — export immutable process snapshots and public Slice 4 service result types.
- `Docs/Development/TTS/TTS_MODULE_GUIDE.md` — document the dormant managed core and the still-deferred UI.

### New focused tests and fixtures

- `Tests/TTS/test_audio_cpp_managed_config.py`
- `Tests/TTS/test_audio_cpp_supervisor.py`
- `Tests/TTS/test_audio_cpp_managed_integration.py`
- `Tests/TTS/fixtures/fake_audiocpp_server.py`

### Existing regression files to extend or run

- `Tests/TTS/test_audio_cpp_config.py`
- `Tests/TTS/test_audio_cpp_adapter.py`
- `Tests/TTS/test_adapter_types.py`
- `Tests/TTS/test_adapter_registry.py`
- `Tests/TTS/test_tts_registry_service.py`
- `Tests/TTS/test_tts_request_admission.py`
- `Tests/TTS/test_stts_settings_reconfiguration.py`
- `Tests/TTS/test_effective_settings.py`
- `Tests/TTS/test_tts_profile_capabilities.py`
- `Tests/TTS/test_tts_settings_capability_observations.py`
- `Tests/TTS/test_tts_logging_privacy.py`
- `Tests/UI/test_settings_audio_cpp_experience_model.py`
- `Tests/UI/test_settings_speech_tts_panel.py`
- `Tests/Subscriptions/test_briefing_audio_synthesis.py`
- `Tests/TTS_Events/test_spoken_feedback_streaming.py`
- `Tests/TTS_Events/test_utterance_speech_entry.py`
- `Tests/UI/test_speech_tts_settings_ownership_closeout.py`
- `Tests/UI/test_uat_first_time_character_chat.py`

## Produced Interfaces

The implementation uses these exact public contracts:

`TTSOperationCode` adds only these provider-neutral process outcomes:

```python
"port_in_use"
"process_spawn_failed"
"process_startup_timeout"
"process_exited"
"runtime_unhealthy"
```

Managed configuration/path failures continue to use `configuration_invalid`;
contract and zero-model outcomes continue to use `contract_incompatible` and
`not_configured`; an exclusive transition continues to use
`TTSProviderReconfiguringError`. This keeps the failure vocabulary bounded
without disguising distinct process remedies as an ordinary connection error.

Every managed failure maps exactly as follows. Messages stored in process state
are fixed application strings, never exception text, paths, origins, request
text, or child output.

| Condition | Code/result | Retryable | Recovery action |
|---|---|---:|---|
| Active managed mapping or launch file is invalid | `configuration_invalid` | No | `open_settings` |
| Configured loopback port is occupied or preflight is ambiguous | `port_in_use` | Yes | `open_settings` |
| The exact executable cannot be spawned | `process_spawn_failed` | Yes | `retry` |
| Adapter-owned generation HTTP resources cannot be initialized transactionally | `process_spawn_failed` | Yes | `retry` |
| Readiness exceeds the one startup deadline | `process_startup_timeout` | Yes | `open_diagnostics` |
| Child exits early or unexpectedly, or output supervision fails and the child is stopped | `process_exited` | Yes | `open_diagnostics` |
| `/health` plus `/v1/models` is incompatible | `contract_incompatible` | No | `open_settings` |
| Contract is valid but exposes no TTS model | `not_configured` | No | `open_settings` |
| A live child remains unhealthy after the immediate probe | `runtime_unhealthy` | Yes | `restart_managed` |
| Provider is Draining or Stopping | existing `TTSProviderReconfiguringError` | Yes | `retry` |

The supervisor stores only the latest process-lifecycle failure. A new process
generation clears the prior failure; a failed new generation records its own
fixed result; first successful Running leaves it clear; a successful health
probe that restores Unhealthy to Running clears `runtime_unhealthy`; terminal
close clears it. A successful retry of a previously failed registry transition
clears the slot's sealed Unavailable result.

~~~python
# tldw_chatbook/TTS/audio_cpp_managed_config.py
@dataclass(frozen=True, slots=True)
class AudioCppManagedLaunchConfig:
    binary_path: Path
    server_json_path: Path
    working_directory: Path
    base_url: str
    startup_timeout_seconds: float
    health_check_interval_seconds: float
    termination_grace_seconds: float


def validate_audio_cpp_managed_launch(
    config: AudioCppConfig,
) -> AudioCppManagedLaunchConfig: ...


def build_audio_cpp_child_environment(
    source: Mapping[str, str],
    *,
    provider_credential_names: AbstractSet[str],
) -> dict[str, str]: ...


def collect_provider_credential_environment_names(
    app_config: Mapping[str, Any],
) -> frozenset[str]: ...
~~~

~~~python
# tldw_chatbook/TTS/audio_cpp_supervisor.py
AudioCppProcessState = Literal[
    "stopped",
    "starting",
    "running",
    "unhealthy",
    "draining",
    "stopping",
    "unavailable",
]
AudioCppTTSCapability = Literal["available", "not_configured", "unknown"]


@dataclass(frozen=True, slots=True)
class AudioCppReadyEndpoint:
    base_url: str
    process_generation: int
    observation_version: int


@dataclass(frozen=True, slots=True)
class AudioCppProcessFailure:
    process_generation: int | None
    code: TTSOperationCode
    message: str
    retryable: bool
    recovery_action: str | None


@dataclass(frozen=True, slots=True)
class AudioCppProcessAdmissionSnapshot:
    lifecycle_epoch: int
    process_generation: int
    state: AudioCppProcessState
    stage_application_eligible: bool


@dataclass(frozen=True, slots=True)
class AudioCppGenerationHooks:
    contract_probe: AudioCppContractProbe
    health_probe: AudioCppHealthProbe
    cleanup: AudioCppGenerationCleanup


@dataclass(frozen=True, slots=True)
class AudioCppProcessSnapshot:
    state: AudioCppProcessState
    process_generation: int
    observation_version: int
    endpoint: str | None
    tts_capability: AudioCppTTSCapability
    consecutive_health_failures: int
    last_failure: AudioCppProcessFailure | None
    diagnostics: tuple[AudioCppDiagnosticLine, ...]
    dropped_diagnostic_lines: int


class AudioCppSupervisor:
    def snapshot(self) -> AudioCppProcessSnapshot: ...

    def admission_snapshot(self) -> AudioCppProcessAdmissionSnapshot: ...

    async def ensure_running(
        self,
        launch: AudioCppManagedLaunchConfig,
        *,
        generation_hooks_factory: Callable[
            [int], Awaitable[AudioCppGenerationHooks]
        ],
        require_existing: AudioCppProcessAdmissionSnapshot | None = None,
    ) -> AudioCppReadyEndpoint: ...

    async def begin_draining(self) -> None: ...

    async def stop(
        self,
        *,
        application_shutdown: bool = False,
        expected_process_generation: int | None = None,
    ) -> None: ...

    async def close(self) -> None: ...

    async def wait_closed(self) -> None: ...
~~~

`stage_application_eligible` is true only in Stopped or Unavailable with no
owned child and no retained startup, stop, or close transition. When
`require_existing` is supplied, `ensure_running()` may return only the same
still-live process generation. If its lifecycle epoch or generation changed,
it raises one private retry signal before spawning; the service releases its
lease and repeats preparation so an obsolete applied mapping can never launch.
The supervisor awaits `generation_hooks_factory` exactly once only after a
successful spawn has received its process generation and its exit/output tasks
exist; an already Running generation never invokes a caller's replacement factory. The adapter
therefore owns all HTTP resources while the supervisor owns their scheduling
and exactly-once cleanup boundary. When `expected_process_generation` is
provided, `stop()` is a no-op unless that exact generation is still current;
manual lifecycle calls omit it and stop the current owned generation.

The registry adds these exact methods:

~~~python
@dataclass(frozen=True, slots=True)
class TTSProviderConfigurationSnapshot:
    revision: int
    applied_generation: int
    applied_config: Mapping[str, Any]
    staged_generation: int | None
    staged_config: Mapping[str, Any] | None


async def provider_configuration_snapshot(
    self,
    provider_id: str,
) -> TTSProviderConfigurationSnapshot: ...


async def stage_provider_configuration(
    self,
    provider_id: str,
    config: Mapping[str, Any],
    *,
    generation: int,
) -> ReconfigureResult: ...


async def run_exclusive_provider_transition(
    self,
    provider_id: str,
    *,
    on_draining: Callable[[], Awaitable[None]],
    action: Callable[[], Awaitable[None]],
    apply_staged: bool,
) -> ReconfigureResult: ...
~~~

The service adds these exact public lifecycle methods:

~~~python
def audio_cpp_process_snapshot(self) -> AudioCppProcessSnapshot: ...


async def start_and_test_audio_cpp(self) -> TTSProviderCatalog: ...


async def restart_audio_cpp(self) -> TTSProviderCatalog | None: ...


async def shutdown_audio_cpp(self) -> None: ...
~~~

`restart_audio_cpp()` returns `None` when the latest staged mode is External; that operation applies External and stops the owned child without connecting. `start_and_test_audio_cpp()` uses the active applied mode while a child exists; when no child exists, it first promotes the latest eligible stage. Managed then launches lazily, while External performs the existing explicit catalog check.

---

### Task 1: Extend active-mode configuration and strict managed launch validation

**Files:**

- Create: `tldw_chatbook/TTS/audio_cpp_managed_config.py`
- Modify: `tldw_chatbook/TTS/audio_cpp_config.py`
- Test: `Tests/TTS/test_audio_cpp_managed_config.py`
- Test: `Tests/TTS/test_audio_cpp_config.py`

**Interfaces:**

- Consumes: existing `AudioCppConfig.from_mapping()`, `AudioCppConfig.to_mapping()`, common HTTP and response limits.
- Produces: `AudioCppManagedLaunchConfig`, `validate_audio_cpp_managed_launch()`, and active-mode mappings consumed by the supervisor and adapter factory.

- [ ] **Step 1: Add red tests for backward-compatible active-mode projection**

Add tests with these names:

~~~python
def test_missing_mode_projects_the_existing_external_mapping() -> None: ...
def test_external_projection_ignores_malformed_dormant_managed_fields() -> None: ...
def test_managed_projection_ignores_malformed_dormant_external_origin() -> None: ...
def test_to_mapping_contains_only_active_mode_fields_and_common_limits() -> None: ...
def test_managed_timing_defaults_and_finite_bounds_are_exact() -> None: ...
def test_managed_timing_rejects_booleans_nan_and_infinities() -> None: ...
~~~

Assert that `AudioCppConfig().to_mapping()` remains equal to the pre-Slice-4 external dictionary and contains no managed key.

- [ ] **Step 2: Run the projection tests and verify the intended failure**

Run:

~~~bash
.venv/bin/python -m pytest \
  Tests/TTS/test_audio_cpp_config.py \
  Tests/TTS/test_audio_cpp_managed_config.py \
  -q
~~~

Expected: the new tests fail because `managed` and its fields are not accepted.

- [ ] **Step 3: Implement the minimal active-mode model**

Extend `AudioCppConfig` with these fields and defaults:

~~~python
mode: Literal["external", "managed"] = "external"
base_url: str = "http://127.0.0.1:8080"
managed_binary_path: str = ""
managed_server_json_path: str = ""
managed_startup_timeout_seconds: float = 30.0
managed_health_check_interval_seconds: float = 10.0
managed_termination_grace_seconds: float = 5.0
~~~

`from_mapping()` must inspect `mode` first and copy only:

- `mode`, `base_url`, and common fields for External;
- `mode`, the five managed fields, and common fields for Managed.

`to_mapping()` must emit the same active registry set. Preserve the current
projection behavior for unknown keys: ignore keys outside the approved active
field set and never forward them to a provider or process. Do not introduce a
new rejection rule in Slice 4. Retain fixed value-independent diagnostics for
approved fields. The raw Settings owner remains responsible for retaining a
dormant External `base_url`; this registry projection must never rewrite the
raw provider table.

- [ ] **Step 4: Add red tests for paths and bounded JSON**

Add these cases:

~~~python
def test_managed_binary_requires_absolute_executable_regular_file(tmp_path: Path) -> None: ...
def test_managed_binary_preserves_an_approved_symlink_path(tmp_path: Path) -> None: ...
def test_server_json_requires_readable_regular_utf8_object(tmp_path: Path) -> None: ...
def test_server_json_rejects_more_than_one_mib_before_parsing(tmp_path: Path) -> None: ...
def test_server_json_rejects_duplicate_keys_at_every_depth(tmp_path: Path) -> None: ...
@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_server_json_rejects_non_json_numeric_constants_at_every_depth(
    constant: str,
    tmp_path: Path,
) -> None: ...
@pytest.mark.parametrize("host", ["localhost", "::1", "0.0.0.0", "127.0.0.2", "example.test"])
def test_server_json_requires_exact_ipv4_loopback(host: str, tmp_path: Path) -> None: ...
@pytest.mark.parametrize("port", [None, True, False, 0, 65536, "8080", 3.5])
def test_server_json_requires_explicit_integer_port(port: object, tmp_path: Path) -> None: ...
def test_launch_snapshot_uses_json_parent_as_cwd_and_derives_origin(tmp_path: Path) -> None: ...
~~~

- [ ] **Step 5: Implement launch validation without side effects**

Use `Path.expanduser()`, require an absolute selected path, use `Path.is_file()` so a valid symlink target is accepted, and use `os.access(path, os.X_OK)` immediately before returning. Persist and return the selected path rather than `resolve()`.

Read with `open("rb")` after a size check, cap the read at `1_048_577` bytes,
reject more than `1_048_576`, decode strict UTF-8, and parse with an
`object_pairs_hook` that raises on any duplicate key plus a `parse_constant`
callback that rejects `NaN`, `Infinity`, and `-Infinity`. Python's default
`json.loads()` acceptance of those constants is not strict JSON. Require one
top-level dictionary and derive `http://127.0.0.1:<port>`.

Do not open a socket, execute the file, reinterpret model/backend fields, or rewrite JSON.

- [ ] **Step 6: Run configuration tests**

Run:

~~~bash
.venv/bin/python -m pytest \
  Tests/TTS/test_audio_cpp_config.py \
  Tests/TTS/test_audio_cpp_managed_config.py \
  -q
~~~

Expected: PASS.

- [ ] **Step 7: Commit configuration validation**

~~~bash
git add \
  tldw_chatbook/TTS/audio_cpp_config.py \
  tldw_chatbook/TTS/audio_cpp_managed_config.py \
  Tests/TTS/test_audio_cpp_config.py \
  Tests/TTS/test_audio_cpp_managed_config.py
git commit -m "feat(tts): validate managed audio.cpp launch config"
~~~

---

### Task 2: Add sanitized child environment and bounded diagnostics

**Files:**

- Modify: `tldw_chatbook/TTS/audio_cpp_managed_config.py`
- Create: `tldw_chatbook/TTS/audio_cpp_supervisor.py`
- Test: `Tests/TTS/test_audio_cpp_managed_config.py`
- Test: `Tests/TTS/test_audio_cpp_supervisor.py`
- Test: `Tests/TTS/test_tts_logging_privacy.py`

**Interfaces:**

- Consumes: a caller-supplied environment mapping and the repository credential-name set.
- Produces: `collect_provider_credential_environment_names()`, `build_audio_cpp_child_environment()`, immutable diagnostic lines, and the private bounded ring consumed by `AudioCppSupervisor`.

- [ ] **Step 1: Add red environment allowlist tests**

Add `test_credential_inventory_includes_fixed_and_configured_provider_names()`
before the allowlist cases below.

The allowlist is exactly:

~~~python
{
    "PATH", "PATHEXT", "SystemRoot", "SYSTEMROOT", "WINDIR", "COMSPEC",
    "HOME", "USER", "LOGNAME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
    "APPDATA", "LOCALAPPDATA", "PROGRAMDATA",
    "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE",
    "TMPDIR", "TMP", "TEMP",
    "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH",
    "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUDA_PATH",
    "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "HIP_PATH", "ROCM_PATH",
    "VK_ICD_FILENAMES", "VK_LAYER_PATH",
    "GGML_METAL_PATH_RESOURCES", "GGML_VK_VISIBLE_DEVICES",
}
~~~

Test that an allowlisted name is still dropped when it is in
`provider_credential_names` or its case-folded name contains `api_key`,
`apikey`, `token`, `secret`, `password`, `credential`, `authorization`, or
`auth`. The production caller supplies a frozen inventory containing every
current provider credential environment name; the pattern check remains the
defense for future names. Test that values and omitted names never appear in
logs.

`collect_provider_credential_environment_names()` returns the union of the
fixed current LLM/TTS provider names and valid `api_key_env_var` names found in
the bounded `api_settings` provider mappings. At minimum, pin the current
OpenAI, Anthropic, Cohere, DeepSeek, Google, Groq, Hugging Face, Mistral,
Moonshot, OpenRouter, ZAI, and ElevenLabs environment variables. Reject
non-string, blank, or non-environment-identifier configured names. This helper
returns names only, never resolves or retains their values.

- [ ] **Step 2: Run the environment tests and verify failure**

Run:

~~~bash
.venv/bin/python -m pytest \
  Tests/TTS/test_audio_cpp_managed_config.py \
  Tests/TTS/test_tts_logging_privacy.py \
  -q
~~~

Expected: new allowlist tests fail because the builder does not exist.

- [ ] **Step 3: Implement the environment builder**

Start from an empty dictionary, iterate only the fixed allowlist, drop credential collisions and conservative secret-pattern names, and copy exact string values. Never log a source name, copied value, or omitted value.

- [ ] **Step 4: Add red diagnostic-ring tests**

Add deterministic tests named:

~~~python
def test_diagnostics_bound_lines_total_utf8_bytes_and_each_line() -> None: ...
def test_diagnostics_flush_an_overlong_stream_without_waiting_for_newline() -> None: ...
def test_diagnostics_replacement_decode_invalid_utf8() -> None: ...
def test_diagnostics_remove_ansi_controls_and_escape_rich_markup() -> None: ...
def test_diagnostics_redact_credentials_and_normalize_home_prefix() -> None: ...
def test_diagnostics_report_eviction_count_and_clear_per_generation() -> None: ...
def test_diagnostics_never_emit_to_python_or_loguru_logs(caplog: pytest.LogCaptureFixture) -> None: ...
~~~

Use synthetic tokens and paths only.

- [ ] **Step 5: Implement the bounded ring and immutable snapshots**

Add:

~~~python
@dataclass(frozen=True, slots=True)
class AudioCppDiagnosticLine:
    stream: Literal["stdout", "stderr"]
    text: str
~~~

The private ring must increment a dropped-line counter for evictions, retain no raw bytes after sanitization, clear at each new process generation and terminal close, and return tuple snapshots. The later UI derives its visible truncation marker from `dropped_diagnostic_lines`; do not insert a fake stdout/stderr line. Process chunks incrementally so a child that never writes a newline cannot grow the pending buffer beyond 4 KiB plus one decoder fragment. Redact assignment-labeled secret values and bearer tokens with fixed patterns, normalize the current user's home prefix to `~`, and keep the required potentially-sensitive warning for Slice 5.

- [ ] **Step 6: Run privacy and diagnostic tests**

Run:

~~~bash
.venv/bin/python -m pytest \
  Tests/TTS/test_audio_cpp_managed_config.py \
  Tests/TTS/test_audio_cpp_supervisor.py \
  Tests/TTS/test_tts_logging_privacy.py \
  -q
~~~

Expected: PASS.

- [ ] **Step 7: Commit environment and diagnostic primitives**

~~~bash
git add \
  tldw_chatbook/TTS/audio_cpp_managed_config.py \
  tldw_chatbook/TTS/audio_cpp_supervisor.py \
  Tests/TTS/test_audio_cpp_managed_config.py \
  Tests/TTS/test_audio_cpp_supervisor.py \
  Tests/TTS/test_tts_logging_privacy.py
git commit -m "feat(tts): bound audio.cpp child diagnostics"
~~~

---

### Task 3: Implement one generation-safe AudioCppSupervisor

**Files:**

- Modify: `tldw_chatbook/TTS/audio_cpp_supervisor.py`
- Modify: `tldw_chatbook/TTS/adapter_types.py`
- Test: `Tests/TTS/test_audio_cpp_supervisor.py`
- Test: `Tests/TTS/test_adapter_types.py`

**Interfaces:**

- Consumes: `AudioCppManagedLaunchConfig`, injected process launcher, port preflight, monotonic clock/sleep, and an adapter-owned generation-hooks factory.
- Produces: `AudioCppSupervisor.ensure_running()`, immutable state snapshots, exact-child stop/close, and stable managed `TTSOperationError` results.

- [ ] **Step 1: Add red state and lazy-construction tests**

Add:

~~~python
def test_construction_is_stopped_and_performs_no_io() -> None: ...
async def test_first_deliberate_use_starts_one_generation() -> None: ...
async def test_concurrent_first_use_shares_one_start_task() -> None: ...
async def test_one_waiter_cancellation_does_not_cancel_shared_start() -> None: ...
async def test_generation_hooks_factory_runs_once_only_for_new_generation() -> None: ...
async def test_generation_hooks_factory_failure_rolls_back_child_and_uses_safe_code() -> None: ...
async def test_new_generation_clears_prior_diagnostics() -> None: ...
async def test_starting_and_stopping_are_never_stage_application_eligible() -> None: ...
async def test_process_snapshot_retains_only_fixed_safe_last_failure() -> None: ...
async def test_successful_new_generation_clears_prior_failure() -> None: ...
~~~

Use injected futures/events; do not use wall-clock sleeps.

- [ ] **Step 2: Add red launch and rollback tests**

Cover:

~~~python
async def test_occupied_port_fails_closed_without_spawn_or_adoption() -> None: ...
async def test_spawn_uses_exact_argv_cwd_stdin_and_environment() -> None: ...
async def test_early_exit_rolls_back_monitor_and_both_drains() -> None: ...
async def test_startup_timeout_kills_exact_child_and_joins_generation_tasks() -> None: ...
async def test_contract_failure_rolls_back_before_running() -> None: ...
async def test_zero_tts_models_reaches_running_not_configured() -> None: ...
async def test_stale_generation_cannot_publish_running() -> None: ...
async def test_require_existing_generation_refuses_to_spawn_after_concurrent_exit() -> None: ...
async def test_stop_during_pre_spawn_startup_invalidates_and_joins_it() -> None: ...
async def test_stop_during_post_spawn_startup_rolls_back_exact_child() -> None: ...
~~~

- [ ] **Step 3: Run the launch tests and verify failure**

Run:

~~~bash
.venv/bin/python -m pytest Tests/TTS/test_audio_cpp_supervisor.py -q
~~~

Expected: failures identify missing supervisor methods and state transitions.

- [ ] **Step 4: Implement direct process creation and shared startup**

The default launcher must call:

~~~python
await asyncio.create_subprocess_exec(
    str(launch.binary_path),
    "--config",
    str(launch.server_json_path),
    cwd=str(launch.working_directory),
    env=child_environment,
    stdin=asyncio.subprocess.DEVNULL,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
)
~~~

One retained startup task captures the immutable launch snapshot and lifecycle
epoch; a process generation is assigned only after spawn succeeds. Each
ordinary waiter uses `asyncio.shield()` so
its cancellation cannot cancel shared startup. Accepted manual stop, restart,
and terminal close increment the lifecycle epoch, cancel the retained startup,
and join it under the retained cleanup path. Startup rechecks its epoch
immediately before spawn and before Running publication; cancellation after
spawn enters complete exact-child rollback. Revalidate files and strict JSON
immediately before port preflight and spawn.

The injected and default launchers return one private owned-process handle that
contains the exact `asyncio.subprocess.Process` plus an idempotent parent-pipe
closer. The default implementation still calls `create_subprocess_exec()` with
the exact arguments below and encapsulates any runtime-specific pipe-transport
access inside that one tested helper.

- [ ] **Step 5: Implement advisory loopback port preflight**

Use an injected preflight in tests. The default implementation performs one
`asyncio.open_connection("127.0.0.1", port)` attempt capped by the lesser of
one second and the remaining startup deadline; proxies are irrelevant. A
successful connection means occupied; connection refusal means advisory
availability; any ambiguous OS error returns a fixed safe preflight failure
instead of adopting or probing the listener. Close and join the successful
probe writer before returning the occupied-port result.

- [ ] **Step 6: Implement readiness deadline and complete rollback**

Use one monotonic deadline through health polling and `contract_probe`. The probe returns `"available"` or `"not_configured"`. Before publishing Running, confirm the generation is still current and the exact child is alive.

Immediately after spawn, start one retained exit-monitor/finalizer. It is the
only code path that awaits `process.wait()`. Startup, stop, rollback, and close
may inspect `returncode`, signal only the stored child, and await the retained
monitor; none performs a second wait. Under the lifecycle lock, the monitor
classifies its generation as current expected exit, current unexpected exit, or
stale before it publishes state.

On error or cancellation before Running: mark expected exit, terminate then
kill under the remaining deadline, await the sole exit monitor, allow at most
one second (and never beyond the outer deadline) for natural stdout/stderr EOF,
then cancel unfinished drains and invoke the owned handle's parent-pipe closer.
Invoke generation cleanup once, clear exact-child references, and publish
Unavailable with the fixed mapped failure. An inherited descriptor held by an
unowned descendant must not extend this cleanup and the descendant is never
signalled.

- [ ] **Step 7: Add red health and exit tests**

~~~python
async def test_periodic_health_probes_never_overlap() -> None: ...
async def test_periodic_and_immediate_health_probes_share_one_inflight_probe() -> None: ...
async def test_concurrent_unhealthy_requests_share_one_immediate_probe() -> None: ...
async def test_two_failures_mark_unhealthy_and_one_success_recovers() -> None: ...
async def test_request_probe_failure_does_not_kill_unhealthy_child() -> None: ...
async def test_unexpected_exit_invalidates_generation_without_restart() -> None: ...
async def test_later_deliberate_use_starts_one_replacement() -> None: ...
async def test_expected_stop_is_not_reported_as_a_crash() -> None: ...
async def test_health_probe_waiter_cancellation_does_not_cancel_shared_probe() -> None: ...
~~~

- [ ] **Step 8: Implement health scheduling and exit monitoring**

Delay the configured interval after each completed probe. Bind every probe and
publication to the current process generation. The supervisor schedules the
adapter-supplied probe but never imports `httpx` or parses HTTP. A retained
generation-bound in-flight probe task is shared by periodic, immediate, and
concurrent request callers; waiters shield it from individual cancellation and
no two probes overlap. The adapter marks catalog evidence stale on the first
failed result. If `ensure_running()` encounters an owned Unhealthy child,
perform exactly one immediate bounded shared probe: success restores Running
and proceeds, while failure records/raises `runtime_unhealthy` without stopping
or replacing the child. The exit monitor/finalizer never schedules a restart.

- [ ] **Step 9: Add red stop and terminal-close tests**

~~~python
async def test_stop_terminates_then_kills_only_the_owned_child() -> None: ...
async def test_stop_waiter_cancellation_does_not_abandon_cleanup() -> None: ...
async def test_expected_generation_stop_is_noop_for_replacement() -> None: ...
async def test_stop_joins_inflight_probe_before_generation_cleanup() -> None: ...
async def test_application_deadline_caps_termination_grace() -> None: ...
async def test_close_and_wait_closed_are_idempotent() -> None: ...
async def test_terminal_close_leaves_no_child_or_task_reference() -> None: ...
async def test_exit_monitor_is_the_only_process_wait_owner() -> None: ...
async def test_stale_exit_monitor_cannot_reap_or_mutate_replacement() -> None: ...
async def test_output_drain_failure_stops_child_and_records_safe_failure() -> None: ...
async def test_inherited_pipe_descriptor_cannot_block_generation_cleanup() -> None: ...
~~~

- [ ] **Step 10: Implement retained stop and close**

`begin_draining()` publishes Draining for an owned child without terminating it
and is idempotent. A registry lifecycle transition calls it immediately after
new leases are blocked and before waiting for admitted leases. On entry,
`stop()` atomically checks `expected_process_generation` under its lifecycle
lock before any state or epoch change; a missing or different current child is
a no-op so a retiring adapter cannot stop its replacement. An accepted stop
calls `begin_draining()` itself when no registry transition did so, increments
the lifecycle epoch, invalidates/cancels any retained startup, and joins that
task before declaring cleanup complete. It publishes Stopping only after the caller has completed
lease draining, cancels and joins the periodic scheduler plus any shared
in-flight health probe before generation cleanup can close its client, signals
only the stored child handle, force-kills after the effective grace, and awaits
the sole exit-monitor/finalizer. The finalizer bounds natural pipe EOF,
cancels/joins unfinished drains, closes the parent pipe transports, invokes
generation cleanup exactly once, and publishes Stopped for manual shutdown.

An output-drain task failure while the child is alive schedules one retained
expected-stop cleanup, records the fixed `process_exited` failure, and cannot
leave an undrained live child. `close()` is retained and cancellation-shielded,
clears diagnostics/failure only at its terminal boundary, and uses
`current_shutdown_deadline()` to cap grace. A later successful start clears
Unavailable state and the prior failure; repeated stop/close remains
idempotent.

- [ ] **Step 11: Run supervisor tests**

Run:

~~~bash
.venv/bin/python -m pytest \
  Tests/TTS/test_audio_cpp_supervisor.py \
  Tests/TTS/test_adapter_types.py \
  -q
~~~

Expected: PASS without timing sleeps.

- [ ] **Step 12: Commit the supervisor**

~~~bash
git add \
  tldw_chatbook/TTS/audio_cpp_supervisor.py \
  tldw_chatbook/TTS/adapter_types.py \
  Tests/TTS/test_audio_cpp_supervisor.py \
  Tests/TTS/test_adapter_types.py
git commit -m "feat(tts): supervise one managed audio.cpp child"
~~~

---

### Task 4: Add staged configuration and an exclusive provider lifecycle transition

**Files:**

- Modify: `tldw_chatbook/TTS/adapter_registry.py`
- Test: `Tests/TTS/test_adapter_registry.py`
- Test: `Tests/TTS/test_tts_registry_service.py`

**Interfaces:**

- Consumes: existing exclusive provider slots, configuration generations, leases, and close semantics.
- Produces: immutable provider configuration snapshots, latest-wins staging, and `run_exclusive_provider_transition()`.

- [ ] **Step 1: Add red staging tests**

~~~python
async def test_stage_keeps_applied_config_revision_and_active_adapter() -> None: ...
async def test_newer_stage_supersedes_older_without_starting_handoff() -> None: ...
async def test_equal_config_advances_generation_without_restart_required() -> None: ...
async def test_reverting_to_applied_config_clears_an_older_stage() -> None: ...
async def test_immediate_reconfigure_clears_every_older_stage() -> None: ...
async def test_configuration_snapshot_is_deeply_immutable() -> None: ...
~~~

Assert that staging neither calls the provider factory nor closes the active adapter.

- [ ] **Step 2: Add red exclusive-transition tests**

~~~python
async def test_transition_rejects_new_leases_and_drains_admitted_lease() -> None: ...
async def test_transition_publishes_draining_before_waiting_for_admitted_lease() -> None: ...
async def test_transition_action_runs_after_last_lease_without_holding_slot_lock() -> None: ...
async def test_transition_promotes_only_latest_staged_config() -> None: ...
async def test_transition_without_stage_keeps_config_revision_and_adapter() -> None: ...
async def test_transition_failure_seals_provider_unavailable_and_releases_waiters() -> None: ...
async def test_action_failure_retry_clears_unavailable_after_success() -> None: ...
async def test_adapter_close_failure_retains_record_and_uses_fresh_retry_task() -> None: ...
async def test_successful_adapter_close_is_never_repeated_on_retry() -> None: ...
async def test_registry_close_joins_an_in_progress_transition() -> None: ...
~~~

- [ ] **Step 3: Run focused registry tests and verify failure**

Run:

~~~bash
.venv/bin/python -m pytest \
  Tests/TTS/test_adapter_registry.py \
  Tests/TTS/test_tts_registry_service.py \
  -q
~~~

Expected: new tests fail because no durable staged slot state or lifecycle transition exists.

- [ ] **Step 4: Add staged fields and immutable snapshots**

Add `staged_config` and `staged_generation` to `_ProviderSlot` rather than
overloading the existing in-flight handoff `pending_*` fields.
`stage_provider_configuration()` validates monotonically increasing
generations and deep-copies input. Older/equal generations return
`SUPERSEDED`. A newer mapping equal to the applied mapping clears any obsolete
stage, advances `applied_generation` and `highest_generation`, and returns
`UNCHANGED` without changing the configuration revision or adapter. Any other
newer mapping replaces only the latest staged pair and advances
`highest_generation`; staging the same values again still replaces the staged
generation so the newest saved identity wins.

Do not change `revision`, `applied_generation`, `active`, or `reconfiguring` for a true stage.

Any newer configuration accepted through the existing immediate
`begin_reconfigure_provider()` path atomically clears an older durable stage
before that configuration can become applied. This is part of the same
`transition_lock` decision, not a separate best-effort cleanup. Therefore the
sequence applied External A -> staged Managed B -> saved immediate External C
can leave only C applied and no reference to B.

- [ ] **Step 5: Implement the exclusive lifecycle transition**

Serialize through `transition_lock`, set `reconfiguring`, and reject new
leases. Release `slot.lock`, await `on_draining()` exactly once so process state
becomes Draining while admitted work is still visible, and then wait for the
active record's lease count to reach zero without holding `slot.lock`. Await
`action()` only after the last lease releases. Neither callback may call back
into the registry.

When `apply_staged=True` and a staged mapping exists: run the provider action,
close the old adapter using generation-local cleanup, and only after both
succeed promote the latest staged mapping, increment configuration revision
once, set applied generation to the staged generation, clear staged fields,
and leave the replacement lazy. Adapter retirement never terminal-closes an
app-scoped provider supervisor.

When no staged mapping exists: retain the adapter and both configuration
identities. Always clear `reconfiguring` and wake waiters. On action or cleanup
failure, make a best-effort generation-local close of the now-unusable active
adapter, seal the slot Unavailable with fixed safe errors, retain the exact
record plus latest staged pair, and permit a later deliberate retry. Keep
action failure and adapter-close failure as separate tested states.

Existing `_AdapterRecord.close_task` caches a failed task, so add an explicit
retry path under `record.close_lock`: only a completed failed close task may be
observed and replaced by one fresh task, and only for a retained failed
exclusive transition. A running or successful task is always reused. The
audio.cpp adapter used by this transition must likewise clear only its failed
attempt task while keeping admission sealed and making each remaining cleanup
step idempotent. No general registry-close path blindly retries arbitrary
legacy adapters.

On a later retry, rerun the provider action idempotently, retry only an actually
failed audio.cpp adapter cleanup, and promote/clear Unavailable only after both
succeed. When no stage exists and a prior successful best-effort close removed
the active adapter, success leaves the same applied mapping with no active
instance so its factory materializes lazily. Never repeat a successful adapter
close, promote after failed action/cleanup, silently restore an older mapping,
or strand a waiter.

- [ ] **Step 6: Run registry tests**

Run:

~~~bash
.venv/bin/python -m pytest \
  Tests/TTS/test_adapter_registry.py \
  Tests/TTS/test_tts_registry_service.py \
  -q
~~~

Expected: PASS.

- [ ] **Step 7: Commit staged lifecycle authority**

~~~bash
git add \
  tldw_chatbook/TTS/adapter_registry.py \
  Tests/TTS/test_adapter_registry.py \
  Tests/TTS/test_tts_registry_service.py
git commit -m "feat(tts): stage managed provider transitions"
~~~

---

### Task 5: Bind the native adapter and service to the one supervisor

**Files:**

- Modify: `tldw_chatbook/TTS/adapters/audio_cpp.py`
- Modify: `tldw_chatbook/TTS/adapter_bootstrap.py`
- Modify: `tldw_chatbook/TTS/TTS_Generation.py`
- Modify: `tldw_chatbook/TTS/request_admission.py`
- Modify: `tldw_chatbook/TTS/effective_settings.py`
- Modify: `tldw_chatbook/TTS/__init__.py`
- Test: `Tests/TTS/test_audio_cpp_adapter.py`
- Test: `Tests/TTS/test_audio_cpp_managed_integration.py`
- Test: `Tests/TTS/test_tts_request_admission.py`
- Test: `Tests/TTS/test_stts_settings_reconfiguration.py`
- Test: `Tests/TTS/test_effective_settings.py`
- Test: `Tests/TTS/test_tts_profile_capabilities.py`
- Test: `Tests/TTS/test_tts_settings_capability_observations.py`

**Interfaces:**

- Consumes: Tasks 1–4 interfaces plus existing catalog, voice, synthesis, response lease, settings publication, and shutdown paths.
- Produces: generation-bound managed HTTP behavior and the public service lifecycle methods specified above.

- [ ] **Step 1: Add red adapter-generation tests**

~~~python
async def test_external_adapter_never_calls_supervisor() -> None: ...
async def test_managed_refresh_and_synthesis_ensure_supervisor_running() -> None: ...
async def test_passive_catalog_voice_and_capability_reads_do_not_launch() -> None: ...
async def test_managed_contract_probe_reuses_existing_health_and_models_parsers() -> None: ...
async def test_process_generation_change_closes_client_and_clears_catalog_voice_caches() -> None: ...
async def test_adapter_owns_and_closes_generation_health_client() -> None: ...
async def test_running_generation_does_not_construct_unused_hook_bundle() -> None: ...
async def test_managed_catalog_preserves_multiple_configured_tts_models() -> None: ...
async def test_managed_synthesis_remains_one_validated_complete_wav_item() -> None: ...
async def test_managed_adapter_close_is_generation_local_and_keeps_supervisor_open() -> None: ...
async def test_managed_adapter_close_stops_bound_generation_before_client_close() -> None: ...
async def test_managed_adapter_close_failure_retries_only_remaining_cleanup() -> None: ...
async def test_managed_a_to_b_retirement_reuses_supervisor_for_new_generation() -> None: ...
~~~

- [ ] **Step 2: Refactor HTTP client binding without changing External behavior**

Keep the existing External client construction and tests. For Managed, supply
an adapter-owned asynchronous generation-hooks factory to `ensure_running()`.
After direct spawn succeeds, the supervisor assigns the process generation,
immediately starts its sole exit monitor and output drains, and awaits the
factory exactly once under the startup deadline. The adapter factory uses one
transactional cleanup owner (for example, `AsyncExitStack`): it registers the
generation request client immediately after creating it, then the dedicated
health client, and transfers both to the returned cleanup closure only after
the complete bundle exists. Failure or cancellation at any intermediate point
closes every already-created resource before raising the fixed
`process_spawn_failed` startup-resource result; the supervisor then performs
the same complete exact-child rollback. The prepared contract probe can
therefore validate
`/v1/models` during startup without recursively calling `ensure_running()`.
Store the bundle by process generation and never reuse it after the supervisor
reports a different generation. If `ensure_running()` reuses an existing
Running child, it must reuse that generation's existing bundle and must not
construct or leak an unused caller bundle.

The adapter also owns one dedicated direct-loopback health client and the
health-probe closure supplied to the supervisor. Configure it with
`trust_env=False`, `follow_redirects=False`, and a per-probe timeout no greater
than the lesser of the configured connect timeout and health interval. The
supervisor owns only scheduling/shared-task state; it never owns or parses an
HTTP client. Generation cleanup closes both clients exactly once.

Add a partial-construction regression in which the first client is created and
the second allocation fails: the first closes, no hook bundle is retained, the
exact child is reaped, and no raw exception text reaches the process snapshot.

`AudioCppAdapter.close()` seals only that adapter instance and uses the process
generation recorded when its hook bundle was bound. After registry lease drain
it first awaits
`supervisor.stop(expected_process_generation=bound_generation)`. That
nonterminal stop is idempotent, joins the periodic scheduler and shared
in-flight probe before generation cleanup closes either HTTP client, and is a
no-op if a replacement generation is already current. The adapter then
invalidates catalog/voice state and joins retry-safe remaining
generation-resource cleanup. An adapter that never bound a process generation
does not stop whatever generation may now be current. Adapter close never calls
`AudioCppSupervisor.close()`. Ordinary A -> B retirement may already have
performed the same stop as its registry action; the generation-bound repeat is
safe and leaves the one app-scoped supervisor reusable for B. Only terminal
service shutdown closes the supervisor. The adapter's retained close-attempt
task remains cached on success; on failure it is cleared under the adapter close
lock only after the failed task is observed, so an explicit registry retry can
rerun idempotent remaining cleanup without reopening admission or repeating
already-complete resource work.

Only deliberate operations may call `ensure_running()`: synthesis and explicit
catalog/voice refresh (`refresh=True`), including Start & Test. Non-refresh
catalog/voice/native-capability observation uses generation-matching cached
evidence or returns Unverified/unavailable evidence while stopped. It never
applies a stage or launches merely because Settings, Profiles, or a passive
status reader requested capability information.

Split catalog refresh into a prepared internal method so the supervisor contract probe can validate `/health` plus `/v1/models` without recursively calling `ensure_running()`.

- [ ] **Step 3: Add red bootstrap singleton tests**

~~~python
def test_default_service_constructs_one_supervisor_without_launch() -> None: ...
async def test_reconfigured_audio_cpp_adapters_share_the_app_supervisor() -> None: ...
async def test_legacy_provider_materialization_never_touches_supervisor() -> None: ...
~~~

- [ ] **Step 4: Inject one app-scoped supervisor**

`build_default_tts_service()` collects credential environment names from its
configuration without resolving values, then constructs one
`AudioCppSupervisor` with that frozen inventory. Pass the supervisor into the
closure used by `audio_cpp_provider_spec()` and into `TTSService`. Descriptor
reads and service construction must remain process- and network-free.

- [ ] **Step 5: Add red managed settings staging tests**

~~~python
async def test_managed_save_while_running_finishes_as_pending_without_stopping_child() -> None: ...
async def test_latest_managed_save_wins_before_explicit_apply() -> None: ...
async def test_external_to_managed_save_is_staged_until_deliberate_operation() -> None: ...
async def test_external_to_external_save_keeps_existing_immediate_handoff() -> None: ...
async def test_external_a_staged_managed_b_then_external_c_cannot_retain_b() -> None: ...
async def test_reverting_a_stage_to_applied_values_finishes_unchanged() -> None: ...
async def test_staged_exact_selection_is_unverified_against_active_catalog() -> None: ...
async def test_dynamic_selection_can_continue_against_clearly_applied_generation() -> None: ...
~~~

- [ ] **Step 6: Stage only managed-boundary publications**

Inside the existing serialized publication lock, reconcile every newer save
against applied and staged state as one latest-wins decision:

- an exact return to the applied mapping clears any stage and advances the saved/applied identity without reconfiguration;
- a changed mapping is staged when either the applied or desired mode is Managed;
- External-to-External retains the existing immediate handoff, and that accepted handoff atomically clears every older stage before C can become applied;
- publish the new preferences and persisted generation only after the registry accepted that same generation; and
- return provider status `"pending"` as the terminal staged result without creating a never-finishing handoff task.

Pin the A/B/C regression: applied External A -> stage Managed B -> save different
External C ends with C applied, no staged mapping, and no later lifecycle call
can materialize B.

When saved and applied generations differ, `get_native_capability_snapshot()` returns an Unverified snapshot for exact validation so active catalog evidence cannot reject or validate a staged exact model/voice. Dynamic modes may still use the clearly labeled applied generation. A save that exactly reverts to the applied mapping clears the stage and truthfully aligns the applied generation without restarting.

- [ ] **Step 7: Add red deliberate-operation preparation tests**

~~~python
async def test_catalog_refresh_applies_latest_stage_before_adapter_lease() -> None: ...
async def test_console_and_roleplay_admission_apply_stage_before_read_gate() -> None: ...
async def test_direct_service_launch_paths_prepare_before_adapter_lease() -> None: ...
async def test_direct_admit_prepares_stage_before_returning_operation() -> None: ...
async def test_passive_service_capability_paths_neither_apply_nor_launch() -> None: ...
async def test_profile_capability_validation_stays_unverified_without_launch() -> None: ...
async def test_settings_capability_observation_stays_passive_while_stopped() -> None: ...
async def test_concurrent_preparation_applies_one_latest_generation() -> None: ...
async def test_crash_with_external_staged_applies_external_without_child() -> None: ...
async def test_crash_with_managed_staged_starts_one_latest_replacement() -> None: ...
async def test_exit_between_live_check_and_ensure_retries_before_old_launch() -> None: ...
async def test_pre_spawn_starting_and_post_exit_stopping_do_not_apply_stage() -> None: ...
async def test_stage_eligibility_is_rechecked_if_starting_appears_before_writer() -> None: ...
def test_provider_only_projection_matches_full_resolution_precedence() -> None: ...
~~~

- [ ] **Step 8: Prepare before the request-admission read gate**

Extract one synchronous provider-only projection in
`TTSEffectiveSettingsResolver`; full non-Studio and Studio resolution must call
that same helper. It evaluates explicit, character/default profile, Studio, and
global/fallback precedence without catalog I/O. Do not implement a second
provider-precedence chain in request admission.

Add one private preparation loop below every service path that could otherwise
reach managed HTTP: direct `admit()`, direct/effective/exact synthesis, refreshed catalog,
refreshed `get_voices()`/`observe_voices()`, Start & Test, and the corresponding
Console/Roleplay admission paths. Passive catalog, voice, and native-capability
calls enter the same seam with a no-launch intent and never apply staging.

For a deliberate operation the loop:

1. obtains the selected provider through the shared provider-only projection;
2. under the publication lock, reads the registry stage and one supervisor admission snapshot;
3. takes the writer side only when audio.cpp has a stage and that snapshot says stage application is eligible, then re-reads admission under the writer before mutating anything; if startup/stop state appeared meanwhile, it releases and retries without applying the stage;
4. explicit restart/shutdown always take the writer and apply the newest stage after draining;
5. runs the registry exclusive transition, publishes supervisor Draining before lease wait, stops after the last lease, and only then promotes the latest stage;
6. when a live/Starting generation made the stage ineligible, enters the read side and passes that admission snapshot to `ensure_running(require_existing=...)`;
7. if the lifecycle epoch/generation changed before `ensure_running()`, releases the lease/read side on the private retry signal and starts again rather than launching the old applied mapping;
8. rechecks provider choice after entering the read side; and
9. uses internal already-prepared catalog/capability readers while the read side is held.

This prevents read-to-write lock upgrades and keeps supervisor locks out of registry lease waits.

- [ ] **Step 9: Add red lifecycle API tests**

~~~python
async def test_start_and_test_managed_starts_and_refreshes_catalog() -> None: ...
async def test_restart_drains_work_stops_old_generation_and_starts_one_new_generation() -> None: ...
async def test_restart_applies_latest_stage_not_earlier_stage() -> None: ...
async def test_shutdown_drains_work_stops_child_and_promotes_stage_without_launch() -> None: ...
async def test_applying_external_stops_child_and_never_relaunches_managed() -> None: ...
async def test_cancelled_lifecycle_waiter_does_not_abandon_accepted_transition() -> None: ...
async def test_service_close_keeps_child_alive_until_admitted_lease_releases() -> None: ...
async def test_service_close_joins_inflight_probe_before_client_and_terminal_close() -> None: ...
async def test_service_close_uses_one_deadline_for_registry_then_supervisor() -> None: ...
async def test_lifecycle_retry_clears_prior_transition_unavailable_state() -> None: ...
~~~

- [ ] **Step 10: Implement service lifecycle APIs**

`start_and_test_audio_cpp()` performs prepared `get_catalog(refresh=True)`. It
applies a staged mapping first only when the supervisor admission snapshot is
stage-application eligible. With a Running or Unhealthy owned child it probes
the clearly applied generation and leaves Restart-required staging untouched.
Starting, Draining, or Stopping returns the existing retryable transition
result; it is never treated as child-free.

`restart_audio_cpp()` takes the publication lock and admission writer side, runs the registry exclusive transition with `on_draining=supervisor.begin_draining`, `action=supervisor.stop`, and `apply_staged=True`, then:

- returns `None` if the applied mode is External;
- otherwise performs one prepared catalog refresh and returns it.

`shutdown_audio_cpp()` performs the same exclusive drain/stop/apply but never materializes or launches a replacement.

Retain accepted lifecycle tasks so cancelling one UI/service waiter cannot abandon them.

- [ ] **Step 11: Integrate terminal service shutdown**

Create one retained service-shutdown task that establishes the existing
monotonic TTS shutdown deadline before creating any nested cleanup task. Amend
registry close to reuse an active `current_shutdown_deadline()` and create its
default only when no outer deadline exists.

Await retained registry close first. Registry admission is sealed and admitted
response/provider leases drain. Managed adapter retirement then performs its
generation-bound nonterminal stop, which cancels and joins the health scheduler
and any shared in-flight probe before generation cleanup closes the HTTP
clients; only afterward does adapter cleanup finish. In a `finally` block after
registry close returns, times out, or fails, always call the one app-owned
`AudioCppSupervisor.close()` under the same remaining deadline and then await
`wait_closed()`. This `TTSService` call is the sole terminal supervisor owner;
it is never conditional fallback behavior and adapter close never duplicates
it. Do not start registry and supervisor close concurrently.
Sanitize/aggregate failures through the current service shutdown boundary.
`TTSService.wait_closed()` must not finish while the supervisor owns a child or
task even if another adapter cleanup fails. A regression holds an admitted
lease and proves the child stays alive until that lease releases or the single
outer deadline expires. A successful-shutdown regression holds one health probe
in flight and proves this order: lease drain, generation-bound nonterminal stop,
probe join, generation client cleanup, terminal supervisor close, then
`wait_closed()` completion.

- [ ] **Step 12: Run focused managed integration**

Run:

~~~bash
.venv/bin/python -m pytest \
  Tests/TTS/test_audio_cpp_adapter.py \
  Tests/TTS/test_audio_cpp_managed_integration.py \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_stts_settings_reconfiguration.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_effective_settings.py \
  Tests/TTS/test_tts_profile_capabilities.py \
  Tests/TTS/test_tts_settings_capability_observations.py \
  -q
~~~

Expected: PASS.

- [ ] **Step 13: Commit adapter/service integration**

~~~bash
git add \
  tldw_chatbook/TTS/adapters/audio_cpp.py \
  tldw_chatbook/TTS/adapter_bootstrap.py \
  tldw_chatbook/TTS/TTS_Generation.py \
  tldw_chatbook/TTS/request_admission.py \
  tldw_chatbook/TTS/effective_settings.py \
  tldw_chatbook/TTS/__init__.py \
  Tests/TTS/test_audio_cpp_adapter.py \
  Tests/TTS/test_audio_cpp_managed_integration.py \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_stts_settings_reconfiguration.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_effective_settings.py \
  Tests/TTS/test_tts_profile_capabilities.py \
  Tests/TTS/test_tts_settings_capability_observations.py
git commit -m "feat(tts): integrate managed audio.cpp lifecycle"
~~~

---

### Task 6: Prove the real subprocess boundary and preserve the dormant UI seam

**Files:**

- Create: `Tests/TTS/fixtures/fake_audiocpp_server.py`
- Modify: `Tests/TTS/test_audio_cpp_supervisor.py`
- Modify: `Tests/TTS/test_audio_cpp_managed_integration.py`
- Modify: `Tests/UI/test_settings_audio_cpp_experience_model.py`
- Modify: `Tests/UI/test_settings_speech_tts_panel.py`

**Interfaces:**

- Consumes: completed supervisor, adapter, registry, and service APIs.
- Produces: a controlled-child characterization on platforms that can execute a temporary shebang wrapper, plus injected cross-platform boundary coverage and proof that Slice 4 remains invisible.

- [ ] **Step 1: Add a controlled fake server executable**

The fixture accepts only:

~~~text
fake_audiocpp_server --config /absolute/path/server.json
~~~

It reads the supplied loopback port, exposes deterministic `/health`,
`/v1/models`, optional voices, and one small PCM16 WAV response, and supports
test-only behavior selected inside the temporary JSON: early exit, delayed
readiness, stdout/stderr chunks, ignore
terminate, normal shutdown, and a POSIX descendant that inherits stdout/stderr
after the exact server child exits. It writes no environment or request text to
disk. The descendant case may write only its numeric PID to a test-owned
`tmp_path` control file so the test finalizer can remove that fixture process.

On POSIX, tests create an executable shebang wrapper in `tmp_path` that invokes the current absolute Python interpreter and fixture path. On a platform that cannot directly execute that wrapper without a shell, mark only the real-child characterization unsupported with an explicit reason; the injected launcher suite remains mandatory. Production always receives exactly one executable path followed by `--config` and the JSON path and never uses a shell.

- [ ] **Step 2: Add real subprocess tests**

~~~python
@pytest.mark.asyncio
async def test_real_child_argv_cwd_environment_readiness_and_cleanup(tmp_path: Path) -> None: ...
@pytest.mark.asyncio
async def test_real_child_early_exit_drains_output_and_leaves_no_process(tmp_path: Path) -> None: ...
@pytest.mark.asyncio
async def test_real_child_force_kill_and_monitor_cleanup(tmp_path: Path) -> None: ...
@pytest.mark.asyncio
async def test_real_child_inherited_pipes_finish_cleanup_without_killing_descendant(
    tmp_path: Path,
) -> None: ...
@pytest.mark.asyncio
async def test_repeated_real_spawn_stop_has_one_reaper_and_no_retained_generation(
    tmp_path: Path,
) -> None: ...
~~~

Assert exact PID disappearance with bounded polling and register a finalizer
that kills only captured fixture PIDs if a test fails. In the inherited-pipe
case, assert supervisor cleanup completes before the descriptor-holder exits,
assert that descendant remains alive (proving production did not signal it),
then let the test finalizer terminate and reap that explicit test process. Run
enough real spawn/stop cycles to expose competing-wait/reaping failures without
using arbitrary sleeps.

- [ ] **Step 3: Run the real subprocess tests**

Run:

~~~bash
.venv/bin/python -m pytest \
  Tests/TTS/test_audio_cpp_supervisor.py \
  Tests/TTS/test_audio_cpp_managed_integration.py \
  -q
~~~

Expected: PASS on the current platform. A platform skip must name the unsupported primitive and retain injected-boundary coverage.

Record a native-release gate in the TTS module guide: Slice 5 must not expose
Managed mode on Windows until CI proves direct execution of a user-supplied
native binary, terminate/force-kill, sole reaping, and bounded parent-pipe
closure there. External audio.cpp remains available; injected supervisor tests
remain mandatory on every platform.

- [ ] **Step 4: Add UI boundary regressions**

Assert on Slice 4 that:

- `AudioCppConfig().to_mapping()` still drives the External-only mounted form;
- the Settings provider model exposes no managed path or timing field;
- no Managed selector, Use detected action, Start/Restart/Shutdown control, process row, or diagnostics panel is mounted;
- mounting Settings and Speech Lab launches no process.

- [ ] **Step 5: Run dormant-seam UI tests**

Run:

~~~bash
.venv/bin/python -m pytest \
  Tests/UI/test_settings_audio_cpp_experience_model.py \
  Tests/UI/test_settings_speech_tts_panel.py \
  Tests/UI/test_speech_playground_pane_lifecycle.py \
  -q
~~~

Expected: PASS.

- [ ] **Step 6: Commit subprocess and UI-boundary proof**

~~~bash
git add \
  Tests/TTS/fixtures/fake_audiocpp_server.py \
  Tests/TTS/test_audio_cpp_supervisor.py \
  Tests/TTS/test_audio_cpp_managed_integration.py \
  Tests/UI/test_settings_audio_cpp_experience_model.py \
  Tests/UI/test_settings_speech_tts_panel.py
git commit -m "test(tts): prove managed audio.cpp process ownership"
~~~

---

### Task 7: Documentation, full verification, and TASK-3792 closeout

**Files:**

- Modify: `Docs/Development/TTS/TTS_MODULE_GUIDE.md`
- Modify: `backlog/tasks/task-3792 - Add-dormant-managed-audio.cpp-runtime-core.md`

**Interfaces:**

- Consumes: the completed Slice 4 implementation and all verification evidence.
- Produces: current developer guidance and a Definition-of-Done-complete Backlog record.

- [ ] **Step 1: Update the TTS module guide**

Document:

- External remains the default and current visible mode;
- one app-scoped provider-specific supervisor exists;
- which deliberate service operations may launch;
- registry-before-supervisor lock ordering and generation-local adapter retirement;
- single-deadline registry-drain-before-supervisor terminal shutdown;
- saved/applied/process generation distinctions;
- lifecycle-epoch launch fencing, sole process reaping, exact-child ownership, and no adoption/restart loop;
- child environment and diagnostic privacy boundaries;
- stable lifecycle failure/recovery mapping and bounded inherited-pipe cleanup;
- complete-WAV behavior remains unchanged; and
- Slice 5 still owns Managed Settings/Lab UI and live audible UAT, with the Windows native-release gate documented.

- [ ] **Step 2: Run focused production quality checks**

~~~bash
.venv/bin/python -m ruff check \
  tldw_chatbook/TTS/audio_cpp_config.py \
  tldw_chatbook/TTS/audio_cpp_managed_config.py \
  tldw_chatbook/TTS/audio_cpp_supervisor.py \
  tldw_chatbook/TTS/adapter_types.py \
  tldw_chatbook/TTS/adapters/audio_cpp.py \
  tldw_chatbook/TTS/adapter_bootstrap.py \
  tldw_chatbook/TTS/adapter_registry.py \
  tldw_chatbook/TTS/TTS_Generation.py \
  tldw_chatbook/TTS/request_admission.py \
  tldw_chatbook/TTS/effective_settings.py \
  tldw_chatbook/TTS/__init__.py \
  Tests/TTS/test_audio_cpp_config.py \
  Tests/TTS/test_audio_cpp_managed_config.py \
  Tests/TTS/test_audio_cpp_supervisor.py \
  Tests/TTS/test_audio_cpp_adapter.py \
  Tests/TTS/test_audio_cpp_managed_integration.py \
  Tests/TTS/test_adapter_types.py \
  Tests/TTS/test_adapter_registry.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_stts_settings_reconfiguration.py \
  Tests/TTS/test_effective_settings.py \
  Tests/TTS/test_tts_profile_capabilities.py \
  Tests/TTS/test_tts_settings_capability_observations.py \
  Tests/TTS/test_tts_logging_privacy.py \
  Tests/TTS/fixtures/fake_audiocpp_server.py \
  Tests/UI/test_settings_audio_cpp_experience_model.py \
  Tests/UI/test_settings_speech_tts_panel.py
.venv/bin/python -m ruff format --check \
  tldw_chatbook/TTS/audio_cpp_config.py \
  tldw_chatbook/TTS/audio_cpp_managed_config.py \
  tldw_chatbook/TTS/audio_cpp_supervisor.py \
  tldw_chatbook/TTS/adapter_types.py \
  tldw_chatbook/TTS/adapters/audio_cpp.py \
  tldw_chatbook/TTS/adapter_bootstrap.py \
  tldw_chatbook/TTS/adapter_registry.py \
  tldw_chatbook/TTS/TTS_Generation.py \
  tldw_chatbook/TTS/request_admission.py \
  tldw_chatbook/TTS/effective_settings.py \
  tldw_chatbook/TTS/__init__.py \
  Tests/TTS/test_audio_cpp_config.py \
  Tests/TTS/test_audio_cpp_managed_config.py \
  Tests/TTS/test_audio_cpp_supervisor.py \
  Tests/TTS/test_audio_cpp_adapter.py \
  Tests/TTS/test_audio_cpp_managed_integration.py \
  Tests/TTS/test_adapter_types.py \
  Tests/TTS/test_adapter_registry.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_stts_settings_reconfiguration.py \
  Tests/TTS/test_effective_settings.py \
  Tests/TTS/test_tts_profile_capabilities.py \
  Tests/TTS/test_tts_settings_capability_observations.py \
  Tests/TTS/test_tts_logging_privacy.py \
  Tests/TTS/fixtures/fake_audiocpp_server.py \
  Tests/UI/test_settings_audio_cpp_experience_model.py \
  Tests/UI/test_settings_speech_tts_panel.py
.venv/bin/python -m compileall -q \
  tldw_chatbook/TTS \
  Tests/TTS/fixtures/fake_audiocpp_server.py
.venv/bin/python -m mypy \
  tldw_chatbook/TTS/audio_cpp_config.py \
  tldw_chatbook/TTS/audio_cpp_managed_config.py \
  tldw_chatbook/TTS/audio_cpp_supervisor.py \
  tldw_chatbook/TTS/adapter_types.py \
  tldw_chatbook/TTS/adapters/audio_cpp.py \
  tldw_chatbook/TTS/adapter_bootstrap.py \
  tldw_chatbook/TTS/adapter_registry.py \
  tldw_chatbook/TTS/request_admission.py \
  tldw_chatbook/TTS/effective_settings.py \
  tldw_chatbook/TTS/TTS_Generation.py \
  tldw_chatbook/TTS/__init__.py
~~~

Expected: all exit zero.

- [ ] **Step 3: Run focused lifecycle and privacy coverage**

~~~bash
.venv/bin/python -m pytest \
  Tests/TTS/test_audio_cpp_config.py \
  Tests/TTS/test_audio_cpp_managed_config.py \
  Tests/TTS/test_audio_cpp_supervisor.py \
  Tests/TTS/test_audio_cpp_adapter.py \
  Tests/TTS/test_audio_cpp_managed_integration.py \
  Tests/TTS/test_adapter_types.py \
  Tests/TTS/test_adapter_registry.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_stts_settings_reconfiguration.py \
  Tests/TTS/test_effective_settings.py \
  Tests/TTS/test_tts_profile_capabilities.py \
  Tests/TTS/test_tts_settings_capability_observations.py \
  Tests/TTS/test_tts_logging_privacy.py \
  Tests/UI/test_settings_audio_cpp_experience_model.py \
  Tests/UI/test_settings_speech_tts_panel.py \
  Tests/Subscriptions/test_briefing_audio_synthesis.py \
  Tests/TTS_Events/test_spoken_feedback_streaming.py \
  Tests/TTS_Events/test_utterance_speech_entry.py \
  Tests/UI/test_speech_tts_settings_ownership_closeout.py \
  Tests/UI/test_uat_first_time_character_chat.py \
  -q
~~~

Expected: PASS.

- [ ] **Step 4: Run the full TTS suite**

Run outside the filesystem sandbox if the one real-socket test receives a sandbox `PermissionError`:

~~~bash
.venv/bin/python -m pytest Tests/TTS -q
~~~

Expected: zero new failures. Compare exact failed node IDs against the same `origin/dev` command if the repository baseline is not green.

- [ ] **Step 5: Run boundary and pre-closeout worktree checks**

~~~bash
rg -n "Managed local server|Use detected|Restart & Apply|Shut down server|Recent diagnostics" \
  tldw_chatbook/UI tldw_chatbook/Widgets
rg -n "create_subprocess|Popen|subprocess\\.|--config" \
  tldw_chatbook/TTS
rg -n "OPENAI_API_KEY|ELEVENLABS_API_KEY|TOKEN|PASSWORD|SECRET" \
  Tests/TTS/test_audio_cpp_managed_config.py \
  Tests/TTS/test_audio_cpp_supervisor.py \
  Tests/TTS/test_audio_cpp_managed_integration.py
.venv/bin/python -m pytest \
  Tests/TTS/test_adapter_types.py::test_tts_operation_code_contains_only_stable_values \
  -q
git diff --check
git diff --cached --check
~~~

Expected:

- the UI-copy search finds no newly added Slice 4 UI;
- subprocess creation exists only in the provider-specific supervisor;
- tests contain synthetic credential names/values only;
- the exact bounded operation-code tuple includes the managed additions and no accidental values; and
- tracked/staged worktree whitespace checks exit zero. A clean status is not expected yet because Step 1's module-guide update is intentionally committed with task closeout in Step 8.

- [ ] **Step 6: Self-review against the spec**

Check every `ML-AC-001` through `ML-AC-012`, `ML-AC-014`, `ML-AC-016`, and `ML-AC-017` requirement assigned to Slice 4. Confirm no code claims Slice 5 UI or audible UAT completion.

Run a placeholder-language audit over the implementation plan and task; remove every hit that represents unfinished thinking.

- [ ] **Step 7: Complete the Backlog record**

After all evidence exists:

1. check every TASK-3792 acceptance criterion and Definition of Done item;
2. add concise Implementation Notes with exact tests and any baseline comparison;
3. state ADR-023 and ADR-039 conformance and no-new-ADR reasoning;
4. add a lessons entry only if implementation produced a new evidenced reusable trap; and
5. set TASK-3792 to Done with the Backlog CLI.

- [ ] **Step 8: Commit documentation and task closeout**

~~~bash
git add \
  Docs/Development/TTS/TTS_MODULE_GUIDE.md \
  "backlog/tasks/task-3792 - Add-dormant-managed-audio.cpp-runtime-core.md"
git commit -m "docs(tts): close managed audio.cpp runtime core"
~~~

- [ ] **Step 9: Run final cumulative branch and task-identity checks**

After Step 8, fetch current refs and run:

~~~bash
git diff origin/dev...HEAD --check
git status --short
backlog task 3792 --plain
~~~

Expected: the cumulative branch diff exits zero, status is clean, and TASK-3792
renders with the intended task, plan, ADRs, completed acceptance criteria, and
implementation evidence.

Immediately before PR/merge, repeat the repository's
`lessons-backlog-hygiene.md` sweep across every fetched local/remote ref and
checked-out worktree, plus an open-PR search for `TASK-3792`. Multiple refs that
contain this same branch/file are expected; any different task title or path
using 3792 is a collision and must be resolved before proceeding.

## Plan Self-Review Record

- Spec coverage: Tasks 1–6 cover every Slice 4 requirement and intentionally leave ML-AC-013 and ML-AC-015 to Slice 5.
- Scope: no Global Settings or Speech Lab managed UI is implemented; UI files appear only in negative boundary tests.
- Type consistency: configuration, supervisor, registry, and service signatures in later tasks match the Produced Interfaces section.
- Concurrency: lifecycle epochs prevent stale applied mappings from launching; one exit monitor reaps each child; one probe task serializes health; registry lease draining precedes child termination.
- Failure truthfulness: latest-wins save reconciliation cannot retain obsolete staging, stable process failures are memory-only, and successful retries clear sealed Unavailable state.
- Evidence: exact operation-code, non-TTS caller, repeated real-spawn, inherited-pipe, passive-no-launch, and cumulative-diff checks are explicit rather than implied by the broad suite.
- Dependency order: validation precedes supervision; supervision precedes registry/service integration; the real subprocess and UI boundary tests consume the completed core.
- ADR check: ADR-023 and ADR-039 are accepted, amended, landed, and linked; no new decision is introduced.
- Rollback: External remains the default, managed fields are inert without Managed mode, and removing the later UI leaves the dormant core unreachable through normal product navigation.
