# vLLM Lab-to-Console Complete Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Lab's vLLM pane into a guided launch-or-connect workflow that proves API/model readiness, hands the exact target to Console, restores non-secret profiles, and remains operable at 80x24.

**Architecture:** Pure vLLM contracts validate drafts, build the public `vllm serve` command, and fence results by generation and semantic fingerprint. `LLMScreen` owns asynchronous preflight/readiness workers and an app-scoped connection owner; a focused `VllmSetupView` renders that state, typed memory-only handoffs let Console and Settings retain mutation authority, and a separate atomic JSON repository owns device-local profiles.

**Tech Stack:** Python 3.11+, Textual 8.x, `httpx`, stdlib `dataclasses`/`enum`/`json`/`pathlib`/`socket`/`subprocess`, pytest, pytest-asyncio, Textual Pilot, generated TCSS.

**Spec:** `Docs/superpowers/specs/2026-09-03-vllm-lab-console-complete-redesign.md`

## Global Constraints

- Use the public command shape `vllm serve <model>`; do not depend on `vllm.entrypoints.*`.
- A Chatbook-owned launch includes exactly one `--served-model-name chatbook-vllm`.
- Initial bind and port values are exactly `127.0.0.1` and `8000`.
- Ready requires current-generation `/health` and `/v1/models` evidence; process liveness alone is insufficient.
- Never persist or copy credentials, raw arguments, environment-variable values, process IDs, model paths, raw HTTP bodies, or unrestricted child output.
- Raw arguments are launch-only and may not override model, host, port, served model name, API key, or Hugging Face token.
- `Use in Console` changes only the active Console session; `Make default for new chats` delegates to Settings' full provider mutation transaction.
- Profiles are device-local, versioned JSON, capped at 32, and preserve an unsupported future version byte-for-byte.
- At 80x24, 100x30, and 120x40 every visible focusable descendant must remain inside its owning pane.
- Use `tldw_chatbook/css/features/_lab.tcss` as source and regenerate `tldw_chatbook/css/tldw_cli_modular.tcss`; never edit the generated bundle directly.
- Run focused tests for each backlog task. Ask before a repository-wide sweep.

## File and ownership map

| File | Responsibility |
|---|---|
| `tldw_chatbook/UI/LLM_Management/vllm_setup.py` | Immutable draft/preflight/snapshot/target/state types, validation, fingerprints, endpoint normalization, CLI construction. |
| `tldw_chatbook/UI/LLM_Management/vllm_connection.py` | App-scoped generation owner, bounded activity, credential-aware health/model probes. |
| `tldw_chatbook/UI/LLM_Management/vllm_profiles.py` | Profile schema, normalization, future-version-safe reads, atomic writes, CRUD. |
| `tldw_chatbook/UI/LLM_Management/vllm_setup_view.py` | vLLM-only Textual composition, events, and state projection. No subprocesses, HTTP, config writes, or profile I/O. |
| `tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_vllm.py` | File-picker adapters and temporary compatibility routing; delegates validation and lifecycle ownership. |
| `tldw_chatbook/UI/Screens/llm_screen.py` | Preflight/readiness/profile workers, generation invalidation, restart coordination, inspector projection, handoff staging. |
| `tldw_chatbook/UI/Navigation/vllm_handoff.py` | Secret-free Console and Settings intents. |
| `tldw_chatbook/UI/Navigation/pending_handoff_store.py` | Type-check and detach vLLM intents in memory. |
| `tldw_chatbook/UI/Screens/chat_screen.py` | Claim session intent and atomically replace active session settings without config writes. |
| `tldw_chatbook/UI/Screens/settings_screen.py` | Claim default intent, prefill provider/model/endpoint, and leave Save as the only durable commit. |
| `tldw_chatbook/UI/LLM_Management_Window.py` | Lazy-mount `VllmSetupView`; remove the legacy inline vLLM form and conflicting provider-view keybindings. |
| `tldw_chatbook/css/features/_lab.tcss` | Wide/medium/compact vLLM layout and focus-visible styling. |
| `Tests/LLM_Management/test_vllm_setup.py` | Pure validation, endpoint, fingerprint, and command contract. |
| `Tests/LLM_Management/test_vllm_connection.py` | Probe state machine, stale settlement, privacy, restart sequencing. |
| `Tests/LLM_Management/test_vllm_profiles.py` | Exact schema/CRUD/atomic/future-version behavior. |
| `Tests/UI/test_vllm_lab_workflow.py` | Mounted first-run, launch/connect, readiness, handoff, profiles, and focus behavior. |
| `Tests/UI/test_vllm_lab_geometry.py` | Production-stylesheet geometry and full Tab-order matrix. |

---

### Task 1: TASK-31388 — Guided environment, model, network, and command preflight

**Files:**
- Create: `tldw_chatbook/UI/LLM_Management/__init__.py`
- Create: `tldw_chatbook/UI/LLM_Management/vllm_setup.py`
- Create: `tldw_chatbook/UI/LLM_Management/vllm_setup_view.py`
- Modify: `tldw_chatbook/UI/LLM_Management_Window.py`
- Modify: `tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_vllm.py`
- Test: `Tests/LLM_Management/test_vllm_setup.py`
- Test: `Tests/UI/test_vllm_lab_workflow.py`
- Modify: `backlog/tasks/task-31388 - Add-guided-vLLM-environment-and-model-preflight.md`

**Interfaces:**
- Consumes: `server_lifecycle.reserve_server_launch()` and the existing deferred pane contract.
- Produces: `VllmLaunchDraft`, `VllmPreflightResult`, `VllmLaunchSnapshot`, `VllmConnectionTarget`, `VllmReadinessState`, `semantic_fingerprint()`, `run_vllm_preflight()`, `build_vllm_command()`, and `VllmSetupView` events used by Tasks 2–5.

- [ ] **Step 1: Put TASK-31388 in progress and attach this plan section**

Run:

```bash
backlog task edit 31264 -a @codex -s "In Progress"
```

Add an Implementation Plan naming this file, ADR-117, the red/green test order below, and:

```text
ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: This task directly implements the accepted runtime and UX boundaries.
```

- [ ] **Step 2: Write the failing pure contract tests**

Create tests that construct both modes and pin the public interfaces:

```python
def test_local_command_uses_public_cli_and_one_served_alias(tmp_path):
    draft = local_draft(python_environment=tmp_path / "venv/bin/python")
    result = passing_preflight(draft, cli_path=tmp_path / "venv/bin/vllm")
    command = build_vllm_command(draft, result)
    assert command[:3] == (str(result.cli_path), "serve", "org/model")
    assert command.count("--served-model-name") == 1
    assert command[command.index("--served-model-name") + 1] == "chatbook-vllm"
    assert "vllm.entrypoints" not in " ".join(command)

@pytest.mark.parametrize(
    "raw,flag",
    [("--host 0.0.0.0", "--host"), ("--port=9000", "--port"),
     ("--model other", "--model"),
     ("--served-model-name other", "--served-model-name"),
     ("--api-key secret", "--api-key"), ("--hf-token secret", "--hf-token")],
)
def test_raw_arguments_cannot_override_managed_or_secret_flags(raw, flag):
    errors = validate_raw_arguments(raw)
    assert errors == (VllmIssue("arguments_conflict", "raw_arguments", flag),)
```

Also cover HF IDs, local directories, IPv4/IPv6 wildcard-to-loopback client URLs, real default values, bounded integers/floats, `trust_remote_code=False`, and semantic fingerprints changing for every launch field but not profile name.

- [ ] **Step 3: Run the pure tests and record the expected red state**

Run:

```bash
.venv/bin/python -m pytest -q Tests/LLM_Management/test_vllm_setup.py
```

Expected: collection fails because `tldw_chatbook.UI.LLM_Management.vllm_setup` does not exist.

- [ ] **Step 4: Implement immutable contracts and validation**

Use these exact public shapes:

```python
class VllmMode(StrEnum):
    LOCAL = "local"
    EXISTING = "existing"

class VllmModelSource(StrEnum):
    HUGGING_FACE = "hugging_face"
    LOCAL_DIRECTORY = "local_directory"

class VllmReadinessState(StrEnum):
    NOT_CONFIGURED = "not_configured"
    CHECKING = "checking"
    READY_TO_START = "ready_to_start"
    LAUNCHING = "launching"
    LOADING_MODEL = "loading_model"
    READY = "ready"
    STOPPING = "stopping"
    NEEDS_ATTENTION = "needs_attention"

@dataclass(frozen=True, slots=True)
class VllmLaunchDraft:
    mode: VllmMode
    python_environment: str
    model_source: VllmModelSource
    model_value: str
    bind_address: str = "127.0.0.1"
    port: int = 8000
    existing_server_url: str = ""
    dtype: str = ""
    tensor_parallel_size: int | None = None
    maximum_model_length: int | None = None
    gpu_memory_utilization: float | None = None
    trust_remote_code: bool = False
    raw_arguments: str = field(default="", repr=False, compare=False)

@dataclass(frozen=True, slots=True)
class VllmIssue:
    code: str
    field: str
    detail: str = ""

@dataclass(frozen=True, slots=True)
class VllmPreflightResult:
    generation: int
    fingerprint: str
    issues: tuple[VllmIssue, ...]
    python_version: str | None = None
    vllm_version: str | None = None
    cli_path: Path | None = field(default=None, repr=False)
    network_exposed: bool = False

@dataclass(frozen=True, slots=True)
class VllmLaunchSnapshot:
    generation: int
    fingerprint: str
    client_api_url: str
    served_model: str
    display_profile_name: str

@dataclass(frozen=True, slots=True)
class VllmConnectionTarget:
    provider_key: Literal["vllm"]
    api_url: str
    model_id: str
    runtime_owner: Literal["chatbook", "external"]
    generation: int
    credential_source: Literal["none", "configured", "environment"]
```

`run_vllm_preflight(draft, generation, *, run=subprocess.run, which=shutil.which, port_available=is_port_available)` executes bounded argv-only probes, validates a selected local directory with `path_validation`, checks port availability without binding persistently, and returns allowlisted issue codes. `build_vllm_command()` accepts only a successful, matching fingerprint and appends approved structured flags before `shlex.split(raw_arguments)`.

- [ ] **Step 5: Replace the inline vLLM pane with a focused view and mounted red tests**

Define view messages with immutable payloads:

```python
class VllmSetupView(VerticalScroll):
    class CheckRequested(Message):
        def __init__(self, draft: VllmLaunchDraft) -> None:
            super().__init__()
            self.draft = draft
    class StartRequested(Message):
        def __init__(self, draft: VllmLaunchDraft) -> None:
            super().__init__()
            self.draft = draft
    class StopRequested(Message):
        """Request settlement of the exact Chatbook-owned process."""
    class DraftChanged(Message):
        def __init__(self, draft: VllmLaunchDraft) -> None:
            super().__init__()
            self.draft = draft

    def apply_state(
        self,
        *,
        draft: VllmLaunchDraft,
        state: VllmReadinessState,
        preflight: VllmPreflightResult | None,
    ) -> None:
        self._draft = draft
        self._state = state
        self._preflight = preflight
        self._render_projection()
```

Mounted tests must assert exact initial copy, actual `8000` value, source-specific controls, mode-draft preservation, visible field-adjacent blocker, disabled Start before success, and no legacy `GGUF` or `checkpoint` copy.

- [ ] **Step 6: Run the Task 1 focused tests green**

Run:

```bash
.venv/bin/python -m pytest -q Tests/LLM_Management/test_vllm_setup.py Tests/UI/test_vllm_lab_workflow.py -k 'preflight or initial or mode or command or source'
.venv/bin/python -m pytest -q Tests/LLM_Management/test_gguf_server_sources.py Tests/UI/test_llm_deferred_views.py
git diff --check
```

Expected: all selected nodes pass; the incumbent llama.cpp/llamafile/MLX behavior remains green.

- [ ] **Step 7: Close and commit TASK-31388**

Check every AC, add exact focused-test evidence and no-ADR rationale to Implementation Notes, mark Done through Backlog CLI, then commit only Task 1 files:

```bash
git commit -m "feat(models): add guided vllm setup preflight"
```

---

### Task 2: TASK-31389 — Generation-fenced API and model readiness

**Files:**
- Create: `tldw_chatbook/UI/LLM_Management/vllm_connection.py`
- Modify: `tldw_chatbook/UI/LLM_Management/vllm_setup.py`
- Modify: `tldw_chatbook/UI/LLM_Management/vllm_setup_view.py`
- Modify: `tldw_chatbook/UI/Screens/llm_screen.py`
- Modify: `tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_vllm.py`
- Test: `Tests/LLM_Management/test_vllm_connection.py`
- Test: `Tests/UI/test_vllm_lab_workflow.py`
- Modify: `backlog/tasks/task-31389 - Add-generation-fenced-vLLM-API-and-model-readiness.md`

**Interfaces:**
- Consumes: Task 1 contracts and shared `server_lifecycle` claims.
- Produces: `VllmConnectionOwner.begin()`, `.settle()`, `.invalidate()`, `.snapshot()`, `probe_vllm_target()`, and current-generation `VllmConnectionTarget` for Tasks 3–5.

- [ ] **Step 1: Start TASK-31389 and copy this section plus ADR-117 rationale into its task file**

Use `backlog task edit 31265 -a @codex -s "In Progress"`; state that no new ADR is required because ADR-117 fixes owner, fencing, privacy, and rollback.

- [ ] **Step 2: Write red owner, probe, cancellation, and privacy tests**

Pin these behaviors with a controllable loopback ASGI/server fixture:

```python
async def test_ready_requires_health_and_exact_models_identity(loopback_vllm):
    loopback_vllm.health_status = 200
    loopback_vllm.models = [{"id": "chatbook-vllm"}]
    result = await probe_vllm_target(probe_request(loopback_vllm.url))
    assert result.target.model_id == "chatbook-vllm"

def test_older_generation_cannot_replace_newer_owner_state():
    owner = VllmConnectionOwner()
    old = owner.begin(local_draft(), runtime_owner="chatbook")
    current = owner.begin(replace(local_draft(), port=8001), runtime_owner="chatbook")
    assert owner.settle(old, ready_result(old)) is False
    assert owner.snapshot().generation == current.generation
```

Also assert timeout, auth-required, healthy-but-model-missing, malformed JSON, process exit, draft edit, screen detach, and recomposition. Search captured logs/activity/notifications for credential, path, command, raw response, and model-source canaries.

- [ ] **Step 3: Implement the app-scoped owner and bounded activity**

Use:

```python
@dataclass(frozen=True, slots=True)
class VllmOperationToken:
    generation: int
    fingerprint: str
    runtime_owner: Literal["chatbook", "external"]

@dataclass(frozen=True, slots=True)
class VllmActivityEvent:
    code: str
    elapsed_bucket: str
    exit_code: int | None = None

class VllmConnectionOwner:
    """Own the latest operation token and immutable settled snapshot."""
```

Implement these exact methods: `begin(draft, *, runtime_owner) -> VllmOperationToken` increments the generation and clears settled evidence; `invalidate(reason) -> int` increments the generation and appends one allowlisted event; `settle(token, result) -> bool` accepts only the exact current generation/fingerprint/owner tuple; `snapshot() -> VllmConnectionSnapshot` returns an immutable copy. Define `VllmProbeResult` with `token`, `state`, `target`, `issue`, and bounded activity fields, and `VllmConnectionSnapshot` with the current token, readiness state, launch snapshot, target, issue, and activity tuple.

Keep at most 32 allowlisted activity events for the current operation. `probe_vllm_target()` uses `httpx.AsyncClient` with explicit connect/read/total bounds, configured vLLM authorization resolution, `/health`, then `/v1/models`; it accepts only bounded printable model IDs and rejects path-like IDs.

- [ ] **Step 4: Move lifecycle orchestration to LLMScreen**

`LLMScreen` lazily installs one owner on `app_instance._vllm_connection_owner`, captures the successful Task 1 preflight, reserves the server claim, calls `run_server_subprocess`, and probes until ready/timeout/cancellation. `llm_management_events_vllm.py` becomes picker/compatibility glue; it must no longer build the legacy Python-module command or equate worker completion with readiness.

On every semantic field change:

```python
token = self._vllm_owner.invalidate("target_changed")
self._cancel_vllm_workers()
self._apply_vllm_view_state(focus=False)
```

Only explicit Check/Start/Retry moves focus. Start focuses Stop; ready focuses Use in Console once Task 3 mounts it; failure expands Activity details and focuses its recovery action.

- [ ] **Step 5: Run focused readiness and incumbent lifecycle tests**

```bash
.venv/bin/python -m pytest -q Tests/LLM_Management/test_vllm_connection.py Tests/UI/test_vllm_lab_workflow.py -k 'readiness or generation or activity or loading or cancellation'
.venv/bin/python -m pytest -q Tests/LLM_Management/test_server_lifecycle_resources.py Tests/UI/test_lab_server_status.py
git diff --check
```

- [ ] **Step 6: Close and commit TASK-31389**

Check ACs, record loopback/claim/privacy evidence, mark Done, and commit:

```bash
git commit -m "feat(models): verify vllm API and model readiness"
```

---

### Task 3: TASK-31390 — Verified session adoption and Settings delegation

**Files:**
- Create: `tldw_chatbook/UI/Navigation/vllm_handoff.py`
- Modify: `tldw_chatbook/UI/Navigation/pending_handoff_store.py`
- Modify: `tldw_chatbook/UI/LLM_Management/vllm_setup_view.py`
- Modify: `tldw_chatbook/UI/Screens/llm_screen.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/app.py`
- Test: `Tests/UI/test_vllm_lab_workflow.py`
- Test: `Tests/Chat/test_console_session_settings.py`
- Test: `Tests/UI/test_console_provider_apply_defaults_flow.py`
- Modify: `backlog/tasks/task-31390 - Adopt-verified-vLLM-targets-into-Console.md`

**Interfaces:**
- Consumes: current `VllmConnectionTarget` from Task 2 and existing Console/Settings mutation owners.
- Produces: `VllmConsoleIntent`, `VllmDefaultIntent`, two typed handoff channels, Console claim settlement, and Settings prefill settlement.

- [ ] **Step 1: Start TASK-31390 and record ADR-117 as the accepted cross-screen contract**

Use the same Backlog transition pattern. Do not mark the task done until both no-config-write and Settings-rollback tests pass.

- [ ] **Step 2: Write red typed-handoff and active-session tests**

```python
def test_vllm_console_intent_is_secret_free_and_exact():
    intent = VllmConsoleIntent.from_target(ready_target())
    assert intent == VllmConsoleIntent(
        api_url="http://127.0.0.1:8000/v1",
        model_id="chatbook-vllm",
        generation=7,
    )

async def test_use_in_console_replaces_session_without_config_write(app, monkeypatch):
    write = monkeypatch.spy(config, "save_settings_to_cli_config")
    await launch_ready_vllm_and_press_use(app)
    settings = active_console_settings(app)
    assert (settings.provider, settings.model, settings.base_url, settings.source) == (
        "vllm", "chatbook-vllm", "http://127.0.0.1:8000/v1", "user"
    )
    assert write.call_count == 0
```

Add stale-generation rejection, inactive/detached origin rollback, pending-claim replay, and configured-different-endpoint preservation.

- [ ] **Step 3: Add secret-free intents and detach validation**

```python
@dataclass(frozen=True, slots=True)
class VllmConsoleIntent:
    api_url: str
    model_id: str
    generation: int

@dataclass(frozen=True, slots=True)
class VllmDefaultIntent:
    api_url: str
    model_id: str
    generation: int
```

Add `VLLM_CONSOLE` and `VLLM_DEFAULT` to `HandoffChannel`. `_detached_value()` must accept only exact intent instances, reconstruct them field-by-field, and reject userinfo/query/fragment URLs, invalid model IDs, booleans-as-generations, and extra mutable data.

- [ ] **Step 4: Implement Console adoption as one active-session transaction**

On Console mount, claim `VLLM_CONSOLE`, verify the app-scoped owner still exposes the same generation/endpoint/model, build from the current active session, then call the existing session replacement once:

```python
next_settings = replace(
    current,
    provider="vllm",
    model=intent.model_id,
    base_url=intent.api_url,
    source="user",
)
validate_console_session_settings(next_settings, app_config)
self._session._replace_active_console_session_settings(next_settings)
```

Acknowledge only after UI synchronization succeeds; release the claim on failure. The originating Lab button stages the intent and navigates through the normal `NavigateToScreen(TAB_CHAT)` path.

- [ ] **Step 5: Implement Settings prefill without durable mutation**

`Make default for new chats` stages `VllmDefaultIntent` and navigates to Settings' canonical Providers category. Settings claims it, stages provider `vllm`, model, and endpoint through `_stage_provider_value`, displays existing endpoint-difference review copy, and leaves its ordinary Save button as the only durable action. Acknowledge after the staged draft renders; release on failure. Cancel/back must leave config byte-identical.

- [ ] **Step 6: Run focused handoff, persistence, and rollback tests**

```bash
.venv/bin/python -m pytest -q Tests/UI/test_vllm_lab_workflow.py -k 'console or default or handoff or stale'
.venv/bin/python -m pytest -q Tests/Chat/test_console_session_settings.py Tests/UI/test_console_provider_apply_defaults_flow.py Tests/Chat/test_provider_setup_persistence.py -k 'vllm or endpoint or rollback'
git diff --check
```

- [ ] **Step 7: Close and commit TASK-31390**

Record the config fingerprint evidence and commit:

```bash
git commit -m "feat(models): hand verified vllm targets to Console"
```

---

### Task 4: TASK-31391 — Non-secret profiles and honest restart drafts

**Files:**
- Create: `tldw_chatbook/UI/LLM_Management/vllm_profiles.py`
- Modify: `tldw_chatbook/UI/LLM_Management/vllm_setup.py`
- Modify: `tldw_chatbook/UI/LLM_Management/vllm_setup_view.py`
- Modify: `tldw_chatbook/UI/Screens/llm_screen.py`
- Test: `Tests/LLM_Management/test_vllm_profiles.py`
- Test: `Tests/LLM_Management/test_vllm_connection.py`
- Test: `Tests/UI/test_vllm_lab_workflow.py`
- Modify: `backlog/tasks/task-31391 - Add-current-server-snapshots-and-reusable-vLLM-launch-profiles.md`

**Interfaces:**
- Consumes: Task 2 launch snapshots/owner and Task 1 validated structured draft.
- Produces: `VllmLaunchProfileV1`, `VllmProfileDocumentV1`, `VllmProfileRepository`, selected-profile restoration, dirty comparison, and two-generation restart.

- [ ] **Step 1: Start TASK-31391 with ADR-117 persistence rationale**

Record that this is the accepted device-local JSON owner, not a database/schema migration.

- [ ] **Step 2: Write red exact-schema, corruption, atomicity, and privacy tests**

```python
def test_profile_round_trip_excludes_launch_only_and_secret_fields(tmp_path):
    repo = VllmProfileRepository(tmp_path / "vllm_profiles.json")
    saved = repo.save(profile_named("GPU 0"), expected_revision=0)
    raw = (tmp_path / "vllm_profiles.json").read_text()
    assert "raw_arguments" not in raw
    assert "credential" not in raw
    assert repo.load().profiles == (saved.profile,)

def test_future_version_is_preserved_byte_for_byte(tmp_path):
    path = tmp_path / "vllm_profiles.json"
    original = b'{"version":2,"opaque":"keep"}\n'
    path.write_bytes(original)
    with pytest.raises(VllmProfileFutureVersion):
        VllmProfileRepository(path).save(profile_named("No overwrite"), expected_revision=0)
    assert path.read_bytes() == original
```

Cover 32-profile cap, exact key set, invalid types, >120-character names, control characters, Unicode casefold/canonical-whitespace collision, duplicate suffixes, delete-last recreating `Default vLLM`, write failure preserving old bytes, and selected-profile restoration.

- [ ] **Step 3: Implement the repository using the shared atomic writer**

```python
@dataclass(frozen=True, slots=True)
class VllmLaunchProfileV1:
    profile_id: str
    name: str
    python_environment: str
    model_source: VllmModelSource
    model_value: str
    bind_address: str
    port: int
    dtype: str
    tensor_parallel_size: int | None
    maximum_model_length: int | None
    gpu_memory_utilization: float | None
    trust_remote_code: bool

@dataclass(frozen=True, slots=True)
class VllmProfileDocumentV1:
    version: Literal[1]
    revision: int
    selected_profile_id: str
    profiles: tuple[VllmLaunchProfileV1, ...]
```

Resolve the path as `get_user_data_dir() / "vllm_launch_profiles.json"`. Validate before calling `atomic_write_json`; use compare-and-swap `expected_revision` so two app instances cannot silently overwrite each other's profile set.

- [ ] **Step 4: Render Current server separately from Next restart configuration**

The current card reads only `VllmLaunchSnapshot`; fields remain immutable until the exact process claim settles dead. The editable draft reads the selected profile plus launch-only raw arguments. Compare semantic fingerprints and render `Modified for next restart` plus per-field markers without exposing model paths in the Inspector or confirmation.

Profile actions post exact messages (`CreateProfileRequested`, `SaveProfileRequested`, `RenameProfileRequested`, `DuplicateProfileRequested`, `DeleteProfileRequested`) and LLMScreen performs repository I/O in a thread worker. Selection never starts/stops a process.

- [ ] **Step 5: Implement safe Restart with draft sequencing**

Require a matching successful preflight. Confirmation lists allowlisted changed field labels only. Stop the old claim, await proof `poll() is not None`, release it, then reserve a new generation and launch. If termination fails, remain in Needs attention with the old current snapshot and no second process.

- [ ] **Step 6: Run focused profile/restart tests**

```bash
.venv/bin/python -m pytest -q Tests/LLM_Management/test_vllm_profiles.py Tests/LLM_Management/test_vllm_connection.py -k 'profile or restart or snapshot'
.venv/bin/python -m pytest -q Tests/UI/test_vllm_lab_workflow.py -k 'profile or current_server or next_restart or restart'
git diff --check
```

- [ ] **Step 7: Close and commit TASK-31391**

Record storage path isolation, future-version, atomic-failure, and two-generation evidence, then commit:

```bash
git commit -m "feat(models): add vllm launch profiles and safe restart"
```

---

### Task 5: TASK-31392 — Responsive, keyboard-contained completion and live evidence

**Files:**
- Modify: `tldw_chatbook/UI/LLM_Management/vllm_setup_view.py`
- Modify: `tldw_chatbook/UI/LLM_Management_Window.py`
- Modify: `tldw_chatbook/UI/Screens/llm_screen.py`
- Modify: `tldw_chatbook/UI/Screens/lab_frame.py` only if the production geometry proves the existing collapse API insufficient
- Modify: `tldw_chatbook/css/features/_lab.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Create: `Tests/UI/test_vllm_lab_geometry.py`
- Modify: `Tests/UI/test_vllm_lab_workflow.py`
- Modify: `Docs/superpowers/specs/2026-09-03-vllm-lab-console-complete-redesign.md`
- Modify: `backlog/tasks/task-31392 - Make-vLLM-setup-responsive-and-keyboard-contained.md`

**Interfaces:**
- Consumes: complete Task 1–4 state projection and Lab rail-collapse primitives.
- Produces: production-stylesheet width classes, deterministic focus targets, final UI matrix, and honest live-qualification record.

- [ ] **Step 1: Read the impeccable craft-floor instructions immediately before UI edits**

Read `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.agents/skills/impeccable/references/craft-floor.md` completely. Apply it within the already-approved Operate direction and surface brief; do not revise the established product visual language.

- [ ] **Step 2: Start TASK-31392 and pin the no-new-ADR rationale**

The task implements ADR-117's responsive/focus contract. Add the exact size/state matrix below to its Implementation Plan.

- [ ] **Step 3: Write red production-stylesheet compositor tests**

Mount through the real `TldwCli.CSS_PATH` stack at `(80, 24)`, `(100, 30)`, and `(120, 40)`. Parameterize these states:

```python
VLLM_GEOMETRY_STATES = (
    "setup_incomplete", "preflight_ready", "launching", "loading",
    "ready", "failed", "dirty_restart", "profile_management",
)

@pytest.mark.parametrize("size", [(80, 24), (100, 30), (120, 40)])
@pytest.mark.parametrize("state", VLLM_GEOMETRY_STATES)
async def test_every_visible_focusable_is_inside_its_owner(size, state):
    app = VllmProductionCssHarness(state=state)
    async with app.run_test(size=size) as pilot:
        view = await mounted_vllm_view(app, pilot)
        for widget in visible_focusables(view, app.screen._compositor):
            assert widget.region.intersection(owner_region(widget)) == widget.region
```

Add a complete Tab walk asserting visited IDs equal the state-specific expected tuple and never include hidden provider-body controls.

- [ ] **Step 4: Remove conflicting child bindings and implement measured composition modes**

Delete `LLMManagementWindow`'s bracket and digit view bindings; Lab's documented `[`/`]` mode-focus binding remains authoritative. Derive one of `vllm-wide`, `vllm-medium`, `vllm-compact` from the mounted body width, not global terminal width. At compact size, collapse the catalog after vLLM selection through the existing rail store/action and keep its reopen control painted.

Apply stable focus transitions:

```python
VLLM_FOCUS_TARGET = {
    VllmReadinessState.READY_TO_START: "vllm-start",
    VllmReadinessState.LAUNCHING: "vllm-stop",
    VllmReadinessState.LOADING_MODEL: "vllm-stop",
    VllmReadinessState.READY: "vllm-use-console",
    VllmReadinessState.NEEDS_ATTENTION: "vllm-recovery-primary",
}
```

Only explicit transitions call `.focus()`; timer refreshes and background invalidation preserve current focus.

- [ ] **Step 5: Add source TCSS and regenerate the bundle**

Use semantic tokens and these layout invariants: wide setup rows may be horizontal; medium hides Inspector and stacks any overflowing control group; compact makes label/input/action complete rows, uses full-width Browse, keeps Readiness plus next action in the first viewport, and shows a fold cue when Activity/Console actions are below.

Run:

```bash
cd tldw_chatbook/css && ../../.venv/bin/python build_css.py
```

Review `git diff --name-only`; keep `_lab.tcss` and regenerated semantic outputs, and discard timestamp-only generated churn through a narrow patch rather than editing the bundle by hand.

- [ ] **Step 6: Run the full vLLM focused verification matrix**

```bash
.venv/bin/python -m pytest -q Tests/LLM_Management/test_vllm_setup.py Tests/LLM_Management/test_vllm_connection.py Tests/LLM_Management/test_vllm_profiles.py Tests/UI/test_vllm_lab_workflow.py Tests/UI/test_vllm_lab_geometry.py
.venv/bin/python -m pytest -q Tests/LLM_Management/test_gguf_server_sources.py Tests/LLM_Management/test_server_lifecycle_resources.py Tests/UI/test_lab_frame.py Tests/UI/test_lab_frame_mode_keys.py Tests/UI/test_lab_rail_layout.py Tests/UI/test_lab_server_status.py Tests/UI/test_llm_deferred_views.py Tests/Chat/test_console_session_settings.py Tests/UI/test_console_provider_apply_defaults_flow.py
git diff --check
```

- [ ] **Step 7: Perform isolated live verification without overstating capability**

Create a disposable `HOME`, XDG config/data/cache directories, and `TLDW_CONFIG_PATH`; set `[paths].data_dir` to the scratch data root. Fingerprint the real config/data files before and after. Drive Lab → Models → vLLM at 80x24, 100x30, and 120x40 through incomplete setup and recovery.

If an eligible local vLLM environment and chat model exist, verify Check setup → Start → loading → `/health` → `/v1/models` → Use in Console → one response → Stop. Otherwise record the exact missing prerequisite and label loopback tests as contract verification, not real-vLLM qualification.

- [ ] **Step 8: Self-review against the approved spec and close TASK-31392**

For every Goals, State model, Validation, Responsive, and Testing bullet in the spec, name the implementing test node in Implementation Notes. Record the live environment result, check all ACs only when evidence exists, mark Done, and commit:

```bash
git commit -m "feat(models): complete responsive vllm Lab workflow"
```

---

### Task 6: Branch integration checkpoint

**Files:**
- Modify only documentation/task evidence found stale by the checks below.

**Interfaces:**
- Consumes: all five implementation commits.
- Produces: one reviewable branch based on the then-current `origin/dev`.

- [ ] **Step 1: Re-sweep task/ADR IDs and fetch current dev**

Use every remote ref and worktree as collision discovery inputs, then verify the merge candidate itself has one owner per ID. Fetch `origin/dev`; inspect upstream changes touching every file in the File and ownership map.

- [ ] **Step 2: Rebase without destructive worktree operations**

```bash
git rebase origin/dev
```

Resolve conflicts file-by-file and stage exact paths only. Preserve newer shipped task/ADR allocations according to add-commit provenance.

- [ ] **Step 3: Re-run the complete focused matrix after rebase**

Run the five feature suites and incumbent integration suites named in Task 5 Step 6 again after the rebase. Then run the repository's generated CSS and diagnostic inventory gates affected by the diff, and inspect `git diff origin/dev...HEAD --check` plus `git status --short`.

- [ ] **Step 4: Request code review before PR/merge**

Use `superpowers:requesting-code-review`. Address evidence-backed findings with `superpowers:receiving-code-review`; rerun the smallest affected red/green set after every fix. Do not merge with unresolved review threads or required checks.

## Plan self-review

- Spec coverage: Tasks 1–5 map every approved goal, non-goal, state, error category, profile rule, handoff scope, responsive breakpoint, and evidence requirement.
- Placeholder scan: the plan contains no deferred implementation markers; every code owner has an exact interface and test command.
- Type consistency: `VllmConnectionTarget` is produced by Task 2; Task 3 derives secret-free intents from it; Task 4 reuses `VllmLaunchSnapshot`; Task 5 consumes the same `VllmReadinessState` enum and stable widget IDs.
- ADR required: no new ADR. ADR-117 is accepted and governs all implementation tasks.
