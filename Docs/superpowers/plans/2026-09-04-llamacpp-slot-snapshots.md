# llama.cpp Prompt-cache Snapshots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Manually save, restore, and delete timestamp-named prompt-cache snapshots for a Chatbook-launched llama.cpp server, retaining the newest 10 per profile by default.

**Architecture:** One app-owned service coordinates a private file store and a dedicated loopback HTTP client. Immutable launch identity, verified compatibility, and acknowledged server results govern publication and cleanup. The Models widget and canonical F9 settings project this service; neither owns subprocesses or writes snapshot files.

**Tech Stack:** Python >=3.11, Textual >=8.0.0,<9, existing httpx, Pydantic, portalocker, and ADR-029 private-path utilities. No new dependency or database.

**Spec:** [Approved design](../specs/2026-09-04-llamacpp-slot-snapshots-design.md)

**Backlog:** [TASK-31552](../../../backlog/tasks/task-31552%20-%20llama.cpp-manual-prompt-cache-snapshot-manager.md)

**Status:** Implementation in progress. Units 1–6 have passed their scoped implementation reviews; Task6 adds the opt-in harness and [evidence closeout](../reviews/2026-09-04-llamacpp-slot-snapshots-verification.md). The integrated final review's I1/I2/I3/M1/M2 findings are addressed in one bounded fix wave (156 affected tests passed); its scoped re-review closed all five with no new breakage. Live AC5 remains open because no eligible asset set or measured counters was supplied. See TASK-31552 for feature completion status.

ADR required: yes

ADR path: [ADR-119](../../../backlog/decisions/119-llamacpp-prompt-cache-snapshot-ownership.md)

Reason: existing accepted decision covers the new private file lifecycle, retention, and management boundary. Also follow [ADR-029](../../../backlog/decisions/029-local-private-data-boundary.md) and [ADR-036](../../../backlog/decisions/036-application-service-composition-lifecycle.md); no additional architectural decision is introduced here.

## Global constraints

- Persist `llamacpp_snapshots.enabled = false` and `llamacpp_snapshots.keep_count = 10`.
- Accept integer keep counts from 1 through 1000; invalid input leaves the previous value intact.
- The section is named **Prompt-cache snapshots**.
- Explanation: "Save processed context to reuse later. Restoring does not change your conversations."
- Beside Save: "Keeps the newest 10 across all models", with the effective count substituted.
- No automatic conversation binding, chat-payload changes, imports, exports, renaming, pinning, remote/router management, or in-memory Erase action.
- No implicit `--swa-full`; no alteration to ordinary launches when snapshots are off.
- TLS, custom API-prefix, router, and non-loopback configurations remain valid ordinary launches with snapshot management disabled and specific guidance; do not strip advanced arguments. Conflicting owned slot flags still cause explicit preflight failure when snapshots are enabled.
- Missing slot token counts mean Unknown, not zero. Compatibility unknown disables Save/Restore, not catalog browsing/deletion.
- Five-second readiness/observation deadlines. Mutation timeouts: connection/pool 5 seconds, write 30 seconds, read inactivity and overall submission deadline 600 seconds.
- No retry after possible submission; an unknown outcome keeps Save/Restore disabled for that launch until acknowledged completion or confirmed stop.
- Profile-local storage, owner-only POSIX files/directories, no process-global umask changes, honest unverified Windows ACL posture.
- V1 storage operations require POSIX descriptor-relative filesystem primitives. On unsupported platforms, return a fixed `unsupported_platform` reason before creating snapshot data; ordinary server launches remain available. Do not substitute weaker path-based mutations or claim verified Windows support.
- Only a complete, compatible, acknowledged and committed save can prune. Failed or uncertain saves cannot remove earlier snapshots.
- SHA-256 and byte-length verification precede every Restore POST. Working copies are cleaned only when no local/server operation may still use them.
- Targeted tests only. No full suite without user authorization. Real-model evidence is required before marking the feature Done.

## Execution boundaries and current code map

This is one vertical feature with six independently testable implementation units, not six new public subsystems. The Backlog task remains In Progress until the live criterion and all other criteria are satisfied. Do not mark an internal unit as a completed feature. Do not create unrelated Backlog tasks or a service registry.

Before execution, use the worktree skill to select safe isolation. The planning checkout has unrelated staged and unstaged work, including changes to Models and app composition. Preserve it; do not copy the entire dirty tree or commit it wholesale. Re-read the listed integration functions in the execution checkout and resolve overlaps before editing them.

Verified integration points:

- `Event_Handlers/LLM_Management_Events/llm_management_events.py`: `_build_gguf_server_command()` appends extra arguments after form host/port. `_run_gguf_server_worker()` resolves external/managed model sources and retains the existing managed lease on the launch claim.
- `Event_Handlers/LLM_Management_Events/server_lifecycle.py`: `ServerLaunchClaim`, `current_server_claim()`, `publish_server_process()`, `release_server_claim()`, `stop_server_process()`, and `run_server_subprocess()`. Process publication currently precedes API readiness. The claim's single `_resource` is already used by managed-model leases; do not commandeer it for snapshots.
- `app.py`: compose after configuration and local launch state exist. Integrate idempotent teardown into `_shutdown_app_owned_lifecycles()`, which runs before Textual closes screens and is called again defensively on unmount.
- `UI/LLM_Management_Window.py`: `compose()` and `_sync_process_controls()`. In the execution checkout, six server views are eager and five Ollama/library views arrive via `_finish_deferred_mount()`. Preserve that boundary and the existing Start/Stop IDs; do not introduce an unrelated lazy-pane refactor.
- `UI/Screens/settings_screen.py`: use Providers & Models for the snapshot preferences. Its `SettingsConfigAdapter.save_sections()` delegates to `config.save_settings_to_cli_config()`; do not use the per-key `save_values()` loop for this pair of settings.
- `config.py`: add defaults to `CONFIG_TOML_CONTENT`; use `get_cli_setting()` at action boundaries and `get_user_data_dir()` off-thread when creating the store.
- Tests already cover source authority, model leases, lazy panes, production CSS, and config ownership. Extend them instead of replacing their harnesses.

All source paths below are repository-relative under `tldw_chatbook/` unless prefixed with `Tests/`, `Docs/`, or `backlog/`.

## File responsibilities and shared interfaces

Create the `LLM_Management/` package; it does not exist in this checkout. Keep its `__init__.py` dependency-light.

| File | Responsibility |
| --- | --- |
| `LLM_Management/snapshot_models.py` | Strict metadata/receipt validation and immutable operation projections; no I/O. |
| `LLM_Management/snapshot_settings.py` | Preference validation and one adapter to the existing config owner. |
| `LLM_Management/snapshot_admission.py` | Effective args/environment, file identities, loopback endpoint, compatibility evidence. |
| `LLM_Management/snapshot_store.py` | Private working files, commit/delete transactions, retention, integrity checks, reconciliation. |
| `LLM_Management/snapshot_client.py` | Bounded management HTTP requests and safe typed errors. |
| `LLM_Management/snapshot_service.py` | One operation per launch, app-lifetime tasks, readiness and view projections. |
| `Widgets/llamacpp_snapshot_manager.py` | Slot/snapshot selection, confirmations, rendering; no file/HTTP ownership. |

Define these concrete types in `snapshot_models.py`. Use frozen Pydantic models with forbidden extra fields for disk/HTTP boundaries; use dataclasses for in-memory descriptors containing process claims and paths. Validate integer fields strictly (bool is not an integer here).

- `SnapshotPreferences`: `enabled: bool`, `keep_count: int` with the constraints above; defined in `snapshot_settings.py`.
- `CompatibilityEvidence`: `model_sha256: str`, `projector_sha256: str | None`, `runtime_sha256: str`, `build_info: str`, `state_settings: tuple[tuple[str, str], ...]`. Absence of a projector is explicit; missing model/runtime identity is not a valid instance. Settings are sorted, unique canonical keys from Task 1.
- `FileIdentity`: `path: Path`, `device: int`, `inode: int`, `size_bytes: int`, `mtime_ns: int`, `ctime_ns: int`, `sha256: str`. Paths stay in memory; metadata contains digests and safe labels, not an argv dump.
- `LaunchDescriptor`: `launch_id: str`, `claim: ServerLaunchClaim`, `base_url: str`, `bearer_token: str | None`, `child_env: dict[str, str]`, `files: tuple[FileIdentity, ...]`, `compatibility: CompatibilityEvidence | None`, `disabled_reason: str | None`. Exclude credentials, environment, file paths and claim from repr/serialization. Treat the captured environment as immutable after construction.
- `SlotObservation`: `slot_id: int`, `busy: bool | None`, `tokens: int | None`, `context_size: int | None`, `observed_at: float` (monotonic).
- `ReadinessObservation`: `slots: tuple[SlotObservation, ...]`, `build_info: str`, `model_path: str`, `runtime_values: tuple[tuple[str, str], ...]`. This internal observation contains only whitelisted properties; it is not a raw `/props` response or a UI projection.
- `SlotReceipt`: `slot_id: int`, `filename: str`, `tokens: int`, `bytes: int`. Parser maps save `n_saved/n_written` and restore `n_restored/n_read` explicitly.
- `SnapshotRecord`: `schema_version: int = 1`, `snapshot_id: str`, `filename: str`, `created_utc: str`, `publication_sequence: int`, `source_slot: int`, `tokens: int`, `bytes: int`, `sha256: str`, `model_label: str`, `compatibility: CompatibilityEvidence`.
- `WorkingFile`: `launch_id: str`, `operation_id: str`, `path: Path`, `source_record: SnapshotRecord | None`. It represents an owned reservation, not a user-supplied filename.
- `SaveResult`: `record: SnapshotRecord`, `removed_ids: tuple[str, ...]`, `cleanup_failed_ids: tuple[str, ...]`.
- `CatalogPage`: `records: tuple[SnapshotRecord, ...]`, `next_offset: int | None`, `stored_bytes: int | None`, `residual_bytes: int | None`, `scan_complete: bool`. Never label a partial scan as a complete total.
- `ManagerView`: `launch_id: str | None`, `status: str`, `operation_id: str | None`, `started_at: float | None`, `slots: tuple[SlotObservation, ...]`, `catalog: CatalogPage`, `disabled_reason: str | None`, `message: str | None`. Only bounded, payload-free copy reaches the widget.
- `SnapshotError(Exception)`: `code: str`, `submission_possible: bool`; its exception message is the fixed code, never a raw HTTP response or path.

Use these APIs consistently across the tasks:

```text
snapshot_settings.load_snapshot_preferences() -> SnapshotPreferences
snapshot_settings.save_snapshot_preferences(value: SnapshotPreferences) -> bool
snapshot_admission.prepare_launch(command: tuple[str, ...], env: Mapping[str, str],
    claim: ServerLaunchClaim, launch_id: str) -> LaunchDescriptor
snapshot_admission.revalidate_files(descriptor: LaunchDescriptor) -> bool
snapshot_admission.finalize_launch(descriptor: LaunchDescriptor,
    observation: ReadinessObservation) -> LaunchDescriptor
snapshot_admission.compatibility_matches(saved: CompatibilityEvidence,
    current: CompatibilityEvidence) -> bool
SnapshotStore(root: Path)
SnapshotStore.prepare_launch_directory(launch_id: str) -> Path
SnapshotStore.reserve_save(launch_id: str, slot_id: int) -> WorkingFile
SnapshotStore.commit_save(working: WorkingFile, receipt: SlotReceipt,
    evidence: CompatibilityEvidence, model_label: str, keep_count: int,
    *, validate_publication: Callable[[], bool] | None = None) -> SaveResult
SnapshotStore.stage_restore(snapshot_id: str, launch_id: str) -> WorkingFile
SnapshotStore.list_records(offset: int = 0, limit: int = 50) -> CatalogPage
SnapshotStore.delete(snapshot_id: str) -> tuple[str, ...]
SnapshotStore.cleanup(working: WorkingFile) -> tuple[str, ...]
SnapshotStore.set_operation_state(working: WorkingFile,
    state: Literal["unknown", "acknowledged", "terminal"]) -> None
SnapshotStore.reconcile(terminated_launch_ids: frozenset[str]) -> tuple[str, ...]
SnapshotClient(descriptor: LaunchDescriptor, *, transport: httpx.AsyncBaseTransport | None = None)
SnapshotClient.readiness() -> async ReadinessObservation
SnapshotClient.slots() -> async tuple[SlotObservation, ...]
SnapshotClient.save(slot_id: int, filename: str) -> async SlotReceipt
SnapshotClient.restore(slot_id: int, filename: str) -> async SlotReceipt
SnapshotClient.aclose() -> async None
LlamaCppSnapshotService(store: SnapshotStore | None,
    is_current: Callable[[ServerLaunchClaim], bool])
LlamaCppSnapshotService.initialize(root_factory: Callable[[], Path]) -> async None
LlamaCppSnapshotService.attach(descriptor: LaunchDescriptor) -> None
LlamaCppSnapshotService.refresh() -> async None
LlamaCppSnapshotService.browse_catalog(offset: int = 0, limit: int = 50) -> async None
LlamaCppSnapshotService.start_save(slot_id: int) -> str
LlamaCppSnapshotService.start_restore(snapshot_id: str, slot_id: int) -> str
LlamaCppSnapshotService.delete_snapshot(snapshot_id: str) -> async None
LlamaCppSnapshotService.server_stopped(claim: ServerLaunchClaim, confirmed: bool) -> async None
LlamaCppSnapshotService.view() -> ManagerView
LlamaCppSnapshotService.subscribe(listener: Callable[[], None]) -> Callable[[], None]
LlamaCppSnapshotService.shutdown() -> async None
```

`start_*` synchronously admits or rejects and returns an operation ID, then owns the coroutine through a strongly held `asyncio.Task`. The widget never awaits that task as its worker. `subscribe()` returns an unsubscribe callback. Store APIs are synchronous and run through the service's off-thread boundary; client/service async APIs run on the app loop. Cleanup returns bounded failure codes, not arbitrary paths. `delete()` acts only on the selected verified record.

## Task 1: Validated configuration and launch admission

**Files:** Create package `__init__.py`, `snapshot_models.py`, `snapshot_settings.py`, `snapshot_admission.py`; modify `config.py`; create `Tests/LLM_Management/test_snapshot_admission.py`, `Tests/LLM_Management/test_snapshot_settings.py`.

**Interfaces:** Produces the shared types and the settings/admission APIs above. Consumes existing `ServerLaunchClaim`, config owner, path validation and managed GGUF identity primitives.

- [x] Write strict preference tests before adding defaults or modules:

```python
import pytest
from pydantic import ValidationError
from tldw_chatbook.LLM_Management.snapshot_settings import SnapshotPreferences

@pytest.mark.parametrize("bad", [0, 1001, True, 1.5, "10"])
def test_keep_count_rejects_non_integer_or_out_of_range(bad):
    with pytest.raises(ValidationError):
        SnapshotPreferences(keep_count=bad)

def test_snapshot_defaults_are_opt_in_and_keep_ten():
    assert SnapshotPreferences().model_dump() == {"enabled": False, "keep_count": 10}
```

- [x] Run `python -m pytest Tests/LLM_Management/test_snapshot_settings.py -q`; record RED due to the absent feature module. Use the project's Python >=3.11 environment; do not silently use a system interpreter below the version floor.
- [x] Implement the constrained preferences and one persistence call. Parse UI text to a strict integer before constructing the model; invalid input leaves disk and effective settings unchanged. Missing settings get defaults; malformed configured settings yield a visible validation error rather than silently granting Save.

```python
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, StrictBool
from tldw_chatbook.config import save_settings_to_cli_config

class SnapshotPreferences(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: StrictBool = False
    keep_count: Annotated[int, Field(strict=True, ge=1, le=1000)] = 10

def save_snapshot_preferences(value: SnapshotPreferences) -> bool:
    return save_settings_to_cli_config({"llamacpp_snapshots": value.model_dump()})
```

- [x] Add `[llamacpp_snapshots]` to `CONFIG_TOML_CONTENT`, keeping the class/template values identical. Test loading through the real isolated config owner, a failed save, and changed count reload; a class-default-only assertion does not test the shipping template.
- [x] Add admission RED cases: last host/port argument wins; `--key=value` and aliases; environment fallback; empty/invalid key file; IPv4/IPv6 loopback; `localhost` with a non-loopback result; invalid port; conflicting slot flags; model/projector file replacement; missing compatibility evidence; credentials absent from repr and persisted records.
- [x] Implement a bounded parser for the already-built command, not a second launcher. Explicit CLI values override recognized upstream environment values; unknown options or unknown `LLAMA_ARG_*` settings cannot be silently ignored when declaring compatibility. Unknown management transport disables the manager; conflicting owned slot flags fail snapshot-enabled preflight as specified. Keep ordinary launches available with snapshots off.

Canonical state-setting groups to represent from the pinned upstream baseline:

| Group | Canonical options/identity that must be captured |
| --- | --- |
| Model files | Resolved primary GGUF and every split shard actually loaded; resolved projector (`--mmproj`) and its digest. Managed lease remains owned by the existing claim resource. |
| Context | `--ctx-size`, `--parallel`, `--cont-batching`/`--no-cont-batching`, `--context-shift`/`--no-context-shift`, `--keep`; validate actual per-slot `n_ctx` from readiness, not a form estimate. |
| Cache/attention | `--cache-type-k`, `--cache-type-v`, `--swa-full`, `--flash-attn`, `--kv-offload`/`--no-kv-offload`, `--batch-size`, `--ubatch-size`. |
| Position | `--rope-scaling`, `--rope-scale`, `--rope-freq-base`, `--rope-freq-scale`, `--yarn-orig-ctx`, `--yarn-ext-factor`, `--yarn-attn-factor`, `--yarn-beta-slow`, `--yarn-beta-fast`. |
| Execution identity | Executable digest, `/props` build identity, device/layer/split settings, `--fit` inputs and observed effective context. If effective state cannot be established, disable Save/Restore instead of asserting a match. |
| Media | `--mmproj`, `--mmproj-auto`/`--no-mmproj`/`--no-mmproj-auto`, `--mmproj-offload`/`--no-mmproj-offload`, `--mmproj-device`, `--image-min-tokens`, `--image-max-tokens`, `--mtmd-batch-max-tokens`. Include `MTMD_BACKEND_DEVICE` when resolving the projector device. Projector URLs or video-specific overrides are unsupported snapshot configurations in v1, not silently ignored. |
| Excluded mutable/special modes | LoRA (including per-request scaling), control vectors, model-metadata overrides, speculative/draft models, RPC, router, custom prefix, TLS. Reject snapshot mutation for these v1 configurations with a specific reason; do not strip flags or break ordinary launches. |
| Non-state controls | Host/port/key, sampling-only options, threads, verbosity, and metrics do not by themselves invalidate a file. Consume their known arity correctly so values cannot masquerade as flags. |

Store unset model-derived values as explicit canonical sentinels paired with identical model/build identity, not guessed numeric defaults. Never treat an unknown flag as an unset known option. For `auto` settings whose effective result affects compatibility, use verified runtime observations or mark evidence unavailable. This conservative admission implements the spec's unknown-evidence rule; it is not a blanket promise of support for every llama.cpp flag.

For split models, `model_sha256` is SHA-256 of a canonical ordered manifest of shard numbers, lengths and verified content digests; the single-file case uses its content digest directly. Missing shards or an unresolvable projector leave evidence unavailable. `prepare_launch()` creates the pre-readiness descriptor; `finalize_launch()` combines its verified file/argument identity with the observed build, model path and effective slot context, returning an immutable replacement for the same claim/launch ID. Refresh never borrows current UI values to complete missing evidence.

- [x] Verify aliases, environment names and supported value domains against `common/arg.cpp` and the server README at `427291b5b34cd914a31b3fd3b61a68f6184f4b9f`; retain focused argument fixtures in the admission test file. Do not fetch GitHub at app runtime. Hash missing artifact identities off-thread once per unchanged file identity. Revalidation compares device/inode/size/mtime/ctime; a changed identity requires re-admission, not rebinding a running server to new bytes.
- [x] Run `python -m pytest Tests/LLM_Management/test_snapshot_settings.py Tests/LLM_Management/test_snapshot_admission.py Tests/LLM_Management/test_gguf_server_sources.py -q`; record GREEN. Commit only these task files and the config change: `feat: validate llama.cpp snapshot settings and launch identity`.

## Task 2: Private snapshot store and commit-before-prune retention

**Files:** Create `snapshot_store.py`, `Tests/LLM_Management/test_snapshot_store.py`, `Tests/LLM_Management/snapshot_fixtures.py`; extend `snapshot_models.py` only for the declared types.

**Interfaces:** Produces `SnapshotStore` APIs, `SnapshotRecord`, `WorkingFile`, `SaveResult`, and `CatalogPage`; consumes validated compatibility evidence and receipts. The store never decides whether an HTTP request completed.

- [x] Add reusable test evidence and an integrity regression. The settings in this fixture must be the complete canonical minimal-state set validated by Task 1, constructed by its real parser from a recorded minimal launch; do not bypass the evidence validator with `model_construct()`.
- [x] Write RED tests for reserve/commit, newest-N across models, rapid timestamp collision, clock rollback, keep=1, invalid metadata, foreign files, symlinks/hardlinks, failed binary flush, failed sidecar commit, failed prune, and two-process publication/deletion. Use real private temp directories and `multiprocessing` events, not only mocked locks.
- [x] Implement the on-disk layout:

```text
llamacpp_snapshots/
  catalog.lock                  # private cross-process lock
  publication.json              # private durable monotonic sequence counter
  catalog/                      # committed .bin and schema-v1 .json pairs
  working/<launch_id>/           # only this directory is passed to llama-server
    <operation_id>.json          # owned operation reservation/state
    slot-<id>-<UTC>-<uuid>.bin    # save or staged restore copy
```

Use generated IDs and basename validation. Reservation manifests hold only the generated IDs, operation kind, expected member identity, and acknowledged/unknown/terminal state; never serialize credentials or request bodies. POSIX directory/file creation goes through the existing private-path primitives. Precreate child write targets as 0600 and use the POSIX `Popen(umask=0o077)` parameter for snapshot-enabled launches; do not call process-global `os.umask()` or `preexec_fn`.

- [x] Implement save publication: validate positive receipt counts and exact basename/slot; verify regular-file identity and receipt byte size; flush/fsync binary; hash off-thread; under `portalocker` durably allocate `max(counter, observed valid publication_sequence)+1`, move binary into catalog, then atomically write metadata as commit marker. Initialize/recover a missing or invalid counter only after a complete safe catalog scan; otherwise fail publication with a fixed ordering-unavailable error and preserve earlier records. Allocation gaps after interrupted saves are harmless. Keep binary and metadata in the same committed directory and make its publication durable before deleting old records. Metadata-first tombstoning makes deletion crash-recoverable. A failed publication never runs pruning.
- [x] Bound metadata to 64 KiB and scans to 10,000 entries per pass; page visible results at 50. If a complete safe catalog scan cannot be obtained, publish without pruning and report cleanup incomplete; mark totals unknown. Do not silently prune only the first page. Foreign/malformed entries are untouched, not adopted or deleted.
- [x] Implement restore staging under the catalog lock, using chunked reads/writes through verified handles, checking length and SHA-256. No hard links. On short write, corrupt bytes, changed identity or full disk, close handles and clean only that owned partial file. Confirm the staged member's final identity/length before handing it to the service.

```python
import pytest

def test_corrupt_snapshot_never_produces_restore_staging(tmp_path):
    from tldw_chatbook.LLM_Management.snapshot_models import SnapshotError
    from tldw_chatbook.LLM_Management.snapshot_store import SnapshotStore
    from Tests.LLM_Management.snapshot_fixtures import commit_test_snapshot

    store = SnapshotStore(tmp_path / "snapshots")
    record = commit_test_snapshot(store, payload=b"original", slot_id=0)
    binary = tmp_path / "snapshots" / "catalog" / record.filename
    binary.write_bytes(b"modified")  # Same length: size checking alone cannot pass.
    with pytest.raises(SnapshotError, match="integrity_mismatch"):
        store.stage_restore(record.snapshot_id, "test-launch-b")
    assert binary.read_bytes() == b"modified"  # Source retained for inspection.
```

Define `commit_test_snapshot(store: SnapshotStore, *, payload: bytes, slot_id: int) -> SnapshotRecord` in `snapshot_fixtures.py`: reserve a save, write the supplied test bytes to its private file, build the matching positive `SlotReceipt`, and call the real `commit_save` with Task 1's complete evidence and `keep_count=10`. The fixture bypasses only llama-server, not validation/publication/retention.

- [x] Implement cleanup/reconciliation for the exact reservation states. Repeated successful restore cycles leave no staged binaries. Unknown writers stay untouched unless their launch is in `terminated_launch_ids`; after an app crash an unverifiable launch is not added merely because its PID is absent or old. Surface residual storage and never automatically promote an unacknowledged binary. Test pre-submission abort, empty acknowledged save, acknowledged failure, cleanup warning, and unknown operation retention separately.

The service durably marks `unknown` before possible POST, `acknowledged` after a valid successful receipt, and `terminal` only after local file work settles and no server operation may use the member. Acknowledgement alone is not cleanup eligibility for another process: reconciliation cleans terminal reservations, or other reservations only after confirmed launch termination and settled local work. Store filesystem methods hold the catalog lock for their owned handle lifetimes; direct owner cleanup is allowed for safely settled reserved/acknowledged/terminal work. `commit_save` may acknowledge its supplied valid receipt internally. Failed state persistence prevents submission.
- [x] Run `python -m pytest Tests/LLM_Management/test_snapshot_store.py Tests/Utils/test_private_paths.py -q`; record RED/GREEN and mutation-check removal of the checksum/commit-before-prune guards. Commit only this unit: `feat: store private prompt-cache snapshots with safe retention`.

## Task 3: Loopback-only management transport

**Files:** Create `snapshot_client.py`, `Tests/LLM_Management/test_snapshot_client.py`; extend receipt/error validators in `snapshot_models.py`.

**Interfaces:** Produces `SnapshotClient` APIs and safe `SnapshotError` codes; consumes immutable `LaunchDescriptor`. It neither publishes files nor retries mutations.

- [x] Write recording `httpx.MockTransport` tests for GET health/props/slots and exact POST path/query/body; slot array with absent metrics; malformed/oversized JSON; unsupported route; unauthorized response; unexpected redirect; mismatched receipt basename/slot; and 200/error bodies containing secret/prompt canaries. Assert canaries are absent from logs, exception text, and projections.
- [x] Write RED for explicit defaults and proxy isolation:

```python
import httpx

def test_mutation_timeouts_do_not_inherit_probe_timeout():
    from tldw_chatbook.LLM_Management.snapshot_client import MUTATION_TIMEOUT
    assert isinstance(MUTATION_TIMEOUT, httpx.Timeout)
    assert MUTATION_TIMEOUT.connect == 5
    assert MUTATION_TIMEOUT.pool == 5
    assert MUTATION_TIMEOUT.write == 30
    assert MUTATION_TIMEOUT.read == 600
```

- [x] Implement a dedicated client with `trust_env=False`, `follow_redirects=False`, no proxy, and the descriptor's validated numeric loopback URL. Give each GET an overall 5-second `asyncio.timeout` and bound the complete health/props/slots readiness observation by the same 5-second overall budget; test the aggregate with a short injected deadline. Wrap each POST in an overall 600-second timeout and the explicit per-phase values. Stream responses into a capped 1 MiB buffer; parse only whitelisted fields and discard raw bodies. Unsupported, auth and protocol errors become fixed codes. On a possibly submitted mutation with no valid terminal response, raise `SnapshotError("outcome_unknown", submission_possible=True)`.

```python
PROBE_SECONDS = 5.0
MUTATION_SECONDS = 600.0
MUTATION_TIMEOUT = httpx.Timeout(connect=5.0, pool=5.0, write=30.0, read=600.0)

client = httpx.AsyncClient(
    base_url=descriptor.base_url,
    headers=({"Authorization": f"Bearer {descriptor.bearer_token}"}
             if descriptor.bearer_token else {}),
    trust_env=False,
    follow_redirects=False,
    timeout=MUTATION_TIMEOUT,
    transport=transport,
)
```

- [x] Prove proxy isolation beyond MockTransport: use two owned numeric-loopback recording listeners, set HTTP_PROXY/ALL_PROXY to the decoy, leave NO_PROXY empty, and assert only the real endpoint sees traffic. Return a redirect to the decoy and assert no second request or credential forwarding. Mark this test `loopback_network`, not unrestricted `allow_network`.
- [x] Test timeout behavior using short injectable constants/events instead of waiting ten minutes. A response after the probe threshold but before the mutation deadline succeeds; expiration after dispatch reports possible submission; a connection failure before dispatch does not. Verify exactly one POST, response/client close, and no raw HTTP exception retained in long-lived service state.
- [x] Run `python -m pytest Tests/LLM_Management/test_snapshot_client.py -q`; record GREEN and commit: `feat: add bounded loopback llama.cpp snapshot transport`.

## Task 4: App-owned operation and subprocess lifecycle

**Files:** Create `snapshot_service.py`, `Tests/LLM_Management/test_snapshot_service.py`; modify `app.py`, `llm_management_events.py`, `server_lifecycle.py`, `snapshot_store.py`; extend `Tests/LLM_Management/test_snapshot_store.py`, `Tests/LLM_Management/test_server_lifecycle_resources.py` and `Tests/LLM_Management/test_gguf_server_sources.py`.

**Interfaces:** Produces `LlamaCppSnapshotService` APIs. Consumes store, client, admission and existing claim identity. Stores snapshot launch context separately from `claim._resource`, preserving managed-model lease teardown.

- [x] Write event-barrier RED tests for duplicate Save; Stop before Popen; Stop during staging; navigation away during POST; old response after replacement launch; unknown response followed by another mutation; old generation cleanup while a new server is alive; and double shutdown. Assert calls/retained records after the race, not exceptions swallowed by callbacks.
- [x] Make the critical keep=1 test cross the service/store boundary: start with one committed usable snapshot, invalidate launch compatibility immediately after the fake server acknowledgement but before publication, then assert no new record and no removed old record. A lower-level metadata validator alone cannot prove this rule.

The optional `commit_save(validate_publication=...)` predicate runs under the catalog lock immediately before publication, after hash/fsync work. The service supplies a thread-safe captured-generation/closing/current-claim predicate plus file revalidation. A false result or failed validation publishes/prunes nothing. This is publication admission: invalidation before the predicate wins and rejects; after admission an acknowledged local commit may finish, with stale UI updates still suppressed. Test invalidation during the file worker, not only before calling it, and keep lifecycle/store lock ordering acyclic.

- [x] Add optional keyword-only `env: Mapping[str, str] | None = None` and `private_umask: int | None = None` to `run_server_subprocess()`. Add Popen kwargs only when supplied and platform-supported, so existing runtime call sites remain unchanged. Snapshot-enabled llama.cpp prepares its working directory and frozen environment after resolving the actual GGUF source but before spawning. Pass owned slot flags once; record the exact claim and directory. Keep the existing model lease attached to that claim.
- [x] Compose the service once in `TldwCli`; construction itself performs no heavy I/O. Initialize its store in an app-owned off-thread setup task using the effective profile directory. Until ready, render an explicit preparing/unavailable state. Marshal lifecycle publish/stop callbacks to the app loop and call `attach()`/`server_stopped()` only for their captured claims. Preflight refuses an existing listener and readiness requires the current child still alive; never infer ownership from HTTP success alone.

The stable service may start with `store=None`; its idempotent `initialize(root_factory)` resolves the effective root and constructs the store off-thread in retained app-owned work. Shutdown settles that local setup. Tests may still inject a ready store through the existing constructor seam.

- [x] In `refresh()`, obtain `ReadinessObservation`, recheck the current claim, and pass it with the captured descriptor to `finalize_launch()`. Store the resulting same-generation descriptor and project only safe slot/status fields. Later differing build/model/context observations invalidate compatibility; they do not silently relabel saved files or another generation as matching.
- [x] Expose catalog-only pagination through `browse_catalog(offset=0, limit=50)`: retained off-thread store reads publish a safe `CatalogPage` to subscribers without HTTP. Validate page arguments through the store contract and reject stale concurrent browse results. Catalog mutations may reset to page zero. Test pagination, absence of HTTP, and newer-page precedence; widgets never access storage directly.
- [x] Implement the state transitions below. Operation admission occurs before returning from `start_*`, so rapid duplicate clicks cannot replace workers. Store file tasks remain strongly owned even when awaiting coroutines are cancelled; cleanup waits until their local handles close.

```text
idle -> preparing -> awaiting_ack -> validating -> publishing (Save) -> idle
idle -> staging_and_verifying (Restore) -> awaiting_ack -> idle
preparing/staging failure with no POST -> safe cleanup -> idle + error
awaiting_ack without a valid terminal response -> outcome_unknown
outcome_unknown -> acknowledged completion OR confirmed stop -> safe cleanup
any stale-generation completion -> original bookkeeping only; no new-generation paint
```

- [x] Save sequence: load keep count at admission, recheck eligibility, reserve, submit once, validate receipt/evidence/current claim, commit, prune, cleanup, refresh. Restore sequence: capture snapshot/slot IDs, stage/verify, recheck eligibility/current claim, submit once, validate receipt, cleanup, refresh even on acknowledged restore failure. Delete operates on the catalog and remains available when mutations are blocked. Serialize local file publication/deletion with the store lock, not a second app-wide lock.
- [x] For shutdown: reject new work, stop subscriptions/readiness retries, settle tracked local filesystem work, close HTTP clients, and record possibly submitted requests as unknown if acknowledgement is lost. Do not delete their working files while the child may still run; normal explicit Stop/confirmed process exit permits cleanup. Do not hold UI teardown for the 600-second network deadline. Both `_shutdown()` and `on_unmount()` may invoke the owner shutdown path; it must be idempotent.
- [x] Run `python -m pytest Tests/LLM_Management/test_snapshot_service.py Tests/LLM_Management/test_snapshot_store.py Tests/LLM_Management/test_server_lifecycle_resources.py Tests/LLM_Management/test_gguf_server_sources.py Tests/UI/test_llm_gguf_source_modes.py -q`. Record GREEN; verify snapshot-disabled command/Popen behavior is unchanged for llama.cpp and llamafile. Commit: `feat: coordinate snapshot operations with managed server lifecycle`.

## Task 5: Manual manager and canonical settings

**Files:** Create `Widgets/llamacpp_snapshot_manager.py`, `Tests/UI/test_llamacpp_snapshot_manager.py`, `Tests/UI/test_llamacpp_snapshot_settings.py`; modify `UI/LLM_Management_Window.py`, `UI/Screens/settings_screen.py`, `UI/Screens/settings_config_models.py` only if its category ownership record needs the new section; extend `Tests/UI/test_llm_deferred_views.py` and `Tests/UI/test_llm_screen_lab_adoption.py`.

**Interfaces:** Widget constructor `LlamaCppSnapshotManager(service: LlamaCppSnapshotService)`. Settings and launcher both call `save_snapshot_preferences()` off-thread; only the existing config module writes TOML. Views subscribe/unsubscribe without cancelling operations.

**Styles:** Use `css/features/_llm-management.tcss` for manager-specific rules and rebuild generated sheets with `python -m tldw_chatbook.css.build_css`; commit changed source and generated outputs, never hand-edit the bundle. Use `service.browse_catalog()` for paging, not store access. Preserve the existing six eager server views and five deferred Ollama/library views.

**Projection completion:** Extend `ManagerView` in `snapshot_models.py` with a defaulted `snapshot_compatibility: tuple[tuple[str, Literal["matching", "different", "unknown"]], ...] = ()` for the visible catalog page. `snapshot_service.py::view()` derives these bounded path-free statuses using the existing compatibility predicate and current valid/ready generation; absent or uncertain current evidence means unknown. The widget translates statuses to the spec's three labels, never reads private service descriptors or performs file checks. Add focused service projection tests in `Tests/LLM_Management/test_snapshot_service.py` for matching, different, unknown, and invalidated launch state before the UI tests. Mutation admission still revalidates independently.

**Preference concurrency:** Extend only `snapshot_settings.py` with `save_snapshot_preferences(value: SnapshotPreferences, *, expected: SnapshotPreferences | None = None) -> bool` and payload-free `SnapshotPreferencesConflict(ValueError)`. Without expected, preserve the existing API behavior. With expected, compare both strictly validated values (same missing-key defaults as load) through the config owner's existing `locked_snapshot_precondition`, not a separate read-then-write check; raise the fixed conflict on `result.conflict` and preserve existing successful no-op return semantics. F9 and launcher use their loaded pair as expected. Add real-config tests in `Tests/LLM_Management/test_snapshot_settings.py` for a competing update before locked apply, matching pair, missing defaults, no-op, and unchanged current values on conflict. F9 snapshot edits join its existing category Save/Revert and dirty-state handling; snapshot-only saves skip unrelated provider validation, while combined partial persistence is reported honestly.

**Explicit Details path:** To satisfy design section 3, add `ManagerView.storage_location: str | None = field(default=None, repr=False)`, derived from initialized `store.root` without additional I/O. This is a narrowly scoped exception to path-free display state: show the full read-only root only in user-expanded Details, never in default summary, notifications, logs, persisted snapshot metadata, or repr. Launch/model/credential paths remain excluded. Verify collapsed/expanded frame behavior; the widget never accesses the store.

- [x] Write production-shaped RED tests using `Tests/UI/app_factory.py::_build_test_app`, the real `LLMScreen` route, and `TldwCli.CSS_PATH`. Inject the real service with a recording transport/private store. Assert rendered frame text and action containment, not only `widget.render()` or computed styles.
- [x] Compose slots and snapshots as selectable tables with shared action rows. Use stable IDs: `snapshot-slots`, `snapshot-records`, `snapshot-save`, `snapshot-restore`, `snapshot-delete`, `snapshot-refresh`, `snapshot-retention`, `snapshot-operation-status`, `snapshot-disabled-reason`. Keep selected opaque IDs stable across refresh; do not select a different record silently when a confirmation is open.
- [x] Add compact controls for Enable snapshots and keep count in F9 Providers & Models and the launcher. Persist through the shared adapter; only update effective UI values after a successful config mutation. Label enable/disable changes as applying on next launch. Keep preferences editing out of legacy Tools Settings/sidebar files. If F9 has dirty snapshot preference fields when another surface updates the pair, detect the stale values before Save and ask the user to reload that draft rather than silently overwriting newer values.
- [x] Implement explicit actions and display stages:

```python
def retention_copy(keep_count: int) -> str:
    return f"Keeps the newest {keep_count} across all models"

def token_copy(tokens: int | None) -> str:
    return "Unknown" if tokens is None else str(tokens)
```

Save has no modal; Restore confirms the timestamp, selected destination, replacement and failure-clearing warning; Delete confirms permanent removal and size. Revalidate the selected IDs/claim after confirmation. Escape cancels the dialog only, never an already-submitted server operation. Prefer known-empty idle destinations; never automatically select a busy one.

- [x] Use widget-local single-letter shortcuts only while its non-text controls have focus; do not shadow global bindings or consume letters from launcher inputs. Tab/Shift-Tab, arrows and Enter must fully operate the manager even if no accelerator is available. Advertise only implemented actions. Use existing design tokens; keep secondary fields in details at 80x24.
- [x] Use a UI timer solely for elapsed-time rendering; never create a continuous server polling loop. Refresh on entry/re-entry, readiness, acknowledgement and explicit Refresh. Show observation time, partial cleanup warnings, unknown operation recovery, compatibility reasons and stored/residual bytes. Screen detach removes subscribers/timers, not app-owned file/HTTP tasks; old callbacks check current attachment before painting.
- [x] Run `python -m pytest Tests/UI/test_llamacpp_snapshot_manager.py Tests/UI/test_llamacpp_snapshot_settings.py Tests/UI/test_llm_deferred_views.py Tests/UI/test_llm_screen_lab_adoption.py Tests/UI/test_llm_gguf_source_modes.py -q`. Exercise both 80x24 and 140x45; assert the cross-model count and primary actions actually paint. Commit: `feat: expose manual prompt-cache snapshots in Models and Settings`.

## Task 6: Real persistence/reuse evidence and feature closeout

**Files:** Create `Tests/LLM_Management/test_snapshot_live.py`, `Docs/LLMs/llamacpp-snapshots.md`; update the task's final notes only after implementation and verification. Create `Docs/superpowers/reviews/2026-09-04-llamacpp-slot-snapshots-verification.md` for sanitized evidence and explicit remaining gaps. Replace stale pre-implementation status prose in this plan, the linked design, and the Backlog task with the verified implementation state; keep missing live evidence and the In Progress status explicit.

**Interfaces:** Uses the real launcher/service/store/client/widget; no additional production API or cache-routing change. The live test accepts existing local model assets only and never downloads models or contacts a cloud provider.

**Narrow admission completion:** Recognize pinned `--cache-ram` / `-cram` as one-value performance-only options and `LLAMA_ARG_CACHE_RAM` as the corresponding environment option in `snapshot_admission.py`, matching the existing ignored-performance option policy. This RAM prompt-cache budget is not serialized compatibility state. Add admission regressions proving these options no longer disable management, do not change compatibility identity, and preserve owned-slot conflict detection after their consumed value. The live harness explicitly selects zero; do not inject or change the ordinary launcher's default.

**Guarded inventory:** Review and update only the new `snapshot_store.py` persistent-sink entry and corresponding count in `Docs/security/production-diagnostic-inventory.json`; run the inventory checker and a focused row comparison. Controller's read-only check found unchanged pre-existing owner drift in `DB/Client_Media_DB_v2.py` and `UI/Screens/library_screen.py`; preserve unrelated rows and report remaining baseline drift rather than bulk rewriting the inventory.

- [x] Add an exact opt-in gate, checked before any process/network action:

```python
import os
from pathlib import Path
import pytest

def live_inputs() -> dict[str, Path]:
    if os.environ.get("TLDW_LLAMA_SNAPSHOT_LIVE") != "1":
        pytest.skip("Set TLDW_LLAMA_SNAPSHOT_LIVE=1 with local server/model/media assets")
    names = ("SERVER", "MODEL", "MMPROJ", "IMAGE_A", "IMAGE_B")
    result = {}
    for name in names:
        raw = os.environ.get(f"TLDW_LLAMA_SNAPSHOT_{name}")
        if not raw or not Path(raw).is_file():
            pytest.fail(f"Missing local snapshot live input: {name}")
        result[name] = Path(raw)
    return result
```

Keep the test inside `Tests/` so root conftest isolates config and installs the network guard. Use `loopback_network` and a justified per-test timeout, not the paid `live` marker or optional/slow gates that skip before this contract. Use a fresh owned port, `tmp_path` data/config, no user history, and fixed benign prompts. Stop/reap all owned children in `finally`; retain only sanitized evidence, not snapshot binaries or prompt/media content in the repo.

- [ ] Prove text and image save/restart/restore with the same executable/model/projector/settings. Send matching OpenAI-compatible requests without `id_slot`, comparing `timings.cache_n` (or the pinned server's verified equivalent cache counter) against a cold control. Record actual reported field names; missing counters fail the evidence gate, not default to zero. A different-image control may reuse preceding text, but must not count the mismatched media prefix as reused. Test requests are ordinary HTTP clients, not production chat-routing modifications.

Use a native in-memory A→B control to measure the different-media prefix boundary, not a guessed text-template token count. Pin one slot, disable the server's separate RAM prompt cache, and require distinct SHA-256 image byte digests. With identical ordinary requests, compare cold A, native A→B, restored A→A, and separately restored A→B. Restored same-A cache reuse must exceed both cold A and native A→B; restored different-B reuse must not exceed native A→B. Validate matching prompt totals and strict counters. This oracle is grounded in the pinned server's byte-hash media IDs and whole-chunk prefix comparison; unsupported counters/configuration or absent live assets leave the evidence gate open. Record the actual controls, not just a pass label.
- [ ] Verify the production Models action path separately: launch through Chatbook, populate a slot, Save, stop/start, Restore, and return to the normal chat request path. This proves the service is reachable rather than only testing an alternate harness entry point. Add audio coverage only when claiming tested audio support. If the chosen model/build cannot demonstrate reuse, leave AC5 open and record the limitation.
- [x] Document prerequisites, enable-next-launch semantics, timestamp naming, global per-profile count, count-versus-bytes, confirmations, matching-config restrictions, required SWA configuration when applicable, and recovery after unknown outcome. Show this command with the five input variables pointing at user-selected existing files:

```bash
TLDW_LLAMA_SNAPSHOT_LIVE=1 python -m pytest Tests/LLM_Management/test_snapshot_live.py -q
```

- [x] Run targeted automated tests across the exact new modules and the existing regression files listed in Tasks 1–5. Do not substitute a broad `-k snapshot` selection that misses launcher regressions. Run Python compilation, a scoped linter/formatter check using the execution environment's installed tools, and `git diff --check`. If lint tools are absent, report that missing verification rather than adding dependencies or claiming lint success. Do not reformat unrelated legacy modules.
- [x] Self-review the complete feature diff against all 11 ACs. Record the commands, exit codes, RED/GREEN evidence, platform coverage and actual live counters. Keep fixture-only tests explicitly separate from real-server evidence. Check each AC only when its evidence exists; if live/hardware/platform evidence is missing, report it and leave the task In Progress.
- [ ] Commit docs/tests with `test: verify llama.cpp snapshot persistence and cache reuse`. After all DoD requirements are met, add concise Implementation Notes linking ADR-119 and the evidence report, then use `backlog task edit 31552 -s Done --plain` and inspect the resulting file. Recheck task/ADR allocation before integration. Do not merge or push as part of this plan without authorization.

## Coverage and review checkpoints

| Contract | Implemented by | Evidence |
| --- | --- | --- |
| Manual save/restore + timestamp names (AC1) | Tasks 2, 4, 5 | Store and real-route Pilot verified; real-model restart remains AC5 gate |
| Profile-global newest-N retention (AC2) | Tasks 1, 2, 4 | Multi-model/clock/keep=1/failed-save tests |
| Claim/readiness/privacy/compatibility (AC3) | Tasks 1–4 | Admission, process race, private-path and client tests |
| Honest keyboard UI (AC4) | Task 5 | Production CSS at two sizes; confirmations/errors |
| Real image reuse (AC5) | Task 6 | Harness implemented; measured live controls missing, AC5 open |
| Unknown evidence cannot evict useful files (AC6) | Tasks 1, 2, 4 | Cross-layer invalidation before publication |
| Integrity before Restore POST (AC7) | Tasks 2–4 | Same-length corruption, zero recorded POSTs |
| Local-only management (AC8) | Tasks 1, 3 | Numeric addresses, decoy proxy, redirect recording |
| Working-file cleanup (AC9) | Tasks 2, 4 | Repeated restore, acknowledged failures, unknown writer |
| Distinct deadlines and elapsed status (AC10) | Tasks 3–5 | Injected deadline tests and visible pending state |
| Cross-model retention copy (AC11) | Task 5 | Rendered count and narrow layout tests |

Review after each unit before advancing. Task 1's effective-configuration table and Task 4's claim/worker interleavings are the highest-risk review points. Implementation must not weaken a blocked/unknown state simply to make the happy-path test pass.

## Plan self-review result

Integrated-review follow-through: preference admission failures are payload-free
before reservation and recover safely in Models; initialization/entry performs
retained off-thread safe reconciliation without claiming writer termination;
bounded nested JSON preserves counter/tombstone policies and cannot bypass Stop
teardown. Details uses an absolute observation time, and the aggregate-deadline
regression uses a deterministic budget with a verified per-request-reset mutant.
Final affected run: 156 passed, 1 existing warning; no broad rerun or new runtime
evidence. See the linked verification record for exact boundaries and commands.

- All nine spec sections and all 11 task criteria have implementation and verification owners above.
- All new public type and method names are defined in the shared interface section; task boundaries do not introduce a second client, config writer, or process owner.
- The six approved review amendments are included in code steps and regression oracles, not only documentation.
- Planning originally ended before implementation. Units 1–5 are now implemented and reviewed, and Task6 records targeted automated evidence separately from the still-missing real-server counters. Missing eligible local assets and live AC5 remain execution gates, not claimed successes.

## Source anchors

- [Pinned upstream API and options](https://github.com/ggml-org/llama.cpp/blob/427291b5b34cd914a31b3fd3b61a68f6184f4b9f/tools/server/README.md)
- [Pinned upstream argument parser](https://github.com/ggml-org/llama.cpp/blob/427291b5b34cd914a31b3fd3b61a68f6184f4b9f/common/arg.cpp)
- [Testing evidence lessons](../../../backlog/docs/lessons-testing-evidence.md): real production hierarchy/CSS, explicit live gate, mutation-tested guards and cancellation barriers.
- [Live verification lessons](../../../backlog/docs/lessons-live-verification.md): scratch profile isolation and keeping app-importing probes within `Tests/`.
- [Backlog hygiene lessons](../../../backlog/docs/lessons-backlog-hygiene.md): CLI section preservation, high-ID verification and collision checks.
