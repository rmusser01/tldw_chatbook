# External Parakeet Bundles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users run catalog-known Parakeet v2/v3 INT8/F32 models directly from user-owned directories while Chatbook supplies and leases the managed Silero VAD dependency.

**Architecture:** Add one descriptor-backed external-root verifier and one app-owned source service that owns exact source preferences, in-process verification retention, VAD readiness, and optional managed copy. Extend the existing artifact/executor contract only enough to lease a managed dependency beside a verified local root, then route First Run, Lab Models, Library, and Console through that same service without introducing downloads in transcription paths.

**Tech Stack:** Python 3.11+, Textual 8.x workers/messages, existing `ModelArtifactService` and `ArtifactAcquisitionService`, existing STT executor/coordinator, TOML config helpers, pytest, Ruff.

**Design:** `Docs/superpowers/specs/2026-08-09-task-598-external-parakeet-bundles-design.md`

**Decision:** `backlog/decisions/050-external-parakeet-roots-with-managed-vad.md`

**ADR required:** yes

**ADR path:** `backlog/decisions/050-external-parakeet-roots-with-managed-vad.md`

**Reason:** ADR-050 defines the mixed external-root/managed-dependency ownership, identity, provenance, configuration, and deletion boundaries implemented here.

**Local verification constraint:** Run only the focused commands listed below. Do not run the unrelated full suite. Use the repository virtualenv's absolute Python path and isolate every live profile with scratch `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, and `TLDW_CONFIG_PATH` as required by `backlog/docs/lessons-testing-evidence.md` and `backlog/docs/lessons-live-verification.md`.

---

## File and responsibility map

- Create `tldw_chatbook/STT/parakeet_external.py`: path-safe descriptor verification, coalesced hashing, cancellation, metadata snapshots, and bounded process-lifetime retention.
- Create `tldw_chatbook/STT/parakeet_sources.py`: exact external-source records, preference resolution, atomic config commits, VAD readiness, batch scopes, optional managed copy, and deletion guard.
- Create `tldw_chatbook/UI/Screens/model_external_view.py`: user-owned external-source inventory and its Change/Stop/Copy actions.
- Create `Tests/STT/test_parakeet_external.py`: tiny-descriptor verifier tests.
- Create `Tests/STT/test_parakeet_sources.py`: source preference, migration, VAD atomicity, cache lifetime, copy, and path-private failure tests.
- Create `Tests/UI/test_model_external_view.py`: external Lab Models section behavior.
- Create `.github/scripts/task598_external_parakeet_evidence.py`: supervised,
  path-private native platform probe using the production acquisition, source,
  coordinator, executor, and ONNX CPU boundaries.
- Create `.github/workflows/task-598-platform-evidence.yml`: label-gated bootstrap
  and later manual v2/v3 INT8 evidence matrix for the four remaining targets.
- Create `Tests/CI/test_task598_external_parakeet_evidence.py`: workflow-shape,
  timeout-result, validation, and path-privacy checks for the evidence seam.
- Modify `tldw_chatbook/Model_Artifacts/service.py` and `tldw_chatbook/Model_Artifacts/__init__.py`: exact dependency-only shared leases; no fake root readiness.
- Modify `tldw_chatbook/Local_Ingestion/parakeet_v2_artifact.py`: VAD-only catalog/preflight/provision helpers with `activate=False`.
- Modify `tldw_chatbook/STT/executor.py`, `tldw_chatbook/STT/executor_worker.py`, and `tldw_chatbook/STT/dispatch_coordinator.py`: carry and retain exact managed dependency references alongside a verified local root.
- Modify `tldw_chatbook/STT/parakeet_dispatch.py` and `tldw_chatbook/Local_Ingestion/transcription_service.py`: consume the shared source service and preserve download-free dispatch.
- Modify `tldw_chatbook/app.py`: own one source service, bind it to Library and Console, retain/release Library scopes, and close it before executor shutdown.
- Modify `tldw_chatbook/UI/Screens/model_browser_state.py`, `tldw_chatbook/UI/Screens/model_installed_view.py`, `tldw_chatbook/Widgets/ModelArtifacts/activation_controls.py`, `tldw_chatbook/UI/Screens/model_curated_view.py`, `tldw_chatbook/UI/Screens/llm_screen.py`, and `tldw_chatbook/UI/LLM_Management_Window.py`: correct managed inventory semantics and add the Lab external-source flow.
- Modify `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py` and `tldw_chatbook/UI/Wizards/first_run_speech_step_state.py`: external selection, VAD-only consent, and atomic first-run commit.
- Modify `tldw_chatbook/Library/ingest_capabilities.py` and `tldw_chatbook/UI/Screens/library_screen.py`: directory picker and pre-enqueue validation/consent for job-scoped overrides.
- Modify the focused test files named in each task; do not broaden unrelated fixtures or format untouched files.

### Task 1: Verify external Parakeet roots without trusting or parsing them

**Files:**
- Create: `tldw_chatbook/STT/parakeet_external.py`
- Create: `Tests/STT/test_parakeet_external.py`
- Reference: `tldw_chatbook/Local_Ingestion/parakeet_v2_artifact.py:145-363`
- Reference: `tldw_chatbook/STT/executor.py:75-190`

- [ ] **Step 1: Write the tiny-descriptor RED tests**

Use injected descriptors whose files are only a few bytes. Cover one v2-like INT8 bundle, one v3-like F32 bundle with an external-data file, extra ignored files, missing/wrong-size/wrong-hash files, a required symlink, a FIFO or directory node, containment rejection, mutation after verification, and `repr()`/error strings that contain no selected path.

```python
def test_verify_external_root_hashes_every_declared_regular_file(tmp_path):
    descriptor = tiny_parakeet_descriptor(
        files={"encoder.onnx": b"enc", "encoder.onnx.data": b"weights"}
    )
    materialize_descriptor(tmp_path, descriptor)

    verified = ExternalParakeetVerifier().verify(descriptor, tmp_path)

    assert verified.reference == descriptor.reference
    assert {path.name for path in verified.snapshot.paths} == {
        "encoder.onnx",
        "encoder.onnx.data",
    }
    assert str(tmp_path) not in repr(verified)
```

- [ ] **Step 2: Run the verifier tests and confirm RED**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/STT/test_parakeet_external.py -q`

Expected: FAIL during collection because `tldw_chatbook.STT.parakeet_external` does not exist.

- [ ] **Step 3: Implement the minimal verifier and stable errors**

Implement only the catalog-declared-file boundary. Do not parse ONNX and do not scan or hash unrelated files.

```python
class ExternalParakeetErrorCode(str, Enum):
    UNSUPPORTED = "unsupported_descriptor"
    MISSING = "missing_file"
    IRREGULAR = "irregular_file"
    CHANGED = "changed_file"
    CORRUPT = "corrupt_file"
    CANCELLED = "cancelled"


@dataclass(frozen=True, repr=False)
class VerifiedExternalParakeet:
    reference: ArtifactRef
    directory: Path = field(repr=False)
    snapshot: LocalSourceSnapshot = field(repr=False)

    def __repr__(self) -> str:
        return f"VerifiedExternalParakeet(reference={self.reference!r})"
```

For each `ArtifactFile`, form `directory / filename`, resolve the parent once through the existing path-validation boundary, use `lstat()` to reject symlinks and non-regular nodes, prove the resolved file remains under the selected directory, compare `st_size`, hash in fixed chunks while polling cancellation, then take the existing path-private `LocalSourceSnapshot`. Re-stat after hashing and reject if device/inode/mode/size/mtime/ctime changed.

- [ ] **Step 4: Mutation-test the guards and make the file GREEN**

Temporarily invert each of the following guards one at a time and prove the named test fails: containment, symlink rejection, size, SHA-256, and post-hash metadata equality. Restore each guard.

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/STT/test_parakeet_external.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add tldw_chatbook/STT/parakeet_external.py Tests/STT/test_parakeet_external.py
git commit -m "feat(stt): verify external Parakeet roots"
```

### Task 2: Coalesce verification and resolve exact source preferences

**Files:**
- Create: `tldw_chatbook/STT/parakeet_sources.py`
- Modify: `tldw_chatbook/STT/parakeet_external.py`
- Modify: `tldw_chatbook/config.py:175-260`
- Modify: `tldw_chatbook/Local_Ingestion/parakeet_v2_artifact.py:364-640`
- Create: `Tests/STT/test_parakeet_sources.py`
- Modify: `Tests/STT/test_parakeet_external.py`
- Modify: `Tests/Local_Ingestion/test_parakeet_v2_artifact.py`

- [ ] **Step 1: Write RED tests for coalescing, retention, preferences, and VAD-only acquisition**

Pin these outcomes:

- two concurrent waiters for the same descriptor and unchanged snapshot perform one hash pass;
- cancelling one waiter does not cancel another; cancelling the last waiter stops hashing;
- a configured selection survives cache pruning, a job-scoped entry survives while its scope is live, `release_scope()` removes only its ownership, a metadata change forces rehash, and `close()` cancels work;
- records are keyed independently as v2/v3 × INT8/F32 and repeat the expected model/precision values;
- preferring managed creates an exact preference even with no external directory, preserves a remembered directory when present, and Stop using removes the directory while retaining an existing managed preference;
- per-job override wins; preferred external and preferred managed are authoritative; a remembered non-preferred external record is ineligible; active managed and legacy are consulted only with no preference;
- an invalid explicit/preferred source fails without fallback and without a path in the error;
- the legacy singular path is considered only for v2 INT8 with no exact preference and is persisted only after descriptor verification plus VAD readiness;
- `prepare_config_commit()` is write-free, ordinary commit writes once, and `accept_committed()` refreshes configured cache ownership only when the persisted record matches its prepared patch;
- `run_parakeet_vad_preflight()` reports only the VAD descriptor and `run_parakeet_vad_provision()` passes `activate=False`.

```python
def test_preferred_managed_does_not_try_remembered_external(source_service):
    source_service.seed_record(V2_INT8, "/external", preference="managed")
    source_service.seed_active_managed(V2_INT8, "/managed")

    dispatch = source_service.resolve(V2_INT8)

    assert dispatch.source_kind is ParakeetSourceKind.MANAGED
    assert source_service.verifier.calls == []
```

- [ ] **Step 2: Run the focused RED tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/STT/test_parakeet_external.py Tests/STT/test_parakeet_sources.py Tests/Local_Ingestion/test_parakeet_v2_artifact.py -q`

Expected: FAIL for missing cache/source-service/VAD-only APIs.

- [ ] **Step 3: Add the bounded verifier cache and exact config model**

Use a small owned `ThreadPoolExecutor` plus one shared future per `(ArtifactRef, metadata_token)`. The in-flight entry retains each waiter's progress callback and fans byte updates out without letting one broken callback fail verification. A caller polls its cancellation token while waiting and unregisters only itself; the hash future receives a shared stop event that is set only after the final waiter leaves. Cache ownership is explicit, not a TTL:

```python
class ExternalParakeetVerifier:
    def verify(
        self,
        descriptor: ArtifactDescriptor,
        directory: Path,
        *,
        owner: tuple[Literal["configured", "scope"], str] | None = None,
        cancelled: Callable[[], bool] = lambda: False,
        progress: Callable[[int, int], None] | None = None,
    ) -> VerifiedExternalParakeet: ...

    def set_configured_owners(self, owners: Mapping[str, tuple[ArtifactRef, Path]]) -> None: ...
    def release_scope(self, scope_id: str) -> None: ...
    def close(self) -> None: ...
```

Persist one nested record per stable key (`v2_int8`, `v2_f32`, `v3_int8`, `v3_f32`) under `transcription.parakeet_external_sources`. The directory is optional so an explicit managed preference can exist before any external directory has been selected; preferring managed preserves an existing directory, and the External view lists only records whose directory is present. An external preference without a directory is invalid, and an entry with neither directory nor preference is omitted. Parse strictly: a record whose repeated `model_id` or `precision` disagrees with its key is ignored as invalid. Use injected `read_setting`/`write_settings` callables in tests and `get_cli_setting`/one `save_settings_to_cli_config` batched write in production.

- [ ] **Step 4: Implement the shared source service and VAD-only helpers**

```python
class ParakeetSourcePreference(str, Enum):
    EXTERNAL = "external"
    MANAGED = "managed"


@dataclass(frozen=True)
class ParakeetSourceRecord:
    model_id: str
    precision: str
    directory: Path | None = field(default=None, repr=False)
    preferred_source: ParakeetSourcePreference | None = None


@dataclass(frozen=True, repr=False)
class PreparedExternalSelection:
    key: ParakeetSourceKey
    verified: VerifiedExternalParakeet = field(repr=False)


@dataclass(frozen=True, repr=False)
class ExternalSourceConfigCommit:
    prepared: PreparedExternalSelection = field(repr=False)
    section_values: Mapping[str, Mapping[str, object]] = field(repr=False)


class ParakeetSourceService:
    def prepare_external(self, key, directory, *, owner, cancelled, progress) -> PreparedExternalSelection: ...
    def retain_prepared(self, scope_id: str, prepared: PreparedExternalSelection) -> None: ...
    def prepare_config_commit(self, prepared: PreparedExternalSelection) -> ExternalSourceConfigCommit: ...
    def accept_committed(self, commit: ExternalSourceConfigCommit) -> None: ...
    def commit_external(self, prepared: PreparedExternalSelection) -> None: ...
    def prefer_managed(self, key) -> None: ...
    def stop_using_external(self, key) -> None: ...
    def resolve(self, key, *, override=None, scope_id=None) -> ParakeetDispatch: ...
    def release_scope(self, scope_id: str) -> None: ...
    def release_scopes_except(self, active_scope_ids: Collection[str]) -> None: ...
    def close(self) -> None: ...
```

`prepare_config_commit()` must recheck the prepared root snapshot and exact managed VAD and return the full `{"transcription": {"parakeet_external_sources": ...}}` section mutation without writing. `commit_external()` is the ordinary one-write wrapper: prepare, pass `section_values` to one batched config write, then call `accept_committed()`. The wizard merges the returned nested `transcription` values with its speech defaults in its own existing atomic write, then calls `accept_committed()`; that method reads back/matches the committed record before refreshing configured verifier ownership, and performs no second write. It must never download. `retain_prepared()` must first prove the prepared key/path/snapshot still match the request it is being adopted for. Add `ParakeetVadCatalog`, `run_parakeet_vad_preflight`, and `run_parakeet_vad_provision`; the provision helper roots acquisition at `parakeet_vad_reference()` and passes `activate=False`.

- [ ] **Step 5: Run focused tests, mutation checks, and commit**

Mutation checks: swap preferred-source order, let remembered external fall through, persist before VAD readiness, and retain a released scope; each corresponding test must fail.

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/STT/test_parakeet_external.py Tests/STT/test_parakeet_sources.py Tests/Local_Ingestion/test_parakeet_v2_artifact.py -q`

Expected: PASS.

```bash
git add tldw_chatbook/STT/parakeet_external.py tldw_chatbook/STT/parakeet_sources.py tldw_chatbook/config.py tldw_chatbook/Local_Ingestion/parakeet_v2_artifact.py Tests/STT/test_parakeet_external.py Tests/STT/test_parakeet_sources.py Tests/Local_Ingestion/test_parakeet_v2_artifact.py
git commit -m "feat(stt): resolve persistent external Parakeet sources"
```

### Task 3: Lease an exact managed VAD without fake readiness

**Files:**
- Modify: `tldw_chatbook/Model_Artifacts/service.py:891-985,1963-2091`
- Modify: `tldw_chatbook/Model_Artifacts/__init__.py`
- Modify: `Tests/Model_Artifacts/test_service.py`

- [ ] **Step 1: Write RED service tests**

Cover one exact dependency acquisition, multiple exact dependencies in canonical order, missing/wrong-role/corrupt dependencies, mutation blocked by a live shared lease, close/context-manager idempotence, rollback when verification fails, and no readiness/active files created.

```python
def test_acquire_dependencies_verifies_and_leases_without_readiness(store, vad):
    install_dependency(store, vad)

    with store.acquire_dependencies((vad.reference,)) as leased:
        assert leased.handle.paths == ((vad.reference, store.artifact_path(vad.reference)),)
        with pytest.raises(ArtifactInUseError):
            store.delete(vad.reference)

    assert not store.readiness_path(vad.reference).exists()
```

- [ ] **Step 2: Run the exact RED nodes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_service.py -q`

Expected: FAIL because `acquire_dependencies` and its handle types do not exist.

- [ ] **Step 3: Implement the dependency-only lease surface**

```python
@dataclass(frozen=True)
class ArtifactDependencyHandle:
    references: tuple[ArtifactRef, ...]
    paths: tuple[tuple[ArtifactRef, Path], ...]

    @property
    def lease_keys(self) -> tuple[ArtifactLeaseKey, ...]:
        return tuple(ref.lease_key() for ref in self.references)


def acquire_dependencies(
    self, references: tuple[ArtifactRef, ...]
) -> LeasedArtifactDependencyHandle:
    # normalize/sort/uniquify, acquire one shared lease set, then call
    # _verify_installed(ref, ArtifactRole.DEPENDENCY) while leases are held
```

Do not call `activate()`, `_write_readiness()`, or `_write_active()`. Export the two public handle types.

- [ ] **Step 4: Mutation-test lease and role guards, then run GREEN**

Remove shared-lease retention and change expected role to ROOT one at a time; the live-deletion and wrong-role tests must fail. Restore.

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_service.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add tldw_chatbook/Model_Artifacts/service.py tldw_chatbook/Model_Artifacts/__init__.py Tests/Model_Artifacts/test_service.py
git commit -m "feat(artifacts): lease exact managed dependencies"
```

### Task 4: Carry the mixed external-root/managed-VAD closure through the executor

**Files:**
- Modify: `tldw_chatbook/STT/executor.py:193-265,490-550`
- Modify: `tldw_chatbook/STT/executor_worker.py:35-70,208-310,417-535`
- Modify: `tldw_chatbook/STT/dispatch_coordinator.py:221-290,500-550`
- Modify: `tldw_chatbook/STT/parakeet_dispatch.py`
- Modify: `Tests/STT/test_local_stt_executor.py`
- Modify: `Tests/STT/test_dispatch_coordinator.py`
- Modify: `Tests/STT/test_parakeet_dispatch.py`
- Modify: `Tests/STT/test_parakeet_onnx.py`

- [ ] **Step 1: Write RED request, coordinator, worker, and provenance tests**

Pin these contracts:

- `local_source + managed_dependency_refs + managed_store_root` is valid;
- `local_source + managed root artifact ref` remains invalid;
- dependency refs without a store root are invalid;
- CPU retry preserves dependency refs;
- the coordinator passes dependency refs unchanged;
- the worker acquires the VAD before native load, supplies its directory, retains the lease across resident reuse, closes it on recycle/shutdown, and recycles if dependency identity changes;
- external `ModelIdentity` carries the catalog reference's exact revision alongside model ID, precision, and local snapshot token, while result provenance still keeps `artifact_root is None`;
- a file metadata change before initial load or reuse rejects/recycles;
- external provenance has `artifact_root is None`, the exact VAD lease key in `artifact_dependencies`, and no path.

```python
request = ExecutorRequest(
    identity=external_identity,
    source=source,
    local_source=model_snapshot,
    managed_store_root=store.root,
    managed_dependency_refs=(vad_ref.as_tuple(),),
)
assert request.managed_artifact_ref is None
```

- [ ] **Step 2: Run the focused executor RED suite**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/STT/test_local_stt_executor.py Tests/STT/test_dispatch_coordinator.py Tests/STT/test_parakeet_dispatch.py Tests/STT/test_parakeet_onnx.py -q`

Expected: FAIL on the absent dependency-reference request field and mixed-load behavior.

- [ ] **Step 3: Extend the public request and coordinator contract minimally**

Add one immutable field only:

```python
managed_dependency_refs: tuple[tuple[str, str, str], ...] = ()
```

Require a managed store root when either `managed_artifact_ref` or dependency refs exist. Preserve the existing root-ref/local-source exclusion, but permit local source with dependency refs. Thread the field through `LocalSTTExecutor.submit`, CPU retry via `dataclasses.replace`, and coordinator submit/buffer paths.

- [ ] **Step 4: Retain both source classes in `_ResidentRuntime`**

Acquire a full `LeasedArtifactHandle` only for managed roots. For an external root, call `ModelArtifactService.acquire_dependencies()` and retain that leased handle in `_ResidentRuntime` until recycle/close. Include the canonical dependency refs in resident matching, pass the leased VAD path and lease identities to `_parakeet_provider`, and validate the local snapshot immediately before load and reuse.

Keep provider provenance unchanged for managed roots. Build external `ModelIdentity.root_revision` from the trusted descriptor reference (not filesystem input), then for external results pass `artifact_root=None` and only dependency lease keys. Never serialize the external directory or turn that identity revision into a managed-root provenance claim.

- [ ] **Step 5: Mutation-test ownership and run GREEN**

Temporarily close the dependency handle immediately after load, omit refs from reuse identity, and set `artifact_root` from the external directory; the lease, recycle, and provenance tests must fail. Restore.

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/STT/test_local_stt_executor.py Tests/STT/test_dispatch_coordinator.py Tests/STT/test_parakeet_dispatch.py Tests/STT/test_parakeet_onnx.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add tldw_chatbook/STT/executor.py tldw_chatbook/STT/executor_worker.py tldw_chatbook/STT/dispatch_coordinator.py tldw_chatbook/STT/parakeet_dispatch.py Tests/STT/test_local_stt_executor.py Tests/STT/test_dispatch_coordinator.py Tests/STT/test_parakeet_dispatch.py Tests/STT/test_parakeet_onnx.py
git commit -m "feat(stt): run external Parakeet with managed VAD"
```

### Task 5: Make one app-owned source service authoritative for Library and Console

**Files:**
- Modify: `tldw_chatbook/app.py:2640-2885,2990-3180,3566-3625,9250-9490`
- Modify: `tldw_chatbook/Local_Ingestion/transcription_service.py:4105-4290`
- Modify: `Tests/App/test_submit_library_ingest_job.py`
- Modify: `Tests/STT/test_transcription_service_facade.py`
- Modify: `Tests/Local_Ingestion/test_transcription_service_parakeet_buffer_wav.py`

- [ ] **Step 1: Write RED app/facade integration tests**

Prove:

- app lazily constructs exactly one `ParakeetSourceService` and passes it to both Library dispatch and the Console `TranscriptionService` factory;
- `_build_local_stt_dispatch` resolves explicit job override through the service and passes the verified root plus exact VAD dependency refs;
- a preferred-source failure becomes a stable structured local-STT failure and never calls a downloader or a fallback resolver;
- repeated items in one folder batch reuse one verified snapshot;
- registry changes release a batch scope only after every job in that batch is terminal, cancellation releases it, and headless jobs without a prevalidated scope fall back to batch/job ID ownership;
- a newly retained pre-enqueue scope is not released by an unrelated registry mutation before its first job exists, while an abandoned submission explicitly releases it;
- shutdown closes the source service before the coordinator/executor;
- Console buffer/dictation paths resolve through the injected service and remain download-free.

Use the registry's documented read-only listener contract rather than adding callbacks to every terminal mutation:

```python
def _sync_parakeet_source_scopes(self) -> None:
    active = {
        self._parakeet_scope_id_for_job(job)
        for job in self.library_ingest_jobs.jobs()
        if job.state in {
            IngestJobState.QUEUED,
            IngestJobState.PARSING,
            IngestJobState.WRITING,
        }
    }
    self._parakeet_source_service.release_scopes_except(active)
```

`_parakeet_scope_id_for_job()` first reads the internal, path-free `transcription_external_scope_id` captured in the job's audio/video options, then falls back to `job.batch_id or job.job_id` for headless/unprepared submissions. This lets one pre-enqueue Library verification owner cover either a single job or every item in a folder batch without predicting a registry job ID.

`release_scopes_except()` tracks scopes that the registry has actually observed live and releases only an observed scope that later disappears from the active set. It must not sweep a newly retained pre-enqueue scope that has not produced a job yet; the submitting UI explicitly releases that scope if validation is cancelled or submission raises before registry adoption.

- [ ] **Step 2: Run the focused RED tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/App/test_submit_library_ingest_job.py Tests/STT/test_transcription_service_facade.py Tests/Local_Ingestion/test_transcription_service_parakeet_buffer_wav.py -q`

Expected: FAIL because the app/facade still call `resolve_parakeet_dispatch` directly and do not own scopes.

- [ ] **Step 3: Add lazy ownership, resolution, and scope cleanup**

Add `_ensure_parakeet_source_service()` beside `_ensure_local_stt_executor()`. Register one read-only `LibraryIngestJobRegistry` listener when the service is created; the listener computes live local scope IDs and releases stale verifier ownership. Remove the listener and close the service during ingest shutdown before the executor is closed.

Pass `scope_id=self._parakeet_scope_id_for_job(job)` and the per-job override into source resolution. Add `managed_dependency_refs` to the dispatch dict and coordinator submission. Do not perform VAD acquisition here; missing VAD is a path-safe failure.

- [ ] **Step 4: Inject the service into the Console facade**

Add an optional `parakeet_source_service` constructor dependency to `TranscriptionService`. The app factory supplies its singleton. Direct test/legacy construction may create a small download-free default service, but no facade or transcription method may call acquisition.

- [ ] **Step 5: Mutation-test batch release and run GREEN**

Release on the first terminal sibling and omit cancel/shutdown cleanup one at a time; the lifecycle tests must fail. Restore.

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/App/test_submit_library_ingest_job.py Tests/STT/test_transcription_service_facade.py Tests/Local_Ingestion/test_transcription_service_parakeet_buffer_wav.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

```bash
git add tldw_chatbook/app.py tldw_chatbook/Local_Ingestion/transcription_service.py Tests/App/test_submit_library_ingest_job.py Tests/STT/test_transcription_service_facade.py Tests/Local_Ingestion/test_transcription_service_parakeet_buffer_wav.py
git commit -m "feat(stt): share external Parakeet source ownership"
```

### Task 6: Preserve managed lifecycle semantics and add optional copy

**Files:**
- Modify: `tldw_chatbook/STT/parakeet_sources.py`
- Modify: `tldw_chatbook/Model_Artifacts/service.py:2932-3075,3495-3645`
- Modify: `tldw_chatbook/UI/Screens/model_browser_state.py:195-255`
- Modify: `tldw_chatbook/UI/Screens/model_installed_view.py:250-390`
- Modify: `tldw_chatbook/Widgets/ModelArtifacts/activation_controls.py`
- Modify: `Tests/STT/test_parakeet_sources.py`
- Modify: `Tests/Model_Artifacts/test_service.py`
- Modify: `Tests/UI/test_model_browser_state.py`
- Modify: `Tests/UI/test_model_installed_view.py`
- Modify: `Tests/UI/test_model_artifact_widgets.py`

- [ ] **Step 1: Write RED lifecycle and copy tests**

Cover:

- dependency rows say `Managed dependency` and never expose Activate;
- a valid ROOT manifest with no readiness says `Installed · activation required` and permits Activate;
- activation calls the existing `service.activate(root)`, then and only then changes that exact source preference to managed;
- failed activation leaves preference unchanged;
- VAD deletion is refused while any configured external source requires it and permitted after the final such source is removed, while the core still protects live leases;
- copy preflight reports only missing root bytes/destination/free space;
- explicit copy consent calls `ModelArtifactService.install(root_descriptor, external_dir, declared_files_only=True)` and does not call `activate`, write readiness/active state, or change preference;
- already-installed root is a no-op; copy failure/cancel leaves external selection untouched.

- [ ] **Step 2: Run focused RED tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/STT/test_parakeet_sources.py Tests/Model_Artifacts/test_service.py Tests/UI/test_model_browser_state.py Tests/UI/test_model_installed_view.py Tests/UI/test_model_artifact_widgets.py -q`

Expected: FAIL on activation-required state, dependency role gating, deletion guard, and copy APIs.

- [ ] **Step 3: Correct inventory and activation behavior**

Derive activation from `descriptor.role is ArtifactRole.ROOT`, not `consumer != "unassigned"`. Separate `allow_activation` from `ready`; readiness absence on a valid root is an actionable state. Inject two narrow callbacks into `InstalledView`: `on_root_activated(ref)` and `may_delete(ref) -> str | None`. The Lab host supplies them from the app-owned source service.

- [ ] **Step 4: Add local-copy planning and execution to the source service**

```python
@dataclass(frozen=True)
class ManagedCopyPlan:
    reference: ArtifactRef
    additional_bytes: int
    destination: Path
    free_bytes: int
    already_installed: bool


def copy_into_managed(
    self, verified: VerifiedExternalParakeet, consent: ManagedCopyConsent
) -> ArtifactRef:
    # revalidate consent/metadata, then
    # core.install(root_descriptor, directory, declared_files_only=True)
    # intentionally do not call core.activate or change source preference
```

Reuse the core's existing staging and validation; do not add another copier. The plan must not include VAD bytes because VAD readiness is already required.

Add a default-false `declared_files_only` option to `ModelArtifactService.install`. In that mode, snapshot and recheck only descriptor-declared paths and their real, non-symlink ancestors; `_copy_payload` already copies only those files, and staging remains strictly validated. This lets a verified external directory contain an unrelated README without allocating a second full temporary copy. The default install path must remain strict and continue rejecting undeclared source entries.

- [ ] **Step 5: Mutation-test ordering and run GREEN**

Change preference before activation, permit dependency activation, and call `activate()` from copy one at a time; the corresponding tests must fail. Restore.

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/STT/test_parakeet_sources.py Tests/Model_Artifacts/test_service.py Tests/UI/test_model_browser_state.py Tests/UI/test_model_installed_view.py Tests/UI/test_model_artifact_widgets.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 6**

```bash
git add tldw_chatbook/STT/parakeet_sources.py tldw_chatbook/Model_Artifacts/service.py tldw_chatbook/UI/Screens/model_browser_state.py tldw_chatbook/UI/Screens/model_installed_view.py tldw_chatbook/Widgets/ModelArtifacts/activation_controls.py Tests/STT/test_parakeet_sources.py Tests/Model_Artifacts/test_service.py Tests/UI/test_model_browser_state.py Tests/UI/test_model_installed_view.py Tests/UI/test_model_artifact_widgets.py
git commit -m "feat(models): preserve external and managed Parakeet ownership"
```

### Task 7: Add the user-owned source section to Lab Models

**Files:**
- Create: `tldw_chatbook/UI/Screens/model_external_view.py`
- Modify: `tldw_chatbook/UI/Screens/model_curated_view.py`
- Modify: `tldw_chatbook/UI/Screens/llm_screen.py:39-80,321-470,1096-1235`
- Modify: `tldw_chatbook/UI/LLM_Management_Window.py:250-270,560-620`
- Create: `Tests/UI/test_model_external_view.py`
- Modify: `Tests/UI/test_model_curated_view.py`
- Modify: `Tests/UI/test_llm_screen_lab_adoption.py`
- Modify: `Tests/UI/test_destination_shells.py`

- [ ] **Step 1: Write RED state/event and mounted picker tests**

Require each exact Parakeet ROOT catalog row to emit `UseFromDiskRequested(ref)`. The external view lists only configured external records with `External source · descriptor verified`, retains a visible exact path only on this edit surface, and emits Change/Stop/Copy actions. A dependency row has no disk-selection action.

Mount one real `SelectDirectory` flow under the production Lab screen. Prove verification happens off-loop with byte progress, stale picker/worker results are fenced, cancel changes nothing, missing VAD opens a VAD-only consent plan, success commits through `ParakeetSourceService`, and no Parakeet-root URL is present in the VAD plan. Also prove successful curated Parakeet installation/activation calls `prefer_managed()` only after provision succeeds; failure leaves the prior preference unchanged.

- [ ] **Step 2: Run the Lab RED tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_model_external_view.py Tests/UI/test_model_curated_view.py Tests/UI/test_llm_screen_lab_adoption.py Tests/UI/test_destination_shells.py -q`

Expected: FAIL for absent external view/events/rail destination.

- [ ] **Step 3: Add the deferred External view and exact catalog actions**

Add `External` to `MODELS_RAIL_SECTIONS` and mount `ExternalModelView` through `LLMManagementWindow`'s existing deferred-view pattern. Keep the screen as worker owner so view recomposition cannot orphan verification, copying, or VAD provision. Use the exact `ArtifactRef` from the catalog row; never infer model/precision from display text.

- [ ] **Step 4: Implement one shared interactive selection state machine**

In `LLMScreen`: picker → verifier worker → runtime/VAD readiness result → optional VAD consent/provision worker → `commit_external`. Stamp every callback with `(selection_generation, screen identity)`. On VAD failure/cancel, retain the prior record and preference. Wire Change through the same flow, Stop through `stop_using_external`, and Copy through Task 6's consented copy plan.

- [ ] **Step 5: Run real-CSS geometry and behavior checks**

Use `TldwCli.CSS_PATH` at 80 columns for the new row/action surface. Assert controls remain reachable, progress/error copy is visible, and paths appear only inside the dedicated external edit view—not notifications or generic row errors.

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_model_external_view.py Tests/UI/test_model_curated_view.py Tests/UI/test_llm_screen_lab_adoption.py Tests/UI/test_destination_shells.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 7**

```bash
git add tldw_chatbook/UI/Screens/model_external_view.py tldw_chatbook/UI/Screens/model_curated_view.py tldw_chatbook/UI/Screens/llm_screen.py tldw_chatbook/UI/LLM_Management_Window.py Tests/UI/test_model_external_view.py Tests/UI/test_model_curated_view.py Tests/UI/test_llm_screen_lab_adoption.py Tests/UI/test_destination_shells.py
git commit -m "feat(models): configure external Parakeet sources"
```

### Task 8: Add external selection to First Run without partial config

**Files:**
- Modify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`
- Modify: `tldw_chatbook/UI/Wizards/first_run_speech_step_state.py`
- Modify: `Tests/Wizards/test_first_run_speech_step.py`
- Modify: `Tests/Wizards/test_first_run_speech_step_state.py`
- Modify: `Tests/UI/test_first_run_wizard_live_contract.py`

- [ ] **Step 1: Write RED state and mounted-flow tests**

For the selected exact model/precision, show both `Use model from disk…` and `Review and install…`. Cover picker cancel, hash progress, corrupt source, missing runtime (`Runtime required`; selection may persist but dispatch remains unusable), VAD-only consent, VAD failure/cancel leaving the old source untouched, stale callback after model/precision change, successful Next committing the exact external record plus speech defaults in one wizard config transaction, configured verifier ownership updating after that write, write failure leaving both config and service ownership unchanged, and managed install/activation preferring managed only after success.

- [ ] **Step 2: Run the First Run RED tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Wizards/test_first_run_speech_step.py Tests/Wizards/test_first_run_speech_step_state.py Tests/UI/test_first_run_wizard_live_contract.py -q`

Expected: FAIL because the speech step is managed-install-only.

- [ ] **Step 3: Add pending external state to the pure step model**

The state helper should merge a source-service-prepared patch with the existing speech defaults but never write it:

```python
def speech_config_patch(
    state: SpeechSetupState,
    source_commit: ExternalSourceConfigCommit,
) -> dict[str, object]:
    speech_values = build_speech_transcription_commit(
        provider_id=state.provider_id,
        model_id=state.model_id,
        language=state.language,
        precision=state.precision,
    )["transcription"]
    source_values = source_commit.section_values["transcription"]
    return {
        "transcription": {
            **speech_values,
            **source_values,
        }
    }
```

Before that nested merge, the wizard calls `ParakeetSourceService.prepare_config_commit(selection)`, which rechecks the root snapshot and VAD without writing. It keeps both the selection and returned commit pending until its existing atomic `commit_config` boundary. The test must assert that the resulting single `transcription` mapping contains all four speech defaults and the complete `parakeet_external_sources` table. After that one write succeeds, the wizard calls `accept_committed(source_commit)` to synchronize configured cache ownership without another write; a failed wizard write never calls `accept_committed`. Runtime availability is a separate usability state: the config commit requires verified root bytes and managed VAD readiness, but does not discard an otherwise valid external selection merely because the optional ONNX runtime is not installed yet.

- [ ] **Step 4: Wire the picker/worker/consent flow through the app-owned service**

Reuse Task 7's selection service calls, not its widgets. Fence results by generation and mount lifetime. `commit()` must refuse a pending external selection unless `prepare_config_commit()` succeeds, merge that returned patch into the wizard's single config transaction, and call `accept_committed()` only after the transaction succeeds. Cancellation or write failure restores the previous UI/service state. If runtime readiness is absent, commit the selection but keep the stable `Runtime required` usability state so dispatch fails clearly until installation.

- [ ] **Step 5: Run mounted real-contract GREEN tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Wizards/test_first_run_speech_step.py Tests/Wizards/test_first_run_speech_step_state.py Tests/UI/test_first_run_wizard_live_contract.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 8**

```bash
git add tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py tldw_chatbook/UI/Wizards/first_run_speech_step_state.py Tests/Wizards/test_first_run_speech_step.py Tests/Wizards/test_first_run_speech_step_state.py Tests/UI/test_first_run_wizard_live_contract.py
git commit -m "feat(wizard): select external Parakeet models"
```

### Task 9: Validate Library per-job overrides before enqueue

**Files:**
- Modify: `tldw_chatbook/Library/ingest_capabilities.py:555-585`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:18280-18620,18840-19020`
- Modify: `tldw_chatbook/app.py:1997-2190,2530-2635`
- Modify: `Tests/UI/test_library_ingest_canvas.py`
- Modify: `Tests/App/test_submit_library_ingest_job.py`
- Modify: `Tests/Library/test_library_ingest_runner.py`

- [ ] **Step 1: Write RED picker, pre-enqueue, headless, and batch tests**

Cover:

- the Parakeet-only model-directory field has a real directory picker;
- selected path stays only in that submission's options and never changes the persistent source record;
- Start validates the exact route's model/precision off-loop before `submit_library_ingest_job` is called;
- missing VAD opens VAD-only consent before any job exists;
- cancel/failure creates zero jobs and leaves the form/path available for correction;
- a folder batch validates/hashes once, submits each item with one batch scope, and releases after the last terminal/cancelled sibling;
- direct/headless app submission never downloads and yields structured `ModelNotInstalled`/artifact-incompatible failure with recovery actions;
- retries preserve the job-local path, but errors/logs/provenance omit it;
- successful use of the existing Library managed-install action prefers managed for v2 INT8 only after its activating provision succeeds.

```python
assert registry.jobs() == ()  # while validation or VAD consent is pending
assert persistent_source_service.records() == prior_records
```

- [ ] **Step 2: Run the Library RED tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_ingest_canvas.py Tests/App/test_submit_library_ingest_job.py Tests/Library/test_library_ingest_runner.py -q`

Expected: FAIL because the field has no picker and submission occurs before external verification/VAD readiness.

- [ ] **Step 3: Add a browse action without changing the capability system**

Keep `transcription_model_dir` as the existing text option and add one adjacent Browse button in the audio/video panel renderer. Its callback opens `SelectDirectory`, updates only `form.type_options["audio_video"]["transcription_model_dir"]`, and does not write global source config.

- [ ] **Step 4: Gate interactive submit through the shared service**

In `_do_submit_ingest`, when the resolved route is Parakeet and an override is present, capture the full form snapshot, mint one path-free `transcription_external_scope_id`, and start a generation-fenced verification worker owned by that provisional scope. If VAD is absent, show the existing acquisition consent with the VAD-only report. After both checks succeed, call `retain_prepared(scope_id, prepared)` **before** any job creation; this performs the required final key/path/snapshot recheck. Put only the scope ID (not the prepared object) into the captured audio/video options, then call the existing `submit_library_ingest_job`. If that call raises before a registry job exists, explicitly `release_scope(scope_id)`. Leave direct app submission side-effect free.

The app carries the same internal scope ID into every recursively created folder job and into resolved transcription context. A single file uses that same pre-adopted scope, so it needs no predicted job ID and no post-enqueue validation. The registry listener marks it observed when the first job appears and releases it after the final terminal sibling. Do not hash once per recursive `submit_library_ingest_job` call, and do not persist the prepared object in job options.

- [ ] **Step 5: Mutation-test the enqueue and batch boundaries**

Move job creation before `retain_prepared`, drop the explicit override after the first item, release scope after the first terminal item, and let an unrelated registry mutation clear an unsubmitted scope one at a time; tests must fail. Restore.

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_ingest_canvas.py Tests/App/test_submit_library_ingest_job.py Tests/Library/test_library_ingest_runner.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 9**

```bash
git add tldw_chatbook/Library/ingest_capabilities.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/app.py Tests/UI/test_library_ingest_canvas.py Tests/App/test_submit_library_ingest_job.py Tests/Library/test_library_ingest_runner.py
git commit -m "feat(library): validate external Parakeet overrides"
```

### Task 10: Prove the joined path, record macOS evidence, and update TASK-598 honestly

**Files:**
- Create: `Docs/STT_Evaluation/task-598/README.md`
- Create: `Docs/STT_Evaluation/task-598/macos-evidence.json`
- Modify: `backlog/tasks/task-598 - Use-descriptor-verified-external-Parakeet-ONNX-bundles.md`
- Modify only if this task exposed a repeatable trap: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md`

- [ ] **Step 1: Run the exact changed-test union once**

Build the union from the files changed since the rebased `origin/dev`, then run exactly those focused test files in one command. Do not use keyword filtering and do not rerun the entire union merely to hide an environment abort. If one concrete node fails, diagnose and rerun only that node before changing code.

Expected: all affected tests pass, or the report records the exact honest failure and narrower diagnosis.

- [ ] **Step 2: Run scoped static checks**

Run Ruff check and format-check only on changed Python files, compile changed package modules, scan the diff for path leakage/placeholders, and run:

```bash
git diff --check origin/dev...HEAD
```

Expected: zero exit status except any precisely documented, proven pre-existing format debt.

- [ ] **Step 3: Run one isolated real macOS external-mode smoke**

Create a scratch profile before importing `TldwCli`. Point the scratch config at a materialized descriptor-valid external Parakeet root and the existing verified managed VAD; do not copy the root. Exercise the production app-owned source service → coordinator → executor → Parakeet ONNX CPU path with a known PCM fixture. Record:

- host/architecture, Python, `onnx-asr`, ORT and provider;
- descriptor/model/precision and external snapshot token (never the path in committed evidence);
- exact VAD reference/lease identity;
- transcript, timing, result provenance, `artifact_root=null`;
- proof no managed Parakeet root was installed/activated and the external tree bytes/mtimes were unchanged;
- real profile and managed-store before/after hashes;
- whether this was an actual mounted user-surface/mic pass or only an in-memory production-path fallback.

Use a bounded probe and terminate it if it hangs. Do not infer unavailable platforms.

- [ ] **Step 4: Write evidence and perform a requirement-by-requirement self-review**

`README.md` must distinguish automated evidence, live macOS evidence, and open platform gates. `macos-evidence.json` must be machine-readable and validate with `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m json.tool`.

Review every TASK-598 AC and ADR-050 consequence against the diff. Search for any generic logging of `directory`, accidental `activate()` during copy, downloads inside dispatch/transcription, unmanaged VAD inputs, and external paths in provenance.

- [ ] **Step 5: Update Backlog through the CLI without falsely closing platform gates**

Add concise Implementation Notes with the plan, ADR, changed boundaries, focused commands, macOS evidence, and open Linux/Windows/macOS-x86_64 gates. Check only acceptance criteria fully evidenced on the available host. Keep TASK-598 `In Progress` until all five wheel-supported platform gates pass.

Run: `backlog task 598 --plain`

Expected: plan/notes present, evidence-linked criteria accurate, status still `In Progress` if any platform gate is open.

- [ ] **Step 6: Commit Task 10**

```bash
git add Docs/STT_Evaluation/task-598/README.md Docs/STT_Evaluation/task-598/macos-evidence.json 'backlog/tasks/task-598 - Use-descriptor-verified-external-Parakeet-ONNX-bundles.md'
git commit -m "docs(stt): record external Parakeet evidence"
```

### Task 11: Final focused review gate

**Files:**
- Review: all files changed in `origin/dev...HEAD`
- Modify: only files required to address concrete review findings

- [ ] **Step 1: Run a correctness review against the spec and ADR**

Use `superpowers:requesting-code-review`. Require the reviewer to trace one exact path end-to-end: Lab/First Run/Library selection → descriptor hashing → managed VAD readiness/lease → shared executor → path-private provenance → teardown. Also inspect preference switching, optional-copy activation ordering, and every cancellation/stale-callback path.

- [ ] **Step 2: Run a complexity pass**

Use `ponytail-review` on `origin/dev...HEAD`. Reject speculative registries, durable verification receipts, duplicate download/copy code, ONNX parsing, or per-surface rule copies. Keep only abstractions exercised by at least two approved surfaces or required for ownership/lifetime safety.

- [ ] **Step 3: Address concrete findings with focused RED→GREEN tests**

For every accepted finding, add the narrowest reproducing test first, verify RED, implement the smallest fix, and rerun only the impacted focused files. Use `superpowers:receiving-code-review` before changing code in response to unclear feedback.

- [ ] **Step 4: Re-run only the impacted verification and diff checks**

Run the exact changed test files affected by review fixes, scoped Ruff, and `git diff --check origin/dev...HEAD`. Do not run unrelated suites.

- [ ] **Step 5: Commit review fixes, if any**

```bash
git add <only-reviewed-files>
git commit -m "fix(stt): address external Parakeet review"
```

If there are no findings, do not create an empty commit.

### Task 12: Close the remaining native external-Parakeet evidence gates

**Files:**
- Create: .github/scripts/task598_external_parakeet_evidence.py
- Create: .github/workflows/task-598-platform-evidence.yml
- Create: Tests/CI/test_task598_external_parakeet_evidence.py
- Modify after successful remote evidence: Docs/STT_Evaluation/task-598/README.md
- Create after successful remote evidence: Docs/STT_Evaluation/task-598/platform-evidence.json
- Modify after all lanes pass: backlog/tasks/task-598 - Use-descriptor-verified-external-Parakeet-ONNX-bundles.md

- [x] **Step 1: Write the focused RED tests before the probe or workflow exists**

Add one CI test file. Read the workflow as text, following the existing CI-shape
tests, and import the probe only after asserting its path exists. Cover:

- only pull_request labeled activity plus workflow_dispatch can trigger it;
- the job requires task-598-platform-evidence for a PR and permits manual dispatch;
- the matrix is exactly ubuntu-24.04, ubuntu-24.04-arm, windows-2022, and
  macos-15-intel, uses Python 3.12, fail-fast false, and max-parallel 2;
- installation uses transcription_parakeet_onnx without a CI-only ORT pin;
- dependency-install failure is converted into a valid path-private lane JSON
  before the job fails, rather than skipping artifact creation;
- worker timeout is shorter than job timeout, validation always runs, and the
  uniquely named result artifact always uploads;
- supervising a sleeping child writes a valid timeout result with no scratch
  path or username;
- validation requires the tested SHA, run ID/attempt, exact v2/v3 keys, CPU
  provider, null artifact roots, exact VAD, unchanged external/cache/store
  invariants, and successful shutdown;
- validation rejects any string value containing the supplied scratch root.
- successful validation requires platform, architecture, Python, onnx-asr, and
  ONNX Runtime versions; an install failure records the unresolved stage instead.

Representative timeout test:

    def test_supervisor_records_a_path_private_timeout(tmp_path):
        output = tmp_path / "result.json"
        result = evidence.supervise(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            output=output,
            timeout_seconds=0.05,
            run_identity={
                "tested_commit": "a" * 40,
                "workflow_run_id": "1",
                "workflow_run_attempt": "1",
            },
            forbidden_roots=(tmp_path,),
        )
        assert result["failure_code"] == "timeout"
        assert str(tmp_path) not in output.read_text(encoding="utf-8")

Run: ../../.venv/bin/python -m pytest -q Tests/CI/test_task598_external_parakeet_evidence.py --confcutdir=Tests/CI

Expected: FAIL because the probe and workflow do not exist.

- [x] **Step 2: Implement the smallest supervised, path-private probe**

Use only stdlib imports at module import time. Provide parent, worker, and
validate CLI modes. Parent mode creates the scratch profile/config/data/cache
tree, sets HOME, XDG_CONFIG_HOME, XDG_DATA_HOME, XDG_CACHE_HOME, HF_HOME, and
TLDW_CONFIG_PATH before application imports, then launches the same script in
worker mode with subprocess.run and a timeout. Capture child output. Timeout
or nonzero exit writes only a bounded failure code and safe exception class;
never copy exception messages or child output into JSON.

Worker mode checks the resolved managed root is beneath the explicit scratch
paths.data_dir and binds the injected ModelArtifactService to that same root,
then processes Parakeet v2 INT8 and v3 INT8 sequentially:

1. Run production preflight/provision with an injected ModelArtifactService.
2. Copy only declared files with shutil.copy2 to the external directory and
   delete the temporary managed root through ModelArtifactService.delete.
3. Snapshot external sizes, hashes, modes, and mtimes.
4. Enable the Hugging Face offline variables, then create fresh TldwCli-owned
   source service/coordinator/executor resources for that model so import-time
   environment checks observe offline mode; verify the
   root, run plan_managed_copy/copy_into_managed, prove no readiness or source
   preference was created, and delete that managed copy.
5. Enable HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE; snapshot HF_HOME,
   XDG_CACHE_HOME, and the managed store; transcribe four seconds of generated
   16 kHz mono 16-bit zero PCM through TranscriptionService.transcribe_buffer,
   passing provider="parakeet-onnx", the exact model ID, language="en",
   precision="int8", and model_dir=str(external_directory).
6. Require a dictionary result with a string text field, CPUExecutionProvider,
   null artifact_root, and the exact VAD dependency. Close source service,
   coordinator, then executor before removing that model's external root.
7. Require external/cache/store invariants, record path-free timings and
   identities, remove the external directory, and process the other model.

The final store may contain only the exact VAD dependency. Success JSON records
platform, architecture, Python, onnx-asr, and ONNX Runtime versions plus the
checked-out git revision and workflow run identity. Always write JSON atomically.
Validation walks every string recursively and rejects forbidden roots,
credentials, usernames, and temporary-directory names.

- [x] **Step 3: Make RED tests GREEN and mutation-check the guards**

Implement the label-gated workflow with contents read permission, the exact
matrix, max-parallel 2, bounded job timeout, and Python 3.12. Checkout the
pull-request head SHA explicitly (falling back to github.sha for manual runs),
then initialize a path-private pending JSON before dependency installation.
Run installation with continue-on-error, convert a failed install outcome into
a dependency_install failure JSON, and run the supervised probe only after a
successful install. Always run validation and artifact upload; validation keeps
the lane failed when the recorded result is not successful.

Run the Step 1 command. Expected: PASS.

Mutation-check and restore: remove the label condition; use the default PR
merge checkout instead of the head SHA; skip dependency-install failure JSON;
remove macOS Intel or v3; leak the scratch path in timeout JSON; accept a
non-CPU provider or non-null root; omit the offline cache/store invariant. Each
focused test must fail under its matching mutation.

- [x] **Step 4: Run focused local verification and commit the CI seam**

Run:

    ../../.venv/bin/python -m pytest -q Tests/CI/test_task598_external_parakeet_evidence.py Tests/CI/test_github_actions_test_workflow.py --confcutdir=Tests/CI
    ../../.venv/bin/python -m ruff check .github/scripts/task598_external_parakeet_evidence.py Tests/CI/test_task598_external_parakeet_evidence.py
    ../../.venv/bin/python -m ruff format --check .github/scripts/task598_external_parakeet_evidence.py Tests/CI/test_task598_external_parakeet_evidence.py
    ../../.venv/bin/python -m py_compile .github/scripts/task598_external_parakeet_evidence.py
    git diff --check

Review for absolute paths, credentials, exception text, post-provision network
access, model caching, and activation/preference state. Commit the three new
files and this updated plan as ci(stt): verify external Parakeet platforms.

- [x] **Step 5: Push, bootstrap the labeled draft PR, and monitor the matrix**

Push the rebased branch, create or reuse a draft PR targeting dev, and apply
task-598-platform-evidence only after the workflow commit is visible remotely.
Confirm the run tests branch HEAD on all four runner labels. Wait once with a
bound. If a lane fails, inspect only that lane and its JSON. Fix only proven
product/packaging or probe defects; remove and reapply the label for a rerun.
Never substitute an architecture or accelerator for a failed CPU lane.

- [x] **Step 6: Persist evidence and close TASK-598 only if honest**

After all lanes pass, download and validate all four JSON artifacts and confirm
their SHA/run identity. Commit one normalized platform-evidence.json with the
run URL and four results. Update the README with resolved versions, timings,
and the limitation that generated zero PCM is a runtime—not quality—smoke.

Use Backlog CLI to check AC7 and mark TASK-598 Done only if these four results
plus existing macOS arm64 evidence satisfy the five-target matrix and every
Definition-of-Done item. Otherwise record the exact open lane and keep In
Progress.

Run:

    ../../.venv/bin/python -m json.tool Docs/STT_Evaluation/task-598/platform-evidence.json
    backlog task 598 --plain
    git diff --check

Commit only durable evidence and Backlog updates as
docs(stt): record native Parakeet platform evidence.
