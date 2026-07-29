# Shared Model Artifact Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the smallest offline, provider-neutral service that verifies,
activates, leases, inventories, reconciles, and safely deletes immutable ONNX
and GGUF artifact revisions.

**Architecture:** One new `service.py` owns descriptors and filesystem lifecycle.
It reuses `leases.py` for interprocess coordination and
`Utils/atomic_file_ops.py` for state records. One focused test module covers the
new behavior; the existing lease/process tests remain unchanged.

**Tech Stack:** Python 3.11 stdlib (`dataclasses`, `enum`, `hashlib`, `json`,
`pathlib`, `shutil`, `tempfile`, `urllib.parse`), existing `portalocker` wrapper,
pytest.

**ADR required:** no
**ADR path:** `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`
**Reason:** TASK-594 directly implements the already accepted artifact/runtime
boundary without changing it.

---

## File map

- Create `tldw_chatbook/Model_Artifacts/service.py`: descriptor validation,
  serialization, closure fingerprints, managed filesystem lifecycle, handles,
  inventory, reconciliation, and errors.
- Create `Tests/Model_Artifacts/test_service.py`: all new offline unit and
  spawn-process lifecycle coverage.
- Modify `tldw_chatbook/Model_Artifacts/__init__.py`: public re-exports only.
- Modify `backlog/docs/model-artifact-operation-leases.md`: document the exact
  lifecycle/exact-artifact lock split implemented here.
- Modify
  `backlog/tasks/task-594 - Build-shared-model-artifact-descriptors-and-lifecycle.md`:
  plan, verification, checked criteria, and implementation notes.

Do not create `descriptors.py`, `store.py`, a database, a repository interface,
download code, model catalogs, runtime adapters, background cleanup, or migration
logic.

### Task 0: Record the implementation plan before code

**Files:**

- Modify:
  `backlog/tasks/task-594 - Build-shared-model-artifact-descriptors-and-lifecycle.md`

- [ ] **Step 1: Attach this plan with the required ADR decision**

Use Backlog CLI before starting Task 1 to add the implementation plan, including:

```text
ADR required: no
ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
Reason: TASK-594 directly implements the accepted provider-neutral artifact
boundary without changing it.
```

Link this plan and ADR-025 from the task. Do not change any acceptance criterion
or mark the task Done at this point.

### Task 1: Typed descriptors and canonical closure identity

**Files:**

- Create: `tldw_chatbook/Model_Artifacts/service.py`
- Create: `Tests/Model_Artifacts/test_service.py`
- Modify: `tldw_chatbook/Model_Artifacts/__init__.py`

- [ ] **Step 1: Write failing descriptor tests**

Add tests for:

```python
def test_ref_requires_canonical_portable_components() -> None:
    assert ArtifactRef("parakeet-v2", "a" * 40, "int8").variant == "int8"
    for value in ("../x", "Parakeet", "con", "x ", "x/y", r"x\\y"):
        with pytest.raises(ValueError):
            ArtifactRef(value, "a" * 40, "int8")
    with pytest.raises(ValueError):
        ArtifactRef("parakeet-v2", "../revision", "int8")
    with pytest.raises(ValueError):
        ArtifactRef("parakeet-v2", "a" * 40, "INT8")


def test_descriptor_rejects_inconsistent_or_unsafe_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="installed bytes"):
        descriptor(expected_installed_bytes=2, files=(artifact_file(b"x"),))
    with pytest.raises(ValueError, match="provenance"):
        descriptor(
            provenance=(
                ProvenanceClass.INTEGRITY_VERIFIED,
                ProvenanceClass.LOCAL_INTEGRITY_RECORDED,
            )
        )
    with pytest.raises(ValueError, match="source_url"):
        descriptor(source_url="https://token@example.test/model?sig=secret")


def test_required_files_reject_unsafe_paths_and_casefold_aliases() -> None:
    for path in (
        "../model.onnx",
        "/model.onnx",
        r"nested\\model.onnx",
        "manifest.json",
        "active/state.json",
    ):
        with pytest.raises(ValueError):
            ArtifactFile(path, 1, "0" * 64)
    with pytest.raises(ValueError, match="case-insensitive"):
        descriptor(
            files=(
                ArtifactFile("Model.onnx", 1, "0" * 64),
                ArtifactFile("model.onnx", 1, "1" * 64),
            ),
            expected_installed_bytes=2,
        )
    with pytest.raises(ValueError, match="duplicate"):
        descriptor(
            files=(
                ArtifactFile("model.onnx", 1, "0" * 64),
                ArtifactFile("model.onnx", 1, "1" * 64),
            ),
            expected_installed_bytes=2,
        )


def test_file_and_dependency_metadata_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        ArtifactFile("model.onnx", -1, "0" * 64)
    with pytest.raises(ValueError):
        ArtifactFile("model.onnx", 1, "not-a-sha256")
    conflicting = (
        ArtifactRef("silero-vad", "a" * 40, "int8"),
        ArtifactRef("silero-vad", "b" * 40, "int8"),
    )
    with pytest.raises(ValueError, match="conflicting"):
        descriptor(dependencies=conflicting)


def test_descriptor_round_trip_and_fingerprint_are_stable() -> None:
    root = ref("parakeet-v2")
    vad = ref("silero-vad")
    encoded = descriptor(reference=root, dependencies=(vad,)).to_dict()
    assert ArtifactDescriptor.from_dict(encoded).to_dict() == encoded
    assert closure_fingerprint(root, (vad,)) == closure_fingerprint(root, (vad,))
    assert closure_fingerprint(root, (vad,)) != closure_fingerprint(root, ())
```

Use local helper factories in the test module so later tests create small
descriptor-backed payloads without fixtures or framework code.

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Model_Artifacts/test_service.py -q
```

Expected: collection fails because `service.py` and its public types do not exist.

- [ ] **Step 3: Implement the minimal descriptor contracts**

Create frozen dataclasses and string enums:

```python
class ArtifactRole(str, Enum):
    ROOT = "root"
    DEPENDENCY = "dependency"


class ArtifactFormat(str, Enum):
    ONNX = "onnx"
    GGUF = "gguf"


class ProvenanceClass(str, Enum):
    CHATBOOK_CURATED = "chatbook_curated"
    INTEGRITY_VERIFIED = "integrity_verified"
    LOCAL_INTEGRITY_RECORDED = "local_integrity_recorded"


@dataclass(frozen=True, order=True)
class ArtifactRef:
    artifact_id: str
    revision: str
    variant: str

    def lease_key(self) -> ArtifactLeaseKey:
        return ArtifactLeaseKey(self.artifact_id, self.revision, self.variant)


@dataclass(frozen=True)
class ArtifactFile:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ArtifactDescriptor:
    reference: ArtifactRef
    model_id: str
    role: ArtifactRole
    format: ArtifactFormat
    consumer: str
    model_family: str
    upstream_repository: str
    upstream_revision: str
    source_url: str
    precision: str
    expected_installed_bytes: int
    license_id: str
    license_url: str
    usage_notice: str
    runtime_name: str
    runtime_version_constraint: str
    supported_os: tuple[str, ...]
    supported_architectures: tuple[str, ...]
    provenance: tuple[ProvenanceClass, ...]
    files: tuple[ArtifactFile, ...]
    dependencies: tuple[ArtifactRef, ...] = ()

    def to_dict(self) -> dict[str, object]: ...

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ArtifactDescriptor": ...
```

Keep validation private and explicit. Use canonical compact JSON plus a version
prefix for:

```python
def closure_fingerprint(
    root: ArtifactRef, dependencies: Iterable[ArtifactRef]
) -> str:
    refs = sorted({root, *dependencies})
    payload = json.dumps(
        [ref.to_dict() for ref in refs],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(b"artifact-closure-v1\0" + payload).hexdigest()
```

Reject unknown manifest schema versions and unknown/mistyped required fields.
Validate every `ArtifactRef` component, file path, size, hash, and dependency
identity on direct construction and deserialization. All three ref components use
the same canonical lowercase ASCII path-component grammar; revisions are not
restricted to hashes. Reserve managed state names, reject
absolute/traversal/backslash paths and Windows reserved/casefold aliases, and
reject two dependency revisions for the same artifact identity. Do not add a
generic schema framework.

- [ ] **Step 4: Export and run GREEN**

Re-export only public descriptor types, `closure_fingerprint`, and stable error
types from `Model_Artifacts/__init__.py`.

Run:

```bash
../../.venv/bin/python -m pytest Tests/Model_Artifacts/test_service.py -q
../../.venv/bin/python -m ruff check tldw_chatbook/Model_Artifacts/service.py Tests/Model_Artifacts/test_service.py
```

Expected: descriptor tests pass; Ruff passes.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Model_Artifacts/service.py \
  tldw_chatbook/Model_Artifacts/__init__.py \
  Tests/Model_Artifacts/test_service.py
git commit -m "feat(artifacts): add immutable descriptor contracts"
```

### Task 2: Verified promotion, inventory, and disk totals

**Files:**

- Modify: `tldw_chatbook/Model_Artifacts/service.py`
- Modify: `Tests/Model_Artifacts/test_service.py`

- [ ] **Step 1: Write failing install tests**

Cover these outcomes with one-byte or few-byte local payloads:

```python
def test_install_verifies_then_promotes_immutable_directory(tmp_path: Path) -> None:
    service = ModelArtifactService(tmp_path / "store")
    source, expected = source_tree(tmp_path, {"model.onnx": b"model"})
    item = descriptor(files=expected)

    assert service.install(item, source) == item.reference
    final = service.artifact_path(item.reference)
    assert (final / "model.onnx").read_bytes() == b"model"
    manifest = read_json(final / "manifest.json")
    assert set(manifest) == {"schema_version", "descriptor"}
    assert manifest["schema_version"] == 1
    assert ArtifactDescriptor.from_dict(manifest["descriptor"]) == item


def test_install_hash_failure_never_creates_final_directory(tmp_path: Path) -> None:
    service = ModelArtifactService(tmp_path / "store")
    source, _expected = source_tree(tmp_path, {"model.onnx": b"wrong"})
    item = descriptor(files=(ArtifactFile("model.onnx", 5, "0" * 64),))

    with pytest.raises(ArtifactIntegrityError):
        service.install(item, source)
    assert service.artifact_path(item.reference).exists() is False


def test_idempotent_install_rehashes_existing_payload(tmp_path: Path) -> None:
    service, item, final = installed_artifact(tmp_path)
    (final / item.files[0].path).write_bytes(b"x" * item.files[0].size_bytes)

    with pytest.raises(ArtifactIntegrityError):
        service.install(item, source_for(item))


@pytest.mark.parametrize("unsafe_entry", ("extra", "symlink"))
def test_install_rejects_extra_files_and_symlinks(
    tmp_path: Path, unsafe_entry: str
) -> None:
    service, item, source = install_inputs(tmp_path)
    if unsafe_entry == "extra":
        (source / "extra.bin").write_bytes(b"extra")
    else:
        (source / "alias.onnx").symlink_to(source / "model.onnx")
    with pytest.raises(ArtifactPathError):
        service.install(item, source)
    assert service.artifact_path(item.reference).exists() is False


def test_install_rejects_nested_source_symlink(tmp_path: Path) -> None:
    service, item, source = install_inputs(tmp_path)
    (source / "nested").mkdir()
    (source / "nested" / "model.onnx").symlink_to(source / "model.onnx")
    with pytest.raises(ArtifactPathError):
        service.install(item, source)


@pytest.mark.parametrize("failure", ("copy", "hash", "promotion"))
def test_failed_install_removes_only_owned_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    service, item, source = install_inputs(tmp_path)
    abandoned = service.staging_path / "pre-existing"
    abandoned.mkdir(parents=True)
    arrange_install_failure(service, monkeypatch, failure)
    with pytest.raises(ArtifactError):
        service.install(item, source)
    assert tuple(service.staging_path.iterdir()) == (abandoned,)
    assert service.artifact_path(item.reference).exists() is False


@pytest.mark.parametrize("populated", (False, True))
def test_promotion_never_replaces_existing_destination(
    tmp_path: Path, populated: bool
) -> None:
    service, item, source = install_inputs(tmp_path)
    destination = service.artifact_path(item.reference)
    destination.mkdir(parents=True)
    if populated:
        (destination / "keep").write_text("existing")
    with pytest.raises(ArtifactConflictError):
        service.install(item, source)
    assert destination.exists()
    if populated:
        assert (destination / "keep").read_text() == "existing"
    else:
        assert tuple(destination.iterdir()) == ()


def test_inventory_and_exact_disk_usage_need_no_runtime_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, item, _ = installed_artifact(tmp_path)
    abandoned = service.staging_path / "abandoned" / "part"
    abandoned.parent.mkdir(parents=True)
    abandoned.write_bytes(b"staging")
    monkeypatch.setattr(shutil, "disk_usage", lambda _: (100, 40, 60))
    inventory = service.list_installed()
    assert inventory[0].descriptor == item
    totals = service.disk_usage()
    assert totals.installed_bytes == tree_size(service.artifacts_path)
    assert totals.staging_bytes == len(b"staging")
    assert totals.free_bytes == 60
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Model_Artifacts/test_service.py \
  -q -k "install or inventory or disk"
```

Expected: fails because `ModelArtifactService` is not implemented.

- [ ] **Step 3: Implement service layout and verified promotion**

Add:

```python
@dataclass(frozen=True)
class InstalledArtifact:
    path: Path
    descriptor: ArtifactDescriptor | None
    ready: bool
    active: bool
    error: str | None = None


@dataclass(frozen=True)
class ArtifactDiskUsage:
    installed_bytes: int
    staging_bytes: int
    free_bytes: int


class ModelArtifactService:
    def __init__(self, root: Path, *, lease_timeout_seconds: float = 5.0) -> None: ...

    def artifact_path(self, reference: ArtifactRef) -> Path: ...

    def install(
        self, descriptor: ArtifactDescriptor, source_directory: Path
    ) -> ArtifactRef: ...

    def list_installed(self) -> tuple[InstalledArtifact, ...]: ...

    def disk_usage(self) -> ArtifactDiskUsage: ...
```

Implementation constraints:

- validate and resolve the service root once;
- use service-owned `artifacts`, `active`, `ready`, `staging`, and `locks`;
- use the private lease key `ArtifactLeaseKey("!lifecycle", "1", "writer")`;
  `!` is forbidden by `ArtifactRef` grammar, so it cannot collide with an
  artifact lease key;
- copy declared regular files into a unique same-filesystem staging directory;
- reject missing, extra, and symlinked entries without following links;
- acquire the private lifecycle exclusive lease, then the target exclusive lease;
- recheck the destination after locking;
- rehash staging or an existing idempotent destination;
- write `manifest.json` with `atomic_write_json`;
- promote only after an authoritative absent-destination check under both locks;
  use `os.rename(staging, final)` under the documented sole-writer invariant and
  treat any pre-existing empty or populated destination as conflict/idempotence,
  never as a replace target;
- remove only the operation-owned staging directory on failure;
- scan regular files without following symlinks for disk totals.

Mark the global lock tradeoff once:

```python
# ponytail: one lifecycle writer lock is enough until measured install throughput
# justifies per-artifact writer coordination.
```

- [ ] **Step 4: Run GREEN and the existing lease regression suite**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Model_Artifacts/test_service.py -q
../../.venv/bin/python -m pytest \
  Tests/Model_Artifacts/test_operation_leases.py \
  Tests/Model_Artifacts/test_operation_leases_process.py -q
```

Expected: new tests pass; existing result remains `53 passed`.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Model_Artifacts/service.py Tests/Model_Artifacts/test_service.py
git commit -m "feat(artifacts): verify and promote immutable artifacts"
```

### Task 3: Dependency readiness, activation, and leased handles

**Files:**

- Modify: `tldw_chatbook/Model_Artifacts/service.py`
- Modify: `Tests/Model_Artifacts/test_service.py`

- [ ] **Step 1: Write failing closure and handle tests**

Add tests for:

```python
def test_activate_writes_readiness_only_after_exact_dependency_verification(
    tmp_path: Path,
) -> None:
    service, root, vad = installed_root_and_vad(tmp_path)
    service.activate(root.reference)

    handle = service.acquire(root.reference)
    with handle as leased:
        assert leased.handle.root == root.reference
        assert leased.handle.closure == tuple(sorted((root.reference, vad.reference)))
        assert leased.handle.closure_fingerprint == closure_fingerprint(
            root.reference, (vad.reference,)
        )
        assert leased.handle.resident_identity == (
            root.reference,
            leased.handle.closure_fingerprint,
        )


def test_activate_rejects_missing_dependency_and_cycle(tmp_path: Path) -> None:
    service, root, _missing = installed_root_with_missing_dependency(tmp_path)
    with pytest.raises(ArtifactDependencyError, match="missing"):
        service.activate(root.reference)
    assert service.readiness_path(root.reference).exists() is False

    service, first, _second = installed_dependency_cycle(tmp_path / "cycle")
    with pytest.raises(ArtifactDependencyError, match="cycle"):
        service.activate(first.reference)
    assert service.readiness_path(first.reference).exists() is False


def test_activate_reuses_matching_readiness_without_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, root, _ = ready_root_and_vad(tmp_path)
    monkeypatch.setattr(service, "_verify_installed", fail_if_called)
    service.activate(root.reference)


@pytest.mark.parametrize(
    "mutation",
    ("malformed", "unsupported-version", "wrong-fingerprint", "wrong-closure"),
)
def test_activate_never_trusts_invalid_readiness(
    tmp_path: Path, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, root, _ = ready_root_and_vad(tmp_path)
    mutate_readiness(service, root.reference, mutation)
    calls = 0
    original = service._verify_installed

    def counting_verifier(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "_verify_installed", counting_verifier)
    service.activate(root.reference)
    assert calls > 0


def test_active_write_failure_preserves_previous_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, previous, replacement = installed_two_root_revisions(tmp_path)
    service.activate(previous.reference)
    previous_state = service.active_path(previous.reference.artifact_id).read_bytes()
    fail_atomic_write_for_active(service, monkeypatch)
    with pytest.raises(ArtifactStateError):
        service.activate(replacement.reference)
    assert service.active_path(previous.reference.artifact_id).read_bytes() == previous_state
    assert service.readiness_path(replacement.reference).exists()


def test_acquire_releases_when_readiness_changes_during_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, root, _ = ready_root_and_vad(tmp_path)
    change_readiness_after_first_read(service, root.reference, monkeypatch)
    with pytest.raises(ArtifactStateError, match="changed"):
        service.acquire(root.reference)
    service.delete(root.reference)
    assert service.artifact_path(root.reference).exists() is False
```

Also assert no readiness record appears when any dependency verification fails.

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Model_Artifacts/test_service.py \
  -q -k "activate or readiness or acquire or closure"
```

Expected: fails because activation and acquisition do not exist.

- [ ] **Step 3: Implement closure state and leased handles**

Add:

```python
@dataclass(frozen=True)
class ArtifactHandle:
    root: ArtifactRef
    closure: tuple[ArtifactRef, ...]
    closure_fingerprint: str
    paths: tuple[tuple[ArtifactRef, Path], ...]

    @property
    def lease_keys(self) -> tuple[ArtifactLeaseKey, ...]: ...

    @property
    def resident_identity(self) -> tuple[ArtifactRef, str]: ...


class LeasedArtifactHandle:
    handle: ArtifactHandle

    def __enter__(self) -> "LeasedArtifactHandle": ...
    def __exit__(self, ...) -> None: ...
    def close(self) -> None: ...
```

Implement:

- strict manifest loading;
- recursive exact dependency resolution with cycle detection;
- canonical sorted closure and readiness JSON;
- versioned, strict record shapes:
  - manifest: `schema_version`, `descriptor`;
  - readiness: `schema_version`, `root`, `closure`, `closure_fingerprint`;
  - active selector: `schema_version`, `root`;
- require integer `schema_version == 1`, reject unknown/mistyped/extra fields,
  recompute the fingerprint, resolve the exact closure again from strict
  manifests, and compare it before readiness reuse;
- derive every filesystem path from validated `ArtifactRef` values only;
- full hash verification only when readiness must be built/rebuilt;
- lifecycle-exclusive plus shared closure leases for activation;
- readiness atomically replaced before the active selector, with both writes
  using `atomic_write_json`, and only after verification;
- active selector failure leaving prior selector intact;
- `acquire()` read → shared lease set → re-read comparison;
- guaranteed lease release on every mismatch/error path.

- [ ] **Step 4: Run GREEN**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Model_Artifacts/test_service.py -q
../../.venv/bin/python -m ruff check tldw_chatbook/Model_Artifacts/service.py Tests/Model_Artifacts/test_service.py
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Model_Artifacts/service.py Tests/Model_Artifacts/test_service.py
git commit -m "feat(artifacts): activate and lease verified closures"
```

### Task 4: Safe deletion and crash reconciliation

**Files:**

- Modify: `tldw_chatbook/Model_Artifacts/service.py`
- Modify: `Tests/Model_Artifacts/test_service.py`

- [ ] **Step 1: Write failing deletion and reconciliation tests**

Cover:

```python
def test_delete_refuses_loaded_root_then_succeeds_after_release(
    tmp_path: Path,
) -> None:
    service, root, _ = ready_root_and_vad(tmp_path, lease_timeout_seconds=0.05)
    leased = service.acquire(root.reference)
    with pytest.raises(ArtifactInUseError):
        service.delete(root.reference)
    leased.close()
    service.delete(root.reference)
    assert service.artifact_path(root.reference).exists() is False


def test_delete_dependency_clears_affected_root_readiness_and_active(
    tmp_path: Path,
) -> None:
    service, root, dependency = ready_root_and_vad(tmp_path)
    service.delete(dependency.reference)
    assert service.readiness_path(root.reference).exists() is False
    assert service.active_path(root.reference.artifact_id).exists() is False
    with pytest.raises(ArtifactNotReadyError):
        service.acquire(root.reference)


def test_delete_succeeds_after_spawned_reader_process_dies(
    tmp_path: Path,
) -> None:
    service, root, _ = ready_root_and_vad(tmp_path, lease_timeout_seconds=0.05)
    process, ready, _release = spawn_closure_reader(service, root.reference)
    assert ready.wait(5)
    with pytest.raises(ArtifactInUseError):
        service.delete(root.reference)
    process.terminate()
    process.join(5)
    service.delete(root.reference)
    assert service.artifact_path(root.reference).exists() is False


def test_reconcile_rebuilds_valid_readiness_and_invalidates_corruption(
    tmp_path: Path,
) -> None:
    service, root, dependency = installed_root_and_vad(tmp_path)
    service.activate(root.reference)
    service.readiness_path(root.reference).unlink()
    report = service.reconcile()
    assert report.readiness_created == 1
    (service.artifact_path(dependency.reference) / "model.onnx").write_bytes(b"bad")
    report = service.reconcile()
    assert service.readiness_path(root.reference).exists() is False
    assert service.active_path(root.reference.artifact_id).exists() is False
    assert service.artifact_path(dependency.reference) in report.corrupt_artifacts
    assert service.artifact_path(dependency.reference).exists()
    installed = {
        entry.descriptor.reference: entry
        for entry in service.list_installed()
        if entry.descriptor is not None
    }
    assert installed[dependency.reference].ready is False


def test_reconcile_removes_malformed_state_without_deleting_payload(
    tmp_path: Path,
) -> None:
    service, root, _ = installed_root_and_vad(tmp_path)
    service.readiness_path(root.reference).write_text("{bad")
    service.active_path(root.reference.artifact_id).write_text("{bad")
    report = service.reconcile()
    assert report.state_removed == 2
    assert service.artifact_path(root.reference).exists()


def test_reconcile_reports_observed_staging_entries_without_removing_them(
    tmp_path: Path,
) -> None:
    service = ModelArtifactService(tmp_path / "store")
    observed = service.staging_path / "interrupted"
    observed.mkdir(parents=True)
    report = service.reconcile()
    assert report.staging_entries == (observed,)
    assert observed.exists()


@pytest.mark.parametrize(
    "state_kind,contents",
    (
        ("readiness", b""),
        ("readiness", b'{"schema_version":2}'),
        ("active", b""),
        ("active", b'{"schema_version":2}'),
    ),
)
def test_reconcile_removes_interrupted_or_unsupported_state(
    tmp_path: Path, state_kind: str, contents: bytes
) -> None:
    service, root, _ = installed_root_and_vad(tmp_path)
    state_path = state_path_for(service, root.reference, state_kind)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_bytes(contents)
    report = service.reconcile()
    assert report.state_removed == 1
    assert service.artifact_path(root.reference).exists()
    if state_kind == "readiness":
        assert report.readiness_created == 1
        assert state_path.exists()
    else:
        assert state_path.exists() is False
    with service.acquire(root.reference):
        pass


def test_reconcile_rejects_installed_payload_symlink_without_deleting_it(
    tmp_path: Path,
) -> None:
    service, item, final = installed_artifact(tmp_path)
    payload = final / item.files[0].path
    payload.unlink()
    payload.symlink_to(tmp_path / "outside")
    report = service.reconcile()
    assert final in report.corrupt_artifacts
    assert final.exists()
    with pytest.raises(ArtifactNotReadyError):
        service.acquire(item.reference)
```

These cases must leave installed payload visible but unloadable until a verified
readiness record exists.

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Model_Artifacts/test_service.py \
  -q -k "delete or reconcile or process"
```

Expected: fails because deletion and reconciliation do not exist.

- [ ] **Step 3: Implement delete and reconcile**

Add:

```python
@dataclass(frozen=True)
class ReconcileReport:
    readiness_created: int
    state_removed: int
    corrupt_artifacts: tuple[Path, ...]
    staging_entries: tuple[Path, ...]


def delete(self, reference: ArtifactRef) -> None: ...
def reconcile(self) -> ReconcileReport: ...
```

Deletion:

- lifecycle-exclusive, then target-exclusive;
- convert target lease timeout to `ArtifactInUseError`;
- change no state when the target lease cannot be acquired;
- remove every readiness record whose closure contains the target;
- clear active selectors selecting the target or any invalidated root;
- delete only the contained immutable target directory.

Reconciliation:

- lifecycle-exclusive for the complete operation;
- lifecycle-only removal for malformed/orphaned derived records whose closure
  cannot be known;
- shared exact closure leases when valid references exist;
- full verification before readiness reconstruction;
- clear active state for missing/corrupt/unready roots;
- never delete corrupt payload or observed staging entries automatically;
  observed entries may include active pre-lifecycle installs.

- [ ] **Step 4: Run GREEN including process tests**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Model_Artifacts -q
```

Expected: all existing 53 lease tests plus the new lifecycle tests pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Model_Artifacts/service.py Tests/Model_Artifacts/test_service.py
git commit -m "feat(artifacts): reconcile and delete leased artifacts safely"
```

### Task 5: Boundary verification and task closeout

**Files:**

- Modify: `tldw_chatbook/Model_Artifacts/__init__.py`
- Modify: `Tests/Model_Artifacts/test_service.py`
- Modify: `backlog/docs/model-artifact-operation-leases.md`
- Modify:
  `backlog/tasks/task-594 - Build-shared-model-artifact-descriptors-and-lifecycle.md`

- [ ] **Step 1: Add the runtime-import boundary test**

Use a clean subprocess so already-imported test modules cannot hide eager imports:

```python
def test_package_import_does_not_load_inference_or_http_runtimes() -> None:
    code = """
import sys
import tldw_chatbook.Model_Artifacts
for name in ("onnxruntime", "onnx_asr", "ctranslate2", "faster_whisper", "httpx"):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", code], check=True)
```

- [ ] **Step 2: Update public exports and lease documentation**

Export only the approved public API. Update
`backlog/docs/model-artifact-operation-leases.md` to state:

- promotion/deletion: lifecycle-exclusive plus target-exclusive;
- activation/reconciliation verification: lifecycle-exclusive plus shared closure;
- load: shared closure for resident lifetime;
- malformed derived-state cleanup: lifecycle-exclusive only;
- fixed acquisition order: lifecycle, then sorted artifact keys.

Do not mark TASK-505 complete or claim Windows/Linux proof.

- [ ] **Step 3: Run complete focused verification**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Model_Artifacts -q
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Model_Artifacts \
  Tests/Model_Artifacts
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Model_Artifacts \
  Tests/Model_Artifacts
../../.venv/bin/python -m compileall -q tldw_chatbook/Model_Artifacts
git diff --check
```

Expected: all commands pass. Record the exact test count.

- [ ] **Step 4: Complete backlog hygiene**

Use Backlog CLI to:

- check TASK-594 AC #1–#7;
- add concise implementation notes naming the one-module approach, lock behavior,
  tests, deliberate non-goals, and modified files;
- record that ADR-025 was reused and no new ADR was required;
- set TASK-594 to Done only after every command above passes.

- [ ] **Step 5: Self-review scope**

Confirm:

```bash
git diff --stat origin/dev...HEAD
git diff --name-only origin/dev...HEAD
```

The implementation must not contain download clients, Textual UI, native runtime
imports, model catalogs, content deduplication, LLM migration, or edits to the
active first-run wizard branch.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Model_Artifacts/__init__.py \
  Tests/Model_Artifacts/test_service.py \
  backlog/docs/model-artifact-operation-leases.md \
  'backlog/tasks/task-594 - Build-shared-model-artifact-descriptors-and-lifecycle.md'
git commit -m "docs(artifacts): complete shared lifecycle task"
```
