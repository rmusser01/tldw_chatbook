# TASK-2062.1 Managed Local GGUF Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user copy one local GGUF into the managed artifact store with path-private, full-digest identity, authoritative staged validation, safe cancellation, and Installed-view recovery while leaving the original untouched.

**Architecture:** Split generic GGUF-v3 structure inspection from transcribe.cpp compatibility, then add one `ModelArtifactService.import_local_gguf()` operation that streams an already-open regular file directly into existing service-owned staging. The Installed view owns picker, consent, progress, cancellation, activation, and recovery presentation; the service remains the sole managed-store writer.

**Tech Stack:** Python 3.11+, stdlib `os`/`stat`/`hashlib`/`threading`, existing `ModelArtifactService` leases and staging, Textual 8.x workers/Pilot, pytest, Ruff.

## Global Constraints

- Read `AGENTS.md`, TASK-2062.1, the approved TASK-2062 design, amended ADR-025, `backlog/docs/lessons-testing-evidence.md`, and `backlog/docs/lessons-live-verification.md` before editing.
- Use strict TDD: write every focused test first, run it red for the intended missing behavior, then implement the smallest green change.
- Do not add a GGUF parser dependency, import a native inference runtime, or activate `_deferred_gguf_managed_import.py`.
- The source file is never opened for write, renamed, moved, deleted, or persisted as provenance.
- `source_url == ""` is valid only for exact `LOCAL_INTEGRITY_RECORDED`; `license_url == ""` additionally requires `license_id == "unknown"`.
- The exact revision is `sha256-` plus all 64 lowercase digest characters. Filename is never identity.
- Copy the source once into existing operation-owned staging as `model.gguf`; inspect and verify the staged bytes before promotion.
- Cancellation is honored before promotion. Promotion begins the non-cancellable **Finalizing** point of no return.
- Successful import activates readiness but never changes llama.cpp or llamafile preference or source selection.
- Keep external GGUF usage, vLLM, MLX, Hugging Face model IDs, and every download-removal path out of this PR.
- UI workers have static path-free descriptions. Notifications and logs never contain selected paths, commands, stderr, raw exceptions, or exception strings.
- Progress updates existing widgets in place; recompose only when controls appear/disappear. Verify real production CSS at 80 columns.
- ADR required: yes. ADR path: `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md` (already amended by the approved design).

---

## File responsibility map

- `tldw_chatbook/Model_Artifacts/gguf_admission.py` — generic bounded GGUF structure inspection and cross-platform safe local-file handle.
- `tldw_chatbook/Model_Artifacts/service.py` — local descriptor constraints, import progress/result types, one-copy staging transaction, exact promotion, and existing-store convergence.
- `tldw_chatbook/Model_Artifacts/__init__.py` — export only the public import result/progress types needed by UI callers.
- `tldw_chatbook/Widgets/ModelArtifacts/local_gguf_import.py` — intent-only unmanaged-row control and consent-only modal.
- `tldw_chatbook/Widgets/ModelArtifacts/install_progress.py` — render import phases through the incumbent stable progress widget.
- `tldw_chatbook/Widgets/ModelArtifacts/__init__.py` — export the new intent/modal types.
- `tldw_chatbook/UI/Screens/model_browser_state.py` — pure unmanaged-row action state.
- `tldw_chatbook/UI/Screens/model_installed_view.py` — picker, generation/cancellation state, worker orchestration, activation recovery, and path-private copy.
- `Tests/Model_Artifacts/test_gguf_admission.py` — structural-policy split and safe local-handle contracts.
- `Tests/Model_Artifacts/test_service.py` — descriptor and managed import transaction contracts.
- `Tests/Model_Artifacts/test_acquisition_types.py` — prove an uninstalled local descriptor cannot enter the download planner.
- `Tests/UI/test_model_browser_state.py` — pure row state.
- `Tests/UI/test_model_artifact_widgets.py` — modal, intent, progress, and focus behavior.
- `Tests/UI/test_model_installed_view.py` — mounted picker/import/cancel/recovery/path-privacy behavior.

## Acceptance-criteria traceability

| Acceptance criterion | Planned evidence |
|---|---|
| AC1 immutable managed copy/original untouched | Tasks 1, 3, and 4 source-handle, write-target, mutation, I/O-failure, and cleanup tests |
| AC2 path-private full-digest convergence | Tasks 2–4 descriptor round-trip, manifest scan, rename convergence, and changed-byte revision tests |
| AC3 staged structure and digest authority | Tasks 1, 3, and 4 generic parser plus staged-corruption mutation |
| AC4 cancellation/cleanup/activation recovery | Tasks 3, 4, and 6 cancellation, concurrent winner, reconcile, Finalizing, and recovery tests |
| AC5 readiness without source preference mutation | Task 6 mounted activation and preference-isolation mutation |
| AC6 progress/recovery/80-column keyboard behavior | Tasks 5–7 mounted production-CSS, focus-identity, physical Cancel, and native-lane evidence |

### Task 1: Split structural GGUF inspection from speech compatibility

**Files:**
- Modify: `tldw_chatbook/Model_Artifacts/gguf_admission.py:143-177,450-653`
- Test: `Tests/Model_Artifacts/test_gguf_admission.py`

**Interfaces:**
- Consumes: existing `GGUFMetadata`, `GGUFSourceIdentity`, `require_transcribe_cpp_architecture()`, and bounded parser constants.
- Produces: `inspect_gguf_structure(handle: BinaryIO, *, file_size: int) -> GGUFMetadata`, `OpenedLocalGGUF`, `open_local_gguf(path: str | Path)`, and `validate_local_gguf_structure(path: str | Path) -> LocalGGUFInspection`.
- Preserves: `inspect_gguf()` and `validate_local_gguf()` continue enforcing the exact transcribe.cpp allowlist and wheel platform.

Add the generic inspection result beside `LocalGGUFAdmission`:

```python
@dataclass(frozen=True)
class LocalGGUFInspection:
    """Bounded structure result for one safely opened local GGUF."""

    path: Path = field(repr=False)
    metadata: GGUFMetadata
    source_identity: GGUFSourceIdentity
```

- [ ] **Step 1: Write the generic-versus-speech policy test**

Add a test proving that an ordinary LLM architecture is structurally valid but remains invalid for transcribe.cpp:

```python
def test_generic_structure_accepts_llama_without_weakening_transcribe_policy():
    payload = make_gguf(architecture="llama", name="Local LLM", file_type=7)

    metadata = gguf.inspect_gguf_structure(
        io.BytesIO(payload),
        file_size=len(payload),
    )

    assert metadata.architecture == "llama"
    assert metadata.model_name == "Local LLM"
    with pytest.raises(gguf.GGUFArchitectureError):
        gguf.inspect_gguf(io.BytesIO(payload), file_size=len(payload))

    malformed = make_gguf(architecture="../private")
    with pytest.raises(gguf.GGUFArchitectureError, match="identifier"):
        gguf.inspect_gguf_structure(
            io.BytesIO(malformed),
            file_size=len(malformed),
        )
```

- [ ] **Step 2: Run the policy test and record genuine RED**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/Model_Artifacts/test_gguf_admission.py::test_generic_structure_accepts_llama_without_weakening_transcribe_policy
```

Expected: collection or attribute failure because `inspect_gguf_structure` does not exist.

- [ ] **Step 3: Extract the structural parser and keep the old wrapper strict**

Rename the current `inspect_gguf()` implementation to `inspect_gguf_structure()`.
Keep every statement from the initial handle-position check through construction of
`GGUFMetadata`, but delete the single
`require_transcribe_cpp_architecture(architecture)` call from that renamed function.
Rename `_validate_architecture()` to `_validate_architecture_identifier()`, retain
its exact ASCII/character grammar, give malformed identifiers the generic stable
message `GGUF general.architecture is not a valid identifier`, and call it from the
structural parser. `require_transcribe_cpp_architecture()` calls that identifier
validator before its incumbent exact allowlist check and keeps the existing
transcribe.cpp compatibility message for valid-but-unsupported architectures. Then
add this strict incumbent wrapper immediately below the parser:

```python
def inspect_gguf(handle: BinaryIO, *, file_size: int) -> GGUFMetadata:
    """Inspect one GGUF accepted by the pinned transcribe.cpp runtime."""
    metadata = inspect_gguf_structure(handle, file_size=file_size)
    require_transcribe_cpp_architecture(metadata.architecture)
    return metadata
```

Do not copy the parser body or alter any existing bounds.

- [ ] **Step 4: Write safe-handle replacement and reparse-point tests**

Add tests that:

```python
def test_open_local_gguf_rejects_symlink(tmp_path: Path):
    target = tmp_path / "model.gguf"
    target.write_bytes(make_gguf(architecture="llama"))
    link = tmp_path / "link.gguf"
    link.symlink_to(target)

    with pytest.raises(gguf.GGUFPathError, match="regular file"):
        with gguf.open_local_gguf(link):
            pytest.fail("symlink must not open")


def test_open_local_gguf_recheck_detects_same_path_replacement(tmp_path: Path):
    source = tmp_path / "model.gguf"
    replacement = tmp_path / "replacement.gguf"
    source.write_bytes(make_gguf(architecture="llama", name="first"))
    replacement.write_bytes(make_gguf(architecture="llama", name="second"))

    with gguf.open_local_gguf(source) as opened:
        source.unlink()
        replacement.rename(source)
        with pytest.raises(gguf.GGUFSourceChangedError):
            opened.recheck()
```

On Windows, add a test-local `os.lstat` result with
`stat.FILE_ATTRIBUTE_REPARSE_POINT` and assert the same fail-closed result without
requiring developer-mode symlink privileges.

- [ ] **Step 5: Implement the shared safe local handle**

Add these exact shapes:

```python
@dataclass
class OpenedLocalGGUF:
    path: Path
    handle: BinaryIO
    descriptor: int
    identity: GGUFSourceIdentity

    def recheck(self) -> None:
        """Fail if the open node or selected name changed after admission."""
        try:
            opened = _source_identity(os.fstat(self.descriptor))
            named = _source_identity(_checked_regular_source_info(self.path))
        except OSError:
            raise GGUFPathError(
                "Selected local GGUF identity could not be verified"
            ) from None
        if opened != self.identity or named != self.identity:
            raise GGUFSourceChangedError(
                "Selected local GGUF changed during validation"
            )


@contextmanager
def open_local_gguf(path: str | Path) -> Iterator[OpenedLocalGGUF]:
    """Yield one no-follow regular GGUF handle with stable identity."""
    try:
        validated = validate_path_simple(path, require_exists=False, probe_existing=False)
        selected = Path(validated).absolute()
    except (OSError, ValueError):
        raise GGUFPathError("Selected local GGUF path is invalid") from None
    if selected.suffix.casefold() != ".gguf":
        raise GGUFPathError("Selected local file must have a .gguf extension")
    initial = _checked_regular_source_info(selected)
    try:
        descriptor = os.open(selected, _read_only_no_follow_flags())
    except OSError:
        raise GGUFPathError("Selected local GGUF could not be opened safely") from None
    try:
        handle = os.fdopen(descriptor, "rb", buffering=0, closefd=False)
    except OSError:
        os.close(descriptor)
        raise GGUFPathError("Selected local GGUF could not be opened safely") from None
    try:
        opened = _source_identity(os.fstat(descriptor))
    except OSError:
        handle.close()
        os.close(descriptor)
        raise GGUFPathError(
            "Selected local GGUF identity could not be verified"
        ) from None
    try:
        if opened != _source_identity(initial) or not stat.S_ISREG(opened.mode):
            raise GGUFSourceChangedError(
                "Selected local GGUF changed during validation"
            )
        local = OpenedLocalGGUF(selected, handle, descriptor, opened)
        local.recheck()
        yield local
        local.recheck()
    finally:
        handle.close()
        os.close(descriptor)
```

Keep the setup-only `OSError` mapping before `yield`; never catch an `OSError`
raised by the caller's copy body, because `ENOSPC` and other destination failures
must retain the artifact service's stable installation-I/O classification.

Add `_checked_regular_source_info()` by moving the incumbent `lstat`, symlink,
regular-file, and stable-error mapping into one helper. Use
`getattr(info, "st_file_attributes", 0)` and
`getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)` for the Windows check. Refactor
`validate_local_gguf()` to use this context manager. Add the generic validator with
this exact policy boundary:

```python
def validate_local_gguf_structure(path: str | Path) -> LocalGGUFInspection:
    with open_local_gguf(path) as opened:
        metadata = inspect_gguf_structure(
            opened.handle,
            file_size=opened.identity.size_bytes,
        )
        opened.recheck()
        return LocalGGUFInspection(opened.path, metadata, opened.identity)


def validate_local_gguf(path: str | Path) -> LocalGGUFAdmission:
    inspected = validate_local_gguf_structure(path)
    require_transcribe_cpp_architecture(inspected.metadata.architecture)
    platform_target = normalize_platform_target(platform.system(), platform.machine())
    return LocalGGUFAdmission(
        inspected.path,
        inspected.metadata,
        inspected.source_identity,
        platform_target,
    )
```

Keep the original function's stable error classes and messages. The strict wrapper
must continue applying both architecture and wheel-platform policy.

- [ ] **Step 6: Run the complete admission suite**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/Model_Artifacts/test_gguf_admission.py
```

Expected: all existing speech admission tests and the new generic/local-handle tests pass.

- [ ] **Step 7: Mutation-check the policy split**

Temporarily make `inspect_gguf()` return `inspect_gguf_structure()` without calling
`require_transcribe_cpp_architecture()`. Re-run the policy test and the incumbent
near-miss parameterization. Expected: both fail because `llama` and existing near
misses are accepted. Restore and rerun green.

- [ ] **Step 8: Commit the structural boundary**

```bash
git add Tests/Model_Artifacts/test_gguf_admission.py tldw_chatbook/Model_Artifacts/gguf_admission.py
git commit -m "refactor(models): split generic GGUF inspection"
```

### Task 2: Admit truthful path-private local descriptors

**Files:**
- Modify: `tldw_chatbook/Model_Artifacts/service.py:523-620`
- Test: `Tests/Model_Artifacts/test_service.py`
- Test: `Tests/Model_Artifacts/test_acquisition_types.py`

**Interfaces:**
- Consumes: `ArtifactDescriptor`, `ProvenanceClass.LOCAL_INTEGRITY_RECORDED`.
- Produces: a cross-field descriptor rule for blank local source/license URLs; no new descriptor version or field.

- [ ] **Step 1: Write exact accepted and rejected descriptor tests**

```python
def test_local_integrity_descriptor_accepts_truthful_empty_urls():
    local = descriptor(
        source_url="",
        license_id="unknown",
        license_url="",
        provenance=(ProvenanceClass.LOCAL_INTEGRITY_RECORDED,),
    )

    assert ArtifactDescriptor.from_dict(local.to_dict()) == local


@pytest.mark.parametrize(
    "overrides",
    (
        {"source_url": ""},
        {
            "source_url": "",
            "license_id": "unknown",
            "license_url": "",
            "provenance": (ProvenanceClass.CHATBOOK_CURATED,),
        },
        {
            "source_url": "",
            "license_id": "cc-by-4.0",
            "license_url": "",
            "provenance": (ProvenanceClass.LOCAL_INTEGRITY_RECORDED,),
        },
    ),
)
def test_empty_urls_fail_outside_exact_local_provenance(overrides):
    with pytest.raises(ArtifactDescriptorValidationError):
        descriptor(**overrides)
```

Also assert that `file://…` and `https://local.invalid/…` are absent from the
canonical local descriptor fixture.

Add `test_uninstalled_local_integrity_descriptor_cannot_enter_download_plan` in
`test_acquisition_types.py`: place the accepted local descriptor in a real
`DictCatalog`, leave the store empty, call `ArtifactAcquisitionService.preflight()`,
and assert a stable `CatalogError` before any gating probe/fetch double is called.

- [ ] **Step 2: Run the descriptor tests and record RED**

Run the two exact nodes. Expected: the accepted local descriptor fails current URL
validation.

- [ ] **Step 3: Implement the cross-field gate**

Replace the two unconditional URL calls with:

```python
local_only = self.provenance == (
    ProvenanceClass.LOCAL_INTEGRITY_RECORDED,
)
if self.source_url:
    _validate_url("source_url", self.source_url)
elif not local_only:
    raise ArtifactDescriptorValidationError(
        "source_url may be empty only for local integrity provenance"
    )

if self.license_url:
    _validate_url("license_url", self.license_url)
elif not (local_only and self.license_id == "unknown"):
    raise ArtifactDescriptorValidationError(
        "license_url may be empty only for unknown local-import licensing"
    )
```

Do not weaken `_validate_url()` or remote descriptor parsing.

- [ ] **Step 4: Run descriptor serialization and acquisition boundary tests**

```bash
../../.venv/bin/python -m pytest -q Tests/Model_Artifacts/test_service.py -k 'descriptor or source_url or license_url'
../../.venv/bin/python -m pytest -q Tests/Model_Artifacts/test_acquisition_types.py -k 'local_integrity_descriptor'
../../.venv/bin/python -m pytest -q Tests/Model_Artifacts/test_credentials_and_boundaries.py
```

Expected: local round-trip and all incumbent remote URL/credential boundaries pass.

- [ ] **Step 5: Mutation-check provenance strictness**

Temporarily replace `local_only` with membership testing. The rejected mixed/curated
cases must fail because blank URLs become accepted. Restore and rerun green.

- [ ] **Step 6: Commit the descriptor accommodation**

```bash
git add Tests/Model_Artifacts/test_service.py Tests/Model_Artifacts/test_acquisition_types.py tldw_chatbook/Model_Artifacts/service.py
git commit -m "feat(models): admit local GGUF provenance"
```

### Task 3: Add one-copy managed GGUF import to the artifact service

**Files:**
- Modify: `tldw_chatbook/Model_Artifacts/service.py:1032-3235`
- Modify: `tldw_chatbook/Model_Artifacts/__init__.py`
- Test: `Tests/Model_Artifacts/test_service.py`

**Interfaces:**
- Consumes: `open_local_gguf()`, `inspect_gguf_structure()`, existing install staging leases, `_verify_payload()`, `_promote()`, `activate()`.
- Produces:

```python
@dataclass(frozen=True)
class LocalGGUFImportProgress:
    """Path-private progress emitted by one local GGUF import."""

    phase: Literal["copy", "inspect", "verify", "finalize"]
    file: str | None
    bytes_done: int
    bytes_total: int


@dataclass(frozen=True)
class LocalGGUFImportResult:
    """Exact managed reference and convergence outcome for an import."""

    reference: ArtifactRef
    already_installed: bool


def import_local_gguf(
    self,
    source_file: Path,
    *,
    cancelled: Callable[[], bool] = _never_cancelled,
    progress: Callable[[LocalGGUFImportProgress], None] = _ignore_local_import_progress,
) -> LocalGGUFImportResult:
```

- [ ] **Step 1: Write the end-to-end one-copy import test**

Use a real synthetic `llama` GGUF and patch `builtins.open`/`os.open` only to record
write targets, not to fake the service:

```python
def test_import_local_gguf_promotes_path_private_full_digest_artifact(tmp_path):
    source = tmp_path / "private-name.gguf"
    payload = make_gguf(architecture="llama", name="Local LLM", file_type=7)
    source.write_bytes(payload)
    before = source.stat()
    service = ModelArtifactService(tmp_path / "store")

    result = service.import_local_gguf(source)

    digest = hashlib.sha256(payload).hexdigest()
    assert result.reference.revision == f"sha256-{digest}"
    installed = service.artifact_path(result.reference)
    assert (installed / "model.gguf").read_bytes() == payload
    manifest = json.loads((installed / "manifest.json").read_text())
    rendered = json.dumps(manifest)
    assert str(source) not in rendered
    assert "file://" not in rendered
    assert "local.invalid" not in rendered
    assert source.read_bytes() == payload
    assert source.stat().st_mtime_ns == before.st_mtime_ns
```

Wrap `gguf_admission.os.open` with the real function and count calls whose resolved
path equals `source`; assert exactly one source descriptor is opened. Separately
record write targets and assert the only new payload write is a service-owned
`staging/<operation-id>/model.gguf`, never `source`.

- [ ] **Step 2: Run the exact import test and record RED**

Expected: `ModelArtifactService` has no `import_local_gguf` method.

- [ ] **Step 3: Extract incumbent install-stage primitives without behavior change**

Extract the staging creation and exact locked promotion blocks already inside
`install()` into private helpers used by both paths.

`_create_install_staging() -> tuple[Path, ArtifactOperationLease]` owns the existing
name-before-create lease sequence and returns only after `os.mkdir(staging, 0o700)`.
`_commit_verified_staging(descriptor, staging, *, cancelled: Callable[[], bool],
on_finalizing: Callable[[], None] | None = None) -> bool` owns the existing
lifecycle/per-reference lock order,
destination convergence, `_verify_payload`, manifest write, parent preparation, and
promotion. It returns `True` when an already-valid immutable destination wins and
`False` when it promotes the caller's stage. Its exact promotion tail is:

```python
_raise_if_install_cancelled(cancelled)
if on_finalizing is not None:
    on_finalizing()
self._promote(staging, destination)
```

No cancellation probe is permitted after `on_finalizing()` because that callback
marks the point of no return. Move these incumbent blocks without changing their
lock order or stable exception mapping. Run the incumbent install tests before
adding import.

- [ ] **Step 4: Implement progress/result types and deterministic descriptor builder**

Use the staged digest and bounded metadata only:

```python
def _local_gguf_descriptor(
    metadata: GGUFMetadata,
    *,
    digest: str,
    size_bytes: int,
) -> ArtifactDescriptor:
    revision = f"sha256-{digest}"
    variant = f"filetype-{metadata.file_type}" if metadata.file_type is not None else "imported"
    label = (metadata.model_name or "").strip() or f"Imported GGUF {digest[:8]}"
    reference = ArtifactRef(f"local-gguf-{digest[:16]}", revision, variant)
    return ArtifactDescriptor(
        reference=reference,
        model_id=label,
        role=ArtifactRole.ROOT,
        format=ArtifactFormat.GGUF,
        consumer="unassigned",
        model_family=metadata.architecture,
        upstream_repository="local-import",
        upstream_revision=revision,
        source_url="",
        precision=variant,
        expected_installed_bytes=size_bytes,
        license_id="unknown",
        license_url="",
        usage_notice="Imported from a user-selected local file; license and runtime compatibility are not verified.",
        runtime_name="unassigned",
        runtime_version_constraint="none",
        supported_os=("unassigned",),
        supported_architectures=("unassigned",),
        provenance=(ProvenanceClass.LOCAL_INTEGRITY_RECORDED,),
        files=(ArtifactFile("model.gguf", size_bytes, digest),),
    )
```

`metadata.model_name` has already passed `_optional_display()`'s control-character
removal and character bound in the generic parser. Strip edge whitespace here and
fall back to the digest label when no canonical display value remains; do not create
a second display sanitizer.

- [ ] **Step 5: Implement the one-copy import transaction**

The method must have this ownership shape:

```python
def import_local_gguf(
    self,
    source_file: Path,
    *,
    cancelled: Callable[[], bool] = _never_cancelled,
    progress: Callable[[LocalGGUFImportProgress], None] = _ignore_local_import_progress,
) -> LocalGGUFImportResult:
    _raise_if_install_cancelled(cancelled)
    staging, staging_lease = self._create_install_staging()
    try:
        target = staging / "model.gguf"
        with open_local_gguf(source_file) as opened, target.open("xb") as output:
            digest = hashlib.sha256()
            copied = 0
            while chunk := opened.handle.read(_LOCAL_COPY_CHUNK_BYTES):
                _raise_if_install_cancelled(cancelled)
                output.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
                progress(LocalGGUFImportProgress("copy", "model.gguf", copied, opened.identity.size_bytes))
            output.flush()
            os.fsync(output.fileno())
            opened.recheck()
        _raise_if_install_cancelled(cancelled)
        progress(LocalGGUFImportProgress("inspect", None, 0, copied))
        with target.open("rb", buffering=0) as staged:
            metadata = inspect_gguf_structure(staged, file_size=copied)
        descriptor = _local_gguf_descriptor(
            metadata,
            digest=digest.hexdigest(),
            size_bytes=copied,
        )
        progress(LocalGGUFImportProgress("verify", "model.gguf", 0, copied))
        already_installed = self._commit_verified_staging(
            descriptor,
            staging,
            cancelled=cancelled,
            on_finalizing=lambda: progress(
                LocalGGUFImportProgress("finalize", None, copied, copied)
            ),
        )
        if not already_installed:
            staging = None
        return LocalGGUFImportResult(descriptor.reference, already_installed)
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        staging_lease.release()
```

Use the incumbent `install()` `sys.exception()` cleanup/error-note pattern around
the two ownership operations shown in `finally`; do not copy those bare calls into
production. The actual implementation must keep the final cancellation check
immediately before the synchronous Finalizing callback and `_promote`; it must not
poll cancellation after Finalizing or report rollback.

The conditional `staging = None` is essential: promotion transfers ownership to the
immutable destination, while convergence leaves the losing stage operation-owned so
its `finally` removes it. Apply the same distinction when refactoring incumbent
`install()`.

- [ ] **Step 6: Run import and incumbent install suites**

```bash
../../.venv/bin/python -m pytest -q Tests/Model_Artifacts/test_service.py -k 'install or import_local_gguf or descriptor'
```

Expected: existing install behavior and new local import pass.

- [ ] **Step 7: Commit the service import**

```bash
git add Tests/Model_Artifacts/test_service.py tldw_chatbook/Model_Artifacts/service.py tldw_chatbook/Model_Artifacts/__init__.py
git commit -m "feat(models): import local GGUF artifacts"
```

### Task 4: Prove cancellation, convergence, mutation detection, and recovery

**Files:**
- Modify: `Tests/Model_Artifacts/test_service.py`
- Modify only if a test exposes a defect: `tldw_chatbook/Model_Artifacts/service.py`

**Interfaces:**
- Consumes: `ModelArtifactService.import_local_gguf()` from Task 3.
- Produces: mutation-discriminating service evidence; no new public API.

- [ ] **Step 1: Add transaction-edge tests**

Add these exact named tests and observations:

| Test | Required observation |
|---|---|
| `test_import_cancel_during_copy_removes_only_its_stage` | Flip the callback after the first real chunk; stable cancellation error, source unchanged, no new destination, only this operation's stage removed. |
| `test_import_source_mutation_before_recheck_never_promotes` | Replace the selected name after copying begins; `GGUFSourceChangedError`, no destination. |
| `test_import_same_bytes_under_two_names_returns_same_reference` | Copy identical bytes from two filenames; exact `ArtifactRef` and one manifest are equal. |
| `test_import_changed_bytes_returns_different_full_revision` | Change one payload byte while preserving valid structure; full revisions differ. |
| `test_concurrent_identical_imports_converge_on_one_manifest` | Barrier two real threads before the exact-reference lock; both return the same ref, one immutable destination remains. |
| `test_import_cancel_after_other_writer_promotes_never_deletes_destination` | Cancel the losing call after the winner promotes; winner remains byte-for-byte valid. |
| `test_import_copy_io_failure_preserves_source_and_prior_artifacts` | Parameterize `ENOSPC` and `EACCES` on the staging write; source and prior artifacts are unchanged and the stage is gone. |
| `test_reconcile_removes_only_abandoned_import_stage` | Retain one live stage lease and create one abandoned stage; reconcile deletes only the abandoned stage. |

For cancellation, flip the real callback after observing a partial staged
`model.gguf`; assert a stable `ArtifactStateError`, unchanged source bytes/mtime,
unchanged pre-existing artifact, and no operation stage. For concurrency, use two
threads and a barrier immediately before the exact-reference lock; assert one final
manifest and identical results.

- [ ] **Step 2: Run the edge tests before any corrective edit**

Expected: tests either pass with Task 3 or expose a concrete ownership gap. Fix only
the demonstrated gap and rerun each exact node.

- [ ] **Step 3: Prove the authoritative staged verification mutation**

Temporarily skip the `_verify_payload(staging, descriptor.files,
cancelled=cancelled)` call inside
`_commit_verified_staging`, mutate one staged byte after the streaming digest but
before commit, and run the corruption node. Expected: the test fails because corrupt
bytes promote. Restore verification and rerun green.

- [ ] **Step 4: Prove filename-independent identity mutation**

Temporarily add `source_file.stem` to `artifact_id`. Run the renamed-identical test.
Expected: references differ. Restore and rerun green.

- [ ] **Step 5: Prove the adjacent pre-promotion cancellation guard**

Temporarily remove the cancellation check immediately before `_promote`. Run the
pre-promotion cancellation node. Expected: the destination is published instead of
raising. Restore and rerun green.

- [ ] **Step 6: Run the complete artifact-service file**

```bash
../../.venv/bin/python -m pytest -q Tests/Model_Artifacts/test_service.py
```

- [ ] **Step 7: Commit transaction hardening tests/fixes**

```bash
git add Tests/Model_Artifacts/test_service.py tldw_chatbook/Model_Artifacts/service.py
git commit -m "test(models): harden local GGUF import"
```

### Task 5: Add intent-only Import controls and consent UI

**Files:**
- Create: `tldw_chatbook/Widgets/ModelArtifacts/local_gguf_import.py`
- Modify: `tldw_chatbook/Widgets/ModelArtifacts/__init__.py`
- Modify: `tldw_chatbook/Widgets/ModelArtifacts/install_progress.py`
- Test: `Tests/UI/test_model_artifact_widgets.py`

**Interfaces:**
- Consumes: `LocalGGUFImportProgress`, Textual `Message`, `ModalScreen`, `Button`.
- Produces: `LocalGGUFImportRequested(path: Path)`, `LocalGGUFImportControls`, and `LocalGGUFImportConsentModal(source: Path, size_bytes: int) -> bool`.

- [ ] **Step 1: Write mounted intent and consent tests**

```python
@pytest.mark.asyncio
async def test_unmanaged_import_control_posts_exact_path(tmp_path: Path):
    source = tmp_path / "outside.gguf"
    app = _ImportControlApp(source)
    async with app.run_test() as pilot:
        await pilot.click(".model-import")
    assert app.received == [source]


@pytest.mark.asyncio
async def test_local_import_modal_states_copy_original_and_compatibility_truth(tmp_path):
    source = tmp_path / "outside.gguf"
    app = _ModalApp()
    async with app.run_test() as pilot:
        await app.push_screen(LocalGGUFImportConsentModal(source, 4_194_304))
        text = "\n".join(str(widget.renderable) for widget in app.screen.query(Static))
    assert source.name in text
    assert "managed copy" in text
    assert "original" in text
    assert "License and runtime compatibility are not verified" in text
```

Define `_ImportControlApp.received` in its constructor and capture the message with
an `@on(LocalGGUFImportRequested)` method; do not add a production callback seam just
for the test.

Also assert the full selected path appears only on this modal, renders with
`markup=False`, both actions are keyboard-reachable, and Escape returns `False`.

- [ ] **Step 2: Run widget tests and record RED**

Expected: import widget/module does not exist.

- [ ] **Step 3: Implement the intent control and consent-only modal**

```python
class LocalGGUFImportRequested(Message):
    """Request consent for importing one explicitly selected GGUF."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path


class LocalGGUFImportControls(Widget):
    """Render one intent-only Import action for an unmanaged GGUF row."""

    def __init__(self, path: Path, *, pending: bool = False) -> None:
        self.path = path
        self.pending = pending
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Button("Import…", classes="model-import", disabled=self.pending)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if not self.pending:
            self.post_message(LocalGGUFImportRequested(self.path))


class LocalGGUFImportConsentModal(ModalScreen[bool]):
    """Return consent for one local managed copy without performing I/O."""

    BINDINGS = [("escape", "cancel", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical(classes="local-gguf-import-modal"):
            yield Static(self.source.name, markup=False)
            yield Static(str(self.source), markup=False)
            yield Static(format_mib(self.size_bytes), markup=False)
            yield Static(
                "Chatbook will create a managed copy. The original stays in place.",
                markup=False,
            )
            yield Static(
                "License and runtime compatibility are not verified.",
                markup=False,
            )
            with Horizontal(classes="model-install-actions"):
                yield Button("Cancel", id="local-gguf-import-cancel")
                yield Button(
                    "Import",
                    id="local-gguf-import-confirm",
                    variant="primary",
                )
```

Store `source` and `size_bytes` in `__init__`; use the incumbent modal's button
handler and `action_cancel()` pattern so the screen only dismisses `True` or `False`.

- [ ] **Step 4: Extend the stable progress widget with import phases**

Add exact labels:

```python
_PHASE_LABELS.update(
    {
        "copy": "Copying model into Chatbook",
        "inspect": "Checking GGUF structure",
        "verify": "Verifying managed copy",
        "finalize": "Finalizing managed model",
    }
)
```

Treat `copy` as a byte phase. The other phases hide the determinate bar. Change the
type annotation to accept `AcquisitionProgress | LocalGGUFImportProgress`; do not add
a second progress widget.

- [ ] **Step 5: Run all shared widget tests**

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_model_artifact_widgets.py
```

- [ ] **Step 6: Commit the reusable UI boundary**

```bash
git add Tests/UI/test_model_artifact_widgets.py tldw_chatbook/Widgets/ModelArtifacts
git commit -m "feat(models): add local GGUF import controls"
```

### Task 6: Orchestrate Import in the Installed view

**Files:**
- Modify: `tldw_chatbook/UI/Screens/model_browser_state.py:31-58,180-260`
- Modify: `tldw_chatbook/UI/Screens/model_installed_view.py:1-680`
- Test: `Tests/UI/test_model_browser_state.py`
- Test: `Tests/UI/test_model_installed_view.py`

**Interfaces:**
- Consumes: Task 3 service API and Task 5 intent/modal/progress types.
- Produces: one Installed-view import lane with generation fencing, a real cancellation event, activation recovery, and no preference mutation.

- [ ] **Step 1: Write pure inventory-state tests**

Change the unmanaged action expectation from the placeholder to Import and prove the
managed store is excluded from legacy results:

```python
def test_unmanaged_gguf_row_offers_import_without_managed_reference(tmp_path: Path):
    source = tmp_path / "outside.gguf"
    source.write_bytes(b"x" * 1_048_577)
    usage = ArtifactDiskUsage(0, 0, 0)
    row = inventory_rows((), usage, (UnmanagedRow(source, source.stat().st_size),))[0]
    assert row.is_unmanaged is True
    assert row.reference is None
    assert row.action_hint == "Outside Chatbook · integrity unknown"


def test_scan_unmanaged_excludes_managed_artifacts_root(tmp_path: Path):
    legacy_root = tmp_path / "legacy"
    store = ModelArtifactService(legacy_root / "managed")
    managed_payload = store.artifacts_path / "local" / "model.gguf"
    managed_payload.parent.mkdir(parents=True)
    managed_payload.write_bytes(b"x" * 1_048_577)
    rows = InstalledView.scan_unmanaged(legacy_root, excluded_root=store.artifacts_path)
    assert managed_payload not in {row.path for row in rows}
```

- [ ] **Step 2: Write mounted picker/consent/worker tests before production edits**

Use a real mounted `InstalledView`, a fake service with the real method signatures,
and an app that records screen callbacks. Add these exact nodes:

- `test_header_and_unmanaged_row_open_real_gguf_picker`: both controls open
  `EnhancedFileOpen` restricted to `.gguf` and feed only the returned path onward.
- `test_declined_consent_performs_no_service_call`: decline leaves the original row
  and records zero import/activation calls.
- `test_import_progress_updates_without_replacing_focused_cancel`: two byte events
  update painted counts while `query_one("#installed-gguf-import-cancel")` retains
  object identity and focus.
- `test_physical_cancel_sets_service_probe_and_preserves_source`: Enter on the real
  Cancel button makes the service callback true; source bytes and mtime stay equal.
- `test_finalizing_disables_cancel_before_promotion`: a synchronous Finalizing
  callback paints the label and disables Cancel before the fake promotion gate opens.
- `test_import_success_activates_but_does_not_change_source_preference`: exact ref is
  activated once; preference/source callbacks remain untouched.
- `test_activation_failure_keeps_installed_row_and_offers_activate`: refresh shows
  the installed-but-unready artifact with incumbent recovery activation control.
- `test_stale_import_callback_cannot_replace_newer_status`: generation N completion
  cannot replace generation N+1 status.
- `test_import_failure_logs_only_stable_category_and_never_selected_path`: Loguru
  sink and notifications contain the stable category but not the sentinel path.
- `test_cancelled_and_failed_import_offer_retry_and_choose_another`: both outcomes
  paint path-free recovery copy; Retry reuses only the retained in-memory selection,
  Choose another opens a new picker, and neither control paints the absolute path.

The fake must implement the exact public signatures verified with
`inspect.signature(ModelArtifactService.import_local_gguf)` and
`inspect.signature(ModelArtifactService.activate)`.

- [ ] **Step 3: Run the new Installed-view selection and orchestration nodes**

Expected: failures for missing buttons, intent handler, state, and worker.

- [ ] **Step 4: Add retained import state and stable controls**

Initialize:

```python
self._import_generation = 0
self._import_active = False
self._import_cancelable = False
self._import_cancel_event: threading.Event | None = None
self._import_progress: LocalGGUFImportProgress | None = None
self._pending_import_path: Path | None = None
self._import_status: str | None = None
self._import_retry_available = False
```

Compose a header `Import GGUF…` button, unmanaged-row
`LocalGGUFImportControls`, the existing `ModelInstallProgress`, and exactly one
physical `Cancel import` button while active. Disable Refresh, Repair, row lifecycle,
and new Import controls during the import lane. Update progress in place when active
and cancelability do not change. After cancellation or a pre-promotion failure,
replace Cancel with path-free `Retry` and `Choose another file` controls. After
success paint `Imported and ready` or `Already imported`; after activation failure
paint `Installed — activation required` and rely on the incumbent row Activate action.

- [ ] **Step 5: Add picker and consent routing**

Use the incumbent `EnhancedFileOpen` with a `.gguf` filter. The picker callback must
verify the view is still attached and no newer generation owns the lane before
opening `LocalGGUFImportConsentModal`. Retain the selected `Path` only in this mounted
view through consent, active work, and retryable cancellation/failure. Clear it on
decline, success, activation-required completion, Choose another, or unmount. Never
render it outside the consent modal and never persist it.

- [ ] **Step 6: Add the cancellable worker and fenced callbacks**

```python
@work(
    thread=True,
    group="installed_gguf_import",
    exclusive=True,
    exit_on_error=False,
    description="Importing local GGUF model",
)
def _import_local_gguf(
    self,
    generation: int,
    source: Path,
    cancel_event: threading.Event,
) -> None:
    try:
        result = self._service_for_worker().import_local_gguf(
            source,
            cancelled=cancel_event.is_set,
            progress=lambda event: self.app.call_from_thread(
                self._apply_import_progress,
                generation,
                event,
            ),
        )
    except Exception as error:
        self.app.call_from_thread(
            self._apply_import_failure,
            generation,
            (
                "Import cancelled. The original file and prior models are unchanged."
                if cancel_event.is_set()
                else local_import_failure_message(error)
            ),
            cancel_event.is_set(),
        )
        return
    try:
        self._service_for_worker().activate(result.reference)
    except Exception:
        self.app.call_from_thread(
            self._apply_import_activation_required,
            generation,
            result.reference,
        )
        return
    self.app.call_from_thread(
        self._apply_import_success,
        generation,
        result,
    )
```

`local_import_failure_message()` maps typed path, parse/bounds/version, integrity,
cancellation, contention, and generic failures to fixed copy. It never formats the
exception. Logging records only `error_type=type(error).__name__` and phase without
`logger.opt(exception=True)`.

The three terminal callbacks verify attachment and generation before touching state.
Success and activation-required callbacks clear the retained path and force one
inventory reload so the immutable row becomes visible. Failure records whether the
event was cancellation, retains the path only for Retry, and recomposes once to add
the two recovery controls.

Cancel keeps the owning generation, sets its event, paints “Cancelling import…”, and
keeps the stable control until that worker reports cancellation. Starting a
replacement import first sets the prior event and then increments the generation so
late callbacks cannot settle the new lane. `on_unmount` sets the event, clears the
retained path, and invalidates the generation without touching widgets.

- [ ] **Step 7: Run pure and mounted Installed-view files**

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_model_browser_state.py Tests/UI/test_model_installed_view.py
```

- [ ] **Step 8: Mutation-check focus, fencing, and preference isolation**

Perform and restore these mutations individually:

1. Recompose on every byte event — focused Cancel identity test fails.
2. Remove generation equality — stale-result test fails.
3. Ignore `cancel_event` in service call — physical-cancel test fails.
4. Call any runtime-preference callback after activation — preference-isolation test fails.
5. Log the raw exception — path-sentinel log test fails.

- [ ] **Step 9: Commit Installed-view orchestration**

```bash
git add Tests/UI/test_model_browser_state.py Tests/UI/test_model_installed_view.py tldw_chatbook/UI/Screens/model_browser_state.py tldw_chatbook/UI/Screens/model_installed_view.py
git commit -m "feat(models): import GGUF from Installed view"
```

### Task 7: Run the complete PR gate and close TASK-2062.1

**Files:**
- Modify through Backlog CLI: `backlog/tasks/task-2062.1 - Import-local-GGUF-files-into-the-managed-artifact-store.md`
- Verify all files listed in the responsibility map.

**Interfaces:**
- Consumes: completed Tasks 1-6.
- Produces: a reviewable TASK-2062.1 PR with no TASK-2062.2 or TASK-2062.3 behavior.

- [ ] **Step 1: Run the exact focused test union**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Model_Artifacts/test_gguf_admission.py \
  Tests/Model_Artifacts/test_service.py \
  Tests/Model_Artifacts/test_acquisition_types.py \
  Tests/Model_Artifacts/test_credentials_and_boundaries.py \
  Tests/UI/test_model_browser_state.py \
  Tests/UI/test_model_artifact_widgets.py \
  Tests/UI/test_model_installed_view.py
```

Expected: all pass. The only permissible skip is a test explicitly limited to a
different native platform.

- [ ] **Step 2: Run a real production-CSS 80-column finish slice**

Run the mounted nodes that prove:

- the header Import and unmanaged-row Import actions are in bounds;
- the consent copy and selected filename are painted;
- byte progress preserves the focused Cancel widget identity;
- physical Enter activates Cancel;
- terminal state restores focus to Import or the imported row;
- selected path never appears in notifications, worker descriptions, or captured logs.

Use `TldwCli.CSS_PATH`; do not substitute isolated widget CSS as the geometry oracle.

- [ ] **Step 3: Verify native supported-platform evidence**

Push the implementation commit and run the repository's applicable PR checks on
Linux, macOS, and Windows. Record the exact job URLs and outcomes for the GGUF
admission, artifact-service, and mounted Installed-view nodes. The Windows
reparse-point node and each platform's replacement/cancellation/cleanup coverage must
execute rather than skip; do not close the task on a red or missing supported lane.

- [ ] **Step 4: Run static and privacy gates**

```bash
../../.venv/bin/ruff check \
  tldw_chatbook/Model_Artifacts/gguf_admission.py \
  tldw_chatbook/Model_Artifacts/service.py \
  tldw_chatbook/Model_Artifacts/__init__.py \
  tldw_chatbook/Widgets/ModelArtifacts \
  tldw_chatbook/UI/Screens/model_browser_state.py \
  tldw_chatbook/UI/Screens/model_installed_view.py \
  Tests/Model_Artifacts/test_gguf_admission.py \
  Tests/Model_Artifacts/test_service.py \
  Tests/Model_Artifacts/test_acquisition_types.py \
  Tests/UI/test_model_browser_state.py \
  Tests/UI/test_model_artifact_widgets.py \
  Tests/UI/test_model_installed_view.py

../../.venv/bin/ruff format --check \
  tldw_chatbook/Widgets/ModelArtifacts/local_gguf_import.py

../../.venv/bin/python -m py_compile \
  tldw_chatbook/Model_Artifacts/gguf_admission.py \
  tldw_chatbook/Model_Artifacts/service.py \
  tldw_chatbook/UI/Screens/model_installed_view.py \
  tldw_chatbook/Widgets/ModelArtifacts/local_gguf_import.py

git diff --check
```

For legacy files that retain baseline formatter debt, run Ruff format checks over
each changed range and record the commands/results rather than formatting unrelated
lines.

- [ ] **Step 5: Run final source/privacy scans**

Confirm the added production lines contain no `file://`, `local.invalid`, raw selected
path logging, dynamic Textual worker description, new network client, new dependency,
runtime preference mutation, vLLM/MLX edit, or downloader removal. Confirm
`_deferred_gguf_managed_import.py` remains dormant and unexported.

- [ ] **Step 6: Review the complete diff against TASK-2062.1 only**

Reject edits that implement runtime source selectors or remove legacy downloaders.
Verify source open/copy/cleanup, staging ownership, activation recovery, and every
user-visible claim directly against the approved spec.

- [ ] **Step 7: Update TASK-2062.1 through Backlog CLI**

After all gates pass:

```bash
backlog task edit 2062.1 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-ac 6
backlog task edit 2062.1 --notes '<concise implementation summary, verification commands/results, mutation evidence, ADR-025 link, and modified files>'
backlog task edit 2062.1 -s Done
```

Replace the quoted notes argument with the concrete results from this implementation;
do not mark Done if any Definition-of-Done item remains open.

- [ ] **Step 8: Commit the task closeout**

```bash
git add 'backlog/tasks/task-2062.1 - Import-local-GGUF-files-into-the-managed-artifact-store.md'
git commit -m "docs(models): close task 2062.1"
```

- [ ] **Step 9: Stop at the PR boundary**

Open/review TASK-2062.1 independently. Do not begin TASK-2062.2 until this child is
approved and its service interfaces are stable on the target branch; write the
TASK-2062.2 implementation plan against that merged baseline rather than reviving the
deleted monolithic plan.
