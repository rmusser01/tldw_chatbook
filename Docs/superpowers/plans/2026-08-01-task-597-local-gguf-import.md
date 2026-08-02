# TASK-597 Bounded Local GGUF Import Implementation Plan

> **SUPERSEDED 2026-08-02:** Do not execute this store-first plan. The user
> approved direct local GGUF paths before managed acquisition. See
> `Docs/superpowers/specs/2026-08-01-task-597-local-gguf-import-design.md` and
> ADR-040. A replacement implementation plan will be written after the revised
> spec completes review.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user explicitly select a compatible local GGUF, inspect it safely, copy and verify it in managed storage, and register an exact manual-only transcribe.cpp artifact from the Installed model view.

**Architecture:** Add one Textual-free `gguf_import.py` module that owns bounded GGUF admission, source identity, managed copying, descriptor creation, and orchestration over the existing `ModelArtifactService`. Make one narrow staging-GC addition to the core, then connect the importer to the existing Installed view and progress widget without changing semantic STT routing or loading a native runtime.

**Tech Stack:** Python 3.11+, stdlib `struct`/`os`/`hashlib`/`threading`, existing model-artifact service and leases, Textual, pytest/Pilot, Ruff, mypy.

---

## Preconditions and authority

- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-597-local-gguf-import` on `codex/task-597-local-gguf-import`.
- Read `Docs/superpowers/specs/2026-08-01-task-597-local-gguf-import-design.md` before implementation.
- Governing ADR: `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`.
- ADR required: no.
- ADR path: `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`.
- Reason: this directly implements ADR-025's managed local GGUF import and curated-only routing boundary.
- Use `superpowers:test-driven-development` for every production change.
- Do not implement TASK-604's provider, catalog, inference, settings, or evaluation.
- Do not touch the user's dirty main checkout or unrelated worktrees.

Before Task 1, after this plan passes plan-document review, record the implementation plan on TASK-597 with Backlog CLI:

```bash
backlog task edit 597 --plan "Plan: Docs/superpowers/plans/2026-08-01-task-597-local-gguf-import.md\n\nADR required: no\nADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md\nReason: direct implementation of ADR-025's approved managed local GGUF import boundary.\n\n1. Add bounded GGUF v3 parsing and pinned transcribe.cpp admission.\n2. Add content-derived descriptors and curated matching.\n3. Add lease-protected staging, copy/hash, cancellation, immutable install, and activation.\n4. Add Installed-view selection, progress, cancellation, and precise errors.\n5. Verify focused/regression/static gates and complete review." --plain
```

Do not begin production implementation until the task file contains this plan.

## File map

### Create

- `tldw_chatbook/Model_Artifacts/gguf_import.py` — bounded reader, pinned compatibility, typed values/errors, source identity, managed copy, descriptor construction, and orchestration.
- `Tests/Model_Artifacts/gguf_test_helpers.py` — deterministic GGUF v3 fixtures shared by TASK-597 tests.
- `Tests/Model_Artifacts/test_gguf_import.py` — parser, compatibility, path/TOCTOU, disk, staging, provenance, cancellation, installation, and activation tests.

### Modify

- `tldw_chatbook/Model_Artifacts/service.py` — recognize and reconcile abandoned `local-import-*` stages under the acquisition-session lease.
- `Tests/Model_Artifacts/test_reconcile_staging_gc.py` — live preservation, abandoned removal, symlink containment, and unrelated-entry tests.
- `tldw_chatbook/Widgets/ModelArtifacts/install_progress.py` — structural progress display and local-import phases.
- `Tests/UI/test_model_artifact_widgets.py` — new progress phases plus acquisition regressions.
- `tldw_chatbook/UI/Screens/model_browser_state.py` — GGUF-specific unmanaged guidance.
- `tldw_chatbook/UI/Screens/model_installed_view.py` — picker, worker, cancellation, progress, errors, notifications, and refresh.
- `Tests/UI/test_model_installed_view.py` — the complete user-facing import flow.
- `backlog/tasks/task-597 - Add-bounded-local-GGUF-artifact-import.md` — plan, completed ACs, notes, evidence, and Done status only after all gates pass.

## Public contracts

Keep the surface small. Internal helper names may move during TDD, but behavior should converge on:

```python
GGUF_VERSION = 3
TRANSCRIBE_CPP_VERSION = "0.1.3"
LOCAL_IMPORT_SPACE_MARGIN_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class GGUFMetadata:
    architecture: str
    variant: str | None
    model_name: str | None
    file_type: int | None
    data_offset: int


@dataclass(frozen=True)
class GGUFImportProgress:
    phase: Literal["copy", "verify", "install", "activate"]
    file: str | None
    bytes_done: int
    bytes_total: int


@dataclass(frozen=True)
class GGUFImportResult:
    reference: ArtifactRef
    descriptor: ArtifactDescriptor


def inspect_gguf(handle: BinaryIO, *, file_size: int) -> GGUFMetadata: ...


def import_local_gguf(
    selected_path: Path,
    *,
    service: ModelArtifactService,
    curated_descriptors: Iterable[ArtifactDescriptor] = (),
    cancel: threading.Event | None = None,
    progress: Callable[[GGUFImportProgress], None] | None = None,
    free_bytes_probe: Callable[[Path], int] | None = None,
) -> GGUFImportResult: ...
```

Typed errors cover unsafe selection, malformed/excessive GGUF, unsupported version/architecture/platform, insufficient space, busy operation, source mutation, cancellation, ambiguous curated matches, installation, and activation carrying the installed reference.

## Task 1: Build deterministic GGUF fixtures and the bounded reader

**Files:**

- Create: `Tests/Model_Artifacts/gguf_test_helpers.py`
- Create: `Tests/Model_Artifacts/test_gguf_import.py`
- Create: `tldw_chatbook/Model_Artifacts/gguf_import.py`

- [ ] **Step 1: Write the fixture builder and first failing reader test**

Build only real v3 structures needed by tests:

```python
def gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def make_gguf(
    *,
    architecture: str = "whisper",
    variant: str | None = None,
    name: str | None = None,
    file_type: int | None = 7,
    tensors: tuple[TensorFixture, ...] = (),
    extra_metadata: tuple[MetadataFixture, ...] = (),
) -> bytes:
    # magic/version/counts, KVs, tensor infos, alignment padding, tiny data.
```

First assertion:

```python
def test_inspect_gguf_reads_supported_identity_without_tensor_payload(tmp_path):
    payload = make_gguf(architecture="whisper", variant="small", name="Whisper Small")
    path = tmp_path / "model.gguf"
    path.write_bytes(payload)
    with path.open("rb") as handle:
        metadata = inspect_gguf(handle, file_size=len(payload))
    assert metadata.architecture == "whisper"
    assert metadata.variant == "small"
    assert metadata.model_name == "Whisper Small"
```

- [ ] **Step 2: Run the reader test and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_gguf_import.py::test_inspect_gguf_reads_supported_identity_without_tensor_payload -v
```

Expected: FAIL because `gguf_import`/`inspect_gguf` does not exist.

- [ ] **Step 3: Implement only the happy-path cursor and v3 reader**

Use a cursor that budgets every read before it occurs:

```python
class _GGUFCursor:
    def read_exact(self, size: int) -> bytes:
        if size < 0 or self.header_bytes + size > MAX_HEADER_BYTES:
            raise GGUFBoundsError("GGUF header exceeds inspection limit")
        data = self.handle.read(size)
        if len(data) != size:
            raise GGUFParseError("GGUF header is truncated")
        self.header_bytes += size
        return data

    def unpack(self, fmt: str) -> tuple[object, ...]:
        return struct.unpack(fmt, self.read_exact(struct.calcsize(fmt)))
```

Implement only enough v3 scalar/string metadata and tensor-info traversal to parse the first valid fixture while budgeting each `read_exact` call. Retain `general.architecture`, `stt.variant`, `general.name`, `general.file_type`, and internal `general.alignment`. Never read tensor bytes. Do not add the complete rejection/limit matrix until its failing tests exist.

- [ ] **Step 4: Run the first test and verify GREEN**

Run Step 2 again. Expected: PASS.

- [ ] **Step 5: Add the complete parser-boundary matrix**

Parameterize magic/version, truncation at each structural section, all approved limits, unknown value type, invalid UTF-8, duplicate required key, bad alignment, data offset beyond EOF, display sanitization, and a handle that raises if the reader crosses `data_offset`.

Assert typed errors, never raw `struct.error`, `UnicodeDecodeError`, `MemoryError`, or unbounded allocation.

- [ ] **Step 6: Run the boundary matrix and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_gguf_import.py -k 'inspect or parser or bounds' -v
```

Expected: the new malformed/excessive cases FAIL against the happy-path reader.

- [ ] **Step 7: Implement the tested parser boundaries minimally**

Add the remaining v3 scalar types, homogeneous arrays, strict retained-field typing, duplicate-key rejection, alignment/data-offset validation, and every named count/length/nesting/header budget. Check sizes before allocation, multiplication, or iteration. Convert decoding/struct failures to the typed public errors without leaking raw exceptions.

- [ ] **Step 8: Run parser tests and verify GREEN**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_gguf_import.py -k 'inspect or parser or bounds' -v
```

Expected: all selected tests PASS.

- [ ] **Step 9: Commit the bounded reader**

```bash
git add Tests/Model_Artifacts/gguf_test_helpers.py Tests/Model_Artifacts/test_gguf_import.py tldw_chatbook/Model_Artifacts/gguf_import.py
git commit -m "feat: add bounded GGUF metadata reader"
```

## Task 2: Pin runtime compatibility and deterministic descriptors

**Files:**

- Modify: `tldw_chatbook/Model_Artifacts/gguf_import.py`
- Modify: `Tests/Model_Artifacts/test_gguf_import.py`

- [ ] **Step 1: Write failing architecture and platform tests**

Assert the 16 exact v0.1.3 names are accepted; reject near misses `cohere`, `granite`, `qwen3-asr`, and `llama`. Test these exact wheel pairs:

```python
WHEEL_TARGETS = {
    ("linux", "x86_64"),
    ("linux", "aarch64"),
    ("windows", "x86_64"),
    ("darwin", "arm64"),
    ("darwin", "x86_64"),
}
```

Reject Windows arm64 and unknown system/machine values before staging.

- [ ] **Step 2: Run compatibility tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_gguf_import.py -k 'architecture or platform' -v
```

Expected: FAIL until the declaration/normalizer exists.

- [ ] **Step 3: Implement exact compatibility declarations**

Define architecture and wheel-pair `frozenset` values plus a pure platform normalizer. Keep them import-safe and free of native probing. `inspect_gguf` rejects architectures outside the declaration.

- [ ] **Step 4: Write failing curated/local descriptor tests**

Cover exact eligible curated reuse; an exact `==0.1.3` constraint; a compatible range such as `>=0.1,<0.2`; incompatible and malformed constraints falling back to local; ineligible digest matches falling back to local; ambiguous eligible matches; deterministic content identity; local-only provenance; no embedded license/path persistence; numeric `filetype-<n>` or `unknown`; current platform pair only; and no dependencies.

- [ ] **Step 5: Run descriptor tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_gguf_import.py -k 'descriptor or curated or provenance' -v
```

Expected: FAIL until descriptor selection/building exists.

- [ ] **Step 6: Implement descriptor selection/building**

```python
source_url = "https://local.invalid/gguf-import"
license_url = "https://local.invalid/noassertion"
artifact_id = f"local-gguf-{metadata.architecture}-{sha256[:16]}"
revision = sha256
precision = f"filetype-{metadata.file_type}" if metadata.file_type is not None else "unknown"
```

Sanitize/cap the label and fall back to `f"{architecture} local GGUF"`. Curated eligibility requires root/GGUF/transcribe.cpp/one file/no dependencies/exact size+digest and a runtime constraint that admits pinned version `0.1.3`; never compare the constraint text for equality.

Keep constraint evaluation Textual/native/dependency-free with a small pure helper for the registry's bounded release grammar: comma-separated `==`, `!=`, `<`, `<=`, `>`, `>=`, or `~=` clauses over one-to-three decimal release components. Trim clause whitespace, evaluate all clauses deterministically against `(0, 1, 3)`, and treat empty, unknown-operator, nonnumeric, or otherwise malformed constraints as ineligible rather than raising or reusing a curated descriptor.

- [ ] **Step 7: Run compatibility and descriptor tests**

Run Steps 2 and 5. Expected: all selected tests PASS.

- [ ] **Step 8: Commit compatibility and identity**

```bash
git add Tests/Model_Artifacts/test_gguf_import.py tldw_chatbook/Model_Artifacts/gguf_import.py
git commit -m "feat: declare compatible local GGUF artifacts"
```

## Task 3: Add contained local-import staging reconciliation

**Files:**

- Modify: `tldw_chatbook/Model_Artifacts/service.py`
- Modify: `Tests/Model_Artifacts/test_reconcile_staging_gc.py`

- [ ] **Step 1: Write failing staging-GC tests**

```python
def test_abandoned_local_import_stage_is_removed_and_reported(service): ...
def test_live_local_import_stage_survives_while_acquisition_lease_is_held(service): ...
def test_local_import_symlink_is_unlinked_without_touching_target(service, tmp_path): ...
def test_unrecognized_staging_entry_remains_untouched(service): ...
```

The live test holds `ACQUISITION_SESSION_LEASE_KEY`; after release, a second reconciliation removes the stage.

- [ ] **Step 2: Run the four tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_reconcile_staging_gc.py -k 'local_import or unrecognized' -v
```

Expected: abandoned `local-import-*` is not removed yet.

- [ ] **Step 3: Implement one prefix and lease-gated collector**

Add `_LOCAL_IMPORT_STAGE_PREFIX = "local-import-"`. Collect matching top-level entries in `_gc_staging`. `_gc_local_import_staging` acquires the global session lease non-blocking; busy returns `()`, free removes only collected contained entries with `_remove_state_path`. Do not change install/download/managed/unknown handling.

- [ ] **Step 4: Run all staging reconciliation tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_reconcile_staging_gc.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit staging ownership**

```bash
git add Tests/Model_Artifacts/test_reconcile_staging_gc.py tldw_chatbook/Model_Artifacts/service.py
git commit -m "feat: reconcile abandoned local GGUF imports"
```

## Task 4: Implement the secure source and managed-copy boundary

**Files:**

- Modify: `tldw_chatbook/Model_Artifacts/gguf_import.py`
- Modify: `Tests/Model_Artifacts/test_gguf_import.py`

- [ ] **Step 1: Write failing unsafe-source tests**

Cover `.gguf` suffix, `validate_path_simple(..., probe_existing=False)`, missing file, directory, FIFO/irregular file where supported, final symlink, and pre-open replacement. Use snapshot `(st_dev, st_ino, file type, size, mtime_ns, ctime_ns)` and assert errors contain no selected path.

- [ ] **Step 2: Run unsafe-source tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_gguf_import.py -k 'path or symlink or irregular or replacement' -v
```

Expected: FAIL until the importer opens/pins the source.

- [ ] **Step 3: Implement no-follow open and pinned identity**

Apply the central validator without resolving the final component; `lstat`; reject non-regular/symlink; `os.open` with `O_RDONLY`, optional `O_BINARY`, optional `O_NOFOLLOW`; compare `fstat`; retain that descriptor through parse/copy. Normalize OS errors without path text.

- [ ] **Step 4: Write failing disk/lease/copy/cancel/containment tests**

Cover first/second `size + 64 MiB` space probes; session-lease contention; 0700 stage; one-MiB chunks; `copy` progress; cancellation before/during/after copy but before commit; truncate/append/in-place mutation; `ENOSPC`; staged reparse failure; and cleanup limited to the owned stage. Add deterministic substitution cases for both the operation directory and `payload` directory: replace each with a symlink and with a different inode after identity capture, assert import fails, and assert an external marker is neither written nor removed. Use injected filesystem hooks/probes/callbacks, not sleeps or huge files.

- [ ] **Step 5: Run copy tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_gguf_import.py -k 'space or lease or copy or cancel or mutation or cleanup' -v
```

Expected: FAIL until staging/copy orchestration exists.

- [ ] **Step 6: Implement identity-pinned pre-commit staging and hashing**

Acquire `ArtifactOperationLease(service.locks_path, ACQUISITION_SESSION_LEASE_KEY, EXCLUSIVE, timeout=NONBLOCKING_LEASE_TIMEOUT_SECONDS)` before the second space probe. Create a fresh `local-import-<uuid>` with `os.mkdir(..., 0o700)` and a fresh `payload` directory, rejecting pre-existence. Capture non-following stat identities for the staging root, operation directory, and payload directory; require real directories and compare those identities before every pathname-based create, handoff, or cleanup.

Where supported, open the operation/payload directories with `O_DIRECTORY | O_NOFOLLOW`, compare `fstat` with the captured identity, and create `model.gguf` relative to the trusted directory handle. On platforms without directory-descriptor operations, perform the same non-following identity checks immediately before and after the exclusive file create. Open `model.gguf` with `O_CREAT | O_EXCL | O_WRONLY` plus `O_NOFOLLOW`/`O_BINARY` where available, retain that descriptor through copy/hash/fsync, and verify it remains the captured regular file.

Copy from the pinned source descriptor with SHA-256/progress, flush/fsync, compare the final source snapshot, reparse the staged descriptor, compare admission identity, and check cancellation immediately before finalization.

Cleanup must be an exact bottom-up unlink/rmdir of `model.gguf`, `payload`, then the operation directory after non-following identity/type checks; never recursively follow or remove a substituted node. If any operation/payload/file identity changed, stop cleanup, report a containment error without path text, and leave the suspect entry for lease-gated reconciliation. Release the lease while preserving the primary exception.

- [ ] **Step 7: Run source/copy tests**

Run Steps 2 and 5. Expected: all selected tests PASS.

- [ ] **Step 8: Commit the secure copy boundary**

```bash
git add Tests/Model_Artifacts/test_gguf_import.py tldw_chatbook/Model_Artifacts/gguf_import.py
git commit -m "feat: stage local GGUF files safely"
```

## Task 5: Finalize through immutable install and exact activation

**Files:**

- Modify: `tldw_chatbook/Model_Artifacts/gguf_import.py`
- Modify: `Tests/Model_Artifacts/test_gguf_import.py`

- [ ] **Step 1: Write failing end-to-end importer tests**

With real `ModelArtifactService`, assert uncurated install/activation/local manifest; curated exact reuse; duplicate idempotency; core detection of post-hash corruption; unchanged source; no path/embedded-license persistence; no semantic STT defaults; last cancellation check before install; and activation failure retaining a complete inactive artifact while carrying its ref.

- [ ] **Step 2: Run finalization tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_gguf_import.py -k 'end_to_end or installs or activation or idempotent or corruption' -v
```

Expected: FAIL until finalization delegates to the core.

- [ ] **Step 3: Implement the commit point**

After the last cancellation check: select/rewrite curated payload path; emit `install`; call `service.install(descriptor, payload_directory, consume_source=True)`; emit `activate`; call `service.activate(reference)`; return `GGUFImportResult`.

Translate activation failure to `GGUFActivationError(reference)` without deleting the installed artifact. Preserve typed causes for logs while keeping public messages path-free.

- [ ] **Step 4: Run the complete importer suite**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_gguf_import.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Run import-boundary regression**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_credentials_and_boundaries.py Tests/Model_Artifacts/test_service.py -k 'never_import or package_import' -v
```

Expected: PASS; local import does not pull acquisition/fetch/httpx/native STT into worker imports.

- [ ] **Step 6: Commit backend finalization**

```bash
git add Tests/Model_Artifacts/test_gguf_import.py tldw_chatbook/Model_Artifacts/gguf_import.py
git commit -m "feat: install and activate imported GGUF artifacts"
```

## Task 6: Generalize the progress display

**Files:**

- Modify: `tldw_chatbook/Widgets/ModelArtifacts/install_progress.py`
- Modify: `Tests/UI/test_model_artifact_widgets.py`

- [ ] **Step 1: Write failing structural-progress tests**

Use a test value with `phase/file/bytes_done/bytes_total` but no `ArtifactRef`. Assert `copy` is determinate and `verify/install/activate` labels hide the bar; existing `AcquisitionProgress("fetch", ...)` remains unchanged.

- [ ] **Step 2: Run widget tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_model_artifact_widgets.py -k 'progress' -v
```

Expected: local phases fail or the type contract still requires acquisition progress.

- [ ] **Step 3: Implement a structural display protocol**

```python
class ModelProgressDisplay(Protocol):
    phase: str
    file: str | None
    bytes_done: int
    bytes_total: int
```

Use it only for `ModelInstallProgress.__init__`/`update_progress`. Keep messages and callback acquisition-specific. Add labels and include only `fetch`, `pre-verify`, and `copy` in determinate phases.

- [ ] **Step 4: Run affected consumers**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_model_artifact_widgets.py Tests/UI/test_model_installed_view.py Tests/UI/test_model_remote_view.py Tests/UI/test_parakeet_v2_install_ui.py Tests/Wizards/test_first_run_speech_step.py -q
```

Expected: all tests PASS.

- [ ] **Step 5: Commit progress support**

```bash
git add Tests/UI/test_model_artifact_widgets.py tldw_chatbook/Widgets/ModelArtifacts/install_progress.py
git commit -m "feat: display local model import progress"
```

## Task 7: Add the Installed-view import flow

**Files:**

- Modify: `tldw_chatbook/UI/Screens/model_browser_state.py`
- Modify: `tldw_chatbook/UI/Screens/model_installed_view.py`
- Modify: `Tests/UI/test_model_installed_view.py`

- [ ] **Step 1: Write failing state and picker tests**

Assert unmanaged `.gguf` guidance; unchanged other-format guidance; header button `#installed-models-import-gguf`; `EnhancedFileOpen` with callable `.gguf` filter and `context="model_gguf_import"`; cancel starts no worker; and composition/mount calls neither service, registry, importer, nor scan.

- [ ] **Step 2: Run picker tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_model_installed_view.py -k 'unmanaged or picker or compose' -v
```

Expected: FAIL until button/callback/hint exists.

- [ ] **Step 3: Implement picker and idle state**

Use `Filters(("GGUF models", lambda path: path.suffix.casefold() == ".gguf"))`. Inject `registry_factory=curated_registry` and `importer=import_local_gguf`, called only in the worker. A selected path sets `_operation_name="import"`, creates a fresh `threading.Event`, clears progress, recomposes, and starts the worker.

- [ ] **Step 4: Write failing operation-state tests**

Cover path only passed to worker, never rendered/logged; registry/importer off-loop; progress through `call_from_thread` without row recompose; Cancel state and commit-phase disabling; lifecycle reentry fencing; sanitized typed errors; provider-setup-required success; activation-retry failure; and terminal state cleanup/forced refresh.

- [ ] **Step 5: Run operation tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_model_installed_view.py -k 'import or cancel or progress or failure' -v
```

Expected: FAIL until worker/state/result mapping exists.

- [ ] **Step 6: Implement worker, cancellation, and mapping**

Add `@work(thread=True, group="installed_models_import", exclusive=True, exit_on_error=False)`. Progress calls `app.call_from_thread`. Catch typed errors, log category/phase only, notify sanitized text, and refresh Installed on every terminal outcome. Keep external path values out of `Static`, `notify`, and logger arguments.

- [ ] **Step 7: Run complete Installed tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_model_installed_view.py -v
```

Expected: all tests PASS.

- [ ] **Step 8: Commit the vertical slice**

```bash
git add Tests/UI/test_model_installed_view.py tldw_chatbook/UI/Screens/model_browser_state.py tldw_chatbook/UI/Screens/model_installed_view.py
git commit -m "feat: import local GGUF models from Installed"
```

## Task 8: Run gates, review, and close TASK-597

**Files:**

- Modify: `backlog/tasks/task-597 - Add-bounded-local-GGUF-artifact-import.md`
- Review: all files above

- [ ] **Step 1: Run focused tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts/test_gguf_import.py Tests/Model_Artifacts/test_reconcile_staging_gc.py Tests/UI/test_model_artifact_widgets.py Tests/UI/test_model_installed_view.py -q
```

Expected: all tests PASS.

- [ ] **Step 2: Run affected regression tests**

Local HTTP fixtures may require loopback-bind permission:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Model_Artifacts Tests/UI/test_model_browser_state.py Tests/UI/test_model_installed_view.py Tests/UI/test_model_artifact_widgets.py Tests/UI/test_model_remote_view.py -q
```

Expected: all tests PASS; record the exact count. Pre-change baseline: 532 focused artifact/state/Installed tests.

- [ ] **Step 3: Run the full repository test suite**

The repository Definition of Done requires the full suite. Local HTTP fixtures may require loopback-bind permission:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q
```

Expected: the complete suite PASS. Record the exact passed/skipped/xfailed counts. Do not check ACs or set TASK-597 to Done if this gate fails; isolate a proven pre-existing baseline failure rather than silently excluding it.

- [ ] **Step 4: Run static checks**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/Model_Artifacts/gguf_import.py tldw_chatbook/Model_Artifacts/service.py tldw_chatbook/Widgets/ModelArtifacts/install_progress.py tldw_chatbook/UI/Screens/model_browser_state.py tldw_chatbook/UI/Screens/model_installed_view.py Tests/Model_Artifacts/gguf_test_helpers.py Tests/Model_Artifacts/test_gguf_import.py Tests/Model_Artifacts/test_reconcile_staging_gc.py Tests/UI/test_model_artifact_widgets.py Tests/UI/test_model_installed_view.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check tldw_chatbook/Model_Artifacts/gguf_import.py tldw_chatbook/Model_Artifacts/service.py tldw_chatbook/Widgets/ModelArtifacts/install_progress.py tldw_chatbook/UI/Screens/model_browser_state.py tldw_chatbook/UI/Screens/model_installed_view.py Tests/Model_Artifacts/gguf_test_helpers.py Tests/Model_Artifacts/test_gguf_import.py Tests/Model_Artifacts/test_reconcile_staging_gc.py Tests/UI/test_model_artifact_widgets.py Tests/UI/test_model_installed_view.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy tldw_chatbook/Model_Artifacts/gguf_import.py tldw_chatbook/Widgets/ModelArtifacts/install_progress.py tldw_chatbook/UI/Screens/model_browser_state.py tldw_chatbook/UI/Screens/model_installed_view.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q tldw_chatbook/Model_Artifacts/gguf_import.py tldw_chatbook/Widgets/ModelArtifacts/install_progress.py tldw_chatbook/UI/Screens/model_installed_view.py
git diff --check origin/dev...HEAD
```

Expected: all checks pass. Isolate/document any repo baseline issue instead of changing unrelated code.

- [ ] **Step 5: Self-review every AC**

Confirm no native import/tensor-data read; no path persistence/log/render; no uncurated automatic routing/default mutation; cancellation before `install`; activation failure retains only a complete artifact; cleanup cannot escape staging; remote acquisition/progress events remain compatible; and no TASK-604 work leaked in.

- [ ] **Step 6: Request code review**

Use `superpowers:requesting-code-review`. Address all Critical/Important findings with focused tests and rerun affected gates. Do not merge or mark Done with findings outstanding.

- [ ] **Step 7: Close Backlog only after gates pass**

Check ACs 1–6, add concise Implementation Notes with exact test/static evidence and file summary, then set Done. Never mark Done if a DoD item is incomplete. The implementation plan must already be present from the pre-Task-1 adoption step; do not defer it to closeout.

- [ ] **Step 8: Commit closeout metadata**

```bash
git add 'backlog/tasks/task-597 - Add-bounded-local-GGUF-artifact-import.md'
git commit -m "docs: close task 597"
```

## Completion boundary

TASK-597 is complete when a user can import a compatible local GGUF from Installed, the managed artifact is precisely labeled/manual-only, cancellation and failure cannot expose a partial artifact, gates and review pass, and the Backlog task satisfies its Definition of Done.

The next work remains TASK-604 or a separately approved Phase 3 browser slice. Do not absorb provider execution, catalog download, server-path migration, legacy-browser retirement, or transcription evaluation.
