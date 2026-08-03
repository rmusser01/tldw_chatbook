# TASK-2062 Model Browser Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** GGUF adoption and legacy-downloader retirement: import unmanaged files into the managed store (copy + verify), give the five local-server sections managed-model pickers, and delete `Widgets/HuggingFace/`.

**Architecture:** One deliberate sealed-core accommodation first (validators), then a pure identity module, a service-side import engine, UI over the TASK-1803/1914 ownership shape, pickers that write resolved paths into the existing launch Inputs, and retirement last.

**Tech Stack:** Python 3.11+, Textual 8.x, pytest. No new dependencies.

**Spec:** `Docs/superpowers/specs/2026-08-03-task-2062-model-browser-phase-3-design.md` — read its "Sealed-core accommodation" and "Per-server launch-path mapping" sections before Tasks 1 and 5.

## Global Constraints

- **Import invocation for every task, ABSOLUTE paths for PYTHONPATH and test files** (a relative path silently tests the main checkout).
- `Model_Artifacts.service` at module scope; `acquisition`/`fetch` only inside functions (subprocess import-recording test + per-module AST tests enforce this).
- Views post intents; `LLMScreen` owns every worker (TASK-1803/1914 shape). No widget calls `install`/`activate`/`delete`.
- No user-visible string contains the word "artifact"; the UI says "model".
- No I/O in `compose()`/`on_mount()`; never write the user's real config/data dirs; tests use `tmp_path`.
- Google-style `Returns:` on public non-None functions; `Args:` on parameterised tests.
- Sabotage-verify each load-bearing new test: exact-string edit with a count assertion, confirm red, revert. Detectors get positive controls.
- The import dialog's copy notice is **product copy, verbatim** (Task 4); do not paraphrase it.

## File Structure

**Create:** `tldw_chatbook/Model_Artifacts/local_import.py` (identity + engine), `tldw_chatbook/Widgets/ModelArtifacts/managed_model_picker.py`, `tldw_chatbook/Widgets/ModelArtifacts/import_model_modal.py`, `Tests/Model_Artifacts/test_local_import.py`, `Tests/UI/test_import_model_flow.py`, `Tests/UI/test_managed_model_picker.py`

**Modify:** `Model_Artifacts/service.py` (validators only), `UI/Screens/model_installed_view.py`, `UI/Screens/llm_screen.py`, `UI/LLM_Management_Window.py`, `UI/Screens/model_browser_state.py`

**Delete (Task 7):** `tldw_chatbook/Widgets/HuggingFace/` (five modules), the `download-models` rail row and view mount.

---

### Task 1: Sealed-core accommodation for local-origin descriptors

**Files:** Modify `tldw_chatbook/Model_Artifacts/service.py`; test in `Tests/Model_Artifacts/test_service.py` and `Tests/Model_Artifacts/test_local_import.py` (new).

**Interfaces — Produces:** descriptor validation accepting (a) `source_url` of the form `file:///<percent-encoded absolute path>` **iff** `provenance == (ProvenanceClass.LOCAL_INTEGRITY_RECORDED,)`; (b) `license_url == ""` **iff** `license_id == "unknown"`. Everything else unchanged.

- [ ] Failing tests first: a `file://` source_url descriptor with LOCAL_INTEGRITY_RECORDED-only provenance constructs; the same URL with `CHATBOOK_CURATED` in provenance raises `ArtifactDescriptorValidationError`; empty `license_url` with `license_id="unknown"` constructs; empty with `license_id="mit"` raises; a `file://` URL containing `?`, whitespace, or an invalid percent-escape raises regardless of provenance.
- [ ] Implement in the descriptor's validation path: branch `_validate_url` handling for these two fields (a `file://` value must be absolute — path starts `/` — percent-escape-valid, and free of `?`/`#`/whitespace/backslash). Keep the strict http(s) rule for every other case and every other field.
- [ ] **Egress defense test** (required by the spec): `evaluate_url_policy("file:///etc/passwd")` (and the module's check-or-raise wrapper) refuses with `reason="scheme"` — pinning that a `file://` descriptor can never become a local-file-read in any fetch path. Confirm the existing behavior, then pin it.
- [ ] Sabotage: widen the provenance gate to allow `file://` for any provenance — the cross-field test must go red. Revert.
- [ ] Run `Tests/Model_Artifacts/` full; commit `feat(artifacts): local-origin descriptor accommodation (file:// source, unknown license)`.

### Task 2: Import identity (pure)

**Files:** Create `tldw_chatbook/Model_Artifacts/local_import.py` (identity half), `Tests/Model_Artifacts/test_local_import.py` (extend).

**Interfaces — Produces:**
- `import_identity(path: Path, sha256_hex: str) -> ImportIdentity` (frozen dataclass: `artifact_id`, `revision`, `variant`)
- `build_import_descriptor(path: Path, size_bytes: int, sha256_hex: str) -> ArtifactDescriptor`

Rules (from the spec): `artifact_id` = sanitized lowercase stem satisfying `_validate_canonical_component` (map `.`/`_`/uppercase/runs of illegal chars → single `-`; reject empty/Windows-reserved results with a typed error, do not "fix" them); `revision` = `f"sha256-{sha256_hex[:12]}"`; `variant`/`precision` = parsed quant tag lowercased (`Q4_K_M` → `q4-k-m`; recognize `[qQ]\d[_-].*` and `f16/f32/bf16`) else `"imported"`; `source_url` = `Path.as_uri()` result (percent-encoded `file://`); `upstream_repository="local-import"`, `license_id="unknown"`, `license_url=""`, `format=ArtifactFormat.GGUF`, `role=ROOT`, `consumer="llm"`, `model_family=artifact_id`, `runtime_name="llama-cpp"`, `runtime_version_constraint=">=0"`, `supported_os=("linux","darwin","windows")`, `supported_architectures=("x86-64","arm64")`, `provenance=(LOCAL_INTEGRITY_RECORDED,)`, one `ArtifactFile` (`path=<original filename>`, declared size + digest), `usage_notice` naming the original path and the unknown license.

- [ ] Failing tests: quant parsing table (incl. `f16`, no-tag → `imported`), sanitization edges (spaces, unicode, leading dots, `CON.gguf` rejected), revision format, `as_uri` percent-encoding for a path with spaces, and **a round-trip test: the built descriptor passes real `ArtifactDescriptor` validation** (this is what pins Task 1's accommodation to real use).
- [ ] Implement; module imports `service` at module scope only. Sabotage the round-trip test via a bad `license_url`; revert.
- [ ] Commit `feat(artifacts): import identity and descriptor builder`.

### Task 3: Import engine (service-side, no Textual)

**Files:** `local_import.py` (engine half), tests in `Tests/Model_Artifacts/test_local_import.py`.

**Interfaces — Produces:** `import_local_model(source: Path, service: ModelArtifactService, *, progress: Callable[[int, int], None] | None = None, cancel: threading.Event | None = None) -> ArtifactRef`, raising typed errors (`ImportCancelledError`, `ImportSourceError`, insufficient-space via existing service errors). Progress is plain `(bytes_done, bytes_total)` — **not** `AcquisitionProgress`; this module must never import `acquisition`.

Flow: free-space check (~1× file size under the store root) → create an import temp **inside `service.staging_path`** named `import-<uuid4hex>/` containing an ownership marker file (JSON: source path, pid, schema) and a `payload/` subdir → stream-copy in 1 MiB chunks while hashing sha256, honoring `cancel` between chunks → build descriptor (Task 2) from the computed digest → `service.install(descriptor, temp/payload, consume_source=True)` (the core verifies the store-side copy and promotes; on `ArtifactIntegrityError` the copy raced a concurrent writer — surface it, never retry silently) → remove the temp scaffold → return the ref. On cancel or any failure: remove this run's temp entirely. On engine entry: sweep `staging/import-*` dirs whose marker parses and whose pid is dead — remove them (self-cleaning); a marker that does not parse is left for `reconcile()` to report, never guessed at.

- [ ] Failing tests: happy path installs and is listed by `list_installed()`; re-import same bytes → same ref, already-installed no-op behavior; cancel mid-copy leaves no temp and no artifact; a stale dead-pid temp is swept on next run while a live-pid temp is not; garbage marker left alone; digest-mismatch path (mutate the temp payload between hash and install via a test hook) surfaces `ArtifactIntegrityError` with the store unmodified; progress called with monotonically increasing `bytes_done` ending at `bytes_total`.
- [ ] Sabotage: skip the marker write — the stale-sweep test must go red. Revert. Commit `feat(artifacts): local import engine (copy, hash, verify, promote)`.

### Task 4: Import UI — dialog, worker, unmanaged-row action

**Files:** Create `Widgets/ModelArtifacts/import_model_modal.py`; modify `model_installed_view.py`, `llm_screen.py`, `model_browser_state.py`; test `Tests/UI/test_import_model_flow.py`.

**Interfaces:** `UnmanagedRow` gains an Import action posting `ImportRequested(path)`; `LLMScreen` owns the `@work(thread=True)` import worker (validates the event payload; refuses when `_install_in_progress()` — imports share the one install lock); `ImportModelModal(ModalScreen[bool])` shows source path, `format_mib` size, destination, free-space result, and this **verbatim** notice:

> This copies the file into Chatbook's managed store — integrity can only be guaranteed for bytes the store owns. Your original file is untouched; once the import completes you may delete it to reclaim the space.

Success state offers Activate and repeats: "The import is complete. Your original file at `<path>` is no longer needed by Chatbook — you may delete it to reclaim the space." Progress renders via the existing `ModelInstallProgress` (the screen adapts `(done, total)` into the widget's event type function-locally) and must survive recompose via the existing hydration path. Escape binds `dismiss(False)` (capital). Failure text goes through a typed-error mapper in `model_browser_state` (extend `install_failure_message` family; never `str(exc)` verbatim).

- [ ] Failing tests: Import on an unmanaged row opens the modal with the verbatim notice (assert the exact sentence); cancel never calls the engine; confirm starts the screen-owned worker; a second request during import is refused leaving the first untouched; progress survives `refresh(recompose=True)`; failure notifies mapped text with an injected marker asserted absent; success row appears in Installed after refresh.
- [ ] Sabotage: bypass the confirm gate — the cancel-never-imports test must go red. Revert. Commit `feat(models): import flow UI over the screen-owned worker`.

### Task 5: Shared managed-model picker; wire llama.cpp / llamafile / ONNX

**Files:** Create `Widgets/ModelArtifacts/managed_model_picker.py`; modify `LLM_Management_Window.py`; test `Tests/UI/test_managed_model_picker.py`.

**Interfaces:** `ManagedModelPicker(format_filter: ArtifactFormat, target_input_id: str)` — lazy-loads ready installed models of the format on first Show (per `InstalledView`'s `_loaded` pattern, threaded, no I/O at compose), renders label + revision + size via `format_mib`, and on selection resolves `<artifact_path>/<files[0].path>` and **sets the target Input's value** — the launch handlers in `llm_management_events.py` are NOT modified. Free-text Input + its browse button move behind an "Use an unmanaged path…" collapsed disclosure for these three sections only.

- [ ] Failing tests: picker lists only ready models of its format; selection writes the resolved payload path into `#llamacpp-model-path`; the disclosure reveals the legacy Input which still accepts arbitrary text; no service call at compose time (assert lazily); empty store renders a "no managed models yet — Curated tab" hint (no "artifact" wording).
- [ ] Wire into the llamacpp, llamafile, and onnx sections. Run the existing launch-handler tests untouched. Sabotage: make selection write the artifact dir instead of the payload file path — the resolution test must go red. Revert. Commit.

### Task 6: vLLM and MLX — picker alongside first-class free text

**Files:** Modify `LLM_Management_Window.py`; extend `Tests/UI/test_managed_model_picker.py`.

Same picker, same Input-writing seam, but the free-text Input stays visible and primary (HF repo ids are legitimate input, per spec). No disclosure. Tests: both sections show picker AND visible Input; typing a repo id is untouched by the picker's presence; picker selection overwrites the Input like any browse would.

- [ ] Implement, test, commit.

### Task 7: Retirement

**Files:** Delete `tldw_chatbook/Widgets/HuggingFace/` (all five modules); modify `llm_screen.py` (`MODELS_RAIL_SECTIONS` drops `download-models`), `LLM_Management_Window.py` (view mount + `view_mapping` entry), `config.py` only if `model_download_dir`'s default is dead after this; update every test that referenced the removed row/widgets **deliberately** (list each in the report — a broken-then-updated test is how regressions hide).

- [ ] First: `grep -rn "HuggingFace\|download-models\|DownloadManager" tldw_chatbook/ Tests/` and enumerate every site in the report before deleting anything.
- [ ] The unmanaged scan keeps working (it lives in `model_installed_view`, not the deleted package — verify, don't assume).
- [ ] Rail test updated: `download-models` absent, `curated`/`installed` present, no `remote` regression.
- [ ] Full sweep: `Tests/UI/`, `Tests/Model_Artifacts/`, `Tests/STT/test_boundaries.py`, plus every file the grep touched. Commit `feat(models)!: retire the legacy HuggingFace downloader (Phase 3 complete)`.

---

## Self-review notes

- Spec coverage: accommodation → T1; identity → T2; engine+abort safety → T3; dialog/notice/ownership → T4; pickers → T5–6; retirement + ADR-025 → T7. The spec's "plan decides the abort mechanism" is decided in T3: marker-owned temps + pid-based self-sweep, `reconcile()` reporting anything unparseable.
- Type consistency: `import_local_model` (T3) consumed by T4's worker; `ImportIdentity`/`build_import_descriptor` (T2) consumed by T3; picker (T5) reused verbatim in T6.
- Ordering: T1 before T2 (round-trip test needs the accommodation); T2 before T3; T3 before T4; T5 before T6; T7 last, after nothing imports the doomed package.
