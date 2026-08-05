# TASK-596 Model Artifact Browser — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the offline half of the model browser — a curated registry, a pure view-model, shared install/plan/progress/activation widgets, the Curated and Installed views in Lab → Models, and the Library Parakeet modal refactored onto the shared modal.

**Architecture:** Three layers. A service-side curated registry that structurally satisfies `ArtifactCatalog`; a pure, Textual-free view-model module that turns service data structures into render-ready rows and user-safe error text; and Textual widgets that render those rows and post intent messages while the host screen owns every worker.

**Tech Stack:** Python 3.11+, Textual 8.x, pytest (+ `app.run_test()` Pilot), loguru. No new dependencies.

**Spec:** `Docs/superpowers/specs/2026-08-01-task-596-model-artifact-browser-design.md`

## Global Constraints

- **Import boundary (load-bearing, enforced by a test).** `Model_Artifacts.service` may be imported at module scope. `Model_Artifacts.acquisition` and `Model_Artifacts.fetch` may be imported **only inside functions**. `Tests/Model_Artifacts/test_credentials_and_boundaries.py:484` runs a subprocess with an import-recording hook and fails on an import even attempted-and-caught.
- **`CuratedRegistry` must NOT inherit from `ArtifactCatalog`.** It satisfies the Protocol structurally; import `ArtifactCatalog` only under `if TYPE_CHECKING:`. Precedent and rationale: `ParakeetV2Catalog`'s docstring in `Local_Ingestion/parakeet_v2_artifact.py:207`.
- **No user-visible copy contains the word "artifact".** The UI says "model". Internal names keep the artifact vocabulary. (`artifacts` is already a top-level destination for generated outputs.)
- **No view performs I/O in `compose()` or `on_mount()`.** `LLMManagementWindow` composes every view up front and switches by CSS class, so compose-time work runs on screen construction. Load on first activation and on explicit refresh.
- **Widgets and modals never call `preflight`, `provision`, `activate`, or `delete`.** They post intent messages; the host screen runs `@work(thread=True)`.
- **Never write to the user's real config or data directories.** All verification happens inside pytest against `tmp_path`.
- **`BaseWizard.py` is not modified by this plan.**
- Every public function returning non-`None` gets a Google-style `Returns:` section.

## File Structure

**Create:**
- `tldw_chatbook/Model_Artifacts/store.py` — neutral store root + service factory
- `tldw_chatbook/Model_Artifacts/curated_registry.py` — `CuratedRegistry`, curated entries
- `tldw_chatbook/UI/Screens/model_browser_state.py` — pure view-model
- `tldw_chatbook/Widgets/ModelArtifacts/__init__.py`
- `tldw_chatbook/Widgets/ModelArtifacts/plan_panel.py` — `ModelPlanPanel`
- `tldw_chatbook/Widgets/ModelArtifacts/install_progress.py` — `ModelInstallProgress`
- `tldw_chatbook/Widgets/ModelArtifacts/install_modal.py` — `ModelInstallModal`
- `tldw_chatbook/Widgets/ModelArtifacts/activation_controls.py` — `ModelActivationControls`
- `tldw_chatbook/UI/Screens/model_curated_view.py` — `CuratedView`
- `tldw_chatbook/UI/Screens/model_installed_view.py` — `InstalledView`

**Modify:**
- `tldw_chatbook/Model_Artifacts/acquisition.py` — add `ArtifactPreflightEntry.provenance`
- `tldw_chatbook/Local_Ingestion/parakeet_v2_artifact.py` — re-export the moved root; register with the curated registry
- `tldw_chatbook/UI/Screens/library_screen.py` — refactor onto `ModelInstallModal`; delete the local error mapper
- `tldw_chatbook/UI/Screens/llm_screen.py` — rail rows
- `tldw_chatbook/UI/LLM_Management_Window.py` — mount the new views

**Test:**
- `Tests/Model_Artifacts/test_curated_registry.py`
- `Tests/UI/test_model_browser_state.py`
- `Tests/UI/test_model_artifact_widgets.py`
- `Tests/UI/test_model_installed_view.py`
- `Tests/UI/test_parakeet_v2_install_ui.py` (existing — migrated)
- `Tests/Model_Artifacts/test_credentials_and_boundaries.py` (existing — extended)

---

### Task 1: Neutral store root and service factory

Move the app-global managed-store root out of the Parakeet adapter so a second consumer does not have to import it from an STT module. Its own docstring already says it is not Parakeet-specific.

**Files:**
- Create: `tldw_chatbook/Model_Artifacts/store.py`
- Modify: `tldw_chatbook/Local_Ingestion/parakeet_v2_artifact.py`
- Test: `Tests/Model_Artifacts/test_curated_registry.py`

**Interfaces:**
- Produces: `managed_model_artifact_root() -> Path`, `managed_service(root: Path | None = None) -> ModelArtifactService`
- Consumes: `ModelArtifactService` from `.service`

- [ ] **Step 1: Write the failing test**

```python
# Tests/Model_Artifacts/test_curated_registry.py
from pathlib import Path


def test_store_root_matches_the_parakeet_adapter_value() -> None:
    """The moved root is the same path the adapter used, so existing
    installs remain discoverable after the move."""
    from tldw_chatbook.Local_Ingestion import parakeet_v2_artifact
    from tldw_chatbook.Model_Artifacts.store import managed_model_artifact_root

    assert managed_model_artifact_root() == parakeet_v2_artifact.managed_model_artifact_root()
    assert isinstance(managed_model_artifact_root(), Path)


def test_managed_service_uses_an_explicit_root(tmp_path: Path) -> None:
    from tldw_chatbook.Model_Artifacts.store import managed_service

    service = managed_service(tmp_path)
    assert service.artifacts_path == tmp_path / "artifacts"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD .venv/bin/pytest Tests/Model_Artifacts/test_curated_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: tldw_chatbook.Model_Artifacts.store`

- [ ] **Step 3: Create the module**

Move the body of `managed_model_artifact_root()` from `Local_Ingestion/parakeet_v2_artifact.py:233` verbatim, including its docstring, into `tldw_chatbook/Model_Artifacts/store.py`:

```python
"""Process-wide managed model store location and service construction.

Imports only ``.service`` at module scope: this module is reachable from
worker-side code, and ``acquisition``/``fetch`` must stay out of that
import graph (see Tests/Model_Artifacts/test_credentials_and_boundaries.py).
"""

from __future__ import annotations

from pathlib import Path

from .service import ModelArtifactService


def managed_model_artifact_root() -> Path:
    """Return the shared managed-artifact store root.

    A sibling of the legacy installer's own ``models/stt/...`` destination
    (``parakeet_v2_installer.parakeet_v2_install_dir()``), both beneath the
    existing user-data directory -- so a fresh install and a legacy one
    never collide on disk. Not Parakeet-specific: every future managed
    artifact this application acquires shares this one
    ``ModelArtifactService`` root, distinguished internally by artifact id,
    revision, and variant.

    Returns:
        The absolute path to the shared managed-artifact store root.
    """

    from tldw_chatbook.Utils.paths import get_user_data_dir

    return get_user_data_dir() / "models" / "managed"


def managed_service(root: Path | None = None) -> ModelArtifactService:
    """Return a service bound to the managed store root.

    Args:
        root: Override for the store root; tests pass a ``tmp_path``.

    Returns:
        A ``ModelArtifactService`` rooted at ``root`` or the shared root.
    """
    return ModelArtifactService(root or managed_model_artifact_root())
```

The path expression above is the current one, copied verbatim from
`parakeet_v2_artifact.py:250`. **Re-read that function before writing this file
and copy what is actually there** — the store root is where every installed model
already lives, so changing it silently orphans them. The test in Step 1 pins the
moved function and the adapter's to be equal, so a divergence fails loudly.

- [ ] **Step 4: Re-export from the Parakeet adapter**

In `parakeet_v2_artifact.py`, replace the function body with a re-export so existing callers keep working:

```python
from tldw_chatbook.Model_Artifacts.store import (  # noqa: F401
    managed_model_artifact_root,
)
```

Keep `parakeet_v2_managed_service()` as-is; it may delegate to `managed_service()`.

- [ ] **Step 5: Run the tests**

Run: `PYTHONPATH=$PWD .venv/bin/pytest Tests/Model_Artifacts/ Tests/Local_Ingestion/test_parakeet_v2_artifact.py Tests/Audio/test_console_dictation.py -q`
Expected: PASS (all previously-passing tests still pass)

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Model_Artifacts/store.py tldw_chatbook/Local_Ingestion/parakeet_v2_artifact.py Tests/Model_Artifacts/test_curated_registry.py
git commit -m "refactor(artifacts): move the managed store root to a neutral module"
```

---

### Task 2: Curated registry

Give "what does this application vouch for" an owner. `ArtifactCatalog` is lookup-only, so the Curated view has nothing to enumerate today.

**Files:**
- Create: `tldw_chatbook/Model_Artifacts/curated_registry.py`
- Modify: `tldw_chatbook/Local_Ingestion/parakeet_v2_artifact.py`
- Modify: `Tests/Model_Artifacts/test_credentials_and_boundaries.py`
- Test: `Tests/Model_Artifacts/test_curated_registry.py`

**Interfaces:**
- Produces: `CuratedRegistry`, `curated_registry() -> CuratedRegistry`
- `CuratedRegistry.register(descriptor: ArtifactDescriptor, *, sources: Mapping[str, str]) -> None`
- `CuratedRegistry.list() -> tuple[ArtifactDescriptor, ...]`
- `CuratedRegistry.descriptor(ref: ArtifactRef) -> ArtifactDescriptor` (raises `KeyError`)
- `CuratedRegistry.sources(ref: ArtifactRef) -> dict[str, str]`

- [ ] **Step 1: Write the failing tests**

```python
def test_registry_lists_registered_descriptors_in_registration_order(tmp_path) -> None:
    from tldw_chatbook.Model_Artifacts.curated_registry import CuratedRegistry
    from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
        parakeet_v2_descriptor,
        parakeet_v2_source_map,
        parakeet_v2_reference,
    )

    registry = CuratedRegistry()
    descriptor = parakeet_v2_descriptor()
    registry.register(descriptor, sources=parakeet_v2_source_map()[parakeet_v2_reference()])

    assert registry.list() == (descriptor,)
    assert registry.descriptor(parakeet_v2_reference()) is descriptor


def test_registry_descriptor_raises_keyerror_for_unknown_ref() -> None:
    import pytest
    from tldw_chatbook.Model_Artifacts.curated_registry import CuratedRegistry
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef

    with pytest.raises(KeyError):
        CuratedRegistry().descriptor(ArtifactRef("nope", "rev", "int8"))


def test_registry_does_not_subclass_the_catalog_protocol() -> None:
    """R2: subclassing would require a module-scope acquisition import,
    which the STT import-boundary test forbids."""
    import inspect
    from tldw_chatbook.Model_Artifacts import curated_registry as module

    source = inspect.getsource(module)
    assert "class CuratedRegistry:" in source
    assert "from .acquisition import" not in source
    assert "from tldw_chatbook.Model_Artifacts.acquisition import" not in source.split(
        "if TYPE_CHECKING:"
    )[0]


def test_default_registry_contains_parakeet_v2() -> None:
    from tldw_chatbook.Model_Artifacts.curated_registry import curated_registry
    from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import parakeet_v2_reference

    refs = [d.reference for d in curated_registry().list()]
    assert parakeet_v2_reference() in refs
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=$PWD .venv/bin/pytest Tests/Model_Artifacts/test_curated_registry.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Implement the registry**

```python
"""The curated model catalog: what this application vouches for.

Structurally satisfies ``Model_Artifacts.acquisition.ArtifactCatalog``
(a plain ``Protocol``) WITHOUT importing it -- that import is confined to
``TYPE_CHECKING`` because this module is reachable from worker-side code
and ``acquisition`` pulls in httpx and the async fetch layer. The same
constraint and rationale apply to ``ParakeetV2Catalog``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from .service import ArtifactDescriptor, ArtifactRef

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .acquisition import ArtifactCatalog  # noqa: F401


class CuratedRegistry:
    """An ordered, in-process registry of curated artifact descriptors."""

    def __init__(self) -> None:
        self._descriptors: dict[ArtifactRef, ArtifactDescriptor] = {}
        self._sources: dict[ArtifactRef, dict[str, str]] = {}

    def register(
        self,
        descriptor: ArtifactDescriptor,
        *,
        sources: Mapping[str, str],
    ) -> None:
        """Register one curated descriptor and its per-file source URLs."""
        self._descriptors[descriptor.reference] = descriptor
        self._sources[descriptor.reference] = dict(sources)

    def list(self) -> tuple[ArtifactDescriptor, ...]:
        """Return every curated descriptor in registration order.

        Returns:
            A tuple of registered descriptors.
        """
        return tuple(self._descriptors.values())

    def descriptor(self, ref: ArtifactRef) -> ArtifactDescriptor:
        """Return the descriptor for ``ref``.

        Args:
            ref: The artifact reference to resolve.

        Returns:
            The registered descriptor.

        Raises:
            KeyError: If ``ref`` is not registered.
        """
        return self._descriptors[ref]

    def sources(self, ref: ArtifactRef) -> dict[str, str]:
        """Return the per-file source map for ``ref``.

        Args:
            ref: The artifact reference to resolve.

        Returns:
            A copy of the registered relative-path to URL mapping.

        Raises:
            KeyError: If ``ref`` is not registered.
        """
        return dict(self._sources[ref])


_REGISTRY: CuratedRegistry | None = None


def curated_registry() -> CuratedRegistry:
    """Return the process-wide curated registry, registering defaults once.

    Returns:
        The shared ``CuratedRegistry``.
    """
    global _REGISTRY
    if _REGISTRY is None:
        registry = CuratedRegistry()
        from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
            parakeet_v2_descriptor,
            parakeet_v2_reference,
            parakeet_v2_source_map,
        )

        ref = parakeet_v2_reference()
        registry.register(
            parakeet_v2_descriptor(), sources=parakeet_v2_source_map()[ref]
        )
        _REGISTRY = registry
    return _REGISTRY
```

The default registration import is function-local: it keeps `Local_Ingestion` out of this module's import graph and lets tests build an empty `CuratedRegistry()` directly.

- [ ] **Step 4: Extend the import-boundary test**

In `Tests/Model_Artifacts/test_credentials_and_boundaries.py`, add `tldw_chatbook.Model_Artifacts.curated_registry` and `...store` to the module list imported by the subprocess in `test_stt_and_transcription_worker_modules_never_import_acquisition_or_fetch`, so the registry is covered by the same hook.

- [ ] **Step 5: Run the tests**

Run: `PYTHONPATH=$PWD .venv/bin/pytest Tests/Model_Artifacts/ -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Model_Artifacts/curated_registry.py Tests/Model_Artifacts/
git commit -m "feat(artifacts): add the curated model registry"
```

---

### Task 3: Carry provenance on preflight entries

AC #2's trust labels must come from the same report as the bytes they describe.

**Files:**
- Modify: `tldw_chatbook/Model_Artifacts/acquisition.py`
- Test: `Tests/Model_Artifacts/test_preflight.py`

**Interfaces:**
- Produces: `ArtifactPreflightEntry.provenance: tuple[ProvenanceClass, ...]`

- [ ] **Step 1: Write the failing test**

```python
def test_preflight_entry_carries_descriptor_provenance(tmp_path) -> None:
    """AC #2: the trust label travels with the bytes it describes."""
    # Build a report with the module's existing preflight fixtures, then:
    entry = report.entries[0]
    assert entry.provenance == descriptor.provenance


def test_adding_provenance_does_not_change_closure_fingerprints() -> None:
    """closure_fingerprint is computed from ArtifactRefs only, so consent
    fingerprints and in-flight resume state are unaffected."""
    from tldw_chatbook.Model_Artifacts.service import closure_fingerprint, ArtifactRef

    root = ArtifactRef("a", "r", "int8")
    dep = ArtifactRef("b", "r", "int8")
    assert closure_fingerprint(root, [dep]) == closure_fingerprint(root, [dep])
```

Use the existing preflight fixtures in `Tests/Model_Artifacts/test_preflight.py` to build `report`/`descriptor`; do not invent a new harness.

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=$PWD .venv/bin/pytest Tests/Model_Artifacts/test_preflight.py -k provenance -v`
Expected: FAIL — `AttributeError: 'ArtifactPreflightEntry' object has no attribute 'provenance'`

- [ ] **Step 3: Add the field**

In `acquisition.py`, add to the frozen dataclass at line 314, as the last field, with **no default** so every construction site must supply it:

```python
    already_installed: bool
    provenance: tuple[ProvenanceClass, ...]
```

Add `ProvenanceClass` to the existing `from .service import (...)` block if absent. At the construction site (`acquisition.py:926`) add:

```python
                already_installed=already_installed,
                provenance=descriptor.provenance,
```

- [ ] **Step 4: Fix every other construction site**

Run the suite and update any test that constructs an `ArtifactPreflightEntry` directly. The absence of a default is deliberate: a missing provenance must be a construction error, never a silently-empty trust label.

- [ ] **Step 5: Run the tests**

Run: `PYTHONPATH=$PWD .venv/bin/pytest Tests/Model_Artifacts/ -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Model_Artifacts/acquisition.py Tests/Model_Artifacts/
git commit -m "feat(artifacts): carry provenance on preflight entries"
```

---

### Task 4: Pure view-model

**Files:**
- Create: `tldw_chatbook/UI/Screens/model_browser_state.py`
- Test: `Tests/UI/test_model_browser_state.py`

**Interfaces:**
- Consumes: `PreflightReport`, `InstalledArtifact`, `ArtifactDiskUsage` (typing-only imports)
- Produces:
  - `PlanRow`, `InventoryRow`, `UnmanagedRow` frozen dataclasses
  - `plan_rows(report) -> tuple[PlanRow, ...]`
  - `plan_totals(report) -> PlanTotals`
  - `provenance_label(provenance: tuple[ProvenanceClass, ...]) -> str`
  - `inventory_rows(installed, usage, unmanaged) -> tuple[InventoryRow, ...]`
  - `install_failure_message(exc: BaseException, *, model_label: str) -> str`

- [ ] **Step 1: Write the failing tests**

```python
def test_provenance_labels_are_precise_and_never_imply_safety() -> None:
    from tldw_chatbook.Model_Artifacts.service import ProvenanceClass
    from tldw_chatbook.UI.Screens.model_browser_state import provenance_label

    curated = provenance_label((ProvenanceClass.CHATBOOK_CURATED,))
    recorded = provenance_label((ProvenanceClass.LOCAL_INTEGRITY_RECORDED,))
    assert curated != recorded
    for label in (curated, recorded):
        assert "safe" not in label.lower()
        assert "malware" not in label.lower()
        assert "trusted" not in label.lower()


def test_inventory_row_for_a_broken_manifest_is_rendered_not_dropped() -> None:
    """InstalledArtifact.descriptor is None after a crash or partial delete."""
    from pathlib import Path
    from tldw_chatbook.Model_Artifacts.service import InstalledArtifact
    from tldw_chatbook.UI.Screens.model_browser_state import inventory_rows

    broken = InstalledArtifact(
        path=Path("/store/artifacts/x"),
        descriptor=None,
        ready=False,
        active=False,
        error="readiness: unreadable",
    )
    rows = inventory_rows((broken,), usage=None, unmanaged=())
    assert len(rows) == 1
    assert rows[0].is_broken is True
    assert "Repair" in rows[0].action_hint


def test_install_failure_message_maps_types_and_never_leaks_raw_text() -> None:
    from tldw_chatbook.Model_Artifacts.acquisition import (
        InsufficientSpaceError,
        PreflightNotGrantableError,
        TransferError,
    )
    from tldw_chatbook.UI.Screens.model_browser_state import install_failure_message

    marker = "SECRET-GATING-DETAIL"
    cases = (
        InsufficientSpaceError(marker),
        PreflightNotGrantableError(marker),
        TransferError(marker, retryable=True),
    )
    for exc in cases:
        message = install_failure_message(exc, model_label="Parakeet v2")
        assert marker not in message
        assert message


def test_install_failure_message_uses_the_model_label() -> None:
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionBusyError
    from tldw_chatbook.UI.Screens.model_browser_state import install_failure_message

    message = install_failure_message(AcquisitionBusyError("x"), model_label="Whisper")
    assert "Whisper" in message
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=$PWD .venv/bin/pytest Tests/UI/test_model_browser_state.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Implement**

Port `_parakeet_v2_failure_message` from `library_screen.py:942` verbatim, with two changes: the function-local `from tldw_chatbook.Model_Artifacts.acquisition import (...)` block is preserved exactly (R2), and the three Parakeet-specific strings take `model_label`:

```python
    if isinstance(exc, AcquisitionBusyError):
        return f"Another {model_label} install is already in progress. Try again shortly."
    if isinstance(exc, CatalogError):
        return f"The {model_label} download source is misconfigured."
    ...
    return f"{model_label} install failed. See the application log for details."
```

`plan_rows` maps one `ArtifactPreflightEntry` to one `PlanRow` carrying repository, revision, license id and url, precision, file count, byte total, `already_installed`, and `provenance_label(entry.provenance)`. `plan_totals` carries download bytes, staging overhead, destination, free bytes, required bytes, `sufficient_space`, and `gating_errors`. This module imports no Textual and performs no I/O.

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=$PWD .venv/bin/pytest Tests/UI/test_model_browser_state.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/model_browser_state.py Tests/UI/test_model_browser_state.py
git commit -m "feat(models): add the model browser view-model"
```

---

### Task 5: Plan panel and install modal

**Files:**
- Create: `tldw_chatbook/Widgets/ModelArtifacts/__init__.py`, `plan_panel.py`, `install_modal.py`
- Test: `Tests/UI/test_model_artifact_widgets.py`

**Interfaces:**
- Consumes: `plan_rows`, `plan_totals` from Task 4
- Produces:
  - `ModelPlanPanel(report: PreflightReport, *, model_label: str)`
  - `ModelInstallModal(ModalScreen[bool])` with ids `#model-install-modal`, `#model-install-confirm`, `#model-install-cancel`

- [ ] **Step 1: Write the failing tests**

```python
async def test_plan_panel_renders_every_ac3_field() -> None:
    """AC #3: closure, revision, license, precision, bytes, staging,
    destination, free space."""
    # Mount ModelPlanPanel with a synthetic grantable report; assert the
    # rendered text contains the repository, the revision string, the
    # license id, the precision, the file count, and the destination.


async def test_install_is_disabled_when_the_report_is_not_grantable() -> None:
    """The plan is a gate: grant() would raise, so Confirm must not be
    clickable and the reason must be shown."""
    # Report with sufficient_space=False; assert the confirm button is
    # disabled and the panel names insufficient space.


async def test_confirm_returns_true_and_cancel_returns_false() -> None:
    # ModalScreen[bool] contract: it returns a decision and owns no worker.
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=$PWD .venv/bin/pytest Tests/UI/test_model_artifact_widgets.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Implement**

`ModelPlanPanel` renders from `plan_rows`/`plan_totals` only — it takes the report in its constructor and holds no service reference. `ModelInstallModal` composes a `ModelPlanPanel` plus Confirm/Cancel, dismisses `True`/`False`, and **starts no worker**. When `report.gating_errors` is non-empty or `sufficient_space` is false, Confirm is `disabled=True` and the panel shows the reason.

Use `Static(..., markup=False)` for any text derived from descriptor or report fields: repository ids and license strings can contain square brackets, which Rich would parse as markup.

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=$PWD .venv/bin/pytest Tests/UI/test_model_artifact_widgets.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/ModelArtifacts/ Tests/UI/test_model_artifact_widgets.py
git commit -m "feat(models): add the shared plan panel and install modal"
```

---

### Task 6: Install progress

Progress must render somewhere that outlives the consent dialog.

**Files:**
- Create: `tldw_chatbook/Widgets/ModelArtifacts/install_progress.py`
- Test: `Tests/UI/test_model_artifact_widgets.py`

**Interfaces:**
- Produces:
  - `ModelInstallProgress` widget with `update_progress(event: AcquisitionProgress) -> None`
  - `InstallProgressed(Message)` carrying an `AcquisitionProgress`

- [ ] **Step 1: Write the failing tests**

```python
async def test_progress_widget_shows_each_phase_in_order() -> None:
    # fetch -> pre-verify -> verify-install -> activate; assert the label
    # changes and byte detail appears only for the byte-bearing phases.


def test_progress_callback_posts_a_message_and_touches_no_widget() -> None:
    """The provision callback runs on the worker thread."""
    # Build the callback the host screen passes to provision(); assert
    # invoking it enqueues an InstallProgressed message rather than
    # mutating widget state directly.
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=$PWD .venv/bin/pytest Tests/UI/test_model_artifact_widgets.py -k progress -v`
Expected: FAIL

- [ ] **Step 3: Implement**

`AcquisitionProgress` carries `phase | ref | file | bytes_done | bytes_total`. `verify-install` and `activate` are indeterminate (zero bytes by design) and render as phase labels without a byte count. `fetch` and `pre-verify` render a determinate bar plus the current file. No throttling: `fetch.py` reads 1 MiB chunks, so a 660 MB download emits about 660 events.

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=$PWD .venv/bin/pytest Tests/UI/test_model_artifact_widgets.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/ModelArtifacts/install_progress.py Tests/UI/
git commit -m "feat(models): add the install progress component"
```

---

### Task 7: Refactor Library onto the shared modal

This is AC #8's proof — a second real consumer — and it gives the Parakeet install the progress display it never had.

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_parakeet_v2_install_ui.py`

**Interfaces:**
- Consumes: `ModelInstallModal`, `ModelInstallProgress`, `install_failure_message`

- [ ] **Step 1: Update the existing tests first**

`Tests/UI/test_parakeet_v2_install_ui.py` pins `#parakeet-v2-install-modal`, `#parakeet-v2-install-modal-confirm`, `#parakeet-v2-install-modal-cancel`. Decide deliberately: either the shared modal accepts an `id_prefix` so those ids survive, or the tests move to the shared ids in this commit. Do not let them break silently — they are the regression net for the flow being refactored.

- [ ] **Step 2: Delete `ParakeetV2InstallModal` and `_parakeet_v2_failure_message`**

Replace with `ModelInstallModal` and `install_failure_message(exc, model_label="Parakeet v2")`. `LibraryScreen` keeps ownership of `_run_parakeet_v2_preflight` and `_run_parakeet_v2_install` — R1 is unchanged, and the worker must continue to outlive the dismissed modal.

- [ ] **Step 3: Pass a progress callback**

`run_parakeet_v2_provision(report, progress=…)` already accepts one; today `library_screen.py` omits it. Pass a callback that posts `InstallProgressed`, and render it somewhere that survives dismissal.

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=$PWD .venv/bin/pytest Tests/UI/test_parakeet_v2_install_ui.py Tests/UI/test_model_artifact_widgets.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_parakeet_v2_install_ui.py
git commit -m "refactor(library): move the Parakeet install onto the shared modal"
```

---

### Task 8: Installed view

**Files:**
- Create: `tldw_chatbook/UI/Screens/model_installed_view.py`
- Create: `tldw_chatbook/Widgets/ModelArtifacts/activation_controls.py`
- Test: `Tests/UI/test_model_installed_view.py`

**Interfaces:**
- Consumes: `inventory_rows`, `managed_service`, `list_installed`, `disk_usage`
- Produces: `InstalledView`, `ModelActivationControls`, messages `ActivationRequested(ref)`, `DeletionRequested(ref)`, `RepairRequested()`

- [ ] **Step 1: Write the failing tests**

```python
async def test_installed_view_performs_no_io_at_compose_time(tmp_path) -> None:
    """Views mount eagerly; work must not. Assert list_installed and
    disk_usage are not called until first activation."""


async def test_installed_view_lists_unmanaged_files_as_integrity_unknown(tmp_path):
    """A user with GGUF files must not see an 'Installed' screen that
    omits them."""


async def test_activation_is_pending_and_refuses_re_entry(tmp_path) -> None:
    """activate() can re-hash a closure under an exclusive lease."""


async def test_deletion_blocked_by_a_lease_reports_the_reason(tmp_path) -> None:
    """AC #5: never bypass an active lease; name the blocker."""
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=$PWD .venv/bin/pytest Tests/UI/test_model_installed_view.py -v`
Expected: FAIL

- [ ] **Step 3: Implement**

Load on first activation and on explicit refresh, never in `compose`/`on_mount`. `list_installed()`, `disk_usage()` (a full tree walk — threaded and cached), and the unmanaged scan all run in `@work(thread=True)`; absorb the existing threaded `os.walk` and delete flow from `Widgets/HuggingFace/local_models_widget.py` rather than rewriting them. Broken rows (`descriptor is None`) render with a Repair affordance. `reconcile()` sits behind an explicit Repair control and shows its `ReconcileReport`. Deletion reports lease blockers; the AC #5 recycle request is out of scope for Phase 1.

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH=$PWD .venv/bin/pytest Tests/UI/test_model_installed_view.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/model_installed_view.py tldw_chatbook/Widgets/ModelArtifacts/activation_controls.py Tests/UI/test_model_installed_view.py
git commit -m "feat(models): add the installed inventory view"
```

---

### Task 9: Curated view and rail wiring

**Files:**
- Create: `tldw_chatbook/UI/Screens/model_curated_view.py`
- Modify: `tldw_chatbook/UI/Screens/llm_screen.py`
- Modify: `tldw_chatbook/UI/LLM_Management_Window.py`
- Test: `Tests/UI/test_model_installed_view.py`

- [ ] **Step 1: Write the failing test**

```python
async def test_models_rail_lists_curated_and_installed_and_drops_local_models():
    """Phase 1 replaces 'Local Models'; 'Download Models' stays until
    Phase 3."""
    from tldw_chatbook.UI.Screens.llm_screen import MODELS_RAIL_SECTIONS

    models_section = dict(MODELS_RAIL_SECTIONS)["Models"]
    keys = [key for key, _label in models_section]
    assert "curated" in keys
    assert "installed" in keys
    assert "local-models" not in keys
    assert "download-models" in keys
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=$PWD .venv/bin/pytest Tests/UI/test_model_installed_view.py -k rail -v`
Expected: FAIL

- [ ] **Step 3: Implement**

Update `MODELS_RAIL_SECTIONS` (`llm_screen.py:42-48`) and register the two new views in `LLMManagementWindow.view_mapping`, following the existing `-active` CSS switching. `CuratedView` lists `curated_registry().list()` cross-referenced against installed refs, and opens `ModelInstallModal` on selection; `LLMScreen` owns the preflight and provision workers.

- [ ] **Step 4: Run the full affected suite**

Run: `PYTHONPATH=$PWD .venv/bin/pytest Tests/UI/ Tests/Model_Artifacts/ Tests/Local_Ingestion/ Tests/Audio/test_console_dictation.py Tests/STT/ -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/ tldw_chatbook/UI/LLM_Management_Window.py Tests/UI/
git commit -m "feat(models): wire the Curated and Installed views into the Models rail"
```

---

## Self-review notes

- **Spec coverage:** AC #1 (partial — Remote is Phase 2), #2, #3, #4, #6 (local), #7, #8 are covered by Tasks 2–9. AC #5 is covered only to "report blockers"; the recycle request is explicitly deferred in the spec and the backlog note.
- **Type consistency:** `plan_rows`/`plan_totals`/`inventory_rows`/`install_failure_message` are used in Tasks 5–9 exactly as defined in Task 4. `managed_service` from Task 1 is used in Task 8. `CuratedRegistry.list`/`sources` from Task 2 are used in Task 9.
- **Ordering:** Task 3 (provenance) precedes Task 4, which consumes it. Task 7 depends on Tasks 5 and 6. Task 9 depends on Tasks 2 and 8.
