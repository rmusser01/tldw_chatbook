"""Tests for the first-run wizard's Speech transcription step (TASK-1301).

Reuses the TASK-596 shared model-artifact controls (ModelInstallModal,
ModelInstallProgress, ModelActivationControls) and the Local_Ingestion
.parakeet_v2_artifact convenience wrappers -- the same ones LibraryScreen's
own Parakeet v2 install surface uses (Tests/UI/test_parakeet_v2_install_ui.py)
-- so these tests follow that file's decomposed unit-testing style: exercise
each handler/callback directly with mocked collaborators, plus one true
end-to-end test that drives the real (module-patched) preflight/provision
functions through the actual @work-decorated bodies.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, RadioButton, Static

from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
    parakeet_v2_descriptor,
    parakeet_v2_reference,
)
from tldw_chatbook.Model_Artifacts import ArtifactRef, ProvenanceClass
from tldw_chatbook.Model_Artifacts.acquisition import (
    ArtifactPreflightEntry,
    PreflightReport,
)
from tldw_chatbook.Model_Artifacts.service import InstalledArtifact
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import SpeechSetupStep
from tldw_chatbook.Widgets.ModelArtifacts import (
    ModelActivationControls,
    ModelInstallModal,
)


def _installed_item(*, active: bool, ready: bool = True, error=None) -> InstalledArtifact:
    return InstalledArtifact(
        path=Path("/fake/models/parakeet-v2"),
        descriptor=parakeet_v2_descriptor(),
        ready=ready,
        active=active,
        error=error,
    )


class _FakeService:
    """Minimal stand-in for ModelArtifactService; no filesystem/network I/O."""

    def __init__(self, *, installed=()):
        self.installed = list(installed)
        self.activate_calls: list[ArtifactRef] = []
        self.delete_calls: list[ArtifactRef] = []

    def list_installed(self):
        return tuple(self.installed)

    def activate(self, reference: ArtifactRef) -> None:
        self.activate_calls.append(reference)

    def delete(self, reference: ArtifactRef) -> None:
        self.delete_calls.append(reference)


def _report(*, destination: Path) -> PreflightReport:
    ref = parakeet_v2_reference()
    entry = ArtifactPreflightEntry(
        ref=ref,
        source_url="https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/x/config.json",
        repository="istupakov/parakeet-tdt-0.6b-v2-onnx",
        revision=ref.revision,
        license_id="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        precision="int8",
        total_bytes=661_191_781,
        file_count=4,
        already_installed=False,
        provenance=(ProvenanceClass.CHATBOOK_CURATED,),
    )
    return PreflightReport(
        root=ref,
        closure_fingerprint="f" * 64,
        entries=(entry,),
        download_bytes=661_191_781,
        already_staged_bytes=0,
        staging_overhead_bytes=0,
        retained_bytes=0,
        destination=destination,
        free_bytes=10**12,
        required_bytes=900_000_000,
        sufficient_space=True,
        gating_errors=(),
    )


def _wizard(**overrides):
    base = dict(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _step(*, installed=(), wizard=None) -> SpeechSetupStep:
    return SpeechSetupStep(
        wizard=wizard or _wizard(),
        config=WizardStepConfig(id="speech", title="Speech", step_number=5),
        service_factory=lambda: _FakeService(installed=installed),
    )


class _StepHost(App):
    def __init__(self, step):
        super().__init__()
        self._step = step

    def compose(self) -> ComposeResult:
        yield self._step


def _patch_app(monkeypatch) -> MagicMock:
    """SpeechSetupStep.app is a read-only Textual property; replace it on
    the class for this test only (monkeypatch reverts automatically) --
    same technique Tests/UI/test_parakeet_v2_install_ui.py uses for
    LibraryScreen.app."""
    fake_app = MagicMock()
    monkeypatch.setattr(SpeechSetupStep, "app", property(lambda self: fake_app))
    return fake_app


# ---------------------------------------------------------------------------
# Rendering: options come from the STT policy/catalog, gated by curated
# availability (AC#2); no I/O happens until on_show's worker runs.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compose_shows_recommended_defaults_and_disables_unavailable_options():
    step = _step(installed=())
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        text = "\n".join(str(s.render()) for s in step.query(Static))
        assert "English" in text
        assert "recommended" in text.lower()
        assert "INT8" in text
        # A v3-only language and F32 are declared by the catalog (so the UI
        # is honest about what the policy defines) but neither has a curated
        # descriptor to download yet -- they must render disabled.
        radio_buttons = list(step.query(RadioButton))
        disabled = [b for b in radio_buttons if b.disabled]
        enabled = [b for b in radio_buttons if not b.disabled]
        assert disabled, "unavailable languages/precisions must still be listed, disabled"
        assert enabled, "English/INT8 must be selectable"
        assert not any("English" in str(b.label) for b in disabled)
        assert not any(str(b.label).upper().startswith("INT8") for b in disabled)


def test_compose_step_alone_does_no_io():
    """compose_step() itself (before any I/O-triggering lifecycle hook runs)
    must build entirely from pure catalog/curated-registry state -- neither
    touches the service factory."""
    factory = MagicMock(return_value=_FakeService())
    step = SpeechSetupStep(
        wizard=_wizard(),
        config=WizardStepConfig(id="speech", title="Speech", step_number=5),
        service_factory=factory,
    )
    status_text, action_widget = step._status_and_action()
    assert "Checking" in status_text
    assert action_widget is None
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_not_installed_shows_install_button_not_activation_controls():
    step = _step(installed=())
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.2)
        assert step.query_one("#setup-speech-install", Button)
        assert not step.query(ModelActivationControls)


@pytest.mark.asyncio
async def test_already_active_shows_installed_state_no_install_button():
    """AC#4: already-installed models show installed state + activation
    controls, never a re-download offer."""
    step = _step(installed=[_installed_item(active=True, ready=True)])
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.2)
        assert not step.query("#setup-speech-install")
        controls = step.query_one(ModelActivationControls)
        assert controls.active is True


@pytest.mark.asyncio
async def test_installed_but_not_active_shows_activation_controls():
    step = _step(installed=[_installed_item(active=False, ready=True)])
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.2)
        controls = step.query_one(ModelActivationControls)
        assert controls.active is False
        assert controls.ready is True


# ---------------------------------------------------------------------------
# Install trigger + consent flow (AC#3/#4): unit-level, mirroring
# Tests/UI/test_parakeet_v2_install_ui.py's decomposed style.
# ---------------------------------------------------------------------------


def test_install_button_triggers_preflight_worker():
    step = _step()
    step._preflight_install = MagicMock()
    step.refresh = MagicMock()
    step._install_pressed()
    step._preflight_install.assert_called_once_with()
    assert step._operation == "install"


def test_install_button_no_op_while_an_operation_is_pending():
    step = _step()
    step._operation = "install"
    step._preflight_install = MagicMock()
    step._install_pressed()
    step._preflight_install.assert_not_called()


def test_preflight_result_pushes_modal_built_from_the_report(tmp_path, monkeypatch):
    fake_app = _patch_app(monkeypatch)
    step = _step()
    report = _report(destination=tmp_path / "d")

    step._apply_preflight_result(report, None)

    assert step._pending_report is report
    fake_app.push_screen.assert_called_once()
    modal, callback = fake_app.push_screen.call_args[0]
    assert isinstance(modal, ModelInstallModal)
    assert modal.report is report
    assert callback == step._confirm_install


def test_preflight_failure_notifies_and_does_not_push_modal(monkeypatch):
    fake_app = _patch_app(monkeypatch)
    step = _step()
    step.notify = MagicMock()
    step.refresh = MagicMock()
    step._operation = "install"

    step._apply_preflight_result(None, "boom")

    assert step._operation is None
    step.notify.assert_called_once()
    assert step.notify.call_args.kwargs.get("severity") == "error"
    fake_app.push_screen.assert_not_called()


def test_confirming_install_starts_provision_worker(tmp_path):
    step = _step()
    step._pending_report = _report(destination=tmp_path / "d")
    step._provision_install = MagicMock()
    step.refresh = MagicMock()

    step._confirm_install(True)

    step._provision_install.assert_called_once_with()


def test_declining_install_resets_state_without_provisioning(tmp_path):
    step = _step()
    step._pending_report = _report(destination=tmp_path / "d")
    step._operation = "install"
    step._provision_install = MagicMock()
    step.refresh = MagicMock()

    step._confirm_install(False)

    step._provision_install.assert_not_called()
    assert step._pending_report is None
    assert step._operation is None


def test_provision_success_notifies_and_reloads_installed_state():
    step = _step()
    step._operation = "install"
    step._pending_report = object()
    step.notify = MagicMock()
    step._ensure_loaded = MagicMock()

    step._apply_provision_result(None)

    assert step._operation is None
    assert step._pending_report is None
    step.notify.assert_called_once()
    assert step.notify.call_args.kwargs.get("severity") == "information"
    step._ensure_loaded.assert_called_once_with(force=True)


def test_provision_failure_notifies_error_but_still_reloads_never_traps():
    """AC#6: a failed download must not leave the step stuck -- it still
    refreshes installed state so the wizard stays fully navigable."""
    step = _step()
    step._operation = "install"
    step.notify = MagicMock()
    step._ensure_loaded = MagicMock()

    step._apply_provision_result("disk full")

    step.notify.assert_called_once()
    assert step.notify.call_args.kwargs.get("severity") == "error"
    step._ensure_loaded.assert_called_once_with(force=True)


# ---------------------------------------------------------------------------
# Activation / deletion (AC#4): reused verbatim from the 596 controls.
# ---------------------------------------------------------------------------


def test_activation_requested_triggers_activate_worker():
    step = _step()
    step._activate_model = MagicMock()
    step.refresh = MagicMock()
    event = MagicMock(reference=parakeet_v2_reference())

    step._activation_requested(event)

    event.stop.assert_called_once_with()
    step._activate_model.assert_called_once_with()
    assert step._operation == "activate"


def test_activate_worker_calls_service_activate_with_the_exact_reference(monkeypatch):
    fake_app = _patch_app(monkeypatch)
    fake_app.call_from_thread.side_effect = lambda fn, *a, **kw: fn(*a, **kw)
    service = _FakeService()
    step = SpeechSetupStep(
        wizard=_wizard(),
        config=WizardStepConfig(id="speech", title="Speech", step_number=5),
        service_factory=lambda: service,
    )

    SpeechSetupStep._activate_model.__wrapped__(step)

    assert service.activate_calls == [parakeet_v2_reference()]


def test_deletion_requested_pushes_confirmation_dialog(monkeypatch):
    fake_app = _patch_app(monkeypatch)
    step = _step()
    event = MagicMock(reference=parakeet_v2_reference())

    step._deletion_requested(event)

    event.stop.assert_called_once_with()
    fake_app.push_screen.assert_called_once()
    dialog, callback = fake_app.push_screen.call_args[0]
    assert callback == step._confirm_deletion


def test_confirmed_deletion_starts_delete_worker():
    step = _step()
    step._delete_model = MagicMock()
    step.refresh = MagicMock()

    step._confirm_deletion(True)

    step._delete_model.assert_called_once_with()
    assert step._operation == "delete"


def test_declined_deletion_does_not_delete():
    step = _step()
    step._delete_model = MagicMock()

    step._confirm_deletion(False)

    step._delete_model.assert_not_called()


# ---------------------------------------------------------------------------
# Persistence gate (AC#5): commit() only writes [transcription] after a
# fresh, off-loop re-verification that the managed artifact is active.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_commit_is_skip_safe_when_never_verified_active(monkeypatch):
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    monkeypatch.setattr(
        wizard_module, "active_managed_parakeet_v2_dir", lambda service: None
    )
    wizard = _wizard()
    step = _step(wizard=wizard)

    ok, error = await step.commit()

    assert ok, error
    wizard.commit_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_commit_persists_the_recommended_selection_once_verified_active(
    tmp_path, monkeypatch
):
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    active_dir = tmp_path / "installed"
    monkeypatch.setattr(
        wizard_module, "active_managed_parakeet_v2_dir", lambda service: active_dir
    )
    wizard = _wizard()
    step = _step(wizard=wizard)

    ok, error = await step.commit()

    assert ok, error
    committed = wizard.commit_config.call_args.args[0]
    assert committed == {
        "transcription": {
            "default_provider": "parakeet-onnx",
            "default_model": "nemo-parakeet-tdt-0.6b-v2",
            "default_language": "en",
        }
    }


@pytest.mark.asyncio
async def test_commit_reports_failure_when_persistence_write_fails(monkeypatch):
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    monkeypatch.setattr(
        wizard_module,
        "active_managed_parakeet_v2_dir",
        lambda service: Path("/fake/active"),
    )
    wizard = _wizard(commit_config=AsyncMock(return_value=False))
    step = _step(wizard=wizard)

    ok, error = await step.commit()

    assert ok is False
    assert error


# ---------------------------------------------------------------------------
# One true end-to-end pass: drives the actual @work-decorated bodies through
# the real (module-patched) preflight/provision functions, proving the
# button -> preflight -> modal -> provision -> reload wiring is real, not
# just individually-mocked assertions above.
# ---------------------------------------------------------------------------


def test_end_to_end_install_flow_calls_the_real_wrapped_functions(
    tmp_path, monkeypatch
):
    """Deliberately a plain (non-async) test: __wrapped__ bypasses @work and
    calls the body directly, which internally does asyncio.run() -- exactly
    like a real worker thread -- so this must NOT run inside pytest-asyncio's
    own event loop (mirrors Tests/UI/test_parakeet_v2_install_ui.py::
    test_install_worker_passes_a_progress_callback, the same technique for
    Library's own Parakeet v2 install worker)."""
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    report = _report(destination=tmp_path / "d")
    preflight_calls: list[None] = []
    provision_calls: list[PreflightReport] = []

    async def fake_preflight():
        preflight_calls.append(None)
        return report

    async def fake_provision(passed_report, *, progress=None):
        provision_calls.append(passed_report)
        return tmp_path / "installed"

    monkeypatch.setattr(wizard_module, "run_parakeet_v2_preflight", fake_preflight)
    monkeypatch.setattr(wizard_module, "run_parakeet_v2_provision", fake_provision)
    fake_app = _patch_app(monkeypatch)
    fake_app.call_from_thread.side_effect = lambda fn, *a, **kw: fn(*a, **kw)

    step = _step()
    step.notify = MagicMock()
    step._ensure_loaded = MagicMock()
    step.refresh = MagicMock()

    SpeechSetupStep._preflight_install.__wrapped__(step)

    assert preflight_calls == [None]
    assert step._pending_report is report
    fake_app.push_screen.assert_called_once()

    SpeechSetupStep._provision_install.__wrapped__(step)

    assert provision_calls == [report]
    step.notify.assert_called_once()
    assert step.notify.call_args.kwargs.get("severity") == "information"
    step._ensure_loaded.assert_called_once_with(force=True)
