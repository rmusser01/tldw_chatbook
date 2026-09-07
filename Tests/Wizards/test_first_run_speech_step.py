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

import asyncio
import builtins
from pathlib import Path
import threading
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from loguru import logger
from textual.app import ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button, RadioButton, Static

from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
    PARAKEET_PRECISIONS,
    parakeet_descriptor,
    parakeet_reference,
    parakeet_v2_descriptor,
    parakeet_v2_reference,
    parakeet_vad_descriptor,
    parakeet_vad_reference,
)
from tldw_chatbook.Model_Artifacts import ArtifactRef, ProvenanceClass
from tldw_chatbook.Model_Artifacts.acquisition import (
    ArtifactPreflightEntry,
    PreflightReport,
)
from tldw_chatbook.Model_Artifacts.service import InstalledArtifact
from tldw_chatbook.STT.parakeet_external import (
    ExternalParakeetErrorCode,
    ExternalParakeetVerificationError,
)
from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey
from tldw_chatbook.Third_Party.textual_fspicker import SelectDirectory
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
    SetupWizardContainer,
    SpeechSetupStep,
    SummaryStep,
)
from tldw_chatbook.Widgets.ModelArtifacts import (
    InstallProgressed,
    ModelActivationControls,
    ModelInstallModal,
)


def test_curated_speech_helpers_do_not_import_the_full_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def reject_curated_registry(name: str, *args: object, **kwargs: object):
        if name == "tldw_chatbook.Model_Artifacts.curated_registry":
            raise AssertionError("speech helpers imported the full curated registry")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_curated_registry)
    policy = __import__(
        "tldw_chatbook.UI.Wizards.first_run_speech_step_state",
        fromlist=["routing_policy"],
    ).routing_policy()
    model_ids = frozenset({"nemo-parakeet-tdt-0.6b-v2", "nemo-parakeet-tdt-0.6b-v3"})

    assert SpeechSetupStep._curated_model_ids() == model_ids
    assert SpeechSetupStep._curated_selections() == frozenset(
        (model_id, precision) for model_id in model_ids for precision in ("int8", "f32")
    )
    assert model_ids == frozenset(
        {policy.parakeet_v2_model_id, policy.parakeet_v3_model_id}
    )
    assert PARAKEET_PRECISIONS == ("int8", "f32")


def _installed_item(
    *, active: bool, ready: bool = True, error=None
) -> InstalledArtifact:
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


def _report(
    *,
    destination: Path,
    model: str = "nemo-parakeet-tdt-0.6b-v2",
    precision: str = "int8",
) -> PreflightReport:
    descriptor = parakeet_descriptor(model, precision)
    ref = descriptor.reference
    entry = ArtifactPreflightEntry(
        ref=ref,
        source_url=descriptor.source_url,
        repository=descriptor.upstream_repository,
        revision=ref.revision,
        license_id=descriptor.license_id,
        license_url=descriptor.license_url,
        precision=precision,
        total_bytes=descriptor.expected_installed_bytes,
        file_count=len(descriptor.files),
        already_installed=False,
        provenance=(ProvenanceClass.CHATBOOK_CURATED,),
    )
    return PreflightReport(
        root=ref,
        closure_fingerprint="f" * 64,
        entries=(entry,),
        download_bytes=descriptor.expected_installed_bytes,
        already_staged_bytes=0,
        staging_overhead_bytes=0,
        retained_bytes=0,
        destination=destination,
        free_bytes=10**12,
        required_bytes=900_000_000,
        sufficient_space=True,
        gating_errors=(),
    )


def _vad_report(*, destination: Path) -> PreflightReport:
    descriptor = parakeet_vad_descriptor()
    ref = descriptor.reference
    entry = ArtifactPreflightEntry(
        ref=ref,
        source_url=descriptor.source_url,
        repository=descriptor.upstream_repository,
        revision=ref.revision,
        license_id=descriptor.license_id,
        license_url=descriptor.license_url,
        precision=descriptor.precision,
        total_bytes=descriptor.expected_installed_bytes,
        file_count=len(descriptor.files),
        already_installed=False,
        provenance=(ProvenanceClass.CHATBOOK_CURATED,),
    )
    return PreflightReport(
        root=ref,
        closure_fingerprint="v" * 64,
        entries=(entry,),
        download_bytes=descriptor.expected_installed_bytes,
        already_staged_bytes=0,
        staging_overhead_bytes=0,
        retained_bytes=0,
        destination=destination,
        free_bytes=10**12,
        required_bytes=descriptor.expected_installed_bytes,
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


def _wizard_with_source(source_service, **overrides):
    app_instance = SimpleNamespace(
        app_config={},
        _parakeet_source_service=source_service,
        _ensure_parakeet_source_service=lambda: source_service,
    )
    return _wizard(app_instance=app_instance, **overrides)


def _wizard_with_real_commit(source_service):
    wizard = _wizard_with_source(source_service)
    wizard._mirror_into_app_config = MethodType(
        SetupWizardContainer._mirror_into_app_config,
        wizard,
    )
    wizard.commit_config = MethodType(SetupWizardContainer.commit_config, wizard)
    return wizard


class _OwnerTrackingSource:
    """Stateful source double: releases and accepts change observable state."""

    def __init__(self, records=None):
        self._records = dict(records or {})
        self.owners: set[str] = set()
        self.accepted = []
        self.prepared_commit = None

    def records(self):
        return dict(self._records)

    def release_scope(self, scope_id):
        self.owners.discard(scope_id)

    def prepare_config_commit(self, prepared):
        return self.prepared_commit

    def accept_committed(self, commit):
        self.accepted.append(commit)


def _step(*, installed=(), wizard=None, runtime_installed=None) -> SpeechSetupStep:
    # runtime_installed defaults to True (not the real probe) so these tests
    # are deterministic regardless of whether onnx-asr happens to be
    # installed in the environment running the suite -- see Important 4.
    return SpeechSetupStep(
        wizard=wizard or _wizard(),
        config=WizardStepConfig(id="speech", title="Speech", step_number=5),
        service_factory=lambda: _FakeService(installed=installed),
        runtime_installed=runtime_installed or (lambda: True),
    )


def _active_lookup(result):
    return lambda model, precision, *, service: result


class _StepHost(ConsolidatedCSSApp):
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
async def test_compose_shows_all_curated_parakeet_options():
    step = _step(installed=())
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        text = "\n".join(str(s.render()) for s in step.query(Static))
        assert "English" in text
        assert "recommended" in text.lower()
        assert "INT8" in text
        radio_buttons = list(step.query(RadioButton))
        enabled = [b for b in radio_buttons if not b.disabled]
        assert len(enabled) == len(radio_buttons)
        assert any("English" in str(b.label) for b in enabled)
        assert any(str(b.label).upper().startswith("F32") for b in enabled)


@pytest.mark.asyncio
async def test_non_english_f32_selection_changes_the_exact_managed_target():
    from tldw_chatbook.UI.Wizards import first_run_speech_step_state as speech_state

    policy = speech_state.routing_policy()
    language = sorted(policy.validated_v3_languages)[0]
    step = _step()
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click(f"#setup-speech-language-{language}")
        await pilot.pause()
        step.query_one("#setup-speech-precision-f32", RadioButton).value = True
        await pilot.pause()

        assert step._reference == parakeet_reference(policy.parakeet_v3_model_id, "f32")
        assert "Parakeet v3" in step._model_label()
        assert "F32" in step._model_label()


@pytest.mark.asyncio
async def test_summary_checks_the_configured_exact_artifact(tmp_path, monkeypatch):
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module
    from tldw_chatbook.UI.Wizards import first_run_speech_step_state as speech_state

    policy = speech_state.routing_policy()
    language = sorted(policy.validated_v3_languages)[0]
    calls = []

    def exact_active(model, precision):
        calls.append((model, precision))
        return tmp_path

    monkeypatch.setattr(wizard_module, "managed_model_artifact_root", lambda: tmp_path)
    monkeypatch.setattr(wizard_module, "active_managed_parakeet_dir", exact_active)
    wizard = _wizard(wizard_data={"welcome": {"track": "quick"}})
    step = SummaryStep(
        wizard=wizard,
        config=WizardStepConfig(id="summary", title="Summary", step_number=9),
        load_config=lambda: {
            "transcription": {
                "default_provider": policy.parakeet_provider_id,
                "default_model": policy.parakeet_v3_model_id,
                "default_language": language,
                "default_precision": "f32",
            }
        },
        rag_deps_installed=lambda: False,
        speech_runtime_installed=lambda: True,
    )
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)

    assert calls
    assert set(calls) == {(policy.parakeet_v3_model_id, "f32")}


@pytest.mark.asyncio
async def test_compose_exposes_path_free_existing_gguf_configuration():
    wizard = _wizard(
        app_instance=MagicMock(
            app_config={
                "transcription": {
                    "transcribe_cpp": {"model_path": "/private/model.gguf"}
                }
            }
        )
    )
    step = _step(wizard=wizard)
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        button = step.query_one("#setup-speech-choose-transcribe-cpp-gguf", Button)
        status = step.query_one("#setup-speech-transcribe-cpp-status", Static)
        assert "Choose another GGUF" in str(button.label)
        assert "configured" in str(status.renderable).lower()
        assert "/private/model.gguf" not in str(status.renderable)


def test_transcribe_cpp_config_worker_reports_path_free_success(tmp_path, monkeypatch):
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    selected = tmp_path / "private-model.gguf"
    configured: list[Path] = []
    fake_app = _patch_app(monkeypatch)
    monkeypatch.setattr(
        wizard_module,
        "configure_transcribe_cpp_model_path",
        lambda path: configured.append(path),
    )
    step = _step()
    step._apply_transcribe_cpp_gguf_result = MagicMock()

    SpeechSetupStep._configure_transcribe_cpp_gguf.__wrapped__(step, selected)

    assert configured == [selected]
    fake_app.call_from_thread.assert_called_once_with(
        step._apply_transcribe_cpp_gguf_result,
        True,
    )
    assert str(selected) not in repr(fake_app.call_from_thread.call_args)


def test_compose_step_alone_does_no_io():
    """compose_step() itself (before any I/O-triggering lifecycle hook runs)
    must build entirely from pure catalog/curated-registry state -- neither
    touches the service factory."""
    factory = MagicMock(return_value=_FakeService())
    step = SpeechSetupStep(
        wizard=_wizard(),
        config=WizardStepConfig(id="speech", title="Speech", step_number=5),
        service_factory=factory,
        runtime_installed=lambda: True,
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


@pytest.mark.asyncio
async def test_broken_or_not_ready_artifact_still_offers_activation_controls():
    """Important 5: a broken/not-ready installed item must not be a dead
    end -- ModelActivationControls(ready=False) already keeps Delete
    enabled while disabling Activate, so wire it instead of returning no
    action widget at all."""
    step = _step(
        installed=[_installed_item(active=False, ready=False, error="corrupt manifest")]
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.2)
        controls = step.query_one(ModelActivationControls)
        assert controls.ready is False
        assert controls.active is False
        # The 596 widget itself keeps Delete enabled and Activate disabled
        # for ready=False -- confirm the real rendered buttons agree.
        activate = step.query_one(".model-activate", Button)
        delete = step.query_one(".model-delete", Button)
        assert activate.disabled is True
        assert delete.disabled is False


# ---------------------------------------------------------------------------
# Review NEW-2: installed+active+configured-elsewhere must not promise an
# action ("installing or activating") that no control on screen offers.
# Chosen fix: a real "Use Parakeet v2 as my default" affordance that sets
# _acted_this_run -- makes the prefill sentence's promise true instead of
# just rewording it away.
# ---------------------------------------------------------------------------

_USE_AS_DEFAULT_ID = "#setup-speech-use-as-default"


def _elsewhere_wizard() -> SimpleNamespace:
    return _wizard(
        app_instance=MagicMock(
            app_config={
                "transcription": {
                    "default_provider": "remote-whisper",
                    "default_model": "whisper-1",
                    "default_language": "auto",
                }
            }
        )
    )


@pytest.mark.asyncio
async def test_use_as_default_offered_when_active_and_configured_elsewhere():
    step = _step(
        installed=[_installed_item(active=True, ready=True)],
        wizard=_elsewhere_wizard(),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.2)
        button = step.query_one(_USE_AS_DEFAULT_ID, Button)
        assert not button.disabled
        assert "Use Parakeet v2 (English, INT8) as my default" in str(button.label)
        text = "\n".join(str(s.render()) for s in step.query(Static))
        assert "installing or activating" not in text


@pytest.mark.asyncio
async def test_use_as_default_not_offered_when_already_the_default():
    """prefill already matches parakeet-onnx -- nothing to switch."""
    wizard = _wizard(
        app_instance=MagicMock(
            app_config={
                "transcription": {
                    "default_provider": "parakeet-onnx",
                    "default_model": "nemo-parakeet-tdt-0.6b-v2",
                    "default_language": "en",
                }
            }
        )
    )
    step = _step(installed=[_installed_item(active=True, ready=True)], wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.2)
        assert not step.query(_USE_AS_DEFAULT_ID)


@pytest.mark.asyncio
async def test_use_as_default_not_offered_when_not_active():
    """Installed but not yet active: Activate is the real path, not this."""
    step = _step(
        installed=[_installed_item(active=False, ready=True)],
        wizard=_elsewhere_wizard(),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.2)
        assert not step.query(_USE_AS_DEFAULT_ID)


@pytest.mark.asyncio
async def test_use_as_default_not_offered_when_nothing_installed():
    step = _step(installed=(), wizard=_elsewhere_wizard())
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.2)
        assert not step.query(_USE_AS_DEFAULT_ID)


@pytest.mark.asyncio
async def test_use_as_default_not_offered_when_runtime_missing():
    """Never offer switching to a provider the runtime cannot execute."""
    step = _step(
        installed=[_installed_item(active=True, ready=True)],
        wizard=_elsewhere_wizard(),
        runtime_installed=lambda: False,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.2)
        assert not step.query(_USE_AS_DEFAULT_ID)


@pytest.mark.asyncio
async def test_use_as_default_not_offered_once_already_acted_this_run():
    step = _step(
        installed=[_installed_item(active=True, ready=True)],
        wizard=_elsewhere_wizard(),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.2)
        assert step.query(_USE_AS_DEFAULT_ID)
        step._acted_this_run = True
        step.refresh(recompose=True)
        await pilot.pause(0.2)
        assert not step.query(_USE_AS_DEFAULT_ID)


@pytest.mark.asyncio
async def test_pressing_use_as_default_sets_acted_flag_and_updates_copy():
    step = _step(
        installed=[_installed_item(active=True, ready=True)],
        wizard=_elsewhere_wizard(),
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.2)
        assert step._acted_this_run is False

        step.query_one(_USE_AS_DEFAULT_ID, Button).press()
        await pilot.pause(0.2)

        assert step._acted_this_run is True
        assert not step.query(_USE_AS_DEFAULT_ID)
        text = "\n".join(str(s.render()) for s in step.query(Static))
        assert "will become your default" in text.lower()


@pytest.mark.asyncio
async def test_commit_persists_after_use_as_default_without_reinstalling(monkeypatch):
    """End-to-end proof the affordance actually makes the promise true:
    pressing it, then Next (commit()), persists the recommended selection
    with no install/activate worker involved."""
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    active_dir = Path("/fake/active")
    monkeypatch.setattr(
        wizard_module, "active_managed_parakeet_dir", _active_lookup(active_dir)
    )
    wizard = _elsewhere_wizard()
    step = _step(installed=[_installed_item(active=True, ready=True)], wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.2)
        step.query_one(_USE_AS_DEFAULT_ID, Button).press()
        await pilot.pause(0.2)

    ok, error = await step.commit()

    assert ok, error
    committed = wizard.commit_config.call_args.args[0]
    assert committed == {
        "transcription": {
            "default_provider": "parakeet-onnx",
            "default_model": "nemo-parakeet-tdt-0.6b-v2",
            "default_language": "en",
            "default_precision": "int8",
        }
    }


# ---------------------------------------------------------------------------
# Runtime-dependency gate (Important 4): mirrors RagStep's own
# embeddings_rag_deps_installed() gate exactly -- missing extra means no
# download remains disabled, regardless of curated/artifact-service state.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_runtime_extra_shows_install_instructions_no_download_offered():
    step = _step(installed=(), runtime_installed=lambda: False)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.2)
        text = "\n".join(str(s.render()) for s in step.query(Static))
        assert "onnx-asr" in text
        assert step.query_one("#setup-speech-install", Button).disabled
        assert not step.query(ModelActivationControls)


@pytest.mark.asyncio
async def test_missing_runtime_extra_message_shown_immediately_without_waiting_for_load():
    """The gate is checked BEFORE the installed-state load completes, so a
    user on a minimal install sees the real reason immediately instead of a
    "Checking installed models…" placeholder that never resolves into
    anything actionable."""
    step = _step(runtime_installed=lambda: False)
    status_text, action_widget = step._status_and_action()
    assert "onnx-asr" in status_text
    assert isinstance(action_widget, Button)
    assert action_widget.id == "setup-speech-install"
    assert action_widget.disabled


@pytest.mark.asyncio
async def test_runtime_extra_present_offers_the_normal_install_flow():
    step = _step(installed=(), runtime_installed=lambda: True)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.2)
        assert step.query_one("#setup-speech-install", Button)


@pytest.mark.asyncio
async def test_missing_runtime_status_is_not_markup_parsed():
    """Review NEW-1: the extras name and pinned profile contain literal
    brackets ("tldw_chatbook[transcription_parakeet_onnx]",
    "onnx-asr[cpu]==0.12.0"). Rich markup parses "[...]" as a tag and
    silently deletes unrecognized ones unless the Static is built with
    markup=False -- the same fix this file already applies to
    "#setup-summary-rows" for the identical trap. Without it the user is
    told to install an extras package with no extra named, and to run
    pip install 'onnx-asr==0.12.0' (missing the pinned [cpu] profile)."""
    step = _step(runtime_installed=lambda: False)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        status = step.query_one("#setup-speech-status", Static)
        rendered = str(status.render())
        assert "tldw_chatbook[transcription_parakeet_onnx]" in rendered
        assert "onnx-asr[cpu]==0.12.0" in rendered


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


def test_provision_success_marks_the_step_as_acted_on_this_run():
    """Important 3: only a successful install/activation THIS run may make
    commit() persist -- this is where that flag gets set."""
    step = _step()
    step.notify = MagicMock()
    step._ensure_loaded = MagicMock()
    assert step._acted_this_run is False

    step._apply_provision_result(None)

    assert step._acted_this_run is True


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


def test_provision_failure_does_not_mark_the_step_as_acted():
    """A failed install must not later let commit() persist as if the user
    had successfully set anything up."""
    step = _step()
    step.notify = MagicMock()
    step._ensure_loaded = MagicMock()

    step._apply_provision_result("disk full")

    assert step._acted_this_run is False


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


def test_successful_activation_marks_the_step_as_acted_on_this_run():
    """Important 3: activating counts as engagement, same as installing."""
    step = _step()
    step._operation = "activate"
    step.notify = MagicMock()
    step._ensure_loaded = MagicMock()

    step._apply_lifecycle_result(None)

    assert step._acted_this_run is True


def test_successful_deletion_does_not_mark_the_step_as_acted():
    """Deleting is not "opting in" -- it must not make commit() persist
    the recommended selection (the artifact will not be active afterwards
    anyway, but this pins the flag directly)."""
    step = _step()
    step._operation = "delete"
    step.notify = MagicMock()
    step._ensure_loaded = MagicMock()

    step._apply_lifecycle_result(None)

    assert step._acted_this_run is False


def test_failed_activation_does_not_mark_the_step_as_acted():
    step = _step()
    step._operation = "activate"
    step.notify = MagicMock()
    step._ensure_loaded = MagicMock()

    step._apply_lifecycle_result("boom")

    assert step._acted_this_run is False


# ---------------------------------------------------------------------------
# Persistence gate (AC#5): commit() only writes [transcription] after a
# fresh, off-loop re-verification that the managed artifact is active.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_commit_is_skip_safe_when_never_verified_active(monkeypatch):
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    monkeypatch.setattr(
        wizard_module, "active_managed_parakeet_dir", _active_lookup(None)
    )
    wizard = _wizard()
    step = _step(wizard=wizard)
    step._acted_this_run = (
        True  # even "acted" must not matter without an active artifact
    )

    ok, error = await step.commit()

    assert ok, error
    wizard.commit_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_commit_does_not_verify_or_persist_an_unavailable_selection(monkeypatch):
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    active_lookup = MagicMock(return_value=Path("/should-not-be-used"))
    monkeypatch.setattr(wizard_module, "active_managed_parakeet_dir", active_lookup)
    wizard = _wizard()
    step = _step(wizard=wizard)
    step._curated_selections = lambda: frozenset()
    step._acted_this_run = True

    ok, error = await step.commit()

    assert ok, error
    active_lookup.assert_not_called()
    wizard.commit_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_commit_does_not_persist_when_active_but_user_did_not_act_this_run(
    tmp_path, monkeypatch
):
    """Important 3 (the core clobbering fix): an artifact installed in an
    EARLIER session (e.g. via the Library screen) being active must NOT be
    enough on its own. A re-run that just presses Next through this step
    (never installed/activated anything here) must leave whatever is
    already persisted in [transcription] completely untouched -- proven at
    the correct boundary: wizard.commit_config, the only write path, is
    never even awaited, so no bytes of the real config file can change."""
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    active_dir = tmp_path / "installed"
    monkeypatch.setattr(
        wizard_module, "active_managed_parakeet_dir", _active_lookup(active_dir)
    )
    wizard = _wizard()
    step = _step(wizard=wizard)
    assert step._acted_this_run is False  # sanity: nothing was done this run

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
        wizard_module, "active_managed_parakeet_dir", _active_lookup(active_dir)
    )
    wizard = _wizard()
    step = _step(wizard=wizard)
    step._acted_this_run = True  # the user just installed/activated it THIS run

    ok, error = await step.commit()

    assert ok, error
    committed = wizard.commit_config.call_args.args[0]
    assert committed == {
        "transcription": {
            "default_provider": "parakeet-onnx",
            "default_model": "nemo-parakeet-tdt-0.6b-v2",
            "default_language": "en",
            "default_precision": "int8",
        }
    }


@pytest.mark.asyncio
async def test_commit_persists_the_live_pressed_default_selection(
    tmp_path, monkeypatch
):
    """PR #1184 review (finding 2), widget-level proof: commit() must read
    the SELECTED language/precision from the mounted step, not a hardcoded
    constant -- ``test_commit_persists_the_recommended_selection_once_verified_active``
    above uses an UNMOUNTED step, so it only ever exercises
    resolve_speech_selection's "" fallback branch. This test mounts the
    real step (real curated_registry, nothing monkeypatched about
    selectability) so English/INT8 is genuinely the only pre-pressed radio,
    and proves the live-radio path is byte-identical to the old constant."""
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    active_dir = tmp_path / "installed"
    monkeypatch.setattr(
        wizard_module, "active_managed_parakeet_dir", _active_lookup(active_dir)
    )
    wizard = _wizard()
    step = _step(wizard=wizard)
    step._acted_this_run = True
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # commit() must run while the step is still mounted -- it reads the
        # live RadioSet's pressed_button, which is gone once the App tears
        # its widget tree down on exit (proven by _effective_language()
        # falling back to "" post-exit; see the unmounted-fallback tests).
        ok, error = await step.commit()

    assert ok, error
    committed = wizard.commit_config.call_args.args[0]
    assert committed == {
        "transcription": {
            "default_provider": "parakeet-onnx",
            "default_model": "nemo-parakeet-tdt-0.6b-v2",
            "default_language": "en",
            "default_precision": "int8",
        }
    }


@pytest.mark.asyncio
async def test_commit_follows_a_hypothetical_second_selectable_language(
    tmp_path, monkeypatch
):
    """Divergence-proofing (finding 2): make a v3 language selectable (as a
    future curated descriptor would), press it, and prove commit() persists
    THAT selection instead of silently keeping the English/v2 default --
    the exact scenario the review flagged as "will silently diverge the
    moment a second combination becomes selectable"."""
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module
    from tldw_chatbook.UI.Wizards import first_run_speech_step_state as speech_state

    policy = speech_state.routing_policy()
    v3_language = sorted(policy.validated_v3_languages)[0]

    active_dir = tmp_path / "installed"
    monkeypatch.setattr(
        wizard_module, "active_managed_parakeet_dir", _active_lookup(active_dir)
    )
    wizard = _wizard()
    step = _step(wizard=wizard)
    step._acted_this_run = True
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click(f"#setup-speech-language-{v3_language}")
        await pilot.pause()
        # commit() must run before the App tears its widget tree down (see
        # the sibling byte-identical test's comment for why).
        ok, error = await step.commit()

    assert ok, error
    committed = wizard.commit_config.call_args.args[0]
    assert committed == {
        "transcription": {
            "default_provider": "parakeet-onnx",
            "default_model": policy.parakeet_v3_model_id,
            "default_language": v3_language,
            "default_precision": "int8",
        }
    }


@pytest.mark.asyncio
async def test_commit_requires_and_persists_the_exact_v3_f32_artifact(
    tmp_path, monkeypatch
):
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module
    from tldw_chatbook.UI.Wizards import first_run_speech_step_state as speech_state

    policy = speech_state.routing_policy()
    language = sorted(policy.validated_v3_languages)[0]
    active_calls = []

    def exact_active(model, precision, *, service):
        active_calls.append((model, precision))
        return (
            tmp_path / "installed"
            if (model, precision)
            == (
                policy.parakeet_v3_model_id,
                "f32",
            )
            else None
        )

    monkeypatch.setattr(wizard_module, "active_managed_parakeet_dir", exact_active)
    wizard = _wizard()
    step = _step(wizard=wizard)
    step._acted_this_run = True
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.click(f"#setup-speech-language-{language}")
        await pilot.pause()
        step.query_one("#setup-speech-precision-f32", RadioButton).value = True
        await pilot.pause()
        ok, error = await step.commit()

    assert ok, error
    assert active_calls == [(policy.parakeet_v3_model_id, "f32")]
    assert wizard.commit_config.call_args.args[0] == {
        "transcription": {
            "default_provider": policy.parakeet_provider_id,
            "default_model": policy.parakeet_v3_model_id,
            "default_language": language,
            "default_precision": "f32",
        }
    }


@pytest.mark.asyncio
async def test_commit_reports_failure_when_persistence_write_fails(monkeypatch):
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    monkeypatch.setattr(
        wizard_module,
        "active_managed_parakeet_dir",
        _active_lookup(Path("/fake/active")),
    )
    wizard = _wizard(commit_config=AsyncMock(return_value=False))
    step = _step(wizard=wizard)
    step._acted_this_run = True

    ok, error = await step.commit()

    assert ok is False
    assert error


@pytest.mark.asyncio
async def test_commit_never_persists_when_the_runtime_extra_is_missing(monkeypatch):
    """Important 4, commit()-side belt-and-suspenders: even if somehow both
    active and acted this run, a missing onnx-asr runtime must still block
    the write (the UI-side gate is the primary defense; this is the second
    independent check, mirroring RagStep's own commit() re-check of
    deps_installed())."""
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    monkeypatch.setattr(
        wizard_module,
        "active_managed_parakeet_dir",
        _active_lookup(Path("/fake/active")),
    )
    wizard = _wizard()
    step = _step(wizard=wizard, runtime_installed=lambda: False)
    step._acted_this_run = True

    ok, error = await step.commit()

    assert ok, error
    wizard.commit_config.assert_not_awaited()


# ---------------------------------------------------------------------------
# AC#5 prefill (Important 3, UI half): the step shows what is already
# persisted before the user acts.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prefill_shown_when_a_different_provider_is_already_configured():
    wizard = _wizard(
        app_instance=MagicMock(
            app_config={
                "transcription": {
                    "default_provider": "remote-whisper",
                    "default_model": "whisper-1",
                    "default_language": "auto",
                }
            }
        )
    )
    step = _step(wizard=wizard)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        text = "\n".join(str(s.render()) for s in step.query(Static))
        assert "remote-whisper" in text


@pytest.mark.asyncio
async def test_no_prefill_line_when_nothing_is_persisted():
    step = _step()  # default app_config={}
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert not step.query("#setup-speech-prefill")


# ---------------------------------------------------------------------------
# Minor 9: progress messages must not keep bubbling past this step.
# ---------------------------------------------------------------------------


def test_install_progressed_stops_the_message():
    from textual.css.query import NoMatches

    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress

    step = _step()
    step.refresh = MagicMock()
    step.query_one = MagicMock(side_effect=NoMatches("not mounted"))
    progress = AcquisitionProgress(
        "fetch", parakeet_v2_reference(), "encoder.onnx", 1, 2
    )
    event = InstallProgressed(progress)
    event.stop = MagicMock()

    step._install_progressed(event)

    event.stop.assert_called_once_with()


# ---------------------------------------------------------------------------
# Minor 11: a forced reload requested while a load is already in flight must
# not be silently dropped.
# ---------------------------------------------------------------------------


def test_forced_reload_requested_while_loading_is_not_dropped():
    step = _step()
    step._loading = True
    step._loaded = True
    step._load_installed_state = MagicMock()
    step.refresh = MagicMock()

    step._ensure_loaded(force=True)

    # Still "loading" from the caller's point of view -- no second worker
    # dispatched yet -- but the request must be remembered...
    step._load_installed_state.assert_not_called()

    # ...and honored once the in-flight load actually completes.
    step._loading = False
    step._apply_installed_state(None, None)
    step._load_installed_state.assert_called_once()


# ---------------------------------------------------------------------------
# Review NEW-3: during a forced reload, the stale _installed_item must not
# render enabled controls -- InstalledView's own pending computation
# includes its own loading flag for the identical reason.
# ---------------------------------------------------------------------------


def test_lifecycle_pending_true_while_loading_even_with_no_operation():
    step = _step()
    assert step._operation is None
    step._loading = True
    assert step._lifecycle_pending is True


def test_lifecycle_pending_false_when_idle():
    step = _step()
    assert step._lifecycle_pending is False


@pytest.mark.asyncio
async def test_install_button_disabled_during_a_forced_reload():
    """A just-completed install/activate leaves _operation=None but
    _loading=True until the reload's own callback runs -- the stale
    "Not installed." + Install button state must not be clickable then."""
    step = _step(installed=())
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.2)
        step._loading = True  # simulate a forced reload in flight
        step.refresh(recompose=True)
        await pilot.pause(0.1)
        button = step.query_one("#setup-speech-install", Button)
        assert button.disabled is True


@pytest.mark.asyncio
async def test_activation_controls_pending_during_a_forced_reload():
    step = _step(installed=[_installed_item(active=False, ready=True)])
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.2)
        step._loading = True
        step.refresh(recompose=True)
        await pilot.pause(0.1)
        controls = step.query_one(ModelActivationControls)
        assert controls.pending is True


def test_deletion_requested_ignored_while_a_reload_is_in_flight(monkeypatch):
    fake_app = _patch_app(monkeypatch)
    step = _step()
    step._loading = True
    event = MagicMock(reference=parakeet_v2_reference())

    step._deletion_requested(event)

    fake_app.push_screen.assert_not_called()


def test_activation_requested_ignored_while_a_reload_is_in_flight():
    step = _step()
    step._activate_model = MagicMock()
    step._loading = True
    event = MagicMock(reference=parakeet_v2_reference())

    step._activation_requested(event)

    step._activate_model.assert_not_called()


def test_install_pressed_ignored_while_a_reload_is_in_flight():
    step = _step()
    step._preflight_install = MagicMock()
    step._loading = True

    step._install_pressed()

    step._preflight_install.assert_not_called()


# ---------------------------------------------------------------------------
# Minor 8: only the ONE recommended precision is pre-pressed (a second
# curated precision must never silently pre-press two radio buttons).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_precision_radioset_pre_presses_only_the_recommended_option():
    step = _step()
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        pressed = [
            b
            for b in step.query("#setup-speech-precision-choice RadioButton")
            if getattr(b, "value", False)
        ]
        assert len(pressed) == 1
        assert "INT8" in str(pressed[0].label).upper()


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
    from tldw_chatbook.UI.Wizards import first_run_speech_step_state as speech_state

    policy = speech_state.routing_policy()
    language = sorted(policy.validated_v3_languages)[0]
    report = _report(
        destination=tmp_path / "d",
        model=policy.parakeet_v3_model_id,
        precision="f32",
    )
    preflight_calls: list[tuple[str, str]] = []
    provision_calls: list[tuple[str, str, PreflightReport]] = []

    async def fake_preflight(model, precision):
        preflight_calls.append((model, precision))
        return report

    async def fake_provision(model, precision, passed_report, *, progress=None):
        provision_calls.append((model, precision, passed_report))
        return tmp_path / "installed"

    monkeypatch.setattr(wizard_module, "run_parakeet_preflight", fake_preflight)
    monkeypatch.setattr(wizard_module, "run_parakeet_provision", fake_provision)
    fake_app = _patch_app(monkeypatch)
    fake_app.call_from_thread.side_effect = lambda fn, *a, **kw: fn(*a, **kw)

    step = _step()
    step._selected_language = language
    step._selected_precision = "f32"
    step.notify = MagicMock()
    step._ensure_loaded = MagicMock()
    step.refresh = MagicMock()

    SpeechSetupStep._preflight_install.__wrapped__(step)

    assert preflight_calls == [(policy.parakeet_v3_model_id, "f32")]
    assert step._pending_report is report
    fake_app.push_screen.assert_called_once()

    SpeechSetupStep._provision_install.__wrapped__(step)

    assert provision_calls == [(policy.parakeet_v3_model_id, "f32", report)]
    step.notify.assert_called_once()
    assert step.notify.call_args.kwargs.get("severity") == "information"
    step._ensure_loaded.assert_called_once_with(force=True)


# ---------------------------------------------------------------------------
# TASK-598 Task 8: user-owned external Parakeet roots in First Run.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speech_step_shows_disk_and_managed_actions_for_exact_selection():
    step = _step(installed=(), runtime_installed=lambda: True)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        await pilot.pause(0.2)

        assert step.query_one("#setup-speech-use-from-disk", Button)
        assert step.query_one("#setup-speech-install", Button)


@pytest.mark.asyncio
async def test_runtime_missing_keeps_disk_selection_available():
    step = _step(installed=(), runtime_installed=lambda: False)
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        assert step.query_one("#setup-speech-use-from-disk", Button)
        assert step.query_one("#setup-speech-install", Button)


@pytest.mark.asyncio
async def test_busy_external_setup_has_keyboard_cancel_with_path_private_copy():
    step = _step()
    worker = MagicMock(is_finished=False)
    step._external_selection_worker = worker
    step._external_selection_token = (1, id(step))
    step._external_busy = True
    step._external_status = "Verifying model files…"
    app = _StepHost(step)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        cancel = step.query_one("#setup-speech-cancel-external", Button)
        assert cancel.region.width > 0 and cancel.region.height > 0
        assert cancel.region.right <= 80 and cancel.region.bottom <= 24
        cancel.focus()
        await pilot.press("enter")
        await pilot.pause()

        assert step._external_busy is False
        assert step._external_status == (
            "External setup cancelled. The prior source is unchanged."
        )
        worker.cancel.assert_called_once_with()


@pytest.mark.asyncio
async def test_commit_pending_physically_blocks_every_source_control(monkeypatch):
    prepared = SimpleNamespace(key=ParakeetSourceKey.V2_INT8)
    source_commit = SimpleNamespace(
        section_values={"transcription": {"parakeet_external_sources": {}}}
    )
    source_service = MagicMock()
    source_service.prepare_config_commit.return_value = source_commit
    owners = {"setup-speech-pending-controls"}
    source_service.release_scope.side_effect = owners.discard
    write_started = asyncio.Event()
    allow_write = asyncio.Event()

    async def delayed_commit(values, *, after_write=None):
        write_started.set()
        await allow_write.wait()
        if after_write is not None:
            after_write()
        return True

    wizard = _wizard_with_source(source_service, commit_config=delayed_commit)
    step = _step(wizard=wizard)
    app = _StepHost(step)
    preflight = MagicMock()
    activate = MagicMock()
    step._preflight_install = preflight
    step._activate_model = activate
    picker = MagicMock()
    monkeypatch.setattr(app, "push_screen", picker)

    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause(0.2)
        step._loading = False
        step._loaded = True
        step._installed_item = None
        token = step._next_external_token()
        scope_id = step._external_scope_ids[token]
        source_service.owners = owners
        owners.add(scope_id)
        worker = MagicMock(is_finished=True)
        step._external_selection_worker = worker
        step._pending_external_selection = prepared
        scope_snapshot = dict(step._external_scope_ids)

        task = asyncio.create_task(step.commit())
        await asyncio.wait_for(write_started.wait(), timeout=2)
        try:
            await pilot.pause()
            assert step._external_commit_pending is True
            disk = step.query_one("#setup-speech-use-from-disk", Button)
            install = step.query_one("#setup-speech-install", Button)
            language = step.query_one("#setup-speech-language-fr", RadioButton)
            precision = step.query_one("#setup-speech-precision-int8", RadioButton)
            assert all(
                control.disabled for control in (disk, install, language, precision)
            )

            for control in (disk, install, language, precision):
                control.disabled = False
                control.focus()
                await pilot.press(
                    "space" if isinstance(control, RadioButton) else "enter"
                )
                await pilot.pause()

            step._installed_item = _installed_item(active=False)
            step.refresh(recompose=True)
            await pilot.pause()
            activation_controls = step.query_one(ModelActivationControls)
            activation = activation_controls.query_one(".model-activate", Button)
            assert activation.disabled
            activation_controls.pending = False
            activation.disabled = False
            activation.focus()
            await pilot.press("enter")
            await pilot.pause()

            assert picker.call_count == 0
            preflight.assert_not_called()
            activate.assert_not_called()
            source_service.prefer_managed.assert_not_called()
            assert step._external_selection_token == token
            assert step._external_scope_ids == scope_snapshot
            assert step._external_selection_worker is worker
            assert scope_id in owners
            source_service.accept_committed.assert_not_called()
        finally:
            allow_write.set()
            result = await task

        ok, error = result
        await pilot.pause()
        assert ok, error
        assert step._external_commit_pending is False
        assert not step.query_one("#setup-speech-use-from-disk", Button).disabled
        assert not step.query_one("#setup-speech-language-en", RadioButton).disabled
        restored = step.query_one(ModelActivationControls)
        assert not restored.query_one(".model-activate", Button).disabled


@pytest.mark.asyncio
async def test_slow_external_prepare_fences_every_source_control(monkeypatch):
    prepared = SimpleNamespace(key=ParakeetSourceKey.V2_INT8)
    source_commit = SimpleNamespace(
        section_values={"transcription": {"parakeet_external_sources": {}}}
    )
    prepare_started = threading.Event()
    allow_prepare = threading.Event()
    source_service = MagicMock()

    def delayed_prepare(selection):
        prepare_started.set()
        allow_prepare.wait(timeout=2)
        return source_commit

    source_service.prepare_config_commit.side_effect = delayed_prepare
    owners = set()
    source_service.release_scope.side_effect = owners.discard
    step = _step(wizard=_wizard_with_source(source_service))
    app = _StepHost(step)
    preflight = MagicMock()
    activate = MagicMock()
    step._preflight_install = preflight
    step._activate_model = activate
    picker = MagicMock()
    monkeypatch.setattr(app, "push_screen", picker)

    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause(0.2)
        step._loading = False
        step._loaded = True
        step._installed_item = None
        token = step._next_external_token()
        scope_id = step._external_scope_ids[token]
        owners.add(scope_id)
        worker = MagicMock(is_finished=True)
        step._external_selection_worker = worker
        step._pending_external_selection = prepared
        scope_snapshot = dict(step._external_scope_ids)

        task = asyncio.create_task(step.commit())
        for _ in range(100):
            if prepare_started.is_set():
                break
            await pilot.pause(0.01)
        assert prepare_started.is_set()
        try:
            await pilot.pause()
            assert step._external_commit_pending is True
            disk = step.query_one("#setup-speech-use-from-disk", Button)
            install = step.query_one("#setup-speech-install", Button)
            language = step.query_one("#setup-speech-language-fr", RadioButton)
            precision = step.query_one("#setup-speech-precision-f32", RadioButton)
            controls = (disk, install, language, precision)
            assert all(control.disabled for control in controls)

            for control in controls:
                control.disabled = False
                control.focus()
                await pilot.press(
                    "space" if isinstance(control, RadioButton) else "enter"
                )
                await pilot.pause()

            step._installed_item = _installed_item(active=False)
            step.refresh(recompose=True)
            await pilot.pause()
            activation_controls = step.query_one(ModelActivationControls)
            activation = activation_controls.query_one(".model-activate", Button)
            assert activation.disabled
            activation_controls.pending = False
            activation.disabled = False
            activation.focus()
            await pilot.press("enter")
            await pilot.pause()

            assert picker.call_count == 0
            preflight.assert_not_called()
            activate.assert_not_called()
            source_service.prefer_managed.assert_not_called()
            source_service.accept_committed.assert_not_called()
            assert step._external_selection_token == token
            assert step._external_scope_ids == scope_snapshot
            assert step._external_selection_worker is worker
            assert scope_id in owners
        finally:
            allow_prepare.set()
            result = await task

        ok, error = result
        assert ok, error
        assert step._external_commit_pending is False


@pytest.mark.asyncio
async def test_vad_provision_cancellation_reaches_underlying_coroutine(
    monkeypatch,
    tmp_path,
):
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    started = threading.Event()
    cancelled = threading.Event()
    release = threading.Event()
    terminal_results = []

    async def gated_provision(report, *, progress=None):
        started.set()
        try:
            while not release.is_set():
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(wizard_module, "run_parakeet_vad_provision", gated_provision)
    prepared = SimpleNamespace(key=ParakeetSourceKey.V2_INT8)
    step = _step(wizard=_wizard_with_source(_OwnerTrackingSource()))
    step._apply_external_vad_provision_result = lambda *args: terminal_results.append(
        args
    )
    app = _StepHost(step)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        token = step._next_external_token()
        worker = step._provision_external_vad(
            token,
            prepared,
            _vad_report(destination=tmp_path / "vad"),
        )
        step._external_selection_worker = worker
        for _ in range(100):
            if started.is_set():
                break
            await pilot.pause(0.01)
        assert started.is_set()
        step._discard_external_selection()
        try:
            for _ in range(20):
                if cancelled.is_set():
                    break
                await pilot.pause(0.01)
            assert cancelled.is_set()
        finally:
            release.set()
            await pilot.pause(0.05)

    assert terminal_results == []
    assert step._pending_external_selection is None


def test_disk_action_opens_real_directory_picker_and_cancel_changes_nothing(
    monkeypatch,
):
    fake_app = _patch_app(monkeypatch)
    source_service = MagicMock()
    step = _step(wizard=_wizard_with_source(source_service))
    step._verify_external_source = MagicMock()

    step._use_external_pressed()

    picker, callback = fake_app.push_screen.call_args.args
    assert isinstance(picker, SelectDirectory)
    callback(None)
    step._verify_external_source.assert_not_called()
    source_service.prepare_external.assert_not_called()
    source_service.prepare_config_commit.assert_not_called()
    source_service.accept_committed.assert_not_called()


def test_stale_picker_cancel_does_not_discard_new_external_selection():
    step = _step()
    stale_token = (1, id(step))
    current_token = (2, id(step))
    pending = SimpleNamespace(key=ParakeetSourceKey.V2_INT8)
    step._external_selection_token = current_token
    step._pending_external_selection = pending
    step._owns_external_token = lambda token: token == current_token

    step._external_directory_selected(
        stale_token,
        ParakeetSourceKey.V2_INT8,
        None,
    )

    assert step._external_selection_token == current_token
    assert step._pending_external_selection is pending


@pytest.mark.asyncio
async def test_external_verification_worker_hashes_off_the_event_loop(tmp_path):
    prepared = SimpleNamespace(key=ParakeetSourceKey.V2_INT8)
    source_service = MagicMock()
    worker_threads: list[int] = []

    def prepare_external(*args, **kwargs):
        worker_threads.append(threading.get_ident())
        assert kwargs["cancelled"]() is False
        return prepared

    source_service.prepare_external.side_effect = prepare_external
    step = _step(wizard=_wizard_with_source(source_service))
    step._apply_external_verification_result = MagicMock()
    app = _StepHost(step)
    event_loop_thread = threading.get_ident()
    original_owns = step._owns_external_token

    def loop_only_ownership(token):
        assert threading.get_ident() == event_loop_thread
        return original_owns(token)

    step._owns_external_token = loop_only_ownership

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        token = step._next_external_token()
        worker = step._verify_external_source(
            token,
            ParakeetSourceKey.V2_INT8,
            tmp_path,
            step._external_scope_ids[token],
        )
        await worker.wait()

    assert worker_threads and worker_threads[0] != event_loop_thread
    source_service.prepare_external.assert_called_once()


def test_verifier_uses_scope_captured_on_loop_after_scope_map_changes(
    monkeypatch,
    tmp_path,
):
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    _patch_app(monkeypatch)
    prepared = SimpleNamespace(key=ParakeetSourceKey.V2_INT8)
    owners = []
    source_service = MagicMock()
    source_service.prepare_external.side_effect = lambda *args, **kwargs: (
        owners.append(kwargs["owner"]) or prepared
    )
    step = _step(wizard=_wizard_with_source(source_service))
    step._apply_external_verification_result = MagicMock()
    token = (1, id(step))
    captured_scope = "setup-speech-captured"
    step._external_scope_ids[token] = "setup-speech-later-map-value"
    monkeypatch.setattr(
        wizard_module,
        "get_current_worker",
        lambda: SimpleNamespace(is_cancelled=False),
    )

    SpeechSetupStep._verify_external_source.__wrapped__(
        step,
        token,
        ParakeetSourceKey.V2_INT8,
        tmp_path,
        captured_scope,
    )

    assert owners == [("scope", captured_scope)]


def test_external_hash_progress_is_path_private(monkeypatch, tmp_path):
    _patch_app(monkeypatch)
    step = _step()
    step.refresh = MagicMock()
    token = (1, id(step))
    step._external_selection_token = token
    step._owns_external_token = lambda candidate: candidate == token

    step._apply_external_hash_progress(token, 12, 100)

    assert "12" in step._external_status
    assert "100" in step._external_status
    assert str(tmp_path) not in step._external_status


def test_changed_external_source_uses_shared_path_private_error(
    monkeypatch,
    tmp_path,
):
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    fake_app = _patch_app(monkeypatch)
    fake_app.call_from_thread.side_effect = lambda fn, *args: fn(*args)
    monkeypatch.setattr(
        wizard_module,
        "get_current_worker",
        lambda: SimpleNamespace(is_cancelled=False),
        raising=False,
    )
    source_service = MagicMock()
    source_service.prepare_external.side_effect = ExternalParakeetVerificationError(
        ExternalParakeetErrorCode.CHANGED
    )
    step = _step(wizard=_wizard_with_source(source_service))
    step.notify = MagicMock()
    step.refresh = MagicMock()
    token = (1, id(step))
    step._external_selection_token = token
    step._external_scope_ids[token] = "setup-speech-scope"
    step._owns_external_token = lambda candidate: candidate == token

    SpeechSetupStep._verify_external_source.__wrapped__(
        step,
        token,
        ParakeetSourceKey.V2_INT8,
        tmp_path / "private-model-dir",
        step._external_scope_ids[token],
    )

    assert step._external_status == (
        "Model files changed during verification. "
        "Wait for file changes to finish, then retry."
    )
    assert str(tmp_path) not in step._external_status
    assert str(tmp_path) not in str(step.notify.call_args)
    source_service.prepare_config_commit.assert_not_called()


@pytest.mark.asyncio
async def test_external_worker_never_logs_or_describes_exception_path(tmp_path):
    sentinel = str(tmp_path / "private-owner-model")
    source_service = MagicMock()
    source_service.prepare_external.side_effect = RuntimeError(sentinel)
    step = _step(wizard=_wizard_with_source(source_service))
    app = _StepHost(step)
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)))

    try:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            token = step._next_external_token()
            worker = step._verify_external_source(
                token,
                ParakeetSourceKey.V2_INT8,
                Path(sentinel),
                step._external_scope_ids[token],
            )
            await worker.wait()
            await pilot.pause()

            assert sentinel not in worker.description
            assert sentinel not in step._external_status
    finally:
        logger.remove(sink)

    assert sentinel not in "".join(messages)


@pytest.mark.asyncio
async def test_stale_verification_after_exact_selection_change_is_ignored():
    source_service = MagicMock()
    step = _step(wizard=_wizard_with_source(source_service))
    step._ensure_loaded = MagicMock()
    language = sorted(
        speech_language.code
        for speech_language in __import__(
            "tldw_chatbook.UI.Wizards.first_run_speech_step_state",
            fromlist=["speech_language_options"],
        ).speech_language_options(curated_model_ids=step._curated_model_ids())
        if speech_language.code != "en"
    )[0]
    app = _StepHost(step)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        old_token = step._next_external_token()
        step._external_status = "External model verified. Continue to save."
        step._set_exact_selection(language, "f32")
        assert step._owns_external_token(old_token) is False
        assert step._external_status == ""
        step._apply_external_verification_result(
            old_token,
            SimpleNamespace(key=ParakeetSourceKey.V2_INT8),
            None,
        )

        assert step._pending_external_selection is None
        source_service.prepare_config_commit.assert_not_called()


def test_vad_only_consent_plan_never_contains_parakeet_root(monkeypatch, tmp_path):
    fake_app = _patch_app(monkeypatch)
    report = _vad_report(destination=tmp_path / "managed")
    prepared = SimpleNamespace(key=ParakeetSourceKey.V2_INT8)
    step = _step(wizard=_wizard_with_source(MagicMock()))
    token = (1, id(step))
    step._external_selection_token = token
    step._external_scope_ids[token] = "setup-speech-scope"
    step._owns_external_token = lambda candidate: candidate == token

    step._apply_external_vad_preflight_result(token, prepared, report, None)

    modal, callback = fake_app.push_screen.call_args.args
    assert isinstance(modal, ModelInstallModal)
    assert modal.report.root == parakeet_vad_reference()
    assert {entry.ref for entry in modal.report.entries} == {parakeet_vad_reference()}
    root_urls = {
        parakeet_descriptor(key.model_id, key.precision).source_url
        for key in ParakeetSourceKey
    }
    assert not root_urls.intersection(
        entry.source_url for entry in modal.report.entries
    )
    assert callable(callback)


def test_vad_cancel_leaves_prior_source_and_pending_selection_untouched(
    monkeypatch,
    tmp_path,
):
    _patch_app(monkeypatch)
    prior_records = {ParakeetSourceKey.V2_INT8: object()}
    source_service = _OwnerTrackingSource(prior_records)
    prepared = SimpleNamespace(key=ParakeetSourceKey.V2_INT8)
    step = _step(wizard=_wizard_with_source(source_service))
    step.refresh = MagicMock()
    report = _vad_report(destination=tmp_path / "managed")
    token = (1, id(step))
    step._external_selection_token = token
    scope_id = "setup-speech-scope"
    step._external_scope_ids[token] = scope_id
    source_service.owners.add(scope_id)
    step._owns_external_token = lambda candidate: candidate == token

    step._confirm_external_vad(False, token, prepared, report)

    assert step._pending_external_selection is None
    assert source_service.records() == prior_records
    assert source_service.accepted == []
    assert source_service.owners == set()
    assert "prior source is unchanged" in step._external_status.lower()


def test_vad_failure_leaves_prior_source_untouched(monkeypatch):
    _patch_app(monkeypatch)
    prior_records = {ParakeetSourceKey.V2_INT8: object()}
    source_service = _OwnerTrackingSource(prior_records)
    prepared = SimpleNamespace(key=ParakeetSourceKey.V2_INT8)
    step = _step(wizard=_wizard_with_source(source_service))
    step.notify = MagicMock()
    step.refresh = MagicMock()
    token = (1, id(step))
    step._external_selection_token = token
    scope_id = "setup-speech-scope"
    step._external_scope_ids[token] = scope_id
    source_service.owners.add(scope_id)
    step._owns_external_token = lambda candidate: candidate == token

    step._apply_external_vad_provision_result(
        token,
        prepared,
        "The managed VAD dependency could not be installed.",
    )

    assert step._pending_external_selection is None
    assert source_service.records() == prior_records
    assert source_service.accepted == []
    assert source_service.owners == set()
    assert step.notify.call_args.kwargs["severity"] == "error"


@pytest.mark.asyncio
async def test_external_next_commits_defaults_and_source_once_then_accepts():
    events: list[str] = []
    prepared = SimpleNamespace(key=ParakeetSourceKey.V2_INT8)
    source_commit = SimpleNamespace(
        section_values={
            "transcription": {
                "parakeet_external_sources": {
                    "v2_int8": {
                        "model_id": ParakeetSourceKey.V2_INT8.model_id,
                        "precision": "int8",
                        "directory": "/user-owned/parakeet-v2",
                        "preferred_source": "external",
                    }
                }
            }
        }
    )
    source_service = MagicMock()
    source_service.prepare_config_commit.side_effect = lambda value: (
        events.append("prepare") or source_commit
    )
    source_service.accept_committed.side_effect = lambda value: events.append("accept")

    async def commit_config(values, *, after_write=None):
        events.append("write")
        if after_write is not None:
            after_write()
        return True

    wizard = _wizard_with_source(source_service)
    wizard.commit_config = AsyncMock(side_effect=commit_config)
    step = _step(wizard=wizard)
    step._pending_external_selection = prepared

    ok, error = await step.commit()

    assert ok, error
    assert events == ["prepare", "write", "accept"]
    wizard.commit_config.assert_awaited_once()
    assert callable(wizard.commit_config.call_args.kwargs["after_write"])
    assert wizard.commit_config.call_args.args == (
        {
            "transcription": {
                "default_provider": "parakeet-onnx",
                "default_model": ParakeetSourceKey.V2_INT8.model_id,
                "default_language": "en",
                "default_precision": "int8",
                "parakeet_external_sources": {
                    "v2_int8": {
                        "model_id": ParakeetSourceKey.V2_INT8.model_id,
                        "precision": "int8",
                        "directory": "/user-owned/parakeet-v2",
                        "preferred_source": "external",
                    }
                },
            }
        },
    )


@pytest.mark.asyncio
async def test_external_write_failure_never_accepts_or_drops_pending_selection():
    prepared = SimpleNamespace(key=ParakeetSourceKey.V2_INT8)
    source_commit = SimpleNamespace(
        section_values={"transcription": {"parakeet_external_sources": {}}}
    )
    source_service = _OwnerTrackingSource()
    source_service.prepared_commit = source_commit
    persisted = {"transcription": {"default_provider": "remote-whisper"}}
    app_instance = SimpleNamespace(
        app_config=persisted,
        _parakeet_source_service=source_service,
        _ensure_parakeet_source_service=lambda: source_service,
    )
    wizard = _wizard(
        app_instance=app_instance, commit_config=AsyncMock(return_value=False)
    )
    step = _step(wizard=wizard)
    step._pending_external_selection = prepared
    token = (1, id(step))
    scope_id = "setup-speech-write"
    step._external_selection_token = token
    step._external_scope_ids[token] = scope_id
    source_service.owners.add(scope_id)

    ok, error = await step.commit()

    assert ok is False
    assert error
    assert persisted == {"transcription": {"default_provider": "remote-whisper"}}
    assert source_service.accepted == []
    assert source_service.owners == {scope_id}
    assert step._pending_external_selection is prepared


@pytest.mark.asyncio
async def test_cancelled_external_commit_settles_write_and_accept_before_scope_release():
    prepared = SimpleNamespace(key=ParakeetSourceKey.V2_INT8)
    source_commit = SimpleNamespace(
        section_values={"transcription": {"parakeet_external_sources": {}}}
    )
    source_service = _OwnerTrackingSource()
    source_service.prepared_commit = source_commit
    persisted = {"transcription": {"default_provider": "remote-whisper"}}
    write_started = threading.Event()
    allow_write = threading.Event()
    write_finished = threading.Event()

    async def delayed_commit(values, *, after_write=None):
        def write():
            write_started.set()
            allow_write.wait(timeout=2)
            persisted.clear()
            persisted.update(values)
            if after_write is not None:
                after_write()
            write_finished.set()
            return True

        return await asyncio.get_running_loop().run_in_executor(None, write)

    app_instance = SimpleNamespace(
        app_config=persisted,
        _parakeet_source_service=source_service,
        _ensure_parakeet_source_service=lambda: source_service,
    )
    wizard = _wizard(app_instance=app_instance, commit_config=delayed_commit)
    step = _step(wizard=wizard)
    step._pending_external_selection = prepared
    token = (1, id(step))
    scope_id = "setup-speech-handoff"
    step._external_selection_token = token
    step._external_scope_ids[token] = scope_id
    source_service.owners.add(scope_id)

    task = asyncio.create_task(step.commit())
    for _ in range(100):
        if write_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert write_started.is_set()
    task.cancel()
    step.on_unmount()
    owner_retained_while_writing = scope_id in source_service.owners
    allow_write.set()
    try:
        await task
    except asyncio.CancelledError:
        pass
    for _ in range(100):
        if write_finished.is_set():
            break
        await asyncio.sleep(0.01)

    assert write_finished.is_set()
    assert owner_retained_while_writing
    assert source_service.accepted == [source_commit]
    assert source_service.owners == set()
    assert persisted != {"transcription": {"default_provider": "remote-whisper"}}


@pytest.mark.parametrize("retry_succeeds", [True, False])
@pytest.mark.asyncio
async def test_cancelled_durable_write_reconciles_or_retains_scope_after_accept_error(
    monkeypatch,
    tmp_path,
    retry_succeeds,
):
    import tldw_chatbook.config as config_module

    prepared = SimpleNamespace(key=ParakeetSourceKey.V2_INT8)
    source_commit = SimpleNamespace(
        section_values={
            "transcription": {
                "parakeet_external_sources": {
                    "v2_int8": {
                        "model_id": ParakeetSourceKey.V2_INT8.model_id,
                        "precision": "int8",
                        "directory": "/user-owned/parakeet-v2",
                        "preferred_source": "external",
                    }
                }
            }
        }
    )
    source_service = _OwnerTrackingSource()
    source_service.prepared_commit = source_commit
    accept_started = threading.Event()
    allow_first_accept = threading.Event()
    accept_attempts = 0
    reconciled = False
    sentinel = str(tmp_path / "private-source-root")

    def accept_committed(commit):
        nonlocal accept_attempts, reconciled
        accept_attempts += 1
        if accept_attempts == 1:
            accept_started.set()
            allow_first_accept.wait(timeout=2)
            raise RuntimeError(sentinel)
        if not retry_succeeds:
            raise RuntimeError(sentinel)
        reconciled = True

    source_service.accept_committed = accept_committed
    durable = {}

    def save(values):
        durable.update(values)
        return True

    monkeypatch.setattr(config_module, "save_settings_to_cli_config", save)
    wizard = _wizard_with_real_commit(source_service)
    prior_config = {"transcription": {"default_provider": "remote-whisper"}}
    wizard.app_instance.app_config = prior_config
    step = _step(wizard=wizard)
    step._pending_external_selection = prepared
    token = (1, id(step))
    scope_id = "setup-speech-durable-recovery"
    step._external_selection_token = token
    step._external_scope_ids[token] = scope_id
    source_service.owners.add(scope_id)

    task = asyncio.create_task(step.commit())
    for _ in range(100):
        if accept_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert accept_started.is_set()
    task.cancel()
    step.on_unmount()
    owner_retained_during_accept = scope_id in source_service.owners
    allow_first_accept.set()

    if retry_succeeds:
        with pytest.raises(asyncio.CancelledError):
            await task
        assert reconciled is True
        assert source_service.owners == set()
    else:
        ok, error = await task
        assert ok is False
        recovery = error.lower()
        assert "saved" in recovery and "restart" in recovery and "retry" in recovery
        assert sentinel not in error
        assert step._external_status == error
        assert source_service.owners == {scope_id}

    assert owner_retained_during_accept
    assert accept_attempts == 2
    assert durable["transcription"]["default_provider"] == "parakeet-onnx"
    assert prior_config["transcription"] == durable["transcription"]


@pytest.mark.asyncio
async def test_external_selection_persists_when_runtime_is_missing():
    prepared = SimpleNamespace(key=ParakeetSourceKey.V2_INT8)
    source_commit = SimpleNamespace(
        section_values={
            "transcription": {
                "parakeet_external_sources": {
                    "v2_int8": {
                        "model_id": ParakeetSourceKey.V2_INT8.model_id,
                        "precision": "int8",
                        "directory": "/user-owned/parakeet-v2",
                        "preferred_source": "external",
                    }
                }
            }
        }
    )
    source_service = MagicMock()
    source_service.prepare_config_commit.return_value = source_commit
    wizard = _wizard_with_source(source_service)

    async def commit_config(values, *, after_write=None):
        if after_write is not None:
            after_write()
        return True

    wizard.commit_config.side_effect = commit_config
    step = _step(wizard=wizard, runtime_installed=lambda: False)
    step._pending_external_selection = prepared

    ok, error = await step.commit()

    assert ok, error
    wizard.commit_config.assert_awaited_once()
    source_service.accept_committed.assert_called_once_with(source_commit)
    assert step._external_status == "Runtime required"


def test_managed_install_prefers_managed_only_after_provision_succeeds(
    monkeypatch,
    tmp_path,
):
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    events: list[str] = []
    report = _report(destination=tmp_path / "managed")
    source_service = MagicMock()
    source_service.prefer_managed.side_effect = lambda key: events.append("prefer")

    async def provision(*args, **kwargs):
        events.append("provision")
        return tmp_path / "managed" / "root"

    monkeypatch.setattr(wizard_module, "run_parakeet_provision", provision)
    fake_app = _patch_app(monkeypatch)
    fake_app.call_from_thread.side_effect = lambda fn, *args: fn(*args)
    step = _step(wizard=_wizard_with_source(source_service))
    step._pending_report = report
    step.notify = MagicMock()
    step._ensure_loaded = MagicMock()

    SpeechSetupStep._provision_install.__wrapped__(step)

    assert events == ["provision", "prefer"]
    source_service.prefer_managed.assert_called_once_with(ParakeetSourceKey.V2_INT8)


def test_managed_activation_prefers_managed_only_after_activation_succeeds(monkeypatch):
    events: list[str] = []
    managed_service = _FakeService()
    managed_service.activate = lambda reference: events.append("activate")
    source_service = MagicMock()
    source_service.prefer_managed.side_effect = lambda key: events.append("prefer")
    fake_app = _patch_app(monkeypatch)
    fake_app.call_from_thread.side_effect = lambda fn, *args: fn(*args)
    step = SpeechSetupStep(
        wizard=_wizard_with_source(source_service),
        config=WizardStepConfig(id="speech", title="Speech", step_number=5),
        service_factory=lambda: managed_service,
        runtime_installed=lambda: True,
    )
    step.notify = MagicMock()
    step._ensure_loaded = MagicMock()

    SpeechSetupStep._activate_model.__wrapped__(step)

    assert events == ["activate", "prefer"]


def test_managed_provision_failure_never_prefers_managed(monkeypatch, tmp_path):
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    async def fail_provision(*args, **kwargs):
        raise OSError("download failed")

    monkeypatch.setattr(wizard_module, "run_parakeet_provision", fail_provision)
    fake_app = _patch_app(monkeypatch)
    fake_app.call_from_thread.side_effect = lambda fn, *args: fn(*args)
    source_service = MagicMock()
    step = _step(wizard=_wizard_with_source(source_service))
    step._pending_report = _report(destination=tmp_path / "managed")
    step.notify = MagicMock()
    step._ensure_loaded = MagicMock()

    SpeechSetupStep._provision_install.__wrapped__(step)

    source_service.prefer_managed.assert_not_called()


def test_managed_activation_failure_never_prefers_managed(monkeypatch):
    fake_app = _patch_app(monkeypatch)
    fake_app.call_from_thread.side_effect = lambda fn, *args: fn(*args)
    managed_service = _FakeService()

    def fail_activation(reference):
        raise OSError("activation failed")

    managed_service.activate = fail_activation
    source_service = MagicMock()
    step = SpeechSetupStep(
        wizard=_wizard_with_source(source_service),
        config=WizardStepConfig(id="speech", title="Speech", step_number=5),
        service_factory=lambda: managed_service,
        runtime_installed=lambda: True,
    )
    step.notify = MagicMock()
    step._ensure_loaded = MagicMock()

    SpeechSetupStep._activate_model.__wrapped__(step)

    source_service.prefer_managed.assert_not_called()


def test_managed_install_preference_failure_reports_partial_success(
    monkeypatch,
    tmp_path,
):
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    async def provision(*args, **kwargs):
        return tmp_path / "managed" / "root"

    monkeypatch.setattr(wizard_module, "run_parakeet_provision", provision)
    fake_app = _patch_app(monkeypatch)
    fake_app.call_from_thread.side_effect = lambda fn, *args: fn(*args)
    source_service = MagicMock()
    source_service.prefer_managed.side_effect = OSError("preference write failed")
    step = _step(wizard=_wizard_with_source(source_service))
    step._pending_report = _report(destination=tmp_path / "managed")
    step.notify = MagicMock()
    step._ensure_loaded = MagicMock()

    SpeechSetupStep._provision_install.__wrapped__(step)

    assert step._acted_this_run is True
    message = str(step.notify.call_args.args[0]).lower()
    assert "installed" in message and "preference" in message
    assert "installation failed" not in message


def test_managed_activation_preference_failure_reports_partial_success(monkeypatch):
    fake_app = _patch_app(monkeypatch)
    fake_app.call_from_thread.side_effect = lambda fn, *args: fn(*args)
    managed_service = _FakeService()
    source_service = MagicMock()
    source_service.prefer_managed.side_effect = OSError("preference write failed")
    step = SpeechSetupStep(
        wizard=_wizard_with_source(source_service),
        config=WizardStepConfig(id="speech", title="Speech", step_number=5),
        service_factory=lambda: managed_service,
        runtime_installed=lambda: True,
    )
    step._operation = "activate"
    step.notify = MagicMock()
    step._ensure_loaded = MagicMock()

    SpeechSetupStep._activate_model.__wrapped__(step)

    assert step._acted_this_run is True
    message = str(step.notify.call_args.args[0]).lower()
    assert "activated" in message and "preference" in message
    assert "activation failed" not in message
