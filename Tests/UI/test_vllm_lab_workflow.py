from __future__ import annotations

import json
import logging
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, Collapsible, Input, Label, Select, Static, TextArea

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.config import get_cli_setting as _real_get_cli_setting
from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
    ServerLaunchClaim,
    attach_server_claim_resource,
    clear_server_process,
    current_server_claim,
    publish_server_process,
    release_server_claim,
    reserve_server_launch,
    server_lifecycle_snapshot,
)
from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
    stop_server_process as real_stop_server_process,
)
from tldw_chatbook.UI.LLM_Management.vllm_connection import (
    VllmActivityEvent,
    VllmConnectionOwner,
    VllmProbeResult,
)
from tldw_chatbook.UI.LLM_Management.vllm_profiles import (
    VllmLaunchProfileV1,
    VllmProfileCorrupt,
    VllmProfileDocumentV1,
    VllmProfileRepository,
    VllmProfileValidationError,
    default_vllm_profile,
    draft_from_profile,
    profile_from_draft,
)
from tldw_chatbook.UI.LLM_Management.vllm_setup import (
    VllmConnectionTarget,
    VllmIssue,
    VllmLaunchDraft,
    VllmMode,
    VllmModelSource,
    VllmPreflightResult,
    VllmReadinessState,
    launch_snapshot_from_draft,
    semantic_fingerprint,
)
from tldw_chatbook.UI.LLM_Management.vllm_setup_view import VllmSetupView
from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

pytestmark = pytest.mark.asyncio

#: TASK-31809: iteration budget for the loops that poll for the lazily-mounted
#: vLLM pane / VllmSetupView in the navigation handoff test. Bare `pilot.pause()`
#: per iteration, so this is a generous ceiling for a machine under concurrent
#: test load, not a performance target -- the loops break the instant their
#: condition holds.
_LAZY_MOUNT_POLL_ITERATIONS = 120


@pytest.fixture(autouse=True)
def _no_splash(monkeypatch):
    def fake_get_cli_setting(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return _real_get_cli_setting(section, key, default)

    monkeypatch.setattr("tldw_chatbook.app.get_cli_setting", fake_get_cli_setting)


class _VllmHost(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.profile_events = []
        self.action_events = []

    def compose(self) -> ComposeResult:
        yield VllmSetupView(id="vllm-setup")

    @on(VllmSetupView.CreateProfileRequested)
    @on(VllmSetupView.SaveProfileRequested)
    @on(VllmSetupView.RenameProfileRequested)
    @on(VllmSetupView.DuplicateProfileRequested)
    @on(VllmSetupView.DeleteProfileRequested)
    def _capture_profile_event(self, event) -> None:
        self.profile_events.append(event)

    @on(VllmSetupView.CheckRequested)
    @on(VllmSetupView.StartRequested)
    @on(VllmSetupView.RestartRequested)
    def _capture_action_event(self, event) -> None:
        self.action_events.append(event)


class _RunningProcess:
    pid = 12345

    def __init__(self) -> None:
        self.running = True

    def poll(self):
        return None if self.running else 0

    def terminate(self) -> None:
        self.running = False

    def wait(self, timeout=None) -> int:
        self.running = False
        return 0

    def kill(self) -> None:
        self.running = False


async def test_initial_vllm_setup_is_guided_and_blocks_start():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)):
        view = app.query_one(VllmSetupView)
        copy = " ".join(str(label.renderable) for label in view.query(Label))
        buttons = " ".join(str(button.label) for button in view.query(Button))
        assert "Set up vLLM" in copy
        assert "Start on this computer" in buttons
        assert "Connect to existing server" in buttons
        assert app.query_one("#vllm-port", Input).value == "8000"
        assert app.query_one("#vllm-start", Button).disabled
        assert "GGUF" not in copy
        assert "checkpoint" not in copy.lower()


async def test_guided_readiness_keeps_four_recoverable_rows_and_python_browse():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)):
        view = app.query_one(VllmSetupView)
        expected_rows = {
            "vllm-check-environment": "Environment",
            "vllm-check-installation": "vLLM installation",
            "vllm-check-model": "Model",
            "vllm-check-network": "Network",
        }
        for row_id, label in expected_rows.items():
            row = view.query_one(f"#{row_id}", Label)
            assert row.display
            assert label in str(row.renderable)
        assert view.query_one("#vllm-browse-python-environment", Button).display

        draft = replace(
            view.draft,
            model_value="org/model",
        )
        view.apply_state(
            draft=draft,
            state=VllmReadinessState.READY_TO_START,
            profiles_ready=True,
            preflight=VllmPreflightResult(
                generation=1,
                fingerprint=semantic_fingerprint(draft),
                issues=(),
                python_version="Python 3.12.8",
                vllm_version="vLLM 0.9.1",
            ),
        )
        assert "Python 3.12.8" in str(
            view.query_one("#vllm-check-environment", Label).renderable
        )
        assert "vLLM 0.9.1" in str(
            view.query_one("#vllm-check-installation", Label).renderable
        )


async def test_advanced_structured_profile_values_are_visible_editable_and_adjacent():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)):
        view = app.query_one(VllmSetupView)
        draft = replace(
            view.draft,
            dtype="bfloat16",
            tensor_parallel_size=2,
            maximum_model_length=8192,
            gpu_memory_utilization=0.75,
            trust_remote_code=True,
            raw_arguments="--enable-prefix-caching",
        )
        view.apply_state(
            draft=draft,
            state=VllmReadinessState.NOT_CONFIGURED,
            preflight=None,
            profiles_ready=True,
        )
        advanced = view.query_one("#vllm-advanced-options", Collapsible)
        advanced.collapsed = False

        assert view.query_one("#vllm-dtype", Select).value == "bfloat16"
        assert view.query_one("#vllm-tensor-parallel-size", Input).value == "2"
        assert view.query_one("#vllm-maximum-model-length", Input).value == "8192"
        assert view.query_one("#vllm-gpu-memory-utilization", Input).value == "0.75"
        assert "Enabled" in str(view.query_one("#vllm-trust-remote-code", Button).label)
        consequence = str(
            view.query_one("#vllm-trust-remote-code-help", Label).renderable
        )
        assert "model code" in consequence
        assert view.query_one("#vllm-advanced-arguments", Collapsible).collapsed
        assert view.query_one("#vllm-raw-arguments", TextArea).text == (
            "--enable-prefix-caching"
        )

        view.apply_state(
            draft=draft,
            state=VllmReadinessState.NEEDS_ATTENTION,
            profiles_ready=True,
            preflight=VllmPreflightResult(
                generation=2,
                fingerprint=semantic_fingerprint(draft),
                issues=(
                    VllmIssue("invalid_tensor_parallel_size", "tensor_parallel_size"),
                ),
            ),
        )
        help_copy = view.query_one("#vllm-tensor-parallel-size-help", Label)
        assert help_copy.display
        assert "positive whole number" in str(help_copy.renderable)


async def test_existing_server_discovery_requires_explicit_model_selection():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)):
        view = app.query_one(VllmSetupView)
        draft = replace(
            view.draft,
            mode=VllmMode.EXISTING,
            existing_server_url="http://127.0.0.1:8000/v1",
        )
        owner = VllmConnectionOwner()
        token = owner.begin(draft, runtime_owner="external")
        assert owner.settle(
            token,
            VllmProbeResult(
                token=token,
                state=VllmReadinessState.NOT_CONFIGURED,
                target=None,
                issue=None,
                activity=(VllmActivityEvent("models_discovered", "under_1s"),),
                discovered_model_ids=("org/first", "org/second"),
            ),
        )
        view.apply_state(
            draft=draft,
            state=VllmReadinessState.NOT_CONFIGURED,
            preflight=None,
            connection=owner.snapshot(),
            discovered_model_ids=("org/first", "org/second"),
            credential_configured=True,
            profiles_ready=True,
        )

        selector = view.query_one("#vllm-existing-model", Select)
        assert selector.value is Select.NULL
        assert "Select a returned model" in str(
            view.query_one("#vllm-existing-model-help", Label).renderable
        )
        assert (
            "configured"
            in str(view.query_one("#vllm-credential-status", Label).renderable).lower()
        )
        assert "Check connection" in str(
            view.query_one("#vllm-check-setup", Button).label
        )
        assert owner.snapshot().target is None


async def test_checking_exposes_generation_bound_cancel_action():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)):
        view = app.query_one(VllmSetupView)
        owner = VllmConnectionOwner()
        token = owner.begin(view.draft, runtime_owner="chatbook")
        view.apply_state(
            draft=view.draft,
            state=VllmReadinessState.CHECKING,
            preflight=None,
            connection=owner.snapshot(),
            profiles_ready=True,
        )
        assert not view.query_one("#vllm-cancel-check", Button).disabled
        assert view.query_one("#vllm-cancel-check", Button).display
        assert not view.query_one("#vllm-check-setup", Button).display
        assert token.generation > 0


async def test_current_server_is_separate_from_modified_next_restart_without_path_leak():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)):
        view = app.query_one(VllmSetupView)
        current = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="/private/PATH_CANARY/bin/python",
            model_source=VllmModelSource.LOCAL_DIRECTORY,
            model_value="/private/MODEL_CANARY",
            raw_arguments="--adapter COMMAND_CANARY",
        )
        profile = profile_from_draft("Local GPU", current)
        document = VllmProfileDocumentV1(1, 3, profile.profile_id, (profile,))
        draft = replace(
            draft_from_profile(profile, raw_arguments="--other-launch-value"),
            model_value="/private/NEXT_MODEL_CANARY",
            port=8001,
        )
        preflight = VllmPreflightResult(
            generation=8,
            fingerprint=semantic_fingerprint(draft),
            issues=(),
            cli_path=Path("/safe/vllm"),
        )
        snapshot = launch_snapshot_from_draft(
            current,
            generation=7,
            profile_id=profile.profile_id,
            profile_name=profile.name,
        )

        view.apply_state(
            draft=draft,
            state=VllmReadinessState.READY_TO_START,
            preflight=preflight,
            current_launch_snapshot=snapshot,
            profiles=document,
            runtime_active=True,
            profiles_ready=True,
        )

        current_copy = str(
            app.query_one("#vllm-current-server-summary", Label).renderable
        )
        next_copy = str(app.query_one("#vllm-next-restart-state", Label).renderable)
        changed_copy = str(
            app.query_one("#vllm-next-restart-changes", Label).renderable
        )
        assert "Current server" in current_copy
        assert "Local GPU" in current_copy
        assert "Modified for next restart" in next_copy
        assert changed_copy == "Changed: Model · Port · Advanced arguments"
        assert app.query_one("#vllm-check-setup", Button).display
        assert "Check draft" in str(app.query_one("#vllm-check-setup", Button).label)
        assert app.query_one("#vllm-stop", Button).display
        assert not app.query_one("#vllm-restart", Button).disabled
        safe_projection = f"{current_copy} {next_copy} {changed_copy}"
        assert not any(
            canary in safe_projection
            for canary in (
                "PATH_CANARY",
                "MODEL_CANARY",
                "NEXT_MODEL_CANARY",
                "COMMAND_CANARY",
            )
        )


async def test_profile_buttons_post_exact_actions_and_raw_arguments_are_launch_only():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(VllmSetupView)
        profile = profile_from_draft(
            "GPU 0", replace(view.draft, model_value="org/model")
        )
        document = VllmProfileDocumentV1(1, 1, profile.profile_id, (profile,))
        view.apply_state(
            draft=draft_from_profile(profile, raw_arguments="--launch-only"),
            state=VllmReadinessState.NOT_CONFIGURED,
            preflight=None,
            profiles=document,
            profiles_ready=True,
        )
        assert app.query_one("#vllm-profile-select", Select).value == profile.profile_id
        assert "Launch only · not saved in profiles." in " ".join(
            str(label.renderable) for label in view.query(Label)
        )

        app.query_one("#vllm-profile-name", Input).value = "Renamed GPU"
        for selector in (
            "#vllm-profile-create-button",
            "#vllm-profile-save-button",
            "#vllm-profile-rename-button",
            "#vllm-profile-duplicate-button",
            "#vllm-profile-delete-button",
        ):
            await pilot.click(selector)
            await pilot.pause()

        assert [type(message) for message in app.profile_events] == [
            VllmSetupView.CreateProfileRequested,
            VllmSetupView.SaveProfileRequested,
            VllmSetupView.RenameProfileRequested,
            VllmSetupView.DuplicateProfileRequested,
            VllmSetupView.DeleteProfileRequested,
        ]


async def test_existing_server_mode_disables_local_profile_mutations_with_explanation():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(VllmSetupView)
        view.apply_state(
            draft=replace(
                view.draft,
                mode=VllmMode.EXISTING,
                existing_server_url="http://127.0.0.1:8000/v1",
            ),
            state=VllmReadinessState.NOT_CONFIGURED,
            preflight=None,
            profiles_ready=True,
        )

        profile_select = view.query_one("#vllm-profile-select", Select)
        assert profile_select.disabled
        assert view.query_one("#vllm-profile-name", Input).disabled
        for selector in (
            "#vllm-profile-create-button",
            "#vllm-profile-save-button",
            "#vllm-profile-rename-button",
            "#vllm-profile-duplicate-button",
            "#vllm-profile-delete-button",
        ):
            button = view.query_one(selector, Button)
            assert button.disabled
            button.press()
        await pilot.pause()

        assert app.profile_events == []
        profile_help = view.query_one("#vllm-profile-help", Label)
        assert profile_help.display
        assert "Start on this computer" in str(profile_help.renderable)

        view.apply_state(
            draft=replace(view.draft, mode=VllmMode.LOCAL),
            state=VllmReadinessState.NOT_CONFIGURED,
            preflight=None,
            profiles_ready=True,
        )
        assert not profile_select.disabled


async def test_existing_mode_forged_profile_events_preserve_repository(
    monkeypatch,
    tmp_path: Path,
):
    """Disabled controls are backed by authoritative view/controller guards."""

    repository = VllmProfileRepository(tmp_path / "profiles.json")
    first = repository.save(
        profile_from_draft(
            "First",
            VllmLaunchDraft(
                mode=VllmMode.LOCAL,
                python_environment="python",
                model_source=VllmModelSource.HUGGING_FACE,
                model_value="org/first",
            ),
        ),
        expected_revision=0,
    )
    second = repository.save(
        profile_from_draft(
            "Second",
            replace(draft_from_profile(first.profile), model_value="org/second"),
        ),
        expected_revision=first.document.revision,
    )
    selected = repository.select(
        first.profile.profile_id,
        expected_revision=second.document.revision,
    )
    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        screen._vllm_profile_repository = repository
        screen._accept_vllm_profiles(selected.document)
        local_draft = screen._vllm_draft
        screen._vllm_draft = replace(
            local_draft,
            mode=VllmMode.EXISTING,
            existing_server_url="http://127.0.0.1:8000/v1",
        )
        screen._apply_vllm_view_state(focus=False)
        await pilot.pause()
        profile_select = view.query_one("#vllm-profile-select", Select)
        assert profile_select.disabled

        posted = []
        original_post_message = view.post_message
        monkeypatch.setattr(view, "post_message", posted.append)
        view._on_profile_selected(
            Select.Changed(profile_select, second.profile.profile_id)
        )
        monkeypatch.setattr(view, "post_message", original_post_message)
        assert posted == []

        scheduled = []
        pushed = []
        original_start_mutation = screen._start_vllm_profile_mutation
        monkeypatch.setattr(screen, "_start_vllm_profile_mutation", scheduled.append)
        original_push_screen = app.push_screen
        monkeypatch.setattr(
            app, "push_screen", lambda *args, **kwargs: pushed.append(args)
        )
        before_bytes = repository.path.read_bytes()
        before_document = repository.load()
        profile_id = first.profile.profile_id
        screen._on_vllm_profile_selected(
            VllmSetupView.ProfileSelected(second.profile.profile_id)
        )
        screen._on_vllm_create_profile(
            VllmSetupView.CreateProfileRequested("Forged", local_draft)
        )
        screen._on_vllm_save_profile(
            VllmSetupView.SaveProfileRequested(profile_id, local_draft)
        )
        screen._on_vllm_rename_profile(
            VllmSetupView.RenameProfileRequested(profile_id, "Forged rename")
        )
        screen._on_vllm_duplicate_profile(
            VllmSetupView.DuplicateProfileRequested(profile_id)
        )
        screen._on_vllm_delete_profile(VllmSetupView.DeleteProfileRequested(profile_id))
        monkeypatch.setattr(app, "push_screen", original_push_screen)
        screen._confirm_vllm_profile_delete(
            True,
            profile_id,
            selected.document.revision,
        )
        monkeypatch.setattr(
            screen,
            "_start_vllm_profile_mutation",
            original_start_mutation,
        )

        assert scheduled == []
        assert pushed == []
        assert repository.path.read_bytes() == before_bytes
        assert repository.load() == before_document


@pytest.mark.parametrize(
    ("source", "expected_control", "expected_copy"),
    [
        (VllmModelSource.HUGGING_FACE, "vllm-hf-model", "Hugging Face"),
        (
            VllmModelSource.LOCAL_DIRECTORY,
            "vllm-local-model-directory",
            "local model directory",
        ),
    ],
)
async def test_profile_model_repair_focuses_visible_source_specific_control(
    source, expected_control, expected_copy, tmp_path: Path
):
    app = _VllmHost()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(VllmSetupView)
        view.apply_state(
            draft=replace(
                view.draft,
                model_source=source,
                model_value=(
                    "invalid"
                    if source is VllmModelSource.HUGGING_FACE
                    else str(tmp_path)
                ),
            ),
            state=VllmReadinessState.NEEDS_ATTENTION,
            preflight=None,
            profiles_ready=True,
        )

        view.show_profile_validation_error(
            "model_value",
            "invalid_hugging_face_model"
            if source is VllmModelSource.HUGGING_FACE
            else "invalid_model_directory",
        )
        await pilot.pause()

        control = view.query_one(f"#{expected_control}", Input)
        assert control.display
        assert app.focused is control
        assert expected_copy in str(
            view.query_one("#vllm-model-help", Label).renderable
        )


@pytest.mark.parametrize(
    ("message", "expected_help", "expected_control", "expected_copy"),
    [
        (
            "profile names must be unique",
            "vllm-profile-name-help",
            "vllm-profile-name",
            "unique profile name",
        ),
        (
            "profile store is capped at 32",
            "vllm-profile-help",
            "vllm-profile-select",
            "Profile limit reached",
        ),
        (
            "profile is unavailable",
            "vllm-profile-help",
            "vllm-profile-select",
            "no longer available",
        ),
    ],
)
async def test_async_profile_validation_routes_to_adjacent_recovery(
    message, expected_help, expected_control, expected_copy
):
    def fail_profile_change():
        raise VllmProfileValidationError(message)

    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)

        await screen._run_vllm_profile_mutation(fail_profile_change)
        await pilot.pause()

        help_label = view.query_one(f"#{expected_help}", Label)
        assert help_label.display
        assert expected_copy in str(help_label.renderable)
        assert app.focused is view.query_one(f"#{expected_control}")


async def test_mounted_rename_and_duplicate_validation_stays_action_adjacent(
    monkeypatch, tmp_path: Path
):
    repo = VllmProfileRepository(tmp_path / "vllm_launch_profiles.json")
    local_draft = VllmLaunchDraft(
        mode=VllmMode.LOCAL,
        python_environment="python",
        model_source=VllmModelSource.HUGGING_FACE,
        model_value="org/model",
    )
    first = profile_from_draft("First", local_draft)
    second = profile_from_draft("Second", local_draft)
    document = repo.save(first, expected_revision=0).document
    document = repo.save(second, expected_revision=document.revision).document
    document = repo.select(
        first.profile_id, expected_revision=document.revision
    ).document

    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        screen._vllm_profile_repository = repo
        screen._accept_vllm_profiles(document)
        await pilot.pause()

        view.query_one("#vllm-profile-name", Input).value = "Second"
        await pilot.click("#vllm-profile-rename-button")
        await _wait_for_profile_mutation_idle(screen, pilot)
        name_help = view.query_one("#vllm-profile-name-help", Label)
        assert name_help.display
        assert "unique profile name" in str(name_help.renderable)
        assert app.focused is view.query_one("#vllm-profile-name", Input)

        def reject_duplicate(*_args, **_kwargs):
            raise VllmProfileValidationError("profile store is capped at 32")

        monkeypatch.setattr(repo, "duplicate", reject_duplicate)
        await pilot.click("#vllm-profile-duplicate-button")
        await _wait_for_profile_mutation_idle(screen, pilot)
        profile_help = view.query_one("#vllm-profile-help", Label)
        assert profile_help.display
        assert "Profile limit reached" in str(profile_help.renderable)
        assert app.focused is view.query_one("#vllm-profile-select", Select)


async def test_profile_validation_classifier_maps_every_editable_schema_field():
    from tldw_chatbook.UI.Screens.llm_screen import (
        _classify_vllm_profile_validation,
    )

    cases = (
        ("name must not be empty", "name"),
        ("python_environment must be a string", "python_environment"),
        ("model_source is invalid", "model_source"),
        ("model_value is invalid for model_source", "model_value"),
        ("bind_address must not be empty", "bind_address"),
        ("port must be an integer from 1 to 65535", "port"),
        ("dtype is not supported", "dtype"),
        ("tensor_parallel_size must be positive", "tensor_parallel_size"),
        ("maximum_model_length must be positive", "maximum_model_length"),
        ("gpu_memory_utilization must be finite", "gpu_memory_utilization"),
        ("trust_remote_code must be a boolean", "trust_remote_code"),
        ("only local launch drafts can be profiled", "mode"),
        ("profile store is capped at 32", "profile"),
        ("profile is unavailable", "profile"),
        ("selected_profile_id must identify a profile", "profile"),
        ("document keys do not match V1", "profile"),
    )

    for message, expected_field in cases:
        field, classification = _classify_vllm_profile_validation(
            VllmProfileValidationError(message)
        )
        assert field == expected_field, message
        assert classification, message


async def _wait_for_profile_confirmation(app, pilot) -> ConfirmationDialog:
    for _ in range(40):
        await pilot.pause()
        if isinstance(app.screen, ConfirmationDialog):
            return app.screen
    raise AssertionError("profile deletion confirmation did not mount")


async def _wait_for_profile_mutation_idle(screen: LLMScreen, pilot) -> None:
    for _ in range(4):
        await pilot.pause()
    for _ in range(80):
        worker = screen._vllm_profile_worker
        if worker is None or worker.is_finished:
            await pilot.pause()
            worker = screen._vllm_profile_worker
            if worker is None or worker.is_finished:
                return
        await pilot.pause()
    raise AssertionError("profile mutation did not settle")


@pytest.mark.parametrize("repair", ["python", "local_model"])
async def test_selected_profile_immediately_projects_local_repair_without_probing(
    repair, tmp_path: Path
):
    python_path = tmp_path / "venv/bin/python"
    python_path.parent.mkdir(parents=True)
    python_path.touch()
    python_path.chmod(0o755)
    model_path = tmp_path / "missing-model"
    draft = VllmLaunchDraft(
        mode=VllmMode.LOCAL,
        python_environment=(
            str(tmp_path / "missing-python") if repair == "python" else str(python_path)
        ),
        model_source=(
            VllmModelSource.HUGGING_FACE
            if repair == "python"
            else VllmModelSource.LOCAL_DIRECTORY
        ),
        model_value="org/model" if repair == "python" else str(model_path),
    )
    profile = profile_from_draft("Repair me", draft)
    document = VllmProfileDocumentV1(1, 1, profile.profile_id, (profile,))

    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        screen._accept_vllm_profiles(document)
        await pilot.pause()

        assert screen._vllm_preflight is not None
        assert screen._vllm_preflight.repair_only is True
        assert str(view.query_one("#vllm-readiness-state", Label).renderable) == (
            "Needs attention"
        )
        assert view.query_one("#vllm-recovery-primary", Button).display
        assert "not checked" in str(
            view.query_one("#vllm-check-installation", Label).renderable
        )
        if repair == "python":
            help_label = view.query_one("#vllm-python-environment-help", Label)
            control = view.query_one("#vllm-python-environment", Input)
            assert "not found" in str(help_label.renderable)
        else:
            help_label = view.query_one("#vllm-model-help", Label)
            control = view.query_one("#vllm-local-model-directory", Input)
            assert "existing local model directory" in str(help_label.renderable)
            assert not view.query_one("#vllm-hf-model", Input).display
        assert help_label.display
        assert control.display and control.can_focus


async def test_selected_profile_projects_invalid_bind_repair_without_any_probe(
    monkeypatch,
) -> None:
    app = _build_test_app()
    profile = replace(default_vllm_profile(), bind_address="not a host")
    document = VllmProfileDocumentV1(1, 7, profile.profile_id, (profile,))
    runtime_calls: list[str] = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.llm_screen.run_vllm_preflight",
        lambda *_args, **_kwargs: runtime_calls.append("preflight"),
    )
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        screen._accept_vllm_profiles(document)
        await pilot.pause()

        assert runtime_calls == []
        assert screen._vllm_preflight is not None
        assert screen._vllm_preflight.repair_only is True
        assert screen._vllm_preflight.issues == (
            VllmIssue("invalid_bind_address", "bind_address"),
        )
        help_copy = view.query_one("#vllm-bind-address-help", Label)
        assert help_copy.display
        assert "IP address" in str(help_copy.renderable)
        assert view.query_one("#vllm-check-setup", Button).display
        assert view.query_one("#vllm-start", Button).disabled


def _write_multi_profile_legacy_bind_store(
    path: Path,
) -> tuple[
    VllmProfileRepository,
    VllmProfileDocumentV1,
    object,
    object,
    bytes,
]:
    valid = default_vllm_profile()
    invalid = replace(
        default_vllm_profile(),
        profile_id="00000000-0000-4000-8000-000000000002",
        name="Repair bind",
        bind_address="not a host",
    )

    def payload(profile) -> dict[str, object]:
        return {
            "profile_id": profile.profile_id,
            "name": profile.name,
            "python_environment": profile.python_environment,
            "model_source": profile.model_source.value,
            "model_value": profile.model_value,
            "bind_address": profile.bind_address,
            "port": profile.port,
            "dtype": profile.dtype,
            "tensor_parallel_size": profile.tensor_parallel_size,
            "maximum_model_length": profile.maximum_model_length,
            "gpu_memory_utilization": profile.gpu_memory_utilization,
            "trust_remote_code": profile.trust_remote_code,
        }

    path.write_text(
        json.dumps(
            {
                "version": 1,
                "revision": 7,
                "selected_profile_id": valid.profile_id,
                "profiles": [payload(valid), payload(invalid)],
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    repository = VllmProfileRepository(path)
    original = path.read_bytes()
    return repository, repository.load(), valid, invalid, original


async def test_invalid_nonselected_profile_enters_nonpersisting_repair_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repository, document, valid, invalid, original = (
        _write_multi_profile_legacy_bind_store(tmp_path / "profiles.json")
    )
    runtime_calls: list[str] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.llm_screen.run_vllm_preflight",
        lambda *_args, **_kwargs: runtime_calls.append("preflight"),
    )
    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        screen._vllm_profile_repository = repository
        screen._accept_vllm_profiles(document)
        scheduled: list[object] = []
        monkeypatch.setattr(screen, "_start_vllm_profile_mutation", scheduled.append)

        screen._on_vllm_profile_selected(
            VllmSetupView.ProfileSelected(invalid.profile_id)
        )
        await pilot.pause()

        assert scheduled == []
        assert runtime_calls == []
        assert screen._vllm_repair_profile_id == invalid.profile_id
        assert screen._vllm_draft.bind_address == "not a host"
        assert screen._vllm_preflight is not None
        assert screen._vllm_preflight.repair_only
        assert (
            view.query_one("#vllm-profile-select", Select).value == invalid.profile_id
        )
        bind_help = view.query_one("#vllm-bind-address-help", Label)
        assert bind_help.display
        assert "IP address" in str(bind_help.renderable)
        assert not view.query_one("#vllm-profile-save-button", Button).disabled
        assert not view.query_one("#vllm-profile-delete-button", Button).disabled
        assert view.query_one("#vllm-profile-create-button", Button).disabled
        assert view.query_one("#vllm-profile-rename-button", Button).disabled
        assert view.query_one("#vllm-profile-duplicate-button", Button).disabled
        assert view.query_one("#vllm-check-setup", Button).disabled
        assert repository.path.read_bytes() == original
        reopened = VllmProfileRepository(repository.path).load()
        assert reopened.selected_profile_id == valid.profile_id
        assert reopened.profiles == document.profiles

        screen._on_vllm_create_profile(
            VllmSetupView.CreateProfileRequested("Blocked", screen._vllm_draft)
        )
        screen._on_vllm_save_profile(
            VllmSetupView.SaveProfileRequested(valid.profile_id, screen._vllm_draft)
        )
        screen._on_vllm_rename_profile(
            VllmSetupView.RenameProfileRequested(valid.profile_id, "Blocked")
        )
        screen._on_vllm_duplicate_profile(
            VllmSetupView.DuplicateProfileRequested(valid.profile_id)
        )
        screen._on_vllm_delete_profile(
            VllmSetupView.DeleteProfileRequested(valid.profile_id)
        )
        assert scheduled == []
        assert repository.path.read_bytes() == original


@pytest.mark.parametrize("repair_action", ["save", "delete"])
async def test_invalid_nonselected_profile_can_be_saved_or_deleted_from_repair_state(
    repair_action: str,
    tmp_path: Path,
) -> None:
    repository, document, valid, invalid, _ = _write_multi_profile_legacy_bind_store(
        tmp_path / "profiles.json"
    )
    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, _ = await _mount_vllm_screen(app, pilot)
        screen._vllm_profile_repository = repository
        screen._accept_vllm_profiles(document)
        screen._on_vllm_profile_selected(
            VllmSetupView.ProfileSelected(invalid.profile_id)
        )
        await pilot.pause()

        if repair_action == "save":
            repaired_draft = replace(
                screen._vllm_draft,
                bind_address="127.0.0.1",
            )
            screen._on_vllm_draft_changed(VllmSetupView.DraftChanged(repaired_draft))
            screen._on_vllm_save_profile(
                VllmSetupView.SaveProfileRequested(
                    invalid.profile_id,
                    repaired_draft,
                )
            )
        else:
            screen._on_vllm_delete_profile(
                VllmSetupView.DeleteProfileRequested(invalid.profile_id)
            )
            await _wait_for_profile_confirmation(app, pilot)
            await pilot.click("#confirm-button")
        await _wait_for_profile_mutation_idle(screen, pilot)

        reopened = VllmProfileRepository(repository.path).load()
        assert screen._vllm_repair_profile_id is None
        assert reopened.profiles[0] == valid
        if repair_action == "save":
            assert reopened.selected_profile_id == invalid.profile_id
            assert reopened.profiles[1].bind_address == "127.0.0.1"
        else:
            assert reopened.selected_profile_id == valid.profile_id
            assert reopened.profiles == (valid,)


def _write_two_invalid_profile_store(
    path: Path,
) -> tuple[
    VllmProfileRepository,
    VllmProfileDocumentV1,
    VllmLaunchProfileV1,
    VllmLaunchProfileV1,
    VllmLaunchProfileV1,
]:
    valid = default_vllm_profile()
    first = replace(
        valid,
        profile_id="00000000-0000-4000-8000-000000000002",
        name="Repair first",
        bind_address="not a host",
    )
    second = replace(
        valid,
        profile_id="00000000-0000-4000-8000-000000000003",
        name="Repair second",
        bind_address="https://example.test",
    )

    def payload(profile) -> dict[str, object]:
        return {
            "profile_id": profile.profile_id,
            "name": profile.name,
            "python_environment": profile.python_environment,
            "model_source": profile.model_source.value,
            "model_value": profile.model_value,
            "bind_address": profile.bind_address,
            "port": profile.port,
            "dtype": profile.dtype,
            "tensor_parallel_size": profile.tensor_parallel_size,
            "maximum_model_length": profile.maximum_model_length,
            "gpu_memory_utilization": profile.gpu_memory_utilization,
            "trust_remote_code": profile.trust_remote_code,
        }

    path.write_text(
        json.dumps(
            {
                "version": 1,
                "revision": 11,
                "selected_profile_id": valid.profile_id,
                "profiles": [payload(valid), payload(first), payload(second)],
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    repository = VllmProfileRepository(path)
    return repository, repository.load(), valid, first, second


async def test_multiple_invalid_profiles_can_be_repaired_one_at_a_time_without_probe(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repository, document, valid, first, second = _write_two_invalid_profile_store(
        tmp_path / "profiles.json"
    )
    runtime_calls: list[str] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.llm_screen.run_vllm_preflight",
        lambda *_args, **_kwargs: runtime_calls.append("preflight"),
    )
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.llm_screen.probe_vllm_target",
        lambda *_args, **_kwargs: runtime_calls.append("probe"),
    )
    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        screen._vllm_profile_repository = repository
        screen._accept_vllm_profiles(document)

        screen._on_vllm_profile_selected(
            VllmSetupView.ProfileSelected(first.profile_id)
        )
        await pilot.pause()
        repaired_first = replace(screen._vllm_draft, bind_address="127.0.0.1")
        screen._on_vllm_draft_changed(VllmSetupView.DraftChanged(repaired_first))
        screen._on_vllm_save_profile(
            VllmSetupView.SaveProfileRequested(first.profile_id, repaired_first)
        )
        await _wait_for_profile_mutation_idle(screen, pilot)

        reopened_after_first = VllmProfileRepository(repository.path).load()
        assert reopened_after_first.revision == 12
        assert reopened_after_first.selected_profile_id == first.profile_id
        assert reopened_after_first.profiles[0] == valid
        assert reopened_after_first.profiles[2] == second
        assert screen._vllm_profiles_require_repair()

        screen._on_vllm_profile_selected(
            VllmSetupView.ProfileSelected(second.profile_id)
        )
        await pilot.pause()
        assert screen._vllm_repair_profile_id == second.profile_id
        assert view.query_one("#vllm-profile-select", Select).value == second.profile_id
        repaired_second = replace(screen._vllm_draft, bind_address="localhost")
        screen._on_vllm_draft_changed(VllmSetupView.DraftChanged(repaired_second))
        screen._on_vllm_save_profile(
            VllmSetupView.SaveProfileRequested(second.profile_id, repaired_second)
        )
        await _wait_for_profile_mutation_idle(screen, pilot)

        final = VllmProfileRepository(repository.path).load()
        assert final.revision == 13
        assert final.selected_profile_id == second.profile_id
        assert not screen._vllm_profiles_require_repair()
        assert screen._vllm_repair_profile_id is None
        assert runtime_calls == []


async def test_multiple_invalid_profiles_can_be_deleted_sequentially_after_reopen(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repository, document, valid, first, second = _write_two_invalid_profile_store(
        tmp_path / "profiles.json"
    )
    runtime_calls: list[str] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.llm_screen.run_vllm_preflight",
        lambda *_args, **_kwargs: runtime_calls.append("preflight"),
    )
    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, _ = await _mount_vllm_screen(app, pilot)
        screen._vllm_profile_repository = repository
        screen._accept_vllm_profiles(document)
        screen._on_vllm_profile_selected(
            VllmSetupView.ProfileSelected(first.profile_id)
        )
        screen._on_vllm_delete_profile(
            VllmSetupView.DeleteProfileRequested(first.profile_id)
        )
        await _wait_for_profile_confirmation(app, pilot)
        await pilot.click("#confirm-button")
        await _wait_for_profile_mutation_idle(screen, pilot)

        reopened_after_first = VllmProfileRepository(repository.path).load()
        assert reopened_after_first.selected_profile_id == valid.profile_id
        assert reopened_after_first.profiles == (valid, second)
        screen._accept_vllm_profiles(reopened_after_first)
        screen._on_vllm_profile_selected(
            VllmSetupView.ProfileSelected(second.profile_id)
        )
        screen._on_vllm_delete_profile(
            VllmSetupView.DeleteProfileRequested(second.profile_id)
        )
        await _wait_for_profile_confirmation(app, pilot)
        await pilot.click("#confirm-button")
        await _wait_for_profile_mutation_idle(screen, pilot)

        final = VllmProfileRepository(repository.path).load()
        assert final.selected_profile_id == valid.profile_id
        assert final.profiles == (valid,)
        assert runtime_calls == []


async def test_forged_repair_save_draft_is_rejected_before_worker_or_write(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repository, document, valid, invalid, original = (
        _write_multi_profile_legacy_bind_store(tmp_path / "profiles.json")
    )
    runtime_calls: list[str] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.llm_screen.run_vllm_preflight",
        lambda *_args, **_kwargs: runtime_calls.append("preflight"),
    )
    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, _ = await _mount_vllm_screen(app, pilot)
        screen._vllm_profile_repository = repository
        screen._accept_vllm_profiles(document)
        screen._on_vllm_profile_selected(
            VllmSetupView.ProfileSelected(invalid.profile_id)
        )
        await pilot.pause()
        worker_before = screen._vllm_profile_worker
        forged = replace(screen._vllm_draft, bind_address="127.0.0.1")

        screen._on_vllm_save_profile(
            VllmSetupView.SaveProfileRequested(invalid.profile_id, forged)
        )
        await pilot.pause()

        assert screen._vllm_profile_worker is worker_before
        assert repository.path.read_bytes() == original
        unchanged = VllmProfileRepository(repository.path).load()
        assert unchanged.revision == document.revision
        assert unchanged.selected_profile_id == valid.profile_id

        screen._on_vllm_draft_changed(VllmSetupView.DraftChanged(forged))
        screen._on_vllm_save_profile(
            VllmSetupView.SaveProfileRequested(invalid.profile_id, forged)
        )
        await _wait_for_profile_mutation_idle(screen, pilot)

        repaired = VllmProfileRepository(repository.path).load()
        assert repaired.revision == document.revision + 1
        assert repaired.selected_profile_id == invalid.profile_id
        assert repaired.profiles[1].bind_address == "127.0.0.1"
        assert runtime_calls == []


async def test_preflight_exception_settles_current_generation_for_retry(
    monkeypatch,
) -> None:
    app = _build_test_app()
    private_canary = "PRIVATE_PREFLIGHT_EXCEPTION_CANARY"
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        screen._vllm_profiles_loaded = True
        draft = replace(screen._vllm_draft, model_value="org/model")
        screen._vllm_draft = draft

        def fail_preflight(*_args, **_kwargs):
            raise RuntimeError(private_canary)

        monkeypatch.setattr(
            "tldw_chatbook.UI.Screens.llm_screen.run_vllm_preflight",
            fail_preflight,
        )
        screen._on_vllm_check_requested(VllmSetupView.CheckRequested(draft))
        for _ in range(40):
            await pilot.pause()
            if screen._vllm_owner.snapshot().state is not VllmReadinessState.CHECKING:
                break

        snapshot = screen._vllm_owner.snapshot()
        assert snapshot.state is VllmReadinessState.NEEDS_ATTENTION
        assert snapshot.issue == VllmIssue("launch_failed", "preflight")
        assert not view.query_one("#vllm-recovery-primary", Button).disabled
        assert not view.query_one("#vllm-cancel-check", Button).display
        assert private_canary not in " ".join(
            str(label.renderable) for label in view.query(Label)
        )


@pytest.mark.parametrize(
    "existing_server_url",
    ["http://[", "https://example.test/" + "x" * 2049],
    ids=("malformed-ipv6", "oversized"),
)
async def test_invalid_existing_url_settles_without_probe_dispatch(
    monkeypatch,
    existing_server_url: str,
) -> None:
    app = _build_test_app()
    probe_calls: list[object] = []
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        screen._vllm_profiles_loaded = True
        draft = replace(
            screen._vllm_draft,
            mode=VllmMode.EXISTING,
            existing_server_url=existing_server_url,
        )
        screen._vllm_draft = draft
        monkeypatch.setattr(
            "tldw_chatbook.UI.Screens.llm_screen.probe_vllm_target",
            lambda request: probe_calls.append(request),
        )

        screen._on_vllm_check_requested(VllmSetupView.CheckRequested(draft))
        for _ in range(40):
            await pilot.pause()
            if screen._vllm_owner.snapshot().state is not VllmReadinessState.CHECKING:
                break

        snapshot = screen._vllm_owner.snapshot()
        assert probe_calls == []
        assert snapshot.state is VllmReadinessState.NEEDS_ATTENTION
        assert snapshot.issue == VllmIssue(
            "invalid_existing_server_url", "existing_server_url"
        )
        assert not view.query_one("#vllm-recovery-primary", Button).disabled
        assert not view.query_one("#vllm-cancel-check", Button).display
        rendered = " ".join(str(label.renderable) for label in view.query(Label))
        assert existing_server_url not in rendered
        assert len(rendered) < 20_000


async def test_probe_request_construction_exception_settles_without_dispatch(
    monkeypatch,
) -> None:
    app = _build_test_app()
    probe_calls: list[object] = []
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        screen._vllm_profiles_loaded = True
        draft = replace(
            screen._vllm_draft,
            mode=VllmMode.EXISTING,
            existing_server_url="http://127.0.0.1:8000/v1",
        )
        token = screen._vllm_owner.begin(draft, runtime_owner="external")

        monkeypatch.setattr(
            "tldw_chatbook.UI.Screens.llm_screen.VllmProbeRequest",
            lambda **_kwargs: (_ for _ in ()).throw(ValueError("PRIVATE_URL_CANARY")),
        )
        monkeypatch.setattr(
            "tldw_chatbook.UI.Screens.llm_screen.probe_vllm_target",
            lambda request: probe_calls.append(request),
        )
        await screen._probe_vllm_generation(token, draft, claim=None)
        await pilot.pause()

        snapshot = screen._vllm_owner.snapshot()
        assert probe_calls == []
        assert snapshot.state is VllmReadinessState.NEEDS_ATTENTION
        assert snapshot.issue == VllmIssue("invalid_endpoint", "connection")
        assert not view.query_one("#vllm-recovery-primary", Button).disabled
        assert not view.query_one("#vllm-cancel-check", Button).display
        assert "PRIVATE_URL_CANARY" not in " ".join(
            str(label.renderable) for label in view.query(Label)
        )


async def test_name_only_profile_refresh_preserves_stronger_full_preflight():
    draft = VllmLaunchDraft(
        mode=VllmMode.LOCAL,
        python_environment="python",
        model_source=VllmModelSource.HUGGING_FACE,
        model_value="org/model",
    )
    profile = profile_from_draft("Renamed only", draft)
    document = VllmProfileDocumentV1(1, 1, profile.profile_id, (profile,))

    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        full_preflight = VllmPreflightResult(
            generation=token.generation,
            fingerprint=token.fingerprint,
            issues=(),
            python_version="Python 3.12.0",
            vllm_version="vLLM 0.9.0",
            cli_path=Path("/safe/vllm"),
        )
        screen._vllm_draft = draft
        screen._vllm_preflight = full_preflight
        screen._settle_vllm_state(
            token,
            VllmReadinessState.READY_TO_START,
            activity_code="checking",
        )

        screen._accept_vllm_profiles(document)
        await pilot.pause()

        assert screen._vllm_preflight is full_preflight
        assert screen._vllm_preflight.repair_only is False
        assert "Python 3.12.0" in str(
            view.query_one("#vllm-check-environment", Label).renderable
        )
        assert "vLLM 0.9.0" in str(
            view.query_one("#vllm-check-installation", Label).renderable
        )
        assert not view.query_one("#vllm-start", Button).disabled


@pytest.mark.parametrize("dismissal", ["cancel", "escape", "backdrop"])
async def test_profile_delete_cancel_or_escape_preserves_exact_document(
    dismissal, monkeypatch, tmp_path: Path
):
    """Removing the confirmation gate must let Delete mutate before consent."""

    repo = VllmProfileRepository(tmp_path / "vllm_launch_profiles.json")
    profile = profile_from_draft(
        "PROFILE_SECRET_CANARY",
        VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="/private/PYTHON_PATH_CANARY",
            model_source=VllmModelSource.LOCAL_DIRECTORY,
            model_value="/private/MODEL_PATH_CANARY",
        ),
    )
    saved = repo.save(profile, expected_revision=0)
    calls = []
    real_delete = repo.delete

    def observed_delete(profile_id, *, expected_revision):
        calls.append((profile_id, expected_revision))
        return real_delete(profile_id, expected_revision=expected_revision)

    monkeypatch.setattr(repo, "delete", observed_delete)
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        screen._vllm_profile_repository = repo
        screen._accept_vllm_profiles(saved.document)
        await _wait_for_profile_mutation_idle(screen, pilot)
        baseline = screen._vllm_profiles
        before = repo.path.read_bytes()

        await pilot.click("#vllm-profile-delete-button")
        dialog = await _wait_for_profile_confirmation(app, pilot)
        assert repo.path.read_bytes() == before
        assert calls == []
        copy = " ".join(
            str(widget.renderable) for widget in dialog.query("Label, Static")
        )
        assert not any(
            canary in copy
            for canary in (
                "PROFILE_SECRET_CANARY",
                "PYTHON_PATH_CANARY",
                "MODEL_PATH_CANARY",
            )
        )

        if dismissal == "cancel":
            await pilot.click("#cancel-button")
        elif dismissal == "escape":
            await pilot.press("escape")
        else:
            await pilot.click(offset=(0, 0))
        await pilot.pause()

        assert app.screen is screen
        assert dialog.result is False
        assert repo.path.read_bytes() == before
        assert repo.load() == baseline
        assert calls == []
        assert app.focused is view.query_one("#vllm-profile-delete-button", Button)


async def test_confirmed_profile_delete_executes_selected_claim_once_and_recreates_default(
    monkeypatch, tmp_path: Path
):
    """Bypassing confirm or replaying its callback must break the exact call count."""

    repo = VllmProfileRepository(tmp_path / "vllm_launch_profiles.json")
    profile = profile_from_draft(
        "Only profile",
        VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="org/model",
        ),
    )
    saved = repo.save(profile, expected_revision=0)
    calls = []
    real_delete = repo.delete

    def observed_delete(profile_id, *, expected_revision):
        calls.append((profile_id, expected_revision))
        return real_delete(profile_id, expected_revision=expected_revision)

    monkeypatch.setattr(repo, "delete", observed_delete)
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen, _, _ = await _mount_vllm_screen(app, pilot)
        screen._vllm_profile_repository = repo
        screen._accept_vllm_profiles(saved.document)
        await _wait_for_profile_mutation_idle(screen, pilot)
        claim = screen._vllm_profiles

        await pilot.click("#vllm-profile-delete-button")
        await _wait_for_profile_confirmation(app, pilot)
        assert calls == []
        await pilot.click("#confirm-button")
        for _ in range(80):
            await pilot.pause()
            if calls and screen._vllm_profiles.revision > claim.revision:
                break
        else:
            raise AssertionError("confirmed profile deletion did not settle")

        assert calls == [(profile.profile_id, claim.revision)]
        restored = repo.load()
        assert restored == screen._vllm_profiles
        assert len(restored.profiles) == 1
        assert restored.profiles[0].name == "Default vLLM"


@pytest.mark.parametrize(
    ("terminal_actions", "confirmed"),
    [
        (("confirm", "confirm"), True),
        (("confirm", "cancel"), True),
        (("cancel", "confirm"), False),
    ],
)
async def test_profile_delete_queued_terminal_actions_settle_once(
    terminal_actions, confirmed, monkeypatch, tmp_path: Path
):
    """Removing the one-shot terminal guard must pop or settle more than once."""

    repo = VllmProfileRepository(tmp_path / "vllm_launch_profiles.json")
    profile = profile_from_draft(
        "Only profile",
        VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="org/model",
        ),
    )
    saved = repo.save(profile, expected_revision=0)
    delete_calls = []
    real_delete = repo.delete

    def observed_delete(profile_id, *, expected_revision):
        delete_calls.append((profile_id, expected_revision))
        return real_delete(profile_id, expected_revision=expected_revision)

    monkeypatch.setattr(repo, "delete", observed_delete)
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen, _, _ = await _mount_vllm_screen(app, pilot)
        screen._vllm_profile_repository = repo
        screen._accept_vllm_profiles(saved.document)
        await _wait_for_profile_mutation_idle(screen, pilot)
        claim = screen._vllm_profiles
        before = repo.path.read_bytes()
        underlying_stack = tuple(app.screen_stack)

        await pilot.click("#vllm-profile-delete-button")
        dialog = await _wait_for_profile_confirmation(app, pilot)
        callback_calls = []
        real_confirm = screen._confirm_vllm_profile_delete

        def observed_confirm(result, profile_id, revision):
            callback_calls.append((result, profile_id, revision))
            real_confirm(result, profile_id, revision)

        monkeypatch.setattr(screen, "_confirm_vllm_profile_delete", observed_confirm)
        for action in terminal_actions:
            dialog.query_one(f"#{action}-button", Button).press()

        for _ in range(100):
            await pilot.pause()
            mutation_settled = (
                not confirmed or screen._vllm_profiles.revision > claim.revision
            )
            if app.screen is screen and callback_calls and mutation_settled:
                break
        else:
            raise AssertionError(
                "queued terminal actions did not settle: "
                f"screen={type(app.screen).__name__}, "
                f"stack={[type(item).__name__ for item in app.screen_stack]}, "
                f"callbacks={callback_calls}, deletes={delete_calls}, "
                f"revision={screen._vllm_profiles.revision}"
            )

        assert app.screen is screen
        assert tuple(app.screen_stack) == underlying_stack
        assert app.screen is not dialog
        assert callback_calls == [(confirmed, profile.profile_id, claim.revision)]
        assert dialog.result is confirmed
        if confirmed:
            assert delete_calls == [(profile.profile_id, claim.revision)]
            restored = repo.load()
            assert restored == screen._vllm_profiles
            assert len(restored.profiles) == 1
            assert restored.profiles[0].name == "Default vLLM"
        else:
            assert delete_calls == []
            assert repo.path.read_bytes() == before
            assert repo.load() == claim


async def test_profile_delete_confirmation_rejects_stale_selection_claim(
    monkeypatch, tmp_path: Path
):
    """Dropping the revision/selection recheck must delete the wrong profile."""

    repo = VllmProfileRepository(tmp_path / "vllm_launch_profiles.json")
    first = repo.save(
        profile_from_draft(
            "First",
            VllmLaunchDraft(
                mode=VllmMode.LOCAL,
                python_environment="python",
                model_source=VllmModelSource.HUGGING_FACE,
                model_value="org/first",
            ),
        ),
        expected_revision=0,
    )
    second = repo.save(
        profile_from_draft(
            "Second",
            replace(draft_from_profile(first.profile), model_value="org/second"),
        ),
        expected_revision=first.document.revision,
    )
    selected_first = repo.select(
        first.profile.profile_id,
        expected_revision=second.document.revision,
    )
    calls = []
    real_delete = repo.delete

    def observed_delete(profile_id, *, expected_revision):
        calls.append((profile_id, expected_revision))
        return real_delete(profile_id, expected_revision=expected_revision)

    monkeypatch.setattr(repo, "delete", observed_delete)
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        screen._vllm_profile_repository = repo
        screen._accept_vllm_profiles(selected_first.document)
        await _wait_for_profile_mutation_idle(screen, pilot)
        claim = screen._vllm_profiles

        await pilot.click("#vllm-profile-delete-button")
        await _wait_for_profile_confirmation(app, pilot)
        newer = repo.select(
            second.profile.profile_id,
            expected_revision=claim.revision,
        )
        screen._accept_vllm_profiles(newer.document)
        before_confirm = repo.path.read_bytes()
        await pilot.click("#confirm-button")
        await pilot.pause()

        assert calls == []
        assert repo.path.read_bytes() == before_confirm
        assert repo.load() == newer.document
        assert app.focused is view.query_one("#vllm-profile-delete-button", Button)


async def test_profile_repository_io_is_threaded_and_selected_profile_restores(
    monkeypatch, tmp_path: Path
):
    path = tmp_path / "vllm_launch_profiles.json"
    repo = VllmProfileRepository(path)
    first = repo.save(
        profile_from_draft(
            "First",
            VllmLaunchDraft(
                mode=VllmMode.LOCAL,
                python_environment="python",
                model_source=VllmModelSource.HUGGING_FACE,
                model_value="org/first",
            ),
        ),
        expected_revision=0,
    )
    second = repo.save(
        profile_from_draft(
            "Second",
            replace(draft_from_profile(first.profile), model_value="org/second"),
        ),
        expected_revision=first.document.revision,
    )
    repo.select(first.profile.profile_id, expected_revision=second.document.revision)
    caller_thread = threading.get_ident()
    io_threads = []
    real_load = repo.load

    def observed_load():
        io_threads.append(threading.get_ident())
        return real_load()

    monkeypatch.setattr(repo, "load", observed_load)
    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        screen._vllm_profile_repository = repo
        await screen._load_vllm_profiles()

        assert io_threads and all(thread != caller_thread for thread in io_threads)
        assert screen._vllm_profiles.selected_profile_id == first.profile.profile_id
        assert screen._vllm_draft.model_value == "org/first"
        assert view.query_one("#vllm-profile-select", Select).value == (
            first.profile.profile_id
        )
        assert current_server_claim(app, "vllm") is None


async def test_restart_proves_old_process_dead_and_released_before_new_generation(
    monkeypatch,
):
    app = _build_test_app()
    process = _RunningProcess()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, _ = await _mount_vllm_screen(app, pilot)
        current = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="org/current",
        )
        old_token = screen._vllm_owner.begin(current, runtime_owner="chatbook")
        old_claim = reserve_server_launch(app, "vllm", authority="chatbook-vllm")
        assert old_claim is not None
        assert screen._vllm_owner.bind_launch_claim(old_token, old_claim)
        assert publish_server_process(app, "vllm", old_claim, process)
        assert screen._vllm_owner.settle(old_token, _ready_result(old_token))

        draft = replace(
            current,
            model_source=VllmModelSource.LOCAL_DIRECTORY,
            model_value="/private/NEXT_MODEL_CANARY",
            port=8001,
        )
        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        preflight = VllmPreflightResult(
            generation=token.generation,
            fingerprint=token.fingerprint,
            issues=(),
            cli_path=Path("/safe/vllm"),
        )
        screen._vllm_draft = draft
        screen._vllm_preflight = preflight
        screen._settle_vllm_state(
            token,
            VllmReadinessState.READY_TO_START,
            activity_code="checking",
        )

        confirmations = []

        def capture_confirmation(dialog, callback):
            confirmations.append((dialog, callback))

        monkeypatch.setattr(app, "push_screen", capture_confirmation)
        screen._on_vllm_restart_requested(
            VllmSetupView.RestartRequested(
                draft,
                ("Model source", "Model", "Port"),
            )
        )
        assert len(confirmations) == 1
        assert confirmations[0][0].message == (
            "Restart vLLM with changes to: Model source, Model, Port?"
        )
        assert "NEXT_MODEL_CANARY" not in confirmations[0][0].message

        order = []

        async def observed_stop(*args, **kwargs):
            order.append("stop")
            return await real_stop_server_process(*args, **kwargs)

        real_reserve = reserve_server_launch

        def observed_reserve(*args, **kwargs):
            assert process.poll() is not None
            assert server_lifecycle_snapshot(app, "vllm") == (None, None)
            order.append("reserve")
            return real_reserve(*args, **kwargs)

        def observed_launch(*args, **kwargs):
            order.append("launch")

        monkeypatch.setattr(
            "tldw_chatbook.UI.Screens.llm_screen.stop_server_process", observed_stop
        )
        monkeypatch.setattr(
            "tldw_chatbook.UI.Screens.llm_screen.reserve_server_launch",
            observed_reserve,
        )
        monkeypatch.setattr(screen, "_start_vllm_process_workers", observed_launch)

        assert await screen._restart_vllm_with_draft(draft)

        new_claim, new_process = server_lifecycle_snapshot(app, "vllm")
        assert new_claim is not None and new_claim is not old_claim
        assert new_process is None
        assert screen._vllm_owner.owns_launch_claim(new_claim)
        assert screen._vllm_owner.snapshot().generation > token.generation
        assert order == ["stop", "reserve", "launch"]


async def test_mounted_edit_check_and_restart_uses_exact_live_claim(monkeypatch):
    """Exercise the user-visible edit → Check draft → Restart path."""

    app = _build_test_app()
    process = _RunningProcess()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        screen._vllm_profiles_loaded = True
        current = replace(
            screen._vllm_draft,
            model_value="org/current",
        )
        old_token = screen._vllm_owner.begin(current, runtime_owner="chatbook")
        old_claim = reserve_server_launch(app, "vllm", authority="chatbook-vllm")
        assert old_claim is not None
        assert screen._vllm_owner.bind_launch_claim(old_token, old_claim)
        assert publish_server_process(app, "vllm", old_claim, process)
        assert screen._vllm_owner.settle(old_token, _ready_result(old_token))
        screen._vllm_draft = current
        screen._apply_vllm_view_state()

        view.query_one("#vllm-hf-model", Input).value = "org/next"
        await pilot.pause()
        assert view.query_one("#vllm-check-setup", Button).display
        assert "Check draft" in str(view.query_one("#vllm-check-setup", Button).label)
        assert view.query_one("#vllm-stop", Button).display

        monkeypatch.setattr(
            "tldw_chatbook.UI.Screens.llm_screen.run_vllm_preflight",
            lambda candidate, generation: VllmPreflightResult(
                generation=generation,
                fingerprint=semantic_fingerprint(candidate),
                issues=(),
                cli_path=Path("/safe/vllm"),
            ),
        )
        await pilot.click("#vllm-check-setup")
        for _ in range(40):
            await pilot.pause()
            if screen._vllm_owner.snapshot().state is (
                VllmReadinessState.READY_TO_START
            ):
                break
        assert not view.query_one("#vllm-restart", Button).disabled

        launches = []
        monkeypatch.setattr(
            screen,
            "_start_vllm_process_workers",
            lambda command, claim, token, draft: launches.append(
                (command, claim, token, draft)
            ),
        )
        await pilot.click("#vllm-restart")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmationDialog)
        await pilot.click("#confirm-button")
        for _ in range(60):
            await pilot.pause()
            if launches:
                break

        assert not process.running
        assert launches
        new_claim, _ = server_lifecycle_snapshot(app, "vllm")
        assert new_claim is launches[0][1]
        assert new_claim is not old_claim
        assert launches[0][3].model_value == "org/next"
        assert release_server_claim(app, "vllm", new_claim)


async def test_restart_termination_failure_keeps_old_snapshot_and_never_reserves(
    monkeypatch,
):
    app = _build_test_app()
    process = _RunningProcess()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, _ = await _mount_vllm_screen(app, pilot)
        current = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="org/current",
        )
        old_token = screen._vllm_owner.begin(current, runtime_owner="chatbook")
        old_claim = reserve_server_launch(app, "vllm", authority="chatbook-vllm")
        assert old_claim is not None
        assert screen._vllm_owner.bind_launch_claim(old_token, old_claim)
        old_snapshot = screen._vllm_owner.bound_launch_snapshot(old_claim)
        assert old_snapshot is not None
        assert publish_server_process(app, "vllm", old_claim, process)

        draft = replace(current, model_value="/private/NEXT_MODEL_CANARY")
        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        screen._vllm_draft = draft
        screen._vllm_preflight = VllmPreflightResult(
            generation=token.generation,
            fingerprint=token.fingerprint,
            issues=(),
            cli_path=Path("/safe/vllm"),
        )
        screen._settle_vllm_state(
            token,
            VllmReadinessState.READY_TO_START,
            activity_code="checking",
        )
        reserve_calls = []

        async def stubborn_stop(*args, **kwargs):
            return False

        def forbidden_reserve(*args, **kwargs):
            reserve_calls.append(True)
            raise AssertionError("restart reserved a second process")

        monkeypatch.setattr(
            "tldw_chatbook.UI.Screens.llm_screen.stop_server_process", stubborn_stop
        )
        monkeypatch.setattr(
            "tldw_chatbook.UI.Screens.llm_screen.reserve_server_launch",
            forbidden_reserve,
        )

        assert not await screen._restart_vllm_with_draft(draft)
        assert reserve_calls == []
        assert server_lifecycle_snapshot(app, "vllm") == (old_claim, process)
        assert screen._vllm_owner.bound_launch_snapshot(old_claim) == old_snapshot
        assert screen._vllm_owner.snapshot().state is VllmReadinessState.NEEDS_ATTENTION


async def test_source_specific_controls_and_mode_drafts_are_preserved():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(VllmSetupView)
        view.apply_state(
            draft=view.draft,
            state=VllmReadinessState.NOT_CONFIGURED,
            preflight=None,
            profiles_ready=True,
        )
        hf_input = app.query_one("#vllm-hf-model", Input)
        local_input = app.query_one("#vllm-local-model-directory", Input)
        assert hf_input.display
        assert not local_input.display

        hf_input.value = "org/kept-model"
        await pilot.pause()
        await pilot.click("#vllm-connect-existing-button")
        assert app.query_one("#vllm-existing-server-url", Input).display
        await pilot.click("#vllm-start-local-button")
        assert app.query_one("#vllm-hf-model", Input).value == "org/kept-model"

        await pilot.click("#vllm-local-model-source-button")
        assert local_input.display
        assert not hf_input.display
        assert app.query_one(
            "#vllm-browse-local-model-directory-button", Button
        ).display


async def test_vllm_inputs_use_shared_lexical_caps_and_restore_rejected_events():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(VllmSetupView)
        view.apply_state(
            draft=view.draft,
            state=VllmReadinessState.NOT_CONFIGURED,
            preflight=None,
            profiles_ready=True,
        )
        port = view.query_one("#vllm-port", Input)
        port.value = "-"
        await pilot.pause()
        assert view.draft.port == "-"

        forged = type(
            "ForgedInputChanged",
            (),
            {"input": port, "value": "123456"},
        )()
        view._on_input_changed(forged)

        assert view.draft.port == "-"
        assert port.value == "-"
        help_label = view.query_one("#vllm-port-help", Label)
        assert help_label.display
        assert "unsupported control characters" in str(help_label.renderable)
        assert "123456" not in str(help_label.renderable)


@pytest.mark.parametrize(
    ("selector", "field", "lexeme"),
    (
        ("#vllm-port", "port", ""),
        ("#vllm-port", "port", "-"),
        ("#vllm-port", "port", "08000"),
        ("#vllm-tensor-parallel-size", "tensor_parallel_size", ""),
        ("#vllm-tensor-parallel-size", "tensor_parallel_size", "-"),
        ("#vllm-tensor-parallel-size", "tensor_parallel_size", "02"),
        ("#vllm-maximum-model-length", "maximum_model_length", ""),
        ("#vllm-maximum-model-length", "maximum_model_length", "1."),
        ("#vllm-maximum-model-length", "maximum_model_length", "08192"),
        ("#vllm-gpu-memory-utilization", "gpu_memory_utilization", ""),
        ("#vllm-gpu-memory-utilization", "gpu_memory_utilization", "."),
        ("#vllm-gpu-memory-utilization", "gpu_memory_utilization", "1."),
    ),
)
async def test_numeric_edits_preserve_exact_lexeme_and_invalidate_readiness(
    selector: str,
    field: str,
    lexeme: str,
):
    app = _VllmHost()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(VllmSetupView)
        view.apply_state(
            draft=replace(
                view.draft,
                model_value="org/model",
                tensor_parallel_size=4,
                maximum_model_length=8192,
                gpu_memory_utilization=0.75,
            ),
            state=VllmReadinessState.READY,
            preflight=None,
            profiles_ready=True,
        )
        control = view.query_one(selector, Input)

        control.value = lexeme
        await pilot.pause()

        assert control.value == lexeme
        assert getattr(view.draft, field) == lexeme
        assert view._state is VllmReadinessState.NOT_CONFIGURED
        assert view.preflight is None


async def test_numeric_action_boundary_normalizes_exact_values_before_messages():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(VllmSetupView)
        view.apply_state(
            draft=replace(view.draft, model_value="org/model"),
            state=VllmReadinessState.NOT_CONFIGURED,
            preflight=None,
            profiles_ready=True,
        )
        edits = {
            "#vllm-port": "08000",
            "#vllm-tensor-parallel-size": "02",
            "#vllm-maximum-model-length": "08192",
            "#vllm-gpu-memory-utilization": "1.",
        }
        for selector, lexeme in edits.items():
            view.query_one(selector, Input).value = lexeme
            await pilot.pause()

        await pilot.click("#vllm-check-setup")
        await pilot.pause()

        checked = app.action_events[-1].draft
        assert checked.port == 8000
        assert checked.tensor_parallel_size == 2
        assert checked.maximum_model_length == 8192
        assert checked.gpu_memory_utilization == 1.0
        assert view.draft == checked
        assert view.query_one("#vllm-port", Input).value == "8000"
        assert view.query_one("#vllm-tensor-parallel-size", Input).value == "2"
        assert view.query_one("#vllm-maximum-model-length", Input).value == "8192"
        assert view.query_one("#vllm-gpu-memory-utilization", Input).value == "1.0"

        view.query_one("#vllm-port", Input).value = "08000"
        await pilot.pause()
        await pilot.click("#vllm-profile-save-button")
        await pilot.pause()

        saved = app.profile_events[-1].draft
        assert saved.port == 8000
        assert type(saved.port) is int
        assert view.draft == saved

        view.query_one("#vllm-port", Input).value = "08000"
        await pilot.pause()
        start = view.query_one("#vllm-start", Button)
        start.disabled = False
        view._on_button_pressed(Button.Pressed(start))
        await pilot.pause()

        started = app.action_events[-1].draft
        assert isinstance(app.action_events[-1], VllmSetupView.StartRequested)
        assert started.port == 8000
        assert type(started.port) is int
        assert view.draft == started


async def test_invalid_numeric_action_stays_adjacent_and_posts_no_raw_draft():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(VllmSetupView)
        view.apply_state(
            draft=replace(view.draft, model_value="org/model"),
            state=VllmReadinessState.NOT_CONFIGURED,
            preflight=None,
            profiles_ready=True,
        )
        port = view.query_one("#vllm-port", Input)
        port.value = "-"
        await pilot.pause()

        await pilot.click("#vllm-check-setup")
        await pilot.pause()

        assert app.action_events == []
        assert view.draft.port == "-"
        assert port.value == "-"
        help_label = view.query_one("#vllm-port-help", Label)
        assert help_label.display
        assert "1 to 65535" in str(help_label.renderable)


async def test_programmatic_state_projection_resets_numeric_lexemes_from_draft():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(VllmSetupView)
        view.apply_state(
            draft=view.draft,
            state=VllmReadinessState.NOT_CONFIGURED,
            preflight=None,
            profiles_ready=True,
        )
        port = view.query_one("#vllm-port", Input)
        port.value = "08000"
        await pilot.pause()
        assert view.draft.port == "08000"

        hydrated = replace(
            view.draft,
            port=9000,
            tensor_parallel_size=4,
            maximum_model_length=16384,
            gpu_memory_utilization=0.75,
        )
        view.apply_state(
            draft=hydrated,
            state=VllmReadinessState.NOT_CONFIGURED,
            preflight=None,
            profiles_ready=True,
        )

        assert port.value == "9000"
        assert view.query_one("#vllm-tensor-parallel-size", Input).value == "4"
        assert view.query_one("#vllm-maximum-model-length", Input).value == "16384"
        assert view.query_one("#vllm-gpu-memory-utilization", Input).value == "0.75"


async def test_numeric_action_revalidates_forged_hydration_without_echo():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(VllmSetupView)
        canary = "123456_NUMERIC_CANARY"
        view.apply_state(
            draft=replace(view.draft, model_value="org/model", port=canary),
            state=VllmReadinessState.NOT_CONFIGURED,
            preflight=None,
            profiles_ready=True,
        )

        await pilot.click("#vllm-check-setup")
        await pilot.pause()

        assert app.action_events == []
        assert view.draft.port == canary
        help_label = view.query_one("#vllm-port-help", Label)
        assert help_label.display
        assert "unsupported control characters" in str(help_label.renderable)
        assert "NUMERIC_CANARY" not in str(help_label.renderable)


@pytest.mark.parametrize(
    "invalid_arguments",
    ("RAW_ARGUMENT_CANARY\x00", "RAW_ARGUMENT_CANARY" + "x" * (16 * 1024)),
)
async def test_vllm_raw_arguments_reject_nul_or_oversize_without_echo(
    invalid_arguments: str,
):
    app = _VllmHost()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(VllmSetupView)
        view.apply_state(
            draft=replace(view.draft, raw_arguments="--enable-prefix-caching"),
            state=VllmReadinessState.NOT_CONFIGURED,
            preflight=None,
            profiles_ready=True,
        )
        arguments = view.query_one("#vllm-raw-arguments", TextArea)
        arguments.text = invalid_arguments
        await pilot.pause()

        assert view.draft.raw_arguments == "--enable-prefix-caching"
        assert arguments.text == "--enable-prefix-caching"
        help_label = view.query_one("#vllm-raw-arguments-help", Label)
        assert help_label.display
        assert "16 KiB" in str(help_label.renderable)
        assert "RAW_ARGUMENT_CANARY" not in str(help_label.renderable)


async def test_vllm_raw_arguments_preserve_blank_and_nonblank_text_exactly():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(VllmSetupView)
        view.apply_state(
            draft=view.draft,
            state=VllmReadinessState.NOT_CONFIGURED,
            preflight=None,
            profiles_ready=True,
        )
        arguments = view.query_one("#vllm-raw-arguments", TextArea)
        arguments.text = "   "
        await pilot.pause()
        assert view.draft.raw_arguments == "   "

        arguments.text = "  --enable-prefix-caching  "
        await pilot.pause()
        assert view.draft.raw_arguments == "  --enable-prefix-caching  "


async def test_vllm_semantic_classes_and_input_caps_are_mounted():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)):
        view = app.query_one(VllmSetupView)
        expected_caps = {
            "vllm-profile-name": 120,
            "vllm-python-environment": 4096,
            "vllm-hf-model": 96,
            "vllm-local-model-directory": 4096,
            "vllm-bind-address": 255,
            "vllm-port": 5,
            "vllm-existing-server-url": 2048,
            "vllm-tensor-parallel-size": 10,
            "vllm-maximum-model-length": 10,
            "vllm-gpu-memory-utilization": 32,
        }
        for widget_id, limit in expected_caps.items():
            control = view.query_one(f"#{widget_id}", Input)
            assert control.max_length == limit
            assert control.has_class("vllm-control")
            assert control.has_class("vllm-focus-target")
        for button in view.query(Button):
            assert button.has_class("vllm-button")
            assert button.has_class("vllm-focus-target")


async def test_preflight_blocker_is_adjacent_and_start_enables_only_for_current_success():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(VllmSetupView)
        draft = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="org/model",
        )
        preflight = VllmPreflightResult(
            generation=1,
            fingerprint=semantic_fingerprint(draft),
            issues=(VllmIssue("python_unavailable", "python_environment"),),
        )
        view.apply_state(
            draft=draft,
            state=VllmReadinessState.NEEDS_ATTENTION,
            preflight=preflight,
            profiles_ready=True,
        )
        assert app.query_one("#vllm-start", Button).disabled
        assert str(
            app.query_one("#vllm-python-environment-help", Label).renderable
        ) == (
            "Python environment not found. Choose an available interpreter "
            "or virtual environment."
        )
        blocker = str(app.query_one("#vllm-start-blocker", Label).renderable)
        assert "Fix the highlighted setup field" in blocker
        assert "python_environment" not in blocker

        view.focus_state_action(VllmReadinessState.NEEDS_ATTENTION)
        await pilot.pause()
        assert app.focused.id == "vllm-recovery-primary"
        assert not app.query_one("#vllm-recovery-primary", Button).disabled

        view.apply_state(
            draft=draft,
            state=VllmReadinessState.NEEDS_ATTENTION,
            profiles_ready=True,
            preflight=VllmPreflightResult(
                generation=1,
                fingerprint=semantic_fingerprint(draft),
                issues=(VllmIssue("invalid_arguments", "raw_arguments"),),
            ),
        )
        assert not app.query_one("#vllm-advanced-options", Collapsible).collapsed
        assert app.query_one("#vllm-raw-arguments-help", Label).display


async def test_lifecycle_projection_enables_stop_only_while_runtime_is_active():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)):
        view = app.query_one(VllmSetupView)
        view.project_lifecycle(active=True)
        assert not app.query_one("#vllm-stop", Button).disabled
        view.project_lifecycle(active=False, status="process exited")
        assert app.query_one("#vllm-stop", Button).disabled


def _ready_result(token) -> VllmProbeResult:
    return VllmProbeResult(
        token=token,
        state=VllmReadinessState.READY,
        target=VllmConnectionTarget(
            provider_key="vllm",
            api_url="http://127.0.0.1:8000/v1/chat/completions",
            model_id="chatbook-vllm",
            runtime_owner=token.runtime_owner,
            generation=token.generation,
            credential_source="none",
        ),
        issue=None,
        activity=(VllmActivityEvent("ready", "under_1s"),),
    )


def _bind_local_claim(owner: VllmConnectionOwner, token) -> ServerLaunchClaim:
    claim = ServerLaunchClaim(provider="vllm", authority="chatbook-vllm")
    assert owner.bind_launch_claim(token, claim)
    return claim


async def test_child_view_requires_explicit_profile_hydration_for_ready_projection():
    """A newly mounted child cannot infer that app-scoped READY is reconciled."""

    app = _VllmHost()
    owner = VllmConnectionOwner()
    app._vllm_connection_owner = owner
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(VllmSetupView)
        draft = replace(
            view.draft,
            mode=VllmMode.EXISTING,
            existing_server_url="http://127.0.0.1:8000/v1",
            existing_model_id="chatbook-vllm",
        )
        token = owner.begin(draft, runtime_owner="external")
        assert owner.settle(token, _ready_result(token))

        view.apply_state(
            draft=draft,
            state=VllmReadinessState.READY,
            preflight=None,
            connection=owner.snapshot(),
        )
        await pilot.pause()

        assert str(view.query_one("#vllm-readiness-state", Label).renderable) == (
            "Setup incomplete"
        )
        checklist = " ".join(
            str(view.query_one(f"#vllm-check-{row}", Label).renderable)
            for row in ("environment", "installation", "model", "network")
        )
        assert "✓" not in checklist
        assert str(view.query_one("#vllm-activity-summary", Label).renderable) == (
            "No activity yet."
        )
        assert "API and model are ready" not in str(
            view.query_one("#vllm-activity-events", Label).renderable
        )
        assert not view.query_one("#vllm-use-console", Button).display
        assert not view.query_one("#vllm-make-default", Button).display

        view.apply_state(
            draft=draft,
            state=VllmReadinessState.READY,
            preflight=None,
            connection=owner.snapshot(),
            profiles_ready=True,
        )
        await pilot.pause()

        assert "Ready · Existing vLLM server" == str(
            view.query_one("#vllm-readiness-state", Label).renderable
        )
        assert "✓ Model · exact selection verified" in str(
            view.query_one("#vllm-check-model", Label).renderable
        )
        assert "✓ Network · API reachable" in str(
            view.query_one("#vllm-check-network", Label).renderable
        )
        assert "API and model are ready" in str(
            view.query_one("#vllm-activity-summary", Label).renderable
        )
        assert view.query_one("#vllm-use-console", Button).display
        assert view.query_one("#vllm-make-default", Button).display


async def test_mounted_activity_renders_ready_and_expands_bounded_failure():
    app = _VllmHost()
    async with app.run_test(size=(120, 40)):
        view = app.query_one(VllmSetupView)
        draft = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="/private/PATH_CANARY/bin/python",
            model_source=VllmModelSource.LOCAL_DIRECTORY,
            model_value="/private/MODEL_SOURCE_CANARY",
            raw_arguments="--flag COMMAND_CANARY",
        )
        owner = VllmConnectionOwner()
        token = owner.begin(draft, runtime_owner="chatbook")
        _bind_local_claim(owner, token)
        assert owner.settle(token, _ready_result(token))
        view.apply_state(
            draft=draft,
            state=VllmReadinessState.READY,
            preflight=None,
            connection=owner.snapshot(),
            profiles_ready=True,
        )
        assert "Ready at" in str(
            app.query_one("#vllm-readiness-state", Label).renderable
        )

        token = owner.begin(draft, runtime_owner="chatbook")
        failure = VllmProbeResult(
            token=token,
            state=VllmReadinessState.NEEDS_ATTENTION,
            target=None,
            issue=VllmIssue("model_missing", "model"),
            activity=(VllmActivityEvent("model_missing", "1_to_4s"),),
        )
        assert owner.settle(token, failure)
        view.apply_state(
            draft=draft,
            state=failure.state,
            preflight=None,
            connection=owner.snapshot(),
            profiles_ready=True,
        )
        assert not app.query_one("#vllm-activity-details", Collapsible).collapsed
        assert not app.query_one("#vllm-recovery-primary", Button).disabled
        visible = " ".join(str(label.renderable) for label in view.query(Label))
        assert "Expected chat model is unavailable" in visible
        assert not any(
            canary in visible
            for canary in ("PATH_CANARY", "MODEL_SOURCE_CANARY", "COMMAND_CANARY")
        )


async def _mount_vllm_screen(
    app, pilot
) -> tuple[LLMScreen, LLMManagementWindow, VllmSetupView]:
    screen = LLMScreen(app)
    await app.push_screen(screen)
    for _ in range(8):
        await pilot.pause()
    window = screen.query_one(LLMManagementWindow)
    window.active_view = "vllm"
    for _ in range(12):
        await pilot.pause()
        views = list(screen.query(VllmSetupView))
        if views:
            return screen, window, views[0]
    raise AssertionError("vLLM setup view did not mount")


async def test_fresh_screen_initial_hydration_preserves_exact_live_ready_target():
    """Loading the bound saved profile must not treat its placeholder as an edit."""

    app = _build_test_app()
    owner = VllmConnectionOwner()
    app._vllm_connection_owner = owner
    profile = default_vllm_profile()
    draft = draft_from_profile(profile)
    token = owner.begin(
        draft,
        runtime_owner="chatbook",
        profile_id=profile.profile_id,
        profile_name=profile.name,
    )
    claim = reserve_server_launch(app, "vllm", authority="chatbook-vllm")
    assert claim is not None
    assert owner.bind_launch_claim(token, claim)
    assert publish_server_process(app, "vllm", claim, _RunningProcess())
    assert owner.settle(token, _ready_result(token))

    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)

        assert screen._vllm_profiles_loaded
        snapshot = owner.snapshot()
        assert snapshot.current_token == token
        assert snapshot.state is VllmReadinessState.READY
        assert snapshot.target == _ready_result(token).target
        assert not view.query_one("#vllm-use-console", Button).disabled


async def test_navigation_to_fresh_models_screen_preserves_exact_ready_handoff(
    monkeypatch,
    tmp_path: Path,
):
    """A real Console handoff and fresh Models instance retain exact evidence."""

    from tldw_chatbook.Constants import TAB_CHAT, TAB_LLM
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

    repository = VllmProfileRepository(tmp_path / "profiles.json")
    block_next_load = threading.Event()
    second_load_entered = threading.Event()
    release_second_load = threading.Event()

    class _NavigationRepository:
        def load(self):
            if block_next_load.is_set():
                block_next_load.clear()
                second_load_entered.set()
                # TASK-31809: this wait keeps the fresh screen's profile load
                # pending until the test explicitly releases it, so the
                # pre-hydration "Setup incomplete" assertions below observe a
                # genuinely un-loaded screen. The timeout is only a failsafe
                # against a never-released hang -- it must comfortably exceed
                # the wall-clock cost of the pause-loops between block_next_load
                # and release_second_load. On a machine under concurrent test
                # load those loops were measured at ~5.6s, so the old 5s
                # failsafe fired early, unblocking the load, flipping
                # `_vllm_profiles_loaded` True, and rendering "Ready at ..."
                # where the test asserts "Setup incomplete".
                release_second_load.wait(60)
            return repository.load()

        def __getattr__(self, name):
            return getattr(repository, name)

    shared_repository = _NavigationRepository()
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.llm_screen.VllmProfileRepository",
        lambda: shared_repository,
    )
    app = _build_test_app()
    process = _RunningProcess()
    async with app.run_test(size=(235, 52)) as pilot:
        for _ in range(150):
            await pilot.pause(0.02)
            if getattr(app, "_initial_screen_pushed", False):
                break
        assert app._initial_screen_pushed
        await app.handle_screen_navigation(NavigateToScreen(TAB_LLM))
        first_screen = app.screen
        assert isinstance(first_screen, LLMScreen)
        for _ in range(100):
            await pilot.pause(0.02)
            if list(first_screen.query(LLMManagementWindow)):
                break
        first_window = first_screen.query_one(LLMManagementWindow)
        # TASK-31809: the vLLM pane mounts lazily as part of the window's
        # compose, and switching `active_view` before that pane exists makes
        # `watch_active_view` hit a QueryError that is only logged -- the
        # deferred VllmSetupView mount worker is never started, so the view
        # never appears and every later `query_one(VllmSetupView)` raises
        # NoMatches. Under concurrent test load the window's compose lags its
        # bare mount by enough for the old code to lose this race
        # intermittently. Wait for the pane before switching.
        for _ in range(_LAZY_MOUNT_POLL_ITERATIONS):
            if list(first_window.query("#llm-view-vllm")):
                break
            await pilot.pause()
        # Fail loud if the budget expired: switching active_view now would let
        # watch_active_view swallow the missing pane and the test would proceed
        # in a broken state, masking the very flake this guards.
        assert list(first_window.query("#llm-view-vllm")), (
            "vLLM pane #llm-view-vllm did not mount within "
            f"{_LAZY_MOUNT_POLL_ITERATIONS} poll iterations"
        )
        first_window.active_view = "vllm"
        # The loop's exit predicate and the post-loop assertion must be the
        # SAME condition. The old loop broke on
        # `_vllm_profiles_loaded and query(VllmSetupView)` but asserted only
        # `_vllm_profiles_loaded`, then called `query_one(VllmSetupView)` five
        # bare pauses later -- so a run whose budget expired with profiles
        # loaded but the view not yet mounted passed the assert and then
        # raised NoMatches. Wait for BOTH and assert BOTH before querying.
        for _ in range(_LAZY_MOUNT_POLL_ITERATIONS):
            await pilot.pause()
            if first_screen._vllm_profiles_loaded and list(
                first_screen.query(VllmSetupView)
            ):
                break
        assert first_screen._vllm_profiles_loaded
        assert list(first_screen.query(VllmSetupView))
        first_view = first_screen.query_one(VllmSetupView)
        profile = first_screen._selected_vllm_profile()
        draft = draft_from_profile(profile)
        token = first_screen._vllm_owner.begin(
            draft,
            runtime_owner="chatbook",
            profile_id=profile.profile_id,
            profile_name=profile.name,
        )
        claim = reserve_server_launch(app, "vllm", authority="chatbook-vllm")
        assert claim is not None
        assert first_screen._vllm_owner.bind_launch_claim(token, claim)
        assert publish_server_process(app, "vllm", claim, process)
        assert first_screen._vllm_owner.settle(token, _ready_result(token))
        first_screen._vllm_draft = draft
        first_screen._apply_vllm_view_state(focus=False)
        await pilot.pause()
        assert not first_view.query_one("#vllm-use-console", Button).disabled

        await pilot.click("#vllm-use-console")
        for _ in range(200):
            await pilot.pause(0.02)
            if (
                app.current_tab == TAB_CHAT
                and type(app.screen).__name__ == "ChatScreen"
            ):
                if not app.pending_handoffs.has_pending(HandoffChannel.VLLM_CONSOLE):
                    break
        assert app.current_tab == TAB_CHAT
        assert type(app.screen).__name__ == "ChatScreen"
        assert not app.pending_handoffs.has_pending(HandoffChannel.VLLM_CONSOLE)
        assert app.screen.current_console_provider_for_command() == "vllm"

        # TASK-31809 (Qodo): the fresh-screen profile load blocks a worker
        # thread on release_second_load until the mid-test set() below. If any
        # assertion here raised before that set(), the worker would hang on the
        # 60s failsafe -- so guarantee the release on every exit path.
        try:
            block_next_load.set()
            await app.handle_screen_navigation(NavigateToScreen(TAB_LLM))
            fresh_screen = app.screen
            assert isinstance(fresh_screen, LLMScreen)
            assert fresh_screen is not first_screen
            for _ in range(100):
                await pilot.pause(0.02)
                if list(fresh_screen.query(LLMManagementWindow)):
                    break
            fresh_window = fresh_screen.query_one(LLMManagementWindow)
            # TASK-31809: mirror the first-visit fix -- wait for the lazily
            # composed vLLM pane before switching `active_view`, or the deferred
            # mount worker never starts and the view never appears.
            for _ in range(_LAZY_MOUNT_POLL_ITERATIONS):
                if list(fresh_window.query("#llm-view-vllm")):
                    break
                await pilot.pause()
            assert list(fresh_window.query("#llm-view-vllm")), (
                "vLLM pane #llm-view-vllm did not mount within "
                f"{_LAZY_MOUNT_POLL_ITERATIONS} poll iterations"
            )
            fresh_window.active_view = "vllm"
            # The load stays pending here (see `_NavigationRepository.load`), so a
            # generous budget cannot flip `_vllm_profiles_loaded` -- it only gives
            # the mount time to land. Wait for the view, then assert it is present
            # before querying it.
            for _ in range(_LAZY_MOUNT_POLL_ITERATIONS):
                await pilot.pause()
                if second_load_entered.is_set() and list(fresh_screen.query(VllmSetupView)):
                    break
            assert second_load_entered.is_set()
            assert not fresh_screen._vllm_profiles_loaded
            assert list(fresh_screen.query(VllmSetupView))
            fresh_view = fresh_screen.query_one(VllmSetupView)
            for _ in range(5):
                await pilot.pause()
            assert (
                str(fresh_view.query_one("#vllm-readiness-state", Label).renderable)
                == "Setup incomplete"
            )
            assert "✓" not in " ".join(
                str(fresh_view.query_one(f"#vllm-check-{row}", Label).renderable)
                for row in ("environment", "installation", "model", "network")
            )
            assert (
                str(fresh_view.query_one("#vllm-activity-summary", Label).renderable)
                == "No activity yet."
            )
            assert "API and model are ready" not in str(
                fresh_view.query_one("#vllm-activity-events", Label).renderable
            )
            fresh_view.project_lifecycle(active=True)
            await pilot.pause()
            assert (
                str(fresh_view.query_one("#vllm-readiness-state", Label).renderable)
                == "Setup incomplete"
            )
            assert "✓" not in " ".join(
                str(fresh_view.query_one(f"#vllm-check-{row}", Label).renderable)
                for row in ("environment", "installation", "model", "network")
            )
            assert (
                str(fresh_view.query_one("#vllm-activity-summary", Label).renderable)
                == "No activity yet."
            )
            assert "API and model are ready" not in str(
                fresh_view.query_one("#vllm-activity-events", Label).renderable
            )
            assert not fresh_view.query_one("#vllm-use-console", Button).display
            assert fresh_view.query_one("#vllm-use-console", Button).disabled
            assert not fresh_view.query_one("#vllm-make-default", Button).display
            assert not fresh_view.query_one("#vllm-recovery-primary", Button).display
            assert not fresh_view.query_one("#vllm-stop", Button).disabled
            assert "Next: Stop vLLM" in str(
                fresh_screen.query_one("#lab-status-chip-model-install", Static).renderable
            )

            from tldw_chatbook.UI.Navigation.vllm_handoff import VllmConsoleIntent

            original_post_message = fresh_screen.post_message
            monkeypatch.setattr(fresh_screen, "post_message", lambda *_args: True)
            staged_before_hydration = fresh_screen._stage_vllm_handoff(
                channel=HandoffChannel.VLLM_CONSOLE,
                intent_type=VllmConsoleIntent,
                route=TAB_CHAT,
            )
            monkeypatch.setattr(fresh_screen, "post_message", original_post_message)
            assert not staged_before_hydration
            assert not app.pending_handoffs.has_pending(HandoffChannel.VLLM_CONSOLE)
            focus_target = app.focused
            assert focus_target is not None

            release_second_load.set()
            for _ in range(100):
                await pilot.pause(0.02)
                if fresh_screen._vllm_profiles_loaded:
                    break
            assert fresh_screen._vllm_profiles_loaded
            assert app.focused is focus_target

            for _ in range(5):
                await pilot.pause()
            snapshot = fresh_screen._vllm_owner.snapshot()
            assert snapshot.current_token == token
            assert snapshot.state is VllmReadinessState.READY
            assert snapshot.target == _ready_result(token).target
            assert "Ready at http://127.0.0.1:8000/v1" in str(
                fresh_view.query_one("#vllm-readiness-state", Label).renderable
            )
            assert "✓ Model · exact selection verified" in str(
                fresh_view.query_one("#vllm-check-model", Label).renderable
            )
            assert "✓ Network · API reachable" in str(
                fresh_view.query_one("#vllm-check-network", Label).renderable
            )
            assert "API and model are ready" in str(
                fresh_view.query_one("#vllm-activity-summary", Label).renderable
            )
            assert fresh_view.query_one("#vllm-use-console", Button).display
            assert not fresh_view.query_one("#vllm-use-console", Button).disabled
            assert fresh_view.query_one("#vllm-make-default", Button).display
            assert not fresh_view.query_one("#vllm-make-default", Button).disabled
            assert not fresh_view.query_one("#vllm-stop", Button).disabled
            assert not fresh_view.query_one("#vllm-recovery-primary", Button).display
        finally:
            release_second_load.set()


async def test_delayed_profile_hydration_interaction_fence_preserves_exact_ready(
    monkeypatch,
    tmp_path: Path,
):
    """Pending profile reconciliation rejects mounted and forged draft actions."""

    repository = VllmProfileRepository(tmp_path / "profiles.json")
    load_entered = threading.Event()
    release_load = threading.Event()

    class _DelayedRepository:
        def load(self):
            load_entered.set()
            release_load.wait(5)
            return repository.load()

        def __getattr__(self, name):
            return getattr(repository, name)

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.llm_screen.VllmProfileRepository",
        _DelayedRepository,
    )

    async def _no_network_preflight(self, token, draft):
        return None

    monkeypatch.setattr(
        LLMScreen,
        "_run_vllm_preflight_generation",
        _no_network_preflight,
    )
    app = _build_test_app()
    owner = VllmConnectionOwner()
    app._vllm_connection_owner = owner
    profile = default_vllm_profile()
    draft = draft_from_profile(profile)
    token = owner.begin(
        draft,
        runtime_owner="chatbook",
        profile_id=profile.profile_id,
        profile_name=profile.name,
    )
    claim = reserve_server_launch(app, "vllm", authority="chatbook-vllm")
    assert claim is not None
    assert owner.bind_launch_claim(token, claim)
    assert publish_server_process(app, "vllm", claim, _RunningProcess())
    expected_ready = _ready_result(token)
    assert owner.settle(token, expected_ready)

    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        assert load_entered.wait(1)
        assert not screen._vllm_profiles_loaded
        before = owner.snapshot()
        before_draft = screen._vllm_draft
        profile_worker = screen._vllm_profile_worker
        mutation_control_ids = (
            "vllm-check-setup",
            "vllm-start",
            "vllm-restart",
            "vllm-start-local-button",
            "vllm-connect-existing-button",
            "vllm-profile-select",
            "vllm-profile-name",
            "vllm-profile-create-button",
            "vllm-profile-save-button",
            "vllm-profile-rename-button",
            "vllm-profile-duplicate-button",
            "vllm-profile-delete-button",
            "vllm-python-environment",
            "vllm-browse-python-environment",
            "vllm-hugging-face-source-button",
            "vllm-local-model-source-button",
            "vllm-hf-model",
            "vllm-local-model-directory",
            "vllm-browse-local-model-directory-button",
            "vllm-bind-address",
            "vllm-port",
            "vllm-existing-server-url",
            "vllm-existing-model",
            "vllm-dtype",
            "vllm-tensor-parallel-size",
            "vllm-maximum-model-length",
            "vllm-gpu-memory-utilization",
            "vllm-trust-remote-code",
            "vllm-raw-arguments",
        )
        enabled_before_hydration = tuple(
            control_id
            for control_id in mutation_control_ids
            if not view.query_one(f"#{control_id}").disabled
        )
        forged_draft = replace(before_draft, port=before_draft.port + 1)
        try:
            await pilot.click("#vllm-connect-existing-button")
            view.query_one("#vllm-check-setup", Button).press()
            view.query_one(
                "#vllm-python-environment", Input
            ).value = "/private/HYDRATION_RACE/bin/python"
            view.query_one("#vllm-raw-arguments", TextArea).text = "--HYDRATION_RACE"
            screen._on_vllm_draft_changed(VllmSetupView.DraftChanged(forged_draft))
            screen._on_vllm_check_requested(VllmSetupView.CheckRequested(forged_draft))
            screen._on_vllm_retry_requested(VllmSetupView.RetryRequested())
            screen._on_vllm_start_requested(VllmSetupView.StartRequested(forged_draft))
            screen._on_vllm_restart_requested(
                VllmSetupView.RestartRequested(forged_draft, ("Port",))
            )
            await pilot.pause()
            during = owner.snapshot()
            draft_during = screen._vllm_draft
            worker_during = screen._vllm_profile_worker
        finally:
            release_load.set()

        assert enabled_before_hydration == ()
        assert not view.query_one("#vllm-stop", Button).disabled
        assert during.current_token == before.current_token == token
        assert during.state is before.state is VllmReadinessState.READY
        assert during.target == before.target == expected_ready.target
        assert draft_during == before_draft
        assert worker_during is profile_worker

        for _ in range(100):
            await pilot.pause(0.02)
            if screen._vllm_profiles_loaded:
                break
        assert screen._vllm_profiles_loaded
        after = owner.snapshot()
        assert after.current_token == token
        assert after.state is VllmReadinessState.READY
        assert after.target == expected_ready.target
        assert not view.query_one("#vllm-use-console", Button).disabled


async def test_fresh_screen_profile_load_failure_invalidates_ready_with_recovery(
    monkeypatch,
):
    """An unreadable profile store cannot leave inherited READY actions usable."""

    load_entered = threading.Event()
    release_load = threading.Event()

    class _FailingRepository:
        def load(self):
            load_entered.set()
            release_load.wait(5)
            raise VllmProfileCorrupt("profile document is unavailable")

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.llm_screen.VllmProfileRepository",
        _FailingRepository,
    )
    app = _build_test_app()
    owner = VllmConnectionOwner()
    app._vllm_connection_owner = owner
    profile = default_vllm_profile()
    draft = draft_from_profile(profile)
    token = owner.begin(
        draft,
        runtime_owner="chatbook",
        profile_id=profile.profile_id,
        profile_name=profile.name,
    )
    claim = reserve_server_launch(app, "vllm", authority="chatbook-vllm")
    assert claim is not None
    assert owner.bind_launch_claim(token, claim)
    assert publish_server_process(app, "vllm", claim, _RunningProcess())
    assert owner.settle(token, _ready_result(token))

    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        assert load_entered.wait(1)
        focused = view.query_one("#vllm-stop", Button)
        focused.focus()
        await pilot.pause()
        assert app.focused is focused

        release_load.set()
        for _ in range(100):
            await pilot.pause(0.02)
            worker = screen._vllm_profile_worker
            if worker is not None and worker.is_finished:
                break

        snapshot = owner.snapshot()
        assert snapshot.state is VllmReadinessState.NOT_CONFIGURED
        assert snapshot.target is None
        assert not screen._vllm_profiles_loaded
        assert not view.query_one("#vllm-use-console", Button).display
        assert view.query_one("#vllm-use-console", Button).disabled
        assert not view.query_one("#vllm-stop", Button).disabled
        assert view.query_one("#vllm-profile-select", Select).disabled
        profile_help = view.query_one("#vllm-profile-help", Label)
        assert profile_help.display
        assert "repair or reload" in str(profile_help.renderable)
        assert app.focused is focused


@pytest.mark.parametrize("liveness", ("cancelled", "dead", "poll_exception"))
async def test_staged_owned_handoff_is_discarded_without_positive_liveness(
    monkeypatch,
    liveness: str,
):
    """Unmount cannot preserve a pending owned target on uncertain evidence."""

    from tldw_chatbook.Constants import TAB_CHAT
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
    from tldw_chatbook.UI.Navigation.vllm_handoff import (
        VllmConsoleIntent,
        VllmDefaultIntent,
    )

    class _ControllableProcess(_RunningProcess):
        poll_raises = False

        def poll(self):
            if self.poll_raises:
                raise RuntimeError("liveness unavailable")
            return super().poll()

    app = _build_test_app()
    process = _ControllableProcess()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, _ = await _mount_vllm_screen(app, pilot)
        for _ in range(50):
            await pilot.pause(0.02)
            if screen._vllm_profiles_loaded:
                break
        assert screen._vllm_profiles_loaded
        profile = screen._selected_vllm_profile()
        external_draft = replace(
            screen._vllm_draft,
            mode=VllmMode.EXISTING,
            existing_server_url="http://127.0.0.1:8000/v1",
            existing_model_id="chatbook-vllm",
        )
        external_token = screen._vllm_owner.begin(
            external_draft,
            runtime_owner="external",
        )
        assert screen._vllm_owner.settle(
            external_token,
            _ready_result(external_token),
        )
        screen._vllm_draft = external_draft
        original_post_message = screen.post_message
        monkeypatch.setattr(screen, "post_message", lambda *_args: True)
        assert screen._stage_vllm_handoff(
            channel=HandoffChannel.VLLM_DEFAULT,
            intent_type=VllmDefaultIntent,
            route="settings",
        )
        monkeypatch.setattr(screen, "post_message", original_post_message)
        assert app.pending_handoffs.has_pending(HandoffChannel.VLLM_DEFAULT)

        draft = draft_from_profile(profile)
        token = screen._vllm_owner.begin(
            draft,
            runtime_owner="chatbook",
            profile_id=profile.profile_id,
            profile_name=profile.name,
        )
        claim = reserve_server_launch(app, "vllm", authority="chatbook-vllm")
        assert claim is not None
        assert screen._vllm_owner.bind_launch_claim(token, claim)
        assert publish_server_process(app, "vllm", claim, process)
        assert screen._vllm_owner.settle(token, _ready_result(token))
        screen._vllm_draft = draft
        screen._apply_vllm_view_state(focus=False)

        monkeypatch.setattr(screen, "post_message", lambda *_args: True)
        staged = screen._stage_vllm_handoff(
            channel=HandoffChannel.VLLM_CONSOLE,
            intent_type=VllmConsoleIntent,
            route=TAB_CHAT,
        )
        monkeypatch.setattr(screen, "post_message", original_post_message)
        assert staged
        assert app.pending_handoffs.has_pending(HandoffChannel.VLLM_CONSOLE)

        if liveness == "cancelled":
            claim.cancel_event.set()
        elif liveness == "dead":
            process.running = False
        else:
            process.poll_raises = True

        screen.on_unmount()
        assert app.pending_handoffs.claim(HandoffChannel.VLLM_CONSOLE) is None
        assert app.pending_handoffs.claim(HandoffChannel.VLLM_DEFAULT) is None
        snapshot = screen._vllm_owner.snapshot()
        assert snapshot.state is VllmReadinessState.NOT_CONFIGURED
        assert snapshot.target is None

        process.poll_raises = False
        process.running = False
        assert clear_server_process(app, "vllm", claim, process)


@pytest.mark.parametrize("receipt_state", ("pending", "in_flight"))
async def test_exact_external_handoff_preserves_ready_departure(
    monkeypatch,
    receipt_state: str,
):
    """Only the exact current external transfer may preserve READY on departure."""

    from tldw_chatbook.Constants import TAB_CHAT
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
    from tldw_chatbook.UI.Navigation.vllm_handoff import VllmConsoleIntent

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, _ = await _mount_vllm_screen(app, pilot)
        for _ in range(50):
            await pilot.pause(0.02)
            if screen._vllm_profiles_loaded:
                break
        assert screen._vllm_profiles_loaded
        draft = replace(
            screen._vllm_draft,
            mode=VllmMode.EXISTING,
            existing_server_url="http://127.0.0.1:8000/v1",
            existing_model_id="chatbook-vllm",
        )
        token = screen._vllm_owner.begin(draft, runtime_owner="external")
        assert screen._vllm_owner.settle(token, _ready_result(token))
        screen._vllm_draft = draft

        original_post_message = screen.post_message
        monkeypatch.setattr(screen, "post_message", lambda *_args: True)
        assert screen._stage_vllm_handoff(
            channel=HandoffChannel.VLLM_CONSOLE,
            intent_type=VllmConsoleIntent,
            route=TAB_CHAT,
        )
        monkeypatch.setattr(screen, "post_message", original_post_message)
        claim = (
            app.pending_handoffs.claim(HandoffChannel.VLLM_CONSOLE)
            if receipt_state == "in_flight"
            else None
        )

        screen.on_unmount()

        if claim is None:
            claim = app.pending_handoffs.claim(HandoffChannel.VLLM_CONSOLE)
        assert claim is not None
        assert (
            claim.revision
            == screen._vllm_staged_handoffs[HandoffChannel.VLLM_CONSOLE][0]
        )
        assert claim.value == VllmConsoleIntent.from_target(_ready_result(token).target)
        assert screen._vllm_owner.snapshot().target == _ready_result(token).target
        assert app.pending_handoffs.acknowledge(claim)


async def test_superseded_external_handoff_cannot_preserve_ready_departure(monkeypatch):
    """A newer unrelated value on the same channel does not own this departure."""

    from tldw_chatbook.Constants import TAB_CHAT
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
    from tldw_chatbook.UI.Navigation.vllm_handoff import VllmConsoleIntent

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, _ = await _mount_vllm_screen(app, pilot)
        for _ in range(50):
            await pilot.pause(0.02)
            if screen._vllm_profiles_loaded:
                break
        assert screen._vllm_profiles_loaded
        draft = replace(
            screen._vllm_draft,
            mode=VllmMode.EXISTING,
            existing_server_url="http://127.0.0.1:8000/v1",
            existing_model_id="chatbook-vllm",
        )
        token = screen._vllm_owner.begin(draft, runtime_owner="external")
        assert screen._vllm_owner.settle(token, _ready_result(token))
        screen._vllm_draft = draft

        original_post_message = screen.post_message
        monkeypatch.setattr(screen, "post_message", lambda *_args: True)
        assert screen._stage_vllm_handoff(
            channel=HandoffChannel.VLLM_CONSOLE,
            intent_type=VllmConsoleIntent,
            route=TAB_CHAT,
        )
        monkeypatch.setattr(screen, "post_message", original_post_message)
        unrelated = VllmConsoleIntent(
            api_url="http://127.0.0.1:9000/v1/chat/completions",
            model_id="org/unrelated",
            generation=token.generation,
        )
        app.pending_handoffs.stage(HandoffChannel.VLLM_CONSOLE, unrelated)

        screen.on_unmount()

        snapshot = screen._vllm_owner.snapshot()
        assert snapshot.state is VllmReadinessState.NOT_CONFIGURED
        assert snapshot.target is None
        unrelated_claim = app.pending_handoffs.claim(HandoffChannel.VLLM_CONSOLE)
        assert unrelated_claim is not None
        assert unrelated_claim.value == unrelated
        assert app.pending_handoffs.acknowledge(unrelated_claim)


async def test_external_handoff_revision_lookup_error_fails_closed(monkeypatch):
    """An indeterminate exact-revision lookup cannot preserve external READY."""

    from tldw_chatbook.Constants import TAB_CHAT
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
    from tldw_chatbook.UI.Navigation.vllm_handoff import VllmConsoleIntent

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, _ = await _mount_vllm_screen(app, pilot)
        for _ in range(50):
            await pilot.pause(0.02)
            if screen._vllm_profiles_loaded:
                break
        assert screen._vllm_profiles_loaded
        draft = replace(
            screen._vllm_draft,
            mode=VllmMode.EXISTING,
            existing_server_url="http://127.0.0.1:8000/v1",
            existing_model_id="chatbook-vllm",
        )
        token = screen._vllm_owner.begin(draft, runtime_owner="external")
        assert screen._vllm_owner.settle(token, _ready_result(token))
        screen._vllm_draft = draft

        original_post_message = screen.post_message
        monkeypatch.setattr(screen, "post_message", lambda *_args: True)
        assert screen._stage_vllm_handoff(
            channel=HandoffChannel.VLLM_CONSOLE,
            intent_type=VllmConsoleIntent,
            route=TAB_CHAT,
        )
        monkeypatch.setattr(screen, "post_message", original_post_message)

        def fail_lookup(*_args):
            raise RuntimeError("revision lookup unavailable")

        monkeypatch.setattr(
            app.pending_handoffs,
            "exact_revision_status",
            fail_lookup,
        )
        screen.on_unmount()

        snapshot = screen._vllm_owner.snapshot()
        assert snapshot.state is VllmReadinessState.NOT_CONFIGURED
        assert snapshot.target is None
        assert app.pending_handoffs.claim(HandoffChannel.VLLM_CONSOLE) is None


@pytest.mark.parametrize(
    "receipt_case",
    ("mixed_stale_valid", "lookup_error_valid", "no_valid"),
)
async def test_external_departure_receipts_are_validated_independently(
    monkeypatch,
    receipt_case: str,
):
    """Each external receipt fails closed without discarding an exact peer."""

    from tldw_chatbook.Constants import TAB_CHAT, TAB_SETTINGS
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
    from tldw_chatbook.UI.Navigation.vllm_handoff import (
        VllmConsoleIntent,
        VllmDefaultIntent,
    )

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, _ = await _mount_vllm_screen(app, pilot)
        for _ in range(50):
            await pilot.pause(0.02)
            if screen._vllm_profiles_loaded:
                break
        assert screen._vllm_profiles_loaded
        draft = replace(
            screen._vllm_draft,
            mode=VllmMode.EXISTING,
            existing_server_url="http://127.0.0.1:8000/v1",
            existing_model_id="chatbook-vllm",
        )
        token = screen._vllm_owner.begin(draft, runtime_owner="external")
        expected_ready = _ready_result(token)
        assert screen._vllm_owner.settle(token, expected_ready)
        screen._vllm_draft = draft

        original_post_message = screen.post_message
        monkeypatch.setattr(screen, "post_message", lambda *_args: True)
        assert screen._stage_vllm_handoff(
            channel=HandoffChannel.VLLM_DEFAULT,
            intent_type=VllmDefaultIntent,
            route=TAB_SETTINGS,
        )
        assert screen._stage_vllm_handoff(
            channel=HandoffChannel.VLLM_CONSOLE,
            intent_type=VllmConsoleIntent,
            route=TAB_CHAT,
        )
        monkeypatch.setattr(screen, "post_message", original_post_message)

        if receipt_case == "mixed_stale_valid":
            unrelated_default = VllmDefaultIntent(
                api_url="http://127.0.0.1:9000/v1/chat/completions",
                model_id="org/unrelated",
                generation=token.generation,
            )
            app.pending_handoffs.stage(
                HandoffChannel.VLLM_DEFAULT,
                unrelated_default,
            )
        elif receipt_case == "lookup_error_valid":
            exact_revision_status = app.pending_handoffs.exact_revision_status

            def fail_one_lookup(channel, revision):
                if channel is HandoffChannel.VLLM_DEFAULT:
                    raise RuntimeError("default receipt lookup unavailable")
                return exact_revision_status(channel, revision)

            monkeypatch.setattr(
                app.pending_handoffs,
                "exact_revision_status",
                fail_one_lookup,
            )
        else:
            for channel in (
                HandoffChannel.VLLM_DEFAULT,
                HandoffChannel.VLLM_CONSOLE,
            ):
                claim = app.pending_handoffs.claim(channel)
                assert claim is not None
                assert app.pending_handoffs.acknowledge(claim)

        screen.on_unmount()

        snapshot = screen._vllm_owner.snapshot()
        if receipt_case == "no_valid":
            assert snapshot.state is VllmReadinessState.NOT_CONFIGURED
            assert snapshot.target is None
            assert screen._vllm_staged_handoffs == {}
            assert app.pending_handoffs.claim(HandoffChannel.VLLM_DEFAULT) is None
            assert app.pending_handoffs.claim(HandoffChannel.VLLM_CONSOLE) is None
            return

        assert snapshot.state is VllmReadinessState.READY
        assert snapshot.target == expected_ready.target
        assert HandoffChannel.VLLM_DEFAULT not in screen._vllm_staged_handoffs
        assert HandoffChannel.VLLM_CONSOLE in screen._vllm_staged_handoffs
        console_claim = app.pending_handoffs.claim(HandoffChannel.VLLM_CONSOLE)
        assert console_claim is not None
        assert console_claim.value == VllmConsoleIntent.from_target(
            expected_ready.target
        )
        assert app.pending_handoffs.acknowledge(console_claim)
        if receipt_case == "mixed_stale_valid":
            default_claim = app.pending_handoffs.claim(HandoffChannel.VLLM_DEFAULT)
            assert default_claim is not None
            assert default_claim.value == unrelated_default
            assert app.pending_handoffs.acknowledge(default_claim)
        else:
            assert app.pending_handoffs.claim(HandoffChannel.VLLM_DEFAULT) is None


async def test_staged_owned_handoff_survives_exact_live_unmount_for_consumption(
    monkeypatch,
):
    """The exact uncancelled live claim remains consumable after departure."""

    from tldw_chatbook.Constants import TAB_CHAT
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
    from tldw_chatbook.UI.Navigation.vllm_handoff import VllmConsoleIntent

    app = _build_test_app()
    process = _RunningProcess()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, _ = await _mount_vllm_screen(app, pilot)
        for _ in range(50):
            await pilot.pause(0.02)
            if screen._vllm_profiles_loaded:
                break
        assert screen._vllm_profiles_loaded
        profile = screen._selected_vllm_profile()
        draft = draft_from_profile(profile)
        token = screen._vllm_owner.begin(
            draft,
            runtime_owner="chatbook",
            profile_id=profile.profile_id,
            profile_name=profile.name,
        )
        claim = reserve_server_launch(app, "vllm", authority="chatbook-vllm")
        assert claim is not None
        assert screen._vllm_owner.bind_launch_claim(token, claim)
        assert publish_server_process(app, "vllm", claim, process)
        assert screen._vllm_owner.settle(token, _ready_result(token))
        screen._vllm_draft = draft

        original_post_message = screen.post_message
        monkeypatch.setattr(screen, "post_message", lambda *_args: True)
        staged = screen._stage_vllm_handoff(
            channel=HandoffChannel.VLLM_CONSOLE,
            intent_type=VllmConsoleIntent,
            route=TAB_CHAT,
        )
        monkeypatch.setattr(screen, "post_message", original_post_message)
        assert staged

        screen.on_unmount()
        pending_claim = app.pending_handoffs.claim(HandoffChannel.VLLM_CONSOLE)
        assert pending_claim is not None
        assert pending_claim.value.generation == token.generation
        assert screen._vllm_owner.snapshot().target == _ready_result(token).target
        assert app.pending_handoffs.acknowledge(pending_claim)

        process.running = False
        assert clear_server_process(app, "vllm", claim, process)


async def test_fresh_screen_mismatched_profile_invalidates_ready_target_safely(
    monkeypatch,
    tmp_path: Path,
):
    """A different restored profile never inherits another launch's READY proof."""

    repository = VllmProfileRepository(tmp_path / "profiles.json")
    launched_profile = default_vllm_profile()
    launched_draft = draft_from_profile(launched_profile)
    mismatched = repository.save(
        profile_from_draft(
            "Different launch",
            replace(launched_draft, model_value="org/different-model"),
        ),
        expected_revision=0,
    )
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.llm_screen.VllmProfileRepository",
        lambda: repository,
    )
    app = _build_test_app()
    owner = VllmConnectionOwner()
    app._vllm_connection_owner = owner
    token = owner.begin(
        launched_draft,
        runtime_owner="chatbook",
        profile_id=launched_profile.profile_id,
        profile_name=launched_profile.name,
    )
    claim = reserve_server_launch(app, "vllm", authority="chatbook-vllm")
    assert claim is not None
    assert owner.bind_launch_claim(token, claim)
    assert publish_server_process(app, "vllm", claim, _RunningProcess())
    assert owner.settle(token, _ready_result(token))

    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)

        assert screen._vllm_profiles == mismatched.document
        snapshot = owner.snapshot()
        assert snapshot.current_token is not None
        assert snapshot.current_token.generation > token.generation
        assert snapshot.state is VllmReadinessState.NOT_CONFIGURED
        assert snapshot.target is None
        assert view.query_one("#vllm-use-console", Button).disabled
        assert not view.query_one("#vllm-use-console", Button).display
        assert not view.query_one("#vllm-stop", Button).disabled
        recovery_actions = (
            view.query_one("#vllm-check-setup", Button),
            view.query_one("#vllm-recovery-primary", Button),
        )
        assert any(
            action.display and not action.disabled for action in recovery_actions
        )


async def test_mounted_draft_edit_fences_old_readiness_generation():
    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        assert app._vllm_connection_owner is screen._vllm_owner
        draft = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="org/model",
        )
        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        claim = _bind_local_claim(screen._vllm_owner, token)
        assert screen._vllm_owner.settle(token, _ready_result(token))
        screen._vllm_draft = draft
        screen._apply_vllm_view_state()

        view.query_one("#vllm-hf-model", Input).value = "org/changed-model"
        await pilot.pause()

        snapshot = screen._vllm_owner.snapshot()
        assert snapshot.generation == token.generation + 1
        assert snapshot.target is None
        assert snapshot.activity[-1].code == "target_changed"
        assert screen._vllm_owner.settle(token, _ready_result(token)) is False

        current_draft = screen._vllm_draft
        token = screen._vllm_owner.begin(current_draft, runtime_owner="chatbook")
        assert screen._vllm_owner.bind_launch_claim(token, claim)
        assert screen._vllm_owner.settle(token, _ready_result(token))
        screen._apply_vllm_view_state()
        view.query_one("#vllm-raw-arguments", TextArea).text = "--enable-prefix-caching"
        await pilot.pause()
        raw_edit = screen._vllm_owner.snapshot()
        assert raw_edit.generation == token.generation + 1
        assert raw_edit.activity[-1].code == "target_changed"


async def test_mounted_external_selection_starts_fresh_exact_probe(monkeypatch):
    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        draft = VllmLaunchDraft(
            mode=VllmMode.EXISTING,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="",
            existing_server_url="http://127.0.0.1:8000/v1",
        )
        token = screen._vllm_owner.begin(draft, runtime_owner="external")
        screen._vllm_draft = draft
        requests = []

        async def fake_probe(request):
            requests.append(request)
            if request.expected_model_id is None:
                return VllmProbeResult(
                    token=request.token,
                    state=VllmReadinessState.NOT_CONFIGURED,
                    target=None,
                    issue=None,
                    activity=(VllmActivityEvent("models_discovered", "under_1s"),),
                    discovered_model_ids=("org/first", "org/second"),
                )
            return VllmProbeResult(
                token=request.token,
                state=VllmReadinessState.READY,
                target=VllmConnectionTarget(
                    provider_key="vllm",
                    api_url="http://127.0.0.1:8000/v1/chat/completions",
                    model_id=request.expected_model_id,
                    runtime_owner="external",
                    generation=request.token.generation,
                    credential_source="none",
                ),
                issue=None,
                activity=(VllmActivityEvent("ready", "under_1s"),),
                discovered_model_ids=(request.expected_model_id,),
            )

        monkeypatch.setattr(
            "tldw_chatbook.UI.Screens.llm_screen.probe_vllm_target", fake_probe
        )
        await screen._probe_vllm_generation(token, draft, None)
        assert screen._vllm_owner.snapshot().target is None
        selector = view.query_one("#vllm-existing-model", Select)
        assert selector.value is Select.NULL

        selector.value = "org/second"
        for _ in range(20):
            await pilot.pause()
            if screen._vllm_owner.snapshot().target is not None:
                break

        snapshot = screen._vllm_owner.snapshot()
        assert [request.expected_model_id for request in requests] == [
            None,
            "org/second",
        ]
        assert requests[1].token.generation == token.generation + 1
        assert snapshot.state is VllmReadinessState.READY
        assert snapshot.target is not None
        assert snapshot.target.model_id == "org/second"
        assert screen._vllm_external_models == ("org/second",)
        assert selector.value == "org/second"


async def test_mounted_external_changed_list_requires_fresh_bounded_rediscovery(
    monkeypatch,
):
    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        draft = VllmLaunchDraft(
            mode=VllmMode.EXISTING,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="",
            existing_server_url="http://127.0.0.1:8000/v1",
            existing_model_id="org/old",
        )
        token = screen._vllm_owner.begin(draft, runtime_owner="external")
        screen._vllm_draft = draft
        screen._vllm_external_models = ("org/old",)

        async def changed_probe(request):
            if request.expected_model_id is None:
                return VllmProbeResult(
                    token=request.token,
                    state=VllmReadinessState.NOT_CONFIGURED,
                    target=None,
                    issue=None,
                    activity=(VllmActivityEvent("models_discovered", "under_1s"),),
                    discovered_model_ids=("org/new",),
                )
            return VllmProbeResult(
                token=request.token,
                state=VllmReadinessState.NEEDS_ATTENTION,
                target=None,
                issue=VllmIssue("model_missing", "model"),
                activity=(VllmActivityEvent("model_missing", "under_1s"),),
            )

        monkeypatch.setattr(
            "tldw_chatbook.UI.Screens.llm_screen.probe_vllm_target",
            changed_probe,
        )
        await screen._probe_vllm_generation(token, draft, None)

        assert screen._vllm_draft.existing_model_id == ""
        assert screen._vllm_external_models == ()
        assert screen._vllm_owner.snapshot().target is None
        selector = view.query_one("#vllm-existing-model", Select)
        assert selector.value is Select.NULL
        assert selector.disabled
        assert "Check connection" in str(
            view.query_one("#vllm-existing-model-help", Label).renderable
        )

        stale_generation = screen._vllm_owner.snapshot().generation
        screen._on_vllm_external_model_selected(
            VllmSetupView.ExternalModelSelected("org/old")
        )
        assert screen._vllm_owner.snapshot().generation == stale_generation

        discovery_draft = screen._vllm_draft
        discovery_token = screen._vllm_owner.begin(
            discovery_draft, runtime_owner="external"
        )
        await screen._probe_vllm_generation(discovery_token, discovery_draft, None)
        assert screen._vllm_external_models == ("org/new",)
        assert not selector.disabled
        generation = screen._vllm_owner.snapshot().generation

        exact_starts = []
        monkeypatch.setattr(
            screen,
            "_start_vllm_probe",
            lambda exact_token, exact_draft, claim: exact_starts.append(
                (exact_token, exact_draft, claim)
            ),
        )
        screen._on_vllm_external_model_selected(
            VllmSetupView.ExternalModelSelected("org/new")
        )
        assert len(exact_starts) == 1
        assert exact_starts[0][0].generation == generation + 1
        assert exact_starts[0][1].existing_model_id == "org/new"
        assert exact_starts[0][2] is None


async def test_mounted_cancel_check_only_cancels_current_generation():
    class _PendingWorker:
        is_finished = False

        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, _ = await _mount_vllm_screen(app, pilot)
        token = screen._vllm_owner.begin(screen._vllm_draft, runtime_owner="chatbook")
        worker = _PendingWorker()
        screen._vllm_preflight_worker = worker

        screen._on_vllm_cancel_check_requested(
            VllmSetupView.CancelCheckRequested(token.generation - 1)
        )
        assert not worker.cancelled
        assert screen._vllm_owner.snapshot().generation == token.generation

        screen._on_vllm_cancel_check_requested(
            VllmSetupView.CancelCheckRequested(token.generation)
        )
        snapshot = screen._vllm_owner.snapshot()
        assert worker.cancelled
        assert snapshot.generation == token.generation + 1
        assert snapshot.state is VllmReadinessState.NOT_CONFIGURED
        assert snapshot.activity[-1].code == "cancelled"


async def test_mounted_profile_validation_error_is_field_adjacent():
    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        screen._on_vllm_create_profile(
            VllmSetupView.CreateProfileRequested("", screen._vllm_draft)
        )
        await pilot.pause()

        help_copy = view.query_one("#vllm-profile-name-help", Label)
        assert help_copy.display
        assert "profile name" in str(help_copy.renderable).lower()
        assert app.focused is view.query_one("#vllm-profile-name", Input)


async def test_mounted_python_environment_browse_updates_the_guided_field(
    monkeypatch,
):
    """Use the established file picker and return only to the current pane."""

    from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        _, _, view = await _mount_vllm_screen(app, pilot)
        pushed = []

        async def capture_picker(screen, callback=None):
            pushed.append((screen, callback))

        monkeypatch.setattr(
            app,
            "push_screen",
            capture_picker,
        )

        await pilot.click("#vllm-browse-python-environment")
        await pilot.pause()
        picker, picked = pushed.pop()
        assert isinstance(picker, EnhancedFileOpen)
        assert picker.filters is not None
        assert picker.filters[0](Path("python3.12")) is True
        assert picker.filters[0](Path("pip3")) is False

        selected = Path("/safe/venv/bin/python3.12")
        await picked(selected)
        await pilot.pause()
        assert view.query_one("#vllm-python-environment", Input).value == str(selected)


async def test_outer_lab_chrome_tracks_verified_vllm_context_without_focus_theft():
    """Project the active vLLM profile, target, scope, and next action in Lab."""

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        screen._vllm_profiles_loaded = True
        draft = replace(
            screen._vllm_draft,
            mode=VllmMode.EXISTING,
            existing_server_url="http://127.0.0.1:8000/v1",
            existing_model_id="org/model",
        )
        token = screen._vllm_owner.begin(draft, runtime_owner="external")
        target = VllmConnectionTarget(
            provider_key="vllm",
            api_url="http://127.0.0.1:8000/v1/chat/completions",
            model_id="org/model",
            runtime_owner="external",
            generation=token.generation,
            credential_source="none",
        )
        result = VllmProbeResult(
            token=token,
            state=VllmReadinessState.READY,
            target=target,
            issue=None,
            activity=(VllmActivityEvent("ready", "under_1s"),),
        )
        assert screen._vllm_owner.settle(token, result)
        screen._vllm_draft = draft
        url = view.query_one("#vllm-existing-server-url", Input)
        url.focus()
        screen._apply_vllm_view_state(focus=False)
        await pilot.pause()

        assert app.focused is url
        assert (
            str(
                screen.query_one(
                    "#lab-destination-header #workbench-header-title", Static
                ).renderable
            )
            == "vLLM"
        )
        assert "Ready" in str(
            screen.query_one(
                "#lab-destination-header #workbench-header-status", Static
            ).renderable
        )
        assert "Default vLLM" in str(
            screen.query_one("#lab-status-chip-servers", Static).renderable
        )
        assert "Use in Console" in str(
            screen.query_one("#lab-status-chip-model-install", Static).renderable
        )

        inspector = {
            row.id: str(row.renderable)
            for row in screen.query(".lab-vllm-inspector-row").results(Static)
        }
        assert inspector == {
            "lab-vllm-profile": "Profile · Default vLLM",
            "lab-vllm-ownership": "Ownership · External server",
            "lab-vllm-target": (
                "Verified · http://127.0.0.1:8000/v1/chat/completions · org/model"
            ),
            "lab-vllm-persistence": (
                "Persistence · Console use is session-only; defaults unchanged"
            ),
            "lab-vllm-configuration": "Current · Verified external; Next · Matches",
            "lab-vllm-next-action": "Next action · Use in Console",
        }
        assert all(row.display for row in screen.query(".lab-vllm-inspector-row"))
        assert not any(
            row.display for row in screen.query(".lab-generic-inspector-row")
        )

        url.value = "http://127.0.0.1:8001/v1"
        await pilot.pause()
        assert app.focused is url
        assert "Setup incomplete" in str(
            screen.query_one(
                "#lab-destination-header #workbench-header-status", Static
            ).renderable
        )
        assert (
            str(screen.query_one("#lab-vllm-target", Static).renderable)
            == "Verified · Not available"
        )
        assert (
            str(screen.query_one("#lab-vllm-next-action", Static).renderable)
            == "Next action · Check connection"
        )


async def test_mounted_recomposition_preserves_exact_readiness_but_detach_invalidates():
    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, _ = await _mount_vllm_screen(app, pilot)
        screen._vllm_profiles_loaded = True
        draft = screen._vllm_draft
        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        _bind_local_claim(screen._vllm_owner, token)
        assert screen._vllm_owner.settle(token, _ready_result(token))

        await screen.recompose()
        for _ in range(8):
            await pilot.pause()
        recomposed = screen._vllm_owner.snapshot()
        assert recomposed.generation == token.generation
        assert recomposed.target is not None
        assert recomposed.target.generation == token.generation
        assert screen.llm_window is not None
        screen.llm_window.active_view = "vllm"
        for _ in range(20):
            await pilot.pause()
            replacement_views = list(screen.query(VllmSetupView))
            if replacement_views:
                break
        else:
            raise AssertionError("recomposed vLLM view did not mount")
        replacement_view = screen._vllm_view()
        assert replacement_view is not None
        assert screen._vllm_owner.snapshot().state is VllmReadinessState.READY
        screen._apply_vllm_view_state(focus=False)
        for _ in range(8):
            await pilot.pause()
            if not replacement_view.query_one("#vllm-use-console", Button).disabled:
                break
        assert replacement_view._state is VllmReadinessState.READY
        assert replacement_view._connection is not None
        assert replacement_view._connection.state is VllmReadinessState.READY
        assert replacement_view._connection.target is not None
        assert not replacement_view.query_one("#vllm-use-console", Button).disabled

        await app.pop_screen()
        await pilot.pause()
        detached = screen._vllm_owner.snapshot()
        assert detached.generation == token.generation + 1
        assert detached.target is None
        assert detached.activity[-1].code == "screen_detached"


async def test_stop_before_process_publication_settles_cancel_and_retry_refuses_claim():
    """Catch a cancelled reservation leaving Retry indefinitely loading."""

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, _ = await _mount_vllm_screen(app, pilot)
        draft = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="org/model",
        )
        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        claim = reserve_server_launch(app, "vllm", authority="chatbook-vllm")
        assert claim is not None
        assert screen._vllm_owner.bind_launch_claim(token, claim)
        screen._vllm_claim = claim
        screen._vllm_draft = draft
        screen._settle_vllm_state(
            token,
            VllmReadinessState.LAUNCHING,
            activity_code="launch_reserved",
        )
        screen._apply_vllm_view_state()

        await screen._on_vllm_stop_requested(VllmSetupView.StopRequested())
        stopped = screen._vllm_owner.snapshot()
        assert stopped.state is VllmReadinessState.NEEDS_ATTENTION
        assert stopped.issue == VllmIssue("cancelled", "process")

        screen._on_vllm_retry_requested(VllmSetupView.RetryRequested())
        await pilot.pause()
        retried = screen._vllm_owner.snapshot()
        assert retried.state is VllmReadinessState.NEEDS_ATTENTION
        assert retried.issue == VllmIssue("cancelled", "process")
        assert screen._vllm_probe_worker is None
        assert release_server_claim(app, "vllm", claim)


async def test_live_owned_claim_keeps_stop_enabled_across_edit_and_screen_replacement():
    """Catch editable connection state hiding an app-owned live process."""

    app = _build_test_app()
    process = _RunningProcess()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        draft = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="org/model",
        )
        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        claim = reserve_server_launch(app, "vllm", authority="chatbook-vllm")
        assert claim is not None
        assert screen._vllm_owner.bind_launch_claim(token, claim)
        assert publish_server_process(app, "vllm", claim, process)
        screen._vllm_claim = claim
        screen._vllm_draft = draft
        assert screen._vllm_owner.settle(token, _ready_result(token))
        screen._apply_vllm_view_state()
        assert not view.query_one("#vllm-stop", Button).disabled

        view.query_one("#vllm-hf-model", Input).value = "org/edited-model"
        await pilot.pause()
        assert screen._vllm_owner.snapshot().state is (
            VllmReadinessState.NOT_CONFIGURED
        )
        assert not view.query_one("#vllm-stop", Button).disabled

        await app.pop_screen()
        await pilot.pause()
        replacement = LLMScreen(app)
        assert replacement._vllm_claim is claim
        await app.push_screen(replacement)
        for _ in range(8):
            await pilot.pause()
        replacement_window = replacement.query_one(LLMManagementWindow)
        replacement_window.active_view = "vllm"
        for _ in range(12):
            await pilot.pause()
            replacement_views = list(replacement.query(VllmSetupView))
            if replacement_views:
                break
        else:
            raise AssertionError("replacement vLLM setup view did not mount")
        replacement._apply_vllm_view_state()
        assert not replacement_views[0].query_one("#vllm-stop", Button).disabled

        process.running = False
        assert clear_server_process(app, "vllm", claim, process)


async def test_preflight_issue_settles_owner_view_and_recovery(monkeypatch):
    """Catch a preflight failure existing only in the mounted view."""

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        draft = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="org/model",
        )
        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        failure = VllmPreflightResult(
            generation=token.generation,
            fingerprint=token.fingerprint,
            issues=(VllmIssue("python_unavailable", "python_environment"),),
        )
        monkeypatch.setattr(
            "tldw_chatbook.UI.Screens.llm_screen.run_vllm_preflight",
            lambda candidate, generation: failure,
        )

        await screen._run_vllm_preflight_generation(token, draft)

        snapshot = screen._vllm_owner.snapshot()
        assert snapshot.state is VllmReadinessState.NEEDS_ATTENTION
        assert snapshot.issue == VllmIssue("python_unavailable", "python_environment")
        assert snapshot.activity[-1].code == "preflight_failed"
        assert "Needs attention" in str(
            view.query_one("#vllm-readiness-state", Label).renderable
        )
        assert not view.query_one("#vllm-recovery-primary", Button).disabled

        retry = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        success = VllmPreflightResult(
            generation=retry.generation,
            fingerprint=retry.fingerprint,
            issues=(),
        )
        monkeypatch.setattr(
            "tldw_chatbook.UI.Screens.llm_screen.run_vllm_preflight",
            lambda candidate, generation: success,
        )
        await screen._run_vllm_preflight_generation(retry, draft)
        recovered = screen._vllm_owner.snapshot()
        assert recovered.state is VllmReadinessState.READY_TO_START
        assert recovered.issue is None
        assert "Ready to start" in str(
            view.query_one("#vllm-readiness-state", Label).renderable
        )


async def test_probe_deadline_reports_overall_thirty_second_elapsed_bucket(monkeypatch):
    """A final retry must not report only its own sub-second attempt duration."""

    import tldw_chatbook.UI.Screens.llm_screen as llm_screen_module

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, _ = await _mount_vllm_screen(app, pilot)
        draft = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="org/model",
        )
        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        claim = _bind_local_claim(screen._vllm_owner, token)
        process = object()
        clock = [100.0]

        async def unavailable(request):
            return VllmProbeResult(
                token=request.token,
                state=VllmReadinessState.NEEDS_ATTENTION,
                target=None,
                issue=VllmIssue("health_timeout", "connection"),
                activity=(VllmActivityEvent("health_timeout", "under_1s"),),
            )

        async def reach_deadline(_delay):
            clock[0] = 130.0

        monkeypatch.setattr(llm_screen_module.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(llm_screen_module.asyncio, "sleep", reach_deadline)
        monkeypatch.setattr(llm_screen_module, "probe_vllm_target", unavailable)
        monkeypatch.setattr(
            llm_screen_module,
            "server_lifecycle_snapshot",
            lambda _app, _provider: (claim, process),
        )
        monkeypatch.setattr(llm_screen_module, "process_is_running", lambda _p: True)

        await screen._probe_vllm_generation(token, draft, claim)

        snapshot = screen._vllm_owner.snapshot()
        assert snapshot.state is VllmReadinessState.NEEDS_ATTENTION
        assert snapshot.issue == VllmIssue("health_timeout", "connection")
        assert (
            snapshot.activity[-1].code,
            snapshot.activity[-1].elapsed_bucket,
        ) == ("health_timeout", "30s_or_more")


async def test_vllm_failure_details_never_cross_logs_notifications_or_global_state(
    caplog,
):
    """Exception details must stop below every app-wide lifecycle surface."""

    import tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle as lifecycle

    private_detail = (
        "CREDENTIAL_CANARY /private/PATH_CANARY --api-key RAW_COMMAND_CANARY "
        "https://URL_CANARY.invalid/v1 RESPONSE_CANARY"
    )

    class FailingResource:
        def close(self):
            raise RuntimeError(private_detail)

    def fail_profile_change():
        raise VllmProfileCorrupt(private_detail)

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, view = await _mount_vllm_screen(app, pilot)
        draft = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="org/model",
        )
        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        claim = reserve_server_launch(app, "vllm", authority="chatbook-vllm")
        assert claim is not None
        assert screen._vllm_owner.bind_launch_claim(token, claim)
        assert attach_server_claim_resource(app, "vllm", claim, FailingResource())

        # App startup intentionally rebuilds root handlers, so attach pytest's
        # capture handler directly before exercising the real lifecycle logger.
        lifecycle.logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.ERROR, logger=lifecycle.__name__):
                await screen._run_vllm_profile_mutation(fail_profile_change)
                assert release_server_claim(app, "vllm", claim)
        finally:
            lifecycle.logger.removeHandler(caplog.handler)
        await pilot.pause()

        notification_text = " ".join(
            f"{notification.title} {notification.message}"
            for notification in app._notifications
        )
        visible_text = " ".join(str(label.renderable) for label in view.query(Label))
        state_text = repr(screen._vllm_owner.snapshot()) + repr(
            app._llm_server_launch_claims
        )
        log_text = "\n".join(record.getMessage() for record in caplog.records)
        assert "vLLM profile change was not saved" in notification_text
        assert "category=resource_close_failed" in log_text
        assert current_server_claim(app, "vllm") is None
        all_surfaces = f"{notification_text} {visible_text} {state_text} {log_text}"
        for canary in (
            "CREDENTIAL_CANARY",
            "PATH_CANARY",
            "RAW_COMMAND_CANARY",
            "URL_CANARY",
            "RESPONSE_CANARY",
        ):
            assert canary not in all_surfaces


async def test_vllm_handoff_intents_are_secret_free_exact_and_strict():
    """A looser value type or copied extras would cross the screen boundary."""

    from tldw_chatbook.UI.Navigation.pending_handoff_store import (
        HandoffChannel,
        HandoffValueError,
        PendingHandoffStore,
    )
    from tldw_chatbook.UI.Navigation.vllm_handoff import (
        VllmConsoleIntent,
        VllmDefaultIntent,
    )

    target = VllmConnectionTarget(
        provider_key="vllm",
        api_url="http://127.0.0.1:8000/v1/chat/completions",
        model_id="chatbook-vllm",
        runtime_owner="chatbook",
        generation=7,
        credential_source="environment",
    )

    assert VllmConsoleIntent.from_target(target) == VllmConsoleIntent(
        api_url="http://127.0.0.1:8000/v1/chat/completions",
        model_id="chatbook-vllm",
        generation=7,
    )
    assert VllmDefaultIntent.from_target(target) == VllmDefaultIntent(
        api_url="http://127.0.0.1:8000/v1/chat/completions",
        model_id="chatbook-vllm",
        generation=7,
    )
    assert set(VllmConsoleIntent.__slots__) == {"api_url", "model_id", "generation"}
    assert set(VllmDefaultIntent.__slots__) == {"api_url", "model_id", "generation"}

    for intent_type in (VllmConsoleIntent, VllmDefaultIntent):
        for invalid in (
            {"api_url": "http://user:secret@127.0.0.1:8000/v1", "generation": 7},
            {"api_url": "http://127.0.0.1:8000/v1?secret=yes", "generation": 7},
            {"api_url": "http://127.0.0.1:8000/v1#secret", "generation": 7},
            {"model_id": "/private/model", "generation": 7},
            {"generation": True},
        ):
            values = {
                "api_url": "http://127.0.0.1:8000/v1/chat/completions",
                "model_id": "chatbook-vllm",
                "generation": 7,
            }
            values.update(invalid)
            with pytest.raises((TypeError, ValueError)):
                intent_type(**values)

    mutable_extra = type(
        "MutableVllmIntent",
        (),
        {
            "api_url": target.api_url,
            "model_id": target.model_id,
            "generation": target.generation,
            "extras": [],
        },
    )()
    with pytest.raises(HandoffValueError):
        PendingHandoffStore().stage(
            HandoffChannel.VLLM_CONSOLE,
            mutable_extra,
        )


async def test_vllm_handoff_intents_reject_mutable_string_subclasses():
    """Exact handoff text must not retain subclass attributes or behavior."""

    from tldw_chatbook.UI.Navigation.vllm_handoff import (
        VllmConsoleIntent,
        VllmDefaultIntent,
    )

    class MutableModelId(str):
        def __new__(cls, value: str):
            instance = super().__new__(cls, value)
            instance.extras = []
            return instance

    model_id = MutableModelId("chatbook-vllm")
    for intent_type in (VllmConsoleIntent, VllmDefaultIntent):
        with pytest.raises((TypeError, ValueError)):
            intent_type(
                api_url="http://127.0.0.1:8000/v1/chat/completions",
                model_id=model_id,
                generation=7,
            )


async def test_handoff_buttons_enable_only_for_current_verified_target():
    """Stale readiness must never leave either cross-screen action enabled."""

    app = _VllmHost()
    async with app.run_test(size=(120, 40)):
        view = app.query_one(VllmSetupView)
        use = app.query_one("#vllm-use-console", Button)
        default = app.query_one("#vllm-make-default", Button)
        assert use.disabled and default.disabled

        draft = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="org/model",
        )
        owner = VllmConnectionOwner()
        token = owner.begin(draft, runtime_owner="chatbook")
        _bind_local_claim(owner, token)
        assert owner.settle(token, _ready_result(token))
        view.apply_state(
            draft=draft,
            state=VllmReadinessState.READY,
            preflight=None,
            connection=owner.snapshot(),
            profiles_ready=True,
        )
        assert not use.disabled and not default.disabled

        owner.invalidate("target_changed")
        view.apply_state(
            draft=draft,
            state=VllmReadinessState.NOT_CONFIGURED,
            preflight=None,
            connection=owner.snapshot(),
            profiles_ready=True,
        )
        assert use.disabled and default.disabled


@pytest.mark.parametrize(
    ("state", "target_id"),
    [
        (VllmReadinessState.READY_TO_START, "vllm-start"),
        (VllmReadinessState.LAUNCHING, "vllm-stop"),
        (VllmReadinessState.LOADING_MODEL, "vllm-stop"),
        (VllmReadinessState.READY, "vllm-use-console"),
        (VllmReadinessState.NEEDS_ATTENTION, "vllm-recovery-primary"),
    ],
)
async def test_explicit_vllm_state_transition_focuses_phase_action(state, target_id):
    """Lifecycle focus lands on the action that advances or repairs the state."""

    app = _VllmHost()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(VllmSetupView)
        draft = replace(view.draft, model_value="org/model")
        owner = VllmConnectionOwner()
        token = owner.begin(draft, runtime_owner="chatbook")
        preflight = VllmPreflightResult(
            generation=token.generation,
            fingerprint=token.fingerprint,
            issues=(),
            cli_path=Path("/safe/vllm"),
        )
        runtime_active = state in {
            VllmReadinessState.LAUNCHING,
            VllmReadinessState.LOADING_MODEL,
            VllmReadinessState.READY,
        }
        connection = replace(owner.snapshot(), state=state)
        if state is VllmReadinessState.READY:
            _bind_local_claim(owner, token)
            assert owner.settle(token, _ready_result(token))
            connection = owner.snapshot()
        elif state is VllmReadinessState.NEEDS_ATTENTION:
            preflight = replace(
                preflight,
                issues=(VllmIssue("model_missing", "model"),),
            )
        view.apply_state(
            draft=draft,
            state=state,
            preflight=preflight,
            connection=connection,
            runtime_active=runtime_active,
            profiles_ready=True,
        )
        view.focus_state_action(state)
        await pilot.pause()
        assert app.focused.id == target_id


async def test_vllm_handoff_stages_only_current_target_and_uses_normal_navigation(
    monkeypatch,
):
    """A stale target or failed dispatch must not survive as a pending handoff."""

    from tldw_chatbook.Constants import TAB_CHAT, TAB_SETTINGS
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, _ = await _mount_vllm_screen(app, pilot)
        draft = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="org/model",
        )
        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        claim = reserve_server_launch(app, "vllm", authority="chatbook-vllm")
        assert claim is not None
        assert screen._vllm_owner.bind_launch_claim(token, claim)
        process = _RunningProcess()
        assert publish_server_process(app, "vllm", claim, process)
        assert screen._vllm_owner.settle(token, _ready_result(token))
        seen: list[NavigateToScreen] = []
        original_post_message = screen.post_message
        monkeypatch.setattr(
            screen,
            "post_message",
            lambda message: seen.append(message) or True,
        )

        screen._on_vllm_use_in_console_requested(VllmSetupView.UseInConsoleRequested())
        screen._on_vllm_make_default_requested(VllmSetupView.MakeDefaultRequested())

        assert [(message.screen_name, message.screen_context) for message in seen] == [
            (TAB_CHAT, {}),
            (
                TAB_SETTINGS,
                {"category": "providers-models"},
            ),
        ]
        assert app.pending_handoffs.has_pending(HandoffChannel.VLLM_CONSOLE)
        assert app.pending_handoffs.has_pending(HandoffChannel.VLLM_DEFAULT)
        screen._apply_vllm_view_state(focus=False)
        assert str(screen.query_one("#lab-vllm-persistence", Static).renderable) == (
            "Persistence · Console use is session-only; defaults unchanged"
        )

        screen._vllm_owner.invalidate("target_changed")
        app.pending_handoffs.clear_pending(HandoffChannel.VLLM_CONSOLE)
        screen._on_vllm_use_in_console_requested(VllmSetupView.UseInConsoleRequested())
        assert not app.pending_handoffs.has_pending(HandoffChannel.VLLM_CONSOLE)
        monkeypatch.setattr(screen, "post_message", original_post_message)

        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        assert screen._vllm_owner.bind_launch_claim(token, claim)
        assert screen._vllm_owner.settle(token, _ready_result(token))
        monkeypatch.setattr(screen, "post_message", lambda _message: False)
        screen._on_vllm_use_in_console_requested(VllmSetupView.UseInConsoleRequested())
        assert not app.pending_handoffs.has_pending(HandoffChannel.VLLM_CONSOLE)
        monkeypatch.setattr(screen, "post_message", original_post_message)
        process.running = False
        assert clear_server_process(app, "vllm", claim, process)
