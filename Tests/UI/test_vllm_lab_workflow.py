from __future__ import annotations

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
    VllmProfileCorrupt,
    VllmProfileDocumentV1,
    VllmProfileRepository,
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

    def compose(self) -> ComposeResult:
        yield VllmSetupView(id="vllm-setup")

    @on(VllmSetupView.CreateProfileRequested)
    @on(VllmSetupView.SaveProfileRequested)
    @on(VllmSetupView.RenameProfileRequested)
    @on(VllmSetupView.DuplicateProfileRequested)
    @on(VllmSetupView.DeleteProfileRequested)
    def _capture_profile_event(self, event) -> None:
        self.profile_events.append(event)


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
                discovered_model_ids=("org/first", "org/second"),
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
        assert screen._vllm_external_models == ("org/first", "org/second")
        assert selector.value == "org/second"


async def test_mounted_external_changed_list_clears_and_fences_stale_selection(
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
            return VllmProbeResult(
                token=request.token,
                state=VllmReadinessState.NEEDS_ATTENTION,
                target=None,
                issue=VllmIssue("model_missing", "model"),
                activity=(VllmActivityEvent("model_missing", "under_1s"),),
                discovered_model_ids=("org/new",),
            )

        monkeypatch.setattr(
            "tldw_chatbook.UI.Screens.llm_screen.probe_vllm_target",
            changed_probe,
        )
        await screen._probe_vllm_generation(token, draft, None)

        assert screen._vllm_draft.existing_model_id == ""
        assert screen._vllm_external_models == ("org/new",)
        assert screen._vllm_owner.snapshot().target is None
        selector = view.query_one("#vllm-existing-model", Select)
        assert selector.value is Select.NULL
        assert "Select a returned model" in str(
            view.query_one("#vllm-existing-model-help", Label).renderable
        )

        generation = screen._vllm_owner.snapshot().generation
        screen._on_vllm_external_model_selected(
            VllmSetupView.ExternalModelSelected("org/old")
        )
        assert screen._vllm_owner.snapshot().generation == generation

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
            "lab-vllm-persistence": "Persistence · Not adopted; defaults unchanged",
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
        )
        assert not use.disabled and not default.disabled

        owner.invalidate("target_changed")
        view.apply_state(
            draft=draft,
            state=VllmReadinessState.NOT_CONFIGURED,
            preflight=None,
            connection=owner.snapshot(),
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
        _bind_local_claim(screen._vllm_owner, token)
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

        screen._vllm_owner.invalidate("target_changed")
        app.pending_handoffs.clear_pending(HandoffChannel.VLLM_CONSOLE)
        screen._on_vllm_use_in_console_requested(VllmSetupView.UseInConsoleRequested())
        assert not app.pending_handoffs.has_pending(HandoffChannel.VLLM_CONSOLE)
        monkeypatch.setattr(screen, "post_message", original_post_message)

        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        _bind_local_claim(screen._vllm_owner, token)
        assert screen._vllm_owner.settle(token, _ready_result(token))
        monkeypatch.setattr(screen, "post_message", lambda _message: False)
        screen._on_vllm_use_in_console_requested(VllmSetupView.UseInConsoleRequested())
        assert not app.pending_handoffs.has_pending(HandoffChannel.VLLM_CONSOLE)
        monkeypatch.setattr(screen, "post_message", original_post_message)
