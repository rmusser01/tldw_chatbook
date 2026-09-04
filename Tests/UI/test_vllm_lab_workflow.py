from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, Collapsible, Input, Label, Select, TextArea

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.config import get_cli_setting as _real_get_cli_setting
from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
    ServerLaunchClaim,
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


@pytest.mark.parametrize("dismissal", ["cancel", "escape"])
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
        else:
            await pilot.press("escape")
        await pilot.pause()

        assert app.screen is screen
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
    async with app.run_test(size=(120, 40)):
        view = app.query_one(VllmSetupView)
        draft = VllmLaunchDraft(
            mode=VllmMode.LOCAL,
            python_environment="python",
            model_source=VllmModelSource.HUGGING_FACE,
            model_value="org/model",
        )
        view.apply_state(
            draft=draft,
            state=VllmReadinessState.NEEDS_ATTENTION,
            preflight=None,
        )
        assert app.query_one("#vllm-start", Button).disabled
        assert "Check setup" in str(
            app.query_one("#vllm-start-blocker", Label).renderable
        )


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


async def test_mounted_recomposition_and_detach_invalidate_readiness_generation():
    app = _build_test_app()
    async with app.run_test(size=(235, 52)) as pilot:
        screen, _, _ = await _mount_vllm_screen(app, pilot)
        draft = screen._vllm_draft
        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        claim = _bind_local_claim(screen._vllm_owner, token)
        assert screen._vllm_owner.settle(token, _ready_result(token))

        await screen.recompose()
        for _ in range(8):
            await pilot.pause()
        recomposed = screen._vllm_owner.snapshot()
        assert recomposed.generation == token.generation + 1
        assert recomposed.target is None
        assert recomposed.activity[-1].code == "recomposed"

        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        assert screen._vllm_owner.bind_launch_claim(token, claim)
        assert screen._vllm_owner.settle(token, _ready_result(token))
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
        assert not replacement_views[0].query_one(
            "#vllm-stop", Button
        ).disabled

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
        assert snapshot.issue == VllmIssue(
            "python_unavailable", "python_environment"
        )
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

        screen._on_vllm_use_in_console_requested(
            VllmSetupView.UseInConsoleRequested()
        )
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
        screen._on_vllm_use_in_console_requested(
            VllmSetupView.UseInConsoleRequested()
        )
        assert not app.pending_handoffs.has_pending(HandoffChannel.VLLM_CONSOLE)
        monkeypatch.setattr(screen, "post_message", original_post_message)

        token = screen._vllm_owner.begin(draft, runtime_owner="chatbook")
        _bind_local_claim(screen._vllm_owner, token)
        assert screen._vllm_owner.settle(token, _ready_result(token))
        monkeypatch.setattr(screen, "post_message", lambda _message: False)
        screen._on_vllm_use_in_console_requested(
            VllmSetupView.UseInConsoleRequested()
        )
        assert not app.pending_handoffs.has_pending(HandoffChannel.VLLM_CONSOLE)
        monkeypatch.setattr(screen, "post_message", original_post_message)
