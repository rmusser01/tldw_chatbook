from types import SimpleNamespace

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Personal_Context.interview_coordinator import (
    ProfileInterviewCoordinator,
)
from tldw_chatbook.Personal_Context.interview_draft_repository import (
    InterviewDraftRepository,
)
from tldw_chatbook.Personal_Context.interview_launch import _pack
from tldw_chatbook.Personal_Context.interview_launch import (
    ProfileInterviewLaunchRequest,
    launch_profile_interview_after_commit,
    launch_workspace_profile_interview_after_commit,
)
from tldw_chatbook.Personal_Context.interview_provider import FixedQuestionProvider
from tldw_chatbook.Personal_Context.key_protector import InMemoryProfileKeyProtector
from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
from tldw_chatbook.Personal_Context.service import PersonalContextService
from tldw_profile_core import InterviewAudience
from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.Widgets.workspace_create_modal import WorkspaceCreateResult
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


class _Registry:
    def __init__(self) -> None:
        self.activated: list[str] = []

    def set_active_workspace(self, workspace_id: str) -> None:
        self.activated.append(workspace_id)


class _App:
    def __init__(self, registry: _Registry) -> None:
        self.workspace_registry_service = registry
        self.calls: list[str] = []
        self.callbacks = []

    def prepare_personal_context_interview_request(self, **kwargs):
        self.calls.append(f"prepare:{kwargs['local_workspace_id']}")
        return SimpleNamespace(**kwargs, scope_id="scope-workspace")

    def build_personal_context_interview_screen(self, request):
        return request

    def push_screen(self, _screen, callback) -> None:
        self.calls.append("interview")
        self.callbacks.append(callback)

    def notify(self, message: str, **_kwargs) -> None:
        self.calls.append(f"notify:{message}")


def _result(*, offer: bool) -> WorkspaceCreateResult:
    return WorkspaceCreateResult(
        workspace_id="workspace-local-1",
        name="Workspace 1",
        make_active=False,
        offer_profile_interview=offer,
    )


class _ConsoleHarness:
    def __init__(self, app: _App) -> None:
        self.app_instance = app
        self.calls: list[str] = []

    def _sync_console_workspace_context(self) -> None:
        self.calls.append("context")

    def _continue_workspace_create_result(self, result) -> None:
        ConsoleWorkspaceController._continue_workspace_create_result(self, result)


def test_console_workspace_continuation_runs_once_after_interview() -> None:
    app = _App(_Registry())
    screen = _ConsoleHarness(app)

    ConsoleWorkspaceController._handle_workspace_create_result(
        screen, _result(offer=True)
    )

    assert screen.calls == []
    app.callbacks[0](None)
    app.callbacks[0](None)
    assert screen.calls == ["context"]


class _SettingsHarness:
    def __init__(self, app: _App) -> None:
        self.app_instance = app
        self.app = app
        self._settings_workspaces_result = ""
        self.refreshes = 0

    def _refresh_settings_workspaces_pane(self) -> None:
        self.refreshes += 1

    def _continue_workspace_create_result(self, result) -> None:
        SettingsScreen._continue_workspace_create_result(self, result)


def test_settings_workspace_continuation_runs_once_after_interview() -> None:
    app = _App(_Registry())
    screen = _SettingsHarness(app)

    SettingsScreen._handle_workspace_create_result(screen, _result(offer=True))

    assert screen.refreshes == 0
    app.callbacks[0](None)
    app.callbacks[0](None)
    assert screen.refreshes == 1


class _LibraryHarness:
    def __init__(self, app: _App) -> None:
        self.app_instance = app
        self.app = app
        self.refreshes = 0

    def _invalidate_library_workspace_depth_state(self) -> None:
        pass

    def _preserve_library_rail_scroll(self) -> None:
        pass

    def refresh(self, **_kwargs) -> None:
        self.refreshes += 1

    def _continue_workspace_create_result(self, result) -> None:
        LibraryScreen._continue_workspace_create_result(self, result)


def test_library_workspace_continuation_runs_once_after_interview() -> None:
    app = _App(_Registry())
    screen = _LibraryHarness(app)

    LibraryScreen._handle_workspace_create_result(screen, _result(offer=True))

    assert screen.refreshes == 0
    app.callbacks[0](None)
    app.callbacks[0](None)
    assert screen.refreshes == 1


@pytest.mark.parametrize("owner", ["console", "settings", "library"])
def test_workspace_without_offer_runs_existing_continuation_immediately(
    owner: str,
) -> None:
    app = _App(_Registry())
    result = _result(offer=False)
    if owner == "console":
        screen = _ConsoleHarness(app)
        ConsoleWorkspaceController._handle_workspace_create_result(screen, result)
        assert screen.calls == ["context"]
    elif owner == "settings":
        screen = _SettingsHarness(app)
        SettingsScreen._handle_workspace_create_result(screen, result)
        assert screen.refreshes == 1
    else:
        screen = _LibraryHarness(app)
        LibraryScreen._handle_workspace_create_result(screen, result)
        assert screen.refreshes == 1
    assert app.callbacks == []


def test_workspace_interview_launch_failure_preserves_created_workspace() -> None:
    registry = _Registry()
    app = _App(registry)
    app.build_personal_context_interview_screen = lambda _request: (
        _ for _ in ()
    ).throw(RuntimeError("unavailable"))
    screen = _ConsoleHarness(app)

    ConsoleWorkspaceController._handle_workspace_create_result(
        screen, _result(offer=True)
    )

    assert screen.calls == ["context"]
    assert app.callbacks == []


def test_workspace_launch_failure_copy_does_not_claim_setup() -> None:
    app = _App(_Registry())
    app.build_personal_context_interview_screen = lambda _request: (
        _ for _ in ()
    ).throw(RuntimeError("unavailable"))

    launch_profile_interview_after_commit(
        app,
        ProfileInterviewLaunchRequest(
            kind="workspace",
            scope_id="scope-workspace",
            local_workspace_id="workspace-local-1",
            source="workspace",
        ),
        lambda: app.calls.append("continued"),
    )

    assert "continued" in app.calls
    assert any("Workspace created" in call for call in app.calls)
    assert all("Setup was saved" not in call for call in app.calls)


def test_settings_launch_failure_copy_does_not_claim_setup() -> None:
    app = _App(_Registry())
    app.build_personal_context_interview_screen = lambda _request: (
        _ for _ in ()
    ).throw(RuntimeError("unavailable"))

    launch_profile_interview_after_commit(
        app,
        ProfileInterviewLaunchRequest(
            kind="personal",
            scope_id="scope-global",
            source="settings",
        ),
        lambda: app.calls.append("continued"),
    )

    assert "continued" in app.calls
    assert any("Profile interview" in call for call in app.calls)
    assert all("Setup was saved" not in call for call in app.calls)


def test_workspace_prepare_failure_continues_when_notification_raises() -> None:
    app = _App(_Registry())
    app.prepare_personal_context_interview_request = lambda **_kwargs: (
        _ for _ in ()
    ).throw(RuntimeError("unavailable"))
    app.notify = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("notification unavailable")
    )
    calls: list[str] = []

    launch_workspace_profile_interview_after_commit(
        app,
        workspace_id="workspace-local-1",
        workspace_label="Workspace 1",
        continuation=lambda: calls.append("continued"),
    )

    assert calls == ["continued"]


def test_build_failure_continues_when_notification_raises() -> None:
    app = _App(_Registry())
    app.build_personal_context_interview_screen = lambda _request: (
        _ for _ in ()
    ).throw(RuntimeError("unavailable"))
    app.notify = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("notification unavailable")
    )
    calls: list[str] = []

    launch_profile_interview_after_commit(
        app,
        ProfileInterviewLaunchRequest(
            kind="personal",
            scope_id="scope-global",
            source="settings",
        ),
        lambda: calls.append("continued"),
    )

    assert calls == ["continued"]


def test_cancelled_interview_preserves_real_created_workspace(tmp_path) -> None:
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspace.sqlite", client_id="profile-handoff-test")
    )
    registry.create_workspace(
        workspace_id="workspace-local-1",
        name="Workspace 1",
    )
    app = _App(registry)
    screen = _ConsoleHarness(app)

    ConsoleWorkspaceController._handle_workspace_create_result(
        screen, _result(offer=True)
    )
    app.callbacks[0](SimpleNamespace(status="cancelled"))

    assert registry.get_workspace("workspace-local-1") is not None
    assert screen.calls == ["context"]


def test_fixed_workspace_pack_commits_every_distinct_allowed_record(tmp_path) -> None:
    service = PersonalContextService(
        PersonalContextRepository(
            tmp_path / "profile.sqlite",
            key_protector=InMemoryProfileKeyProtector(),
        )
    )
    service.create_profile()
    scope = service.create_workspace_scope("workspace-local-1", "Workspace 1")
    coordinator = ProfileInterviewCoordinator(
        service=service,
        drafts=InterviewDraftRepository.memory_only(),
        fixed_provider=FixedQuestionProvider(_pack("workspace")),
    )
    session = coordinator.start(
        kind="workspace",
        scope_id=scope.scope_id,
        mode="fixed",
    )
    while session.status == "active":
        session = coordinator.answer(session.session_id, "A durable answer")
    diff = coordinator.finish(session.session_id)

    coordinator.commit(
        session.session_id,
        selections=tuple(change.change_id for change in diff.changes),
        enable_runtime=False,
    )

    records = service.list_records(scope_ids=(scope.scope_id,))
    assert len(records) == len(_pack("workspace").questions)
    assert {record.kind.value for record in records} <= {
        "goal",
        "working_context",
        "convention",
    }
    assert len({record.semantic_key.subject for record in records}) == len(records)


def test_fixed_personal_dislike_question_proposes_dislike_polarity() -> None:
    question = next(
        question
        for question in _pack("personal").questions
        if question.topic.startswith("dislike.")
    )

    change = ProfileInterviewCoordinator._change_for_answer(
        InterviewAudience.PERSONAL,
        question,
        "Avoid excessive headings",
    )

    assert change.proposed_payload.kind == "preference"
    assert change.proposed_payload.polarity == "dislike"
