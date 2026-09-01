from types import SimpleNamespace

import pytest
from textual.widgets import Checkbox

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Constants import TAB_HOME
from tldw_chatbook.Personal_Context import interview_launch
from tldw_chatbook.Personal_Context.interview_launch import (
    ProfileInterviewLaunchRequest,
    build_profile_interview_screen,
)
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import SummaryStep
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
from Tests.Wizards.test_first_run_setup_wizard import _StepHost


class _FirstRunHarness:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.current_tab = "home"
        self.focus_mode = False
        self._deferred_focus_request = False
        self.app_config = {"first_run": {"setup_completed": True}}
        self._profile_interview_launches = []
        self.work = []

    def _schedule_startup_model_catalog_refresh(self, **_kwargs) -> None:
        assert self.app_config["first_run"]["setup_completed"] is True
        self.calls.append("catalog")

    def prepare_personal_context_interview_request(self, **kwargs):
        self.calls.append("prepare")
        return SimpleNamespace(**kwargs, scope_id="scope-global")

    def push_screen(self, _screen, callback) -> None:
        self.calls.append("interview")
        self._profile_interview_launches.append(callback)

    def build_personal_context_interview_screen(self, request):
        return request

    def run_worker(self, work, **_kwargs) -> None:
        self.work.append(work)

    async def handle_screen_navigation(self, event) -> None:
        self.calls.append(f"navigate:{event.screen_name}")


def test_draft_repository_falls_back_when_secure_backend_fails_operational_probe(
    monkeypatch, tmp_path
) -> None:
    class _UnavailableProtector:
        def load_or_create(self, _profile_ref):
            raise RuntimeError("keyring write unavailable")

        def delete(self, _profile_ref):
            raise AssertionError("delete is unreachable after failed creation")

    monkeypatch.setattr(
        interview_launch,
        "get_personal_context_db_path",
        lambda: tmp_path / "personal-context.db",
    )
    monkeypatch.setattr(
        interview_launch,
        "KeyringProfileKeyProtector",
        _UnavailableProtector,
    )

    repository = interview_launch._draft_repository()

    assert repository.is_memory_only is True


def test_adaptive_screen_build_fails_before_push_without_provider_or_model(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "tldw_chatbook.config.get_runtime_config_snapshot",
        lambda: SimpleNamespace(values={}),
    )
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_session_settings.build_default_console_session_settings",
        lambda _values: SimpleNamespace(provider="", model=""),
    )
    monkeypatch.setattr(
        "tldw_chatbook.Chat.provider_readiness.provider_config_key",
        lambda _provider: "",
    )
    app = SimpleNamespace(get_personal_context_service=lambda **_kwargs: object())

    with pytest.raises(RuntimeError, match="configured provider and model"):
        build_profile_interview_screen(
            app,
            ProfileInterviewLaunchRequest(
                kind="personal",
                scope_id="scope-global",
                mode="adaptive",
                source="settings",
            ),
        )


@pytest.mark.asyncio
async def test_summary_offer_control_defaults_false_and_carries_true() -> None:
    wizard = SimpleNamespace(
        app_instance=SimpleNamespace(app_config={}),
        rerun=False,
        wizard_data={"welcome": {"track": "quick"}},
        advance_programmatically=lambda: None,
    )
    step = SummaryStep(
        wizard=wizard,
        config=WizardStepConfig(id="summary", title="Summary", step_number=9),
        load_config=lambda: {},
        rag_deps_installed=lambda: False,
        speech_installed=lambda: False,
        speech_runtime_installed=lambda: False,
    )
    app = _StepHost(step)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        offer = step.query_one("#setup-profile-interview-offer", Checkbox)
        assert offer.value is False
        offer.value = True
        step._exit_home()
        assert step.get_step_data() == {
            "exit_route": TAB_HOME,
            "offer_profile_interview": True,
        }


def test_first_run_marks_setup_complete_before_interview_launch() -> None:
    app = _FirstRunHarness()

    TldwCli._handle_first_run_wizard_result(
        app,
        {
            "completed": True,
            "exit_route": None,
            "offer_profile_interview": True,
        },
    )

    assert app.calls == ["prepare", "interview"]
    app._profile_interview_launches[0](None)
    app._profile_interview_launches[0](None)
    assert app.calls == ["prepare", "interview", "catalog"]


def test_first_run_without_offer_keeps_existing_completion_path() -> None:
    app = _FirstRunHarness()

    TldwCli._handle_first_run_wizard_result(
        app,
        {"completed": True, "exit_route": None},
    )

    assert app.calls == ["catalog"]


def test_first_run_interview_launch_failure_still_continues() -> None:
    app = _FirstRunHarness()

    def _raise(_request):
        raise RuntimeError("screen factory unavailable")

    app.build_personal_context_interview_screen = _raise
    app.notify = lambda *_args, **_kwargs: app.calls.append("warning")

    TldwCli._handle_first_run_wizard_result(
        app,
        {
            "completed": True,
            "exit_route": None,
            "offer_profile_interview": True,
        },
    )

    assert app.calls == ["prepare", "warning", "catalog"]


def test_first_run_prepare_failure_still_continues() -> None:
    app = _FirstRunHarness()
    app.prepare_personal_context_interview_request = lambda **_kwargs: (
        _ for _ in ()
    ).throw(RuntimeError("profile unavailable"))
    app.notify = lambda *_args, **_kwargs: app.calls.append("warning")

    TldwCli._handle_first_run_wizard_result(
        app,
        {
            "completed": True,
            "exit_route": None,
            "offer_profile_interview": True,
        },
    )

    assert app.calls == ["warning", "catalog"]


def test_first_run_prepare_failure_continues_when_notification_raises() -> None:
    app = _FirstRunHarness()
    app.prepare_personal_context_interview_request = lambda **_kwargs: (
        _ for _ in ()
    ).throw(RuntimeError("profile unavailable"))
    app.notify = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("notification unavailable")
    )

    TldwCli._handle_first_run_wizard_result(
        app,
        {
            "completed": True,
            "exit_route": None,
            "offer_profile_interview": True,
        },
    )

    assert app.calls == ["catalog"]


@pytest.mark.parametrize("result", [SimpleNamespace(status="committed"), None])
def test_settings_interview_return_reloads_mounted_profile_once(result) -> None:
    class _Panel:
        def __init__(self) -> None:
            self.loads: list[bool] = []

        def load_records(self, *, retry_locked: bool = False) -> None:
            self.loads.append(retry_locked)

    app = _FirstRunHarness()
    panel = _Panel()
    app.query_one = lambda *_args, **_kwargs: panel
    app.prepare_personal_context_interview_request = lambda **kwargs: SimpleNamespace(
        **kwargs
    )

    TldwCli.launch_personal_context_interview(
        app,
        kind="personal",
        scope_id="scope-global",
    )
    app._profile_interview_launches[0](result)
    app._profile_interview_launches[0](result)

    assert panel.loads == [True]


@pytest.mark.asyncio
async def test_completed_exit_route_continues_once_after_interview() -> None:
    app = _FirstRunHarness()

    TldwCli._handle_first_run_wizard_result(
        app,
        {
            "completed": True,
            "exit_route": TAB_HOME,
            "offer_profile_interview": True,
        },
    )
    assert app.calls == ["prepare", "interview"]
    app._profile_interview_launches[0](None)
    app._profile_interview_launches[0](None)
    assert len(app.work) == 1
    await app.work[0]
    assert app.calls == ["prepare", "interview", f"navigate:{TAB_HOME}", "catalog"]


def test_first_run_rerun_result_can_offer_interview_once() -> None:
    app = _FirstRunHarness()
    app._handle_first_run_wizard_result = lambda result: (
        TldwCli._handle_first_run_wizard_result(app, result)
    )

    TldwCli.handle_first_run_wizard_result(
        app,
        {
            "completed": True,
            "exit_route": None,
            "offer_profile_interview": True,
        },
    )
    app._profile_interview_launches[0](None)

    assert app.calls == ["prepare", "interview", "catalog"]
