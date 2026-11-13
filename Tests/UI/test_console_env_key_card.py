"""TASK-34100.2 AC#12 (E2): the Get started card offers env-found keys.

With only ANTHROPIC_API_KEY exported, the card said "Not ready · no key" and
its one action opened Settings preselected on OpenAI, never mentioning the
key that was found. The card now names each env-keyed provider ("Found
ANTHROPIC_API_KEY · Use Anthropic"); choosing one commits it as the chat
default with ``credential_source = environment`` through the shared
provider-setup owner, in one mutation, and moves this chat onto it.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from textual.app import ComposeResult
from textual.widgets import Button

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_console_local_server_discovery_card import (
    _BLOCKED_CARD_STATE,
    _is_displayed,
)
from tldw_chatbook.Chat import provider_setup_persistence as persistence_module
from tldw_chatbook.Chat.console_onboarding_state import (
    ConsoleSetupCardState,
    build_console_env_key_actions,
)
from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML, ConfigMutationResult
from tldw_chatbook.UI.Console_Modules import setup_card_offers
from tldw_chatbook.UI.Workbench.workbench_widgets import WorkbenchActionRequested
from tldw_chatbook.Widgets.Console.console_setup_modal import (
    CONSOLE_SETUP_MODAL_ENV_KEY_ACTION_ID,
    ConsoleSetupModal,
)

ENVIRON = {
    "ANTHROPIC_API_KEY": "sk-ant-test-env-key-card",
    "GROQ_API_KEY": "gsk-test-env-key-card",
}


def _profile() -> dict:
    return copy.deepcopy(DEFAULT_CONFIG_FROM_TOML)


def _actions():
    return build_console_env_key_actions(_profile(), ENVIRON, card_mode="card")


class EnvKeyCardHarness(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.workbench_actions: list[str] = []

    def compose(self) -> ComposeResult:
        yield ConsoleSetupModal(id="console-setup-modal")

    async def on_mount(self) -> None:
        self.query_one("#console-setup-modal", ConsoleSetupModal).sync_card_state(
            _BLOCKED_CARD_STATE,
            action_label="Set up provider",
            action_tooltip="Open provider settings.",
        )

    def on_workbench_action_requested(self, event: WorkbenchActionRequested) -> None:
        event.stop()
        self.workbench_actions.append(event.action_id)


@pytest.mark.asyncio
async def test_the_card_names_each_found_key_and_routes_its_choice() -> None:
    app = EnvKeyCardHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        modal = app.query_one("#console-setup-modal", ConsoleSetupModal)
        buttons = [
            app.query_one(f"#{CONSOLE_SETUP_MODAL_ENV_KEY_ACTION_ID}-{index}", Button)
            for index in range(3)
        ]
        assert not any(_is_displayed(button) for button in buttons)

        modal.sync_env_key_actions(_actions())
        await pilot.pause()

        assert [str(button.label) for button in buttons[:2]] == [
            "Found ANTHROPIC_API_KEY · Use Anthropic",
            "Found GROQ_API_KEY · Use Groq",
        ]
        assert [_is_displayed(button) for button in buttons] == [True, True, False]
        assert "ANTHROPIC_API_KEY from your environment" in str(buttons[0].tooltip)

        buttons[0].press()
        await pilot.pause()
        assert app.workbench_actions == ["use-env-key:anthropic"]

        modal.sync_card_state(ConsoleSetupCardState(mode="quiet"))
        await pilot.pause()
        assert not any(_is_displayed(button) for button in buttons)


def _fake_screen(events: list) -> SimpleNamespace:
    modal = SimpleNamespace(env_key_actions=_actions())
    screen = SimpleNamespace(
        query_one=lambda *_args: modal,
        _provider_readiness_app_config=_profile,
        _session=SimpleNamespace(
            _replace_active_console_session_settings=lambda settings: events.append(
                ("session", settings.provider, settings.model, settings.source)
            )
        ),
        _sync_console_transcript_guidance=lambda: events.append("guidance"),
        _sync_native_console_chat_ui=lambda: "sync-coroutine",
        run_worker=lambda work, **kwargs: events.append(("worker", work)),
        app_instance=SimpleNamespace(
            app_config=None,
            notify=lambda message, **kwargs: events.append(("notify", message)),
        ),
    )
    return screen


@pytest.mark.asyncio
async def test_choosing_a_found_key_commits_it_once_and_moves_the_chat(monkeypatch):
    events: list = []
    commits: list = []

    def persist(mutation):
        commits.append(mutation)
        return ConfigMutationResult(True, True, None)

    monkeypatch.setattr(persistence_module, "persist_provider_setup", persist)

    applied = await setup_card_offers.apply_env_key_offer(
        _fake_screen(events), "use-env-key:anthropic"
    )

    assert applied is True
    (mutation,) = commits
    section = mutation.section_values["api_settings.anthropic"]
    assert section["credential_source"] == "environment"
    assert section["api_key_env_var"] == "ANTHROPIC_API_KEY"
    assert dict(mutation.section_values["chat_defaults"])["provider"] == "anthropic"
    model = DEFAULT_CONFIG_FROM_TOML["api_settings"]["anthropic"]["model"]
    assert ("session", "anthropic", model, "user") in events
    assert events[-2:] == ["guidance", ("worker", "sync-coroutine")]


@pytest.mark.asyncio
async def test_a_failed_commit_changes_nothing_and_says_so(monkeypatch):
    events: list = []
    monkeypatch.setattr(
        persistence_module,
        "persist_provider_setup",
        lambda mutation: ConfigMutationResult(False, False, "before_replace"),
    )

    applied = await setup_card_offers.apply_env_key_offer(
        _fake_screen(events), "use-env-key:anthropic"
    )

    assert applied is False
    assert events == [
        (
            "notify",
            "Couldn't make Anthropic the chat default; nothing was changed. "
            "Try Set up provider.",
        )
    ]


@pytest.mark.asyncio
async def test_an_offer_no_longer_on_the_card_is_ignored(monkeypatch):
    events: list = []
    monkeypatch.setattr(
        persistence_module,
        "persist_provider_setup",
        lambda mutation: pytest.fail("nothing to commit"),
    )

    assert (
        await setup_card_offers.apply_env_key_offer(
            _fake_screen(events), "use-env-key:mistral"
        )
        is False
    )
    assert events == []


# --- review round 1 ----------------------------------------------------------


@pytest.mark.asyncio
async def test_a_committed_found_key_reaches_every_app_config_reader(monkeypatch):
    """Review round 1 (F2, live): after 'Use Anthropic', Library's analyze-
    after-import still read the template chat default from the app's
    in-memory config (OpenAI · gpt-5.6-terra, no key) until a relaunch. The
    commit now reaches that snapshot exactly as setup's commit does."""
    from tldw_chatbook.Library.ingest_analysis import resolve_ingest_analysis_provider

    events: list = []
    monkeypatch.setattr(
        persistence_module,
        "persist_provider_setup",
        lambda mutation: ConfigMutationResult(True, True, None),
    )
    screen = _fake_screen(events)
    screen.app_instance.app_config = _profile()

    assert await setup_card_offers.apply_env_key_offer(screen, "use-env-key:anthropic")

    snapshot = screen.app_instance.app_config
    model = DEFAULT_CONFIG_FROM_TOML["api_settings"]["anthropic"]["model"]
    assert snapshot["chat_defaults"]["provider"] == "anthropic"
    assert snapshot["chat_defaults"]["model"] == model
    assert snapshot["api_settings"]["anthropic"]["credential_source"] == "environment"
    analysis = resolve_ingest_analysis_provider(snapshot, environ=ENVIRON)
    assert (analysis.provider, analysis.model, analysis.ready) == (
        "anthropic",
        model,
        True,
    )


def _recording_modal(log: list):
    return SimpleNamespace(
        sync_detected_server_action=lambda action: log.append(("server", action)),
        sync_env_key_actions=lambda actions: log.append(
            ("env", tuple(action.label for action in actions))
        ),
    )


@pytest.mark.parametrize(
    ("mode", "offered"),
    [
        ("card", ("Use Anthropic", "Use Groq")),
        ("quiet", ()),
        ("ready_line", ()),
    ],
)
def test_the_card_offers_found_keys_only_while_it_is_the_get_started_card(
    monkeypatch, mode, offered
):
    """Review round 1 (C5): the card shows env-key offers only in 'card' mode
    and withdraws them otherwise, reading the in-memory config snapshot."""
    monkeypatch.setattr(setup_card_offers, "os", SimpleNamespace(environ=ENVIRON))
    screen = SimpleNamespace(
        _console_detected_local_server=None,
        app_instance=SimpleNamespace(app_config=_profile()),
    )
    log: list = []

    setup_card_offers.sync_setup_card_offers(
        screen, _recording_modal(log), ConsoleSetupCardState(mode=mode)
    )

    (env,) = [entry for entry in log if entry[0] == "env"]
    assert env == ("env", offered)


@pytest.mark.asyncio
async def test_the_console_routes_a_found_key_choice_to_the_commit(monkeypatch):
    """Review round 1 (C5): ChatScreen's Workbench dispatch sends
    'use-env-key:<provider>' to the shared commit."""
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    routed: list = []

    async def apply(screen, action_id):
        routed.append((screen, action_id))
        return True

    monkeypatch.setattr(setup_card_offers, "apply_env_key_offer", apply)
    receiver = SimpleNamespace()

    await ChatScreen.on_console_workbench_action_requested(
        receiver, WorkbenchActionRequested("use-env-key:anthropic")
    )

    assert routed == [(receiver, "use-env-key:anthropic")]


def _card_for(snapshot: dict, environ: dict):
    from tldw_chatbook.Chat.console_onboarding_state import (
        build_console_setup_card_state,
    )
    from tldw_chatbook.Chat.console_session_settings import (
        build_console_settings_readiness,
        build_default_console_session_settings,
    )

    settings = build_default_console_session_settings(snapshot)
    readiness = build_console_settings_readiness(
        settings, app_config=snapshot, environ=environ
    )
    card = build_console_setup_card_state(
        readiness=readiness,
        provider_label=readiness.provider_display_name,
        has_model=bool(settings.model),
        first_send_completed=False,
        has_messages=False,
        guidance_dismissed=False,
    )
    return readiness, card


@pytest.mark.asyncio
async def test_after_use_anthropic_the_card_reads_the_shared_verdict(monkeypatch):
    """Review round 1 (C5): the card's readiness line agrees with the shared
    verdict for the provider the found key just made the chat default."""
    from tldw_chatbook.Chat.console_session_settings import readiness_words

    monkeypatch.setattr(
        persistence_module,
        "persist_provider_setup",
        lambda mutation: ConfigMutationResult(True, True, None),
    )
    screen = _fake_screen([])
    screen.app_instance.app_config = _profile()
    before, before_card = _card_for(screen.app_instance.app_config, ENVIRON)
    assert before.provider_display_name == "OpenAI"  # the template default
    assert before_card.mode == "card"
    # Dev's TASK-33008 (faa51c3fb1) names the provider the word is about.
    assert before_card.steps[0].detail == f"OpenAI: {readiness_words(before)}"

    assert await setup_card_offers.apply_env_key_offer(screen, "use-env-key:anthropic")

    after, after_card = _card_for(screen.app_instance.app_config, ENVIRON)
    assert after.provider_display_name == "Anthropic"
    assert after.operability == "ready_to_send"
    assert after_card.mode == "ready_line"


def test_a_card_over_a_full_context_names_the_same_word_as_the_header():
    from tldw_chatbook.Chat.console_session_settings import readiness_words

    snapshot = _profile()
    snapshot["chat_defaults"] = {"provider": "openai", "model": "gpt-4o"}
    snapshot["api_settings"]["openai"].update(
        {"api_key": "sk-test-env-key-card", "max_tokens": 128_000}
    )

    readiness, card = _card_for(snapshot, {})

    assert readiness.blocker == "capacity_exhausted"
    # Pass 2 (final review, major 2): a full context explains itself on the
    # card's one line -- the composer stays usable and the send gate names
    # the fix -- instead of the blocking card's 'Pick a model' step.
    assert card.mode == "ready_line"
    assert card.body_copy.startswith(readiness_words(readiness))
    assert readiness_words(readiness) == "Not ready · window too small"


# --- the env-key toast goes once its claim is false (review round 2, F10) ----

_ENV_NOTICE = (
    "Found ANTHROPIC_API_KEY, but new chats use OpenAI, which has no key. "
    "Choose Use Anthropic on Console's Get started card."
)


@pytest.mark.asyncio
async def test_committing_a_found_key_dismisses_only_the_env_key_toast() -> None:
    """After 'Use Anthropic' committed, the 20 s toast still said 'new chats
    use OpenAI, which has no key' (live, g2-v2-envant/05). Committing the
    offer now removes that toast -- and only that one."""
    from textual.app import App

    from tldw_chatbook.app_env_key_notice import EnvKeyNoticeMixin

    class ToastHost(EnvKeyNoticeMixin, App):
        """The notice mixin is the real app's base, so it owns the toast."""

    app = ToastHost()
    async with app.run_test(size=(120, 30)) as pilot:
        app.notify(_ENV_NOTICE, title="Provider key found — setup needed", timeout=20)
        app.notify("Another copy is open.", title="Profile already open", timeout=20)
        app._env_key_notice_message = _ENV_NOTICE
        await pilot.pause()
        assert len(app._notifications) == 2

        setup_card_offers.dismiss_env_key_notice(app)
        setup_card_offers.dismiss_env_key_notice(app)  # idempotent
        await pilot.pause()
        remaining = [notification.message for notification in app._notifications]

    assert remaining == ["Another copy is open."]


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [True, False])
async def test_the_toast_is_dismissed_only_when_the_offer_commits(
    monkeypatch, committed: bool
) -> None:
    events: list = []
    monkeypatch.setattr(
        persistence_module,
        "persist_provider_setup",
        lambda mutation: ConfigMutationResult(committed, committed, None),
    )
    monkeypatch.setattr(
        setup_card_offers,
        "dismiss_env_key_notice",
        lambda app: events.append("env-key-toast-dismissed"),
    )

    await setup_card_offers.apply_env_key_offer(
        _fake_screen(events), "use-env-key:anthropic"
    )

    assert ("env-key-toast-dismissed" in events) is committed


def test_the_shown_env_key_notice_is_remembered_for_dismissal() -> None:
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat.console_onboarding_state import EnvKeyFirstLaunchNotice

    shown: list = []
    fake = SimpleNamespace(
        _pending_env_key_notice=EnvKeyFirstLaunchNotice(
            title="Provider key found — setup needed", message=_ENV_NOTICE, ready=False
        ),
        notify=lambda message, **kwargs: shown.append(message),
        _persist_env_key_notice_flag=lambda: None,
    )

    TldwCli._show_pending_env_key_notice(fake)

    assert shown == [_ENV_NOTICE]
    assert fake._env_key_notice_message == _ENV_NOTICE
