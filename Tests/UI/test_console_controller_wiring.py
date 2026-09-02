"""Characterisation of the Console controller wiring (decomposition wave 4).

Written BEFORE wave 4 task 1 moved the six `Console*Controller(...)`
constructions out of `ChatScreen.__init__` and into
`UI/Console_Modules/wiring.py`. Every assertion here is *behavioural* -- it
pins what the wiring DOES, never where the source lines live -- so this file
must stay byte-identical across that move. If the extraction needs one of
these to change, the extraction changed behaviour: stop and treat it as a
finding.

Three properties are pinned, each the thing a "pure move" is most likely to
break silently:

1. **Presence, class, and relative construction order.** `vars(screen)`
   preserves assignment order in CPython, so the six controller slots' order
   in the instance dict is a direct observation of the order `__init__` (now
   `build_console_controllers`) built them in. Only the six's order relative
   to each other is pinned -- unrelated attributes around them are free to
   move.
2. **Dependency identity.** Every named dependency is wired as a
   late-binding lambda closing over the screen, so calling the stored
   callable must return the *identical* object the screen's own method
   returns. "is not None" would pass against a lambda rewired to a different
   (but similarly-shaped) source; `is` would not.
3. **Late binding, including across controllers.** Replacing the target on
   the screen *instance* after construction must be observed by the stored
   callable -- that is the binding rule's whole point (see
   `ConsoleDictationController.__init__`'s docstring, the canonical
   statement of it). The cross-controller cases additionally prove the
   sibling is resolved at CALL time, which is why construction order among
   the six cannot matter.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from textual.css.query import QueryError

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_destination_shells import _build_test_app
from tldw_chatbook.UI.Console_Modules.agent import ConsoleAgentController
from tldw_chatbook.UI.Console_Modules.character import ConsoleCharacterController
from tldw_chatbook.UI.Console_Modules.dictation import ConsoleDictationController
from tldw_chatbook.UI.Console_Modules.fleet import ConsoleFleetLifecycleController
from tldw_chatbook.UI.Console_Modules import hands_free as hands_free_module
from tldw_chatbook.UI.Console_Modules.hands_free import ConsoleHandsFreeController
from tldw_chatbook.UI.Console_Modules.image import ConsoleImageController
from tldw_chatbook.UI.Console_Modules.message import ConsoleMessageController
from tldw_chatbook.UI.Console_Modules.prompt_queue import (
    ConsolePromptQueueUIController,
)
from tldw_chatbook.UI.Console_Modules.prompts import ConsolePromptsController
from tldw_chatbook.UI.Console_Modules.realtime import ConsoleRealtimeController
from tldw_chatbook.UI.Console_Modules.review_selection import (
    ConsoleReviewSelectionController,
)
from tldw_chatbook.UI.Console_Modules import wiring as wiring_module
from tldw_chatbook.Chat.console_chat_models import FEEDBACK_ACTIVE_RUN_STATUSES
from tldw_chatbook.UI.Console_Modules.retrieval import ConsoleRetrievalController
from tldw_chatbook.UI.Console_Modules.send_price import ConsoleSendPriceController
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
from tldw_chatbook.UI.Console_Modules.skill import ConsoleSkillController
from tldw_chatbook.UI.Console_Modules.terminal import ConsoleTerminalController
from tldw_chatbook.UI.Console_Modules.video import ConsoleVideoController
from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Workspaces.models import RuntimeBindingKind, RuntimeBindingStatus

#: (screen attribute, controller class), in the order the wiring builds them.
_EXPECTED_SLOTS: list[tuple[str, type]] = [
    ("_workspace", ConsoleWorkspaceController),
    ("_session", ConsoleSessionController),
    ("_dictation", ConsoleDictationController),
    ("_hands_free", ConsoleHandsFreeController),
    ("_message", ConsoleMessageController),
    ("_prompts", ConsolePromptsController),
]

#: Complete controller graph for the Task-3070.8 construction-order contract.
#: Kept separate because `_EXPECTED_SLOTS` is a historical common-interface subset.
_ALL_CONTROLLER_SLOTS: list[tuple[str, type]] = [
    ("_image", ConsoleImageController),
    ("_video", ConsoleVideoController),
    ("_retrieval", ConsoleRetrievalController),
    ("_skill", ConsoleSkillController),
    ("_workspace", ConsoleWorkspaceController),
    ("_character", ConsoleCharacterController),
    ("_fleet", ConsoleFleetLifecycleController),
    ("_session", ConsoleSessionController),
    ("_dictation", ConsoleDictationController),
    ("_hands_free", ConsoleHandsFreeController),
    ("_realtime", ConsoleRealtimeController),
    ("_message", ConsoleMessageController),
    ("_prompts", ConsolePromptsController),
    ("_agent", ConsoleAgentController),
    ("_terminal", ConsoleTerminalController),
    ("_prompt_queue", ConsolePromptQueueUIController),
    ("_review_selection", ConsoleReviewSelectionController),
    ("_send_price", ConsoleSendPriceController),
]

#: Every controller takes `chat_store_accessor=lambda: self._ensure_console_
#: chat_store()`; it is stored as `_chat_store_accessor`. The one dependency
#: shared by all six, so it is the cheapest uniform identity probe.
_SHARED_ACCESSOR = "_chat_store_accessor"


def _unmounted_console() -> ChatScreen:
    """A real, unmounted `ChatScreen`.

    The wiring under test runs entirely in `__init__`, so no pilot/mount is
    needed -- this mirrors the `ChatScreen(app)` idiom already used across
    `test_console_internals_decomposition.py`,
    `test_console_message_controller.py` and friends.
    """
    return ChatScreen(_build_test_app())


def test_all_six_controllers_are_constructed_with_the_right_classes():
    screen = _unmounted_console()

    for attr, cls in _EXPECTED_SLOTS:
        controller = getattr(screen, attr, None)
        assert controller is not None, f"{attr} was never wired"
        assert isinstance(controller, cls), (
            f"{attr} is {type(controller).__name__}, expected {cls.__name__}"
        )


def test_all_eighteen_controllers_are_constructed_with_the_right_classes() -> None:
    screen = _unmounted_console()
    names = [attr for attr, _ in _ALL_CONTROLLER_SLOTS]
    observed = [key for key in vars(screen) if key in set(names)]

    for attr, cls in _ALL_CONTROLLER_SLOTS:
        controller = getattr(screen, attr, None)
        assert controller is not None, f"{attr} was never wired"
        assert isinstance(controller, cls), (
            f"{attr} is {type(controller).__name__}, expected {cls.__name__}"
        )
    assert observed == names, f"controller build order changed: {observed}"


def test_terminal_controller_is_wired_to_late_bound_app_and_console_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screen = _unmounted_console()
    controller = getattr(screen, "_terminal", None)
    assert isinstance(controller, ConsoleTerminalController), (
        "_terminal was never wired"
    )

    manager = object()
    workspace = object()
    selected_root = Path("/late-bound-terminal-root")
    settings_calls: list[None] = []
    screen.app_instance.terminal_session_manager = manager
    screen._console_terminal_workspace = workspace
    screen._open_terminal_privacy_settings = lambda: settings_calls.append(None)
    monkeypatch.setattr(
        wiring_module,
        "_selected_console_local_root",
        lambda current_screen, session_id=None: (
            selected_root,
            current_screen,
            session_id,
        ),
    )

    assert controller._terminal_runtime() is manager
    assert controller._workspace_accessor() is workspace
    assert controller._selected_local_root() == (selected_root, screen, None)
    assert screen._raw_cli._selected_local_root("raw-session") == (
        selected_root,
        screen,
        "raw-session",
    )
    replacement_home = Path("/late-bound-terminal-home")
    monkeypatch.setattr(
        Path,
        "home",
        classmethod(lambda _path_class: replacement_home),
    )
    assert controller._account_home() == replacement_home
    controller._open_privacy_settings()
    assert settings_calls == [None]


def _selected_root_screen(
    root: Path,
    *,
    status: RuntimeBindingStatus | str = RuntimeBindingStatus.READY,
    workspace_id: str = "workspace-one",
    binding_kind: RuntimeBindingKind | str = RuntimeBindingKind.LOCAL_FILESYSTEM,
) -> SimpleNamespace:
    session = SimpleNamespace(
        id="session-one",
        workspace_id="workspace-one",
        project_instruction_state=SimpleNamespace(
            working_folder_binding_id="binding-one"
        ),
    )
    store = SimpleNamespace(
        active_session_id=session.id,
        sessions=lambda: (session,),
    )
    binding = SimpleNamespace(
        workspace_id=workspace_id,
        binding_kind=binding_kind,
        status=status,
        locator=str(root),
    )
    registry = SimpleNamespace(get_runtime_binding=lambda _binding_id: binding)
    return SimpleNamespace(
        _ensure_console_chat_store=lambda: store,
        app_instance=SimpleNamespace(workspace_registry_service=registry),
    )


@pytest.mark.parametrize(
    "status",
    [RuntimeBindingStatus.READY, RuntimeBindingStatus.READY.value],
    ids=["enum", "value"],
)
def test_selected_console_local_root_accepts_only_ready_status_forms(
    tmp_path: Path,
    status: RuntimeBindingStatus | str,
) -> None:
    screen = _selected_root_screen(tmp_path, status=status)

    assert wiring_module._selected_console_local_root(screen) == tmp_path


@pytest.mark.parametrize(
    "status",
    [
        *(
            status
            for status in RuntimeBindingStatus
            if status is not RuntimeBindingStatus.READY
        ),
        "stale",
    ],
    ids=lambda status: getattr(status, "value", status),
)
def test_selected_console_local_root_rejects_every_non_ready_status(
    tmp_path: Path,
    status: RuntimeBindingStatus | str,
) -> None:
    screen = _selected_root_screen(tmp_path, status=status)

    assert wiring_module._selected_console_local_root(screen) is None


def test_selected_console_local_root_rejects_wrong_workspace(tmp_path: Path) -> None:
    screen = _selected_root_screen(tmp_path, workspace_id="workspace-two")

    assert wiring_module._selected_console_local_root(screen) is None


@pytest.mark.parametrize(
    "binding_kind",
    [
        kind
        for kind in RuntimeBindingKind
        if kind is not RuntimeBindingKind.LOCAL_FILESYSTEM
    ],
    ids=lambda kind: kind.value,
)
def test_selected_console_local_root_rejects_non_local_bindings(
    tmp_path: Path,
    binding_kind: RuntimeBindingKind,
) -> None:
    screen = _selected_root_screen(tmp_path, binding_kind=binding_kind)

    assert wiring_module._selected_console_local_root(screen) is None


def test_raw_cli_worker_adapter_uses_its_own_group() -> None:
    screen = _unmounted_console()
    worker = object()
    screen.run_worker = MagicMock(return_value=worker)

    result = screen._raw_cli._start_worker(
        "job",
        thread=True,
        exclusive=True,
        name="console-raw-cli-test",
    )

    assert result is worker
    screen.run_worker.assert_called_once_with(
        "job",
        group="console-raw-cli",
        thread=True,
        exclusive=True,
        name="console-raw-cli-test",
    )


def test_review_selection_controller_is_late_bound_without_sibling_objects(
    monkeypatch,
) -> None:
    screen = _unmounted_console()
    controller = screen._review_selection

    assert controller.annotation_loaded_conversation is None
    assert controller.annotation_previews == {}
    assert controller.selection_feedback_inflight is False

    store = object()
    conversation_id = "conversation"
    provider = object()
    roots = object()
    runs_db = object()
    bindings = object()
    native_messages = object()
    screen._ensure_console_chat_store = lambda: store
    active_status = next(iter(FEEDBACK_ACTIVE_RUN_STATUSES))
    chat_controller = SimpleNamespace(
        store=SimpleNamespace(active_session_id="session"),
        run_state=SimpleNamespace(status=active_status),
        _agent_conversation_id=lambda session_id: (conversation_id, session_id),
        run_active_for_workspace=lambda root: ("active", root),
        resolve_turn_execution_context=lambda session_id: SimpleNamespace(
            workspace_roots=(roots, session_id)
        ),
    )
    screen._console_chat_controller = chat_controller
    screen._ensure_console_chat_controller = lambda: chat_controller
    screen._ensure_console_agent_bridge = lambda: SimpleNamespace(
        runs_db=runs_db,
        change_review_provider=lambda value: (provider, value),
    )
    screen._console_runtime = lambda: SimpleNamespace(chat_controller=chat_controller)
    monkeypatch.setattr(
        wiring_module,
        "build_capture_policy_bindings",
        lambda controller, session_id, conv_id: (
            bindings,
            controller,
            session_id,
            conv_id,
        ),
    )
    screen._message._native_console_messages = lambda: native_messages

    assert controller._store_accessor() is store
    assert controller._agent_conversation_id_accessor() == (
        conversation_id,
        "session",
    )
    assert controller._change_review_provider_accessor("conversation") == (
        provider,
        "conversation",
    )
    assert controller._run_active_accessor() is True
    assert controller._run_active_for_root("root") == ("active", "root")
    assert controller._workspace_roots_accessor() == (roots, "session")
    assert controller._agent_runs_db_accessor() is runs_db
    assert controller._capture_policy_bindings_accessor("session", "conversation") == (
        bindings,
        chat_controller,
        "session",
        "conversation",
    )
    assert controller._native_messages_accessor() is native_messages


def test_send_price_controller_is_constructed_with_late_bound_screen_edges() -> None:
    screen = _unmounted_console()
    controller = getattr(screen, "_send_price", None)
    assert isinstance(controller, ConsoleSendPriceController), (
        "_send_price was never wired"
    )

    settings = object()
    store = object()
    launch = object()
    projection = object()
    screen._session._ensure_active_console_session_settings = lambda: settings
    screen._console_chat_store = store
    screen._pending_console_launch_context = launch
    screen._ensure_console_chat_controller = lambda: SimpleNamespace(
        provider_messages_for_next_send_estimate=(
            lambda session_id: (projection, session_id)
        )
    )

    assert controller._settings_accessor() is settings
    assert controller._chat_store_accessor() is store
    assert controller._pending_launch_accessor() is launch
    assert controller._provider_history_accessor("session-1") == (
        projection,
        "session-1",
    )


def test_realtime_controller_is_wired_late_bound_with_empty_owned_state() -> None:
    screen = _unmounted_console()
    controller = screen._realtime

    assert type(controller) is ConsoleRealtimeController
    assert controller.session is None
    assert controller.close_worker is None

    session_settings = object()
    chat_store = object()
    runtime = object()
    dictation_state = object()
    pipeline_blocker = object()
    recorder_factory = object()
    provider_session_factory = object()
    sink_factory = object()
    ui_thread_id = object()
    event_loop = object()
    interval = object()
    worker = object()
    screen._session = SimpleNamespace(
        _ensure_active_console_session_settings=lambda: session_settings
    )
    screen._ensure_console_chat_store = lambda: chat_store
    screen._console_runtime = lambda: runtime
    screen._console_dictation_state = dictation_state

    dictation_stop_calls: list[None] = []
    pipeline_loop_calls: list[bool] = []
    screen._request_console_dictation_stop = lambda: dictation_stop_calls.append(None)
    screen._hands_free = SimpleNamespace(
        _console_pipeline_hands_free_blocker=lambda: pipeline_blocker,
        _enter_console_hands_free_pipeline_loop=(
            lambda *, capture_live: pipeline_loop_calls.append(capture_live)
        ),
    )
    screen.app_instance.console_realtime_recorder_factory = recorder_factory
    screen.app_instance.console_realtime_session_factory = provider_session_factory
    screen.app_instance.console_realtime_sink_factory = sink_factory
    notify = MagicMock()
    screen.app_instance.notify = notify
    screen.app_instance._thread_id = ui_thread_id
    screen.app_instance._loop = event_loop
    screen.set_interval = MagicMock(return_value=interval)
    screen.run_worker = MagicMock(return_value=worker)
    screen.call_later = MagicMock()
    native_sync = object()
    screen._sync_native_console_chat_ui = native_sync
    screen._repaint_console_realtime_chip = MagicMock()
    screen._restore_console_voice_chip = MagicMock()

    assert controller._ensure_session_settings() is session_settings
    assert controller._chat_store_accessor() is chat_store
    assert controller._runtime_accessor() is runtime
    assert controller._dictation_state_accessor() is dictation_state
    assert controller._pipeline_blocker() is pipeline_blocker
    assert controller._recorder_factory_accessor() is recorder_factory
    assert controller._provider_session_factory_accessor() is provider_session_factory
    assert controller._sink_factory_accessor() is sink_factory
    assert controller._ui_thread_id_accessor() is ui_thread_id
    assert controller._event_loop_accessor() is event_loop

    controller._request_dictation_stop()
    controller._enter_pipeline_loop(True)
    controller._notify("copy", severity="warning")
    assert controller._set_interval(0.1, "tick") is interval
    assert (
        controller._run_worker(
            "job",
            exclusive=True,
            group="console-realtime-test",
        )
        is worker
    )
    controller._defer_native_sync()
    controller._repaint_chip()
    controller._restore_voice_chip()

    assert dictation_stop_calls == [None]
    assert pipeline_loop_calls == [True]
    notify.assert_called_once_with("copy", severity="warning")
    screen.set_interval.assert_called_once_with(0.1, "tick")
    screen.run_worker.assert_called_once_with(
        "job",
        exclusive=True,
        group="console-realtime-test",
    )
    screen.call_later.assert_called_once_with(native_sync)
    screen._repaint_console_realtime_chip.assert_called_once_with()
    screen._restore_console_voice_chip.assert_called_once_with()


def test_realtime_outgoing_edges_target_controller_after_method_move() -> None:
    screen = _unmounted_console()
    transcript_calls: list[str] = []
    entry_calls: list[bool] = []
    screen._realtime._console_realtime_adopt_transcript = lambda transcript: (
        transcript_calls.append(transcript) or True
    )
    screen._realtime._enter_console_realtime_loop = lambda *, capture_live: (
        entry_calls.append(capture_live)
    )

    assert screen._dictation._console_realtime_adopt_transcript("late words") is True
    screen._hands_free._enter_console_realtime_loop(capture_live=True)

    assert transcript_calls == ["late words"]
    assert entry_calls == [True]


def test_fleet_controller_is_constructed_with_late_bound_screen_edges() -> None:
    screen = _unmounted_console()
    controller = getattr(screen, "_fleet", None)
    assert isinstance(controller, ConsoleFleetLifecycleController), (
        "_fleet was never wired"
    )

    composer = SimpleNamespace(draft_text=lambda: " replacement draft ")
    screen._console_composer_or_none = lambda: composer
    displayed_draft = controller._displayed_composer_draft_accessor()
    assert displayed_draft == " replacement draft "
    assert displayed_draft is not composer
    assert controller._console_wake_user_priority("session-a") is True

    pending_handoffs = object()
    sessions = (SimpleNamespace(id="late-session"),)
    store = SimpleNamespace(
        active_session_id="late-session",
        sessions=lambda: sessions,
    )
    screen.app_instance.pending_handoffs = pending_handoffs
    screen._ensure_console_chat_store = lambda: store
    screen._console_chat_store = store

    assert controller._pending_handoffs_accessor() is pending_handoffs
    assert controller._ensure_chat_store() is store
    assert controller._active_session_id_accessor() == "late-session"
    assert controller._chat_sessions_accessor() is sessions

    chat_controller = object()
    screen._ensure_console_chat_controller = lambda: chat_controller
    assert controller._ensure_chat_controller() is chat_controller

    def raise_controller_error() -> None:
        raise RuntimeError("replacement controller unavailable")

    screen._ensure_console_chat_controller = raise_controller_error
    with pytest.raises(RuntimeError, match="replacement controller unavailable"):
        controller._ensure_chat_controller()

    wake_calls: list[object] = []
    wake = SimpleNamespace(
        wire=lambda **kwargs: wake_calls.append(("wire", kwargs.get("app"))) or True,
        seed_from_marks=lambda: wake_calls.append("seed") or True,
        retry_soon=lambda: wake_calls.append("retry"),
        has_pending=lambda conversation_id: conversation_id == "conversation-a",
        delivering_conversation_id=lambda: "conversation-a",
    )
    screen._console_chat_controller = SimpleNamespace(
        fleet_wake=wake,
        fleet_has_unsettled_children=lambda: True,
    )

    assert controller._chat_controller_available() is True
    assert controller._wire_wake_coordinator() is True
    assert controller._seed_wake_from_marks() is True
    controller._retry_wake_soon()
    assert controller._wake_has_pending("conversation-a") is True
    assert controller._wake_delivering_conversation_id() == "conversation-a"
    assert controller._fleet_has_unsettled_children() is True
    assert wake_calls == [("wire", screen.app_instance), "seed", "retry"]


@pytest.mark.asyncio
async def test_session_first_chat_edges_are_late_bound_and_presentation_only(
    monkeypatch,
) -> None:
    screen = _unmounted_console()
    controller = screen._session
    screen._console_control_provider = "late-provider"
    screen._console_control_model = "late-model"
    focus_token = MagicMock()
    focus_token.is_mounted = True

    assert controller._screen_mounted_accessor() is False
    assert controller._first_chat_presentation_snapshot_fn() == (
        "late-provider",
        "late-model",
        None,
    )
    controller._apply_first_chat_control_selection_fn("next-provider", "next-model")
    assert (screen._console_control_provider, screen._console_control_model) == (
        "next-provider",
        "next-model",
    )
    controller._restore_first_chat_focus_fn(focus_token)
    focus_token.focus.assert_not_called()

    host = ConsolidatedCSSApp()
    async with host.run_test(size=(120, 40)) as pilot:
        await host.push_screen(screen)
        await pilot.pause()
        assert screen.is_attached is True
        assert controller._screen_mounted_accessor() is True

        mounted_focus_token = screen.query_one("#console-native-composer")
        focus_spy = MagicMock(wraps=mounted_focus_token.focus)
        monkeypatch.setattr(mounted_focus_token, "focus", focus_spy)
        controller._restore_first_chat_focus_fn(mounted_focus_token)
        focus_spy.assert_called_once_with()

        await host.pop_screen()
        await pilot.pause()
        assert screen.is_attached is False


@pytest.mark.asyncio
async def test_session_first_chat_focus_ignores_opaque_token() -> None:
    screen = _unmounted_console()
    controller = screen._session
    opaque_token = SimpleNamespace(focus=MagicMock())

    controller._restore_first_chat_focus_fn(opaque_token)
    opaque_token.focus.assert_not_called()

    host = ConsolidatedCSSApp()
    async with host.run_test(size=(120, 40)) as pilot:
        await host.push_screen(screen)
        await pilot.pause()

        controller._restore_first_chat_focus_fn(opaque_token)
        opaque_token.focus.assert_not_called()


def test_retrieval_controller_is_constructed_with_late_bound_screen_edges():
    """Wave 6 wires one retrieval owner without freezing screen state."""
    screen = _unmounted_console()

    assert isinstance(screen._retrieval, ConsoleRetrievalController)
    sentinel = object()
    screen._character._current_console_rail_conversation_id = lambda: sentinel
    assert screen._retrieval._current_conversation_id() is sentinel


def test_character_controller_is_constructed_with_late_bound_screen_edges():
    """Wave 6 wires `_character` without changing the six-slot contract."""
    screen = _unmounted_console()

    assert isinstance(screen._character, ConsoleCharacterController)
    native_session = object()
    conversation_id = object()
    default_settings = object()
    character_db = object()
    screen._session = SimpleNamespace(
        _active_native_console_session=lambda: native_session,
        _current_console_conversation_id=lambda: conversation_id,
        _default_console_session_settings=lambda: default_settings,
    )
    screen.app_instance.chachanotes_db = character_db

    assert screen._character._active_native_session_accessor() is native_session
    assert screen._character._current_conversation_id_accessor() is conversation_id
    assert screen._character._default_session_settings() is default_settings
    assert screen._character._character_db_accessor() is character_db


def test_skill_controller_is_constructed_with_late_bound_screen_edges():
    """Wave 6 wires `_skill` without changing the original six-slot contract."""
    screen = _unmounted_console()

    assert isinstance(screen._skill, ConsoleSkillController)
    sentinel = object()
    screen._task_resume_state = sentinel
    assert screen._skill._task_resume_state() is sentinel

    calls: list[str] = []
    screen._sync_console_command_popup = lambda: calls.append("replacement")
    screen._skill._sync_console_command_popup()
    assert calls == ["replacement"]


def test_controllers_are_built_in_the_documented_order():
    """The six slots appear in `vars(screen)` in build order.

    Order is load-bearing documentation (every cross-controller lambda
    resolves its sibling at call time precisely so it *need not* be
    load-bearing behaviour) -- pin it so a reshuffle is a deliberate,
    visible act rather than a silent side effect of moving the block.
    """
    screen = _unmounted_console()

    names = [attr for attr, _ in _EXPECTED_SLOTS]
    observed = [key for key in vars(screen) if key in set(names)]

    assert observed == names, f"controller build order changed: {observed}"


def test_every_controller_holds_the_same_app_instance_as_the_screen():
    screen = _unmounted_console()

    for attr, _ in _EXPECTED_SLOTS:
        controller = getattr(screen, attr)
        assert controller.app_instance is screen.app_instance, (
            f"{attr}.app_instance is not the screen's app_instance"
        )


def test_shared_chat_store_accessor_resolves_to_the_screens_own_store():
    """Identity, not truthiness: all six must reach the SAME store object."""
    screen = _unmounted_console()

    expected = screen._ensure_console_chat_store()
    assert expected is not None, "screen produced no chat store to compare against"

    for attr, _ in _EXPECTED_SLOTS:
        accessor = getattr(getattr(screen, attr), _SHARED_ACCESSOR)
        assert accessor() is expected, (
            f"{attr}.{_SHARED_ACCESSOR}() is not the screen's chat store"
        )


def test_shared_chat_store_accessor_is_late_bound():
    """An instance-level replacement made AFTER construction must be seen."""
    screen = _unmounted_console()
    sentinel = object()
    screen._ensure_console_chat_store = lambda: sentinel

    for attr, _ in _EXPECTED_SLOTS:
        accessor = getattr(getattr(screen, attr), _SHARED_ACCESSOR)
        assert accessor() is sentinel, (
            f"{attr}.{_SHARED_ACCESSOR} froze the constructor-time method"
        )


@pytest.mark.parametrize(
    "attr,accessor_name",
    [
        ("_session", "_composer_accessor"),
        ("_dictation", "_composer_accessor"),
        ("_hands_free", "_composer_accessor"),
        ("_prompts", "_composer_accessor"),
    ],
)
def test_composer_accessor_is_late_bound(attr, accessor_name):
    screen = _unmounted_console()
    sentinel = object()
    screen._console_composer_or_none = lambda: sentinel

    accessor = getattr(getattr(screen, attr), accessor_name)
    assert accessor() is sentinel, f"{attr}.{accessor_name} froze at construction"


@pytest.mark.parametrize(
    "attr,accessor_name",
    [
        ("_workspace", "_current_chat_store_accessor"),
        ("_session", "_current_chat_store_accessor"),
        ("_message", "_current_chat_store_accessor"),
    ],
)
def test_current_chat_store_accessor_reads_the_live_attribute(attr, accessor_name):
    """`lambda: self._console_chat_store` -- a bare attribute read, late."""
    screen = _unmounted_console()
    sentinel = object()
    screen._console_chat_store = sentinel

    accessor = getattr(getattr(screen, attr), accessor_name)
    assert accessor() is sentinel, f"{attr}.{accessor_name} snapshotted the value"


def test_workspace_resolves_the_session_sibling_at_call_time():
    """`_workspace` is built BEFORE `_session`; the lambda must still work.

    Swapping `screen._session` wholesale after construction and seeing the
    workspace accessor follow is the proof that the sibling is looked up per
    call -- the property that makes the six constructions order-independent.
    """
    screen = _unmounted_console()
    sentinel = object()
    screen._session = SimpleNamespace(_current_console_conversation_id=lambda: sentinel)

    assert screen._workspace._current_conversation_id_accessor() is sentinel


def test_dictation_resolves_the_hands_free_sibling_at_call_time():
    """`_dictation` is built BEFORE `_hands_free` -- same proof, other way."""
    screen = _unmounted_console()
    sentinel = object()
    screen._hands_free = SimpleNamespace(_console_hands_free=sentinel)

    assert screen._dictation._hands_free_session_accessor() is sentinel


def test_dictation_session_late_binds_the_app_owned_service_factory():
    """The session must follow app factory replacement after screen wiring."""
    screen = _unmounted_console()
    first = object()
    second = object()
    calls: list[tuple[object, dict]] = []

    def first_factory(**kwargs):
        calls.append((first, kwargs))
        return first

    def second_factory(**kwargs):
        calls.append((second, kwargs))
        return second

    screen.app_instance._create_console_dictation_service = first_factory
    session = screen._dictation._create_console_dictation_session()
    assert session._build_service(language="en") is first

    screen.app_instance._create_console_dictation_service = second_factory
    assert session._build_service(language="fr") is second
    assert calls == [
        (
            first,
            {
                "language": "en",
                "max_buffer_bytes": 1_920_000,
            },
        ),
        (
            second,
            {
                "language": "fr",
                "max_buffer_bytes": 1_920_000,
            },
        ),
    ]


def test_hands_free_reads_dictation_state_through_the_screen_at_call_time():
    screen = _unmounted_console()
    sentinel = object()
    screen._dictation = SimpleNamespace(_console_dictation_state=sentinel)

    assert screen._hands_free._dictation_state_accessor() is sentinel


@pytest.mark.asyncio
async def test_hands_free_auto_speak_edges_are_late_bound_and_mount_safe(monkeypatch):
    """Wave 6 keeps policy on HandsFree without freezing its sibling edge.

    Args:
        monkeypatch: Pytest fixture used to simulate teardown and capture logging.
    """
    screen = _unmounted_console()
    controller = screen._hands_free
    requests: list[tuple[str, object | None]] = []
    screen._console_auto_speak = SimpleNamespace(
        request_enabled=lambda enabled: requests.append(("enabled", enabled)),
        request_resume=lambda: requests.append(("resume", None)),
        request_retry=lambda: requests.append(("retry", None)),
    )

    controller.on_console_auto_speak_changed(SimpleNamespace(enabled=True))
    controller.on_console_auto_speak_resume_requested(SimpleNamespace())
    controller.on_console_auto_speak_retry_requested(SimpleNamespace())

    destination = object()

    async def resolve_destination(assistant_kind, character_ref):
        assert assistant_kind == "character"
        assert character_ref is None
        return destination

    async def ensure_handler():
        return SimpleNamespace(
            resolve_console_speech_destination=resolve_destination,
        )

    screen.app_instance._ensure_tts_handler = ensure_handler
    assert (
        await controller._resolve_console_auto_speak_destination("character", None)
        is destination
    )

    async def fail_destination(assistant_kind, character_ref):
        raise RuntimeError("destination failed")

    async def ensure_failing_handler():
        return SimpleNamespace(
            resolve_console_speech_destination=fail_destination,
        )

    screen.app_instance._ensure_tts_handler = ensure_failing_handler
    logger = MagicMock()
    monkeypatch.setattr(hands_free_module, "logger", logger)
    assert (
        await controller._resolve_console_auto_speak_destination("character", None)
        is None
    )
    logger.opt.assert_called_once_with(exception=True)
    logger.opt.return_value.warning.assert_called_once_with(
        "Failed to resolve the Console auto-speak destination."
    )

    # Both presentation paths must remain harmless before the screen mounts.
    controller._sync_console_auto_speak_controls(True, False, False)
    controller._sync_hands_free_switch(True)

    # Teardown can leave a stale query root that raises a broader QueryError.
    monkeypatch.setattr(screen, "query_one", MagicMock(side_effect=QueryError()))
    controller._sync_console_auto_speak_controls(False, True, True)
    controller._sync_hands_free_switch(False)

    assert requests == [
        ("enabled", True),
        ("resume", None),
        ("retry", None),
    ]


def test_prompts_resolves_the_session_sibling_at_call_time():
    screen = _unmounted_console()
    sentinel = object()
    screen._session = SimpleNamespace(
        _ensure_active_console_session_settings=lambda: sentinel
    )

    assert screen._prompts._ensure_active_console_session_settings_fn() is sentinel
