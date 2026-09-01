"""Ownership and LIFETIME pins for the app-owned Console runtime (task-15860).

`test_console_runtime_is_the_single_construction_site` pins the ownership
move: if a later change re-adds a `ConsoleChatStore(...)` or
`ConsoleChatController(...)` anywhere but `Chat/console_runtime.py`, the app
stops being the owner and Design A quietly loses its premise.

The rest pin the lifetime landing. `test_second_console_visit_reuses_the_
runtime` **replaces** Task 1's `test_second_console_visit_gets_a_new_
runtime`, which asserted the opposite and whose docstring said it must be
rewritten here rather than deleted -- that is the whole reason it existed.

`test_a_terminal_run_state_after_leaving_does_not_reach_the_dead_screen` is
this landing's defect red for the attach/detach seam. Before the seam
existed, every screen-owned hook slot stayed bound to the unmounted
`ChatScreen` (Task 0's P3 measured five of them still bound and none
raising), so a run settling after the navigation called straight into a
dead view.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.events import Key

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _wait_for_selector
from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus
from tldw_chatbook.Chat.console_runtime import (
    CONSOLE_RUNTIME_ATTR,
    CONSOLE_VIEW_HOOK_SLOTS,
    ConsoleRuntime,
)
from tldw_chatbook.Chat.console_scratch_space import ConsoleScratchSpaceManager
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Persona_Buddy.console_adapter import PersonaBuddyConsoleAdapter
from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController
from tldw_chatbook.UI.Console_Modules.fleet import (
    ConsoleFleetLifecycleController,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_composer_bar import (
    ConsoleComposerBar,
    classify_console_raw_draft,
)

#: Constructor calls that must exist in exactly one place: the runtime.
#: `ConsoleProviderGateway(` is deliberately NOT here -- the Personas
#: preview controller builds its own, unrelated to the Console runtime
#: (`UI/Persona_Modules/personas_preview_controller.py`).
_RUNTIME_OWNED_CONSTRUCTIONS = ("ConsoleChatStore", "ConsoleChatController")

_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook"
_RUNTIME_MODULE = "tldw_chatbook/Chat/console_runtime.py"


@pytest.mark.unit
def test_console_runtime_owns_one_screen_free_persona_buddy_sink():
    """The app-owned runtime, not a screen, retains the trusted sink."""
    app = type("App", (), {})()
    app.persona_buddy_controller = PersonaBuddyController()
    runtime = ConsoleRuntime(app)

    assert isinstance(runtime.persona_buddy_sink, PersonaBuddyConsoleAdapter)
    assert runtime.persona_buddy_sink is runtime.persona_buddy_sink
    assert "view" not in vars(runtime.persona_buddy_sink)


@pytest.mark.unit
def test_console_runtime_reuses_one_scratch_manager_across_console_visits():
    runtime = ConsoleRuntime(type("App", (), {})())
    first = runtime.scratch_spaces

    runtime.detach_view(None)

    assert runtime.scratch_spaces is first


@pytest.mark.asyncio
async def test_runtime_injects_its_scratch_manager_into_chat_controller():
    runtime = ConsoleRuntime(type("App", (), {})())

    controller = runtime.ensure_chat_controller(
        store=ConsoleChatStore(),
        provider_gateway=object(),
    )

    assert controller._scratch_spaces is runtime.scratch_spaces
    assert controller._owns_scratch_spaces is False
    await runtime.dispose()


@pytest.mark.asyncio
async def test_leaving_console_preserves_live_session_scratch(tmp_path):
    runtime = ConsoleRuntime(type("App", (), {})())
    runtime._scratch_spaces = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    snapshot = runtime.scratch_spaces.snapshot("session-a")

    assert await runtime.leave_console() is True

    assert runtime.scratch_spaces.is_live(snapshot)
    assert snapshot.root.is_dir()
    await runtime.dispose()
    assert not snapshot.root.exists()


@pytest.mark.asyncio
async def test_raw_cli_refusal_bank_survives_leave_and_clears_on_dispose():
    runtime = ConsoleRuntime(type("App", (), {})())
    stash = object()
    bank = runtime.raw_cli_refusal_stash_bank
    bank["session-a"] = [stash]

    assert runtime.accepts_raw_cli_refusal_callbacks is True
    assert await runtime.leave_console() is True
    assert runtime.accepts_raw_cli_refusal_callbacks is True
    assert runtime.raw_cli_refusal_stash_bank is bank
    assert bank == {"session-a": [stash]}

    await runtime.dispose()
    assert runtime.accepts_raw_cli_refusal_callbacks is False
    assert bank == {}


@pytest.mark.asyncio
async def test_runtime_tombstones_before_shutdown_and_disposes_via_to_thread(
    monkeypatch,
):
    events: list[str] = []

    class ScratchSpaces:
        def tombstone_all(self) -> None:
            events.append("scratch-tombstone")

        def dispose(self) -> bool:
            events.append("scratch-dispose")
            return True

    class Controller:
        async def shutdown(self) -> None:
            events.append("controller-shutdown")

    class Gateway:
        async def aclose(self) -> None:
            events.append("gateway-close")

    async def fake_to_thread(function, *args, **kwargs):
        events.append("to-thread")
        return function(*args, **kwargs)

    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_runtime.asyncio.to_thread",
        fake_to_thread,
    )
    runtime = ConsoleRuntime(type("App", (), {})())
    runtime._scratch_spaces = ScratchSpaces()
    runtime._chat_controller = Controller()
    runtime._provider_gateway = Gateway()

    await runtime.dispose()

    assert events == [
        "scratch-tombstone",
        "controller-shutdown",
        "to-thread",
        "scratch-dispose",
        "gateway-close",
    ]


@pytest.mark.asyncio
async def test_persona_buddy_release_follows_controller_wake_disposal():
    """Runtime shutdown terminally fences wake producers before sink release."""
    events: list[str] = []

    class Sink:
        def dispose(self) -> None:
            events.append("sink-release")

    class Controller:
        async def shutdown(self) -> None:
            events.append("wake-dispose")

    runtime = ConsoleRuntime(type("App", (), {})())
    runtime._persona_buddy_sink = Sink()
    runtime._chat_controller = Controller()
    runtime._provider_gateway = None

    await runtime.dispose()

    assert events == ["wake-dispose", "sink-release"]


def _construction_sites(class_name: str) -> list[str]:
    """Every `<path>:<line>` in the shipped package CALLING `class_name(...)`.

    AST-based on purpose (PR #1648 follow-up): a raw substring scan counts
    innocuous mentions in comments, docstrings, and string literals as
    construction sites and false-fails the pin. Only actual call nodes
    count here.
    """
    sites: list[str] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, UnicodeDecodeError, SyntaxError):  # pragma: no cover
            continue
        rel = path.relative_to(_PACKAGE_ROOT.parent).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if called == class_name:
                line = source.splitlines()[node.lineno - 1].strip()
                sites.append(f"{rel}:{node.lineno}: {line}")
    return sites


@pytest.mark.unit
def test_console_runtime_is_the_single_construction_site():
    """PIN (characterization): only the runtime builds the store/controller."""
    for token in _RUNTIME_OWNED_CONSTRUCTIONS:
        sites = _construction_sites(token)
        assert sites, f"{token} vanished entirely -- the pin is stale."
        foreign = [site for site in sites if not site.startswith(_RUNTIME_MODULE)]
        assert not foreign, (
            f"{token} is constructed outside {_RUNTIME_MODULE}; the app is no "
            "longer the Console runtime's owner:\n  " + "\n  ".join(foreign)
        )


@pytest.mark.unit
def test_attach_and_detach_cover_exactly_the_same_slot_set():
    """The ONE enumerated list runs in both directions.

    `ChatScreen.console_view_hooks()` is what `attach_view` sets and
    `CONSOLE_VIEW_HOOK_SLOTS` is what `detach_view` clears. A slot present
    in one and not the other is either bound and never cleared (a dead
    screen kept alive, answering questions about a view that is gone) or
    cleared and never bound (a live Console with a silently dead hook).
    """
    screen = ChatScreen.__new__(ChatScreen)

    def no_op(*_args, **_kwargs):
        return None

    screen._fleet = ConsoleFleetLifecycleController(
        **{
            name: no_op
            for name in inspect.signature(ConsoleFleetLifecycleController).parameters
        }
    )
    screen._library_activity = SimpleNamespace(build_provider=no_op)
    declared = {slot.name for slot in CONSOLE_VIEW_HOOK_SLOTS}
    provided = set(ChatScreen.console_view_hooks(screen))

    assert provided == declared, (
        "attach-set != detach-set; only in the view: "
        f"{sorted(provided - declared)}; only in the slot list: "
        f"{sorted(declared - provided)}"
    )
    assert len(CONSOLE_VIEW_HOOK_SLOTS) == len(declared), "duplicate slot name"


@pytest.mark.asyncio
async def test_second_console_visit_reuses_the_runtime(tmp_path):
    """The runtime SURVIVES leaving Console -- this landing's central change.

    Replaces Task 1's `test_second_console_visit_gets_a_new_runtime`, which
    pinned the opposite (dispose-at-unmount) and said in its own docstring
    that it must be rewritten here.
    """
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    terminal_manager = app.terminal_session_manager

    async with app.run_test(size=(160, 48)) as pilot:
        chat = ChatScreen(app)
        await app.push_screen(chat)
        app._initial_screen_pushed = True
        app.current_tab = "chat"
        await pilot.pause()
        await _wait_for_selector(chat, pilot, "#console-native-composer")

        controller_one = chat._ensure_console_chat_controller()
        runtime_one = getattr(app, CONSOLE_RUNTIME_ATTR, None)
        assert isinstance(runtime_one, ConsoleRuntime), type(runtime_one).__name__
        # The app -- not the screen -- is what built these.
        assert runtime_one.chat_controller is controller_one
        assert runtime_one.chat_store is chat._console_chat_store
        assert runtime_one.provider_gateway is chat._console_provider_gateway
        bridge_one = runtime_one.agent_bridge
        assert bridge_one is not None, (
            "the real-on-disk-DB rig must actually build an agent bridge, "
            "or this test cannot say anything about bridge identity"
        )
        assert bridge_one is chat._console_agent_bridge
        store_one = runtime_one.chat_store
        visit_one_event = controller_one._shutdown_requested
        assert runtime_one.generation == 0

        # ---- leave Console through the real navigation API ---------------
        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()
        assert chat not in app.screen_stack, "Console must actually unmount"
        # The VISIT ended: its cancellation Event is set (which is also what
        # keeps `_attempt`'s wake gate refusing while nothing is mounted)...
        assert visit_one_event.is_set()
        # ...every screen-owned slot is back at its viewless default...
        assert controller_one.notify_run_outcome is None
        assert controller_one.fleet_wake.delivery_ui_hook is None
        # task-15860 Task 4: the view probe's viewless default is NOT None
        # (its read site reads an unwired probe as IN VIEW). Asserted here
        # through the DECISION the production path makes, after a real
        # navigation -- an unwatched delivery must not be able to report
        # itself as watched and clear the ◈ mark.
        assert (
            controller_one.fleet_wake._conversation_in_view(
                "conv-anything", "sess-anything"
            )
            is False
        ), (
            "with Console genuinely unmounted the runtime still reported the "
            "conversation as being watched"
        )
        assert runtime_one.view is None
        # ...and the runtime itself is untouched and still the app's.
        assert runtime_one.generation == 0, "leaving Console must NOT dispose"
        assert getattr(app, CONSOLE_RUNTIME_ATTR, None) is runtime_one
        assert runtime_one.chat_controller is controller_one
        assert runtime_one.provider_gateway is not None, (
            "the gateway is app-owned now and must not be closed/dropped "
            "on a navigation"
        )

        # ---- return to Console -------------------------------------------
        await app.handle_screen_navigation(NavigateToScreen("chat"))
        await pilot.pause()
        chat_two = app.screen
        assert isinstance(chat_two, ChatScreen), type(chat_two).__name__
        assert chat_two is not chat, "screens are never cached"
        await _wait_for_selector(chat_two, pilot, "#console-native-composer")

        controller_two = chat_two._ensure_console_chat_controller()
        runtime_two = getattr(app, CONSOLE_RUNTIME_ATTR, None)
        assert runtime_two is runtime_one, "the SAME runtime serves visit two"
        assert app.terminal_session_manager is terminal_manager
        assert controller_two is controller_one
        assert runtime_two.chat_store is store_one
        assert runtime_two.agent_bridge is bridge_one
        # A fresh visit Event, and the previous visit's stays set forever.
        assert controller_two._shutdown_requested is not visit_one_event
        assert not controller_two._shutdown_requested.is_set()
        assert visit_one_event.is_set()
        # The hooks now answer for the LIVE screen, not the dead one.
        assert controller_two.notify_run_outcome is not None
        assert controller_two.notify_run_outcome.__self__ is chat_two, (
            "a hook is still bound to the previous, unmounted screen"
        )


@pytest.mark.asyncio
async def test_post_unmount_raw_refusal_restores_on_second_console_visit(tmp_path):
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)

    async with app.run_test(size=(160, 48)) as pilot:
        chat = ChatScreen(app)
        await app.push_screen(chat)
        app._initial_screen_pushed = True
        app.current_tab = "chat"
        await pilot.pause()
        await _wait_for_selector(chat, pilot, "#console-native-composer")

        store = chat._ensure_console_chat_store()
        origin_session_id = store.active_session_id
        assert origin_session_id is not None
        runtime = chat._console_runtime()
        controller_a = chat._raw_cli

        source = ConsoleComposerBar()
        assert source.handle_console_key(Key("exclamation_mark", "!")) is True
        assert source.handle_console_key(Key("space", " ")) is True
        source.insert_pasted_text("pwd")
        stash = source.stash_draft_for_send()
        assert stash is not None

        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()
        assert chat not in app.screen_stack

        controller_a._append_local_error = lambda _session_id, _text: None
        controller_a._refuse(origin_session_id, stash, "test refusal")
        assert runtime.raw_cli_refusal_stash_bank[origin_session_id][0] is stash

        await app.handle_screen_navigation(NavigateToScreen("chat"))
        await pilot.pause()
        chat_two = app.screen
        assert isinstance(chat_two, ChatScreen)
        await _wait_for_selector(chat_two, pilot, "#console-native-composer")

        assert chat_two._console_runtime() is runtime
        assert chat_two._raw_cli is not controller_a
        composer = chat_two.query_one("#console-native-composer", ConsoleComposerBar)
        restored = composer.stash_draft_for_send()
        assert restored is not None
        assert restored.segments == stash.segments
        assert restored.raw_cli_prefix_typed is True
        assert restored.has_paste is True
        classified = classify_console_raw_draft(restored)
        assert classified.kind == "raw"
        assert classified.text == "pwd"
        assert runtime.raw_cli_refusal_stash_bank == {}


@pytest.mark.asyncio
async def test_a_terminal_run_state_after_leaving_does_not_reach_the_dead_screen(
    tmp_path,
):
    """RED (defect): a run settling post-unmount called into a dead view.

    Drives the production path -- `_set_run_state` into a terminal status
    for a NON-active session -- on the controller that SURVIVES the
    navigation, and asserts the unmounted screen's toast never fires.
    Without `detach_view` the slot is still bound to that screen and it
    does.
    """
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)

    async with app.run_test(size=(160, 48)) as pilot:
        chat = ChatScreen(app)
        await app.push_screen(chat)
        app._initial_screen_pushed = True
        app.current_tab = "chat"
        await pilot.pause()
        await _wait_for_selector(chat, pilot, "#console-native-composer")

        controller = chat._ensure_console_chat_controller()
        store = chat._ensure_console_chat_store()
        background = store.create_session(title="background")
        # `create_session` activates what it creates, so this second one
        # leaves a DIFFERENT session viewed -- which is what makes the
        # terminal transition below take the non-active branch that owns
        # `notify_run_outcome`.
        store.create_session(title="viewed")
        assert store.active_session_id != background.id

        reached: list[tuple[str, ConsoleRunStatus]] = []
        original = chat._notify_console_run_outcome
        chat._notify_console_run_outcome = lambda session_id, status: reached.append(
            (session_id, status)
        )
        # Re-bind so the recorder is what the runtime holds for THIS visit.
        chat._console_runtime().attach_view(chat)
        assert controller.notify_run_outcome is not None

        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()
        assert chat not in app.screen_stack, "Console must actually unmount"

        controller._set_run_state(
            ConsoleRunState(status=ConsoleRunStatus.COMPLETED),
            session_id=background.id,
        )
        await pilot.pause()

        assert reached == [], (
            "a run that settled after the navigation reached the UNMOUNTED "
            f"screen's toast: {reached}"
        )
        chat._notify_console_run_outcome = original


@pytest.mark.asyncio
async def test_a_superseded_screen_never_detaches_the_successors_runtime(tmp_path):
    """The restore-before-unmount order, made explicit.

    `_complete_screen_navigation` constructs the incoming screen and calls
    `restore_state` (which reaches `ensure_chat_store`) BEFORE
    `switch_screen` unmounts the outgoing one, so on a SAME-TARGET
    navigation -- reachable through the `coding` route, which aliases to
    Chat -- the incoming screen attaches FIRST and the outgoing screen
    detaches SECOND. The successor's claim must win: its hooks stay bound
    and its visit Event stays unset.
    """
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)

    async with app.run_test(size=(160, 48)) as pilot:
        chat = ChatScreen(app)
        await app.push_screen(chat)
        app._initial_screen_pushed = True
        app.current_tab = "chat"
        await pilot.pause()
        await _wait_for_selector(chat, pilot, "#console-native-composer")

        controller = chat._ensure_console_chat_controller()
        runtime = getattr(app, CONSOLE_RUNTIME_ATTR, None)
        assert runtime.view is chat

        await app.handle_screen_navigation(NavigateToScreen("coding"))
        await pilot.pause()
        chat_two = app.screen
        assert isinstance(chat_two, ChatScreen), type(chat_two).__name__
        assert chat_two is not chat
        await _wait_for_selector(chat_two, pilot, "#console-native-composer")

        assert runtime.view is chat_two, (
            "the outgoing screen's detach ran anyway and dropped the claim "
            "its successor had already made"
        )
        assert controller.notify_run_outcome is not None, (
            "the outgoing screen cleared a hook its successor had bound"
        )
        assert not controller._shutdown_requested.is_set(), (
            "the outgoing screen's leave_console poisoned the incoming "
            "visit -- a dead Console after a same-target navigation"
        )


@pytest.mark.asyncio
async def test_opening_console_during_a_headless_delivery_arms_the_poll(tmp_path):
    """task-15860 Task 4: the mid-delivery freeze, at the REAL mount.

    `delivery_ui_hook` fires exactly once, in `_attempt`, at delivery
    start -- and it is the only thing that arms the 0.2s transcript poll
    for a wake turn (a wake bypasses the user-send worker that normally
    arms it). With the runtime outliving the screen, "delivery start" and
    "view attach" are independent events, so a Console opened DURING a
    delivery that began with no view must be re-armed by the attach
    itself. PR 3a-2 Task 7 measured the cost of not doing it live: a 4+
    minute frozen transcript.

    Driven through the production navigation API, with a control leg
    (return with nothing delivering -> no poll) so the assertion cannot
    pass for the wrong reason.
    """
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)

    async with app.run_test(size=(160, 48)) as pilot:
        chat = ChatScreen(app)
        await app.push_screen(chat)
        app._initial_screen_pushed = True
        app.current_tab = "chat"
        await pilot.pause()
        await _wait_for_selector(chat, pilot, "#console-native-composer")

        controller = chat._ensure_console_chat_controller()
        store = chat._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id, "the rig must have an active session"
        wake = controller.fleet_wake

        # -- control leg: leave and return with NOTHING delivering -------
        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()
        await app.handle_screen_navigation(NavigateToScreen("chat"))
        await pilot.pause()
        control = app.screen
        await _wait_for_selector(control, pilot, "#console-native-composer")
        await pilot.pause()
        assert control._console_transcript_sync_timer is None, (
            "returning to Console with nothing in flight armed the poll "
            "anyway -- the delivery leg below would prove nothing"
        )

        # -- the real leg: a wake begins while Console is unmounted ------
        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()
        assert control not in app.screen_stack, "Console must actually unmount"
        # Harness precondition ONLY: stand in for `_attempt` having marked
        # a delivery in flight. Everything after this line is production.
        wake._delivering = session_id
        wake._delivering_session = session_id
        try:
            await app.handle_screen_navigation(NavigateToScreen("chat"))
            await pilot.pause()
            reopened = app.screen
            assert isinstance(reopened, ChatScreen), type(reopened).__name__
            await _wait_for_selector(reopened, pilot, "#console-native-composer")
            await pilot.pause()

            assert reopened._console_transcript_sync_timer is not None, (
                "Console opened during a wake delivery with no transcript "
                "poll armed -- this is the live 4+ minute freeze"
            )
        finally:
            wake._delivering = None
            wake._delivering_session = None


@pytest.mark.unit
def test_the_runtime_is_disposed_by_the_apps_shutdown_lifecycles():
    """`dispose()` is app-exit work and must be registered as such.

    The runtime survives every navigation now, so nothing else ends it.
    """
    import inspect

    from tldw_chatbook.app import TldwCli

    source = inspect.getsource(TldwCli._shutdown_app_owned_lifecycles)
    assert "_shutdown_console_runtime" in source, source
    disposer = inspect.getsource(TldwCli._shutdown_console_runtime)
    assert "dispose_console_runtime" in disposer, disposer


@pytest.mark.unit
def test_raw_cli_runtime_is_app_owned_unarmed_and_reads_config_replacements():
    """The app owns one launch-local arm bit over its latest config object."""
    from tldw_chatbook.app import TldwCli

    initializer = inspect.getsource(TldwCli.__init__)
    config_load = initializer.index("self.app_config = load_settings()")
    raw_runtime = initializer.index("self.raw_cli_runtime")
    next_owner = initializer.index("self.library_new_profile_admission")
    assert config_load < raw_runtime < next_owner, initializer

    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": True}})
    runtime = app.raw_cli_runtime
    assert runtime.permitted is True
    assert runtime.armed is False

    app.app_config = {"console": {"raw_cli_permitted": "true"}}
    assert runtime.arm().armed is False
    app.app_config = {"console": {"raw_cli_permitted": 1}}
    assert runtime.arm().armed is False
    app.app_config = {"console": {"raw_cli_permitted": True}}
    assert runtime.arm().armed is True
    assert app.raw_cli_runtime is runtime
    runtime.shutdown()


@pytest.mark.unit
def test_terminal_manager_is_app_owned_unarmed_and_reads_config_replacements():
    """The app owns one launch-local Terminal arm over its latest config."""
    from tldw_chatbook.Terminal.session_manager import TerminalSessionManager
    from tldw_chatbook.app import TldwCli

    initializer = inspect.getsource(TldwCli.__init__)
    config_load = initializer.index("self.app_config = load_settings()")
    terminal_manager = initializer.index("self.terminal_session_manager")
    console_runtime = initializer.index("self.console_runtime")
    assert config_load < terminal_manager < console_runtime, initializer

    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": True}})
    manager = app.terminal_session_manager
    assert isinstance(manager, TerminalSessionManager)
    assert manager.permitted is True
    assert manager.armed is False

    app.app_config = {"console": {"raw_cli_permitted": "true"}}
    assert manager.arm(acknowledge_disclosure=True).armed is False
    app.app_config = {"console": {"raw_cli_permitted": 1}}
    assert manager.arm(acknowledge_disclosure=True).armed is False
    app.app_config = {"console": {"raw_cli_permitted": True}}
    assert manager.arm(acknowledge_disclosure=True).armed is True
    assert app.terminal_session_manager is manager
    manager.disarm()


@pytest.mark.asyncio
async def test_raw_cli_runtime_shutdown_is_once_and_precedes_console_shutdown():
    """Both Textual shutdown paths share one raw-runtime shutdown task."""
    from tldw_chatbook.app import TldwCli

    calls: list[str] = []

    class Runtime:
        def shutdown(self) -> object:
            calls.append("raw")
            return object()

    app = object.__new__(TldwCli)
    app.raw_cli_runtime = Runtime()
    app._raw_cli_runtime_shutdown_task = None

    await TldwCli._shutdown_raw_cli_runtime(app)
    await TldwCli._shutdown_raw_cli_runtime(app)

    assert calls == ["raw"]
    source = inspect.getsource(TldwCli._shutdown_app_owned_lifecycles)
    raw = source.index("_shutdown_raw_cli_runtime")
    console = source.index("_shutdown_console_runtime")
    assert raw < console, source


@pytest.mark.asyncio
async def test_terminal_manager_shutdown_is_shared_and_shielded_from_waiter_cancel():
    """One app task owns the five-second drain and final handle closure."""
    from tldw_chatbook.app import TldwCli

    entered = asyncio.Event()
    release = asyncio.Event()
    calls: list[object] = []

    class Manager:
        async def shutdown(self, *, deadline_seconds: float) -> bool:
            calls.append(("shutdown", deadline_seconds))
            entered.set()
            await release.wait()
            return False

        def finalize_shutdown(self) -> None:
            calls.append("finalize")

    app = object.__new__(TldwCli)
    app.terminal_session_manager = Manager()
    app._terminal_session_manager_shutdown_task = None

    first = asyncio.create_task(TldwCli._shutdown_terminal_session_manager(app))
    await asyncio.wait_for(entered.wait(), 1)
    second = asyncio.create_task(TldwCli._shutdown_terminal_session_manager(app))
    await asyncio.sleep(0)

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert calls == [("shutdown", 5.0)]

    release.set()
    await asyncio.wait_for(second, 1)
    assert calls == [("shutdown", 5.0), "finalize"]

    source = inspect.getsource(TldwCli._shutdown_app_owned_lifecycles)
    terminal = source.index("_shutdown_terminal_session_manager")
    console = source.index("_shutdown_console_runtime")
    buddy = source.index("_shutdown_persona_buddy")
    assert terminal < console < buddy, source


@pytest.mark.asyncio
async def test_app_shutdown_drains_and_finalizes_a_real_terminal_manager():
    """The app boundary drives real manager cleanup through finalization."""
    from tldw_chatbook.Terminal.contracts import (
        AdmissionGate,
        BackendIdentity,
        CleanupAttempt,
        CleanupProof,
        TerminalLaunchRequest,
    )
    from tldw_chatbook.Terminal.session_manager import TerminalSessionManager
    from tldw_chatbook.app import TldwCli

    cleanup_entered = threading.Event()
    cleanup_release = threading.Event()

    class Backend:
        def __init__(self) -> None:
            self.finalize_calls = 0

        def start(
            self,
            _request: TerminalLaunchRequest,
            admission: AdmissionGate,
        ) -> BackendIdentity:
            return BackendIdentity(session_id=admission.token)

        def read(self, _maximum: int = 64 * 1024) -> bytes | None:
            return None

        def write(self, _data: bytes) -> None:
            return None

        def resize(self, _columns: int, _rows: int) -> None:
            return None

        def request_priority_close(self) -> None:
            return None

        def cleanup(self, _attempt: CleanupAttempt) -> CleanupProof:
            cleanup_entered.set()
            assert cleanup_release.wait(1)
            return CleanupProof()

        def finalize_shutdown(self) -> None:
            self.finalize_calls += 1

    backend = Backend()
    manager = TerminalSessionManager(lambda: True, lambda: backend)
    manager.arm(acknowledge_disclosure=True)
    created = manager.create_session(
        TerminalLaunchRequest(
            name="app-shutdown-integration",
            shell="default",
            start_directory=str(Path.cwd()),
            columns=80,
            rows=24,
        )
    )
    assert created.admitted is True

    app = object.__new__(TldwCli)
    app.terminal_session_manager = manager
    app._terminal_session_manager_shutdown_task = None

    first = asyncio.create_task(TldwCli._shutdown_terminal_session_manager(app))
    assert await asyncio.to_thread(cleanup_entered.wait, 1)
    second = asyncio.create_task(TldwCli._shutdown_terminal_session_manager(app))

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert backend.finalize_calls == 0

    cleanup_release.set()
    await asyncio.wait_for(second, 1)
    assert backend.finalize_calls == 1


@pytest.mark.unit
def test_persona_buddy_is_app_owned_and_shutdown_after_console_producers():
    """Console producers stop before Buddy drains, which precedes profiles.

    TASK-21103 rewrote what the construction half of this pin means. The
    controller is no longer built inside ``__init__`` — importing it drags
    Persona_Visual and PIL (1.28 s cold) onto the boot path, so the eager
    wiring became the lazy ``persona_buddy_controller`` property over
    ``_build_persona_buddy_controller``. The construction SEMANTICS the old
    pin protected (portrait loader partial over the local persona service)
    moved there intact, and ``__init__`` must stay construction-free. The
    shutdown half is unchanged in meaning: Console producers stop first,
    then Buddy drains — but the disposer must now PEEK the slot rather than
    read the property, or a never-built controller would be constructed
    (importing PIL) purely to be shut down.
    """
    import inspect

    from tldw_chatbook.app import TldwCli

    initializer = inspect.getsource(TldwCli.__init__)
    assert "PersonaBuddyController(" not in initializer, (
        "eager Buddy construction is back in __init__ (TASK-21103 regression)"
    )
    slot = initializer.index("self._persona_buddy_controller")
    console_runtime = initializer.index("= ConsoleRuntime(self)")
    assert slot < console_runtime, (
        "the controller slot must exist before ConsoleRuntime reads the "
        "persona_buddy_controller property"
    )

    assert isinstance(
        inspect.getattr_static(TldwCli, "persona_buddy_controller"), property
    )
    builder = inspect.getsource(TldwCli._build_persona_buddy_controller)
    assert "PersonaBuddyController(" in builder, builder
    assert "portrait_loader=partial(" in builder, builder
    assert "load_local_persona_portrait" in builder, builder
    guard = builder.index('"local_character_persona_service"')
    construction = builder.index("PersonaBuddyController(")
    assert guard < construction, builder

    source = inspect.getsource(TldwCli._shutdown_app_owned_lifecycles)
    buddy = source.index("_shutdown_persona_buddy")
    console = source.index("_shutdown_console_runtime")
    assert console < buddy, source
    disposer = inspect.getsource(TldwCli._shutdown_persona_buddy)
    assert "self._persona_buddy_controller" in disposer, disposer
    assert "controller.shutdown()" in disposer, disposer
    assert "self.persona_buddy_controller.shutdown" not in disposer, (
        "shutdown must peek the slot, never the constructing property"
    )


@pytest.mark.unit
def test_lazy_persona_buddy_property_defers_and_ensure_constructs():
    """The lazy controller property's three states behave as designed.

    TASK-21103 behavior pins, on a skeletal ``TldwCli``:

    - disabled preferences: the passive property returns None WITHOUT
      constructing (the every-screen-mount reconcile early-out stays free of
      the Persona_Visual/PIL import);
    - disabled preferences, explicit feature use:
      ``ensure_persona_buddy_controller()`` constructs anyway (this is what
      lets Workbench "Use for Buddy" enable from a disabled state), and the
      passive property then returns the same cached instance;
    - enabled preferences: the first passive read constructs, and the
      construction is cached (same object on the second read).
    - the setter installs a test double the property returns verbatim.
    """
    from tldw_chatbook.app import TldwCli

    def skeleton(enabled: bool) -> TldwCli:
        app = object.__new__(TldwCli)
        app._persona_buddy_controller = None
        app._persona_buddy_controller_lock = threading.Lock()
        app.app_config = {"persona_buddy": {"enabled": enabled}}
        app.local_character_persona_service = object()
        app.chachanotes_db = object()
        app.call_after_refresh = lambda *args, **kwargs: None
        return app

    disabled = skeleton(enabled=False)
    assert disabled.persona_buddy_controller is None
    assert disabled._persona_buddy_controller is None, (
        "the passive property constructed a controller for a disabled profile"
    )

    ensured = disabled.ensure_persona_buddy_controller()
    assert ensured is not None
    assert disabled.persona_buddy_controller is ensured

    enabled = skeleton(enabled=True)
    first = enabled.persona_buddy_controller
    assert first is not None
    assert enabled.persona_buddy_controller is first

    injected = object()
    enabled.persona_buddy_controller = injected
    assert enabled.persona_buddy_controller is injected


@pytest.mark.unit
def test_actor_pack_recovery_precedes_character_persona_surfaces():
    """Cross-store recovery is gated ahead of every affected surface.

    task-21106 rewrote what this pin means. Recovery no longer runs inside
    ``_wire_character_persona_services`` — synchronous SQLite during
    ``__init__`` cost every boot and crashed the test app factory — so the
    old ``local_service < coordinator < recover() < scope`` source ordering
    is gone by design. The guarantee it protected now has three seams, and
    this test pins all of them:

    - the deferred-startup worker kicks ``ensure_actor_pack_recovery`` on a
      thread right after first paint (ahead of any user-driven Console/Buddy
      persona read);
    - the Personas surface awaits the same idempotent gate before its first
      library read (behavioral proof in test_actor_pack_recovery_seam.py);
    - the coordinator itself runs ``ensure_recovered`` before admitting a
      ``create_persona`` mutation, so no caller ordering can bypass it.
    """
    import inspect

    from tldw_chatbook.Actor_Packs.persona_coordinator import (
        PersonaActorPackCoordinator,
    )
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

    wiring = inspect.getsource(TldwCli._wire_character_persona_services)
    local_service = wiring.index("LocalCharacterPersonaService(")
    coordinator = wiring.index("PersonaActorPackCoordinator(")
    scope = wiring.index("CharacterPersonaScopeService(")
    assert local_service < coordinator < scope, wiring
    assert ".recover()" not in wiring, (
        "recovery is back on the construction path (task-21106 regression)"
    )

    # TASK-22215 moved the deferred-startup kick behind the boot-worker
    # stagger policy: `_schedule_deferred_startup_work` opens the gate, the
    # policy names the row, and the app's starter table calls the gate's
    # `ensure_actor_pack_recovery`. Recovery is still FIRST in that order --
    # it is a prefetch for a surface that would otherwise run SQLite recovery
    # on the event loop -- and the behavioral proof that it actually runs
    # after first paint lives in Tests/UI/test_actor_pack_recovery_seam.py
    # and Tests/App/test_boot_worker_stagger_policy.py.
    from tldw_chatbook.Utils.boot_worker_policy import STAGGERED_BOOT_WORKER_KEYS

    deferred = inspect.getsource(TldwCli._schedule_deferred_startup_work)
    assert "_start_staggered_boot_workers" in deferred, deferred
    assert STAGGERED_BOOT_WORKER_KEYS[0] == "actor_pack_recovery", (
        f"recovery must stay first in the staggered boot order; observed "
        f"{STAGGERED_BOOT_WORKER_KEYS!r}"
    )
    starters = inspect.getsource(TldwCli.boot_worker_starters)
    assert "ensure_actor_pack_recovery" in starters, starters

    personas_load = inspect.getsource(PersonasScreen._load_after_mount)
    assert "ensure_actor_pack_recovery" in personas_load, personas_load

    create = inspect.getsource(PersonaActorPackCoordinator.create_persona)
    assert create.index("self.ensure_recovered()") < create.index(
        "self._blocked_intent_ids"
    ), create

    # TASK-21103 moved Buddy construction out of ``__init__`` into the lazy
    # ``_build_persona_buddy_controller``. The ordering guarantee this stanza
    # pinned — Buddy is only ever wired to a fully constructed local persona
    # service — is now enforced by the builder itself: it reads the service
    # defensively and defers (returns None, retried on next access) until
    # ``_wire_character_persona_services`` has run.
    builder = inspect.getsource(TldwCli._build_persona_buddy_controller)
    guard = builder.index('"local_character_persona_service"')
    buddy = builder.index("PersonaBuddyController(")
    assert guard < buddy, builder
    assert "PersonaBuddyController(" not in inspect.getsource(TldwCli.__init__)


@pytest.mark.asyncio
async def test_app_fences_console_then_drains_buddy_before_profile_teardown(
    monkeypatch: pytest.MonkeyPatch,
):
    """Repeated cancellation cannot skip either ordered app-owned drain."""
    from textual.app import App

    from tldw_chatbook.app import TldwCli

    terminal_entered = asyncio.Event()
    terminal_release = asyncio.Event()
    console_entered = asyncio.Event()
    console_release = asyncio.Event()
    buddy_entered = asyncio.Event()
    buddy_release = asyncio.Event()
    events: list[str] = []

    class Buddy:
        async def shutdown(self) -> None:
            events.append("buddy-start")
            buddy_entered.set()
            await buddy_release.wait()
            events.append("buddy-finished")

    class ProfileService:
        def teardown(self) -> None:
            events.append("profile-teardown")

    class AsyncOwner:
        async def shutdown(self) -> None:
            events.append("later-owner")

    async def later_lifecycle() -> None:
        events.append("later-lifecycle")

    async def no_op_lifecycle() -> None:
        return None

    terminal_task: asyncio.Task[None] | None = None

    async def terminal_runner() -> None:
        events.append("terminal-start")
        terminal_entered.set()
        await terminal_release.wait()
        events.append("terminal-finished")

    async def shutdown_terminal_manager() -> None:
        nonlocal terminal_task
        if terminal_task is None:
            terminal_task = asyncio.create_task(terminal_runner())
        await asyncio.shield(terminal_task)

    console_task: asyncio.Task[None] | None = None

    async def console_runner() -> None:
        events.append("console-start")
        console_entered.set()
        await console_release.wait()
        events.append("console-finished")

    async def shutdown_console_runtime() -> None:
        nonlocal console_task
        if console_task is None:
            console_task = asyncio.create_task(console_runner())
        await asyncio.shield(console_task)

    app = object.__new__(TldwCli)
    app.persona_buddy_controller = Buddy()
    app._persona_buddy_shutdown_task = None
    app._audio_cpp_artifact_lease_coordinator = None
    app.audio_cpp_model_install_owner = AsyncOwner()
    app._shutdown_notes_sync_runtime = no_op_lifecycle
    app._shutdown_raw_cli_runtime = no_op_lifecycle
    app._shutdown_terminal_session_manager = shutdown_terminal_manager
    app._shutdown_console_image_edits = later_lifecycle
    app._shutdown_console_runtime = shutdown_console_runtime
    app._shutdown_file_notes_session_owner = later_lifecycle
    profile_service = ProfileService()

    async def profile_teardown(_app: App[None]) -> None:
        profile_service.teardown()

    monkeypatch.setattr(App, "_shutdown", profile_teardown)
    draining = asyncio.create_task(TldwCli._shutdown(app))
    await asyncio.wait_for(terminal_entered.wait(), 2)
    assert events == ["terminal-start"]
    assert not draining.done()

    draining.cancel()
    await asyncio.sleep(0)
    draining.cancel()
    await asyncio.sleep(0)
    assert not draining.done()
    assert not console_entered.is_set()
    assert not buddy_entered.is_set()

    terminal_release.set()
    await asyncio.wait_for(console_entered.wait(), 2)
    assert events[:3] == [
        "terminal-start",
        "terminal-finished",
        "console-start",
    ]

    console_release.set()
    await asyncio.wait_for(buddy_entered.wait(), 2)
    assert events[2:5] == ["console-start", "console-finished", "buddy-start"]
    assert "profile-teardown" not in events

    buddy_release.set()
    with pytest.raises(asyncio.CancelledError):
        await draining

    assert events.index("terminal-finished") < events.index("console-start")
    assert events.index("console-finished") < events.index("buddy-start")
    assert events.index("buddy-finished") < events.index("profile-teardown")
    assert events[-1] == "profile-teardown"
