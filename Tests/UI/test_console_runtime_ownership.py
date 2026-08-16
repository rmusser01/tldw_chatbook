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
from pathlib import Path

import pytest

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
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

#: Constructor calls that must exist in exactly one place: the runtime.
#: `ConsoleProviderGateway(` is deliberately NOT here -- the Personas
#: preview controller builds its own, unrelated to the Console runtime
#: (`UI/Persona_Modules/personas_preview_controller.py`).
_RUNTIME_OWNED_CONSTRUCTIONS = ("ConsoleChatStore", "ConsoleChatController")

_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook"
_RUNTIME_MODULE = "tldw_chatbook/Chat/console_runtime.py"


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
                func.id if isinstance(func, ast.Name) else func.attr
                if isinstance(func, ast.Attribute) else None
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
        assert controller_two is controller_one
        assert runtime_two.chat_store is store_one
        assert runtime_two.agent_bridge is bridge_one
        # A fresh visit Event, and the previous visit's stays set forever.
        assert controller_two._shutdown_requested is not visit_one_event
        assert not controller_two._shutdown_requested.is_set()
        assert visit_one_event.is_set()
        # The hooks now answer for the LIVE screen, not the dead one.
        assert controller_two.notify_run_outcome is not None
        assert (
            controller_two.notify_run_outcome.__self__ is chat_two
        ), "a hook is still bound to the previous, unmounted screen"


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
        chat._notify_console_run_outcome = (
            lambda session_id, status: reached.append((session_id, status))
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
