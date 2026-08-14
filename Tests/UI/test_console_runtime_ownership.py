"""Characterization pins for the app-owned Console runtime (task-15860, Task 1).

**Neither test here is a defect red.** Task 1 is a pure ownership move with
zero observable behaviour change, so there is no user-visible defect to
reproduce. These two pin what is true the moment the move lands, for two
different reasons:

1. `test_console_runtime_is_the_single_construction_site` pins the move
   itself. If a later change re-adds a `ConsoleChatStore(...)` or
   `ConsoleChatController(...)` anywhere but `Chat/console_runtime.py`, the
   app stops being the owner and Design A quietly loses its premise.

2. `test_second_console_visit_gets_a_new_runtime` pins TODAY's LIFETIME --
   the thing Task 1 deliberately did NOT change. Leaving Console still
   disposes the runtime, so the next visit builds a brand-new
   store/gateway/bridge/controller, exactly as it did when `ChatScreen`
   owned them. Task 2 makes the runtime survive a navigation; when it does,
   this test must go red and be rewritten. That is the point: without it,
   Task 2's central change would be a diff to nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _wait_for_selector
from tldw_chatbook.Chat.console_runtime import (
    CONSOLE_RUNTIME_ATTR,
    ConsoleRuntime,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

#: Constructor calls that must exist in exactly one place: the runtime.
#: `ConsoleProviderGateway(` is deliberately NOT here -- the Personas
#: preview controller builds its own, unrelated to the Console runtime
#: (`UI/Persona_Modules/personas_preview_controller.py`).
_RUNTIME_OWNED_CONSTRUCTIONS = ("ConsoleChatStore(", "ConsoleChatController(")

_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook"
_RUNTIME_MODULE = "tldw_chatbook/Chat/console_runtime.py"


def _construction_sites(token: str) -> list[str]:
    """Every `<path>:<line>` in the shipped package containing `token`."""
    sites: list[str] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        if token not in source:
            continue
        rel = path.relative_to(_PACKAGE_ROOT.parent).as_posix()
        for lineno, line in enumerate(source.splitlines(), start=1):
            if token in line:
                sites.append(f"{rel}:{lineno}: {line.strip()}")
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


@pytest.mark.asyncio
async def test_second_console_visit_gets_a_new_runtime(tmp_path):
    """PIN (characterization) of TODAY's lifetime: leaving Console disposes
    the runtime, so returning builds a brand-new one.

    Task 2 deliberately breaks this. Rewrite it there; do not delete it.
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
        assert runtime_one.generation == 0

        # ---- leave Console through the real navigation API ---------------
        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()
        assert chat not in app.screen_stack, "Console must actually unmount"
        # TODAY: teardown still shuts the controller down ...
        assert controller_one._shutdown_requested.is_set(), (
            "Task 1 must not change what leaving Console does to the "
            "controller -- the shutdown is still on the unmount path"
        )
        # ... and still disposes the runtime with the view.
        assert runtime_one.generation == 1, "on_unmount must dispose the runtime"
        assert getattr(app, CONSOLE_RUNTIME_ATTR, None) is None

        # ---- return to Console -------------------------------------------
        await app.handle_screen_navigation(NavigateToScreen("chat"))
        await pilot.pause()
        chat_two = app.screen
        assert isinstance(chat_two, ChatScreen), type(chat_two).__name__
        assert chat_two is not chat, "screens are never cached"
        await _wait_for_selector(chat_two, pilot, "#console-native-composer")

        controller_two = chat_two._ensure_console_chat_controller()
        runtime_two = getattr(app, CONSOLE_RUNTIME_ATTR, None)
        assert isinstance(runtime_two, ConsoleRuntime), type(runtime_two).__name__
        assert runtime_two is not runtime_one, "a second visit gets a NEW runtime"
        assert controller_two is not controller_one
        assert runtime_two.chat_store is not store_one
        assert runtime_two.agent_bridge is not bridge_one
