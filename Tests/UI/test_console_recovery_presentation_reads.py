"""Recovery repaint reads current owned state without reconfiguring a live run."""

import sys
import threading
from types import MethodType

import pytest

from Tests.UI.test_console_turn_resend_ui import FailThenRecoverGateway, _console_app
from tldw_chatbook import config
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
async def test_actual_recovery_builders_do_not_refresh_established_controller(
    record_property, monkeypatch
):
    host = _console_app(FailThenRecoverGateway())
    runtime = host.app_instance.console_runtime
    assert type(runtime) is ConsoleRuntime and runtime.app is host.app_instance
    try:
        async with host.run_test(size=(211, 44)):
            screen = host.screen_stack[-1]
            assert type(screen) is ChatScreen
            controller = runtime.chat_controller
            assert controller is not None and controller.store is runtime.chat_store
            assert runtime.view is screen
            region = screen._build_console_center()
            before = (region._trace_recovery_state_builder(), region._recovery_state())
            codes = {
                ChatScreen._ensure_console_chat_controller.__code__: "ensure",
                ChatScreen._sync_console_chat_core_state.__code__: "core",
                config.load_settings.__code__: "settings",
            }
            counts = dict.fromkeys(codes.values(), 0)
            main = threading.get_ident()
            old = sys.getprofile()

            def observe(frame, event, _argument):
                if event == "call" and threading.get_ident() == main:
                    name = codes.get(frame.f_code)
                    if name is not None:
                        counts[name] += 1

            sys.setprofile(observe)
            try:
                after = (
                    region._trace_recovery_state_builder(),
                    region._recovery_state(),
                )
            finally:
                sys.setprofile(old)
            assert after == before
            assert runtime.chat_controller is controller and runtime.view is screen
            for name, value in counts.items():
                record_property("recovery_" + name + "_entries", value)
            assert counts == {"ensure": 0, "core": 0, "settings": 0}, (
                "Recovery repaint refreshes live provider/runtime state",
                counts,
            )
            custom_refreshes = []
            original_ensure = screen._ensure_console_chat_controller

            def custom_ensure():
                custom_refreshes.append(True)
                return original_ensure()

            with monkeypatch.context() as custom:
                custom.setattr(screen, "_ensure_console_chat_controller", custom_ensure)
                assert screen._console_recovery_presentation_controller() is controller
            assert custom_refreshes == [
                True
            ], "display skipped a replaced live refresh seam"

            generation = screen._console_runtime_attachment_generation
            try:
                screen._console_runtime_attachment_generation = -1
                counts.update(dict.fromkeys(counts, 0))
                sys.setprofile(observe)
                try:
                    assert (
                        screen._console_recovery_presentation_controller() is controller
                    )
                finally:
                    sys.setprofile(old)
                assert (
                    counts["ensure"] == 1
                ), "obsolete view generation used presentation reuse"
            finally:
                screen._console_runtime_attachment_generation = generation

            original = type(controller).provider_continuation_recovery_message
            owned_store = controller.store
            foreign_store = ConsoleChatStore()
            inspected = []

            def retarget_getter(_controller):
                inspected.append(True)
                _controller.store = foreign_store
                return MethodType(original, _controller)

            try:
                with monkeypatch.context() as custom:
                    custom.setattr(
                        type(controller),
                        "provider_continuation_recovery_message",
                        property(retarget_getter),
                    )
                    assert (
                        screen._console_recovery_presentation_controller() is controller
                    )
                    assert (
                        inspected == []
                    ), "qualification executed a replacement descriptor"
                    assert controller.store is owned_store
            finally:
                controller.store = owned_store

    finally:
        await runtime.dispose()
