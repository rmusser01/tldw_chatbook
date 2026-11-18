"""Observe exact return sites without replacing qualified production callbacks."""

import dis
import sys


def pytest_runtest_setup(item):
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.UI.Console_Modules import (
        config_sync,
        console_spend_projection as spend,
    )

    functions = (
        ChatScreen._sync_native_console_chat_ui,
        spend.ConsoleReadinessConfigProjection.warm,
        config_sync.project_console_tick,
    )
    locations = {
        f.__code__: {i.offset: i.positions.lineno for i in dis.get_instructions(f)}
        for f in functions
    }

    def returned(code, offset, result):
        if code.co_name == "warm" and not result:
            frame = sys._getframe(1)
            owner = frame.f_locals.get("self")
            captured = frame.f_locals.get("key")
            current = owner._key()
            print(
                "WARM_REFUSAL",
                "changed_indices",
                [
                    i
                    for i, (a, b) in enumerate(zip(captured or (), current))
                    if a is not b and a != b
                ],
                "age",
                spend.time.monotonic() - owner.at,
                "value",
                owner.value is not None,
                "pending",
                owner.pending,
                "closed",
                getattr(owner.screen, "_closing", False),
                "captured_current",
                spend._same_readiness_key(captured, current),
                "stored_current",
                spend._same_readiness_key(owner.key, current),
                flush=True,
            )
        print(
            "RETURN",
            code.co_name,
            locations[code].get(offset),
            result,
            "clock",
            config_sync._clock(),
            "poll",
            config_sync._POLL_SYNC.get(),
            flush=True,
        )

    sys.monitoring.use_tool_id(4, "cadence-return-sites")
    sys.monitoring.register_callback(4, sys.monitoring.events.PY_RETURN, returned)
    for function in functions:
        sys.monitoring.set_local_events(
            4, function.__code__, sys.monitoring.events.PY_RETURN
        )
    item._return_codes = tuple(locations)


def pytest_runtest_teardown(item):
    for code in item._return_codes:
        sys.monitoring.set_local_events(4, code, 0)
    sys.monitoring.register_callback(4, sys.monitoring.events.PY_RETURN, None)
    sys.monitoring.free_tool_id(4)
