"""The app asks Windows not to power-throttle it (TASK-40608).

Windows slows a process it takes for background work. On a CPU with performance
and efficiency cores that put every step of a Console turn on the efficiency
cores (i5-14500, Windows 10 22H2, 2026-11-28: 4.4 to 5.8 s a turn, against 2.1
to 2.6 s once the app's own process had made the request tested here).

What Windows did with a request cannot be read back on Windows 10, so these
tests pin the request itself at the kernel boundary (information class and
payload bytes, written out by hand below) and, on Windows, that the real call
is accepted.
"""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Iterator

import pytest
from textual.app import App

from tldw_chatbook.Utils import text_selection_crash_guard as guard

#: ``ProcessPowerThrottling`` in ``PROCESS_INFORMATION_CLASS``.
POWER_THROTTLING = 4
#: ``PROCESS_POWER_THROTTLING_STATE``: version 1, control mask 0x1 (execution
#: speed is the policy being decided), state mask 0 (and it is switched off).
NOT_THROTTLED = bytes.fromhex("01000000" "01000000" "00000000")
SWITCH = "TLDW_POWER_THROTTLING_OPT_OUT"


@pytest.fixture
def requests(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, bytes]]:
    """Every request that reaches the kernel boundary, on a pretend Windows that accepts it."""
    seen: list[tuple[int, bytes]] = []

    def accept(information_class: int, payload: bytes) -> bool:
        seen.append((information_class, bytes(payload)))
        return True

    monkeypatch.setattr(guard, "_set_process_information", accept)
    monkeypatch.setattr(guard, "_is_windows", lambda: True)
    return seen


def leave_pytest(monkeypatch: pytest.MonkeyPatch) -> None:
    """As in the running app: no pytest marker in the environment, no switch.

    Called from the test body: pytest sets ``PYTEST_CURRENT_TEST`` again at
    the start of every phase, so a fixture cannot take it away for the call.
    """
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv(SWITCH, raising=False)


def test_the_request_sent_is_the_documented_opt_out(requests, monkeypatch):
    monkeypatch.setenv(SWITCH, "1")

    assert guard.opt_out_of_windows_power_throttling() is True

    assert requests == [(POWER_THROTTLING, NOT_THROTTLED)]


def test_outside_pytest_the_request_is_made_by_default(requests, monkeypatch):
    leave_pytest(monkeypatch)

    assert guard.opt_out_of_windows_power_throttling() is True

    assert requests == [(POWER_THROTTLING, NOT_THROTTLED)]


def test_nothing_is_asked_under_pytest_by_default(requests, monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)

    assert guard.opt_out_of_windows_power_throttling() is False

    assert requests == []


@pytest.mark.parametrize("value", ["0", "false", "no", ""])
def test_the_switch_turns_the_request_off(requests, monkeypatch, value):
    leave_pytest(monkeypatch)
    monkeypatch.setenv(SWITCH, value)

    assert guard.opt_out_of_windows_power_throttling() is False

    assert requests == []


def test_another_platform_is_asked_nothing(requests, monkeypatch):
    monkeypatch.setenv(SWITCH, "1")
    monkeypatch.setattr(guard, "_is_windows", lambda: False)

    assert guard.opt_out_of_windows_power_throttling() is False

    assert requests == []


@pytest.mark.parametrize(
    "refusal",
    [
        pytest.param(lambda information_class, payload: False, id="windows-refuses"),
        pytest.param(
            lambda information_class, payload: (_ for _ in ()).throw(
                OSError(87, "refused")
            ),
            id="the-call-fails",
        ),
        pytest.param(
            lambda information_class, payload: (_ for _ in ()).throw(
                AttributeError("function 'SetProcessInformation' not found")
            ),
            id="an-older-windows-lacks-the-call",
        ),
    ],
)
def test_a_refusal_never_stops_the_app(requests, monkeypatch, refusal):
    monkeypatch.setenv(SWITCH, "1")
    monkeypatch.setattr(guard, "_set_process_information", refusal)

    assert guard.opt_out_of_windows_power_throttling() is False


class _GuardedApp(guard.ThreadWorkerContextGuard, App[None]):
    """The smallest app that carries the start-up guard."""


@pytest.mark.asyncio
async def test_a_loading_app_makes_the_request(requests, monkeypatch):
    monkeypatch.setenv(SWITCH, "1")

    async with _GuardedApp().run_test():
        assert requests == [(POWER_THROTTLING, NOT_THROTTLED)]


@pytest.fixture
def windows_manages_this_process_again() -> Iterator[None]:
    """Hand the decision back to Windows after a real request (control mask 0)."""
    yield
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetProcessInformation.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    )
    state = (ctypes.c_uint32 * 3)(1, 0, 0)
    assert kernel.SetProcessInformation(
        kernel.GetCurrentProcess(), POWER_THROTTLING, state, ctypes.sizeof(state)
    )


@pytest.mark.skipif(sys.platform != "win32", reason="the real Windows call")
def test_windows_accepts_the_real_request(
    monkeypatch, windows_manages_this_process_again
):
    monkeypatch.setenv(SWITCH, "1")

    assert guard.opt_out_of_windows_power_throttling() is True
