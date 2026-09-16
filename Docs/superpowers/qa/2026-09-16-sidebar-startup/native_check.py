"""Verify sidebar state in two fresh native app processes on one private profile.

Usage: native_check.py PROFILE TMUX_SOCKET SESSION save|restart
Seed PROFILE/ui_state.toml as described in README.md before the save phase.
"""

import asyncio
import hashlib
import json
import os
import runpy
import subprocess
import sys
import traceback
from pathlib import Path


def main():
    root = Path(sys.argv[1]).resolve()
    socket, session, phase = sys.argv[2:5]
    if phase not in {"save", "restart"}:
        raise ValueError("Choose save or restart")
    guard = Path(__file__).resolve().parent.parent / (
        "2026-09-16-ingest-lifecycle/native_check.py"
    )
    runpy.run_path(str(guard))["validate_profile"](root)
    evidence = root / phase
    evidence.mkdir(exist_ok=False)
    os.environ["TLDW_CONFIG_PATH"] = str(root / "config.toml")
    os.environ["XDG_DATA_HOME"] = str(root / "data")
    os.environ["XDG_CONFIG_HOME"] = str(root / "config")
    os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
    os.environ.pop("NO_COLOR", None)

    from textual_image._terminal import probe_terminal

    import tldw_chatbook.app as app_module
    from tldw_chatbook.app import TldwCli

    checkout = Path(__file__).resolve().parents[4]
    assert Path(app_module.__file__).resolve() == checkout / "tldw_chatbook/app.py"
    probe_terminal()
    app = TldwCli()
    result = {
        "phase": phase,
        "pid": os.getpid(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "screen_sha256": hashlib.sha256(
            (checkout / "tldw_chatbook/UI/Screens/chat_screen.py").read_bytes()
        ).hexdigest(),
    }

    def record():
        (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    async def journey(pilot):
        try:
            assert app._instance_lock_status.acquired
            result["exclusive_profile"] = True
            deadline = asyncio.get_running_loop().time() + 60
            while not (
                getattr(app, "_ui_ready", False)
                and type(app.screen).__name__ == "ChatScreen"
                and app.screen.query("#console-native-composer")
            ):
                assert asyncio.get_running_loop().time() < deadline, "Console startup"
                await pilot.pause(0.05)
            await pilot.pause()
            screen = app.screen
            result["driver"] = type(app._driver).__name__
            assert result["driver"] == "LinuxDriver"
            expected = {"notes": True, "chat": False}
            if phase == "restart":
                expected["restart-check"] = True
            assert screen.sidebar_state == expected
            assert screen.ui_state.collapsible_states == expected
            assert screen.ui_state.sidebar_search_query == "saved search"
            assert screen.ui_state.last_active_section == "notes"
            result["restored"] = screen._sidebar_state_snapshot()
            app.save_screenshot("console.svg", path=str(evidence))
            if phase == "save":
                # Exercise the real persistence signal; this is not a visible
                # Collapsible gesture or a claim about Console rail preferences.
                screen.ui_state.set_collapsible_state("restart-check", True)
                screen.sidebar_state = dict(screen.ui_state.collapsible_states)
                assert screen._sidebar_state_dirty
                result["pending_change_at_quit"] = screen._sidebar_state_snapshot()
            result["passed"] = True
        except Exception:  # noqa: BLE001 - record the native failure before quitting
            result["passed"] = False
            result["error"] = traceback.format_exc()
        finally:
            record()
            await asyncio.to_thread(
                subprocess.run,
                [
                    "/opt/homebrew/bin/tmux",
                    "-L",
                    socket,
                    "send-keys",
                    "-t",
                    session,
                    "C-q",
                ],
                check=True,
            )

    app.run(auto_pilot=journey, size=(170, 48))
    result["app_run_returned"] = True
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
