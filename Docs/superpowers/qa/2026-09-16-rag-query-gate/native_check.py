"""Audit the query gate in real TldwCli with deterministic display-state timing.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
The provider names are synthetic display states; no Run/Send action is used.
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
    socket, terminal = sys.argv[2:4]
    qa = Path(__file__).resolve().parents[1]
    runpy.run_path(str(qa / "2026-09-16-ingest-lifecycle/native_check.py"))[
        "validate_profile"
    ](root)
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=False)
    os.environ["TLDW_CONFIG_PATH"] = str(root / "config.toml")
    os.environ["XDG_DATA_HOME"] = str(root / "data")
    os.environ["XDG_CONFIG_HOME"] = str(root / "config")
    os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
    os.environ.pop("NO_COLOR", None)

    from textual import events
    from textual.css.query import NoMatches, QueryError
    from textual.widgets import Button, Input, Static
    from textual_image._terminal import probe_terminal

    import tldw_chatbook.app as app_module
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Library.library_rag_state import LibraryRagPanelState
    from tldw_chatbook.Widgets.Library.library_search_rag_panel import (
        library_rag_query_quiet_text,
    )

    assert Path(app_module.__file__).resolve() == (
        Path(__file__).resolve().parents[4] / "tldw_chatbook/app.py"
    )
    probe_terminal()
    app = TldwCli()
    result = {
        "pid": os.getpid(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "state_seam": "controlled LibraryRagPanelState; real native widgets and refreshes",
        "steps": [],
    }

    def record():
        (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    async def tmux(*args):
        await asyncio.to_thread(
            subprocess.run,
            ["/opt/homebrew/bin/tmux", "-L", socket, *args],
            check=True,
            text=True,
            capture_output=True,
        )

    def painted(widget):
        region = widget.region
        strips = list(app.screen._compositor.render_strips())
        return "\n".join(
            strips[y].crop(region.x, region.right).text
            for y in range(max(0, region.y), min(region.bottom, len(strips)))
        )

    async def journey(pilot):
        async def wait_for(predicate, label):
            result["waiting_for"] = label
            record()
            deadline = asyncio.get_running_loop().time() + 45
            while asyncio.get_running_loop().time() < deadline:
                try:
                    if predicate():
                        await pilot.pause()
                        return
                except (NoMatches, QueryError):
                    pass
                await pilot.pause(0.03)
            raise AssertionError(label)

        async def press(selector):
            await wait_for(lambda: bool(app.screen.query(selector)), selector)
            button = app.screen.query_one(selector, Button)
            button.focus()
            await wait_for(
                lambda: (
                    app.screen.focused is button
                    and str(button.label) in painted(button)
                ),
                "painted " + selector,
            )
            await pilot.press("enter")

        def state(*, count=1, provider="openai"):
            return LibraryRagPanelState.from_values(
                source_counts={"notes": count},
                query="What changed?",
                mode="rag",
                provider_name=provider,
            )

        try:
            assert app._instance_lock_status.acquired
            assert type(app._driver).__name__ == "LinuxDriver"
            assert app.console.file.isatty()
            result["exclusive_profile"] = True
            await wait_for(
                lambda: (
                    getattr(app, "_ui_ready", False)
                    and type(app.screen).__name__ == "ChatScreen"
                ),
                "Console ready",
            )
            await pilot.press("ctrl+3")
            await wait_for(
                lambda: (
                    type(app.screen).__name__ == "LibraryScreen"
                    and app.screen._library_loaded
                ),
                "Library loaded",
            )
            await wait_for(
                lambda: bool(
                    app.screen.query(
                        "#library-hub-explore-all, #library-rail-explore-all, #library-row-browse-search"
                    )
                ),
                "Library navigation mounted",
            )
            if not app.screen.query("#library-row-browse-search"):
                selector = (
                    "#library-hub-explore-all"
                    if app.screen.query("#library-hub-explore-all")
                    else "#library-rail-explore-all"
                )
                await press(selector)
            await press("#library-row-browse-search")
            screen = app.screen
            await wait_for(
                lambda: bool(screen.query("#library-rag-query-input")), "query"
            )
            if screen._rag_search_state.mode != "rag":
                await press("#library-rag-mode-toggle")
            await wait_for(lambda: screen._rag_search_state.mode == "rag", "RAG mode")
            query = screen.query_one("#library-rag-query-input", Input)
            query.focus()
            app.post_message(events.Paste("What changed?"))
            await wait_for(
                lambda: screen._rag_search_state.query == "What changed?", "query text"
            )
            original_state = screen._library_rag_panel_state
            current = [state(provider="")]
            screen._library_rag_panel_state = lambda: current[0]
            try:
                for theme in ("textual-dark", "textual-light"):
                    for width, height in ((170, 48), (80, 24)):
                        app.theme = theme
                        await tmux(
                            "resize-window",
                            "-t",
                            terminal,
                            "-x",
                            str(width),
                            "-y",
                            str(height),
                        )
                        await wait_for(
                            lambda width=width, height=height: (
                                tuple(app.size) == (width, height)
                            ),
                            "matrix size",
                        )
                        current[0] = state(provider="")
                        await screen._refresh_search_rag_panel_state_widgets(
                            include_results_and_history=False
                        )
                        query.focus()
                        await pilot.pause()
                        quiet = screen.query_one(
                            "#library-rag-query-quiet-line", Static
                        )
                        run = screen.query_one("#library-rag-run-query", Button)
                        assert run.disabled
                        callout = screen.query_one(
                            "#library-rag-query-blocked-callout", Static
                        )
                        remove = callout.remove
                        removed, release = asyncio.Event(), asyncio.Event()

                        async def held_remove(
                            remove=remove, removed=removed, release=release
                        ):
                            await remove()
                            removed.set()
                            await release.wait()

                        callout.remove = held_remove
                        current[0] = state(count=0)

                        async def refresh_status():
                            async with screen._rag_search_state.panel_refresh_lock:
                                await screen._refresh_library_rag_query_status_widgets(
                                    current[0]
                                )

                        refresh = asyncio.create_task(refresh_status())
                        try:
                            await asyncio.wait_for(removed.wait(), timeout=5)
                            current[0] = state()
                            screen._sync_library_rag_scope_toggle_and_run_gate_widgets()
                            assert not run.disabled
                            assert (
                                screen.query_one(
                                    "#library-rag-query-quiet-line", Static
                                )
                                is quiet
                            )
                            assert str(quiet.render()) == library_rag_query_quiet_text(
                                current[0]
                            )
                            await pilot.pause()
                            during_paint = painted(quiet)
                            assert during_paint.strip() == library_rag_query_quiet_text(
                                current[0]
                            )
                        finally:
                            release.set()
                            await asyncio.wait_for(refresh, timeout=5)
                        assert not run.disabled
                        assert str(quiet.render()) == library_rag_query_quiet_text(
                            current[0]
                        )
                        assert screen.focused is query
                        assert query.value == "What changed?"
                        await pilot.pause()
                        name = theme + "-" + str(width)
                        app.save_screenshot(name + "-ready.svg", path=str(evidence))
                        ready_paint = painted(quiet)
                        assert ready_paint.strip() == library_rag_query_quiet_text(
                            current[0]
                        )
                        current[0] = state(count=0)
                        screen._sync_library_rag_scope_toggle_and_run_gate_widgets()
                        assert run.disabled
                        assert str(quiet.render()) == ""
                        await pilot.pause()
                        app.save_screenshot(name + "-blocked.svg", path=str(evidence))
                        result["steps"].append(
                            {
                                "theme": theme,
                                "size": [width, height],
                                "during_yield_paint": during_paint,
                                "settled_notice_paint": ready_paint,
                                "coherent_gate_during_and_after_yield": True,
                                "missing_scope_disables_run": True,
                                "query_focus_and_value_retained": True,
                            }
                        )
                        record()
            finally:
                screen._library_rag_panel_state = original_state
            result["passed"] = True
        except Exception:  # noqa: BLE001 - retain evidence and exit normally
            result["passed"] = False
            result["error"] = traceback.format_exc()
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            record()
            await tmux("send-keys", "-t", terminal, "C-q")

    app.run(auto_pilot=journey, size=(170, 48))
    result["app_run_returned"] = True
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
