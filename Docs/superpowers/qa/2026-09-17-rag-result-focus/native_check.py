"""Native local retrieval: retained query and newer keyboard focus stay painted.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
The recording gate delays the real keyword service without changing its results.
"""

import asyncio
import hashlib
import json
import os
import runpy
import subprocess
import sys
import threading
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

    from textual.css.query import NoMatches, QueryError
    from textual.widgets import Button, Input
    from textual_image._terminal import probe_terminal

    import tldw_chatbook.app as app_module
    from tldw_chatbook.app import TldwCli

    assert Path(app_module.__file__).resolve() == (
        Path(__file__).resolve().parents[4] / "tldw_chatbook/app.py"
    )
    probe_terminal()
    app = TldwCli()
    result = {
        "pid": os.getpid(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cells": [],
    }
    release = threading.Event()
    entered = threading.Event()
    completed = []

    def record():
        (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    def painted(widget):
        geometry = app.screen._compositor.visible_widgets.get(widget)
        return (
            geometry is not None
            and geometry[0].intersection(geometry[1]) == geometry[0]
        )

    async def tmux(*args):
        await asyncio.to_thread(
            subprocess.run,
            ["/opt/homebrew/bin/tmux", "-L", socket, *args],
            check=True,
            text=True,
            capture_output=True,
        )

    async def journey(pilot):
        async def wait_for(predicate, label):
            result["waiting_for"] = label
            record()
            deadline = asyncio.get_running_loop().time() + 30
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
            assert not button.disabled
            button.focus()
            await wait_for(
                lambda: app.screen.focused is button and painted(button), selector
            )
            await pilot.press("enter")

        try:
            assert app._instance_lock_status.acquired
            assert type(app._driver).__name__ == "LinuxDriver"
            assert sys.stdout.isatty() and sys.stderr.isatty()
            await wait_for(lambda: getattr(app, "_ui_ready", False), "Console ready")
            query = "auditfocusarrival"
            media_id, _, _ = app.media_db.add_media_with_keywords(
                title="Focus arrival source",
                media_type="document",
                content=query + " exact source body.",
                keywords=[query],
            )
            registry = app.workspace_registry_service
            registry.link_membership(
                registry.get_active_workspace().workspace_id,
                item_type="media",
                item_id=str(media_id),
                title="Focus arrival source",
            )
            before = app.media_db.get_media_by_id(media_id)
            (evidence / "source-before.json").write_text(
                json.dumps(before, indent=2, default=str) + "\n"
            )
            real_search = app.library_rag_search_service.search

            async def held_search(query, source_types, mode, **kwargs):
                assert mode == "search"
                entered.set()
                assert await asyncio.to_thread(release.wait, 20), "service gate timeout"
                outcome = await real_search(query, source_types, mode, **kwargs)
                completed.append(query)
                return outcome

            app.library_rag_search_service.search = held_search
            await pilot.press("ctrl+3")
            await wait_for(
                lambda: (
                    type(app.screen).__name__ == "LibraryScreen"
                    and app.screen._library_loaded
                ),
                "Library loaded",
            )
            if not app.screen.query("#library-row-browse-search"):
                await press(
                    "#library-hub-explore-all"
                    if app.screen.query("#library-hub-explore-all")
                    else "#library-rail-explore-all"
                )
            await press("#library-row-browse-search")
            screen = app.screen
            if screen._rag_search_state.mode != "search":
                await press("#library-rag-mode-toggle")
            for theme in ("textual-dark", "textual-light"):
                app.theme = theme
                for width, height in ((170, 48), (80, 24)):
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
                        "terminal resize",
                    )
                    for newer_focus in (False, True):
                        field = screen.query_one("#library-rag-query-input", Input)
                        field.value = query
                        field.focus()
                        await wait_for(
                            lambda field=field: (
                                screen._rag_search_state.query == query
                                and painted(field)
                                and not screen.query_one(
                                    "#library-rag-run-query", Button
                                ).disabled
                            ),
                            "query ready",
                        )
                        entered.clear()
                        release.clear()
                        previous = len(completed)
                        await pilot.press("enter")
                        await wait_for(entered.is_set, "real service entered")
                        await wait_for(
                            lambda field=field: (
                                screen.focused is field and painted(field)
                            ),
                            "in-flight query visible",
                        )
                        target = field
                        if newer_focus:
                            await pilot.press("shift+tab")
                            target = screen.query_one("#library-rag-mode-toggle")
                            await wait_for(
                                lambda target=target: (
                                    screen.focused is target and painted(target)
                                ),
                                "new keyboard choice",
                            )
                        release.set()
                        await wait_for(
                            lambda previous=previous: (
                                len(completed) > previous
                                and screen._rag_search_state.retrieval_status == "ready"
                            ),
                            "result arrival",
                        )
                        async with screen._rag_search_state.panel_refresh_lock:
                            pass
                        await pilot.wait_for_scheduled_animations()
                        await pilot.pause()
                        assert screen.focused is target and painted(target)
                        assert field.value == query
                        rows = screen._rag_search_state.results
                        assert len(rows) == 1 and rows[0].source_id == str(media_id)
                        assert rows[0].title == "Focus arrival source"
                        name = "newer-focus" if newer_focus else "query-focus"
                        app.save_screenshot(
                            f"{theme}-{width}-{name}.svg", path=str(evidence)
                        )
                        result["cells"].append(
                            {
                                "theme": theme,
                                "size": [width, height],
                                "newer_focus": newer_focus,
                                "focused": target.id,
                                "focused_region": list(target.region),
                                "fully_painted": True,
                                "exact_local_result": True,
                            }
                        )
                        record()
                        for _ in range(40):
                            if (
                                getattr(screen.focused, "id", None)
                                == "library-rag-result-card-0"
                            ):
                                break
                            await pilot.press("tab")
                        else:
                            raise AssertionError("Tab never reached Evidence")
                        await wait_for(
                            lambda: (
                                "Focus arrival source"
                                in " ".join(
                                    strip.text
                                    for strip in screen._compositor.render_strips()
                                )
                            ),
                            "Evidence title visible",
                        )
                        if newer_focus:
                            app.save_screenshot(
                                f"{theme}-{width}-evidence-focus.svg",
                                path=str(evidence),
                            )
            after = app.media_db.get_media_by_id(media_id)
            (evidence / "source-after.json").write_text(
                json.dumps(after, indent=2, default=str) + "\n"
            )
            assert before == after
            assert len(completed) == 8
            result.update(real_keyword_searches=8, source_unchanged=True, passed=True)
        except Exception:  # noqa: BLE001 - record runner failure and quit normally
            result.update(passed=False, error=traceback.format_exc())
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            release.set()
            record()
            await tmux("send-keys", "-t", terminal, "C-q")

    app.run(auto_pilot=journey, size=(170, 48))
    result["app_run_returned"] = True
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
