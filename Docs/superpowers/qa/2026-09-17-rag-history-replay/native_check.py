"""Native keyboard replay and clearing of Recent searches.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
Real local keyword retrieval is adapted to rag mode; the provider seam is controlled.
"""

import asyncio
import hashlib
import json
import os
import runpy
import subprocess
import sys
import tomllib
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
    from textual.widgets import Button, Collapsible, Input
    from textual_image._terminal import probe_terminal

    import tldw_chatbook.app as app_module
    from tldw_chatbook.app import TldwCli

    assert Path(app_module.__file__).resolve() == (
        Path(__file__).resolve().parents[4] / "tldw_chatbook/app.py"
    )
    from tldw_chatbook import config as app_config

    app_config.default_api_endpoint = "openai"
    os.environ["OPENAI_API_KEY"] = "sk-test-native-controlled-answer"
    probe_terminal()
    app = TldwCli()
    result = {
        "pid": os.getpid(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cells": [],
    }
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
            raise AssertionError(
                f"{label}: focus={getattr(app.screen.focused, 'id', None)}"
            )

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
            alpha, beta = "audithistoryalpha", "audithistorybeta"
            title = "Recent search replay source"
            media_id, _, _ = app.media_db.add_media_with_keywords(
                title=title,
                media_type="document",
                content=alpha
                + " "
                + beta
                + " An expired credential caused the incident.",
                keywords=[alpha, beta],
            )
            registry = app.workspace_registry_service
            registry.link_membership(
                registry.get_active_workspace().workspace_id,
                item_type="media",
                item_id=str(media_id),
                title=title,
            )
            before = app.media_db.get_media_by_id(media_id)
            (evidence / "source-before.json").write_text(
                json.dumps(before, indent=2, default=str) + "\n"
            )
            real_search = app.library_rag_search_service.search

            async def keyword_adapter(query, source_types, mode, **kwargs):
                outcome = await real_search(query, source_types, "search", **kwargs)
                completed.append({"query": query, "scope": source_types, "mode": mode})
                return outcome

            app.library_rag_search_service = type(
                "KeywordAdapter", (), {"search": staticmethod(keyword_adapter)}
            )()
            calls = []

            def answer(**kwargs):
                calls.append(kwargs)
                assert title in kwargs["messages_payload"][0]["content"]
                return "An expired credential caused the incident. [S1]"

            app.library_rag_answer_chat = answer
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

            async def settle():
                async with screen._rag_search_state.panel_refresh_lock:
                    pass
                async with screen._rag_search_state.history_refresh_lock:
                    pass
                await pilot.wait_for_scheduled_animations()
                await pilot.pause()

            async def tab_to(predicate, key="tab"):
                for _ in range(30):
                    if predicate(screen.focused):
                        assert painted(screen.focused)
                        return screen.focused
                    await pilot.press(key)
                    await pilot.wait_for_scheduled_animations()
                    await pilot.pause()
                raise AssertionError(f"Keyboard target missing: {screen.focused!r}")

            async def open_history():
                title_widget = await tab_to(
                    lambda focused: (
                        type(focused).__name__ == "CollapsibleTitle"
                        and getattr(focused.parent, "id", None) == "library-rag-history"
                    )
                )
                history = screen.query_one("#library-rag-history", Collapsible)
                if history.collapsed:
                    await pilot.press("enter")
                    await settle()
                assert not history.collapsed
                return title_widget

            async def submit(query):
                field = screen.query_one("#library-rag-query-input", Input)
                field.value = query
                field.focus()
                screen.query_one("#library-search-rag-panel").scroll_to_widget(
                    field, animate=False
                )
                await wait_for(
                    lambda: (
                        screen._rag_search_state.query == query
                        and not screen.query_one(
                            "#library-rag-run-query", Button
                        ).disabled
                        and screen.focused is field
                        and painted(field)
                    ),
                    "query ready",
                )
                count = len(completed)
                await pilot.press("enter")
                await wait_for(
                    lambda: (
                        len(completed) == count + 1
                        and screen._rag_search_state.searched_query == query
                        and screen._rag_search_state.retrieval_status == "ready"
                    ),
                    "keyword result ready",
                )
                await settle()

            for theme in ("textual-dark", "textual-light"):
                app.theme = theme
                for size in ((170, 48), (80, 24)):
                    await tmux(
                        "resize-window",
                        "-t",
                        terminal,
                        "-x",
                        str(size[0]),
                        "-y",
                        str(size[1]),
                    )
                    await wait_for(
                        lambda size=size: tuple(app.size) == size, "terminal size"
                    )
                    for mode in ("search", "rag"):
                        if screen._rag_search_state.mode != "search":
                            await press("#library-rag-mode-toggle")
                            await settle()
                        # Each independent cell seeds history through two actual searches.
                        await submit(alpha)
                        await submit(beta)
                        assert screen._rag_search_state.history == (beta, alpha)
                        if mode == "rag":
                            await press("#library-rag-mode-toggle")
                            await settle()
                        scope = screen._library_rag_panel_state().scope.selected_source_types
                        assert scope == ("media",)
                        await open_history()
                        await tab_to(
                            lambda focused: (
                                getattr(focused, "id", None) == "library-rag-history-1"
                            )
                        )
                        before_calls, before_answers = len(completed), len(calls)
                        await pilot.press("enter")
                        await wait_for(
                            lambda before_calls=before_calls: (
                                len(completed) == before_calls + 1
                                and screen._rag_search_state.searched_query == alpha
                                and screen._rag_search_state.retrieval_status == "ready"
                            ),
                            "replay result ready",
                        )
                        if mode == "rag":
                            await wait_for(
                                lambda: (
                                    screen._rag_search_state.answer is not None
                                    and screen._rag_search_state.answer.status
                                    == "ready"
                                ),
                                "replay answer ready",
                            )
                        await settle()
                        assert completed[-1] == {
                            "query": alpha,
                            "scope": scope,
                            "mode": mode,
                        }
                        assert len(calls) == before_answers + (mode == "rag")
                        assert screen._rag_search_state.history == (alpha, beta)
                        assert (
                            screen.query_one("#library-rag-query-input", Input).value
                            == alpha
                        )
                        assert (
                            screen.query_one("#library-search-input", Input).value
                            == alpha
                        )
                        panel = screen.query_one("#library-search-rag-panel")
                        assert (
                            screen.focused is not None
                            and panel in screen.focused.ancestors
                            and painted(screen.focused)
                        )
                        replay_focus = type(screen.focused).__name__
                        rows, answer_state = (
                            screen._rag_search_state.results,
                            screen._rag_search_state.answer,
                        )
                        assert len(rows) == 1 and rows[0].source_id == str(media_id)
                        await tab_to(
                            lambda focused: (
                                getattr(focused, "id", None)
                                == "library-rag-query-input"
                            ),
                            "shift+tab",
                        )
                        stem = f"{theme}-{size[0]}-{mode}"
                        app.save_screenshot(stem + "-replayed.svg", path=str(evidence))
                        history_title = await open_history()
                        await tab_to(
                            lambda focused: (
                                getattr(focused, "id", None)
                                == "library-rag-history-clear"
                            )
                        )
                        await pilot.press("enter")
                        await wait_for(
                            lambda: bool(screen.query("#library-rag-history-empty")),
                            "history empty",
                        )
                        await settle()
                        assert screen.focused is history_title and painted(
                            history_title
                        )
                        assert painted(screen.query_one("#library-rag-history-empty"))
                        assert screen._rag_search_state.history == ()
                        assert screen._rag_search_state.results is rows
                        assert screen._rag_search_state.answer is answer_state
                        assert screen._rag_search_state.query == alpha
                        assert len(completed) == before_calls + 1
                        assert len(calls) == before_answers + (mode == "rag")
                        await wait_for(
                            lambda: (
                                tomllib.loads((root / "config.toml").read_text())
                                .get("library", {})
                                .get("search", {})
                                .get("history")
                                == []
                            ),
                            "cleared history persisted",
                        )
                        app.save_screenshot(stem + "-cleared.svg", path=str(evidence))
                        result["cells"].append(
                            {
                                "theme": theme,
                                "size": size,
                                "mode": mode,
                                "scope": scope,
                                "replay_focus": replay_focus,
                                "synchronized_inputs": True,
                                "replay_calls": 1,
                                "answer_calls": int(mode == "rag"),
                                "clear_preserves_query_results_answer": True,
                                "clear_focus": "CollapsibleTitle",
                                "empty_notice_painted": True,
                                "clear_persisted": True,
                            }
                        )
                        record()
            after = app.media_db.get_media_by_id(media_id)
            (evidence / "source-after.json").write_text(
                json.dumps(after, indent=2, default=str) + "\n"
            )
            assert before == after
            assert len(completed) == 24 and len(calls) == 4
            result.update(
                real_keyword_searches=24,
                controlled_provider_calls=4,
                source_unchanged=True,
                passed=True,
            )
        except Exception:  # noqa: BLE001 - record failure and quit normally
            result.update(passed=False, error=traceback.format_exc())
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
