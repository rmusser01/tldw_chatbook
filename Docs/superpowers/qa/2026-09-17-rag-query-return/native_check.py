"""Native RAG resize focus retention and keyboard return to the query.

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
            query = "auditqueryreturn"
            title = "Query return source"
            source_text = query + " The incident was caused by an expired credential."
            media_id, _, _ = app.media_db.add_media_with_keywords(
                title=title,
                media_type="document",
                content=source_text,
                keywords=[query],
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

            async def keyword_for_rag(query, source_types, mode, **kwargs):
                assert mode == "rag"
                outcome = await real_search(query, source_types, "search", **kwargs)
                completed.append(query)
                return outcome

            app.library_rag_search_service = type(
                "KeywordAdapter", (), {"search": staticmethod(keyword_for_rag)}
            )()
            calls = []
            reply = (
                "\n".join(
                    f"Answer line {i:03d}: an expired credential caused the incident."
                    for i in range(1, 81)
                )
                + " [S1]"
            )

            def answer(**kwargs):
                calls.append(kwargs)
                assert title in kwargs["messages_payload"][0]["content"]
                return reply

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
            if screen._rag_search_state.mode != "rag":
                await press("#library-rag-mode-toggle")
            for theme in ("textual-dark", "textual-light"):
                app.theme = theme
                for origin in ("query", "evidence"):
                    # Set up each independent cell at a visible focused query.
                    field = screen.query_one("#library-rag-query-input", Input)
                    panel = screen.query_one("#library-search-rag-panel")
                    field.value = query
                    field.focus()
                    panel.scroll_to_widget(field, animate=False)
                    await wait_for(
                        lambda field=field: (
                            screen.focused is field
                            and painted(field)
                            and screen._rag_search_state.query == query
                            and not screen.query_one(
                                "#library-rag-run-query", Button
                            ).disabled
                        ),
                        "query ready",
                    )
                    scope = (
                        screen._library_rag_panel_state().scope.selected_source_types
                    )
                    previous = screen._rag_search_state.answer
                    await pilot.press("enter")
                    await wait_for(
                        lambda previous=previous: (
                            screen._rag_search_state.answer is not None
                            and screen._rag_search_state.answer is not previous
                            and screen._rag_search_state.answer.status == "ready"
                            and bool(screen.query("#library-rag-answer-citation-note"))
                        ),
                        "answer ready",
                    )
                    async with screen._rag_search_state.panel_refresh_lock:
                        pass
                    await pilot.wait_for_scheduled_animations()
                    await pilot.pause()
                    state = screen._rag_search_state.answer
                    assert state.text == reply and state.citation_status == "validated"
                    assert screen.focused is field and painted(field)
                    if origin == "evidence":
                        for _ in range(15):
                            await pilot.press("tab")
                            await pilot.wait_for_scheduled_animations()
                            await pilot.pause()
                            if (
                                getattr(screen.focused, "id", None)
                                == "library-rag-result-card-0"
                            ):
                                break
                        assert (
                            getattr(screen.focused, "id", None)
                            == "library-rag-result-card-0"
                        )
                        await pilot.press(
                            "pageup", "pageup", "pagedown", "pagedown", "tab"
                        )
                        await pilot.wait_for_scheduled_animations()
                        await pilot.pause()
                        assert (
                            getattr(screen.focused, "id", None)
                            == "library-rag-open-result-0"
                        )
                    retained = screen.focused
                    assert painted(retained)
                    selection = field.selection
                    observations = []
                    for width, height in ((80, 24), (170, 48), (170, 24), (170, 48)):
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
                        await wait_for(
                            lambda retained=retained, width=width: (
                                screen._notes_state.compact == (width < 120)
                                and screen.focused is retained
                                and painted(retained)
                            ),
                            "retained focus painted after resize",
                        )
                        await pilot.wait_for_scheduled_animations()
                        await pilot.pause()
                        assert screen.focused is retained and painted(retained)
                        assert (
                            screen.query_one("#library-rag-query-input", Input) is field
                        )
                        assert field.value == query and field.selection == selection
                        assert (
                            screen._rag_search_state.answer is state
                            and state.text == reply
                        )
                        assert screen._rag_search_state.mode == "rag"
                        assert (
                            screen._library_rag_panel_state().scope.selected_source_types
                            == scope
                        )
                        observations.append(
                            {
                                "size": [width, height],
                                "focus": retained.id,
                                "region": list(retained.region),
                                "painted": True,
                            }
                        )
                        if width == 80:
                            app.save_screenshot(
                                f"{theme}-{origin}-compact.svg", path=str(evidence)
                            )
                    reverse_stops = []
                    for _ in range(15):
                        if (
                            getattr(screen.focused, "id", None)
                            == "library-rag-query-input"
                        ):
                            break
                        await pilot.press("shift+tab")
                        await pilot.wait_for_scheduled_animations()
                        await pilot.pause()
                        reverse_stops.append(getattr(screen.focused, "id", None))
                    assert screen.focused is field and painted(field)
                    assert field.value == query
                    assert screen._rag_search_state.answer is state
                    await pilot.press("end", "space", "x")
                    await wait_for(
                        lambda: screen._rag_search_state.query == query + " x",
                        "edited query ready",
                    )
                    assert (
                        field.value == query + " x"
                        and screen.focused is field
                        and painted(field)
                    )
                    assert screen._rag_search_state.mode == "rag"
                    expected = len(result["cells"]) + 1
                    assert len(completed) == len(calls) == expected
                    rows = screen._rag_search_state.results
                    assert (
                        len(rows) == 1
                        and rows[0].source_id == str(media_id)
                        and rows[0].title == title
                    )
                    app.save_screenshot(
                        f"{theme}-{origin}-query-edited.svg", path=str(evidence)
                    )
                    result["cells"].append(
                        {
                            "theme": theme,
                            "origin": origin,
                            "transitions": observations,
                            "reverse_tab_stops": reverse_stops,
                            "query_scope_answer_preserved_during_resize": True,
                            "query_editable_without_submission": True,
                            "no_extra_retrieval_or_provider_calls": True,
                        }
                    )
                    record()
            after = app.media_db.get_media_by_id(media_id)
            (evidence / "source-after.json").write_text(
                json.dumps(after, indent=2, default=str) + "\n"
            )
            assert before == after
            assert len(completed) == len(calls) == 4
            result.update(
                real_keyword_searches=4,
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
