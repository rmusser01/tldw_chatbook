"""Native answer failure/empty reply followed by a controlled cited retry.

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

        release = threading.Event()
        try:
            assert app._instance_lock_status.acquired
            assert type(app._driver).__name__ == "LinuxDriver"
            assert sys.stdout.isatty() and sys.stderr.isatty()
            await wait_for(lambda: getattr(app, "_ui_ready", False), "Console ready")
            query = "auditanswerrecovery"
            title = "Answer recovery source"
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
            cycle = {"count": 0, "failure": "exception"}

            def answer(**kwargs):
                calls.append(kwargs)
                cycle["count"] += 1
                assert title in kwargs["messages_payload"][0]["content"]
                if cycle["count"] == 1:
                    if cycle["failure"] == "exception":
                        raise RuntimeError("Controlled answer provider failure")
                    return ""
                assert release.wait(30), "Retry provider was not released"
                return "An expired credential caused the incident [S1]."

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
                    for failure in ("exception", "empty"):
                        cycle.update(count=0, failure=failure)
                        release.clear()
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
                        scope = screen._library_rag_panel_state().scope.selected_source_types
                        await pilot.press("enter")
                        await wait_for(
                            lambda: (
                                screen._rag_search_state.answer is not None
                                and screen._rag_search_state.answer.status == "failed"
                                and bool(screen.query("#library-rag-answer-error"))
                            ),
                            "answer failed",
                        )
                        async with screen._rag_search_state.panel_refresh_lock:
                            pass
                        await pilot.wait_for_scheduled_animations()
                        await pilot.pause()
                        notice = screen.query_one("#library-rag-retrieval-notice")
                        run = screen.query_one("#library-rag-run-query", Button)
                        disclosure = screen.query_one("#library-rag-query-quiet-line")
                        assert screen.focused is field and painted(field)
                        assert all(
                            painted(widget) for widget in (notice, run, disclosure)
                        )
                        assert (
                            notice.display and not run.disabled and field.value == query
                        )
                        content = " ".join(
                            strip.text for strip in screen._compositor.render_strips()
                        )
                        assert "Answer failed. Run again to retry." in content
                        assert "To openai: question + evidence" in content
                        assert cycle["count"] == 1
                        assert screen._rag_search_state.answer.error == (
                            "Controlled answer provider failure"
                            if failure == "exception"
                            else "The model returned an empty answer."
                        )
                        assert len(screen._rag_search_state.results) == 1
                        app.save_screenshot(
                            f"{theme}-{width}-{failure}.svg", path=str(evidence)
                        )
                        await pilot.press("enter")
                        await wait_for(
                            lambda: cycle["count"] == 2, "retry provider called"
                        )
                        assert screen._rag_search_state.answer_in_flight
                        assert run.disabled and not notice.display
                        assert "Answer failed" not in str(notice.render())
                        release.set()
                        await wait_for(
                            lambda: (
                                screen._rag_search_state.answer is not None
                                and screen._rag_search_state.answer.status == "ready"
                                and bool(
                                    screen.query("#library-rag-answer-citation-note")
                                )
                            ),
                            "cited answer landed",
                        )
                        async with screen._rag_search_state.panel_refresh_lock:
                            pass
                        await pilot.wait_for_scheduled_animations()
                        await pilot.pause()
                        assert screen.focused is field and painted(field)
                        assert (
                            field.value == query
                            and not notice.display
                            and not run.disabled
                        )
                        assert (
                            screen._library_rag_panel_state().scope.selected_source_types
                            == scope
                        )
                        assert not screen.query("#library-rag-answer-error")
                        assert (
                            screen._rag_search_state.answer.citation_status
                            == "validated"
                        )
                        assert (
                            "An expired credential"
                            in screen._rag_search_state.answer.text
                        )
                        rows = screen._rag_search_state.results
                        assert (
                            len(rows) == 1
                            and rows[0].source_id == str(media_id)
                            and rows[0].title == title
                        )
                        result["cells"].append(
                            {
                                "theme": theme,
                                "size": [width, height],
                                "failure": failure,
                                "query_scope_focus_retained": True,
                                "failure_run_disclosure_painted": True,
                                "notice_cleared_while_busy": True,
                                "cited_retry_answer": True,
                                "exact_local_evidence": True,
                            }
                        )
                        record()
                        if failure == "empty":
                            # Explicit inspection scroll; this does not qualify keyboard answer navigation.
                            panel = screen.query_one("#library-search-rag-panel")
                            panel.scroll_to_widget(
                                screen.query_one("#library-rag-answer-heading"),
                                top=True,
                                animate=False,
                            )
                            await pilot.pause()
                            await wait_for(
                                lambda: (
                                    "An expired credential"
                                    in " ".join(
                                        strip.text
                                        for strip in screen._compositor.render_strips()
                                    )
                                ),
                                "answer painted for inspection",
                            )
                            app.save_screenshot(
                                f"{theme}-{width}-retried.svg", path=str(evidence)
                            )
                            # Undo the inspection-only scroll before the next cell.
                            panel.scroll_to_widget(field, animate=False)
                            await pilot.pause()
            after = app.media_db.get_media_by_id(media_id)
            (evidence / "source-after.json").write_text(
                json.dumps(after, indent=2, default=str) + "\n"
            )
            assert before == after
            assert len(completed) == 16 and len(calls) == 16
            result.update(
                real_keyword_searches=16,
                controlled_provider_calls=16,
                source_unchanged=True,
                passed=True,
            )
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
