"""Native keyboard reading of long answers and all three citation feedback states.

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
            query = "auditanswernavigation"
            title = "Answer navigation source"
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
            cycle = {"marker": ""}
            lines = [
                f"Answer line {i:03d}: an expired credential caused the incident."
                for i in range(1, 81)
            ]

            def answer(**kwargs):
                calls.append(kwargs)
                assert title in kwargs["messages_payload"][0]["content"]
                return "\n".join(lines) + cycle["marker"]

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

            def painted_rows(selector):
                widget_region = screen.query_one(selector).content_region
                region = widget_region.intersection(panel.content_region)
                strips = screen._compositor.render_strips()
                return {
                    y - widget_region.y: strips[y].crop(region.x, region.right).text
                    for y in range(max(0, region.y), min(region.bottom, len(strips)))
                }

            def reading(rows):
                return " ".join(" ".join(rows[y] for y in sorted(rows)).split())

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
                    for citation_status, marker, feedback in (
                        (
                            "uncited",
                            "",
                            "The answer does not cite available staged evidence.",
                        ),
                        (
                            "unverified",
                            " [S99]",
                            "Some citation markers do not match available staged evidence.",
                        ),
                        ("validated", " [S1]", "Citations resolve to staged evidence."),
                    ):
                        cycle["marker"] = marker
                        # Each matrix cell starts from a visible focused query.
                        # Setup is programmatic; the reading route below uses keys only.
                        field = screen.query_one("#library-rag-query-input", Input)
                        field.value = query
                        field.focus()
                        screen.query_one("#library-search-rag-panel").scroll_to_widget(
                            field, animate=False
                        )
                        await wait_for(
                            lambda field=field: (
                                screen.focused is field
                                and screen._rag_search_state.query == query
                                and painted(field)
                                and not screen.query_one(
                                    "#library-rag-run-query", Button
                                ).disabled
                            ),
                            "query ready",
                        )
                        scope = screen._library_rag_panel_state().scope.selected_source_types
                        previous_answer = screen._rag_search_state.answer
                        await pilot.press("enter")
                        feedback_id = (
                            "#library-rag-answer-citation-note"
                            if citation_status == "validated"
                            else "#library-rag-answer-caution"
                        )
                        await wait_for(
                            lambda previous_answer=previous_answer, feedback_id=feedback_id: (
                                screen._rag_search_state.answer is not None
                                and screen._rag_search_state.answer
                                is not previous_answer
                                and screen._rag_search_state.answer.status == "ready"
                                and bool(screen.query(feedback_id))
                            ),
                            "answer and citation feedback ready",
                        )
                        async with screen._rag_search_state.panel_refresh_lock:
                            pass
                        await pilot.wait_for_scheduled_animations()
                        await pilot.pause()
                        state = screen._rag_search_state.answer
                        reply = "\n".join(lines) + marker
                        assert state.citation_status == citation_status
                        assert state.text == reply
                        assert screen.focused is field and painted(field)
                        panel = screen.query_one("#library-search-rag-panel")
                        card_id = "library-rag-result-card-0"
                        tab_stops = []
                        for _ in range(15):
                            await pilot.press("tab")
                            await pilot.wait_for_scheduled_animations()
                            await pilot.pause()
                            tab_stops.append(getattr(screen.focused, "id", None))
                            if tab_stops[-1] == card_id:
                                break
                        card = screen.query_one("#" + card_id)
                        assert screen.focused is card
                        trace = []
                        captured_feedback = False
                        captured_tail = False
                        for key, edge in (
                            ("pageup", 0),
                            ("pagedown", panel.max_scroll_y),
                        ):
                            answer_rows, feedback_rows = {}, {}
                            for _ in range(40):
                                current_answer = painted_rows(
                                    "#library-rag-answer-text"
                                )
                                current_feedback = painted_rows(feedback_id)
                                answer_rows.update(current_answer)
                                feedback_rows.update(current_feedback)
                                trace.append({"key": key, "scroll_y": panel.scroll_y})
                                if (
                                    not captured_feedback
                                    and reading(current_feedback) == feedback
                                ):
                                    app.save_screenshot(
                                        f"{theme}-{width}-{citation_status}.svg",
                                        path=str(evidence),
                                    )
                                    captured_feedback = True
                                if (
                                    citation_status == "validated"
                                    and not captured_tail
                                    and lines[-1] + marker in reading(current_answer)
                                ):
                                    app.save_screenshot(
                                        f"{theme}-{width}-answer-tail.svg",
                                        path=str(evidence),
                                    )
                                    captured_tail = True
                                if panel.scroll_y == edge:
                                    break
                                await pilot.press(key)
                                await pilot.wait_for_scheduled_animations()
                                await pilot.pause()
                            assert panel.scroll_y == edge, (key, panel.scroll_y, edge)
                            assert reading(answer_rows) == " ".join(reply.split()), (
                                key,
                                answer_rows,
                            )
                            assert reading(feedback_rows) == feedback, (
                                key,
                                feedback_rows,
                            )
                            assert screen.focused is card
                        assert captured_feedback
                        assert citation_status != "validated" or captured_tail
                        assert field.value == query
                        assert (
                            screen._library_rag_panel_state().scope.selected_source_types
                            == scope
                        )
                        assert screen._rag_search_state.answer is state
                        rows = screen._rag_search_state.results
                        assert (
                            len(rows) == 1
                            and rows[0].source_id == str(media_id)
                            and rows[0].title == title
                        )
                        await pilot.press("tab")
                        await pilot.wait_for_scheduled_animations()
                        await pilot.pause()
                        assert screen.focused is not card
                        assert screen.focused in card.walk_children()
                        assert reading(painted_rows(f"#{screen.focused.id}"))
                        expected = len(result["cells"]) + 1
                        assert len(completed) == len(calls) == expected
                        result["cells"].append(
                            {
                                "theme": theme,
                                "size": [width, height],
                                "citation_status": citation_status,
                                "answer_lines": len(lines),
                                "complete_answer_and_feedback_painted_both_directions": True,
                                "query_scope_answer_preserved": True,
                                "no_extra_retrieval_or_provider_calls": True,
                                "evidence_action_reachable": screen.focused.id,
                                "tab_stops": tab_stops,
                                "scroll_trace": trace,
                            }
                        )
                        record()
            after = app.media_db.get_media_by_id(media_id)
            (evidence / "source-after.json").write_text(
                json.dumps(after, indent=2, default=str) + "\n"
            )
            assert before == after
            assert len(completed) == 12 and len(calls) == 12
            result.update(
                real_keyword_searches=12,
                controlled_provider_calls=12,
                source_unchanged=True,
                passed=True,
            )
        except Exception:  # noqa: BLE001 - preserve diagnostics and quit normally
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
