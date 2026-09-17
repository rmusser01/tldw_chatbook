"""Controlled source snapshots around native navigation, then real local search.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
No provider calls. Snapshot counts are injected; keyword retrieval uses the DB.
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
    probe_terminal()
    app = TldwCli()
    result = {
        "pid": os.getpid(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "injected snapshots; actual navigation, local retrieval and DB",
        "cells": [],
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

        async def press(selector, *, capture=None):
            await wait_for(lambda: bool(app.screen.query(selector)), selector)
            button = app.screen.query_one(selector, Button)
            assert not button.disabled, selector
            button.focus()
            await wait_for(lambda: app.screen.focused is button, "focus " + selector)
            if capture:
                await pilot.pause()
                painted = " ".join(
                    strip.text for strip in app.screen._compositor.render_strips()
                )
                assert "Open Import media" in painted
                assert "No Library sources yet" in painted
                app.save_screenshot(capture, path=str(evidence))
            await pilot.press("enter")

        try:
            assert app._instance_lock_status.acquired
            assert type(app._driver).__name__ == "LinuxDriver"
            assert app.console.file.isatty()
            result["exclusive_profile"] = True
            await wait_for(lambda: getattr(app, "_ui_ready", False), "Console ready")
            query = "auditrecovery"
            media_id, _, _ = app.media_db.add_media_with_keywords(
                title="Recovery audit source",
                media_type="document",
                content=query + " exact source body.",
                keywords=[query],
            )
            assert media_id
            registry = app.workspace_registry_service
            registry.link_membership(
                registry.get_active_workspace().workspace_id,
                item_type="media",
                item_id=str(media_id),
                title="Recovery audit source",
            )
            before = app.media_db.get_media_by_id(media_id)
            (evidence / "source-before.json").write_text(
                json.dumps(before, indent=2, default=str) + "\n"
            )
            searches = []
            search = app.library_rag_search_service.search

            async def record_search(query, source_types, mode, **kwargs):
                assert mode == "search"
                outcome = await search(query, source_types, mode, **kwargs)
                searches.append(query)
                return outcome

            app.library_rag_search_service.search = record_search
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
            screen._refresh_local_source_snapshot = lambda: None
            if screen._rag_search_state.mode != "search":
                await press("#library-rag-mode-toggle")
            screen.query_one("#library-rag-query-input", Input).value = query
            await wait_for(
                lambda: screen._rag_search_state.query == query, "query ready"
            )
            async with screen._rag_search_state.panel_refresh_lock:
                pass

            def snapshot(count):
                screen._apply_local_source_snapshot(
                    {
                        "notes": (),
                        "media": (before,) if count else (),
                        "conversations": (),
                    },
                    {"notes": 0, "media": count, "conversations": 0},
                    {"notes": True, "media": True, "conversations": True},
                )

            async def settled(count):
                await wait_for(
                    lambda: (
                        screen.query_one("#library-rag-source-scope").has_class(
                            "has-recovery"
                        )
                        is (count == 0)
                        and bool(screen.query("#library-rag-scope-recovery"))
                        is (count == 0)
                        and bool(screen.query("#library-rag-open-import-export"))
                        is (count == 0)
                    ),
                    f"scope settled at {count}",
                )

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
                    for initial in (1, 0):
                        snapshot(initial)
                        await settled(initial)
                        original = screen.query_one("#library-rag-source-scope")
                        await press("#library-row-browse-notes")
                        await wait_for(
                            lambda: bool(screen.query("#library-notes-canvas")), "Notes"
                        )
                        snapshot(1 - initial)
                        await pilot.pause()
                        if not screen._notes_state.reader_layout.library_open:
                            await press("#library-browse-library-grip")
                        await press("#library-row-browse-search")
                        await settled(1 - initial)
                        assert (
                            screen.query_one("#library-rag-source-scope")
                            is not original
                        )
                        field = screen.query_one("#library-rag-query-input", Input)
                        field.focus()
                        await wait_for(
                            lambda field=field: screen.focused is field, "query focused"
                        )
                        # No query edit here: a full query refresh would conceal
                        # the stale-cache window this snapshot must exercise.
                        assert (
                            screen._rag_search_state.scope_recovery_key[0]() is original
                        )
                        snapshot(initial)
                        await settled(initial)
                        assert field.value == query and screen.focused is field
                        run = screen.query_one("#library-rag-run-query", Button)
                        assert run.disabled is (initial == 0)
                        scope = screen.query_one("#library-rag-source-scope")
                        children = tuple(scope.children)
                        snapshot(initial)
                        await pilot.pause()
                        assert tuple(scope.children) == children
                        if initial:
                            completed = len(searches)
                            await pilot.press("enter")
                            await wait_for(
                                lambda completed=completed: (
                                    len(searches) > completed
                                    and bool(screen._rag_search_state.results)
                                ),
                                "local result",
                            )
                            rows = screen._rag_search_state.results
                            assert len(rows) == 1 and rows[0].source_id == str(media_id)
                            assert rows[0].title == "Recovery audit source"
                        suffix = "ready" if initial else "empty"
                        app.save_screenshot(
                            f"{theme}-{width}-{suffix}.svg", path=str(evidence)
                        )
                        result["cells"].append(
                            {
                                "theme": theme,
                                "size": [width, height],
                                "final_source_count": initial,
                                "query_focus_retained": True,
                                "steady_snapshot_retains_children": True,
                                "exact_local_result": bool(initial),
                                "focus_after_capture": getattr(
                                    screen.focused, "id", None
                                ),
                                "query_region_after_capture": list(field.region),
                            }
                        )
                        record()
                    await press(
                        "#library-rag-open-import-export",
                        capture=f"{theme}-{width}-recovery-action.svg",
                    )
                    await wait_for(
                        lambda: bool(screen.query("#library-ingest-canvas")),
                        "Import media",
                    )
                    await press("#library-row-browse-search")
                    await settled(0)
            after = app.media_db.get_media_by_id(media_id)
            (evidence / "source-after.json").write_text(
                json.dumps(after, indent=2, default=str) + "\n"
            )
            assert after == before
            assert len(searches) == 4
            result["real_keyword_searches"] = len(searches)
            result["source_record_unchanged"] = True
            result["passed"] = True
        except Exception:  # noqa: BLE001 - retain diagnostics, then normal quit
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
