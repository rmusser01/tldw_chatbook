"""Native Re-chunk continuity across Library canvas and main navigation.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
Runs actual local re-chunking; semantic indexing is disabled in the private profile.
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
from datetime import UTC
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
    from textual.widgets import Button, Static
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
        from datetime import datetime

        from tldw_chatbook.Chunking.Chunk_Lib import ENGINE_VERSION
        from tldw_chatbook.Library.library_rechunk_service import (
            RECHUNK_SLOT,
            bulk_rag_slot_in_flight,
            format_rechunk_summary,
        )
        from tldw_chatbook.RAG_Search.ingestion_indexing import (
            semantic_indexing_available,
        )

        action = "#library-rag-rechunk-legacy"
        receipt = "#library-rag-rechunk-summary"
        report = "#library-rag-legacy-chunk-line"
        sources = {}
        runs = []
        entered = threading.Event()

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
            raise AssertionError(f"{label}: focus={app.screen.focused!r}")

        async def press(selector):
            # Compact Notes readers may collapse the retained Library rail.
            # Reopen it through the visible Nav grip before selecting a row.
            if selector.startswith("#library-row-"):
                grips = app.screen.query("#library-browse-library-grip")
                if grips and grips.first(Button).name == "Expand Library pane":
                    await press("#library-browse-library-grip")
                    await pilot.pause()
            await wait_for(lambda: bool(app.screen.query(selector)), selector)
            button = app.screen.query_one(selector, Button)
            assert not button.disabled
            button.focus()
            await wait_for(
                lambda: app.screen.focused is button and painted(button), selector
            )
            await pilot.press("enter")

        def seed(title, content):
            media_id, _, _ = app.media_db.add_media_with_keywords(
                title=title,
                media_type="plaintext",
                content=content,
                keywords=["rechunkreview"],
                chunks=[{"text": "OLD legacy chunk", "start_char": 0, "end_char": 16}],
                chunk_options={"size": 500, "max_size": 500, "overlap": 100},
            )
            registry = app.workspace_registry_service
            registry.link_membership(
                registry.get_active_workspace().workspace_id,
                item_type="media",
                item_id=str(media_id),
                title=title,
            )
            return int(media_id)

        def chunk_rows(media_id):
            return [
                dict(row)
                for row in app.media_db.get_connection().execute(
                    "SELECT chunk_text, chunk_engine_version FROM UnvectorizedMediaChunks "
                    "WHERE media_id = ? AND deleted = 0 ORDER BY chunk_index",
                    (media_id,),
                )
            ]

        try:
            assert app._instance_lock_status.acquired
            assert type(app._driver).__name__ == "LinuxDriver"
            assert sys.stdout.isatty() and sys.stderr.isatty()
            assert not semantic_indexing_available()
            await wait_for(lambda: getattr(app, "_ui_ready", False), "Console ready")
            skipped_id = seed(
                "Empty source retained for retry", "temporary fixture content"
            )
            now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            with app.media_db.transaction() as conn:
                conn.execute(
                    "UPDATE Media SET content = '', last_modified = ?, "
                    "version = version + 1 WHERE id = ?",
                    (now, skipped_id),
                )
            sources[str(skipped_id)] = app.media_db.get_media_by_id(skipped_id)
            skipped_chunks = chunk_rows(skipped_id)
            scope = app.rag_admin_scope_service
            real_run = scope.rechunk_legacy_media

            async def held_run(**kwargs):
                entered.set()
                assert await asyncio.to_thread(release.wait, 30)
                outcome = await real_run(**kwargs)
                runs.append(outcome)
                return outcome

            scope.rechunk_legacy_media = held_run
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
            panel = screen.query_one("#library-search-rag-panel")

            async def settle():
                async with screen._rag_search_state.panel_refresh_lock:
                    pass
                await pilot.wait_for_scheduled_animations()
                await pilot.pause()

            async def reveal(widget):
                # Real keyboard scrolling; focus setup chooses the scroll owner.
                panel.focus()
                await pilot.pause()
                for _ in range(100):
                    if painted(widget):
                        return
                    key = (
                        "down"
                        if widget.region.bottom > panel.content_region.bottom
                        else "up"
                    )
                    await pilot.press(key)
                    await pilot.wait_for_scheduled_animations()
                    await pilot.pause()
                raise AssertionError(
                    f"Keyboard scroll did not reveal {widget.id}: {widget.region}"
                )

            def painted_text(widget):
                region = widget.region
                return " ".join(
                    " ".join(
                        strip.crop(region.x, region.right).text
                        for strip in screen._compositor.render_strips()[
                            region.y : region.bottom
                        ]
                    ).split()
                )

            for theme in ("textual-dark", "textual-light"):
                app.theme = theme
                for size in ((170, 48), (80, 24)):
                    await wait_for(
                        lambda screen=screen: not screen.query("Toast"),
                        "prior notifications expired",
                    )
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
                    if screen._rag_search_state.mode != "search":
                        await press("#library-rag-mode-toggle")
                        await settle()
                    if (
                        "media"
                        not in screen._library_rag_panel_state().scope.selected_source_types
                    ):
                        await press("#library-rag-scope-toggle-media")
                        await settle()
                    media_id = seed(
                        f"Re-chunk source {len(runs) + 1}",
                        f"Source {len(runs) + 1}. "
                        + "An expired credential caused the incident. " * 30,
                    )
                    sources[str(media_id)] = app.media_db.get_media_by_id(media_id)
                    before_chunks = chunk_rows(media_id)
                    assert before_chunks and all(
                        row["chunk_engine_version"] is None for row in before_chunks
                    )
                    await screen._refresh_local_source_snapshot().wait()
                    panel._request_legacy_chunk_report_refresh()
                    await wait_for(
                        lambda screen=screen: (
                            str(screen.query_one(report, Static).renderable)
                            == "Chunked by an older engine: 2 items"
                        ),
                        "real legacy census",
                    )
                    stem = f"{theme}-{size[0]}"
                    entered.clear()
                    release.clear()
                    screen.query_one("#library-rag-mode-toggle", Button).focus()
                    for _ in range(20):
                        if getattr(screen.focused, "id", None) == action[1:]:
                            break
                        await pilot.press("tab")
                        await pilot.wait_for_scheduled_animations()
                        await pilot.pause()
                    assert screen.focused is screen.query_one(
                        action, Button
                    ) and painted(screen.focused)
                    before_runs = len(runs)
                    await pilot.press("enter")
                    try:
                        await wait_for(entered.is_set, "real Re-chunk worker started")
                        previous_panel = panel
                        await press("#library-row-browse-notes")
                        await wait_for(
                            lambda screen=screen: bool(
                                screen.query("#library-notes-canvas")
                            ),
                            "Notes canvas",
                        )
                        assert not previous_panel.is_attached
                        assert bulk_rag_slot_in_flight(RECHUNK_SLOT)
                        await press("#library-row-browse-search")
                        await wait_for(
                            lambda screen=screen: bool(screen.query(receipt)),
                            "returned Search/RAG",
                        )
                        panel = screen.query_one("#library-search-rag-panel")
                        assert panel is not previous_panel
                        await wait_for(
                            lambda screen=screen: (
                                screen.query_one(action, Button).disabled
                                and str(screen.query_one(receipt, Static).renderable)
                                == "Re-chunking…"
                            ),
                            "returned running feedback",
                        )
                        assert len(runs) == before_runs
                        await reveal(screen.query_one(receipt, Static))
                        assert "Re-chunking…" in painted_text(
                            screen.query_one(receipt, Static)
                        )
                        app.save_screenshot(stem + "-running.svg", path=str(evidence))
                        await pilot.press("ctrl+2")
                        await wait_for(
                            lambda screen=screen: app.screen is not screen,
                            "Console destination",
                        )
                        assert bulk_rag_slot_in_flight(RECHUNK_SLOT)
                    finally:
                        release.set()
                    await wait_for(
                        lambda before_runs=before_runs: (
                            len(runs) == before_runs + 1
                            and not bulk_rag_slot_in_flight(RECHUNK_SLOT)
                        ),
                        "Re-chunk completed while in Console",
                    )
                    await pilot.press("ctrl+3")
                    await wait_for(
                        lambda screen=screen: (
                            type(app.screen).__name__ == "LibraryScreen"
                            and app.screen._library_loaded
                        ),
                        "Library return",
                    )
                    screen = app.screen
                    if not screen.query("#library-row-browse-search"):
                        await press(
                            "#library-hub-explore-all"
                            if screen.query("#library-hub-explore-all")
                            else "#library-rail-explore-all"
                        )
                    await press("#library-row-browse-search")
                    await wait_for(
                        lambda screen=screen: bool(screen.query(receipt)),
                        "Search/RAG return",
                    )
                    panel = screen.query_one("#library-search-rag-panel")
                    await wait_for(
                        lambda screen=screen: (
                            str(screen.query_one(receipt, Static).renderable)
                            == format_rechunk_summary(runs[-1])
                        ),
                        "returned completed receipt",
                    )
                    await wait_for(
                        lambda screen=screen: (
                            str(screen.query_one(report, Static).renderable)
                            == "Chunked by an older engine: 1 items"
                        ),
                        "updated legacy census",
                    )
                    assert (
                        runs[-1]["rechunked"] == 1
                        and runs[-1]["skipped"] == 1
                        and runs[-1]["failed"] == 0
                    )
                    line = format_rechunk_summary(runs[-1])
                    assert "re-index skipped (semantic index unavailable)" in line
                    await settle()
                    summary = screen.query_one(receipt, Static)
                    assert summary.display and str(summary.renderable) == line
                    assert not screen.query_one(action, Button).disabled
                    await wait_for(
                        lambda screen=screen: not screen.query("Toast"),
                        "completion toast expired",
                    )
                    await reveal(summary)
                    assert line in painted_text(summary)
                    app.save_screenshot(stem + "-completed.svg", path=str(evidence))
                    after_chunks = chunk_rows(media_id)
                    assert after_chunks and all(
                        row["chunk_engine_version"] == ENGINE_VERSION
                        for row in after_chunks
                    )
                    assert all(
                        "OLD legacy chunk" not in row["chunk_text"]
                        for row in after_chunks
                    )
                    assert chunk_rows(skipped_id) == skipped_chunks
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "media_id": media_id,
                            "legacy_before": 2,
                            "legacy_after": 1,
                            "summary": line,
                            "running_feedback_survives_notes_roundtrip": True,
                            "completion_in_console_visible_after_return": True,
                            "full_receipt_painted_after_keyboard_scroll": True,
                            "chunks_stamped": len(after_chunks),
                            "skipped_chunks_unchanged": True,
                        }
                    )
                    record()
            after = {key: app.media_db.get_media_by_id(int(key)) for key in sources}
            assert sources == after
            (evidence / "source-before.json").write_text(
                json.dumps(sources, indent=2, default=str) + "\n"
            )
            (evidence / "source-after.json").write_text(
                json.dumps(after, indent=2, default=str) + "\n"
            )
            assert len(runs) == 4
            result.update(
                passed=True,
                real_rechunk_runs=4,
                sources_unchanged=True,
                skipped_id=skipped_id,
                semantic_indexing_enabled=False,
            )
        except Exception:  # noqa: BLE001 - record failure and quit normally
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
