"""Native recovery routes with private SQLite data and no external requests.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
Provider transitions are ephemeral config fixtures, not saved Settings edits.
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
from unittest.mock import Mock


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
    os.environ.pop("OPENAI_API_KEY", None)

    from textual.css.query import NoMatches, QueryError
    from textual.widgets import Button, Input
    from textual_image._terminal import probe_terminal

    import tldw_chatbook.app as app_module
    from tldw_chatbook import config
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Library.library_rag_answer_service import (
        library_rag_answer_provider_gate,
    )
    from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId

    assert Path(app_module.__file__).resolve() == (
        Path(__file__).resolve().parents[4] / "tldw_chatbook/app.py"
    )
    probe_terminal()
    app = TldwCli()
    query = "draft [/archive] résumé"
    # Sentinels qualify navigation only; no retrieval/provider/ingest dispatch
    # may occur during the journey, including after the run gate becomes ready.
    search = Mock(side_effect=AssertionError("Unexpected retrieval"))
    answer = Mock(side_effect=AssertionError("Unexpected answer request"))
    ingest = Mock(side_effect=AssertionError("Unexpected ingest submission"))
    app.library_rag_search_service.search = search
    app.library_rag_answer_chat = answer
    app.submit_library_ingest_job = ingest
    result = {
        "pid": os.getpid(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cells": [],
    }

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
            raise AssertionError(f"{label}: focus={app.screen.focused!r}")

        async def settle():
            screen = app.screen
            if hasattr(screen, "_rag_search_state"):
                async with screen._rag_search_state.panel_refresh_lock:
                    pass
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()

        async def press(selector):
            if selector.startswith("#library-row-"):
                handles = app.screen.query("#library-rail-open")
                if handles and painted(handles.first(Button)):
                    await press("#library-rail-open")
                for grip in app.screen.query(".library-adaptive-reader-pane-grip"):
                    if isinstance(grip, Button) and grip.name == "Expand Library pane":
                        await press(f"#{grip.id}")
                        break
            await wait_for(lambda: bool(app.screen.query(selector)), selector)
            button = app.screen.query_one(selector, Button)
            assert not button.disabled
            button.focus()
            await wait_for(
                lambda: app.screen.focused is button and painted(button), selector
            )
            await pilot.press("enter")
            await settle()

        async def recovery(selector):
            app.screen.query_one("#library-rag-query-input", Input).focus()
            await settle()
            for _ in range(30):
                if getattr(app.screen.focused, "id", None) == selector[1:]:
                    assert painted(app.screen.focused)
                    return
                await pilot.press("tab")
                await settle()
            raise AssertionError(f"Recovery action unreachable: {selector}")

        async def library():
            await pilot.press("ctrl+3")
            await wait_for(
                lambda: (
                    type(app.screen).__name__ == "LibraryScreen"
                    and app.screen._library_loaded
                ),
                "Library ready",
            )
            if not app.screen.query("#library-row-browse-search"):
                await press(
                    "#library-hub-explore-all"
                    if app.screen.query("#library-hub-explore-all")
                    else "#library-rail-explore-all"
                )
            if not app.screen.query("#library-rag-query-input"):
                await press("#library-row-browse-search")
            await wait_for(
                lambda: bool(app.screen.query("#library-rag-query-input")), "Search"
            )

        async def size_cell(theme, size):
            app.theme = theme
            await tmux(
                "resize-window", "-t", terminal, "-x", str(size[0]), "-y", str(size[1])
            )
            await wait_for(lambda: tuple(app.size) == size, "terminal size")
            await settle()

        def draft(mode, scope):
            screen = app.screen
            assert screen.query_one("#library-rag-query-input", Input).value == query
            assert screen.query_one("#library-search-input", Input).value == query
            assert screen._rag_search_state.mode == mode
            assert (
                screen._library_rag_panel_state().scope.selected_source_types == scope
            )

        try:
            assert app._instance_lock_status.acquired
            assert type(app._driver).__name__ == "LinuxDriver"
            assert sys.stdout.isatty() and sys.stderr.isatty()
            await wait_for(lambda: getattr(app, "_ui_ready", False), "Console ready")
            await library()
            config.default_api_endpoint = ""
            assert library_rag_answer_provider_gate().provider is None
            for theme in ("textual-dark", "textual-light"):
                for size in ((170, 48), (80, 24)):
                    await size_cell(theme, size)
                    app.screen.query_one(
                        "#library-rag-query-input", Input
                    ).value = query
                    await settle()
                    await recovery("#library-rag-open-import-export")
                    assert app.screen.query_one(
                        "#library-rag-run-query", Button
                    ).disabled
                    stem = f"{theme}-{size[0]}"
                    app.save_screenshot(stem + "-import-link.svg", path=str(evidence))
                    await pilot.press("enter")
                    await wait_for(
                        lambda: bool(app.screen.query("#library-ingest-canvas")),
                        "Import",
                    )
                    assert app.screen._library_selected_row_id == "ingest-import-media"
                    assert not app.library_ingest_jobs.jobs()
                    await press("#library-row-browse-search")
                    draft("search", ())
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "route": "Import",
                            "retained": True,
                        }
                    )
                    record()

            media_id, _, _ = app.media_db.add_media_with_keywords(
                title="Recovery navigation source",
                media_type="plaintext",
                content="A private source used only to enable the provider gate.",
                keywords=["recoveryreview"],
            )
            registry = app.workspace_registry_service
            registry.link_membership(
                registry.get_active_workspace().workspace_id,
                item_type="media",
                item_id=str(media_id),
                title="Recovery navigation source",
            )
            before = app.media_db.get_media_by_id(media_id)
            await app.screen._refresh_local_source_snapshot().wait()
            await press("#library-rag-mode-toggle")
            for theme in ("textual-dark", "textual-light"):
                for size in ((170, 48), (80, 24)):
                    await size_cell(theme, size)
                    for endpoint in ("", "openai"):
                        config.default_api_endpoint = endpoint
                        gate = library_rag_answer_provider_gate()
                        assert gate.provider is None
                        assert bool(gate.credential_recovery) is bool(endpoint)
                        # Re-enter the canvas to render the changed real config gate.
                        await press("#library-row-browse-notes")
                        await press("#library-row-browse-search")
                        draft("rag", ("media",))
                        await recovery("#library-rag-open-provider-settings")
                        if not endpoint:
                            app.save_screenshot(
                                f"{theme}-{size[0]}-provider-link.svg",
                                path=str(evidence),
                            )
                        await pilot.press("enter")
                        await wait_for(
                            lambda: (
                                type(app.screen).__name__ == "SettingsScreen"
                                and app.screen.active_category
                                == SettingsCategoryId.PROVIDERS_MODELS.value
                                and bool(
                                    app.screen.query("#settings-providers-models-card")
                                )
                            ),
                            "Providers and Models settings",
                        )
                        if endpoint:
                            config.default_api_endpoint = "ollama"
                            assert (
                                library_rag_answer_provider_gate().provider == "ollama"
                            )
                        await library()
                        draft("rag", ("media",))
                        await wait_for(
                            lambda endpoint=endpoint: (
                                app.screen.query_one(
                                    "#library-rag-run-query", Button
                                ).disabled
                                is (not bool(endpoint))
                            ),
                            "current provider gate after return",
                        )
                        assert bool(
                            app.screen.query("#library-rag-open-provider-settings")
                        ) is (not bool(endpoint))
                        result["cells"].append(
                            {
                                "theme": theme,
                                "size": size,
                                "route": "Providers & Models",
                                "blocker": "credential" if endpoint else "unselected",
                                "returned_ready": bool(endpoint),
                                "retained": True,
                            }
                        )
                        record()
            after = app.media_db.get_media_by_id(media_id)
            assert before == after
            search.assert_not_called()
            answer.assert_not_called()
            ingest.assert_not_called()
            assert not app.library_ingest_jobs.jobs()
            result.update(
                passed=True,
                source_unchanged=True,
                retrieval_calls=search.call_count,
                answer_calls=answer.call_count,
                ingest_calls=ingest.call_count,
            )
        except Exception:  # noqa: BLE001 - preserve evidence and quit normally
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
