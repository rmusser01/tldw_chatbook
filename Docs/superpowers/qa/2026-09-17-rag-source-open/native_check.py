"""Real local keyword retrieval/readers with an explicit legacy-ID adapter.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
Only result ID spellings and two malformed rows are injected; no provider is used.
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
    from textual.widgets import Button, Input, TextArea
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
        "seam": "real keyword search; legacy result-ID adapter; real local readers",
        "retrievals": [],
        "steps": [],
    }
    fixtures = {}

    def record():
        (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    def snapshot():
        return {
            query: {
                "media": app.media_db.get_media_by_id(f["media_id"]),
                "prompt": app.prompts_db.get_prompt_by_id(f["prompt_id"]),
            }
            for query, f in fixtures.items()
        }

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
            assert not button.disabled, selector
            button.focus()
            await wait_for(lambda: app.screen.focused is button, "focus " + selector)
            await pilot.press("enter")

        async def return_to_search(kind):
            layout = (
                app.screen._media_state.reader_layout
                if kind == "media"
                else app.screen._prompts_state.reader_layout
            )
            if not layout.library_open:
                await press(
                    "#library-browse-library-grip"
                    if kind == "media"
                    else "#library-prompts-library-grip"
                )
            await press("#library-row-browse-search")
            await wait_for(
                lambda: bool(app.screen.query("#library-rag-result-card-3")),
                "retained results",
            )

        try:
            assert app._instance_lock_status.acquired
            assert type(app._driver).__name__ == "LinuxDriver"
            assert app.console.file.isatty()
            result["exclusive_profile"] = True
            await wait_for(lambda: getattr(app, "_ui_ready", False), "Console ready")
            for query in (
                "auditdarkwide",
                "auditdarkcompact",
                "auditlightwide",
                "auditlightcompact",
            ):
                body = query + " exact stored body: [source], café."
                media_id, _, _ = app.media_db.add_media_with_keywords(
                    title=query + " Media",
                    media_type="document",
                    content=body,
                    keywords=[query],
                )
                prompt_id, _, _ = app.prompts_db.add_prompt(
                    name=query + " Prompt",
                    author=None,
                    details=None,
                    user_prompt=body,
                )
                assert media_id and prompt_id
                fixtures[query] = {
                    "media_id": media_id,
                    "prompt_id": prompt_id,
                    "body": body,
                }
                registry = app.workspace_registry_service
                registry.link_membership(
                    registry.get_active_workspace().workspace_id,
                    item_type="media",
                    item_id=str(media_id),
                    title=query + " Media",
                )
            before = snapshot()
            (evidence / "sources-before.json").write_text(
                json.dumps(before, indent=2, default=str) + "\n"
            )
            search = app.library_rag_search_service.search

            async def legacy_ids(query, source_types, mode, **kwargs):
                assert mode == "search", "Native check must not use a provider"
                outcome = await search(query, source_types, mode, **kwargs)
                assert isinstance(outcome, dict), repr(outcome)
                rows = [dict(row) for row in outcome["results"]]
                ids = {}
                for row in rows:
                    kind = row["provenance"]["source_type"]
                    assert kind in {"media", "prompt"}, row
                    original = row["source_id"]
                    assert original == str(fixtures[query][kind + "_id"]), row
                    row["source_id"] = f"{kind}_{original}"
                    ids[kind] = {"original": original, "wrapped": row["source_id"]}
                assert set(ids) == {"media", "prompt"}, rows
                for kind in ("media", "prompt"):
                    rows.append(
                        {
                            "source_id": f"{kind}_invalid",
                            "title": "Unresolvable " + kind,
                            "snippet": "Invalid identity recovery check",
                            "score": None,
                            "provenance": {"source_type": kind},
                        }
                    )
                result["retrievals"].append({"query": query, "mode": mode, "ids": ids})
                record()
                return {**outcome, "results": rows}

            app.library_rag_search_service.search = legacy_ids
            notices = []
            notify = app.notify

            def capture_notice(message, **kwargs):
                notices.append(str(message))
                return notify(message, **kwargs)

            app.notify = capture_notice
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
                "Library navigation",
            )
            if not app.screen.query("#library-row-browse-search"):
                await press(
                    "#library-hub-explore-all"
                    if app.screen.query("#library-hub-explore-all")
                    else "#library-rail-explore-all"
                )
            await press("#library-row-browse-search")
            screen = app.screen
            await wait_for(
                lambda: bool(screen.query("#library-rag-query-input")), "Search mounted"
            )
            if screen._rag_search_state.mode != "search":
                await press("#library-rag-mode-toggle")
            for index, (query, fixture) in enumerate(fixtures.items()):
                theme = "textual-dark" if index < 2 else "textual-light"
                width, height = (170, 48) if index % 2 == 0 else (80, 24)
                app.theme = theme
                await tmux(
                    "resize-window", "-t", terminal, "-x", str(width), "-y", str(height)
                )
                await wait_for(
                    lambda width=width, height=height: (
                        tuple(app.size) == (width, height)
                    ),
                    "matrix size",
                )
                field = screen.query_one("#library-rag-query-input", Input)
                field.value = query
                field.focus()
                await wait_for(
                    lambda query=query: (
                        screen._rag_search_state.query == query
                        and not screen.query_one(
                            "#library-rag-run-query", Button
                        ).disabled
                    ),
                    "query ready",
                )
                await pilot.press("enter")
                await wait_for(
                    lambda query=query: (
                        len(screen._library_rag_panel_state().results) == 4
                        and screen._library_rag_panel_state()
                        .results[0]
                        .title.startswith(query)
                    ),
                    "four results",
                )
                rows = screen._library_rag_panel_state().results
                for kind in ("media", "prompt"):
                    bad_index = next(
                        i
                        for i, row in enumerate(rows)
                        if row.source_id == kind + "_invalid"
                    )
                    card = screen.query_one(f"#library-rag-result-card-{bad_index}")
                    card.focus()
                    await pilot.pause()
                    count = len(notices)
                    await pilot.press("o")
                    await wait_for(
                        lambda count=count: len(notices) > count,
                        "invalid identity notice",
                    )
                    assert "Can't open" in notices[-1]
                    assert screen.focused is card
                    assert (
                        screen.query_one("#library-rag-query-input", Input).value
                        == query
                    )
                    if kind == "prompt":
                        app.save_screenshot(query + "-refusal.svg", path=str(evidence))
                    await wait_for(lambda: not screen.query("Toast"), "notices expired")
                media_index = next(
                    i
                    for i, row in enumerate(rows)
                    if row.source_id == f"media_{fixture['media_id']}"
                )
                await press(f"#library-rag-open-result-{media_index}")
                await wait_for(
                    lambda fixture=fixture: (
                        screen._media_state.reader_session.loaded_id
                        == f"local:media:{fixture['media_id']}"
                    ),
                    "exact Media record",
                )
                assert screen._media_state.detail["content"] == fixture["body"]
                app.save_screenshot(query + "-media.svg", path=str(evidence))
                await return_to_search("media")
                prompt_index = next(
                    i
                    for i, row in enumerate(rows)
                    if row.source_id == f"prompt_{fixture['prompt_id']}"
                )
                screen.query_one(f"#library-rag-result-card-{prompt_index}").focus()
                await pilot.pause()
                await pilot.press("o")
                await wait_for(
                    lambda query=query: (
                        bool(screen.query("#library-prompt-name"))
                        and screen.query_one("#library-prompt-name", Input).value
                        == query + " Prompt"
                    ),
                    "exact Prompt record",
                )
                assert (
                    screen.query_one("#library-prompt-user", TextArea).text
                    == fixture["body"]
                )
                app.save_screenshot(query + "-prompt.svg", path=str(evidence))
                screen.query_one("#library-prompt-user", TextArea).focus()
                await pilot.pause()
                app.save_screenshot(query + "-prompt-body.svg", path=str(evidence))
                result["steps"].append(
                    {
                        "theme": theme,
                        "size": [width, height],
                        "query": query,
                        "media_id": fixture["media_id"],
                        "prompt_id": fixture["prompt_id"],
                        "exact_source_bodies": True,
                        "refusals_keep_search_and_focus": True,
                    }
                )
                record()
                await return_to_search("prompt")
            after = snapshot()
            (evidence / "sources-after.json").write_text(
                json.dumps(after, indent=2, default=str) + "\n"
            )
            assert before == after
            result["source_records_unchanged"] = True
            result["passed"] = True
        except Exception:  # noqa: BLE001 - retain failure evidence and exit normally
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
