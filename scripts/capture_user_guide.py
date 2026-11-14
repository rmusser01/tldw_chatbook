#!/usr/bin/env python3
"""Capture User Guide screenshots from the real app (TASK-38969, TASK-38970).

Run one target per process from the repository root::

    PYTHONPATH=. .venv/bin/python scripts/capture_user_guide.py <target>

Targets, and the image each one writes under ``Docs/User_Guide/images``:

* ``search-rag`` -- ``library/search-rag.svg``: Library > Search / RAG with
  one note found, scrolled to the panel title.
* ``file-notes`` -- ``library/file-notes.svg``: Library > Notes > Folder
  files, a demo folder chosen through the Choose folder... picker and one
  file open.
* ``watchlists-workflow`` -- ``watchlists/console-workflow-complete.svg``:
  the Console Watchlists workflow run to completion. The model's turns are
  scripted (``Tests/QA/test_console_watchlists_workflow_uat.py``); every
  tool, receipt and database write is the app's own.

Each run boots a real ``TldwCli`` in ``run_test(size=(235, 52))`` on a fresh
temporary profile: HOME, XDG_* and TLDW_CONFIG_PATH point into it, the
keyring is the null backend, the test suite's real-profile guard refuses
writes to the real profile, and sockets may reach loopback only, so no real
provider can be contacted. The SVG is written with its cdnjs font URLs
stripped (``Docs/User_Guide/_template.md``, step 5).
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import re
import shutil
import socket
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES = REPO_ROOT / "Docs" / "User_Guide" / "images"
SIZE = (235, 52)
_DROPPED_VARS = {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "SSH_AUTH_SOCK"}
_CREDENTIAL_VAR = re.compile(r"(KEY|_TOKEN|_SECRET|_PASSWORD|_USERNAME)$")
_REMOTE_FONT = re.compile(
    r',\s*url\("https://cdnjs\.cloudflare\.com/[^"]+"\) format\("woff2?"\)'
)


def strip_remote_fonts(svg: str) -> str:
    """Drop the cdnjs ``@font-face`` sources; ``local(...)`` stays."""
    return _REMOTE_FONT.sub("", svg)


def write_guide_svg(path: Path, svg: str) -> None:
    """Write a User Guide screenshot without remote font sources."""
    path.write_text(strip_remote_fonts(svg), encoding="utf-8")


def _isolate(profile: Path) -> None:
    """Point this process at ``profile`` before tldw_chatbook is imported."""
    from Tests import real_profile_guard

    real_profile_guard.install()
    home = profile / "home"
    config = profile / "config" / "config.toml"
    for directory in (home, profile / "data", config.parent):
        directory.mkdir(mode=0o700, parents=True)
    # Drop the proxies (httpx trusts *_PROXY by default) and every inherited
    # variable named like a credential: *KEY (FAL_KEY, OPENAI_API_KEY),
    # *_TOKEN, *_SECRET, *_PASSWORD, *_USERNAME and the ssh agent socket.
    for name in list(os.environ):
        if name.upper() in _DROPPED_VARS or _CREDENTIAL_VAR.search(name.upper()):
            del os.environ[name]
    os.environ.update(
        HOME=str(home),
        USERPROFILE=str(home),
        XDG_CONFIG_HOME=str(config.parent),
        XDG_DATA_HOME=str(profile / "data"),
        XDG_CACHE_HOME=str(profile / "cache"),
        TLDW_CONFIG_PATH=str(config),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        HF_HUB_OFFLINE="1",
    )
    config.write_text(
        "[general]\n"
        'users_name = "guide"\n'
        'default_tab = "chat"\n\n'
        "[splash_screen]\nenabled = false\n\n"
        "[first_run]\nsetup_completed = true\n\n"
        '[library.rail_state]\nlifecycle = "expanded"\n',
        encoding="utf-8",
    )

    real_connect = socket.socket.connect

    def loopback_only(sock: socket.socket, address) -> None:
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            if not ipaddress.ip_address(address[0]).is_loopback:
                raise ConnectionRefusedError(f"capture refuses {address[0]}")
        return real_connect(sock, address)

    socket.socket.connect = loopback_only


async def _wait(pilot, ready, what: str) -> None:
    deadline = time.monotonic() + 30
    while not ready():
        if time.monotonic() > deadline:
            raise TimeoutError(f"capture never reached: {what}")
        await pilot.pause(0.05)
    await pilot.pause()


def _screen_named(app, name: str) -> bool:
    return type(app.screen).__name__ == name and app.screen.is_mounted


async def _open_library_row(app, pilot, row: str):
    from textual.widgets import Button

    await pilot.press("ctrl+3")
    await _wait(pilot, lambda: _screen_named(app, "LibraryScreen"), "Library")
    screen = app.screen
    await _wait(pilot, lambda: bool(screen.query(f"#library-row-{row}")), row)
    screen.query_one(f"#library-row-{row}", Button).press()
    return screen


async def _search_rag(app, pilot) -> Path:
    from textual.widgets import Button, Input

    # A note made through Library > Create > New note, the way a user would.
    screen = await _open_library_row(app, pilot, "create-note")
    await _wait(
        pilot, lambda: bool(screen.query("#library-notes-create-blank")), "New note"
    )
    screen.query_one("#library-notes-create-blank", Button).press()
    await _wait(pilot, lambda: bool(screen.query("#library-note-body")), "note editor")
    screen.query_one("#library-note-title", Input).focus()
    await pilot.press(*"Release review")
    screen.query_one("#library-note-body").focus()
    await pilot.press(*"Open questions: migration order, rollback plan, reports.")
    screen.query_one("#library-note-save", Button).press()
    await pilot.pause(1.0)

    screen.query_one("#library-row-browse-search", Button).press()
    await _wait(pilot, lambda: bool(screen.query("#library-rag-query-input")), "query")
    screen.query_one("#library-rag-query-input", Input).focus()
    await pilot.press(*"open questions", "enter")
    await _wait(
        pilot, lambda: bool(screen.query(".library-rag-result-card")), "a result"
    )
    # TASK-38969: the panel title stays in the picture with the result.
    screen.query_one("#library-rag-panel-title").scroll_visible(
        top=True, animate=False, immediate=True
    )
    await pilot.pause(0.3)
    return IMAGES / "library" / "search-rag.svg"


async def _file_notes(app, pilot) -> Path:
    from textual.widgets import Button, Tree

    from tldw_chatbook.Third_Party.textual_fspicker.base_dialog import (
        FileSystemPickerScreen,
    )
    from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
        LibraryFileNotesWorkspace,
    )

    vault = Path(os.environ["HOME"]) / "Notes"
    (vault / "projects").mkdir(parents=True)
    (vault / "welcome.md").write_text(
        "# Welcome\n\nThese notes live in a folder on disk.\n"
        "Edits here are saved straight back to the file.\n",
        encoding="utf-8",
    )
    (vault / "ideas.md").write_text("# Ideas\n\n- Weekly review\n", encoding="utf-8")
    (vault / "projects" / "roadmap.md").write_text("# Roadmap\n", encoding="utf-8")

    screen = await _open_library_row(app, pilot, "browse-notes")
    await _wait(
        pilot, lambda: bool(screen.query("#library-notes-source-files")), "source strip"
    )
    screen.query_one("#library-notes-source-files", Button).press()
    await _wait(
        pilot, lambda: bool(screen.query("#file-notes-choose-root")), "Choose folder"
    )
    workspace = screen.query_one(LibraryFileNotesWorkspace)
    workspace.query_one("#file-notes-choose-root", Button).focus()
    await pilot.press("enter")
    await _wait(
        pilot,
        lambda: isinstance(app.screen_stack[-1], FileSystemPickerScreen),
        "the folder picker",
    )
    # The picker opens on its path field: replace it, go there, then Select.
    await pilot.press("ctrl+a", *str(vault), "enter")
    await pilot.pause(0.3)
    await pilot.press("tab", "enter")
    await _wait(pilot, lambda: workspace.root == vault.resolve(), "the chosen folder")
    await _wait(pilot, lambda: bool(workspace.query("#file-notes-tree")), "Files tree")
    tree = workspace.query_one("#file-notes-tree", Tree)
    tree.focus()
    for _ in range(12):
        node = tree.cursor_node
        if node is not None and str(node.label).endswith("welcome.md"):
            break
        await pilot.press("down")
    await pilot.press("enter")
    await _wait(
        pilot, lambda: workspace.current_path == "welcome.md", "welcome.md open"
    )
    await pilot.pause(0.5)
    return IMAGES / "library" / "file-notes.svg"


def guide_gateway(feed_urls, receipt_ready):
    """The mounted UAT's scripted planner, as the real app's Console needs it."""
    from Tests.QA.test_console_watchlists_workflow_uat import (
        BRIEFING_ONLY_MARKER,
        _ScriptedWatchlistsGateway,
    )

    class GuideGateway(_ScriptedWatchlistsGateway):
        def cached_context_window(self, _settings):
            return None  # the real app's context meter asks; nothing is cached

        async def stream_chat(self, *args, **kwargs):
            async for chunk in super().stream_chat(*args, **kwargs):
                yield chunk.replace(f" ({BRIEFING_ONLY_MARKER})", "")

    return GuideGateway(feed_urls, receipt_ready=receipt_ready)


def _arm_watchlists_workflow():
    """Script the model and the feed fetches; return the app-side wiring.

    The provider is a loopback llama.cpp entry that is never called: the
    Console's gateway is the UAT's scripted planner, and feeds and the
    briefing's own model call return canned demo text.
    """
    from Tests.QA.test_console_watchlists_workflow_uat import BRIEFING_ONLY_MARKER
    from tldw_chatbook import config
    from tldw_chatbook.Chat.provider_setup_persistence import persist_provider_setup
    from tldw_chatbook.Subscriptions import briefing_service
    from tldw_chatbook.Subscriptions import watchlists_operation_coordinator
    from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard

    mutation = wizard.build_first_run_provider_commit(
        wizard.FirstRunProviderDraft(
            provider="llama_cpp",
            endpoint="http://127.0.0.1:8791/v1/chat/completions",
            credential=wizard.ProviderCredentialDraft("none", "", 0),
        ),
        "local-model",
        config.load_cli_config_and_ensure_existence(force_reload=True),
    )
    if not persist_provider_setup(mutation).fully_applied:
        raise RuntimeError("could not save the demo provider")
    # Plain chat's light tool surface offers no Watchlists tools.
    if not config.save_setting_to_cli_config("tools", "full_agent_tools_enabled", True):
        raise RuntimeError("could not turn on Full agent tools")

    real_execute = briefing_service.execute_accepted_briefing

    async def scripted_briefing(db, briefing_id, **kwargs):
        return await real_execute(
            db,
            briefing_id,
            chat=lambda **_: (
                f"## Daily signals\n\n{BRIEFING_ONLY_MARKER} Three campaign "
                "signals share one indicator family [item 1] [item 2] [item 3]."
            ),
            **kwargs,
        )

    watchlists_operation_coordinator.execute_accepted_briefing = scripted_briefing

    async def scripted_feed(subscription):
        source_id = int(subscription["id"])
        return {
            "items": [
                {
                    "url": f"https://public.example/item-{source_id}",
                    "title": f"Campaign signal {source_id}",
                    "content": f"Observed indicator family {source_id}.",
                    "content_hash": f"guide-signal-{source_id}",
                    "content_kind": "article",
                    "content_format": "text",
                    "published_date": "2026-11-10T12:00:00+00:00",
                }
            ],
            "stats": {"fixture": "guide-no-network"},
        }

    def arm(app) -> None:
        def receipt_ready(kind: str) -> bool:
            if kind == "checks":
                runs = app.subscriptions_db.list_operations_for_agent(limit=10)
                return len(runs["source_runs"]) == 3 and all(
                    row["status"] == "completed" for row in runs["source_runs"]
                )
            rows = app.subscriptions_db.list_briefings(1)
            return bool(rows and rows[0]["status"] == "complete")

        feeds = [f"https://public.example/feed-{index}.xml" for index in (1, 2, 3)]
        gateway = guide_gateway(feeds, receipt_ready)
        app.console_provider_gateway_factory = lambda: gateway
        app.local_watchlists_service.run_executor = scripted_feed
        app.unified_mcp_service.set_global_default("allow")

    return arm


async def _watchlists_workflow(app, pilot) -> Path:
    from textual.widgets import Button

    gateway = app.console_provider_gateway_factory()

    await _wait(pilot, lambda: _screen_named(app, "ChatScreen"), "Console")
    console = app.screen
    await _wait(
        pilot, lambda: bool(console.query("#console-native-composer")), "composer"
    )
    console.query_one("#console-native-composer").load_draft(
        "Create a daily threat-intel Watchlist from my three feeds, check them, "
        "write a briefing, schedule it every 24 hours, and show me the result."
    )
    console.query_one("#console-send-message", Button).press()
    # Approve each round once, as the mounted UAT does: risky tools ask.
    approved: set[str] = set()
    deadline = time.monotonic() + 90
    while not (gateway.stage >= 12 and "provenance" in app.export_screenshot()):
        if time.monotonic() > deadline:
            raise TimeoutError(f"workflow stalled at stage {gateway.stage}")
        pending = console._task_resume_state.pending_approval
        round_id = str(pending.get("round_id", "")) if isinstance(pending, dict) else ""
        buttons = list(console.query(".approval-row-fast-approve"))
        if (
            round_id
            and round_id not in approved
            and buttons
            and not buttons[0].disabled
        ):
            approved.add(round_id)
            buttons[0].press()
        await pilot.pause(0.1)
    # The reply paints before the turn settles: wait until Stop is gone.
    await _wait(
        pilot,
        lambda: not console.query_one("#console-stop-generation").display,
        "the run to finish",
    )
    console.query_one("#console-native-transcript").scroll_end(
        animate=False, immediate=True
    )
    await pilot.pause(0.5)
    return IMAGES / "watchlists" / "console-workflow-complete.svg"


#: target -> (drive it to its state, optional setup run before the app boots)
TARGETS = {
    "search-rag": (_search_rag, None),
    "file-notes": (_file_notes, None),
    "watchlists-workflow": (_watchlists_workflow, _arm_watchlists_workflow),
}


def main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] not in TARGETS:
        print(f"usage: capture_user_guide.py {{{','.join(TARGETS)}}}", file=sys.stderr)
        return 2
    drive, before_boot = TARGETS[argv[0]]
    profile = Path(tempfile.mkdtemp(prefix="tldw-guide-capture-")).resolve()
    _isolate(profile)

    async def run() -> Path:
        from tldw_chatbook.app import TldwCli
        from tldw_chatbook.Chat.console_exchange_capture import install_safe_capture_trimming
        from tldw_chatbook.Chat.console_semantic_revision import install_semantic_coordination
        from tldw_chatbook.Sync_Interop.chat_sync_source import install_chat_sync_hashing

        # This script builds TldwCli() itself, so it is a composition root and
        # installs the fail-closed DB-layer ports the app entry points install.
        install_semantic_coordination()
        install_chat_sync_hashing()
        install_safe_capture_trimming()
        arm = before_boot() if before_boot else None
        app = TldwCli()
        if arm:
            arm(app)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.5)
            out = await drive(app, pilot)
            write_guide_svg(out, app.export_screenshot())
        return out

    status = 0
    try:
        print(f"wrote {asyncio.run(run()).relative_to(REPO_ROOT)}")
    except BaseException:
        traceback.print_exc()
        status = 1
    shutil.rmtree(profile, ignore_errors=True)
    # os._exit skips the guard's atexit report, so read its refusals here.
    from Tests import real_profile_guard

    if refused := real_profile_guard.take_violations():
        print(
            "refused writes to the real profile:", *refused, sep="\n", file=sys.stderr
        )
        status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # A real TldwCli can wait minutes at interpreter exit for a worker thread
    # after its screenshot (lessons-live-verification.md, TASK-36521).
    os._exit(status)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
