"""Native Test Tool result replacement and raw disclosure through a real private stdio server.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import runpy
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )


async def _save_fixture_profile(
    service: UnifiedMCPControlPlaneService, repo: Path, evidence: Path
) -> str:
    """Create the local fixture only when its ID is unoccupied.

    Args:
        service: Control plane bound to the validated private profile.
        repo: Checkout containing the local JSON-RPC fixture.
        evidence: Newly created exclusive directory for state and wire trace files.

    Returns:
        The successfully created profile ID, owned by this runner.

    Raises:
        ValueError: The fixture ID belongs to an existing profile.
    """
    if service.local_service.store.get_profile("execution-review") is not None:
        raise ValueError("Native fixture profile execution-review already exists")
    await service.save_local_profile(
        {
            "profile_id": "execution-review",
            "command": sys.executable,
            "args": [
                str(repo / "Tests/MCP/fixtures/stdio_tool_result_server.py"),
                str(evidence / "fixture-result.json"),
                str(evidence / "fixture-trace.jsonl"),
                str(evidence.resolve()),
            ],
            "env_placeholders": {},
            "env_literals": {},
        }
    )
    return "execution-review"


def main(*, size: tuple[int, int] = (170, 48)) -> None:
    """Qualify result replacement and raw response access in an unused private terminal profile.

    Args:
        size: Wide default or compact (80, 24) qualification size.

    Raises:
        ValueError: The requested terminal size is outside the two review cells.
        SystemExit: Zero for success, one for journey failure, or two for invalid CLI.
    """
    if size not in ((170, 48), (80, 24)):
        raise ValueError("Unsupported native review size")
    here = Path(__file__).resolve()
    repo = here.parents[5]
    sys.path.insert(0, str(repo))
    args = runpy.run_path(str(repo / "Docs/superpowers/qa/native_runner_args.py"))[
        "parse_native_args"
    ]()
    root, socket, session = args.root, args.tmux_socket, args.session
    tmux_path = args.tmux_path
    os.environ.update(
        HOME=str(root / "home"),
        USERPROFILE=str(root / "home"),
        TLDW_CONFIG_PATH=str(root / "config.toml"),
        XDG_CONFIG_HOME=str(root / "config"),
        XDG_DATA_HOME=str(root / "data"),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
    )
    for key in (
        "NO_COLOR",
        "OPENAI_API_KEY",
        "LLAMA_CPP_API_KEY",
        "SEARX_URL",
        "SERPER_API_KEY",
    ):
        os.environ.pop(key, None)
    sys.path.insert(0, str(repo))
    from loguru import logger
    from textual.containers import VerticalScroll
    from textual.css.query import NoMatches, QueryError
    from textual.widgets import Button, Collapsible, Static

    from Tests.network_guard import blocked_attempts, install

    install()
    logger.remove()
    logger.add(root / "native.log", level="INFO")

    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.MCP_Modules.mcp_inspector import MCPInspector
    from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode
    from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.Utils.app_shutdown import claim_process_exit
    from tldw_chatbook.Utils.terminal_utils import warm_up_image_protocol

    evidence = root / "evidence"
    evidence.mkdir(exist_ok=False)
    (root / "launch.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "started_at": datetime.now(UTC).isoformat(),
                "repo": str(repo),
                "head": subprocess.check_output(
                    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
                ).strip(),
                "dirty_status": subprocess.check_output(
                    ["git", "-C", str(repo), "status", "--short"], text=True
                ),
                "requested_size": list(size),
            },
            indent=2,
        )
        + "\n"
    )
    claim_process_exit()
    warm_up_image_protocol()
    app = TldwCli()
    paths = [
        "tldw_chatbook/UI/MCP_Modules/" + p + ".py"
        for p in ("mcp_servers_mode", "mcp_workbench", "mcp_profile_form")
    ]
    paths += [
        "tldw_chatbook/css/" + p
        for p in (
            "tldw_cli_modular.tcss",
            "widget_defaults_scoped.tcss",
            "widget_defaults_self.tcss",
            "core/_variables.tcss",
            "components/_agentic_terminal.tcss",
        )
    ]
    paths += [
        "tldw_chatbook/MCP/client.py",
        "tldw_chatbook/MCP/local_control_service.py",
        "tldw_chatbook/MCP/unified_control_plane_service.py",
        "tldw_chatbook/Utils/input_validation.py",
        "tldw_chatbook/UI/MCP_Modules/mcp_inspector.py",
        "Tests/MCP/fixtures/stdio_tool_result_server.py",
    ]
    result = {
        "pid": os.getpid(),
        "cells": [],
        "requested_size": list(size),
        "module_origins": {},
        "source_hashes": {
            p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in paths
        },
        "runner_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
    }

    def record():
        (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    async def tmux(*args):
        return await asyncio.to_thread(
            subprocess.run,
            [tmux_path, "-L", socket, *args],
            check=True,
            text=True,
            capture_output=True,
        )

    async def journey(pilot):
        async def settle():
            await pilot.pause()
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()

        async def wait_for(predicate, label):
            result["waiting_for"] = label
            record()
            async with asyncio.timeout(35):
                while True:
                    try:
                        if predicate():
                            break
                    except (NoMatches, QueryError):
                        pass
                    await pilot.pause(0.03)
            await settle()

        def visible(control):
            region, clip = app.screen._compositor.visible_widgets[control]
            assert region.width > 0 and region.height > 0
            assert region.intersection(clip) == region, (control.id, region, clip)
            hit = app.screen.get_widget_at(
                region.x + region.width // 2, region.y + region.height // 2
            )[0]
            assert hit is control or control in hit.ancestors, (control, hit)
            if isinstance(control, Button):
                painted = "".join(
                    app.screen._compositor.render_strips()[row]
                    .crop(region.x, region.right)
                    .text
                    for row in range(region.y, region.bottom)
                )
                assert "".join(str(control.label).split()) in "".join(
                    painted.split()
                ), (
                    control.id,
                    str(control.label),
                    painted,
                )

        async def focus(selector):
            control = app.screen.query_one(selector)
            control.focus()
            control.scroll_visible(immediate=True)
            await settle()
            assert control.has_focus
            visible(control)
            if isinstance(control, Button):
                await wait_for(lambda: not control.has_class("-active"), "button ready")
            return control

        async def capture(stem):
            await wait_for(lambda: not app._notifications, "notices clear")
            app.save_screenshot(stem + ".svg", path=str(evidence))
            pane = await tmux("capture-pane", "-p", "-t", session)
            (evidence / (stem + ".txt")).write_text(pane.stdout)

        async def tab_to(selector):
            control = app.screen.query_one(selector)
            if control.has_focus:
                visible(control)
                return control
            for _ in range(30):
                await pilot.press("tab")
                await settle()
                if control.has_focus:
                    visible(control)
                    return control
            raise AssertionError(f"Keyboard traversal never reached {selector}")

        fixture_id = None
        process = None
        try:
            assert app._instance_lock_status.acquired
            assert type(app._driver).__name__ == "LinuxDriver"
            assert (
                sys.stdout.isatty()
                and sys.stderr.isatty()
                and app.console.file.isatty()
            )
            result.update(driver="LinuxDriver", tty_streams=True, lock_acquired=True)
            await wait_for(lambda: getattr(app, "_ui_ready", False), "startup")
            await app.handle_screen_navigation(NavigateToScreen("mcp"))
            await wait_for(
                lambda: getattr(app.screen, "screen_name", None) == "mcp",
                "MCP destination",
            )
            workbench = app.screen.workbench
            await wait_for(
                lambda: (
                    not workbench.is_loading
                    and not workbench._reloading
                    and bool(workbench.query(MCPServersMode))
                ),
                "MCP loaded",
            )
            service = app.unified_mcp_service
            trace = evidence / "fixture-trace.jsonl"
            state = evidence / "fixture-result.json"
            state.write_text(json.dumps({"content": []}))
            client = service.local_service._get_client()
            fixture_id = await _save_fixture_profile(service, repo, evidence)
            await service.connect_local_profile("execution-review")
            process = client.sessions["execution-review"].process
            result["fixture_pid"] = process.pid
            names = {
                type(app).__module__,
                type(client).__module__,
                type(service).__module__,
                type(service.local_service).__module__,
                type(workbench).__module__,
                MCPInspector.__module__,
                MCPToolsMode.__module__,
                "tldw_chatbook.Utils.input_validation",
            }
            for name in sorted(names):
                origin = Path(sys.modules[name].__file__).resolve()
                assert origin.is_relative_to(repo), (name, origin)
                result["module_origins"][name] = str(origin)
            workbench._snapshots = await workbench._collect_snapshots()
            await workbench._sync_children()
            workbench.set_mode("tools")
            await settle()
            mode = workbench.query_one(MCPToolsMode)
            assert await mode.select_tool_row("local:execution-review::review_echo")
            await focus("#mcp-tools-table")
            await pilot.press("enter")
            inspector = workbench.query_one(MCPInspector)
            await wait_for(
                lambda: (
                    inspector.current_tool is not None
                    and inspector.current_tool.name == "review_echo"
                ),
                "tool selected",
            )
            await tab_to("#mcp-inspector-test-tool")
            await pilot.press("enter")
            await wait_for(lambda: inspector._test_preview is not None, "preview ready")

            def wire_calls():
                return sum(
                    json.loads(line)["method"] == "tools/call"
                    for line in trace.read_text().splitlines()
                )

            async def enter_message(message):
                field = await focus("#mcp-schema-field-0")
                await pilot.press("home", "shift+end", "backspace", *message)
                await settle()
                assert field.value == message
                return field

            async def run_result(payload, message):
                state.write_text(json.dumps(payload, ensure_ascii=False))
                await enter_message(message)
                before = len(service.execution_log.read_recent(50))
                before_wire = wire_calls()
                run = await tab_to("#mcp-inspector-test-run")
                await wait_for(
                    lambda run=run: not run.has_class("-active"), "run ready"
                )
                assert not run.disabled
                await pilot.press("enter")
                await wait_for(
                    lambda: (
                        len(service.execution_log.read_recent(50)) == before + 1
                        and inspector._test_preview is not None
                        and not run.disabled
                    ),
                    "invocation settled",
                )
                assert wire_calls() == before_wire + 1
                summary = inspector.query_one("#mcp-inspector-test-result", Static)
                text = str(summary.renderable)
                assert text.startswith("OK"), text
                assert client.sessions["execution-review"].process is process
                assert process.returncode is None
                audit = service.execution_log.read_recent(1)[0]
                assert audit["ok"] and audit["status"] == "success"
                return {"summary": text, "audit": audit}

            async def open_raw(stem, *, long_response=False):
                disclosure = inspector.query_one(
                    "#mcp-inspector-test-result-raw", Collapsible
                )
                assert disclosure.display
                title = disclosure.query_one("CollapsibleTitle")
                title.focus()
                title.scroll_visible(immediate=True)
                await settle()
                visible(title)
                region = title.region
                painted = " ".join(
                    app.screen._compositor.render_strips()[row]
                    .crop(region.x, region.right)
                    .text.strip()
                    for row in range(region.y, region.bottom)
                )
                assert "Raw response" in " ".join(painted.split()), painted
                if disclosure.collapsed:
                    await pilot.press("enter")
                    await settle()
                assert not disclosure.collapsed
                scroll = await tab_to("#mcp-inspector-test-result-raw-scroll")
                assert isinstance(scroll, VerticalScroll)
                body = inspector.query_one(
                    "#mcp-inspector-test-result-raw-body", Static
                )
                raw = str(body.renderable)
                assert not body._render_markup
                if long_response:
                    assert "[bold] literal [/]" in raw, raw
                    assert "中文" in json.dumps(json.loads(raw), ensure_ascii=False)
                    await pilot.press("end")
                    await settle()
                    visible(scroll)
                    assert scroll.scroll_y == scroll.max_scroll_y > 0
                    painted = " ".join(
                        app.screen._compositor.render_strips()[row]
                        .crop(scroll.region.x, scroll.region.right)
                        .text
                        for row in range(scroll.region.y, scroll.region.bottom)
                    )
                    assert "END" in painted, painted
                await capture(stem)
                return {
                    "raw_title_painted": True,
                    "keyboard_end_visible": long_response,
                }

            for theme in ("textual-dark", "textual-light"):
                for width, height in ((80, 24), (170, 48)):
                    stem = f"{theme}-{width}x{height}"
                    app.theme = theme
                    await tmux(
                        "resize-window",
                        "-t",
                        session,
                        "-x",
                        str(width),
                        "-y",
                        str(height),
                    )
                    await wait_for(
                        lambda width=width, height=height: app.size == (width, height),
                        "resize",
                    )
                    # Use the real empty response first: its interpretation and
                    # raw details must both disappear on local validation failure.
                    payload = {"content": []}
                    previous = await run_result(payload, "previous")
                    note = inspector.query_one(
                        "#mcp-inspector-test-result-note", Static
                    )
                    assert note.display and str(note.renderable)
                    raw_checks = await open_raw(stem + "-previous-raw")
                    previous_context = inspector._current_tool_profile_context
                    audit_before = len(service.execution_log.read_recent(50))
                    wire_before = wire_calls()
                    field = await enter_message("")
                    run = await tab_to("#mcp-inspector-test-run")
                    await wait_for(
                        lambda run=run: not run.has_class("-active"), "run ready"
                    )
                    await pilot.press("enter")
                    await settle()
                    summary = inspector.query_one("#mcp-inspector-test-result", Static)
                    assert str(summary.renderable).startswith("Failed"), (
                        summary.renderable
                    )
                    summary.scroll_visible(immediate=True)
                    await settle()
                    visible(summary)
                    assert not note.display and not str(note.renderable)
                    disclosure = inspector.query_one(
                        "#mcp-inspector-test-result-raw", Collapsible
                    )
                    body = inspector.query_one(
                        "#mcp-inspector-test-result-raw-body", Static
                    )
                    assert not disclosure.display and not str(body.renderable)
                    assert field.value == "" and not run.disabled
                    assert inspector._current_tool_profile_context == previous_context
                    assert len(service.execution_log.read_recent(50)) == audit_before
                    assert wire_calls() == wire_before
                    await capture(stem + "-invalid")
                    payload["content"] = [
                        {"type": "text", "text": text}
                        for text in [
                            "Corrected response",
                            "[bold] literal [/] 中文",
                            *[f"row {i}" for i in range(24)],
                            "END",
                        ]
                    ]
                    current = await run_result(payload, "corrected")
                    assert not note.display
                    current_raw_checks = await open_raw(
                        stem + "-corrected-raw", long_response=True
                    )
                    assert "Corrected response" in str(body.renderable)
                    await focus("#mcp-inspector-test-close")
                    await pilot.press("enter")
                    await settle()
                    assert not inspector.query("#mcp-inspector-test-panel")
                    await focus("#mcp-inspector-test-tool")
                    await pilot.press("enter")
                    await wait_for(
                        lambda: inspector._test_preview is not None, "reopen preview"
                    )
                    summary = inspector.query_one("#mcp-inspector-test-result", Static)
                    assert not str(summary.renderable)
                    assert not inspector.query_one(
                        "#mcp-inspector-test-result-raw", Collapsible
                    ).display
                    await focus("#mcp-schema-field-0")
                    await capture(stem + "-reopened")
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": [width, height],
                            "previous": previous,
                            "corrected": current,
                            "previous_raw_checks": raw_checks,
                            "corrected_raw_checks": current_raw_checks,
                            "invalid_clears_previous_details": True,
                            "invalid_does_not_execute": True,
                            "invalid_retains_draft_and_context": True,
                            "reopened_result_blank": True,
                            "same_connection": True,
                            "keyboard_arguments_and_run": True,
                        }
                    )
                    record()
            result["audit"] = service.execution_log.read_recent(50)
            assert len(result["audit"]) == 8
            requests = [json.loads(line) for line in trace.read_text().splitlines()]
            assert wire_calls() == 8
            result["wire_requests"] = requests
            result["passed"] = True
        except Exception:  # noqa: BLE001 - preserve failure and shut down the native app
            result.update(passed=False, error=traceback.format_exc())
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            if fixture_id is not None:
                try:
                    if (
                        fixture_id in client.sessions
                        or fixture_id in client._pending_connections
                    ):
                        assert await service.disconnect_local_profile(fixture_id)
                    assert fixture_id not in client.sessions
                    assert fixture_id not in client._pending_connections
                    if process is not None:
                        assert process.returncode is not None
                        result["fixture_returncode"] = process.returncode
                    assert await service.delete_local_profile(fixture_id)
                    assert service.local_service.store.get_profile(fixture_id) is None
                    result["fixture_session_absent"] = True
                    result["fixture_profile_absent"] = True
                except Exception:  # noqa: BLE001 - retain cleanup failure before app exit
                    result.update(passed=False, cleanup_error=traceback.format_exc())
            result["network_attempts"] = blocked_attempts()
            if result["network_attempts"]:
                result["passed"] = False
            record()
            await tmux("send-keys", "-t", session, "C-q")

    app.run(auto_pilot=journey, size=size)
    result.update(
        app_run_returned=True,
        app_return_code=app.return_code,
        app_exception=type(app._exception).__name__
        if app._exception is not None
        else None,
    )
    if app.return_code != 0 or app._exception is not None:
        result["passed"] = False
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
