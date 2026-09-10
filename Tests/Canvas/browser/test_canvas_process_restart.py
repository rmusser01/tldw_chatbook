"""Real owned parent/all-child replacement into a revoked profile policy."""

import asyncio
import json
import os
import signal
import sqlite3
import sys
from pathlib import Path

import pytest
from playwright.async_api import async_playwright, expect

from Tests.Canvas.browser.canvas_live_harness import (
    chromium_executable,
    reserve_loopback_port,
)
from Tests.Canvas.browser.test_canvas_served_flow import (
    _login_live_page,
    _send_console_prompt,
    _served_shell_projection,
)
from Tests.Packaging.test_installed_distribution import _sanitized_build_env


async def _wait_file(path):
    for _ in range(1200):
        if path.exists():
            return path.read_text()
        await asyncio.sleep(0.05)
    raise AssertionError(f"owned process acknowledgement absent: {path.name}")


def _gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


async def _wait_owned_parent_exit(process, *, timeout):
    # Every caller explicitly creates this session; never target name matches.
    assert process.returncode is not None or os.getpgid(process.pid) == process.pid
    try:
        return await asyncio.wait_for(process.wait(), timeout)
    except TimeoutError:
        if process.returncode is None:
            assert os.getpgid(process.pid) == process.pid
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
        raise  # Forced cleanup remains a failed qualification, never success.


@pytest.mark.asyncio
async def test_forced_owned_parent_teardown_reaps_its_entire_process_group():
    code = """
import json, subprocess, sys, time
children = [subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']) for _ in range(2)]
print(json.dumps([child.pid for child in children]), flush=True)
time.sleep(60)
"""
    parent = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        code,
        start_new_session=True,
        stdout=asyncio.subprocess.PIPE,
    )
    try:
        children = json.loads(await asyncio.wait_for(parent.stdout.readline(), 5))
        assert os.getpgid(parent.pid) == parent.pid
        assert all(os.getpgid(pid) == parent.pid for pid in children)
        with pytest.raises(TimeoutError):
            await _wait_owned_parent_exit(parent, timeout=0.05)
        for _ in range(100):
            if all(_gone(pid) for pid in children):
                break
            await asyncio.sleep(0.05)
        assert all(_gone(pid) for pid in [parent.pid, *children])
    finally:
        if parent.returncode is None:
            assert os.getpgid(parent.pid) == parent.pid
            os.killpg(parent.pid, signal.SIGKILL)
            await parent.wait()


@pytest.mark.loopback_network
async def test_owned_parent_and_all_children_restart_revokes_exact_durable_v2(tmp_path):
    """Old authorities cannot survive OS replacement; old durable truth survives it."""
    root = Path(__file__).resolve().parents[3]
    control = tmp_path / "control"
    control.mkdir()
    port = reserve_loopback_port()
    token = "owned-release-qualification-access-token"
    process = None
    log = None

    async def start(mode):
        nonlocal process, log
        (control / "stop").unlink(missing_ok=True)
        (control / "state.json").unlink(missing_ok=True)
        (control / "settings.json").write_text(
            json.dumps({"policy": mode, "port": port, "token": token})
        )
        env = _sanitized_build_env(tmp_path / f"environment-{mode}")
        env["TLDW_CANVAS_RELEASE_CONTROL"] = str(control)
        log = (control / f"{mode}.log").open("w")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "--show-capture=no",
            "Tests/Canvas/browser/canvas_release_parent.py",
            "--basetemp",
            str(tmp_path / f"parent-{mode}"),
            cwd=root,
            env=env,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        state = json.loads(await _wait_file(control / "state.json"))
        assert state["parent"] == process.pid
        return state

    async def stop():
        nonlocal process, log
        if process is None:
            return
        (control / "stop").touch()
        try:
            code = await _wait_owned_parent_exit(process, timeout=20)
        finally:
            log.close()
            if process.returncode is not None:
                process = None
        assert code == 0, (control / "settings.json").read_text()

    try:
        first = await start("candidate")
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=True, executable_path=chromium_executable(playwright.chromium)
            )
            contexts = [
                await browser.new_context(ignore_https_errors=True) for _ in range(2)
            ]
            pages = [await context.new_page() for context in contexts]
            for page in pages:
                await _login_live_page(page, origin=first["origin"], access_token=token)
                await expect(page.locator("#terminal")).to_contain_text(
                    "Composer", timeout=45_000
                )
            page = pages[0]
            data = Path(first["data"])
            await _send_console_prompt(
                page,
                "Create the requested Canvas",
                data / "canvas-live-composer-focused",
            )
            await expect(page.locator("#terminal")).to_contain_text(
                "CHATBOOK_CANVAS_CREATED", timeout=45_000
            )
            shell = page.frame_locator("#served-canvas-frame")
            preview = shell.frame_locator("#canvas-preview")
            await expect(preview.locator("svg")).to_be_visible(timeout=45_000)
            original = await _served_shell_projection(page)
            old_url = await page.locator("#served-canvas-frame").get_attribute("src")
            old_renderer_url = await shell.locator("#canvas-preview").get_attribute(
                "src"
            )
            with sqlite3.connect(
                f"file:{data / 'canvas-live-chatbook.sqlite'}?mode=ro", uri=True
            ) as database:
                original_rows = database.execute(
                    "SELECT id, canvas_id, runtime_profile, content_sha256, html FROM canvas_revisions"
                ).fetchall()
            assert len(original_rows) == 1
            await preview.locator("#release-submit").click()
            await expect(shell.locator("#bridge-dialog")).to_be_visible()
            pending = []

            async def capture_confirmation(route):
                pending.append(
                    (
                        route.request.url,
                        await route.request.all_headers(),
                        route.request.post_data_json,
                    )
                )
                await route.abort()

            await page.route("**/api/bridge", capture_confirmation)
            await shell.get_by_role("button", name="Send to composer").click()
            for _ in range(100):
                if pending:
                    break
                await asyncio.sleep(0.02)
            assert len(pending) == 1
            await page.unroute("**/api/bridge", capture_confirmation)
            for _ in range(100):
                before = json.loads((control / "state.json").read_text())
                if before["pending_bridges"] == 1:
                    break
                await asyncio.sleep(0.05)
            assert before["pending_bridges"] == 1
            assert len(before["children"]) == 2
            assert all(
                os.getpgid(pid) == before["parent"] for pid in before["children"]
            )
            await stop()
            assert all(_gone(pid) for pid in [before["parent"], *before["children"]])
            for page in pages:
                await page.goto("about:blank")
            second = await start("revoked")
            assert second["snapshot"] != first["snapshot"]
            fresh = await browser.new_context(ignore_https_errors=True)
            page = await fresh.new_page()
            replacement_workers = []
            page.on("worker", lambda worker: replacement_workers.append(worker.url))
            await _login_live_page(page, origin=second["origin"], access_token=token)
            await expect(page.locator("#terminal")).to_contain_text(
                "Composer", timeout=45_000
            )
            assert (await page.request.get(second["origin"] + old_url)).status == 404
            assert (
                await page.request.get(second["origin"] + old_renderer_url)
            ).status == 404
            url, headers, body = pending[0]
            assert (
                await page.request.post(
                    url,
                    headers={
                        key: value
                        for key, value in headers.items()
                        if key not in {"cookie", "content-length", "host"}
                    },
                    data=body,
                )
            ).status in {403, 404}
            target = page.locator("#terminal .xterm-helper-textarea")
            await target.focus()
            await target.press("F10")
            data = Path(second["data"])
            assert (
                await _wait_file(data / "canvas-live-saved-loaded")
                == "loaded-without-provider"
            )
            await target.press("F12")
            assert (
                await _wait_file(data / "canvas-live-card-pressed") == "selected-pinned"
            )
            shell = page.frame_locator("#served-canvas-frame")
            await expect(shell.locator("#source-view")).to_have_value(
                original_rows[0][4], timeout=30_000
            )
            assert (
                await shell.locator("#canvas-preview").get_attribute("src")
                == "about:blank"
            )
            assert replacement_workers == []
            restored = await _served_shell_projection(page)
            assert restored["selection"] == original["selection"]
            assert restored["metadata"]["content_sha256"] == original_rows[0][3]
            # Explicit new creation under the new policy; no in-place profile substitution.
            await page.context.tracing.start(screenshots=True, snapshots=True)
            http = []
            page.on(
                "response",
                lambda response: http.append(
                    {"route": response.url.split("/")[-1], "status": response.status}
                ),
            )
            await _send_console_prompt(
                page, "Create a new Canvas", data / "canvas-live-composer-focused"
            )
            await expect(page.locator("#terminal")).to_contain_text(
                "CHATBOOK_CANVAS_CREATED", timeout=45_000
            )
            try:
                await expect(
                    shell.frame_locator("#canvas-preview").locator(
                        "#chatbook-app-canvas"
                    )
                ).to_be_visible(timeout=45_000)
                await expect(shell.locator("#source-panel")).to_be_hidden()
                assert not await shell.locator("#canvas-preview").evaluate(
                    "element => Boolean(element.closest('[inert]'))"
                )
            except AssertionError:
                output = root / "output/playwright/mermaid-release"
                output.mkdir(parents=True, exist_ok=True)
                diagnostics = {
                    "http": http[-80:],
                    "state": await _served_shell_projection(page),
                    "provider_calls": (data / "canvas-live-gateway-calls").read_text(),
                    "tool_status": (data / "canvas-live-tool-status").read_text(),
                    "preview": await shell.locator("#preview-state").text_content(),
                    "loading": await shell.locator("#loading-state").text_content(),
                    "frame": await shell.locator("#canvas-preview").get_attribute(
                        "src"
                    ),
                }
                # The state projection is source-free; omit bearer capabilities.
                diagnostics["state"] = {
                    key: value
                    for key, value in diagnostics["state"].items()
                    if key in {"selection", "metadata", "execution", "availability"}
                }
                (output / "restart-failure.json").write_text(
                    json.dumps(diagnostics, indent=2)
                )
                await page.context.tracing.stop(
                    path=output / "restart-failure-trace.zip"
                )
                raise
            else:
                await page.context.tracing.stop()
            with sqlite3.connect(
                f"file:{data / 'canvas-live-chatbook.sqlite'}?mode=ro", uri=True
            ) as database:
                rows = database.execute(
                    "SELECT id, canvas_id, runtime_profile, content_sha256, html FROM canvas_revisions"
                ).fetchall()
            assert original_rows[0] in rows
            assert len(rows) == 2 and {row[2] for row in rows} == {
                "canvas-v1",
                "canvas-v2-mermaid-1",
            }
            await browser.close()
    finally:
        await stop()
        output = root / "output/playwright/mermaid-release"
        output.mkdir(parents=True, exist_ok=True)
        for mode in ("candidate", "revoked"):
            for suffix in ("events", "event-scopes"):
                path = control / f"{mode}-{suffix}.json"
                if path.exists():
                    (output / f"restart-{mode}-{suffix}.json").write_bytes(
                        path.read_bytes()
                    )
