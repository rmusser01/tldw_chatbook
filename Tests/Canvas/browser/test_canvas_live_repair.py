"""Committed preview failure and confirmed unsent repair in a real TldwCli child."""

import asyncio
import hashlib
import json
import sqlite3

import pytest
from playwright.async_api import async_playwright, expect

from Tests.Canvas.browser.canvas_live_harness import (
    chromium_executable,
    start_live_served_stack,
)
from Tests.Canvas.browser.test_canvas_process_restart import _wait_file
from Tests.Canvas.browser.test_canvas_served_flow import (
    _login_live_page,
    _send_console_prompt,
    _wait_for_gateway_calls,
)
from tldw_chatbook.Web_Server import serve


@pytest.mark.loopback_network
async def test_actual_child_failed_preview_repairs_only_confirmed_unsent_draft(
    tmp_path, monkeypatch, candidate_snapshot
):
    monkeypatch.setenv("TLDW_CANVAS_TEST_CANDIDATE", "1")
    monkeypatch.setenv("TLDW_CANVAS_TEST_PREVIEW_FAILURE", "1")
    monkeypatch.setattr(serve, "load_profile_snapshot", lambda: candidate_snapshot)
    token = "owned-failed-preview-test-access-token"
    stack = await start_live_served_stack(
        tmp_path,
        monkeypatch,
        access_token=token,
        child_module="Tests.Canvas.browser.canvas_live_chatbook_child",
    )
    data = tmp_path / "test_data"
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=True, executable_path=chromium_executable(playwright.chromium)
            )
            page = await browser.new_page(ignore_https_errors=True)
            await _login_live_page(page, origin=stack.origin, access_token=token)
            await expect(page.locator("#terminal")).to_contain_text(
                "Composer", timeout=45_000
            )
            await _send_console_prompt(
                page,
                "Create the requested Canvas",
                data / "canvas-live-composer-focused",
            )
            await _wait_for_gateway_calls(data / "canvas-live-gateway-calls", 2)
            await expect(page.locator("#terminal")).to_contain_text(
                "CHATBOOK_CANVAS_CREATED", timeout=45_000
            )
            shell = page.frame_locator("#served-canvas-frame")
            await expect(
                shell.frame_locator("#canvas-preview").locator("svg")
            ).to_be_visible(timeout=45_000)
            await _send_console_prompt(
                page, "Revise the active Canvas", data / "canvas-live-composer-focused"
            )
            await _wait_for_gateway_calls(data / "canvas-live-gateway-calls", 4)
            await expect(page.locator("#terminal")).to_contain_text(
                "CHATBOOK_CANVAS_UPDATED", timeout=45_000
            )
            await expect(shell.locator("#preview-state")).to_have_text(
                "Preview failed", timeout=30_000
            )
            await expect(shell.locator("#loading-state")).to_contain_text(
                "Diagram 1: cycle"
            )
            with sqlite3.connect(
                f"file:{data / 'canvas-live-chatbook.sqlite'}?mode=ro", uri=True
            ) as database:
                rows = database.execute(
                    "SELECT html, runtime_profile FROM canvas_revisions ORDER BY sequence"
                ).fetchall()
            assert len(rows) == 2 and rows[1][1] == "canvas-v2-mermaid-1"
            await shell.locator("#source-button").click()
            await expect(shell.locator("#source-view")).to_have_value(rows[1][0])
            await shell.locator("#source-close-button").click()
            await shell.locator("#notice-previous").click()
            await expect(
                shell.frame_locator("#canvas-preview").locator("svg")
            ).to_be_visible(timeout=30_000)
            await shell.locator("#follow-button").click()
            await expect(shell.locator("#preview-state")).to_have_text(
                "Preview failed", timeout=30_000
            )
            await shell.locator("#repair-button").click()
            await expect(shell.locator("#bridge-dialog")).to_be_visible()
            draft = await shell.locator("#bridge-complete-text").input_value()
            assert "cycle" in draft and "Tea" not in draft
            target = page.locator("#terminal .xterm-helper-textarea")
            receipt = data / "canvas-live-repair-receipt"

            async def composer_receipt():
                receipt.unlink(missing_ok=True)
                await target.focus()
                await target.press("F11")
                return json.loads(await _wait_file(receipt))

            before = await composer_receipt()
            assert before["draft_bytes"] == 0 and before["provider_calls"] == 4
            await shell.get_by_role("button", name="Send to composer").click()
            await expect(shell.locator("#bridge-dialog")).to_be_hidden()
            after = await composer_receipt()
            assert after == {
                "draft_sha256": hashlib.sha256(draft.encode()).hexdigest(),
                "draft_bytes": len(draft.encode()),
                "provider_calls": 4,
            }
            await asyncio.sleep(0.3)
            assert int((data / "canvas-live-gateway-calls").read_text()) == 4
            await browser.close()
    finally:
        await stack.aclose()
