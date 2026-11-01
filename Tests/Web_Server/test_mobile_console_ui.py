"""Real served assets with controlled API facts; not retained-child evidence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer
from playwright.async_api import async_playwright, expect

from Tests.Web_Server.test_mobile_page import (
    _PAGE_ACCESS_TOKEN,
    _PAGE_REQUEST_HEADERS,
    _app_without_control_broker,
    _page_server,
)
from Tests.Web_Server.test_mobile_page import (
    mobile_surface_gate as _mobile_surface_gate,
)
from tldw_chatbook.Canvas.web_auth import build_web_auth_policy
from tldw_chatbook.Chat.console_remote_commands import REVISION_FIELDS
from tldw_chatbook.Chat.console_remote_models import ConsoleHostIdentity, ConsoleTarget
from tldw_chatbook.DB.console_command_receipts import ConsoleCommandReceipt
from tldw_chatbook.Web_Server.mobile_service import MobileEvent

mobile_surface_gate = _mobile_surface_gate
pytestmark = [pytest.mark.loopback_network, pytest.mark.asyncio]

# Pytest isolates HOME before collection; the account home locates binaries only.
_BROWSER_ACCOUNT_HOME = Path.home()
if os.name != "nt":
    import pwd

    _BROWSER_ACCOUNT_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir)
_BROWSER_CACHE_HOME = (
    Path(os.environ.get("LOCALAPPDATA", _BROWSER_ACCOUNT_HOME / "AppData" / "Local"))
    if sys.platform == "win32"
    else _BROWSER_ACCOUNT_HOME / "Library" / "Caches"
    if sys.platform == "darwin"
    else Path(os.environ.get("XDG_CACHE_HOME", _BROWSER_ACCOUNT_HOME / ".cache"))
)
_BROWSER_CACHE = os.environ.get(
    "PLAYWRIGHT_BROWSERS_PATH", str(_BROWSER_CACHE_HOME / "ms-playwright")
)


async def _capture(case, name):
    destination = os.environ.get("TLDW_MOBILE_CAPTURE_DIR")
    if destination:
        directory = Path(destination)
        directory.mkdir(parents=True, exist_ok=True)
        await case.page.wait_for_function(
            "height => Math.abs(document.body.getBoundingClientRect().height-height) <= 1 && Math.abs((window.visualViewport?.height || innerHeight)-height) <= 1",
            arg=case.page.viewport_size["height"],
        )
        image_path = directory / f"{case.engine}-{name}.png"
        await case.page.screenshot(path=str(image_path), full_page=True)
        conditions = {
            "python": sys.version,
            "executable": sys.executable,
            "source_origins": {
                name: sys.modules[name].__file__
                for name in (
                    "tldw_chatbook",
                    "tldw_chatbook.Web_Server.mobile_service",
                    "tldw_chatbook.Chat.console_remote_commands",
                    "tldw_chatbook.DB.console_command_receipts",
                )
            },
            "engine": case.engine,
            "version": case.context.browser.version,
            "viewport": case.page.viewport_size,
            "user_agent": await case.page.evaluate("navigator.userAgent"),
            "device_pixel_ratio": await case.page.evaluate("devicePixelRatio"),
            "assets": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [
                    Path("tldw_chatbook/Web_Server/static/mobile/index.html"),
                    Path("tldw_chatbook/Web_Server/static/mobile/app.js"),
                    Path("tldw_chatbook/Web_Server/static/mobile/app.css"),
                ]
            },
            "scope": "Authenticated served assets, controlled API facts; no physical phone, native child or VPN evidence",
        }
        image_path.with_suffix(".json").write_text(json.dumps(conditions, indent=2))


class RecordingService:
    def __init__(self):
        self.commands = []
        self.tickets = []
        self.receipts = {}
        self.submitted = asyncio.Event()
        self.release = asyncio.Event()
        self.events_queue = asyncio.Queue()
        self.calls = []
        self.receipt_state = "accepted"
        self.epoch = "epoch"
        self.incarnation = "inc-a"
        self.body = '<img src=x onerror="window.injected=true">😺' * 1700
        self.decisions = []
        self.unavailable = False
        self.reset_required = False
        self.worker_run = "worker-run"
        self.attention_b = 0
        self.paired_pages = None
        self.paired_calls = []
        self.execution_admitted = True
        self.metadata_gate = None
        self.continuation_gate = None
        self.changing_history = False
        self.latest_window = None
        self.message_role = "tool"

    def row(self, sid):
        return {
            "server_epoch": self.epoch,
            "console_id": "console",
            "session_id": sid,
            "incarnation_id": self.incarnation if sid == "chat-a" else "inc-b",
            "conversation_id": "saved-" + sid,
            "title": "Research" if sid == "chat-a" else "Planning",
            "pending_decision_count": len(self.decisions)
            if sid == "chat-a"
            else self.attention_b,
            "live": True,
            "status": "streaming",
            "run_id": "run-" + sid,
            "turn_id": "turn-" + sid,
            "scope_type": "global",
            "workspace_id": "global",
            "workspace_name": None,
            "last_modified": "2026-11-01T10:00:00Z",
            "expected_revisions": {key: 0 for key in REVISION_FIELDS},
        }

    async def snapshot(self, target=None, cursor=None, receipt_cursor=None, **_):
        self.calls.append(("snapshot", target))
        if target and cursor is None and self.metadata_gate is not None:
            await self.metadata_gate.wait()
        if target and cursor is not None and self.continuation_gate is not None:
            await self.continuation_gate.wait()
        if self.unavailable:
            from tldw_chatbook.Web_Server.mobile_service import MobileOwnerUnavailable

            raise MobileOwnerUnavailable("console_unavailable")
        if target and self.paired_pages:
            message_pages, receipt_pages = self.paired_pages
            self.paired_calls.append((cursor, receipt_cursor))
            position = int(cursor.removeprefix("native-")) if cursor else 0
            receipt_position = (
                int(receipt_cursor.removeprefix("receipt-")) if receipt_cursor else 0
            )
            size = (len(self.body) + message_pages - 1) // message_pages
            start, end = position * size, min((position + 1) * size, len(self.body))
            message_more = position + 1 < message_pages
            receipt_more = receipt_position + 1 < receipt_pages
            receipt = []
            if receipt_position < receipt_pages:
                command_id = f"old-{receipt_position}"
                native_target = ConsoleTarget(
                    server_epoch=self.epoch,
                    console_id="console",
                    session_id=target.session_id,
                    incarnation_id=target.incarnation_id,
                )
                row = ConsoleCommandReceipt(
                    command_id=command_id,
                    server_epoch=self.epoch,
                    console_id="console",
                    action="send",
                    target=native_target,
                    payload_fingerprint="a" * 64,
                    state="pending",
                    issued_ns=1,
                    expires_ns=2,
                )
                self.receipts[command_id] = row
                receipt = [row.public()]
            return {
                "server_epoch": self.epoch,
                "console_id": "console",
                "sequence": 1,
                "execution_admitted": self.execution_admitted,
                "reset_required": False,
                "chats": [self.row(target.session_id)] if position == 0 else [],
                "messages": [
                    {
                        "message_id": "paired-message",
                        "role": "assistant",
                        "content": self.body[start:end],
                        "offset": start,
                        "complete": not message_more,
                    }
                ]
                if position < message_pages
                else [],
                "fleet": [],
                "decisions": [],
                "recoveries": [],
                "command_receipts": receipt,
                "next_cursor": f"native-{min(position + 1, message_pages)}"
                if message_more or receipt_more
                else None,
                "next_receipt_cursor": f"receipt-{min(receipt_position + 1, receipt_pages)}"
                if message_more or receipt_more
                else None,
            }
        page = {
            "server_epoch": self.epoch,
            "console_id": "console",
            "sequence": 1,
            "execution_admitted": self.execution_admitted,
            "reset_required": self.reset_required,
            "next_cursor": None,
            "next_receipt_cursor": None,
            "command_receipts": [
                r.public()
                for r in self.receipts.values()
                if r.state in {"pending", "outcome_unknown"}
                and (receipt_cursor is None or r.command_id > receipt_cursor)
            ],
            "chats": ([self.row(target.session_id)] if cursor is None else [])
            if target
            else [self.row("chat-a"), self.row("chat-b")],
            "messages": [],
            "fleet": [],
            "decisions": [],
            "recoveries": [],
        }
        if target:
            if self.changing_history and cursor is not None:
                page["chats"] = [self.row(target.session_id)]
                page["reset_required"] = True
                cursor = None
            start = 0 if cursor is None else int(cursor)
            end = min(start + 2000, len(self.body))
            page["messages"] = [
                {
                    "message_id": "msg-a",
                    "role": self.message_role,
                    "content": self.body[start:end],
                    "offset": start,
                    "complete": end == len(self.body),
                    "session_id": target.session_id,
                }
            ]
            page["next_cursor"] = str(end) if end < len(self.body) else None
            page["fleet"] = [
                dict(
                    {
                        key: self.row(target.session_id)[key]
                        for key in (
                            "server_epoch",
                            "console_id",
                            "session_id",
                            "incarnation_id",
                        )
                    },
                    handle_id="worker-a",
                    run_id=self.worker_run,
                    status="running",
                    live=True,
                    agent="Research worker",
                    task="Read source",
                    queued_steering=2,
                    undelivered_steering=0,
                    activity="",
                )
            ]
            if cursor is not None:
                page["fleet"] = []
            if page["next_cursor"] is not None:
                page["next_receipt_cursor"] = (
                    page["command_receipts"][-1]["command_id"]
                    if page["command_receipts"]
                    else receipt_cursor
                )
            page["decisions"] = self.decisions if cursor is None else []
            page["latest_message"] = self.latest_window
        return page

    def sessions_page(self, **_):
        return {
            "items": [
                {
                    "id": "saved-chat",
                    "title": "Saved research",
                    "scope_type": "global",
                    "workspace_id": None,
                }
            ],
            "page": 1,
            "has_more": False,
        }

    async def session_messages(
        self, conversation_id, message_offset=0, char_start=0, **_
    ):
        assert conversation_id == "saved-chat"
        text = self.body[char_start : char_start + 2000]
        return {
            "version": 1,
            "message_epoch": "saved-epoch",
            "message_total": 1,
            "message_offset": message_offset,
            "has_more": False,
            "next_message_offset": None,
            "messages": [
                {
                    "id": "saved-message",
                    "sender": "assistant",
                    "text": text,
                    "returned_chars": len(text),
                    "has_more": char_start + len(text) < len(self.body),
                }
            ],
        }

    def workers_page(self, conversation_id, *, cursor=None):
        self.calls.append(("workers", conversation_id, cursor))
        return {
            "items": [
                {
                    "id": "finished-child-" + str(i),
                    "task": "Finished task " + str(i),
                    "status": "done",
                }
                for i in (range(20) if cursor is None else range(20, 22))
            ],
            "next_cursor": "history-next" if cursor is None else None,
        }

    def worker_result(self, conversation_id, run_id, *, char_start=0, max_chars=2000):
        self.calls.append(("result", conversation_id, run_id, char_start))
        text = self.body[char_start : char_start + max_chars]
        has_more = char_start + len(text) < len(self.body)
        return {
            "run_id": run_id,
            "status": "done",
            "updated_at": "2026-11-01",
            "available": True,
            "text": text,
            "char_start": char_start,
            "total_chars": len(self.body),
            "has_more": has_more,
            "next_offset": char_start + len(text) if has_more else None,
        }

    async def bootstrap_host(self):
        self.calls.append(("bootstrap", None))
        return ConsoleHostIdentity(server_epoch=self.epoch, console_id="console")

    async def issue_command_ticket(self, action, target, payload):
        command_id = "command-" + str(len(self.tickets))
        self.tickets.append(
            {
                "action": action,
                "target": target,
                "payload": payload,
                "command_id": command_id,
            }
        )
        return {"ticket": command_id, "command_id": command_id}

    def receipt(self, ticket, state):
        row = next(row for row in self.tickets if row["command_id"] == ticket)
        receipt = ConsoleCommandReceipt(
            command_id=ticket,
            server_epoch=row["target"].server_epoch,
            console_id="console",
            action=row["action"],
            target=row["target"],
            payload_fingerprint="a" * 64,
            state=state,
            issued_ns=1,
            expires_ns=2,
            reason="Preparing" if state == "pending" else None,
            native_id="chat-b" if row["action"] in {"create", "open"} else None,
        )
        self.receipts[ticket] = receipt
        return receipt

    async def submit_command(self, ticket, payload):
        self.commands.append(
            dict(
                next(row for row in self.tickets if row["command_id"] == ticket),
                ticket=ticket,
                payload=payload,
            )
        )
        self.receipt(ticket, "pending")
        self.submitted.set()
        await self.release.wait()
        return self.receipt(ticket, self.receipt_state)

    async def query_command(self, command_id):
        self.calls.append(("query", command_id))
        return self.receipts.get(command_id)

    async def events(self, **_):
        while True:
            event = await self.events_queue.get()
            if event is None:
                return
            yield event

    async def set_answerable(self, *args):
        self.calls.append(("answerable", args))


@pytest_asyncio.fixture(params=["chromium", "webkit"])
async def mobile_browser_case(tmp_path, monkeypatch, mobile_surface_gate, request):
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", _BROWSER_CACHE)
    service = RecordingService()
    server = _page_server(tmp_path)
    monkeypatch.setattr(server, "_get_mobile_service", lambda: service)
    async with TestClient(
        TestServer(await _app_without_control_broker(server)),
        headers=dict(_PAGE_REQUEST_HEADERS),
    ) as client:
        base = str(client.make_url("/")).rstrip("/")
        server._web_auth.policy = build_web_auth_policy(
            host="127.0.0.1",
            port=client.server.port,
            access_token=_PAGE_ACCESS_TOKEN,
            public_url=base,
        )
        client.session.headers.clear()
        client.session.headers["Origin"] = base
        response = await client.post(
            "/auth/login",
            data={"access_token": _PAGE_ACCESS_TOKEN},
            allow_redirects=False,
        )
        assert response.status == 303
        async with async_playwright() as pw:
            browser = await getattr(pw, request.param).launch()
            context = await browser.new_context(
                viewport={"width": 390, "height": 844}, extra_http_headers={}
            )
            base = str(client.make_url("/")).rstrip("/")
            await context.add_cookies(
                [
                    {"name": c.key, "value": c.value, "url": base}
                    for c in client.session.cookie_jar
                ]
            )
            page = await context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.add_init_script("""window.__mobileInputTrace = [];
                for (const type of ['pointerdown', 'pointerup', 'click', 'submit']) document.addEventListener(type, event => {
                  const target = event.target;
                  window.__mobileInputTrace.push({type, target: target.id, disabled: target.disabled,
                    value: document.querySelector('#chat-draft')?.value,
                    sendDisabled: document.querySelector('#send-command')?.disabled});
                  if (window.__mobileInputTrace.length > 30) window.__mobileInputTrace.shift();
                }, true);""")
            await page.goto(base + "/m")

            async def open_chat(sid):
                await page.locator(
                    f'#chat-list button[data-session-id="{sid}"]'
                ).click()
                await expect(page.locator("#view-chat")).to_be_visible()
                await expect(page.locator("#connection-state")).to_contain_text(
                    "Connected"
                )
                await expect(page.locator("#chat-draft")).to_be_enabled()
                await expect(page.locator("#target-state")).not_to_contain_text(
                    "Original target unavailable"
                )
                await expect(page.locator("#target-state")).not_to_contain_text(
                    "Target not confirmed"
                )

            async def release_receipt(state):
                try:
                    await asyncio.wait_for(service.submitted.wait(), 5)
                except TimeoutError:
                    raise AssertionError(
                        {
                            "input_trace": await page.evaluate(
                                "window.__mobileInputTrace"
                            ),
                            "page_errors": errors,
                            "tickets": service.tickets,
                            "commands": service.commands,
                            "connection": await page.locator(
                                "#connection-state"
                            ).text_content(),
                            "target": await page.locator(
                                "#target-label"
                            ).text_content(),
                            "status": await page.locator("#status-line").text_content(),
                        }
                    ) from None
                service.receipt_state = state
                service.release.set()

            case = SimpleNamespace(
                engine=request.param,
                page=page,
                open_chat=open_chat,
                release_receipt=release_receipt,
                commands=service.commands,
                service=service,
                context=context,
                pw=pw,
                base=base,
                server=server,
            )
            yield case
            service.release.set()
            for gate in (service.metadata_gate, service.continuation_gate):
                if gate is not None:
                    gate.set()
            await context.close()
            await browser.close()


async def test_late_receipt_cannot_clear_a_newer_phone_draft(mobile_browser_case):
    case = mobile_browser_case
    await case.open_chat("chat-a")
    await case.page.locator("#chat-draft").fill("first correction")
    await case.page.locator("#send-command").click()
    await asyncio.wait_for(case.service.submitted.wait(), 5)
    await case.page.locator("#chat-draft").fill("newer unsent correction")
    await case.release_receipt("accepted")
    await expect(case.page.locator("#command-status")).to_contain_text("accepted")
    await expect(case.page.locator("#chat-draft")).to_have_value(
        "newer unsent correction"
    )
    assert case.commands[0]["payload"]["text"] == "first correction"


async def test_chat_and_worker_drafts_and_exact_typed_commands(mobile_browser_case):
    case = mobile_browser_case
    page = case.page
    await case.open_chat("chat-a")
    await page.locator("#chat-draft").fill("draft A")
    await page.locator("#back-button").click()
    await case.open_chat("chat-b")
    await expect(page.locator("#chat-draft")).to_have_value("")
    await page.locator("#chat-draft").fill("draft B")
    await page.locator("#agents-button").click()
    await page.get_by_role("button", name="Worker worker-a").click()
    await expect(page.locator("#target-label")).to_contain_text(
        "Planning · Worker worker-a · run worker-run"
    )
    await expect(page.locator("#chat-draft")).to_have_value("")
    await page.locator("#chat-draft").fill("worker guidance")
    await page.locator("#send-command").click()
    await case.release_receipt("queued")
    await expect(page.locator("#command-status")).to_contain_text(
        "consumption not confirmed"
    )
    await _capture(case, "worker-live-390")
    command = case.commands[0]
    assert command["action"] == "subagent_guidance"
    assert command["target"].session_id == "chat-b"
    assert command["target"].run_id == "worker-run"
    assert command["target"].handle_id == "worker-a"
    await expect(page.locator("#chat-draft")).to_have_value("")
    case.service.receipt_state = "accepted"
    await page.locator("#stop-command").click()
    await expect(page.locator("#command-status")).to_contain_text(
        "physical exit not confirmed"
    )
    assert case.commands[-1]["action"] == "subagent_stop"
    case.service.receipt_state = "queued"
    await page.get_by_role("button", name="Main agent").click()
    await expect(page.locator("#chat-draft")).to_have_value("draft B")
    await page.locator("#steer-command").click()
    await expect(page.locator("#command-status")).to_contain_text("steer: queued")
    assert case.commands[-1]["action"] == "steer"
    assert case.commands[-1]["target"].run_id == "run-chat-b"
    await page.locator("#chat-draft").fill("redirect B")
    await page.locator("#redirect-command").click()
    await expect(page.locator("#command-status")).to_contain_text("redirect: queued")
    assert case.commands[-1]["action"] == "redirect"
    assert case.commands[-1]["payload"] == {"text": "redirect B"}
    await page.locator("#back-button").click()  # close Agents
    await page.locator("#back-button").click()  # Chats
    await case.open_chat("chat-a")
    await expect(page.locator("#chat-draft")).to_have_value("draft A")
    assert not any(call[0] == "bootstrap" for call in case.service.calls)


async def test_preparing_duplicate_submit_and_lost_ack_query_without_replay(
    mobile_browser_case,
):
    case = mobile_browser_case
    page = case.page
    await case.open_chat("chat-a")
    await page.locator("#chat-draft").fill("immutable send")

    # Deliver to the real recording API, then lose only its response.
    async def lose_response(route):
        await route.fetch()
        await route.abort("connectionreset")

    await page.route("**/api/v1/mobile/commands", lose_response)
    await page.locator("#send-command").dblclick()
    await asyncio.wait_for(case.service.submitted.wait(), 5)
    assert len(case.commands) == 1
    await case.release_receipt("pending")
    await expect(page.locator("#command-status")).to_contain_text(
        "Preparing · not accepted yet"
    )
    await expect(page.locator("#chat-draft")).to_have_value("immutable send")
    await expect(page.locator("#send-command")).to_be_disabled()
    assert any(call[0] == "query" for call in case.service.calls)
    await case.service.events_queue.put(None)
    await expect(page.locator("#connection-state")).to_contain_text("Disconnected")
    await expect(page.locator("#stop-command")).to_be_disabled()
    await page.locator("#chat-draft").fill("new offline draft")
    case.service.receipt("command-0", "accepted")
    await expect(page.locator("#connection-state")).to_contain_text("Connected")
    await expect(page.locator("#chat-draft")).to_have_value("new offline draft")
    assert len(case.commands) == len(case.service.tickets) == 1
    assert not any(call[0] == "bootstrap" for call in case.service.calls)


async def test_replaced_incarnation_keeps_original_draft_until_explicit_choice(
    mobile_browser_case,
):
    case = mobile_browser_case
    await case.open_chat("chat-a")
    await case.page.locator("#chat-draft").fill("old incarnation draft")
    case.service.incarnation = "replacement"
    await case.service.events_queue.put(MobileEvent("console.changed", "", {}, 2))
    await expect(case.page.locator("#target-state")).to_contain_text(
        "Original target unavailable"
    )
    await expect(case.page.locator("#chat-draft")).to_have_value(
        "old incarnation draft"
    )
    await expect(case.page.locator("#send-command")).to_be_disabled()
    await case.page.locator("#back-button").click()
    await case.open_chat("chat-a")
    await expect(case.page.locator("#chat-draft")).to_have_value("")
    assert not case.commands


async def test_long_unicode_tool_html_and_supported_native_decisions(
    mobile_browser_case,
):
    case = mobile_browser_case
    base = case.service.row("chat-a")
    case.service.decisions = [
        dict(
            base,
            kind="approval",
            round_id="approval-a",
            supported=True,
            calls=[
                {
                    "call_id": "call-a",
                    "llm_name": "fs_read",
                    "options": ["approve_once", "deny"],
                    "arguments_summary": "/notes/readme",
                }
            ],
        ),
        dict(
            base,
            kind="question",
            round_id="question-a",
            supported=True,
            questions=[
                {
                    "question": "Which branch?",
                    "options": [
                        {"label": "Main", "description": "Current branch"},
                        {"label": "Other", "description": "Alternate branch"},
                    ],
                    "multiSelect": False,
                }
            ],
        ),
        dict(
            base,
            kind="shell",
            round_id="unsupported-a",
            supported=False,
            reason="This review requires the terminal.",
        ),
    ]
    await case.open_chat("chat-a")
    page = case.page
    await page.locator("#chat-transcript summary").click()
    assert await page.locator(".tool-content").text_content() == case.service.body
    assert await page.locator("#chat-transcript img").count() == 0
    assert await page.evaluate("window.injected || false") is False
    assert await page.locator("#chat-decisions .decision").count() == 3
    await expect(
        page.locator("#chat-decisions").get_by_role("link", name="Open terminal")
    ).to_be_visible()
    await expect(page.locator("#approvals-badge")).to_have_text("3")
    for control in (
        page.get_by_label("Decision for fs_read"),
        page.locator("#chat-decisions").get_by_role("link", name="Open terminal"),
    ):
        box = await control.bounding_box()
        assert box and box["height"] >= 44
    await _capture(case, "decisions-390")
    await page.get_by_label("Decision for fs_read").select_option("deny")
    await (
        page.locator(".decision")
        .nth(0)
        .get_by_role("button", name="Submit decision")
        .click()
    )
    await case.release_receipt("accepted")
    await expect(page.locator("#command-status")).to_contain_text("decision: accepted")
    assert case.commands[0]["payload"] == {
        "kind": "approval",
        "answer": {"decisions": {"call-a": "deny"}},
    }
    assert case.commands[0]["target"].round_id == "approval-a"
    await page.locator(".decision").nth(1).get_by_label("Main · Current branch").check()
    await (
        page.locator(".decision")
        .nth(1)
        .get_by_role("button", name="Submit decision")
        .click()
    )
    await expect(page.locator("#command-status")).to_contain_text("decision: accepted")
    await page.wait_for_function(
        "document.querySelector('#command-status').textContent.split('decision: accepted').length === 3"
    )
    assert case.commands[1]["payload"]["answer"]["answers"] == [
        {
            "question": "Which branch?",
            "selected": ["Main"],
            "other_text": None,
            "unanswered": False,
        }
    ]
    assert all(
        command["target"].round_id != "unsupported-a" for command in case.commands
    )


async def test_phone_geometry_keyboard_focus_and_long_content(mobile_browser_case):
    case = mobile_browser_case
    page = case.page
    await expect(page.locator("#chat-list button")).to_have_count(2)
    await expect(page.locator("#connection-state")).to_contain_text("Connected")
    await _capture(case, "chats-390")
    await case.open_chat("chat-a")
    await page.locator("#chat-transcript summary").click()
    await page.locator("#chat-draft").fill("A correction for the named agent")
    for width, height in [(320, 568), (390, 844), (844, 390), (390, 360)]:
        await page.set_viewport_size({"width": width, "height": height})
        await expect(page.locator("#send-command")).to_be_enabled()
        await page.locator("#chat-draft").focus()
        # macOS WebKit follows Safari's Option-Tab all-controls convention.
        await page.keyboard.press("Alt+Tab" if case.engine == "webkit" else "Tab")
        await expect(page.locator("#send-command")).to_be_focused()
        assert (
            await page.locator("#send-command").evaluate(
                "(el) => getComputedStyle(el).outlineStyle"
            )
            == "solid"
        )
        for selector in [
            "#send-command",
            "#stop-command",
            "#chat-draft",
            "#back-button",
        ]:
            box = await page.locator(selector).bounding_box()
            assert box and box["x"] >= 0 and box["y"] >= 0, (selector, box)
            assert box["x"] + box["width"] <= width + 1, (selector, box)
            assert box["y"] + box["height"] <= height + 1, (selector, box)
            if selector != "#chat-draft":
                assert box["height"] >= 44, (selector, box)
        assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert await page.locator("#chat-transcript").evaluate(
            "(el) => el.clientHeight > 0"
        )
        assert await page.locator(".tool-content").evaluate(
            "(el) => el.scrollWidth <= el.clientWidth + 1"
        )
        await _capture(case, f"chat-{width}x{height}")
    await page.locator("#back-button").click()
    await page.set_viewport_size({"width": 1280, "height": 900})
    await expect(page.locator("#chat-list button")).to_have_count(2)
    await _capture(case, "desktop-1280")
    assert not case.commands


async def test_reset_pages_and_worker_run_replacement_do_not_retarget_draft(
    mobile_browser_case,
):
    case = mobile_browser_case
    case.service.reset_required = True
    await case.open_chat("chat-a")
    await case.page.locator("#chat-transcript summary").click()
    assert await case.page.locator(".tool-content").text_content() == case.service.body
    await case.page.locator("#agents-button").click()
    await case.page.get_by_role("button", name="Worker worker-a").click()
    await expect(case.page.locator("#target-label")).to_contain_text("worker-run")
    await case.page.locator("#chat-draft").fill("original worker draft")
    await expect(case.page.locator("#send-command")).to_be_enabled()
    case.service.worker_run = "successor-run"
    await case.service.events_queue.put(
        MobileEvent("reset", "", {"reset_required": True}, 2)
    )
    await expect(case.page.locator("#target-state")).to_contain_text(
        "Original target unavailable"
    )
    await expect(case.page.locator("#chat-draft")).to_have_value(
        "original worker draft"
    )
    await expect(case.page.locator("#send-command")).to_be_disabled()
    await expect(
        case.page.get_by_role("button", name="Worker worker-a")
    ).to_have_attribute("data-run-id", "successor-run")
    await case.page.get_by_role("button", name="Worker worker-a").click()
    await expect(case.page.locator("#target-label")).to_contain_text("successor-run")
    await expect(case.page.locator("#chat-draft")).to_have_value("")
    assert not case.commands


async def test_unavailable_saved_history_is_complete_without_starting_host(
    mobile_browser_case,
):
    case = mobile_browser_case
    case.service.unavailable = True
    # New page has no old live rows; canonical saved reading stays available.
    await case.page.reload()
    await case.page.locator('#chat-list button[data-session-id="saved-chat"]').click()
    await expect(case.page.locator("#chat-transcript .message-body")).to_have_text(
        case.service.body
    )
    await expect(case.page.locator("#send-command")).to_be_disabled()
    await expect(
        case.page.get_by_role("button", name="Open saved chat")
    ).to_be_visible()
    assert not case.commands
    assert not any(call[0] == "bootstrap" for call in case.service.calls)


async def test_expired_authentication_returns_to_existing_sign_in(mobile_browser_case):
    case = mobile_browser_case
    await case.open_chat("chat-a")
    await case.page.route(
        "**/api/v1/mobile/console/snapshot?*",
        lambda route: route.fulfill(status=401, json={"error": "auth_required"}),
    )
    async with case.page.expect_request("**/auth/login"):
        await case.service.events_queue.put(MobileEvent("console.changed", "", {}, 2))
    assert not case.commands


async def test_attention_and_finished_worker_reading_are_explicit_read_only(
    mobile_browser_case,
):
    case = mobile_browser_case
    case.service.attention_b = 1
    await case.service.events_queue.put(MobileEvent("console.changed", "", {}, 2))
    await expect(
        case.page.locator('#chat-list button[data-session-id="chat-b"]')
    ).to_contain_text("Needs a decision · 1")
    await case.page.locator("#approvals-button").click()
    await case.page.get_by_role("button", name="Planning · 1 pending decisions").click()
    await expect(case.page.locator("#chat-title")).to_have_text("Planning")
    await case.page.locator("#agents-button").click()
    await expect(case.page.locator("#worker-history li")).to_have_count(20)
    assert not any(call[0] == "result" for call in case.service.calls)
    await case.page.locator("#workers-more").click()
    await expect(case.page.locator("#worker-history li")).to_have_count(22)
    await case.page.get_by_role(
        "button", name="Finished task 0 · recorded done", exact=True
    ).click()
    await expect(case.page.locator("#target-label")).to_contain_text(
        "Worker finished-child-0 · History (read only)"
    )
    await expect(case.page.locator("#send-command")).to_be_disabled()
    await expect(case.page.locator("#worker-result")).to_have_text(
        case.service.body[:2000]
    )
    await case.page.locator("#worker-result-more").click()
    await expect(case.page.locator("#worker-result")).to_have_text(
        case.service.body[:4000]
    )
    assert await case.page.locator("#worker-result img").count() == 0
    await _capture(case, "worker-history-390")
    assert not case.commands
    assert not any(call[0] == "bootstrap" for call in case.service.calls)


async def test_create_bootstraps_only_after_explicit_click_with_native_workspace_payload(
    mobile_browser_case,
):
    case = mobile_browser_case
    assert not any(call[0] == "bootstrap" for call in case.service.calls)
    case.service.receipt_state = "applied"
    case.service.release.set()
    await case.page.locator("#new-chat-button").click()
    await expect(case.page.locator("#chat-title")).to_have_text("Planning")
    assert [call[0] for call in case.service.calls].count("bootstrap") == 1
    assert len(case.commands) == len(case.service.tickets) == 1
    command = case.commands[0]
    assert command["action"] == "create"
    assert command["payload"] == {"workspace_id": "global"}
    assert command["target"].model_dump() == {
        "server_epoch": "epoch",
        "console_id": "console",
    }


async def test_fresh_snapshot_reconciles_unresolved_receipts_and_unknown_disables_mutation(
    mobile_browser_case,
):
    case = mobile_browser_case
    target = ConsoleTarget(
        server_epoch="epoch",
        console_id="console",
        session_id="chat-a",
        incarnation_id="inc-a",
    )
    ticket = await case.service.issue_command_ticket(
        "send", target, {"text": "not retained in receipt"}
    )
    case.service.receipt(ticket["command_id"], "pending")
    await case.open_chat("chat-a")
    await expect(case.page.locator("#command-status")).to_contain_text(
        "Preparing · not accepted yet"
    )
    await case.page.locator("#chat-draft").fill("new unsent draft")
    case.service.receipt(ticket["command_id"], "outcome_unknown")
    await case.service.events_queue.put(MobileEvent("console.changed", "", {}, 2))
    await expect(case.page.locator("#connection-state")).to_contain_text(
        "unresolved command outcome"
    )
    await expect(case.page.locator("#send-command")).to_be_disabled()
    await expect(case.page.locator("#stop-command")).to_be_disabled()
    await expect(case.page.locator("#chat-draft")).to_have_value("new unsent draft")
    assert not case.commands
    assert len(case.service.tickets) == 1


@pytest.mark.parametrize("pages", [(3, 2), (2, 3)])
async def test_unequal_snapshot_sections_follow_exact_pairs_without_repeated_body(
    mobile_browser_case, pages
):
    case = mobile_browser_case
    case.service.body = "UTF-32 offset 😺 <script>inert</script> " * 170
    case.service.paired_pages = pages
    await case.open_chat("chat-a")
    await expect(case.page.locator("#chat-transcript .message-body")).to_have_text(
        case.service.body
    )
    await expect(case.page.locator("#connection-state")).to_contain_text("Connected")
    await expect(case.page.locator("#connection-state")).not_to_contain_text(
        "loading history"
    )
    assert len(case.service.paired_calls) == max(pages)
    assert case.service.paired_calls == [
        (None, None),
        ("native-1", "receipt-1"),
        ("native-2", "receipt-2"),
    ]
    assert not case.commands
    assert not any(call[0] == "bootstrap" for call in case.service.calls)


async def test_fresh_metadata_enables_exact_stop_during_changing_history_and_reset_reconfirms(
    mobile_browser_case,
):
    case = mobile_browser_case
    page = case.page
    case.service.changing_history = True
    case.service.continuation_gate = asyncio.Event()
    # First page is admitted metadata; the bulk continuation cannot finish yet.
    await case.open_chat("chat-a")
    await expect(page.locator("#connection-state")).to_contain_text("loading history")
    await expect(page.locator("#stop-command")).to_be_enabled()
    case.service.receipt_state = "accepted"
    case.service.release.set()
    await page.locator("#stop-command").click()
    await asyncio.wait_for(case.service.submitted.wait(), 5)
    assert len(case.commands) == 1
    assert case.commands[0]["action"] == "stop"
    assert case.commands[0]["target"].model_dump(exclude_none=True) == {
        "server_epoch": "epoch",
        "console_id": "console",
        "session_id": "chat-a",
        "incarnation_id": "inc-a",
        "run_id": "run-chat-a",
        "turn_id": "turn-chat-a",
    }
    case.service.continuation_gate.set()
    await expect(page.locator("#command-status")).to_contain_text(
        "physical exit not confirmed"
    )
    case.service.metadata_gate = asyncio.Event()
    await case.service.events_queue.put(
        MobileEvent("reset", "", {"reset_required": True}, 2)
    )
    await expect(page.locator("#stop-command")).to_be_disabled()
    await page.locator("#chat-draft").fill("kept while target confirmation is delayed")
    # Replacement metadata cannot silently inherit the old incarnation draft.
    case.service.incarnation = "replacement"
    case.service.metadata_gate.set()
    await expect(page.locator("#target-state")).to_contain_text(
        "Original target unavailable"
    )
    await expect(page.locator("#chat-draft")).to_have_value(
        "kept while target confirmation is delayed"
    )
    await expect(page.locator("#stop-command")).to_be_disabled()
    assert len(case.commands) == 1


async def test_unadmitted_native_metadata_cannot_enable_controls(mobile_browser_case):
    case = mobile_browser_case
    case.service.execution_admitted = False
    await case.page.locator('#chat-list button[data-session-id="chat-a"]').click()
    await case.page.locator("#chat-draft").fill("held draft")
    await expect(case.page.locator("#target-state")).to_contain_text(
        "Target not confirmed"
    )
    await expect(case.page.locator("#send-command")).to_be_disabled()
    await expect(case.page.locator("#stop-command")).to_be_disabled()
    assert not case.commands


async def test_late_ack_after_navigation_clears_only_original_target(
    mobile_browser_case,
):
    case = mobile_browser_case
    await case.open_chat("chat-a")
    await case.page.locator("#chat-draft").fill("submitted A")
    await case.page.locator("#send-command").click()
    await asyncio.wait_for(case.service.submitted.wait(), 5)
    await case.page.locator("#back-button").click()
    await case.open_chat("chat-b")
    await case.page.locator("#chat-draft").fill("unsent B")
    async with case.page.expect_response("**/api/v1/mobile/commands"):
        await case.release_receipt("accepted")
    await expect(case.page.locator("#chat-draft")).to_have_value("unsent B")
    await case.page.locator("#back-button").click()
    await case.open_chat("chat-a")
    await expect(case.page.locator("#chat-draft")).to_have_value("")
    assert case.commands[0]["target"].session_id == "chat-a"
    assert case.commands[0]["payload"]["text"] == "submitted A"


async def test_latest_excerpt_is_inert_scoped_and_removed_only_after_exact_full_body(
    mobile_browser_case,
):
    case = mobile_browser_case
    case.service.body = (
        '<img src=x onerror="window.injected=true">😺' * 2500 + " NEWEST REPLY END"
    )
    case.service.message_role = "assistant"
    case.service.continuation_gate = asyncio.Event()
    row = case.service.row("chat-a")
    case.service.latest_window = {
        key: row[key]
        for key in ("server_epoch", "console_id", "session_id", "incarnation_id")
    }
    case.service.latest_window.update(
        message_id="msg-a",
        role="assistant",
        status="streaming",
        offset=len(case.service.body) - 300,
        total_chars=len(case.service.body),
        content=case.service.body[-300:],
    )
    await case.open_chat("chat-a")
    await expect(case.page.locator("#latest-output")).to_be_visible()
    await expect(case.page.locator("#latest-output-label")).to_contain_text("excerpt")
    await expect(case.page.locator("#latest-output-content")).to_have_text(
        case.service.body[-300:]
    )
    assert not await case.page.locator("#latest-output img").count()
    await expect(case.page.locator("#stop-command")).to_be_enabled()
    region = await case.page.locator("#latest-output").bounding_box()
    label = await case.page.locator("#latest-output-label").bounding_box()
    assert (
        region
        and label
        and label["y"] >= region["y"]
        and label["y"] + label["height"] <= region["y"] + region["height"]
    )
    await _capture(case, "latest-output-390")
    case.service.continuation_gate.set()
    await expect(case.page.locator("#chat-transcript .message-body")).to_have_text(
        case.service.body
    )
    await expect(case.page.locator("#latest-output")).to_be_hidden()
    await case.page.locator("#back-button").click()
    await case.open_chat("chat-b")
    # An old chat's independently delivered excerpt cannot follow local selection.
    await expect(case.page.locator("#latest-output")).to_be_hidden()
    assert not case.commands


async def test_fix1_lost_ack_reconciles_before_endlessly_resetting_history(
    mobile_browser_case,
):
    case = mobile_browser_case
    page = case.page
    await case.open_chat("chat-a")
    failed_query = asyncio.Event()
    query_ids = []
    snapshots = []
    original_snapshot = case.service.snapshot

    async def witnessed_snapshot(**coordinates):
        result = await original_snapshot(**coordinates)
        snapshots.append((coordinates, result))
        return result

    case.service.snapshot = witnessed_snapshot

    async def lose_response(route):
        await route.fetch()
        await route.abort("connectionreset")

    async def query_once_fails(route):
        query_ids.append(route.request.url.rsplit("/", 1)[-1])
        if len(query_ids) == 1:
            await route.fulfill(
                status=503, json={"error": "query_temporarily_unavailable"}
            )
            failed_query.set()
        else:
            await route.continue_()

    await page.route("**/api/v1/mobile/commands", lose_response)
    await page.route("**/api/v1/mobile/commands/command-0", query_once_fails)
    await page.locator("#chat-draft").fill("accepted but acknowledgement lost")
    await page.locator("#send-command").click()
    await case.release_receipt("accepted")
    await asyncio.wait_for(failed_query.wait(), 5)
    await expect(page.locator("#stop-command")).to_be_disabled()
    assert case.service.receipts["command-0"].state == "accepted"
    await page.unroute("**/api/v1/mobile/commands", lose_response)
    case.service.changing_history = True
    await case.service.events_queue.put(None)
    await expect(page.locator("#connection-state")).to_contain_text("loading history")
    await expect(page.locator("#stop-command")).to_be_enabled()
    await expect(page.locator("#chat-draft")).to_have_value("")
    assert query_ids == ["command-0", "command-0"]
    assert any(coords.get("cursor") for coords, _ in snapshots)
    assert all(not row["command_receipts"] for _, row in snapshots)
    await page.locator("#stop-command").click()
    await expect(page.locator("#command-status")).to_contain_text("Stop requested")
    assert [row["action"] for row in case.commands] == ["send", "stop"]
    assert len(case.service.tickets) == 2
    assert case.commands[1]["target"].model_dump(exclude_none=True) == {
        "server_epoch": "epoch",
        "console_id": "console",
        "session_id": "chat-a",
        "incarnation_id": "inc-a",
        "run_id": "run-chat-a",
        "turn_id": "turn-chat-a",
    }
    assert not any(call[0] == "bootstrap" for call in case.service.calls)


async def test_fix1_cached_live_chat_reads_saved_history_after_owner_loss(
    mobile_browser_case,
):
    case = mobile_browser_case
    page = case.page
    await case.open_chat("chat-a")
    await page.locator("#chat-draft").fill("original live draft kept")
    reads = []
    original_messages = case.service.session_messages

    async def saved_messages(conversation_id, **coordinates):
        reads.append(conversation_id)
        assert conversation_id == "saved-chat-a"
        return await original_messages("saved-chat", **coordinates)

    case.service.session_messages = saved_messages
    case.service.unavailable = True
    await case.service.events_queue.put(MobileEvent("console.changed", "", {}, 2))
    await expect(page.locator("#connection-state")).to_contain_text("State uncertain")
    await page.locator("#back-button").click()
    await expect(
        page.locator('#chat-list button[data-session-id="chat-a"]')
    ).to_contain_text("Unavailable")
    await page.locator('#chat-list button[data-session-id="chat-a"]').click()
    await expect(page.locator("#chat-transcript .message-body")).to_have_text(
        case.service.body
    )
    await expect(page.locator("#send-command")).to_be_disabled()
    await expect(page.get_by_role("button", name="Open saved chat")).to_be_visible()
    assert reads and set(reads) == {"saved-chat-a"}
    case.service.unavailable = False
    await page.locator("#back-button").click()
    await case.open_chat("chat-a")
    await expect(page.locator("#chat-draft")).to_have_value("original live draft kept")
    assert not case.commands
    assert not any(call[0] == "bootstrap" for call in case.service.calls)


@pytest.mark.bootstrap_profile
async def test_fix1_canonical_sqlite_history_follows_filtered_message_pages(
    mobile_browser_case,
    tmp_path,
):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Web_Server.mobile_service import CanonicalMobileReads

    case = mobile_browser_case
    path = tmp_path / "filtered-history.sqlite"
    core = CharactersRAGDB(path, "mobile-filtered-history")
    conversation_id = core.add_conversation({"title": "Filtered canonical history"})
    user_body = "visible user 😺" * 210
    assistant_body = (
        '<img src=x onerror="window.filteredInjected=true">visible assistant 🦊'
    )
    for sender, content in [
        ("system", "HIDDEN_SYSTEM_BODY"),
        ("user", user_body),
        ("other", "HIDDEN_OTHER_BODY"),
        ("assistant", assistant_body),
    ]:
        core.add_message(
            {"conversation_id": conversation_id, "sender": sender, "content": content}
        )
    core.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    canonical = CanonicalMobileReads(
        path, tmp_path / "absent-media", tmp_path / "absent-runs"
    )
    coordinates_seen = []
    responses = []

    async def messages(identity, **coordinates):
        coordinates_seen.append(
            (coordinates["message_offset"], coordinates["char_start"])
        )
        result = await asyncio.to_thread(
            canonical.session_messages, identity, **coordinates
        )
        responses.append(result)
        return result

    case.service.sessions_page = canonical.sessions_page
    case.service.session_messages = messages
    case.service.unavailable = True
    try:
        await case.page.reload()
        await case.page.locator(
            f'#chat-list button[data-session-id="{conversation_id}"]'
        ).click()
        await expect(case.page.locator("#chat-transcript .message-body")).to_have_text(
            [user_body, assistant_body]
        )
        assert coordinates_seen == [(0, 0), (1, 0), (1, 2000), (2, 0), (3, 0)]
        assert responses[0]["messages"] == responses[3]["messages"] == []
        assert responses[0]["next_message_offset"] == 1
        assert responses[3]["next_message_offset"] == 3
        assert all("HIDDEN_" not in json.dumps(row) for row in responses)
        assert await case.page.evaluate("window.filteredInjected") is None
        assert not await case.page.locator("#chat-transcript img").count()
        await expect(case.page.locator("#send-command")).to_be_disabled()
        assert not case.commands
        assert not any(call[0] == "bootstrap" for call in case.service.calls)
        assert not (tmp_path / "absent-runs").exists()
        assert not (tmp_path / "absent-media").exists()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    finally:
        canonical.close()


async def test_fix1_saved_chat_discovery_loads_an_explicit_second_page(
    mobile_browser_case,
):
    case = mobile_browser_case
    # Finish the fixture's initial live discovery before replacing its reader.
    await expect(
        case.page.locator('#chat-list button[data-session-id="chat-a"]')
    ).to_be_visible()
    pages = []

    def saved_page(*, page=1, per_page=50):
        pages.append((page, per_page))
        assert page in {1, 2}
        items = (
            [
                {
                    "id": "recent-" + str(i),
                    "title": "Recent " + str(i),
                    "scope_type": "global",
                    "workspace_id": None,
                }
                for i in range(50)
            ]
            if page == 1
            else [
                {
                    "id": "saved-chat",
                    "title": "Older saved chat",
                    "scope_type": "global",
                    "workspace_id": None,
                }
            ]
        )
        return {"items": items, "page": page, "has_more": page == 1}

    case.service.sessions_page = saved_page
    case.service.unavailable = True
    await case.page.reload()
    await expect(case.page.locator("#chat-list button")).to_have_count(50)
    assert pages == [(1, 50)]
    more = case.page.get_by_role("button", name="Load more saved chats")
    await more.scroll_into_view_if_needed()
    await more.focus()
    await expect(more).to_be_focused()
    bounds = await more.bounding_box()
    assert bounds["height"] >= 44
    assert bounds["x"] >= 0 and bounds["x"] + bounds["width"] <= 390
    await more.press("Enter")
    await expect(case.page.locator("#chat-list button")).to_have_count(51)
    assert pages == [(1, 50), (2, 50)]
    await expect(
        case.page.get_by_role("button", name="Load more saved chats")
    ).to_be_hidden()
    await case.page.locator('#chat-list button[data-session-id="saved-chat"]').click()
    await expect(case.page.locator("#chat-transcript .message-body")).to_have_text(
        case.service.body
    )
    await expect(case.page.locator("#send-command")).to_be_disabled()
    assert not case.commands
    assert not any(call[0] == "bootstrap" for call in case.service.calls)


@pytest.mark.parametrize("fault", ["nonadvancing_offset", "changed_epoch"])
async def test_fix1_saved_history_refuses_invalid_or_stale_filtered_continuation(
    mobile_browser_case,
    fault,
):
    case = mobile_browser_case
    await expect(
        case.page.locator('#chat-list button[data-session-id="chat-a"]')
    ).to_be_visible()
    offsets = []

    async def messages(_identity, *, message_offset=0, **_):
        offsets.append(message_offset)
        return {
            "version": 1,
            "message_epoch": "changed" if message_offset else "initial",
            "message_total": 2,
            "message_offset": message_offset,
            "has_more": message_offset == 0,
            "next_message_offset": 0 if fault == "nonadvancing_offset" else 1,
            "messages": []
            if message_offset == 0
            else [
                {
                    "id": "visible-message",
                    "sender": "assistant",
                    "text": "stale visible body",
                    "returned_chars": 18,
                    "has_more": False,
                }
            ],
        }

    case.service.session_messages = messages
    case.service.unavailable = True
    await case.page.reload()
    await case.page.locator('#chat-list button[data-session-id="saved-chat"]').click()
    await expect(case.page.locator("#status-line")).to_contain_text(
        "invalid_history_page" if fault == "nonadvancing_offset" else "history_changed"
    )
    assert offsets == ([0] if fault == "nonadvancing_offset" else [0, 1])
    await expect(case.page.locator("#chat-transcript .message-body")).to_have_count(0)
    await expect(case.page.locator("#send-command")).to_be_disabled()
    assert not case.commands
    assert not any(call[0] == "bootstrap" for call in case.service.calls)


@pytest.mark.parametrize(
    "outcome",
    [
        "delayed",
        "lost-response",
        "older-owner",
        "bootstrap-owner",
        "navigation",
        "replacement-owner",
        "replacement-console",
        "newer-action",
    ],
)
async def test_host_selection_waits_for_exact_original_receipt_and_current_row(
    mobile_browser_case, monkeypatch, outcome
):
    from dataclasses import replace

    case = mobile_browser_case
    original_snapshot = case.service.snapshot
    original_receipt = case.service.receipt

    def receipt(ticket, state):
        row = original_receipt(ticket, state)
        if outcome == "newer-action" and ticket == "command-1":
            row = replace(row, native_id="chat-c")
            case.service.receipts[ticket] = row
        return row

    monkeypatch.setattr(case.service, "receipt", receipt)
    visible = False
    replacement = False
    restored_owner = False
    unrelated_delivered = asyncio.Event()

    async def observe_snapshot(response):
        if (
            "/api/v1/mobile/console/snapshot?" in response.url
            and response.status == 200
        ):
            body = await response.json()
            if any(
                row["command_id"] == "unrelated-command"
                for row in body["command_receipts"]
            ):
                unrelated_delivered.set()

    case.page.on("response", observe_snapshot)

    async def snapshot(*args, **kwargs):
        data = await original_snapshot(*args, **kwargs)
        if not visible:
            data["chats"] = [
                row for row in data["chats"] if row["session_id"] != "chat-b"
            ]
        if visible and outcome == "newer-action":
            if not args and not kwargs.get("target"):
                data["chats"].append(case.service.row("chat-c"))
            for row in data["chats"]:
                if row["session_id"] == "chat-c":
                    row["title"] = "Newer explicit chat"
        if replacement:
            field = "console_id" if outcome == "replacement-console" else "server_epoch"
            data[field] = "replacement-owner"
            for row in data["chats"]:
                row[field] = "replacement-owner"
        unrelated = case.service.receipts.get("unrelated-command")
        if unrelated is not None:
            data["command_receipts"].append(unrelated.public())
        if restored_owner:
            for row in data["chats"]:
                if row["session_id"] == "chat-b":
                    row["title"] = "Planning restored"
        return data

    monkeypatch.setattr(case.service, "snapshot", snapshot)
    await case.page.reload()
    await expect(
        case.page.locator('#chat-list button[data-session-id="chat-a"]')
    ).to_be_visible()
    await expect(
        case.page.locator('#chat-list button[data-session-id="chat-b"]')
    ).to_have_count(0)
    await expect(case.page.locator("#connection-state")).to_contain_text("Connected")
    old_ready, release_old, effect_response = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    gated = False
    host_started, release_host = asyncio.Event(), asyncio.Event()

    async def delay_host(route):
        host_started.set()
        await release_host.wait()
        await route.continue_()

    async def delay_old_list(route):
        nonlocal gated
        if gated or "session_id=" in route.request.url:
            await route.continue_()
            return
        gated = True
        response = await route.fetch()
        body = await response.json()
        assert all(row["session_id"] != "chat-b" for row in body["chats"])
        old_ready.set()
        await release_old.wait()
        await route.fulfill(response=response, json=body)

    async def observe_effect(route):
        response = await route.fetch()
        assert (await response.json())["native_id"] in {"chat-b", "chat-c"}
        effect_response.set()
        if outcome == "lost-response":
            await route.abort("failed")
        else:
            await route.fulfill(response=response)

    await case.page.route("**/api/v1/mobile/console/snapshot?*", delay_old_list)
    await case.page.route("**/api/v1/mobile/commands", observe_effect)
    if outcome == "bootstrap-owner":
        await case.page.route("**/api/v1/mobile/console/host", delay_host)
    case.service.receipt_state = "applied"
    case.service.release.set()
    try:
        if outcome == "bootstrap-owner":
            await case.page.locator("#new-chat-button").click()
            await asyncio.wait_for(host_started.wait(), 5)
        # Foreground refresh retains the connected state while its list is in flight.
        await case.page.evaluate(
            "document.dispatchEvent(new Event('visibilitychange'))"
        )
        await asyncio.wait_for(old_ready.wait(), 5)
        if outcome in {"older-owner", "bootstrap-owner"}:
            case.service.epoch = "new-explicit-owner"
        if outcome == "bootstrap-owner":
            release_host.set()
        else:
            await case.page.locator("#new-chat-button").click()
        await asyncio.wait_for(effect_response.wait(), 5)
        if outcome == "lost-response":
            await expect(case.page.locator("#status-line")).to_contain_text(
                "Unable to complete"
            )
        else:
            await expect(case.page.locator("#new-chat-button")).to_be_enabled()
        original = case.service.receipts["command-0"]
        case.service.receipts["unrelated-command"] = replace(
            original, command_id="unrelated-command", native_id="chat-a"
        )
        if outcome == "navigation":
            await case.page.locator(
                '#chat-list button[data-session-id="chat-a"]'
            ).click()
            await case.page.locator("#chat-draft").fill("KEEP_EXPLICIT_CHAT_A_DRAFT")
        if outcome == "newer-action":
            effect_response.clear()
            await case.page.locator("#new-chat-button").click()
            await asyncio.wait_for(effect_response.wait(), 5)
            await expect(case.page.locator("#new-chat-button")).to_be_enabled()
        replacement = outcome in {"replacement-owner", "replacement-console"}
        visible = True
        release_old.set()
        if outcome in {
            "delayed",
            "lost-response",
            "newer-action",
            "older-owner",
            "bootstrap-owner",
        }:
            await expect(case.page.locator("#chat-title")).to_have_text(
                "Newer explicit chat" if outcome == "newer-action" else "Planning"
            )
            await expect(case.page.locator("#chat-draft")).to_be_enabled()
            await expect(case.page.locator("#target-state")).not_to_contain_text(
                "Target not confirmed"
            )
        elif outcome == "navigation":
            await expect(case.page.locator("#chat-title")).to_have_text("Research")
            await expect(case.page.locator("#target-state")).not_to_contain_text(
                "Target not confirmed"
            )
            await expect(case.page.locator("#chat-draft")).to_have_value(
                "KEEP_EXPLICIT_CHAT_A_DRAFT"
            )
        else:
            await expect(
                case.page.locator('#chat-list button[data-session-id="chat-b"]')
            ).to_be_visible()
            await expect(case.page.locator("#chat-draft")).to_be_disabled()
            replacement = False
            restored_owner = True
            await case.service.events_queue.put(
                MobileEvent("console.changed", "", {}, 11)
            )
            await expect(
                case.page.locator('#chat-list button[data-session-id="chat-b"]')
            ).to_contain_text("Planning restored")
            await expect(case.page.locator("#chat-draft")).to_be_disabled()
        assert (
            len(case.commands)
            == len(case.service.tickets)
            == (2 if outcome == "newer-action" else 1)
        )
        assert case.commands[0]["command_id"] == "command-0"
        assert case.commands[0]["payload"] == {"workspace_id": "global"}
        if outcome == "lost-response":
            assert ("query", "command-0") in case.service.calls
    finally:
        release_old.set()
        release_host.set()
        case.page.remove_listener("response", observe_snapshot)


async def test_retired_answerability_reset_does_not_starve_current_metadata(
    mobile_browser_case,
):
    case = mobile_browser_case
    service, page = case.service, case.page
    service.body = "Current transcript"

    def approval(session_id, round_id):
        return dict(
            service.row(session_id),
            kind="approval",
            round_id=round_id,
            supported=True,
            calls=[
                {
                    "call_id": round_id,
                    "llm_name": "fs_read",
                    "options": ["approve_once", "deny"],
                    "arguments_summary": "/owned/readme",
                }
            ],
        )

    service.decisions = [approval("chat-a", "retired-round")]
    retired = False
    rejected = []
    original_answerable = service.set_answerable
    original_query = service.query_command

    async def answerable(*args):
        await original_answerable(*args)
        if retired and args[1].round_id == "retired-round":
            rejected.append(args[-1])
            raise ValueError("Decision is no longer remotely answerable.")

    async def delayed_query(command_id):
        # Keep the original receipt request in flight while native stale-round
        # rejection delivers reset; no mutation or alternate command is issued.
        await asyncio.sleep(0.05)
        return await original_query(command_id)

    service.set_answerable = answerable
    service.query_command = delayed_query
    await case.open_chat("chat-a")
    decision = page.get_by_role("button", name="Submit decision", exact=True)
    await expect(decision).to_be_enabled()
    service.release.set()
    await page.locator("#chat-draft").fill("One original accepted prompt")
    await page.get_by_role("button", name="Send", exact=True).click()
    await expect(page.locator("#command-status")).to_contain_text("accepted")
    assert len(service.commands) == 1
    retired = True
    service.decisions = []
    await service.events_queue.put(
        MobileEvent("reset", "", {"reset_required": True}, 2)
    )
    await expect(decision).to_have_count(0)
    await expect(page.locator("#connection-state")).to_contain_text("Connected")
    assert rejected and all(value is False for value in rejected), rejected
    assert len(rejected) <= 2, rejected
    assert len(service.commands) == 1 and len(service.tickets) == 1

    # Only current native metadata can restore a true claim.
    service.decisions = [approval("chat-a", "current-round")]
    await service.events_queue.put(MobileEvent("console.changed", "", {}, 3))
    await expect(decision).to_be_enabled()

    def true_claims(round_id):
        return [
            args
            for name, args in service.calls
            if name == "answerable"
            and args[1].round_id == round_id
            and args[-1] is True
        ]

    async with asyncio.timeout(5):
        while not true_claims("current-round"):
            await asyncio.sleep(0.01)
    first_client = true_claims("current-round")[-1][0]
    await service.events_queue.put(None)  # End only this genuine socket stream.
    async with asyncio.timeout(5):
        while not any(args[0] != first_client for args in true_claims("current-round")):
            await asyncio.sleep(0.01)
    await expect(decision).to_be_enabled()

    # A new rendered target is advertised on that current socket as well.
    service.decisions = [approval("chat-b", "new-target-round")]
    await page.locator("#back-button").click()
    await case.open_chat("chat-b")
    await expect(decision).to_be_enabled()
    async with asyncio.timeout(5):
        while not true_claims("new-target-round"):
            await asyncio.sleep(0.01)
    assert all(
        args[1].session_id == "chat-b" for args in true_claims("new-target-round")
    )
    assert len(service.commands) == 1
