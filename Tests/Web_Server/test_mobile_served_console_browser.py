"""Actual retained TldwCli, canonical stores and TLS routes; never a fake server."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("section", ["paths", "Paths", None])
def test_server_storage_selection_keeps_selected_data_directory(tmp_path, section):
    home = tmp_path / "contained-home"
    home.mkdir(mode=0o700)
    (home / "selected-data").mkdir(mode=0o700)
    (home / "selected-core").mkdir(mode=0o700)
    selector = home / "selected" / "config.toml"
    selector.parent.mkdir(mode=0o700)
    selector.write_text(
        (f'[{section}]\ndata_dir="{home / "selected-data"}"\n' if section else "")
        + '[general]\nusers_name="selected-operator"\n'
        f'[database]\nchachanotes_db_path="{home / "selected-core/chat.db"}"\n'
        f'workspaces_db_path="{home / "selected-core/workspaces.db"}"\n'
    )
    selector.chmod(0o600)
    environment = dict(
        os.environ,
        HOME=str(home),
        USERPROFILE=str(home),
        TLDW_CONFIG_PATH=str(selector),
        TLDW_TEST_CONFIG_ROOT=str(home / ".test-config"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        PYTHONPATH=str(REPOSITORY),
    )
    script = r"""
from pathlib import Path
from Tests.real_profile_guard import install as guard, take_violations
guard()
from Tests.network_guard import install, blocked_attempts
install()
from tldw_chatbook.config import get_atomic_config_snapshot
from tldw_chatbook.Backup_Recovery.profile_paths import canonical_console_paths, database_path, legacy_mobile_dir
from tldw_chatbook.Web_Server.serve import ChatbookWebServerMixin
selected = get_atomic_config_snapshot().values
stores, media, source = ChatbookWebServerMixin._canonical_console_selection(None)
assert stores.paths == canonical_console_paths(selected)
assert media == database_path(selected, "media_db_path"), (media, database_path(selected, "media_db_path"))
assert source == legacy_mobile_dir(selected)
assert source.name == "mobile" and source.parent.name == "selected-operator"
assert not blocked_attempts() and not take_violations()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    (tmp_path / "selection.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr


def _prepare_profile(tmp_path, *, mobile_surface=True):
    """Cold profile selection precedes every parent/child application import."""
    from Tests.Backup_Recovery.test_mobile_public_operator import (
        _NO_DATA,
        _START_CANONICAL,
    )

    parent = Path(tempfile.gettempdir()).resolve()
    assert parent.stat().st_uid == os.geteuid() and not parent.stat().st_mode & 0o077
    root = Path(tempfile.mkdtemp(prefix="task11-served-", dir=parent))
    (tmp_path / "retained-fixture.json").write_text(json.dumps({"root": str(root)}))
    home = root / "home"
    home.mkdir(mode=0o700)
    selector = home / ".config" / "tldw_cli" / "config.toml"
    selector.parent.mkdir(parents=True, mode=0o700)
    (home / "data").mkdir(mode=0o700)
    (home / "canonical").mkdir(mode=0o700)
    selector.write_text(
        '[general]\nusers_name="operator"\ndefault_tab="chat"\n'
        "[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n"
        "[model_catalog]\nauto_refresh_enabled=false\nrefresh_consent_recorded=true\n"
        '[tldw_api]\nbase_url=""\napi_url=""\nurl=""\n'
        '[chat_defaults]\nprovider="llama_cpp"\nmodel="gpt-4o"\n'
        '[api_settings.llama_cpp]\napi_url="http://127.0.0.1:9099"\nmodel="gpt-4o"\n'
        f'[paths]\ndata_dir="{home / "data"}"\n'
        f'[database]\nchachanotes_db_path="{home / "canonical/chat.db"}"\n'
        f'workspaces_db_path="{home / "canonical/workspaces.db"}"\n'
        + ("[web_server]\nmobile_surface=false\n" if not mobile_surface else "")
    )
    selector.chmod(0o600)
    assert not (selector.parent / "tokenizers").exists()
    environment = dict(
        os.environ,
        HOME=str(home),
        USERPROFILE=str(home),
        TLDW_TEST_CONFIG_ROOT=str(home / ".test-config"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        PYTHONPATH=str(REPOSITORY),
    )
    for name in (
        "TLDW_CONFIG_PATH",
        "TLDW_TEST_MOBILE_CLI",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
    ):
        environment.pop(name, None)
    for name, script in (("startup", _START_CANONICAL), ("cutover", _NO_DATA)):
        with (root / f"{name}.log").open("w") as output:
            result = subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "faulthandler",
                    "-c",
                    script,
                    "--config",
                    str(selector),
                    str(root / f"{name}.json"),
                ],
                cwd=REPOSITORY,
                env=environment,
                stdout=output,
                stderr=output,
                timeout=180,
                check=False,
            )
        assert result.returncode == 0, (
            f"{root}/{name}.log\n" + (root / f"{name}.log").read_text()[-6000:]
        )
    assert (
        json.loads((root / "cutover.json").read_text())["finished_state"] == "succeeded"
    )
    environment["TLDW_CONFIG_PATH"] = str(selector)
    environment["TLDW_MOBILE_CANONICAL_FIXTURE"] = "1"
    return root, environment


class LiveMobileConsole:
    """One isolated fixture owner, with browser operations in its child process.

    The subprocess boundary ensures profile selection precedes app imports. The
    same owner type carries the TLS stack and browser contexts in that process.
    """

    def __init__(self, root, environment=None):
        self.root = root
        self.environment = environment
        self.data = root / "test_data"
        self.prompts = []
        self.stack = None
        self.desktop = None
        self.phone = None

    async def start_gated_chat(self, text):
        from Tests.Canvas.browser.canvas_live_harness import _send_console_prompt

        await _send_console_prompt(
            self.desktop, text, self.data / "canvas-live-composer-focused"
        )
        self.prompts.append(text)
        entry = json.loads(
            await _wait_file(
                self.data
                / (
                    "console-entry-"
                    + hashlib.sha256(text.encode()).hexdigest()
                    + ".json"
                )
            )
        )
        terminal = json.loads((self.data / "canvas-live-terminal-state").read_text())
        return entry, terminal

    async def phone_send(self, text):
        from playwright.async_api import expect

        await self.phone.locator("#chat-draft").fill(text)
        await self.phone.get_by_role("button", name="Send", exact=True).click()
        await expect(self.phone.locator("#command-status")).to_contain_text(
            "queued", timeout=30000
        )
        self.prompts.append(text)

    async def release_agent(self):
        for prompt in self.prompts:
            (
                self.data
                / ("console-release-" + hashlib.sha256(prompt.encode()).hexdigest())
            ).touch()

    async def snapshot(self):
        response = await self.phone.request.get(
            self.stack.origin + "/api/v1/mobile/console/snapshot?protocol_version=1"
        )
        assert response.status == 200, await response.text()
        return await response.json()

    async def close_terminal(self):
        await self.desktop.goto("about:blank")

    async def reattach_terminal(self):
        from Tests.Canvas.browser.canvas_live_harness import (
            _wait_for_initial_console_ready,
        )

        await self.desktop.goto(self.stack.origin)
        await _wait_for_initial_console_ready(self.desktop)
        (self.data / "canvas-live-terminal-state").unlink()
        await self.desktop.locator("#terminal .xterm-helper-textarea").press("F11")
        return json.loads(await _wait_file(self.data / "canvas-live-terminal-state"))


@pytest.fixture
def live_mobile_console(tmp_path):
    return LiveMobileConsole(*_prepare_profile(tmp_path))


def test_phone_reconnect_keeps_the_same_console_run(live_mobile_console):
    root, environment = live_mobile_console.root, live_mobile_console.environment
    selector = Path(environment["TLDW_CONFIG_PATH"])
    with (root / "retention.log").open("w") as output:
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-m", __name__, str(root)],
            cwd=REPOSITORY,
            env=environment,
            stdout=output,
            stderr=output,
            timeout=180,
            check=False,
        )
    assert result.returncode == 0, (
        f"{root}/retention.log\n" + (root / "retention.log").read_text()[-10000:]
    )
    evidence = json.loads((root / "retention.json").read_text())
    assert evidence["gateway_calls"] == 2
    assert evidence["physical_exit"] == [0]
    assert not (selector.parent / "tokenizers").exists()


async def _wait_file(path, timeout=45):
    async with asyncio.timeout(timeout):
        while not path.exists():
            await asyncio.sleep(0.05)
    return path.read_text()


async def _run_retention(root):
    from playwright.async_api import async_playwright, expect

    from Tests.Canvas.browser.canvas_live_harness import (
        _login_live_page,
        _wait_for_initial_console_ready,
        chromium_executable,
        start_live_served_stack,
    )
    from Tests.network_guard import blocked_attempts
    from Tests.real_profile_guard import take_violations
    from tldw_chatbook.Backup_Recovery.profile_paths import canonical_console_paths
    from tldw_chatbook.config import load_settings

    case = LiveMobileConsole(root)
    token = secrets.token_urlsafe(24)
    patch = pytest.MonkeyPatch()
    stack = await start_live_served_stack(
        root,
        patch,
        access_token=token,
        child_module="Tests.Canvas.browser.canvas_live_chatbook_child",
    )
    case.stack = stack
    data = case.data
    result = {}
    source_files = [
        "Tests/Web_Server/test_mobile_served_console_browser.py",
        "Tests/Canvas/browser/canvas_live_harness.py",
        "Tests/Canvas/browser/canvas_live_chatbook_child.py",
        "tldw_chatbook/Web_Server/serve.py",
        "tldw_chatbook/Web_Server/mobile_service.py",
        "tldw_chatbook/Web_Server/static/mobile/app.js",
        "tldw_chatbook/app.py",
        "tldw_chatbook/app_served.py",
        "tldw_chatbook/Chat/console_chat_controller.py",
        "tldw_chatbook/Utils/custom_tokenizers.py",
        "tldw_chatbook/Backup_Recovery/mobile_enrollment.py",
        "tldw_chatbook/Backup_Recovery/storage_admission.py",
    ]
    (root / "source-before.json").write_text(
        json.dumps(
            {
                "python": sys.executable,
                "origins": {
                    name: hashlib.sha256((REPOSITORY / name).read_bytes()).hexdigest()
                    for name in source_files
                },
            },
            indent=2,
        )
    )
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=chromium_executable(playwright.chromium), headless=True
            )
            desktop_context = await browser.new_context(
                ignore_https_errors=True, viewport={"width": 1280, "height": 900}
            )
            phone_context = await browser.new_context(
                ignore_https_errors=True, viewport={"width": 390, "height": 844}
            )
            case.desktop_context, case.phone_context = desktop_context, phone_context
            desktop = case.desktop = await desktop_context.new_page()
            await _login_live_page(desktop, origin=stack.origin, access_token=token)
            await _wait_for_initial_console_ready(desktop)
            entry, before_terminal = await case.start_gated_chat("EXACT_SHARED_PROMPT")
            assert not (Path.home() / ".config/tldw_cli/tokenizers").exists()
            process = stack.services[0]._process
            assert entry["pid"] == process.pid == before_terminal["pid"]
            assert (
                Path(before_terminal["core_path"])
                == canonical_console_paths(load_settings())[0]
            )
            phone = case.phone = await phone_context.new_page()

            async def record_response(response):
                if "/api/v1/mobile/" in response.url:
                    with (root / "phone-http.jsonl").open("a") as output:
                        output.write(
                            json.dumps(
                                {
                                    "url": response.url,
                                    "status": response.status,
                                    "body": await response.text(),
                                }
                            )
                            + "\n"
                        )

            phone.on("response", record_response)
            # Authenticate without attaching a second terminal.
            login = await phone.request.post(
                stack.origin + "/auth/login",
                form={"access_token": token},
                headers={"Origin": stack.origin},
                max_redirects=0,
            )
            assert login.status == 303
            await phone.goto(stack.origin + "/m")
            await expect(phone.locator("#chat-list button").first).to_be_visible(
                timeout=30000
            )

            snapshot = case.snapshot

            before = await snapshot()
            result.update(
                pid=process.pid, before=before, terminal_before=before_terminal
            )
            (root / "snapshot-before.json").write_text(json.dumps(before, indent=2))
            await case.close_terminal()
            await phone.reload()
            await expect(phone.locator("#chat-list button").first).to_be_visible()
            during = await snapshot()
            assert process.returncode is None and len(stack.services) == 1
            assert before["chats"] == during["chats"]
            result["during"] = during
            await phone.locator("#chat-list button").first.click()
            await case.phone_send("EXACT_PHONE_FOLLOWUP")
            await case.release_agent()
            await _wait_file(
                data
                / (
                    "console-entry-"
                    + hashlib.sha256(b"EXACT_PHONE_FOLLOWUP").hexdigest()
                    + ".json"
                )
            )
            await expect(phone.locator("#chat-transcript")).to_contain_text(
                "EXACT_REPLY:EXACT_PHONE_FOLLOWUP", timeout=30000
            )
            assert not (Path.home() / ".config/tldw_cli/tokenizers").exists()
            after = await case.reattach_terminal()
            keyboard = desktop.locator("#terminal .xterm-helper-textarea")
            assert process.pid == after["pid"]
            assert after["input_thread"] == before_terminal["input_thread"]
            assert (
                after["input_thread_object"] == before_terminal["input_thread_object"]
            )
            assert after["input_thread_objects_observed"] == 1
            result.update(
                pid=process.pid,
                before=before,
                during=during,
                terminal=after,
                gateway_calls=int((data / "console-gateway-calls").read_text()),
            )
            state_path = data / "canvas-live-terminal-state"

            async def terminal_receipt():
                state_path.unlink(missing_ok=True)
                await keyboard.press("F11")
                return json.loads(await _wait_file(state_path, timeout=10))

            await desktop.set_viewport_size({"width": 1100, "height": 800})
            async with asyncio.timeout(10):
                while True:
                    resized = await terminal_receipt()
                    if resized["width"] < after["width"]:
                        break
                    await asyncio.sleep(0.05)
            region = resized["composer_region"]
            bounds = await desktop.locator("#terminal .xterm-screen").bounding_box()
            await desktop.mouse.click(
                bounds["x"] + (region[0] + 1) * bounds["width"] / resized["width"],
                bounds["y"] + (region[1] + 0.5) * bounds["height"] / resized["height"],
            )
            clicked = await terminal_receipt()
            assert clicked["focused_id"] == "console-native-composer"
            assert clicked["mouse"] == [region[0] + 1, region[1]]
            paste = "desktop retained first line\ndesktop retained second line"
            await keyboard.evaluate(
                """(element, text) => {
                const clipboard = new DataTransfer(); clipboard.setData('text/plain', text);
                element.dispatchEvent(new ClipboardEvent('paste', {clipboardData: clipboard, bubbles: true}));
            }""",
                paste,
            )
            async with asyncio.timeout(10):
                while True:
                    pasted = await terminal_receipt()
                    if pasted["draft"].splitlines() == paste.splitlines():
                        break
                    await asyncio.sleep(0.05)
            assert pasted["input_thread"] == before_terminal["input_thread"]
            assert pasted["input_threads"] == 1
            assert (
                pasted["input_thread_object"] == before_terminal["input_thread_object"]
            )
            assert pasted["input_thread_objects_observed"] == 1
            result["terminal_interaction"] = {
                "resized": resized,
                "clicked": clicked,
                "pasted": pasted,
            }
            csrf = await phone.locator('meta[name="chatbook-csrf"]').get_attribute(
                "content"
            )
            endpoint = stack.origin + "/api/v1/mobile/commands/tickets"
            unauth = await browser.new_context(ignore_https_errors=True)
            response = await unauth.request.post(endpoint, data={"protocol_version": 1})
            assert response.status == 401
            await unauth.close()
            response = await phone.request.post(endpoint, data={"protocol_version": 1})
            assert response.status == 401
            response = await phone.request.post(
                endpoint,
                data={"protocol_version": 1},
                headers={"Origin": "https://wrong.invalid", "X-Chatbook-CSRF": csrf},
            )
            assert response.status == 401
            await stack.server._disable_canvas_runtime()
            current = await snapshot()
            assert current["console_id"] == before["console_id"]
            assert process.returncode is None
            after_disable = await terminal_receipt()
            assert after_disable["draft"] == pasted["draft"]
            assert after_disable["session_id"] == before_terminal["session_id"]
            result["canvas_disabled_retained"] = after_disable
            await desktop.screenshot(path=str(root / "retained-terminal.png"))
            await phone.screenshot(path=str(root / "retained-phone-viewport.png"))
            await browser.close()
    except BaseException:
        import traceback

        (root / "journey-error.txt").write_text(traceback.format_exc())
        if "phone" in locals() and not phone.is_closed():
            (root / "phone-failure.html").write_text(await phone.content())
        raise
    finally:
        processes = [service._process for service in stack.services]
        for entry_path in data.glob("console-entry-*.json"):
            (
                data
                / entry_path.name.replace(
                    "console-entry-", "console-release-"
                ).removesuffix(".json")
            ).touch()
        import shutil

        if data.exists():
            shutil.copytree(data, root / "child-artifacts", dirs_exist_ok=True)
        try:
            await stack.aclose()
        finally:
            result["physical_exit"] = [process.returncode for process in processes]
            (root / "retention.json").write_text(json.dumps(result, indent=2))
            (root / "source-after.json").write_text(
                json.dumps(
                    {
                        name: hashlib.sha256(
                            (REPOSITORY / name).read_bytes()
                        ).hexdigest()
                        for name in source_files
                    },
                    indent=2,
                )
            )
            patch.undo()
    import sqlite3

    database = Path(result["terminal"]["core_path"])
    assert result["physical_exit"] == [0]
    assert not database.with_name(database.name + "-wal").exists()
    with sqlite3.connect(
        database.as_uri() + "?mode=ro&immutable=1", uri=True
    ) as connection:
        rows = connection.execute(
            "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY rowid",
            (result["before"]["chats"][0]["conversation_id"],),
        ).fetchall()
    assert rows == [
        ("user", "EXACT_SHARED_PROMPT"),
        ("assistant", "EXACT_REPLY:EXACT_SHARED_PROMPT"),
        ("user", "EXACT_PHONE_FOLLOWUP"),
        ("assistant", "EXACT_REPLY:EXACT_PHONE_FOLLOWUP"),
    ]
    result["persisted_rows"] = rows
    (root / "retention.json").write_text(json.dumps(result, indent=2))
    assert not blocked_attempts(), blocked_attempts()
    assert not take_violations()


@pytest.mark.parametrize("trusted_proxy", [False, True])
def test_cold_phone_create_then_immediate_send(live_mobile_console, trusted_proxy):
    root, environment = live_mobile_console.root, live_mobile_console.environment
    environment = dict(
        environment, TLDW_TEST_TRUSTED_PROXY="1" if trusted_proxy else "0"
    )
    with (root / "cold-phone.log").open("w") as output:
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-m", __name__, str(root), "cold"],
            cwd=REPOSITORY,
            env=environment,
            stdout=output,
            stderr=output,
            timeout=120,
            check=False,
        )
    assert result.returncode == 0, (root / "cold-phone.log").read_text()[-10000:]
    evidence = json.loads((root / "cold-phone.json").read_text())
    assert evidence["physical_exit"] == [0]
    assert evidence["gateway_calls"] == 1


async def _run_cold_phone(root):
    import shutil
    import sqlite3

    from playwright.async_api import async_playwright, expect

    from Tests.Canvas.browser.canvas_live_harness import (
        chromium_executable,
        start_live_served_stack,
    )
    from Tests.network_guard import blocked_attempts
    from Tests.real_profile_guard import take_violations
    from tldw_chatbook.Backup_Recovery.profile_paths import canonical_console_paths
    from tldw_chatbook.config import get_atomic_config_snapshot

    patch = pytest.MonkeyPatch()
    token = secrets.token_urlsafe(24)
    stack = await start_live_served_stack(
        root,
        patch,
        access_token=token,
        child_module="Tests.Canvas.browser.canvas_live_chatbook_child",
        trusted_proxy=os.environ.get("TLDW_TEST_TRUSTED_PROXY") == "1",
    )
    case = LiveMobileConsole(root)
    case.stack = stack
    result = {"initial_services": len(stack.services)}
    assert result["initial_services"] == 0
    responses = []
    tasks = set()

    async def record(response):
        if "/api/v1/mobile/" in response.url:
            row = {
                "url": response.url,
                "status": response.status,
                "body": await response.text(),
            }
            responses.append(row)
            with (root / "cold-phone-http.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")

    def observe(response):
        task = asyncio.create_task(record(response))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=chromium_executable(playwright.chromium),
                headless=True,
            )
            context = await browser.new_context(
                ignore_https_errors=True, viewport={"width": 390, "height": 844}
            )
            phone = case.phone = await context.new_page()
            phone.on("response", observe)
            login = await phone.request.post(
                stack.origin + "/auth/login",
                form={"access_token": token},
                headers={"Origin": stack.origin},
                max_redirects=0,
            )
            assert login.status == 303
            await phone.goto(stack.origin + "/m")
            assert not stack.services  # Authenticated reads do not start a child.
            # Only shipped controls determine readiness. No terminal attachment,
            # post-mount receipt, execution-admitted poll, or ticket retry.
            await phone.get_by_role("button", name="New chat", exact=True).click()
            await expect(phone.locator("#chat-draft")).to_be_enabled(timeout=30000)
            await phone.locator("#chat-draft").fill("EXACT_COLD_PHONE_PROMPT")
            await phone.get_by_role("button", name="Send", exact=True).click()
            entry = json.loads(
                await _wait_file(
                    case.data
                    / (
                        "console-entry-"
                        + hashlib.sha256(b"EXACT_COLD_PHONE_PROMPT").hexdigest()
                        + ".json"
                    )
                )
            )
            assert (
                len(stack.services) == 1
                and entry["pid"] == stack.services[0]._process.pid
            )
            assert not (Path.home() / ".config/tldw_cli/tokenizers").exists()
            case.prompts.append("EXACT_COLD_PHONE_PROMPT")
            await case.release_agent()
            await expect(phone.locator("#chat-transcript")).to_contain_text(
                "EXACT_REPLY:EXACT_COLD_PHONE_PROMPT",
                timeout=30000,
            )
            result.update(
                entry=entry,
                snapshot=await case.snapshot(),
                gateway_calls=int((case.data / "console-gateway-calls").read_text()),
            )
            await phone.screenshot(path=str(root / "cold-phone.png"))
            if tasks:
                await asyncio.gather(*tasks)
            commands = [row for row in responses if row["url"].endswith("/commands")]
            assert len(commands) == 2 and all(
                row["status"] in {200, 202} for row in commands
            ), commands
            send_id = json.loads(commands[1]["body"])["command_id"]
            receipt_response = await phone.request.get(
                stack.origin + "/api/v1/mobile/commands/" + send_id
            )
            assert receipt_response.status == 200
            receipt = await receipt_response.json()
            assert receipt["command_id"] == send_id and receipt["state"] == "accepted"
            result["send_receipt"] = receipt
            result["commands"] = commands
            result["trusted_proxy"] = os.environ.get("TLDW_TEST_TRUSTED_PROXY") == "1"
            result["proxy_counts"] = stack.proxy_counts
            await browser.close()
    except BaseException:
        import traceback

        (root / "cold-phone-error.txt").write_text(traceback.format_exc())
        raise
    finally:
        for entry_path in case.data.glob("console-entry-*.json"):
            (
                case.data
                / entry_path.name.replace(
                    "console-entry-", "console-release-"
                ).removesuffix(".json")
            ).touch()
        if case.data.exists():
            shutil.copytree(case.data, root / "child-artifacts", dirs_exist_ok=True)
        await stack.aclose()
        result["physical_exit"] = [
            service._process.returncode
            for service in stack.services
            if service._process is not None
        ]
        (root / "cold-phone.json").write_text(json.dumps(result, indent=2))
        patch.undo()
    assert result["physical_exit"] == [0]
    database = canonical_console_paths(get_atomic_config_snapshot().values)[0]
    with sqlite3.connect(
        database.as_uri() + "?mode=ro&immutable=1", uri=True
    ) as connection:
        rows = connection.execute(
            "SELECT role,content FROM messages ORDER BY rowid"
        ).fetchall()
    assert rows == [
        ("user", "EXACT_COLD_PHONE_PROMPT"),
        ("assistant", "EXACT_REPLY:EXACT_COLD_PHONE_PROMPT"),
    ]
    (root / "cold-phone-persisted.json").write_text(json.dumps(rows))
    assert not (Path.home() / ".config/tldw_cli/tokenizers").exists()
    assert not blocked_attempts(), blocked_attempts()
    assert not take_violations()


def test_two_chats_workers_and_exact_phone_controls(live_mobile_console):
    root, environment = live_mobile_console.root, live_mobile_console.environment
    environment = dict(environment, TLDW_FLEET_FIXTURE="1")
    with (root / "fleet.log").open("w") as output:
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-m", __name__, str(root), "fleet"],
            cwd=REPOSITORY,
            env=environment,
            stdout=output,
            stderr=output,
            timeout=180,
            check=False,
        )
    assert result.returncode == 0, (root / "fleet.log").read_text()[-12000:]
    result = json.loads((root / "fleet.json").read_text())
    assert result["physical_exit"] == [0]


async def _run_fleet(root):
    import shutil

    from playwright.async_api import async_playwright, expect

    from Tests.Canvas.browser.canvas_live_harness import (
        _login_live_page,
        _wait_for_initial_console_ready,
        chromium_executable,
        start_live_served_stack,
    )
    from Tests.network_guard import blocked_attempts
    from Tests.real_profile_guard import take_violations

    patch = pytest.MonkeyPatch()
    token = secrets.token_urlsafe(24)
    stack = await start_live_served_stack(
        root,
        patch,
        access_token=token,
        child_module="Tests.Canvas.browser.canvas_live_chatbook_child",
    )
    case = LiveMobileConsole(root)
    case.stack = stack
    result = {}
    data = case.data
    from tldw_chatbook.Web_Server import mobile_api

    original_error = mobile_api._command_error

    async def observed_error(error):
        import traceback

        with (root / "fleet-api-errors.txt").open("a") as output:
            output.write("".join(traceback.format_exception(error)))
        return await original_error(error)

    patch.setattr(mobile_api, "_command_error", observed_error)
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=chromium_executable(playwright.chromium), headless=True
            )
            desktop_context = await browser.new_context(
                ignore_https_errors=True, viewport={"width": 1280, "height": 900}
            )
            phone_context = await browser.new_context(
                ignore_https_errors=True, viewport={"width": 390, "height": 844}
            )
            desktop = case.desktop = await desktop_context.new_page()
            await _login_live_page(desktop, origin=stack.origin, access_token=token)
            await _wait_for_initial_console_ready(desktop)
            entry_a, terminal_a = await case.start_gated_chat("EXACT_UNRELATED_CHAT_A")
            keyboard = desktop.locator("#terminal .xterm-helper-textarea")
            await keyboard.evaluate("""(element) => {
                const clipboard = new DataTransfer(); clipboard.setData('text/plain', 'EXACT_DESKTOP_DRAFT');
                element.dispatchEvent(new ClipboardEvent('paste', {clipboardData:clipboard,bubbles:true}));
            }""")
            phone = case.phone = await phone_context.new_page()
            login = await phone.request.post(
                stack.origin + "/auth/login",
                form={"access_token": token},
                headers={"Origin": stack.origin},
                max_redirects=0,
            )
            assert login.status == 303
            await phone.goto(stack.origin + "/m")
            await phone.get_by_role("button", name="New chat", exact=True).click()
            await expect(phone.locator("#chat-draft")).to_be_enabled(timeout=30000)
            before = await case.snapshot()
            chat_b = next(
                row
                for row in before["chats"]
                if row["session_id"] != terminal_a["session_id"]
            )
            target_b = {
                key: chat_b[key]
                for key in (
                    "server_epoch",
                    "console_id",
                    "session_id",
                    "incarnation_id",
                )
            }
            await phone.locator("#chat-draft").fill("EXACT_FLEET_PARENT")
            await phone.get_by_role("button", name="Send", exact=True).click()
            await _wait_file(data / "fleet-entry-EXACT_FLEET_PARENT-0.json")
            csrf = await phone.locator('meta[name="chatbook-csrf"]').get_attribute(
                "content"
            )
            headers = {"Origin": stack.origin, "X-Chatbook-CSRF": csrf}

            async def targeted():
                response = await phone.request.get(
                    stack.origin + "/api/v1/mobile/console/snapshot",
                    params={"protocol_version": "1", "target": json.dumps(target_b)},
                )
                assert response.status == 200, await response.text()
                snapshot = await response.json()
                (root / "fleet-latest-snapshot.json").write_text(
                    json.dumps(snapshot, indent=2)
                )
                return snapshot

            async def command(action, target, payload):
                response = await phone.request.post(
                    stack.origin + "/api/v1/mobile/commands/tickets",
                    headers=headers,
                    data={
                        "protocol_version": 1,
                        "action": action,
                        "target": target,
                        "payload": payload,
                    },
                )
                assert response.status == 200, await response.text()
                ticket = await response.json()
                response = await phone.request.post(
                    stack.origin + "/api/v1/mobile/commands",
                    headers=headers,
                    data={
                        "protocol_version": 1,
                        "ticket": ticket["ticket"],
                        "payload": payload,
                    },
                )
                assert response.status in {200, 202}, await response.text()
                receipt = await response.json()
                result.setdefault("commands", []).append(
                    {
                        "action": action,
                        "target": target,
                        "payload": payload,
                        "receipt": receipt,
                    }
                )
                assert receipt["state"] in {"accepted", "queued", "applied"}, receipt
                return receipt

            # Native tool policy is exercised as rendered: approve only the exact
            # ordinary canonical character-search read, and answer the real question.
            approved_rounds = set()
            async with asyncio.timeout(45):
                while True:
                    snapshot = await targeted()
                    rounds = snapshot.get("decisions", [])
                    if rounds:
                        decision = rounds[0]
                        if (
                            decision["kind"] == "approval"
                            and decision["supported"]
                            and decision["round_id"] not in approved_rounds
                        ):
                            approved_rounds.add(decision["round_id"])
                            assert all(
                                call.get("llm_name", call.get("tool_name"))
                                == "character_search"
                                for call in decision["calls"]
                            ), decision
                            result.setdefault("approvals", []).append(decision)
                            await expect(
                                phone.locator("#chat-decisions select").first
                            ).to_be_visible()
                            for select in await phone.locator(
                                "#chat-decisions select"
                            ).all():
                                await select.select_option("approve_once")
                            await phone.get_by_role(
                                "button", name="Submit decision", exact=True
                            ).click()
                        elif decision["kind"] == "question" and decision["supported"]:
                            break
                    await asyncio.sleep(0.1)
            result["question"] = decision
            await _wait_file(data / "fleet-entry-EXACT_WORKER_A-0.json")
            await _wait_file(data / "fleet-entry-EXACT_WORKER_B-0.json")
            snapshot = await targeted()
            assert len(snapshot["fleet"]) == 2, snapshot
            result["workers_before"] = snapshot["fleet"]
            workers = sorted(
                snapshot["fleet"],
                key=lambda row: row.get("task", row.get("handle_id", "")),
            )
            # Match task labels rather than relying on ordering of async starts.
            worker_a = next(
                row for row in workers if "EXACT_WORKER_A" in json.dumps(row)
            )
            worker_b = next(
                row for row in workers if "EXACT_WORKER_B" in json.dumps(row)
            )
            keys = (
                "server_epoch",
                "console_id",
                "session_id",
                "incarnation_id",
                "run_id",
                "handle_id",
            )
            worker_target = {key: worker_a[key] for key in keys}
            ui_tickets = []

            def record_ui_ticket(request):
                if (
                    request.url.endswith("/commands/tickets")
                    and request.method == "POST"
                ):
                    ui_tickets.append(request.post_data_json)

            phone.on("request", record_ui_ticket)
            await phone.locator("#chat-draft").fill("EXACT_PHONE_MAIN_DRAFT")
            await phone.locator("#agents-button").click()
            await phone.locator(
                f'#agent-list button[data-agent-key="{worker_a["handle_id"]}"]'
            ).click()
            await expect(phone.locator("#chat-draft")).to_have_value("")
            await phone.locator("#chat-draft").fill("EXACT_WORKER_GUIDANCE")
            await phone.get_by_role("button", name="Send guidance", exact=True).click()
            await expect(phone.locator("#command-status")).to_contain_text(
                "consumption not confirmed", timeout=15000
            )
            issued = [row for row in ui_tickets if row["action"] == "subagent_guidance"]
            assert len(issued) == 1 and issued[0]["payload"] == {
                "text": "EXACT_WORKER_GUIDANCE"
            }
            assert all(
                issued[0]["target"][key] == value
                for key, value in worker_target.items()
            )
            result["worker_ui_ticket"] = issued[0]
            await phone.locator('#agent-list button[data-agent-key="main"]').click()
            await expect(phone.locator("#chat-draft")).to_have_value(
                "EXACT_PHONE_MAIN_DRAFT"
            )
            (data / "fleet-release-EXACT_WORKER_A-0").touch()
            consumed = json.loads(
                await _wait_file(data / "fleet-entry-EXACT_WORKER_A-1.json")
            )
            assert any(
                "EXACT_WORKER_GUIDANCE" in text for text in consumed["received"]
            ), consumed
            result["guidance_consumed"] = consumed
            await command("subagent_stop", {key: worker_b[key] for key in keys}, {})
            # Another phone may render the same supported decision and disappear.
            second_context = await browser.new_context(
                ignore_https_errors=True, viewport={"width": 390, "height": 844}
            )
            second = await second_context.new_page()
            login = await second.request.post(
                stack.origin + "/auth/login",
                form={"access_token": token},
                headers={"Origin": stack.origin},
                max_redirects=0,
            )
            assert login.status == 303
            await second.goto(stack.origin + "/m")
            await second.locator(
                f'#chat-list button[data-session-id="{target_b["session_id"]}"]'
            ).click()
            await expect(
                second.get_by_role("button", name="Submit decision", exact=True)
            ).to_be_visible(timeout=15000)
            await second_context.close()
            await phone.get_by_label("EXACT_ANSWER_A", exact=True).check()
            await phone.get_by_role(
                "button", name="Submit decision", exact=True
            ).click()
            final_parent = json.loads(
                await _wait_file(data / "fleet-entry-EXACT_FLEET_PARENT-6.json")
            )
            assert any("EXACT_ANSWER_A" in text for text in final_parent["received"]), (
                final_parent
            )
            tool_result = next(
                text
                for text in final_parent["received"]
                if text.startswith("Tool result for character_search: ")
            )
            tool_result = json.loads(tool_result.split(": ", 1)[1])
            assert tool_result["status"] == "ok" and tool_result["items"], tool_result
            assert len(result["approvals"]) == 1
            result["local_tool_result"] = tool_result
            result["question_consumed"] = final_parent
            snapshot = await targeted()
            main = next(
                row
                for row in snapshot["chats"]
                if row["session_id"] == target_b["session_id"]
            )
            run_target = dict(target_b, run_id=main["run_id"], turn_id=main["turn_id"])
            await command("steer", run_target, {"text": "EXACT_PARENT_STEER"})
            await command("redirect", run_target, {"text": "EXACT_PARENT_REDIRECT"})
            current = await targeted()
            stopped = next(
                row
                for row in current["chats"]
                if row["session_id"] == target_b["session_id"]
            )
            assert stopped["run_id"] or stopped["turn_id"]
            await command(
                "stop",
                dict(target_b, run_id=stopped["run_id"], turn_id=stopped["turn_id"]),
                {},
            )
            state = data / "canvas-live-terminal-state"
            state.unlink(missing_ok=True)
            await keyboard.press("F11")
            terminal_after = json.loads(await _wait_file(state))
            assert terminal_after["session_id"] == terminal_a["session_id"]
            assert terminal_after["draft"] == "EXACT_DESKTOP_DRAFT"
            assert (
                terminal_after["input_thread_object"]
                == terminal_a["input_thread_object"]
            )
            assert stack.services[0]._process.pid == entry_a["pid"]
            result["desktop_before"] = terminal_a
            result["desktop_after"] = terminal_after
            case.prompts.append("EXACT_UNRELATED_CHAT_A")
            await case.release_agent()
            (data / "fleet-release-EXACT_WORKER_A-1").touch()
            (data / "fleet-release-parent").touch()
            await browser.close()
    except BaseException:
        import traceback

        (root / "fleet-error.txt").write_text(traceback.format_exc())
        raise
    finally:
        for name in ("EXACT_WORKER_A", "EXACT_WORKER_B"):
            for phase in (0, 1):
                (data / f"fleet-release-{name}-{phase}").touch()
        (data / "fleet-release-parent").touch()
        for entry_path in data.glob("console-entry-*.json"):
            (
                data
                / entry_path.name.replace(
                    "console-entry-", "console-release-"
                ).removesuffix(".json")
            ).touch()
        if data.exists():
            shutil.copytree(data, root / "child-artifacts", dirs_exist_ok=True)
        await stack.aclose()
        patch.undo()
        result["physical_exit"] = [
            service._process.returncode
            for service in stack.services
            if service._process is not None
        ]
        (root / "fleet.json").write_text(json.dumps(result, indent=2))
    assert not blocked_attempts(), blocked_attempts()
    assert not take_violations()


def test_phone_lost_ack_queries_original_commands_without_replay(live_mobile_console):
    root, environment = live_mobile_console.root, live_mobile_console.environment
    with (root / "transport.log").open("w") as output:
        result = subprocess.run(
            [
                sys.executable,
                "-X",
                "faulthandler",
                "-m",
                __name__,
                str(root),
                "transport",
            ],
            cwd=REPOSITORY,
            env=environment,
            stdout=output,
            stderr=output,
            timeout=180,
            check=False,
        )
    assert result.returncode == 0, (root / "transport.log").read_text()[-10000:]
    evidence = json.loads((root / "transport.json").read_text())
    assert evidence["physical_exit"] == [0]
    assert evidence["provider_calls"] == 2


async def _run_transport(root):
    import base64
    import shutil
    import sqlite3

    from playwright.async_api import async_playwright, expect

    from Tests.Canvas.browser.canvas_live_harness import (
        chromium_executable,
        start_live_served_stack,
    )
    from Tests.network_guard import blocked_attempts
    from Tests.real_profile_guard import take_violations
    from tldw_chatbook.Backup_Recovery.profile_paths import canonical_console_paths
    from tldw_chatbook.config import get_atomic_config_snapshot

    patch = pytest.MonkeyPatch()
    token = secrets.token_urlsafe(24)
    stack = await start_live_served_stack(
        root,
        patch,
        access_token=token,
        child_module="Tests.Canvas.browser.canvas_live_chatbook_child",
    )
    case = LiveMobileConsole(root)
    case.stack = stack
    data = case.data
    result = {"cases": []}
    http_tasks = set()

    async def record_http(response):
        if "/api/v1/mobile/" in response.url:
            row = {
                "url": response.url,
                "status": response.status,
                "body": await response.text(),
            }
            with (root / "transport-http.jsonl").open("a") as output:
                output.write(json.dumps(row) + "\n")

    def observe_http(response):
        task = asyncio.create_task(record_http(response))
        http_tasks.add(task)
        task.add_done_callback(http_tasks.discard)

    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=chromium_executable(playwright.chromium), headless=True
            )
            context = await browser.new_context(
                ignore_https_errors=True, viewport={"width": 390, "height": 844}
            )
            await context.add_init_script("""window.__task11TransportInput = [];
                for (const type of ['pointerdown', 'pointerup', 'click']) document.addEventListener(type, event => {
                  if (event.target.id === 'new-chat-button' || event.target.id === 'tab-chats')
                    window.__task11TransportInput.push({type, target:event.target.id, disabled:event.target.disabled,
                      connection:document.querySelector('#connection-state')?.textContent});
                }, true);""")
            phone = case.phone = await context.new_page()
            phone.on("response", observe_http)
            login = await phone.request.post(
                stack.origin + "/auth/login",
                form={"access_token": token},
                headers={"Origin": stack.origin},
                max_redirects=0,
            )
            assert login.status == 303
            await phone.goto(stack.origin + "/m")
            for fault in ("before-admission", "after-acceptance", "after-effect"):
                await phone.locator("#tab-chats").click()
                await phone.get_by_role("button", name="New chat", exact=True).click()
                await expect(phone.locator("#chat-draft")).to_be_enabled(timeout=30000)
                assert len(stack.services) == 1
                pid = stack.services[0]._process.pid
                assert result.setdefault("pid", pid) == pid
                prompt = "EXACT_TRANSPORT_" + fault
                digest = hashlib.sha256(prompt.encode()).hexdigest()
                witness = data / ("console-entry-" + digest + ".json")
                release = data / ("console-release-" + digest)
                interrupted = asyncio.Event()
                attempts = []
                route_errors = []

                async def lose_response(
                    route,
                    _request,
                    *,
                    fault=fault,
                    attempts=attempts,
                    prompt=prompt,
                    witness=witness,
                    pid=pid,
                    release=release,
                    route_errors=route_errors,
                    interrupted=interrupted,
                ):
                    try:
                        body = route.request.post_data_json
                        ticket = json.loads(
                            base64.urlsafe_b64decode(body["ticket"].split(".")[0])
                        )
                        row = {
                            "fault": fault,
                            "command_id": ticket["command_id"],
                            "target": ticket["target"],
                            "payload": body["payload"],
                        }
                        attempts.append(row)
                        assert body["payload"]["text"] == prompt
                        if fault != "before-admission":
                            response = await route.fetch()
                            row["server_status"] = response.status
                            row["server_receipt"] = await response.json()
                            assert response.status in {200, 202}
                            row["provider_entry"] = json.loads(
                                await _wait_file(witness)
                            )
                            assert row["provider_entry"]["pid"] == pid
                            if fault == "after-effect":
                                release.touch()
                                await expect(
                                    phone.locator("#chat-transcript")
                                ).to_contain_text(
                                    "EXACT_REPLY:" + prompt, timeout=30000
                                )
                                row["effect_visible_before_response_loss"] = True
                        await route.abort("connectionreset")
                    except BaseException as error:
                        route_errors.append(repr(error))
                        raise
                    finally:
                        interrupted.set()

                await phone.route("**/api/v1/mobile/commands", lose_response)
                await phone.locator("#chat-draft").fill(prompt)
                await phone.get_by_role("button", name="Send", exact=True).click()
                await asyncio.wait_for(interrupted.wait(), 45)
                assert not route_errors, route_errors
                assert len(attempts) == 1
                row = attempts[0]
                response = await phone.request.get(
                    stack.origin + "/api/v1/mobile/commands/" + row["command_id"]
                )
                row["queried_status"] = response.status
                row["queried_receipt"] = await response.json()
                if fault == "before-admission":
                    assert (
                        response.status == 404
                        and row["queried_receipt"]["error"] == "unknown_command"
                    )
                    assert not witness.exists()
                    await expect(phone.locator("#chat-draft")).to_have_value(prompt)
                else:
                    assert (
                        response.status == 200
                        and row["queried_receipt"]["state"] == "accepted"
                    )
                    release.touch()
                    await expect(phone.locator("#chat-transcript")).to_contain_text(
                        "EXACT_REPLY:" + prompt, timeout=30000
                    )
                # Reconnection is read-only; it must not mint/replay this command.
                await phone.reload()
                await expect(phone.locator("#new-chat-button")).to_be_visible(
                    timeout=15000
                )
                assert len(attempts) == 1
                assert (
                    stack.services[0]._process.pid == pid
                    and stack.services[0]._process.returncode is None
                )
                await phone.unroute("**/api/v1/mobile/commands", lose_response)
                result["cases"].append(row)
            result["provider_calls"] = int((data / "console-gateway-calls").read_text())
            assert result["provider_calls"] == 2
            if http_tasks:
                await asyncio.gather(*http_tasks)
            await browser.close()
    except BaseException:
        import traceback

        (root / "transport-error.txt").write_text(traceback.format_exc())
        if case.phone is not None and not case.phone.is_closed():
            result["failed_dom"] = await case.phone.evaluate("""() => ({
                inputs:window.__task11TransportInput,
                connection:document.querySelector('#connection-state')?.textContent,
                status:document.querySelector('#status-line')?.textContent,
                target:document.querySelector('#target-label')?.textContent,
                draftDisabled:document.querySelector('#chat-draft')?.disabled,
                newDisabled:document.querySelector('#new-chat-button')?.disabled
            })""")
            await case.phone.screenshot(path=str(root / "transport-failed.png"))
        raise
    finally:
        for path in data.glob("console-entry-*.json"):
            (
                data
                / path.name.replace("console-entry-", "console-release-").removesuffix(
                    ".json"
                )
            ).touch()
        if data.exists():
            shutil.copytree(data, root / "child-artifacts", dirs_exist_ok=True)
        await stack.aclose()
        patch.undo()
        result["physical_exit"] = [
            service._process.returncode
            for service in stack.services
            if service._process is not None
        ]
        (root / "transport.json").write_text(json.dumps(result, indent=2))
    database = canonical_console_paths(get_atomic_config_snapshot().values)[0]
    with sqlite3.connect(
        database.as_uri() + "?mode=ro&immutable=1", uri=True
    ) as connection:
        rows = connection.execute(
            "SELECT role,content FROM messages ORDER BY rowid"
        ).fetchall()
    assert rows == [
        (role, text)
        for fault in ("after-acceptance", "after-effect")
        for role, text in (
            ("user", "EXACT_TRANSPORT_" + fault),
            ("assistant", "EXACT_REPLY:EXACT_TRANSPORT_" + fault),
        )
    ]
    (root / "transport-persisted.json").write_text(json.dumps(rows))
    assert not blocked_attempts(), blocked_attempts()
    assert not take_violations()


@pytest.mark.parametrize("mobile_off", [False, True])
def test_actual_security_projection_and_mobile_off(tmp_path, mobile_off):
    root, environment = _prepare_profile(tmp_path, mobile_surface=not mobile_off)
    environment = dict(environment, TLDW_SECURITY_FIXTURE="1")
    with (root / "security.log").open("w") as output:
        result = subprocess.run(
            [
                sys.executable,
                "-X",
                "faulthandler",
                "-m",
                __name__,
                str(root),
                "mobile-off" if mobile_off else "security",
            ],
            cwd=REPOSITORY,
            env=environment,
            stdout=output,
            stderr=output,
            timeout=180,
            check=False,
        )
    assert result.returncode == 0, (root / "security.log").read_text()[-10000:]
    assert json.loads((root / "security.json").read_text())["physical_exit"] == [0]


async def _run_security(root, mobile_off):
    import shutil

    from aiohttp import ClientSession, CookieJar, TCPConnector
    from playwright.async_api import async_playwright, expect
    from yarl import URL

    from Tests.Canvas.browser.canvas_live_harness import (
        _login_live_page,
        _wait_for_initial_console_ready,
        chromium_executable,
        start_live_served_stack,
    )
    from Tests.network_guard import blocked_attempts
    from Tests.real_profile_guard import take_violations
    from tldw_chatbook.Web_Server.served_console_hosts import StaleTerminalAttachment

    patch = pytest.MonkeyPatch()
    token = secrets.token_urlsafe(24)
    stack = await start_live_served_stack(
        root,
        patch,
        access_token=token,
        child_module="Tests.Canvas.browser.canvas_live_chatbook_child",
    )
    case = LiveMobileConsole(root)
    case.stack = stack
    data = case.data
    result = {"mobile_off": mobile_off}
    slow_session = slow_socket = None
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=chromium_executable(playwright.chromium), headless=True
            )
            desktop_context = await browser.new_context(
                ignore_https_errors=True, viewport={"width": 1280, "height": 900}
            )
            desktop = case.desktop = await desktop_context.new_page()
            await _login_live_page(desktop, origin=stack.origin, access_token=token)
            await _wait_for_initial_console_ready(desktop)
            entry, terminal = await case.start_gated_chat("EXACT_SECURITY_HELD_RUN")
            result.update(entry=entry, terminal=terminal)
            service = stack.services[0]
            pid = service._process.pid
            stores, _, _ = stack.server._canonical_console_selection()
            host = stack.server._served_console_hosts.get_existing(stores.identity)
            assert host is not None
            old_attachment = host.current_attachment
            buffer = stack.server._canvas_control_broker.console_events(
                service.app_service_id
            )
            for index in range(80):
                buffer.note_changed(
                    buffer.publish("console.changed", {"qualification": index})
                )
            before = buffer.read(0)
            assert before.reset_required and len(before.events) <= 64
            result["before_first_phone_overflow"] = {
                "sequence": before.sequence,
                "retained": len(before.events),
                "reset_required": before.reset_required,
            }
            if mobile_off:
                for path in (
                    "/m",
                    "/api/v1/mobile/console/snapshot?protocol_version=1",
                    "/api/v1/mobile/stream",
                ):
                    response = await desktop.request.get(stack.origin + path)
                    assert response.status == 404, (path, response.status)
                assert (
                    service._process.pid == pid and service._process.returncode is None
                )
                assert host.current_attachment is old_attachment
                result["mobile_routes_absent_owner_retained"] = True
            else:
                phone_context = await browser.new_context(
                    ignore_https_errors=True, viewport={"width": 390, "height": 844}
                )
                phone = case.phone = await phone_context.new_page()
                login = await phone.request.post(
                    stack.origin + "/auth/login",
                    form={"access_token": token},
                    headers={"Origin": stack.origin},
                    max_redirects=0,
                )
                assert login.status == 303
                await phone.goto(stack.origin + "/m")
                await phone.locator(
                    f'#chat-list button[data-session-id="{terminal["session_id"]}"]'
                ).click()
                await expect(phone.locator("#chat-draft")).to_be_enabled(timeout=15000)
                snapshot = await case.snapshot()
                chat = next(
                    row
                    for row in snapshot["chats"]
                    if row["session_id"] == terminal["session_id"]
                )
                target = {
                    key: chat[key]
                    for key in (
                        "server_epoch",
                        "console_id",
                        "session_id",
                        "incarnation_id",
                        "run_id",
                        "turn_id",
                    )
                }
                result["native_target"] = target
                await desktop.locator("#terminal .xterm-helper-textarea").press("F8")
                native_note = json.loads(
                    await _wait_file(data / "fixture-native-note.json")
                )
                assert native_note["core_path"] == terminal["core_path"]
                note_page = await phone.request.get(
                    stack.origin + "/api/v1/mobile/notes"
                )
                assert note_page.status == 200
                assert any(
                    row["id"] == native_note["id"]
                    and row["title"] == "EXACT_CANONICAL_LIBRARY_NOTE"
                    for row in (await note_page.json())["items"]
                )
                note_detail = await phone.request.get(
                    stack.origin + "/api/v1/mobile/notes/" + native_note["id"]
                )
                assert note_detail.status == 200
                literal_body = (
                    'EXACT_CANONICAL_LIBRARY_BODY<img src=x onerror="window.__task11Executed=1">'
                    "<script>window.__task11Executed=2</script>"
                )
                assert (await note_detail.json())["note"]["content"] == literal_body
                await phone.locator("#back-button").click()
                await phone.locator("#tab-library").click()
                await phone.locator(
                    f'#notes-list li[data-note-id="{native_note["id"]}"]'
                ).click()
                await expect(phone.locator("#note-content")).to_have_text(literal_body)
                assert (
                    await phone.locator(
                        "#note-content script, #note-content img"
                    ).count()
                    == 0
                )
                assert (
                    await phone.evaluate("typeof window.__task11Executed")
                    == "undefined"
                )
                result["canonical_library_note"] = native_note
                result["library_content_inert"] = True
                await phone.locator("#back-button").click()
                await phone.locator("#tab-chats").click()
                await phone.locator(
                    f'#chat-list button[data-session-id="{terminal["session_id"]}"]'
                ).click()
                await expect(phone.locator("#chat-draft")).to_be_enabled(timeout=15000)
                csrf = await phone.locator('meta[name="chatbook-csrf"]').get_attribute(
                    "content"
                )
                headers = {"Origin": stack.origin, "X-Chatbook-CSRF": csrf}
                endpoint = stack.origin + "/api/v1/mobile/commands/tickets"
                response = await phone.request.post(
                    endpoint,
                    headers=headers,
                    data={
                        "protocol_version": 1,
                        "action": "steer",
                        "target": dict(target, server_epoch="stale-epoch"),
                        "payload": {"text": "MUST_NOT_STEER"},
                    },
                )
                assert response.status == 409, await response.text()
                result["stale_epoch_status"] = response.status
                payload = {"text": "MUST_NOT_EXPIRE_INTO_EFFECT"}
                response = await phone.request.post(
                    endpoint,
                    headers=headers,
                    data={
                        "protocol_version": 1,
                        "action": "steer",
                        "target": target,
                        "payload": payload,
                    },
                )
                assert response.status == 200, await response.text()
                ticket = await response.json()
                (data / "fixture-command-clock-offset-ns").write_text(
                    str(301 * 1_000_000_000)
                )
                response = await phone.request.post(
                    stack.origin + "/api/v1/mobile/commands",
                    headers=headers,
                    data={
                        "protocol_version": 1,
                        "ticket": ticket["ticket"],
                        "payload": payload,
                    },
                )
                assert response.status == 410, await response.text()
                result["expired_ticket"] = {
                    "command_id": ticket["command_id"],
                    "status": response.status,
                }
                response = await phone.request.get(
                    stack.origin + "/api/v1/mobile/commands/" + ticket["command_id"]
                )
                assert response.status == 404
                # A second authenticated terminal takes over through the actual
                # browser path. Old attachment calls cannot reach the child.
                successor_context = await browser.new_context(
                    ignore_https_errors=True, viewport={"width": 1000, "height": 800}
                )
                successor = await successor_context.new_page()
                await _login_live_page(
                    successor, origin=stack.origin, access_token=token
                )
                await _wait_for_initial_console_ready(successor)
                current_attachment = host.current_attachment
                assert current_attachment is not old_attachment
                with pytest.raises(StaleTerminalAttachment):
                    await old_attachment.send_bytes(b"MUST_NOT_ENTER\r")
                with pytest.raises(StaleTerminalAttachment):
                    await old_attachment.resize(1, 1)
                await old_attachment.detach()
                assert host.current_attachment is current_attachment
                state = data / "canvas-live-terminal-state"
                state.unlink(missing_ok=True)
                await successor.locator("#terminal .xterm-helper-textarea").press("F11")
                after = json.loads(await _wait_file(state))
                assert (
                    after["input_thread_object"] == terminal["input_thread_object"]
                    and after["session_id"] == terminal["session_id"]
                    and not after["draft"]
                )
                result["takeover_terminal"] = after
                # Invoke the real retained unsupported binding-review producer.
                # No folder is selected, and the phone cannot grant authority.
                await successor.locator("#terminal .xterm-helper-textarea").press("F9")
                await expect(phone.locator("#chat-decisions")).to_contain_text(
                    "Open terminal", timeout=15000
                )
                assert (
                    await phone.locator("#chat-decisions")
                    .get_by_role("button", name="Submit decision")
                    .count()
                    == 0
                )
                response = await phone.request.get(
                    stack.origin + "/api/v1/mobile/console/snapshot",
                    params={
                        "protocol_version": "1",
                        "target": json.dumps(
                            {
                                k: v
                                for k, v in target.items()
                                if k not in {"run_id", "turn_id"}
                            }
                        ),
                    },
                )
                assert response.status == 200
                unsupported = next(
                    row
                    for row in (await response.json())["decisions"]
                    if row["kind"] == "project_instruction_binding"
                )
                assert not unsupported["supported"] and not unsupported["answerable"]
                await phone.reload()
                await phone.locator(
                    f'#chat-list button[data-session-id="{terminal["session_id"]}"]'
                ).click()
                await expect(phone.locator("#chat-decisions")).to_contain_text(
                    "Open terminal", timeout=15000
                )
                result["unsupported_pending"] = unsupported
                # Leave one real authenticated WebSocket unread while another
                # phone reconnects and the shared projection loses history.
                slow_session = ClientSession(
                    cookie_jar=CookieJar(unsafe=True), connector=TCPConnector(ssl=False)
                )
                slow_session.cookie_jar.update_cookies(
                    {
                        row["name"]: row["value"]
                        for row in await phone_context.cookies()
                    },
                    response_url=URL(stack.origin),
                )
                slow_socket = await slow_session.ws_connect(
                    stack.origin + "/api/v1/mobile/stream",
                    protocols=("chatbook-v1", "csrf." + csrf),
                    headers={"Origin": stack.origin},
                )
                await phone_context.set_offline(True)
                for index in range(160):
                    buffer.note_changed(
                        buffer.publish("console.changed", {"qualification_slow": index})
                    )
                assert (
                    buffer.read(0).reset_required and len(buffer.read(0).events) <= 64
                )
                await phone_context.set_offline(False)
                await expect(phone.locator("#connection-state")).to_contain_text(
                    "Connected", timeout=20000
                )
                converged = await case.snapshot()
                assert converged["console_id"] == target["console_id"]
                assert (
                    service._process.pid == pid and service._process.returncode is None
                )
                result["overflow_converged"] = converged
                result[
                    "slow_subscription_unread_during_overflow"
                ] = not slow_socket.closed
                assert not slow_socket.closed
                await slow_socket.close()
                await slow_session.close()
                # Revoke only this phone using the shipped HTTP session route.
                csrf = await phone.locator('meta[name="chatbook-csrf"]').get_attribute(
                    "content"
                )
                response = await phone.request.post(
                    stack.origin + "/auth/logout",
                    headers={"Origin": stack.origin, "X-Chatbook-CSRF": csrf},
                    max_redirects=0,
                )
                assert response.status == 303
                response = await phone.request.get(
                    stack.origin + "/api/v1/mobile/console/snapshot?protocol_version=1"
                )
                assert response.status == 401
                result["revoked_phone_status"] = response.status
                assert (
                    host.current_attachment is current_attachment
                    and service._process.returncode is None
                )
                # Native session expiry is exercised with its own expiry field,
                # leaving application and command clocks unchanged.
                login = await phone.request.post(
                    stack.origin + "/auth/login",
                    form={"access_token": token},
                    headers={"Origin": stack.origin},
                    max_redirects=0,
                )
                assert login.status == 303
                await phone.goto(stack.origin + "/m")
                csrf = await phone.locator('meta[name="chatbook-csrf"]').get_attribute(
                    "content"
                )
                session = next(
                    row
                    for row in stack.server._web_auth.sessions_snapshot()
                    if row.csrf_token == csrf
                )
                session.absolute_expires_at = stack.server._web_auth._clock() - 1
                response = await phone.request.get(
                    stack.origin + "/api/v1/mobile/console/snapshot?protocol_version=1"
                )
                assert response.status == 401
                result["expired_phone_status"] = response.status
                assert service._process.returncode is None
            assert int((data / "console-gateway-calls").read_text()) == 1
            result["provider_calls_before_explicit_release"] = 1
            await case.release_agent()
            await browser.close()
    except BaseException:
        import traceback

        (root / "security-error.txt").write_text(traceback.format_exc())
        raise
    finally:
        if slow_socket is not None:
            await slow_socket.close()
        if slow_session is not None:
            await slow_session.close()
        for path in data.glob("console-entry-*.json"):
            (
                data
                / path.name.replace("console-entry-", "console-release-").removesuffix(
                    ".json"
                )
            ).touch()
        if data.exists():
            shutil.copytree(data, root / "child-artifacts", dirs_exist_ok=True)
        await stack.aclose()
        patch.undo()
        result["physical_exit"] = [
            service._process.returncode
            for service in stack.services
            if service._process is not None
        ]
        (root / "security.json").write_text(json.dumps(result, indent=2))
    assert not blocked_attempts(), blocked_attempts()
    assert not take_violations()


_REPLACEMENT_PROBE = r"""
import hashlib,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
import psutil
from Tests.real_profile_guard import install as guard,take_violations
guard()
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionBusy
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Backup_Recovery.profile_paths import canonical_console_paths
from tldw_chatbook.config import get_atomic_config_snapshot
root=Path(sys.argv[1]);paths=canonical_console_paths(get_atomic_config_snapshot().values)
runtime=ConsoleRuntime(SimpleNamespace(chachanotes_db=SimpleNamespace(db_path=paths[0])))
(root/'replacement-ready').touch()
deadline=time.monotonic()+90
while not (root/'replacement-go').exists():
 assert time.monotonic()<deadline
 time.sleep(.01)
record=json.loads((root/'lifecycle-ready.json').read_text())
assert Path(record['run_path'])==paths[2]
def alive():
 try:
  p=psutil.Process(record['child_pid'])
  return p.create_time()==record['child_created'] and p.status()!=psutil.STATUS_ZOMBIE
 except psutil.NoSuchProcess:return False
assert alive()
reader=AgentRunsDB.open_read_only(paths[2])
def rows():
 with reader.connection() as connection:
  return {name:[list(row) for row in connection.execute('SELECT * FROM '+name+' ORDER BY rowid')] for name in ('agent_runs','console_command_receipts')}
before=rows()
assert before['agent_runs'] and before['console_command_receipts']
counts={'constructor':0,'reconcile':0,'store_factory':0,'provider_factory':0}
original_init=AgentRunsDB.__init__;original_reconcile=AgentRunsDB.reconcile_orphaned_runs
def observed_init(self,*a,**k):
 counts['constructor']+=1
 return original_init(self,*a,**k)
def observed_reconcile(self,*a,**k):
 counts['reconcile']+=1
 return original_reconcile(self,*a,**k)
def factory(name):
 counts[name]+=1
 raise AssertionError('Replacement reached '+name)
AgentRunsDB.__init__=observed_init;AgentRunsDB.reconcile_orphaned_runs=observed_reconcile
gateway=(root/'test_data/console-gateway-calls').read_text()
try:
 try:
  runtime.ensure_agent_bridge(store_factory=lambda:factory('store_factory'),provider_gateway_factory=lambda:factory('provider_factory'))
 except ConsoleExecutionBusy:
  refused=True
 else:raise AssertionError('Replacement execution admitted while old child alive')
 assert alive()
 after=rows()
 assert before==after
 assert (root/'test_data/console-gateway-calls').read_text()==gateway
 assert all(value==0 for value in counts.values()),counts
 assert runtime._agent_runs_db is None and runtime._agent_bridge is None and runtime._activity_receipts is None
 result={'refused':refused,'old_child_live_before_and_after':True,'child_pid':record['child_pid'],'child_created':record['child_created'],'counts':counts,'run_and_receipt_rows_unchanged':True,'before_sha256':hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),'after_sha256':hashlib.sha256(json.dumps(after,sort_keys=True).encode()).hexdigest(),'gateway_calls':int(gateway)}
 (root/'replacement-result.json').write_text(json.dumps(result,indent=2))
finally:
 AgentRunsDB.__init__=original_init;AgentRunsDB.reconcile_orphaned_runs=original_reconcile
 reader.close()
assert not blocked_attempts() and not take_violations()
"""


def _same_owned_process(record):
    import psutil

    try:
        process = psutil.Process(record["child_pid"])
        return (
            process.create_time() == record["child_created"]
            and process.status() != psutil.STATUS_ZOMBIE
        )
    except psutil.NoSuchProcess:
        return False


@pytest.mark.parametrize(
    "mode", ["clean-stop", "control-eof", "child-crash", "parent-death"]
)
def test_owned_actual_served_process_lifecycle(tmp_path, mode):
    import time

    from tldw_chatbook.Chat.console_execution_owner import (
        ConsoleExecutionBusy,
        ConsoleExecutionOwner,
    )

    root, environment = _prepare_profile(tmp_path)
    environment["TLDW_LIFECYCLE_BLOCK"] = "1"
    replacement = None
    replacement_output = None
    if mode == "parent-death":
        replacement_output = (root / "replacement.log").open("w")
        replacement = subprocess.Popen(
            [sys.executable, "-c", _REPLACEMENT_PROBE, str(root)],
            cwd=REPOSITORY,
            env=environment,
            stdout=replacement_output,
            stderr=replacement_output,
        )
        deadline = time.monotonic() + 30
        while (
            not (root / "replacement-ready").exists()
            and replacement.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        if not (root / "replacement-ready").exists():
            if replacement.poll() is None:
                replacement.kill()
            replacement.wait(timeout=10)
            replacement_output.close()
            raise AssertionError((root / "replacement.log").read_text())
    record = None
    with (root / "lifecycle.log").open("w") as output:
        parent = subprocess.Popen(
            [sys.executable, "-X", "faulthandler", "-m", __name__, str(root), mode],
            cwd=REPOSITORY,
            env=environment,
            stdout=output,
            stderr=output,
        )
        try:
            deadline = time.monotonic() + 60
            ready = root / "lifecycle-ready.json"
            while (
                not ready.exists()
                and parent.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)
            assert ready.exists(), (root / "lifecycle.log").read_text()[-7000:]
            record = json.loads(ready.read_text())
            assert record["parent_pid"] == parent.pid and _same_owned_process(record)
            if mode == "parent-death":
                parent.kill()  # Exact Popen-owned standalone server, never pytest/shell.
                assert parent.wait(timeout=10) == -9
                assert _same_owned_process(record)
                with pytest.raises(ConsoleExecutionBusy):
                    ConsoleExecutionOwner.acquire(Path(record["run_path"]))
                record["survivor_fence_refused"] = True
                (root / "replacement-go").touch()
                assert replacement.wait(timeout=10) == 0, (
                    root / "replacement.log"
                ).read_text()
                record["native_replacement"] = json.loads(
                    (root / "replacement-result.json").read_text()
                )
                deadline = time.monotonic() + 20
                while _same_owned_process(record) and time.monotonic() < deadline:
                    time.sleep(0.02)
                assert not _same_owned_process(record), (
                    "owned child did not retire after parent EOF"
                )
                with ConsoleExecutionOwner.acquire(Path(record["run_path"])):
                    pass
                record.update(
                    parent_exit=-9, child_physically_absent=True, fence_reacquired=True
                )
                (root / "lifecycle-result.json").write_text(
                    json.dumps(record, indent=2)
                )
            else:
                (root / "lifecycle-trigger").touch()
                assert parent.wait(timeout=40) == 0, (
                    root / "lifecycle.log"
                ).read_text()[-7000:]
                result = json.loads((root / "lifecycle-result.json").read_text())
                assert result["child_exit"] == (-9 if mode == "child-crash" else 0)
                assert not _same_owned_process(record)
                with ConsoleExecutionOwner.acquire(Path(record["run_path"])):
                    pass
        finally:
            if replacement is not None:
                if replacement.poll() is None:
                    replacement.kill()
                replacement.wait(timeout=10)
                replacement_output.close()
            # The standalone parent records its own actual AppService process as
            # soon as it exists, before command readiness. Early failures still
            # have exact PID/create-time custody for the fallback teardown.
            owned = root / "lifecycle-owned.json"
            if record is None and owned.exists():
                record = json.loads(owned.read_text())
                assert record["parent_pid"] == parent.pid
            if parent.poll() is None:
                parent.kill()
            parent.wait(timeout=10)
            if record and _same_owned_process(record):
                import psutil

                psutil.Process(record["child_pid"]).kill()
                deadline = time.monotonic() + 10
                while _same_owned_process(record) and time.monotonic() < deadline:
                    time.sleep(0.02)
                assert not _same_owned_process(record), (
                    "owned fixture child survived teardown"
                )


async def _run_lifecycle(root, mode):
    import re
    import shutil

    import psutil
    from aiohttp import ClientSession, CookieJar, TCPConnector

    from Tests.Canvas.browser.canvas_live_harness import start_live_served_stack
    from Tests.network_guard import blocked_attempts
    from Tests.real_profile_guard import take_violations
    from tldw_chatbook.Backup_Recovery import mobile_consolidation
    from tldw_chatbook.Backup_Recovery.profile_paths import canonical_console_paths
    from tldw_chatbook.config import get_atomic_config_snapshot

    original_gate = mobile_consolidation.require_mobile_cutover

    def observed_gate(*args, **kwargs):
        try:
            return original_gate(*args, **kwargs)
        except BaseException:
            import traceback

            with (root / "cutover-gate-errors.txt").open("a") as output:
                output.write(traceback.format_exc())
            raise

    mobile_consolidation.require_mobile_cutover = observed_gate
    patch = pytest.MonkeyPatch()
    token = secrets.token_urlsafe(24)
    stack = await start_live_served_stack(
        root,
        patch,
        access_token=token,
        child_module="Tests.Canvas.browser.canvas_live_chatbook_child",
    )
    pending = None
    result = {"case": mode, "parent_pid": os.getpid()}
    source_files = (
        "Tests/Web_Server/test_mobile_served_console_browser.py",
        "Tests/Canvas/browser/canvas_live_harness.py",
        "Tests/Canvas/browser/canvas_live_chatbook_child.py",
        "tldw_chatbook/Backup_Recovery/mobile_consolidation_recovery.py",
        "tldw_chatbook/Backup_Recovery/storage_admission.py",
        "tldw_chatbook/DB/private_sqlite.py",
        "tldw_chatbook/Web_Server/mobile_service.py",
        "tldw_chatbook/Web_Server/serve.py",
    )

    def source_epoch():
        return {
            name: hashlib.sha256((REPOSITORY / name).read_bytes()).hexdigest()
            for name in source_files
        }

    (root / "source-before.json").write_text(json.dumps(source_epoch(), indent=2))
    data = root / "test_data"

    async def record_owned_child():
        while True:
            for service in stack.services:
                process = service._process
                if process is not None:
                    record = {
                        "parent_pid": os.getpid(),
                        "child_pid": process.pid,
                        "child_created": psutil.Process(process.pid).create_time(),
                    }
                    temporary = root / "lifecycle-owned.pending"
                    temporary.write_text(json.dumps(record))
                    temporary.replace(root / "lifecycle-owned.json")
                    return
            await asyncio.sleep(0.01)

    owned_observer = asyncio.create_task(record_owned_child())
    try:
        async with ClientSession(
            cookie_jar=CookieJar(unsafe=True), connector=TCPConnector(ssl=False)
        ) as client:
            async with client.post(
                stack.origin + "/auth/login",
                data={"access_token": token},
                headers={"Origin": stack.origin},
                allow_redirects=False,
            ) as response:
                assert response.status == 303
            async with client.get(stack.origin + "/m") as response:
                csrf = re.search(
                    r'<meta name="chatbook-csrf" content="([^"]*)">',
                    await response.text(),
                ).group(1)
            headers = {"Origin": stack.origin, "X-Chatbook-CSRF": csrf}

            async def post(path, body):
                async with client.post(
                    stack.origin + "/api/v1/mobile/" + path,
                    json={"protocol_version": 1, **body},
                    headers=headers,
                ) as response:
                    value = await response.json()
                    assert response.status in (200, 202), (response.status, value)
                    return value

            host = (await post("console/host", {}))["host"]
            ready = json.loads(await _wait_file(data / "console-app-ready.json"))
            assert ready["ui_ready"] and ready["pid"] == stack.services[0]._process.pid
            async with asyncio.timeout(30):
                while True:
                    async with client.get(
                        stack.origin
                        + "/api/v1/mobile/console/snapshot?protocol_version=1"
                    ) as response:
                        current = await response.json()
                    if current.get("execution_admitted"):
                        break
                    await asyncio.sleep(0.05)
            ticket = await post(
                "commands/tickets",
                {
                    "action": "create",
                    "target": host,
                    "payload": {"workspace_id": "global"},
                },
            )
            receipt = await post(
                "commands",
                {"ticket": ticket["ticket"], "payload": {"workspace_id": "global"}},
            )
            async with asyncio.timeout(20):
                while True:
                    async with client.get(
                        stack.origin
                        + "/api/v1/mobile/console/snapshot?protocol_version=1"
                    ) as response:
                        current = await response.json()
                    chat = next(
                        (
                            row
                            for row in current.get("chats", [])
                            if row["session_id"] == receipt["native_id"]
                        ),
                        None,
                    )
                    if chat:
                        break
                    await asyncio.sleep(0.05)
            target = {
                key: chat[key]
                for key in (
                    "server_epoch",
                    "console_id",
                    "session_id",
                    "incarnation_id",
                )
            }
            payload = {
                "text": "EXACT_LIFECYCLE_PROMPT",
                "expected_revisions": chat["expected_revisions"],
            }
            ticket = await post(
                "commands/tickets",
                {"action": "send", "target": target, "payload": payload},
            )
            pending = asyncio.create_task(
                post("commands", {"ticket": ticket["ticket"], "payload": payload})
            )
            entry = json.loads(
                await _wait_file(
                    data
                    / (
                        "console-entry-"
                        + hashlib.sha256(b"EXACT_LIFECYCLE_PROMPT").hexdigest()
                        + ".json"
                    )
                )
            )
            process = stack.services[0]._process
            assert entry["pid"] == process.pid
            result.update(
                child_pid=process.pid,
                child_created=psutil.Process(process.pid).create_time(),
                run_path=str(
                    canonical_console_paths(get_atomic_config_snapshot().values)[2]
                ),
                target=target,
            )
            (root / "lifecycle-ready.json").write_text(json.dumps(result, indent=2))
            await _wait_file(root / "lifecycle-trigger", timeout=60)
            if mode == "child-crash":
                process.kill()
                await asyncio.wait_for(process.wait(), 15)
            elif mode == "control-eof":
                await stack.server._canvas_control_broker.aclose()
                await asyncio.wait_for(process.wait(), 15)
            elif mode == "clean-stop":
                await stack.aclose()
            else:
                raise AssertionError("parent-death must be signaled externally")
            result["child_exit"] = process.returncode
    except BaseException:
        import traceback

        (root / "lifecycle-error.txt").write_text(traceback.format_exc())
        raise
    finally:
        owned_observer.cancel()
        await asyncio.gather(owned_observer, return_exceptions=True)
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        if data.exists():
            shutil.copytree(data, root / "child-artifacts", dirs_exist_ok=True)
        if mode != "clean-stop" or result.get("child_exit") != 0:
            await stack.aclose()
        result["physical_exits"] = [
            service._process.returncode
            for service in stack.services
            if service._process is not None
        ]
        result["owned_child_pids"] = [
            service._process.pid
            for service in stack.services
            if service._process is not None
        ]
        (root / "lifecycle-result.json").write_text(json.dumps(result, indent=2))
        (root / "source-after.json").write_text(json.dumps(source_epoch(), indent=2))
        mobile_consolidation.require_mobile_cutover = original_gate
        patch.undo()
    assert not blocked_attempts(), blocked_attempts()
    assert not take_violations()


if __name__ == "__main__":
    from Tests.network_guard import NetworkMode, install, set_mode
    from Tests.real_profile_guard import install as install_profile_guard

    install_profile_guard()
    install()
    set_mode(NetworkMode.LOOPBACK_ONLY)
    root = Path(sys.argv[1])
    asyncio.run(
        _run_security(root, sys.argv[2] == "mobile-off")
        if len(sys.argv) > 2 and sys.argv[2] in {"security", "mobile-off"}
        else _run_transport(root)
        if len(sys.argv) > 2 and sys.argv[2] == "transport"
        else _run_fleet(root)
        if len(sys.argv) > 2 and sys.argv[2] == "fleet"
        else _run_cold_phone(root)
        if len(sys.argv) > 2 and sys.argv[2] == "cold"
        else _run_lifecycle(root, sys.argv[2])
        if len(sys.argv) > 2
        else _run_retention(root)
    )
