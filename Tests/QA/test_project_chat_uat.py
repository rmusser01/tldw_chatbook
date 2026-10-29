"""TASK-37270: actual-app project creation, binding, chat, and file evidence UAT.

Only model HTTP responses are scripted. Run with the CI dependency set; each
variant starts in its own interpreter/profile via private_profile_test.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
import traceback
import uuid

import pytest
from textual.events import Paste

from Tests.private_profile import private_profile_test
from Tests.uat_project_chat_support import (
    MODEL,
    VARIANTS,
    assert_project_evidence,
    assert_project_ownership,
    disclosed_schemas,
    only_owned_server,
    project_chat_server,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.loopback_network]


@pytest.fixture
def uat_environment(monkeypatch, tmp_path):
    """Remove credentials/overrides before the parent launches its fresh child."""
    keep = {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "SYSTEMDRIVE",
        "LANG",
        "LC_ALL",
        "TERM",
        "COLORTERM",
        "HOME",
        "USERPROFILE",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TLDW_CONFIG_PATH",
        "TLDW_TEST_MODE",
        "PYTHON_KEYRING_BACKEND",
        "HF_HUB_OFFLINE",
        "PYTHONUTF8",
        "PYTHONIOENCODING",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUNBUFFERED",
        "TLDW_UAT_ARTIFACTS",
        "TLDW_UAT_NEGATIVE_CONTROL",
    }
    for key in list(os.environ):
        if key not in keep and not key.startswith(
            ("PYTEST_", "TLDW_TEST_", "COVERAGE_", "COV_CORE_")
        ):
            monkeypatch.delenv(key)
    private_temp = tmp_path / "temp"
    private_temp.mkdir(mode=0o700)
    for key in ("TEMP", "TMP", "TMPDIR"):
        monkeypatch.setenv(key, str(private_temp))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")


async def wait_for(pilot, condition, phase, server, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assert not server.errors, f"{phase}: model protocol failed: {server.errors}"
        result = condition()
        if result:
            return result
        await pilot.pause(0.05)
    raise AssertionError(f"{phase}: condition not reached in {timeout}s")


async def click(pilot, selector, server):
    def actionable():
        matches = list(pilot.app.screen.query(selector))
        if not matches:
            return None
        widget = matches[0]
        return (
            widget
            if widget.is_mounted
            and widget.display
            and not widget.disabled
            and widget.region
            else None
        )

    widget = await wait_for(pilot, actionable, f"actionable {selector}", server)
    widget.scroll_visible(animate=False, immediate=True)
    await pilot.pause()
    # Re-query after layout/recomposition, never use a detached pre-await button.
    widget = await wait_for(pilot, actionable, f"settled {selector}", server)
    assert await pilot.click(widget), f"click was not delivered: {selector}"


async def fill(pilot, selector, text, server):
    await click(pilot, selector, server)
    await pilot.press("home", "shift+end", "backspace")
    assert pilot.app.focused.post_message(
        Paste(text)
    ), f"paste not accepted: {selector}"
    await wait_for(
        pilot,
        lambda: pilot.app.screen.query_one(selector).value == text,
        f"pasted {selector}",
        server,
    )


def painted_text(screen, region=None):
    lines = [strip.text for strip in screen._compositor.render_strips()]
    if region is None:
        return "\n".join(lines)
    clipped = region.intersection(screen.region)
    return "\n".join(
        line[clipped.x : clipped.right] for line in lines[clipped.y : clipped.bottom]
    )


def preparation_stacks():
    """Capture suspended app coroutines as well as the native worker threads."""
    names = {thread.ident: thread.name for thread in threading.enumerate()}
    parts = []
    for ident, frame in sys._current_frames().items():
        if ident == threading.main_thread().ident:
            continue
        parts.append(f"\n--- thread {names.get(ident, ident)}\n")
        parts.extend(traceback.format_stack(frame, limit=20))
    for task in asyncio.all_tasks():
        chain = [f"\n--- asyncio task {task.get_name()}\n"]
        awaiting = task.get_coro()
        seen = set()
        while awaiting is not None and id(awaiting) not in seen:
            seen.add(id(awaiting))
            frame = getattr(awaiting, "cr_frame", None) or getattr(
                awaiting, "gi_frame", None
            )
            if frame is not None:
                chain.extend(traceback.format_stack(frame, limit=1))
            awaiting = getattr(awaiting, "cr_await", None) or getattr(
                awaiting, "gi_yieldfrom", None
            )
        text = "".join(chain)
        if "tldw_chatbook" in text:
            parts.append(text)
    return "".join(parts)[:100000]


@pytest.mark.asyncio
# Cold Windows storage admission is substantially slower than the Linux CI lane.
@pytest.mark.timeout(600 if os.name == "nt" else 240)
@pytest.mark.parametrize("variant", VARIANTS)
@private_profile_test
async def test_project_chat_uat(
    variant, request, tmp_path, monkeypatch, uat_environment
):
    code = "uat-" + uuid.uuid4().hex
    project = tmp_path / "sample project \u00e9"
    project.mkdir()
    readme = project / "README.md"
    readme.write_text(
        "This application is a work/rest timer for the terminal.\n"
        "It stores completed sessions in a local SQLite database.\n"
        f"Verification code: {code}\n",
        encoding="utf-8",
    )
    (project / "AGENTS.md").write_text(
        "Consult README.md for the project description.\n", encoding="utf-8"
    )
    (project / "pyproject.toml").write_text(
        '[project]\nname = "sample-project"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    (project / "main.py").write_text(
        '"""Synthetic work/rest timer fixture."""\n', encoding="utf-8"
    )
    artifacts = (
        Path(os.environ.get("TLDW_UAT_ARTIFACTS", str(tmp_path / "artifacts")))
        / f"{variant}-{uuid.uuid4().hex[:8]}"
    )
    artifacts.mkdir(parents=True)
    record = {
        "case": "UAT-02",
        "variant": variant,
        "status": "running",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "phases": [],
        "substitutions": [
            "scripted model HTTP responses",
            "preconfigured provider/onboarding/read permissions",
            "root pytest fixtures block unrelated default endpoint discovery",
        ],
    }
    start = time.monotonic()
    app = None

    def phase(name):
        record["phase"] = name
        record["phases"].append(
            {"name": name, "seconds": round(time.monotonic() - start, 3)}
        )
        (artifacts / "receipt.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )

    phase("configure")
    try:
        with project_chat_server(variant) as server, only_owned_server(
            server.port
        ) as blocked:
            # Product imports occur after the private-profile child has selected its
            # config source. No general UI app factory or borrowed Host is used.
            from tldw_chatbook import config
            from tldw_chatbook.Chat.local_reasoning import reasoning_override_key
            from tldw_chatbook.Utils import token_counter

            console_settings = {
                "local_tools_enabled": True,
                "compaction_mode": "off",
                "agent_max_model_turns": 12,
                "agent_max_wall_seconds": 180 if os.name == "nt" else 90,
            }
            if variant == "native":
                key = reasoning_override_key(
                    "llama_cpp", server.url + "/v1/chat/completions", MODEL
                )
                console_settings["reasoning_native_tool_overrides"] = {key: True}
            assert config.save_settings_to_cli_config(
                {
                    "general": {"default_tab": "chat"},
                    "first_run": {"setup_started": True, "setup_completed": True},
                    "splash_screen": {"enabled": False},
                    "model_catalog": {"auto_refresh_enabled": False},
                    "chat_defaults": {
                        "provider": "llama_cpp",
                        "model": MODEL,
                        "rag_auto_retrieve_on_send": False,
                    },
                    "api_settings.llama_cpp": {
                        "api_url": server.url + "/v1/chat/completions",
                        "model": MODEL,
                        "api_key": "",
                        "max_tokens": 256,
                    },
                    "console": console_settings,
                }
            )
            from tldw_chatbook.app import TldwCli
            from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

            record["tokenizer_available"] = token_counter.TIKTOKEN_AVAILABLE
            phase("startup")
            app = TldwCli()
            phase("app-constructed")
            app.unified_mcp_service.set_global_default("allow")
            async with app.run_test(size=(180, 55)) as pilot:
                try:
                    console = await wait_for(
                        pilot,
                        lambda: app.screen
                        if isinstance(app.screen, ChatScreen) and app._ui_ready
                        else None,
                        "Console startup",
                        server,
                    )
                    phase("create-workspace")
                    if not console.query_one(
                        "#console-rail-section-body-workspace"
                    ).display:
                        await click(
                            pilot, "#console-rail-section-toggle-workspace", server
                        )
                    await click(pilot, "#console-new-workspace", server)
                    await fill(pilot, "#workspace-create-name", "UAT project", server)
                    await fill(
                        pilot, "#workspace-create-folder-path", str(project), server
                    )
                    await click(pilot, "#workspace-create-folder-add", server)
                    await wait_for(
                        pilot,
                        lambda: bool(app.screen.query(".workspace-create-folder-item")),
                        "folder added to form",
                        server,
                    )
                    interview = app.screen.query_one(
                        "#workspace-create-profile-interview"
                    )
                    if interview.value:
                        await click(
                            pilot, "#workspace-create-profile-interview", server
                        )
                    await click(pilot, "#workspace-create-confirm", server)
                    await wait_for(
                        pilot,
                        lambda: isinstance(app.screen, ChatScreen),
                        "workspace dialog closed",
                        server,
                    )
                    workspace = await wait_for(
                        pilot,
                        lambda: app.workspace_registry_service.get_active_workspace(),
                        "workspace activated",
                        server,
                    )
                    bindings = await asyncio.to_thread(
                        app.workspace_registry_service.list_folder_bindings,
                        workspace.workspace_id,
                    )
                    assert (
                        len(bindings) == 1
                        and Path(bindings[0].locator) == project.resolve()
                    )
                    alias = bindings[0].binding_id
                    record.update(workspace_id=workspace.workspace_id, binding_id=alias)
                    phase("new-chat")
                    console = app.screen
                    await click(pilot, "#console-new-chat-tab", server)
                    controller = console._console_chat_controller
                    store = controller.store
                    session = next(
                        item
                        for item in store.sessions()
                        if item.id == store.active_session_id
                    )
                    assert (
                        session.workspace_id == workspace.workspace_id
                    ), "new chat did not inherit the active workspace"
                    record["session_id"] = session.id
                    if os.environ.get("TLDW_UAT_NEGATIVE_CONTROL") == "missing-readme":
                        readme.write_text(
                            "README deliberately lacks verification evidence.\n",
                            encoding="utf-8",
                        )
                        record["negative_control"] = "missing-readme"
                    phase("send")
                    await click(pilot, "#console-native-composer", server)
                    prompt = "Inspect this project's README. Summarize what it does and include its verification code."
                    assert app.focused.post_message(Paste(prompt))
                    await wait_for(
                        pilot,
                        lambda: console.query_one(
                            "#console-native-composer"
                        ).draft_text()
                        == prompt,
                        "pasted prompt",
                        server,
                    )
                    await click(pilot, "#console-send-message", server)

                    last_snapshot = 0.0

                    def capture_progress():
                        nonlocal last_snapshot
                        if time.monotonic() - last_snapshot >= 10:
                            last_snapshot = time.monotonic()
                            (artifacts / "task-stacks.txt").write_text(
                                preparation_stacks(), encoding="utf-8"
                            )
                            (artifacts / "viewport.txt").write_text(
                                painted_text(app.screen), encoding="utf-8"
                            )
                            (artifacts / "requests.json").write_text(
                                json.dumps(server.model.received, indent=2),
                                encoding="utf-8",
                            )

                    def instruction_notice():
                        capture_progress()
                        return bool(app.screen.query("#console-project-notice-proceed"))

                    phase("project-instructions")
                    await wait_for(
                        pilot,
                        instruction_notice,
                        "project instruction disclosure",
                        server,
                        timeout=360 if os.name == "nt" else 100,
                    )
                    assert app.screen.notice.relative_source == "AGENTS.md"
                    assert app.screen.notice.session_id == session.id
                    assert not server.model.received
                    await click(pilot, "#console-project-notice-proceed", server)
                    await wait_for(
                        pilot,
                        lambda: app.screen is console,
                        "project disclosure dismissed",
                        server,
                    )

                    def final_message():
                        capture_progress()
                        for message in reversed(store.messages_for_session(session.id)):
                            if (
                                message.role == "assistant"
                                and message.status == "complete"
                                and message.persisted_message_id
                                and server.model.final
                                and message.content == server.model.final
                            ):
                                return message
                        return None

                    phase("await-answer")
                    message = await wait_for(
                        pilot,
                        final_message,
                        "completed persisted assistant answer",
                        server,
                        timeout=360 if os.name == "nt" else 100,
                    )
                    assert (
                        len(server.model.received)
                        == len(server.model.requests)
                        == len(server.model.sequence)
                    )
                    assert code not in json.dumps(server.model.requests[0])
                    assert server.model.alias == alias
                    if variant == "compact-recovery":
                        initial = server.model.requests[0]
                        assert set(disclosed_schemas(initial)) <= {
                            "find_tools",
                            "load_tools",
                            "read_agent_messages",
                        }
                        assert (
                            "Use find_tools to discover available tools, then call load_tools"
                            in json.dumps(initial)
                        )
                        from tldw_chatbook.Utils.token_counter import (
                            count_tokens_messages,
                        )

                        estimate = count_tokens_messages(
                            initial["messages"], model=MODEL
                        )
                        record["compact_input_estimate"] = estimate
                        assert estimate + initial.get("max_tokens", 0) <= 4096
                    phase("verify-evidence")
                    row = await wait_for(
                        pilot,
                        lambda: next(
                            (
                                row
                                for row in console.query(
                                    f"#console-message-{message.id}"
                                )
                            ),
                            None,
                        ),
                        "final assistant row",
                        server,
                    )
                    row.scroll_visible(animate=False, immediate=True)
                    await wait_for(
                        pilot,
                        lambda: code in painted_text(console, row.region),
                        "painted final assistant code",
                        server,
                    )
                    saved = await asyncio.to_thread(
                        app.chachanotes_db.get_message_by_id,
                        message.persisted_message_id,
                    )
                    runs = await asyncio.to_thread(
                        controller._agent_bridge._db.list_runs,
                        session.persisted_conversation_id,
                    )
                    assert (
                        len(runs) == 1
                    ), "expected one owning run for the one user send"
                    run = runs[0]
                    record.update(
                        run_id=run["id"],
                        conversation_id=session.persisted_conversation_id,
                        assistant_message_id=message.persisted_message_id,
                        steps=run["steps"],
                        answer=message.content,
                    )
                    assert run["status"] == "done"
                    assert_project_ownership(
                        run,
                        saved,
                        session.persisted_conversation_id,
                        message.persisted_message_id,
                    )
                    assert_project_evidence(
                        code=code,
                        alias=alias,
                        steps=run["steps"],
                        provider_request=server.model.received[-1],
                        final_text=message.content,
                        persisted_text=saved["content"],
                        painted_text=painted_text(console, row.region),
                    )
                    assert not blocked, f"unexpected network destinations: {blocked}"
                    phase("shutdown")
                finally:
                    record.update(
                        request_count=len(server.model.received),
                        requests=server.model.received,
                        model_errors=server.errors,
                        probes=server.probes,
                        blocked_destinations=blocked,
                    )
                    (artifacts / "viewport.txt").write_text(
                        painted_text(app.screen), encoding="utf-8"
                    )
                    (artifacts / "viewport.svg").write_text(
                        app.export_screenshot(), encoding="utf-8"
                    )
            assert not blocked, f"unexpected network during shutdown: {blocked}"
        record["status"] = "passed"
        phase("complete")
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"[:4000]
        raise
    finally:
        record["duration_seconds"] = round(time.monotonic() - start, 3)
        (artifacts / "receipt.json").write_text(
            json.dumps(record, indent=2, default=str), encoding="utf-8"
        )
