import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# Keep the native child's PTY dimensions stable when the tool transport yields.
if len(sys.argv) > 1 and sys.argv[1] == "--pty-child":
    sys.argv.pop(1)
else:
    import errno
    import fcntl
    import pty
    import struct
    import subprocess
    import termios

    master, slave = pty.openpty()
    columns, rows = map(int, sys.argv[1:3])
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))
    process = subprocess.Popen(
        [sys.executable, __file__, "--pty-child", *sys.argv[1:]],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        start_new_session=True,
    )
    os.close(slave)
    try:
        while True:
            try:
                output = os.read(master, 65536)
            except OSError as error:
                if error.errno != errno.EIO:
                    raise
                break
            if not output:
                break
            sys.stdout.buffer.write(output)
            sys.stdout.buffer.flush()
    finally:
        os.close(master)
        if process.poll() is None:
            process.terminate()
    raise SystemExit(process.wait(timeout=5))

root = Path(tempfile.mkdtemp(prefix="hook-review-native-", dir="/private/tmp"))
root.chmod(0o700)
# The app recovery bootstrap is rooted under HOME; isolate it with this QA profile.
qa_home = root / "home"
qa_home.mkdir(mode=0o700)
os.environ["HOME"] = str(qa_home)
data = root / "data"
data.mkdir(mode=0o700)
os.environ["TLDW_CONFIG_PATH"] = str(root / "config.toml")
(root / "config.toml").write_text(f'''
[general]
default_tab = "chat"
users_name = "hook-review-qa"
[paths]
data_dir = "{data}"
[first_run]
setup_completed = true
[splash_screen]
enabled = false
[model_catalog]
auto_refresh_enabled = false
[embeddings]
enabled = false
[chat_defaults]
provider = "llama_cpp"
model = "local-model"
[api_settings.llama_cpp]
api_url = "http://127.0.0.1:9099"
model = "local-model"
[console]
agent_runtime = false
native_tool_calls = false
[hooks]
enabled = true
[[hooks.hook]]
id = "qa-hook"
event = "UserPromptSubmit"
command = ["{sys.executable}", "-c", "from pathlib import Path; p=Path('{root}/hook-marker'); p.write_text((p.read_text() if p.exists() else '')+'run;')", "a b", "[bold]"]
timeout_s = 5
''')
sys.path.insert(0, str(Path.cwd()))
from loguru import logger

logger.remove()
logger.add(str(root / "application.log"), level="WARNING")
from tldw_chatbook import config
from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.Widgets.Console.console_hooks_review_modal import (
    ConsoleHooksReviewModal,
)

assert (
    config.read_hooks_config_snapshot().config_path == (root / "config.toml").resolve()
)
assert Path(config.get_user_data_dir()).resolve() == (data / "hook-review-qa").resolve()
for getter in (
    config.get_chachanotes_db_path,
    config.get_prompts_db_path,
    config.get_media_db_path,
    config.get_workspaces_db_path,
    config.get_evals_db_path,
):
    assert getter().resolve().is_relative_to(root.resolve())
app = TldwCli()
from Tests.Chat.test_console_chat_controller import RecordingStreamingGateway
from Tests.UI.test_console_workbench_contract import _configure_native_ready_console

gateway = RecordingStreamingGateway()
from tldw_chatbook.Utils.token_counter import resolve_context_window

# Native first paint now asks the gateway for cached serving metadata.
gateway.cached_context_window = lambda settings: resolve_context_window(
    settings.provider, settings.model or ""
)
app.console_provider_gateway_factory = lambda: gateway
original_app_config = app.app_config
_configure_native_ready_console(app)
app.app_config = {**original_app_config, **app.app_config}
size = (int(sys.argv[1]), int(sys.argv[2]))
results = {"size": size, "profile": str(root), "checks": []}


async def wait_for(pilot, condition, timeout=30):
    async with asyncio.timeout(timeout):
        while not condition():
            await pilot.pause(0.05)


async def qa(pilot):
    try:
        await wait_for(
            pilot,
            lambda: (
                isinstance(app.screen, ChatScreen)
                and bool(app.screen.query("#console-control-hooks"))
            ),
        )
        assert (app.size.width, app.size.height) == size, (app.size, size)
        results["actual_size"] = [app.size.width, app.size.height]
        from tldw_chatbook import config

        assert (
            config.read_hooks_config_snapshot().config_path
            == (root / "config.toml").resolve()
        )
        assert (
            Path(config.get_user_data_dir()).resolve()
            == (data / "hook-review-qa").resolve()
        )
        results["checks"].append(
            "config and data resolved inside the private QA profile"
        )
        console = app.screen
        results["stage"] = "refresh hooks"
        await console._refresh_console_hooks()
        assert str(console.query_one("#console-control-hooks").label).endswith(" 1")
        results["stage"] = "open review"
        console.query_one("#console-control-hooks").focus()
        await pilot.press("enter")
        await wait_for(
            pilot,
            lambda: (
                isinstance(app.screen, ConsoleHooksReviewModal)
                and bool(app.screen.query("#hook-review-details-0"))
            ),
        )
        results["stage"] = "expand review"
        modal = app.screen
        await pilot.pause()
        assert not modal._selected
        modal.query_one("#hook-review-details-0").focus()
        await pilot.pause()
        assert modal.focused.id == "hook-review-details-0"
        await pilot.press("enter")
        await wait_for(pilot, lambda: bool(modal.query(".hook-review-detail")))
        await pilot.pause()
        for selector in (
            "#console-hooks-cancel",
            "#console-hooks-settings",
            "#console-hooks-allow-all",
        ):
            control = modal.query_one(selector)
            assert (
                control.region.right <= size[0] and control.region.bottom <= size[1]
            ), selector
        assert (app.size.width, app.size.height) == size, (app.size, size)
        (root / "review.svg").write_text(app.export_screenshot())
        results["checks"].append(
            "native Console icon, expanded exact argv, modal action bounds"
        )
        await pilot.press("escape")
        await wait_for(pilot, lambda: app.screen is console)
        composer = console._console_composer_or_none()
        composer.load_draft("native hook review draft")
        await pilot.pause()
        marker = root / "hook-marker"
        send = asyncio.create_task(
            console._dispatch_console_draft_send("native hook review draft")
        )
        await wait_for(pilot, lambda: isinstance(app.screen, ConsoleHooksReviewModal))
        await pilot.pause()
        assert not marker.exists() and gateway.messages_seen is None
        await pilot.press("escape")
        assert not await asyncio.wait_for(send, 10)
        assert composer.draft_text() == "native hook review draft"
        assert not marker.exists() and gateway.messages_seen is None
        results["checks"].append(
            "pending next Send and Escape retain the draft without hook or provider execution"
        )
        send = asyncio.create_task(
            console._dispatch_console_draft_send("native hook review draft")
        )
        await wait_for(pilot, lambda: isinstance(app.screen, ConsoleHooksReviewModal))
        await pilot.pause()
        app.screen.query_one("#console-hooks-allow-all").focus()
        await pilot.press("enter")
        assert await asyncio.wait_for(send, 15)
        assert marker.read_text() == "run;"
        await wait_for(pilot, lambda: gateway.messages_seen is not None)
        controller = console._ensure_console_chat_controller()
        await wait_for(pilot, lambda: controller.run_state.is_send_allowed)
        from tldw_chatbook.Agents.hook_permissions import HookPermissions

        assert HookPermissions().snapshot().ready
        results["checks"].append(
            "explicit approval launches the real hook once and admits the recording gateway; persisted consent reloads"
        )
        console.query_one("#console-control-hooks").focus()
        await pilot.press("enter")
        await wait_for(pilot, lambda: isinstance(app.screen, ConsoleHooksReviewModal))
        await pilot.pause()
        app.screen.query_one("#console-hooks-settings").focus()
        await pilot.press("enter")
        await wait_for(
            pilot,
            lambda: (
                isinstance(app.screen, SettingsScreen)
                and bool(app.screen.query("#settings-hooks-enabled"))
            ),
        )
        assert app.screen.active_category == "hooks"
        await pilot.pause()
        draft = app.screen._settings_drafts[app.screen._active_category_id()]
        results["initial_settings_dirty_keys"] = sorted(draft.dirty_keys)
        assert (app.size.width, app.size.height) == size, (app.size, size)
        app.clear_notifications()
        await pilot.pause()
        (root / "settings.svg").write_text(app.export_screenshot())
        results["checks"].append("modal deep link to canonical Hooks Settings")
        checkbox = app.screen.query_one("#settings-hooks-enabled")
        checkbox.focus()
        await pilot.press("space")
        await pilot.pause()
        assert app.screen._category_has_unsaved_changes(
            app.screen._active_category_id()
        )
        app.screen.query_one("#settings-hooks-save").focus()
        await pilot.press("enter")
        await wait_for(pilot, lambda: not app.screen._hooks_saving)
        await pilot.pause()
        import tomllib

        assert (
            tomllib.loads((root / "config.toml").read_text())["hooks"]["enabled"]
            is False
        )
        results["checks"].append("keyboard staged master switch and guarded Save")
    except BaseException as error:
        results["error"] = repr(error)
        results["screen"] = type(app.screen).__name__
        try:
            (root / "failure.svg").write_text(app.export_screenshot())
        except Exception as screenshot_error:  # noqa: BLE001 -- capture must preserve the original failure
            results["screenshot_error"] = type(screenshot_error).__name__
        raise
    finally:
        (root / "result.json").write_text(json.dumps(results, indent=2))
        app.exit()


app.run(size=size, auto_pilot=qa)
print(json.dumps(results, indent=2))
