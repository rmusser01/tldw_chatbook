#!/usr/bin/env python3
"""Isolated, headless UAT for Console context and memory controls.

The harness drives the real Textual app and widgets through the user-facing
paths. Provider calls are replaced with a deterministic local fake. Test app
data lives in the suite's per-app temporary directories, and the environment
is redirected before importing Chatbook.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from html import unescape
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CAPTURES = HERE / "captures"
SANDBOX = HERE / "sandbox"
SANDBOX.mkdir(parents=True, exist_ok=True)
CAPTURES.mkdir(parents=True, exist_ok=True)
(SANDBOX / "config").mkdir(parents=True, exist_ok=True)
(SANDBOX / "data").mkdir(parents=True, exist_ok=True)

os.environ["TLDW_TEST_MODE"] = "1"
os.environ["HOME"] = str(SANDBOX)
os.environ["USERPROFILE"] = str(SANDBOX)
os.environ["APPDATA"] = str(SANDBOX / "appdata" / "roaming")
os.environ["LOCALAPPDATA"] = str(SANDBOX / "appdata" / "local")
os.environ["XDG_CONFIG_HOME"] = str(SANDBOX / "config")
os.environ["XDG_DATA_HOME"] = str(SANDBOX / "data")
os.environ["TLDW_CONFIG_PATH"] = str(SANDBOX / "config" / "config.toml")

sys.path.insert(0, str(REPO))

from Tests.UI.app_factory import (  # noqa: E402
    _build_test_app,
    drain_active_service_patches,
    drain_created_dirs,
)
from tldw_chatbook.Chat.console_context_policy import (  # noqa: E402
    ContextBudgetMode,
    ContextCompactionMode,
)
from tldw_chatbook.Chat.console_provider_gateway import (  # noqa: E402
    AuxiliaryCompletionResult,
    ConsoleProviderResolution,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen  # noqa: E402
from tldw_chatbook.UI.Screens.settings_config_models import (  # noqa: E402
    SettingsCategoryId,
)
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen  # noqa: E402
from tldw_chatbook.Widgets.Console.console_model_popover import (  # noqa: E402
    ConsoleModelPopover,
)
from tldw_chatbook.Widgets.Console.console_settings_modal import (  # noqa: E402
    ConsoleSettingsModal,
)
from textual.widgets import Button, Input, Select, Static  # noqa: E402


class FakeGateway:
    """Ready local provider with a deterministic response and unknown window."""

    async def resolve_for_send(self, _selection):
        return ConsoleProviderResolution(
            provider="llama_cpp",
            base_url="http://127.0.0.1:9099",
            model="uat-unknown-window-model",
            ready=True,
            readiness_key="llama_cpp",
            execution_key="llama_cpp",
        )

    async def complete_auxiliary(self, _request):
        return AuxiliaryCompletionResult(
            provider="llama_cpp",
            model="uat-unknown-window-model",
            text="A bounded summary.",
        )

    async def stream_chat(self, _resolution, _messages, tools=None, signals=None):
        del tools, signals
        for chunk in ("The bounded ", "conversation sent ", "successfully."):
            yield chunk
            await asyncio.sleep(0.01)


def configure_ready(app) -> None:
    app.app_config["chat_defaults"] = {
        "provider": "llama_cpp",
        "model": "uat-unknown-window-model",
    }
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://127.0.0.1:9099",
            "model": "uat-unknown-window-model",
        }
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "uat-unknown-window-model"
    app.console_provider_gateway_factory = FakeGateway


def _svg_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    rows: dict[int, list[tuple[int, str]]] = {}
    for match in re.finditer(
        r'<text[^>]*\bx="([\d.]+)"[^>]*\by="([\d.]+)"[^>]*>(.*?)</text>',
        raw,
        re.S,
    ):
        x, y = float(match.group(1)), float(match.group(2))
        value = unescape(re.sub(r"<[^>]+>", "", match.group(3)))
        rows.setdefault(int(y), []).append((int(x), value))
    return "\n".join(
        "".join(value for _, value in sorted(rows[y])).rstrip() for y in sorted(rows)
    )


def capture(app, name: str) -> None:
    svg_path = CAPTURES / f"{name}.svg"
    app.save_screenshot(str(svg_path))
    (CAPTURES / f"{name}.txt").write_text(_svg_text(svg_path), encoding="utf-8")


async def wait_for(predicate, *, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except Exception:
            pass
        await asyncio.sleep(0.05)
    raise TimeoutError("UAT state did not appear before the timeout")


async def settle(pilot, seconds: float = 0.4) -> None:
    for _ in range(max(1, int(seconds / 0.05))):
        await pilot.pause(0.05)


async def goto_console(pilot, app) -> ChatScreen:
    if not isinstance(app.screen, ChatScreen):
        await pilot.press("ctrl+2")
    await wait_for(lambda: isinstance(app.screen, ChatScreen))
    await settle(pilot, 1.0)
    return app.screen


def static_text(screen, selector: str) -> str:
    renderable = screen.query_one(selector, Static).renderable
    return getattr(renderable, "plain", str(renderable))


async def wide_journey() -> dict[str, object]:
    app = _build_test_app()
    configure_ready(app)
    evidence: dict[str, object] = {}
    try:
        async with app.run_test(size=(120, 42)) as pilot:
            screen = await goto_console(pilot, app)
            capture(app, "01-console-ready-120x42")

            await pilot.press("alt+m")
            await wait_for(lambda: isinstance(app.screen, ConsoleModelPopover))
            capture(app, "02-quick-model-settings-default-120x42")
            evidence["quick_default"] = {
                "response_max": static_text(
                    app.screen,
                    "#console-popover-response-max",
                ),
                "request": static_text(app.screen, "#console-popover-request-usage"),
                "conversation": static_text(
                    app.screen, "#console-popover-conversation-usage"
                ),
                "threshold": static_text(
                    app.screen, "#console-popover-compaction-threshold"
                ),
                "compaction_help": static_text(
                    app.screen,
                    "#console-popover-compaction-help",
                ),
            }
            assert "next reply" in evidence["quick_default"]["response_max"]
            assert "8,001 max tokens" in evidence["quick_default"]["conversation"]
            assert "extra model call" in evidence["quick_default"]["compaction_help"]

            await pilot.click("#console-popover-full-settings")
            await wait_for(lambda: isinstance(app.screen, ConsoleSettingsModal))
            await settle(pilot)
            capture(app, "03-context-memory-default-120x42")
            evidence["default_capacity_status"] = static_text(
                app.screen, "#console-context-capacity-status"
            )
            evidence["context_scope"] = static_text(
                app.screen,
                "#console-settings-scope",
            )
            assert "model capacity is unverified" in evidence["default_capacity_status"]
            assert "this conversation" in evidence["context_scope"]
            assert not app.screen.query_one(
                "#console-settings-save-default",
                Button,
            ).display

            app.screen.query_one(
                "#console-context-budget-mode", Select
            ).value = ContextBudgetMode.CUSTOM.value
            budget = app.screen.query_one("#console-context-custom-budget", Input)
            await pilot.click("#console-settings-save")
            await settle(pilot)
            assert isinstance(app.screen, ConsoleSettingsModal)
            evidence["blank_custom_budget_error"] = static_text(
                app.screen, "#console-settings-error"
            )
            capture(app, "04a-custom-budget-required-error-120x42")
            budget.value = "12000"
            app.screen.query_one(
                "#console-context-compaction-mode", Select
            ).value = ContextCompactionMode.ASK.value
            await settle(pilot)
            capture(app, "04b-context-memory-bounded-before-save-120x42")
            await pilot.click("#console-settings-save")
            await wait_for(lambda: isinstance(app.screen, ChatScreen))
            await settle(pilot, 0.8)

            store = screen._ensure_console_chat_store()
            session_id = store.active_session_id
            assert session_id is not None
            overrides = store.session_context_policy_overrides(session_id)
            evidence["saved_overrides"] = overrides.to_dict()
            assert overrides.custom_budget_tokens == 12_000
            assert overrides.budget_mode is ContextBudgetMode.CUSTOM
            capture(app, "05-console-after-policy-save-120x42")

            await pilot.click("#console-native-composer")
            await pilot.press(*list("Send with my bounded context limit"))
            await pilot.click("#console-send-message")
            await wait_for(
                lambda: len(store.messages_for_session(session_id)) >= 2,
                timeout=15.0,
            )
            await settle(pilot, 0.5)
            capture(app, "06-bounded-first-send-succeeds-120x42")
            messages = store.messages_for_session(session_id)
            evidence["send_result"] = [
                {
                    "role": message.role.value,
                    "status": str(message.status),
                    "content": message.content,
                }
                for message in messages
            ]
            assert any("sent successfully" in message.content for message in messages)

            await pilot.press("alt+m")
            await wait_for(lambda: isinstance(app.screen, ConsoleModelPopover))
            capture(app, "07-quick-model-settings-restored-120x42")
            evidence["quick_restored"] = {
                "request": static_text(app.screen, "#console-popover-request-usage"),
                "conversation": static_text(
                    app.screen, "#console-popover-conversation-usage"
                ),
                "threshold": static_text(
                    app.screen, "#console-popover-compaction-threshold"
                ),
            }
    finally:
        drain_active_service_patches()
        drain_created_dirs()
    return evidence


async def narrow_and_keyboard_journey() -> dict[str, object]:
    app = _build_test_app()
    configure_ready(app)
    evidence: dict[str, object] = {}
    try:
        async with app.run_test(size=(72, 24)) as pilot:
            await goto_console(pilot, app)
            await pilot.press("alt+m")
            await wait_for(lambda: isinstance(app.screen, ConsoleModelPopover))
            capture(app, "08-quick-model-settings-72x24")
            quick_focus_order: list[str] = []
            for _ in range(16):
                focused = app.focused
                focused_id = getattr(focused, "id", None) or type(focused).__name__
                quick_focus_order.append(focused_id)
                if focused_id == "console-popover-full-settings":
                    break
                await pilot.press("tab")
                await pilot.pause(0.03)
            evidence["quick_focus_order"] = quick_focus_order
            assert app.focused is not None
            assert app.focused.id == "console-popover-full-settings"
            assert "console-popover-temperature" in quick_focus_order
            assert "console-popover-streaming" in quick_focus_order
            assert "console-popover-compaction-mode" in quick_focus_order
            assert app.screen.query_one(
                "#console-popover-fold-hint",
                Static,
            ).display
            capture(app, "08b-quick-full-settings-keyboard-focus-72x24")
            await pilot.press("enter")
            await wait_for(lambda: isinstance(app.screen, ConsoleSettingsModal))
            await settle(pilot)
            capture(app, "09-context-memory-72x24")
            modal = app.screen.query_one("#console-settings-modal")
            evidence["modal_region"] = {
                "x": modal.region.x,
                "y": modal.region.y,
                "width": modal.region.width,
                "height": modal.region.height,
            }
            assert app.focused is not None
            assert app.focused.id == "console-context-budget-mode"
            assert app.screen.query_one(
                "#console-settings-fold-hint",
                Static,
            ).display
            focus_order: list[str] = []
            for _ in range(18):
                await pilot.press("tab")
                await pilot.pause(0.03)
                focused = app.focused
                focus_order.append(
                    getattr(focused, "id", None) or type(focused).__name__
                )
            evidence["focus_order"] = focus_order
    finally:
        drain_active_service_patches()
        drain_created_dirs()
    return evidence


async def global_settings_journey() -> dict[str, object]:
    app = _build_test_app()
    configure_ready(app)
    evidence: dict[str, object] = {}
    try:
        async with app.run_test(size=(120, 42)) as pilot:
            await goto_console(pilot, app)
            app.action_shell_destination(12)
            await wait_for(lambda: isinstance(app.screen, SettingsScreen))
            await settle(pilot, 1.5)
            screen: SettingsScreen = app.screen
            screen._select_category(SettingsCategoryId.CONSOLE_BEHAVIOR.value)
            await settle(pilot)
            capture(app, "10-global-console-behavior-120x42")
            jump = screen.query_one("#settings-console-context-memory-jump", Button)
            assert "Conversation context and memory" in str(jump.label)
            jump.press()
            await settle(pilot)
            capture(app, "11-global-context-memory-focused-120x42")
            assert screen.focused is not None
            assert screen.focused.id == "settings-console-context-budget-mode"
            evidence["safety_copy"] = static_text(
                screen, "#settings-console-context-safety-copy"
            )
            evidence["budget_mode"] = str(
                screen.query_one("#settings-console-context-budget-mode", Select).value
            )
            evidence["compaction_mode"] = str(
                screen.query_one(
                    "#settings-console-context-compaction-mode", Select
                ).value
            )
            evidence["conversation_max_label"] = static_text(
                screen.query_one("#settings-console-context-budget-tokens").parent,
                ".settings-input-label",
            )
            evidence["response_max_label"] = static_text(
                screen.query_one("#settings-console-default-max-tokens").parent,
                ".settings-input-label",
            )
            assert evidence["conversation_max_label"] == "Conversation max tokens"
            assert evidence["response_max_label"] == "Response max tokens"
    finally:
        drain_active_service_patches()
        drain_created_dirs()
    return evidence


async def main() -> None:
    scenarios = {
        "wide": wide_journey,
        "narrow": narrow_and_keyboard_journey,
        "settings": global_settings_journey,
    }
    requested = sys.argv[1:] or list(scenarios)
    unknown = [name for name in requested if name not in scenarios]
    if unknown:
        raise SystemExit(f"Unknown scenarios: {', '.join(unknown)}")
    result = {name: await scenarios[name]() for name in requested}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
