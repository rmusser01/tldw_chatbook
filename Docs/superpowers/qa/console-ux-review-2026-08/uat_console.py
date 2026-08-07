#!/usr/bin/env python3
"""Headless UAT harness v2 for the Console screen UX review (2026-08-04).

Fully sandboxed: TLDW_CONFIG_PATH + HOME/XDG redirects point at ./sandbox so
no real user config or data is read or written (v1 learned this the hard way —
runtime code re-resolves paths after the test factory's init-time patches
expire).

Drives the real TldwCli app through first-run and power-user scenarios, saving
Textual SVG screenshots + visible-text dumps into ./captures/.

Usage:
    ../../.venv/bin/python uat_console.py all
    ../../.venv/bin/python uat_console.py f1_first_run p1_ready
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
import time
from html import unescape
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SANDBOX = HERE / "sandbox"
CAP = HERE / "captures"
CAP.mkdir(parents=True, exist_ok=True)

# --- sandbox BEFORE any tldw_chatbook import --------------------------------
SANDBOX_HOME = SANDBOX / "home"
SANDBOX_HOME.mkdir(parents=True, exist_ok=True)
SANDBOX_CONFIG = SANDBOX / "config" / "config.toml"
SANDBOX_CONFIG.parent.mkdir(parents=True, exist_ok=True)
os.environ["HOME"] = str(SANDBOX_HOME)
os.environ["XDG_CONFIG_HOME"] = str(SANDBOX_HOME / ".config")
os.environ["XDG_DATA_HOME"] = str(SANDBOX_HOME / ".local" / "share")
os.environ["TLDW_CONFIG_PATH"] = str(SANDBOX_CONFIG)

sys.path.insert(0, str(REPO))

from Tests.UI.app_factory import (  # noqa: E402
    _build_test_app,
    drain_active_service_patches,
    drain_created_dirs,
)
from tldw_chatbook.Chat.console_provider_gateway import (  # noqa: E402
    AuxiliaryCompletionResult,
    ConsoleProviderResolution,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen  # noqa: E402

REPLY = (
    "Here's a **markdown** reply with structure:\n\n"
    "- first point\n- second point\n\n"
    "And a sentence with `inline code` to check styling."
)

SEEDED_PROVIDER_CONFIG = """\
[chat_defaults]
provider = "llama_cpp"
model = "local-model"

[api_settings.llama_cpp]
api_url = "http://127.0.0.1:9099"
model = "local-model"
"""


class FakeGateway:
    """Ready provider that streams a canned reply (or raises)."""

    def __init__(self, reply: str = REPLY, fail: str | None = None) -> None:
        self.reply = reply
        self.fail = fail

    async def resolve_for_send(self, selection):
        return ConsoleProviderResolution(
            provider="llama_cpp",
            base_url="http://127.0.0.1:9099",
            model="local-model",
            ready=True,
            readiness_key="llama_cpp",
            execution_key="llama_cpp",
        )

    async def complete_auxiliary(self, request):
        return AuxiliaryCompletionResult(
            provider="llama_cpp", model="local-model", text="{}"
        )

    async def stream_chat(self, resolution, messages, tools=None, signals=None):
        if self.fail:
            raise RuntimeError(self.fail)
        for i in range(0, len(self.reply), 8):
            yield self.reply[i : i + 8]
            await asyncio.sleep(0.005)


def configure_ready(app) -> None:
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "local-model"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "local-model"}
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"


def seed_config(provider: bool) -> None:
    SANDBOX_CONFIG.write_text(SEEDED_PROVIDER_CONFIG if provider else "")
    stale = SANDBOX_HOME / ".config" / "tldw_cli" / "config.toml"
    if stale.exists():
        stale.unlink()


# ---------------------------------------------------------------- helpers


def svg_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    rows: dict[int, list[tuple[int, str]]] = {}
    for m in re.finditer(
        r'<text[^>]*\bx="([\d.]+)"[^>]*\by="([\d.]+)"[^>]*>(.*?)</text>', raw, re.S
    ):
        x, y = float(m.group(1)), float(m.group(2))
        txt = unescape(re.sub(r"<[^>]+>", "", m.group(3)))
        rows.setdefault(int(y), []).append((int(x), txt))
    return "\n".join(
        "".join(t for _, t in sorted(rows[y])).rstrip() for y in sorted(rows)
    )


def save(app, name: str) -> None:
    svg = CAP / f"{name}.svg"
    app.save_screenshot(str(svg))
    try:
        (CAP / f"{name}.txt").write_text(svg_text(svg), encoding="utf-8")
    except Exception as exc:
        (CAP / f"{name}.txt").write_text(f"<text extraction failed: {exc}>")


async def settle(pilot, seconds: float = 0.5, step: float = 0.1) -> None:
    for _ in range(max(1, int(seconds / step))):
        await pilot.pause(step)


def chat_screen(app) -> ChatScreen | None:
    for s in app.screen_stack:
        if isinstance(s, ChatScreen):
            return s
    return None


def wizard_present(app) -> bool:
    return any(type(s).__name__ == "FirstRunSetupWizard" for s in app.screen_stack)


async def wait_for(predicate, timeout: float = 45.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass
        await asyncio.sleep(0.15)
    return False


async def dismiss_wizard(pilot, app) -> str:
    """Dismiss the first-run wizard via its visible 'Skip' affordance."""
    from textual.widgets import Button

    if not wizard_present(app):
        return "no wizard"
    for b in app.screen.query(Button):
        if any(k in str(b.label).lower() for k in ("skip", "later", "dismiss")):
            bid = b.id
            await pilot.click(f"#{bid}")
            await settle(pilot, 1.0)
            return f"clicked {bid}"
    await pilot.press("escape")
    await settle(pilot, 1.0)
    return "pressed escape"


async def goto_console(pilot, app) -> bool:
    """Navigate to the Console tab if not already there (UI-level)."""
    if isinstance(app.screen, ChatScreen):
        return True
    for key in ("2", "ctrl+2"):
        try:
            await pilot.press(key)
        except Exception:
            pass
        ok = await wait_for(lambda: isinstance(app.screen, ChatScreen), timeout=6)
        if ok:
            return True
    return isinstance(app.screen, ChatScreen)


# ---------------------------------------------------------------- scenarios


async def f1_first_run():
    """FT: fresh app — wizard auto-offer over Home."""
    seed_config(provider=False)
    app = _build_test_app(first_run_setup_completed=False)
    async with app.run_test(size=(140, 42)) as pilot:
        await wait_for(lambda: wizard_present(app), timeout=60)
        await settle(pilot, 1.0)
        save(app, "f1-first-run-wizard")


async def f2_after_skip():
    """FT: skip wizard, then navigate to Console → its own setup blocker."""
    seed_config(provider=False)
    app = _build_test_app(first_run_setup_completed=False)
    async with app.run_test(size=(140, 42)) as pilot:
        await wait_for(lambda: wizard_present(app), timeout=60)
        await settle(pilot, 1.0)
        how = await dismiss_wizard(pilot, app)
        (CAP / "f2-dismissed-via.txt").write_text(how)
        await settle(pilot, 1.0)
        save(app, "f2a-home-after-skip")
        ok = await goto_console(pilot, app)
        await settle(pilot, 2.0)
        save(app, "f2b-console-after-skip")
        (CAP / "f2b-got-console.txt").write_text(str(ok))


async def f3_send_blocked():
    """FT: in the unconfigured Console, try to type + send."""
    seed_config(provider=False)
    app = _build_test_app(first_run_setup_completed=True)
    async with app.run_test(size=(140, 42)) as pilot:
        if await wait_for(lambda: wizard_present(app), timeout=8):
            await dismiss_wizard(pilot, app)
        ok = await goto_console(pilot, app)
        await settle(pilot, 2.0)
        save(app, "f3a-console-unconfigured")
        cs = chat_screen(app)
        blocked_note = ""
        if ok and cs is not None:
            try:
                await pilot.click("#console-native-composer")
                await pilot.press(*list("hello there"))
                await settle(pilot, 0.5)
                save(app, "f3b-typed-no-provider")
                await pilot.press("enter")
                await settle(pilot, 1.5)
                save(app, "f3c-enter-no-provider")
                await pilot.click("#console-send-message")
                await settle(pilot, 1.5)
                save(app, "f3d-send-click-no-provider")
            except Exception as exc:
                blocked_note = repr(exc)
        (CAP / "f3-notes.txt").write_text(f"goto_console={ok}\n{blocked_note}")


async def f4_slash_popup():
    """FT/PU: slash popup (needs focus in composer)."""
    seed_config(provider=True)
    app = _build_test_app()
    configure_ready(app)
    async with app.run_test(size=(140, 42)) as pilot:
        await goto_console(pilot, app)
        await wait_for(lambda: chat_screen(app) is not None, timeout=30)
        await settle(pilot, 2.0)
        try:
            await pilot.click("#console-native-composer")
            await pilot.press("/")
            await settle(pilot, 1.2)
        except Exception as exc:
            (CAP / "f4-error.txt").write_text(repr(exc))
        save(app, "f4-slash-popup")


async def f5_palette():
    """PU: command palette from Console."""
    seed_config(provider=True)
    app = _build_test_app()
    configure_ready(app)
    async with app.run_test(size=(140, 42)) as pilot:
        await goto_console(pilot, app)
        await settle(pilot, 2.0)
        await pilot.press("ctrl+p")
        await settle(pilot, 1.5)
        save(app, "f5-command-palette")


async def p1_ready():
    """PU: send-ready Console, wide."""
    seed_config(provider=True)
    app = _build_test_app()
    configure_ready(app)
    async with app.run_test(size=(160, 48)) as pilot:
        await goto_console(pilot, app)
        await settle(pilot, 2.5)
        save(app, "p1-ready-console")


async def p2_send():
    """PU: type + send, streaming reply lands."""
    seed_config(provider=True)
    app = _build_test_app()
    configure_ready(app)
    app.console_provider_gateway_factory = lambda: FakeGateway()
    async with app.run_test(size=(160, 48)) as pilot:
        ok = await goto_console(pilot, app)
        await settle(pilot, 2.0)
        try:
            await pilot.click("#console-native-composer")
            await pilot.press(*list("summarize the UX review plan"))
            await settle(pilot, 0.4)
            save(app, "p2a-draft-typed")
            await pilot.click("#console-send-message")
            await settle(pilot, 0.5)
            save(app, "p2b-streaming")
            await settle(pilot, 5.0)
            save(app, "p2c-reply-complete")
        except Exception as exc:
            (CAP / "p2-error.txt").write_text(f"goto={ok} {exc!r}")


async def p3_overlays():
    """PU: help panel, session switcher, model popover, chat-context viewer."""
    seed_config(provider=True)
    app = _build_test_app()
    configure_ready(app)
    async with app.run_test(size=(160, 48)) as pilot:
        await goto_console(pilot, app)
        await settle(pilot, 2.0)
        await pilot.press("f1")
        await settle(pilot, 1.2)
        save(app, "p3a-help-panel")
        await pilot.press("escape")
        await settle(pilot, 0.6)
        await pilot.press("ctrl+k")
        await settle(pilot, 1.2)
        save(app, "p3b-session-switcher")
        await pilot.press("escape")
        await settle(pilot, 0.6)
        try:
            await pilot.press("alt+m")
        except Exception:
            pass
        await settle(pilot, 1.2)
        save(app, "p3c-model-popover")
        await pilot.press("escape")
        await settle(pilot, 0.6)
        await pilot.press("ctrl+shift+p")
        await settle(pilot, 1.5)
        save(app, "p3d-context-viewer")
        await pilot.press("escape")
        await settle(pilot, 0.6)


async def p4_narrow():
    """PU: responsive behavior at smaller sizes."""
    for w, h in ((140, 42), (110, 32), (80, 24), (60, 18)):
        seed_config(provider=True)
        app = _build_test_app()
        configure_ready(app)
        async with app.run_test(size=(w, h)) as pilot:
            await goto_console(pilot, app)
            await settle(pilot, 2.0)
            save(app, f"p4-narrow-{w}x{h}")


async def p5_focus_tour():
    """A11y: tab-focus order."""
    seed_config(provider=True)
    app = _build_test_app()
    configure_ready(app)
    async with app.run_test(size=(160, 48)) as pilot:
        await goto_console(pilot, app)
        await settle(pilot, 2.0)
        seen = []
        for _ in range(30):
            await pilot.press("tab")
            await pilot.pause(0.1)
            w = app.screen.focused
            seen.append("<none>" if w is None else f"{type(w).__name__}#{w.id}")
        (CAP / "p5-focus-tour.txt").write_text("\n".join(seen), encoding="utf-8")


async def p6_send_failure():
    """PU: provider errors on send — feedback + recovery."""
    seed_config(provider=True)
    app = _build_test_app()
    configure_ready(app)
    app.console_provider_gateway_factory = lambda: FakeGateway(
        fail="Connection refused: llama.cpp server not reachable at http://127.0.0.1:9099"
    )
    async with app.run_test(size=(160, 48)) as pilot:
        await goto_console(pilot, app)
        await settle(pilot, 2.0)
        try:
            await pilot.click("#console-native-composer")
            await pilot.press(*list("will this fail"))
            await pilot.click("#console-send-message")
            await settle(pilot, 3.0)
        except Exception as exc:
            (CAP / "p6-error.txt").write_text(repr(exc))
        save(app, "p6-send-failure")


async def p7_inspector_handle():
    """PU: try to open the Inspector rail at 140 cols (below the 150-col force-collapse)."""
    seed_config(provider=True)
    app = _build_test_app()
    configure_ready(app)
    async with app.run_test(size=(140, 42)) as pilot:
        await goto_console(pilot, app)
        await settle(pilot, 2.0)
        save(app, "p7a-140-before-inspector")
        try:
            await pilot.click("#console-inspector-rail-open")
            await settle(pilot, 1.2)
        except Exception as exc:
            (CAP / "p7-error.txt").write_text(repr(exc))
        save(app, "p7b-140-after-inspector-click")


SCENARIOS = {
    "f1_first_run": f1_first_run,
    "f2_after_skip": f2_after_skip,
    "f3_send_blocked": f3_send_blocked,
    "f4_slash_popup": f4_slash_popup,
    "f5_palette": f5_palette,
    "p1_ready": p1_ready,
    "p2_send": p2_send,
    "p3_overlays": p3_overlays,
    "p4_narrow": p4_narrow,
    "p5_focus_tour": p5_focus_tour,
    "p6_send_failure": p6_send_failure,
    "p7_inspector_handle": p7_inspector_handle,
}


async def main(names: list[str]) -> None:
    if not names or names == ["all"]:
        names = list(SCENARIOS)
    for name in names:
        print(f"== {name} ==", flush=True)
        try:
            await SCENARIOS[name]()
            print("   ok", flush=True)
        except Exception as exc:
            print(f"   FAILED: {exc!r}", flush=True)
            (CAP / f"{name}.error.txt").write_text(repr(exc))
        finally:
            drain_created_dirs()
            drain_active_service_patches()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
