"""TASK-33003.3 scratch probe (committed for reproducibility, not collected).

Copy to Tests/UI/test_zz_task3_probe_tmp.py and run from the tree root with
PROBE_OUT=<dir> PROBE_TAG=<before|after> python -m pytest <that file> -n0.
Writes painted-text captures and every shown field's width (Model view with
all sections expanded, and the Context view) at 211x44 and 235x52, for a
llama.cpp chat, an OpenAI gpt-5 chat (Reasoning effort, Reasoning summary
and Verbosity) and an Anthropic chat (Thinking effort; fix round 1).
"""
import io
import os

import pytest
from rich.console import Console
from textual.widgets import Button, Collapsible

from Tests.UI.test_console_session_settings import StyledModalHarness
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Widgets.Console.console_settings_modal import (
    ConsoleSettingsContextEstimate,
    ConsoleSettingsModal,
)

OUT = os.environ.get("PROBE_OUT", "/tmp")
TAG = os.environ.get("PROBE_TAG", "x")
CHATS = {
    "llamacpp": dict(provider="llama_cpp", model="model-a", base_url="http://127.0.0.1:9099", temperature=0.7, max_tokens=4096),
    "openai": dict(provider="openai", model="gpt-5", temperature=1.0, max_tokens=8192),
    "anthropic": dict(provider="anthropic", model="claude-opus-4-8", max_tokens=8192),
}


def _text(app) -> str:
    width, height = app.size
    console = Console(width=width, height=height, file=io.StringIO(), record=True,
                      legacy_windows=False, safe_box=False, color_system=None)
    console.print(app.screen._compositor.render_update(full=True, screen_stack=app._background_screens))
    return console.export_text()


def _dump(screen, lines, view):
    for w in screen.query("Input, Select, ConsoleProviderPicker, ModelSearchPicker"):
        if w.id and w.region.area:
            lines.append(f"{view} {type(w).__name__:28} id={w.id:48} x={w.region.x:3} width={w.region.width}")


@pytest.mark.parametrize("chat", sorted(CHATS))
@pytest.mark.parametrize("size", [(211, 44), (235, 52)])
@pytest.mark.asyncio
async def test_probe(size, chat):
    app = StyledModalHarness()
    async with app.run_test(size=size) as pilot:
        await app.push_screen(ConsoleSettingsModal(
            settings=ConsoleSessionSettings(**CHATS[chat]),
            app_config=app.app_config,
            providers_models={"llama_cpp": ["model-a", "model-b"], "openai": ["gpt-5", "gpt-4.1"], "anthropic": ["claude-opus-4-8"]},
            context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
            can_save=True,
        ))
        await pilot.pause()
        await pilot.pause()
        screen = app.screen
        tag = f"{TAG}-{chat}-{size[0]}x{size[1]}"
        lines = []
        screen.query_one("#console-settings-view-model").focus()
        await pilot.pause(0.3); await pilot.pause()
        for c in screen.query(Collapsible):
            c.collapsed = False
        await pilot.pause(); await pilot.pause()
        _dump(screen, lines, "model-expanded")
        body = screen.query_one("#console-settings-body")
        body.scroll_to(y=body.max_scroll_y // 2 if chat != "llamacpp" else 14, animate=False)
        await pilot.pause(); await pilot.pause()
        with open(os.path.join(OUT, f"modal-model-expanded-{tag}.txt"), "w") as fh:
            fh.write(_text(app))
        for c in screen.query(Collapsible):
            c.collapsed = True
        screen.query_one("#console-settings-view-context", Button).press()
        await pilot.pause(); await pilot.pause()
        _dump(screen, lines, "context")
        with open(os.path.join(OUT, f"modal-context-{tag}.txt"), "w") as fh:
            fh.write(_text(app))
        with open(os.path.join(OUT, f"widths-{tag}.txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
