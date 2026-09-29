"""TASK-33003.2 scratch probe (committed for reproducibility, not collected).

Copy to Tests/UI/test_zz_task2_probe_tmp.py and run from the tree root with
PROBE_OUT=<dir> PROBE_TAG=<before|after> python -m pytest <that file> -n0.
Writes painted-text captures and per-control geometry (heights, label rows,
editable Model-view controls fully visible in the body without scrolling).
"""
import io
import os

import pytest
from rich.console import Console
from textual.widgets import Button, Checkbox, Input, Select, Static, Switch, TextArea

from Tests.UI.test_console_session_settings import StyledModalHarness
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Widgets.Console.console_settings_modal import (
    ConsoleSettingsContextEstimate,
    ConsoleSettingsModal,
)

OUT = os.environ.get("PROBE_OUT", "/tmp")
TAG = os.environ.get("PROBE_TAG", "x")


def _text(app) -> str:
    width, height = app.size
    console = Console(width=width, height=height, file=io.StringIO(), record=True,
                      legacy_windows=False, safe_box=False, color_system=None)
    console.print(app.screen._compositor.render_update(full=True, screen_stack=app._background_screens))
    return console.export_text()


def _dump(screen, lines, view):
    body = screen.query_one("#console-settings-body")
    visible = body.region
    footer = screen.query_one("#console-settings-actions")
    editable = 0
    for w in screen.query("Input, Select, Button, Checkbox, Switch, TextArea"):
        if not w.region.area:
            continue
        in_footer = footer in w.ancestors
        kind = type(w).__name__
        full = visible.contains_region(w.region)
        lines.append(f"{view} {kind:28} id={w.id} region={tuple(w.region)} h={w.region.height} footer={in_footer} visible={full}")
        if (not in_footer and full and not w.disabled
                and isinstance(w, (Input, Select, Checkbox, Switch, TextArea))):
            editable += 1
    for lab in screen.query(".console-settings-modal-label"):
        if lab.region.area:
            row = lab.parent
            ctrls = [c for c in row.children if c is not lab and c.region.area]
            tallest = max((c.region.height for c in ctrls), default=0)
            lines.append(f"{view} LABEL {str(lab.render())[:24]!r:28} y={lab.region.y} h={lab.region.height} row_h={row.region.height} tallest_ctrl={tallest} ctrl_y={[c.region.y for c in ctrls]}")
    for sel in ("#console-settings-readiness", "#console-settings-scope", "#console-settings-view-tabs", "#console-settings-default-recovery-actions"):
        w = screen.query_one(sel)
        lines.append(f"{view} SUMMARY {sel} region={tuple(w.region)}")
    lines.append(f"{view} EDITABLE_VISIBLE_WITHOUT_SCROLL={editable}")


@pytest.mark.parametrize("size", [(211, 44), (235, 52)])
@pytest.mark.asyncio
async def test_probe(size):
    app = StyledModalHarness()
    async with app.run_test(size=size) as pilot:
        await app.push_screen(ConsoleSettingsModal(
            settings=ConsoleSessionSettings(provider="llama_cpp", model="model-a", base_url="http://127.0.0.1:9099"),
            app_config=app.app_config,
            providers_models={"llama_cpp": ["model-a", "model-b"]},
            context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
            can_save=True,
        ))
        await pilot.pause()
        await pilot.pause()
        screen = app.screen
        lines = []
        tag = f"{TAG}-{size[0]}x{size[1]}"
        _dump(screen, lines, "model")
        with open(os.path.join(OUT, f"modal-model-{tag}.txt"), "w") as fh:
            fh.write(_text(app))
        # settled: focus parked on the view tab so the picker lists close
        screen.query_one("#console-settings-view-model").focus()
        await pilot.pause(0.3); await pilot.pause()
        screen.query_one("#console-settings-body").scroll_home(animate=False)
        await pilot.pause(); await pilot.pause()
        _dump(screen, lines, "model-settled")
        with open(os.path.join(OUT, f"modal-model-settled-{tag}.txt"), "w") as fh:
            fh.write(_text(app))
        # expanded: open Advanced generation only, scroll to top
        from textual.widgets import Collapsible
        screen.query_one("#console-settings-generation-advanced", Collapsible).collapsed = False
        await pilot.pause(); await pilot.pause()
        screen.query_one("#console-settings-body").scroll_home(animate=False)
        await pilot.pause(); await pilot.pause()
        _dump(screen, lines, "model-expanded")
        with open(os.path.join(OUT, f"modal-model-expanded-{tag}.txt"), "w") as fh:
            fh.write(_text(app))
        for c in screen.query(Collapsible):
            c.collapsed = True
        screen.query_one("#console-settings-view-context").press()
        await pilot.pause(); await pilot.pause()
        _dump(screen, lines, "context")
        with open(os.path.join(OUT, f"modal-context-{tag}.txt"), "w") as fh:
            fh.write(_text(app))
        with open(os.path.join(OUT, f"geometry-{tag}.txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")


@pytest.mark.asyncio
async def test_probe_focus():
    app = StyledModalHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        await app.push_screen(ConsoleSettingsModal(
            settings=ConsoleSessionSettings(provider="llama_cpp", model="model-a", base_url="http://127.0.0.1:9099", temperature=0.7),
            app_config=app.app_config,
            providers_models={"llama_cpp": ["model-a", "model-b"]},
            context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
            can_save=True,
        ))
        await pilot.pause()
        screen = app.screen
        from textual.widgets import Collapsible
        screen.query_one("#console-settings-generation-advanced", Collapsible).collapsed = False
        await pilot.pause(); await pilot.pause()
        temp = screen.query_one("#console-settings-temperature")
        temp.focus()
        await pilot.pause()
        await pilot.press("end", "backspace", "backspace", "backspace", "0", ".", "4", "2")
        await pilot.pause(); await pilot.pause()
        out = []
        txt = _text(app).splitlines()
        for wid in ("#console-settings-temperature", "#console-settings-top-p"):
            w = screen.query_one(wid)
            r = w.region
            for y in range(r.y, r.y + r.height):
                out.append(f"{wid} y={y} {txt[y][r.x - 1:r.right + 1]!r}")
        screen.query_one("#console-settings-view-context").press()
        await pilot.pause(); await pilot.pause()
        sel = screen.query_one("#console-context-budget-mode")
        sel.focus()
        await pilot.pause()
        txt = _text(app).splitlines()
        r = sel.region
        for y in range(r.y - 1, r.y + r.height + 1):
            out.append(f"select y={y} {txt[y][r.x - 1:r.right + 1]!r}")
        tab = screen.query_one("#console-settings-view-model")
        tab.focus()
        await pilot.pause()
        txt = _text(app).splitlines()
        r = tab.region
        for y in range(r.y, r.y + r.height):
            out.append(f"tab y={y} {txt[y][r.x - 1:r.right + 1]!r}")
        with open(os.path.join(OUT, f"focus-{TAG}-211x44.txt"), "w") as fh:
            fh.write("\n".join(out) + "\n")


@pytest.mark.asyncio
async def test_probe_tabs():
    app = StyledModalHarness()
    async with app.run_test(size=(235, 52)) as pilot:
        await app.push_screen(ConsoleSettingsModal(
            settings=ConsoleSessionSettings(provider="llama_cpp", model="model-a", base_url="http://127.0.0.1:9099"),
            app_config=app.app_config,
            providers_models={"llama_cpp": ["model-a", "model-b"]},
            context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
            can_save=True,
        ))
        await pilot.pause()
        screen = app.screen
        out = []
        await pilot.click("#console-settings-view-context")
        await pilot.pause(0.3)
        t = screen.query_one("#console-settings-view-tabs")
        out.append(repr(_text(app).splitlines()[t.region.y][t.region.x:t.region.x+80]))
        await pilot.click("#console-settings-view-model")
        await pilot.pause(0.3)
        out.append(repr(_text(app).splitlines()[t.region.y][t.region.x:t.region.x+80]))
        for b in t.children:
            out.append(f"{b.id} {b.region} {str(b.label)!r}")
        with open(os.path.join(OUT, f"tabs-{TAG}.txt"), "w") as fh:
            fh.write("\n".join(out) + "\n")


@pytest.mark.asyncio
async def test_probe_overlay():
    app = StyledModalHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        await app.push_screen(ConsoleSettingsModal(
            settings=ConsoleSessionSettings(provider="llama_cpp", model="model-a", base_url="http://127.0.0.1:9099"),
            app_config=app.app_config,
            providers_models={"llama_cpp": ["model-a", "model-b"]},
            context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
            can_save=True,
        ))
        await pilot.pause()
        screen = app.screen
        screen.query_one("#console-settings-view-context").press()
        await pilot.pause()
        sel = screen.query_one("#console-context-compaction-mode")
        sel.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause(0.3)
        with open(os.path.join(OUT, f"overlay-{TAG}-211x44.txt"), "w") as fh:
            fh.write(_text(app))
