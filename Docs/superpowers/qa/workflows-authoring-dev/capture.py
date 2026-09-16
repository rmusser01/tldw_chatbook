"""Authoring-port visual evidence only; no runtime, model, server or installed user data.

Run from the worktree with the existing interpreter and process-only
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib. Functional tests run separately.
"""

import asyncio
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

# Bootstrap isolated config and deny network BEFORE application imports.
importlib.import_module("Tests.conftest")
importlib.import_module("Tests.UI.conftest")

from rich.console import CONSOLE_SVG_FORMAT, Console

from Tests.UI.app_factory import (
    _build_test_app,
    drain_active_service_patches,
    drain_created_dirs,
)
from Tests.UI.test_screen_navigation import _wait_for_initial_screen
from Tests.UI.test_workflows_editor import svg_text_contrast
from tldw_chatbook.config import save_setting_to_cli_config
from tldw_chatbook.DB.Workflows_DB import WorkflowsDB
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.workflows_screen import WorkflowsScreen
from tldw_chatbook.UI.Workflows_Modules.editor import WorkflowEditor
from tldw_chatbook.Utils.optional_deps import ensure_svg_rendering
from tldw_chatbook.Workflows.document_service import DocumentService


def export_menlo(console, **kwargs):
    """Capture-only public Rich format override; keep all cell geometry intact."""
    template = re.sub(
        r"@font-face\s*\{\{.*?\}\}", "", CONSOLE_SVG_FORMAT, flags=re.DOTALL
    )
    kwargs["code_format"] = template.replace(
        "font-family: Fira Code, monospace;", "font-family: Menlo, monospace;"
    )
    return RICH_EXPORT_SVG(console, **kwargs)


RICH_EXPORT_SVG = Console.export_svg


async def capture() -> None:
    if not ensure_svg_rendering():
        raise ImportError(
            "Workflow visual capture requires SVG rendering. Install "
            "tldw_chatbook[svg] and the native Cairo library."
        )
    # Import after process-local fontconfig setup, before Cairo resolves fonts.
    import cairocffi as cairo
    import cairosvg

    context = cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1))
    for weight in (cairo.FONT_WEIGHT_NORMAL, cairo.FONT_WEIGHT_BOLD):
        context.select_font_face("Menlo", cairo.FONT_SLANT_NORMAL, weight)
        context.set_font_size(20)
        thin, wide = (context.text_extents(text)[4] for text in ("iiii", "WWWW"))
        assert thin == wide, (thin, wide)
        print(
            f"Cairo Menlo weight={weight}: iiii={thin}, WWWW={wide} (fixed pitch)",
            flush=True,
        )
    # The tool host exports NO_COLOR; capture the real app's color theme instead.
    os.environ.pop("NO_COLOR", None)
    assert (
        Path(importlib.import_module("tldw_chatbook").__file__).resolve().parent
        == ROOT / "tldw_chatbook"
    )
    output = ROOT / ".impeccable/review/workflows-authoring-dev"
    output.mkdir(parents=True, exist_ok=True)
    # The constructor-only factory patch does not cover delayed splash reads.
    # This writes only the conftest's disposable profile established above.
    save_setting_to_cli_config("splash_screen", "enabled", False)
    for width, height in ((160, 48), (110, 36), (60, 20)):
        with TemporaryDirectory(prefix="workflow-authoring-visual-") as directory:
            path = Path(directory).resolve(strict=True) / "workflows.sqlite3"
            database = WorkflowsDB(path)
            try:
                fixture = ROOT / "Tests/fixtures/workflows/file_to_note.json"
                DocumentService(database).create(fixture.read_text(encoding="utf-8"))
            finally:
                database.close()
            # Real TldwCli, navigation, workflow lifecycle, database and CSS.
            # The established factory substitutes unrelated service startup.
            app = _build_test_app(
                "home", config_overrides={"splash_screen": {"enabled": False}}
            )
            app._refresh_model_catalogs = AsyncMock()
            with patch("tldw_chatbook.config.get_workflows_db_path", return_value=path):
                await capture_app(app, width, height, output, cairosvg)
            drain_active_service_patches()
            drain_created_dirs()


async def capture_app(app, width, height, output, cairosvg):
    async with app.run_test(size=(width, height)) as pilot:
        await _wait_for_initial_screen(pilot)
        app.post_message(NavigateToScreen("workflows"))
        await pilot.pause()
        await asyncio.gather(
            *(w.wait() for w in app.workers if w.group.startswith("workflows-"))
        )
        await pilot.pause()
        assert isinstance(app.screen, WorkflowsScreen)
        if width == 160:
            export_frame(
                app, width, height, output, "workflows-overview-160x48", cairosvg
            )
        app.screen.query_one(WorkflowEditor).show_step("summarize")
        await pilot.pause()
        await asyncio.gather(
            *(w.wait() for w in app.workers if w.group.startswith("workflows-"))
        )
        await pilot.pause()
        assert app.screen.controller.section == "step:summarize"
        assert not app.screen._busy
        export_frame(
            app, width, height, output, f"workflows-{width}x{height}", cairosvg
        )


def export_frame(app, width, height, output, stem, cairosvg):
    svg = output / f"{stem}.svg"
    title = f"Workflows · {width}×{height}"
    original = app.export_screenshot(title=title, simplify=True)
    if "overview" not in stem:
        painted_text = " ".join(
            "".join(element.itertext())
            for element in ElementTree.fromstring(original).iter(
                "{http://www.w3.org/2000/svg}text"
            )
        )
        assert "Prompt" in painted_text, (width, height, "missing Prompt label")
        assert "prepare.text" in painted_text, (
            width,
            height,
            "missing populated Prompt value",
        )
    checks = [("Run", 3)]
    if width == 160:
        checks.append(("Search workflows", 4.5))
    for label, floor in checks:
        foreground, background, ratio = svg_text_contrast(original, label)
        assert ratio >= floor, (label, foreground, background, ratio)
        print(
            f"{width}x{height} {label}: {foreground} / {background} = {ratio:.4f}:1",
            flush=True,
        )
    svg.with_suffix(".textual.svg").write_text(
        "\n".join(line.rstrip() for line in original.splitlines()) + "\n"
    )
    # Textual calls this public Rich method but exposes no format
    # argument itself. Override only for this synchronous export.
    with patch.object(Console, "export_svg", export_menlo):
        exported = app.export_screenshot(title=title, simplify=True)
    trees = [ElementTree.fromstring(value) for value in (original, exported)]
    for tree in trees:
        tree.remove(tree.find("{http://www.w3.org/2000/svg}style"))
    assert ElementTree.tostring(trees[0]) == ElementTree.tostring(trees[1])
    # Strip whitespace-only XML indentation, not rendered content.
    svg.write_text("\n".join(line.rstrip() for line in exported.splitlines()) + "\n")
    png = svg.with_suffix(".png")
    cairosvg.svg2png(url=str(svg), write_to=str(png))
    print(f"{width}x{height}: {svg} -> {png}", flush=True)


if __name__ == "__main__":
    with TemporaryDirectory(prefix="workflow-authoring-fontcache-") as font_cache:
        os.environ["FONTCONFIG_FILE"] = str(Path(__file__).with_name("fontconfig.xml"))
        os.environ["FONTCONFIG_PATH"] = str(Path(__file__).resolve().parent)
        os.environ["XDG_CACHE_HOME"] = font_cache
        for pattern in ("Fira Code", "Fira Code:style=Bold", "monospace"):
            match = subprocess.run(
                [
                    "/opt/homebrew/bin/fc-match",
                    "--format",
                    "%{family}|%{file}|%{spacing}",
                    pattern,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            assert not match.stderr, match.stderr
            assert match.stdout == "Menlo|/System/Library/Fonts/Menlo.ttc|100", (
                match.stdout
            )
            print(f"{pattern} -> {match.stdout}", flush=True)
        asyncio.run(capture())
