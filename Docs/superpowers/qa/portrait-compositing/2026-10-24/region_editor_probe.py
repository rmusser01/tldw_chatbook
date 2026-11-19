"""Actual token-backed modal captures with existing artwork; no provider or DB writes."""

import asyncio
import json
from pathlib import Path

import cairosvg
from textual.widgets import Button, Input, Select

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.Character_Chat.portrait_compositing import (
    PixelRect,
    PortraitCompositionRecipe,
    prepared_base_pixel_sha256,
)
from tldw_chatbook.Character_Chat.portrait_performance_authoring import (
    performance_reference,
)
from tldw_chatbook.css.Themes.themes import ThemeVariableDefaultsMixin
from tldw_chatbook.Widgets.Persona_Widgets.portrait_region_editor import (
    PortraitRegionEditor,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


class CaptureApp(ThemeVariableDefaultsMixin, ConsolidatedCSSApp):
    pass


async def main() -> None:
    """Capture stable focus/scroll states using existing portrait artwork."""
    source = REPO / "tldw_chatbook/assets/characters/samira/expressions/neutral.webp"
    import hashlib

    raw = source.read_bytes()
    reference = performance_reference(raw, hashlib.sha256(raw).hexdigest())
    recipe = PortraitCompositionRecipe(
        1,
        reference.width,
        reference.height,
        prepared_base_pixel_sha256(reference.content),
        PixelRect(526, 330, 86, 58),
        PixelRect(635, 344, 70, 58),
        PixelRect(560, 440, 110, 85),
        True,
    )
    observations = {}
    for name, size in [("wide", (140, 72)), ("narrow", (48, 34))]:
        app = CaptureApp(css_path=[BUNDLED_STYLESHEET])
        app.no_color = False
        app.console.no_color = False
        app._filters = [
            item
            for item in app._filters
            if type(item).__name__ not in {"Monochrome", "NoColor"}
        ]
        async with app.run_test(size=size) as pilot:

            async def save(_):
                return True

            modal = PortraitRegionEditor(
                reference.content,
                width=reference.width,
                height=reference.height,
                pixel_sha256=recipe.base_pixel_sha256,
                recipe=recipe,
                save=save,
                is_current=lambda: True,
            )
            await app.push_screen(modal)
            modal.query_one("#portrait-regions-role", Select).value = "mouth"
            async with asyncio.timeout(15):
                while modal._transform is None or (
                    modal._preview_task and not modal._preview_task.done()
                ):
                    await pilot.pause()
            await pilot.pause()
            states = (
                [("source", "#portrait-regions-role")]
                if name == "wide"
                else [
                    ("fields", "#portrait-region-x"),
                    ("save", "#portrait-regions-save"),
                ]
            )
            for stage, focus in states:
                modal.query_one(focus).focus()
                await pilot.pause()
                await pilot.wait_for_scheduled_animations()
                svg = app.export_screenshot(
                    title=f"Portrait regions · {name} · {stage}"
                )
                target = HERE / f"regions-{name}-{stage}"
                target.with_suffix(".svg").write_text(svg)
                cairosvg.svg2png(
                    bytestring=svg.replace("Fira Code", "Menlo").encode(),
                    write_to=str(target.with_suffix(".png")),
                )
            observations[name] = {
                "size": size,
                "compact": modal.has_class("region-compact"),
                "source_transform": str(modal._transform),
                "save_enabled": not modal.query_one(
                    "#portrait-regions-save", Button
                ).disabled,
                "fields": {field.id: str(field.region) for field in modal.query(Input)},
                "rectangles": str(modal._rectangles),
            }
            modal.action_cancel()
    (HERE / "region-editor-observations.json").write_text(
        json.dumps(observations, indent=2)
    )


if __name__ == "__main__":
    asyncio.run(main())
