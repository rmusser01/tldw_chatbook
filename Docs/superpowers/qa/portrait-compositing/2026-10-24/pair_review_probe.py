"""Inspect the actual pair viewer using existing generated edits; no provider calls."""

import asyncio
import hashlib
import json

import cairosvg
from region_editor_probe import HERE, REPO, CaptureApp
from textual.widgets import Select

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET
from tldw_chatbook.Character_Chat.portrait_compositing import (
    PixelRect,
    PortraitCompositionRecipe,
    assemble_components,
    compose_component,
    prepared_base_pixel_sha256,
    recover_components,
)
from tldw_chatbook.Character_Chat.portrait_performance_authoring import (
    performance_reference,
)
from tldw_chatbook.Widgets.Persona_Widgets.portrait_composition_review import (
    PortraitCompositionReview,
)


async def main() -> None:
    """Capture original-frame overview/detail and narrow keyboard presentation."""
    source = REPO / "tldw_chatbook/assets/characters/samira/expressions/neutral.webp"
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
    root = REPO / "output/imagegen/console-radio-samira-2026-10-23/ready"
    files = {"eyelids": "blink_rest.png", "small": "small.png", "open": "open.png"}
    components = {
        role: compose_component(
            reference.content, recipe, role, (root / name).read_bytes()
        )
        for role, name in files.items()
    }
    frames = assemble_components(reference.content, recipe, components)
    assert recover_components(reference.content, recipe, frames)
    observations = {
        "source_sha256": reference.file_id,
        "base_pixel_sha256": recipe.base_pixel_sha256,
        "inputs": {
            role: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for role, name in files.items()
        },
        "provider_calls": 0,
        "views": {},
    }
    for name, size, region in [
        ("wide-overview", (140, 72), "overview"),
        ("wide-mouth", (140, 72), "mouth"),
        ("narrow-eye", (48, 34), "left_eye"),
    ]:
        app = CaptureApp(css_path=[BUNDLED_STYLESHEET])
        app.no_color = app.console.no_color = False
        app._filters = [
            item
            for item in app._filters
            if type(item).__name__ not in {"Monochrome", "NoColor"}
        ]
        async with app.run_test(size=size) as pilot:
            review = PortraitCompositionReview(frames, recipe, lambda: True)
            await app.push_screen(review)
            review.query_one("#portrait-pairs-region", Select).value = region
            await pilot.pause()
            await review.wait_for_render()
            review.query_one("#portrait-pairs-canvas").focus()
            await pilot.pause()
            await pilot.wait_for_scheduled_animations()
            assert review._raster is not None
            svg = app.export_screenshot(title=f"Portrait pairs · {name}")
            target = HERE / f"pairs-{name}"
            target.with_suffix(".svg").write_text(svg)
            cairosvg.svg2png(
                bytestring=svg.replace("Fira Code", "Menlo").encode(),
                write_to=str(target.with_suffix(".png")),
            )
            raster = review._raster
            (HERE / f"pairs-{name}-raw-raster.png").write_bytes(raster.png)
            observations["views"][name] = {
                "crop": raster.crop_box,
                "raster_size": [raster.width, raster.height],
                "sha256": hashlib.sha256(raster.png).hexdigest(),
                "roles": raster.roles,
            }
            review.retire()
    (HERE / "pair-review-observations.json").write_text(
        json.dumps(observations, indent=2)
    )


if __name__ == "__main__":
    asyncio.run(main())
