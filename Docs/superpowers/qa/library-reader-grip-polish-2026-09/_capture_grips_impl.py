"""Capture production-shaped Library Media grip frames at supported sizes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import cairosvg

from Tests.UI.test_library_media_reader_shell import (
    _build_media_test_app,
    _open_media_shell,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _wait_for_condition,
)
from tldw_chatbook.Widgets.Library import LibraryMediaReaderShell
from tldw_chatbook.Widgets.Library.library_adaptive_reader_shell import (
    LIBRARY_ARROW_UPPER_POSITION_RATIO,
)


OUT = Path(__file__).parent
SIZES = ((160, 50), (120, 35), (100, 30), (80, 24))


def _painted_rows(host, grip, token: str) -> list[int]:
    """Return grip-relative compositor rows containing ``token``."""
    strips = list(host.screen._compositor.render_strips())
    return [
        y - grip.region.y
        for y in range(grip.region.y, grip.region.bottom)
        if token in strips[y].crop(grip.region.x, grip.region.right).text
    ]


def _background_cells(host, grip) -> tuple[tuple[object, ...], ...]:
    """Return the final background color for every painted grip cell."""
    strips = list(host.screen._compositor.render_strips())
    rows: list[tuple[object, ...]] = []
    for y in range(grip.region.y, grip.region.bottom):
        cells: list[object] = []
        for segment in strips[y].crop(grip.region.x, grip.region.right):
            background = segment.style.bgcolor if segment.style is not None else None
            cells.extend([background] * len(segment.text))
        rows.append(tuple(cells))
    return tuple(rows)


async def capture() -> None:
    """Write focused PNG/SVG captures plus a painted-geometry ledger."""
    ledger: list[dict[str, object]] = []
    for width, height in SIZES:
        host = LibraryProductionCSSHarness(_build_media_test_app())
        async with host.run_test(size=(width, height)) as pilot:
            screen, shell = await _open_media_shell(host, pilot)
            shell.items.query_one("#library-media-row-0").press()
            await _wait_for_condition(
                pilot,
                lambda: screen._library_media_reader_session.pending_request is None
                and screen._library_media_reader_session.loaded_id is not None
                and screen._library_media_reader_session.selected_id
                == screen._library_media_reader_session.loaded_id,
                message=f"Reader did not settle at {width}x{height}.",
            )
            await pilot.pause()

            shell = screen.query_one(
                "#library-media-reader-shell", LibraryMediaReaderShell
            )
            library_grip = shell.library_grip
            items_grip = shell.items_grip
            rest_backgrounds = _background_cells(host, library_grip)
            screen.set_focus(library_grip, scroll_visible=False)
            await pilot.pause()

            last_row = library_grip.region.height - 1
            expected_upper = round(
                last_row * LIBRARY_ARROW_UPPER_POSITION_RATIO
            )
            library_rows = _painted_rows(host, library_grip, library_grip.label.plain)
            items_rows = _painted_rows(host, items_grip, items_grip.label.plain)
            assert library_rows == [expected_upper, last_row - expected_upper]
            assert items_rows in ([last_row // 2], [(last_row + 1) // 2])
            assert _background_cells(host, library_grip) == rest_backgrounds
            # task-31276 retired the accent end-caps: the grip is as tall as
            # the shell, so outline-top painted a five-cell rule across the
            # Reader's own first row at the pane join. Focus is the arrow's
            # colour and weight now, and no edge carries an outline.
            focused_endcaps = [
                edge
                for edge in ("top", "bottom", "left", "right")
                if getattr(library_grip.styles, f"outline_{edge}")[0]
                not in {"", "none"}
            ]
            assert focused_endcaps == [], focused_endcaps
            assert not library_grip.get_visual_style().reverse
            assert shell.content_region.contains_region(shell.reader.region)
            assert all(
                shell.content_region.contains_region(child.region)
                for child in shell.children
                if child.display
            )

            stem = f"library-reader-grips-focused-{width}x{height}"
            svg = host.export_screenshot(
                title=f"Library reader grips focused {width}x{height}", simplify=True
            )
            clean_svg = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
            svg_path = OUT / f"{stem}.svg"
            png_path = OUT / f"{stem}.png"
            svg_path.write_text(clean_svg, encoding="utf-8")
            cairosvg.svg2png(
                bytestring=clean_svg.encode("utf-8"), write_to=str(png_path)
            )

            layout = shell.effective_layout
            ledger.append(
                {
                    "terminal": [width, height],
                    "shell": [shell.region.width, shell.region.height],
                    "effective_panes": {
                        "library_open": layout.library_open,
                        "items_open": layout.items_open,
                    },
                    "grip_widths": [
                        library_grip.region.width,
                        items_grip.region.width,
                    ],
                    "library_arrow_rows": library_rows,
                    "items_arrow_rows": items_rows,
                    "focused_background_unchanged": True,
                    "focused_endcaps": focused_endcaps,
                    "visible_children_inside_shell": True,
                    "reader_inside_shell": True,
                }
            )

    (OUT / "geometry.json").write_text(
        json.dumps(ledger, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    asyncio.run(capture())
