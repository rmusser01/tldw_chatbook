"""Capture production-CSS Library Media Reader frames at supported sizes."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
QA_CONFIG = Path("/private/tmp/tldw-chatbook-library-media-reader-qa/config.toml")
QA_CONFIG.parent.mkdir(parents=True, exist_ok=True)
os.environ["TLDW_CONFIG_PATH"] = str(QA_CONFIG)

from Tests.UI.test_library_media_reader_shell import (  # noqa: E402
    _build_media_test_app,
    _open_media_shell,
)
from Tests.UI.test_library_shell import (  # noqa: E402
    LibraryProductionCSSHarness,
    _wait_for_condition,
)
from tldw_chatbook.Widgets.Library import LibraryMediaReaderShell  # noqa: E402


OUT = Path(__file__).parent
SIZES = ((160, 50), (120, 35), (100, 30), (80, 24))


async def capture() -> None:
    """Write SVG/PNG captures plus the resolved pane geometry ledger."""
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
            layout = shell.effective_layout
            assert shell.content_region.contains_region(shell.reader.region)
            assert shell.library_grip.region.width == 5
            assert shell.items_grip.region.width == 5
            assert shell.reader.region.width > 0

            stem = f"library-media-reader-{width}x{height}"
            svg = host.export_screenshot(
                title=f"Library Media Reader {width}x{height}", simplify=True
            )
            (OUT / f"{stem}.svg").write_text(svg, encoding="utf-8")
            ledger.append(
                {
                    "terminal": [width, height],
                    "shell_width": shell.region.width,
                    "library_open": layout.library_open,
                    "items_open": layout.items_open,
                    "library_width": layout.library_width,
                    "items_width": layout.items_width,
                    "reader_width": layout.reader_width,
                    "reader_inside_shell": True,
                    "grip_widths": [
                        shell.library_grip.region.width,
                        shell.items_grip.region.width,
                    ],
                }
            )

    (OUT / "geometry.json").write_text(
        json.dumps(ledger, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    asyncio.run(capture())
