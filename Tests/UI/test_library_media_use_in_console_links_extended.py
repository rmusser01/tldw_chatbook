"""TASK-34000.24, extended arms (outside the UI PR-gate lane).

Review Focus 5: a very long workspace name. The receipt row under the
Reader's primary toolbar must stay ONE row -- the text ellipsises -- and the
"Undo link" button must keep its own cells, at the narrow 80x24 Reader as
well as the wide one. The lane file pins the behaviour at the standard size;
these arms add the sizes and the name length.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_library_media_use_in_console_links import (
    _media_host,
    _memberships,
    _open_media_reader,
    _painted,
    _press_use_in_console,
    _receipt_widget_text,
)
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

LONG_NAME = "Thesis chapter four interview analysis pass"  # 43 characters


@pytest.mark.parametrize("size", ((80, 24), LIBRARY_TEST_SIZE))
async def test_a_long_workspace_name_keeps_the_receipt_on_one_row(size):
    host, app = _media_host()
    registry = app.workspace_registry_service
    long = registry.create_workspace(workspace_id="ws-long-name", name=LONG_NAME)
    registry.set_active_workspace(long.workspace_id)

    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_media_reader(screen, pilot)

        await _press_use_in_console(screen, pilot)

        assert [row.workspace_id for row in _memberships(app)] == ["ws-long-name"]
        full_line = f"Linked to {LONG_NAME} · staged in Console"
        # The widget carries the whole line; what is PAINTED is the head of
        # it, ellipsised to the row the Reader can give it beside "Undo link"
        # (at 170x48 the Reader row has ~45 cells for it, fewer than the
        # 53-character head "Linked to <name>"), so the painted check is the
        # prefix and the first word of the name, never the full sentence.
        await _wait_for_condition(
            pilot,
            lambda: _receipt_widget_text(screen) == full_line
            and "Linked to Thesis" in _painted(screen),
            message=lambda: (
                f"long-name receipt never painted; widget="
                f"{_receipt_widget_text(screen)!r}"
            ),
        )
        receipt = screen.query_one("#library-media-link-receipt", Static)
        undo = screen.query_one("#library-media-link-undo", Button)
        assert receipt.region.height == 1, receipt.region
        assert undo.region.height == 1 and undo.region.width >= len("Undo link"), (
            undo.region
        )
        assert receipt.region.y == undo.region.y, (receipt.region, undo.region)
        assert undo.region.right <= screen.size.width
        painted = _painted(screen)
        assert "Undo link" in painted
        assert "ws-long-name" not in painted
