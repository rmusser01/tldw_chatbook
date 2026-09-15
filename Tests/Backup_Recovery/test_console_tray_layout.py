"""Retained Console trays wait for layout while backup screens cover them."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_LAYOUT = r'''
import asyncio
import sys
from dataclasses import replace

from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Widgets.Console.console_workspace_context import (
    ConsoleWorkspaceContextTray,
)
from tldw_chatbook.Workspaces.display_state import ConsoleWorkspaceContextState

route = sys.argv[1]
state = ConsoleWorkspaceContextState(heading='Workspace', workspace_label='Local',
    authority_label='Local', sync_label='Local', runtime_label='None',
    conversation_rows=(), conversation_empty_copy='No conversations',
    change_workspace_enabled=False, change_workspace_recovery='',
    new_conversation_enabled=False, new_conversation_recovery='', recovery_copy='')

class ObservedTray(ConsoleWorkspaceContextTray):
    fits = 0
    def _fit_height_to_content(self):
        self.fits += 1
        return super()._fit_height_to_content()

class LayoutApp(ConsolidatedCSSApp):
    CSS = """
    #console-left-rail-body { width: 30; scrollbar-gutter: stable; }
    .console-rail-section-body { height: auto; }
    #following-section { height: 1; }
    """
    def compose(self):
        with VerticalScroll(id='console-left-rail-body'):
            with Vertical(classes='console-rail-section-body', id='section'):
                yield ObservedTray(state, content='workspace', id='tray')
            yield Static('Following section', id='following-section')

async def main():
    app = LayoutApp()
    async with app.run_test(size=(80,40)) as pilot:
        tray = app.query_one(ObservedTray)
        section = app.query_one('#section')
        owner = tray.screen
        await pilot.pause()
        initial_height = tray.region.height
        assert initial_height > 1
        if route in ('covered', 'unmount', 'transparent'):
            middle = Screen(Static('Backup destination'))
            middle.styles.background = '#000000' if route != 'transparent' else 'transparent'
            await app.push_screen(middle)
            await app.push_screen(Screen(Static('Backup options')))
            await pilot.pause()
            assert owner.is_current == (route == 'transparent')
        elif route == 'hidden':
            section.display = False
            await pilot.pause()

        # Real recomposition replaces the children; no geometry or layout
        # method is mocked. Repeat while covered to exercise coalescing.
        for count in (35, 45):
            tray.state = replace(state, recovery_copy='\n'.join(['Recovery detail'] * count))
            await tray.recompose()
            tray._schedule_recomposed_content_fit()
            if route in ('covered', 'unmount'):
                assert tray.region.height == initial_height
                assert all(child.virtual_region.height == 0 for child in tray.children)
                await asyncio.sleep(.03)
                before = tray.fits
                await asyncio.sleep(.04)
                assert tray.fits == before, 'covered tray repeatedly fits without an owner layout'

        if route == 'unmount':
            await tray.remove()
            before = tray.fits
            await app.pop_screen()
            await app.pop_screen()
            await pilot.pause()
            assert tray.fits == before
            assert not tray.is_attached
            return
        if route in ('covered', 'transparent'):
            await app.pop_screen()
            await app.pop_screen()
        elif route == 'hidden':
            section.display = True
        await pilot.pause()
        await pilot.pause()

        # Same-size return must grow the tray and keep its following section
        # reachable; stopping the retry alone would leave stale clipping.
        recovery = tray.query_one('#console-workspace-recovery')
        assert recovery.virtual_region.height == 45
        assert tray.region.height > initial_height
        assert tray.region.height >= recovery.virtual_region.bottom
        following = app.query_one('#following-section')
        assert following.virtual_region.y >= section.virtual_region.bottom
        scroll = app.query_one('#console-left-rail-body')
        scroll.scroll_end(animate=False)
        await pilot.pause()
        assert following.region.overlaps(scroll.content_region)
        before = tray.fits
        await asyncio.sleep(.04)
        assert tray.fits == before, 'settled layout keeps scheduling fit callbacks'

asyncio.run(main())
print('retired and reopened')  # Shared subprocess helper's completion sentinel.
'''


@pytest.mark.parametrize('route', ['covered', 'transparent', 'hidden', 'visible', 'unmount'])
def test_tray_recomposition_waits_for_its_screen_layout(tmp_path, route):
    """Background changes settle when shown and leave no callback after removal."""
    _run(tmp_path, route, '', script=_LAYOUT)
