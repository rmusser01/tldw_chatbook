"""Covered backup screens do not spin a retained Console section's layout."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_BOUNDED = r'''
import asyncio
import sys

from textual.screen import Screen
from textual.widgets import Static

from Tests.UI.test_console_bounded_section import _Harness, _lines, _settle, _workspace
from tldw_chatbook.UI.Console_Modules.left_rail import _ContextBoundedSection
from tldw_chatbook.Widgets.Console.console_workspace_tree import ConsoleWorkspaceTree

mode, route = sys.argv[1:]

class Owner:
    allocation_requests = 0
    def request_allocation_reconcile(self):
        self.allocation_requests += 1
    def recover_section_focus(self, section_id):
        pass

async def main():
    owner = Owner()
    tree = None
    content = Static(_lines(30 if mode == 'wrapped' else 2))
    if mode == 'native':
        tree = ConsoleWorkspaceTree()
        tree.sync_projection(
            (_workspace('w', 'Workspace',
                *((f'c{i}', f'Conversation {i}') for i in range(29))),),
            expanded_workspace_ids={'w'})
    section = _ContextBoundedSection(content, section_id='workspace',
        owner=owner, native_scroll_owner=tree)
    section.set_allocation(8)
    reconciles = []
    original = section._reconcile
    def reconcile():
        reconciles.append(True)
        return original()
    section._reconcile = reconcile
    app = _Harness(section)
    async with app.run_test(size=(60,30)) as pilot:
        await _settle(pilot)
        viewport = section._viewport
        viewport.scroll_to(y=4, animate=False, immediate=True)
        await _settle(pilot)
        initial_height = viewport.content_region.height
        assert initial_height == (8 if mode == 'wrapped' else 6)
        assert viewport.scroll_y == 4
        owner.allocation_requests = 0
        if route != 'hidden':
            middle = Screen(Static('Backup destination'))
            middle.styles.background = 'transparent' if route == 'transparent' else '#000000'
            await app.push_screen(middle)
            await app.push_screen(Screen(Static('Backup options')))
            await pilot.pause()
            assert section.screen.is_current == (route == 'transparent')
        else:
            section.set_presented(False)
            await pilot.pause()
        section.set_allocation(5)
        content.update(_lines(45 if mode == 'wrapped' else 4))
        section.request_scoped_reconcile()
        if route in ('covered', 'unmount'):
            await asyncio.sleep(.03)
            assert viewport.content_region.height == initial_height
            section.request_reconcile()
            section.request_scoped_reconcile()
            await asyncio.sleep(.03)
            before = len(reconciles)
            await asyncio.sleep(.04)
            assert len(reconciles) == before, 'covered section retries without an owner layout'
            assert not section._reconcile_scoped, 'scoped request narrowed a pending full pass'
        if route == 'unmount':
            await section.remove()
            before = len(reconciles)
            await app.pop_screen()
            await app.pop_screen()
            await _settle(pilot)
            assert len(reconciles) == before
            assert not section.is_attached
            return
        if route == 'hidden':
            section.set_presented(True)
        else:
            await app.pop_screen()
            await app.pop_screen()
        await _settle(pilot)
        assert viewport.content_region.height == (5 if mode == 'wrapped' else 1)
        assert section.region.height == 6
        assert viewport.scroll_y == 4
        assert section._hint.display
        assert viewport.can_focus
        assert owner.allocation_requests > 0, 'real changed demand never reached the allocator'
        assert not section._reconcile_scheduled
        before = len(reconciles)
        await asyncio.sleep(.04)
        assert len(reconciles) == before

asyncio.run(main())
print('retired and reopened')
'''


@pytest.mark.parametrize('mode', ['wrapped', 'native'])
@pytest.mark.parametrize('route', ['covered', 'transparent', 'hidden', 'unmount'])
def test_bounded_section_waits_for_owner_layout(tmp_path, mode, route):
    """Retained sections resume height, scrolling, focus and allocator updates."""
    _run(tmp_path, mode, route, script=_BOUNDED)
