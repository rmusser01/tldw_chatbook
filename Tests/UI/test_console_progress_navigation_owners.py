"""Progress counts follow each navigation owner without retaining stale counts."""

from dataclasses import replace

import pytest

from Tests.UI.test_console_character_context import (
    _CharacterIdentityApp,
    _controller,
    _resolved,
)
from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    LocalCharacterConversationTarget,
    ResolvedLocalCharacterKey,
)
from tldw_chatbook.Widgets.Console.console_character_context import (
    CharacterConversationButton,
    ConsoleCharacterContext,
)
from tldw_chatbook.Widgets.Console.console_workspace_tree import ConsoleWorkspaceTree
from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
    build_console_conversation_browser_state,
)
from tldw_chatbook.Workspaces.workspace_tree_state import build_workspace_tree_state


def test_workspace_counts_use_native_owner_and_clear_after_owner_release():
    row = ConsoleConversationBrowserInputRow(
        row_key="saved",
        conversation_id="saved",
        native_session_id="native",
        title="[literal]",
        scope_type="workspace",
        workspace_id="named",
        workspace_label="Named",
    )

    def project(counts, current=row):
        return build_workspace_tree_state(
            workspaces=[("named", "Named")], rows=[current], progress_counts=counts
        )[0].conversations[0]

    flat = build_console_conversation_browser_state(
        rows=[row], active_workspace_id="named", progress_counts={"native": 8}
    )
    assert not any(section.rows for section in flat.sections)
    projected = project({"native": 8, "saved": 99})
    assert projected.progress_count == 8
    label = ConsoleWorkspaceTree._conversation_label(projected)
    assert label.plain == "Progress: 8 · [literal]"
    assert not label.spans
    assert project({}).progress_count == 0
    assert (
        project(
            {"native": 8}, replace(row, native_session_id="replacement")
        ).progress_count
        == 0
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("search", [False, True])
async def test_character_browse_and_search_refresh_progress_without_collection(search):
    conversation_id = "search-0" if search else "1-0"
    counts = {conversation_id: 3}
    controller = _controller(progress_counts=lambda: dict(counts))
    app = _CharacterIdentityApp(controller)
    async with app.run_test(size=(80, 35)) as pilot:
        await pilot.pause()
        if search:
            await controller.search("needle")
        else:
            controller._publish(
                replace(controller.state, expanded_key=controller.state.groups[0].key)
            )
        await pilot.pause()
        widget = app.query_one(ConsoleCharacterContext)
        widget._sync_progress_counts()
        button = next(
            item
            for item in widget.query(CharacterConversationButton)
            if item.character_row.target.conversation_id == conversation_id
        )
        assert str(button.label).startswith("Progress: 3 · ")
        await pilot.pause()
        region = button.region
        painted = "\n".join(
            strip.crop(region.x, region.right).text
            for strip in app.screen._compositor.render_strips()[region.y : region.bottom]
        )
        assert "Progress: 3" in painted
        assert counts == {conversation_id: 3}
        counts.clear()
        widget._sync_progress_counts()
        assert "Progress:" not in str(button.label)
        counts[conversation_id] = 5
        controller._database_accessor = lambda: object()
        widget._sync_progress_counts()
        assert "Progress:" not in str(button.label)


@pytest.mark.asyncio
async def test_character_foreign_authority_never_receives_same_id_count():
    controller = _controller(progress_counts=lambda: {"same-id": 4})
    await controller.refresh()
    row = _resolved(1, "same-id")
    assert controller.pending_progress_count(row) == 4
    foreign = replace(
        row,
        target=LocalCharacterConversationTarget(
            ResolvedLocalCharacterKey("foreign", 1), "same-id"
        ),
    )
    assert controller.pending_progress_count(foreign) == 0
