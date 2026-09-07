"""Mounted interaction contract for the shared Prompt/Recipe block editor."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Checkbox, Collapsible, Input, Select, Static, TextArea

from tldw_chatbook.Prompt_Management.prompt_artifact_models import (
    BlockArtifactDefinition,
    PromptBlock,
    PromptLane,
)
from tldw_chatbook.Widgets.Prompts.prompt_block_editor import PromptBlockEditor
from tldw_chatbook.Widgets.Prompts.prompt_block_editor_state import (
    ADDITIONAL_CONTEXT_RESERVED_PREFIX,
    PromptBlockEditorState,
    delete_block,
    update_block,
)


LONG_CONTEXT = "\n".join(f"evidence line {index}" for index in range(30))


def _state(
    *,
    system_content: str = "Be exact.",
    user_content: str = "Explain the result.",
    context_content: str = LONG_CONTEXT,
    mapped_additional_context: bool = False,
) -> PromptBlockEditorState:
    user_blocks = [
        PromptBlock(
            id="goal",
            title="Goal",
            syntax="freeform",
            content=user_content,
        ),
        PromptBlock(
            id="context",
            title="Context",
            syntax="freeform",
            content=context_content,
        ),
    ]
    if mapped_additional_context:
        user_blocks.append(
            PromptBlock(
                id=ADDITIONAL_CONTEXT_RESERVED_PREFIX,
                title="Additional context",
                syntax="markdown",
                content="Unmatched evidence.",
            )
        )
    definition = BlockArtifactDefinition(
        kind="block_prompt",
        schema_version=2,
        lanes=(
            PromptLane(
                id="system",
                blocks=(
                    PromptBlock(
                        id="role",
                        title="Role",
                        syntax="markdown",
                        content=system_content,
                    ),
                ),
            ),
            PromptLane(
                id="user",
                blocks=tuple(user_blocks),
            ),
        ),
    )
    return PromptBlockEditorState.from_definition(
        artifact_type="prompt",
        definition=definition,
    )


class BlockEditorHarness(App[None]):
    def __init__(
        self,
        state: PromptBlockEditorState,
        *,
        can_update_original: bool = False,
        embedded: bool = False,
        host_owned_lifecycle: bool = False,
        initially_hidden_block_ids: frozenset[str] = frozenset(),
    ) -> None:
        super().__init__()
        self.state = state
        self.can_update_original = can_update_original
        self.embedded = embedded
        self.host_owned_lifecycle = host_owned_lifecycle
        self.initially_hidden_block_ids = initially_hidden_block_ids
        self.messages: list[object] = []

    def compose(self) -> ComposeResult:
        yield PromptBlockEditor(
            self.state,
            can_update_original=self.can_update_original,
            embedded=self.embedded,
            host_owned_lifecycle=self.host_owned_lifecycle,
            initially_hidden_block_ids=self.initially_hidden_block_ids,
            id="editor",
        )

    def on_prompt_block_editor_block_field_changed(
        self, message: PromptBlockEditor.BlockFieldChanged
    ) -> None:
        self.messages.append(message)

    def on_prompt_block_editor_apply_requested(
        self, message: PromptBlockEditor.ApplyRequested
    ) -> None:
        self.messages.append(message)

    def on_prompt_block_editor_block_action_requested(
        self, message: PromptBlockEditor.BlockActionRequested
    ) -> None:
        self.messages.append(message)

    def on_prompt_block_editor_back_requested(
        self, message: PromptBlockEditor.BackRequested
    ) -> None:
        self.messages.append(message)

    def on_prompt_block_editor_save_as_prompt_requested(
        self, message: PromptBlockEditor.SaveAsPromptRequested
    ) -> None:
        self.messages.append(message)

    def on_prompt_block_editor_save_as_recipe_requested(
        self, message: PromptBlockEditor.SaveAsRecipeRequested
    ) -> None:
        self.messages.append(message)

    def on_prompt_block_editor_update_original_requested(
        self, message: PromptBlockEditor.UpdateOriginalRequested
    ) -> None:
        self.messages.append(message)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (80, 24)])
async def test_lanes_stack_and_nonempty_lanes_start_expanded(
    size: tuple[int, int],
) -> None:
    app = BlockEditorHarness(_state())

    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        editor = app.query_one("#editor", PromptBlockEditor)
        system = app.query_one("#prompt-lane-system", Collapsible)
        user = app.query_one("#prompt-lane-user", Collapsible)

        assert system.region.y < user.region.y
        assert system.collapsed is False
        assert user.collapsed is False
        assert editor.has_class("-narrow") is (size[0] < 90)
        assert editor.query_one("#prompt-editor-footer").has_class("two-row") is (
            size[0] < 90
        )
        assert len(editor.query(".prompt-block-card")) == 3
        for selector in (
            "#prompt-editor-apply",
            "#prompt-editor-save-menu",
        ):
            region = editor.query_one(selector).region
            assert region.x >= 0 and region.right <= size[0]
            assert region.y >= 0 and region.bottom <= size[1]
        assert not editor.query("#prompt-editor-save-prompt")
        assert not editor.query("#prompt-editor-save-recipe")
        assert not editor.query("#prompt-editor-update-original")


@pytest.mark.asyncio
async def test_optional_blocks_use_keyboard_reachable_progressive_disclosure() -> None:
    app = BlockEditorHarness(
        _state(),
        initially_hidden_block_ids=frozenset({"role", "context"}),
    )

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()

        summary = app.query_one("#prompt-editor-guided-summary", Static)
        reveal = app.query_one("#prompt-editor-show-optional", Button)
        assert "Start with the four essentials" in str(summary.renderable)
        assert "Show 2 optional blocks" == str(reveal.label)
        assert app.query_one("#prompt-block-role").display is False
        assert app.query_one("#prompt-block-context").display is False

        reveal.focus()
        await pilot.press("enter")
        await pilot.pause()

        assert reveal.display is False
        assert app.query_one("#prompt-block-role").display is True
        assert app.query_one("#prompt-block-context").display is True
        assert "Optional blocks are shown" in str(summary.renderable)


@pytest.mark.asyncio
async def test_genuinely_wide_editor_keeps_readable_single_row_footer() -> None:
    app = BlockEditorHarness(_state())

    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one("#editor", PromptBlockEditor)
        footer = editor.query_one("#prompt-editor-footer")
        lane_options = editor.query_one("#prompt-editor-lane-options")
        actions = editor.query_one("#prompt-editor-actions")

        assert footer.has_class("two-row") is False
        assert lane_options.region.y == actions.region.y
        for checkbox, label in (
            (
                editor.query_one("#prompt-editor-apply-system", Checkbox),
                "Replace this session's System prompt",
            ),
            (
                editor.query_one("#prompt-editor-apply-user", Checkbox),
                "Apply User",
            ),
        ):
            painted = "\n".join(
                checkbox.render_line(row).text for row in range(checkbox.size.height)
            )
            assert "▐X▌" in painted
            assert label in painted

        await pilot.resize_terminal(100, 30)
        await pilot.pause()
        assert footer.has_class("two-row") is True

        await pilot.resize_terminal(140, 40)
        await pilot.pause()
        assert footer.has_class("two-row") is False
        assert lane_options.region.y == actions.region.y


@pytest.mark.asyncio
async def test_embedded_host_owned_lifecycle_keeps_only_structured_save_menu():
    app = BlockEditorHarness(
        _state(),
        embedded=True,
        host_owned_lifecycle=True,
    )

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one("#editor", PromptBlockEditor)

        assert editor.query_one("#prompt-editor-save-menu", Select).display
        for selector in (
            "#prompt-editor-back",
            "#prompt-editor-apply",
            "#prompt-editor-lane-options",
        ):
            assert editor.query_one(selector).display is False


@pytest.mark.asyncio
async def test_embedded_editor_uses_a_plain_naturally_sized_body() -> None:
    """An embedding scroll owner must not receive a nested block-editor scroll."""
    app = BlockEditorHarness(_state(), embedded=True)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one("#editor", PromptBlockEditor)

        assert editor.has_class("embedded")
        assert isinstance(editor.query_one("#prompt-editor-body"), Vertical)
        assert list(editor.query(VerticalScroll)) == []


@pytest.mark.asyncio
async def test_xml_tag_control_is_visible_only_for_xml_blocks() -> None:
    state = update_block(_state(), "goal", syntax="xml", xml_tag="goal")
    app = BlockEditorHarness(state)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        assert app.query_one("#prompt-block-xml-tag-goal", Input).display is True
        assert app.query_one("#prompt-block-xml-tag-context", Input).display is False


@pytest.mark.asyncio
async def test_clearing_xml_tag_keeps_editor_open_with_recovery_issue() -> None:
    state = update_block(_state(), "goal", syntax="xml", xml_tag="goal")
    app = BlockEditorHarness(state)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        xml_tag = app.query_one("#prompt-block-xml-tag-goal", Input)
        xml_tag.value = ""
        await pilot.pause()

        editor = app.query_one("#editor", PromptBlockEditor)
        assert editor.state.definition.lanes[1].blocks[0].xml_tag == ""
        assert editor.state.definition.lanes[1].blocks[0].content == (
            "Explain the result."
        )
        assert "Invalid" in str(
            app.query_one("#prompt-block-issue-goal", Static).renderable
        )


@pytest.mark.asyncio
async def test_cleared_xml_tag_can_be_duplicated_without_losing_draft() -> None:
    state = update_block(_state(), "goal", syntax="xml", xml_tag="goal")
    app = BlockEditorHarness(state)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one("#prompt-block-xml-tag-goal", Input).value = ""
        await pilot.pause()
        app.query_one("#prompt-block-duplicate-goal", Button).press()
        await pilot.pause()

        editor = app.query_one("#editor", PromptBlockEditor)
        source, copy, _context = editor.state.definition.lanes[1].blocks
        assert source.xml_tag == copy.xml_tag == ""
        assert source.content == copy.content == "Explain the result."
        assert app.query_one("#prompt-block-content-goal-copy", TextArea).text == (
            "Explain the result."
        )
        assert "Invalid" in str(
            app.query_one("#prompt-block-issue-goal-copy", Static).renderable
        )


@pytest.mark.asyncio
async def test_mapped_additional_context_duplicate_click_is_disabled_and_safe() -> None:
    app = BlockEditorHarness(_state(mapped_additional_context=True))

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one("#editor", PromptBlockEditor)
        original_state = editor.state
        duplicate = app.query_one(
            "#prompt-block-duplicate-additional-context",
            Button,
        )

        duplicate.press()
        await pilot.pause()

        assert duplicate.disabled is True
        assert editor.state is original_state
        assert not [
            message
            for message in app.messages
            if isinstance(message, PromptBlockEditor.BlockActionRequested)
            and message.action == "duplicate"
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (80, 24)])
async def test_adding_to_empty_lane_updates_count_expands_and_reveals_card(
    size: tuple[int, int],
) -> None:
    app = BlockEditorHarness(delete_block(_state(), "role"))

    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        system = app.query_one("#prompt-lane-system", Collapsible)
        assert system.collapsed is True
        assert str(system.title) == "System · 0 blocks"

        app.query_one("#prompt-lane-add-system", Button).press()
        await pilot.pause()

        card = app.query_one("#prompt-block-block")
        assert str(system.title) == "System · 1 blocks"
        assert system.collapsed is False
        assert card.display is True


@pytest.mark.asyncio
async def test_reconcile_does_not_expand_an_intentionally_collapsed_nonempty_lane() -> (
    None
):
    app = BlockEditorHarness(_state())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        user = app.query_one("#prompt-lane-user", Collapsible)
        user.collapsed = True
        await pilot.pause()

        app.query_one("#prompt-lane-add-system", Button).press()
        await pilot.pause()

        assert user.collapsed is True
        assert str(user.title) == "User · 2 blocks"


@pytest.mark.asyncio
async def test_validation_is_visible_beside_block_and_action_focuses_first_error() -> (
    None
):
    state = update_block(_state(), "goal", syntax="xml", xml_tag="bad tag")
    app = BlockEditorHarness(state)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        issue = app.query_one("#prompt-block-issue-goal", Static)
        editor = app.query_one("#editor", PromptBlockEditor)

        assert "Invalid" in str(issue.renderable)
        assert "XML tag" in str(issue.renderable)
        assert app.query_one("#prompt-editor-apply", Button).disabled

        editor.action_apply()
        await pilot.pause()
        assert app.focused is app.query_one("#prompt-block-xml-tag-goal", Input)


@pytest.mark.asyncio
async def test_update_original_validation_reason_explains_recovery() -> None:
    invalid = update_block(_state(), "goal", syntax="xml", xml_tag="bad tag")
    app = BlockEditorHarness(invalid, can_update_original=True)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()

        assert app.query_one("#prompt-editor-save-menu", Select).disabled
        reason = str(
            app.query_one("#prompt-editor-update-reason", Static).renderable
        ).lower()
        assert "resolve" in reason
        assert "block" in reason


@pytest.mark.asyncio
async def test_update_original_source_unavailable_reason_is_preserved() -> None:
    app = BlockEditorHarness(_state(), can_update_original=False)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        menu = app.query_one("#prompt-editor-save-menu", Select)
        assert "update" not in [
            value for _label, value in menu._options if value is not Select.NULL
        ]
        reason = str(
            app.query_one("#prompt-editor-update-reason", Static).renderable
        ).lower()
        assert "no guarded version update" in reason
        assert "save as new" in reason


@pytest.mark.asyncio
async def test_host_can_enable_guarded_update_without_reconstructing_editor() -> None:
    app = BlockEditorHarness(_state(), can_update_original=False)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one("#editor", PromptBlockEditor)
        menu = app.query_one("#prompt-editor-save-menu", Select)

        assert "update" not in [
            value for _label, value in menu._options if value is not Select.NULL
        ]
        assert (
            "no guarded version update"
            in str(
                app.query_one("#prompt-editor-update-reason", Static).renderable
            ).lower()
        )

        editor.set_update_original_available(True)
        await pilot.pause()

        assert "update" in [
            value for _label, value in menu._options if value is not Select.NULL
        ]
        assert (
            str(app.query_one("#prompt-editor-update-reason", Static).renderable) == ""
        )
        menu.value = "update"
        await pilot.pause()

        assert [type(message) for message in app.messages] == [
            PromptBlockEditor.UpdateOriginalRequested
        ]


@pytest.mark.asyncio
async def test_update_original_source_unavailable_reason_precedes_validation() -> None:
    invalid = update_block(_state(), "goal", syntax="xml", xml_tag="bad tag")
    app = BlockEditorHarness(invalid, can_update_original=False)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()

        assert app.query_one("#prompt-editor-save-menu", Select).disabled
        reason = str(
            app.query_one("#prompt-editor-update-reason", Static).renderable
        ).lower()
        assert "no guarded version update" in reason
        assert "save as new" in reason
        assert "resolve the block errors" not in reason


@pytest.mark.asyncio
async def test_apply_defaults_and_all_empty_disabled_reason_are_explicit() -> None:
    app = BlockEditorHarness(_state())
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        system = app.query_one("#prompt-editor-apply-system", Checkbox)
        user = app.query_one("#prompt-editor-apply-user", Checkbox)

        assert system.value is False
        assert user.value is True
        assert not app.query_one("#prompt-editor-apply", Button).disabled

        await pilot.click("#prompt-editor-apply")
        await pilot.pause()
        [message] = [
            message
            for message in app.messages
            if isinstance(message, PromptBlockEditor.ApplyRequested)
        ]
        assert message.apply_system is False
        assert message.system_prompt is None
        assert message.apply_user is True
        assert message.user_prompt == _state().compiled_user

    empty = BlockEditorHarness(
        _state(system_content="", user_content="", context_content="")
    )
    async with empty.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert empty.query_one("#prompt-editor-apply", Button).disabled
        reason = empty.query_one("#prompt-editor-apply-reason", Static)
        assert "add content" in str(reason.renderable).lower()


@pytest.mark.asyncio
async def test_unselecting_nonempty_lanes_is_noop_and_disables_apply_with_recovery() -> (
    None
):
    app = BlockEditorHarness(_state(system_content="", user_content="Draft"))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.click("#prompt-editor-apply-user")
        await pilot.pause()

        assert app.query_one("#prompt-editor-apply", Button).disabled
        reason = app.query_one("#prompt-editor-apply-reason", Static)
        assert "select a non-empty lane" in str(reason.renderable).lower()


@pytest.mark.asyncio
async def test_replace_block_state_patches_only_changed_controls() -> None:
    app = BlockEditorHarness(_state())

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one("#editor", PromptBlockEditor)
        untouched = app.query_one("#prompt-block-content-context", TextArea)
        untouched.insert("!", (0, 0))
        await pilot.pause()
        original_before_insert = LONG_CONTEXT
        untouched.cursor_location = (8, 4)
        untouched.selection = untouched.selection.__class__((7, 1), (8, 4))
        untouched.scroll_to(y=6, animate=False, force=True, immediate=True)
        await pilot.pause()

        original_identity = id(untouched)
        original_cursor = untouched.cursor_location
        original_selection = untouched.selection
        original_scroll = untouched.scroll_offset

        changed = update_block(editor.state, "goal", content="Changed elsewhere")
        await editor.replace_block_state("goal", changed)
        await pilot.pause()

        same = app.query_one("#prompt-block-content-context", TextArea)
        assert id(same) == original_identity
        assert same.cursor_location == original_cursor
        assert same.selection == original_selection
        assert same.scroll_offset == original_scroll

        same.undo()
        await pilot.pause()
        assert same.text == original_before_insert


@pytest.mark.asyncio
async def test_typing_in_one_block_preserves_sibling_textarea_identity_and_state() -> (
    None
):
    app = BlockEditorHarness(_state())

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        untouched = app.query_one("#prompt-block-content-context", TextArea)
        untouched.cursor_location = (5, 3)
        untouched.selection = untouched.selection.__class__((4, 1), (5, 3))
        untouched.scroll_to(y=5, animate=False, force=True, immediate=True)
        await pilot.pause()
        identity = id(untouched)
        cursor = untouched.cursor_location
        selection = untouched.selection
        scroll = untouched.scroll_offset

        goal = app.query_one("#prompt-block-content-goal", TextArea)
        goal.focus()
        app.query_one("#prompt-editor-scroll").scroll_to_widget(goal, animate=False)
        await pilot.pause()
        await pilot.press("end")
        await pilot.press("x")
        await pilot.pause()

        same = app.query_one("#prompt-block-content-context", TextArea)
        assert id(same) == identity
        assert same.cursor_location == cursor
        assert same.selection == selection
        assert same.scroll_offset == scroll
        field_messages = [
            message
            for message in app.messages
            if isinstance(message, PromptBlockEditor.BlockFieldChanged)
        ]
        assert field_messages[-1].block_id == "goal"
        assert field_messages[-1].field == "content"


@pytest.mark.asyncio
async def test_reorder_moves_existing_cards_without_reconstructing_textareas() -> None:
    app = BlockEditorHarness(_state())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one("#editor", PromptBlockEditor)
        goal = app.query_one("#prompt-block-content-goal", TextArea)
        context = app.query_one("#prompt-block-content-context", TextArea)

        app.query_one("#prompt-block-move-down-goal", Button).press()
        await pilot.pause()

        assert app.query_one("#prompt-block-content-goal", TextArea) is goal
        assert app.query_one("#prompt-block-content-context", TextArea) is context
        body = editor.query_one("#prompt-lane-user-blocks")
        assert [
            card.block_id for card in body.children if hasattr(card, "block_id")
        ] == ["context", "goal"]


@pytest.mark.asyncio
async def test_structural_controls_emit_typed_messages_with_immutable_state() -> None:
    app = BlockEditorHarness(_state())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one("#prompt-lane-add-system", Button).press()
        await pilot.pause()
        app.query_one("#prompt-block-duplicate-goal", Button).press()
        await pilot.pause()
        app.query_one("#prompt-block-delete-goal-copy", Button).press()
        await pilot.pause()

        actions = [
            message
            for message in app.messages
            if isinstance(message, PromptBlockEditor.BlockActionRequested)
        ]
        assert [message.action for message in actions] == [
            "add",
            "duplicate",
            "delete",
        ]
        assert all(
            isinstance(message.state, PromptBlockEditorState) for message in actions
        )
        assert actions[-1].state is app.query_one("#editor", PromptBlockEditor).state


@pytest.mark.asyncio
async def test_footer_controls_emit_distinct_typed_navigation_and_save_messages() -> (
    None
):
    app = BlockEditorHarness(_state(), can_update_original=True)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one("#prompt-editor-back", Button).press()
        await pilot.pause()
        save_menu = app.query_one("#prompt-editor-save-menu", Select)
        for value in ("prompt", "recipe", "update"):
            save_menu.value = value
            await pilot.pause()

        assert [type(message) for message in app.messages] == [
            PromptBlockEditor.BackRequested,
            PromptBlockEditor.SaveAsPromptRequested,
            PromptBlockEditor.SaveAsRecipeRequested,
            PromptBlockEditor.UpdateOriginalRequested,
        ]


@pytest.mark.asyncio
async def test_ctrl_s_opens_save_menu_and_keyboard_chooses_first_action() -> None:
    app = BlockEditorHarness(_state())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        menu = app.query_one("#prompt-editor-save-menu", Select)

        await pilot.press("ctrl+s")
        await pilot.pause()

        assert menu.has_focus_within
        assert menu.expanded is True

        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()

        assert menu.expanded is False
        assert [type(message) for message in app.messages] == [
            PromptBlockEditor.SaveAsPromptRequested
        ]


def test_widget_ids_are_stable_from_block_ids_not_titles_or_indexes() -> None:
    first = PromptBlockEditor.widget_token_for_block_id("goal")
    renamed = PromptBlockEditor.widget_token_for_block_id("goal")
    unsafe = PromptBlockEditor.widget_token_for_block_id("server:goal/1")

    assert first == renamed == "goal"
    assert unsafe.startswith("encoded-")
    assert ":" not in unsafe and "/" not in unsafe
    assert unsafe != PromptBlockEditor.widget_token_for_block_id("server:goal/2")


def _count_footer_writes(app, editor: PromptBlockEditor) -> dict[str, list]:
    """Instrument the footer's three Statics and the save-menu tooltip write.

    TASK-22228 item 5. ``Widget.tooltip``'s setter is not a plain
    assignment -- it calls ``self.screen._update_tooltip(self)`` -- so the
    tooltip arm counts through the screen seam the setter actually reaches.
    """
    log: dict[str, list] = {"statics": [], "tooltips": []}
    for static_id in (
        "prompt-editor-validation",
        "prompt-editor-apply-reason",
        "prompt-editor-update-reason",
    ):
        static = editor.query_one(f"#{static_id}", Static)
        original = static.update

        def patched(renderable="", _original=original, _id=static_id, **kwargs):
            log["statics"].append((_id, str(renderable)))
            return _original(renderable, **kwargs)

        static.update = patched  # type: ignore[method-assign]

    screen = app.screen
    original_tooltip = screen._update_tooltip

    def counting_tooltip(widget, _original=original_tooltip):
        log["tooltips"].append(getattr(widget, "id", None))
        return _original(widget)

    screen._update_tooltip = counting_tooltip  # type: ignore[method-assign]
    return log


@pytest.mark.asyncio
async def test_footer_writes_nothing_when_a_keystroke_changes_no_footer_copy() -> None:
    """TASK-22228 item 5: typing re-syncs the footer without repainting it.

    ``_sync_footer`` runs on every keystroke in every block
    (``_change_field`` -> ``replace_block_state``). Before the guards it
    made three ``Static.update`` calls (each ``layout=True``) plus a
    tooltip write per keystroke, every one of them writing exactly the copy
    already rendered: measured 15 updates for 5 keystrokes, all 15 no-ops.
    """
    app = BlockEditorHarness(_state())

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one("#editor", PromptBlockEditor)
        goal = app.query_one("#prompt-block-content-goal", TextArea)
        goal.focus()
        app.query_one("#prompt-editor-scroll").scroll_to_widget(goal, animate=False)
        await pilot.pause()
        await pilot.press("end")
        await pilot.pause()

        log = _count_footer_writes(app, editor)
        for _ in range(5):
            await pilot.press("x")
            await pilot.pause()

        # The keystrokes really did reach the editor (else this is vacuous).
        assert goal.text.endswith("xxxxx"), goal.text
        assert log["statics"] == []
        assert log["tooltips"] == []


@pytest.mark.asyncio
async def test_footer_still_repaints_when_the_copy_actually_changes() -> None:
    """Control arm: a real validation transition writes, and only once.

    The second ``_sync_footer`` for the SAME state must write nothing --
    that is the property the guard depends on (the rendered text reads back
    equal to what was written), and it is what makes the guard safe rather
    than merely quiet.
    """
    app = BlockEditorHarness(_state())

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one("#editor", PromptBlockEditor)
        validation = app.query_one("#prompt-editor-validation", Static)
        before = str(validation.renderable)
        assert before.startswith("Valid ·"), before

        log = _count_footer_writes(app, editor)
        await editor.replace_block_state(
            "goal", delete_block(editor.state, "goal")
        )
        await pilot.pause()

        written = [entry for entry in log["statics"] if entry[0] == "prompt-editor-validation"]
        assert written, log["statics"]
        assert str(validation.renderable) != before
        assert str(validation.renderable) == written[-1][1]

        # ...and a repeat sync for the unchanged state writes nothing.
        log["statics"].clear()
        log["tooltips"].clear()
        editor._sync_footer()
        assert log["statics"] == []
        assert log["tooltips"] == []
