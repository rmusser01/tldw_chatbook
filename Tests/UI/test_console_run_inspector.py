"""TASK-259: ConsoleRunInspector updates changed rows only.

Row-level state changes (same rendered structure, different text/status)
must update the mounted row Statics in place; structural changes (rows or
actions added/removed, dictionary section shape changes) still recompose
the widget -- and only the widget, never the owning screen.
"""

from collections import Counter
from dataclasses import replace
import importlib

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_display_state import (
    ConsoleDisplayRow,
    ConsoleInspectorAction,
    ConsoleInspectorState,
)
from tldw_chatbook.Widgets.Console.console_run_inspector import (
    _ACTION_GROUPS,
    _ROW_GROUPS,
    ConsoleRunInspector,
    inspector_group_is_actionable,
)
from tldw_chatbook.Widgets.Console.console_bounded_section import (
    ConsoleBoundedSection,
)


EXPECTED_ROW_OWNERS = {
    "Run recipe": "Run",
    "Live work": "Run",
    "Setup": "Run",
    "Send blocked": "Run",
    "Recovery action": "Run",
    "Blocked impact": "Run",
    "Next action": "Run",
    "Provider": "Run",
    "Sources": "Source Readiness",
    "RAG/source": "Source Readiness",
    "Evidence": "Source Readiness",
    "Authority": "Source Readiness",
    "Tools": "Tools",
    "MCP": "Tools",
    "Approvals": "Approvals",
    "Artifacts": "Artifacts",
    "Selected conversation": "Selected Conversation",
    "Conversation source": "Selected Conversation",
    "Workspace": "Selected Conversation",
    "Resume state": "Selected Conversation",
    "Prefill (next send only)": "Selected Conversation",
    "Prefill (pinned)": "Selected Conversation",
    "Session provider": "Session Defaults",
    "Session model": "Session Defaults",
    "Session endpoint": "Session Defaults",
    "Session sampling": "Session Defaults",
    "Session persona": "Session Defaults",
    "Selected message": "Selected Message",
    "Message actions": "Selected Message",
    "Keyboard": "Selected Message",
    "Variants": "Selected Message",
    "Excerpt": "Selected Message",
    "Delete confirmation": "Selected Message",
}

EXPECTED_ACTION_OWNERS = {
    "console-inspector-save-chatbook": "Artifacts",
    "console-inspector-review-approval": "Approvals",
    "console-inspector-review-changes": "Changes",
}


def _ownership_module():
    return importlib.import_module(
        "tldw_chatbook.Widgets.Console.console_inspector_ownership"
    )


def _base_state(**overrides) -> ConsoleInspectorState:
    values = {
        "rows": (
            ConsoleDisplayRow("Run recipe", "Chat with provider"),
            ConsoleDisplayRow("Provider", "OpenAI / gpt-4o", status="ready"),
            ConsoleDisplayRow("Sources", "1 staged", status="ready"),
        ),
        "actions": (),
        "dictionary_rows": (),
        "dictionary_actions": (),
    }
    values.update(overrides)
    return ConsoleInspectorState(**values)


class InspectorHarness(App):
    def __init__(
        self,
        state: ConsoleInspectorState,
        *,
        ownership_policy=None,
        more_open: bool = False,
    ) -> None:
        super().__init__()
        self._state = state
        self._more_open = more_open
        self.more_events: list[bool] = []
        ownership = _ownership_module()
        self._ownership_policy = (
            ownership_policy or ownership.InspectorOwnershipPolicy.STRICT
        )

    def compose(self) -> ComposeResult:
        yield ConsoleRunInspector(
            self._state,
            ownership_policy=self._ownership_policy,
            more_open=self._more_open,
            id="inspector",
        )

    def on_console_run_inspector_more_toggled(
        self, event: ConsoleRunInspector.MoreToggled
    ) -> None:
        self.more_events.append(event.open)


def test_inspector_row_and_action_inventory_has_exactly_one_owner():
    ownership = _ownership_module()

    grouped_rows = [
        (label, owner) for owner, _heading_id, labels in _ROW_GROUPS for label in labels
    ]
    grouped_actions = [
        (action_id, owner)
        for owner, action_ids in _ACTION_GROUPS.items()
        for action_id in action_ids
    ]

    assert ownership.ROW_OWNERS == EXPECTED_ROW_OWNERS
    assert set(ownership.ROW_IDS) == set(EXPECTED_ROW_OWNERS)
    assert Counter(label for label, _owner in grouped_rows) == Counter(
        {label: 1 for label in EXPECTED_ROW_OWNERS}
    )
    assert dict(grouped_rows) == EXPECTED_ROW_OWNERS
    assert ownership.ACTION_OWNERS == EXPECTED_ACTION_OWNERS
    assert Counter(action_id for action_id, _owner in grouped_actions) == Counter(
        {action_id: 1 for action_id in EXPECTED_ACTION_OWNERS}
    )
    assert dict(grouped_actions) == EXPECTED_ACTION_OWNERS


@pytest.mark.parametrize(
    ("collection_name", "expected_owner"),
    [
        ("dictionary_rows", "Chat Dictionaries"),
        ("dictionary_actions", "Chat Dictionaries"),
        ("world_book_rows", "World Books"),
        ("world_book_actions", "World Books"),
    ],
)
def test_dynamic_inspector_collection_items_have_explicit_owner(
    collection_name, expected_owner
):
    ownership = _ownership_module()
    item = (
        ConsoleDisplayRow("dynamic label", "dynamic value")
        if collection_name.endswith("rows")
        else ConsoleInspectorAction("dynamic-action", "dynamic action", True)
    )
    state = _base_state(**{collection_name: (item,)})
    classified = ownership.classify_inspector_content(
        state, ownership.InspectorOwnershipPolicy.STRICT
    )
    projected = getattr(classified, collection_name)

    assert ownership.DYNAMIC_COLLECTION_OWNERS[collection_name] == expected_owner
    assert projected[0].owner == expected_owner


@pytest.mark.parametrize(
    "unknown_content",
    [
        {"rows": (ConsoleDisplayRow("Unknown row", "private row text"),)},
        {
            "actions": (
                ConsoleInspectorAction(
                    "console-inspector-unknown-action",
                    "private action copy",
                    True,
                ),
            )
        },
    ],
)
def test_strict_ownership_rejects_unknown_content_before_mount(unknown_content):
    ownership = _ownership_module()
    state = _base_state(**{"rows": (), **unknown_content})

    with pytest.raises(ownership.UnownedInspectorContentError):
        ConsoleRunInspector(
            state,
            ownership_policy=ownership.InspectorOwnershipPolicy.STRICT,
        )


@pytest.mark.asyncio
async def test_strict_sync_rejects_unknown_content_before_replacing_valid_tree():
    ownership = _ownership_module()
    initial = _base_state()
    app = InspectorHarness(initial)

    async with app.run_test(size=(80, 32)):
        inspector = app.query_one("#inspector", ConsoleRunInspector)
        provider_before = inspector.query_one("#console-inspector-provider", Static)

        with pytest.raises(ownership.UnownedInspectorContentError):
            inspector.sync_state(
                _base_state(
                    rows=initial.rows
                    + (ConsoleDisplayRow("Unknown row", "private row text"),)
                )
            )

        assert inspector.state is initial
        assert inspector.recompose_count == 0
        assert (
            inspector.query_one("#console-inspector-provider", Static)
            is provider_before
        )


@pytest.mark.parametrize(
    "state",
    [
        _base_state(
            rows=(
                ConsoleDisplayRow("Provider", "first"),
                ConsoleDisplayRow("Provider", "second"),
            )
        ),
        _base_state(
            rows=(),
            actions=(
                ConsoleInspectorAction(
                    "console-inspector-save-chatbook", "first", True
                ),
                ConsoleInspectorAction(
                    "console-inspector-save-chatbook", "second", True
                ),
            ),
        ),
        _base_state(
            rows=(),
            actions=(
                ConsoleInspectorAction(
                    "console-inspector-save-chatbook", "ordinary", True
                ),
            ),
            dictionary_actions=(
                ConsoleInspectorAction(
                    "console-inspector-save-chatbook", "dictionary", True
                ),
            ),
        ),
        _base_state(
            rows=(),
            dictionary_actions=(
                ConsoleInspectorAction("dynamic-collision", "dictionary", True),
            ),
            world_book_actions=(
                ConsoleInspectorAction("dynamic-collision", "world book", True),
            ),
        ),
        _base_state(
            rows=(),
            dictionary_rows=(ConsoleDisplayRow("Dictionary", "attached"),),
            world_book_actions=(
                ConsoleInspectorAction(
                    "console-inspector-dictionaries-row-0", "collision", True
                ),
            ),
        ),
    ],
)
def test_strict_ownership_rejects_duplicate_or_colliding_stable_ids(state):
    ownership = _ownership_module()

    with pytest.raises(ownership.UnownedInspectorContentError):
        ConsoleRunInspector(
            state,
            ownership_policy=ownership.InspectorOwnershipPolicy.STRICT,
        )


@pytest.mark.asyncio
async def test_resilient_collision_projection_keeps_first_instance_consistently(
    monkeypatch,
):
    ownership = _ownership_module()
    inspector_module = importlib.import_module(
        "tldw_chatbook.Widgets.Console.console_run_inspector"
    )
    diagnostics = []
    monkeypatch.setattr(
        inspector_module.logger,
        "warning",
        lambda message, fingerprint: diagnostics.append((message, fingerprint)),
    )
    state = _base_state(
        rows=(
            ConsoleDisplayRow("Provider", "first provider"),
            ConsoleDisplayRow("Provider", "SECOND PROVIDER SECRET"),
        ),
        actions=(
            ConsoleInspectorAction(
                "console-inspector-save-chatbook", "first save", True
            ),
            ConsoleInspectorAction(
                "console-inspector-save-chatbook", "SECOND SAVE SECRET", True
            ),
        ),
        dictionary_rows=(ConsoleDisplayRow("Dictionary", "attached"),),
        dictionary_actions=(
            ConsoleInspectorAction(
                "console-inspector-save-chatbook", "DICTIONARY SECRET", True
            ),
        ),
        world_book_actions=(
            ConsoleInspectorAction(
                "console-inspector-dictionaries-row-0", "WORLD SECRET", True
            ),
        ),
    )
    app = InspectorHarness(
        state,
        ownership_policy=ownership.InspectorOwnershipPolicy.RESILIENT,
    )

    async with app.run_test(size=(80, 32)) as pilot:
        inspector = app.query_one("#inspector", ConsoleRunInspector)
        provider = inspector.query_one("#console-inspector-provider", Static)
        assert str(provider.renderable) == "Provider: first provider"
        assert len(app.query("#console-inspector-save-chatbook")) == 1
        assert len(app.query("#console-inspector-dictionaries-row-0")) == 1
        assert not list(app.query("#console-inspector-run-status-summary"))
        assert diagnostics == [
            (
                "Inspector ownership incomplete: {}",
                (
                    "action:console-inspector-dictionaries-row-0",
                    "action:console-inspector-save-chatbook",
                    "row:Provider",
                ),
            )
        ]
        assert not any(
            secret in repr(diagnostics)
            for secret in (
                "SECOND PROVIDER SECRET",
                "SECOND SAVE SECRET",
                "DICTIONARY SECRET",
                "WORLD SECRET",
            )
        )

        inspector.sync_state(
            _base_state(
                rows=(
                    ConsoleDisplayRow("Provider", "updated first"),
                    ConsoleDisplayRow("Provider", "changed duplicate"),
                ),
                actions=state.actions,
                dictionary_rows=state.dictionary_rows,
                dictionary_actions=state.dictionary_actions,
                world_book_actions=state.world_book_actions,
            )
        )
        await pilot.pause()
        assert inspector.recompose_count == 0
        assert inspector.query_one("#console-inspector-provider", Static) is provider
        assert str(provider.renderable) == "Provider: updated first"
        assert len(diagnostics) == 1


@pytest.mark.asyncio
async def test_review_changes_mounts_under_changes_before_dictionaries():
    state = _base_state(
        actions=(
            ConsoleInspectorAction(
                "console-inspector-review-changes", "Review changes", True
            ),
        ),
        dictionary_rows=(ConsoleDisplayRow("Dictionary", "attached"),),
    )
    app = InspectorHarness(state)

    async with app.run_test(size=(80, 32)):
        inspector = app.query_one("#inspector", ConsoleRunInspector)
        mounted_ids = [widget.id for widget in inspector.query("*") if widget.id]

        assert mounted_ids.index(
            "console-inspector-changes-heading"
        ) < mounted_ids.index("console-inspector-review-changes")
        assert mounted_ids.index(
            "console-inspector-review-changes"
        ) < mounted_ids.index("console-inspector-dictionaries-heading")


@pytest.mark.asyncio
async def test_each_run_group_has_external_heading_and_one_bounded_body():
    state = _base_state(
        rows=tuple(
            ConsoleDisplayRow(label, f"value for {label}")
            for label in EXPECTED_ROW_OWNERS
        ),
        actions=tuple(
            ConsoleInspectorAction(action_id, owner, True)
            for action_id, owner in EXPECTED_ACTION_OWNERS.items()
        ),
        dictionary_rows=(ConsoleDisplayRow("Dictionary", "attached"),),
        world_book_rows=(ConsoleDisplayRow("World Book", "attached"),),
    )

    async with InspectorHarness(state).run_test(size=(100, 60)) as pilot:
        await pilot.pause()
        inspector = pilot.app.query_one("#inspector", ConsoleRunInspector)
        assert not list(inspector.query("#console-inspector-run-status-summary"))

        expected_sections = (
            *(owner for owner, _heading_id, _labels in _ROW_GROUPS),
            "Chat Dictionaries",
            "World Books",
        )
        assert tuple(
            section.section_id for section in inspector.query(ConsoleBoundedSection)
        ) == tuple(owner.lower().replace(" ", "-") for owner in expected_sections)

        for owner, heading_id, labels in _ROW_GROUPS:
            heading = inspector.query_one(f"#{heading_id}")
            body = inspector.query_one(
                f"#console-bounded-section-{owner.lower().replace(' ', '-')}",
                ConsoleBoundedSection,
            )
            assert heading.parent is inspector
            assert body.parent is inspector
            children = list(inspector.children)
            assert children.index(body) == children.index(heading) + 1
            assert len(body.query(ConsoleBoundedSection)) == 0
            for label in labels:
                if label in {"Selected conversation", "Workspace"}:
                    assert not body.query(f"#{_ownership_module().ROW_IDS[label]}")
                    continue
                row_id = _ownership_module().ROW_IDS[label]
                assert body.query_one(f"#{row_id}")


@pytest.mark.asyncio
async def test_pinned_run_authority_removes_only_the_duplicate_status_summary():
    state = _base_state(
        rows=_base_state().rows
        + (
            ConsoleDisplayRow("Recovery action", "Reconnect provider"),
            ConsoleDisplayRow("Next action", "Open Settings"),
        )
    )

    async with InspectorHarness(state).run_test(size=(80, 32)) as pilot:
        inspector = pilot.app.query_one("#inspector", ConsoleRunInspector)

        assert not list(inspector.query("#console-inspector-run-status-summary"))
        assert inspector.query_one("#console-inspector-run-recipe", Static)
        assert inspector.query_one("#console-inspector-recovery-action", Static)
        assert inspector.query_one("#console-inspector-next-action", Static)


@pytest.mark.asyncio
async def test_run_section_uses_exact_twenty_line_content_ceiling():
    state = _base_state()
    async with InspectorHarness(state).run_test(size=(80, 40)) as pilot:
        section = pilot.app.query_one(
            "#console-bounded-section-run", ConsoleBoundedSection
        )
        await section.viewport.remove_children()
        content = Static("\n".join(f"row {index}" for index in range(20)))
        await section.viewport.mount(content)
        section.request_reconcile()
        for _ in range(4):
            await pilot.pause()
        assert section.viewport.content_region.height == 20
        assert section.hint.display is False

        content.update("\n".join(f"row {index}" for index in range(21)))
        content.refresh(layout=True)
        section.request_reconcile()
        for _ in range(4):
            await pilot.pause()
        assert section.viewport.content_region.height == 20
        assert section.hint.display is True
        assert section.hint.region.height == 1


@pytest.mark.asyncio
async def test_resilient_ownership_omits_unknowns_deduplicates_safe_diagnostic_and_clears_in_place(
    monkeypatch,
):
    ownership = _ownership_module()
    inspector_module = importlib.import_module(
        "tldw_chatbook.Widgets.Console.console_run_inspector"
    )
    diagnostics = []
    monkeypatch.setattr(
        inspector_module.logger,
        "warning",
        lambda message, fingerprint: diagnostics.append((message, fingerprint)),
    )
    state = _base_state(
        rows=(
            ConsoleDisplayRow("Provider", "known provider", status="ready"),
            ConsoleDisplayRow("Unknown row", "ROW SECRET"),
        ),
        actions=(
            ConsoleInspectorAction(
                "console-inspector-save-chatbook", "Save as Chatbook", True
            ),
            ConsoleInspectorAction(
                "console-inspector-unknown-action", "ACTION SECRET", True
            ),
        ),
    )
    app = InspectorHarness(
        state,
        ownership_policy=ownership.InspectorOwnershipPolicy.RESILIENT,
    )

    async with app.run_test(size=(80, 32)) as pilot:
        inspector = app.query_one("#inspector", ConsoleRunInspector)
        provider_before = inspector.query_one("#console-inspector-provider", Static)
        assert not list(inspector.query("#console-inspector-run-status-summary"))
        assert not app.query("#console-inspector-unknown-action")
        assert not [
            widget
            for widget in app.query(Static)
            if "ROW SECRET" in str(widget.renderable)
            or "ACTION SECRET" in str(widget.renderable)
            or str(widget.renderable) == "Other"
        ]
        assert len(diagnostics) == 1
        assert diagnostics[0][1] == (
            "action:console-inspector-unknown-action",
            "row:Unknown row",
        )
        assert "ROW SECRET" not in repr(diagnostics)
        assert "ACTION SECRET" not in repr(diagnostics)

        inspector.sync_state(
            _base_state(
                rows=(
                    ConsoleDisplayRow("Provider", "updated provider", status="ready"),
                    ConsoleDisplayRow("Unknown row", "DIFFERENT ROW SECRET"),
                ),
                actions=state.actions,
            )
        )
        await pilot.pause()
        assert inspector.recompose_count == 0
        assert len(diagnostics) == 1
        assert (
            inspector.query_one("#console-inspector-provider", Static)
            is provider_before
        )

        inspector.sync_state(
            _base_state(
                rows=(
                    ConsoleDisplayRow("Provider", "updated again", status="ready"),
                    ConsoleDisplayRow("Unknown row", "FIRST DUPLICATE SECRET"),
                    ConsoleDisplayRow("Unknown row", "SECOND DUPLICATE SECRET"),
                ),
                actions=state.actions,
            )
        )
        await pilot.pause()
        assert inspector.recompose_count == 0
        assert len(diagnostics) == 1

        inspector.sync_state(
            _base_state(
                rows=(ConsoleDisplayRow("Provider", "valid provider", status="ready"),),
                actions=(
                    ConsoleInspectorAction(
                        "console-inspector-save-chatbook", "Save as Chatbook", True
                    ),
                ),
            )
        )
        await pilot.pause()
        assert inspector.recompose_count == 0
        inspector.sync_state(
            _base_state(
                rows=(
                    ConsoleDisplayRow("Provider", "valid provider", status="ready"),
                    ConsoleDisplayRow("Sources", "staged", status="ready"),
                ),
                actions=(
                    ConsoleInspectorAction(
                        "console-inspector-save-chatbook", "Save as Chatbook", True
                    ),
                ),
            )
        )
        await pilot.pause()
        assert inspector.recompose_count == 1


@pytest.mark.asyncio
async def test_inspector_row_text_change_updates_rows_in_place():
    app = InspectorHarness(_base_state())

    async with app.run_test(size=(80, 32)) as pilot:
        inspector = app.query_one("#inspector", ConsoleRunInspector)
        provider_row_before = inspector.query_one("#console-inspector-provider", Static)
        recipe_row_before = inspector.query_one("#console-inspector-run-recipe", Static)

        new_state = _base_state(
            rows=(
                ConsoleDisplayRow("Run recipe", "Chat with provider"),
                ConsoleDisplayRow("Provider", "Anthropic / claude", status="ready"),
                ConsoleDisplayRow("Sources", "1 staged", status="ready"),
            )
        )
        inspector.sync_state(new_state)
        await pilot.pause()

        assert inspector.recompose_count == 0
        provider_row_after = inspector.query_one("#console-inspector-provider", Static)
        assert provider_row_after is provider_row_before
        assert str(provider_row_after.renderable) == "Provider: Anthropic / claude"
        # Unchanged rows keep both identity and content.
        assert (
            inspector.query_one("#console-inspector-run-recipe", Static)
            is recipe_row_before
        )


@pytest.mark.asyncio
async def test_inspector_row_status_change_swaps_class_in_place():
    app = InspectorHarness(_base_state())

    async with app.run_test(size=(80, 32)) as pilot:
        inspector = app.query_one("#inspector", ConsoleRunInspector)
        new_state = _base_state(
            rows=(
                ConsoleDisplayRow("Run recipe", "Chat with provider"),
                ConsoleDisplayRow(
                    "Provider",
                    "Missing API key",
                    status="blocked",
                    recovery="Add a key",
                ),
                ConsoleDisplayRow("Sources", "1 staged", status="ready"),
            )
        )
        inspector.sync_state(new_state)
        await pilot.pause()

        assert inspector.recompose_count == 0
        provider_row = inspector.query_one("#console-inspector-provider", Static)
        assert provider_row.has_class("console-inspector-row-blocked")
        assert not provider_row.has_class("console-inspector-row-ready")
        assert str(provider_row.renderable) == "Provider: Missing API key - Add a key"
        assert not list(inspector.query("#console-inspector-run-status-summary"))


@pytest.mark.asyncio
async def test_inspector_structural_row_change_recomposes_widget():
    app = InspectorHarness(_base_state())

    async with app.run_test(size=(80, 32)) as pilot:
        inspector = app.query_one("#inspector", ConsoleRunInspector)

        new_state = _base_state(
            rows=_base_state().rows
            + (ConsoleDisplayRow("Artifacts", "Chatbook available", status="ready"),)
        )
        inspector.sync_state(new_state)
        await pilot.pause()

        assert inspector.recompose_count == 1
        artifacts_row = inspector.query_one("#console-inspector-artifacts", Static)
        assert str(artifacts_row.renderable) == "Artifacts: Chatbook available"


@pytest.mark.asyncio
async def test_inspector_action_change_recomposes_widget():
    app = InspectorHarness(_base_state())

    async with app.run_test(size=(80, 32)) as pilot:
        inspector = app.query_one("#inspector", ConsoleRunInspector)

        new_state = _base_state(
            actions=(
                ConsoleInspectorAction(
                    widget_id="console-inspector-save-chatbook",
                    label="Save as Chatbook",
                    enabled=True,
                ),
            )
        )
        inspector.sync_state(new_state)
        await pilot.pause()

        assert inspector.recompose_count == 1
        assert app.query("#console-inspector-save-chatbook")


@pytest.mark.asyncio
async def test_inspector_dictionary_row_text_change_updates_in_place():
    dict_state = _base_state(
        dictionary_rows=(ConsoleDisplayRow("Dictionaries", "2 active"),)
    )
    app = InspectorHarness(dict_state)

    async with app.run_test(size=(80, 32)) as pilot:
        inspector = app.query_one("#inspector", ConsoleRunInspector)
        row_before = inspector.query_one(
            "#console-inspector-dictionaries-row-0", Static
        )

        inspector.sync_state(
            _base_state(
                dictionary_rows=(ConsoleDisplayRow("Dictionaries", "3 active"),)
            )
        )
        await pilot.pause()

        assert inspector.recompose_count == 0
        row_after = inspector.query_one("#console-inspector-dictionaries-row-0", Static)
        assert row_after is row_before
        assert str(row_after.renderable) == "Dictionaries: 3 active"

        # Changing the dictionary section shape is structural.
        inspector.sync_state(_base_state(dictionary_rows=()))
        await pilot.pause()
        assert inspector.recompose_count == 1
        assert len(app.query("#console-inspector-dictionaries-row-0")) == 0


@pytest.mark.asyncio
async def test_inspector_equal_state_sync_is_noop():
    app = InspectorHarness(_base_state())

    async with app.run_test(size=(80, 32)) as pilot:
        inspector = app.query_one("#inspector", ConsoleRunInspector)
        provider_row_before = inspector.query_one("#console-inspector-provider", Static)

        inspector.sync_state(_base_state())
        await pilot.pause()

        assert inspector.recompose_count == 0
        assert (
            inspector.query_one("#console-inspector-provider", Static)
            is provider_row_before
        )


def test_prefill_rows_route_into_selected_conversation_group():
    """Prefill rows must render inside the Selected Conversation group with
    stable ids -- not fall through to the unrouted tail after Selected
    Message (which also leaves them below the fold in a collapsed rail)."""
    state = _base_state(
        rows=(
            ConsoleDisplayRow("Selected conversation", "Chat 1"),
            ConsoleDisplayRow("Conversation source", "native Console session"),
            ConsoleDisplayRow("Workspace", "Default"),
            ConsoleDisplayRow("Resume state", "local session, not persisted yet"),
            ConsoleDisplayRow("Prefill (next send only)", "Sure thing:"),
            ConsoleDisplayRow("Prefill (pinned)", "*Ship AI:*"),
            ConsoleDisplayRow("Selected message", "None selected"),
        ),
    )
    entries = ConsoleRunInspector._rendered_row_entries(state)
    ids = [entry_id for entry_id, _text, _status in entries]
    assert "console-inspector-prefill-one-shot" in ids
    assert "console-inspector-prefill-pinned" in ids
    # Grouped BEFORE the Selected Message group's rows, not after them.
    assert ids.index("console-inspector-prefill-pinned") < ids.index(
        "console-inspector-selected-message"
    )
    # No prefill row left in the unrouted fallback tail (generic row-N ids).
    assert not [
        entry_id
        for entry_id, text, _status in entries
        if "Prefill" in text and entry_id.startswith("console-inspector-row-")
    ]


@pytest.mark.parametrize(
    ("owner", "value", "status", "expected"),
    [
        ("Tools", "", "ready", False),
        ("Tools", "—", "ready", False),
        ("Tools", "0", "ready", False),
        ("Tools", "0 ready", "ready", False),
        ("Tools", "1 ready", "ready", True),
        ("Approvals", "0 pending", "ready", False),
        ("Approvals", "1 pending", "blocked", True),
        ("Artifacts", "", "ready", False),
        ("Artifacts", "—", "ready", False),
        ("Artifacts", "none", "ready", False),
        ("Artifacts", "unavailable", "ready", False),
        ("Artifacts", "not available for this item", "ready", False),
        ("Artifacts", "Chatbook available", "ready", True),
    ],
)
def test_conditional_group_actionability_pins_current_zero_spellings(
    owner, value, status, expected
):
    ownership = _ownership_module()
    label = {"Tools": "Tools", "Approvals": "Approvals", "Artifacts": "Artifacts"}[
        owner
    ]
    state = _base_state(rows=(ConsoleDisplayRow(label, value, status=status),))
    owned = ownership.classify_inspector_content(
        state, ownership.InspectorOwnershipPolicy.STRICT
    )

    assert inspector_group_is_actionable(owner, owned) is expected


def test_conditional_group_enabled_action_is_always_actionable():
    ownership = _ownership_module()
    state = _base_state(
        rows=(ConsoleDisplayRow("Artifacts", "unavailable"),),
        actions=(
            ConsoleInspectorAction(
                "console-inspector-save-chatbook", "Save as Chatbook", True
            ),
        ),
    )
    owned = ownership.classify_inspector_content(
        state, ownership.InspectorOwnershipPolicy.STRICT
    )
    assert inspector_group_is_actionable("Artifacts", owned)
    with pytest.raises(ValueError):
        inspector_group_is_actionable("Run", owned)


def _conditional_state(*, tools=False, approval=False, artifact=False):
    return _base_state(
        rows=(
            ConsoleDisplayRow("Sources", "0 staged", status="ready"),
            ConsoleDisplayRow("Tools", "1 ready" if tools else "0 ready"),
            ConsoleDisplayRow(
                "Approvals",
                "1 pending" if approval else "0 pending",
                status="blocked" if approval else "ready",
            ),
            ConsoleDisplayRow(
                "Artifacts", "Chatbook available" if artifact else "unavailable"
            ),
            ConsoleDisplayRow("Session provider", "OpenAI"),
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("flags", "promoted", "under_more"),
    [
        ({}, (), ("Tools", "Approvals", "Artifacts")),
        ({"tools": True}, ("Tools",), ("Approvals", "Artifacts")),
        ({"approval": True}, ("Approvals",), ("Tools", "Artifacts")),
        ({"artifact": True}, ("Artifacts",), ("Tools", "Approvals")),
        (
            {"tools": True, "approval": True, "artifact": True},
            ("Tools", "Approvals", "Artifacts"),
            (),
        ),
    ],
)
async def test_conditional_groups_promote_in_fixed_order_and_never_hide_actions(
    flags, promoted, under_more
):
    app = InspectorHarness(_conditional_state(**flags), more_open=True)
    async with app.run_test(size=(80, 40)):
        inspector = app.query_one("#inspector", ConsoleRunInspector)
        direct_headings = tuple(
            str(child.renderable)
            for child in inspector.children
            if isinstance(child, Static)
            and child.has_class("console-inspector-group-heading")
        )
        source_index = direct_headings.index("Source Readiness")
        after_source = direct_headings[source_index + 1 :]
        assert after_source[: len(promoted)] == promoted
        assert "Source Readiness" in direct_headings
        more = list(inspector.query("#console-inspector-more-toggle"))
        if under_more:
            assert len(more) == 1
            more_body = inspector.query_one("#console-inspector-more-body")
            assert (
                tuple(
                    str(heading.renderable)
                    for heading in more_body.query(".console-inspector-group-heading")
                )
                == under_more
            )
            for owner in promoted:
                assert not list(
                    more_body.query(f"#console-inspector-{owner.lower()}-heading")
                )
        else:
            assert not more


@pytest.mark.asyncio
async def test_more_real_pointer_and_key_grammar_posts_once_only_for_user_changes():
    app = InspectorHarness(_conditional_state())
    async with app.run_test(size=(80, 40)) as pilot:
        inspector = app.query_one("#inspector", ConsoleRunInspector)
        toggle = inspector.query_one("#console-inspector-more-toggle", Button)
        body = inspector.query_one("#console-inspector-more-body")
        assert body.display is False, app.more_events

        assert await pilot.click(toggle)
        await pilot.pause()
        assert body.display is True
        assert app.more_events == [True]

        toggle.focus()
        await pilot.pause()
        assert app.focused is toggle
        await pilot.press("space")
        await pilot.pause()
        assert body.display is False, app.more_events
        await pilot.press("enter")
        await pilot.pause()
        assert body.display is True
        await pilot.press("left")
        await pilot.pause()
        assert body.display is False
        await pilot.press("right")
        await pilot.pause()
        assert body.display is True
        assert app.more_events == [True, False, True, False, True]

        inspector.set_more_open(False)
        await pilot.pause()
        assert body.display is False
        assert app.more_events == [True, False, True, False, True]


@pytest.mark.asyncio
async def test_conditional_focus_demotes_to_heading_then_closed_more_toggle():
    active = _conditional_state(artifact=True)
    active = replace(
        active,
        actions=(
            ConsoleInspectorAction(
                "console-inspector-save-chatbook", "Save as Chatbook", True
            ),
        ),
    )
    app = InspectorHarness(active, more_open=True)
    async with app.run_test(size=(80, 40)) as pilot:
        inspector = app.query_one("#inspector", ConsoleRunInspector)
        action = inspector.query_one("#console-inspector-save-chatbook", Button)
        action.focus()

        inspector.sync_state(_conditional_state())
        await pilot.pause()
        await pilot.pause()
        assert app.focused.id == "console-inspector-artifacts-heading"
        more_body = inspector.query_one("#console-inspector-more-body")
        assert app.focused in tuple(more_body.query("*"))

        inspector.set_more_open(False)
        await pilot.pause()
        assert app.focused.id == "console-inspector-more-toggle"


@pytest.mark.asyncio
async def test_conditional_focus_keeps_same_more_heading_across_other_promotion():
    app = InspectorHarness(_conditional_state(), more_open=True)
    async with app.run_test(size=(80, 40)) as pilot:
        inspector = app.query_one("#inspector", ConsoleRunInspector)
        heading = inspector.query_one("#console-inspector-tools-heading", Static)
        heading.focus()
        await pilot.pause()
        assert app.focused is heading

        inspector.sync_state(_conditional_state(artifact=True))
        await pilot.pause()
        await pilot.pause()

        assert app.focused.id == "console-inspector-tools-heading"
        assert app.focused in tuple(
            inspector.query_one("#console-inspector-more-body").query("*")
        )


@pytest.mark.asyncio
async def test_disabled_approval_recovers_to_closed_more_toggle():
    active = replace(
        _conditional_state(approval=True),
        actions=(
            ConsoleInspectorAction(
                "console-inspector-review-approval", "Review approval", True
            ),
        ),
    )
    disabled = replace(
        _conditional_state(),
        actions=(
            ConsoleInspectorAction(
                "console-inspector-review-approval", "Review approval", False
            ),
        ),
    )
    app = InspectorHarness(active, more_open=False)
    async with app.run_test(size=(80, 40)) as pilot:
        inspector = app.query_one("#inspector", ConsoleRunInspector)
        inspector.query_one("#console-inspector-review-approval", Button).focus()

        inspector.sync_state(disabled)
        await pilot.pause()
        await pilot.pause()

        assert app.focused.id == "console-inspector-more-toggle"


@pytest.mark.asyncio
async def test_promotion_does_not_proactively_move_focus_from_more_toggle():
    app = InspectorHarness(_conditional_state(), more_open=True)
    async with app.run_test(size=(80, 40)) as pilot:
        inspector = app.query_one("#inspector", ConsoleRunInspector)
        toggle = inspector.query_one("#console-inspector-more-toggle", Button)
        toggle.focus()
        await pilot.pause()
        assert app.focused is toggle

        inspector.sync_state(_conditional_state(tools=True))
        await pilot.pause()
        await pilot.pause()

        promoted = inspector.query_one("#console-inspector-tools-heading")
        assert app.focused is not promoted
        assert promoted not in tuple(
            inspector.query_one("#console-inspector-more-body").query("*")
        )
