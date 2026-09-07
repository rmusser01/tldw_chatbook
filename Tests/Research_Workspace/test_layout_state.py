from dataclasses import replace

import pytest

from tldw_chatbook.Research_Workspace.layout_state import (
    ResearchPanePreferences,
    derive_research_pane_layout,
    toggle_research_pane,
)


@pytest.mark.parametrize(
    ("width", "mode", "visible_panes"),
    [
        (160, "wide", ("sources", "chat", "studio")),
        (150, "wide", ("sources", "chat", "studio")),
        (149, "medium", ("sources", "chat")),
        (120, "medium", ("sources", "chat")),
        (100, "medium", ("sources", "chat")),
        (99, "narrow", ("chat",)),
        (84, "narrow", ("chat",)),
        (80, "narrow", ("chat",)),
        (60, "narrow", ("chat",)),
    ],
)
def test_width_boundaries_choose_the_expected_panes(
    width: int,
    mode: str,
    visible_panes: tuple[str, ...],
) -> None:
    layout = derive_research_pane_layout(width, ResearchPanePreferences())

    assert layout.mode == mode
    assert layout.visible_panes == visible_panes


@pytest.mark.parametrize(
    ("preferences", "visible_panes", "forced"),
    [
        (
            ResearchPanePreferences(sources_open=True, studio_open=False),
            ("sources", "chat"),
            (False, False),
        ),
        (
            ResearchPanePreferences(sources_open=False, studio_open=True),
            ("chat", "studio"),
            (False, False),
        ),
        (
            ResearchPanePreferences(sources_open=False, studio_open=False),
            ("chat",),
            (False, False),
        ),
        (
            ResearchPanePreferences(
                sources_open=True,
                studio_open=True,
                preferred_companion="studio",
            ),
            ("chat", "studio"),
            (True, False),
        ),
    ],
)
def test_medium_layout_honors_side_preferences_and_companion_priority(
    preferences: ResearchPanePreferences,
    visible_panes: tuple[str, ...],
    forced: tuple[bool, bool],
) -> None:
    layout = derive_research_pane_layout(120, preferences)

    assert layout.visible_panes == visible_panes
    assert (layout.sources_forced_closed, layout.studio_forced_closed) == forced


def test_revealing_hidden_medium_companion_replaces_it_without_closing_wide_preference() -> (
    None
):
    original = ResearchPanePreferences(
        sources_open=True,
        studio_open=True,
        preferred_companion="sources",
    )

    updated = toggle_research_pane(original, "studio", width=120)

    assert updated == replace(original, preferred_companion="studio")
    assert derive_research_pane_layout(120, updated).visible_panes == (
        "chat",
        "studio",
    )
    assert derive_research_pane_layout(160, updated).visible_panes == (
        "sources",
        "chat",
        "studio",
    )


@pytest.mark.parametrize("active_pane", ["sources", "chat", "studio"])
def test_narrow_mode_mounts_exactly_the_selected_pane_without_mutating_preferences(
    active_pane: str,
) -> None:
    preferences = ResearchPanePreferences()

    layout = derive_research_pane_layout(
        84,
        preferences,
        active_pane=active_pane,
    )

    assert layout.visible_panes == (active_pane,)
    assert preferences == ResearchPanePreferences()


def test_responsive_forced_collapse_restores_stored_preferences_when_width_returns() -> (
    None
):
    preferences = ResearchPanePreferences()

    narrow = derive_research_pane_layout(60, preferences)
    restored = derive_research_pane_layout(160, preferences)

    assert narrow.sources_forced_closed is True
    assert narrow.studio_forced_closed is True
    assert restored.visible_panes == ("sources", "chat", "studio")
    assert preferences == ResearchPanePreferences()


def test_explicit_medium_collapse_and_reveal_each_change_the_effective_layout() -> None:
    original = ResearchPanePreferences(
        sources_open=True,
        studio_open=True,
        preferred_companion="sources",
    )

    collapsed = toggle_research_pane(original, "sources", width=120)
    revealed = toggle_research_pane(collapsed, "sources", width=120)

    assert derive_research_pane_layout(120, collapsed).visible_panes == (
        "chat",
        "studio",
    )
    assert derive_research_pane_layout(120, revealed).visible_panes == (
        "sources",
        "chat",
    )
    assert collapsed.studio_open is True
    assert revealed.studio_open is True


def test_invalid_preference_and_reducer_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="preferred_companion"):
        ResearchPanePreferences(preferred_companion="chat")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="width"):
        derive_research_pane_layout(0, ResearchPanePreferences())
    with pytest.raises(ValueError, match="active_pane"):
        derive_research_pane_layout(
            80,
            ResearchPanePreferences(),
            active_pane="other",  # type: ignore[arg-type]
        )
