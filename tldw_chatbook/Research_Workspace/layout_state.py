"""Pure responsive layout state for the Research Workspace."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


ResearchPane = Literal["sources", "chat", "studio"]
ResearchCompanion = Literal["sources", "studio"]
ResearchLayoutMode = Literal["wide", "medium", "narrow"]


@dataclass(frozen=True, slots=True)
class ResearchPanePreferences:
    """Stored side-pane preferences, never responsive effective state."""

    sources_open: bool = True
    studio_open: bool = True
    preferred_companion: ResearchCompanion = "sources"

    def __post_init__(self) -> None:
        if type(self.sources_open) is not bool:
            raise TypeError("sources_open must be bool")
        if type(self.studio_open) is not bool:
            raise TypeError("studio_open must be bool")
        if self.preferred_companion not in {"sources", "studio"}:
            raise ValueError("preferred_companion must be sources or studio")


@dataclass(frozen=True, slots=True)
class ResearchPaneLayout:
    """Effective pane visibility derived for one viewport width."""

    mode: ResearchLayoutMode
    visible_panes: tuple[ResearchPane, ...]
    sources_forced_closed: bool
    studio_forced_closed: bool


def derive_research_pane_layout(
    width: int,
    preferences: ResearchPanePreferences,
    *,
    active_pane: ResearchPane = "chat",
) -> ResearchPaneLayout:
    """Derive effective visibility without changing stored preferences."""

    if type(width) is not int or width < 1:
        raise ValueError("width must be a positive integer")
    if not isinstance(preferences, ResearchPanePreferences):
        raise TypeError("preferences must be ResearchPanePreferences")
    if active_pane not in {"sources", "chat", "studio"}:
        raise ValueError("active_pane must be sources, chat, or studio")

    if width >= 150:
        visible: list[ResearchPane] = []
        if preferences.sources_open:
            visible.append("sources")
        visible.append("chat")
        if preferences.studio_open:
            visible.append("studio")
        return ResearchPaneLayout("wide", tuple(visible), False, False)

    if width >= 100:
        companion = _medium_companion(preferences)
        if companion == "sources":
            visible_panes = ("sources", "chat")
        elif companion == "studio":
            visible_panes = ("chat", "studio")
        else:
            visible_panes = ("chat",)
        return ResearchPaneLayout(
            "medium",
            visible_panes,
            preferences.sources_open and companion != "sources",
            preferences.studio_open and companion != "studio",
        )

    return ResearchPaneLayout(
        "narrow",
        (active_pane,),
        preferences.sources_open and active_pane != "sources",
        preferences.studio_open and active_pane != "studio",
    )


def toggle_research_pane(
    preferences: ResearchPanePreferences,
    pane: ResearchCompanion,
    *,
    width: int,
) -> ResearchPanePreferences:
    """Apply a side-pane toggle so its effective medium/wide state changes."""

    if pane not in {"sources", "studio"}:
        raise ValueError("pane must be sources or studio")
    if type(width) is not int or width < 1:
        raise ValueError("width must be a positive integer")
    if width < 100:
        raise ValueError("narrow layout uses active_pane selection")

    layout = derive_research_pane_layout(width, preferences)
    if pane in layout.visible_panes:
        other: ResearchCompanion = "studio" if pane == "sources" else "sources"
        changes: dict[str, object] = {f"{pane}_open": False}
        if getattr(preferences, f"{other}_open"):
            changes["preferred_companion"] = other
        return replace(preferences, **changes)

    return replace(
        preferences,
        **{f"{pane}_open": True, "preferred_companion": pane},
    )


def _medium_companion(
    preferences: ResearchPanePreferences,
) -> ResearchCompanion | None:
    preferred = preferences.preferred_companion
    if getattr(preferences, f"{preferred}_open"):
        return preferred
    other: ResearchCompanion = "studio" if preferred == "sources" else "sources"
    return other if getattr(preferences, f"{other}_open") else None
