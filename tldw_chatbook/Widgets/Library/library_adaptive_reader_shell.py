"""Library names for the shared adaptive pane shell (Roleplay frame B0).

The retained three-role structure that lived here -- grip, shell and their
messages -- moved unchanged to ``tldw_chatbook.Widgets.adaptive_pane_shell``,
so a second destination (Roleplay) can share it without importing this
package. This module keeps every Library name working:

- The messages are the SAME objects (``LibraryPaneVisibilityChanged is
  PaneVisibilityChanged``). Every Library handler binds with ``@on(...)``,
  which matches on class identity, so nothing re-routes; a subclass alias
  would silently stop matching the shared shell's posts.
- The widgets are thin subclasses that supply only the Library's destination
  classes (``LIBRARY_ADAPTIVE_READER_CLASSES``) and painted names. They stay
  real classes because ``query(LibraryAdaptiveReaderPaneGrip)`` matches on the
  class NAME, and because Library code and tests construct them directly.
- Neither subclass defines ``on_mount``, ``on_resize`` or
  ``on_descendant_focus``: Textual dispatches those once per class in the MRO
  that defines one, so a redefinition here would run the shared body twice.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from textual.widget import Widget

from tldw_chatbook.Utils.adaptive_reader_state import (
    PANE_GRIP_WIDTH,
    AdaptiveReaderEffectiveLayout,
    PaneName,
)
from tldw_chatbook.Widgets.adaptive_pane_shell import (
    ARROW_UPPER_POSITION_RATIO,
    AdaptivePaneClasses,
    AdaptivePaneGrip,
    AdaptivePaneShell,
    AdaptivePaneShellResized,
    PaneToggleRequested,
    PaneVisibilityChanged,
)

__all__ = [
    "LIBRARY_ADAPTIVE_READER_CLASSES",
    "LIBRARY_ADAPTIVE_READER_GRIP_CLASS",
    "LIBRARY_ARROW_UPPER_POSITION_RATIO",
    "LIBRARY_PANE_GRIP_NAMES",
    "AdaptiveReaderShellResized",
    "LibraryAdaptiveReaderPaneGrip",
    "LibraryAdaptiveReaderShell",
    "LibraryPaneVisibilityChanged",
    "PaneToggleRequested",
]

LIBRARY_ARROW_UPPER_POSITION_RATIO = ARROW_UPPER_POSITION_RATIO

#: The Library's destination classes (Roleplay frame B0). These are the class names the
#: Library always used, so ``css/features/_library.tcss`` matches them
#: unchanged, every token keeps the Library split prefix (the rules stay in the
#: lazy ``screen_agentic_library.tcss``), and boot CSS does not move.
LIBRARY_ADAPTIVE_READER_CLASSES = AdaptivePaneClasses(
    shell="library-adaptive-reader-shell",
    nav="library-adaptive-reader-library",
    items="library-adaptive-reader-items",
    work="library-adaptive-reader-work",
    grip="library-adaptive-reader-pane-grip",
)

#: Shared class every Library adaptive reader shell puts on BOTH of its pane
#: grips. Named here so focus code can recognise a grip without a magic string
#: (task-31567: the grips are the shell's first focusable widgets, so a
#: recompose hands them focus unless someone puts it back).
LIBRARY_ADAPTIVE_READER_GRIP_CLASS = LIBRARY_ADAPTIVE_READER_CLASSES.grip

#: (task-32355) What a grip PAINTS where its pane's spoken name would not fit
#: or would not be the name the guide uses. The Library pane's tooltip says
#: "Expand Library pane"; the handle itself is the "Nav" handle
#: (``Docs/User_Guide/library.md``), which is also the only form that reads in
#: a five-cell column.
LIBRARY_PANE_GRIP_NAMES = {"Library": "Nav"}

#: Same objects as the shared messages, never subclasses (module docstring).
AdaptiveReaderShellResized = AdaptivePaneShellResized
LibraryPaneVisibilityChanged = PaneVisibilityChanged


class LibraryAdaptiveReaderPaneGrip(AdaptivePaneGrip):
    """The Library's pane grip: the shared grip with the Library's class and names."""

    def __init__(
        self,
        pane: PaneName,
        *,
        open: bool,
        pane_label: str,
        extra_classes: str = "",
        width: int = PANE_GRIP_WIDTH,
        destination_class: str = LIBRARY_ADAPTIVE_READER_GRIP_CLASS,
        painted_names: Mapping[str, str] | None = LIBRARY_PANE_GRIP_NAMES,
        **kwargs: Any,
    ) -> None:
        """Build one Library grip; see ``AdaptivePaneGrip.__init__``.

        The two Library defaults are the only additions: the grip always
        carries ``LIBRARY_ADAPTIVE_READER_GRIP_CLASS`` (a bare grip built
        outside any shell still gets its ``:focus`` rule) and paints "Nav" for
        the Library pane.
        """
        super().__init__(
            pane,
            open=open,
            pane_label=pane_label,
            destination_class=destination_class,
            painted_names=painted_names,
            extra_classes=extra_classes,
            width=width,
            **kwargs,
        )


class LibraryAdaptiveReaderShell(AdaptivePaneShell):
    """The Library's adaptive reader shell: the shared shell, Library-keyed."""

    grip_type: ClassVar[type[AdaptivePaneGrip]] = LibraryAdaptiveReaderPaneGrip

    def __init__(
        self,
        library: Widget,
        items: Widget,
        work: Widget,
        layout: AdaptiveReaderEffectiveLayout,
        *,
        id_prefix: str,
        library_label: str,
        items_label: str,
        grip_classes: str = "",
        **kwargs: Any,
    ) -> None:
        """Assemble a Library shell; see ``AdaptivePaneShell.__init__``.

        Keeps the pre-B0 signature: every Library caller and subclass
        (``LibraryBrowseReaderShell``, ``LibraryArtifactsReaderShell``) builds
        it exactly as before.
        """
        super().__init__(
            library,
            items,
            work,
            layout,
            id_prefix=id_prefix,
            library_label=library_label,
            items_label=items_label,
            destination=LIBRARY_ADAPTIVE_READER_CLASSES,
            painted_names=LIBRARY_PANE_GRIP_NAMES,
            grip_classes=grip_classes,
            **kwargs,
        )
