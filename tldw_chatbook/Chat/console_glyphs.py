"""Shared Console glyph language (spec §4)."""

# The two disclosure glyphs are owned by the shared rail widget that draws
# them, and re-exported here so Console keeps one import site for its glyph
# vocabulary. They are not Console-specific -- Evals' library rail uses the
# same triangles. See ADR-034 (backlog/decisions/034-shared-rail-disclosure-glyphs.md).
from tldw_chatbook.Widgets.destination_rail import (  # noqa: E402
    GLYPH_COLLAPSED,
    GLYPH_EXPANDED,
)

GLYPH_ACTIVE = "▸"
GLYPH_IN_PROGRESS = "●"
GLYPH_DONE = "✓"
GLYPH_CLOSE = "✕"
GLYPH_COLLAPSE_LEFT = "◂"
GLYPH_COLLAPSE_RIGHT = "▸"

#: Per-source-type glyphs for RAG scope items (media vs. note) -- used by
#: ConsoleScopePickerModal's checkbox rows (rag-scope-narrowing spec §4).
GLYPH_SOURCE_MEDIA = "▦"
GLYPH_SOURCE_NOTE = "✎"

#: Temporary (never-saved) Console session tab marker. A dotted ring reads as
#: "outline of a thing, not the thing" -- deliberately unlike the solid run
#: markers above, which mean a run is happening. Decoded in the tab tooltip;
#: the 19-cell tab label has no room for a word.
GLYPH_TEMPORARY = "◌"
