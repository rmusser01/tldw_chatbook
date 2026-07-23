"""Shared Console glyph language (spec §4)."""

GLYPH_EXPANDED = "▾"
GLYPH_COLLAPSED = "▸"
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
