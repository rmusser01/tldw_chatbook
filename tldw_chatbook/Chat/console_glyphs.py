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

#: Voice capture is live (composer mic button's recording state and the inline
#: voice chip's timer head). CN-05 (TASK-2154.13): recording used to share
#: ``GLYPH_IN_PROGRESS``'s "●" with agent-run markers, one glyph with two
#: meanings in adjacent regions; "●" is now agent-run-only, and the ringed
#: dot reads as "the mic is on" without copying the run marker.
GLYPH_VOICE_RECORDING = "◉"

#: The voice pipeline is working WITHOUT capturing right now: model warmup /
#: preparing messages (prefixed by the screen) and transcribing phase
#: (composer voice chip). CN-05 (TASK-2154.13): these used to share
#: ``GLYPH_TEMPORARY``'s "◌" with temporary-session tabs; "◌" is now
#: temporary-only, and the half-filled circle reads as "in progress" rather
#: than "outline of a thing". TASK-2154.19 hangs ASCII fallbacks off both
#: voice glyphs, same as the rest of this vocabulary.
GLYPH_VOICE_WORKING = "◐"

#: task-31207: the dim placeholder drawn by a conversation row's icon control
#: when no custom icon is set -- an empty slot inviting the click that opens
#: the appearance picker.
GLYPH_APPEARANCE_PLACEHOLDER = "▢"

#: task-31207: what the icon control renders in ASCII-glyph mode. Arbitrary
#: user emoji cannot map meaningfully to ASCII (and emoji rendering is what
#: that mode exists to avoid), so a set icon degrades to a colored ``*`` and
#: the unset placeholder to a dim ``+`` -- two distinguishable affordances
#: without any font-dependent glyph.
GLYPH_APPEARANCE_ASCII_SET = "*"
GLYPH_APPEARANCE_ASCII_UNSET = "+"

#: Section-row status -> glyph, the ONE vocabulary every Console rail
#: section shares (TASK-32334; the fleet panel's own map moved here).
#: Unknown/empty statuses get no glyph from ``status_glyph``; call sites
#: that need a running default use ``STATUS_GLYPHS.get(status, GLYPH_IN_PROGRESS)``.
STATUS_GLYPHS = {
    "done": GLYPH_DONE,
    "running": GLYPH_IN_PROGRESS,
    "error": "✗",  # the fleet's failure mark (NOT GLYPH_CLOSE ✕, the
    "cancelled": "✗",  # close-button glyph); ASCII fallback "[X]"
    "stuck": "⚠",
    "blocked": "⚠",
}


def status_glyph(status: str) -> str:
    """Return the shared glyph for a section-row status, "" when none.

    Args:
        status: Row status class ("" means no status -> no glyph).
    """
    return STATUS_GLYPHS.get(str(status or ""), "")
