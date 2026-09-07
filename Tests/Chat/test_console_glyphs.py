"""Glyph-per-meaning guard for the Console glyph vocabulary.

CN-05 (TASK-2154.13): "◌" used to mark temporary session tabs AND
voice-pipeline working states; "●" used to mark agent runs AND live voice
capture. One glyph must mean one thing everywhere it appears, so the voice
lifecycle got its own pair (◉ recording / ◐ working). These tests lock the
separation so a future reuse of a glyph re-collides loudly, here, instead of
silently in two adjacent UI regions again.
"""

from tldw_chatbook.Chat.console_chat_models import CONSOLE_RUN_MARKER_GLYPHS
from tldw_chatbook.Chat.console_glyphs import (
    GLYPH_IN_PROGRESS,
    GLYPH_TEMPORARY,
    GLYPH_VOICE_RECORDING,
    GLYPH_VOICE_WORKING,
)

_RUN_GLYPHS = {glyph for glyph in CONSOLE_RUN_MARKER_GLYPHS.values() if glyph}


def test_voice_glyphs_do_not_collide_with_run_or_temporary_markers():
    """◉/◐ belong to the voice lifecycle only."""
    assert GLYPH_VOICE_RECORDING not in _RUN_GLYPHS
    assert GLYPH_VOICE_WORKING not in _RUN_GLYPHS
    assert GLYPH_VOICE_RECORDING != GLYPH_TEMPORARY
    assert GLYPH_VOICE_WORKING != GLYPH_TEMPORARY
    assert GLYPH_VOICE_RECORDING != GLYPH_VOICE_WORKING


def test_run_and_temporary_glyphs_stay_distinct():
    """The original CN-05 collisions must not be reintroduced at the source."""
    assert GLYPH_IN_PROGRESS != GLYPH_TEMPORARY
    assert GLYPH_TEMPORARY not in _RUN_GLYPHS


# ---------------------------------------------------------------------------
# AC-01 (TASK-2154.19): ASCII-safe fallback mode.
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from tldw_chatbook.Chat.console_chat_models import ConsoleRunMarker  # noqa: E402
from tldw_chatbook.Chat.console_glyphs import (  # noqa: E402
    GLYPH_ACTIVE,
    GLYPH_CLOSE,
    GLYPH_COLLAPSE_LEFT,
    GLYPH_COLLAPSE_RIGHT,
    GLYPH_COLLAPSED,
    GLYPH_DONE,
    GLYPH_EXPANDED,
    GLYPH_SOURCE_MEDIA,
    GLYPH_SOURCE_NOTE,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import (  # noqa: E402
    ConsoleComposerBar,
)
from tldw_chatbook.Widgets.Console.console_session_surface import (  # noqa: E402
    ConsoleSessionSurface,
)
from tldw_chatbook.Widgets.glyph_fallback import (  # noqa: E402
    ASCII_GLYPH_FALLBACKS,
    ascii_glyph_mode,
    resolve_glyph,
    resolve_glyph_text,
    set_ascii_glyph_mode,
)

_VOCABULARY = {
    GLYPH_ACTIVE,
    GLYPH_CLOSE,
    GLYPH_COLLAPSE_LEFT,
    GLYPH_COLLAPSE_RIGHT,
    GLYPH_COLLAPSED,
    GLYPH_DONE,
    GLYPH_EXPANDED,
    GLYPH_IN_PROGRESS,
    GLYPH_SOURCE_MEDIA,
    GLYPH_SOURCE_NOTE,
    GLYPH_TEMPORARY,
    GLYPH_VOICE_RECORDING,
    GLYPH_VOICE_WORKING,
    ConsoleComposerBar.CURSOR_GLYPH,
    "📎",
} | _RUN_GLYPHS


@pytest.fixture
def ascii_mode():
    """Run one test with ASCII mode on and ALWAYS restore the off default."""
    set_ascii_glyph_mode(True)
    try:
        yield
    finally:
        set_ascii_glyph_mode(False)


def test_ascii_fallback_map_covers_the_console_vocabulary():
    """Every status marker has an ASCII substitute assigned."""
    missing = _VOCABULARY - set(ASCII_GLYPH_FALLBACKS)
    assert not missing, f"glyphs without an ASCII fallback: {sorted(missing)}"


def test_ascii_fallbacks_are_pure_ascii_and_nonempty():
    for glyph, fallback in ASCII_GLYPH_FALLBACKS.items():
        assert fallback, f"{glyph!r} maps to an empty fallback"
        assert fallback.isascii(), f"{glyph!r} fallback {fallback!r} is not ASCII"


def test_resolve_glyph_is_identity_in_default_mode():
    """The out-of-the-box glyph set is unchanged (AC-01 default)."""
    assert ascii_glyph_mode() is False
    for glyph in _VOCABULARY:
        assert resolve_glyph(glyph) == glyph
    assert (
        resolve_glyph_text("◌ scratch ◐ Transcribing…") == "◌ scratch ◐ Transcribing…"
    )


def test_resolve_glyph_substitutes_in_ascii_mode(ascii_mode):
    assert resolve_glyph("●") == "[*]"
    assert resolve_glyph("◌") == "[~]"
    assert resolve_glyph("✕") == "x"
    assert resolve_glyph("▾") == "v"
    # Unknown glyphs pass through untouched in both modes.
    assert resolve_glyph("?") == "?"
    assert resolve_glyph("") == ""


def test_resolve_glyph_text_maps_embedded_markers(ascii_mode):
    assert (
        resolve_glyph_text(ConsoleComposerBar.VOICE_CHIP_TRANSCRIBING_LABEL)
        == "(~) Transcribing…"
    )
    assert resolve_glyph_text("Composer ▾") == "Composer v"
    assert resolve_glyph_text("📎 2 files") == "[+] 2 files"
    # Plain ASCII text is identity even with the mode on.
    assert resolve_glyph_text("Send") == "Send"


def test_tab_label_uses_ascii_markers_in_ascii_mode(ascii_mode):
    label = ConsoleSessionSurface._tab_label(
        "scratch", marker=ConsoleRunMarker.RUNNING, ephemeral=True
    )
    # Temporary and running stack, temporary outermost: "[~] [*] scratch".
    assert label.startswith("[~]")
    assert "[*]" in label
    assert not any(char in label for char in ("●", "◌"))
