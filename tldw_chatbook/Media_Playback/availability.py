"""Optional-dependency probing for video playback (task-3401.9).

``av`` (PyAV) decodes frames; ``textual-canvas`` backs the full player
screen's scrubber (task-3401.10). Both are optional: every surface in this
package degrades gracefully and points at the ``video_playback`` extra
instead of raising at import or render time.
"""

from __future__ import annotations

import importlib.util

#: Install guidance shown wherever a missing extra blocks playback.
VIDEO_PLAYBACK_EXTRA = "video_playback"
VIDEO_PLAYBACK_INSTALL_GUIDANCE = (
    "In-app video playback needs the video_playback extra: "
    'pip install "tldw_chatbook[video_playback]" '
    "(or: pip install av textual-canvas)"
)


def av_available() -> bool:
    """Whether PyAV (``av``) can be imported (frame decoding)."""
    return importlib.util.find_spec("av") is not None


def textual_canvas_available() -> bool:
    """Whether ``textual-canvas`` can be imported (player scrubber)."""
    return importlib.util.find_spec("textual_canvas") is not None


def playback_missing_reason() -> str | None:
    """Return the install-guidance reason when playback deps are incomplete.

    Returns:
        The guidance string when ``av`` is missing (the decode dependency --
        without it no playback surface works at all), else ``None``.
        ``textual-canvas`` alone being absent does not block the preview.
    """
    if not av_available():
        return VIDEO_PLAYBACK_INSTALL_GUIDANCE
    return None
