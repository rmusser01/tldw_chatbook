"""Can this machine actually play Console reply speech?

Two independent playback paths exist: the in-process streaming sink
(`Audio/streaming_sink.py`, `sounddevice`/PortAudio, PCM and WAV only)
and the external player binaries (`TTS/audio_player.py`'s format-aware
catalogue). Neither knows about the other, and each can be absent on a
given machine -- the sink when `sounddevice` isn't installed, every
capable player binary on a stock Fedora Workstation (no mpv/mplayer/
ffplay, and paplay/pw-play are WAV/FLAC-only by the conservative
catalogue). This module is the single predicate the Console speech
pipeline consults so a request is never synthesized into a format this
machine provably cannot play.

The probes are deliberately re-read at call time (no caching): both are
cheap (`find_spec` / PATH lookups) and call-time reads keep a mid-run
install -- or a test's monkeypatch -- immediately effective.
"""

from __future__ import annotations

from tldw_chatbook.Audio.streaming_sink import sink_available
from tldw_chatbook.TTS.audio_player import find_player_for_format

#: Formats the streaming sink can play when it is available. PCM is
#: sink-only: raw PCM16 has no container, so no external player binary in
#: the catalogue can decode a bare ".pcm" artifact.
_SINK_FORMATS: frozenset[str] = frozenset({"pcm", "wav"})

#: The adaptive fallback format: playable through the sink on any machine
#: where `sounddevice` resolves, decodable by every catalogue player, and
#: supported as an output format by every TTS provider that supports more
#: than one format.
_ADAPTIVE_FORMAT = "wav"


def locally_playable_formats() -> frozenset[str]:
    """Return every audio format this machine can play right now.

    The union of the sink's formats (when `sounddevice` is importable)
    and the formats some available external player can definitely decode.
    """
    formats: set[str] = set()
    if sink_available():
        formats |= _SINK_FORMATS
    # External players can only serve file formats; PCM is sink-only.
    for candidate in ("mp3", "opus", "aac", "flac", "wav"):
        if find_player_for_format(candidate) is not None:
            formats.add(candidate)
    return frozenset(formats)


def format_playable_locally(audio_format: str | None) -> bool:
    """Return whether `audio_format` can be played on this machine."""
    if not isinstance(audio_format, str) or not audio_format:
        return False
    return audio_format in locally_playable_formats()


def adapt_console_speech_format(resolved_format: str | None) -> str:
    """Return the format a Console speech request should actually use.

    Unchanged when `resolved_format` is already playable here. Otherwise
    `"wav"` when WAV would be playable (the streaming-sink path -- this
    is the branch that keeps reply speech working on a machine whose
    player binaries cannot decode the configured format). Otherwise the
    input unchanged: when nothing is playable there is no better format
    to request, and the playback-failure surfacing owns telling the user
    what to install or reconfigure.
    """
    if not isinstance(resolved_format, str) or not resolved_format:
        return resolved_format  # type: ignore[return-value]
    if format_playable_locally(resolved_format):
        return resolved_format
    if format_playable_locally(_ADAPTIVE_FORMAT):
        return _ADAPTIVE_FORMAT
    return resolved_format


def playback_remedy(audio_format: str | None) -> str:
    """UI-ready remedy copy for audio this machine provably cannot play.

    One sentence, two concrete outs: install a decoding player, or move
    the TTS output format to WAV (which the in-process sink plays without
    any system binary).
    """
    shown = audio_format if isinstance(audio_format, str) and audio_format else "audio"
    return (
        f"Reply audio can't be played on this machine: no player for '{shown}' "
        "and no built-in audio output. Install a player (e.g. `dnf install mpv` "
        "or `apt install mpv`), or set Settings ▸ Speech & TTS ▸ Output format "
        "to WAV."
    )
