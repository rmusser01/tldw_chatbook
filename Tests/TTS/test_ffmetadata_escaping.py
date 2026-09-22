"""FFMETADATA1 values carry user text and must be escaped (tier-2 S03/S04 P3).

`AudioService.create_m4b_with_chapters` interpolates the audiobook's title,
artist, album, genre, date and description -- plus every chapter title --
straight into an FFMETADATA1 document. The format reserves ``=``, ``;``,
``#``, ``\\`` and newline; a description carrying a newline followed by
``[CHAPTER]`` injects chapter records into the user's own M4B.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.TTS.audio_service import escape_ffmetadata


@pytest.mark.unit
def test_reserved_characters_are_backslash_escaped():
    assert escape_ffmetadata("a=b") == r"a\=b"
    assert escape_ffmetadata("a;b") == r"a\;b"
    assert escape_ffmetadata("a#b") == r"a\#b"


@pytest.mark.unit
def test_backslash_is_escaped_before_the_other_characters():
    # A single trailing backslash must not escape the delimiter ffmpeg
    # appends after the value.
    assert escape_ffmetadata("a\\") == "a\\\\"
    assert escape_ffmetadata("a\\=b") == "a\\\\\\=b"


@pytest.mark.unit
def test_newline_cannot_open_a_new_ffmetadata_record():
    injected = "Chapter notes\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0"
    escaped = escape_ffmetadata(injected)
    # Every newline survives, but each is now preceded by a backslash, which
    # makes it a continuation rather than the start of a new record or key.
    assert escaped.count("\n") == injected.count("\n")
    assert all(
        escaped[index - 1] == "\\"
        for index, character in enumerate(escaped)
        if character == "\n"
    ), escaped
    assert r"TIMEBASE\=1/1000" in escaped


@pytest.mark.unit
def test_ordinary_text_is_returned_unchanged():
    assert escape_ffmetadata("The Hobbit") == "The Hobbit"
    assert escape_ffmetadata("") == ""
