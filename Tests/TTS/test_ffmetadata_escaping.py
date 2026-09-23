"""FFMETADATA1 values carry user text and must be escaped (tier-2 S03/S04 P3).

`AudioService.create_m4b_with_chapters` interpolates the audiobook's title,
artist, album, genre, date and description -- plus every chapter title --
straight into an FFMETADATA1 document. The format reserves ``=``, ``;``,
``#``, ``\\`` and newline; a description carrying a newline followed by
``[CHAPTER]`` injects chapter records into the user's own M4B.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tldw_chatbook.TTS.audio_service import (
    build_ffmetadata_document,
    escape_ffmetadata,
    write_ffmetadata_file,
)


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


@pytest.mark.unit
def test_the_document_escapes_every_user_controlled_field():
    """Escaping is only worth testing where it is actually applied.

    `escape_ffmetadata` had unit coverage, but nothing exercised the
    document that reaches ffmpeg, so a field that simply forgot to call it
    would not have been caught. `create_m4b_with_chapters` itself needs
    pydub and an ffmpeg binary (pydub is not even installed here), so an
    end-to-end test of it would skip on most machines and cover nothing --
    `build_ffmetadata_document` is the same code path without them.
    """
    document = build_ffmetadata_document(
        {
            "tit=le": "Book=One",
            "description": "notes\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0",
        },
        ["Ch#1", "Ch;2"],
        [0, 1000, 2000],
        2000,
    )

    # Keys are user text too, not just values.
    assert r"tit\=le=Book\=One" in document
    assert r"title=Ch\#1" in document
    assert r"title=Ch\;2" in document
    # The description tried to open a third chapter. Exactly the two real
    # records survive, because every newline it carried is now a
    # continuation rather than a record boundary.
    assert document.count("\n[CHAPTER]\n") == 2


@pytest.mark.unit
def test_the_metadata_file_is_opened_as_utf8_with_no_newline_translation():
    """Pins the two open() settings the escaping depends on.

    Default text mode rewrites every ``\\n`` to ``os.linesep``, so on
    Windows an escaped newline (backslash immediately followed by LF)
    becomes backslash, CR, LF: the backslash escapes the CR, the LF ends
    the record, and the injection `escape_ffmetadata` prevents comes back
    one layer down. The default encoding is the platform's preferred one,
    not UTF-8, so a Japanese or emoji chapter title raises
    UnicodeEncodeError on a cp1252 box.

    Asserting the settings rather than the bytes is deliberate: on POSIX
    ``os.linesep`` is already LF, so a byte-level test passes here whether
    or not the bug is fixed and would only ever fail on Windows CI.
    """
    with patch(
        "tldw_chatbook.TTS.audio_service.tempfile.NamedTemporaryFile"
    ) as opener:
        write_ffmetadata_file(";FFMETADATA1\n")

    assert opener.call_args.kwargs["encoding"] == "utf-8"
    assert opener.call_args.kwargs["newline"] == "\n"


@pytest.mark.unit
def test_the_written_file_round_trips_escaped_newlines_and_non_latin1_text():
    path = Path(
        write_ffmetadata_file(
            build_ffmetadata_document(
                {"description": "notes\n[CHAPTER]"}, ["第一章 🎧"], [0, 1000], 1000
            )
        )
    )
    try:
        raw = path.read_bytes()
        # The escape survives as backslash-LF, never backslash-CR-LF.
        assert b"\\\n" in raw
        assert b"\r" not in raw
        assert "第一章 🎧" in raw.decode("utf-8")
    finally:
        path.unlink()
