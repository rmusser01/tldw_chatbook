"""TASK-34000.25: the ``[<title>](media://<media-uuid>)`` source line is pure
and refuses anything that is not a stable uuid (reading-desk design §5.1).

Fast lane (``Tests/CI/test_ci_queue_pressure_contract.py``): no app, no DB.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Library.library_media_source_link import (
    MediaSourceLinkError,
    media_source_line,
    media_source_uuid,
)

UUID = "4B0E1C2D-7A53-4F0E-9B61-2C8D5E7F9A10"


def test_source_line_names_the_title_and_the_media_uuid():
    assert (
        media_source_line("A field guide to Markdown tables", UUID)
        == "[A field guide to Markdown tables](media://4b0e1c2d-7a53-4f0e-9b61-2c8d5e7f9a10)"
    )


def test_source_line_is_one_line_with_brackets_escaped():
    line = media_source_line("Notes [draft]\nwith\tbreaks  ", f"  {UUID} ")
    assert "\n" not in line
    assert line.startswith("[Notes \\[draft\\] with breaks](media://")
    assert line.endswith(")")


def test_blank_title_still_renders_a_readable_link():
    assert media_source_line("", UUID).startswith("[Untitled media](media://")
    assert media_source_line(None, UUID).startswith("[Untitled media](media://")


@pytest.mark.parametrize(
    "value",
    [7, "7", "media-7", "", "   ", None, True, "local:media:7", "not-a-uuid"],
)
def test_an_integer_or_malformed_id_is_refused_never_linked(value):
    """A link that named a device-local row would read as portable and break
    elsewhere -- so the integer fallback is refused loudly, never silent."""
    with pytest.raises(MediaSourceLinkError):
        media_source_uuid(value)
    with pytest.raises(MediaSourceLinkError):
        media_source_line("Title", value)
