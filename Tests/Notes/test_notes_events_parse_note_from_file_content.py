"""Unit coverage for ``notes_events._parse_note_from_file_content`` --
the file-import parser shared by the standalone Notes screen's Import
button and the Library canvas's "Import note" action (L2b.2 task 4)."""
from pathlib import Path

from tldw_chatbook.Event_Handlers.notes_events import _parse_note_from_file_content


def test_parse_note_from_file_content_json_with_title():
    path = Path("notes.json")
    content = '{"title": "From JSON", "content": "json body"}'

    title, body = _parse_note_from_file_content(path, content)

    assert title == "From JSON"
    assert body == "json body"


def test_parse_note_from_file_content_json_missing_content_falls_back_to_full_text():
    path = Path("notes.json")
    content = '{"title": "Title only"}'

    title, body = _parse_note_from_file_content(path, content)

    assert title == "Title only"
    assert body == content


def test_parse_note_from_file_content_markdown_uses_filename_stem():
    """``.md``/``.txt`` files never go through JSON/YAML parsing -- the
    filename stem is always the title and the full text is the body."""
    path = Path("My Great Notes.md")
    content = "# Heading\n\nSome body text."

    title, body = _parse_note_from_file_content(path, content)

    assert title == "My Great Notes"
    assert body == content


def test_parse_note_from_file_content_txt_uses_filename_stem():
    path = Path("plain.txt")
    content = "just plain text, no title key"

    title, body = _parse_note_from_file_content(path, content)

    assert title == "plain"
    assert body == content


def test_parse_note_from_file_content_empty_file_uses_filename_stem_and_empty_body():
    path = Path("empty.json")

    title, body = _parse_note_from_file_content(path, "   ")

    assert title == "empty"
    assert body == ""


def test_parse_note_from_file_content_unparsable_falls_back_to_filename():
    """A file with a JSON-like suffix but invalid JSON/YAML content still
    degrades to the filename-as-title fallback instead of raising."""
    path = Path("broken.json")
    content = "{not valid json or yaml: ["

    title, body = _parse_note_from_file_content(path, content)

    assert title == "broken"
    assert body == content
