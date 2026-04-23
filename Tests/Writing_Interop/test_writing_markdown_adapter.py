from tldw_chatbook.Writing_Interop.writing_markdown_adapter import (
    markdown_to_plain_text,
    markdown_to_server_content,
    parse_server_content_json,
    server_content_to_markdown,
)


def test_markdown_round_trips_through_deterministic_wrapper():
    markdown = "# Title\n\n- one\n- two\n\nParagraph."
    content = markdown_to_server_content(markdown)

    assert content == {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "attrs": {
                    "tldw_chatbook_markdown": True,
                    "format": "markdown",
                    "version": 1,
                },
                "content": [{"type": "text", "text": markdown}],
            }
        ],
    }
    assert server_content_to_markdown(content, None) == markdown


def test_markdown_to_plain_text_keeps_content():
    markdown = "# Heading\n\n- item one\n- item two\n\n**Body** text"
    plain = markdown_to_plain_text(markdown)
    assert "Heading" in plain
    assert "item one" in plain
    assert "Body text" in plain


def test_server_tiptap_without_wrapper_falls_back_to_content_plain():
    content = {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "not wrapped markdown"}]}
        ],
    }
    assert server_content_to_markdown(content, "fallback plain") == "fallback plain"


def test_parse_server_content_json_invalid_or_unknown_falls_back():
    assert parse_server_content_json("{not json}") is None
    assert parse_server_content_json('["not-a-dict"]') is None

    parsed = parse_server_content_json('{"type":"doc","content":[]}')
    assert parsed == {"type": "doc", "content": []}
