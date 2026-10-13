"""Scalar-coordinate and source-safe boundary controls."""

import pytest


@pytest.mark.parametrize(
    "line,column,expected",
    [(0, 0, 0), (0, 1, 1), (0, 3, 3), (1, 0, 5), (1, 1, 6), (2, 0, 8), (9, 9, 8)],
)
def test_scalar_offsets_crlf_tabs_and_eof(line, column, expected):
    from tldw_chatbook.Document_Editing.source_positions import scalar_offset

    assert scalar_offset("😀é\r\n\tx\n", line, column) == expected


def test_real_parser_unicode_and_eof_positions(request_for):
    from tldw_chatbook.Document_Editing.engine import check_document

    for source, language in [('"😀"\r\n[', "jsonl"), ('a: "😀"\r\nb: [', "yaml")]:
        result = check_document(request_for(source, language=language))
        assert result.diagnostics[0].start == len(source)
        assert result.diagnostics[0].end == len(source)


def test_lone_surrogate_is_bounded_source_error(request_for):
    from tldw_chatbook.Document_Editing.engine import check_document

    result = check_document(request_for('"\ud800"'))
    assert result.diagnostics[0].code == "syntax"
    assert result.diagnostics[0].start == 1


def test_out_of_line_columns_do_not_jump_to_next_line():
    from tldw_chatbook.Document_Editing.source_positions import scalar_offset

    assert scalar_offset("😀é\r\nnext", 0, 99) == 3
    assert scalar_offset("😀é\r\nnext", -1, 99) == 3
