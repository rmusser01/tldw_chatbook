from tldw_chatbook.Chunking.engine import Chunker


def test_chunk_with_metadata_json_offsets_match_source():

    text = '{ "data": [ {"a": 1}, {"b": 2} ], "meta": "x" }'
    chunker = Chunker()

    results = chunker.chunk_text_with_metadata(
        text,
        method="json",
        max_size=1,
        overlap=0,
        align_text_to_source=True,
    )

    assert results, "Expected JSON chunking to return chunks"
    for res in results:
        s = res.metadata.start_char
        e = res.metadata.end_char
        assert res.text == text[s:e]


def test_chunk_with_metadata_xml_offsets_match_source():

    text = "<root><a>Alpha one</a><b>Beta two</b></root>"
    chunker = Chunker()

    results = chunker.chunk_text_with_metadata(
        text,
        method="xml",
        max_size=1,
        overlap=0,
        align_text_to_source=True,
    )

    assert results, "Expected XML chunking to return chunks"
    for res in results:
        s = res.metadata.start_char
        e = res.metadata.end_char
        assert res.text == text[s:e]
