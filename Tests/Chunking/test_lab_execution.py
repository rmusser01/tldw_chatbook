"""Golden real-processor fixtures for reports, attribution, and saved apply."""

import json
import socket
import urllib.request

import pytest

from tldw_chatbook.Chunking import template_runtime as runtime
from tldw_chatbook.Chunking.lab_preflight import current_local_runtime, prepare_recipe


@pytest.fixture(autouse=True)
def no_network_or_download(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("Local execution attempted network or a download")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def run(body, text):
    assert hasattr(runtime, "execute_prepared"), (
        "Prepared execution reports are missing"
    )
    return runtime.execute_prepared(
        prepare_recipe(body, runtime=current_local_runtime()), text
    )


def words(*post, pre=()):
    return {
        "chunking": {"method": "words", "config": {"max_size": 2, "overlap": 0}},
        "preprocessing": list(pre),
        "postprocessing": list(post),
    }


def test_full_pipeline_has_identical_saved_and_unsaved_outputs():
    body = words(
        {"operation": "filter_empty", "config": {"min_length": 4}},
        {"operation": "add_metadata", "config": {"prefix": "{index}/{total}: "}},
        pre=[
            {"operation": "clean_markdown", "config": {"remove_formatting": True}},
            {"operation": "normalize_whitespace"},
        ],
    )
    source = "**one**  two three four x"
    report = run(body, source)
    assert report.transformed_text == "one two three four x"
    assert [c["text"] for c in report.chunks] == ["0/2: one two", "1/2: three four"]
    saved = {"name": "saved recipe", "template_json": json.dumps(body)}
    assert [c["text"] for c in runtime.apply_template(saved, source)] == [
        "0/2: one two",
        "1/2: three four",
    ]
    assert all("span" not in c for c in report.chunks)


def test_legitimate_empty_filter_and_preprocess_results_remain_empty():
    body = words({"operation": "filter_empty", "config": {"min_length": 100}})
    assert run(body, "one two").chunks == ()
    assert runtime.apply_template(body, "one two") == []
    empty = words(pre=[{"operation": "remove_headers", "config": {"patterns": [".*"]}}])
    assert run(empty, "one two").chunks == ()


def test_exact_span_survives_filter_and_repeated_text_is_not_guessed():
    report = run(
        words({"operation": "filter_empty", "config": {"min_length": 8}}),
        "one two three four",
    )
    assert report.chunks[0]["span"] == {
        "start": 8,
        "end": 18,
        "coordinate_space": "source",
    }
    repeated = run(words(), "one two one two")
    assert [c["text"] for c in repeated.chunks] == ["one two", "one two"]
    assert all("span" not in c for c in repeated.chunks)
    assert all(
        c["provenance"]["mapping"]["status"] == "unavailable" for c in repeated.chunks
    )


@pytest.mark.parametrize(
    "source", ["one  two one two", "one two one\ttwo", "one\ntwo one two"]
)
def test_normalized_word_chunks_cannot_borrow_another_occurrences_span(source):
    report = run(words(), source)
    assert [chunk["text"] for chunk in report.chunks] == ["one two", "one two"]
    assert all("span" not in chunk for chunk in report.chunks)
    assert all(
        chunk["provenance"]["mapping"]["status"] == "unavailable"
        for chunk in report.chunks
    )


def test_fixed_size_preserves_exact_whitespace_without_word_normalization():
    body = {
        "chunking": {"method": "fixed_size", "config": {"max_size": 2, "overlap": 0}}
    }
    report = run(body, "a  b")
    assert [chunk["text"] for chunk in report.chunks] == ["a ", " b"]
    assert [chunk["span"] for chunk in report.chunks] == [
        {"start": 0, "end": 2, "coordinate_space": "source"},
        {"start": 2, "end": 4, "coordinate_space": "source"},
    ]


def test_preprocessing_metadata_and_transformed_coordinates_survive():
    report = run(
        words(
            pre=[
                {"operation": "detect_language"},
                {"operation": "normalize_whitespace"},
            ]
        ),
        "one  two three four",
    )
    assert report.transformed_text == "one two three four"
    assert report.chunks[0]["span"] == {
        "start": 0,
        "end": 7,
        "coordinate_space": "transformed",
    }
    assert report.chunks[0]["provenance"]["preprocessing"][0]["metadata"] == {
        "detected_language": "en"
    }


@pytest.mark.parametrize(
    "operation,expected",
    [
        (
            {"operation": "merge_small", "config": {"min_size": 10, "separator": "|"}},
            ["one two|three four"],
        ),
        (
            {"operation": "add_overlap", "config": {"size": 3, "marker": "ctx"}},
            ["one two", "ctx\ntwo\nctx\nthree four"],
        ),
        (
            {"operation": "format_chunks", "config": {"template": "[{index}] {chunk}"}},
            ["[0] one two", "[1] three four"],
        ),
    ],
)
def test_postprocessing_uses_real_operations_and_invalidates_rewritten_maps(
    operation, expected
):
    report = run(words(operation), "one two three four")
    assert [c["text"] for c in report.chunks] == expected
    assert "span" not in report.chunks[-1]
    assert (
        report.chunks[-1]["provenance"]["operations"][-1]["operation"]
        == operation["operation"]
    )


@pytest.mark.parametrize(
    "operation,expected",
    [
        ({"operation": "filter_empty", "config": {"min_length": 8}}, ["three four"]),
        (
            {"operation": "merge_small", "config": {"min_size": 10, "separator": "|"}},
            ["one two|three four"],
        ),
        (
            {"operation": "add_overlap", "config": {"size": 3}},
            ["one two", "twothree four"],
        ),
    ],
)
def test_real_dict_engine_output_retains_contributors_through_postprocessing(
    operation, expected
):
    body = words(operation)
    body["chunking"]["config"]["hierarchical"] = True
    output = runtime.apply_template(body, "one two three four")
    assert [c["text"] for c in output] == expected
    assert "provenance" in output[-1], (
        "Shared execution discarded structured provenance"
    )
    contributors = output[-1]["provenance"]["contributors"]
    assert contributors[-1]["metadata"]["chunk_type"] == "text"
    # Engine counters remain engine metadata; authoritative final counters differ.
    assert contributors[-1]["metadata"]["chunk_index"] == 2
    assert output[-1]["chunk_index"] == len(output) - 1
    if operation["operation"] != "filter_empty":
        assert [item["metadata"]["start_offset"] for item in contributors] == [0, 8]


def test_fixed_size_and_sanitized_text_are_reported_honestly():
    body = {
        "chunking": {"method": "fixed_size", "config": {"max_size": 3, "overlap": 0}}
    }
    report = run(body, "abc\u202edef")
    assert report.transformed_text == "abc def"
    assert [c["text"] for c in report.chunks] == ["abc", " de", "f"]
    assert report.chunks[1]["span"]["coordinate_space"] == "transformed"


def test_prepared_runtime_or_document_tampering_is_refused():
    assert hasattr(runtime, "execute_prepared"), (
        "Prepared execution reports are missing"
    )
    recipe = prepare_recipe(words(), runtime=current_local_runtime())
    changed = recipe.model_copy(
        update={"effective_json": '{"chunking":{"method":"rolling_summarize"}}'}
    )
    with pytest.raises(ValueError, match="snapshot"):
        runtime.execute_prepared(changed, "one two")


def test_publication_copies_nested_metadata_and_rejects_invalid_chunks():
    from tldw_chatbook.Chunking.lab_models import ExecutionReport

    chunks = ({"text": "ok", "metadata": {"nested": [1]}, "provenance": {}},)
    report = ExecutionReport(chunks=chunks, transformed_text="ok")
    chunks[0]["metadata"]["nested"].append(2)
    assert report.model_dump()["chunks"][0]["metadata"] == {"nested": [1]}
    with pytest.raises(ValueError, match="text"):
        ExecutionReport(chunks=({"text": {"wrong": "shape"}},), transformed_text="ok")


def test_merging_preserves_each_contributors_prior_transformation():
    report = run(
        words(
            {"operation": "format_chunks", "config": {"template": "[{index}] {chunk}"}},
            {"operation": "merge_small", "config": {"min_size": 100, "separator": "|"}},
        ),
        "one two three four",
    )
    assert report.chunks[0]["text"] == "[0] one two|[1] three four"
    history = report.chunks[0]["provenance"]["operations"]
    assert [event["operation"] for event in history] == [
        "format_chunks",
        "format_chunks",
        "merge_small",
    ]
    assert [event["output_index"] for event in history[:2]] == [0, 1]


def test_hierarchical_saved_apply_offsets_use_whole_transformed_document():
    body = words()
    body["chunking"]["config"]["hierarchical"] = True
    result = runtime.apply_template(body, "one two\n\nthree four")
    assert [chunk["metadata"]["offset_basis"] for chunk in result] == [
        "source",
        "source",
    ]
    assert [(chunk["start_char"], chunk["end_char"]) for chunk in result] == [
        (0, 7),
        (9, 19),
    ]


def test_publication_revalidates_a_mutated_nested_record():
    report = run(words(), "one two")
    report.chunks[0]["text"] = {"not": "text"}
    with pytest.raises(ValueError, match="text"):
        report.model_dump_json()


def test_merge_after_empty_text_formatter_is_refused_at_admission():
    from tldw_chatbook.Chunking.lab_preflight import PreviewUnsupportedError

    body = words(
        {"operation": "format_chunks", "config": {"template": ""}},
        {"operation": "merge_small"},
    )
    with pytest.raises(PreviewUnsupportedError, match="postprocessing.1"):
        prepare_recipe(body, runtime=current_local_runtime())


@pytest.mark.parametrize(
    "operation,source,transformed,metadata",
    [
        (
            {"operation": "normalize_whitespace", "config": {"max_line_breaks": 1}},
            "a  b\n\n\nc",
            "a b\nc",
            {},
        ),
        (
            {"operation": "remove_headers", "config": {"patterns": ["^HEADER\\n"]}},
            "HEADER\na b",
            "a b",
            {},
        ),
        (
            {"operation": "extract_sections"},
            "# Heading\na b",
            "# Heading\na b",
            {"sections": [{"title": "Heading", "position": 0}]},
        ),
        (
            {"operation": "extract_sections", "config": {"pattern": "TITLE: (.+)"}},
            "TITLE: hello",
            "TITLE: hello",
            {"sections": [{"title": "hello", "position": 0}]},
        ),
        (
            {"operation": "clean_markdown", "config": {"remove_links": True}},
            "[label](url) text",
            "label text",
            {},
        ),
        (
            {"operation": "clean_markdown", "config": {"remove_images": True}},
            "![alt](url)text",
            "text",
            {},
        ),
    ],
)
def test_qualified_preprocessing_options_have_real_effects(
    operation, source, transformed, metadata
):
    report = run(words(pre=[operation]), source)
    assert report.transformed_text == transformed
    assert report.diagnostics[0]["operations"][0]["metadata"] == metadata


def test_qualified_word_size_overlap_and_tail_options_have_real_effects():
    body = words()
    body["chunking"]["config"].update(max_size=3, overlap=1, min_chunk_size=2)
    assert [c["text"] for c in run(body, "a b c d e").chunks] == ["a b c", "c d e"]


def test_vendor_sentence_preservation_word_loss_explains_lab_refusal():
    # Qualification evidence only: saved apply retains the pinned algorithm.
    # Lab refuses this value rather than quietly pretending it is faithful.
    body = words()
    body["chunking"]["config"].update(
        max_size=4, overlap=0, min_chunk_size=0, preserve_sentences=True
    )
    assert [
        c["text"]
        for c in runtime.apply_template(body, "aaaaaaaa bbbbbbbb cccccccc. d e")
    ] == ["aaaaaaaa bbbbbbbb cccccccc.", "e"]


def test_word_whitespace_normalization_does_not_claim_exact_source_span():
    report = run(words(), "one  two")
    assert report.chunks[0]["text"] == "one two"
    assert "span" not in report.chunks[0]
