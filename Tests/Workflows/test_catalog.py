"""Discovery is not execution, and unknown outputs are not typed references."""

from tldw_chatbook.Workflows import catalog


def test_discovery_keeps_unavailable_types_out_of_default_choices():
    assert {item.step_type for item in catalog.discover()} == {
        "media_ingest",
        "prompt",
        "llm",
        "wait_for_human",
        "notes",
    }
    all_types = {item.step_type: item for item in catalog.discover(show_all=True)}
    assert len(all_types) == 130
    assert all_types["branch"].disposition == "V2"
    assert all_types["map"].disposition == "V3"
    assert not all_types["rag_search"].available
    assert all_types["rag_search"].disposition == "V1 minimum"


def test_types_do_not_promise_human_edited_text():
    assert catalog.output_types("prompt") == (("text", "string"),)
    assert ("text", "unverified") in catalog.output_types("wait_for_human")
    assert catalog.output_types("unknown") == ()
