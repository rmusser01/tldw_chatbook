"""Reference compatibility follows the pinned server adapter's output contract."""

import pytest

from tldw_chatbook.UI.Workflows_Modules.reference_picker import reference_choices


@pytest.fixture
def media_document():
    # tldw_server dev 2e1a5e58d3344a1efd578efb4dbfb1c9465e8767:
    # adapters/media/ingest.py:74-79 initializes metadata as a list;
    # :109-111 emits text only for nonempty extracted content.
    return {
        "steps": [
            {"id": "ingest", "type": "media_ingest"},
            {"id": "consumer", "type": "prompt"},
        ]
    }


def test_media_text_reference_warns_that_runtime_validation_is_required(media_document):
    choices = {
        expression: label
        for label, expression in reference_choices(
            media_document, "consumer", "string", "steps"
        )
    }
    assert "unverified — runtime validation required" in choices["{{ ingest.text }}"]


@pytest.mark.parametrize("target_type", ["array", "object", "string"])
def test_media_metadata_reference_is_offered_only_for_array_targets(
    media_document, target_type
):
    choices = {
        expression: label
        for label, expression in reference_choices(
            media_document, "consumer", target_type, "steps"
        )
    }
    if target_type == "array":
        assert choices["{{ ingest.metadata }}"].endswith(" · array")
    else:
        assert "{{ ingest.metadata }}" not in choices
