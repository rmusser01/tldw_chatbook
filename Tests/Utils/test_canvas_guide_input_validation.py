"""Closed shared boundary for model-issued Canvas guide arguments."""

import pytest

from tldw_chatbook.Canvas.guide import CANVAS_GUIDE_PATHS
from tldw_chatbook.Utils import input_validation


@pytest.mark.parametrize("topic", CANVAS_GUIDE_PATHS)
def test_canvas_guide_arguments_accept_exact_packaged_topics(topic):
    assert input_validation.validate_canvas_guide_arguments({"topic": topic}) == {
        "topic": topic
    }


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"topic": "basics", "extra": "private-canary"},
        {"topic": "private-canary"},
        {"topic": " basics"},
        {"topic": b"basics"},
        {"topic": True},
        {"topic": ["basics"]},
    ],
)
def test_canvas_guide_arguments_reject_without_echoing_input(value):
    with pytest.raises(ValueError, match="invalid Canvas guide arguments") as error:
        input_validation.validate_canvas_guide_arguments(value)
    assert "private-canary" not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__
