"""One strict-JSON contract for wire and storage boundaries (TASK-32805.5).

The wire family (hosted_chat/qwencloud) used to accept duplicate object keys
last-wins while the storage family (provider_continuation/thinking_blocks)
rejected them, so a dup-key tool-call argument accepted on the wire raised an
uncaught error when its continuation checkpoint was built. Both now delegate to
``strict_json_loads`` and agree.
"""

import pytest

from tldw_chatbook.Utils.input_validation import StrictJSONError, strict_json_loads


def test_accepts_a_valid_document():
    assert strict_json_loads('{"a": 1, "b": [1, 2, 3.5]}') == {"a": 1, "b": [1, 2, 3.5]}


def test_rejects_duplicate_keys_by_default():
    with pytest.raises(StrictJSONError):
        strict_json_loads('{"a": 1, "a": 2}')


def test_tolerates_duplicate_keys_when_asked():
    assert strict_json_loads('{"a": 1, "a": 2}', reject_duplicate_keys=False) == {"a": 2}


def test_rejects_non_finite_constants():
    for raw in ("NaN", '{"x": Infinity}', '[-Infinity]'):
        with pytest.raises(StrictJSONError):
            strict_json_loads(raw)


def test_rejects_shape_past_the_caps():
    with pytest.raises(StrictJSONError):
        strict_json_loads("[" * 200 + "]" * 200, max_depth=10)


def test_rejects_malformed_json():
    with pytest.raises(StrictJSONError):
        strict_json_loads("{not json")


def test_wire_and_storage_agree_on_a_duplicate_key_payload():
    """The exact drift: a dup-key payload must be refused at BOTH the wire and
    the storage boundary, so a wire-accepted value cannot fail at storage."""
    dup = '{"a": 1, "a": 2}'

    # Public contract: rejected.
    with pytest.raises(StrictJSONError):
        strict_json_loads(dup)

    # Wire family (hosted_chat): rejected via its sentinel, not last-wins.
    from tldw_chatbook.LLM_Calls import hosted_chat as hc

    assert hc._strict_json_loads(dup) is hc._JSON_DECODE_FAILED

    # Storage family (provider_continuation): rejected by raising.
    from tldw_chatbook.Chat import provider_continuation as pc

    with pytest.raises(pc._InvalidContinuation):
        pc._strict_json_loads(dup)


def test_wire_and_storage_agree_on_a_valid_payload():
    """The complement: an ordinary tool-argument object is accepted at both."""
    valid = '{"expression": "2 + 2", "note": "ok"}'

    from tldw_chatbook.LLM_Calls import hosted_chat as hc
    from tldw_chatbook.Chat import provider_continuation as pc

    decoded = strict_json_loads(valid)
    assert hc._strict_json_loads(valid) == decoded
    assert pc._strict_json_loads(valid) == decoded
