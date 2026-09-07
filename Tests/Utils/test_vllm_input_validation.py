from __future__ import annotations

import pytest

from tldw_chatbook.Utils.input_validation import validate_vllm_draft_input


@pytest.mark.parametrize(
    ("control_id", "value"),
    [
        ("port", ""),
        ("port", "-"),
        ("gpu_memory_utilization", "."),
        ("gpu_memory_utilization", "1."),
        ("tensor_parallel_size", "+"),
        ("profile_name", "  exact spacing  "),
    ],
)
def test_vllm_lexical_boundary_preserves_partial_edits_exactly(
    control_id: str, value: str
) -> None:
    assert validate_vllm_draft_input(control_id, value) == value


@pytest.mark.parametrize(
    ("control_id", "value"),
    [
        ("profile_name", "x" * 121),
        ("python_environment", "x" * 4097),
        ("hugging_face_model", "x" * 97),
        ("bind_address", "x" * 256),
        ("existing_server_url", "x" * 2049),
        ("port", "123456"),
        ("profile_name", "bad\x00name"),
        ("profile_name", "bad\nname"),
        ("profile_name", "bad\x7fname"),
    ],
)
def test_vllm_lexical_boundary_rejects_oversize_or_control_text(
    control_id: str, value: str
) -> None:
    with pytest.raises(ValueError, match="vLLM setup value is invalid"):
        validate_vllm_draft_input(control_id, value)


@pytest.mark.parametrize("control_id,value", [(1, "text"), ("profile_name", 1)])
def test_vllm_lexical_boundary_rejects_forged_types(
    control_id: object, value: object
) -> None:
    with pytest.raises(ValueError, match="vLLM setup value is invalid"):
        validate_vllm_draft_input(control_id, value)
