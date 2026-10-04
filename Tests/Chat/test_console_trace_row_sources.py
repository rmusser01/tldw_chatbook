"""TASK-33621.2: unsaved provider rows get the source their request category admits.

Each case builds a real Capture-On request with ``build_console_request`` (the
aggregate that refused every system-prompt chat before the provider was
contacted) from the sources ``unsaved_row_sources`` assigns, and pairs it with
the labelling that the aggregate rejects for the same rows.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_prepared_request import (
    build_console_request,
    tagged_memory_message,
)
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleTraceCaptureMode,
    ProviderArtifactTraceProvenance,
    TraceProvenanceAlignmentError,
    TraceProvenanceSource,
)
from tldw_chatbook.Chat.console_trace_row_sources import (
    unsaved_row_sources,
    unsaved_trace_artifact_source,
)

pytestmark = pytest.mark.unit

SYSTEM = {"role": "system", "content": "You are terse."}
USER = {"role": "user", "content": "hello"}
REPLY = {"role": "assistant", "content": "answer"}
CALL = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
    ],
}
RESULT = {"role": "tool", "tool_call_id": "call-1", "content": "42"}
ACTIVE = {"role": "user", "content": "and now?"}


def _build(rows, sources):
    policy = FrozenTracePolicy(
        policy_id=new_opaque_id(),
        credential_filter_version="credentials-v1",
        pii_redaction_enabled=False,
        pii_ruleset_revision_id=None,
    )
    return build_console_request(
        rows,
        message_provenance=tuple(
            ProviderArtifactTraceProvenance(source, policy) for source in sources
        ),
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        capture_policy=policy,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )


def _by_role_only(rows):
    """Every row labelled by its role alone, ignoring its position."""
    return tuple(
        unsaved_trace_artifact_source(row, is_last=index == len(rows) - 1)
        for index, row in enumerate(rows)
    )


def test_a_leading_system_row_is_rendered_system_context():
    rows = [SYSTEM, USER, REPLY, ACTIVE]

    sources = unsaved_row_sources(rows)

    assert sources[0] is TraceProvenanceSource.RENDERED_SYSTEM
    request = _build(rows, sources)
    assert request.provenance is not None
    assert len(request.provenance.system) == 1
    # Negative control: the pre-fix label for every unsaved row.
    with pytest.raises(TraceProvenanceAlignmentError, match="mismatch: system"):
        _build(rows, (TraceProvenanceSource.ACTIVE_REQUEST,) * len(rows))


def test_a_marked_memory_row_is_conversation_memory():
    rows = [SYSTEM, dict(tagged_memory_message("Earlier: hello")), USER, ACTIVE]

    sources = unsaved_row_sources(rows)

    assert sources[:2] == (
        TraceProvenanceSource.RENDERED_SYSTEM,
        TraceProvenanceSource.CONVERSATION_MEMORY,
    )
    request = _build(rows, sources)
    assert request.provenance is not None
    assert len(request.provenance.memory) == 1
    with pytest.raises(TraceProvenanceAlignmentError, match="mismatch: memory"):
        _build(rows, _by_role_only(rows))


def test_a_system_row_inside_the_history_is_history():
    rows = [USER, REPLY, SYSTEM, ACTIVE]

    sources = unsaved_row_sources(rows)

    assert sources[2] is TraceProvenanceSource.ACTIVE_REQUEST
    _build(rows, sources)
    with pytest.raises(TraceProvenanceAlignmentError, match="category mismatch"):
        _build(rows, _by_role_only(rows))


def test_tool_rows_keep_their_tool_sources():
    rows = [SYSTEM, USER, CALL, RESULT, REPLY, ACTIVE]

    sources = unsaved_row_sources(rows)

    assert sources[2:4] == (
        TraceProvenanceSource.TOOL_CALL,
        TraceProvenanceSource.TOOL_RESULT,
    )
    _build(rows, sources)
    with pytest.raises(TraceProvenanceAlignmentError, match="tool_loop"):
        _build(
            rows,
            (TraceProvenanceSource.RENDERED_SYSTEM,)
            + (TraceProvenanceSource.ACTIVE_REQUEST,) * (len(rows) - 1),
        )


def test_the_final_row_is_the_active_request():
    rows = [SYSTEM, USER]

    assert unsaved_row_sources(rows) == (
        TraceProvenanceSource.RENDERED_SYSTEM,
        TraceProvenanceSource.ACTIVE_REQUEST,
    )
