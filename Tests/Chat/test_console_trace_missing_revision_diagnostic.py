"""TASK-34100.5 AC#7 (entry-exit-handoff-06), diagnostics half.

The first send after some setup exits failed with 'Trace capture blocked'
(trace_reservation failed, trace_revision_unavailable) in 2 of 5 live runs,
and nothing recorded WHICH revision the reservation could not find, so the
trigger stayed unidentified. The refusal now logs the missing opaque revision
ids (no content), which of them is the active turn's, and how many were found.
"""

from __future__ import annotations

import pytest
from loguru import logger

from Tests.Chat.test_console_trace_runtime import (  # noqa: F401 - fixture
    _saved_message,
    _semantic_request,
    make_database,
)
from tldw_chatbook.Chat.console_prepared_request import (
    prepare_provider_request,
    resolve_request_capacity,
)
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    SavedRevisionTraceProvenance,
)
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory


def test_a_missing_revision_row_is_named_in_the_diagnostic(
    tmp_path, make_database
) -> None:
    database = make_database(tmp_path / "trace-missing.sqlite", "trace-missing")
    conversation_id = database.add_conversation({"title": "missing revision"})
    assert conversation_id is not None
    _prior_id, prior = _saved_message(database, conversation_id, "prior")
    ghost = SavedRevisionTraceProvenance(new_opaque_id())
    policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
    semantic = _semantic_request(
        [
            {"role": "user", "content": "prior"},
            {"role": "user", "content": "active turn whose row is missing"},
        ],
        [prior, ghost],
        policy,
    )
    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        provider="openai",
        model="gpt-test",
        capacity=resolve_request_capacity(context_window_tokens=None),
        apply_safety_window=False,
    )
    records: list[str] = []
    sink = logger.add(lambda message: records.append(str(message)), level="WARNING")
    try:
        with pytest.raises(ValueError, match="trace_revision_unavailable"):
            ConsoleTraceBoundaryFactory(database)(
                prepared, None, ConsoleRequestRoute.FRESH
            )
    finally:
        logger.remove(sink)

    named = [line for line in records if "no saved revision row" in line]
    assert len(named) == 1, records
    assert ghost.revision_id in named[0]
    assert "active_missing=['" + ghost.revision_id in named[0]
    assert prior.revision_id not in named[0].split("present=")[0]
    assert "active turn whose row is missing" not in named[0]
