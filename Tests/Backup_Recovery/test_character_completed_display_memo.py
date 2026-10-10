"""A completed original Character display read establishes its current memo."""

import time

import pytest

from Tests.Backup_Recovery.test_character_refresh_finite_batch import (
    _original_calls,
    _receipt,
    _retired,
    _source_hashes,
)
from Tests.Backup_Recovery.test_console_presentation_cadence import _character, _display
from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.usefixtures("local_root")]


@pytest.mark.asyncio
async def test_completed_changed_character_display_does_not_issue_immediate_verification(
    tmp_path, record_property
):
    database = CharactersRAGDB(
        tmp_path / "completed-display.sqlite", "completed-display"
    )
    controller, screen, *_ = _character(database)
    source = _source_hashes()
    try:
        with _original_calls(database) as initial:
            start = time.monotonic()
            assert await _display(controller, screen)
            initial_elapsed = time.monotonic() - start
        _retired(database, initial)
        assert (
            not controller.state.error
            and controller.state.scope_fingerprint is not None
        )
        assert initial["pairs"] == 2 and initial["recent"] == 1
        assert initial["callbacks"] == 1 and len(initial["connections"]) == 1
        expected_state = controller.state
        after_first = controller._presentation_scope_key
        with _original_calls(database) as following:
            start = time.monotonic()
            assert await _display(controller, screen) is False
            follow_elapsed = time.monotonic() - start
        if following["connections"]:
            _retired(database, following)
        assert controller.state is expected_state
        assert source == _source_hashes()
        record_property("original_changed_display", _receipt(initial, initial_elapsed))
        record_property(
            "immediate_same_owner_display", _receipt(following, follow_elapsed)
        )
        record_property(
            "completed_display_memo_was_current",
            after_first == controller._presentation_owner_key(screen),
        )
        # Actual original readers/retirement qualify before this work-count oracle.
        assert (
            following["pairs"] == 0 and following["callbacks"] == 0
        ), "Completed current Character display issued an immediate native verification"
        assert after_first == controller._presentation_owner_key(screen)
    finally:
        database.close()
