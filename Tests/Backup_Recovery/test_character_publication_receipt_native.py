"""Original stock publication refuses memo after entry/success generation drift."""

from contextlib import contextmanager
import inspect
import sys
import threading

import pytest

from Tests.Backup_Recovery.test_character_refresh_finite_batch import (
    _original_calls,
    _retired,
    _source_hashes,
)
from Tests.Backup_Recovery.test_console_presentation_cadence import _character, _display
from Tests.Backup_Recovery.test_participant_lifetimes import local_root as local_root  # noqa: PLC0414
from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules import character_context as module

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.usefixtures("local_root")]


@contextmanager
def _actual_publication_edge(controller, screen, edge, character_id):
    witness = OriginalStorageUnitObserver({}, lambda: True, lambda unit: None)
    refresh = inspect.getattr_static(
        module.ConsoleCharacterContextController, "refresh"
    )
    batch = inspect.getattr_static(
        module.ConsoleCharacterContextController, "_refresh_recent_batch"
    )
    for function in (refresh, batch):
        witness._pin(function)
    witness.slots.extend(
        [
            (module.ConsoleCharacterContextController, "refresh", refresh),
            (module.ConsoleCharacterContextController, "_refresh_recent_batch", batch),
        ]
    )
    witness.codes = {
        refresh.__code__: "refresh_invocation",
        batch.__code__: "stock_batch",
    }
    monitor = sys.monitoring
    witness.tool = next(
        slot
        for slot in range(5, 0, -1)
        if slot != monitor.DEBUGGER_ID and monitor.get_tool(slot) is None
    )
    monitor.use_tool_id(witness.tool, "character-publication-invocation")
    witness.active = witness.installed = True
    main, events = threading.current_thread(), []

    def start(code, offset):
        witness._start(code, offset)
        if code is refresh.__code__ and edge == "entry_refused":
            frame = witness._frame(code)
            if frame.f_locals.get("self") is controller:
                assert threading.current_thread() is main
                screen.app_instance.app_config = dict(screen.app_instance.app_config)
                events.append("original_refresh_entered_then_owner_retargeted")

    def returned(code, offset, value):
        witness._return(code, offset, value)
        if code is batch.__code__ and edge == "generation_after_publication":
            frame = witness._frame(code)
            if frame.f_locals.get("self") is controller:
                assert threading.current_thread() is main
                assert (
                    not controller.state.error
                    and controller.state.scope_fingerprint is not None
                )
                assert (
                    controller.state.scope_fingerprint.current_character_id
                    == character_id
                )
                controller.invalidate_scope()  # Original public generation fence.
                events.append("original_batch_return_then_generation_invalidated")

    witness.registered = {
        monitor.events.PY_START: start,
        monitor.events.PY_RETURN: returned,
    }
    try:
        for event, callback in witness.registered.items():
            assert monitor.register_callback(witness.tool, event, callback) is None
        for code in witness.codes:
            monitor.set_local_events(
                witness.tool, code, monitor.events.PY_START | monitor.events.PY_RETURN
            )
        assert monitor.get_events(witness.tool) == 0
        yield events
    finally:
        receipt = witness.close()
        assert receipt["complete"] and receipt["original_source_current"], receipt
        assert (
            receipt["global_events"] == 0 and receipt["hooks_retired_before_inactive"]
        ), receipt
        assert monitor.get_tool(witness.tool) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("edge", ["entry_refused", "generation_after_publication"])
async def test_prior_success_or_superseding_generation_cannot_establish_memo(
    tmp_path, edge
):
    database = CharactersRAGDB(tmp_path / "receipt-negative.sqlite", "receipt-negative")
    controller, screen, _active, current, _conversation = _character(database)
    hashes = _source_hashes()
    try:
        character_id = database.add_character_card({"name": "Receipt selection"})
        assert type(character_id) is int and character_id > 0  # noqa: E721 - actual persisted integer identity.
        with _original_calls(database) as initial:
            assert await _display(controller, screen)
        _retired(database, initial)
        assert (
            not controller.state.error
            and controller.state.scope_fingerprint is not None
        )
        prior_state = controller.state
        # An actual ambient selection change forces the existing changed branch.
        current[0] = (character_id, "Receipt selection")
        with _actual_publication_edge(controller, screen, edge, character_id) as events:
            with _original_calls(database) as observed:
                await _display(controller, screen)
        if observed["connections"]:
            _retired(database, observed)
        assert len(events) == 1, events
        assert controller._presentation_scope_key is None
        if edge == "entry_refused":
            assert controller.state is prior_state and not controller.state.error
        else:
            assert controller.state.scope_fingerprint is None
        assert hashes == _source_hashes()
    finally:
        database.close()
