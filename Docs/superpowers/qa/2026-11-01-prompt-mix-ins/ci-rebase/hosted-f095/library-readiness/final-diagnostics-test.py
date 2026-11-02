"""Failed Library waits report metadata without changing their decisions."""

from types import SimpleNamespace

import pytest

from Tests.UI import library_note_links_support as support

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("row_present", [False, True])
async def test_source_dom_failure_keeps_original_assertion_and_reports_flags(
    monkeypatch, row_present
):
    row = SimpleNamespace(is_parent_active=False)
    queried_one = []

    def query(selector):
        return [row] if selector == "#library-row-browse-notes" and row_present else []

    def query_one(selector):
        queried_one.append(selector)
        assert row_present
        return row

    screen = SimpleNamespace(
        _library_loaded=True,
        _library_lookup_error=None,
        _library_snapshot_rendered_generation=2,
        _library_snapshot_state_generation=3,
        _library_entry_reconcile_dirty=True,
        query=query,
        query_one=query_one,
    )
    app = SimpleNamespace(screen=SimpleNamespace(id="library"))
    failure = AssertionError("timed out waiting for the healthy Library source DOM")
    calls = []

    async def mounted(_app, _pilot):
        return screen

    async def refused(_pilot, predicate, what):
        calls.append((predicate(), what))
        raise failure

    monkeypatch.setattr(support, "_mounted_library", mounted)
    monkeypatch.setattr(support, "_until", refused)
    with pytest.raises(AssertionError) as caught:
        await support._library(app, object())
    assert caught.value is failure
    assert calls == [(False, "the healthy Library source DOM")]
    assert queried_one == (["#library-row-browse-notes"] if row_present else [])
    (note,) = failure.__notes__
    assert "'rendered_generation': 2" in note
    assert "'state_generation': 3" in note
    assert "'entry_reconcile_dirty': True" in note
    assert f"'notes_row_count': {int(row_present)}" in note
    assert f"'notes_row_parent_active': {False if row_present else None}" in note
    assert "'library_is_current_screen': False" in note
    assert "'current_screen_id': 'library'" in note
