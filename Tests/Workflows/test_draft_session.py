"""Draft ownership: losing pending text or publishing stale Saved is a bug."""

import asyncio
import json
from threading import Event

import pytest

from Tests.Workflows.helpers import prompt_definition
from tldw_chatbook.DB.Workflows_DB import WorkflowsDB
from tldw_chatbook.Workflows import draft_session
from tldw_chatbook.Workflows.document_service import DocumentService
from tldw_chatbook.Workflows.draft_session import DraftSession
from tldw_chatbook.Workflows.models import (
    DraftConflict,
    DraftWriteFailed,
    InvalidDraft,
    RevisionConflict,
)


@pytest.fixture
def owner(tmp_path):
    db = WorkflowsDB(tmp_path / "drafts.sqlite3")
    documents = DocumentService(db)
    revision = documents.create(json.dumps(prompt_definition()))
    yield documents, revision
    db.close()


@pytest.mark.parametrize("fragment", ["", "tru", "{", '0,"injected":true'])
async def test_owned_fragment_repairs_only_its_field_and_restart_requires_explicit_repair(
    owner, fragment
):
    documents, base = owner
    session = DraftSession(documents)
    await session.select(base.workflow_id, base.revision_id)
    original = session.current
    pending = session.update_field("/steps/0/retry", fragment)
    assert pending.error and pending.last_valid_json == original.last_valid_json
    with pytest.raises(InvalidDraft):
        session.update_field("/steps/1/retry", "1")
    assert session.current == pending
    await session.flush()
    # More typing after persistence uses the same exact original field source.
    repaired = session.update_field("/steps/0/retry", "2")
    assert repaired.error is None
    # Switching to raw typing before debounce must not lose the field-repair proof.
    named = session.update(
        documents.edit_field(repaired.raw_text, "/name", "After repair")
    )
    await session.flush()
    assert session.current == named
    pending = session.update_field("/steps/0/retry", '0,"injected":true')
    await session.close()
    reopened = DraftSession(documents)
    await reopened.select(base.workflow_id, base.revision_id)
    assert reopened.current == pending
    assert reopened.field_edit is None and reopened.raw_repair_required
    assert reopened.update(pending.raw_text) == pending
    await reopened.flush()
    with pytest.raises(InvalidDraft):
        await reopened.save_revision()
    raw = documents.edit_field(
        named.last_valid_json, "/steps/0/retry", "3", as_json=True
    )
    changed = reopened.update(raw)
    assert changed.error and changed.last_valid_json == named.last_valid_json
    version = reopened.confirmation_version
    reopened.update(raw + " ")
    with pytest.raises(DraftConflict):
        await reopened.repair_raw(changed, confirmation_version=version)
    source = reopened.current
    fixed = await reopened.repair_raw(
        source, confirmation_version=reopened.confirmation_version
    )
    assert fixed.generation > source.generation and fixed.error is None
    assert fixed.raw_text == source.raw_text
    assert documents.field_text(fixed.last_valid_json, "/steps/0/retry") == "3"
    assert documents.field_text(fixed.last_valid_json, "/steps/0/injected") == ""
    await reopened.save_revision()
    await reopened.close()


@pytest.mark.parametrize("fail", [False, True])
async def test_cancelled_fragment_write_retains_exact_repair_provenance(
    owner, monkeypatch, fail
):
    documents, base = owner
    session = DraftSession(documents)
    await session.select(base.workflow_id, base.revision_id)
    session.update_field("/steps/0/retry", "")
    started, release = Event(), Event()
    put = documents.put_draft

    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(5)
        if fail:
            raise OSError("test disk failure")
        return put(*args, **kwargs)

    monkeypatch.setattr(documents, "put_draft", blocked)
    caller = asyncio.create_task(session.flush())
    try:
        assert await asyncio.to_thread(started.wait, 3)
        caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)
        session.update_field("/steps/0/retry", "t")
        proof = session.field_edit
    finally:
        release.set()
    if fail:
        with pytest.raises(DraftWriteFailed):
            await session.flush()
        assert session.field_edit == proof
    monkeypatch.setattr(documents, "put_draft", put)
    session.update_field("/steps/0/retry", "2")
    await session.close()
    assert (
        documents.field_text(
            documents.get_draft(base.workflow_id, base.revision_id).raw_text,
            "/steps/0/retry",
        )
        == "2"
    )


async def test_new_json_typing_during_valid_field_flush_and_full_discard(
    owner, monkeypatch
):
    documents, base = owner
    session = DraftSession(documents)
    await session.select(base.workflow_id, base.revision_id)
    session.update_field("/steps/0/retry", "1")
    started, release = Event(), Event()
    put = documents.put_draft

    def hold_completion(*args, **kwargs):
        result = put(*args, **kwargs)
        started.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(documents, "put_draft", hold_completion)
    flush = asyncio.create_task(session.flush())
    try:
        assert await asyncio.to_thread(started.wait, 3)
        session.update_field("/steps/0/retry", "2")
    finally:
        release.set()
    await flush
    monkeypatch.setattr(documents, "put_draft", put)
    session.update_field("/steps/0/retry", "")
    await session.flush()
    discarded = await session.discard_draft()
    assert discarded.raw_text == base.raw_json and discarded.error is None
    await session.close()


async def test_discard_failed_pending_fragment_restores_durable_field_provenance(
    owner, monkeypatch
):
    documents, base = owner
    session = DraftSession(documents)
    await session.select(base.workflow_id, base.revision_id)
    durable = session.update_field("/steps/0/retry", "")
    await session.flush()
    session.update_field("/steps/0/retry", "t")
    put = documents.put_draft

    def fail(*args, **kwargs):
        raise OSError("test write failure")

    monkeypatch.setattr(documents, "put_draft", fail)
    with pytest.raises(DraftWriteFailed):
        await session.flush()
    assert await session.discard_pending() == durable
    assert session.field_edit.text == ""
    monkeypatch.setattr(documents, "put_draft", put)
    session.update_field("/steps/0/retry", "1")
    await session.close()


@pytest.mark.parametrize("operation", ["history", "raw"])
@pytest.mark.parametrize("fail", [False, True])
async def test_cancelled_accepted_authoring_transaction_locks_until_close_drains(
    owner, monkeypatch, operation, fail
):
    documents, base = owner
    session = DraftSession(documents)
    await session.select(base.workflow_id, base.revision_id)
    head = await session.save_revision()
    await session.select(base.workflow_id, base.revision_id)
    if operation == "raw":
        session.update_field("/steps/0/retry", '0,"injected":true')
        await session.flush()
        session.update(base.raw_json)
    await session.flush()
    old = session.current
    started, release = Event(), Event()
    method = "copy_revision_to_head" if operation == "history" else "put_draft"
    original = getattr(documents, method)

    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(5)
        if fail:
            raise OSError("test transaction failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(documents, method, blocked)
    caller = asyncio.create_task(
        session.copy_revision_to_head(base, head)
        if operation == "history"
        else session.repair_raw(old, confirmation_version=session.confirmation_version)
    )
    try:
        assert await asyncio.to_thread(started.wait, 3)
        caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)
        assert session.editing_locked
        with pytest.raises(DraftWriteFailed):
            session.update(base.raw_json + " ")
        closing = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        assert not closing.done()
    finally:
        release.set()
    await closing
    assert not session.editing_locked
    if fail:
        assert session.current == old
    else:
        assert session.current.error is None
        assert session.current.base_revision_id == (
            head.revision_id if operation == "history" else base.revision_id
        )
    assert (
        documents.get_draft(
            session.current.workflow_id, session.current.base_revision_id
        )
        == session.current
    )


@pytest.mark.parametrize("invalid_newer", [False, True])
async def test_blocked_save_new_typing_explicit_recovery_second_save_and_reopen(
    owner,
    monkeypatch,
    invalid_newer,
):
    documents, base = owner
    session = DraftSession(documents)
    await session.select(base.workflow_id, base.revision_id)
    session.update(documents.edit_field(base.raw_json, "/name", "First saved"))
    started, release = Event(), Event()
    save = documents.save_revision

    def hold_completion(*args):
        saved = save(*args)
        started.set()
        assert release.wait(5)
        return saved

    monkeypatch.setattr(documents, "save_revision", hold_completion)
    saving = asyncio.create_task(session.save_revision())
    try:
        assert await asyncio.to_thread(started.wait, 3)
        newer = session.update(
            documents.edit_field(session.current.raw_text, "/name", "Typed during save")
        )
        pending = session.update('{"unfinished":') if invalid_newer else newer
    finally:
        release.set()
    head = await saving
    assert session.current == pending
    assert session.base == base
    await session.flush()
    if invalid_newer:
        await session.close()
        session = DraftSession(documents)
        await session.select(base.workflow_id, base.revision_id)
        assert session.current == pending
        assert session.current.last_valid_json == newer.last_valid_json
        with pytest.raises(InvalidDraft):
            await session.recover_to_head(
                pending, head, confirmation_version=session.confirmation_version
            )
        assert session.current == pending
        newer = session.update(newer.raw_text)
    copied = await session.recover_to_head(
        newer,
        head,
        confirmation_version=session.confirmation_version,
    )
    assert session.current == copied
    assert documents.get_draft(base.workflow_id, base.revision_id) == newer
    second = await session.save_revision()
    assert second.parent_revision_ids == (head.revision_id,)
    assert (
        documents.field_text(second.raw_json, "/name", as_json=False)
        == "Typed during save"
    )
    await session.close()
    reopened = DraftSession(documents)
    await reopened.select(base.workflow_id, base.revision_id)
    assert reopened.current == newer
    await reopened.select(head.workflow_id, head.revision_id)
    assert reopened.current == copied
    await reopened.close()


def stale_source(documents, base):
    documents.put_draft(base.workflow_id, base.revision_id, base.raw_json, 0)
    head = documents.save_revision(base.workflow_id, base.revision_id, 0)
    source = documents.put_draft(
        base.workflow_id,
        base.revision_id,
        documents.edit_field(base.raw_json, "/name", "Newer"),
        1,
    )
    return source, head


async def test_recovery_refuses_an_outstanding_owner_transition(owner, monkeypatch):
    documents, base = owner
    session = DraftSession(documents)
    await session.select(base.workflow_id, base.revision_id)
    started, release = Event(), Event()
    save = documents.save_revision

    def blocked(*args):
        head = save(*args)
        started.set()
        assert release.wait(5)
        return head

    monkeypatch.setattr(documents, "save_revision", blocked)
    saving = asyncio.create_task(session.save_revision())
    try:
        assert await asyncio.to_thread(started.wait, 3)
        recovering = asyncio.create_task(
            session.recover_to_head(
                session.current,
                documents.list_workflows()[0],
                confirmation_version=session.confirmation_version,
            )
        )
        await asyncio.sleep(0)
        rejected_immediately = recovering.done()
    finally:
        release.set()
    await saving
    result = (await asyncio.gather(recovering, return_exceptions=True))[0]
    await session.close()
    assert rejected_immediately
    assert isinstance(result, DraftWriteFailed)


@pytest.mark.parametrize("change", ["typing", "selection", "invalid"])
async def test_recovery_confirmation_cannot_survive_changes(owner, change):
    documents, base = owner
    source, head = stale_source(documents, base)
    session = DraftSession(documents)
    await session.select(base.workflow_id, base.revision_id)
    version = session.confirmation_version
    if change == "selection":
        await session.select(head.workflow_id, head.revision_id)
        await session.select(base.workflow_id, base.revision_id)
    else:
        session.update('{"unfinished":')
        if change == "typing":
            session.update(source.raw_text)
    current = session.current
    with pytest.raises(DraftConflict):
        await session.recover_to_head(source, head, confirmation_version=version)
    assert session.current == current
    if change == "invalid":
        with pytest.raises(InvalidDraft):
            await session.recover_to_head(
                current, head, confirmation_version=session.confirmation_version
            )
        assert current.last_valid_json == source.last_valid_json
    await session.close()


@pytest.mark.parametrize("fail", [False, True])
async def test_cancelled_recovery_keeps_owner_lock_until_close_drains_transaction(
    owner,
    monkeypatch,
    fail,
):
    documents, base = owner
    source, head = stale_source(documents, base)
    session = DraftSession(documents)
    await session.select(base.workflow_id, base.revision_id)
    copy = documents.copy_draft_to_head
    started, release = Event(), Event()

    def blocked(*args):
        started.set()
        assert release.wait(5)
        if fail:
            raise OSError("private store detail")
        return copy(*args)

    monkeypatch.setattr(documents, "copy_draft_to_head", blocked)
    recovering = asyncio.create_task(
        session.recover_to_head(
            source,
            head,
            confirmation_version=session.confirmation_version,
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 3)
        recovering.cancel()
        with pytest.raises(asyncio.CancelledError):
            await recovering
        assert session.editing_locked
        with pytest.raises(DraftWriteFailed):
            session.update('{"must_not_replace":')
        closing = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        assert not closing.done()
    finally:
        release.set()
    await closing
    assert not session.editing_locked
    assert documents.get_draft(base.workflow_id, base.revision_id) == source
    assert "private store detail" not in session.status
    if fail:
        assert session.current == source
        assert documents.get_draft(head.workflow_id, head.revision_id) is None
    else:
        assert session.current.base_revision_id == head.revision_id
        assert (
            documents.get_draft(head.workflow_id, head.revision_id) == session.current
        )
    reopened = DraftSession(documents)
    await reopened.select(session.current.workflow_id, session.current.base_revision_id)
    assert reopened.current == session.current
    await reopened.close()


async def test_update_is_pure_and_invalid_text_preserves_last_valid_projection(owner):
    documents, revision = owner
    session = DraftSession(documents)
    await session.select(revision.workflow_id, revision.revision_id)
    raw = documents.edit_field(revision.raw_json, "/steps/0/config/template", "changed")
    valid = session.update(raw)
    invalid = session.update('{"steps": [')
    assert invalid.error is not None
    assert invalid.last_valid_json == valid.last_valid_json
    assert documents.get_draft(revision.workflow_id, revision.revision_id) is None
    await session.flush()
    assert session.status == "Saved locally"
    assert session.current.last_valid_json == valid.last_valid_json
    assert (
        documents.get_draft(revision.workflow_id, revision.revision_id).raw_text
        == '{"steps": ['
    )
    await session.close()


async def test_failed_flush_vetoes_selection_retry_and_pending_discard_are_distinct(
    owner, monkeypatch
):
    documents, revision = owner
    session = DraftSession(documents)
    await session.select(revision.workflow_id, revision.revision_id)
    durable = session.update(
        documents.edit_field(revision.raw_json, "/name", "Durable")
    )
    await session.flush()
    pending = session.update('{"steps": [')
    write = documents.put_draft

    def fail(*args, **kwargs):
        raise OSError("private path must not appear in status")

    monkeypatch.setattr(documents, "put_draft", fail)
    with pytest.raises(DraftWriteFailed):
        await session.select(revision.workflow_id, revision.revision_id)
    assert session.current == pending
    assert session.status == "Not saved locally — Retry"
    restored = await session.discard_pending()
    assert restored.raw_text == durable.raw_text
    monkeypatch.setattr(documents, "put_draft", write)
    discarded = await session.discard_draft()
    assert discarded.raw_text == revision.raw_json
    assert discarded.generation > pending.generation
    await session.close()


@pytest.mark.parametrize("operation", ["discard_pending", "discard_draft"])
@pytest.mark.parametrize("same_workflow", [False, True])
async def test_discard_queued_behind_selection_cannot_retarget_or_lose_new_edits(
    owner, monkeypatch, operation, same_workflow
):
    documents, base = owner
    session = DraftSession(documents)
    await session.select(base.workflow_id, base.revision_id)
    if same_workflow:
        target = await session.save_revision()
        await session.select(base.workflow_id, base.revision_id)
    else:
        definition = prompt_definition()
        del definition["metadata"]["tldw_workflow"]
        target = documents.create(json.dumps(definition))
    target_draft = documents.put_draft(
        target.workflow_id, target.revision_id, target.raw_json + " \n", 1
    )
    started, release = Event(), Event()
    read = documents.get_revision

    def blocked(*args):
        started.set()
        assert release.wait(5)
        return read(*args)

    monkeypatch.setattr(documents, "get_revision", blocked)
    selecting = asyncio.create_task(
        session.select(target.workflow_id, target.revision_id)
    )
    try:
        assert await asyncio.to_thread(started.wait, 3)
        latest = session.update('{"newer opaque text": [')
        discarding = asyncio.create_task(getattr(session, operation)())
        await asyncio.sleep(0)
        waited_for_selection = not discarding.done()
    finally:
        release.set()
    await selecting
    result = (await asyncio.gather(discarding, return_exceptions=True))[0]
    await session.close()
    assert documents.get_draft(base.workflow_id, base.revision_id) == latest
    assert session.current == target_draft
    assert documents.get_draft(target.workflow_id, target.revision_id) == target_draft
    assert waited_for_selection
    assert isinstance(result, DraftConflict)


@pytest.mark.parametrize("operation", ["discard_pending", "discard_draft"])
async def test_selection_intent_during_discard_flush_invalidates_discard(
    owner, monkeypatch, operation
):
    documents, base = owner
    definition = prompt_definition()
    del definition["metadata"]["tldw_workflow"]
    target = documents.create(json.dumps(definition))
    session = DraftSession(documents)
    await session.select(base.workflow_id, base.revision_id)
    session.update(base.raw_json + " \n")
    started, release = Event(), Event()
    put = documents.put_draft

    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return put(*args, **kwargs)

    monkeypatch.setattr(documents, "put_draft", blocked)
    flushing = asyncio.create_task(session.flush())
    try:
        assert await asyncio.to_thread(started.wait, 3)
        discarding = asyncio.create_task(getattr(session, operation)())
        await asyncio.sleep(0)
        selecting = asyncio.create_task(
            session.select(target.workflow_id, target.revision_id)
        )
        await asyncio.sleep(0)
    finally:
        release.set()
    await flushing
    result = (await asyncio.gather(discarding, return_exceptions=True))[0]
    selected = await selecting
    await session.close()
    assert session.current == selected
    assert session.base == target
    assert isinstance(result, DraftConflict)


@pytest.mark.parametrize("field_edit", [False, True])
async def test_pending_discard_preserves_edits_made_during_failed_flush(
    owner, monkeypatch, field_edit
):
    documents, base = owner
    session = DraftSession(documents)
    await session.select(base.workflow_id, base.revision_id)
    await session.flush()
    session.update(base.raw_json + " \n")
    started, release = Event(), Event()
    put = documents.put_draft

    def failed_write(*args, **kwargs):
        started.set()
        assert release.wait(5)
        raise OSError("private test write failure")

    monkeypatch.setattr(documents, "put_draft", failed_write)
    flushing = asyncio.create_task(session.flush())
    try:
        assert await asyncio.to_thread(started.wait, 3)
        discarding = asyncio.create_task(session.discard_pending())
        await asyncio.sleep(0)
        latest = (
            session.update_field("/steps/0/retry", '0,"opaque":true')
            if field_edit
            else session.update('{"newer opaque text": [')
        )
        proof = session.field_edit
    finally:
        release.set()
    with pytest.raises(DraftWriteFailed):
        await flushing
    result = (await asyncio.gather(discarding, return_exceptions=True))[0]
    current, retained_proof, status = (
        session.current,
        session.field_edit,
        session.status,
    )
    monkeypatch.setattr(documents, "put_draft", put)
    await session.close()
    assert current == latest
    assert retained_proof == proof
    assert status == "Not saved locally — Retry"
    assert documents.get_draft(base.workflow_id, base.revision_id) == latest
    assert isinstance(result, DraftConflict)


async def test_full_discard_does_not_adopt_newer_protected_raw_after_flush(
    owner, monkeypatch
):
    documents, base = owner
    session = DraftSession(documents)
    await session.select(base.workflow_id, base.revision_id)
    session.update_field("/steps/0/retry", '0,"opaque":true')
    await session.flush()
    started, release = Event(), Event()
    put = documents.put_draft

    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return put(*args, **kwargs)

    monkeypatch.setattr(documents, "put_draft", blocked)
    discarding = asyncio.create_task(session.discard_draft())
    try:
        assert await asyncio.to_thread(started.wait, 3)
        latest = session.update(base.raw_json + " \n")
        assert latest.error
    finally:
        release.set()
    result = (await asyncio.gather(discarding, return_exceptions=True))[0]
    await session.close()
    assert session.current == latest
    assert session.raw_repair_required
    assert documents.get_draft(base.workflow_id, base.revision_id) == latest
    assert isinstance(result, DraftConflict)


@pytest.mark.parametrize("operation", ["discard_pending", "discard_draft"])
async def test_cancelled_discard_leaves_physical_flush_owned_and_newer_text_intact(
    owner, monkeypatch, operation
):
    documents, base = owner
    session = DraftSession(documents)
    await session.select(base.workflow_id, base.revision_id)
    session.update(base.raw_json + " \n")
    started, release = Event(), Event()
    put = documents.put_draft

    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return put(*args, **kwargs)

    monkeypatch.setattr(documents, "put_draft", blocked)
    flushing = asyncio.create_task(session.flush())
    try:
        assert await asyncio.to_thread(started.wait, 3)
        discarding = asyncio.create_task(getattr(session, operation)())
        await asyncio.sleep(0)
        discarding.cancel()
        with pytest.raises(asyncio.CancelledError):
            await discarding
        latest = session.update('{"newer opaque text": [')
        closing = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        assert not closing.done()
    finally:
        release.set()
    await flushing
    await closing
    assert session.current == latest
    assert documents.get_draft(base.workflow_id, base.revision_id) == latest


@pytest.mark.parametrize("stage", ["flush", "repair"])
async def test_cancelled_full_discard_drains_only_the_accepted_protected_raw_stage(
    owner, monkeypatch, stage
):
    documents, base = owner
    session = DraftSession(documents)
    await session.select(base.workflow_id, base.revision_id)
    session.update_field("/steps/0/retry", '0,"opaque":true')
    await session.flush()
    started, release = Event(), Event()
    put = documents.put_draft

    def blocked(*args, **kwargs):
        if (kwargs.get("repair_source") is not None) == (stage == "repair"):
            started.set()
            assert release.wait(5)
        return put(*args, **kwargs)

    monkeypatch.setattr(documents, "put_draft", blocked)
    discarding = asyncio.create_task(session.discard_draft())
    try:
        assert await asyncio.to_thread(started.wait, 3)
        discarding.cancel()
        with pytest.raises(asyncio.CancelledError):
            await discarding
        assert session.editing_locked == (stage == "repair")
        if stage == "flush":
            latest = session.update(base.raw_json + " \n")
        else:
            with pytest.raises(DraftWriteFailed):
                session.update(base.raw_json + " \n")
        closing = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        assert not closing.done()
    finally:
        release.set()
    await closing
    assert not session.editing_locked
    if stage == "flush":
        assert session.current == latest
        assert session.raw_repair_required
    else:
        assert session.current.raw_text == base.raw_json
        assert session.current.error is None
    reopened = DraftSession(documents)
    recovered = await reopened.select(base.workflow_id, base.revision_id)
    assert recovered == session.current
    await reopened.close()


async def test_blocked_write_never_marks_new_generation_saved_and_close_drains(
    owner, monkeypatch
):
    documents, revision = owner
    session = DraftSession(documents)
    await session.select(revision.workflow_id, revision.revision_id)
    started, release = Event(), Event()
    write = documents.put_draft

    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return write(*args, **kwargs)

    monkeypatch.setattr(documents, "put_draft", blocked)
    session.update('{"first":')
    flush = asyncio.create_task(session.flush())
    try:
        assert await asyncio.to_thread(started.wait, 3)
        latest = session.update('{"second":')
        assert session.status != "Saved locally"
    finally:
        release.set()
    await flush
    assert session.current == latest
    assert (
        documents.get_draft(revision.workflow_id, revision.revision_id).raw_text
        == '{"second":'
    )
    await session.close()


async def test_debounce_writes_after_500ms_and_reopen_recovers(owner):
    documents, revision = owner
    session = DraftSession(documents)
    await session.select(revision.workflow_id, revision.revision_id)
    session.update('{"steps": [')
    await asyncio.sleep(0.2)
    assert documents.get_draft(revision.workflow_id, revision.revision_id) is None
    await asyncio.sleep(0.4)
    assert (
        documents.get_draft(revision.workflow_id, revision.revision_id).raw_text
        == '{"steps": ['
    )
    await session.close()
    reopened = DraftSession(documents)
    recovered = await reopened.select(revision.workflow_id, revision.revision_id)
    assert recovered.raw_text == '{"steps": ['
    assert reopened.status == "Draft recovered"
    await reopened.close()


async def test_debounce_policy_controls_status_and_actual_write_deadline(
    owner, monkeypatch
):
    documents, base = owner
    monkeypatch.setattr(draft_session, "DRAFT_DEBOUNCE_SECONDS", 0.05, raising=False)
    session = DraftSession(documents)
    await session.select(base.workflow_id, base.revision_id)
    saved = asyncio.Event()
    session.subscribe(
        lambda: saved.set() if session.status == "Saved locally" else None
    )
    try:
        pending = session.update('{"unfinished":')
        status = session.status
        await asyncio.wait_for(saved.wait(), 0.3)
        assert status == "Pending — saved after 50 ms"
        assert documents.get_draft(base.workflow_id, base.revision_id) == pending
    finally:
        await session.close()


async def test_cancelled_flush_keeps_physical_write_and_selected_identity(
    owner, monkeypatch
):
    documents, revision = owner
    session = DraftSession(documents)
    await session.select(revision.workflow_id, revision.revision_id)
    started, release = Event(), Event()
    write = documents.put_draft

    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return write(*args, **kwargs)

    monkeypatch.setattr(documents, "put_draft", blocked)
    session.update('{"pending":')
    flush = asyncio.create_task(session.flush())
    try:
        assert await asyncio.to_thread(started.wait, 3)
        flush.cancel()
        with pytest.raises(asyncio.CancelledError):
            await flush
        assert session.current.workflow_id == revision.workflow_id
        assert session.status != "Saved locally"
    finally:
        release.set()
    await session.close()
    assert (
        documents.get_draft(revision.workflow_id, revision.revision_id).raw_text
        == '{"pending":'
    )


async def test_cancelled_save_still_publishes_exact_new_base(owner, monkeypatch):
    documents, revision = owner
    session = DraftSession(documents)
    await session.select(revision.workflow_id, revision.revision_id)
    session.update(documents.edit_field(revision.raw_json, "/name", "Saved name"))
    await session.flush()
    started, release = Event(), Event()
    save = documents.save_revision

    def blocked(*args):
        started.set()
        assert release.wait(5)
        return save(*args)

    monkeypatch.setattr(documents, "save_revision", blocked)
    saving = asyncio.create_task(session.save_revision())
    try:
        assert await asyncio.to_thread(started.wait, 3)
        saving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await saving
    finally:
        release.set()
    await session.close()
    assert session.current.base_revision_id == documents.list_workflows()[0].revision_id


async def test_close_drains_a_cancelled_selection_read_before_store_teardown(
    owner, monkeypatch
):
    documents, revision = owner
    session = DraftSession(documents)
    started, release = Event(), Event()
    read = documents.get_revision

    def blocked(*args):
        started.set()
        assert release.wait(5)
        return read(*args)

    monkeypatch.setattr(documents, "get_revision", blocked)
    selecting = asyncio.create_task(
        session.select(revision.workflow_id, revision.revision_id)
    )
    try:
        assert await asyncio.to_thread(started.wait, 3)
        selecting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await selecting
        closing = asyncio.create_task(session.close())
        await asyncio.sleep(0.05)
        assert not closing.done(), (
            "Store teardown must wait for the physical owner read"
        )
    finally:
        release.set()
    await closing
    assert session.current.workflow_id == revision.workflow_id


async def test_cancelled_close_retains_committed_save_and_reconciles_base(
    owner, monkeypatch
):
    documents, revision = owner
    session = DraftSession(documents)
    await session.select(revision.workflow_id, revision.revision_id)
    session.update(documents.edit_field(revision.raw_json, "/name", "Committed"))
    await session.flush()
    committed, release = Event(), Event()
    save = documents.save_revision

    def blocked(*args):
        result = save(*args)
        committed.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(documents, "save_revision", blocked)
    saving = asyncio.create_task(session.save_revision())
    try:
        assert await asyncio.to_thread(committed.wait, 3)
        closing = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        assert not closing.done()
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert not saving.done()
    finally:
        release.set()
    saved = await saving
    await session.close()
    assert session.base == saved
    assert session.current.base_revision_id == saved.revision_id
    assert documents.list_workflows()[0] == saved


@pytest.mark.parametrize("rejection", [InvalidDraft, RevisionConflict, DraftConflict])
@pytest.mark.parametrize("fail_final_flush", [False, True])
async def test_close_flushes_pending_text_after_rejected_revision_save(
    owner, monkeypatch, rejection, fail_final_flush
):
    documents, revision = owner
    session = DraftSession(documents)
    await session.select(revision.workflow_id, revision.revision_id)
    session.update(documents.edit_field(revision.raw_json, "/name", "Save attempt"))
    await session.flush()
    started, release = Event(), Event()
    write = documents.put_draft

    def rejected(*args):
        started.set()
        assert release.wait(5)
        raise rejection("Revision save rejected")

    def failed_write(*args, **kwargs):
        raise OSError("Final write unavailable")

    monkeypatch.setattr(documents, "save_revision", rejected)
    saving = asyncio.create_task(session.save_revision())
    try:
        assert await asyncio.to_thread(started.wait, 3)
        pending = session.update('{"newer pending text":')
        if fail_final_flush:
            monkeypatch.setattr(documents, "put_draft", failed_write)
        closing = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        assert not closing.done()
    finally:
        release.set()
    with pytest.raises(rejection):
        await saving
    if fail_final_flush:
        with pytest.raises(DraftWriteFailed, match="pending changes remain") as failure:
            await closing
        assert str(failure.value.__cause__) == "Final write unavailable"
        assert session.current == pending
        monkeypatch.setattr(documents, "put_draft", write)
        await session.close()
    else:
        await closing
    assert documents.get_draft(revision.workflow_id, revision.revision_id) == pending
