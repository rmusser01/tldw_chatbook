"""Behavioral contract for process-only temporary Canvas histories."""

from __future__ import annotations

from dataclasses import fields

import pytest

from tldw_chatbook.Canvas.compiler import compile_canvas_document
from tldw_chatbook.Canvas.limits import CanvasRepositoryLimits
from tldw_chatbook.Canvas.models import CanvasRenderPlan
from tldw_chatbook.Canvas.staging import CanvasStagingError, CanvasStagingStore


def _html(text: str) -> str:
    return f"<!doctype html><html><body><main>{text}</main></body></html>"


def _create(
    store: CanvasStagingStore,
    *,
    session_id: str = "session-a",
    run_id: str = "run-a",
    tool_call_id: str = "call-create",
    source: str | None = None,
):
    try:
        owner = store.session_owner(session_id)
    except CanvasStagingError:
        owner = store.activate_session(session_id)
    return store.create_canvas(
        owner=owner,
        run_id=run_id,
        tool_call_id=tool_call_id,
        title="Trip planner",
        source=source or _html("one"),
        origin_message_id="assistant-native-1",
    )


def test_temporary_create_update_and_historical_rename_form_immutable_chain() -> None:
    """Overwriting a parent or carrying source in mutation metadata breaks this."""

    store = CanvasStagingStore()
    created = _create(store)
    updated = store.update_canvas(
        owner=store.session_owner("session-a"),
        run_id="run-b",
        tool_call_id="call-update",
        canvas_id=created.revision.canvas_id,
        expected_parent_revision_id=created.revision.revision_id,
        source=_html("two"),
        origin_message_id="assistant-native-2",
    )
    renamed = store.rename_canvas(
        owner=store.session_owner("session-a"),
        run_id="run-c",
        tool_call_id="call-rename",
        canvas_id=created.revision.canvas_id,
        expected_parent_revision_id=created.revision.revision_id,
        title="Historical fork",
        origin_message_id="user-native-3",
    )

    assert created.revision.sequence == 1
    assert updated.revision.parent_revision_id == created.revision.revision_id
    assert updated.revision.sequence == 2
    assert renamed.revision.parent_revision_id == created.revision.revision_id
    assert renamed.revision.sequence == 3
    assert renamed.revision.title == "Historical fork"
    assert not ({"source", "render_plan"} & {item.name for item in fields(renamed)})

    original = store.read_revision(
        session_id="session-a", revision_id=created.revision.revision_id
    )
    update = store.read_revision(
        session_id="session-a", revision_id=updated.revision.revision_id
    )
    rename = store.read_revision(
        session_id="session-a", revision_id=renamed.revision.revision_id
    )
    assert original.source == _html("one")
    assert update.source == _html("two")
    assert rename.source == _html("one")
    assert isinstance(rename.render_plan, CanvasRenderPlan)
    assert rename.render_plan is original.render_plan


def test_exact_idempotency_key_returns_same_result_without_recompiling() -> None:
    """Compiling or allocating again on replay makes provider retries non-idempotent."""

    calls: list[str] = []

    def counting_compiler(source: str) -> CanvasRenderPlan:
        calls.append(source)
        return compile_canvas_document(source)

    store = CanvasStagingStore(compiler=counting_compiler)
    first = _create(store)
    replay = _create(store)

    assert replay == first
    assert calls == [_html("one")]
    assert store.staged_revision_count("session-a") == 1


def test_update_replay_returns_same_revision_without_recompiling() -> None:
    """An update retry allocating a sibling would make tool retries nondeterministic."""

    calls: list[str] = []

    def counting_compiler(source: str) -> CanvasRenderPlan:
        calls.append(source)
        return compile_canvas_document(source)

    store = CanvasStagingStore(compiler=counting_compiler)
    created = _create(store)
    arguments = {
        "owner": store.session_owner("session-a"),
        "run_id": "run-update",
        "tool_call_id": "call-update",
        "canvas_id": created.revision.canvas_id,
        "expected_parent_revision_id": created.revision.revision_id,
        "source": _html("two"),
        "origin_message_id": "assistant-native-2",
    }

    first = store.update_canvas(**arguments)
    replay = store.update_canvas(**arguments)

    assert replay == first
    assert calls == [_html("one"), _html("two")]
    assert store.staged_revision_count("session-a") == 2


def test_reused_idempotency_key_with_different_request_fails_source_free() -> None:
    """Accepting a changed replay would alias two tool requests to one identity."""

    store = CanvasStagingStore()
    _create(store)

    with pytest.raises(CanvasStagingError) as captured:
        _create(store, source=_html("secret changed payload"))

    assert captured.value.code == "idempotency_conflict"
    assert "secret changed payload" not in str(captured.value)
    assert "secret changed payload" not in repr(captured.value)
    assert store.staged_revision_count("session-a") == 1


def test_aggregate_source_cap_is_per_session_and_failure_does_not_stage() -> None:
    """Counting only the newest head would let retained snapshots exceed memory."""

    first = _html("one")
    second = _html("two")
    store = CanvasStagingStore(
        max_staged_source_bytes=len(first.encode()) + len(second.encode()) - 1
    )
    created = _create(store, source=first)

    with pytest.raises(CanvasStagingError) as captured:
        store.update_canvas(
            owner=store.session_owner("session-a"),
            run_id="run-b",
            tool_call_id="call-update",
            canvas_id=created.revision.canvas_id,
            expected_parent_revision_id=created.revision.revision_id,
            source=second,
            origin_message_id="assistant-native-2",
        )

    assert captured.value.code == "session_source_bytes"
    assert store.staged_revision_count("session-a") == 1
    other = _create(
        store,
        session_id="session-b",
        run_id="run-b",
        tool_call_id="call-b",
        source=second,
    )
    assert other.revision.sequence == 1


def test_default_aggregate_source_cap_accepts_eight_mib_and_rejects_next_snapshot() -> (
    None
):
    """Changing the production default away from 8 MiB breaks this boundary."""

    prefix = "<!doctype html><html><body>"
    suffix = "</body></html>"
    source = prefix + ("x" * (512 * 1024 - len(prefix) - len(suffix))) + suffix
    assert len(source.encode()) == 512 * 1024
    store = CanvasStagingStore()
    created = _create(store, source=source)
    parent_id = created.revision.revision_id
    for index in range(15):
        renamed = store.rename_canvas(
            owner=store.session_owner("session-a"),
            run_id=f"rename-{index}",
            tool_call_id=f"call-{index}",
            canvas_id=created.revision.canvas_id,
            expected_parent_revision_id=parent_id,
            title=f"Title {index}",
            origin_message_id="user-native",
        )
        parent_id = renamed.revision.revision_id

    with pytest.raises(CanvasStagingError, match="session_source_bytes"):
        store.rename_canvas(
            owner=store.session_owner("session-a"),
            run_id="rename-overflow",
            tool_call_id="call-overflow",
            canvas_id=created.revision.canvas_id,
            expected_parent_revision_id=parent_id,
            title="Overflow",
            origin_message_id="user-native",
        )

    assert store.staged_revision_count("session-a") == 16


def test_staging_respects_central_canvas_and_revision_count_limits() -> None:
    """Ignoring repository ceilings would create a graph promotion must reject."""

    limits = CanvasRepositoryLimits(
        max_canvases_per_conversation=1,
        max_revisions_per_canvas=2,
    )
    store = CanvasStagingStore(repository_limits=limits)
    created = _create(store)
    updated = store.update_canvas(
        owner=store.session_owner("session-a"),
        run_id="run-update",
        tool_call_id="call-update",
        canvas_id=created.revision.canvas_id,
        expected_parent_revision_id=created.revision.revision_id,
        source=_html("two"),
        origin_message_id="assistant-native-2",
    )

    with pytest.raises(CanvasStagingError, match="revision_count"):
        store.rename_canvas(
            owner=store.session_owner("session-a"),
            run_id="run-rename",
            tool_call_id="call-rename",
            canvas_id=created.revision.canvas_id,
            expected_parent_revision_id=updated.revision.revision_id,
            title="Too many",
            origin_message_id="user-native",
        )
    with pytest.raises(CanvasStagingError, match="canvas_count"):
        _create(
            store,
            run_id="run-second",
            tool_call_id="call-second",
        )


def test_staging_rejects_source_over_the_central_per_revision_limit() -> None:
    """Delegating only to a looser compiler would violate durable graph limits."""

    store = CanvasStagingStore(
        repository_limits=CanvasRepositoryLimits(max_source_bytes_per_revision=64)
    )

    with pytest.raises(CanvasStagingError, match="revision_source_bytes"):
        _create(store, source=_html("x" * 64))

    assert store.staged_revision_count("session-a") == 0


def test_session_and_process_destruction_remove_source_and_plans() -> None:
    """Leaving staged state reachable after close/teardown violates temporary scope."""

    store = CanvasStagingStore()
    first = _create(store, session_id="session-a")
    _create(store, session_id="session-b", tool_call_id="call-b")

    store.discard_session("session-a")

    with pytest.raises(CanvasStagingError, match="revision_not_found"):
        store.read_revision(
            session_id="session-a", revision_id=first.revision.revision_id
        )
    assert store.staged_revision_count("session-b") == 1

    store.discard_all()

    assert store.staged_revision_count("session-b") == 0


def test_compiler_failure_is_source_free_and_does_not_stage() -> None:
    """Forwarding an untrusted compiler exception can disclose the complete HTML."""

    source = _html("private compiler payload")

    def failing_compiler(value: str) -> CanvasRenderPlan:
        raise RuntimeError(value)

    store = CanvasStagingStore(compiler=failing_compiler)

    with pytest.raises(CanvasStagingError) as captured:
        _create(store, source=source)

    assert captured.value.code == "compile_failed"
    assert "private compiler payload" not in str(captured.value)
    assert "private compiler payload" not in repr(captured.value)
    assert store.staged_revision_count("session-a") == 0


def test_exact_promotion_confirm_clears_leased_staged_state() -> None:
    """A successful transaction must clear its exact frozen contribution."""

    store = CanvasStagingStore()
    _create(store)
    contribution = store.promotion_contribution("session-a")
    assert contribution is not None
    assert store.confirm_contribution("session-a", contribution) is True
    assert store.staged_revision_count("session-a") == 0
    assert "one" not in repr(contribution)


def test_promotion_lease_blocks_exact_session_mutation_until_abort() -> None:
    """A snapshot without a mutation fence can strand a post-snapshot delta."""

    store = CanvasStagingStore()
    owner = store.activate_session("session-a")
    created = store.create_canvas(
        owner=owner,
        run_id="run-a",
        tool_call_id="call-create",
        title="Trip planner",
        source=_html("one"),
        origin_message_id="assistant-native-1",
    )
    contribution = store.promotion_contribution("session-a")
    assert contribution is not None

    with pytest.raises(CanvasStagingError, match="promotion_in_flight"):
        store.update_canvas(
            owner=owner,
            run_id="run-b",
            tool_call_id="call-update",
            canvas_id=created.revision.canvas_id,
            expected_parent_revision_id=created.revision.revision_id,
            source=_html("two"),
            origin_message_id="assistant-native-2",
        )

    assert store.abort_contribution("session-a", contribution) is True
    updated = store.update_canvas(
        owner=owner,
        run_id="run-b",
        tool_call_id="call-update",
        canvas_id=created.revision.canvas_id,
        expected_parent_revision_id=created.revision.revision_id,
        source=_html("two"),
        origin_message_id="assistant-native-2",
    )
    assert updated.revision.sequence == 2


def test_exact_retire_clears_only_the_contribution_owner_and_lease() -> None:
    """A postcommit fallback keyed only by session id can erase a replacement."""

    store = CanvasStagingStore()
    _create(store)
    old_contribution = store.promotion_contribution("session-a")
    assert old_contribution is not None
    store.discard_session("session-a")

    replacement_owner = store.activate_session("session-a")
    replacement = store.create_canvas(
        owner=replacement_owner,
        run_id="run-replacement",
        tool_call_id="call-replacement",
        title="Replacement",
        source=_html("replacement"),
        origin_message_id="assistant-replacement",
    )
    replacement_contribution = store.promotion_contribution("session-a")
    assert replacement_contribution is not None

    assert store.retire_contribution("session-a", old_contribution) is False
    assert store.staged_revision_count("session-a") == 1
    assert store.read_revision(
        session_id="session-a",
        revision_id=replacement.revision.revision_id,
    ).source == _html("replacement")
    assert store.retire_contribution("session-a", replacement_contribution) is True
    assert store.staged_revision_count("session-a") == 0


def test_retired_owner_cannot_resurrect_or_mutate_reused_session_id() -> None:
    """A bare session id cannot distinguish a late callback from a new owner."""

    store = CanvasStagingStore()
    first_owner = store.activate_session("session-a")
    first = store.create_canvas(
        owner=first_owner,
        run_id="run-first",
        tool_call_id="call-first",
        title="First",
        source=_html("first"),
        origin_message_id="assistant-first",
    )
    store.discard_session("session-a")
    second_owner = store.activate_session("session-a")

    with pytest.raises(CanvasStagingError, match="session_retired"):
        store.update_canvas(
            owner=first_owner,
            run_id="run-late",
            tool_call_id="call-late",
            canvas_id=first.revision.canvas_id,
            expected_parent_revision_id=first.revision.revision_id,
            source=_html("late"),
            origin_message_id="assistant-late",
        )

    second = store.create_canvas(
        owner=second_owner,
        run_id="run-second",
        tool_call_id="call-second",
        title="Second",
        source=_html("second"),
        origin_message_id="assistant-second",
    )
    assert second.revision.sequence == 1
    assert store.staged_revision_count("session-a") == 1


def test_runtime_close_permanently_rejects_activation_and_late_mutation() -> None:
    """Runtime teardown must be a terminal fence, not only dictionary cleanup."""

    store = CanvasStagingStore()
    owner = store.activate_session("session-a")
    store.close_runtime()

    with pytest.raises(CanvasStagingError, match="runtime_closed"):
        store.create_canvas(
            owner=owner,
            run_id="run-late",
            tool_call_id="call-late",
            title="Late",
            source=_html("late"),
            origin_message_id="assistant-late",
        )
    with pytest.raises(CanvasStagingError, match="runtime_closed"):
        store.create_canvas(
            owner=owner,
            run_id="run-invalid-late",
            tool_call_id="call-invalid-late",
            title="Late",
            source="\ud800",
            origin_message_id="assistant-late",
        )
    with pytest.raises(CanvasStagingError, match="runtime_closed"):
        store.activate_session("session-b")


def test_staging_uses_lower_durable_conversation_source_ceiling() -> None:
    """Promotion bypasses public repository validation, so staging owns this cap."""

    first = _html("one")
    second = _html("two")
    durable_limit = len(first.encode()) + len(second.encode()) - 1
    store = CanvasStagingStore(
        repository_limits=CanvasRepositoryLimits(
            max_source_bytes_per_conversation=durable_limit
        ),
        max_staged_source_bytes=durable_limit + 1024,
    )
    owner = store.activate_session("session-a")
    created = store.create_canvas(
        owner=owner,
        run_id="run-a",
        tool_call_id="call-create",
        title="Trip planner",
        source=first,
        origin_message_id="assistant-native-1",
    )

    with pytest.raises(CanvasStagingError, match="session_source_bytes"):
        store.update_canvas(
            owner=owner,
            run_id="run-b",
            tool_call_id="call-update",
            canvas_id=created.revision.canvas_id,
            expected_parent_revision_id=created.revision.revision_id,
            source=second,
            origin_message_id="assistant-native-2",
        )
