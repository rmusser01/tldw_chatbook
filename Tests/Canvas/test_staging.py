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
    return store.create_canvas(
        session_id=session_id,
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
        session_id="session-a",
        run_id="run-b",
        tool_call_id="call-update",
        canvas_id=created.revision.canvas_id,
        expected_parent_revision_id=created.revision.revision_id,
        source=_html("two"),
        origin_message_id="assistant-native-2",
    )
    renamed = store.rename_canvas(
        session_id="session-a",
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
        "session_id": "session-a",
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
            session_id="session-a",
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
            session_id="session-a",
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
            session_id="session-a",
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
        session_id="session-a",
        run_id="run-update",
        tool_call_id="call-update",
        canvas_id=created.revision.canvas_id,
        expected_parent_revision_id=created.revision.revision_id,
        source=_html("two"),
        origin_message_id="assistant-native-2",
    )

    with pytest.raises(CanvasStagingError, match="revision_count"):
        store.rename_canvas(
            session_id="session-a",
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


def test_compare_and_confirm_does_not_discard_newer_staged_state() -> None:
    """Confirming an old snapshot unconditionally can erase a concurrent revision."""

    store = CanvasStagingStore()
    created = _create(store)
    contribution = store.promotion_contribution("session-a")
    assert contribution is not None
    store.update_canvas(
        session_id="session-a",
        run_id="run-b",
        tool_call_id="call-update",
        canvas_id=created.revision.canvas_id,
        expected_parent_revision_id=created.revision.revision_id,
        source=_html("newer"),
        origin_message_id="assistant-native-2",
    )

    assert store.confirm_contribution("session-a", contribution) is False
    assert store.staged_revision_count("session-a") == 2
    assert "one" not in repr(contribution)
