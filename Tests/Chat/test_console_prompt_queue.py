from __future__ import annotations

import ast
import inspect
import threading
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from rich.cells import cell_len
from rich.text import Text

from tldw_chatbook.Chat.console_prompt_queue import (
    MAX_CONSOLE_QUEUED_PROMPT_LENGTH,
    PROMPT_PREVIEW_CELL_BUDGET,
    ConsolePromptQueueRegistry,
    PromptQueueEntryPhase,
    PromptQueueMode,
    PromptQueuePauseReason,
    PromptQueueReservation,
    QueueMutationStatus,
    QueueThreadViolation,
    QueuedPrompt,
    make_prompt_preview,
)


class _DeterministicValues:
    def __init__(self) -> None:
        self._id = 0
        self._time = 100.0

    def next_id(self) -> str:
        self._id += 1
        return f"queue-entry-{self._id}"

    def monotonic(self) -> float:
        self._time += 1.0
        return self._time


@pytest.fixture
def registry() -> ConsolePromptQueueRegistry:
    values = _DeterministicValues()
    return ConsolePromptQueueRegistry(
        id_factory=values.next_id,
        monotonic=values.monotonic,
    )


def _begin(
    registry: ConsolePromptQueueRegistry,
    session_id: str = "session-a",
    *,
    context_epoch: int = 7,
) -> int:
    result = registry.begin_chain(
        session_id,
        context_epoch=context_epoch,
        expected_revision=registry.snapshot(session_id).revision,
    )
    assert result.status is QueueMutationStatus.APPLIED
    return result.snapshot.revision


def _admit(
    registry: ConsolePromptQueueRegistry,
    text: str,
    session_id: str = "session-a",
) -> str:
    result = registry.admit(
        session_id,
        text=text,
        expected_revision=registry.snapshot(session_id).revision,
    )
    assert result.status is QueueMutationStatus.APPLIED
    assert result.entry_id is not None
    return result.entry_id


def test_entries_are_immutable_redacted_and_fifo_with_stable_ids(
    registry: ConsolePromptQueueRegistry,
) -> None:
    _begin(registry)
    first_id = _admit(registry, "first private prompt")
    second_id = _admit(registry, "second private prompt")

    first_claim = registry.claim_next(
        "session-a",
        expected_revision=registry.snapshot("session-a").revision,
    )
    assert first_claim.status is QueueMutationStatus.APPLIED
    assert first_claim.claim is not None
    assert first_claim.claim.prompt.entry_id == first_id
    assert first_claim.claim.prompt.text == "first private prompt"
    assert "first private prompt" not in repr(first_claim.claim)
    assert "first private prompt" not in repr(first_claim)
    with pytest.raises(FrozenInstanceError):
        first_claim.claim.prompt.text = "changed"  # type: ignore[misc]

    settled = registry.settle_claim(
        "session-a",
        entry_id=first_id,
        expected_revision=first_claim.snapshot.revision,
    )
    second_claim = registry.claim_next(
        "session-a",
        expected_revision=settled.snapshot.revision,
    )
    assert second_claim.claim is not None
    assert second_claim.claim.prompt.entry_id == second_id


def test_sessions_are_isolated(registry: ConsolePromptQueueRegistry) -> None:
    _begin(registry, "session-a", context_epoch=1)
    _begin(registry, "session-b", context_epoch=20)
    a_id = _admit(registry, "alpha", "session-a")
    b_id = _admit(registry, "beta", "session-b")

    a = registry.snapshot("session-a")
    b = registry.snapshot("session-b")
    assert [entry.entry_id for entry in a.entries] == [a_id]
    assert [entry.entry_id for entry in b.entries] == [b_id]
    assert a.expected_context_epoch == 1
    assert b.expected_context_epoch == 20


def test_capacity_counts_waiting_plus_claimed(
    registry: ConsolePromptQueueRegistry,
) -> None:
    _begin(registry)
    ids = [_admit(registry, f"prompt {index}") for index in range(10)]
    claim = registry.claim_next(
        "session-a",
        expected_revision=registry.snapshot("session-a").revision,
    )
    assert claim.claim is not None
    assert claim.claim.prompt.entry_id == ids[0]
    assert claim.snapshot.waiting_count == 9
    assert claim.snapshot.claimed_count == 1
    assert claim.snapshot.total_count == 10

    full = registry.admit(
        "session-a",
        text="must not fit while one entry is starting",
        expected_revision=claim.snapshot.revision,
    )
    assert full.status is QueueMutationStatus.FULL
    assert full.snapshot.revision == claim.snapshot.revision
    assert full.snapshot.total_count == 10

    settled = registry.settle_claim(
        "session-a",
        entry_id=ids[0],
        expected_revision=claim.snapshot.revision,
    )
    admitted = registry.admit(
        "session-a",
        text="now there is room",
        expected_revision=settled.snapshot.revision,
    )
    assert admitted.status is QueueMutationStatus.APPLIED
    assert admitted.snapshot.total_count == 10


def test_stale_revision_rejects_without_calling_id_or_clock_producers() -> None:
    calls = {"id": 0, "clock": 0}

    def next_id() -> str:
        calls["id"] += 1
        return f"id-{calls['id']}"

    def monotonic() -> float:
        calls["clock"] += 1
        return float(calls["clock"])

    registry = ConsolePromptQueueRegistry(id_factory=next_id, monotonic=monotonic)
    revision = _begin(registry)
    _admit(registry, "older")
    before = registry.snapshot("session-a")
    calls_before = dict(calls)

    stale = registry.admit(
        "session-a",
        text="private stale draft",
        expected_revision=revision,
    )

    assert stale.status is QueueMutationStatus.STALE_REVISION
    assert stale.snapshot is before
    assert calls == calls_before
    assert "private stale draft" not in repr(stale)


def test_edit_recomputes_only_one_preview_and_preserves_id_and_order(
    registry: ConsolePromptQueueRegistry,
) -> None:
    _begin(registry)
    first_id = _admit(registry, "first")
    second_id = _admit(registry, "second")
    before = registry.snapshot("session-a")
    second_before = before.entries[1]

    edited = registry.edit(
        "session-a",
        entry_id=first_id,
        text="edited [bold]private[/bold] text",
        expected_revision=before.revision,
    )

    assert edited.status is QueueMutationStatus.APPLIED
    assert [entry.entry_id for entry in edited.snapshot.entries] == [
        first_id,
        second_id,
    ]
    assert edited.snapshot.entries[1] is second_before
    assert Text.from_markup(edited.snapshot.entries[0].preview).plain == (
        "edited [bold]private[/bold] text"
    )


def test_full_text_read_materializes_only_selected_waiting_entry_and_redacts_repr(
    registry: ConsolePromptQueueRegistry,
) -> None:
    _begin(registry)
    first_id = _admit(registry, "first private body")
    second_id = _admit(registry, "second private body")
    current = registry.snapshot("session-a")

    selected = registry.read_waiting_text(
        "session-a",
        entry_id=second_id,
        expected_revision=current.revision,
    )
    assert selected.status is QueueMutationStatus.APPLIED
    assert selected.entry_id == second_id
    assert selected.text == "second private body"
    assert "second private body" not in repr(selected)
    stale = registry.read_waiting_text(
        "session-a",
        entry_id=first_id,
        expected_revision=current.revision - 1,
    )
    assert stale.status is QueueMutationStatus.STALE_REVISION
    assert stale.text is None

    claim = registry.claim_next(
        "session-a",
        expected_revision=current.revision,
    )
    locked = registry.read_waiting_text(
        "session-a",
        entry_id=first_id,
        expected_revision=claim.snapshot.revision,
    )
    assert locked.status is QueueMutationStatus.LOCKED
    assert locked.text is None


def test_move_remove_and_clear_waiting_never_mutate_claimed_work(
    registry: ConsolePromptQueueRegistry,
) -> None:
    _begin(registry)
    first_id = _admit(registry, "first")
    second_id = _admit(registry, "second")
    third_id = _admit(registry, "third")
    claim = registry.claim_next(
        "session-a",
        expected_revision=registry.snapshot("session-a").revision,
    )

    for operation in (
        lambda revision: registry.edit(
            "session-a",
            entry_id=first_id,
            text="cannot edit a starting prompt",
            expected_revision=revision,
        ),
        lambda revision: registry.move(
            "session-a",
            entry_id=first_id,
            new_index=0,
            expected_revision=revision,
        ),
        lambda revision: registry.remove(
            "session-a",
            entry_id=first_id,
            expected_revision=revision,
        ),
    ):
        result = operation(claim.snapshot.revision)
        assert result.status is QueueMutationStatus.LOCKED
        assert result.snapshot.revision == claim.snapshot.revision

    moved = registry.move(
        "session-a",
        entry_id=third_id,
        new_index=0,
        expected_revision=claim.snapshot.revision,
    )
    assert [entry.entry_id for entry in moved.snapshot.entries] == [
        first_id,
        third_id,
        second_id,
    ]
    removed = registry.remove(
        "session-a",
        entry_id=second_id,
        expected_revision=moved.snapshot.revision,
    )
    cleared = registry.clear_waiting(
        "session-a",
        expected_revision=removed.snapshot.revision,
    )
    assert cleared.snapshot.waiting_count == 0
    assert cleared.snapshot.claimed_count == 1
    assert cleared.snapshot.entries[0].entry_id == first_id
    assert cleared.snapshot.entries[0].phase is PromptQueueEntryPhase.STARTING


@pytest.mark.parametrize(
    "reason",
    [
        PromptQueuePauseReason.MANUAL,
        PromptQueuePauseReason.FAILED,
        PromptQueuePauseReason.STOPPED,
        PromptQueuePauseReason.CONTEXT_CHANGED,
        PromptQueuePauseReason.DISPATCH_REFUSED,
    ],
)
def test_each_pause_reason_releases_reservation_and_accepts_new_work_at_tail(
    registry: ConsolePromptQueueRegistry,
    reason: PromptQueuePauseReason,
) -> None:
    _begin(registry)
    first_id = _admit(registry, "first")
    paused = registry.pause(
        "session-a",
        reason=reason,
        expected_revision=registry.snapshot("session-a").revision,
    )
    assert paused.status is QueueMutationStatus.APPLIED
    assert paused.snapshot.mode is PromptQueueMode.PAUSED
    assert paused.snapshot.pause_reason is reason
    assert paused.snapshot.reservation is PromptQueueReservation.RELEASED

    second_id = _admit(registry, "second while paused")
    assert [entry.entry_id for entry in registry.snapshot("session-a").entries] == [
        first_id,
        second_id,
    ]


def test_pause_after_turn_keep_draining_and_resume_require_explicit_reservation(
    registry: ConsolePromptQueueRegistry,
) -> None:
    _begin(registry)
    _admit(registry, "later")
    requested = registry.request_pause_after_turn(
        "session-a",
        expected_revision=registry.snapshot("session-a").revision,
    )
    assert requested.snapshot.mode is PromptQueueMode.PAUSE_AFTER_TURN
    assert requested.snapshot.reservation is PromptQueueReservation.HELD

    kept = registry.keep_draining(
        "session-a",
        expected_revision=requested.snapshot.revision,
    )
    assert kept.snapshot.mode is PromptQueueMode.DRAINING
    paused = registry.pause(
        "session-a",
        reason=PromptQueuePauseReason.MANUAL,
        expected_revision=kept.snapshot.revision,
    )
    refused = registry.resume(
        "session-a",
        expected_revision=paused.snapshot.revision,
    )
    assert refused.status is QueueMutationStatus.INVALID
    assert refused.snapshot.mode is PromptQueueMode.PAUSED

    reserved = registry.reserve(
        "session-a",
        expected_revision=paused.snapshot.revision,
    )
    resumed = registry.resume(
        "session-a",
        expected_revision=reserved.snapshot.revision,
    )
    assert resumed.status is QueueMutationStatus.APPLIED
    assert resumed.snapshot.mode is PromptQueueMode.DRAINING
    assert resumed.snapshot.pause_reason is None
    assert resumed.snapshot.reservation is PromptQueueReservation.HELD


def test_every_session_transition_checks_revision_before_mutating(
    registry: ConsolePromptQueueRegistry,
) -> None:
    _begin(registry)
    entry_id = _admit(registry, "body that must remain untouched")
    current = registry.snapshot("session-a")
    stale_revision = current.revision - 1
    operations = [
        lambda: registry.begin_chain(
            "session-a", context_epoch=8, expected_revision=stale_revision
        ),
        lambda: registry.admit(
            "session-a", text="new", expected_revision=stale_revision
        ),
        lambda: registry.edit(
            "session-a",
            entry_id=entry_id,
            text="edited",
            expected_revision=stale_revision,
        ),
        lambda: registry.move(
            "session-a",
            entry_id=entry_id,
            new_index=0,
            expected_revision=stale_revision,
        ),
        lambda: registry.remove(
            "session-a", entry_id=entry_id, expected_revision=stale_revision
        ),
        lambda: registry.clear_waiting("session-a", expected_revision=stale_revision),
        lambda: registry.claim_next("session-a", expected_revision=stale_revision),
        lambda: registry.settle_claim(
            "session-a", entry_id=entry_id, expected_revision=stale_revision
        ),
        lambda: registry.return_claim_to_head(
            "session-a",
            entry_id=entry_id,
            reason=PromptQueuePauseReason.DISPATCH_REFUSED,
            expected_revision=stale_revision,
        ),
        lambda: registry.request_pause_after_turn(
            "session-a", expected_revision=stale_revision
        ),
        lambda: registry.keep_draining("session-a", expected_revision=stale_revision),
        lambda: registry.pause(
            "session-a",
            reason=PromptQueuePauseReason.MANUAL,
            expected_revision=stale_revision,
        ),
        lambda: registry.reserve("session-a", expected_revision=stale_revision),
        lambda: registry.release_reservation(
            "session-a", expected_revision=stale_revision
        ),
        lambda: registry.resume(
            "session-a",
            expected_revision=stale_revision,
        ),
        lambda: registry.adopt_context_baseline(
            "session-a", context_epoch=9, expected_revision=stale_revision
        ),
        lambda: registry.finalize_empty_chain(
            "session-a", expected_revision=stale_revision
        ),
        lambda: registry.mark_closing("session-a", expected_revision=stale_revision),
        lambda: registry.remove_session("session-a", expected_revision=stale_revision),
    ]

    for operation in operations:
        result = operation()
        assert result.status is QueueMutationStatus.STALE_REVISION
        assert result.snapshot is current


def test_return_claim_to_head_preserves_fifo_and_pauses(
    registry: ConsolePromptQueueRegistry,
) -> None:
    _begin(registry)
    first_id = _admit(registry, "first")
    second_id = _admit(registry, "second")
    claim = registry.claim_next(
        "session-a",
        expected_revision=registry.snapshot("session-a").revision,
    )
    returned = registry.return_claim_to_head(
        "session-a",
        entry_id=first_id,
        reason=PromptQueuePauseReason.DISPATCH_REFUSED,
        expected_revision=claim.snapshot.revision,
    )
    assert returned.status is QueueMutationStatus.APPLIED
    assert [entry.entry_id for entry in returned.snapshot.entries] == [
        first_id,
        second_id,
    ]
    assert returned.snapshot.claimed_count == 0
    assert returned.snapshot.mode is PromptQueueMode.PAUSED
    assert returned.snapshot.reservation is PromptQueueReservation.RELEASED


def test_claim_guards_mode_reservation_and_single_claim(
    registry: ConsolePromptQueueRegistry,
) -> None:
    _begin(registry)
    _admit(registry, "first")
    _admit(registry, "second")
    first = registry.claim_next(
        "session-a",
        expected_revision=registry.snapshot("session-a").revision,
    )
    duplicate = registry.claim_next(
        "session-a",
        expected_revision=first.snapshot.revision,
    )
    assert duplicate.status is QueueMutationStatus.LOCKED
    assert duplicate.snapshot.waiting_count == 1

    returned = registry.return_claim_to_head(
        "session-a",
        entry_id=first.claim.prompt.entry_id,  # type: ignore[union-attr]
        reason=PromptQueuePauseReason.MANUAL,
        expected_revision=first.snapshot.revision,
    )
    paused_claim = registry.claim_next(
        "session-a",
        expected_revision=returned.snapshot.revision,
    )
    assert paused_claim.status is QueueMutationStatus.INVALID


def test_starting_claim_cannot_be_paused_or_lost_on_invalid_clock() -> None:
    values = _DeterministicValues()
    clock_values: list[object] = [101.0, "invalid claim clock"]
    registry = ConsolePromptQueueRegistry(
        id_factory=values.next_id,
        monotonic=lambda: clock_values.pop(0),
    )
    _begin(registry)
    entry_id = _admit(registry, "first")
    before = registry.snapshot("session-a")
    invalid_claim = registry.claim_next(
        "session-a",
        expected_revision=before.revision,
    )
    assert invalid_claim.status is QueueMutationStatus.INVALID
    assert invalid_claim.snapshot is before
    assert invalid_claim.snapshot.entries[0].entry_id == entry_id

    healthy_registry = ConsolePromptQueueRegistry()
    _begin(healthy_registry)
    claimed_id = _admit(healthy_registry, "starting")
    claimed = healthy_registry.claim_next(
        "session-a",
        expected_revision=healthy_registry.snapshot("session-a").revision,
    )
    paused = healthy_registry.pause(
        "session-a",
        reason=PromptQueuePauseReason.MANUAL,
        expected_revision=claimed.snapshot.revision,
    )
    assert paused.status is QueueMutationStatus.LOCKED
    assert paused.snapshot.entries[0].entry_id == claimed_id
    assert paused.snapshot.entries[0].phase is PromptQueueEntryPhase.STARTING
    assert paused.snapshot.reservation is PromptQueueReservation.HELD


def test_context_baseline_can_only_be_adopted_from_context_review_pause(
    registry: ConsolePromptQueueRegistry,
) -> None:
    _begin(registry, context_epoch=11)
    _admit(registry, "future")
    before = registry.snapshot("session-a")
    refused = registry.adopt_context_baseline(
        "session-a",
        context_epoch=12,
        expected_revision=before.revision,
    )
    assert refused.status is QueueMutationStatus.INVALID
    assert refused.snapshot.expected_context_epoch == 11

    paused = registry.pause(
        "session-a",
        reason=PromptQueuePauseReason.CONTEXT_CHANGED,
        expected_revision=before.revision,
    )
    adopted = registry.adopt_context_baseline(
        "session-a",
        context_epoch=12,
        expected_revision=paused.snapshot.revision,
    )
    assert adopted.status is QueueMutationStatus.APPLIED
    assert adopted.snapshot.expected_context_epoch == 12
    assert adopted.snapshot.mode is PromptQueueMode.PAUSED


def test_admission_and_final_empty_release_have_two_revision_orderings(
    registry: ConsolePromptQueueRegistry,
) -> None:
    initial_revision = _begin(registry)
    admitted = registry.admit(
        "session-a",
        text="admission wins",
        expected_revision=initial_revision,
    )
    stale_release = registry.finalize_empty_chain(
        "session-a",
        expected_revision=initial_revision,
    )
    assert admitted.status is QueueMutationStatus.APPLIED
    assert stale_release.status is QueueMutationStatus.STALE_REVISION
    assert stale_release.snapshot.total_count == 1

    other = ConsolePromptQueueRegistry()
    release_revision = _begin(other, context_epoch=3)
    released = other.finalize_empty_chain(
        "session-a",
        expected_revision=release_revision,
    )
    rerouted = other.admit(
        "session-a",
        text="release wins and this remains a manual draft",
        expected_revision=release_revision,
    )
    assert released.status is QueueMutationStatus.APPLIED
    assert rerouted.status is QueueMutationStatus.REROUTE_NORMAL_SEND
    assert rerouted.snapshot.total_count == 0
    older_stale = other.admit(
        "session-a",
        text="an older stale intent must not be rerouted",
        expected_revision=0,
    )
    assert older_stale.status is QueueMutationStatus.STALE_REVISION


def test_closing_tombstone_rejects_work_until_exact_session_removal(
    registry: ConsolePromptQueueRegistry,
) -> None:
    _begin(registry, "session-a")
    _begin(registry, "session-b")
    _admit(registry, "a", "session-a")
    _admit(registry, "b", "session-b")
    closing = registry.mark_closing(
        "session-a",
        expected_revision=registry.snapshot("session-a").revision,
    )
    assert closing.snapshot.closing is True
    assert closing.snapshot.reservation is PromptQueueReservation.RELEASED

    refused = registry.admit(
        "session-a",
        text="must not enter closing session",
        expected_revision=closing.snapshot.revision,
    )
    assert refused.status is QueueMutationStatus.CLOSING
    removed = registry.remove_session(
        "session-a",
        expected_revision=closing.snapshot.revision,
    )
    assert removed.status is QueueMutationStatus.APPLIED
    assert registry.snapshot("session-a").total_count == 0
    assert registry.snapshot("session-b").total_count == 1


def test_shutdown_is_revision_checked_and_suppresses_all_later_claims(
    registry: ConsolePromptQueueRegistry,
) -> None:
    _begin(registry)
    _admit(registry, "private queued prompt")
    before = registry.snapshot("session-a")
    stale = registry.shutdown(expected_registry_revision=registry.registry_revision - 1)
    assert stale.status is QueueMutationStatus.STALE_REVISION
    assert registry.snapshot("session-a") is before

    stopped = registry.shutdown(expected_registry_revision=registry.registry_revision)
    assert stopped.status is QueueMutationStatus.APPLIED
    assert stopped.removed_sessions == 1
    assert stopped.removed_prompts == 1
    claim = registry.claim_next("session-a", expected_revision=0)
    assert claim.status is QueueMutationStatus.SHUTTING_DOWN
    assert registry.snapshot("session-a").total_count == 0
    assert "private queued prompt" not in repr(stopped)


def test_foreign_thread_access_is_rejected(
    registry: ConsolePromptQueueRegistry,
) -> None:
    errors: list[BaseException] = []

    def mutate() -> None:
        try:
            registry.snapshot("session-a")
        except BaseException as exc:  # noqa: BLE001 - asserting boundary type
            errors.append(exc)

    thread = threading.Thread(target=mutate)
    thread.start()
    thread.join()
    assert len(errors) == 1
    assert isinstance(errors[0], QueueThreadViolation)


@pytest.mark.parametrize(
    ("raw", "plain"),
    [
        ("first\r\nsecond\tthird", "first second third"),
        ("\x1b[31mred\x1b[0m text", "red text"),
        ("\x1b]8;;https://evil.example\x07label\x1b]8;;\x07", "label"),
        ("safe\x00\x08\x7f\x9ftext", "safetext"),
        ("[bold]not markup[/bold]", "[bold]not markup[/bold]"),
        ("wide 界 and e\u0301", "wide 界 and e\u0301"),
    ],
)
def test_preview_is_one_line_terminal_safe_and_markup_escaped(
    raw: str, plain: str
) -> None:
    preview = make_prompt_preview(raw)
    rendered = Text.from_markup(preview).plain
    assert rendered == plain
    assert "\n" not in preview
    assert "\r" not in preview
    assert "\x1b" not in preview
    assert cell_len(rendered) <= PROMPT_PREVIEW_CELL_BUDGET


@pytest.mark.parametrize(
    ("raw", "complete_suffix"),
    [
        ("界" * (PROMPT_PREVIEW_CELL_BUDGET + 5), "界…"),
        ("e\u0301" * (PROMPT_PREVIEW_CELL_BUDGET + 5), "e\u0301…"),
    ],
)
def test_preview_truncates_by_cells_without_splitting_graphemes(
    raw: str,
    complete_suffix: str,
) -> None:
    rendered = Text.from_markup(make_prompt_preview(raw)).plain
    assert rendered.endswith("…")
    assert cell_len(rendered) <= PROMPT_PREVIEW_CELL_BUDGET
    assert rendered.endswith(complete_suffix)


def test_snapshot_is_body_free_cached_and_reuses_unchanged_entry_views(
    registry: ConsolePromptQueueRegistry,
) -> None:
    secret = "DO-NOT-COPY-THIS-BODY-" + ("x" * 10_000)
    _begin(registry)
    _admit(registry, secret)
    first = registry.snapshot("session-a")
    second = registry.snapshot("session-a")

    assert second is first
    assert first.entries[0].preview == make_prompt_preview(secret)
    assert secret not in repr(first)
    assert not hasattr(first.entries[0], "text")
    assert not hasattr(first, "prompt_texts")

    class _ExplodingBody(str):
        def __str__(self) -> str:
            raise AssertionError("snapshot traversed the canonical body")

        def __repr__(self) -> str:
            raise AssertionError("snapshot represented the canonical body")

        def __len__(self) -> int:
            raise AssertionError("snapshot measured the canonical body")

        def __iter__(self):
            raise AssertionError("snapshot iterated the canonical body")

    stored_prompt = registry._states["session-a"].waiting[0]
    object.__setattr__(stored_prompt, "text", _ExplodingBody(secret))
    pause = registry.request_pause_after_turn(
        "session-a",
        expected_revision=first.revision,
    )
    assert pause.snapshot.entries[0] is first.entries[0]


def test_invalid_text_and_duplicate_id_fail_without_partial_mutation() -> None:
    registry = ConsolePromptQueueRegistry(id_factory=lambda: "same-id")
    _begin(registry)
    first = _admit(registry, "first")
    before = registry.snapshot("session-a")
    duplicate = registry.admit(
        "session-a",
        text="second",
        expected_revision=before.revision,
    )
    assert duplicate.status is QueueMutationStatus.INVALID
    assert duplicate.snapshot is before
    assert [entry.entry_id for entry in duplicate.snapshot.entries] == [first]

    too_large = registry.edit(
        "session-a",
        entry_id=first,
        text="x" * (MAX_CONSOLE_QUEUED_PROMPT_LENGTH + 1),
        expected_revision=before.revision,
    )
    assert too_large.status is QueueMutationStatus.INVALID
    assert too_large.snapshot is before

    unsafe_markup = registry.edit(
        "session-a",
        entry_id=first,
        text="<script>alert('queued')</script>",
        expected_revision=before.revision,
    )
    assert unsafe_markup.status is QueueMutationStatus.INVALID
    assert unsafe_markup.snapshot is before


def test_entry_identity_tracking_is_bounded_to_active_prompts() -> None:
    registry = ConsolePromptQueueRegistry(id_factory=lambda: "reusable-id")
    _begin(registry)

    first = _admit(registry, "first")
    removed = registry.remove(
        "session-a",
        entry_id=first,
        expected_revision=registry.snapshot("session-a").revision,
    )
    assert removed.status is QueueMutationStatus.APPLIED
    assert registry._active_entry_ids == set()
    released = registry.release_reservation(
        "session-a",
        expected_revision=removed.snapshot.revision,
    )
    assert released.status is QueueMutationStatus.APPLIED

    _begin(registry)
    second = _admit(registry, "second")
    assert second == first
    cleared = registry.clear_waiting(
        "session-a",
        expected_revision=registry.snapshot("session-a").revision,
    )
    assert cleared.status is QueueMutationStatus.APPLIED
    assert registry._active_entry_ids == set()
    released = registry.release_reservation(
        "session-a",
        expected_revision=cleared.snapshot.revision,
    )
    assert released.status is QueueMutationStatus.APPLIED

    _begin(registry)
    third = _admit(registry, "third")
    claimed = registry.claim_next(
        "session-a",
        expected_revision=registry.snapshot("session-a").revision,
    )
    settled = registry.settle_claim(
        "session-a",
        entry_id=third,
        expected_revision=claimed.snapshot.revision,
    )
    assert settled.status is QueueMutationStatus.APPLIED
    assert registry._active_entry_ids == set()

    fourth = _admit(registry, "fourth")
    assert fourth == first
    removed_session = registry.remove_session(
        "session-a",
        expected_revision=registry.snapshot("session-a").revision,
    )
    assert removed_session.status is QueueMutationStatus.APPLIED
    assert registry._active_entry_ids == set()

    _begin(registry)
    _admit(registry, "fifth")
    stopped = registry.shutdown(expected_registry_revision=registry.registry_revision)
    assert stopped.status is QueueMutationStatus.APPLIED
    assert registry._active_entry_ids == set()


def test_registry_is_sync_and_has_no_forbidden_runtime_dependencies() -> None:
    module_path = (
        Path(__file__).parents[2] / "tldw_chatbook" / "Chat" / "console_prompt_queue.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    forbidden_fragments = {
        "textual",
        "provider",
        "database",
        ".DB",
        "prompt_history",
        "diagnostic",
        "logging",
        "chat_screen",
    }
    assert not {
        imported
        for imported in imports
        if any(fragment in imported.lower() for fragment in forbidden_fragments)
    }

    public_transitions = [
        ConsolePromptQueueRegistry.begin_chain,
        ConsolePromptQueueRegistry.admit,
        ConsolePromptQueueRegistry.edit,
        ConsolePromptQueueRegistry.move,
        ConsolePromptQueueRegistry.remove,
        ConsolePromptQueueRegistry.clear_waiting,
        ConsolePromptQueueRegistry.claim_next,
        ConsolePromptQueueRegistry.settle_claim,
        ConsolePromptQueueRegistry.return_claim_to_head,
        ConsolePromptQueueRegistry.request_pause_after_turn,
        ConsolePromptQueueRegistry.keep_draining,
        ConsolePromptQueueRegistry.pause,
        ConsolePromptQueueRegistry.reserve,
        ConsolePromptQueueRegistry.resume,
        ConsolePromptQueueRegistry.adopt_context_baseline,
        ConsolePromptQueueRegistry.finalize_empty_chain,
        ConsolePromptQueueRegistry.mark_closing,
        ConsolePromptQueueRegistry.remove_session,
        ConsolePromptQueueRegistry.shutdown,
    ]
    assert not any(inspect.iscoroutinefunction(method) for method in public_transitions)


def test_prompt_model_explicit_repr_never_contains_body() -> None:
    prompt = QueuedPrompt(
        entry_id="entry",
        text="PRIVATE BODY",
        preview="PRIVATE PREVIEW",
        insertion_order=1,
        admitted_at=2.0,
    )
    assert "PRIVATE BODY" not in repr(prompt)
    assert "PRIVATE PREVIEW" not in repr(prompt)
    assert "redacted" in repr(prompt).lower()
