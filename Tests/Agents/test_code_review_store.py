"""Real filesystem and threaded controls for the session review ledger."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from Tests.Agents.code_review_helpers import coverage_payload, finding_payload
from tldw_chatbook.Agents.code_review_models import (
    ReviewLimits,
    ReviewRefused,
    ReviewRunBinding,
)
from tldw_chatbook.Agents.code_review_store import CapturedSource, ReviewStore
from tldw_chatbook.Agents.run_context import use_run_id
from tldw_chatbook.Chat.console_scratch_space import ConsoleScratchSpaceManager

_ALIASES = weakref.WeakKeyDictionary()


def create_context(store, alias="review-1", **kwargs):
    context = store.create(**kwargs)
    _ALIASES.setdefault(store, {})[alias] = context.review_id
    return context


def rid(store, alias="review-1"):
    return _ALIASES.get(store, {}).get(alias, alias)


@pytest.fixture
def manager(tmp_path):
    owner = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    yield owner
    assert owner.dispose()


@pytest.fixture
def store(manager):
    result = ReviewStore(manager, "session")
    context = create_context(result, "review-1", target_digest="b" * 64)
    content = b"one\ntwo\nthree\nfour\n"
    with result.reserve(
        rid(result, "review-1"), source_bytes=len(content), report_bytes=0, files=1
    ) as reservation:
        path = reservation.write(content)
        reservation.commit(len(content), 0, 1)
    context.sources["src-1"] = CapturedSource(
        path=path, digest=hashlib.sha256(content).hexdigest(), line_count=4
    )
    result.bind_attempt("default", binding(result))
    with use_run_id("default"):
        yield result


def binding(store, slot="R1", attempt="a1", role="reviewer", **overrides):
    data = dict(
        review_id=rid(store),
        slot_id=slot,
        attempt_id=attempt,
        role=role,
        target_digest="b" * 64,
        source_epoch=0,
    )
    if role == "verifier":
        data["check_feedback_mode"] = "unavailable_inline"
    return ReviewRunBinding(**(data | overrides))


@pytest.fixture
def bound_attempt(store):
    @contextmanager
    def bound(slot_id, attempt_id):
        run = f"run-{slot_id}-{attempt_id}"
        store.bind_attempt(run, binding(store, slot_id, attempt_id))
        with use_run_id(run):
            yield

    return bound


def payload(store):
    record = finding_payload()
    with store.lease(rid(store, "review-1")) as context:
        record["evidence"][0]["digest"] = context.sources["src-1"].digest
    return record


def publish(store, **kwargs):
    return store.publish(
        rid(store, "review-1"),
        expected_revision=0,
        submission_id="s1",
        records=[payload(store)],
        **kwargs,
    )


def test_same_submission_replays_before_stale_revision_check(store):
    record = payload(store)
    first = publish(store)
    before = store.usage()
    assert publish(store) == first
    assert store.usage() == before
    with pytest.raises(ReviewRefused, match="submission_conflict"):
        store.publish(
            rid(store, "review-1"),
            expected_revision=0,
            submission_id="s1",
            records=[record | {"impact": "Different claim"}],
        )
    assert store.usage().retained_bytes > before.retained_bytes
    with store.lease(rid(store)) as context:
        assert context.publication_sequence == 1
        assert tuple(context.original_index) == first.accepted_ids
        assert (
            store.read_record(context, context.original_index[first.accepted_ids[0]])
            == record
        )
        assert context.streams[("R1", "a1")].revision == 1


def test_participants_append_without_invalidating_each_other(store, bound_attempt):
    barrier = threading.Barrier(2)
    record = payload(store)

    def submit(slot):
        with bound_attempt(slot, f"attempt-{slot}"):
            barrier.wait(timeout=5)
            return store.publish(
                rid(store, "review-1"),
                expected_revision=0,
                submission_id="s1",
                records=[record],
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        acks = [
            future.result(timeout=10)
            for future in [pool.submit(submit, slot) for slot in ("R1", "R2")]
        ]
    assert [ack.submission_revision for ack in acks] == [1, 1]
    assert len({ack.accepted_ids[0] for ack in acks}) == 2
    with bound_attempt("R1", "attempt-R1"):
        with pytest.raises(
            ReviewRefused, match="submission_revision_conflict"
        ) as refused:
            store.publish(
                rid(store, "review-1"),
                expected_revision=0,
                submission_id="s2",
                records=[record],
            )
        assert refused.value.submission_revision == 1
    with bound_attempt("R2", "attempt-R2"):
        assert publish(store) == acks[1]
    with store.lease(rid(store, "review-1")) as context:
        assert context.revision == 1  # R1 replaced the fixture's initial attempt.
        assert context.publication_sequence == 2
        assert len(context.original_index) == 2


@pytest.mark.parametrize(
    "field,source,report,files,participant",
    [
        ("source_max_bytes", 10, 0, 1, None),
        ("reports_max_bytes", 0, 10, 1, None),
        ("participant_max_bytes", 0, 10, 1, "R1"),
        ("review_max_bytes", 5, 5, 2, None),
        ("review_max_files", 0, 0, 10, None),
        ("tab_max_bytes", 5, 5, 2, None),
    ],
)
def test_exact_byte_and_file_caps(manager, field, source, report, files, participant):
    result = ReviewStore(
        manager, "session", limits=replace(ReviewLimits(), **{field: 10})
    )
    create_context(result, "r", target_digest="b" * 64)
    with result.reserve(
        rid(result, "r"),
        source_bytes=source,
        report_bytes=report,
        files=files,
        participant_id=participant,
    ) as reservation:
        with pytest.raises(ReviewRefused, match="quota"):
            result.reserve(
                rid(result, "r"),
                source_bytes=int(source > 0),
                report_bytes=int(not source and report > 0),
                files=1,
                participant_id=participant,
            )
        for index in range(files):
            if index == 0 and source:
                reservation.write(b"x" * source)
            elif report and (not source or index == 1):
                reservation.write(b"x" * report, category="report")
            else:
                reservation.write(b"")
        reservation.commit(source, report, files)
    usage = result.usage()
    assert usage.retained_bytes == source + report
    assert usage.retained_files == files
    assert usage.reserved_bytes == 0
    with pytest.raises(ReviewRefused, match="quota"):
        result.reserve(
            rid(result, "r"),
            source_bytes=int(source > 0),
            report_bytes=int(not source and report > 0),
            files=1,
            participant_id=participant,
        )


def test_context_cap_never_evicts_retained_context(manager):
    result = ReviewStore(
        manager, "session", limits=replace(ReviewLimits(), tab_max_contexts=2)
    )
    for review_id in ("one", "two"):
        create_context(result, review_id, target_digest="b" * 64)
    with pytest.raises(ReviewRefused, match="quota"):
        create_context(result, "three", target_digest="b" * 64)
    assert result.usage().retained_contexts == 2
    with result.lease(rid(result, "one")):
        pass


def test_reservations_race_atomically(manager):
    result = ReviewStore(
        manager, "session", limits=replace(ReviewLimits(), tab_max_bytes=10)
    )
    create_context(result, "r", target_digest="b" * 64)
    start, held = threading.Barrier(2), threading.Barrier(2)

    def reserve():
        start.wait(timeout=5)
        try:
            reservation = result.reserve(
                rid(result, "r"), source_bytes=6, report_bytes=0, files=1
            )
        except ReviewRefused:
            held.wait(timeout=5)
            return False
        with reservation:
            held.wait(timeout=5)
            return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(reserve) for _ in range(2)]
        assert sorted(future.result(timeout=10) for future in futures) == [False, True]
    assert result.usage().reserved_bytes == 0


def test_write_cannot_grow_beyond_reservation_and_copies_are_charged(manager):
    result = ReviewStore(manager, "s")
    create_context(result, "r", target_digest="b" * 64)
    with result.reserve(
        rid(result, "r"), source_bytes=6, report_bytes=0, files=2
    ) as reservation:
        first = reservation.write(b"abc")
        second = reservation.write(first.read_bytes())
        assert result.usage().reserved_bytes + result.usage().retained_bytes == 6
        with pytest.raises(ReviewRefused, match="reservation_exceeded"):
            reservation.write(b"x")
        reservation.commit(6, 0, 2)
    assert first != second
    assert first.stat().st_mode & 0o777 == 0o600
    assert result.usage().retained_bytes == 6


def test_commit_rejects_misdeclared_actual_usage(manager):
    result = ReviewStore(manager, "s")
    create_context(result, "r", target_digest="b" * 64)
    with pytest.raises(ReviewRefused, match="reservation_actual_mismatch"):
        with result.reserve(
            rid(result, "r"), source_bytes=3, report_bytes=0, files=1
        ) as reservation:
            reservation.write(b"abc")
            reservation.commit(0, 0, 0)
    assert result.usage().retained_bytes == 0


def test_release_refuses_active_lease_then_is_idempotent(store):
    before = store.usage()
    with store.lease(rid(store, "review-1")):
        with pytest.raises(ReviewRefused, match="review_busy"):
            store.release(rid(store, "review-1"), expected_revision=0)
    assert store.usage() == before
    first = store.release(rid(store, "review-1"), expected_revision=0)
    assert first.tombstoned and first.cleaned
    assert store.release(rid(store, "review-1"), expected_revision=0) == first
    assert store.usage().retained_contexts == 0
    with pytest.raises(ReviewRefused, match="review_expired"):
        publish(store)
    old_id = rid(store)
    fresh = create_context(store, "review-1", target_digest="b" * 64)
    assert fresh.review_id != old_id


def test_release_failed_delete_stays_charged_until_retry(store, monkeypatch):
    before = store.usage()
    original = Path.unlink
    monkeypatch.setattr(
        Path, "unlink", lambda *a, **k: (_ for _ in ()).throw(OSError("disk error"))
    )
    receipt = store.release(rid(store, "review-1"), expected_revision=0)
    assert not receipt.cleaned
    assert store.usage().retained_bytes == before.retained_bytes
    assert store.usage().retained_contexts == 1
    monkeypatch.setattr(Path, "unlink", original)
    assert store.release(rid(store, "review-1"), expected_revision=0).cleaned
    assert store.usage().retained_bytes == 0


@pytest.mark.parametrize("fault", ["write", "replace"])
def test_publication_fault_preserves_old_manifest_and_charges_orphans(
    store, monkeypatch, fault
):
    first = publish(store)
    before = store.usage()
    with store.lease(rid(store, "review-1")) as context:
        manifest = context.manifest_path.read_bytes()
    original_unlink = Path.unlink
    monkeypatch.setattr(
        Path, "unlink", lambda *a, **k: (_ for _ in ()).throw(OSError("delete error"))
    )
    function = "write" if fault == "write" else "replace"
    original = getattr(os, function)

    def fail(*args):
        if fault == "write":
            original(args[0], args[1][:3])
        raise OSError("disk error")

    monkeypatch.setattr(os, function, fail)
    with pytest.raises(ReviewRefused, match="publication_failed"):
        store.publish(
            rid(store, "review-1"),
            expected_revision=1,
            submission_id="s2",
            records=[payload(store) | {"local_key": "second"}],
        )
    assert store.usage().retained_bytes > before.retained_bytes
    assert store.usage().reserved_bytes == 0
    with store.lease(rid(store, "review-1")) as context:
        assert context.manifest_path.read_bytes() == manifest
        assert context.publication_sequence == 1
        assert len(context.original_index) == 1
    monkeypatch.setattr(os, function, original)
    monkeypatch.setattr(Path, "unlink", original_unlink)
    assert publish(store) == first


def test_binding_cannot_be_reassociated_or_cross_review(store):
    with pytest.raises(ReviewRefused, match="run_binding_conflict"):
        store.bind_attempt("default", binding(store, "R2"))
    with use_run_id("unbound"):
        with pytest.raises(ReviewRefused, match="run_not_bound"):
            publish(store)
    create_context(store, "other", target_digest="b" * 64)
    with pytest.raises(ReviewRefused, match="review_scope_mismatch"):
        store.publish(
            rid(store, "other"),
            expected_revision=0,
            submission_id="s",
            records=[payload(store)],
        )
    with pytest.raises(ReviewRefused, match="source_epoch_mismatch"):
        store.bind_attempt("old", binding(store, source_epoch=1))


def test_new_attempt_revokes_old_callback_without_changing_other_slots(
    store, bound_attempt
):
    publish(store)
    with bound_attempt("R2", "a2"):
        other = publish(store)
    with bound_attempt("R1", "retry"):
        assert publish(store).submission_revision == 1
    with pytest.raises(ReviewRefused, match="stale_attempt"):
        publish(store)
    with bound_attempt("R2", "a2"):
        assert publish(store) == other


@pytest.mark.parametrize("mutation", ["source", "line", "digest", "version"])
def test_invalid_records_never_change_accepted_data(store, mutation):
    record = payload(store)
    if mutation == "source":
        record["locations"][0]["source_ref"] = "unknown"
    elif mutation == "line":
        record["locations"][0]["end_line"] = 5
    elif mutation == "digest":
        record["evidence"][0]["digest"] = "c" * 64
    else:
        record["schema_version"] = 2
    before = store.usage()
    with pytest.raises(ReviewRefused):
        store.publish(
            rid(store, "review-1"),
            expected_revision=0,
            submission_id="s1",
            records=[record],
        )
    assert store.usage().retained_bytes > before.retained_bytes
    with store.lease(rid(store)) as context:
        assert not context.original_index
        assert context.publication_sequence == 0
        assert context.streams[("R1", "a1")].revision == 0
        assert context.streams[("R1", "a1")].rejected


def test_kind_is_scoped_and_completion_is_only_a_request(store):
    with pytest.raises(ReviewRefused, match="role_kind_mismatch"):
        store.publish(
            rid(store, "review-1"),
            kind="verifier_complete",
            expected_revision=0,
            submission_id="s",
            records=[],
        )
    first = store.publish(
        rid(store, "review-1"),
        kind="coverage",
        expected_revision=0,
        submission_id="s",
        records=[coverage_payload()],
    )
    with pytest.raises(ReviewRefused, match="submission_conflict"):
        store.publish(
            rid(store, "review-1"),
            kind="reviewer_complete",
            expected_revision=0,
            submission_id="s",
            records=[coverage_payload()],
        )
    assert first.submission_revision == 1
    with store.lease(rid(store, "review-1")) as context:
        assert context.state == "live"


def test_close_drains_cancelled_but_live_worker_and_revokes_callbacks(store, manager):
    started, finish = threading.Event(), threading.Event()

    def worker():
        with store.reserve(
            rid(store, "review-1"), source_bytes=10, report_bytes=0, files=1
        ) as reservation:
            reservation.write(b"pending")
            started.set()
            assert finish.wait(5)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker)
        assert started.wait(5)
        assert not future.cancel()
        store.close()
        assert store.usage().reserved_bytes + store.usage().retained_bytes >= 10
        assert not manager.wait_for_cleanup(0.01)
        with pytest.raises(ReviewRefused, match="review_expired"):
            publish(store)
        finish.set()
        future.result(timeout=10)
    assert manager.wait_for_cleanup(2)
    assert store.usage().retained_bytes == 0
    assert store.usage().retained_contexts == 0


def test_scratch_close_during_chunk_write_refuses_publication(
    store, manager, monkeypatch
):
    first = publish(store)
    original = os.write
    fired = False

    def close_during_write(fd, data):
        nonlocal fired
        result = original(fd, data)
        if not fired:
            fired = True
            manager.close("session")
        return result

    monkeypatch.setattr(os, "write", close_during_write)
    with pytest.raises(ReviewRefused, match="review_expired"):
        store.publish(
            rid(store, "review-1"),
            expected_revision=1,
            submission_id="s2",
            records=[payload(store) | {"local_key": "second"}],
        )
    assert first.submission_revision == 1
    assert manager.wait_for_cleanup(2)
    with pytest.raises(ReviewRefused, match="review_expired"):
        publish(store)


def test_stores_for_same_tab_share_quota(manager):
    limits = replace(ReviewLimits(), tab_max_bytes=10)
    first = ReviewStore(manager, "s", limits=limits)
    second = ReviewStore(manager, "s", limits=limits)
    create_context(first, "one", target_digest="b" * 64)
    create_context(second, "two", target_digest="b" * 64)
    with first.reserve(rid(first, "one"), source_bytes=10, report_bytes=0, files=1):
        with pytest.raises(ReviewRefused, match="quota"):
            second.reserve(rid(second, "two"), source_bytes=1, report_bytes=0, files=1)


def test_reopened_session_does_not_reanimate_old_store(store, manager):
    manager.close("session")
    assert manager.wait_for_cleanup(2)
    fresh = ReviewStore(manager, "session")
    create_context(fresh, "review-1", target_digest="b" * 64)
    with pytest.raises(ReviewRefused, match="review_expired"):
        publish(store)


def test_old_store_close_does_not_close_reopened_session(store, manager):
    manager.close("session")
    assert manager.wait_for_cleanup(2)
    fresh = ReviewStore(manager, "session")
    create_context(fresh, "fresh", target_digest="b" * 64)
    store.close()
    with fresh.lease(rid(fresh, "fresh")):
        pass


def test_partial_owned_cleanup_refunds_only_successful_unlinks_then_retries(
    store, manager, monkeypatch
):
    import tldw_chatbook.Chat.console_scratch_space as scratch_module

    with store.reserve(
        rid(store), source_bytes=3, report_bytes=0, files=1
    ) as reservation:
        remaining = reservation.write(b"abc")
        reservation.commit(3, 0, 1)
    unlink = Path.unlink

    def fail_one(path, *args, **kwargs):
        if path == remaining:
            raise OSError("owned delete interrupted")
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "unlink", fail_one)
        fault.setattr(
            scratch_module.shutil,
            "rmtree",
            lambda path: (_ for _ in ()).throw(OSError("cleanup interrupted")),
        )
        store.close()
        assert not manager.wait_for_cleanup(0.05)
        assert remaining.read_bytes() == b"abc"
        assert store.usage().retained_bytes == 3
        assert store.usage().retained_files == 1
    assert manager.dispose()
    assert store.usage().retained_bytes == 0


def test_repeated_release_does_not_delete_replaced_generation(
    store, manager, monkeypatch, tmp_path
):
    with store.lease(rid(store, "review-1")) as context:
        root = context.scratch.root
        relative = context.sources["src-1"].path.relative_to(root)
    unlink = Path.unlink
    monkeypatch.setattr(
        Path, "unlink", lambda *a, **k: (_ for _ in ()).throw(OSError("delete error"))
    )
    assert not store.release(rid(store, "review-1"), expected_revision=0).cleaned
    monkeypatch.setattr(Path, "unlink", unlink)
    displaced = tmp_path / "displaced"
    root.rename(displaced)
    replacement = root / relative
    replacement.parent.mkdir(parents=True)
    replacement.write_bytes(b"keep replacement")
    assert not store.release(rid(store, "review-1"), expected_revision=0).cleaned
    assert replacement.read_bytes() == b"keep replacement"
    assert store.usage().retained_bytes > 0
    assert manager.wait_for_cleanup(2)


def test_concurrent_output_writers_count_actual_bytes_once(store):
    barrier = threading.Barrier(2)

    def writer(participant):
        with store.reserve(
            rid(store, "review-1"),
            source_bytes=0,
            report_bytes=5,
            files=1,
            participant_id=participant,
        ) as reservation:
            barrier.wait(timeout=5)
            path = reservation.write(b"hello", category="report")
            reservation.commit(0, 5, 1)
            return path

    before = store.usage()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(writer, actor) for actor in ("R1", "R2")]
        paths = [future.result(timeout=10) for future in futures]
    assert paths[0] != paths[1]
    assert all(path.read_bytes() == b"hello" for path in paths)
    assert store.usage().retained_bytes == before.retained_bytes + 10
    assert store.usage().reserved_bytes == 0


def test_append_keeps_prior_chunk_immutable_and_manifest_small(store):
    first = publish(store)
    with store.lease(rid(store, "review-1")) as context:
        first_path, _ = context.original_index[first.accepted_ids[0]]
        first_bytes = first_path.read_bytes()
    for revision in range(1, 5):
        store.publish(
            rid(store, "review-1"),
            expected_revision=revision,
            submission_id=f"s{revision + 1}",
            records=[payload(store) | {"local_key": f"key-{revision}"}],
        )
    with store.lease(rid(store, "review-1")) as context:
        assert first_path.read_bytes() == first_bytes
        pointer = json.loads(context.manifest_path.read_bytes())
        assert pointer["sequence"] == 5
        assert context.manifest_path.stat().st_size < 100
        current = context.bundle_path / pointer["chunk"]
        count = 0
        while current:
            chunk = json.loads(current.read_bytes())
            count += len(chunk["records"])
            current = (
                context.bundle_path / chunk["previous"] if chunk["previous"] else None
            )
        assert count == 5
        assert store.usage().retained_bytes == sum(
            path.stat().st_size for path in context.bundle_path.iterdir()
        )


def test_changed_source_epoch_revokes_even_an_identical_replay(store):
    publish(store)
    with store.lease(rid(store, "review-1")) as context:
        context.source_epoch += 1
    with pytest.raises(ReviewRefused, match="source_epoch_mismatch"):
        publish(store)


def test_local_key_reuse_is_rejected_but_idempotent_replay_survives(store):
    first = publish(store)
    with pytest.raises(ReviewRefused, match="duplicate_local_key"):
        store.publish(
            rid(store, "review-1"),
            expected_revision=1,
            submission_id="s2",
            records=[payload(store)],
        )
    assert publish(store) == first


def test_release_uses_context_revision_not_submission_revision(store):
    publish(store)
    with pytest.raises(ReviewRefused, match="context_revision_conflict"):
        store.release(rid(store, "review-1"), expected_revision=1)
    assert store.release(rid(store, "review-1"), expected_revision=0).cleaned


@pytest.mark.parametrize(
    "counts", [(True, 0, 0), (-1, 0, 0), (0, 1.0, 0), (0, 0, False)]
)
def test_reservations_reject_coerced_or_negative_counts(store, counts):
    with pytest.raises(ReviewRefused, match="invalid_storage_counts"):
        store.reserve(
            rid(store, "review-1"),
            source_bytes=counts[0],
            report_bytes=counts[1],
            files=counts[2],
        )


def test_production_storage_caps_are_exact():
    limits = ReviewLimits()
    assert (
        limits.source_max_bytes,
        limits.reports_max_bytes,
        limits.participant_max_bytes,
        limits.review_max_bytes,
        limits.review_max_files,
        limits.tab_max_bytes,
        limits.tab_max_contexts,
    ) == (67108864, 16777216, 1048576, 83886080, 2048, 167772160, 4)


def test_scratch_guard_serializes_commit_and_close(manager):
    snapshot = manager.snapshot("s")
    started, closed = threading.Event(), threading.Event()

    def close():
        started.set()
        manager.close("s")
        closed.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with manager.lease(rid(manager, snapshot)), manager.live_guard(snapshot):
            future = pool.submit(close)
            assert started.wait(5)
            assert not closed.wait(0.05)
            assert manager.is_live(snapshot)
        future.result(timeout=10)
    assert closed.is_set()
    assert not manager.is_live(snapshot)


def test_reservation_rollback_cannot_delete_replaced_generation(
    store, manager, tmp_path
):
    with store.reserve(
        rid(store, "review-1"), source_bytes=3, report_bytes=0, files=1
    ) as reservation:
        pending = reservation.write(b"abc")
        with store.lease(rid(store, "review-1")) as context:
            root = context.scratch.root
        relative = pending.relative_to(root)
        root.rename(tmp_path / "displaced")
        replacement = root / relative
        replacement.parent.mkdir(parents=True)
        replacement.write_bytes(b"keep replacement")
    assert replacement.read_bytes() == b"keep replacement"
    assert store.usage().retained_bytes >= 3
    assert manager.wait_for_cleanup(2)


def test_displaced_root_without_replacement_is_not_deletion_proof(
    store, manager, tmp_path
):
    before = store.usage()
    with store.lease(rid(store, "review-1")) as context:
        root = context.scratch.root
    root.rename(tmp_path / "displaced")
    store.close()
    assert manager.wait_for_cleanup(2)
    assert store.usage().retained_bytes == before.retained_bytes
    assert store.usage().retained_contexts == 1


def test_retry_retains_prior_original_ids_and_chunks(store, bound_attempt):
    first = publish(store)
    with store.lease(rid(store, "review-1")) as context:
        first_entry = context.original_index[first.accepted_ids[0]]
        original_bytes = first_entry[0].read_bytes()
    with bound_attempt("R2", "a2"):
        other = publish(store)
    with bound_attempt("R1", "retry"):
        retried = publish(store)
        assert retried.submission_revision == 1
        assert retried.accepted_ids != first.accepted_ids
    with store.lease(rid(store, "review-1")) as context:
        assert context.original_index[first.accepted_ids[0]] == first_entry
        assert first_entry[0].read_bytes() == original_bytes
        assert len(context.original_index) == 3
    with bound_attempt("R2", "a2"):
        assert publish(store) == other
    with pytest.raises(ReviewRefused, match="stale_attempt"):
        store.bind_attempt("new-run-old-attempt", binding(store))


def test_peak_staging_headroom_is_reserved_before_manifest_write(manager):
    record = {**coverage_payload(), "inspected_refs": [], "empty_findings": True}
    result = ReviewStore(
        manager, "s", limits=replace(ReviewLimits(), review_max_files=2)
    )
    create_context(result, "r", target_digest="b" * 64)
    result.bind_attempt(
        "r1", binding(result, review_id=rid(result, "r"), slot="R1", role="reviewer")
    )
    with use_run_id("r1"):
        first = result.publish(
            rid(result, "r"),
            kind="reviewer_complete",
            records=[record],
            expected_revision=0,
            submission_id="one",
        )
        assert result.usage().retained_files == 2
        with pytest.raises(ReviewRefused, match="quota"):
            result.publish(
                rid(result, "r"),
                kind="reviewer_complete",
                records=[record],
                expected_revision=1,
                submission_id="two",
            )
        assert (
            result.publish(
                rid(result, "r"),
                kind="reviewer_complete",
                records=[record],
                expected_revision=0,
                submission_id="one",
            )
            == first
        )


@pytest.mark.parametrize("declared_bytes", [3, 6])
def test_commit_cannot_legitimize_outside_writer_growth(
    store, monkeypatch, declared_bytes
):
    before = store.usage().retained_bytes
    with pytest.raises(ReviewRefused, match="reservation_exceeded"):
        with store.reserve(
            rid(store, "review-1"), source_bytes=3, report_bytes=0, files=1
        ) as reservation:
            path = reservation.write(b"abc")
            path.write_bytes(b"abcdef")  # A broken host writer bypassed write().
            monkeypatch.setattr(
                Path,
                "unlink",
                lambda *a, **k: (_ for _ in ()).throw(OSError("cannot delete")),
            )
            reservation.commit(declared_bytes, 0, 1)
    assert store.usage().retained_bytes == before + 6
    assert store.usage().reserved_bytes == 0


def test_cleanup_receipt_stays_pending_after_failure_and_settles_on_retry(
    manager, monkeypatch
):
    import tldw_chatbook.Chat.console_scratch_space as scratch_module

    snapshot = manager.snapshot("s")
    receipt = manager.observe_cleanup(snapshot)
    assert receipt.outcome is None
    original = scratch_module.shutil.rmtree
    with monkeypatch.context() as faults:
        faults.setattr(
            scratch_module.shutil,
            "rmtree",
            lambda path: (_ for _ in ()).throw(OSError("cannot delete")),
        )
        manager.close("s")
        assert not manager.wait_for_cleanup(0.05)
        assert receipt.outcome is None
    assert scratch_module.shutil.rmtree == original
    assert manager.dispose()
    assert receipt.outcome == "deleted"


@pytest.mark.parametrize(
    "replace_root, expected", [(False, "already-missing"), (True, "identity-changed")]
)
def test_cleanup_receipt_distinguishes_revocation_from_deletion(
    manager, tmp_path, replace_root, expected
):
    snapshot = manager.snapshot("s")
    receipt = manager.observe_cleanup(snapshot)
    snapshot.root.rename(tmp_path / "displaced")
    if replace_root:
        snapshot.root.mkdir()
    manager.close("s")
    assert manager.wait_for_cleanup(2)
    assert receipt.outcome == expected


def test_deleting_pending_file_does_not_refund_live_worker_reservation(store):
    before = store.usage().retained_bytes
    with store.reserve(
        rid(store, "review-1"), source_bytes=3, report_bytes=0, files=1
    ) as reservation:
        path = reservation.write(b"abc")
        path.unlink()
        usage = store.usage()
        assert usage.retained_bytes + usage.reserved_bytes == before + 3
        assert usage.retained_files + usage.reserved_files == 2
    assert store.usage().reserved_bytes == 0


def test_primary_action_rejects_unknown_proposal_reference(store):
    store.bind_attempt("primary", binding(store, slot="primary", role="primary"))
    store.bind_attempt(
        "verifier", binding(store, slot="V", attempt="v1", role="verifier")
    )
    store.bind_check_actor("primary", rid(store), "v1")
    with (
        use_run_id("primary"),
        pytest.raises(ReviewRefused, match="invalid_check_proposal"),
    ):
        store.publish(
            rid(store, "review-1"),
            kind="check_actions",
            expected_revision=0,
            submission_id="s",
            records=[
                {"schema_version": 1, "proposal_id": "unknown", "action": "claim"}
            ],
        )


@pytest.mark.parametrize(
    "kind, record, reason",
    [
        (
            "coverage",
            {
                "schema_version": 1,
                "inspected_refs": ["unknown"],
                "gaps": [],
                "empty_findings": False,
            },
            "invalid_source_reference",
        ),
        (
            "dispositions",
            {
                "schema_version": 1,
                "verification_unit_id": "unknown",
                "verdict": "confirmed",
                "reason": "read source",
                "evidence": [],
            },
            "invalid_verification_unit",
        ),
        (
            "check_proposals",
            {
                "schema_version": 1,
                "argv": ["python", "check.py"],
                "intent": "check",
                "verification_unit_ids": ["unknown"],
            },
            "invalid_verification_unit",
        ),
    ],
)
def test_nonfinding_references_are_checked(store, kind, record, reason):
    role = "reviewer" if kind == "coverage" else "verifier"
    store.bind_attempt("participant", binding(store, slot="other", role=role))
    with use_run_id("participant"), pytest.raises(ReviewRefused, match=reason):
        store.publish(
            rid(store, "review-1"),
            kind=kind,
            expected_revision=0,
            submission_id="s",
            records=[record],
        )


def test_check_receipt_evidence_must_be_host_registered(store):
    record = payload(store)
    record["evidence"] = [{"kind": "check", "receipt_id": "unknown"}]
    with pytest.raises(ReviewRefused, match="invalid_check_receipt"):
        store.publish(
            rid(store, "review-1"),
            expected_revision=0,
            submission_id="s",
            records=[record],
        )


def test_store_close_revokes_without_waiting_for_stalled_writer(
    store, manager, monkeypatch
):
    entered, unblock = threading.Event(), threading.Event()
    record = payload(store)
    with store.lease(rid(store, "review-1")) as context:
        snapshot = context.scratch
    original = os.write

    def stalled_write(fd, data):
        entered.set()
        assert unblock.wait(5)
        return original(fd, data)

    monkeypatch.setattr(os, "write", stalled_write)

    def writer():
        with use_run_id("default"):
            return store.publish(
                rid(store, "review-1"),
                expected_revision=0,
                submission_id="s",
                records=[record],
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        writing = pool.submit(writer)
        assert entered.wait(5)
        closing = pool.submit(store.close)
        try:
            closing.result(timeout=1)
            assert not manager.is_live(snapshot)
            assert not manager.wait_for_cleanup(0.01)
        finally:
            unblock.set()
        with pytest.raises(ReviewRefused, match="review_expired"):
            writing.result(timeout=5)
    assert manager.wait_for_cleanup(2)
    assert store.usage().retained_bytes == 0


@pytest.mark.parametrize("move_bundle", [True, False])
def test_unlocated_contents_stay_conservatively_charged_after_root_cleanup(
    store, manager, move_bundle
):
    before = store.usage()
    with store.lease(rid(store, "review-1")) as context:
        source = context.sources["src-1"].path
        bundle = context.bundle_path
    moved = (
        bundle.with_name("displaced-bundle")
        if move_bundle
        else source.with_name("displaced-file")
    )
    (bundle if move_bundle else source).rename(moved)
    assert store.usage() == before
    assert not store.release(rid(store, "review-1"), expected_revision=0).cleaned
    assert store.usage() == before
    assert moved.exists()
    store.close()
    assert manager.wait_for_cleanup(2)
    assert not moved.exists()
    # Root deletion does not establish which displaced owned identities it covered.
    assert store.usage().retained_bytes == 19
    assert store.usage().retained_contexts == 1


@pytest.mark.parametrize("move_bundle", [True, False])
@pytest.mark.parametrize("prior_access", ["none", "usage", "release"])
def test_outside_root_displacement_survives_close_and_stays_charged(
    store, manager, tmp_path, move_bundle, prior_access
):
    with store.lease(rid(store)) as context:
        source = context.sources["src-1"].path
        bundle = context.bundle_path
        root = context.scratch.root
    moved = tmp_path / "outside-owned-content"
    (bundle if move_bundle else source).rename(moved)
    if prior_access == "usage":
        assert store.usage().retained_bytes == 19
    elif prior_access == "release":
        assert not store.release(rid(store), expected_revision=0).cleaned
    store.close()
    assert manager.wait_for_cleanup(2)
    assert not root.exists()
    surviving_source = moved / source.name if move_bundle else moved
    assert surviving_source.read_bytes() == b"one\ntwo\nthree\nfour\n"
    assert store.usage().retained_bytes == 19
    assert store.usage().retained_files == 1
    assert store.usage().retained_contexts == 1
    assert not context.sources and not context.streams


def test_replacement_file_is_never_deleted_or_counted_as_owned(store, manager):
    before = store.usage()
    with store.lease(rid(store, "review-1")) as context:
        source = context.sources["src-1"].path
    moved = source.with_name("displaced-file")
    source.rename(moved)
    source.write_bytes(b"replacement")
    assert store.usage() == before
    assert not store.release(rid(store, "review-1"), expected_revision=0).cleaned
    assert source.read_bytes() == b"replacement"
    assert moved.exists()
    store.close()
    assert manager.wait_for_cleanup(2)
    assert store.usage().retained_bytes == 19


def test_released_context_payloads_are_collectable_and_receipts_bounded(manager):
    import gc
    import weakref

    record = {**coverage_payload(), "inspected_refs": [], "empty_findings": True}
    result = ReviewStore(manager, "s")
    references = []
    for index in range(12):
        context = create_context(result, f"review-{index}", target_digest="b" * 64)
        references.append(weakref.ref(context))
        result.bind_attempt(
            f"run-{index}",
            binding(result, review_id=context.review_id, slot="R1", role="reviewer"),
        )
        with use_run_id(f"run-{index}"):
            result.publish(
                context.review_id,
                kind="reviewer_complete",
                expected_revision=0,
                submission_id="s",
                records=[record],
            )
        result.release(context.review_id, expected_revision=0)
        assert not context.streams
        assert not context.active_attempts
        del context
    gc.collect()
    assert all(reference() is None for reference in references)
    assert not result._tab.runs
    assert not result._tab.retired_attempts
    assert len(result._tab.release_receipts) <= 4


def test_closed_generations_retire_without_another_store_call(manager):
    import gc
    import weakref

    import tldw_chatbook.Agents.code_review_store as module

    references = []
    for index in range(6):
        result = ReviewStore(manager, "s")
        context = create_context(result, f"review-{index}", target_digest="b" * 64)
        references.append(weakref.ref(context))
        result.close()
        assert manager.wait_for_cleanup(2)
        assert not module._TABS.get(manager)
        assert not context.sources
        del context, result
    gc.collect()
    assert all(reference() is None for reference in references)


def test_live_generation_remembers_quotas_without_a_store_handle(manager):
    import gc

    result = ReviewStore(manager, "s")
    context = create_context(result, "r", target_digest="b" * 64)
    with result.reserve(
        context.review_id, source_bytes=3, report_bytes=0, files=1
    ) as reservation:
        reservation.write(b"abc")
        reservation.commit(3, 0, 1)
    del reservation, context, result
    gc.collect()
    fresh = ReviewStore(manager, "s")
    assert fresh.usage().retained_bytes == 3
    assert fresh.usage().retained_contexts == 1


def test_expired_release_receipt_cannot_recreate_an_old_identity(manager):
    result = ReviewStore(manager, "s")
    issued = []
    for _ in range(6):
        context = result.create(target_digest="b" * 64)
        issued.append(context.review_id)
        result.release(context.review_id, expected_revision=0)
    assert len(set(issued)) == 6
    assert result.release(issued[-1], expected_revision=0).cleaned
    with pytest.raises(ReviewRefused, match="unknown_review|review_expired"):
        result.release(issued[0], expected_revision=0)
    with pytest.raises(ReviewRefused, match="unknown_review|review_expired"):
        with result.lease(issued[0]):
            pass
    with pytest.raises(TypeError):
        result.create(issued[0], target_digest="b" * 64)
    assert result.create(target_digest="b" * 64).review_id not in issued


def test_failed_release_discards_payload_but_retains_compact_orphan_charge(
    store, monkeypatch
):
    publish(store)
    with store.lease(rid(store)) as context:
        source = context.sources["src-1"].path
    source.rename(source.with_name("displaced-file"))
    receipt = store.release(rid(store), expected_revision=0)
    assert not receipt.cleaned
    assert not context.sources
    assert not context.original_index
    assert not context.streams
    assert not context.active_attempts
    assert not store._tab.runs
    assert store.usage().retained_bytes == 19
    assert store.usage().retained_files == 1
    assert store.usage().retained_contexts == 1


def test_terminal_displacement_removes_registry_but_stale_handle_keeps_charges(
    store, manager, tmp_path
):
    import tldw_chatbook.Agents.code_review_store as module

    publish(store)
    before = store.usage()
    with store.lease(rid(store)) as context:
        root = context.scratch.root
    root.rename(tmp_path / "displaced")
    store.close()
    assert manager.wait_for_cleanup(2)
    assert not module._TABS.get(manager)
    assert store.usage() == before
    assert not context.sources
    assert not context.original_index
    assert not context.streams
    with pytest.raises(ReviewRefused, match="review_expired"):
        store.create(target_digest="b" * 64)


def test_cleanup_callbacks_run_unlocked_and_one_failure_does_not_strand_cleanup(
    manager,
):
    snapshot = manager.snapshot("s")
    receipt = manager.observe_cleanup(snapshot)
    observations = []

    def callback(outcome):
        with ThreadPoolExecutor(max_workers=1) as pool:
            observed = pool.submit(lambda: (manager.is_live(snapshot), receipt.outcome))
            observations.append(observed.result(timeout=1))
        assert outcome == "deleted"

    def failing(outcome):
        raise RuntimeError("private report body must not be logged")

    preparations = []

    def prepare():
        with ThreadPoolExecutor(max_workers=1) as pool:
            observed = pool.submit(lambda: (manager.is_live(snapshot), receipt.outcome))
            preparations.append(observed.result(timeout=1))
        assert snapshot.root.exists()

    receipt.add_cleanup_callback(lambda: failing(None))
    receipt.add_cleanup_callback(prepare)
    receipt.add_done_callback(failing)
    receipt.add_done_callback(callback)
    manager.close("s")
    assert manager.wait_for_cleanup(2)
    assert observations == [(False, "deleted")]
    assert preparations == [(False, None)]
    receipt.add_cleanup_callback(prepare)
    assert preparations == [(False, None)]
    receipt.add_done_callback(callback)
    assert observations == [(False, "deleted"), (False, "deleted")]


def test_cleanup_callbacks_wait_for_actual_successful_retry(manager, monkeypatch):
    import tldw_chatbook.Chat.console_scratch_space as scratch_module

    snapshot = manager.snapshot("s")
    receipt = manager.observe_cleanup(snapshot)
    observed = []
    preparations = []
    receipt.add_cleanup_callback(lambda: preparations.append(snapshot.root.exists()))
    receipt.add_done_callback(observed.append)
    with monkeypatch.context() as fault:
        fault.setattr(
            scratch_module.shutil,
            "rmtree",
            lambda path: (_ for _ in ()).throw(OSError("delete failed")),
        )
        manager.close("s")
        assert not manager.wait_for_cleanup(0.05)
        assert observed == []
        assert preparations == [True]
    assert manager.dispose()
    assert preparations == [True, True]
    assert observed == ["deleted"]


def test_constructor_cleanup_before_callback_registration_cannot_resurrect_registry(
    manager, monkeypatch
):
    import tldw_chatbook.Agents.code_review_store as module

    observe = manager.observe_cleanup

    def settle_before_registration(snapshot):
        receipt = observe(snapshot)
        manager.close("s")
        assert manager.wait_for_cleanup(2)
        return receipt

    monkeypatch.setattr(manager, "observe_cleanup", settle_before_registration)
    finished = threading.Event()
    results = []

    def construct():
        try:
            results.append(ReviewStore(manager, "s"))
        finally:
            finished.set()

    worker = threading.Thread(target=construct, daemon=True)
    worker.start()
    assert finished.wait(2), "immediate cleanup callback deadlocked during construction"
    worker.join()
    assert len(results) == 1
    assert not module._TABS.get(manager)
    with pytest.raises(ReviewRefused, match="review_expired"):
        results[0].create(target_digest="b" * 64)


def test_failed_owned_precleanup_never_infers_deletion_from_root_receipt(
    store, manager, monkeypatch
):
    with monkeypatch.context() as fault:
        fault.setattr(
            ReviewStore,
            "_cleanup",
            lambda context: (_ for _ in ()).throw(OSError("private cleanup failure")),
        )
        store.close()
        assert manager.wait_for_cleanup(2)
    assert store._cleanup_receipt.outcome == "deleted"
    assert store.usage().retained_bytes == 19
    assert store.usage().retained_files == 1


def test_retire_unstarted_attempt_requires_exact_empty_binding(store):
    active = binding(store)
    with pytest.raises(ReviewRefused, match="run_binding_conflict"):
        store.retire_unstarted_attempt("other-run", active)
    with pytest.raises(ReviewRefused, match="run_binding_conflict"):
        store.retire_unstarted_attempt("default", binding(store, attempt="other"))
    publish(store)
    with pytest.raises(ReviewRefused, match="review_attempt_has_publications"):
        store.retire_unstarted_attempt("default", active)
    with store.lease(rid(store)) as context:
        assert store._bound(context, "default") == active
        assert context.streams[("R1", "a1")].revision == 1


def test_host_capture_interleaves_slot_appends_without_spending_revisions(
    store, bound_attempt
):
    barrier = threading.Barrier(2)
    record = payload(store)

    def submit(slot):
        attempt = "capture-" + slot
        run = f"run-{slot}-{attempt}"
        with bound_attempt(slot, attempt):
            first = store.publish(
                rid(store), expected_revision=0, submission_id="first", records=[record]
            )
            barrier.wait(timeout=5)
            store.capture_review_output(run, "raw " + slot, terminal=False)
            second = store.publish(
                rid(store),
                expected_revision=1,
                submission_id="second",
                records=[record | {"local_key": "second"}],
            )
            store.capture_review_output(run, "", terminal=True)
            store.settle_review_run(
                run, terminal_status="done", receipt_id="terminal-" + slot
            )
            return first, second

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            f.result(timeout=10)
            for f in [pool.submit(submit, slot) for slot in ("R1", "R2")]
        ]
    assert [[ack.submission_revision for ack in pair] for pair in results] == [
        [1, 2],
        [1, 2],
    ]
    with store.lease(rid(store)) as context:
        assert set(context.original_index) == {"R1-F1", "R1-F2", "R2-F1", "R2-F2"}
        for slot in ("R1", "R2"):
            stream = context.streams[(slot, "capture-" + slot)]
            assert stream.host_sequence == 3
            assert (
                store.read_file(context, stream.raw_chunks[0])
                == ("raw " + slot).encode()
            )
        assert store._totals(context)[1] == sum(
            f.size for f in context._files.values() if f.category == "report"
        )
    assert store.usage().reserved_bytes == 0


def test_raw_capture_aggregate_16_mib_limit_keeps_accepted_chunks(store):
    data = "x" * 900_000
    for index in range(19):
        run = f"aggregate-{index}"
        store.bind_attempt(run, binding(store, slot=f"R{index + 2}", attempt=run))
        store.capture_review_output(run, data, terminal=False)
    with store.lease(rid(store)) as context:
        streams = [
            stream
            for stream in context.streams.values()
            if stream.run_id.startswith("aggregate-")
        ]
        accepted = [stream for stream in streams if stream.raw_chunks]
        assert len(accepted) == 18
        assert streams[-1].capture_problem == "storage_quota_exceeded"
        assert all(
            store.read_file(context, stream.raw_chunks[0]) == data.encode()
            for stream in accepted
        )
        assert store._totals(context)[1] <= 16 * 1024 * 1024
        assert all(
            store._totals(context, f"R{i + 2}")[1] <= 1024 * 1024 for i in range(19)
        )
    assert store.usage().reserved_bytes == 0
