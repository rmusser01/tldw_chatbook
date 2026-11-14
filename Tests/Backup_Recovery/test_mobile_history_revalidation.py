"""Native history revalidation retains first-parse authority and four seal hashes."""

import sqlite3
import sys
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from functools import partial
from threading import Event

import pytest

from Tests.Backup_Recovery.test_mobile_captured_sqlite import _fingerprints
from Tests.Backup_Recovery.test_mobile_native_operation import captured_operation
from Tests.Backup_Recovery.test_mobile_owning_transactions import _small_native_seed


class _HistoryInputsReady(Exception):
    """Stop the real prerequisite planner before its first history qualification."""


@contextmanager
def _history_case(
    tmp_path, monkeypatch, *, framed=True, forged_document=False, archive_holder=None
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Backup_Recovery import (
        mobile_captured_sqlite,
        mobile_runs_history,
    )
    from tldw_chatbook.Backup_Recovery.mobile_runs_preflight import _plan_runs

    monkeypatch.setattr(
        mobile_consolidation_fixture,
        "_seed",
        partial(_small_native_seed, source_case="history"),
    )
    with captured_operation(tmp_path, monkeypatch) as captured:
        before = _fingerprints(tmp_path / "profile")
        with captured.journal._locked(exclusive=False) as fd:
            events = tuple(captured.journal._records(fd))
        reads = None
        connections = []
        try:
            with captured.sqlite_reads(
                password=b"test-private-mobile-rollback", cancel=Event()
            ) as reads:
                connections = [member[0] for member in reads._connections.values()]
                inputs = []

                def take_inputs(actual, runs, identities, *, _read_budget):
                    assert actual is reads and _read_budget is reads.read_budget
                    inputs.append((runs, identities))
                    raise _HistoryInputsReady

                # Production supplies the real SQL rows and identity reservations;
                # the component ends before unrelated complete-plan serialization.
                with monkeypatch.context() as selection:
                    selection.setattr(
                        mobile_runs_history, "_qualify_history_census", take_inputs
                    )
                    with pytest.raises(_HistoryInputsReady):
                        _plan_runs(reads, _read_budget=reads.read_budget)
                ((runs, identities),) = inputs
                archive = reads._source_archive
                with monkeypatch.context() as substitution:
                    if forged_document:
                        partition, document, sealed = archive
                        substitution.setattr(
                            reads,
                            "_source_archive",
                            (
                                partition,
                                document.model_copy(update={"captured_at": "forged"}),
                                sealed,
                            ),
                        )
                    if archive_holder is not None:
                        partition, document, sealed = archive
                        copied = (partition, document.model_copy(), sealed)
                        substitution.setattr(
                            reads,
                            "_source_archive",
                            list(copied) if archive_holder == "list" else copied,
                        )
                    frame = (
                        mobile_captured_sqlite._plan_row_data(reads, reads.read_budget)
                        if framed
                        else nullcontext()
                    )
                    with frame:
                        yield reads, runs, identities
        finally:
            assert reads is not None and not reads.active and reads.resources == []
            for connection in connections:
                with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                    connection.execute("SELECT 1")
            assert _fingerprints(tmp_path / "profile") == before
            with captured.journal._locked(exclusive=False) as fd:
                assert tuple(captured.journal._records(fd)) == events
                assert not any(row.event == "mobile_owner_opened" for row in events)


@contextmanager
def _history_calls(
    reads, monkeypatch, *, after_first=None, after_payload=None, after_late_hash=None
):
    from tldw_chatbook.Backup_Recovery import archive_reader, mobile_runs_history

    sealed = reads._source_archive[2]
    verify = archive_reader.verify_sealed
    native_hash = archive_reader._hash
    regular = archive_reader._regular
    qualify_code = mobile_runs_history._qualify_history_census.__code__
    calls = {"generic": [], "hashes": [], "payload_closed": False, "late_opens": []}

    def checked(seal, cancel=None, *, _custody=None):
        if seal is sealed:
            calls["generic"].append((cancel, _custody))
        result = verify(seal, cancel, _custody=_custody)
        if seal is sealed and len(calls["generic"]) == 1 and after_first is not None:
            after_first()
        return result

    def hashed(path, cancel, *, _custody=None, _read_budget=None):
        before = None if _read_budget is None else _read_budget._remaining
        result = native_hash(path, cancel, _custody=_custody, _read_budget=_read_budget)
        if path == sealed.path:
            calls["hashes"].append((cancel, _custody, _read_budget, before))
            if _read_budget is not None:
                assert _read_budget._remaining < before
                if len(calls["hashes"]) == 3 and after_late_hash is not None:
                    after_late_hash()
        return result

    @contextmanager
    def opened(path, **kwargs):
        # contextlib.__enter__ resumes this generator from the actual caller.
        payload = sys._getframe(2).f_code is qualify_code and path == sealed.path
        if path == sealed.path and calls["payload_closed"]:
            calls["late_opens"].append(path)
        with regular(path, **kwargs) as stream:
            yield stream
        if payload:
            assert stream.closed
            assert not calls["payload_closed"]
            calls["payload_closed"] = True
            if after_payload is not None:
                after_payload()

    with monkeypatch.context() as observation:
        observation.setattr(archive_reader, "verify_sealed", checked)
        observation.setattr(archive_reader, "_hash", hashed)
        observation.setattr(archive_reader, "_regular", opened)
        yield calls


@pytest.mark.parametrize("framed", [True, False], ids=["native", "unframed"])
def test_history_revalidation_keeps_four_hashes_without_reparsing_native_data(
    tmp_path, monkeypatch, framed
):
    from tldw_chatbook.Backup_Recovery.mobile_runs_history import (
        _qualify_history_census,
    )

    with _history_case(tmp_path, monkeypatch, framed=framed) as (
        reads,
        runs,
        identities,
    ):
        budget = reads.read_budget
        before = budget._remaining
        with _history_calls(reads, monkeypatch) as calls:
            proofs, blockers, members = _qualify_history_census(
                reads, runs, identities, _read_budget=budget
            )
        assert proofs and members and not blockers
        assert any(proof.run_ids for proof in proofs)
        assert len(calls["generic"]) == (1 if framed else 2)
        assert len(calls["hashes"]) == 4
        assert all(
            cancel is reads.cancel and custody is reads.custody
            for cancel, custody in calls["generic"]
        )
        assert all(
            cancel is reads.cancel and custody is reads.custody
            for cancel, custody, _, _ in calls["hashes"]
        )
        assert all(counter is None for _, _, counter, _ in calls["hashes"][:2])
        if framed:
            assert all(counter is budget for _, _, counter, _ in calls["hashes"][2:])
        else:
            assert all(counter is None for _, _, counter, _ in calls["hashes"])
        assert reads.read_budget is budget and 0 <= budget._remaining < before


def test_pre_frame_forged_document_fails_first_verification_before_members(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import mobile_runs_history

    with _history_case(tmp_path, monkeypatch, forged_document=True) as (
        reads,
        runs,
        identities,
    ):

        def no_members(*args, **kwargs):
            pytest.fail("forged pre-frame document reached native history members")

        with (
            monkeypatch.context() as boundary,
            _history_calls(reads, monkeypatch) as calls,
        ):
            boundary.setattr(mobile_runs_history, "_members", no_members)
            with pytest.raises(
                ValueError, match="^mobile_runs_history_capture_changed$"
            ):
                mobile_runs_history._qualify_history_census(
                    reads, runs, identities, _read_budget=reads.read_budget
                )
        assert len(calls["generic"]) == 1
        assert len(calls["hashes"]) == 2
        assert not calls["payload_closed"]


@pytest.mark.parametrize(
    "changed",
    [
        "archive_tuple",
        "source",
        "owner",
        "marker",
        "budget",
        "partition",
        "encrypted_plaintext",
        "encrypted_manifest",
    ],
)
def test_late_native_history_binding_loss_refuses_inside_qualifier(
    tmp_path, monkeypatch, changed
):
    from tldw_chatbook.Backup_Recovery import (
        mobile_captured_sqlite,
        mobile_runs_history,
    )

    with _history_case(tmp_path, monkeypatch) as (reads, runs, identities):
        budget = reads.read_budget
        archive = reads._source_archive
        partition, _, sealed = archive
        owner = mobile_captured_sqlite._row_data(reads, budget)
        assert owner is not None
        with ExitStack() as restore, monkeypatch.context() as altered:
            injected = []

            def change_after_payload():
                if changed == "archive_tuple":
                    altered.setattr(reads, "_source_archive", (*archive,))
                elif changed == "source":
                    altered.setattr(
                        reads.captured, "source", replace(reads.captured.source)
                    )
                elif changed == "owner":
                    altered.setattr(reads, "_row_data_owner", None)
                elif changed == "marker":
                    token = mobile_captured_sqlite._row_data_marker.set(None)
                    restore.callback(
                        mobile_captured_sqlite._row_data_marker.reset, token
                    )
                elif changed == "budget":
                    altered.setattr(reads, "read_budget", None)
                elif changed == "partition":
                    altered.setattr(partition, "review_only", True)
                else:
                    encrypted = sealed.encrypted_source
                    field = (
                        "plaintext_digest"
                        if changed == "encrypted_plaintext"
                        else "manifest_digest"
                    )
                    previous = getattr(encrypted, field)
                    object.__setattr__(encrypted, field, "0" * 64)
                    restore.callback(object.__setattr__, encrypted, field, previous)
                injected.append(changed)

            with (
                _history_calls(
                    reads, monkeypatch, after_payload=change_after_payload
                ) as calls,
                pytest.raises(ValueError, match="mobile_.*(changed|inactive|retired)"),
            ):
                # The refusal must occur here, while the real issuer is active;
                # its eventual exit check cannot make this assertion pass.
                mobile_runs_history._qualify_history_census(
                    reads, runs, identities, _read_budget=budget
                )
            assert injected == [changed] and calls["payload_closed"]
            assert len(calls["generic"]) == 1
            assert len(calls["hashes"]) == 2
            assert calls["late_opens"] == []
        assert mobile_captured_sqlite._row_data(reads, budget) is owner


def test_late_actual_sealed_byte_mutation_refuses_before_owner_writes(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery.mobile_runs_history import (
        _qualify_history_census,
    )

    with _history_case(tmp_path, monkeypatch) as (reads, runs, identities):
        sealed = reads._source_archive[2]
        original = sealed.path.read_bytes()
        mutated = []

        def corrupt_after_payload():
            with sealed.path.open("r+b") as stream:
                stream.seek(-1, 2)
                byte = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([byte[0] ^ 1]))
            mutated.append(sealed.path)

        try:
            with (
                _history_calls(
                    reads, monkeypatch, after_payload=corrupt_after_payload
                ) as calls,
                pytest.raises(ValueError, match="^sealed_changed$"),
            ):
                _qualify_history_census(
                    reads, runs, identities, _read_budget=reads.read_budget
                )
            assert mutated == [sealed.path] and calls["payload_closed"]
            assert len(calls["generic"]) == 1
            assert len(calls["hashes"]) == 3
        finally:
            sealed.path.write_bytes(original)
        assert sealed.path.read_bytes() == original


def test_late_native_hash_exhaustion_refuses_before_open_or_buffer_allocation(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import mobile_runs_history
    from tldw_chatbook.Backup_Recovery.limits import _TraceReadLimit

    exhausted, allocations = [], []
    with (
        pytest.raises(_TraceReadLimit),
        _history_case(tmp_path, monkeypatch) as (reads, runs, identities),
    ):
        budget = reads.read_budget
        sealed = reads._source_archive[2]
        sha256 = mobile_runs_history.hashlib.sha256
        with monkeypatch.context() as allocation:

            def no_manifest_allocation(data=b"", *args, **kwargs):
                if data is sealed.manifest_bytes:
                    allocations.append(data)
                    pytest.fail("exhausted native proof allocated a manifest digest")
                return sha256(data, *args, **kwargs)

            def consume_original_allowance():
                assert budget is reads.read_budget and budget._remaining > 0
                budget.reserve(budget._remaining)
                exhausted.append(budget)
                allocation.setattr(
                    mobile_runs_history.hashlib, "sha256", no_manifest_allocation
                )

            with _history_calls(
                reads, monkeypatch, after_payload=consume_original_allowance
            ) as calls:
                try:
                    mobile_runs_history._qualify_history_census(
                        reads, runs, identities, _read_budget=budget
                    )
                except _TraceReadLimit:
                    assert exhausted == [budget] and not allocations
                    assert calls["payload_closed"]
                    assert calls["late_opens"] == []
                    assert budget is reads.read_budget and budget._remaining == 0
                    raise
                pytest.fail(
                    "exhausted original native counter admitted late verification"
                )
    assert exhausted and not allocations


@pytest.mark.parametrize("when", ["first-verification", "first-late-hash"])
def test_native_history_owner_loss_cannot_escape_its_next_boundary(
    tmp_path, monkeypatch, when
):
    from tldw_chatbook.Backup_Recovery import (
        mobile_captured_sqlite,
        mobile_runs_history,
    )

    with _history_case(tmp_path, monkeypatch) as (reads, runs, identities):
        budget = reads.read_budget
        owner = mobile_captured_sqlite._row_data(reads, budget)
        changed = []
        with monkeypatch.context() as alteration:

            def lose_owner():
                alteration.setattr(reads, "_row_data_owner", None)
                changed.append(when)

            def no_members(*args, **kwargs):
                pytest.fail(
                    "native owner loss after first verification reached members"
                )

            if when == "first-verification":
                alteration.setattr(mobile_runs_history, "_members", no_members)
            hooks = {
                "after_first"
                if when == "first-verification"
                else "after_late_hash": lose_owner
            }
            with (
                _history_calls(reads, monkeypatch, **hooks) as calls,
                pytest.raises(ValueError, match="mobile_.*changed"),
            ):
                mobile_runs_history._qualify_history_census(
                    reads, runs, identities, _read_budget=budget
                )
            assert changed == [when]
            assert len(calls["generic"]) == 1
            assert len(calls["hashes"]) == (2 if when == "first-verification" else 3)
        assert mobile_captured_sqlite._row_data(reads, budget) is owner


def test_native_history_cancellation_after_payload_precedes_late_hash_io(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery.mobile_runs_history import (
        _qualify_history_census,
    )

    with _history_case(tmp_path, monkeypatch) as (reads, runs, identities):
        cancel = reads.cancel
        try:
            with (
                _history_calls(reads, monkeypatch, after_payload=cancel.set) as calls,
                pytest.raises(InterruptedError, match="^cancelled$"),
            ):
                _qualify_history_census(
                    reads, runs, identities, _read_budget=reads.read_budget
                )
            assert calls["payload_closed"] and cancel.is_set()
            assert len(calls["generic"]) == 1 and len(calls["hashes"]) == 2
            assert calls["late_opens"] == []
        finally:
            cancel.clear()


@pytest.mark.parametrize(
    "mutable", [True, False], ids=["mutable-list", "copied-immutable"]
)
def test_pre_frame_history_holder_requires_native_immutable_tuple(
    tmp_path, monkeypatch, mutable
):
    from tldw_chatbook.Backup_Recovery import mobile_runs_history

    with _history_case(
        tmp_path, monkeypatch, archive_holder="list" if mutable else "copy"
    ) as (reads, runs, identities):

        def no_members(*args, **kwargs):
            pytest.fail("mutable pre-frame archive holder reached native members")

        with (
            monkeypatch.context() as boundary,
            _history_calls(reads, monkeypatch) as calls,
        ):
            if mutable:
                boundary.setattr(mobile_runs_history, "_members", no_members)
                with pytest.raises(
                    ValueError, match="^mobile_runs_history_capture_changed$"
                ):
                    mobile_runs_history._qualify_history_census(
                        reads, runs, identities, _read_budget=reads.read_budget
                    )
            else:
                proofs, blockers, members = mobile_runs_history._qualify_history_census(
                    reads, runs, identities, _read_budget=reads.read_budget
                )
                assert proofs and members and not blockers
        assert len(calls["generic"]) == 1
        assert len(calls["hashes"]) == (2 if mutable else 4)
