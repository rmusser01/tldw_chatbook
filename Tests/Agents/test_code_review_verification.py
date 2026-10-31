"""Blinded claims retain substance, never reviewer provenance."""

import json
from dataclasses import replace

from Tests.Agents.code_review_helpers import finding_payload
from tldw_chatbook.Agents.code_review_models import FindingRecord
from tldw_chatbook.Agents.code_review_source import SourceDocument, SourceManifest


def manifest():
    return SourceManifest(
        "review",
        "b" * 64,
        "root",
        0,
        "working_tree",
        ".",
        "",
        (),
        (SourceDocument("src-1", "a.py", "worktree", "a" * 64, 100, 10),),
        (),
        (),
    )


def test_only_exact_substantive_duplicates_share_units():
    from tldw_chatbook.Agents.code_review_verification import build_verification_units

    original = finding_payload()
    repeated = dict(
        original, severity="critical", local_key="different", title="Another title"
    )
    changed = dict(original, trigger="Call parse with a null value")
    records = tuple(
        {"original_id": key, "finding": finding}
        for key, finding in [
            ("R1-F1", original),
            ("R2-F1", repeated),
            ("R3-F1", changed),
        ]
    )
    units = build_verification_units(records, manifest())
    assert len(units.public_units) == 2
    assert units.original_to_unit["R1-F1"] == units.original_to_unit["R2-F1"]
    assert units.original_to_unit["R3-F1"] != units.original_to_unit["R1-F1"]
    public = json.dumps(units.public_units)
    for excluded in (
        "R1-F1",
        "R2-F1",
        "severity",
        "multiplicity",
        "local_key",
        "title",
    ):
        assert excluded not in public
    assert [u["verification_unit_id"] for u in units.public_units] == sorted(
        u["verification_unit_id"] for u in units.public_units
    )
    shuffled = build_verification_units(tuple(reversed(records)), manifest())
    assert sorted(
        json.dumps(u["claim"], sort_keys=True) for u in units.public_units
    ) == sorted(json.dumps(u["claim"], sort_keys=True) for u in shuffled.public_units)


def test_projection_retains_target_source_digest_and_uncertainty():
    from tldw_chatbook.Agents.code_review_verification import project_claim

    claim = project_claim(FindingRecord.model_validate(finding_payload()), manifest())
    assert claim["target_digest"] == "b" * 64
    assert claim["evidence"][0]["digest"] == "a" * 64
    assert claim["uncertainty"] == "No execution performed"
    assert claim != project_claim(
        FindingRecord.model_validate(finding_payload()),
        replace(manifest(), target_digest="c" * 64),
    )


import pytest

from Tests.Agents import code_review_helpers
from Tests.Agents.test_code_review_reports import finding
from tldw_chatbook.Agents.code_review_models import ReviewRefused
from tldw_chatbook.Agents.run_context import use_run_actor

pytestmark = pytest.mark.bootstrap_profile

review_case = code_review_helpers.review_case
source_case = code_review_helpers.source_case


def seed(c, count=3):
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        provider.service.submit(
            prepared.review_id,
            "findings",
            [
                finding(c, prepared, str(i), trigger="trigger " + str(i))
                for i in range(count)
            ],
            0,
            "seed",
        )
    return prepared


def test_batches_checkpoint_before_expansion_and_replay(review_case):
    c = review_case
    prepared = seed(c)
    with c.participant(prepared, role="verifier", slot="V1") as provider:
        service = provider.service
        page = service.next_verification_batch(
            prepared.review_id,
            remaining_input_tokens=10000,
            estimate=lambda text: len(text) // 4,
            max_result_chars=1024,
            cursor=None,
        )
        assert 0 < len(page.records) < 3
        assert len(page.model_dump_json()) <= 1024
        assert (
            service.next_verification_batch(
                prepared.review_id,
                remaining_input_tokens=10000,
                estimate=lambda text: len(text) // 4,
                max_result_chars=1024,
                cursor=None,
            )
            == page
        )
        row = page.records[0]
        disposition = {
            "schema_version": 1,
            "verification_unit_id": row["verification_unit_id"],
            "verdict": "confirmed",
            "reason": "source proves failure",
            "evidence": row["claim"]["evidence"],
        }
        ack = service.submit(
            prepared.review_id, "dispositions", [disposition], 0, "decision"
        )
        assert (
            service.submit(
                prepared.review_id, "dispositions", [disposition], 0, "decision"
            )
            == ack
        )
        checkpoint = service.verifier_checkpoint(prepared.review_id)
        assert checkpoint.submission_revision == 1
        assert ack.cursor in checkpoint.acknowledged_cursors
        with pytest.raises(ReviewRefused, match="disposition_conflict"):
            service.submit(
                prepared.review_id,
                "dispositions",
                [dict(disposition, verdict="refuted")],
                1,
                "changed",
            )
        service.settle_review_run(
            service.actor.run_id, terminal_status="done", receipt_id="terminal"
        )
        with pytest.raises(ReviewRefused, match="verification_attempt_closed"):
            service.submit(
                prepared.review_id, "dispositions", [disposition], 1, "after"
            )
    with use_run_actor(c.actor):
        expanded = c.primary.expand_dispositions(prepared.review_id)
    assert len(expanded) == 3
    assert sum(row["disposition"]["verdict"] == "confirmed" for row in expanded) == 1
    assert (
        sum(
            row["disposition"]["reason"] == "verification_incomplete"
            for row in expanded
        )
        == 2
    )


def test_transport_gap_preserves_original_and_model_gap_is_distinct(review_case):
    c = review_case
    prepared = seed(c, 1)
    with c.participant(prepared, role="verifier", slot="V1") as provider:
        service = provider.service
        page = service.next_verification_batch(
            prepared.review_id,
            remaining_input_tokens=10000,
            estimate=lambda text: len(text) // 4,
            max_result_chars=600,
            cursor=None,
        )
        assert not page.records and page.cursor is None
        assert page.gaps[0].reason == "review_record_too_large_for_tool_result"
    with use_run_actor(c.actor):
        expanded = c.primary.expand_dispositions(prepared.review_id)
    assert expanded[0]["original"]["trigger"] == "trigger 0"
    assert (
        expanded[0]["disposition"]["reason"]
        == "review_record_too_large_for_tool_result"
    )


def test_batch_rechecks_current_source_permission(review_case):
    c = review_case
    prepared = seed(c, 1)
    with c.participant(prepared, role="verifier", slot="V1") as provider:
        service = provider.service
        calls = []
        service.source_permission = lambda name, args: (
            calls.append((name, args)) or "permission_revoked"
        )
        with pytest.raises(ReviewRefused, match="permission_revoked"):
            service.next_verification_batch(
                prepared.review_id,
                remaining_input_tokens=10000,
                estimate=lambda text: len(text) // 4,
                max_result_chars=16000,
                cursor=None,
            )
        assert calls[0][0] == "fs_read"


@pytest.mark.parametrize("required_owner, scale", [(False, 2), (True, 20)])
def test_individually_oversized_claim_is_unresolved_without_truncation(
    review_case, required_owner, scale
):
    c = review_case
    prepared = seed(c, 1)
    with c.participant(prepared, role="verifier", slot="V1") as provider:
        provider.service.verifier_required_continuation = required_owner
        page = provider.service.next_verification_batch(
            prepared.review_id,
            remaining_input_tokens=100000 if required_owner else 1024,
            estimate=lambda text: len(text) * scale,
            max_result_chars=16000,
            cursor=None,
        )
        assert page.cursor is None
        assert page.gaps[0].reason == "claim_too_large"
    with use_run_actor(c.actor):
        expanded = c.primary.expand_dispositions(prepared.review_id)
    assert expanded[0]["original"]["trigger"] == "trigger 0"


def test_capacity_failure_is_request_gap_not_claim_size(review_case):
    c = review_case
    prepared = seed(c, 1)
    with c.participant(prepared, role="verifier", slot="V1") as provider:
        with pytest.raises(ReviewRefused, match="review_context_reduction_blocked"):
            provider.service.next_verification_batch(
                prepared.review_id,
                remaining_input_tokens=0,
                estimate=len,
                max_result_chars=16000,
                cursor=None,
            )
        provider.service.settle_review_run(
            provider.service.actor.run_id,
            terminal_status="error",
            receipt_id="terminal",
        )
    with use_run_actor(c.actor):
        expanded = c.primary.expand_dispositions(prepared.review_id)
    assert expanded[0]["disposition"]["reason"] == "verification_incomplete"


def test_exact_repeat_expansion_keeps_all_originals_and_requires_changed_trigger(
    review_case,
):
    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        provider.service.submit(
            prepared.review_id,
            "findings",
            [
                finding(c, prepared, "one"),
                finding(c, prepared, "two", severity="critical"),
                finding(c, prepared, "three", trigger="different trigger"),
            ],
            0,
            "originals",
        )
    with c.participant(prepared, role="verifier", slot="V1") as provider:
        page = provider.service.next_verification_batch(
            prepared.review_id,
            remaining_input_tokens=10000,
            estimate=lambda text: len(text) // 4,
            max_result_chars=16000,
            cursor=None,
        )
        assert len(page.records) == 2
        repeated = next(
            row
            for row in page.records
            if row["claim"]["trigger"] != "different trigger"
        )
        provider.service.submit(
            prepared.review_id,
            "dispositions",
            [
                {
                    "schema_version": 1,
                    "verification_unit_id": repeated["verification_unit_id"],
                    "verdict": "confirmed",
                    "reason": "captured evidence",
                    "evidence": repeated["claim"]["evidence"],
                }
            ],
            0,
            "verified",
        )
        assert not provider.service.verifier_checkpoint(
            prepared.review_id
        ).acknowledged_cursors
        provider.service.settle_review_run(
            provider.service.actor.run_id,
            terminal_status="cancelled",
            receipt_id="terminal",
        )
    with use_run_actor(c.actor):
        expanded = c.primary.expand_dispositions(prepared.review_id)
    assert {row["original_id"] for row in expanded} == {"R1-F1", "R1-F2", "R1-F3"}
    assert [row["disposition"]["verdict"] for row in expanded] == [
        "confirmed",
        "confirmed",
        "unresolved",
    ]


@pytest.mark.parametrize("limit", [0, 20000])
def test_native_ceiling_applies_even_when_ordinary_limit_is_unbounded(
    review_case, limit
):
    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        provider.service.submit(
            prepared.review_id,
            "findings",
            [
                finding(c, prepared, str(i), trigger=str(i) + "word " * 1000)
                for i in range(4)
            ],
            0,
            "originals",
        )
    with c.participant(prepared, role="verifier", slot="V1", limit=limit) as provider:
        page = provider.service.next_verification_batch(
            prepared.review_id,
            remaining_input_tokens=50000,
            estimate=lambda text: len(text) // 20,
            max_result_chars=limit,
            cursor=None,
        )
        assert len(page.model_dump_json()) <= 16000
        assert 0 < len(page.records) < 4


def test_empty_originals_return_explicit_zero_finding_coverage(review_case):
    c = review_case
    prepared = c.prepare()
    with c.participant(prepared, role="verifier", slot="V1") as provider:
        page = provider.service.next_verification_batch(
            prepared.review_id,
            remaining_input_tokens=10000,
            estimate=lambda text: len(text) // 4,
            max_result_chars=16000,
            cursor=None,
        )
        assert not page.records and page.cursor is None
        assert page.text == "Zero findings: no verification units."


def test_checkpoint_storage_failure_still_closes_native_verifier(
    review_case, monkeypatch
):
    c = review_case
    prepared = seed(c, 1)
    with c.participant(prepared, role="verifier", slot="V1") as provider:
        service = provider.service
        service.verifier_checkpoint(prepared.review_id)

        def unavailable(*args, **kwargs):
            raise ReviewRefused("review_storage_limit")

        monkeypatch.setattr(service.store, "write_verification_state", unavailable)
        service.settle_review_run(
            service.actor.run_id, terminal_status="error", receipt_id="native-error"
        )
        with pytest.raises(ReviewRefused, match="settled"):
            service.verifier_checkpoint(prepared.review_id)
    with use_run_actor(c.actor):
        expanded = c.primary.expand_dispositions(prepared.review_id)
    assert expanded[0]["disposition"]["reason"] == "verification_incomplete"


def test_public_batch_read_releases_store_lock_during_source_approval(review_case):
    from threading import Event, Thread

    c = review_case
    prepared = seed(c, 1)
    entered, release, acquired = Event(), Event(), Event()
    errors = []
    with c.participant(prepared, role="verifier", slot="V1") as provider:
        service = provider.service
        service.verifier_input_tokens = 10000
        service.verifier_estimate = lambda text: len(text) // 4

        def approve(*args):
            entered.set()
            assert release.wait(5)

        service.source_permission = approve

        def read():
            try:
                with use_run_actor(service.actor):
                    service.read(prepared.review_id, view="batch")
            except Exception as exc:  # noqa: BLE001 — assert worker errors in caller
                errors.append(exc)

        def probe():
            with service.store._tab.lock:
                acquired.set()

        reader = Thread(target=read)
        reader.start()
        probe_thread = Thread(target=probe)
        try:
            assert entered.wait(5), "public read never reached source approval"
            probe_thread.start()
            assert acquired.wait(1), "public read held store lock during approval"
        finally:
            release.set()
            reader.join(5)
            if probe_thread.ident is not None:
                probe_thread.join(5)
        assert not reader.is_alive() and not probe_thread.is_alive()
        assert not errors
