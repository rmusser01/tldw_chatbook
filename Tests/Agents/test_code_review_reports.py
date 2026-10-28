"""Original reports retain exact structured records and honest capture receipts."""

import json

import pytest

from Tests.Agents import code_review_helpers, test_code_review_dispatch
from tldw_chatbook.Agents.run_context import use_run_actor

review_case = code_review_helpers.review_case
source_case = code_review_helpers.source_case
dispatch_case = test_code_review_dispatch.dispatch_case


def coverage(c, prepared, *, empty=True):
    with c.store.lease(prepared.review_id) as context:
        return code_review_helpers.coverage_payload() | {
            "inspected_refs": list(context.sources),
            "empty_findings": empty,
        }


def summary(c, prepared):
    with use_run_actor(c.actor):
        return c.primary.read(prepared.review_id, "summary").records[0]


@pytest.mark.parametrize("status", ["done", "error", "cancelled", "stuck"])
def test_terminal_without_coverage_never_means_empty(review_case, status):
    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        run_id = provider.service.actor.run_id
        provider.service.capture_review_output(
            run_id, "Unstructured report", terminal=False
        )
        provider.service.capture_review_output(run_id, "", terminal=True)
        provider.service.settle_review_run(
            run_id, terminal_status=status, receipt_id="actual-terminal"
        )
    slot = summary(c, prepared)["slots"][0]
    assert slot["report_status"] == "incomplete"
    assert slot["empty_findings"] is False


@pytest.mark.parametrize("capture_first", [True, False])
def test_sealing_needs_coverage_terminal_and_capture_receipt(
    review_case, capture_first
):
    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        service, run_id = provider.service, provider.service.actor.run_id
        service.submit(
            prepared.review_id,
            "reviewer_complete",
            [coverage(c, prepared)],
            0,
            "complete",
        )
        if capture_first:
            service.capture_review_output(run_id, "", terminal=True)
        assert summary(c, prepared)["slots"][0]["report_status"] == "pending"
        service.settle_review_run(
            run_id, terminal_status="done", receipt_id="actual-terminal"
        )
        if not capture_first:
            assert summary(c, prepared)["slots"][0]["report_status"] == "incomplete"
            service.capture_review_output(run_id, "", terminal=True)
    slot = summary(c, prepared)["slots"][0]
    assert slot["report_status"] == "sealed"
    assert slot["empty_findings"] is True
    assert slot["terminal_receipt"] == "actual-terminal"
    assert slot["capture_receipt"]


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("inline", [True, False])
def test_actual_child_capture_precedes_fleet_and_run_summary_caps(
    dispatch_case, native, inline
):
    c = dispatch_case
    original = "UNIQUE-REPORT-CANARY " + ("original report line\n" * 1200)
    c.run(native=native, inline=inline, child_script=[original], limit=1024)
    with use_run_actor(c.actor):
        c.primary.result_limit = 16000
        envelope = c.primary.read(c.prepared.review_id, "summary").records[0]["slots"][
            0
        ]
        chunks, cursor = [], None
        while True:
            page = c.primary.read(
                c.prepared.review_id, "report", envelope["raw_ref"], cursor
            )
            chunks.append(page.text)
            cursor = page.cursor
            if cursor is None:
                break
        assert "".join(chunks) == original
    assert envelope["capture_status"] == "complete"
    assert envelope["report_status"] == "incomplete"
    assert envelope["actual_routes"] == (
        {"provider": "openai", "model": "review-model-0"},
    )
    child = next(
        row for row in c.db.list_runs("chat") if row["agent_kind"] == "subagent"
    )
    assert "UNIQUE-REPORT-CANARY" not in json.dumps(c.db.get_run(child["id"]))
    assert all(
        "UNIQUE-REPORT-CANARY" not in json.dumps(c.db.get_run(r["id"]))
        for r in c.db.list_runs("chat")
    )


def finding(c, prepared, key="one", **changes):
    with c.store.lease(prepared.review_id) as context:
        ref, source = next(iter(context.sources.items()))
    location = {"source_ref": ref, "start_line": 1, "end_line": 1}
    return code_review_helpers.finding_payload() | {
        "local_key": key,
        "locations": [location],
        "evidence": [location | {"kind": "source", "digest": source.digest}],
        **changes,
    }


def public_pages(c, prepared, *, view, document_ref=None, limit=1024):
    with use_run_actor(c.actor):
        c.primary.result_limit = limit
        cursor, pages = None, []
        for _ in range(100):
            result = c.provider.invoke(
                "code_review:code_review_read",
                {
                    "review_id": prepared.review_id,
                    "view": view,
                    "document_ref": document_ref,
                    "cursor": cursor,
                },
            )
            assert result.ok, result.error
            assert len(result.content) <= limit
            page = json.loads(result.content)
            assert page["revision"] >= 0 and "submission_revision" in page
            pages.append(page)
            if cursor:
                replay = c.provider.invoke(
                    "code_review:code_review_read",
                    {
                        "review_id": prepared.review_id,
                        "view": view,
                        "document_ref": document_ref,
                        "cursor": cursor,
                    },
                )
                assert replay.content == result.content
            cursor = page["cursor"]
            if cursor is None:
                return pages
        pytest.fail("cursor did not progress")


@pytest.mark.parametrize("corrected", [True, False])
def test_rejected_record_is_gap_until_same_submission_corrected(review_case, corrected):
    from tldw_chatbook.Agents.code_review_models import ReviewRefused

    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        service, run_id = provider.service, provider.service.actor.run_id
        with pytest.raises(ReviewRefused):
            service.submit(
                prepared.review_id,
                "findings",
                [finding(c, prepared, severity="invented")],
                0,
                "repair",
            )
        revision = 0
        if corrected:
            service.submit(
                prepared.review_id,
                "findings",
                [finding(c, prepared)],
                revision,
                "repair",
            )
            revision += 1
        service.submit(
            prepared.review_id,
            "reviewer_complete",
            [coverage(c, prepared, empty=not corrected)],
            revision,
            "complete",
        )
        service.capture_review_output(
            run_id, "Original inspection prose", terminal=True
        )
        service.settle_review_run(
            run_id, terminal_status="done", receipt_id="actual-terminal"
        )
    slot = summary(c, prepared)["slots"][0]
    assert slot["report_status"] == ("sealed" if corrected else "incomplete")
    assert bool(slot["rejected_submissions"]) is not corrected


@pytest.mark.parametrize("status", ["error", "cancelled", "stuck"])
def test_failed_roles_preserve_originals_and_reject_late_writes(review_case, status):
    from tldw_chatbook.Agents.code_review_models import ReviewRefused

    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        service, run_id = provider.service, provider.service.actor.run_id
        service.submit(prepared.review_id, "findings", [finding(c, prepared)], 0, "one")
        service.capture_review_output(run_id, "Original", terminal=True)
        service.settle_review_run(
            run_id, terminal_status=status, receipt_id="actual-terminal"
        )
        with pytest.raises(ReviewRefused, match="review_run_settled"):
            service.submit(
                prepared.review_id,
                "reviewer_complete",
                [coverage(c, prepared, empty=False)],
                1,
                "late",
            )
        with pytest.raises(ReviewRefused, match="review_capture_closed"):
            service.capture_review_output(run_id, "late", terminal=False)
    pages = public_pages(c, prepared, view="report", limit=2048)
    assert pages[0]["records"][0]["record"] == finding(c, prepared)
    assert pages[0]["records"][0]["report_status"] == "incomplete"


def test_complete_index_and_originals_pack_without_skipping_candidates(review_case):
    c = review_case
    prepared = c.prepare()
    originals = [
        finding(c, prepared, str(i), impact="long claim " * (300 if i == 2 else 1))
        for i in range(8)
    ]
    with c.participant(prepared) as provider:
        ack = provider.service.submit(
            prepared.review_id, "findings", originals, 0, "eight"
        )
    pages = public_pages(c, prepared, view="index")
    assert [row["original_id"] for page in pages for row in page["records"]] == list(
        ack.accepted_ids
    )
    pages = public_pages(c, prepared, view="report", limit=1400)
    rows = [row for page in pages for row in page["records"]]
    assert [row["record_ref"] for row in rows] == list(ack.accepted_ids)
    assert rows[2] == {"record_ref": "R1-F3", "gap": "model_read_record_too_large"}
    with c.store.lease(prepared.review_id) as context:
        assert (
            c.store.read_record(context, context.original_index["R1-F3"])
            == originals[2]
        )
    assert [row["record"] for row in rows if "record" in row] == originals[
        :2
    ] + originals[3:]


def test_raw_capture_preserves_whitespace_redacts_secrets_and_discloses_withholding(
    review_case,
):
    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        service, run_id = provider.service, provider.service.actor.run_id
        service.capture_review_output(run_id, "ordinary\n" * 2000, terminal=False)
        service.capture_review_output(
            run_id,
            "Bearer very-secret-token\n<think>private reasoning</think>\n",
            terminal=True,
        )
    slot = summary(c, prepared)["slots"][0]
    pages = public_pages(c, prepared, view="report", document_ref=slot["raw_ref"])
    text = "".join(p["text"] for p in pages)
    assert text.startswith("ordinary\n" * 2000)
    assert "very-secret-token" not in text and "private reasoning" not in text
    assert slot["capture_status"] == "withheld"


def test_capture_overflow_preserves_prior_chunks_and_combined_participant_charge(
    review_case,
):
    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        service, run_id = provider.service, provider.service.actor.run_id
        service.submit(prepared.review_id, "findings", [finding(c, prepared)], 0, "one")
        service.capture_review_output(run_id, "early original", terminal=False)
        service.capture_review_output(run_id, "x" * (1024 * 1024), terminal=False)
        service.capture_review_output(run_id, "", terminal=True)
        service.settle_review_run(
            run_id, terminal_status="done", receipt_id="actual-terminal"
        )
    slot = summary(c, prepared)["slots"][0]
    assert slot["capture_status"] == "storage_quota_exceeded"
    pages = public_pages(c, prepared, view="report", document_ref=slot["raw_ref"])
    assert "".join(p["text"] for p in pages) == "early original"
    with c.store.lease(prepared.review_id) as context:
        assert len(context.original_index) == 1
        assert c.store._totals(context, "R1")[1] < 1024 * 1024
        assert c.store.usage().reserved_bytes == 0


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("native", [True, False])
def test_threaded_children_capture_interleaves_independent_appends(
    dispatch_case, native
):
    import threading

    c = dispatch_case
    barrier = threading.Barrier(2)

    def submit(which):
        barrier.wait(timeout=10)
        data = dict(
            review_id=c.prepared.review_id,
            submission_id=str(which),
            expected_revision=which,
        )
        if which < 2:
            data.update(kind="findings", records=[finding(c, c.prepared, str(which))])
        else:
            data.update(
                kind="reviewer_complete", records=[coverage(c, c.prepared, empty=False)]
            )
        result = c.reply("code_review_submit", data)
        return (
            (result | {"content": f"raw turn {which}\n"})
            if native
            else f"raw turn {which}\n" + result
        )

    c.run(
        reviewers=2,
        native=native,
        child_script=[
            lambda: submit(0),
            lambda: submit(1),
            lambda: submit(2),
            "Final raw original",
        ],
    )
    with use_run_actor(c.actor):
        rows = c.primary.finding_index(c.prepared.review_id).records
    assert {r["original_id"] for r in rows} == {"R1-F1", "R1-F2", "R2-F1", "R2-F2"}
    assert all(r["report_status"] == "sealed" for r in rows)
    with c.store.lease(c.prepared.review_id) as context:
        for (slot, _), stream in context.streams.items():
            if slot == "primary":
                continue
            assert stream.revision == 3 and stream.host_sequence > 3
            raw = "".join(
                c.store.read_file(context, p).decode() for p in stream.raw_chunks
            )
            assert raw.count("Final raw original") == 1
            assert raw.count("raw turn 1") == 1
            assert len(stream.local_keys) == 2
            assert c.store._totals(context, slot)[1] == sum(
                f.size for f in context._files.values() if f.participant_id == slot
            )


def test_successful_terminal_still_rejects_new_completion(review_case):
    from tldw_chatbook.Agents.code_review_models import ReviewRefused

    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        service, run_id = provider.service, provider.service.actor.run_id
        service.capture_review_output(run_id, "Original", terminal=True)
        service.settle_review_run(run_id, terminal_status="done", receipt_id="terminal")
        with pytest.raises(ReviewRefused, match="review_run_settled"):
            service.submit(
                prepared.review_id,
                "reviewer_complete",
                [coverage(c, prepared)],
                0,
                "late",
            )
    assert summary(c, prepared)["slots"][0]["report_status"] == "incomplete"


def test_capture_disk_failure_and_missing_receipt_prevent_empty_seal(
    review_case, monkeypatch
):
    from tldw_chatbook.Agents.code_review_store import ReviewReservation

    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        service, run_id = provider.service, provider.service.actor.run_id
        service.submit(
            prepared.review_id,
            "reviewer_complete",
            [coverage(c, prepared)],
            0,
            "complete",
        )
        with monkeypatch.context() as patch:
            patch.setattr(
                ReviewReservation,
                "write",
                lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk failed")),
            )
            service.capture_review_output(run_id, "Original", terminal=False)
        service.capture_review_output(run_id, "", terminal=True)
        service.settle_review_run(run_id, terminal_status="done", receipt_id="")
    slot = summary(c, prepared)["slots"][0]
    assert slot["report_status"] == "incomplete" and not slot["empty_findings"]
    assert slot["capture_status"] == "capture_receipt_unavailable"


def test_late_capture_after_release_cannot_recreate_files(review_case):
    from tldw_chatbook.Agents.code_review_models import ReviewRefused

    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        service, run_id = provider.service, provider.service.actor.run_id
        service.capture_review_output(run_id, "early", terminal=False)
    with use_run_actor(c.actor):
        c.primary.release(prepared.review_id, prepared.revision)
    before = c.store.usage()
    with pytest.raises(ReviewRefused, match="review_expired|run_not_bound"):
        service.capture_review_output(run_id, "late", terminal=False)
    assert c.store.usage() == before


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("native", [True, False])
def test_accepted_preloop_failure_settles_report_without_originals(
    dispatch_case, monkeypatch, native
):
    from tldw_chatbook.Agents.agent_service import AgentService

    c = dispatch_case
    original = AgentService._fork_review_child

    def fail(self, service, **kwargs):
        original(self, service, **kwargs)
        raise RuntimeError("injected preloop failure")

    monkeypatch.setattr(AgentService, "_fork_review_child", fail)
    c.run(native=native)
    slot = summary(c, c.prepared)["slots"][0]
    assert slot["report_status"] == "incomplete"
    assert slot["terminal_status"] == "error"
    assert slot["terminal_receipt"]
    assert slot["empty_findings"] is False


def test_malformed_public_submission_leaves_format_gap(review_case):
    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        result = provider.invoke(
            "code_review:code_review_submit",
            {
                "review_id": prepared.review_id,
                "kind": "findings",
                "records": [],
                "submission_id": "bad",
                "expected_revision": 0,
                "unexpected": True,
            },
        )
        assert not result.ok
        service, run_id = provider.service, provider.service.actor.run_id
        service.submit(
            prepared.review_id,
            "reviewer_complete",
            [coverage(c, prepared)],
            0,
            "complete",
        )
        service.capture_review_output(run_id, "", terminal=True)
        service.settle_review_run(run_id, terminal_status="done", receipt_id="terminal")
    assert summary(c, prepared)["slots"][0]["report_status"] == "incomplete"


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("native", [True, False])
def test_actual_primary_pages_complete_index_at_live_result_limit(
    dispatch_case, native
):
    from Tests.Agents.test_code_review_dispatch import model_pages

    c = dispatch_case
    received = []

    def child_submit():
        return c.reply(
            "code_review_submit",
            {
                "review_id": c.prepared.review_id,
                "kind": "findings",
                "records": [finding(c, c.prepared, str(i)) for i in range(8)],
                "expected_revision": 0,
                "submission_id": "eight",
            },
        )

    def next_page():
        page = model_pages(c.calls[-1][1])[-1]
        received.extend(row["original_id"] for row in page["records"])
        assert len(json.dumps(page, ensure_ascii=False, separators=(",", ":"))) <= 1024
        if page["cursor"]:
            c.parent_script.insert(0, next_page)
            return c.reply(
                "code_review_read",
                {
                    "review_id": c.prepared.review_id,
                    "view": "index",
                    "cursor": page["cursor"],
                },
            )
        return "Index delivered"

    def parent(c):
        return [
            c.reply(
                "spawn_subagent",
                {"review_id": c.prepared.review_id, "review_slot_id": "R1"},
            ),
            c.reply("wait_agents", {}),
            c.reply(
                "code_review_read", {"review_id": c.prepared.review_id, "view": "index"}
            ),
            next_page,
        ]

    c.run(
        native=native,
        limit=1024,
        child_script=[child_submit, "Done"],
        parent_script=parent,
    )
    assert received == [f"R1-F{i}" for i in range(1, 9)]


@pytest.mark.bootstrap_profile
def test_existing_lossy_renderers_do_not_define_originals(dispatch_case):
    from tldw_chatbook.Agents.agent_models import RunBudget
    from tldw_chatbook.Agents.agent_service import (
        TRUNCATION_NOTICE,
        _safe_terminal_result,
    )

    c = dispatch_case
    original = "Full report line\n" * 1500
    c.run(reviewers=2, child_script=[original])
    fleet = c.runtime._fleet
    rendered = c.runtime._format_wait_result(
        fleet,
        [h.handle_id for h in fleet.snapshot()],
        RunBudget(max_tool_result_chars=500, max_subagent_result_chars=16),
        "",
    )
    assert TRUNCATION_NOTICE in rendered and len(rendered) <= 500
    assert len(_safe_terminal_result(original)) < len(original)
    with use_run_actor(c.actor):
        page = c.primary.read(c.prepared.review_id, "summary")
    captured = [row["slots"][0] for row in page.records if "raw_ref" in row["slots"][0]]
    assert len(captured) == 2
    for slot in captured:
        raw = public_pages(
            c, c.prepared, view="report", document_ref=slot["raw_ref"], limit=16000
        )
        assert "".join(p["text"] for p in raw) == original
    assert all(
        "Full report line" not in json.dumps(c.db.get_run(r["id"]))
        for r in c.db.list_runs("chat")
    )


@pytest.mark.bootstrap_profile
def test_lost_exact_terminal_trace_cannot_seal_valid_empty_report(
    dispatch_case, monkeypatch
):
    from tldw_chatbook.Agents.agent_models import AGENT_LIFECYCLE_INDEX_BASE

    c = dispatch_case
    original = c.db.insert_steps_at_indices

    def omit(run_id, steps):
        return original(
            run_id,
            [
                (i, value)
                for i, value in steps
                if run_id == "primary" or i >= AGENT_LIFECYCLE_INDEX_BASE
            ],
        )

    monkeypatch.setattr(c.db, "insert_steps_at_indices", omit)

    def complete():
        return c.reply(
            "code_review_submit",
            {
                "review_id": c.prepared.review_id,
                "kind": "reviewer_complete",
                "records": [coverage(c, c.prepared)],
                "expected_revision": 0,
                "submission_id": "complete",
            },
        )

    c.run(child_script=[complete, "Done"])
    slot = summary(c, c.prepared)["slots"][0]
    assert slot["terminal_status"] == "done"
    assert slot["terminal_receipt"] == ""
    assert slot["report_status"] == "incomplete" and not slot["empty_findings"]


@pytest.mark.bootstrap_profile
def test_native_cancellation_closes_late_model_capture_without_claiming_drain(
    dispatch_case,
):
    from tldw_chatbook.Agents.run_context import current_run_actor

    c = dispatch_case

    def first():
        return c.reply(
            "code_review_submit",
            {
                "review_id": c.prepared.review_id,
                "kind": "findings",
                "records": [finding(c, c.prepared)],
                "expected_revision": 0,
                "submission_id": "one",
            },
        )

    def cancelled():
        run_id = current_run_actor().run_id
        c.runtime._set_terminal_status(run_id, "cancelled")
        c.runtime._settle_review_capture(c.primary, run_id, close_capture=False)
        return "LATE-RAW-CANARY"

    c.run(child_script=[first, cancelled])
    slot = summary(c, c.prepared)["slots"][0]
    assert (
        slot["terminal_status"] == "cancelled" and slot["report_status"] == "incomplete"
    )
    assert slot["capture_status"] == "pending"
    raw = public_pages(c, c.prepared, view="report", document_ref=slot["raw_ref"])
    assert "LATE-RAW-CANARY" not in "".join(p["text"] for p in raw)
    with c.store.lease(c.prepared.review_id) as context:
        assert len(context.original_index) == 1


def test_retry_keeps_prior_raw_and_original_ids_without_reopening_old_attempt(
    review_case,
):
    from tldw_chatbook.Agents.code_review_models import ReviewRefused

    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        old_service, run_id = provider.service, provider.service.actor.run_id
        old_service.submit(
            prepared.review_id, "findings", [finding(c, prepared)], 0, "one"
        )
        old_service.capture_review_output(run_id, "old original", terminal=True)
        old_service.settle_review_run(
            run_id, terminal_status="error", receipt_id="old-terminal"
        )
        with c.store.lease(prepared.review_id) as context:
            old_binding = c.store._bound(context, run_id)
        new_binding = old_binding.model_copy(update={"attempt_id": "retry"})
        c.store.bind_attempt("new-run", new_binding)
        from tldw_chatbook.Agents.run_context import CurrentRunActor

        actor = CurrentRunActor("subagent", "new-run", "primary")
        service = old_service.for_run(actor)
        with use_run_actor(actor):
            service.submit(
                prepared.review_id, "findings", [finding(c, prepared)], 0, "one"
            )
            service.capture_review_output("new-run", "new original", terminal=True)
        with pytest.raises(ReviewRefused, match="stale_attempt"):
            old_service.capture_review_output(run_id, "late", terminal=False)
    with use_run_actor(c.actor):
        records = c.primary.finding_index(prepared.review_id).records
        slots = [
            r["slots"][0] for r in c.primary.read(prepared.review_id, "summary").records
        ]
    assert [r["original_id"] for r in records] == ["R1-F1", "R1-F2"]
    assert len({s["raw_ref"] for s in slots if s["slot_id"] == "R1"}) == 2


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("inline", [True, False])
def test_review_live_callback_omits_bodies_but_preserves_identity(
    dispatch_case, native, inline
):
    from dataclasses import asdict
    from hashlib import sha256

    from tldw_chatbook.Agents.agent_models import STEP_MODEL, STEP_TOOL_RESULT
    from tldw_chatbook.Chat.console_agent_bridge import (
        AgentLiveStep,
        ConsoleAgentBridge,
    )

    c = dispatch_case
    source_body = "SOURCE-BODY-CALLBACK-CANARY"
    report_body = "REPORT-BODY-CALLBACK-CANARY"
    secret = "review-callback-secret-token"
    original = f"{report_body}\nBearer {secret}\n" + "full permitted report\n" * 40
    c.file.write_text(source_body + "\n")
    observed = []
    live = []

    def callback(step, agent_kind, run_id):
        observed.append((step, agent_kind, run_id))
        live.append(
            AgentLiveStep(step.kind, ConsoleAgentBridge._summarize(step), agent_kind)
        )

    def read_source():
        with c.store.lease(c.prepared.review_id) as context:
            ref = next(
                ref
                for ref, source in context.sources.items()
                if source.digest == sha256((source_body + "\n").encode()).hexdigest()
            )
        return c.reply(
            "code_review_read",
            {
                "review_id": c.prepared.review_id,
                "view": "source",
                "document_ref": ref,
            },
        )

    c.run(
        native=native,
        inline=inline,
        child_script=[read_source, original],
        on_step=callback,
    )
    child_steps = [step for step, kind, _ in observed if kind == "subagent"]
    assert child_steps and all(step.created_at for step in child_steps)
    assert any(step.kind == STEP_MODEL for step in child_steps)
    assert any(
        step.kind == STEP_TOOL_RESULT and step.tool_name == "code_review_read"
        for step in child_steps
    )
    # Prove the actual tool read returned source bytes to the child model.
    assert any(
        source_body in json.dumps(kwargs)
        for actor, kwargs in c.calls
        if actor.kind == "subagent"
    )
    projected = json.dumps(
        [asdict(step) for step in child_steps] + [asdict(step) for step in live]
    )
    assert all(canary not in projected for canary in (source_body, report_body, secret))
    assert all(
        step.summary == step.result == "" and step.args is None for step in child_steps
    )
    assert all(step.field_states["payload"] == "omitted" for step in child_steps)
    # Ordinary primary callbacks still receive their original response content.
    assert any(
        step.kind == STEP_MODEL and step.summary == "finished"
        for step, kind, _ in observed
        if kind == "primary"
    )
    slot = summary(c, c.prepared)["slots"][0]
    pages = public_pages(c, c.prepared, view="report", document_ref=slot["raw_ref"])
    captured = "".join(page["text"] for page in pages)
    assert captured.endswith("full permitted report\n" * 40)
    assert report_body in captured and secret not in captured
    assert source_body not in captured


@pytest.mark.parametrize("view", ["raw", "report", "index", "summary"])
@pytest.mark.parametrize("revocation", ["guard", "exclusion", "permission"])
def test_model_reports_revalidate_current_authority(review_case, view, revocation):
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace

    c = review_case
    allowed, excluded = [True], []
    c.primary.source_authority = replace(
        c.authority,
        admitted=replace(
            c.authority.admitted,
            guard=lambda write: allowed[0] and not write,
            exclusions_provider=lambda: tuple(excluded),
        ),
    )
    c.authority = c.primary.source_authority
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        provider.service.submit(
            prepared.review_id, "findings", [finding(c, prepared)], 0, "finding"
        )
        provider.service.capture_review_output(
            provider.service.actor.run_id,
            "SENSITIVE_REPORT_CANARY " * 2000,
            terminal=True,
        )
    with use_run_actor(c.actor):
        raw_ref = c.primary.read(prepared.review_id, "summary").records[0]["slots"][0][
            "raw_ref"
        ]
        args = {
            "review_id": prepared.review_id,
            "view": "report" if view == "raw" else view,
        }
        if view == "raw":
            args["document_ref"] = raw_ref
        first = c.provider.invoke("code_review:code_review_read", args)
        assert first.ok
        if view == "raw":
            import json

            args["cursor"] = json.loads(first.content)["cursor"]
            assert args["cursor"]
            assert c.provider.invoke("code_review:code_review_read", args).ok

        def permission(name, parameters):
            # A real competing owner must acquire the lock while policy is checked.
            def probe():
                acquired = c.store._tab.lock.acquire(timeout=0.5)
                if acquired:
                    c.store._tab.lock.release()
                return acquired

            with ThreadPoolExecutor(1) as pool:
                assert pool.submit(probe).result(1)
            return "source_permission_deny" if revocation == "permission" else None

        c.primary.source_permission = permission
        allowed[0] = revocation != "guard"
        if revocation == "exclusion":
            excluded.append(c.file)
        result = c.provider.invoke("code_review:code_review_read", args)
    assert not result.ok
    assert result.error in {
        "source_authority_unavailable",
        "source_path_unavailable",
        "source_permission_deny",
    }
    assert "SENSITIVE_REPORT_CANARY" not in str(result)


@pytest.mark.parametrize(
    "case",
    [
        "complete",
        "reasoned_unresolved",
        "premature",
        "missing",
        "no_intent",
        "no_capture",
        "error",
        "cancelled",
    ],
)
def test_verifier_seal_requires_own_complete_accounting(review_case, case):
    from Tests.Agents.test_code_review_verification import seed
    from tldw_chatbook.Agents.code_review_models import ReviewRefused

    c = review_case
    prepared = seed(c, count=2)
    with c.participant(prepared, role="verifier", slot="V1") as provider:
        service, run_id = provider.service, provider.service.actor.run_id
        page = service.next_verification_batch(
            prepared.review_id,
            remaining_input_tokens=10000,
            estimate=lambda text: len(text) // 4,
            max_result_chars=16000,
            cursor=None,
        )
        if case == "premature":
            with pytest.raises(ReviewRefused, match="verification_incomplete"):
                service.submit(
                    prepared.review_id, "verifier_complete", [], 0, "complete"
                )
        decisions = [
            {
                "schema_version": 1,
                "verification_unit_id": row["verification_unit_id"],
                "verdict": "unresolved"
                if case == "reasoned_unresolved"
                else "confirmed",
                "reason": "Source leaves remote caller unknown"
                if case == "reasoned_unresolved"
                else "Source proves failure",
                "evidence": row["claim"]["evidence"],
                "next_check": "Inspect remote caller"
                if case == "reasoned_unresolved"
                else None,
            }
            for row in page.records
        ]
        assert len(decisions) == 2
        if case == "missing":
            decisions.pop()
        service.submit(prepared.review_id, "dispositions", decisions, 0, "decisions")
        if case == "missing":
            with pytest.raises(ReviewRefused, match="verification_incomplete"):
                service.submit(
                    prepared.review_id, "verifier_complete", [], 1, "complete"
                )
        elif case != "no_intent":
            service.submit(prepared.review_id, "verifier_complete", [], 1, "complete")
        if case != "no_capture":
            service.capture_review_output(run_id, "Verifier report", terminal=True)
        service.settle_review_run(
            run_id,
            terminal_status=case if case in {"error", "cancelled"} else "done",
            receipt_id="native-terminal",
        )
        if case == "missing":
            with c.store.lease(prepared.review_id) as context:
                stream = context.streams[("V1", "attempt")]
                assert not stream.completion_requested
                assert len(stream.verification.dispositions) == 2
                assert any(
                    row["reason"] == "verification_incomplete"
                    for row in stream.verification.dispositions.values()
                )
    with use_run_actor(c.actor):
        envelope = next(
            slot
            for row in c.primary.read(prepared.review_id, "summary").records
            for slot in row["slots"]
            if slot["slot_id"] == "V1"
        )
    complete = case in {"complete", "reasoned_unresolved", "premature"}
    assert envelope["report_status"] == ("sealed" if complete else "incomplete")
    assert bool(envelope["seal_receipt"]) == complete
