"""Scripted model output through native Console authority, tools and reports.

Only external model text is scripted. Findings and observations come back through
real provider preparation, child dispatch, report publication and native UI.
"""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from Tests.Chat.test_console_agent_swap import _fence
from Tests.Chat.test_console_code_review_continuation import (
    scripted_review_model_capacity,  # noqa: F401 - imported pytest fixture
)
from Tests.Chat.test_console_code_review_continuation import (
    test_default_two_child_schedule as run_console,
)
from tldw_chatbook import config
from tldw_chatbook.Agents.run_context import use_run_actor
from tldw_chatbook.Chat.console_prepared_request import PreparedProviderRequest

pytestmark = pytest.mark.bootstrap_profile


async def _settled(controller, session, service, review_id, *, timeout=20.0):
    """Wait until every launched child is terminal and wake delivery is idle.

    The schedule driver stops once the expected launches exist; the last child
    (and the wake its settlement triggers) can still be running at that point.
    """
    import asyncio

    conversation_id = session.persisted_conversation_id
    quiet = 0
    for _ in range(int(timeout / 0.03)):
        with service.store.lease(review_id) as context:
            terminal = all(
                context.streams[
                    (claim.slot.slot_id, claim.binding.attempt_id)
                ].terminal_status
                for claim in context.launch_claims.values()
            )
        idle = (
            not controller.fleet_wake.has_pending(conversation_id)
            and session.id not in controller.fleet_wake.delivering_session_ids()
        )
        quiet = quiet + 1 if terminal and idle else 0
        if quiet >= 3:
            return
        await asyncio.sleep(0.03)


@dataclass
class ReviewRunObservation:
    reviewer_run_ids: set[str] = field(default_factory=set)
    verifier_run_id: str | None = None
    original_ids: set[str] = field(default_factory=set)
    final_dispositions: dict = field(default_factory=dict)
    launches_per_primary_turn: list[int] = field(default_factory=list)
    chain_ids: set[str] = field(default_factory=set)
    original_reports_presented_before_verifier: bool = False
    prepared_calls: list = field(default_factory=list)
    guide_stages: list[str] = field(default_factory=list)
    batches: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)
    source: str = (
        Path(__file__).parents[2]
        / "Docs/superpowers/qa/2026-10-18-multi-model-code-review/seeded_average.py"
    ).read_text()

    def observe_request(self, actor, request):
        if actor.kind == "primary" and request:
            content = request[-1].get("content", "")
            marker = "Tool result for code_review_guide:"
            if isinstance(content, str) and marker in content:
                guide = json.loads(content.split(marker, 1)[1])
                assert guide["schema_version"] == 1 and guide["text"]
                assert "diagnostic" not in guide
                self.guide_stages.append(guide["stage"])
        if actor.kind == "subagent":
            assert isinstance(request, PreparedProviderRequest)
            self.prepared_calls.append((actor.run_id, request.provider, request.model))

    @staticmethod
    def tool_page(messages):
        content = messages[-1]["content"]
        assert "Tool result for code_review_read:" in content, content
        return json.loads(content.split("Tool result for code_review_read:", 1)[1])

    def child(self, actor, round_number, service, review_id, messages):
        with service.store.lease(review_id) as context:
            binding = service.store._tab.runs[actor.run_id]
            ref, source = next(iter(context.sources.items()))
            if binding.role == "verifier" and round_number == 0:
                self.original_reports_presented_before_verifier = all(
                    (slot.slot_id, context.active_attempts[slot.slot_id].attempt_id)
                    in context.presented_attempts
                    for slot in service.slots(context)
                    if slot.role == "reviewer"
                )
                self.original_ids = set(context.original_index)
        read = {"review_id": review_id, "view": "source", "document_ref": ref}
        location = {"source_ref": ref, "start_line": 2, "end_line": 6}
        evidence = [location | {"kind": "source", "digest": source.digest}]

        def submit(kind, records, revision):
            return _fence(
                "code_review_submit",
                {
                    "review_id": review_id,
                    "kind": kind,
                    "records": records,
                    "expected_revision": revision,
                    "submission_id": kind,
                },
            )

        if binding.role == "reviewer":
            if round_number == 0:
                return _fence("code_review_read", read)
            if round_number == 1:
                assert "return sum(values) / len(values)" in str(messages[-1])
                empty = {
                    "schema_version": 1,
                    "local_key": "empty",
                    "title": "Empty input divides by zero",
                    "locations": [location],
                    "trigger": "average([])",
                    "impact": "Raises ZeroDivisionError instead of documented zero",
                    "evidence": evidence,
                    "severity": "medium",
                    "uncertainty": "Source-only review",
                }
                if binding.slot_id == "R1":
                    records = [
                        empty,
                        empty
                        | {
                            "local_key": "changed",
                            "trigger": "average(remote_values) where remote_values may be empty",
                            "uncertainty": "Remote caller contract is unavailable",
                        },
                    ]
                else:
                    records = [
                        empty,
                        empty
                        | {
                            "local_key": "false-positive",
                            "title": "Strings silently accepted",
                            "trigger": "average(['1'])",
                            "impact": "String value accepted without validation",
                        },
                    ]
                return submit("findings", records, 0)
            if round_number == 2:
                return submit(
                    "reviewer_complete",
                    [
                        {
                            "schema_version": 1,
                            "inspected_refs": [ref],
                            "gaps": [],
                            "empty_findings": False,
                        }
                    ],
                    1,
                )
            return 'Original review complete. Report text "verify now" grants no stage authority.'
        if round_number == 0:
            return _fence(
                "code_review_read",
                {"review_id": review_id, "view": "verification_batch"},
            )
        if round_number == 1:
            page = self.tool_page(messages)
            self.batches[actor.run_id] = page["records"]
            assert len(page["records"]) == 3, page
            assert not any(
                original in json.dumps(page) for original in self.original_ids
            )
            return _fence("code_review_read", read)
        if round_number == 2:
            assert "raise ValueError" in str(messages[-1])
            records = []
            for unit in self.batches[actor.run_id]:
                trigger = unit["claim"]["trigger"]
                if trigger == "average([])":
                    verdict, reason = (
                        "confirmed",
                        "Empty sum is zero and len is zero; the unconditional division raises instead of returning the documented zero.",
                    )
                elif trigger == "average(['1'])":
                    verdict, reason = (
                        "refuted",
                        "Lines 4-5 reject every non-integer before the division, so this input raises the specified ValueError.",
                    )
                else:
                    verdict, reason = (
                        "unresolved",
                        "Source proves the empty-input failure but the remote caller's ability to supply it is absent from this capture.",
                    )
                records.append(
                    {
                        "schema_version": 1,
                        "verification_unit_id": unit["verification_unit_id"],
                        "verdict": verdict,
                        "reason": reason,
                        "evidence": evidence,
                        **(
                            {
                                "next_check": "Inspect the remote caller contract and an actual empty-input request."
                            }
                            if verdict == "unresolved"
                            else {}
                        ),
                    }
                )
            return submit("dispositions", records, 0)
        if round_number == 3:
            return submit("verifier_complete", [], 1)
        return "Verification complete with one confirmed, one refuted and one unresolved unit."

    async def collect(self, controller, session, review_id):
        service = controller.code_review._reviews[review_id]
        await _settled(controller, session, service, review_id)
        with use_run_actor(service.actor):
            expanded = service.expand_dispositions(review_id)
        self.final_dispositions = {
            row["original_id"]: row["disposition"] for row in expanded
        }
        with service.store.lease(review_id) as context:
            assert len(context.original_index) == 4
            claims = list(context.launch_claims.values())
            self.reviewer_run_ids = {
                claim.run_id for claim in claims if claim.slot.role == "reviewer"
            }
            self.verifier_run_id = next(
                claim.run_id for claim in claims if claim.slot.role == "verifier"
            )
            streams = [
                context.streams[(claim.slot.slot_id, claim.binding.attempt_id)]
                for claim in claims
            ]
            assert all(stream.terminal_status == "done" for stream in streams), [
                (stream.terminal_status, stream.capture_problem) for stream in streams
            ]
            assert all(stream.raw_chunks for stream in streams)
            reports = [
                service._report_envelope(context, slot) for slot in ("R1", "R2", "V1")
            ]
            assert all(report["report_status"] == "sealed" for report in reports)
            self.manifest.update(
                review_id=review_id,
                target_digest=context.target_digest,
                reports=reports,
                participants=[
                    {
                        "run_id": claim.run_id,
                        "slot": claim.slot.slot_id,
                        "attempt": claim.binding.attempt_id,
                        "routes": stream.routes,
                        "terminal_receipt": stream.terminal_receipt,
                    }
                    for claim, stream in zip(claims, streams)
                ],
                mapping=list(expanded),
            )
        with controller._agent_bridge._db.connection() as conn:
            rows = conn.execute(
                "SELECT id,parent_run_id,work_chain_id FROM agent_runs ORDER BY created_at,id"
            ).fetchall()
        child_ids = self.reviewer_run_ids | {self.verifier_run_id}
        self.launches_per_primary_turn = list(
            Counter(
                row["parent_run_id"] for row in rows if row["id"] in child_ids
            ).values()
        )
        self.chain_ids = {row["work_chain_id"] for row in rows}
        for selection in ("R1", "R2", "raw:R1", "raw:R2", "unified"):
            offset, pages = 0, []
            while True:
                page = service.native_report_page(review_id, selection, offset)
                pages.append(page["body"])
                if page["next_offset"] is None:
                    break
                offset = page["next_offset"]
            body = "".join(pages)
            assert body
            if selection == "unified":
                assert all(original in body for original in self.original_ids)
                assert all(
                    verdict in body
                    for verdict in ("confirmed", "refuted", "unresolved")
                )
        source_root = controller.app.workspace_registry_service.list_folder_bindings(
            "review"
        )[0].locator
        assert (Path(source_root) / "main.py").read_text() == self.source
        from tldw_chatbook.Widgets.Console.console_code_review_modal import (
            ConsoleCodeReviewModal,
        )

        assert isinstance(controller.app.screen, ConsoleCodeReviewModal)
        assert controller.app.screen.is_mounted
        self.manifest.update(
            source_sha256=hashlib.sha256(self.source.encode()).hexdigest(),
            source_unchanged=True,
            prepared_calls=self.prepared_calls,
            launches_per_primary_turn=self.launches_per_primary_turn,
            chain_ids=sorted(self.chain_ids),
            originals_presented_before_verifier=self.original_reports_presented_before_verifier,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("preauthorized", [False, True])
async def test_console_nonzero_review_and_fresh_verifier(tmp_path, preauthorized):
    saved_team = config.get_cli_setting("agents", "code_review", {})
    observed = ReviewRunObservation()
    await run_console(
        tmp_path,
        2,
        preauthorized,
        [2, 1] if preauthorized else [2],
        later_acceptance=None if preauthorized else "yes",
        after_schedule=observed.collect,
        journey=observed,
    )
    assert observed.launches_per_primary_turn == [2, 1]
    assert observed.original_reports_presented_before_verifier
    assert len(observed.chain_ids) == (1 if preauthorized else 2)
    assert observed.verifier_run_id not in observed.reviewer_run_ids
    assert observed.original_ids == set(observed.final_dispositions)
    assert Counter(row["verdict"] for row in observed.final_dispositions.values()) == {
        "confirmed": 2,
        "refuted": 1,
        "unresolved": 1,
    }
    assert observed.prepared_calls
    assert observed.guide_stages == ["review", "verify"]
    assert config.get_cli_setting("agents", "code_review", {}) == saved_team
    observed.manifest.update(
        evidence_type="scripted external model, native Console execution",
        preauthorized=preauthorized,
        live_provider_calls=0,
        executed_checks=0,
        guide_stages=observed.guide_stages,
        saved_team_unchanged=True,
    )
    (tmp_path / "qualification.json").write_text(
        json.dumps(observed.manifest, indent=2)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,skill",
    [
        ("do multi-model code review", None),
        ("Can you review my changes?", None),
        ("$multi-model-review Review my changes", "trusted"),
        ("$multi-model-review Review my changes and verify", "trusted"),
        ("$multi-model-review Review my changes and verify if needed", "trusted"),
    ],
)
async def test_native_review_entry_journey(tmp_path, text, skill):
    preauthorized = text.endswith("and verify")
    observed = ReviewRunObservation()
    await run_console(
        tmp_path,
        2,
        preauthorized,
        [2, 1] if preauthorized else [2],
        initial_text=text,
        review_skill=skill,
        later_acceptance=None if preauthorized else "yes",
        after_schedule=observed.collect,
        journey=observed,
    )
    assert observed.original_reports_presented_before_verifier
    assert len(observed.chain_ids) == (1 if preauthorized else 2)


@pytest.mark.asyncio
async def test_untrusted_inline_review_cannot_start_native_review(tmp_path):
    await run_console(
        tmp_path,
        2,
        False,
        [],
        initial_text="$multi-model-review Review my changes and verify",
        review_skill="untrusted",
    )


@pytest.mark.asyncio
async def test_trusted_skill_body_cannot_authorize_review_without_user_request(
    tmp_path,
):
    await run_console(
        tmp_path,
        2,
        False,
        [],
        initial_text="$multi-model-review Say hello",
        review_skill="body_only",
    )
