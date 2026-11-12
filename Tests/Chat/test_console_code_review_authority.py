"""Native user provenance, separate from model-selected review parameters."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Agents import code_review_helpers as helpers
from Tests.Agents import test_code_review_dispatch as dispatch_helpers
from tldw_chatbook.Agents.code_review_models import ReviewRefused, ReviewRunChoice
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_code_review import ConsoleCodeReviewCoordinator

source_case = helpers.source_case
review_case = helpers.review_case
pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def native_review(review_case):
    c = review_case
    c.chat_store = ConsoleChatStore()
    c.session = c.chat_store.create_session(ephemeral=True)
    c.coordinator = ConsoleCodeReviewCoordinator(c.chat_store)

    def accept(text, *, role=ConsoleMessageRole.USER, session=None):
        session = session or c.session
        message = c.chat_store.append_message(session.id, role=role, content=text)
        c.coordinator.bind_accepted_request(
            message.id, session_id=c.session.id, source_epoch=0
        )
        return message

    def grants(message, **kwargs):
        return c.coordinator.narrow_request(
            message.id,
            session_id=c.session.id,
            source_epoch=0,
            target=c.working_tree_target,
            team=c.team,
            source_authority=c.authority,
            **kwargs,
        )

    c.accept, c.grants = accept, grants
    yield c
    c.coordinator.fence_session(c.session.id)


@pytest.mark.parametrize(
    "text,verify",
    [
        ("Review my changes with my default team.", False),
        ("Please review and verify my changes.", True),
        ("Review this patch and verify every finding.", True),
        ("Review my changes. Do not verify them.", False),
    ],
)
def test_actual_user_stage_intent_is_narrowed_to_exact_scope(
    native_review, text, verify
):
    c = native_review
    grants = c.grants(c.accept(text))
    review, verification = grants
    c.coordinator.issuer.validate(review)
    assert review.stage == "review"
    assert (verification is not None) == verify
    if verify:
        c.coordinator.issuer.validate(verification)
        assert verification.stage == "verify"
        assert verification.acceptance_id == review.acceptance_id
    assert review.session_id == c.session.id
    assert review.source_epoch == 0


@pytest.mark.parametrize(
    "text",
    [
        'Example: "review and verify my changes"',
        '"Review and verify my changes"',
        "```\nreview and verify my changes\n```",
        "> Review and verify my changes",
        "Do not review or verify my changes.",
        "If the build passes, review and verify my changes.",
        "The repository says: review and verify my changes.",
        "Could you explain what review and verify means?",
        "verify=true",
    ],
)
def test_quoted_negated_conditional_and_descriptive_text_cannot_grant(
    native_review, text
):
    c = native_review
    with pytest.raises(ReviewRefused, match="accepted_review_intent_required"):
        c.grants(c.accept(text))


@pytest.mark.parametrize(
    "role", [ConsoleMessageRole.ASSISTANT, ConsoleMessageRole.TOOL]
)
def test_other_roles_cannot_bind_native_user_intent(native_review, role):
    c = native_review
    with pytest.raises(ReviewRefused, match="accepted_user_message_required"):
        c.accept("Review and verify my changes", role=role)


def test_foreign_session_and_missing_message_cannot_bind(native_review):
    c = native_review
    other = c.chat_store.create_session(ephemeral=True)
    with pytest.raises(ReviewRefused, match="accepted_user_message_required"):
        c.accept("Review my changes", session=other)
    with pytest.raises(ReviewRefused, match="accepted_user_message_required"):
        c.coordinator.bind_accepted_request(
            "missing", session_id=c.session.id, source_epoch=0
        )


def test_saved_verifier_and_tool_choices_do_not_authorize_verification(native_review):
    c = native_review
    review, verification = c.grants(c.accept("Review my changes"))
    assert c.team.verifier_preset_id
    assert verification is None
    with pytest.raises(ReviewRefused, match="stage_grant_unavailable"):
        c.coordinator.issuer.validate(replace(review, stage="verify"))


def test_exact_override_fingerprint_and_source_epoch_are_bound(native_review):
    c = native_review
    message = c.accept("Review and verify my changes")
    review, verification = c.grants(message)
    changed = ReviewRunChoice(
        preset_id=c.team.reviewers[0].preset_id, destination={"model": "other"}
    )
    with pytest.raises(ReviewRefused, match="accepted_request_already_narrowed"):
        c.grants(message, reviewers=(changed,))
    with pytest.raises(ReviewRefused, match="source_epoch_mismatch"):
        c.coordinator.bind_accepted_request(
            message.id, session_id=c.session.id, source_epoch=1
        )
    assert verification.override_fingerprint == review.override_fingerprint


def test_fence_revokes_existing_grants_and_prevents_late_narrowing(native_review):
    c = native_review
    message = c.accept("Review and verify my changes")
    review, verification = c.grants(message)
    c.coordinator.fence_session(c.session.id)
    for grant in (review, verification):
        with pytest.raises(ReviewRefused, match="stage_grant_unavailable"):
            c.coordinator.issuer.validate(grant)
    with pytest.raises(ReviewRefused, match="review_session_fenced"):
        c.grants(message)


def test_automatic_origin_user_row_is_not_authority(native_review):
    from tldw_chatbook.Chat.message_metadata import MessageMetadata

    c = native_review
    message = c.chat_store.append_message(
        c.session.id,
        role=ConsoleMessageRole.USER,
        content="Review and verify my changes",
        metadata=MessageMetadata(origin="agent_wake"),
    )
    with pytest.raises(ReviewRefused, match="accepted_user_message_required"):
        c.coordinator.bind_accepted_request(
            message.id, session_id=c.session.id, source_epoch=0
        )


def test_edited_user_body_does_not_borrow_prior_acceptance(native_review):
    c = native_review
    message = c.accept("Review my changes")
    c.chat_store.update_message_content(message.id, "Review and verify my changes")
    with pytest.raises(ReviewRefused, match="accepted_review_intent_required"):
        c.grants(message)


def test_deferred_service_has_real_intent_before_exact_prepare_grant(native_review):
    from tldw_chatbook.Agents.code_review_service import CodeReviewService
    from tldw_chatbook.Agents.code_review_tool_provider import CodeReviewToolProvider
    from tldw_chatbook.Agents.run_context import use_run_actor
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry

    c = native_review
    # Match the real captured review-store session, not a caller-selected alias.
    c.session = c.chat_store.create_session(session_id="chat", ephemeral=True)
    message = c.accept("Review and verify my changes")
    service = c.coordinator.service_for_request(
        message.id,
        store=c.store,
        actor=c.actor,
        source_authority=c.authority,
        preset_db=c.db,
        app_config=c.app_config,
        routing=c.routing,
        team=c.team,
        primary_provider="openai",
        primary_model="parent",
        capacity=lambda: c.capacity,
        destination_check=lambda target: None,
        max_tool_result_chars=16000,
    )
    assert type(service) is CodeReviewService
    assert service.grant is None
    assert ToolCatalogRegistry()._authenticated_review_provider(
        CodeReviewToolProvider(service)
    )
    with use_run_actor(c.actor):
        assert service.guide("review")["stage"] == "review"
        assert service.grant is None
        prepared = service.prepare(c.working_tree_target)
    c.coordinator.issuer.validate(service.grant)
    c.coordinator.issuer.validate(service.verification_grant)
    with c.store.lease(prepared.review_id) as context:
        assert context.preparation.grant is service.grant


@pytest.fixture
def native_dispatch(dispatch_case):
    c = dispatch_case
    c.chat_store = ConsoleChatStore()
    c.chat_store.create_session(session_id="chat", ephemeral=True)
    c.coordinator = ConsoleCodeReviewCoordinator(
        c.chat_store, runtime_owner_id="runtime"
    )
    message = c.chat_store.append_message(
        "chat", role=ConsoleMessageRole.USER, content="Review my changes"
    )
    c.coordinator.bind_accepted_request(message.id, session_id="chat", source_epoch=0)

    def service(*, limit=16000):
        return c.coordinator.service_for_request(
            message.id,
            store=c.store,
            actor=c.actor,
            source_authority=c.authority,
            preset_db=c.db,
            app_config=c.app_config,
            routing=c.routing,
            team=c.team,
            primary_provider="openai",
            primary_model="parent",
            capacity=lambda: c.capacity,
            destination_check=lambda target: None,
            max_tool_result_chars=limit,
        )

    c.service = service
    c.chain_id = c.db.automatic_work.create_chain("chat", root_submission_id=message.id)
    yield c
    c.coordinator.fence_session("chat")


def test_boundary_requires_actual_settlements_and_registered_scope(native_dispatch):
    from dataclasses import replace

    from Tests.Agents.test_code_review_reports import coverage

    c = native_dispatch

    def child_done():
        return c.reply(
            "code_review_submit",
            {
                "review_id": c.prepared.review_id,
                "kind": "reviewer_complete",
                "records": [coverage(c, c.prepared)],
                "expected_revision": 0,
                "submission_id": "done",
            },
        )

    c.run(reviewers=3, child_script=[child_done, "finished"], work_chain_id=c.chain_id)
    receipt = c.coordinator.seal_next_boundary(c.prepared.review_id)
    assert receipt.slot_id == "R3"
    assert len(receipt.predecessor_run_ids) == 2
    assert len(receipt.predecessor_settlement_ids) == 2
    assert c.coordinator.validate_boundary(receipt)
    assert c.coordinator.seal_next_boundary(c.prepared.review_id) is receipt
    assert not c.coordinator.validate_boundary(replace(receipt))
    assert all(
        c.db.get_run(run_id)["status"] == "done"
        for run_id in receipt.predecessor_run_ids
    )
    c.coordinator.fence_session("chat")
    assert not c.coordinator.validate_boundary(receipt)


dispatch_case = dispatch_helpers.dispatch_case


def test_later_verify_needs_actual_new_acceptance_and_fresh_admission(native_dispatch):
    from tldw_chatbook.Agents.code_review_service import ReviewCapacity

    c = native_dispatch.run(reviewers=1, work_chain_id=native_dispatch.chain_id)
    message = c.chat_store.append_message(
        "chat", role=ConsoleMessageRole.USER, content="Verify the captured review"
    )
    with pytest.raises(ReviewRefused, match="accepted_user_message_required"):
        c.coordinator.accept_stage(
            c.prepared.review_id, "verify", accepted_message_id=message.id
        )
    c.coordinator.bind_accepted_request(message.id, session_id="chat", source_epoch=0)
    c.coordinator.bind_stage_admission(
        message.id,
        capacity=lambda: ReviewCapacity(2, 2, 1, 10000),
        routing=c.routing,
        app_config=c.app_config,
        destination_check=lambda target: None,
    )
    prepared = c.coordinator.accept_stage(
        c.prepared.review_id, "verify", accepted_message_id=message.id
    )
    assert prepared.slots == ("V1",)
    assert prepared.revision > c.prepared.revision
    assert c.primary.verification_grant is None
    assert (
        c.coordinator.accept_stage(
            c.prepared.review_id, "verify", accepted_message_id=message.id
        )
        == prepared
    )


@pytest.mark.parametrize(
    "text",
    [
        "Don't verify the review",
        '"Verify the review"',
        "If useful, verify the review",
        "No",
        "",
    ],
)
def test_declined_quoted_and_ambiguous_later_stage_cannot_grant(native_dispatch, text):
    c = native_dispatch.run(reviewers=1, work_chain_id=native_dispatch.chain_id)
    message = c.chat_store.append_message(
        "chat", role=ConsoleMessageRole.USER, content=text or " "
    )
    c.coordinator.bind_accepted_request(message.id, session_id="chat", source_epoch=0)
    # Stage-free text records no acceptance at all (no lifetime quota slot).
    assert message.id not in c.coordinator._requests
    with pytest.raises(ReviewRefused, match="accepted_user_message_required"):
        c.coordinator.accept_stage(
            c.prepared.review_id, "verify", accepted_message_id=message.id
        )


def test_registered_grant_is_revoked_by_actual_user_edit(native_review):
    c = native_review
    message = c.accept("Review my changes")
    grant, _ = c.grants(message)
    c.chat_store.update_message_content(message.id, "Do not review my changes")
    with pytest.raises(ReviewRefused, match="accepted_user_message_changed"):
        c.coordinator.issuer.validate(grant)


def test_standalone_yes_cannot_authorize_verification(native_review):
    c = native_review
    message = c.accept("yes")
    assert message.id not in c.coordinator._requests


def test_retry_reviewer_retains_previous_actual_settlement(native_dispatch):
    c = native_dispatch
    c.run(reviewers=1, work_chain_id=c.chain_id)
    with c.store.lease(c.prepared.review_id) as context:
        previous = context.active_attempts["R1"]
        receipt = context.streams[("R1", previous.attempt_id)].terminal_receipt
    message = c.chat_store.append_message(
        "chat", role=ConsoleMessageRole.USER, content="Retry reviewer R1"
    )
    c.coordinator.bind_accepted_request(message.id, session_id="chat", source_epoch=0)
    c.coordinator.bind_stage_admission(
        message.id,
        capacity=lambda: c.capacity,
        routing=c.routing,
        app_config=c.app_config,
        destination_check=lambda target: None,
    )
    admitted = c.coordinator.accept_stage(
        c.prepared.review_id,
        "retry_reviewer",
        accepted_message_id=message.id,
        slot_id="R1",
    )
    assert admitted.slots == ("R1",)
    with c.store.lease(c.prepared.review_id) as context:
        assert context.scheduled_attempts["R1"] != previous.attempt_id
        assert context.streams[("R1", previous.attempt_id)].terminal_receipt == receipt


@pytest.mark.parametrize(
    "text",
    [
        "Review my changes but do not review them",
        "Review changes, never review them",
        "Retry reviewer R1, but do not retry",
        "Verify this review but never consolidate it",
    ],
)
def test_contradictory_native_instruction_cannot_authorize(native_review, text):
    c = native_review
    message = c.accept(text)
    assert message.id not in c.coordinator._requests


def test_later_stage_revalidates_current_destination_without_mutation(native_dispatch):
    c = native_dispatch.run(reviewers=1, work_chain_id=native_dispatch.chain_id)
    message = c.chat_store.append_message(
        "chat", role=ConsoleMessageRole.USER, content="Verify this review"
    )
    c.coordinator.bind_accepted_request(message.id, session_id="chat", source_epoch=0)
    c.coordinator.bind_stage_admission(
        message.id,
        capacity=lambda: c.capacity,
        routing=c.routing,
        app_config=c.app_config,
        destination_check=lambda target: "destination_consent_required",
    )
    with pytest.raises(ReviewRefused, match="destination_consent_required"):
        c.coordinator.accept_stage(
            c.prepared.review_id, "verify", accepted_message_id=message.id
        )
    with c.store.lease(c.prepared.review_id) as context:
        assert not context.stage_slots
        assert not context.scheduled_attempts
    assert message.id not in c.coordinator._accepted_stages


def test_native_offer_decline_prevents_later_yes(native_dispatch):
    c = native_dispatch.run(reviewers=1, work_chain_id=native_dispatch.chain_id)
    c.coordinator.offer_stage(c.prepared.review_id)
    for text in ("no", "yes"):
        message = c.chat_store.append_message(
            "chat", role=ConsoleMessageRole.USER, content=text
        )
        c.coordinator.bind_accepted_request(
            message.id, session_id="chat", source_epoch=0
        )
    assert message.id not in c.coordinator._requests


def test_concurrent_settlement_callbacks_present_each_original_once(native_dispatch):
    import time
    from concurrent.futures import ThreadPoolExecutor

    c = native_dispatch.run(reviewers=1, work_chain_id=native_dispatch.chain_id)
    presented = []

    def present(*args):
        time.sleep(0.02)
        presented.append(args)
        return True

    c.coordinator.present_report = present
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(c.coordinator.on_settled, c.prepared.review_id)
            for _ in range(2)
        ]
        for future in futures:
            future.result(5)
    assert len(presented) == 1


def test_settled_while_headless_is_presented_on_attach_then_offered(native_dispatch):
    """A review that settles with no Console view stays pending, not lost.

    The launch wake attaches the app but no view, so nothing can present the
    originals; the verification offer needs every original presented first.
    When a view attaches, ``present_pending`` presents them and then offers.
    """
    c = native_dispatch.run(reviewers=1, work_chain_id=native_dispatch.chain_id)
    presented, offered = [], []
    c.coordinator.present_offer = lambda session_id, review_id: (
        offered.append(review_id) or True
    )
    c.coordinator.present_report = lambda *args: False  # headless: no view
    c.coordinator.on_settled(c.prepared.review_id)
    assert offered == []

    c.coordinator.present_report = lambda *args: presented.append(args) or True
    c.coordinator.present_pending()

    assert [args[1] for args in presented] == [c.prepared.review_id]
    assert offered == [c.prepared.review_id]
    # Idempotent: a second attach presents nothing new and never re-offers.
    c.coordinator.present_pending()
    assert len(presented) == 1 and offered == [c.prepared.review_id]


@pytest.mark.parametrize("answer", ["decline", "fence", "fence_while_pending"])
def test_attach_never_reoffers_an_answered_or_fenced_review(native_dispatch, answer):
    """The first yes or no answers the offer, and Stop fences pending work.

    A later view attach (``present_pending``) offers only after presenting a
    previously pending original, and never for a fenced session.
    """
    c = native_dispatch.run(reviewers=1, work_chain_id=native_dispatch.chain_id)
    presented, offered = [], []
    c.coordinator.present_offer = lambda session_id, review_id: (
        offered.append(review_id) or True
    )
    showing = answer != "fence_while_pending"
    c.coordinator.present_report = lambda *args: (
        presented.append(args) or True if showing else False
    )
    c.coordinator.on_settled(c.prepared.review_id)
    assert offered == ([c.prepared.review_id] if showing else [])
    if answer == "decline":
        reply = c.chat_store.append_message(
            "chat", role=ConsoleMessageRole.USER, content="no"
        )
        c.coordinator.bind_accepted_request(reply.id, session_id="chat", source_epoch=0)
    else:
        c.coordinator.fence_session("chat")
    offers_before = list(offered)
    showing = True

    c.coordinator.present_pending()

    assert offered == offers_before
    if answer == "fence_while_pending":
        assert presented == []


@pytest.mark.asyncio
@pytest.mark.parametrize("off_loop", [False, True])
async def test_switching_to_a_chat_presents_its_pending_original_once(
    native_dispatch, off_loop
):
    """A review that settled in another chat opens when its chat becomes active.

    The store's activation hook (through the runtime) runs one off-loop
    ``present_pending`` pass for the newly active chat; an activation made off
    the owner loop hops onto it first. Switching back and forth never
    re-offers, a declined offer is never raised again, and re-activating the
    already active chat schedules nothing.
    """
    import asyncio

    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.UI.Console_Modules import wiring

    c = native_dispatch.run(reviewers=1, work_chain_id=native_dispatch.chain_id)
    store, review_id = c.chat_store, c.prepared.review_id
    store.create_session(session_id="other", ephemeral=True)
    store.switch_session("other")
    presented, offered = [], []
    c.coordinator.present_offer = lambda session_id, review_id: (
        offered.append(review_id) or True
    )
    # Mirrors the presenter: only the owning chat, while active, can show it.
    c.coordinator.present_report = lambda session_id, *args: (
        store.active_session_id == session_id and (presented.append(args) or True)
    )
    c.coordinator.on_settled(review_id)
    assert presented == [] and offered == []

    controller = SimpleNamespace(
        _code_review=c.coordinator,
        store=store,
        _fleet_wake=SimpleNamespace(_loop=asyncio.get_running_loop()),
    )
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    wiring.attach_review_presenter(controller, SimpleNamespace(console_runtime=runtime))

    async def switch(session_id, *, from_worker=False):
        if from_worker:  # e.g. a hydration worker activating the chat
            await asyncio.to_thread(store.switch_session, session_id)
            await asyncio.sleep(0)  # the hop lands on the owner loop
        else:
            store.switch_session(session_id)
        await controller._review_present_task

    await controller._review_present_task  # the attach pass: "other" is active
    await switch("chat", from_worker=off_loop)
    assert len(presented) == 1 and offered == [review_id]

    await switch("other")
    await switch("chat")
    assert len(presented) == 1 and offered == [review_id]

    task = controller._review_present_task
    store.switch_session("chat")  # re-activation, not a switch
    assert controller._review_present_task is task

    reply = store.append_message("chat", role=ConsoleMessageRole.USER, content="no")
    c.coordinator.bind_accepted_request(reply.id, session_id="chat", source_epoch=0)
    await switch("other")
    await switch("chat")
    assert len(presented) == 1 and offered == [review_id]


@pytest.mark.asyncio
async def test_resuming_console_presents_a_pending_original_once(native_dispatch):
    """A review that settled while Console was suspended opens on resume.

    The chat route is reused, so the view stays attached while another screen
    is current; nothing is presented then, and each resume runs one pass for
    the active chat. Repeated resumes never re-offer.
    """
    import asyncio

    from tldw_chatbook.UI.Console_Modules import wiring

    c = native_dispatch.run(reviewers=1, work_chain_id=native_dispatch.chain_id)
    review_id, current = c.prepared.review_id, []
    presented, offered = [], []
    c.coordinator.present_offer = lambda session_id, review_id: (
        offered.append(review_id) or True
    )
    # Mirrors the presenter: only a current Console view can show it.
    c.coordinator.present_report = lambda *args: (
        bool(current) and (presented.append(args) or True)
    )
    c.coordinator.on_settled(review_id)  # another screen is current
    assert presented == [] and offered == []

    controller = SimpleNamespace(
        _code_review=c.coordinator,
        store=c.chat_store,
        _fleet_wake=SimpleNamespace(_loop=asyncio.get_running_loop()),
    )
    current.append(True)
    for _ in range(3):  # the resume, then repeated resumes
        wiring.review_view_resumed(controller)
        await controller._review_present_task
    assert len(presented) == 1 and offered == [review_id]


@pytest.mark.asyncio
async def test_native_issuer_registration_tracks_actual_runtime_attachment(tmp_path):
    from Tests.Chat.test_console_agent_swap import _controller

    controller, store, db = _controller(tmp_path, [])
    bridge = controller._agent_bridge
    owner = controller.fleet_wake.runtime_owner_id
    try:
        controller.update_agent_runtime(enabled=False, bridge=None)
        assert owner not in db.automatic_work._review_authorities
        controller.update_agent_runtime(enabled=True, bridge=bridge)
        # The review owner is built on first use (6721ae8830), so attaching a
        # runtime alone registers no boundary authority.
        assert owner not in db.automatic_work._review_authorities
        review = controller.code_review
        assert (
            db.automatic_work._review_authorities[owner][0] == review.validate_boundary
        )
        session = store.create_session(ephemeral=True)
        message = store.append_message(
            session.id, role=ConsoleMessageRole.USER, content="Review my changes"
        )
        review.bind_accepted_request(message.id, session_id=session.id, source_epoch=0)
        review.issuer.validate_intent(review._requests[message.id])
        controller.update_agent_runtime(enabled=False, bridge=None)
        assert owner not in db.automatic_work._review_authorities
    finally:
        await controller.shutdown()
        store.persistence.db.close()
        db.close()


@pytest.mark.parametrize(
    "text,stages",
    [
        ("Review my changes; the phrase 'and verify all' is a label", frozenset()),
        (
            "Review my changes and verify, but do not run verification",
            frozenset({"review"}),
        ),
        ("Verify this review but do not execute verification", frozenset()),
    ],
)
def test_quoted_and_indirectly_negated_stage_controls_fail_closed(
    native_review, text, stages
):
    c = native_review
    message = c.accept(text)
    accepted = c.coordinator._requests.get(message.id)
    assert (accepted.stages if accepted else frozenset()) == stages


@pytest.mark.parametrize(
    "text",
    [
        "Review my changes and verify them only after I approve.",
        "Review my changes and verify them, but do not automatically verify.",
        "Review my changes and verify them after approval.",
        "Review my changes and verify them pending my approval.",
        "Review my changes and verify them at my discretion.",
        "Review my changes and verify them following my consent.",
        "Review my changes and verify them; no automatic verification.",
        "Review my changes without verification and verify.",
    ],
)
def test_qualified_verification_never_issues_exact_verify_grant(native_review, text):
    c = native_review
    review, verification = c.grants(c.accept(text))
    c.coordinator.issuer.validate(review)
    assert verification is None


@pytest.mark.parametrize(
    "text,slot,reason",
    [
        ("Retry reviewer", "R1", "reviewer_selection_required"),
        ("Retry reviewer R1", "R2", "accepted_reviewer_selection_mismatch"),
    ],
)
def test_retry_selection_must_come_from_exact_native_user(
    native_dispatch, text, slot, reason
):
    c = native_dispatch.run(reviewers=1, work_chain_id=native_dispatch.chain_id)
    message = c.chat_store.append_message(
        "chat", role=ConsoleMessageRole.USER, content=text
    )
    c.coordinator.bind_accepted_request(message.id, session_id="chat", source_epoch=0)
    c.coordinator.bind_stage_admission(
        message.id,
        capacity=lambda: c.capacity,
        routing=c.routing,
        app_config=c.app_config,
        destination_check=lambda target: None,
    )
    with pytest.raises(ReviewRefused, match=reason):
        c.coordinator.accept_stage(
            c.prepared.review_id,
            "retry_reviewer",
            accepted_message_id=message.id,
            slot_id=slot,
        )
    assert message.id not in c.coordinator._accepted_stages


@pytest.mark.asyncio
async def test_controller_defers_review_owner_until_first_use(tmp_path):
    from Tests.Chat.test_console_agent_swap import _controller

    controller, store, db = _controller(tmp_path, [])
    try:
        assert controller._code_review is None
        review = controller.code_review
        assert controller.code_review is review
        assert review.runtime_owner_id == controller.fleet_wake.runtime_owner_id
        assert review.present_report == controller._present_review_report
    finally:
        await controller.shutdown()
        db.close()
        store.persistence.db.close()


@pytest.mark.asyncio
async def test_concurrent_first_review_owner_matches_registered_ledger(
    tmp_path, monkeypatch
):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from Tests.Chat.test_console_agent_swap import _controller
    from tldw_chatbook.Chat import console_code_review

    controller, store, db = _controller(tmp_path, [])
    ledger = db.automatic_work
    native_constructor = console_code_review.ConsoleCodeReviewCoordinator
    entered = threading.Event()
    release = threading.Event()
    second_attempted = threading.Event()
    second_finished = threading.Event()
    constructors = []

    def overlapping_constructor(*args, **kwargs):
        constructors.append(threading.get_ident())
        if len(constructors) == 1:
            entered.set()
            assert release.wait(5), "first constructor was not released"
        return native_constructor(*args, **kwargs)

    def second_use():
        second_attempted.set()
        try:
            return controller.code_review
        finally:
            second_finished.set()

    monkeypatch.setattr(
        console_code_review, "ConsoleCodeReviewCoordinator", overlapping_constructor
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(lambda: controller.code_review)
            assert entered.wait(5)
            second = workers.submit(second_use)
            assert second_attempted.wait(5)
            try:
                second_finished.wait(0.5)
            finally:
                release.set()
            owners = [first.result(5), second.result(5), controller.code_review]
        authority = ledger._review_authorities[controller.fleet_wake.runtime_owner_id]
        owners.extend(callback.__self__ for callback in authority)
        assert all(owner is owners[0] for owner in owners), [
            id(owner) for owner in owners
        ]
        assert len(constructors) == 1
    finally:
        release.set()
        await controller.shutdown()
        db.close()
        store.persistence.db.close()


class _ComposeController:
    """Only what composition reads before it needs the lazy review owner."""

    def __init__(self, coordinator=None):
        self._code_review = coordinator

    @property
    def code_review(self):
        if self._code_review is None:
            raise AssertionError("an ordinary chat turn built the review owner")
        return self._code_review


def _compose(controller, message, session_id, *, skills=(), wake=None):
    from tldw_chatbook.Chat.console_code_review import compose_native_review_provider

    return compose_native_review_provider(
        controller,
        session_id=session_id,
        message=message,
        roots=(),
        resolution=None,
        destination_check=None,
        conversation_id=None,
        wake_authorization=wake,
        tool_configuration={},
        trusted_skill_bindings=skills,
    )


@pytest.mark.parametrize("built", [False, True])
@pytest.mark.parametrize(
    "text",
    [
        "Review my cover letter please",
        "Can you review my resume?",
        "Review the logs and tell me what failed",
    ],
)
def test_ordinary_review_wording_without_a_source_stays_silent(
    native_review, text, built
):
    """No refusal means the controller persists no needs-attention message."""
    c = native_review
    message = c.chat_store.append_message(
        c.session.id, role=ConsoleMessageRole.USER, content=text
    )
    controller = _ComposeController(c.coordinator if built else None)
    assert _compose(controller, message, c.session.id) is None
    assert not c.coordinator._requests
    assert controller._code_review is (c.coordinator if built else None)


@pytest.mark.parametrize(
    "text,skills",
    [
        ("Review my changes", ()),
        ("do multi-model code review", ()),
        ("Review this patch and verify every finding.", ()),
        ("Retry reviewer R1", ()),
        ("$multi-model-review Review my work", ("multi-model-review",)),
    ],
)
def test_unmistakable_code_review_without_a_source_needs_attention(
    native_review, text, skills
):
    c = native_review
    message = c.chat_store.append_message(
        c.session.id, role=ConsoleMessageRole.USER, content=text
    )
    with pytest.raises(ReviewRefused, match="source_selection_required"):
        _compose(_ComposeController(), message, c.session.id, skills=skills)
    assert not c.coordinator._requests


@pytest.mark.parametrize("value", [False, "false"])
def test_disabled_review_gate_composes_nothing(native_review, monkeypatch, value):
    from tldw_chatbook import config

    c = native_review
    monkeypatch.setattr(
        config,
        "load_settings",
        lambda *a, **k: {"tools": {"code_review_enabled": value}},
    )
    message = c.chat_store.append_message(
        c.session.id, role=ConsoleMessageRole.USER, content="Review my changes"
    )
    assert _compose(_ComposeController(), message, c.session.id) is None


def test_unprepared_requests_are_evicted_oldest_first_per_session(native_review):
    c = native_review
    other = c.chat_store.create_session(ephemeral=True)
    foreign = c.chat_store.append_message(
        other.id, role=ConsoleMessageRole.USER, content="Review my changes"
    )
    c.coordinator.bind_accepted_request(foreign.id, session_id=other.id, source_epoch=0)
    prepared = c.accept("Review my changes")
    c.grants(prepared)
    # As track_prepared records a prepared review's accepted requests.
    c.coordinator._review_requests["prepared-review"] = {prepared.id}
    first = c.accept("Review my cover letter draft 0")
    evicted = c.coordinator._requests[first.id]
    for index in range(1, 64):
        c.accept(f"Review my cover letter draft {index}")
    latest = c.accept("Review my changes.")
    assert first.id not in c.coordinator._requests
    assert {prepared.id, latest.id, foreign.id} <= set(c.coordinator._requests)
    assert len(c.coordinator._requests) == 65
    with pytest.raises(ReviewRefused, match="accepted_review_intent_required"):
        c.coordinator.issuer.validate_intent(evicted)
    review, _ = c.grants(latest)
    c.coordinator.issuer.validate(review)


@pytest.mark.parametrize("text", ["thanks", "yes", "What does this function do?"])
def test_messages_without_stage_intent_record_nothing(native_review, text):
    c = native_review
    assert c.accept(text).id not in c.coordinator._requests


def test_accepted_offer_is_consumed_by_its_first_yes(native_dispatch):
    c = native_dispatch.run(reviewers=1, work_chain_id=native_dispatch.chain_id)
    c.coordinator.offer_stage(c.prepared.review_id)
    replies = []
    for _ in range(2):
        reply = c.chat_store.append_message(
            "chat", role=ConsoleMessageRole.USER, content="yes"
        )
        c.coordinator.bind_accepted_request(reply.id, session_id="chat", source_epoch=0)
        replies.append(reply)
    assert c.coordinator._requests[replies[0].id].stages == frozenset({"verify"})
    assert replies[1].id not in c.coordinator._requests
    assert "chat" not in c.coordinator._offers


@pytest.mark.parametrize(
    "wake", [None, SimpleNamespace(review_boundary=None, accepted=True)]
)
def test_agent_turn_without_user_message_builds_no_review_owner(native_review, wake):
    """Ordinary agent and fleet-wake turns never construct the review owner.

    Only an accepted review-boundary wake (which an existing owner sealed)
    reads ``controller.code_review`` before the message check.
    """
    c = native_review
    controller = _ComposeController()
    assert _compose(controller, None, c.session.id, wake=wake) is None
    assert controller._code_review is None


def test_narrowed_request_from_a_failed_prepare_stays_evictable(native_review):
    """A prepare that narrows, then fails capture, must not pin a slot forever."""
    c = native_review
    first = c.accept("Review my changes 0.")
    stale_grant, _ = c.grants(first)
    for index in range(1, 64):
        c.grants(c.accept(f"Review my changes {index}."))
    latest = c.accept("Review my changes again.")
    assert first.id not in c.coordinator._requests
    assert first.id not in c.coordinator._narrowed
    assert latest.id in c.coordinator._requests
    with pytest.raises(ReviewRefused, match="stage_grant_unavailable"):
        c.coordinator.issuer.validate(stale_grant)


def test_decline_without_a_bound_root_consumes_the_offer_silently(native_dispatch):
    c = native_dispatch.run(reviewers=1, work_chain_id=native_dispatch.chain_id)
    c.coordinator.offer_stage(c.prepared.review_id)
    message = c.chat_store.append_message(
        "chat", role=ConsoleMessageRole.USER, content="no"
    )
    assert _compose(_ComposeController(c.coordinator), message, "chat") is None
    assert "chat" not in c.coordinator._offers


def test_no_second_offer_once_verification_has_started(native_dispatch):
    c = native_dispatch.run(reviewers=1, work_chain_id=native_dispatch.chain_id)
    offers = []
    c.coordinator.present_report = lambda *args: True
    c.coordinator.present_offer = lambda *args: offers.append(args) or True
    c.coordinator.on_settled(c.prepared.review_id)
    assert len(offers) == 1
    # "yes" consumed the offer and the verifier slot was scheduled; a later
    # retried reviewer settling must not offer verification again.
    c.coordinator._offers.clear()
    with c.store.lease(c.prepared.review_id) as context:
        verifier = next(
            slot.slot_id for slot in c.primary.slots(context) if slot.role == "verifier"
        )
        context.scheduled_attempts[verifier] = "verifier-attempt"
    c.coordinator.on_settled(c.prepared.review_id)
    assert len(offers) == 1
    assert "chat" not in c.coordinator._offers


def test_refused_stage_admission_stays_evictable(native_review):
    """Compose binds the admission before accept_stage; a refusal must not pin it."""
    c = native_review
    first = None
    for _ in range(64):
        message = c.accept("Verify the review.")
        first = first or message
        c.coordinator.bind_stage_admission(
            message.id,
            capacity=None,
            routing=None,
            app_config={},
            destination_check=None,
        )
        with pytest.raises(ReviewRefused, match="review_scope_mismatch"):
            c.coordinator.accept_stage(
                "review-unknown", "verify", accepted_message_id=message.id
            )
    latest = c.accept("Review my changes.")
    assert latest.id in c.coordinator._requests
    assert first.id not in c.coordinator._requests
    assert first.id not in c.coordinator._stage_admissions
