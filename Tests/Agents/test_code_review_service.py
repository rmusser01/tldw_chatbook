"""Public review authority and preparation with real source/store/preset owners."""

from dataclasses import replace

import pytest

from Tests.Agents import code_review_helpers as helpers
from tldw_chatbook.Agents.agent_routing import AgentsRoutingConfig
from tldw_chatbook.Agents.code_review_models import ReviewRunChoice
from tldw_chatbook.Agents.code_review_service import (
    ReviewCapacity,
    request_fingerprints,
)
from tldw_chatbook.Agents.code_review_tool_provider import CodeReviewToolProvider
from tldw_chatbook.Agents.run_context import use_run_actor

source_case = helpers.source_case
review_case = helpers.review_case


def test_prepare_freezes_routes_without_model_or_settings_writes(review_case):
    c = review_case
    before = c.db.get_agent_definition(c.team.reviewers[0].preset_id)
    prepared = c.prepare()
    assert prepared.slots == ("R1", "V1")
    with c.store.lease(prepared.review_id) as context:
        assert context.preparation.slots[0].target.model == "test"
        assert context.preparation.slots[0].choice.preset_id == before["id"]
        assert "independently" in context.preparation.guide_text
    assert c.db.get_agent_definition(before["id"]) == before


@pytest.mark.parametrize(
    "role,view",
    [
        ("reviewer", "report"),
        ("reviewer", "index"),
        ("reviewer", "summary"),
        ("verifier", "report"),
        ("verifier", "index"),
        ("verifier", "summary"),
    ],
)
def test_guessed_review_id_does_not_grant_report_access(review_case, role, view):
    c = review_case
    prepared = c.prepare()
    with c.participant(
        prepared, role, "V1" if role == "verifier" else "R1"
    ) as provider:
        result = provider.invoke(
            "code_review:code_review_read",
            {"review_id": prepared.review_id, "view": view},
        )
    assert not result.ok
    assert result.error == "role_denied"


@pytest.mark.parametrize(
    "capacity", [ReviewCapacity(0, 2, 1, 10000), ReviewCapacity(1, 1, 0, 10000)]
)
def test_inadmissible_budget_refuses_before_capture(review_case, capacity):
    c = review_case
    c.capacity = capacity
    c.primary.verification_grant = c.issuer.issue(
        c.owner,
        session_id="chat",
        acceptance_id="accepted",
        stage="verify",
        source_epoch=0,
        **request_fingerprints(c.working_tree_target, c.team),
    )
    with use_run_actor(c.actor):
        result = c.provider.invoke(
            "code_review:code_review_prepare",
            {"target": c.working_tree_target.model_dump(mode="json")},
        )
    assert not result.ok
    assert result.error == "child_budget_unavailable"
    assert c.store.usage().retained_contexts == 0


def test_no_run_or_forged_grant_cannot_prepare(review_case):
    c = review_case
    result = c.provider.invoke(
        "code_review:code_review_prepare",
        {"target": c.working_tree_target.model_dump(mode="json")},
    )
    assert result.error == "run_not_bound"
    forged = replace(c.primary.grant)
    with pytest.raises(ValueError, match="stage_grant_unavailable"):
        c.service(grant=forged)


def test_small_prepare_ack_refuses_before_context_allocation(review_case):
    c = review_case
    with use_run_actor(c.actor):
        result = CodeReviewToolProvider(c.service(limit=32)).invoke(
            "code_review:code_review_prepare",
            {"target": c.working_tree_target.model_dump(mode="json")},
        )
    assert not result.ok
    assert c.store.usage().retained_contexts == 0


@pytest.mark.parametrize(
    "destination",
    [
        {"provider": "openai"},
        {"model": "override"},
        {"provider": "openai", "model": "override"},
    ],
)
@pytest.mark.parametrize("role", ["reviewer", "verifier"])
def test_run_only_routing_overrides_freeze_without_writes(
    review_case, destination, role
):
    from tldw_chatbook.Agents.code_review_models import ReviewDestinationOverride

    c = review_case
    choice = ReviewRunChoice(
        preset_id=c.team.reviewers[0].preset_id
        if role == "reviewer"
        else c.team.verifier_preset_id,
        destination=ReviewDestinationOverride(**destination),
    )
    before = c.db.get_agent_definition(choice.preset_id)
    reviewers, verifier = ((choice,), None) if role == "reviewer" else (None, choice)
    service = c.service(reviewers=reviewers, verifier_choice=verifier)
    with use_run_actor(c.actor):
        prepared = service.prepare(c.working_tree_target, reviewers, verifier)
    with c.store.lease(prepared.review_id) as context:
        slot = next(s for s in context.preparation.slots if s.role == role)
        assert slot.choice == choice
        assert slot.target.provider == destination.get("provider", "openai")
        assert slot.target.model == destination.get("model", before["model"])
    assert c.db.get_agent_definition(choice.preset_id) == before


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("disabled", "override_disabled"),
        ("allowlist", "provider_not_allowlisted"),
        ("unready", "provider_not_ready"),
        ("consent", "destination_consent_required"),
    ],
)
def test_routing_and_host_consent_refuse_before_capture(review_case, mode, reason):
    from tldw_chatbook.Agents.code_review_models import (
        ReviewDestinationOverride,
        ReviewRefused,
    )

    c = review_case
    choice = ReviewRunChoice(
        preset_id=c.team.reviewers[0].preset_id,
        destination=ReviewDestinationOverride(
            provider="anthropic" if mode == "unready" else "openai", model="override"
        ),
    )
    c.routing = AgentsRoutingConfig(
        spawn_override_enabled=mode != "disabled",
        spawn_override_allowlist=("openai/other",)
        if mode == "allowlist"
        else ("openai/*", "anthropic/*"),
    )
    service = c.service(reviewers=(choice,))
    if mode == "consent":
        service.destination_check = lambda target: reason
    with use_run_actor(c.actor), pytest.raises(ReviewRefused, match=reason):
        service.prepare(c.working_tree_target, (choice,))
    assert c.store.usage().retained_contexts == 0


def test_submit_ack_is_atomic_and_stream_local(review_case):
    from Tests.Agents.code_review_helpers import finding_payload

    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        from Tests.Agents.test_code_review_tool_provider import invoke

        page = invoke(
            provider,
            "read",
            review_id=prepared.review_id,
            view="source",
            source_query={
                "op": "resolve",
                "relative_path": "main.py",
                "side": "worktree",
            },
        )
        ref = page["records"][0]["source_ref"]
        with c.store.lease(prepared.review_id) as context:
            digest = context.sources[ref].digest
        finding = finding_payload()
        finding["locations"] = [{"source_ref": ref, "start_line": 1, "end_line": 1}]
        finding["evidence"] = [
            dict(finding["locations"][0], kind="source", digest=digest)
        ]
        args = dict(
            review_id=prepared.review_id,
            kind="findings",
            records=[finding],
            expected_revision=0,
            submission_id="one",
        )
        provider.service.result_limit = 32
        refused = provider.invoke("code_review:code_review_submit", args)
        assert refused.error == "review_response_capacity_unavailable"
        with c.store.lease(prepared.review_id) as context:
            assert not context.original_index
            assert context.publication_sequence == 0
        provider.service.result_limit = 1024
        accepted = invoke(provider, "submit", **args)
        assert accepted["accepted_ids"] == ["R1-F1"]
        assert invoke(provider, "submit", **args) == accepted
        assert (
            invoke(
                provider,
                "read",
                review_id=prepared.review_id,
                view="source",
                document_ref=ref,
            )["submission_revision"]
            == 1
        )
    with use_run_actor(c.actor):
        c.primary.result_limit = 32
        result = c.provider.invoke(
            "code_review:code_review_release",
            {"review_id": prepared.review_id, "expected_revision": 0},
        )
        assert result.error == "native_discard_required"
        with c.store.lease(prepared.review_id) as context:
            assert context.state == "live"
        c.primary.result_limit = 1024
        receipt = c.primary.release(prepared.review_id, 0)
        assert receipt.cleaned and receipt.tombstoned


def test_primary_cannot_submit_reviewer_findings(review_case):
    c = review_case
    prepared = c.prepare()
    with use_run_actor(c.actor):
        result = c.provider.invoke(
            "code_review:code_review_submit",
            dict(
                review_id=prepared.review_id,
                kind="coverage",
                records=[],
                expected_revision=0,
                submission_id="one",
            ),
        )
    assert result.error == "role_denied"


def test_review_only_with_saved_verifier_needs_no_verifier_budget(review_case):
    c = review_case
    c.capacity = ReviewCapacity(1, 1, 0, 10000)
    assert c.prepare().slots == ("R1", "V1")


def test_preauthorized_verification_counts_actual_schedule(review_case):
    from tldw_chatbook.Agents.code_review_models import ReviewRefused

    c = review_case
    c.capacity = ReviewCapacity(1, 1, 0, 10000)
    c.primary.verification_grant = c.issuer.issue(
        c.owner,
        session_id="chat",
        acceptance_id="accepted",
        stage="verify",
        source_epoch=0,
        **request_fingerprints(c.working_tree_target, c.team),
    )
    with pytest.raises(ReviewRefused, match="child_budget_unavailable"):
        c.prepare()


def test_preparation_metadata_is_quota_owned_and_retired(review_case):
    c = review_case
    before = c.store.usage()
    prepared = c.prepare()
    with c.store.lease(prepared.review_id) as context:
        snapshots = [
            item
            for item in context._files.values()
            if item.category == "report" and b'"slots"' in item.path.read_bytes()
        ]
        assert snapshots
        assert b"Review independently" in snapshots[0].path.read_bytes()
        assert c.store.usage().retained_bytes > before.retained_bytes
    with use_run_actor(c.actor):
        c.primary.release(prepared.review_id, 0)
    assert context.preparation is None


def test_submission_rechecks_live_source_exclusions(review_case):
    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        from Tests.Agents.test_code_review_tool_provider import invoke

        ref = invoke(
            provider,
            "read",
            review_id=prepared.review_id,
            view="source",
            source_query={
                "op": "resolve",
                "relative_path": "main.py",
                "side": "worktree",
            },
        )["records"][0]["source_ref"]
        object.__setattr__(
            c.authority.admitted, "exclusions_provider", lambda: (c.file,)
        )
        result = provider.invoke(
            "code_review:code_review_submit",
            dict(
                review_id=prepared.review_id,
                kind="coverage",
                records=[
                    {
                        "schema_version": 1,
                        "inspected_refs": [ref],
                        "gaps": [],
                        "empty_findings": True,
                    }
                ],
                expected_revision=0,
                submission_id="coverage",
            ),
        )
        assert not result.ok
        with c.store.lease(prepared.review_id) as context:
            assert context.publication_sequence == 0


def test_release_replay_is_authorized_and_bounded(review_case):
    c = review_case
    prepared = c.prepare()
    with use_run_actor(c.actor):
        receipt = c.primary.release(prepared.review_id, 0)
        assert c.primary.release(prepared.review_id, 0) == receipt


def test_verifier_requires_its_live_stage_grant(review_case):
    c = review_case
    prepared = c.prepare()
    with c.participant(prepared, "verifier", "V1") as provider:
        provider.service.grant = c.primary.grant
        result = provider.invoke(
            "code_review:code_review_read",
            {
                "review_id": prepared.review_id,
                "view": "source",
                "source_query": {"op": "list", "relative_dir": ".", "side": "worktree"},
            },
        )
        assert not result.ok
        assert result.error == "stage_grant_scope_mismatch"


@pytest.mark.parametrize("state", ["missing", "disabled", "duplicate"])
def test_missing_disabled_and_duplicate_destinations_are_visible(review_case, state):
    from tldw_chatbook.Agents.agent_models import definition_from_row
    from tldw_chatbook.Agents.code_review_models import ReviewRefused

    c = review_case
    identity = c.team.verifier_preset_id
    if state == "missing":
        c.db.soft_delete_agent_definition(identity)
    else:
        definition = definition_from_row(c.db.get_agent_definition(identity))
        c.db.update_agent_definition(
            identity, replace(definition, enabled=state != "disabled", model="test")
        )
    if state == "duplicate":
        assert c.prepare().summary == "duplicate_destinations"
    else:
        with pytest.raises(ReviewRefused, match="preset_" + state):
            c.prepare()
        assert c.store.usage().retained_contexts == 0


def test_actual_other_chat_cannot_use_existing_reference(review_case):
    from tldw_chatbook.Agents.code_review_models import ReviewRefused
    from tldw_chatbook.Agents.code_review_store import ReviewStore
    from tldw_chatbook.Agents.run_context import CurrentRunActor

    c = review_case
    prepared = c.prepare()
    other_store = ReviewStore(c.store._manager, "other-chat")
    actor = CurrentRunActor("primary", "other-run", None)
    other = c.service(actor, store=other_store)
    try:
        with use_run_actor(actor):
            own = other.prepare(c.working_tree_target)
            assert other.read(own.review_id, "summary").records
            with pytest.raises(ReviewRefused):
                other.read(prepared.review_id, "summary")
            with pytest.raises(ReviewRefused):
                other.source_instruction_targets(
                    prepared.review_id, document_ref="forged"
                )
    finally:
        other_store.close()


def test_capacity_change_during_capture_cannot_publish_unacknowledged_context(
    review_case, monkeypatch
):
    from tldw_chatbook.Agents import code_review_service

    c = review_case
    original = code_review_service.capture_review_source

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        c.capacity = replace(c.capacity, remaining_input_tokens=1)
        return result

    monkeypatch.setattr(code_review_service, "capture_review_source", capture)
    with use_run_actor(c.actor):
        result = c.provider.invoke(
            "code_review:code_review_prepare",
            {"target": c.working_tree_target.model_dump(mode="json")},
        )
    assert result.error == "review_response_capacity_unavailable"
    assert c.store.usage().retained_contexts == 0


@pytest.mark.parametrize("role", ["reviewer", "verifier"])
@pytest.mark.parametrize(
    "route,expected_model",
    [
        ("unresolved", None),
        ("preset_pair", "selected"),
        ("preset_provider", "parent"),
        ("preset_model", "selected"),
        ("default_pair", "selected"),
        ("default_model", "selected"),
        ("override_provider", "parent"),
        ("override_model", "selected"),
        ("override_pair", "selected"),
    ],
)
def test_prepare_refuses_unresolved_inheritance_preserving_explicit_routes(
    review_case, monkeypatch, role, route, expected_model
):
    from tldw_chatbook.Agents import code_review_service
    from tldw_chatbook.Agents.agent_models import definition_from_row
    from tldw_chatbook.Agents.code_review_models import (
        ReviewDestinationOverride,
        ReviewRefused,
    )

    c = review_case
    identity = (
        c.team.reviewers[0].preset_id
        if role == "reviewer"
        else c.team.verifier_preset_id
    )
    definition = definition_from_row(c.db.get_agent_definition(identity))
    c.db.update_agent_definition(
        identity,
        replace(
            definition,
            provider="openai" if route in ("preset_pair", "preset_provider") else "",
            model="selected" if route in ("preset_pair", "preset_model") else "",
        ),
    )
    c.routing = replace(
        c.routing,
        subagent_default_provider="openai" if route == "default_pair" else "",
        subagent_default_model="selected"
        if route in ("default_pair", "default_model")
        else "",
    )
    destination = (
        ReviewDestinationOverride(
            provider="openai" if route != "override_model" else None,
            model="selected" if route != "override_provider" else None,
        )
        if route.startswith("override_")
        else None
    )
    choice = ReviewRunChoice(preset_id=identity, destination=destination)
    reviewers, verifier = ((choice,), None) if role == "reviewer" else (None, choice)
    service = c.service(reviewers=reviewers, verifier_choice=verifier)
    before = c.db.get_agent_definition(identity)
    if expected_model is None:
        captures = []
        original_capture = code_review_service.capture_review_source

        def capture(*args, **kwargs):
            captures.append(True)
            return original_capture(*args, **kwargs)

        monkeypatch.setattr(code_review_service, "capture_review_source", capture)
        with (
            use_run_actor(c.actor),
            pytest.raises(ReviewRefused, match="^no_model_resolved$"),
        ):
            service.prepare(c.working_tree_target, reviewers, verifier)
        assert captures == []
        assert c.store.usage().retained_contexts == 0
    else:
        with use_run_actor(c.actor):
            prepared = service.prepare(c.working_tree_target, reviewers, verifier)
        with c.store.lease(prepared.review_id) as context:
            slot = next(slot for slot in context.preparation.slots if slot.role == role)
            assert (slot.target.provider, slot.target.model) == (
                "openai",
                expected_model,
            )
    assert c.db.get_agent_definition(identity) == before
