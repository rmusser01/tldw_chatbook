"""Review capacity is full input capacity with a separately reserved output."""

from tldw_chatbook.Agents.agent_models import AgentConfig
from tldw_chatbook.Agents.agent_service import AgentService, ModelRequest
from tldw_chatbook.Utils.token_counter import ContextWindowResolution


def test_unverified_numeric_fallback_is_unknown(monkeypatch):
    from tldw_chatbook.Utils import token_counter

    monkeypatch.setattr(
        token_counter,
        "resolve_context_window",
        lambda *a, **kw: ContextWindowResolution(128000, "fallback", False),
    )
    service = object.__new__(AgentService)
    assert (
        service.review_input_capacity(
            AgentConfig(model="unknown", system_prompt=""),
            "unknown",
            ModelRequest(messages=({"role": "user", "content": "hello"},), tools=()),
        )
        is None
    )


def test_capacity_counts_native_tools_and_output_reserve(monkeypatch):
    from tldw_chatbook.Utils import token_counter

    monkeypatch.setattr(
        token_counter,
        "resolve_context_window",
        lambda *a, **kw: ContextWindowResolution(8192, "model", True),
    )
    service = object.__new__(AgentService)
    config = AgentConfig(model="gpt-4o", system_prompt="", response_reserve_tokens=1000)
    request = ModelRequest(messages=({"role": "user", "content": "hello"},), tools=())
    capacity = service.review_input_capacity(config, "openai", request)
    assert 0 < capacity < 7192
    with_tools = ModelRequest(
        messages=request.messages, tools=({"description": "large schema " * 400},)
    )
    assert service.review_input_capacity(config, "openai", with_tools) < capacity


import pytest

from Tests.Agents import code_review_helpers, test_code_review_dispatch
from Tests.Agents.test_code_review_verification import seed
from tldw_chatbook.Agents.code_review_models import ReviewRefused

review_case = code_review_helpers.review_case
source_case = code_review_helpers.source_case
dispatch_case = test_code_review_dispatch.dispatch_case
pytestmark = pytest.mark.bootstrap_profile


def test_source_page_uses_actual_estimator_before_publishing_cursor(review_case):
    c = review_case
    prepared = seed(c, 1)
    with c.participant(prepared, role="verifier", slot="V1") as provider:
        service = provider.service
        service.verifier_input_tokens = 10000
        service.verifier_estimate = lambda text: len(text) * 10
        with c.store.lease(prepared.review_id) as context:
            source = next(iter(context.sources))
            pages_before = len(context.source_state.pages)
        with pytest.raises(ReviewRefused):
            service.read(prepared.review_id, "source", source)
        with c.store.lease(prepared.review_id) as context:
            assert len(context.source_state.pages) == pages_before
            assert not context.source_state.cursors


@pytest.mark.parametrize("reject", [False, True])
def test_gateway_admission_sees_final_prepared_request_before_dispatch(
    monkeypatch, reject
):
    from dataclasses import replace

    from Tests.Chat.test_console_agent_bridge import (
        _routed_child_messages,
        _routing_parent_resolution,
        _streaming_adapter,
    )
    from tldw_chatbook.Chat.console_prepared_request import PreparedProviderRequest
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.Utils.token_counter import ContextWindowResolution

    gateway = ConsoleProviderGateway(config_provider=dict, environ={})
    dispatched, admissions = [], []

    async def stream(resolution, request, **kwargs):
        dispatched.append(request)
        yield "done"

    monkeypatch.setattr(gateway, "stream_chat", stream)
    resolution = replace(
        _routing_parent_resolution(),
        context_window=ContextWindowResolution(8192, "server metadata", True),
    )

    def admit(request):
        assert isinstance(request, PreparedProviderRequest)
        assert request.capacity.context_window_tokens == 8192
        assert request.accounting.total_input_tokens > 0
        assert request.model == resolution.model
        admissions.append(request)
        if reject:
            raise ReviewRefused("review_context_reduction_blocked")

    with _streaming_adapter(resolution, gateway) as adapter:
        with adapter.review_request_admission(admit):
            if reject:
                with pytest.raises(
                    ReviewRefused, match="review_context_reduction_blocked"
                ):
                    adapter.chat_call(messages_payload=_routed_child_messages())
            else:
                adapter.chat_call(messages_payload=_routed_child_messages())
        # An ordinary call on the same shared adapter has no leftover callback.
        adapter.chat_call(messages_payload=_routed_child_messages())
    assert len(admissions) == 1
    assert len(dispatched) == (1 if reject else 2)
    if not reject:
        assert dispatched[0] is admissions[0]


def test_gateway_admission_counts_private_continuation_and_effective_fallback(
    monkeypatch,
):
    from dataclasses import replace

    from Tests.Agents.test_provider_continuation_runtime import (
        _checkpoint,
        _pending_call,
    )
    from Tests.Chat.test_console_agent_bridge import (
        _routed_child_messages,
        _routing_parent_resolution,
        _streaming_adapter,
    )
    from tldw_chatbook.Chat.console_prepared_request import CONTINUATION_OWNER_KEY
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.Chat.provider_continuation import (
        ContinuationOwnerGroup,
        ContinuationRestoreTarget,
        ContinuationResult,
    )

    gateway = ConsoleProviderGateway(config_provider=dict, environ={})
    prepared, streamed = [], []

    async def stream(resolution, request, **kwargs):
        streamed.append(request)
        yield "done"

    monkeypatch.setattr(gateway, "stream_chat", stream)
    resolution = replace(
        _routing_parent_resolution(),
        provider="deepseek",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com/v1",
        execution_key="deepseek",
        readiness_key="deepseek",
        # Pinned exactly as resolve_for_send pins deepseek: a child derives its
        # continuation target from the resolution, not the parent's target.
        api_mode="responses",
        continuation_protocol="responses",
        context_window=ContextWindowResolution(8192, "server metadata", True),
    )
    target = ContinuationRestoreTarget(
        "deepseek", "deepseek-v4-flash", "responses", "https://api.deepseek.com/v1"
    )
    call = replace(
        _pending_call("read"),
        state="completed",
        result=ContinuationResult("captured evidence"),
    )
    checkpoint = _checkpoint(call, reasoning=("private continuation " * 5000,))
    group = ContinuationOwnerGroup("owner", checkpoint, checkpoint.rounds)
    messages = _routed_child_messages() + [
        {
            "role": "assistant",
            CONTINUATION_OWNER_KEY: "owner",
            "content": "",
            "tool_calls": [
                {
                    "id": "read",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression":"2+2"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "read", "content": "captured evidence"},
    ]

    def admit(request):
        prepared.append(request)
        assert request.continuation_groups == (group,)
        assert (
            request.accounting.total_input_tokens
            > request.capacity.effective_input_ceiling_tokens
        )
        raise ReviewRefused("review_context_reduction_blocked")

    with _streaming_adapter(
        resolution,
        gateway,
        continuation_target=target,
        continuation_owner_key=CONTINUATION_OWNER_KEY,
    ) as adapter:
        with (
            adapter.review_request_admission(admit),
            pytest.raises(ReviewRefused, match="review_context_reduction_blocked"),
        ):
            adapter.chat_call(messages_payload=messages, continuation_groups=(group,))
        # The selected model on a fallback request, not the parent's, owns admission.
        with adapter.review_request_admission(lambda request: prepared.append(request)):
            adapter.chat_call(
                messages_payload=_routed_child_messages(), model="deepseek-v4-pro"
            )
    assert len(streamed) == 1
    assert prepared[-1] is streamed[0]
    assert prepared[-1].model == "deepseek-v4-pro"


@pytest.mark.bootstrap_profile
def test_unknown_capacity_stops_actual_verifier_without_claim_size_gap(
    dispatch_case, monkeypatch
):
    from tldw_chatbook.Utils import token_counter

    monkeypatch.setattr(
        token_counter,
        "resolve_context_window",
        lambda *a, **kw: ContextWindowResolution(32768, "fallback", False),
    )
    c = dispatch_case.run(verify=True, child_script=["must not dispatch"])
    assert not [actor for actor, request in c.calls if actor.kind == "subagent"]
    with c.store.lease(c.prepared.review_id) as context:
        verifier = next(
            stream for (slot, _), stream in context.streams.items() if slot == "V1"
        )
        assert verifier.verification.request_gaps == ["review_context_capacity_unknown"]
        assert verifier.terminal_status == "error"
        assert not verifier.verification.dispositions


def test_final_prepared_owner_positive_remainder_does_not_consume_claim(
    review_case,
    monkeypatch,
):
    from dataclasses import replace

    from Tests.Agents.test_code_review_reports import finding
    from Tests.Agents.test_provider_continuation_runtime import (
        _checkpoint,
        _pending_call,
    )
    from Tests.Agents.test_run_log_eviction import _make_registry
    from Tests.Chat.test_console_agent_bridge import (
        _routing_parent_resolution,
        _streaming_adapter,
    )
    from tldw_chatbook.Agents.code_review_models import canonical_record_bytes
    from tldw_chatbook.Agents.run_context import use_run_actor
    from tldw_chatbook.Chat.console_prepared_request import CONTINUATION_OWNER_KEY
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.Chat.provider_continuation import (
        ContinuationOwnerGroup,
        ContinuationRestoreTarget,
        ContinuationResult,
    )

    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as reviewer:
        reviewer.service.submit(
            prepared.review_id,
            "findings",
            [finding(c, prepared, "long", trigger="ordinary bounded claim " * 150)],
            0,
            "seed",
        )
    gateway = ConsoleProviderGateway(config_provider=dict, environ={})
    sent = []

    async def stream(resolution, request, **kwargs):
        sent.append(request)
        yield "ready"

    monkeypatch.setattr(gateway, "stream_chat", stream)
    resolution = replace(
        _routing_parent_resolution(),
        provider="deepseek",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com/v1",
        execution_key="deepseek",
        readiness_key="deepseek",
        context_window=ContextWindowResolution(32768, "server metadata", True),
    )
    target = ContinuationRestoreTarget(
        "deepseek", "deepseek-v4-flash", "responses", "https://api.deepseek.com/v1"
    )
    call = replace(
        _pending_call("read"), state="completed", result=ContinuationResult("evidence")
    )
    checkpoint = _checkpoint(
        call, reasoning=("indivisible private continuation " * 1500,)
    )
    group = ContinuationOwnerGroup("owner", checkpoint, checkpoint.rounds)
    messages = [
        {"role": "user", "content": "Verify the next batch."},
        {
            "role": "assistant",
            CONTINUATION_OWNER_KEY: "owner",
            "content": "",
            "tool_calls": [
                {
                    "id": "read",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression":"2+2"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "read", "content": "evidence"},
    ]
    with c.participant(prepared, role="verifier", slot="V1") as provider:
        review = provider.service
        registry = _make_registry(0)
        registry._review_service = review
        binding = c.store._tab.runs[review.actor.run_id]
        config = AgentConfig(
            model=resolution.model,
            provider="deepseek",
            system_prompt="Verify independently",
            native_tools=True,
            review_binding=binding,
        )

        def send(selected):
            with _streaming_adapter(
                selected,
                gateway,
                continuation_target=target,
                continuation_owner_key=CONTINUATION_OWNER_KEY,
            ) as adapter:
                agent = AgentService(
                    db=c.db,
                    registry=registry,
                    chat_call=adapter.chat_call,
                    prepare_provider_continuation_request=True,
                )
                invoke = agent._make_call_model(
                    config,
                    "deepseek",
                    [],
                    continuation_groups=(group,),
                    continuation_owner_key=CONTINUATION_OWNER_KEY,
                    trusted_role="subagent",
                )
                invoke(messages, ())

        send(resolution)
        roomy = sent[-1]
        with c.store.lease(prepared.review_id) as context:
            ledger = c.store.verification_ledger(context, binding)
            cost = review.verifier_estimate(
                canonical_record_bytes(list(ledger.units.public_units)).decode()
            )
        assert 250 < cost <= 4096
        assert cost <= min(4096, review.verifier_input_tokens // 2)
        # Shrink only the real server capacity, retaining exactly the same request/owner.
        window = (
            roomy.accounting.total_input_tokens
            + roomy.capacity.effective_response_tokens
            + roomy.capacity.safety_margin_tokens
            + 500
        )
        constrained = replace(
            resolution,
            context_window=ContextWindowResolution(window, "server metadata", True),
        )
        send(constrained)
        # Counterfactual accounting only: do not send or use this owner-free request.
        owner_free = gateway.prepare_chat_request(
            constrained,
            [{"role": "system", "content": config.system_prompt}, messages[0]],
            apply_safety_window=False,
        )
        owner_free_remaining = (
            owner_free.capacity.effective_input_ceiling_tokens
            - owner_free.accounting.total_input_tokens
        )
        assert cost <= min(4096, owner_free_remaining // 2)
        assert sent[-1].continuation_groups == (group,)
        assert (
            sent[-1].accounting.total_input_tokens
            == roomy.accounting.total_input_tokens
        )
        assert 0 < review.verifier_input_tokens < cost * 2
        assert messages[1][CONTINUATION_OWNER_KEY] == "owner"
        with pytest.raises(ReviewRefused, match="review_context_reduction_blocked"):
            review.read(prepared.review_id, view="batch")
        assert ledger.request_gaps == ["review_context_reduction_blocked"]
        assert not ledger.dispositions and not ledger.delivered and not ledger.pages
        assert ledger.cursor is None and not ledger.positions
        review.settle_review_run(
            review.actor.run_id, terminal_status="error", receipt_id="terminal"
        )
    with use_run_actor(c.actor):
        expanded = c.primary.expand_dispositions(prepared.review_id)
    assert expanded[0]["disposition"]["reason"] == "verification_incomplete"
    assert expanded[0]["original"]["trigger"] == "ordinary bounded claim " * 150
