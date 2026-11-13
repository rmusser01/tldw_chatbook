"""The per-run operations of one ``AgentService._run_one`` call.

``_run_one`` used to define these as closures over its own locals
(TASK-36950.3, architecture review 2026-10-25 §F5). They are methods of
:class:`AgentRunTools` now, and the per-run state they shared is its
attributes, set by ``_run_one`` at the point where the closures used to be
defined. The method bodies are the closure bodies: a closure's ``self`` (the
service, which ``_run_one`` rebinds for a code-review child) is
``self.service``, and each local it shared is ``self.<name>``.

Three kinds of state are not plain snapshots:

* ``sub_agent_spawns`` and ``child_started`` are counters the methods
  update; ``_run_one``'s own code (the review capacity probe and the skill
  branch of ``invoke_tool``) reads ``sub_agent_spawns`` through this object.
* ``hook_run`` and ``progress_reader`` are assigned by ``_run_one`` after
  this object is built; it sets the attributes where it assigns them, before
  the loop can dispatch a tool.
* ``_launch_fleet_child`` is wrapped per instance in ``core_operation`` over
  the run's own database, as the closure was decorated at definition time.

``agent_service._setting``, ``AgentConfig``, ``load_agents_routing_config``
and ``build_first_request_schema_plan`` are read through ``agent_service``
at call time, because tests monkeypatch them on that module.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, Literal
from uuid import uuid4

from tldw_chatbook.Provider_Contract.sampling_params import params_to_tuple

from . import agent_service
from .agent_models import (
    AGENT_KIND_SUBAGENT,
    MAX_STEERING_CHARS,
    RUN_DONE,
    RUN_ERROR,
    SPAWN_TOOL_NAME,
    STEERING_SOURCE_REDIRECT,
    STEERING_SOURCE_SUPERVISOR,
    STEP_TOOL_CALL,
    TERMINAL_RUN_STATUSES,
    PluginContextText,
    SpawnAdmissionRefusal,
    SpawnAdmissionReplay,
    ToolResult,
    WorkOrigin,
    contain_child_budget,
    definition_fingerprint as compute_definition_fingerprint,
)
from .agent_routing import RoutingError, resolve_spawn_target
from .agent_service import (
    _FLEET_POLL_SECONDS,
    _SPAWN_ECHO_CHARS,
    CHILD_MAX_WALL_SECONDS_KEY,
    DEFAULT_CHILD_MAX_WALL_SECONDS,
    DEFAULT_MAX_LIVE_SUBAGENTS,
    DEFAULT_SUBAGENTS_OUTLIVE_TURN,
    MAX_LIVE_SUBAGENTS_KEY,
    SUBAGENTS_OUTLIVE_TURN_KEY,
    TRUNCATION_NOTICE,
    _coerce_child_max_wall_seconds,
    _coerce_subagents_outlive_turn,
    coerce_max_live_subagents,
    get_internal_prompt,
)
from .automatic_work_budget import AutomaticWorkLimits, AutomaticWorkRefused
from .execution_capacity import CapacityRefused, current_execution_owner

if TYPE_CHECKING:
    from .agent_models import AgentConfig
    from .agent_service import AgentService
    from .execution_capacity import ExecutionOwner
    from .fleet_coordinator import FleetCoordinator, FleetHandle, PeerMessenger
    from .fleet_messages import MessageReader, MessageSender
    from .run_log import RunLogWriter


class AgentRunTools:
    """Fleet operations of one agent run, over the state that run shares.

    Built by ``AgentService._run_one`` once per run, after the run row
    exists and its tool surface is decided; its bound methods are what the
    run's ``LoopDeps`` dispatch to.
    """

    def __init__(
        self,
        service: AgentService,
        *,
        run_id: str,
        conversation_id: str,
        config: AgentConfig,
        api_endpoint: str,
        should_cancel: Callable[[], bool],
        fleet: FleetCoordinator | None,
        message_chain_id: str,
        started: float,
        writer: RunLogWriter,
        parent_instruction_snapshot: Any,
        parent_raw_provider: str,
        continuation_durability: Literal["persistent", "ephemeral"],
        chain_id: str,
        request_worktree_merge_confirm: Callable[[dict], dict] | None,
        review_owner: Any,
        owner: ExecutionOwner | None,
        message_automatic: Any,
        progress_sender: MessageSender | None,
        peer_messenger: PeerMessenger | None,
        drain_mailbox: Callable[[], list[tuple[str, str]]] | None,
        drain_mailbox_with_causes: (
            Callable[[], list[tuple[str, str, str | None]]] | None
        ),
        seeded_steering_with_causes: tuple[tuple[str, str, str | None], ...],
    ) -> None:
        from tldw_chatbook.Backup_Recovery.participants import core_operation

        self.service = service
        self.run_id = run_id
        self.conversation_id = conversation_id
        self.config = config
        self.api_endpoint = api_endpoint
        self.should_cancel = should_cancel
        self.fleet = fleet
        self.message_chain_id = message_chain_id
        self.started = started
        self.writer = writer
        self.parent_instruction_snapshot = parent_instruction_snapshot
        self.parent_raw_provider = parent_raw_provider
        self.continuation_durability = continuation_durability
        self.chain_id = chain_id
        self.request_worktree_merge_confirm = request_worktree_merge_confirm
        self.review_owner = review_owner
        self.owner = owner
        self.message_automatic = message_automatic
        self.progress_sender = progress_sender
        self.peer_messenger = peer_messenger
        self.drain_mailbox = drain_mailbox
        self.drain_mailbox_with_causes = drain_mailbox_with_causes
        self.seeded_steering = list(seeded_steering_with_causes)
        self.sub_agent_spawns = 0
        self.child_started = False  # this spawn call left a possibly-run child charged
        # Handles THIS run started, in spawn order. The coordinator is
        # deliberately not run-scoped (PR 3a wants children that outlive
        # their spawning turn), so "all my sub-agents" has to be tracked
        # here rather than read off `snapshot()` -- otherwise a long-lived
        # injected coordinator would show, and make this run wait on,
        # another run's children.
        self.my_handle_ids: list[str] = []
        self.durable_handle_ids: dict[str, str] = (
            fleet.durable_handle_map() if fleet is not None else {}
        )
        # Assigned by _run_one where it assigns its own locals of these names.
        self.hook_run = None
        self.progress_reader: MessageReader | None = None
        # Keep this exact parent repository operation admitted until the child
        # owns its independent scope or the parent finishes refused-launch cleanup.
        self._launch_fleet_child = core_operation(service.db)(self._launch_fleet_child)

    def _reserve_child_execution(self, child_kwargs) -> tuple[
        ExecutionOwner | None, str | None, ToolResult | None
    ]:
        automatic_work = self.service._automatic_work
        reservation_id = None
        child_owner = None
        try:
            if automatic_work is not None:
                automatic_work.check()
                reservation_id = automatic_work.ledger.reserve(
                    automatic_work.chain_id,
                    reservation_id=uuid4().hex,
                    owner_id=automatic_work.owner_id,
                    kind="child_launch",
                    amount=1,
                    limits=AutomaticWorkLimits.from_settings(
                        automatic_work.policy_kind()
                    ),
                ).id
            child_owner = self.service.runtime_capacity.begin_execution(
                origin=self.service.work_origin,
                conversation_id=self.conversation_id,
                child=True,
            )
            tree = getattr(self.service, "_ordinary_tree", None)
            if tree is not None:
                try:
                    tree.attach(child_owner)
                    child_id = child_kwargs.setdefault(
                        "requested_run_id", uuid.uuid4().hex
                    )
                    keys = self.service.__dict__.setdefault("_ordinary_child_keys", {})
                    keys[child_owner] = child_id
                    with child_owner.activate():
                        prepare_legacy = getattr(
                            self.service, "_prepare_child_notification", None
                        )
                        if prepare_legacy is not None and self.fleet is not None:
                            prepare_legacy(child_id)
                        if self.hook_run is not None:
                            event = self.service._hooks_v2_lifecycle.event(
                                "SubagentStop",
                                run_id=self.run_id,
                                data={"child_run_id": child_id},
                                initiator="child",
                            )
                            scope = self.service._hooks_v2_engine.prehold_event(event)
                            self.service.__dict__.setdefault("_ordinary_child_stops", {})[
                                child_id
                            ] = (event, scope)
                except BaseException:
                    self.service._finish_execution(child_owner)
                    raise
            return child_owner, reservation_id, None
        except Exception as exc:
            if reservation_id is not None:
                automatic_work.ledger.release(
                    reservation_id, owner_id=automatic_work.owner_id
                )
            self.sub_agent_spawns -= 1
            if isinstance(exc, AutomaticWorkRefused):
                return (
                    None,
                    None,
                    SpawnAdmissionRefusal(
                        ok=False, error=f"automatic sub-agent launch refused: {exc}"
                    ),
                )
            if not isinstance(exc, CapacityRefused):
                raise
            return (
                None,
                None,
                SpawnAdmissionRefusal(
                    ok=False,
                    error=(
                        f"runtime sub-agent limit reached ({exc}); wait for running "
                        "agents and cleanup to finish, then retry"
                    ),
                ),
            )

    def _launch_fleet_child(
        self,
        spawn_task,
        agent_name,
        child_kwargs,
        isolation=None,
        *,
        definition_wall_seconds=None,
        resumed_managed=None,
        managed_resume_ceiling=None,
    ):
        child_owner, child_reservation, failure = self._reserve_child_execution(child_kwargs)
        if failure is not None:
            return None, failure
        launched = False
        try:
            handle, failure = self.service._launch_prepared_child(
                spawn_task,
                agent_name,
                child_kwargs,
                isolation,
                fleet=self.fleet,
                run_id=self.run_id,
                message_chain_id=self.message_chain_id,
                hook_run=self.hook_run,
                should_cancel=self.should_cancel,
                my_handle_ids=self.my_handle_ids,
                durable_handle_ids=self.durable_handle_ids,
                fleet_cancels=self.service._fleet_cancels,
                fleet_threads=self.service._fleet_threads,
                child_owner=child_owner,
                child_reservation=child_reservation,
                definition_wall_seconds=definition_wall_seconds,
                resumed_managed=resumed_managed,
                managed_resume_ceiling=managed_resume_ceiling,
            )
            launched = handle is not None
            return handle, failure
        finally:
            if not launched:
                self.sub_agent_spawns -= 1

    def initialize_child_hooks(self, child_kwargs, child_task, *, inline=False):
        if self.hook_run is None:
            return None
        child_config = child_kwargs["config"]
        from .hooks_v2.lifecycle import narrow_child
        from .hooks_v2.validation import CAPS
        from .tool_catalog import ToolDefinitionSnapshot

        child_plan = agent_service.build_first_request_schema_plan(
            self.service.registry,
            child_config.allowed_tools,
            child_config,
            child_kwargs["api_endpoint"],
            child_kwargs["messages"],
            skill_file_enabled=bool(
                self.service.skill_file_bindings is not None
                and self.service.skill_file_bindings.authorized
            ),
            install_skill_enabled=False,
            run_skill_script_enabled=self.service._run_skill_script_tool is not None,
            run_log_active=False,
            agent_kind=AGENT_KIND_SUBAGENT,
            reporting_available=bool(self.fleet is not None and not inline),
            peers_available=bool(self.fleet is not None and not inline),
        )
        tool_ids = {
            name: self.service.registry.snapshot_for_hook(name).tool_id
            for name in child_config.allowed_tools
        }
        tool_ids.update(
            {
                schema.name: ToolDefinitionSnapshot.from_schema(
                    schema, "runtime", 0, id(schema)
                ).tool_id
                for schema in child_plan.runtime_schemas
            }
        )
        child_scope = self.service._hooks_v2_lifecycle.open_scope(parent=self.run_id)
        event = self.service._hooks_v2_lifecycle.event(
            "SubagentStart",
            run_id=self.run_id,
            turn_id=self.service._hooks_v2_turn_id,
            data={
                "child_task": child_task,
                "tool_ids": list(tool_ids.values()),
                "budget_caps": {
                    key: getattr(child_config.budget, key) for key in CAPS
                },
                **({"model": child_config.model} if child_config.model else {}),
                **(
                    {"provider": child_config.provider}
                    if child_config.provider
                    else {}
                ),
            },
            initiator="child",
        )
        try:
            result = self.service._hooks_v2_engine._sync(
                self.service._hooks_v2_lifecycle.fire(
                    event, child_scope, current=lambda: not self.should_cancel()
                )
            )
            child_config, selected_ids = narrow_child(
                child_config, result, tool_ids
            )
            child_kwargs["config"] = child_config
            child_kwargs["hook_tool_ids"] = selected_ids
            self.service._hooks_v2_lifecycle.context.refresh_sources(child_scope)
            child_kwargs["messages"] += list(
                self.service._hooks_v2_lifecycle.context.blocks(child_scope, "child")
            )
        except Exception:  # noqa: BLE001 -- hook boundary
            self.sub_agent_spawns -= 1
            return ToolResult(
                ok=False,
                error="sub-agent initialization refused",
                dispatch_state="settled",
            )
        finally:
            self.service._hooks_v2_lifecycle.close_scope(child_scope)
        return None

    def _spawn(
        self,
        spawn_task: str | None = None,
        *,
        review_id: str | None = None,
        review_slot_id: str | None = None,
        allowed_tools: tuple[str, ...] | None = None,
        agent: str | None = None,
        inline: bool = False,
        spawn_step_index: int | None = None,
        isolation: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> ToolResult:
        # Task-12 review Finding 2: this closure is THE single spawn
        # path -- the loop calls it directly for the native
        # spawn_subagent tool (agent_runtime.py), and invoke_tool's
        # skill branch below calls it via skill_runner.run. Gating and
        # incrementing the shared counter HERE, before any child run is
        # created, enforces one combined sub-agent ceiling across both
        # paths regardless of call order. (Previously each path checked
        # its own independent counter -- the loop's own `spawned` and
        # this service's now-removed `skill_spawns` -- so an operator
        # ceiling of 1 could permit 2 sub-agent runs.) The loop's own
        # counter stays untouched as a redundant secondary bound that
        # is never reached first.
        #
        # Fleet spec §4: the skill path (allowed_tools override) and
        # the named-definition path are disjoint by construction --
        # skills never pass `agent`.
        #
        # A structural invariant, not input validation: unreachable in
        # production today (both call sites are internal and neither
        # can supply both kwargs at once). It is an explicit `raise`
        # rather than an `assert` because an `assert` is STRIPPED
        # under `python -O`, and the failure it guards is silent --
        # `resolved.tool_allowlist` would intersect against the
        # skill's own narrowed override below, quietly producing a
        # child with neither party's intended allow-list. A future
        # caller that gets this wrong must fail loudly in every
        # interpreter mode, not just a non-optimised one.
        if agent and allowed_tools is not None:
            raise ValueError(
                "spawn(): `agent` and `allowed_tools` are mutually "
                "exclusive -- a named agent definition supplies its own "
                "tool allow-list, so passing both leaves the child's "
                "permissions ambiguous."
            )
        review_claim = None
        scoped_service = None
        if review_id is not None or review_slot_id is not None:
            if (
                not review_id
                or not review_slot_id
                or spawn_task is not None
                or agent is not None
                or allowed_tools is not None
                or provider is not None
                or model is not None
                or isolation is not None
                or inline
            ):
                return SpawnAdmissionRefusal(ok=False, error="invalid_review_spawn")
            from .code_review_models import ReviewRefused
            from .code_review_tool_provider import CodeReviewToolProvider

            owner = self.service.registry.resolve_owner_for_name("code_review_prepare")
            if owner is None or type(owner[1]) is not CodeReviewToolProvider:
                return SpawnAdmissionRefusal(
                    ok=False, error="stage_grant_unavailable"
                )
            scoped_service = owner[1].service
            try:
                review_claim = scoped_service.claim_review_slot(
                    review_id, review_slot_id
                )
                if review_claim.receipt is not None:
                    return SpawnAdmissionReplay(
                        **dataclasses.asdict(review_claim.receipt)
                    )
                if not review_claim.launch_allowed:
                    return SpawnAdmissionRefusal(
                        ok=False, error="review_launch_pending"
                    )
                slot = review_claim.slot
                override = slot.choice.destination
                resolve_spawn_target(
                    self.service._app_config,
                    parent_provider=self.parent_raw_provider or self.api_endpoint,
                    parent_model=self.config.model,
                    preset=dataclasses.replace(
                        slot.definition,
                        provider=slot.target.provider,
                        model=slot.target.model,
                    ),
                    override_provider=slot.target.provider if override else "",
                    override_model=slot.target.model if override else "",
                    routing=agent_service.load_agents_routing_config(),
                )
                refusal = scoped_service.destination_check(slot.target)
                if refusal:
                    raise ReviewRefused(refusal)
                inline = (
                    self.fleet is None
                    or coerce_max_live_subagents(
                        agent_service._setting(MAX_LIVE_SUBAGENTS_KEY, DEFAULT_MAX_LIVE_SUBAGENTS)
                    )
                    == 1
                )
                if slot.role == "verifier":
                    review_claim.binding = review_claim.binding.model_copy(
                        update={
                            "check_feedback_mode": "unavailable_inline"
                            if self.fleet is None or inline
                            else "primary_handoff"
                        }
                    )
                spawn_task = scoped_service.launch_task(review_claim)
            except (ReviewRefused, RoutingError) as exc:
                return SpawnAdmissionRefusal(
                    ok=False,
                    error=getattr(
                        exc, "reason", getattr(exc, "code", "review_launch_refused")
                    ),
                )
        resolved = review_claim.slot.definition if review_claim else None
        if agent:
            # PR3a-1 Task 6a, recorded because the Task 6 audit got
            # this half-right and the half it missed is the reachable
            # one. `self._turn_definitions` is per-TURN state read HERE
            # AT CALL TIME, and `run_turn` replaces it for the next
            # turn -- the exact shape Task 3 found the run-log writer
            # in. The audit called it unreachable "because the lookup
            # sits behind `max_subagents > 0`": that is true of the
            # SPAWN SCHEMA BUILD (`_make_tool_schemas`), not of this
            # closure body, which no budget check guards. It is
            # unreachable only because BOTH outer gates hold -- a child
            # gets `max_subagents = 0` (`contain_child_budget` /
            # `clamp_child_budget`), so the spawn tool is never offered
            # to it and `spawn` is never dispatched on its behalf. If
            # either gate ever loosens, a SURVIVOR calling
            # `spawn(agent=...)` would resolve against the NEXT turn's
            # roster and leak that roster's agent names in the
            # "available: ..." error string below. Pinned by the
            # depth-1 `budget["max_subagents"] == 0` assertions in
            # `Tests/Agents/test_agent_models.py` and
            # `Tests/Agents/test_fleet_runtime.py`.
            resolved = next(
                (d for d in self.service._turn_definitions if d.name == agent),
                None,
            )
            if resolved is None:
                available = (
                    ", ".join(d.name for d in self.service._turn_definitions) or "none"
                )
                # Refused BEFORE the budget increment: a typo costs no
                # sub-agent slot (mirrors the loop's empty-task refusal).
                return ToolResult(
                    ok=False,
                    error=(f"unknown agent '{agent}'; available: {available}"),
                )
        # ADR-344 (Task 6): resolve WHERE the child runs before touching
        # budget or fleet capacity. A refusal is an admission refusal:
        # no child run, no slot consumed, the coded error back to the
        # supervisor model. `resolved` (the named preset, when given) is
        # an input here; the legacy preset `model`-only override below
        # is gone -- the resolver owns that path now.
        if review_claim is not None:
            target = review_claim.slot.target
            fallback_targets = review_claim.slot.fallbacks
            from .agent_routing import execution_provider_for_target
        else:
            try:
                target = resolve_spawn_target(
                    self.service._app_config,
                    # Raw identity first (custom-ep:<slug>): an inheriting
                    # child of a registry-endpoint parent resolves to the
                    # endpoint itself — snapshotting its registry base URL —
                    # not the flattened execution family (qodo PR-2651 High).
                    parent_provider=(self.parent_raw_provider or self.api_endpoint),
                    parent_model=self.config.model,
                    preset=resolved,
                    override_provider=provider,
                    override_model=model,
                    routing=agent_service.load_agents_routing_config(),
                )
            except RoutingError as err:
                return SpawnAdmissionRefusal(
                    ok=False, error=f"[{err.code}] ({err.level}) {err}"
                )
            from .agent_routing import (
                execution_provider_for_target,
                freeze_spawn_target,
                resolve_preset_fallback_targets,
            )

            try:
                fallback_targets = resolve_preset_fallback_targets(
                    self.service._app_config,
                    resolved,
                    primary_provider=target.provider,
                    primary_model=target.model,
                )
                if fallback_targets:
                    target = freeze_spawn_target(
                        self.service._app_config,
                        target,
                        selected_endpoint=(
                            self.config.base_url
                            if target.provider
                            == (self.parent_raw_provider or self.api_endpoint)
                            else None
                        ),
                    )
            except RoutingError as err:
                return SpawnAdmissionRefusal(
                    ok=False, error=f"[{err.code}] ({err.level}) {err}"
                )
        child_execution_provider = (
            target.execution_provider
            or execution_provider_for_target(self.service._app_config, target.provider)
        )
        if self.sub_agent_spawns >= self.config.budget.max_subagents:
            return ToolResult(
                ok=False,
                error="sub-agent budget exhausted",
                # The runtime spawn callback ran; only child admission failed.
                dispatch_state="settled",
            )
        self.sub_agent_spawns += 1
        child_kwargs, child_budget = self.service._compose_native_child(
            spawn_task, config=self.config, api_endpoint=self.api_endpoint,
            parent_raw_provider=self.parent_raw_provider, fleet=self.fleet, inline=inline,
            started=self.started, resolved=resolved, target=target,
            fallback_targets=fallback_targets,
            child_execution_provider=child_execution_provider,
            allowed_tools=allowed_tools, isolation=isolation, run_id=self.run_id,
            conversation_id=self.conversation_id,
            continuation_durability=self.continuation_durability,
            chain_id=self.chain_id, writer=self.writer, spawn_step_index=spawn_step_index,
            child_sequence=self.sub_agent_spawns,
            startup_instruction_snapshot=self.parent_instruction_snapshot,
            review_claim=review_claim,
        )
        if review_claim is not None:
            child_kwargs.update(
                review_launch=review_claim, review_service=scoped_service
            )
        if isolation not in (None, "worktree"):
            # I2 (TASK-28238 P2 T7 final fix wave): the model-supplied
            # value is an unenforced JSON-fence string, not a schema
            # enum -- without this, anything other than the literal
            # "worktree" (a typo, "sandbox", "work-tree", ...) fell
            # through every isolation check below and silently
            # launched a NORMAL shared-tree child with a success
            # message, the opposite of what was asked. No child was
            # created, so this costs no spawn slot -- same rule as the
            # cap/unknown-agent/no_fleet refusals.
            self.sub_agent_spawns -= 1
            return ToolResult(
                ok=False,
                error=(
                    f'unknown isolation "{isolation}" [invalid_isolation]; '
                    'valid values: "worktree" or omit'
                ),
            )
        if isolation == "worktree" and (self.fleet is None or inline):
            # TASK-28238 P2 T4 (review fix): the INLINE path never
            # reaches `_launch_fleet_child`, so it can never create or
            # admit a worktree -- silently falling into it here would
            # share the tree despite the model asking for isolation.
            # No child was created, so this costs no spawn slot --
            # same rule as the cap/unknown-agent refusals above.
            self.sub_agent_spawns -= 1
            from tldw_chatbook.Agents.agent_worktree import (
                unsupported_execution_boundary,
            )

            refusal = unsupported_execution_boundary()
            return ToolResult(
                ok=False,
                error=(
                    f"worktree isolation refused [{refusal.reason_code}]: "
                    f"{refusal.message}"
                ),
            )
        failure = self.initialize_child_hooks(
            child_kwargs, child_kwargs["task"], inline=inline
        )
        if failure is not None:
            return failure
        if self.fleet is None or inline:
            # -- INLINE path: byte-identical to every release before
            # PR2a. Kept, not merely tolerated: with no fleet there is
            # no wait_agents, so a handle id would be unredeemable and
            # the child's work simply lost.
            #
            # `inline` takes this branch even WITH a live fleet, and is
            # how a SKILL call keeps its contract (PR2a Task 6.5). See
            # `invoke_tool`'s skill branch for why the caller, not the
            # runner, decides.
            #
            # Why `review_state_scope(run_id)` is still sound here even
            # when fleet siblings are running RIGHT NOW: the scope
            # snapshots and restores only THIS parent's slice -- the
            # restore rewrites exactly the keys where `key[0] ==
            # run_id`, under the provider's own `_decisions_lock` (see
            # `MCPToolProvider.stamp_scope`). A concurrent sibling
            # stamps under its OWN run id, so it is never in the
            # snapshot and never in the rewritten set; and the parent
            # itself is BLOCKED inside this call for the whole window,
            # so it cannot stamp anything the restore would roll back.
            # This is precisely the reasoning that does NOT hold on the
            # threaded path -- there the scope would be entered around
            # a child while the parent keeps running and stamping, so
            # Task 6 dropped it there. Writing it down because the two
            # branches now differ deliberately, not accidentally.
            child_owner, child_reservation, failure = self._reserve_child_execution(child_kwargs)
            if failure is not None:
                return failure
            try:
                scope = (
                    self.service.review_state_scope(self.run_id)
                    if self.service.review_state_scope
                    else contextlib.nullcontext()
                )
                with scope:
                    self.child_started = True  # proven refusals below still return
                    _child_id, child_outcome = self.service._run_owned(
                        execution_owner=child_owner,
                        automatic_child_reservation=child_reservation,
                        finish_root=False,
                        should_cancel=self.should_cancel,
                        **child_kwargs,
                    )
                if self.hook_run is not None:
                    self.service._install_child_stop(_child_id, child_outcome.status, self.run_id, self.should_cancel)
            except Exception as exc:
                from .code_review_models import ReviewRefused

                if (
                    review_claim is not None
                    and review_claim.run_id is None
                    and isinstance(exc, ReviewRefused)
                ):
                    # Inline attachment precedes model/tool execution too.
                    # A proven refusal may unwind; an uncertain failure may not.
                    if child_owner.run_id is not None:
                        self.service._set_terminal_status(
                            child_owner.run_id, RUN_ERROR, result=exc.reason
                        )
                    self.sub_agent_spawns -= 1
                    return SpawnAdmissionRefusal(ok=False, error=exc.reason)
                if child_owner.run_id is None:
                    self.sub_agent_spawns -= 1
                    return SpawnAdmissionRefusal(
                        ok=False,
                        error=f"could not start sub-agent ({type(exc).__name__}); retry when available",
                    )
                if review_claim is not None and review_claim.run_id is not None:
                    self.service._set_terminal_status(child_owner.run_id, RUN_ERROR)
                    self.service._settle_review_capture(scoped_service, child_owner.run_id)
                raise
            finally:
                try:
                    if child_reservation is not None:
                        self.service._automatic_work.ledger.release(
                            child_reservation,
                            owner_id=self.service._automatic_work.owner_id,
                        )
                finally:
                    self.service._finish_execution(child_owner)
            text = child_outcome.final_text
            cap = self.config.budget.max_subagent_result_chars
            if len(text) > cap:
                text = text[:cap] + TRUNCATION_NOTICE
            if review_claim is not None:
                self.service._settle_review_capture(scoped_service, _child_id)  # 2nd try
                receipt_text = (
                    f"sub-agent {child_outcome.status}; run {_child_id}; "
                    f"review {review_id}; slot {review_slot_id}; read originals with code_review_read"
                )
                review_claim.receipt = ToolResult(
                    ok=child_outcome.status == RUN_DONE,
                    content=receipt_text
                    if child_outcome.status == RUN_DONE
                    else "",
                    error=receipt_text if child_outcome.status != RUN_DONE else "",
                    dispatch_state="settled",
                )
                return review_claim.receipt
            if child_outcome.status != RUN_DONE:
                return ToolResult(
                    ok=False, error=f"sub-agent {child_outcome.status}: {text}"
                )
            return ToolResult(ok=True, content=text)

        # -- FLEET path: register, launch, return a handle -- via the
        # shared reserve->Event->thread->handle tail (PR3b Task 4
        # extracted it verbatim so the continuation path launches
        # through the exact same machinery).
        handle, failure = self._launch_fleet_child(
            child_kwargs["task"],
            (resolved.name if resolved else None),
            child_kwargs,
            isolation,
            definition_wall_seconds=(
                child_budget.max_wall_seconds
                if resolved is not None and resolved.max_wall_seconds is not None
                else None
            ),
        )
        if failure is not None:
            return failure
        if review_claim is not None:
            review_claim.receipt = ToolResult(
                ok=True,
                content=(
                    f"started {handle.handle_id}; review {review_id}; slot {review_slot_id}"
                ),
            )
            return review_claim.receipt
        snippet = spawn_task[:_SPAWN_ECHO_CHARS]
        return ToolResult(
            ok=True,
            content=(
                f"started {handle.handle_id}: {snippet}\n"
                "It is running now. Call wait_agents to collect its "
                "result before you answer."
            ),
        )

    def spawn(self, *args, **kwargs):
        scoped = "review_id" in kwargs or "review_slot_id" in kwargs
        if scoped and (
            args
            or set(kwargs) != {"review_id", "review_slot_id"}
            or any(
                not isinstance(kwargs.get(key), str) or not kwargs[key].strip()
                for key in ("review_id", "review_slot_id")
            )
        ):
            return SpawnAdmissionRefusal(ok=False, error="invalid_review_spawn")
        spawns_before, result, self.child_started = self.sub_agent_spawns, None, False
        try:
            result = self._spawn(*args, **kwargs)
            return result
        finally:
            # An exception (result None) returns the slot only if no child may
            # have run (the ceiling counts started children); restore, as some
            # paths gave it back. It also releases an unaccepted claim.
            if result is None and not self.child_started:
                self.sub_agent_spawns = spawns_before
            if (
                (result is None or not result.ok)
                and getattr(result, "error", "") != "review_launch_pending"
                and kwargs.get("review_id")
                and kwargs.get("review_slot_id")
            ):
                owner = self.service.registry.resolve_owner_for_name("code_review_prepare")
                if owner is not None:
                    from .code_review_models import ReviewRefused

                    try:
                        owner[1].service.release_unaccepted_claim(
                            kwargs["review_id"], kwargs["review_slot_id"]
                        )
                    except ReviewRefused:
                        # A missing or revoked scope has nothing safe to
                        # release; preserve the original admission refusal.
                        pass

    def wait_agents(self, ids: list[str] | None = None) -> ToolResult:
        """Block until the named (or all) children finish; collect them.

        Bounded by the parent's own remaining wall-clock and polling
        ``should_cancel`` every ``_FLEET_POLL_SECONDS``, because this
        is dispatched IN-LOOP rather than through ``invoke_tool``'s
        per-call daemon wrapper (see ``LoopDeps.wait_agents``).

        Args:
            ids: Handle ids to wait for, or ``None`` for every child
                this run has started.

        Returns:
            One entry per child -- handle id, agent, terminal status,
            and result -- each capped at
            ``max_subagent_result_chars`` and the whole thing
            additionally budgeted to ``max_tool_result_chars`` split
            evenly across the children returned, so five 4000-char
            results are shortened fairly here instead of being cut
            mid-result by the history-append seam. Never raises.
        """
        if self.fleet is None:  # pragma: no cover — not wired without a fleet
            return ToolResult(
                ok=False, error="wait_agents: this run has no sub-agents"
            )
        known = {
            handle.handle_id: handle
            for handle in (self.fleet.get(handle_id) for handle_id in self.my_handle_ids)
            if handle is not None
        }
        if not known:
            return ToolResult(
                ok=False,
                error=(
                    "wait_agents: no sub-agents have been started; call "
                    "spawn_subagent first"
                ),
            )
        if ids is None:
            targets = list(known)
        else:
            targets = [hid for hid in dict.fromkeys(ids) if hid in known]
            unknown = [hid for hid in dict.fromkeys(ids) if hid not in known]
            if unknown:
                return ToolResult(
                    ok=False,
                    error=(
                        "wait_agents: unknown sub-agent id(s): "
                        + ", ".join(unknown)
                        + ". Known ids: "
                        + ", ".join(known)
                    ),
                )
        note = ""
        stopped_children = False
        # See `_settle_fleet`: a real-time bound runs alongside the
        # budget deadline so an injected (possibly frozen) clock can
        # never turn this into an unbounded wait.
        wall_deadline = time.monotonic() + self.config.budget.max_wall_seconds
        while True:
            pending = self.service._pending_handles(self.fleet, targets)
            if not pending:
                break
            if self.should_cancel():
                # PR3b Task 5 (spec Sec 8): a user Stop releases the
                # WAIT, not the children. With the outlive default ON
                # the pending children keep working in the background
                # (the settle spares them too -- `_surviving_handles`);
                # under the kill switch this branch is the pre-Task-5
                # cancel, byte-identical. Read here, at the moment of
                # the Stop, matching `_surviving_handles`' settle-time
                # read -- the two must agree on the SAME Stop. The
                # note promises only what is true on EVERY path from
                # here (Qodo #1808 finding 4): `pending` is non-empty
                # by the loop guard, so children genuinely continue --
                # but result DELIVERY depends on the auto-wake key (a
                # completion is only ever marked when that is off),
                # so the copy does not promise delivery.
                if _coerce_subagents_outlive_turn(
                    agent_service._setting(
                        SUBAGENTS_OUTLIVE_TURN_KEY,
                        DEFAULT_SUBAGENTS_OUTLIVE_TURN,
                    )
                ):
                    note = (
                        "\n(The run was cancelled; sub-agents continue "
                        "in the background.)"
                    )
                    break
                note = "\n(The run was cancelled; sub-agents were stopped.)"
                self.service._cancel_fleet_handles(pending)
                stopped_children = True
                break
            if (
                self.service.clock() - self.started >= self.config.budget.max_wall_seconds
                or time.monotonic() >= wall_deadline
            ):
                note = (
                    "\n(This run's time budget ran out; sub-agents still "
                    "working were stopped.)"
                )
                self.service._cancel_fleet_handles(pending)
                stopped_children = True
                break
            if self.review_owner is not None:
                from .code_review_models import ReviewRefused

                try:
                    notice = self.review_owner[1].service.pending_check_for_wait(
                        self.run_id,
                        tuple(
                            known[hid].run_id
                            for hid in targets
                            if known[hid].run_id
                        ),
                    )
                except ReviewRefused:
                    notice = None
                if notice is not None:
                    terminal = [hid for hid in targets if hid not in pending]
                    payload = {
                        "status": "check_requested",
                        "review_id": notice.review_id,
                        "proposal_id": notice.proposal_id,
                        "child_run_id": notice.child_run_id,
                        "completed": "",
                    }
                    limit = self.review_owner[1].service.result_limit
                    overhead = len(json.dumps(payload, separators=(",", ":")))
                    if overhead > limit:
                        from .agent_request_execution import end_review_checks

                        end_review_checks(
                            self.service,
                            self.run_id,
                            "tool_check_notice_transport_unavailable",
                            unstarted_only=True,
                        )
                        return ToolResult(
                            ok=False, error="check_notice_transport_unavailable"
                        )
                    if terminal:
                        # JSON escaping can expand each character six-fold.
                        remaining = (limit - overhead) // 6
                        if remaining:
                            payload["completed"] = self.service._format_wait_result(
                                self.fleet,
                                terminal,
                                dataclasses.replace(
                                    self.config.budget, max_tool_result_chars=remaining
                                ),
                                "",
                            )
                    return ToolResult(
                        ok=True, content=json.dumps(payload, separators=(",", ":"))
                    )
            time.sleep(_FLEET_POLL_SECONDS)
        if stopped_children:
            # Give the children just cancelled a bounded chance to
            # unwind and record a real status before we report them.
            # Gated on an actual cancel (not the note): the Task 5
            # not-cancelling Stop branch above has nothing to drain,
            # and waiting out the grace for children that are not
            # stopping would hold the stopped turn open for nothing.
            self.service._drain_fleet_handles(self.fleet, targets)
        return ToolResult(
            ok=True,
            content=self.service._format_wait_result(self.fleet, targets, self.config.budget, note),
        )

    def check_agents(self) -> ToolResult:
        """Non-blocking status snapshot of every child of this run.

        PR3a-1 Task 6a: plus any child of an EARLIER turn that is
        still running. With a per-conversation coordinator a survivor
        stays in `fleet` after the turn that spawned it returned, and
        leaving it out of the one surface a supervisor can ask "what
        is still working?" is what the audit called an invisible
        agent -- the worst outcome for a feature whose whole point is
        background work. It is reported in its own labelled section
        rather than mixed into this run's own list, because the two
        differ in what the supervisor may do with them: `wait_agents`
        deliberately stays scoped to `my_handle_ids` (collecting a
        foreign child's RESULT into this turn's history is delivery,
        which is PR 3a-2's job, and blocking this turn on another
        turn's child would be worse still). Terminal foreign handles
        are never listed: they are somebody else's finished business.

        Returns:
            One compact line per child (handle id, agent, status,
            elapsed seconds, task snippet), or a plain sentence when
            nothing has been started yet. Never blocks, never raises.
        """
        if self.fleet is None:  # pragma: no cover — not wired without a fleet
            return ToolResult(
                ok=False, error="check_agents: this run has no sub-agents"
            )
        handles = [
            handle
            for handle in (self.fleet.get(handle_id) for handle_id in self.my_handle_ids)
            if handle is not None
        ]
        mine = set(self.my_handle_ids)
        others = [
            handle
            for handle in self.fleet.snapshot()
            if handle.handle_id not in mine
            and handle.status not in TERMINAL_RUN_STATUSES
        ]
        progress_note = ""
        if self.progress_reader is not None and self.progress_authorized():
            from .fleet_messages import MessageError

            try:
                progress_note = (
                    f"\nQueued progress: {self.progress_reader.pending_count()}"
                )
            except MessageError:
                pass
        if not handles and not others:
            return ToolResult(
                ok=True,
                content="No sub-agents have been started yet." + progress_note,
            )
        now = self.service.clock()

        def _line(handle: FleetHandle) -> str:
            end = handle.finished_at if handle.finished_at is not None else now
            elapsed = max(end - handle.started_at, 0.0)
            return (
                f"[{handle.handle_id}] {handle.agent or 'sub-agent'} — "
                f"{handle.status} ({elapsed:.1f}s) — "
                f"{handle.task[:_SPAWN_ECHO_CHARS]}"
            )

        lines = [_line(handle) for handle in handles]
        if others:
            if lines:
                lines.append("")
            lines.append(
                "Still running from an earlier turn (started by a "
                "previous message; wait_agents cannot collect these):"
            )
            lines.extend(_line(handle) for handle in others)
        return ToolResult(ok=True, content="\n".join(lines) + progress_note)

    def recover_current_worktree(self, handle_id: str, action: str) -> ToolResult:
        from .agent_worktree import WorktreeRefusal
        from .agent_worktree_recovery import recover_agent_worktree

        authority = self.service._worktree_repo_authority
        if authority is None:
            return ToolResult(
                ok=False,
                error="[source_authority_unavailable] Select a writable named repository binding.",
            )
        created = self.service._agent_worktrees.get(handle_id)
        handle = self.fleet.get(handle_id) if self.fleet is not None else None
        if (
            handle_id not in self.my_handle_ids
            or created is None
            or handle is None
            or handle.run_id != created.run_id
        ):
            return ToolResult(
                ok=False,
                error="[worktree_unavailable] This turn does not own that isolated child handle.",
            )

        def confirm(payload):
            return self.request_worktree_merge_confirm(
                {**payload, "handle_id": handle_id}
            )

        result = recover_agent_worktree(
            self.service.db,
            authority=authority,
            conversation_id=self.conversation_id,
            run_id=created.run_id,
            action=action,
            request_confirmation=confirm,
            should_cancel=self.should_cancel,
        )
        if isinstance(result, WorktreeRefusal):
            return ToolResult(
                ok=False, error=f"[{result.reason_code}] {result.message}"
            )
        return ToolResult(ok=True, content=result.message)

    def merge_agent_worktree_tool(
        self,
        handle_id: str, mode: str = "apply"
    ) -> ToolResult:
        return self.recover_current_worktree(handle_id, mode)

    def discard_agent_worktree_tool(self, handle_id: str) -> ToolResult:
        return self.recover_current_worktree(handle_id, "discard")

    def _resume_retained_child(
        self,
        retained, steer_text: str, spawn_step_index: int | None
    ) -> ToolResult:
        """Continuation (PR3b Task 4, spec SS6): a NEW run from a
        retained transcript.

        Order of operations mirrors spawn's own gauntlet so every
        refusal costs what spawn's would: definition re-resolution
        FIRST (a vanished definition costs no slot -- "a typo costs
        no sub-agent slot"), then the spawn-slot check/consume, then
        `_launch_fleet_child` (whose cap refusal and thread-start
        failure both unwind the slot).

        Ruling #1 on the definition: a still-existing one re-resolves
        to its CURRENT form -- the new row's fresh
        `definition_fingerprint` records the change, which is exactly
        what that audit column exists for; a deleted/disabled one
        refuses clearly, because silently downgrading the child to a
        generic spawn would change its behavior with no record.

        The seed is the retained coherent transcript, then any
        undelivered queued steering the retention claimed (ORIGINAL
        labels -- the user's entry stays the user's), then the new
        supervisor-labeled message. The new row's `parent_run_id` is
        THIS (the resuming) primary; `resumed_from_run_id` records
        the old run.
        """
        managed_resume_ceiling = None
        try:
            if retained.managed_required and (
                retained.managed_pin is None or self.service.managed_child_resume is None
            ):
                raise PermissionError("managed child pin unavailable")
            if self.service.managed_child_resume is not None:
                managed_resume_ceiling = self.service.managed_child_resume(retained)
        except (ValueError, PermissionError):
            return ToolResult(
                ok=False,
                error="send_to_agent: managed continuation authority no longer matches; spawn a new child",
            )
        resolved = None
        if retained.agent:
            resolved = next(
                (d for d in self.service._turn_definitions if d.name == retained.agent),
                None,
            )
            if resolved is None:
                available = (
                    ", ".join(d.name for d in self.service._turn_definitions) or "none"
                )
                return ToolResult(
                    ok=False,
                    error=(
                        f"send_to_agent: sub-agent "
                        f"'{retained.handle_id}' was spawned from "
                        f"agent definition '{retained.agent}', which "
                        f"no longer exists (or is disabled), so it "
                        f"cannot be resumed as it was. Spawn a fresh "
                        f"sub-agent instead (available agents: "
                        f"{available})."
                    ),
                )
        if self.sub_agent_spawns >= self.config.budget.max_subagents:
            # Spawn's own budget-refusal shape: a resume starts a NEW
            # run, so it costs a spawn slot like any other spawn.
            return ToolResult(
                ok=False,
                error=(
                    "send_to_agent: sub-agent budget exhausted -- "
                    "resuming a finished sub-agent starts a new run "
                    "and costs a spawn slot."
                ),
            )
        self.sub_agent_spawns += 1
        # A resumed child is a THREADED survivor candidate by
        # construction (it is launched through the fleet tail below),
        # so it gets `contain_child_budget`'s independent ceiling --
        # never the turn-scoped parent-remainder clamp.
        child_max_wall_seconds = _coerce_child_max_wall_seconds(
            agent_service._setting(CHILD_MAX_WALL_SECONDS_KEY, DEFAULT_CHILD_MAX_WALL_SECONDS)
        )
        child_budget = contain_child_budget(self.config.budget, child_max_wall_seconds)
        definition_bounds = [
            bound
            for bound in (
                resolved.max_wall_seconds if resolved is not None else None,
                retained.definition_wall_seconds,
            )
            if bound is not None
        ]
        if definition_bounds:
            child_budget = dataclasses.replace(
                child_budget,
                max_wall_seconds=min(
                    child_budget.max_wall_seconds, *definition_bounds
                ),
            )
        # Composition mirrors spawn's default path exactly (inherit
        # minus the spawn tool and any skill-tool names; a resolved
        # definition APPENDS instructions and INTERSECTS the
        # allow-list -- never grants). Duplicated deliberately rather
        # than extracted: the plan mandates only the launch TAIL be
        # shared (`_launch_fleet_child`); spawn's composition block
        # stays byte-identical where it is, and each site cites the
        # other.
        child_allowed_tools = tuple(
            n
            for n in self.config.allowed_tools
            if n != SPAWN_TOOL_NAME
            and not (
                self.service.skill_runner is not None and self.service.skill_runner.is_skill_tool(n)
            )
        )
        if retained.isolation == "worktree":
            # Finding 6 twin (Qodo round, TASK-28238 P2 T7): mirrors
            # spawn's identical exclusion -- a resumed isolated
            # child must not reinherit shell_exec/virtual_cli either.
            from .raw_shell_tool_provider import RAW_SHELL_TOOL_NAME
            from .virtual_cli_provider import VIRTUAL_CLI_TOOL_NAME

            child_allowed_tools = tuple(
                n
                for n in child_allowed_tools
                if n not in (RAW_SHELL_TOOL_NAME, VIRTUAL_CLI_TOOL_NAME)
            )
        child_system_prompt = get_internal_prompt("agents.subagent_system")
        # ADR-344 (Task 8): WHERE the resumed child runs comes from the
        # v21 snapshot the ORIGINAL run froze at spawn time
        # (`get_run_resolved_target`), never from a live re-resolution
        # -- editing a preset between runs retargets NEW spawns only,
        # and an edit to something invalid cannot break continuing an
        # existing child: the snapshot is not re-validated here (it was
        # validated when it was written), so `resolve_spawn_target`
        # does not run on this path at all. The DEFINITION still
        # re-resolves live below (Ruling #1: instructions/allow-list
        # follow the current form, fingerprinted on the new row) --
        # only the routing is pinned. A legacy row (NULL snapshot --
        # every run from before v21) takes the miss branch, which keeps
        # this path's pre-Task-8 behavior byte-identical: inherit the
        # parent's endpoint, take the definition's CURRENT model.
        snapshot = None
        fallback_state = None
        if retained.run_id is not None:
            try:
                snapshot = self.service.db.get_run_resolved_target(retained.run_id)
                fallback_state = self.service.db.get_run_fallback_state(retained.run_id)
            except Exception:  # noqa: BLE001 — fail closed on frozen route reads
                return SpawnAdmissionRefusal(
                    ok=False, error="could not read frozen child target"
                )
        if snapshot is not None:
            child_endpoint = snapshot["provider"]
            child_model = snapshot["model"] or self.config.model
            child_base_url = snapshot["base_url"]
            child_sampling_params = (
                params_to_tuple(json.loads(snapshot["params_json"]))
                if snapshot["params_json"]
                else ()
            )
        else:
            child_endpoint = self.api_endpoint
            child_model = self.config.model
            child_base_url = None
            child_sampling_params = ()
        plugin_instructions = ()
        if resolved is not None:
            from .agent_models import compose_agent_instructions

            child_system_prompt, child_allowed_tools, plugin_instructions = (
                compose_agent_instructions(
                    resolved, child_system_prompt, child_allowed_tools
                )
            )
            if snapshot is None and resolved.model:
                child_model = resolved.model
        child_config = agent_service.AgentConfig(
            model=child_model,
            # Children report provider-level faults against the same
            # provider the parent is using (review minor, 2026-08-31 --
            # previously they reported "unknown provider").
            provider=child_endpoint,
            execution_provider=(
                fallback_state["active_target"].get("execution_provider", "")
                if fallback_state
                else ""
            ),
            system_prompt=child_system_prompt,
            allowed_tools=child_allowed_tools,
            budget=child_budget,
            native_tools=self.config.native_tools,
            reasoning_replay=(
                self.config.reasoning_replay
                if child_model == self.config.model
                and child_endpoint == (self.parent_raw_provider or self.api_endpoint)
                and not (fallback_state and fallback_state["active_index"])
                else None
            ),
            workspace_context_note=self.config.workspace_context_note,
            personal_context_block=self.config.personal_context_block,
            response_reserve_tokens=self.config.response_reserve_tokens,
            # Snapshot-owned routing (ADR-344, Task 8); both sit at
            # their AgentConfig defaults on the legacy miss branch.
            base_url=child_base_url,
            sampling_params=child_sampling_params,
        )
        # Re-project the current preset once, retaining unrelated context.
        origins = {row["content"].checked_origins() for row in plugin_instructions}
        seed = [
            *plugin_instructions,
            *(
                dict(message)
                for message in retained.messages
                if not (
                    isinstance(message.get("content"), PluginContextText)
                    and message["content"].checked_origins() in origins
                )
            ),
        ]
        retained_steering = retained.steering_with_causes or tuple(
            (source, text, None) for source, text in retained.steering
        )
        resume_event_id = (
            self.service._resolved_control_event_id(
                self.run_id, spawn_step_index, STEP_TOOL_CALL
            )
            if spawn_step_index is not None
            else None
        )
        child_kwargs = dict(
            conversation_id=self.conversation_id,
            messages=seed,
            config=child_config,
            api_endpoint=child_endpoint,
            agent_kind=AGENT_KIND_SUBAGENT,
            task=retained.task,
            parent_run_id=self.run_id,
            spawn_event_id=resume_event_id,
            agent_definition=(resolved.name if resolved else None),
            definition_fingerprint=(
                compute_definition_fingerprint(resolved) if resolved else None
            ),
            continuation_durability=self.continuation_durability,
            run_log_writer=self.writer,
            startup_instruction_snapshot=self.parent_instruction_snapshot,
            resumed_from_run_id=retained.run_id,
            seeded_steering_with_causes=(
                *retained_steering,
                (STEERING_SOURCE_SUPERVISOR, steer_text, resume_event_id),
            ),
            # ADR-344 (Task 8): re-freeze the snapshot onto the NEW row
            # so a later continuation of THIS run reuses the same
            # target. The legacy miss branch leaves them NULL -- a
            # pre-v21 row's continuations keep re-resolving live, as
            # they always have.
            fallback_targets_json=fallback_state["targets_json"]
            if fallback_state
            else None,
            active_fallback_index=fallback_state["active_index"]
            if fallback_state
            else 0,
            resolved_provider=(
                fallback_state["original_target"]["provider"]
                if fallback_state
                else snapshot["provider"]
                if snapshot
                else None
            ),
            resolved_model=(
                fallback_state["original_target"]["model"]
                if fallback_state
                else child_model
                if snapshot
                else None
            ),
            resolved_base_url=(
                fallback_state["original_target"]["base_url"]
                if fallback_state
                else snapshot["base_url"]
                if snapshot
                else None
            ),
            resolved_params_json=(
                fallback_state["original_target"]["params_json"]
                if fallback_state
                else snapshot["params_json"]
                if snapshot
                else None
            ),
        )
        child_kwargs["spawn_parent_event_id"] = (
            child_kwargs["spawn_event_id"] or f"agent-run:{self.run_id}"
        )
        failure = self.initialize_child_hooks(child_kwargs, retained.task)
        if failure is not None:
            return failure
        handle, failure = self._launch_fleet_child(
            retained.task,
            (resolved.name if resolved else None),
            child_kwargs,
            # TASK-28238 P2 T7 Qodo round, finding 7 (was T4: a
            # literal `None` here silently dropped isolation on
            # resume -- `RetainedTranscript` now carries the
            # original child's isolation, recorded at retention time
            # off its `FleetHandle`). A resumed run is a NEW run_id,
            # so re-admitting a FRESH worktree here is correct, not
            # merely tolerated -- the existing `_admit_agent_
            # worktree` refusal machinery (non-git root, no local
            # provider, admit failure) already covers every way that
            # can fail.
            retained.isolation,
            definition_wall_seconds=(
                child_budget.max_wall_seconds if definition_bounds else None
            ),
            resumed_managed=retained,
            managed_resume_ceiling=managed_resume_ceiling,
        )
        if failure is not None:
            return failure
        queued = len(retained.steering)
        queued_note = (
            f" plus {queued} undelivered queued steering "
            f"entr{'y' if queued == 1 else 'ies'}"
            if queued
            else ""
        )
        return ToolResult(
            ok=True,
            content=(
                f"resumed {retained.handle_id} as a NEW run: started "
                f"{handle.handle_id}, seeded with its retained "
                f"transcript ({len(retained.messages)} messages"
                f"{queued_note}) and your message. The finished run "
                f"itself was not restarted. It is running now. Call "
                f"wait_agents to collect its result before you answer."
            ),
        )

    def send_to_agent(
        self,
        target_id: str, message: str, spawn_step_index: int | None = None
    ) -> ToolResult:
        """Queue a steering message for a LIVE child (PR3b Task 2).

        Spec SS6's supervisor path into Task 1's per-child mailbox.
        Text validation happens HERE, at the producer boundary --
        ``post_steering`` deliberately does not validate (Task 1's
        pinned decision), because each producer owes its caller its
        own refusal copy, which the mailbox's silent bool cannot
        carry.

        Resolution speaks BOTH id vocabularies over the WHOLE
        coordinator, not ``my_handle_ids``: a live survivor another
        turn's service spawned is steerable, because the mailbox
        lives on the conversation-lifetime coordinator and needs no
        per-service state -- deliberately unlike ``cancel_subagent``,
        whose retained-owner walk exists only because cancel Events
        are service-local. Handle id resolves FIRST (the coordinator
        minted it as the primary vocabulary -- spawn results,
        check_agents, and the panel rows all speak it), then a live
        handle's run id (the vocabulary completion notices speak); a
        pathological collision therefore lands on the handle-id
        owner.

        Steering never cancels (spec SS3 invariant 4): this posts to
        the mailbox and touches nothing else -- no cancel Event, no
        run row, no coordinator status.

        PR3b Task 4 (continuation): when the id is NOT live, the
        RETENTION store resolves next -- before the coordinator's
        terminal handles -- because `prune_terminal` drops those at
        every turn start while retention survives it (Task 2's
        concern (a)). A retained finished child is RESUMED (a new
        run, seeded; see `_resume_retained_child` above); a real
        finished child with nothing retained, or a run id only the
        database remembers, each get their own honest refusal. Recent
        pruned identities also survive without retaining payloads;
        unknown handles may be older than that bounded window.

        Args:
            target_id: A live child's handle id, or its run id --
                or a RETAINED finished child's (either vocabulary).
            message: The steering text (validated, then posted
                stripped; the drain point prepends the label -- or,
                for a resume, seeds the new run supervisor-labeled).

        Returns:
            ok with honest queued-plus-latency copy naming the
            resolved handle id (live), or the resumed-as-new-run copy
            (retained); a refusal with its own copy for an empty
            message, an oversize one, a finished-but-unretained
            child, a pre-restart run id, or an unknown id. Never
            raises.
        """
        if self.fleet is None:  # pragma: no cover — not wired without a fleet
            return ToolResult(
                ok=False, error="send_to_agent: this run has no sub-agents"
            )
        text = message.strip()
        if not text:
            return ToolResult(
                ok=False,
                error=(
                    "send_to_agent: the message is empty; there is "
                    "nothing to deliver. Put the steering text in "
                    "'message'."
                ),
            )
        if len(text) > MAX_STEERING_CHARS:
            return ToolResult(
                ok=False,
                error=(
                    f"send_to_agent: the message is too long "
                    f"({len(text)} chars; the cap is "
                    f"{MAX_STEERING_CHARS}). Shorten it and send it "
                    f"again."
                ),
            )
        handles = self.fleet.snapshot()
        live = [
            handle
            for handle in handles
            if handle.status not in TERMINAL_RUN_STATUSES
        ]
        target = next((h for h in live if h.handle_id == target_id), None) or next(
            (h for h in live if h.run_id == target_id), None
        )
        queued = False
        if target is not None:
            if spawn_step_index is not None:
                queued = self.fleet.post_steering_with_cause(
                    target.handle_id,
                    STEERING_SOURCE_SUPERVISOR,
                    text,
                    f"agent-step:{self.run_id}:{spawn_step_index}",
                )
            else:
                queued = self.fleet.post_steering(
                    target.handle_id, STEERING_SOURCE_SUPERVISOR, text
                )
        if target is not None and queued:
            return ToolResult(
                ok=True,
                content=(
                    f"Steering for {target.handle_id} queued; it will "
                    f"be delivered before its next model turn if it "
                    f"takes one. If it finishes first, follow up with "
                    f"send_to_agent to resume it when history is retained. If it "
                    f"is inside a long tool call it will see the "
                    f"message once that call returns. It was not "
                    f"cancelled or restarted."
                ),
            )
        if target is not None:
            # A still-live target refused admission because its queue
            # is full. A terminal race continues through retention below.
            current = self.fleet.get(target.handle_id)
            if current is not None and current.status not in TERMINAL_RUN_STATUSES:
                return ToolResult(
                    ok=False,
                    error=(
                        "send_to_agent: the steering queue is full; "
                        "wait for the child to consume queued messages "
                        "before retrying. This message was not queued."
                    ),
                )
            handles = self.fleet.snapshot()
            live = [
                handle
                for handle in handles
                if handle.status not in TERMINAL_RUN_STATUSES
            ]
        live_ids = ", ".join(h.handle_id for h in live) or "none"
        # PR3b Task 4: not live -- try the RETENTION store next, and
        # FIRST among the finished-child surfaces (Task 2's concern
        # (a)): `prune_terminal` drops terminal handles at every turn
        # start, so a finished child's ids may already be gone from
        # the coordinator's handle map -- but its transcript survives
        # in the retention store, keyed by both vocabularies (handle
        # id first, Task 2's pinned order). Retention is ATOMIC with
        # `finish` (one critical section), so a real retainable child
        # can never be observed terminal-but-unretained by this
        # lookup.
        retained = self.fleet.get_retained(target_id)
        if retained is not None:
            self.durable_handle_ids[retained.handle_id] = retained.run_id
            return self._resume_retained_child(retained, text, spawn_step_index)
        finished = next(
            (
                h
                for h in handles
                if h.handle_id == target_id or h.run_id == target_id
            ),
            None,
        )
        if finished is None:
            finished = self.fleet.get_pruned_identity(target_id)
        if finished is not None:
            # A REAL finished child with nothing retained: cancelled/
            # superseded are never retained (the user killed it / it
            # was replaced), and an oversize or evicted transcript is
            # gone too. Honest refusal, never the unknown-id copy.
            return ToolResult(
                ok=False,
                error=(
                    f"send_to_agent: sub-agent '{target_id}' has "
                    f"finished ({finished.status}) and no retained "
                    f"transcript is available for it (a cancelled or "
                    f"superseded child is never retained; an oversize "
                    f"or evicted transcript is not either), so it "
                    f"cannot be resumed. Spawn a fresh sub-agent for "
                    f"follow-up work. Live sub-agent ids: {live_ids}."
                ),
            )
        # A run id the DATABASE still knows but the coordinator does not:
        # an evicted identity or a child from before restart. Retention is
        # in-memory by design (spec SS6: cross-restart resurrection is
        # out of scope), so the honest answer names the real limit
        # instead of pretending the id is unknown. Scoped to THIS
        # conversation's sub-agent runs -- a foreign conversation's
        # run id stays an unknown id here. task-18601 part A (AC#2):
        # only agent_kind/conversation_id/status are read below --
        # get_run_metadata skips the step-log entirely.
        past = None
        try:
            past = self.service.db.get_run_metadata(target_id)
        except Exception:  # noqa: BLE001 — a read failure is "unknown"
            past = None
        if (
            past is not None
            and past.get("agent_kind") == AGENT_KIND_SUBAGENT
            and past.get("conversation_id") == self.conversation_id
            and past.get("status") in TERMINAL_RUN_STATUSES
        ):
            return ToolResult(
                ok=False,
                error=(
                    f"send_to_agent: run '{target_id}' has finished "
                    f"({past.get('status')}) and its transcript is no longer "
                    f"available -- retained transcripts live in memory "
                    f"and do not survive an app restart. Spawn a fresh "
                    f"sub-agent instead."
                ),
            )
        return ToolResult(
            ok=False,
            error=(
                f"send_to_agent: no sub-agent matches id "
                f"'{target_id}' (checked handle ids and run ids). "
                f"Older handle ids may have expired from memory; try "
                f"the run id from run history, or spawn a fresh sub-agent. "
                f"Live sub-agent ids: {live_ids}."
            ),
        )

    def drain_causal_steering(self) -> list[tuple[str, str, str | None]]:
        entries = list(self.seeded_steering)
        self.seeded_steering.clear()
        if self.drain_mailbox_with_causes is not None:
            entries.extend(self.drain_mailbox_with_causes())
        elif self.drain_mailbox is not None:
            entries.extend((source, text, None) for source, text in self.drain_mailbox())
        # Review F5 (defensive): today this closure is wired for
        # subagents only, but if seeded causal steering is ever extended
        # to a primary, a drain that bypassed _primary_drain_for would
        # leave the abort flag up forever -- every re-ask cut until
        # EMPTY_TURN_LIMIT. Clearing here keeps the invariant "a drain
        # that consumed a redirect entry lowers the flag" wiring-proof.
        if any(source == STEERING_SOURCE_REDIRECT for source, _t, _c in entries):
            flag = self.service._primary_redirect_flags.get(self.run_id)
            if flag is not None:
                flag.clear()
        return entries

    def progress_authorized(self) -> bool:
        # All external policy/capacity checks precede queue locking.
        if (
            self.should_cancel()
            or self.owner is None
            or current_execution_owner() is not self.owner
        ):
            return False
        snapshot = self.owner.capacity.snapshot()
        if snapshot.closed or not any(
            item.execution_id == self.owner.execution_id
            and item.run_id == self.run_id
            and not item.root_finished
            for item in snapshot.executions
        ):
            return False
        try:
            if (
                self.service.work_origin is WorkOrigin.AUTOMATIC
                and self.message_automatic is None
            ):
                return False
            if self.message_automatic is not None:
                self.message_automatic.check()
        except Exception:  # noqa: BLE001 -- authority failure must refuse
            return False
        return True

    def report_to_supervisor(self, args: dict) -> ToolResult:
        from tldw_chatbook.Backup_Recovery.participants import (
            run_finite_local_worker,
        )

        from .fleet_message_tools import (
            refused as refuse_progress,
        )
        from .fleet_message_tools import (
            report as report_progress,
        )

        if self.progress_sender is None or not self.progress_authorized():
            return refuse_progress()
        return run_finite_local_worker(report_progress, self.progress_sender, args)

    def list_peer_agents(self, args: dict) -> ToolResult:
        from .fleet_message_tools import list_peers, refused

        if self.peer_messenger is None or not self.progress_authorized():
            return refused()
        return list_peers(self.peer_messenger, args)

    def send_to_peer(self, args: dict) -> ToolResult:
        from .fleet_message_tools import refused, send_peer

        if self.peer_messenger is None or not self.progress_authorized():
            return refused()
        return send_peer(self.peer_messenger, args)

    def read_agent_messages(self, args: dict) -> ToolResult:
        from .fleet_message_tools import (
            collect as collect_progress,
        )
        from .fleet_message_tools import (
            refused as refuse_progress,
        )

        if self.progress_reader is None or not self.progress_authorized():
            return refuse_progress()
        return collect_progress(
            self.progress_reader, args, self.config.budget.max_tool_result_chars
        )
