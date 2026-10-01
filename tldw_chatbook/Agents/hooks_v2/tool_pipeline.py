"""One finite transformation pass bound to the host's exact tool definition."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace

from jsonschema import validators
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry
from referencing.exceptions import Unresolvable

from tldw_chatbook.Utils.timestamps import utc_now_iso

from .engine import HookEngine, HookEventOutcome
from .models import HookEvent
from .validation import parse_event


def freeze_candidate(arguments: dict) -> bytes:
    """Freeze exact JSON arguments without coercion or mutable aliases."""
    return json.dumps(
        arguments, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def definition_digest(engine: HookEngine) -> str:
    return hashlib.sha256(
        freeze_candidate({"handlers": [h.model_dump() for h in engine.definitions]})
    ).hexdigest()


def validate_candidate(arguments: dict, definition) -> None:
    try:
        schema = json.loads(definition.parameters_json)
        validator = validators.validator_for(schema)
        validator.check_schema(schema)
        # The default jsonschema registry retrieves missing URIs. An empty
        # referencing registry resolves embedded resources but never performs I/O.
        validator(schema, registry=Registry()).validate(arguments)
        freeze_candidate(arguments)
    except (ValueError, SchemaError, ValidationError, Unresolvable):
        raise HookPreparationError("tool candidate schema validation failed") from None


@dataclass(frozen=True)
class PreparedHookCall:
    """Frozen reviewed candidate and its captured host authority identities."""

    event: HookEvent
    original_arguments: bytes
    final_arguments: bytes
    definition: object
    hook_definitions: str
    outcome: HookEventOutcome
    effect_events: tuple[tuple[str, HookEvent], ...] = ()
    requirements: tuple[str, ...] = ()
    dependency_requirements: tuple[str, ...] = ()
    pending_scope: object | None = field(default=None, compare=False, repr=False)
    pending_context: tuple[str, ...] = ()


class HookPreparationError(RuntimeError):
    pass


def _candidate_event(event, arguments):
    value = {key: item for key, item in event.model_dump().items() if item is not None}
    value["data"].update(tool_args=arguments, candidate_arguments=arguments)
    return parse_event(value)


def _prepare_steps(event, engine, definition, *, defer_context=False):
    if event.event != "PreToolUse":
        raise ValueError("tool preparation requires PreToolUse")
    if (
        event.data["tool_id"] != definition.tool_id
        or event.data["provider"] != definition.provider
    ):
        raise ValueError("tool event definition identity mismatch")
    original = dict(event.model_dump()["data"].get("tool_args", {}))
    validate_candidate(original, definition)
    original_bytes = freeze_candidate(original)
    identity = definition_digest(engine)
    scope = engine.begin_event(event)
    outcomes = []
    effect_events = []
    candidate = original
    current = event
    retained = False
    try:
        plans = engine.plan_handlers(scope, event)
        if not plans:
            result = yield None, current, scope
            outcomes.append(result)
        phases = (
            ("transform", "validate")
            if defer_context
            else ("transform", "validate", "context")
        )
        for phase in phases:
            for plan in plans:
                if plan.phase != phase:
                    continue
                result = yield plan.handler_id, current, scope
                outcomes.append(result)
                effect_events.extend(
                    (handler_id, current) for handler_id, _ in result.accepted
                )
                if not result.allowed:
                    raise HookPreparationError("controlling tool hook refused")
                for _handler, effects in result.accepted:
                    if effects.updated_input is not None:
                        if phase != "transform":
                            raise HookPreparationError("late transformation refused")
                        candidate = effects.model_dump()["updated_input"]
                        validate_candidate(candidate, definition)
                        current = _candidate_event(current, candidate)
        result = engine._merge(outcomes)
        if not result.allowed or definition_digest(engine) != identity:
            raise HookPreparationError("tool hook definition changed or refused")
        if not engine.effects_current(current, result):
            raise HookPreparationError("tool hook authority changed")
        if not defer_context:
            engine.notify_planned(scope, current)
        prepared = PreparedHookCall(
            current,
            original_bytes,
            freeze_candidate(candidate),
            definition,
            identity,
            result,
            tuple(effect_events),
            tuple(
                plan.handler_id
                for plan in plans
                if plan.explicit_required
                or plan.phase == "transform"
                or "deny"
                in next(
                    h.effects for h in engine.definitions if h.id == plan.handler_id
                )
            ),
            tuple(plan.handler_id for plan in plans if plan.dependency_required),
            scope if defer_context else None,
            (
                tuple(plan.handler_id for plan in plans if plan.phase == "context")
                if defer_context
                else ()
            ),
        )
        retained = defer_context
        return prepared
    finally:
        if not retained:
            scope.close()


async def prepare_tool(
    event: HookEvent, engine: HookEngine, *, definition
) -> PreparedHookCall:
    """Prepare against a host-resolved schema on H2's application owner loop."""
    steps = _prepare_steps(event, engine, definition)
    value = None
    while True:
        try:
            handler_id, candidate, scope = steps.send(value)
        except StopIteration as done:
            return done.value
        try:
            value = (
                await engine.fire_handler_async(scope, handler_id, candidate)
                if handler_id is not None
                else await engine.fire_async(candidate)
            )
        except BaseException:
            steps.close()
            raise


def prepare_tool_sync(
    event: HookEvent, engine: HookEngine, *, definition, defer_context=False
) -> PreparedHookCall:
    """Agent-thread face using H2's owned synchronous handler facade."""
    steps = _prepare_steps(event, engine, definition, defer_context=defer_context)
    value = None
    while True:
        try:
            handler_id, candidate, scope = steps.send(value)
        except StopIteration as done:
            return done.value
        try:
            value = (
                engine.fire_handler(scope, handler_id, candidate)
                if handler_id is not None
                else engine.fire(candidate)
            )
        except BaseException:
            steps.close()
            raise


class ToolHookRun:
    """Run-scoped adapter for the existing synchronous agent and terminal gates."""

    def __init__(
        self,
        engine,
        *,
        run_id,
        session_id,
        turn_id,
        resolve_definition,
        should_cancel=lambda: False,
        required_handler_ids=lambda: (),
        definition_requirements=None,
        render_context=None,
        parent_run_id=None,
        workspace_id=None,
        lifecycle=None,
        parent_scope=None,
        context_owner=None,
        containing_hook=None,
    ):
        from .checkpoints import HookCheckpointStore

        self.engine = engine
        self.run_id = run_id
        self.session_id = session_id
        self.turn_id = turn_id
        self.parent_run_id = parent_run_id
        self.workspace_id = workspace_id
        self.resolve_definition = resolve_definition
        self.should_cancel = should_cancel
        self.required_handler_ids = required_handler_ids
        self.definition_requirements = definition_requirements
        self.render_context = render_context or self._render_user_context
        self.context_owner = context_owner
        self.containing_hook = containing_hook
        self._prepared = {}
        self._event_definitions = {}
        self._post_installed = set()
        self._preaccepted = set()
        self._effect_events = {}
        self._rendered = {}
        self._accepted_rows = []
        self._pending = set()
        self._definitions = definition_digest(engine)
        self.lifecycle = lifecycle
        self.checkpoints = (
            lifecycle.checkpoints
            if lifecycle
            else HookCheckpointStore(
                current=self._current,
                accept_current=self.engine.effects_current,
                stage_context=self._stage_context,
            )
        )
        self.checkpoints.bind_owner(run_id, parent_scope if lifecycle else None)

    def _current(self, event):
        from ..agent_models import ToolCall

        expected = self._event_definitions.get(event.event_id)
        current_definition = self.resolve_definition(
            ToolCall(
                event.data["tool_name"],
                dict(event.model_dump()["data"]["tool_args"]),
                "",
            )
        )
        return (
            expected == current_definition
            and event.run_id == self.run_id
            and not self.should_cancel()
            and definition_digest(self.engine) == self._definitions
        )

    @staticmethod
    def _render_user_context(event, handler_id, block):
        # The host supplies origin and handler identity, never result body fields.
        return (
            "<untrusted-hook-context>\n"
            + json.dumps(
                {
                    "event_id": event.event_id,
                    "handler_id": handler_id,
                    "candidate_sha256": hashlib.sha256(
                        freeze_candidate(
                            event.model_dump()["data"].get("candidate_arguments", {})
                        )
                    ).hexdigest(),
                    "instructions": block.text,
                },
                ensure_ascii=False,
            )
            + "\n</untrusted-hook-context>"
        )

    def _stage_context(self, event, outcome):
        from ..agent_models import (
            HookContextOrigin,
            PluginContextText,
            check_host_context,
        )

        rows = []
        for handler_id, result in outcome.accepted:
            for block in result.context:
                source = self._effect_events.get((event.event_id, handler_id), event)
                rendered = self.render_context(source, handler_id, block)
                origins = (
                    rendered.checked_origins()
                    if isinstance(rendered, PluginContextText)
                    else ()
                )
                origin = HookContextOrigin(
                    event.event_id, handler_id, len(rendered.encode("utf-8")), origins
                )
                rows.append(
                    {
                        "role": "user",
                        "content": PluginContextText(str(rendered), origins, (origin,)),
                    }
                )
        # Whole event and shared turn contributions must fit BEFORE release.
        check_host_context(self._accepted_rows + rows, strip=False)
        if self.containing_hook is not None:
            if self.lifecycle is None or self.context_owner is None:
                if rows:
                    raise ValueError("nested hook context has no input owner")
                return
            container, handler_id = self.containing_hook
            self.lifecycle.context.stage_nested(
                container,
                handler_id,
                self.context_owner,
                event,
                outcome,
                rows,
                current=lambda: self._current(event),
            )
        elif self.lifecycle is not None:
            self.lifecycle.context.commit_nested(
                event,
                outcome,
                current=lambda: self._current(event),
                additional_rows=self._accepted_rows + rows,
            )
        self._accepted_rows.extend(rows)
        self._rendered[event.event_id] = tuple(rows)

    def _event(self, name, definition, data):
        from uuid import uuid4

        from .causality import current_chain

        chain = current_chain()
        value = {
            "protocol_version": 2,
            "event_id": uuid4().hex,
            "event": name,
            "timestamp": utc_now_iso(),
            "runtime_session_id": self.session_id,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "initiator": "child" if self.parent_run_id else "manual",
            "origin": "host_tool_dispatch",
            "causal_chain_id": chain.visits[0][0] if chain.visits else self.run_id,
            "causal_depth": len(chain.visits),
            "data": {
                "tool_id": definition.tool_id,
                "provider": definition.provider,
                "tool_name": definition.name,
                "definition_hash": hashlib.sha256(
                    definition.parameters_json.encode()
                ).hexdigest(),
                **data,
            },
        }
        if self.parent_run_id:
            value["parent_run_id"] = self.parent_run_id
        if self.workspace_id:
            value["workspace_id"] = self.workspace_id
        event = parse_event(value)
        self._event_definitions[event.event_id] = definition
        return event

    def _lookup(self, call):
        retained, prepared = self._prepared[id(call)]
        if retained is not call:
            raise HookPreparationError("unowned tool call")
        return prepared

    def preview_call(self, call):
        """Project prospective dispatch identity without executing any handlers."""
        definition = self.resolve_definition(call)
        return (
            self._event(
                "PreToolUse",
                definition,
                {
                    "tool_args": call.args,
                    "original_arguments": call.args,
                    "candidate_arguments": call.args,
                },
            ),
            definition,
        )

    def requirements_for(self, definition=None):
        legacy = self.required_handler_ids()
        mapped = (
            self.definition_requirements(definition)
            if self.definition_requirements is not None
            else ()
        )
        return (
            None
            if legacy is None or mapped is None
            else tuple(sorted(set(legacy) | set(mapped)))
        )

    def prepare_call(self, call):
        from dataclasses import replace

        definition = self.resolve_definition(call)
        self.checkpoints.wait(
            self.run_id,
            required_handler_ids=self.requirements_for(definition),
            should_cancel=self.should_cancel,
        )
        event = self._event(
            "PreToolUse",
            definition,
            {
                "tool_args": call.args,
                "original_arguments": call.args,
                "candidate_arguments": call.args,
            },
        )
        prepared = prepare_tool_sync(
            event, self.engine, definition=definition, defer_context=True
        )
        self._effect_events.update(
            {
                (prepared.event.event_id, handler): source
                for handler, source in prepared.effect_events
            }
        )
        final_call = replace(
            call, args=json.loads(prepared.final_arguments), raw_arguments=""
        )
        self._prepared[id(final_call)] = (final_call, prepared)
        return final_call

    def bind_dispatch_call(self, call, dispatch_call):
        """Bind an exact host reconstruction to its retained owned preparation."""
        prepared = self._lookup(call)
        if (
            dispatch_call.name != call.name
            or freeze_candidate(dispatch_call.args) != prepared.final_arguments
        ):
            raise HookPreparationError("dispatch reconstruction changed candidate")
        self._prepared[id(dispatch_call)] = (dispatch_call, prepared)

    def _finish_context(self, prepared):
        scope = prepared.pending_scope
        if scope is None:
            return prepared
        outcomes = [prepared.outcome]
        effect_events = list(prepared.effect_events)
        try:
            for handler_id in prepared.pending_context:
                outcome = self.engine.fire_handler(scope, handler_id, prepared.event)
                outcomes.append(outcome)
                effect_events.extend(
                    (handler, prepared.event) for handler, _ in outcome.accepted
                )
            outcome = self.engine._merge(outcomes)
            if not outcome.allowed or not self.engine.effects_current(
                prepared.event, outcome
            ):
                raise HookPreparationError("tool hook authority changed or refused")
            self.engine.notify_planned(scope, prepared.event)
            completed = replace(
                prepared,
                outcome=outcome,
                effect_events=tuple(effect_events),
                pending_scope=None,
                pending_context=(),
            )
            for key, (call, value) in tuple(self._prepared.items()):
                if value.event.event_id == prepared.event.event_id:
                    self._prepared[key] = (call, completed)
            self._effect_events.update(
                {
                    (prepared.event.event_id, handler): source
                    for handler, source in effect_events
                }
            )
            return completed
        finally:
            scope.close()

    def validate_dispatch(self, call, *, accept_context=True):
        prepared = self._lookup(call)
        if (
            freeze_candidate(call.args) != prepared.final_arguments
            or self.resolve_definition(call) != prepared.definition
            or definition_digest(self.engine) != prepared.hook_definitions
            or not self._current(prepared.event)
            or not self.engine.effects_current(prepared.event, prepared.outcome)
        ):
            raise HookPreparationError("reviewed tool identity changed")
        validate_candidate(call.args, prepared.definition)
        self.checkpoints.wait(
            self.run_id,
            required_handler_ids=self.requirements_for(prepared.definition),
            should_cancel=self.should_cancel,
        )
        # The runtime calls this acceptance seam after final legacy guards,
        # before permission review; dispatch revalidation never repeats effects.
        if accept_context and prepared.event.event_id not in self._preaccepted:
            prepared = self._finish_context(prepared)
            token = self.checkpoints.begin(
                prepared.event,
                prepared.requirements,
                dependency_requirements=prepared.dependency_requirements,
                owner_id=self.run_id,
                current=self._current,
                stage_context=self._stage_context,
            )
            self.checkpoints.accept(token, prepared.outcome)
            self._preaccepted.add(prepared.event.event_id)
            self.checkpoints.assert_next_input_allowed(
                self.run_id,
                required_handler_ids=self.requirements_for(prepared.definition),
            )
        return prepared.definition

    def install_result(self, call, result):
        if getattr(result, "dispatch_state", None) == "not_started" or (
            getattr(result, "dispatch_state", None) is None
            and result.outcome == "blocked"
        ):
            return
        prepared = self._lookup(call)
        if prepared.event.event_id in self._post_installed:
            return
        self._post_installed.add(prepared.event.event_id)
        uncertain = getattr(result, "dispatch_state", None) == "uncertain"
        status = "uncertain" if uncertain else ("success" if result.ok else "failed")
        data = {
            "tool_args": call.args,
            "status": status,
            "is_error": not result.ok,
            "result": {"content": result.content, "error": result.error},
        }
        input_failure = False
        events = []
        names = ["PostToolUse"] + (
            ["PostToolUseFailure"] if not result.ok and not uncertain else []
        )
        for name in names:
            event_data = dict(data)
            if name == "PostToolUseFailure":
                event_data["reason"] = (
                    result.outcome
                    if result.outcome in {"timeout", "cancelled"}
                    else "execution_failed"
                )
            try:
                event = self._event(name, prepared.definition, event_data)
            except ValueError:
                # Preserve the settled result while refusing the oversized
                # hook input whole. Never feed truncated result text to a hook.
                input_failure = True
                event_data.pop("result", None)
                event = self._event(name, prepared.definition, event_data)
            events.append(event)
        planned = []
        # Install BOTH events before any work can complete or publish a result.
        for event in events:
            scope = self.engine.begin_event(event)
            plans = self.engine.plan_handlers(scope, event)
            required = tuple(
                plan.handler_id for plan in plans if plan.explicit_required
            )
            dependent = tuple(
                plan.handler_id for plan in plans if plan.dependency_required
            )
            required += tuple(
                f"config:{record.index}"
                for record in self.engine.invalid_admissions
                if record.policy in {"explicit_required", "event_control"}
                and record.event in {None, event.event}
            )
            token = self.checkpoints.begin(
                event,
                required,
                dependency_requirements=dependent,
                owner_id=self.run_id,
                current=self._current,
                stage_context=self._stage_context,
            )
            planned.append((event, scope, plans, token))
        if input_failure:
            for _event, scope, _plans, token in planned:
                self.checkpoints.fail(token, "hook input too large")
                scope.close()
            return
        import asyncio

        future = asyncio.run_coroutine_threadsafe(
            self._complete_post(planned), self.engine.loop
        )
        self._pending.add(future)
        future.add_done_callback(self._pending.discard)

    async def _complete_post(self, planned):
        import asyncio

        for index, (event, scope, plans, token) in enumerate(planned):
            try:
                outcomes = []
                for plan in plans:
                    if plan.phase != "observe":
                        outcomes.append(
                            await self.engine.fire_handler_async(
                                scope, plan.handler_id, event
                            )
                        )
                if not plans:
                    outcomes.append(await self.engine.fire_async(event))
                result = self.engine._merge(outcomes)
                self.engine.notify_planned(scope, event)
                self.checkpoints.accept(token, result)
            except asyncio.CancelledError:
                for _event, remaining_scope, _plans, remaining_token in planned[index:]:
                    self.checkpoints.fail(remaining_token, "hook completion cancelled")
                    remaining_scope.close()
                raise
            except Exception:  # noqa: BLE001 -- host failure retains requirements
                self.checkpoints.fail(token, "hook completion failed")
            finally:
                scope.close()

    def admit_input(self):
        from .checkpoints import HookCheckpointError

        try:
            selected = self.requirements_for()
        except Exception as error:
            raise HookCheckpointError("hook dependency mapping unknown") from error
        self.checkpoints.wait(
            self.run_id, required_handler_ids=selected, should_cancel=self.should_cancel
        )
        rows = []
        if self.lifecycle is not None:
            rows.extend(self.lifecycle.context.blocks(self.run_id, "model"))
        for event, outcome in self.checkpoints.drain_context(self.run_id):
            if event.event not in {"PreToolUse", "PostToolUse", "PostToolUseFailure"}:
                continue
            if not self.engine.effects_current(event, outcome) or not self._current(
                event
            ):
                raise HookCheckpointError("accepted hook context became stale")
            rows.extend(self._rendered.pop(event.event_id, ()))
        return tuple(rows)

    def settle(self):
        """Normal terminal admission joins every required pending event."""
        try:
            self.checkpoints.wait(
                self.run_id, terminal=True, should_cancel=self.should_cancel
            )
        finally:
            for _call, prepared in self._prepared.values():
                if prepared.pending_scope is not None:
                    prepared.pending_scope.close()
            if self.lifecycle is not None:
                self.lifecycle.close_scope(self.run_id)
            else:
                self.checkpoints.close_owner(self.run_id)
            if self.should_cancel():
                for future in tuple(self._pending):
                    future.cancel()
