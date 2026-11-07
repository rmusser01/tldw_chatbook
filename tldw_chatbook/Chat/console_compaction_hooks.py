"""Compaction hook preparation and release using the actual Console owner."""

import asyncio
from dataclasses import replace


async def _hooks_for_compaction(
    self, session_id, resolution, *, reason, current, configuration=None
):
    """Resolve a real context operation under the runtime's one hook owner."""
    runtime = getattr(self, "_hooks_v2_runtime", None)
    if runtime is None:
        return None
    from tldw_chatbook.Agents.hooks_v2.lifecycle import CompactionHooks

    task = asyncio.current_task()
    submissions = getattr(self, "_hooks_v2_submissions", {})
    captured = submissions.get(task)
    operation = None
    if task in submissions:
        if (
            not isinstance(captured, tuple)
            or len(captured) != 3
            or captured[2] != session_id
        ):
            raise PermissionError("compaction hook submission already reserved")
        if captured[0] is None and captured[1] is None:
            return None
        if captured[0] is None or captured[1] is None:
            raise PermissionError("compaction hook submission already reserved")
        lifecycle, scope, _session = captured
    else:
        if configuration is None:
            configuration = await self.capture_turn_configuration_snapshot(session_id)
        from .console_automatic_submission import _begin_hook_submission
        from .console_received_turn import received_turn_claim_for

        if (
            received_turn_claim_for(self.store, session_id) is not None
            or self.store.received_turn_for_session(session_id) is not None
        ):
            raise RuntimeError("console_received_turn_pending")
        busy = self._live_busy_session_ids()
        if session_id in busy or len(busy) >= self.max_parallel_runs:
            raise RuntimeError("console_session_busy")
        prior_state = self.run_state_for(session_id)
        _begin_hook_submission(self, session_id)
        submissions = self._hooks_v2_submissions
        operation = (
            task,
            submissions[task],
            prior_state,
            self.run_state_for(session_id),
        )
        try:
            lifecycle = await runtime.prepare_hooks_v2(
                session_id, reason="resume", configuration=configuration
            )
        except BaseException:
            operation = (task, submissions.get(task, operation[1]), *operation[2:])
            self._release_compaction_submission(operation)
            raise
        operation = (task, submissions.get(task, operation[1]), *operation[2:])
        if lifecycle is None:
            self._release_compaction_submission(operation)
            return None
        scope = lifecycle.turn_scope
        reason = "manual"
    try:
        session = next(row for row in self.store.sessions() if row.id == session_id)
        workspace_id = session.workspace_id

        def memory_current(memory):
            if not lifecycle.current():
                return False
            session_now = next(
                (row for row in self.store.sessions() if row.id == session_id), None
            )
            if session_now is None or session_now.workspace_id != workspace_id:
                return False
            snapshots = self._durable_context_snapshots(session_id)
            if not snapshots:
                return False
            effective = self._select_session_effective_memory(
                session_id,
                memory.conversation_id,
                snapshots,
            )
            selected = effective.memory
            return bool(
                selected is not None
                and selected.memory_id == memory.memory_id
                and selected.revision == memory.revision
                and selected.active
                and selected.summarized_prefix_digest == memory.summarized_prefix_digest
            )

        def prepare(messages, cap):
            return self.provider_gateway.prepare_chat_request(
                replace(resolution, streaming=False, max_tokens=cap),
                list(messages),
                apply_safety_window=False,
            )

        hooks = CompactionHooks(
            lifecycle,
            scope,
            reason=reason,
            prepare=prepare,
            current=current,
            memory_current=memory_current,
        )
        hooks._controller_submission = operation
        return hooks
    except BaseException:
        self._release_compaction_submission(operation)
        raise


def _release_compaction_submission(self, operation):
    """Release only this auxiliary operation's reservation and validating state."""
    if operation is None:
        return
    task, reservation, prior_state, owned_state = operation
    submissions = self._hooks_v2_submissions
    if submissions.get(task) is not reservation:
        return
    lifecycle, scope, session_id = reservation
    if self.run_state_for(session_id) is owned_state:
        self._set_run_state(prior_state, session_id=session_id)
    submissions.pop(task)
    if lifecycle is not None:
        if lifecycle.turn_scope == scope:
            lifecycle.turn_scope = None
        if lifecycle._handoff is None or lifecycle._handoff[0] != scope:
            lifecycle.close_scope(scope)


async def _finish_compaction_hooks(hooks) -> str | None:
    """Return the existing completion refusal without swallowing cancellation."""
    if hooks is None:
        return None
    try:
        await hooks.finish()
        await hooks.lifecycle.wait(hooks.owner)
    except Exception:  # noqa: BLE001 -- existing hook completion boundary
        return "Required compaction hook failed; committed memory is retained."
    return None
