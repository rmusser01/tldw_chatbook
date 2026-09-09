"""Console goal orchestration. Widgets project state; runtime owns execution."""

from __future__ import annotations

import asyncio
from pathlib import Path

from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits
from tldw_chatbook.Agents.goal_models import (
    GoalBindingRef,
    GoalPolicy,
    GoalRequest,
    GoalToolScope,
)
from tldw_chatbook.Chat.console_goal_runs import goal_provider_ref


def change_review_provider(controller, bridge, conversation_id=None):
    """Pin review/revert occupancy to the actual reviewed conversation, including aliases."""
    if controller is None or bridge is None:
        return None
    if conversation_id is None:
        active = controller.store.active_session_id
        if active:
            conversation_id = controller._agent_conversation_id(active)
    provider = (
        bridge.change_review_provider(conversation_id) if conversation_id else None
    )
    if provider is not None:
        provider.run_active = lambda: any(
            controller._agent_conversation_id(sid) == conversation_id
            for sid in controller._live_busy_session_ids()
        )
    return provider


class ConsoleGoalsController:
    """Use late-bound Console dependencies without storing DOM or owning tasks."""

    def __init__(
        self,
        *,
        app_instance,
        get_controller,
        get_coordinator,
        push_screen,
        run_worker,
        open_changes,
    ):
        self.app_instance = app_instance
        self.get_controller = get_controller
        self.get_coordinator = get_coordinator
        self.push_screen = push_screen
        self.run_worker = run_worker
        self.open_changes = open_changes

    def open_history(self, offset: int = 0) -> None:
        self.run_worker(self._open_history(offset))

    async def _open_history(self, offset: int = 0) -> None:
        from tldw_chatbook.Widgets.Console.console_goal_status import ConsoleGoalHistory

        try:
            self.get_controller()
            coordinator = self.get_coordinator()
            await coordinator.recover()
            rows = await asyncio.to_thread(
                coordinator.service.list_goals, limit=50, offset=offset
            )
            self.push_screen(
                ConsoleGoalHistory(
                    rows,
                    open_goal=self.open_goal,
                    new_goal=self.open_setup,
                    older=(lambda: self.open_history(offset + 50))
                    if len(rows) == 50
                    else None,
                    newer=(lambda: self.open_history(max(0, offset - 50)))
                    if offset
                    else None,
                    page=offset // 50 + 1,
                )
            )
        except (RuntimeError, ValueError) as exc:
            self.app_instance.notify(str(exc), severity="warning")

    def open_goal(self, goal_id: str) -> None:
        from tldw_chatbook.Widgets.Console.console_goal_status import ConsoleGoalStatus

        self.push_screen(
            ConsoleGoalStatus(
                self.get_coordinator(),
                goal_id,
                review_changes=self.review_changes,
                app_instance=self.app_instance,
            )
        )

    def review_changes(self, goal, checkpoint) -> None:
        self.run_worker(self._review_changes(goal, checkpoint))

    async def _review_changes(self, goal, checkpoint) -> None:
        # The selected checkpoint is an attempt identity, never a newest-run query.
        coordinator = self.get_coordinator()
        run_id = await asyncio.to_thread(
            coordinator.service.checkpoint_run_id, goal.id, checkpoint.id
        )
        if run_id:
            self.open_changes(run_id, conversation_id=goal.conversation_id)

    def open_setup(self) -> None:
        self.run_worker(self._open_setup())

    async def _open_setup(self) -> None:
        from tldw_chatbook.Agents.run_log import _setting
        from tldw_chatbook.config import coerce_bool_setting
        from tldw_chatbook.Widgets.Console.console_goal_setup_modal import (
            ConsoleGoalSetupModal,
        )

        try:
            controller = self.get_controller()
            coordinator = self.get_coordinator()
            if not coerce_bool_setting(_setting("goal_runs_enabled", False), False):
                raise ValueError(
                    "Enable Goal runs in F9 Settings → Console behavior first."
                )
            active = controller.store.active_session_id
            if not active:
                raise ValueError("Open a Console conversation first.")
            resolution = await controller.provider_gateway.resolve_for_send(
                controller._provider_selection_for_session(active)
            )
            registry = coordinator.service.persistence.workspace_registry
            workspace_id = controller.store.session_workspace_id(active)
            rows = await asyncio.to_thread(registry.list_runtime_bindings, workspace_id)
            bindings = tuple(
                GoalBindingRef(
                    workspace_id=r.workspace_id,
                    binding_id=r.binding_id,
                    locator=r.locator,
                    access=r.metadata.get("access", "ro"),
                )
                for r in rows
                if r.status.value == "ready"
                and r.binding_kind.value == "local-filesystem"
            )
            if not bindings:
                raise ValueError(
                    "Add a ready local folder binding to this workspace in F9 Settings."
                )
            entries = list(controller._agent_bridge._registry.list_catalog())
            mcp = await controller._compose_mcp_provider(active, publish_counts=False)
            mcp_refs = {}
            if mcp:
                for tool_id, (tool, _state) in mcp._entry_by_llm_name.items():
                    if tool.server_key.startswith("local:") and not tool.stale:
                        mcp_refs[tool_id] = (
                            mcp._service.local_service.goal_tool_binding(
                                tool.server_key.removeprefix("local:"),
                                tool.name,
                                tool_id=tool_id,
                            )
                        )
                entries.extend(e for e in mcp.list_catalog() if e.id in mcp_refs)

            async def discover_tools(binding):
                current = await asyncio.to_thread(
                    registry.list_runtime_bindings, workspace_id
                )
                row = next(
                    (r for r in current if r.binding_id == binding.binding_id), None
                )
                if (
                    row is None
                    or row.status.value != "ready"
                    or row.binding_kind.value != "local-filesystem"
                    or row.locator != binding.locator
                    or row.metadata.get("access", "ro") != binding.access
                ):
                    raise ValueError(
                        "Project binding changed. Reopen goal setup to refresh it."
                    )
                local, _ = controller._compose_local_provider(
                    active,
                    project_root=Path(row.locator),
                    allow_write=row.metadata.get("access", "ro") == "rw",
                )
                selected_entries = entries + (
                    list(local.list_catalog()) if local else []
                )
                return tuple(dict.fromkeys(e.id for e in selected_entries))

            tool_ids = await discover_tools(bindings[0])
            defaults = tuple(
                t for t in tool_ids if t in {"local:fs_read", "local:fs_edit"}
            )
            limits = AutomaticWorkLimits.from_settings("goal_iteration")
            request = GoalRequest(
                objective="Describe the result you want",
                criteria="Describe how the result will be checked",
                provider=goal_provider_ref(resolution),
                binding=bindings[0],
                tool_scope=GoalToolScope(
                    catalog_tools=defaults, runtime_tools=("run_skill_script",)
                ),
                policy=GoalPolicy(
                    iterations=limits.generations,
                    model_calls=limits.model_calls,
                    budget_tokens=limits.budget_tokens,
                    output_tokens=limits.output_tokens,
                    wall_seconds=int(limits.wall_seconds),
                ),
            )

            async def configure(values, skill_name, script_path, arguments, inputs):
                selected = values["tool_scope"]["catalog_tools"]
                available = await discover_tools(
                    GoalBindingRef.model_validate(values["binding"])
                )
                if not set(selected).issubset(available):
                    raise ValueError(
                        "Tool availability changed. Refresh the project binding before launch."
                    )
                values["tool_scope"]["mcp_bindings"] = tuple(
                    mcp_refs[t].model_dump() for t in selected if t in mcp_refs
                )
                if skill_name:
                    skills = controller._agent_bridge._skills_service
                    if skills is None:
                        raise ValueError("Local skills service is unavailable.")
                    values["verifiers"] = (
                        (
                            await skills.goal_verifier_reference(
                                skill_name,
                                script_path,
                                arguments=arguments,
                                input_paths=inputs,
                            )
                        ).model_dump(),
                    )
                return GoalRequest.model_validate(values)

            self.push_screen(
                ConsoleGoalSetupModal(
                    request,
                    start=self.start,
                    bindings=bindings,
                    tool_ids=tool_ids,
                    discover_tools=discover_tools,
                    configure=configure,
                ),
                self._started,
            )
        except (RuntimeError, ValueError) as exc:
            self.app_instance.notify(str(exc), severity="warning")

    async def start(self, request: GoalRequest, launch_id: str):
        coordinator = self.get_coordinator()
        return await asyncio.shield(
            coordinator.launch(request, launch_id, app=self.app_instance)
        )

    def _started(self, goal) -> None:
        if goal:
            self.open_goal(goal.id)
