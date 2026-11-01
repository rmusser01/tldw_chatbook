"""Private workflow task narrowing; native owners retain all permission authority."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint, WorktreeScope

from .agent_models import ToolCall
from .automatic_work_budget import AutomaticWorkRefused
from .run_context import current_run_id

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Chat.console_workflow_runs import WorkflowPhaseAuthorization
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeActivityClaim

    from .execution_capacity import ExecutionOwner
    from .local_tool_provider import RunAdmittedWorkspaceRoot
    from .tool_catalog import ToolCatalogRegistry
    from .workflow_models import WorkflowTaskClaim, WorkflowTaskSpec

_KEY = object()
_CURRENT: ContextVar[WorkflowTaskAuthorization | None] = ContextVar(
    "workflow_task", default=None
)
_FILE_READS = frozenset({"fs_read", "fs_list", "fs_glob", "fs_grep"})
_FILE_WRITES = frozenset({"fs_write", "fs_edit", "fs_patch", "fs_mkdir"})
_GIT_READS = frozenset(
    {"git_status", "git_diff", "git_log", "git_blame", "git_branches"}
)
_DISCOVERY = frozenset({"find_tools", "load_tools"})
_PURE_BUILTINS = frozenset({"calculator", "get_current_datetime"})


def _refuse(reason: str) -> None:
    raise AutomaticWorkRefused(f"workflow_task_{reason}")


def _runtime(phase):
    controller = phase._controller
    runtime = getattr(getattr(controller, "app", None), "console_runtime", None)
    if runtime is None:
        runtime = getattr(controller, "_hooks_v2_runtime", None)
    if (
        runtime is None
        or runtime.chat_controller is not controller
        or runtime.agent_bridge is not controller._agent_bridge
    ):
        _refuse("runtime_changed")
    return runtime


def _check_hooks(runtime, session_id: str) -> None:
    # Hook notification workers currently lack the native activity drain contract.
    from tldw_chatbook import config

    from .run_hooks import load_hooks_config

    with config.locked_hooks_config_snapshot() as snapshot:
        configured = load_hooks_config(
            {"hooks": snapshot.section} if snapshot.section_present else {}
        )
    if configured.enabled and (configured.hooks or configured.v2_handlers):
        _refuse("effectful_hooks_unsupported")
    if getattr(runtime, "run_hooks_engine", None) is not None:
        _refuse("effectful_hooks_unsupported")
    engines = getattr(runtime, "_hooks_v2_engines", {})
    engine = engines.get(session_id)
    if engine is not None and engine.definitions:
        _refuse("effectful_hooks_unsupported")


@dataclass(frozen=True, slots=True, init=False)
class WorkflowTaskAuthorization:
    """An original accepted phase's immutable restriction, bound once to native work."""

    phase: WorkflowPhaseAuthorization = field(repr=False)
    claim: WorkflowTaskClaim = field(repr=False)
    task: WorkflowTaskSpec = field(repr=False)
    binding: RunAdmittedWorkspaceRoot = field(repr=False)
    registry: ToolCatalogRegistry = field(repr=False)
    footprint: WorktreeFootprint = field(repr=False)
    _runtime: ConsoleRuntime = field(repr=False)
    _binding_state: list = field(repr=False)
    _native_preparation: list = field(repr=False)
    _native_launch: list = field(repr=False)
    _native_dispatch: list = field(repr=False)
    _required_evidence: list = field(repr=False)
    _lock: Lock = field(repr=False)
    _owners: tuple = field(repr=False)

    def __init__(
        self, *, phase, claim, task, binding, registry, footprint, runtime, _key
    ):
        if _key is not _KEY:
            _refuse("private_custody_required")
        for name, value in locals().copy().items():
            if name not in {"self", "_key", "runtime"}:
                object.__setattr__(self, name, value)
        object.__setattr__(self, "_runtime", runtime)
        object.__setattr__(self, "_binding_state", [])
        object.__setattr__(self, "_native_preparation", [])
        object.__setattr__(self, "_native_launch", [])
        object.__setattr__(self, "_native_dispatch", [])
        object.__setattr__(self, "_required_evidence", [])
        object.__setattr__(self, "_lock", Lock())
        object.__setattr__(
            self,
            "_owners",
            tuple(
                (
                    name,
                    resolved[0],
                    resolved[1],
                    deepcopy(resolved[1].load_schema(resolved[0])),
                )
                for name in task.allowed_tools
                if (resolved := registry.resolve_owner_for_name(name)) is not None
            ),
        )

    @contextmanager
    def scope(self) -> Iterator[WorkflowTaskAuthorization]:
        """Carry original custody explicitly into one native loop/tool thread."""
        token = _CURRENT.set(self)
        try:
            yield self
        finally:
            _CURRENT.reset(token)

    def check_preparation(self) -> None:
        """Revalidate accepted custody and current binding without granting dispatch."""
        self.phase.check()
        if _runtime(self.phase) is not self._runtime:
            _refuse("runtime_changed")
        _check_hooks(self._runtime, self.claim.session_id)
        if not self.binding.guard(False):
            _refuse("binding_changed")
        root = self.footprint.reads + self.footprint.writes
        if (
            root
            and capture_directory_chain(root[0].root.canonical_root) != root[0].root
        ):
            _refuse("binding_changed")

    def bind_native(
        self, *, owner: ExecutionOwner, activity_claim: WorktreeActivityClaim
    ) -> None:
        """Bind only after the existing native row, capacity and launch commit."""
        self.check_preparation()
        from .execution_capacity import ExecutionOwner, WorkOrigin

        if (
            type(owner) is not ExecutionOwner
            or owner.capacity is not self._runtime.execution_capacity
            or owner.run_id != self.claim.launch_run_id
            or owner.conversation_id != self.claim.conversation_id
            or owner.origin is not WorkOrigin.AUTOMATIC
            or not owner.child
        ):
            _refuse("native_owner_required")
        self._runtime.worktree_activity.validate_claim(
            activity_claim, owner=owner, footprint=self.footprint
        )
        ledger = self.phase._ledger
        from .activation import execution

        with execution(ledger), ledger._db.connection() as conn:
            reservation = conn.execute(
                "SELECT state,chain_id,owner_id FROM automatic_work_reservations WHERE id=?",
                (self.claim.reservation_id,),
            ).fetchone()
            if reservation is None or tuple(reservation) != (
                "committed",
                self.claim.chain_id,
                self.claim.owner_id,
            ):
                _refuse("native_launch_required")
        with self._lock:
            if self._binding_state:
                _refuse("already_bound")
            ledger.bind_workflow_task_run(
                self.claim.phase_attempt_id,
                task_id=self.claim.task_id,
                run_id=owner.run_id,
                owner_id=self.claim.owner_id,
            )
            self._binding_state.append((owner, activity_claim))

    def check(self) -> None:
        """Require the same live task, native run, owner and held activity claim."""
        self.check_preparation()
        with self._lock:
            bound = self._binding_state[0] if self._binding_state else None
        if bound is None:
            _refuse("native_binding_required")
        owner, claim = bound
        if (
            current_run_id() != self.claim.launch_run_id
            or owner.run_id != current_run_id()
        ):
            _refuse("run_mismatch")
        from .execution_capacity import current_execution_owner

        if current_execution_owner() is not owner:
            _refuse("owner_mismatch")
        self._runtime.worktree_activity.validate_claim(
            claim, owner=owner, footprint=self.footprint
        )
        from .activation import execution

        with execution(self.phase._ledger), self.phase._ledger._db.connection() as conn:
            row = conn.execute(
                "SELECT t.status,t.run_id,r.status FROM workflow_task_attempts t "
                "JOIN agent_runs r ON r.id=t.run_id WHERE t.id=?",
                (self.claim.id,),
            ).fetchone()
            if row is None or tuple(row) != ("accepted", self.claim.launch_run_id, "running"):
                _refuse("task_settled")

    def permits_name(self, name: str) -> bool:
        """Intersect immutable planned tools with their captured native owners."""
        if name not in self.task.allowed_tools:
            return False
        if name in _DISCOVERY:
            return True
        resolved = self.registry.resolve_owner_for_name(name)
        if resolved is None:
            return False
        _, provider = resolved
        captured = next((entry for entry in self._owners if entry[0] == name), None)
        if (
            captured is None
            or resolved != (captured[1], captured[2])
            or provider.load_schema(resolved[0]) != captured[3]
        ):
            return False
        from .local_tool_provider import LocalToolProvider
        from .mcp_tool_provider import MCPToolProvider
        from .raw_shell_tool_provider import RawShellToolProvider
        from .tool_catalog import BuiltinToolProvider
        from .virtual_cli_provider import VirtualCliProvider

        if type(provider) is LocalToolProvider:
            return (
                name in _FILE_READS | _FILE_WRITES | _GIT_READS
                or self.task.execution_mode == "exclusive"
            )
        if type(provider) is BuiltinToolProvider:
            return name in _PURE_BUILTINS
        if type(provider) is VirtualCliProvider:
            return True
        return self.task.execution_mode == "exclusive" and type(provider) in {
            RawShellToolProvider,
            MCPToolProvider,
        }


def _prepare_workflow_task(
    phase, *, task_id: str, registry, binding
) -> WorkflowTaskAuthorization:
    """Issue at most one private preparation per task from original phase custody."""
    from tldw_chatbook.Chat.console_workflow_runs import WorkflowPhaseAuthorization

    if (
        type(phase) is not WorkflowPhaseAuthorization
        or phase.context.workflow is not phase
    ):
        _refuse("private_phase_required")
    with phase._task_scope_lock:
        if task_id in phase._task_scopes:
            _refuse("already_prepared")
        authorization = _prepare_task(
            phase, task_id=task_id, registry=registry, binding=binding
        )
        phase._task_scopes[task_id] = authorization
        return authorization


def _prepare_task(
    phase, *, task_id: str, registry, binding
) -> WorkflowTaskAuthorization:
    """Host-only preparation from accepted custody, never a supplied claim/ID grant."""
    from tldw_chatbook.Chat.console_workflow_runs import WorkflowPhaseAuthorization
    from tldw_chatbook.Tools.local_tool_impls import resolve_workspace_path
    from tldw_chatbook.Tools.remote_root_types import local_root_path

    from .activation import execution
    from .local_tool_provider import RunAdmittedWorkspaceRoot

    if (
        type(phase) is not WorkflowPhaseAuthorization
        or phase.context.workflow is not phase
    ):
        _refuse("private_phase_required")
    phase.check()
    runtime = _runtime(phase)
    _check_hooks(runtime, phase.attempt.session_id)
    if type(binding) is not RunAdmittedWorkspaceRoot:
        _refuse("binding_required")
    with execution(phase._ledger):
        snapshot = phase._ledger._db.workflow_runs.get(phase.attempt.workflow_id)
        plan = snapshot.plans[phase.attempt.revision_id - 1]
        expected = plan.request.binding
        root = local_root_path(binding.root, site="workflow task preparation")
        if (binding.workspace_id, binding.binding_id, str(root)) != (
            expected.workspace_id,
            expected.binding_id,
            expected.locator,
        ):
            _refuse("binding_mismatch")
        if not binding.guard(False):
            _refuse("binding_changed")
        task = next(
            t
            for p in plan.phases
            if p.id == phase.attempt.phase_id
            for t in p.tasks
            if t.id == task_id
        )
        chain = capture_directory_chain(root)
        reads, writes = [], []
        for entry in task.footprint:
            path = resolve_workspace_path(
                entry.path,
                root,
                intent="write" if entry.access == "write" else "read",
                context=_path_context(binding),
            )
            (writes if entry.access == "write" else reads).append(
                WorktreeScope(path, entry.subtree, chain)
            )
        if writes and (
            not binding.allow_write
            or expected.access != "rw"
            or not binding.guard(True)
        ):
            _refuse("readonly_binding")
        if task.execution_mode == "exclusive":
            if (
                not binding.allow_write
                or expected.access != "rw"
                or not binding.guard(True)
            ):
                _refuse("readonly_binding")
            writes = [WorktreeScope(chain.canonical_root, True, chain)]
        footprint = WorktreeFootprint(
            tuple(reads), tuple(writes), task.execution_mode == "exclusive"
        )
        for path in task.inputs:
            # Structured predecessor references name the approved output path.
            target = resolve_workspace_path(
                path.split(":", 1)[-1],
                root,
                intent="read",
                context=_path_context(binding),
            )
            if not _covered(target, False, footprint.reads + footprint.writes):
                _refuse("input_outside_scope")
        claim = phase._ledger.claim_workflow_task(
            phase.attempt.id, task_id=task_id, owner_id=phase.attempt.owner_id
        )
        with phase._ledger._db.connection() as conn:
            row = conn.execute(
                "SELECT state FROM automatic_work_reservations WHERE id=?",
                (claim.reservation_id,),
            ).fetchone()
            if row is None or row[0] != "reserved" or claim.run_id is not None:
                _refuse("already_launched")
    return WorkflowTaskAuthorization(
        phase=phase,
        claim=claim,
        task=task,
        binding=binding,
        registry=registry,
        footprint=footprint,
        runtime=runtime,
        _key=_KEY,
    )


def _covered(path: Path, subtree: bool, scopes: tuple[WorktreeScope, ...]) -> bool:
    return any(
        (path == s.path and (not subtree or s.subtree))
        or (s.subtree and path.is_relative_to(s.path))
        for s in scopes
    )


def _path_context(binding):
    from tldw_chatbook.Utils.sensitive_paths import (
        merge_sensitive_context,
        resolve_sensitive_context,
    )

    extra = (
        tuple(binding.exclusions_provider())
        if binding.exclusions_provider is not None
        else ()
    )
    return merge_sensitive_context(
        resolve_sensitive_context(),
        extra_files=tuple(path for path in extra if path.is_file()),
        extra_dirs=tuple(path for path in extra if not path.is_file()),
    )


def _check_paths(authorization, provider, name, args):
    from tldw_chatbook.Tools.local_tool_impls import resolve_workspace_path
    from tldw_chatbook.Tools.remote_root_types import local_root_path

    from .local_tool_provider import LocalToolProvider

    if (
        args.get("root_alias", authorization.binding.alias)
        != authorization.binding.alias
    ):
        _refuse("alias_mismatch")
    if type(provider) is LocalToolProvider:
        authority, args = provider._select_admitted_root(name, args)
    else:
        command, argv, authority = provider._validated_args(args)
        from tldw_chatbook.Tools.virtual_cli_impls import (
            _leased_operation,
            parse_request,
        )

        _, parsed = parse_request(command, argv)
        name, args = _leased_operation(command, parsed)
    write = name in _FILE_WRITES
    if (
        authority is not authorization.binding
        or not authority.guard(write)
        or (write and not authority.allow_write)
    ):
        _refuse("binding_changed")
    root = local_root_path(authority.root, site="workflow call scope")
    if name == "fs_patch":
        from tldw_chatbook.Tools.patch_tool_impls import parse_patch_targets

        paths = [(p.new_path, False) for p in parse_patch_targets(args["diff"])]
    elif name in {"fs_glob", "fs_grep"} | _GIT_READS:
        paths = [(".", True)]
    else:
        paths = [(args["path"], name == "fs_list")]
    scopes = (
        authorization.footprint.writes
        if write
        else authorization.footprint.reads + authorization.footprint.writes
    )
    for raw, subtree in paths:
        path = resolve_workspace_path(
            raw,
            root,
            intent="write" if write else "read",
            context=_path_context(authority),
        )
        if not _covered(path, subtree, scopes):
            _refuse("path_outside_scope")
        if name == "fs_mkdir":
            for parent in path.parents:
                if parent.exists():
                    break
                if not _covered(parent, False, scopes):
                    _refuse("parent_outside_scope")


def check_task_call(
    authorization: WorkflowTaskAuthorization,
    *,
    registry: ToolCatalogRegistry,
    call: ToolCall,
) -> None:
    """Apply one stable fail-closed restriction without granting permission."""
    try:
        _check_task_call(authorization, registry=registry, call=call)
    except AutomaticWorkRefused:
        raise
    except Exception:  # noqa: BLE001 - every authority failure refuses
        _refuse("scope_unavailable")


def _check_task_call(authorization, *, registry, call) -> None:
    """Refuse mismatched native custody or effects before existing permission review."""
    if type(authorization) is not WorkflowTaskAuthorization:
        _refuse("private_custody_required")
    authorization.check()
    if registry is not authorization.registry or not authorization.permits_name(
        call.name
    ):
        _refuse("tool_outside_scope")
    if call.name in _DISCOVERY:
        return
    _, provider = registry.resolve_owner_for_name(call.name)
    from .local_tool_provider import LocalToolProvider
    from .virtual_cli_provider import VirtualCliProvider

    if type(provider) in {LocalToolProvider, VirtualCliProvider} and (
        call.name in _FILE_READS | _FILE_WRITES | _GIT_READS
        or type(provider) is VirtualCliProvider
    ):
        _check_paths(authorization, provider, call.name, call.args)


def current_task_authorization() -> WorkflowTaskAuthorization | None:
    return _CURRENT.get()


def check_current_task_call(call: ToolCall, *, registry=None) -> None:
    authorization = _CURRENT.get()
    if authorization is not None:
        check_task_call(
            authorization, registry=registry or authorization.registry, call=call
        )
    elif call.name == "run_skill_script":
        from .automatic_work_runtime import current_automatic_work

        automatic = current_automatic_work()
        if automatic is not None and automatic.workflow is not None:
            _refuse("script_cleanup_unsupported")


def check_task_provider(provider, tool_id: str, args: dict) -> None:
    """Recheck the actual cached owner at the provider's post-wait effect seam."""
    authorization = _CURRENT.get()
    if authorization is None:
        return
    entries = authorization.registry._ensure_catalog_cache().entries
    entry = next((e for e in entries if e.id == tool_id or e.name == tool_id), None)
    if entry is None or authorization.registry.resolve_owner_for_name(entry.name) != (
        entry.id,
        provider,
    ):
        _refuse("provider_changed")
    check_task_call(
        authorization, registry=authorization.registry, call=ToolCall(entry.name, args)
    )
