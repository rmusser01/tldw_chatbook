"""Required native workflow evidence and conservative saved lifecycle (ADR-285)."""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .agent_models import RUN_DONE
from .goal_iteration import capture_manifest, digest, verifier_digest
from .goal_storage_lifetime import goal_storage_operation, owned_goal_storage
from .workflow_models import (
    RECORD_BYTES,
    WorkflowCheck,
    WorkflowCheckpoint,
    WorkflowDecision,
    WorkflowExecutionRecovery,
    WorkflowQualityReview,
    WorkflowSnapshot,
    WorkflowTaskResult,
    WorkflowVerifierEvidence,
)

if TYPE_CHECKING:
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    from .agent_service import WorkflowChildReceipt
    from .automatic_work_budget import RuntimeRecoveryResult
    from .workflow_task_scope import WorkflowTaskAuthorization


def _manifest(root, paths, *, authorization=None, validate_path=None):
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain

    try:
        identity = capture_directory_chain(Path(root))
        manifest = capture_manifest(
            str(root),
            SimpleNamespace(input_paths=tuple(paths)),
            validate_path=(lambda path: _admitted_path(authorization, path))
            if authorization is not None
            else validate_path,
        )
        if manifest is None or capture_directory_chain(Path(root)) != identity:
            return None
        return digest(json.dumps((repr(identity), manifest)))
    except (OSError, ValueError):
        return None


def _admitted_path(authorization, relative):
    from tldw_chatbook.Tools.local_tool_impls import resolve_workspace_path
    from tldw_chatbook.Tools.remote_root_types import local_root_path

    from .workflow_task_scope import _covered, _path_context

    if not authorization.binding.guard(False):
        raise ValueError("evidence_binding_changed")
    path = resolve_workspace_path(
        relative,
        local_root_path(authorization.binding.root, site="workflow evidence"),
        intent="read",
        context=_path_context(authorization.binding),
    )
    if not _covered(
        path, False, authorization.footprint.reads + authorization.footprint.writes
    ):
        raise ValueError("evidence_path_outside_task")
    return path


def _bound_path(binding, root, path):
    from tldw_chatbook.Tools.local_tool_impls import resolve_workspace_path

    from .workflow_task_scope import _path_context

    return resolve_workspace_path(
        path, Path(root), intent="read", context=_path_context(binding)
    )


def _shell_manifest(path):
    return _manifest("/", (str(Path(path).absolute()).lstrip("/"),))


@dataclass
class _ShellObservation:
    spec: object
    request: object
    shell: str
    shell_manifest: str | None
    input_manifest: str | None
    output: object


@dataclass
class _TaskEvidence:
    service: WorkflowRunService
    authorization: object
    inputs: str | None
    records: list = field(default_factory=list)
    receipt: object | None = None
    drain: bool | None = None
    pending: dict = field(default_factory=dict)
    seed_json: str = ""
    dependency_digest: str = ""

    def begin_shell(self, provider, request):
        from tldw_chatbook.Chat.console_raw_cli import RawCliRuntime
        from tldw_chatbook.Tools.raw_cli_executor import (
            RequiredCliOutput,
            resolve_shell_argv,
        )
        from tldw_chatbook.Tools.remote_root_types import local_root_path

        authorization = self.authorization
        authorization.check()
        resolved = authorization.registry.resolve_owner_for_name("shell_exec")
        if resolved is None or resolved[1] is not provider:
            raise ValueError("native_verifier_provider_changed")
        if type(
            provider._runtime
        ) is not RawCliRuntime or provider._runtime is not getattr(
            authorization.phase._controller.app, "raw_cli_runtime", None
        ):
            raise ValueError("native_verifier_runtime_required")
        spec = next(
            (
                v
                for v in authorization.task.verifiers
                if request.command == shlex.join((v.verifier_path, *v.arguments))
            ),
            None,
        )
        if spec is None:
            return None
        root = local_root_path(authorization.binding.root, site="workflow verifier")
        try:
            relative = str(Path(spec.verifier_path).relative_to(root))
        except ValueError:
            raise ValueError("verifier_outside_binding") from None
        _admitted_path(authorization, relative)
        if (
            request.shell != "bash"
            or request.initial_directory
            != local_root_path(authorization.binding.root, site="workflow verifier")
            or request.console_session_id != authorization.claim.session_id
            or len(self.records) + len(self.pending) >= 32
            or self.pending
            or verifier_digest(spec) != spec.verifier_sha256
        ):
            raise ValueError("native_verifier_contract_changed")
        shell = resolve_shell_argv("bash", request.command)[0]
        observation = _ShellObservation(
            spec,
            request,
            shell,
            _shell_manifest(shell),
            _manifest(root, spec.input_paths, authorization=authorization),
            RequiredCliOutput(
                max_bytes=max(
                    0,
                    96 * 1024
                    - sum(len((v.stdout + v.stderr).encode()) for v in self.records),
                )
            ),
        )
        if request.invocation_id in self.pending:
            raise ValueError("native_verifier_invocation_conflict")
        self.pending[request.invocation_id] = observation
        return observation

    def finish_shell(self, observation, result):
        from tldw_chatbook.Tools.raw_cli_executor import (
            RawCliResult,
            resolve_shell_argv,
        )

        authorization = self.authorization
        from .execution_capacity import current_execution_owner
        from .run_context import current_run_id

        if (
            type(result) is not RawCliResult
            or self.pending.get(result.invocation_id) is not observation
            or result.invocation_id != observation.request.invocation_id
            or current_run_id() != authorization.claim.launch_run_id
            or not authorization._binding_state
            or current_execution_owner() is not authorization._binding_state[0][0]
        ):
            raise ValueError("native_verifier_result_mismatch")
        spec = observation.spec
        root = str(observation.request.initial_directory)
        after = _manifest(root, spec.input_paths, authorization=authorization)
        output_manifest = _manifest(
            root, authorization.task.outputs, authorization=authorization
        )
        fresh = bool(
            observation.input_manifest
            and after == observation.input_manifest
            and observation.shell_manifest
            and observation.shell_manifest == _shell_manifest(observation.shell)
            and resolve_shell_argv("bash", observation.request.command)[0]
            == observation.shell
            and result.resolved_shell == observation.shell
            and verifier_digest(spec) == spec.verifier_sha256
        )
        output = observation.output
        self.records.append(
            WorkflowVerifierEvidence(
                verifier_id=spec.id,
                invocation_id=result.invocation_id,
                verifier_sha256=spec.verifier_sha256,
                arguments_digest=digest(json.dumps(spec.arguments)),
                shell_path=observation.shell,
                shell_manifest=observation.shell_manifest,
                input_manifest=after,
                output_manifest=output_manifest,
                exit_code=result.exit_code,
                complete=output.sealed and not output.overflow,
                cleanup_proven=result.cleanup_proven,
                fresh=fresh,
                stdout=output.stdout,
                stderr=output.stderr,
                output_digest=digest(json.dumps((output.stdout, output.stderr))),
            )
        )
        del self.pending[result.invocation_id]
        if not result.cleanup_proven:
            from .execution_capacity import current_execution_owner

            owner = current_execution_owner()
            if owner is not None:
                owner.mark_cleanup_unproven()


class WorkflowRunService:
    """History is passive; only original native custody can publish a checkpoint."""

    def __init__(self, db: AgentRunsDB, persistence: Any) -> None:
        self.db = db
        self.persistence = persistence
        self._tasks = {}

    async def call(
        self, operation: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Any:
        """Offload one named finite service operation with complete storage custody.

        Args:
            operation: An original bound public service operation listed below.
            *args: Positional arguments for that operation.
            **kwargs: Keyword arguments for that operation.

        Returns:
            The operation's result after its owned storage callback completes.

        Raises:
            ValueError: The callback is not an original supported service method.
        """
        from tldw_chatbook.DB.base_db import run_owned_db_call

        allowed = (
            WorkflowRunService.get,
            WorkflowRunService.prepare_task_inputs,
            WorkflowRunService.validate_task_inputs,
            WorkflowRunService.checkpoint_task,
            WorkflowRunService.prepare_decision,
            WorkflowRunService.prepare_quality_review,
            WorkflowRunService.validate_decision,
            WorkflowRunService.project_recovery,
            WorkflowRunService.resume,
            WorkflowRunService.review_quality,
            WorkflowRunService.resolve_recovery,
        )
        if (
            type(operation) is not MethodType
            or getattr(operation, "__self__", None) is not self
            or getattr(operation, "__func__", None) not in allowed
        ):
            raise ValueError("unsupported_workflow_operation")

        def invoke():
            with owned_goal_storage(self):
                return operation(*args, **kwargs)

        return await run_owned_db_call(self.db, invoke)

    @goal_storage_operation
    def get(self, workflow_id: str) -> WorkflowSnapshot:
        """Read saved workflow state without dispatch or startup recovery."""
        return self.db.workflow_runs.get(workflow_id)

    @goal_storage_operation
    def prepare_task_inputs(
        self, authorization: WorkflowTaskAuthorization
    ) -> tuple[dict, ...]:
        """Freeze bounded task-only inputs before Task6's native preparation."""
        from .workflow_task_scope import WorkflowTaskAuthorization

        if (
            type(authorization) is not WorkflowTaskAuthorization
            or authorization.phase._ledger._db is not self.db
            or authorization.phase._task_scopes.get(authorization.task.id)
            is not authorization
        ):
            raise ValueError("native_task_custody_required")
        authorization.check_preparation()
        snapshot = self.get(authorization.claim.workflow_id)
        task = authorization.task
        root = snapshot.request.binding.locator
        paths = tuple(path.split(":", 1)[-1] for path in task.inputs)
        before = _manifest(root, paths, authorization=authorization)
        if before is None:
            raise ValueError("required_inputs_unavailable")
        review = self._prepare_review(
            snapshot.id,
            phase_id=authorization.claim.phase_id,
            quality=False,
            authorization=authorization,
        )
        if (
            type(review.evidence) is not WorkflowQualityReview
            or not review.evidence.ready
        ):
            raise ValueError("required_predecessor_unavailable")
        bodies = []
        definitions = {
            task.id: task for phase in snapshot.plans[-1].phases for task in phase.tasks
        }
        dependencies = {
            task_id: self._fresh_task_result(snapshot, definitions[task_id])
            for task_id in task.dependencies
        }
        for path in task.inputs:
            if ":" in path:
                predecessor, _output = path.split(":", 1)
                result = dependencies[predecessor]
                bodies.append({"source": path, "handoff": result.handoff})
            else:
                bodies.append(
                    {"source": path, "manifest": before, "read_with": "fs_read"}
                )
        if before != _manifest(root, paths, authorization=authorization):
            raise ValueError("required_inputs_changed")
        messages = (
            {"role": "user", "content": task.instructions},
            {"role": "user", "content": json.dumps(bodies, ensure_ascii=False)},
        )
        if len(json.dumps(messages, ensure_ascii=False).encode()) > RECORD_BYTES:
            raise ValueError("required_inputs_oversize")
        with authorization._lock:
            if authorization._required_evidence:
                raise ValueError("native_task_already_prepared")
            tracker = _TaskEvidence(
                self,
                authorization,
                before,
                seed_json=json.dumps(messages, ensure_ascii=False),
                dependency_digest=digest(
                    json.dumps(
                        [result.canonical_json() for result in dependencies.values()]
                    )
                ),
            )
            authorization._required_evidence.append(tracker)
            self._tasks[authorization.claim.id] = tracker
        return messages

    @goal_storage_operation
    def validate_task_inputs(
        self,
        authorization: WorkflowTaskAuthorization,
        messages: tuple[dict, ...] | list[dict] | None = None,
    ) -> None:
        """Recheck the original finite seed and all dependencies before admission."""
        tracker = self._tasks.get(authorization.claim.id)
        if tracker is None or tracker.authorization is not authorization:
            raise ValueError("native_task_custody_required")
        authorization.check_preparation()
        if (
            messages is not None
            and json.dumps(messages, ensure_ascii=False) != tracker.seed_json
        ):
            raise ValueError("required_task_seed_changed")
        snapshot = self.get(authorization.claim.workflow_id)
        paths = tuple(path.split(":", 1)[-1] for path in authorization.task.inputs)
        if tracker.inputs != _manifest(
            snapshot.request.binding.locator, paths, authorization=authorization
        ):
            raise ValueError("required_inputs_changed")
        definitions = {t.id: t for p in snapshot.plans[-1].phases for t in p.tasks}
        results = [
            self._fresh_task_result(snapshot, definitions[identity]).canonical_json()
            for identity in authorization.task.dependencies
        ]
        if tracker.dependency_digest != digest(json.dumps(results)):
            raise ValueError("required_predecessor_changed")

    def bind_native_receipt(
        self, authorization: WorkflowTaskAuthorization, receipt: WorkflowChildReceipt
    ) -> None:
        """Retain the actual original launch owner, including pre-bind failures."""
        from .agent_service import WorkflowChildReceipt

        tracker = self._tasks.get(authorization.claim.id)
        if (
            tracker is None
            or tracker.authorization is not authorization
            or type(receipt) is not WorkflowChildReceipt
            or not authorization._native_launch
            or authorization._native_launch[0] is not receipt
            or receipt.attempt_id != authorization.claim.id
            or receipt.run_id != authorization.claim.launch_run_id
            or (
                receipt.failure != "unknown"
                and receipt.owner.run_id
                != (None if receipt.failure == "no_run" else receipt.run_id)
            )
            or receipt.owner.capacity is not authorization._runtime.execution_capacity
            or receipt.owner.conversation_id != authorization.claim.conversation_id
            or receipt.activity_claim
            not in authorization._native_preparation[0].activity_claims
        ):
            raise ValueError("native_receipt_custody_required")
        if tracker.receipt is not None:
            if tracker.receipt is not receipt:
                raise ValueError("native_receipt_conflict")
            return
        tracker.receipt = receipt
        receipt.owner.on_drained(lambda drained: setattr(tracker, "drain", drained))

    @goal_storage_operation
    def checkpoint_task(
        self, result: WorkflowTaskResult, *, drained: bool
    ) -> WorkflowSnapshot:
        """Atomically settle required native evidence after original physical drain.

        Args:
            result: Complete private task result for the original native run.
            drained: Host observation; the original owner's callback must agree.

        Returns:
            The saved workflow after FULL checkpoint settlement.

        Raises:
            ValueError: Custody, mandatory data, capacity or exact replay differs.
        """
        tracker = self._tasks.get(result.attempt_id)
        if tracker is None or tracker.receipt is None:
            raise ValueError("native_receipt_custody_required")
        auth = tracker.authorization
        if (
            result.workflow_id != auth.claim.workflow_id
            or result.run_id != tracker.receipt.run_id
            or type(drained) is not bool
            or not drained
            or tracker.drain is not True
            or tracker.receipt.failure in {"no_run", "unknown"}
            or tracker.pending
        ):
            raise ValueError("native_retirement_unproven")
        native_run = self.db.get_run(result.run_id)
        if result.status == "completed" and (
            native_run is None
            or native_run["status"] != RUN_DONE
            or tracker.receipt.failure is not None
        ):
            raise ValueError("native_task_not_completed")
        submitted_digest = digest(result.canonical_json())
        if not result.summary.strip() or not result.handoff.strip():
            raise ValueError("required_result_missing")
        with self.db.connection() as conn:
            prior = conn.execute(
                "SELECT payload_json FROM workflow_checkpoints WHERE attempt_id=?",
                (result.attempt_id,),
            ).fetchone()
        if prior is not None:
            saved = WorkflowCheckpoint.model_validate_json(prior[0])
            if saved.submitted_digest != submitted_digest:
                raise ValueError("result_identity_conflict")
            self._checkpoint(result.workflow_id, result.attempt_id)
            tracker.receipt.activity_claim.settle(durable=True)
            return self.get(result.workflow_id)
        root = self.get(result.workflow_id).request.binding.locator
        checks = []
        for check_id in auth.task.expected_checks:
            spec = next((v for v in auth.task.verifiers if v.id == check_id), None)
            evidence = next(
                (e for e in reversed(tracker.records) if e.verifier_id == check_id),
                None,
            )
            fresh = bool(
                spec
                and evidence
                and evidence.fresh
                and evidence.input_manifest
                == _manifest(root, spec.input_paths, authorization=auth)
                and evidence.output_manifest is not None
                and evidence.output_manifest
                == _manifest(root, auth.task.outputs, authorization=auth)
                and evidence.shell_manifest == _shell_manifest(evidence.shell_path)
                and verifier_digest(spec) == spec.verifier_sha256
            )
            passed = bool(
                fresh
                and evidence.complete
                and evidence.cleanup_proven
                and evidence.exit_code == spec.expected_exit_code
            )
            status = "passed" if passed else "missing"
            if evidence is not None and not passed:
                if not fresh:
                    status = "stale"
                elif spec is not None and evidence.exit_code != spec.expected_exit_code:
                    status = "failed"
            checks.append(
                WorkflowCheck(
                    id=check_id,
                    status=status,
                    verifier=check_id,
                    arguments=spec.arguments if spec else (),
                    input_manifest=evidence.input_manifest if evidence else None,
                    output_manifest=evidence.output_manifest if evidence else None,
                    exit_code=evidence.exit_code if evidence else None,
                    reason="native_evidence"
                    if passed
                    else "native_evidence_unavailable",
                )
            )
        result = result.model_copy(update={"checks": tuple(checks)})
        with self.db.connection() as conn:
            windows = tuple(
                row[0]
                for row in conn.execute(
                    "SELECT id FROM change_snapshots WHERE run_id=? ORDER BY id LIMIT 65",
                    (result.run_id,),
                )
            )
        checkpoint = WorkflowCheckpoint(
            workflow_id=result.workflow_id,
            attempt_id=result.attempt_id,
            run_id=result.run_id,
            result_digest=digest(result.canonical_json()),
            submitted_digest=submitted_digest,
            input_manifest=_manifest(
                root,
                tuple(path.split(":", 1)[-1] for path in auth.task.inputs),
                authorization=auth,
            ),
            output_manifest=_manifest(root, auth.task.outputs, authorization=auth),
            verifiers=tuple(tracker.records),
            change_window_refs=windows,
        )
        payload = checkpoint.canonical_json()
        if len(payload.encode()) + len(result.canonical_json().encode()) > RECORD_BYTES:
            raise ValueError("workflow_result_record_capacity")
        with self.db.automatic_work.transaction() as conn:
            reservation = conn.execute(
                "SELECT bytes FROM workflow_payload_reservations WHERE attempt_id=? AND workflow_id=?",
                (result.attempt_id, result.workflow_id),
            ).fetchone()
            if (
                reservation is None
                or len(payload.encode()) + len(result.canonical_json().encode())
                > reservation[0]
            ):
                raise ValueError("workflow_result_record_capacity")
            prior = conn.execute(
                "SELECT payload_json FROM workflow_checkpoints WHERE attempt_id=?",
                (result.attempt_id,),
            ).fetchone()
            if prior and prior[0] != payload:
                raise ValueError("checkpoint_identity_conflict")
            self.db.workflow_runs._record_retirement(
                conn,
                result.workflow_id,
                result.attempt_id,
                run_id=result.run_id,
                drained=True,
            )
            self.db.workflow_runs._publish_result(conn, result)
            self.db.automatic_work._workflow_active_seconds(
                conn, result.workflow_id, observe=True
            )
            if prior is None:
                conn.execute(
                    "INSERT INTO workflow_checkpoints VALUES (?,?,?,?)",
                    (result.attempt_id, result.workflow_id, digest(payload), payload),
                )
            snapshot = self.get(result.workflow_id)
        tracker.receipt.activity_claim.settle(durable=True)
        return snapshot

    def _checkpoint(self, workflow_id, attempt_id):
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT payload_hash,payload_json FROM workflow_checkpoints WHERE workflow_id=? AND attempt_id=?",
                (workflow_id, attempt_id),
            ).fetchone()
        if row is None or digest(row[1]) != row[0]:
            raise ValueError("native_checkpoint_missing")
        checkpoint = WorkflowCheckpoint.model_validate_json(row[1])
        result = self.db.workflow_runs.read_result(workflow_id, attempt_id)
        if checkpoint.run_id != result.run_id or checkpoint.result_digest != digest(
            result.canonical_json()
        ):
            raise ValueError("native_checkpoint_result_changed")
        return checkpoint

    @goal_storage_operation
    def prepare_decision(self, workflow_id: str, *, phase_id: str) -> WorkflowDecision:
        """Inspect approval of an upcoming phase; never accept its eventual result."""
        return self._prepare_review(workflow_id, phase_id=phase_id, quality=False)

    @goal_storage_operation
    def prepare_quality_review(
        self, workflow_id: str, *, phase_id: str
    ) -> WorkflowDecision:
        """Inspect completed phase results separately from permission to start work."""
        return self._prepare_review(workflow_id, phase_id=phase_id, quality=True)

    def _prepare_review(
        self, workflow_id, *, phase_id, quality, authorization=None, partial=False
    ):
        snapshot = self.get(workflow_id)
        plan = snapshot.plans[-1]
        index = next(i for i, p in enumerate(plan.phases) if p.id == phase_id)
        phase = plan.phases[index]
        required = tuple(
            t for p in plan.phases[: index + int(quality)] for t in p.tasks
        )
        same_phase = {task.id: task for task in phase.tasks}
        deferred = tuple(
            (task.id, ref, same_phase[ref.split(":", 1)[0]].canonical_json())
            for task in phase.tasks
            for ref in task.inputs
            if not quality and ":" in ref and ref.split(":", 1)[0] in same_phase
        )
        inputs = tuple(
            path.split(":", 1)[-1]
            for task in phase.tasks
            for path in task.inputs
            if quality or ":" not in path or path.split(":", 1)[0] not in same_phase
        )
        manifest = (
            self._source_manifest(snapshot, inputs)
            if authorization is None
            else _manifest(
                snapshot.request.binding.locator,
                inputs,
                validate_path=lambda path: _bound_path(
                    authorization.binding, snapshot.request.binding.locator, path
                ),
            )
        )
        refs = []
        ready = manifest is not None and (
            authorization is not None or self._binding_reason(snapshot) is None
        )
        reason = "ready" if ready else "binding_or_inputs_unavailable"
        if not quality and any(
            set(t.expected_checks) != {v.id for v in t.verifiers} for t in phase.tasks
        ):
            ready, reason = False, "verifier_definition_missing"
        if snapshot.status == "recovery_required" or snapshot.accounting.uncertain:
            evidence = WorkflowExecutionRecovery(reason="uncertain_execution")
        else:
            for task in required:
                try:
                    result = self._fresh_task_result(snapshot, task, partial=partial)
                    if result is not None:
                        refs.append(result.attempt_id)
                except ValueError as exc:
                    ready, reason = False, str(exc)
            if quality and not partial:
                with self.db.connection() as conn:
                    settled = conn.execute(
                        "SELECT 1 FROM workflow_phase_attempts WHERE workflow_id=? AND plan_revision=? AND phase_id=? AND status='completed' AND retirement='drained'",
                        (workflow_id, len(snapshot.plans), phase_id),
                    ).fetchone()
                if settled is None:
                    ready, reason = False, "phase_not_settled"
            evidence = WorkflowQualityReview(
                predecessor_digest=digest(
                    json.dumps(
                        (
                            [
                                self.db.workflow_runs.read_result(
                                    workflow_id, ref
                                ).canonical_json()
                                for ref in refs
                            ],
                            deferred,
                        )
                    )
                ),
                input_manifest=manifest or "unavailable",
                ready=ready,
                reason=reason,
            )
        return WorkflowDecision(
            id=str(uuid4()),
            workflow_id=workflow_id,
            plan_revision=len(snapshot.plans),
            control_revision=snapshot.control_revision,
            phase_id=phase_id,
            action="complete" if quality else "approve",
            reason=evidence.reason,
            evidence=evidence,
            result_refs=tuple(refs),
        )

    def _fresh_task_result(self, snapshot, task, *, partial=False):
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT id,retirement,run_id FROM workflow_task_attempts WHERE workflow_id=? AND plan_revision=? AND task_id=? ORDER BY ordinal DESC LIMIT 1",
                (snapshot.id, len(snapshot.plans), task.id),
            ).fetchone()
        if partial and (row is None or row["run_id"] is None):
            return None
        if row is None or row["retirement"] != "drained":
            raise ValueError("native_checkpoint_missing")
        checkpoint = self._checkpoint(snapshot.id, row["id"])
        result = self.db.workflow_runs.read_result(snapshot.id, row["id"])
        if result.status != "completed" and not partial:
            raise ValueError("task_not_completed")
        if (
            not partial and checkpoint.output_manifest is None
        ) or checkpoint.output_manifest != self._source_manifest(
            snapshot, task.outputs
        ):
            raise ValueError("stale_task_output")
        if result.status == "completed" and (
            checkpoint.input_manifest is None
            or checkpoint.input_manifest
            != self._source_manifest(
                snapshot, tuple(path.split(":", 1)[-1] for path in task.inputs)
            )
        ):
            raise ValueError("stale_task_input")
        for check_id in task.expected_checks if result.status == "completed" else ():
            spec = next((v for v in task.verifiers if v.id == check_id), None)
            native = next(
                (
                    v
                    for v in reversed(checkpoint.verifiers)
                    if v.verifier_id == check_id
                ),
                None,
            )
            if (
                spec is None
                or not any(
                    check.id == check_id and check.status == "passed"
                    for check in result.checks
                )
                or native is None
                or not native.fresh
                or not native.complete
                or not native.cleanup_proven
                or native.exit_code != spec.expected_exit_code
                or native.verifier_sha256 != spec.verifier_sha256
                or native.arguments_digest != digest(json.dumps(spec.arguments))
                or native.shell_manifest != _shell_manifest(native.shell_path)
                or verifier_digest(spec) != spec.verifier_sha256
                or native.input_manifest
                != self._source_manifest(snapshot, spec.input_paths)
                or native.output_manifest != checkpoint.output_manifest
                or native.output_digest
                != digest(json.dumps((native.stdout, native.stderr)))
            ):
                raise ValueError("native_check_missing_or_stale")
        return result

    def _source_manifest(self, snapshot, paths):
        """Read bounded metadata under the selected binding's current exclusions."""
        from tldw_chatbook.Tools.local_tool_impls import resolve_workspace_path

        from .workflow_task_scope import _path_context

        if self._binding_reason(snapshot) is not None:
            return None
        root = Path(snapshot.request.binding.locator)
        live = next(
            (
                t.authorization.binding
                for t in self._tasks.values()
                if t.authorization.claim.workflow_id == snapshot.id
            ),
            None,
        )
        if live is None:
            registry = self.persistence.workspace_registry
            live = SimpleNamespace(
                exclusions_provider=lambda: tuple(
                    root / entry.path
                    for entry in registry.list_binding_exclusions(
                        snapshot.request.binding.binding_id
                    )
                )
            )
        return _manifest(
            root,
            paths,
            validate_path=lambda path: resolve_workspace_path(
                path, root, intent="read", context=_path_context(live)
            ),
        )

    def _binding_reason(self, snapshot):
        reference = snapshot.request.binding
        for tracker in self._tasks.values():
            auth = tracker.authorization
            if auth.claim.workflow_id == snapshot.id:
                from tldw_chatbook.Utils.filesystem_identity import (
                    capture_directory_chain,
                )

                roots = auth.footprint.reads + auth.footprint.writes
                if (
                    auth.binding.guard(False)
                    and Path(reference.locator).is_dir()
                    and (
                        not roots
                        or capture_directory_chain(Path(reference.locator))
                        == roots[0].root
                    )
                ):
                    return None
                return "binding_changed"
        registry = getattr(self.persistence, "workspace_registry", None)
        if registry is None:
            return "binding_missing"
        workspace = registry.get_workspace(reference.workspace_id)
        binding = registry.get_runtime_binding(reference.binding_id)
        if (
            workspace is None
            or workspace.archived
            or binding is None
            or binding.status.value != "ready"
            or binding.workspace_id != reference.workspace_id
            or binding.binding_kind.value != "local-filesystem"
            or binding.locator != reference.locator
            or binding.metadata.get("access", "ro") != reference.access
            or not Path(reference.locator).is_dir()
            or str(Path(reference.locator).resolve()) != reference.locator
        ):
            return "binding_changed"
        return None

    @goal_storage_operation
    def validate_decision(self, decision: WorkflowDecision) -> WorkflowDecision:
        """Require the exact current approval or quality evidence snapshot."""
        if type(decision.evidence) is not WorkflowQualityReview:
            raise ValueError("quality_decision_required")
        current = self._prepare_review(
            decision.workflow_id,
            phase_id=decision.phase_id,
            quality=decision.action == "complete",
        )
        if (
            decision.action not in {"approve", "complete"}
            or type(current.evidence) is not WorkflowQualityReview
            or not current.evidence.ready
            or current.model_copy(update={"id": decision.id}) != decision
        ):
            raise ValueError("stale_workflow_decision")
        return current

    @goal_storage_operation
    def project_recovery(
        self, audit: RuntimeRecoveryResult
    ) -> tuple[WorkflowSnapshot, ...]:
        """Project only this ledger's original current startup audit, once by CAS."""
        if audit is None or audit is not getattr(
            self.db.automatic_work, "recovery_result", None
        ):
            raise ValueError("untrusted_recovery_audit")
        with self.db.automatic_work.transaction() as conn:
            self.db.automatic_work._check_runtime_owner(conn, audit.owner_id)
            for workflow_id, revision, status in audit.workflows:
                conn.execute(
                    "UPDATE workflow_runs SET status=?,control_revision=control_revision+1 WHERE id=? AND control_revision=?",
                    (status, workflow_id, revision),
                )
            return tuple(self.get(workflow_id) for workflow_id, _, _ in audit.workflows)

    @goal_storage_operation
    def resume(
        self, workflow_id: str, *, expected_control_revision: int, command_id: str
    ) -> WorkflowSnapshot:
        """Request fresh approval; Task8 revalidates native provider/tool preparation."""
        from pydantic import TypeAdapter

        from .automatic_work_budget import AutomaticWorkLimits
        from .workflow_models import Identity

        TypeAdapter(Identity).validate_python(command_id)
        snapshot = self.get(workflow_id)
        with self.db.connection() as conn:
            prior = conn.execute(
                "SELECT workflow_id,control_revision FROM workflow_resume_commands WHERE id=?",
                (command_id,),
            ).fetchone()
        if prior is not None:
            if tuple(prior) != (workflow_id, expected_control_revision):
                raise ValueError("resume_command_conflict")
            return snapshot
        if snapshot.control_revision != expected_control_revision:
            raise ValueError("revision_conflict")
        if snapshot.status != "awaiting_resume" or snapshot.accounting.uncertain:
            raise ValueError("resume_requires_clean_checkpoint")
        if self._binding_reason(snapshot) is not None:
            raise ValueError("binding_changed")
        upcoming_checked = False
        for phase in snapshot.plans[-1].phases if snapshot.plans else ():
            with self.db.connection() as conn:
                rows = conn.execute(
                    "SELECT task_id,status,retirement FROM workflow_task_attempts WHERE workflow_id=? AND plan_revision=? AND phase_id=? ORDER BY ordinal DESC",
                    (workflow_id, len(snapshot.plans), phase.id),
                ).fetchall()
                latest = {}
                for row in rows:
                    latest.setdefault(row["task_id"], row)
                complete = all(
                    task.id in latest
                    and latest[task.id]["status"] == "completed"
                    and latest[task.id]["retirement"] == "drained"
                    for task in phase.tasks
                )
                has_results = conn.execute(
                    "SELECT 1 FROM workflow_task_attempts WHERE workflow_id=? AND plan_revision=? AND phase_id=? AND run_id IS NOT NULL",
                    (workflow_id, len(snapshot.plans), phase.id),
                ).fetchone()
            if has_results:
                review = self._prepare_review(
                    workflow_id, phase_id=phase.id, quality=True, partial=True
                )
                if (
                    type(review.evidence) is not WorkflowQualityReview
                    or not review.evidence.ready
                ):
                    raise ValueError("resume_evidence_stale")
            if not complete and not upcoming_checked:
                upcoming_checked = True
                review = self._prepare_review(
                    workflow_id, phase_id=phase.id, quality=False
                )
                if (
                    type(review.evidence) is not WorkflowQualityReview
                    or not review.evidence.ready
                ):
                    raise ValueError("resume_inputs_unavailable")
        with self.db.automatic_work.transaction() as conn:
            current = self.get(workflow_id)
            if (
                current.control_revision != expected_control_revision
                or current.status != "awaiting_resume"
            ):
                raise ValueError("revision_conflict")
            if conn.execute(
                "SELECT 1 FROM workflow_phase_attempts WHERE workflow_id=? AND status IN ('prepared','accepted','uncertain') LIMIT 1",
                (workflow_id,),
            ).fetchone():
                raise ValueError("resume_requires_clean_checkpoint")
            limits = AutomaticWorkLimits.from_settings("workflow_phase")
            for kind in ("generation", "child_launch", "model_call", "tokens"):
                self.db.automatic_work._check_admission(
                    conn, current.chain_id, kind=kind, amount=1, limits=limits
                )
            self.db.workflow_runs._cas(
                conn, workflow_id, expected_control_revision, "awaiting_approval"
            )
            conn.execute(
                "INSERT INTO workflow_resume_commands VALUES (?,?,?,?)",
                (
                    command_id,
                    workflow_id,
                    expected_control_revision,
                    expected_control_revision + 1,
                ),
            )
            return self.get(workflow_id)

    @goal_storage_operation
    def review_quality(
        self, decision: WorkflowDecision, *, accepted: bool
    ) -> WorkflowSnapshot:
        """Persist explicit quality review without clearing physical uncertainty."""
        if (
            type(accepted) is not bool
            or decision.action != "complete"
            or type(decision.evidence) is not WorkflowQualityReview
        ):
            raise ValueError("quality_decision_required")
        self.validate_decision(decision)
        with self.db.automatic_work.transaction() as conn:
            current = self.validate_decision(decision)
            snapshot = self.get(decision.workflow_id)
            status = (
                "paused"
                if not accepted
                else "completed"
                if snapshot.plans[-1].phases[-1].id == current.phase_id
                else "awaiting_approval"
            )
            if status == "completed" and self.db.workflow_runs._active(
                conn, snapshot.id
            ):
                raise ValueError("workflow_has_active_evidence")
            payload = decision.canonical_json()
            self.db.workflow_runs._capacity(conn, snapshot.id, len(payload.encode()))
            self.db.workflow_runs._cas(
                conn, snapshot.id, current.control_revision, status
            )
            conn.execute(
                "INSERT INTO workflow_decisions VALUES (?,?,?,?,?,?)",
                (
                    decision.id,
                    snapshot.id,
                    decision.plan_revision,
                    decision.phase_id,
                    decision.control_revision,
                    payload,
                ),
            )
            return self.get(snapshot.id)

    @goal_storage_operation
    def resolve_recovery(
        self,
        workflow_id: str,
        *,
        expected_control_revision: int,
        decision: WorkflowExecutionRecovery,
    ) -> WorkflowSnapshot:
        """Close recovery review while retaining unknown physical/accounting fences."""
        if type(decision) is not WorkflowExecutionRecovery:
            raise ValueError("recovery_decision_required")
        # Closing the saved review is not physical cleanup or permission to replay.
        with self.db.automatic_work.transaction() as conn:
            snapshot = self.get(workflow_id)
            if snapshot.status != "recovery_required":
                raise ValueError("recovery_unavailable")
            self.db.workflow_runs._cas(
                conn, workflow_id, expected_control_revision, "closed"
            )
            return self.get(workflow_id)
