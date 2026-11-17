"""Fixed adapters for the contained Console's actual local command owners.

Wire data is never a local source, review, durable plan or received-input proof.
Each capability below is resolved from an actual earlier backend receipt.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

from .console_execution_protocol import (
    ConsoleProtocolError,
    ConsoleWireMessage,
    decode_message,
    encode_message,
)


def _settings(value: Mapping[str, Any] | None):
    from .console_session_settings import ConsoleSessionSettings

    return None if value is None else ConsoleSessionSettings(**value)


def _settings_draft(value: Mapping[str, Any]):
    from .console_context_policy import (
        CompactionFailureBehavior,
        ConsoleContextPolicyOverrides,
        ContextBudgetMode,
        ContextCarryForwardMode,
        ContextCompactionMode,
        ContextCompactionRepresentation,
    )
    from .console_settings_apply import (
        ConsoleEndpointDraft,
        ConsoleModelDraft,
        ConsoleSettingsDraftState,
        ConsoleSettingsFieldDraft,
        ConsoleSettingsFieldProvenance,
    )

    def field_drafts(items):
        return tuple(
            ConsoleSettingsFieldDraft(
                name=item["name"],
                effective_value=item["effective_value"],
                profile_override=item["profile_override"],
                provenance=ConsoleSettingsFieldProvenance(item["provenance"]),
                dirty=item["dirty"],
            )
            for item in items
        )

    def endpoint(item):
        return None if item is None else ConsoleEndpointDraft(**item)

    context = dict(value["context_policy_overrides"])
    for name, enum in (
        ("budget_mode", ContextBudgetMode),
        ("compaction_mode", ContextCompactionMode),
        ("compaction_representation", ContextCompactionRepresentation),
        ("failure_behavior", CompactionFailureBehavior),
        ("carry_forward_mode", ContextCarryForwardMode),
    ):
        if context[name] is not None:
            context[name] = enum(context[name])
    return ConsoleSettingsDraftState(
        settings=_settings(value["settings"]),
        context_policy_overrides=ConsoleContextPolicyOverrides(**context),
        field_drafts=field_drafts(value["field_drafts"]),
        model_drafts=tuple(
            ConsoleModelDraft(
                provider=item["provider"],
                model=item["model"],
                settings=_settings(item["settings"]),
                field_drafts=field_drafts(item["field_drafts"]),
                endpoint_draft=endpoint(item["endpoint_draft"]),
            )
            for item in value["model_drafts"]
        ),
        endpoint_draft=endpoint(value["endpoint_draft"]),
    )


def _submission(value: Mapping[str, Any]):
    from .console_settings_apply import (
        ConsoleSettingsAction,
        ConsoleSettingsOrigin,
        ConsoleSettingsSubmission,
        ConsoleSettingsSurface,
    )

    return ConsoleSettingsSubmission(
        submission_id=value["submission_id"],
        action=ConsoleSettingsAction(value["action"]),
        surface=ConsoleSettingsSurface(value["surface"]),
        origin=ConsoleSettingsOrigin(**value["origin"]),
        draft=_settings_draft(value["draft"]),
        user_display_name_override=value["user_display_name_override"],
        default_field_mask=frozenset(value["default_field_mask"]),
    )


def _selection(value: Mapping[str, Any]):
    from .console_chat_models import (
        ConsoleEndpointProvenance,
        ConsoleProviderSelection,
        ConsoleStagedSource,
        ConsoleWorkspaceContext,
    )
    from .console_configuration_preparation import ConsoleTurnCaptureSelection
    from .console_roleplay_identity import (
        ConsolePresentationContext,
        ConsoleTranscriptStyle,
    )

    provider = dict(value["provider_selection"])
    provider["endpoint_provenance"] = ConsoleEndpointProvenance(
        provider["endpoint_provenance"]
    )
    workspace = provider["workspace_context"]
    if workspace is not None:
        provider["workspace_context"] = ConsoleWorkspaceContext(
            active_workspace_id=workspace["active_workspace_id"],
            staged_sources=tuple(
                ConsoleStagedSource(**item) for item in workspace["staged_sources"]
            ),
            active_run_id=workspace["active_run_id"],
            handoff_id=workspace["handoff_id"],
        )
    presentation = value["presentation_context"]
    if presentation is not None:
        presentation = dict(presentation)
        presentation["transcript_style"] = ConsoleTranscriptStyle(
            presentation["transcript_style"]
        )
    return ConsoleTurnCaptureSelection(
        provider_selection=ConsoleProviderSelection(**provider),
        presentation_context=(
            None if presentation is None else ConsolePresentationContext(**presentation)
        ),
        rag_defaults=value["rag_defaults"],
        tool_configuration=value["tool_configuration"],
        skill_workspace_id=value["skill_workspace_id"],
        project_bindings_eligible=value["project_bindings_eligible"],
        agent_runtime_enabled=value["agent_runtime_enabled"],
    )


class _BackendCommandAdapter:
    """Resolve only actual results of commands dispatched on this backend."""

    def __init__(self, controls: Any) -> None:
        from .console_control_dispatch import ConsoleLocalControls

        if type(controls) is not ConsoleLocalControls:
            raise TypeError("The original Console controls are required")
        controls._require_graph()
        self.controls = controls
        self._receipts: dict[str, tuple[ConsoleWireMessage, Any]] = {}
        self._commands: dict[str, Any] = {}
        self._review_presentations: dict[str, tuple[Any, Any]] = {}

    def _result(self, request_id: str, operations: tuple[str, ...]) -> Any:
        entry = self._receipts.get(request_id)
        if entry is None or entry[0].operation not in operations:
            raise ConsoleProtocolError("unknown_original_result")
        message, receipt = entry
        if (
            message.identity.profile_epoch != self.controls.profile_epoch
            or message.identity.service_epoch != self.controls.service_epoch
            or not receipt.applied.done()
            or receipt.applied.cancelled()
        ):
            raise ConsoleProtocolError("original_result_not_settled")
        self.controls._require_graph()
        return receipt.applied.result()

    def command(self, message: ConsoleWireMessage):
        """Construct one fixed local command from validated detached data."""
        from . import console_control_dispatch as controls
        from . import console_control_publication as publication

        if type(message) is not ConsoleWireMessage or message.kind != "command":
            raise ConsoleProtocolError("expected_command")
        # Thaw only through the closed codec, not a generic object decoder.
        value = encode_message(message)["payload"]
        identity = controls.ControlIdentity(
            **{
                item.name: getattr(message.identity, item.name)
                for item in fields(controls.ControlIdentity)
            }
        )
        operation = message.operation
        if operation == "UpdateInputs":
            prefill = value["prefill"]
            return controls.UpdateInputs(
                identity,
                value["draft"],
                tuple(value["authored_token"]),
                expected_settings_revision=value["expected_settings_revision"],
                expected_settings=_settings(value["expected_settings"]),
                expected_attachment_ids=(
                    None
                    if value["expected_attachment_ids"] is None
                    else tuple(value["expected_attachment_ids"])
                ),
                remove_attachment_ids=tuple(value["remove_attachment_ids"]),
                prefill=None if prefill is None else controls.PrefillUpdate(**prefill),
            )
        if operation == "ApplySettings":
            return controls.ApplySettings(identity, _submission(value["submission"]))
        if operation == "ReceiveTurn":
            return controls.ReceiveTurn(
                identity,
                value["turn_id"],
                value["draft_revision"],
                _selection(value["selection"]),
                expected_settings_revision=value["expected_settings_revision"],
                expected_attachment_revision=value["expected_attachment_revision"],
                expected_prefill_revision=value["expected_prefill_revision"],
                wire_inputs=message,
            )
        if operation == "ReadProjection":
            # The optional wire cache hint is not a local read/check authority.
            return controls.ReadProjection(identity)
        plain = {
            "OpenSession": controls.OpenSession,
            "CloseSession": controls.CloseSession,
            "ProjectDecisionPresentation": controls.ProjectDecisionPresentation,
            "CancelTurn": controls.CancelTurn,
            "FenceSource": controls.FenceSource,
            "CloseIntake": controls.CloseIntake,
            "Drain": controls.Drain,
            "Shutdown": controls.Shutdown,
        }
        if operation in plain:
            return plain[operation](identity, **value)
        if operation == "CommitPlanAck":
            from .console_chat_store import ConsoleDurableAcceptanceFingerprint

            return controls.CommitPlanAck(
                identity,
                value["received_request_id"],
                value["turn_id"],
                value["ordinal"],
                ConsoleDurableAcceptanceFingerprint(**value["fingerprint"]),
            )
        if operation == "ResolveDecision":
            return controls.ResolveDecision(identity, self._decision(value["decision"]))
        if operation == "PermissionAction":
            return controls.PermissionAction(
                identity, self._permission(value["operation"])
            )
        if operation == "ApplySourceRevision":
            result = self._result(
                value["publication_request_id"],
                (
                    "ReplaceConfiguration",
                    "RefreshConfiguration",
                    "SetWorkspaceBindingAccess",
                ),
            )
            return publication.ApplySourceRevision(identity, result)
        fence = self._result(value["fence_request_id"], ("FenceSource",))
        if operation == "RefreshConfiguration":
            return publication.RefreshConfiguration(identity, fence)
        if operation == "SetWorkspaceBindingAccess":
            return publication.SetWorkspaceBindingAccess(
                identity, fence, value["binding_id"], value["allow_write"]
            )
        if operation == "ReplaceConfiguration":
            from tldw_chatbook.config import ConfigFileSnapshot

            expected = value["expected_snapshot"]
            return publication.ReplaceConfiguration(
                identity,
                fence,
                value["serialized"],
                None
                if expected is None
                else ConfigFileSnapshot(Path(expected["path"]), expected["serialized"]),
                value["create_backup"],
            )
        raise ConsoleProtocolError("unsupported_command")

    def retain(self, message: ConsoleWireMessage, receipt: Any, command: Any) -> None:
        """Retain the actual original dispatch result, never a wire substitute."""
        identity = receipt.identity
        if (
            message.identity.request_id in self._receipts
            or identity.request_id != message.identity.request_id
            or identity.profile_epoch != self.controls.profile_epoch
            or identity.service_epoch != self.controls.service_epoch
            or identity.request_id not in self.controls._requests
        ):
            raise ConsoleProtocolError("receipt_association_changed")
        self._receipts[identity.request_id] = (message, receipt)
        if command.identity is not identity:
            raise ConsoleProtocolError("original_command_identity_changed")
        self._commands[identity.request_id] = command

    def _decision(self, value):
        from . import console_control_decisions as decisions

        kind = value["kind"]
        arguments = {name: item for name, item in value.items() if name != "kind"}
        if kind == "InitialHookDecision":
            from .console_hook_review import HookReviewResult

            reference = arguments.pop("presentation_request_id")
            original = self._review_presentations.get(reference)
            if original is None:
                raise ConsoleProtocolError("unknown_hook_presentation")
            result_kind = arguments["result"]["kind"]
            token, snapshot = original
            arguments["result"] = HookReviewResult(
                result_kind, snapshot if result_kind == "ready" else None
            )
            arguments["presentation_token"] = token
            return decisions.InitialHookDecision(**arguments)
        classes = (
            decisions.ApprovalRoundDecision,
            decisions.ApprovalPresentationDecision,
            decisions.ProjectBindingDecision,
            decisions.ProjectDispatchDecision,
            decisions.ChatCreateDecision,
            decisions.QuestionDecision,
            decisions.SkillInstallDecision,
            decisions.SkillScriptDecision,
            decisions.WorktreeMergeDecision,
        )
        if kind in {"ApprovalRoundDecision", "ApprovalPresentationDecision"}:
            arguments["call_ids"] = tuple(arguments["call_ids"])
        selected = next((item for item in classes if item.__name__ == kind), None)
        if selected is None:
            raise ConsoleProtocolError("unknown_decision")
        return selected(**arguments)

    def _permission(self, value):
        from . import console_control_permissions as permissions
        from tldw_chatbook.MCP.hub_tool_catalog import HubTool

        kind = value["kind"]
        arguments = {name: item for name, item in value.items() if name != "kind"}
        selected = next(
            (
                item
                for item in permissions.PERMISSION_OPERATION_TYPES
                if item.__name__ == kind
            ),
            None,
        )
        if selected is None:
            raise ConsoleProtocolError("unknown_permission_operation")
        if kind in {"ImportUnboundProfile", "ConfirmFirstBind"}:
            prior = arguments.pop("review_request_id")
            original = self._result(prior, ("PermissionAction",))
            expected = (
                "InspectImport" if kind == "ImportUnboundProfile" else "ReviewFirstBind"
            )
            if self._receipts[prior][0].payload["operation"]["kind"] != expected:
                raise ConsoleProtocolError("permission_review_kind_changed")
            arguments["review"] = original
        if "tool" in arguments and arguments["tool"] is not None:
            arguments["tool"] = HubTool(**arguments["tool"])
        if "tools" in arguments:
            arguments["tools"] = [HubTool(**item) for item in arguments["tools"]]
        if kind == "InspectImport":
            from tldw_chatbook.Tool_Packs.importer import ServerMapping

            arguments["archive_path"] = Path(arguments["archive_path"])
            arguments["mappings"] = tuple(
                ServerMapping(**item) for item in arguments["mappings"]
            )
        if kind == "ReviewFirstBind":
            from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults

            arguments["intended_defaults"] = WorkspaceAssistantDefaults(
                **arguments["intended_defaults"]
            )
        return selected(**arguments)


import asyncio
import queue
import threading
from typing import BinaryIO

from tldw_chatbook.MCP.permission_process import read_message, write_message
from tldw_chatbook.Chat.console_execution_process import _ConsoleFrameInput


class _BackendChannel:
    """Retain actual pipe workers; Shutdown is the last legal input frame."""

    def __init__(self, input_stream: BinaryIO, output_stream: BinaryIO) -> None:
        self.input = input_stream
        self.output = output_stream
        self.loop = asyncio.get_running_loop()
        self.inbox: asyncio.Queue[ConsoleWireMessage | BaseException] = asyncio.Queue()
        self.outbox: queue.Queue[dict | None] = queue.Queue()
        self.failure: asyncio.Future[BaseException] = self.loop.create_future()
        self.reader = threading.Thread(
            target=self._read, name="console-backend-reader", daemon=True
        )
        self.writer = threading.Thread(
            target=self._write, name="console-backend-writer", daemon=True
        )
        self.started: list[threading.Thread] = []
        self.stopping = False
        self.shutdown_read = False

    def start(self) -> None:
        if self.started:
            raise RuntimeError("Backend channel already started")
        for thread in (self.writer, self.reader):
            try:
                thread.start()
            finally:
                if thread.ident is not None:
                    self.started.append(thread)

    def _failed(self, error: BaseException) -> None:
        if not self.failure.done():
            self.failure.set_result(error)
            self.inbox.put_nowait(error)

    def _report(self, error: BaseException) -> None:
        try:
            self.loop.call_soon_threadsafe(self._failed, error)
        except RuntimeError:
            # Actual thread state is retained by the enclosing worker even if
            # its loop has failed. This path cannot produce a success receipt.
            pass

    def _read(self) -> None:
        try:
            while True:
                message = decode_message(read_message(_ConsoleFrameInput(self.input)))
                if message.kind != "command":
                    raise ConsoleProtocolError("expected_command")
                if message.operation == "Shutdown":
                    self.shutdown_read = True
                self.loop.call_soon_threadsafe(self.inbox.put_nowait, message)
                if self.shutdown_read:
                    return
        except BaseException as error:
            self._report(error)

    def _write(self) -> None:
        try:
            while True:
                value = self.outbox.get()
                if value is None:
                    return
                write_message(self.output, value)
        except BaseException as error:
            self._report(error)

    def send(self, message: ConsoleWireMessage) -> None:
        if self.stopping or self.failure.done():
            raise ConsoleProtocolError("backend_channel_failed")
        self.outbox.put_nowait(encode_message(message))

    async def finish_output(self) -> None:
        """Flush ordered receipts; independently retain any pipe failure."""
        if not self.stopping:
            self.stopping = True
            self.outbox.put_nowait(None)
        if self.writer in self.started:
            await asyncio.to_thread(self.writer.join)
        if self.failure.done():
            raise self.failure.result()
        if not self.shutdown_read:
            raise ConsoleProtocolError("backend_intake_not_closed")
        if self.reader in self.started:
            await asyncio.to_thread(self.reader.join)


def _initialization(value: object):
    """Validate only inert parent-selected paths before original bootstrap."""
    from pathlib import Path
    from tldw_chatbook.Chat.console_execution_protocol import WireIdentity

    fields = {
        "initialize",
        "profile_epoch",
        "service_epoch",
        "request_id",
        "configuration_path",
        "data_directory",
    }
    if (
        type(value) is not dict
        or set(value) != fields
        or type(value["initialize"]) is not int
        or value["initialize"] != 1
    ):
        raise ConsoleProtocolError("invalid_initialization")
    identity = WireIdentity(
        value["profile_epoch"], value["service_epoch"], 1, value["request_id"]
    )
    paths = []
    for name in ("configuration_path", "data_directory"):
        item = value[name]
        if type(item) is not str or not item or not Path(item).is_absolute():
            raise ConsoleProtocolError("invalid_profile_path")
        paths.append(Path(item))
    return identity, paths[0], paths[1]


class _BackendService:
    """Correlate pipe messages with the retained original local commands."""

    def __init__(self, backend, channel: _BackendChannel) -> None:
        if backend._worker_service is not None:
            raise RuntimeError("Backend channel already attached")
        self.backend = backend
        self.channel = channel
        self.sequence = 0
        self.handlers: set[asyncio.Task] = set()
        self.handler_error = None
        self._received_requests: set[str] = set()
        self._projection_messages = None
        self._projection_key = None
        backend._worker_service = self

    def _identity(self, original):
        from dataclasses import replace

        return replace(original, sequence=self.sequence + 1)

    def emit(self, original, kind, operation, payload) -> None:
        message = ConsoleWireMessage(self._identity(original), kind, operation, payload)
        self.channel.send(message)
        self.sequence = message.identity.sequence

    def applied(self, original, command, result) -> None:
        from .console_control_dispatch import PermissionAction, ReadProjection
        from .console_execution_permission_values import encode_permission_result
        from .console_execution_result_values import encode_control_applied

        identity = self._identity(original)
        if type(command) is ReadProjection:
            from .console_execution_projection_result import (
                encode_projection_payload,
                projection_key,
            )
            from .console_execution_projection_values import encode_message_projection

            key = projection_key(result)
            current = key == self._projection_key
            request = self.backend.adapter._receipts[original.request_id][0]
            known = request.payload.get("known_display_revision")
            if not current:
                self._projection_messages = [
                    encode_message_projection(item) for item in result.messages
                ]
                self._projection_key = key
            body = encode_projection_payload(
                result,
                messages=None
                if current and known == result.display_revision
                and request.payload.get("known_marker_revision") == result.marker_revision
                else self._projection_messages,
            )
            body["session_view"] = self.backend.presentation.projection_view(
                command.identity, result
            )
            message = ConsoleWireMessage(identity, "applied", "ReadProjection", body)
            self.channel.send(message)
            self.sequence = message.identity.sequence
            return
        message = (
            encode_permission_result(identity, command.operation, result)
            if type(command) is PermissionAction
            else encode_control_applied(identity, command, result)
        )
        self.channel.send(message)
        self.sequence = message.identity.sequence

    def publish_commit_plan(self, original, pending) -> None:
        from .console_control_dispatch import ReceiveTurn
        from .console_execution_result_values import encode_commit_planned

        request_id = original.request_id
        entry = self.backend.adapter._receipts.get(request_id)
        command = self.backend.adapter._commands.get(request_id)
        if (
            self.backend._worker_service is not self
            or request_id not in self._received_requests
            or entry is None
            or type(command) is not ReceiveTurn
            or command.identity is not original
            or entry[1].identity is not original
            or entry[1].work is not pending.task
            or command.turn_id != pending.projection.turn_id
        ):
            raise ConsoleProtocolError("commit_plan_received_owner_changed")
        message = encode_commit_planned(
            self._identity(entry[0].identity), pending.projection
        )
        self.channel.send(message)
        self.sequence = message.identity.sequence

    def accept(self, message: ConsoleWireMessage):
        if message.operation == "CloseIntake":
            self._pending_turn_locations()
        try:
            receipt = self.backend.dispatch(message)
        except Exception as error:
            self._failure(message.operation, "admission", error)
            self.emit(
                message.identity,
                "failed",
                message.operation,
                {"code": "command_refused", "stage": "admission"},
            )
            return None
        command = self.backend.adapter._commands[message.identity.request_id]
        # ReceiveTurn dispatch returns only after its original claim exists.
        self.emit(
            message.identity,
            "received",
            message.operation,
            {"sequence": message.identity.sequence},
        )
        self._received_requests.add(message.identity.request_id)
        task = asyncio.create_task(self.finish(message, command, receipt))
        self.handlers.add(task)
        task.add_done_callback(self._finished)
        return task

    def _pending_turn_locations(self) -> None:
        """Log bounded code locations before original lifecycle cancellation."""
        import inspect
        import sys
        from loguru import logger

        for original, receipt in tuple(self.backend.adapter._receipts.values()):
            if (
                original.operation != "ReceiveTurn"
                or receipt.work is None
                or receipt.work.done()
            ):
                continue
            current = receipt.work.get_coro()
            frames = []
            for _ in range(16):
                if inspect.iscoroutine(current):
                    frame, following = current.cr_frame, current.cr_await
                elif inspect.isgenerator(current):
                    frame, following = current.gi_frame, current.gi_yieldfrom
                else:
                    break
                if frame is not None:
                    frames.append(f"{frame.f_code.co_name}:{frame.f_lineno}")
                current = following
            logger.warning(
                "Console backend pending ReceiveTurn; frames={}", " <- ".join(frames)
            )
            retirement = receipt.turn_retirement
            if retirement is None:
                continue
            active_frames = sys._current_frames()
            for index, observation in enumerate(retirement.reads):
                if (
                    not observation.entered
                    or observation.closed
                    or observation.creator is None
                ):
                    continue
                thread = observation.creator[1]
                frame = active_frames.get(thread.ident)
                frames = []
                for _ in range(24):
                    if frame is None:
                        break
                    frames.append(f"{frame.f_code.co_name}:{frame.f_lineno}")
                    frame = frame.f_back
                logger.warning(
                    "Console backend active received read {}/{}; frames={}",
                    index + 1,
                    len(retirement.reads),
                    " <- ".join(frames),
                )

    def _finished(self, task) -> None:
        self.handlers.discard(task)
        error = asyncio.CancelledError() if task.cancelled() else task.exception()
        if error is not None:
            if self.handler_error is None:
                self.handler_error = error
            self.channel._failed(error)

    @staticmethod
    def _failure(operation: str, stage: str, error: BaseException) -> None:
        """Retain safe failure locations without exception text or frame locals."""
        from loguru import logger

        frames = []
        current = error.__traceback__
        while current is not None:
            code = current.tb_frame.f_code
            frames.append(f"{code.co_name}:{current.tb_lineno}")
            current = current.tb_next
        logger.warning(
            "Console backend {} {} failed ({}); frames={}",
            operation,
            stage,
            type(error).__name__,
            " <- ".join(frames[-16:]),
        )

    async def finish(self, message, command, receipt) -> None:
        from .console_control_dispatch import (
            ApplySourceRevision,
            CloseIntake,
            CommitPlanAck,
            Drain,
            FenceSource,
            OpenSession,
            ResolveDecision,
            ProjectDecisionPresentation,
            SetWorkspaceBindingAccess,
            Shutdown,
            UpdateInputs,
        )

        result = None
        applied = False
        source_retired = False
        try:
            result = await asyncio.shield(receipt.applied)
            if type(command) is FenceSource:
                # The original finish_source_fence drained actual finite effects;
                # validate its exact issued fence before exporting that outcome.
                self.backend.controls._require_source_publication(result)
                source_retired = True
            elif type(command) is SetWorkspaceBindingAccess:
                # This original publication exists only after its native Registry
                # scope and preparation-read producer have physically retired.
                self.backend.controls._require_publication(result)
                source_retired = True
            self.applied(message.identity, command, result)
            applied = True
        except BaseException as error:
            from .console_backend_view_lifecycle import ConsoleBackendViewSuperseded

            superseded = (
                message.operation == "ReadProjection"
                and "view_checkpoint" in message.payload
                and type(error) is ConsoleBackendViewSuperseded
            )
            self._failure(message.operation, "applied", error)
            self.emit(
                message.identity,
                "failed",
                message.operation,
                {
                    "code": "cancelled"
                    if isinstance(error, asyncio.CancelledError)
                    else "view_checkpoint_superseded" if superseded else "command_failed",
                    "stage": "applied",
                },
            )
        retired = False
        try:
            if type(command) is Shutdown:
                await self.backend.aclose()
                retired = applied and result is True and self.backend.closed
            elif receipt.retired is not None:
                retired = await asyncio.shield(receipt.retired)
                if type(retired) is not bool:
                    raise ConsoleProtocolError("invalid_original_retirement")
            elif type(command) in (FenceSource, SetWorkspaceBindingAccess):
                retired = source_retired
            elif type(command) is Drain:
                # The original drain's successful return follows all original
                # native/Task checks. A bare task-done flag is never this proof.
                if receipt.work is not None:
                    await asyncio.shield(receipt.work)
                retired = applied and result is True
            elif receipt.work is None and type(command) in (
                UpdateInputs,
                ResolveDecision,
                ProjectDecisionPresentation,
                CommitPlanAck,
                CloseIntake,
                ApplySourceRevision,
                OpenSession,
            ):
                # These exact synchronous branches issue no operation-owned
                # native work; decision/ACK targets keep their own receipts.
                retired = applied and (
                    type(command) is not OpenSession or command.conversation_id is None
                )
            if receipt.work is not None:
                try:
                    await asyncio.shield(receipt.work)
                except BaseException as error:
                    self._failure(message.operation, "terminal", error)
                    if message.operation == "ReceiveTurn" and applied:
                        self.emit(
                            message.identity,
                            "failed",
                            message.operation,
                            {
                                "code": "cancelled"
                                if isinstance(error, asyncio.CancelledError)
                                else "turn_failed",
                                "stage": "terminal",
                            },
                        )
        except BaseException as error:
            self._failure(message.operation, "retirement", error)
            retired = False
        if message.operation == "ReceiveTurn" and applied:
            from .console_execution_retirement import ConsoleTurnRetirement, _definitions_current

            terminal = receipt.turn_retirement
            if (
                type(terminal) is ConsoleTurnRetirement
                and terminal.controls is self.backend.controls
                and terminal.identity is command.identity
                and terminal.turn_id == command.turn_id
                and terminal.logical_closed
                and type(terminal.accepted) is bool
                and _definitions_current()
            ):
                # This is the original custody callback's result. Physical
                # retirement remains the separate receipt immediately below.
                self.emit(message.identity, "terminal", "ReceiveTurn", {
                    "turn_id": command.turn_id, "accepted": terminal.accepted,
                })
            else:
                retired = False
        self.emit(
            message.identity,
            "retired",
            message.operation,
            {"operation_id": message.identity.request_id, "retired": retired},
        )
        if retired:
            self._release_result(message)

    def _release_result(self, message) -> None:
        self.backend.presentation._projection_views.pop(
            message.identity.request_id, None
        )
        # Only these actual results can be consumed by a later named command.
        retained = message.operation in {
            "FenceSource",
            "ReplaceConfiguration",
            "RefreshConfiguration",
            "SetWorkspaceBindingAccess",
        } or (
            message.operation == "PermissionAction"
            and message.payload["operation"]["kind"]
            in {"InspectImport", "ReviewFirstBind"}
        )
        self.backend.adapter._commands.pop(message.identity.request_id, None)
        self._received_requests.discard(message.identity.request_id)
        if not retained:
            self.backend.adapter._receipts.pop(message.identity.request_id, None)


async def _serve_backend(initial, input_stream, output_stream) -> None:
    import os
    from .console_execution_backend import ConsoleExecutionBackend
    from .console_execution_profile import open_console_backend_profile

    identity, config_path, directory = _initialization(initial)
    profile = open_console_backend_profile(
        expected_config_path=config_path,
        expected_data_directory=directory,
    )
    backend = channel = service = None
    error = None
    try:
        backend = ConsoleExecutionBackend.__new__(ConsoleExecutionBackend)
        ConsoleExecutionBackend.__init__(
            backend,
            profile,
            profile_epoch=identity.profile_epoch,
            service_epoch=identity.service_epoch,
        )
        channel = _BackendChannel(input_stream, output_stream)
        service = _BackendService(backend, channel)
        backend.start()
        channel.start()
        service.emit(
            identity,
            "ready",
            "Initialize",
            {
                "pid": os.getpid(),
                "source_revision": backend.controls.source_revision,
            },
        )
        while True:
            message = await channel.inbox.get()
            if isinstance(message, BaseException):
                raise message
            task = service.accept(message)
            if message.operation == "Shutdown":
                if task is None:
                    raise ConsoleProtocolError("shutdown_not_admitted")
                await asyncio.shield(task)
                break
    except BaseException as failure:
        error = failure
    finally:
        try:
            if backend is not None:
                await backend.aclose()
            else:
                await profile.aclose()
        except BaseException as failure:
            if error is None:
                error = failure
            # Backend construction itself may not have reached its cleanup
            # fields. Independently close the still-retained profile owner.
            try:
                await profile.aclose()
            except BaseException:
                pass  # First error stays negative; profile retains its own error.
        if service is not None and service.handlers:
            results = await asyncio.gather(
                *tuple(service.handlers), return_exceptions=True
            )
            if error is None:
                error = next(
                    (item for item in results if isinstance(item, BaseException)), None
                )
        try:
            if channel is not None:
                await channel.finish_output()
        except BaseException as failure:
            if error is None:
                error = failure
    if error is None and service is not None:
        error = service.handler_error
    if error is not None:
        raise error


def main() -> int:
    """Run the canonical module once, with original isolated profile selection."""
    import sys

    input_stream, output_stream = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr
    initial = read_message(_ConsoleFrameInput(input_stream))
    asyncio.run(_serve_backend(initial, input_stream, output_stream))
    return 0


if __name__ == "__main__":
    # runpy's __main__ object is not a second authority/definition namespace.
    from tldw_chatbook.Chat.console_execution_worker import main as _canonical_main

    raise SystemExit(_canonical_main())
