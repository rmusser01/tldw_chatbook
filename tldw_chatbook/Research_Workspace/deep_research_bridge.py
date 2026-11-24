"""Bind native Research Runs to their workspace launch and reviewed return."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import json
from sqlite3 import IntegrityError
from uuid import uuid4

from .chat_models import ResearchRetrievalSettings
from .chat_service import MAX_ANSWER_BYTES
from .contracts import ProcessingRoute, QualifiedWorkspaceRef, WorkspaceDataSource
from .deep_research_context import ResearchLaunchContext, ResearchWorkspaceRunRef
from .overlay_store import ResearchPresentationOverlayStore
from .studio_generation import StudioGeneratedOutput
from .studio_models import (
    ResearchGenerationConfig,
    ResearchStudioRequest,
    WorkspaceOutputRef,
)


class ResearchReturnMismatch(ValueError):
    """The native run or bundle belongs to a different launch."""


@dataclass(frozen=True, slots=True)
class DeepResearchReportPreview:
    target: ResearchWorkspaceRunRef
    context: ResearchLaunchContext
    report: str = field(repr=False)
    source_count: int | None = None

    def studio_request(self, *, context_revision: int) -> ResearchStudioRequest:
        identity = json.dumps(asdict(self.target), sort_keys=True)
        operation = "research-run-" + sha256(identity.encode("utf-8")).hexdigest()
        mode = next(
            mode for row in self.context.source_snapshot for mode in row.ready_modes
        )
        return ResearchStudioRequest(
            operation_id=operation,
            workspace=self.target.origin,
            output_kind="report",
            source_snapshot=self.context.source_snapshot,
            generation=ResearchGenerationConfig(
                title=f"Deep Research: {self.context.query}"[:120].rstrip(),
                retrieval=ResearchRetrievalSettings(mode=mode, history_limit=0),
            ),
            processing_route=ProcessingRoute(
                self.target.origin.data_source, "research"
            ),
            context_revision=context_revision,
            research_run_ref=self.target,
        )

    async def save_report(
        self,
        port,
        request: ResearchStudioRequest,
        *,
        supersedes: WorkspaceOutputRef | None = None,
    ) -> WorkspaceOutputRef:
        if request != self.studio_request(context_revision=request.context_revision):
            raise ResearchReturnMismatch(
                "The Report save belongs to a different run or workspace"
            )
        if supersedes is not None:
            request = replace(
                request, operation_id=f"research-run-version-{uuid4().hex}"
            )
        return await port.save_imported_report(
            request,
            StudioGeneratedOutput("complete", text=self.report),
            supersedes=supersedes,
        )


class DeepResearchBridge:
    """Read and write only through the selected native Research scope."""

    def __init__(self, scope, overlay: ResearchPresentationOverlayStore) -> None:
        self.scope = scope
        self.overlay = overlay

    async def launch_local(
        self, context: ResearchLaunchContext
    ) -> ResearchWorkspaceRunRef:
        if context.origin.data_source is not WorkspaceDataSource.LOCAL:
            raise ValueError("Choose a Local workspace for this launch")
        target = context.target(context.launch_id)
        await asyncio.to_thread(self.overlay.remember_research_launch, target, context)
        run = await self.scope.get_run(target.run_id, mode="local")
        if run is None:
            try:
                run = await self.scope.create_run(
                    id=target.run_id,
                    mode="local",
                    query=context.query,
                    chat_handoff={"research_workspace": context.to_payload()},
                )
            except IntegrityError:
                run = await self.scope.get_run(target.run_id, mode="local")
                if run is None:
                    raise
        if self.target_from_run(run) != target or self._local_context(run) != context:
            raise ResearchReturnMismatch(
                "The saved run belongs to a different workspace launch"
            )
        return target

    async def launch_server(
        self, context: ResearchLaunchContext
    ) -> ResearchWorkspaceRunRef:
        if context.origin.data_source is not WorkspaceDataSource.SERVER:
            raise ValueError("Choose a Server workspace for this launch")
        run = await self.scope.create_run(
            mode="server",
            query=context.query,
            chat_handoff={"research_workspace": context.to_payload()},
        )
        target = self.target_from_run(run, server_origin=context.origin)
        if target is None or self._run_context(run, context.origin) != context:
            raise ResearchReturnMismatch(
                "The saved Server run belongs to a different workspace launch"
            )
        await asyncio.to_thread(self.overlay.remember_research_launch, target, context)
        return target

    @staticmethod
    def _local_context(run: Mapping) -> ResearchLaunchContext:
        handoff = run.get("chat_handoff")
        if not isinstance(handoff, Mapping):
            raise ValueError("This run has no workspace launch")
        return ResearchLaunchContext.from_payload(handoff.get("research_workspace"))

    @classmethod
    def _run_context(
        cls, run: Mapping, origin: QualifiedWorkspaceRef | None
    ) -> ResearchLaunchContext:
        if run.get("source") == "local":
            context = cls._local_context(run)
            if context.origin.data_source is not WorkspaceDataSource.LOCAL:
                raise ResearchReturnMismatch("The local run has a different authority")
            return context
        if run.get("source") == "server" and origin is not None:
            return ResearchLaunchContext.from_server_projection(
                origin, run.get("workspace_launch")
            )
        raise ResearchReturnMismatch("The run has no verified workspace launch")

    @classmethod
    def target_from_run(
        cls, run: object, *, server_origin: QualifiedWorkspaceRef | None = None
    ) -> ResearchWorkspaceRunRef | None:
        if not isinstance(run, Mapping):
            return None
        try:
            context = cls._run_context(run, server_origin)
            query = run.get("query", context.query)
            if not isinstance(query, str) or " ".join(query.split()) != context.query:
                return None
            return context.target(run.get("id"))
        except (TypeError, ValueError):
            return None

    async def preview_return(
        self, target: ResearchWorkspaceRunRef
    ) -> DeepResearchReportPreview:
        mode = target.origin.data_source.value
        run = await self.scope.get_run(target.run_id, mode=mode)
        if self.target_from_run(run, server_origin=target.origin) != target:
            raise ResearchReturnMismatch(
                "This run belongs to a different workspace or launch"
            )
        context = self._run_context(run, target.origin)
        overlay = await asyncio.to_thread(self.overlay.load, target.origin)
        remembered = (
            overlay.deep_research_launches.get(target.run_id) if overlay else None
        )
        if remembered is not None and remembered != context:
            raise ResearchReturnMismatch(
                "The native run differs from the saved workspace launch"
            )
        bundle = await self.scope.get_bundle(mode=mode, run_id=target.run_id)
        if not isinstance(bundle, Mapping) or bundle.get("backend") != mode:
            raise ResearchReturnMismatch(
                "The bundle belongs to a different Research backend"
            )
        if (
            "run" in bundle
            and self.target_from_run(bundle["run"], server_origin=target.origin)
            != target
        ):
            raise ResearchReturnMismatch(
                "The bundle belongs to a different Research run"
            )
        if mode == "server" and (
            bundle.get("run_id") != target.run_id
            or bundle.get("workspace_launch") != run.get("workspace_launch")
        ):
            raise ResearchReturnMismatch(
                "The Server bundle belongs to a different run or workspace launch"
            )
        report = self._report_payload(bundle, target)
        question = report.get("query", report.get("question"))
        if not isinstance(question, str) or " ".join(question.split()) != context.query:
            raise ResearchReturnMismatch(
                "The report question differs from this workspace launch"
            )
        body = report.get("report_markdown")
        if not isinstance(body, str) or not body.strip() or "\x00" in body:
            raise ValueError("This run does not have a saved Markdown report")
        if len(body.encode("utf-8")) > MAX_ANSWER_BYTES:
            raise ValueError(
                "This report exceeds Studio's import limit. Open it in Runs."
            )
        count = report.get("source_count")
        if type(count) is not int or count < 0:
            inventory = report.get("source_inventory")
            count = len(inventory) if isinstance(inventory, list) else None
        return DeepResearchReportPreview(target, context, body, count)

    @staticmethod
    def _report_payload(bundle: Mapping, target: ResearchWorkspaceRunRef) -> Mapping:
        if "artifacts" in bundle:
            if not isinstance(bundle["artifacts"], (list, tuple)):
                raise ResearchReturnMismatch("The bundle's artifact list is invalid")
            reports = [
                row
                for row in bundle["artifacts"]
                if isinstance(row, Mapping)
                and row.get("artifact_name") == "bundle.json"
            ]
            if not reports:
                raise ValueError("This run has no completed report bundle yet")
            if len(reports) != 1 or reports[0].get("run_id") != target.run_id:
                raise ResearchReturnMismatch(
                    "The report bundle belongs to a different Research run"
                )
            report = reports[0].get("content")
        else:
            report = bundle.get("bundle.json", bundle)
        if not isinstance(report, Mapping):
            raise ValueError("This run has no completed report bundle yet")
        return report
