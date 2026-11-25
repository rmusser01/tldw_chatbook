"""Canonical Server Studio artifact persistence under one qualified server."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from uuid import NAMESPACE_URL, uuid5

from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.Notes.server_notes_workspace_service import (
    ServerNotesWorkspaceService,
)
from tldw_chatbook.Workspaces.output_membership import output_membership_metadata
from tldw_chatbook.tldw_api.exceptions import APIResponseError
from tldw_chatbook.tldw_api.workspace_data_tables_client import WorkspaceDataTablesClient, canonical_table_uuid
from tldw_chatbook.tldw_api.workspace_artifact_exports import (
    export_accepted_workspace_markdown,
)

from tldw_chatbook.tldw_api.workspace_study_client import WorkspaceStudyClient

from .chat_models import validate_chat_id
from .chat_service import MAX_ANSWER_BYTES
from .contracts import (
    BoundedPageResult,
    QualifiedWorkspaceRef,
    ResearchCapability,
    WorkspaceDataSource,
)
from .server_study import output_creation_time, study_payload, study_result
from .studio_generation import StudioGeneratedOutput, validate_diagram_text
from .studio_mapping import SERVER_DIAGRAM_TYPES, STUDIO_OUTPUT_MAPPINGS, SERVER_TABLE_KINDS
from .studio_models import (
    ResearchStudioRequest,
    StudioOwnerKind,
    WorkspaceOutputRef,
    studio_output_kind,
    validate_output_page,
)
from .work_product_templates import TEXT_WORK_PRODUCT_TEMPLATES
from .server_data_tables import generate_native_table, table_sources
from .studio_provenance import (
    citation_references,
    generation_metadata,
    output_reference,
)


class ServerStudioAdapter:
    def __init__(self, client_for: Callable, *, policy_enforcer=None) -> None:
        self.client_for = client_for
        self.policy_enforcer = policy_enforcer
        self._mutations = asyncio.Lock()

    @staticmethod
    def capabilities(
        chat: ResearchCapability, *, writable: bool, study_supported: bool = False
    ) -> dict[str, ResearchCapability]:
        ready = chat.available and writable
        text = ResearchCapability(
            ready,
            "available" if ready else "server_studio_unavailable",
            "Server workspace artifacts are ready."
            if ready
            else "Server generation or artifact writes are unavailable.",
            "server_artifact",
            "" if ready else "Check server generation and workspace permissions.",
            chat.capability_revision,
        )
        change = ResearchCapability(
            writable,
            "available" if writable else "server_artifact_read_only",
            "Server artifact changes are available."
            if writable
            else "Server artifacts are read only.",
            "server_artifact",
            "" if writable else "Check workspace write permissions.",
            chat.capability_revision,
        )
        study_ready = ready and study_supported
        study = {
            action: ResearchCapability(
                study_ready,
                "available" if study_ready else "server_study_unavailable",
                "Server Study is ready."
                if study_ready
                else "Server generation or workspace writes are unavailable."
                if not ready
                else "This Server does not advertise retry-safe Study creation.",
                owner,
                ""
                if study_ready
                else "Check Server generation and workspace permissions."
                if not ready
                else "Update the selected server to enable these outputs.",
                chat.capability_revision,
            )
            for action, owner in (
                ("create_deck", "server_study_deck"),
                ("create_quiz", "server_quiz"),
            )
        }
        return {
            "save_workspace_artifact": text,
            "update_workspace_artifact": change,
            **{
                f"generate_{kind.value}": text
                for kind in (*SERVER_DIAGRAM_TYPES, *TEXT_WORK_PRODUCT_TEMPLATES)
            },
            **study,
        }

    @staticmethod
    def _study(ref, client) -> WorkspaceStudyClient:
        return WorkspaceStudyClient(client, ref.workspace_id, ref.principal_id)

    async def study_available(self, ref: QualifiedWorkspaceRef) -> bool:
        client, _ = self._connection(ref)
        result = await self._study(ref, client).page(limit=1, offset=0)
        self._check_connection(ref, client)
        return result is not None

    @staticmethod
    def _tables(ref, client) -> WorkspaceDataTablesClient:
        return WorkspaceDataTablesClient(client, ref.workspace_id, ref.principal_id)

    async def native_capabilities(self, ref, route, sources, revision) -> dict[str, ResearchCapability]:
        if route is None or route.data_source is not ref.data_source:
            return {}
        client, _ = self._connection(ref)
        reply = await self._tables(ref, client).capability(route.provider, route.model)
        self._check_connection(ref, client)
        message = "Choose an explicit Server provider and model with Data Tables access."
        reason = "server_data_tables_unavailable"
        if reply and reply.reason == "metered_workspace_table_route_unavailable":
            message = "Native Workspace Data Tables requires a Server route with usage quotas disabled."
        elif reply and reply.reason == "server_provider_credentials_required":
            message = "The selected provider needs a Server-managed credential for Data Tables."
        elif reply and reply.reason == "workspace_table_jobs_unavailable":
            message = "The native Data Tables queue is unavailable."
        available = bool(reply and reply.available)
        if available:
            try:
                table_sources(sources, maximum=reply.max_sources)
            except ValueError as exc:
                available, reason, message = False, "studio_source_owner_unavailable", str(exc)
        capability = ResearchCapability(
            available, "available" if available else reason,
            "Server Data Tables is ready." if available else message, StudioOwnerKind.SERVER_DATA_TABLE,
            "" if available else "Check the selected route and refresh the versions of the selected Server Media.",
            revision,
        )
        return {
            "get_data_table": capability,
            **{f"generate_{kind.value}": capability for kind in SERVER_TABLE_KINDS},
        }

    async def generate_native(self, request, *, cancel_event) -> StudioGeneratedOutput | None:
        if request.output_kind not in SERVER_TABLE_KINDS:
            return None
        ref = request.workspace
        client, _ = self._connection(ref)
        result = await generate_native_table(self._tables(ref, client), request, cancel_event)
        self._check_connection(ref, client)
        return result

    def _connection(self, ref: QualifiedWorkspaceRef):
        if ref.data_source is not WorkspaceDataSource.SERVER:
            raise ValueError("This Studio owner requires a Server workspace")
        client = self.client_for(ref)
        service = ServerNotesWorkspaceService(
            client, policy_enforcer=self.policy_enforcer
        )
        return client, NotesScopeService(
            local_notes_service=None,
            server_service=service,
            policy_enforcer=self.policy_enforcer,
        )

    def _check_connection(self, ref, client) -> None:
        if self.client_for(ref) is not client:
            raise ValueError("The selected server changed. Reopen the workspace")

    async def _records(self, ref, client, service):
        rows = await service.list_workspace_artifacts(workspace_id=ref.workspace_id)
        self._check_connection(ref, client)
        if any(row.get("workspace_id") != ref.workspace_id for row in rows):
            raise ValueError("Server returned outputs from a different workspace")
        return rows

    @staticmethod
    def _metadata(row):
        producer = row.get("producer_metadata")
        if not isinstance(producer, Mapping) or "research_generation" not in producer:
            return None
        return output_membership_metadata(
            {"research_generation": producer["research_generation"]}, "output"
        )

    def _reference(self, ref, row, metadata) -> WorkspaceOutputRef:
        details = metadata["research_generation"]
        kind = studio_output_kind(details["output_kind"])
        owner = STUDIO_OUTPUT_MAPPINGS[WorkspaceDataSource.SERVER, kind]
        if (
            owner.artifact_type is None
            or row.get("artifact_type") != owner.artifact_type
        ):
            raise ValueError(
                "The Server artifact does not match its Studio output kind"
            )
        if (
            row.get("status") != "complete"
            or type(row.get("version")) is not int
            or row["version"] < 1
        ):
            raise ValueError("The Server artifact has no completed revision")
        if owner.owner_kind is StudioOwnerKind.SERVER_DATA_TABLE:
            native = row["producer_metadata"].get("research_native_owner")
            if not isinstance(native, Mapping) or native.get("owner_kind") != owner.owner_kind.value:
                raise ValueError("The workspace link has no native Data Table")
            owner_id = canonical_table_uuid(native.get("owner_id"))
            version = native.get("owner_version")
            if type(version) is not int or version < 1:
                raise ValueError("The workspace link has no native Data Table revision")
            return output_reference(ref, owner_id, owner.owner_kind, version, details)
        return output_reference(
            ref,
            validate_chat_id(row.get("id"), "artifact ID"),
            owner.owner_kind,
            row["version"],
            details,
        )

    async def list_outputs(self, ref, *, limit=20, offset=0):
        page, _ = await self.list_output_records(ref, limit=limit, offset=offset)
        return page

    async def list_output_records(self, ref, *, limit=20, offset=0):
        if (
            type(limit) is not int
            or not 1 <= limit <= 100
            or type(offset) is not int
            or offset < 0
        ):
            raise ValueError("Choose a bounded output page")
        client, service = self._connection(ref)
        rows = await self._records(ref, client, service)
        entries = tuple(
            (self._reference(ref, row, metadata), row)
            for row in rows
            if (metadata := self._metadata(row))
        )
        study = self._study(ref, client)
        if entries:
            study_rows, study_total = await self._study_prefix(
                ref, client, study, offset + limit
            )
            combined = sorted(
                (*entries, *study_rows),
                key=lambda entry: output_creation_time(entry[1]),
                reverse=True,
            )
            selected = combined[offset : offset + limit]
        else:
            study_page = await study.page(limit=limit, offset=offset)
            self._check_connection(ref, client)
            study_total = study_page.total if study_page else 0
            selected = (
                [study_result(ref, row) for row in study_page.items]
                if study_page
                else []
            )
        total = len(entries) + study_total
        page = BoundedPageResult(
            tuple(output for output, _ in selected),
            limit,
            offset,
            total,
            offset + limit < total,
        )
        return validate_output_page(ref, page), dict(selected)

    async def _study_prefix(self, ref, client, study, count):
        rows = []
        total = None
        while total is None or len(rows) < min(count, total):
            page = await study.page(limit=min(100, count - len(rows)), offset=len(rows))
            self._check_connection(ref, client)
            if page is None and total is None:
                return [], 0
            if page is None or (total is not None and page.total != total):
                raise ValueError(
                    "Study output history changed. Reload it before continuing"
                )
            total = page.total
            rows.extend(study_result(ref, item) for item in page.items)
        return rows, total

    async def resolve_output(self, output: WorkspaceOutputRef):
        client, service = self._connection(output.workspace)
        if output.owner_kind in {
            StudioOwnerKind.SERVER_STUDY_DECK,
            StudioOwnerKind.SERVER_QUIZ,
        }:
            row = await self._study(output.workspace, client).get(
                output.output_kind.value, output.owner_id
            )
            self._check_connection(output.workspace, client)
            return study_result(output.workspace, row)[1] if row else None
        if output.owner_kind is StudioOwnerKind.SERVER_DATA_TABLE:
            links = await self._records(output.workspace, client, service)
            if not any(
                row.get("artifact_type") == "data_table" and (metadata := self._metadata(row))
                and self._reference(output.workspace, row, metadata) == output for row in links
            ):
                return None
            detail = await self._tables(output.workspace, client).detail(output.owner_id, rows_limit=12)
            self._check_connection(output.workspace, client)
            if not detail or detail.table.version != output.owner_version or detail.table.status != "ready":
                return None
            return {"title": detail.table.name, **detail.model_dump(mode="json")}
        for row in await self._records(output.workspace, client, service):
            if row.get("id") == output.owner_id and (metadata := self._metadata(row)):
                selected = self._reference(output.workspace, row, metadata)
                if selected.output_kind is output.output_kind:
                    return row
        return None

    @staticmethod
    def _artifact_id(request):
        ref = request.workspace
        identity = json.dumps(
            (
                ref.server_profile_id,
                ref.principal_id,
                ref.workspace_id,
                request.operation_id,
            )
        )
        return str(uuid5(NAMESPACE_URL, "tldw:research-studio:" + identity))

    async def get_imported_report(
        self, request: ResearchStudioRequest
    ) -> WorkspaceOutputRef | None:
        if request.research_run_ref is None:
            raise ValueError("Choose a verified Research run to import")
        ref = request.workspace
        client, service = self._connection(ref)
        owner_id = self._artifact_id(request)
        for row in await self._records(ref, client, service):
            if row.get("id") == owner_id:
                metadata = self._metadata(row)
                if (
                    not metadata
                    or metadata["research_generation"].get("research_run_ref")
                    != request.research_run_ref.provenance()
                ):
                    raise ValueError(
                        "The import already belongs to a different Research run"
                    )
                return self._reference(ref, row, metadata)
        return None

    async def edit_output(self, output: WorkspaceOutputRef, body: str) -> None:
        if not body.strip() or len(body.encode("utf-8")) > MAX_ANSWER_BYTES:
            raise ValueError("Enter a non-empty output within the size limit")
        if output.output_kind in SERVER_DIAGRAM_TYPES:
            body = validate_diagram_text(output.output_kind, body)
        await self._update_selected(output, content=body, review_state="draft")

    async def accept_output(self, output: WorkspaceOutputRef) -> None:
        await self._update_selected(output, review_state="accepted")

    async def _update_selected(self, output: WorkspaceOutputRef, **fields) -> None:
        ref = output.workspace
        client, service = self._connection(ref)
        async with self._mutations:
            row = await self.resolve_output(output)
            self._check_connection(ref, client)
            if (
                not row
                or not output.owner_version
                or row.get("version") != output.owner_version
            ):
                raise ValueError("The output changed. Reload it before updating")
            saved = await service.save_workspace_artifact(
                workspace_id=ref.workspace_id,
                artifact_id=output.owner_id,
                version=output.owner_version,
                **fields,
            )
            self._check_connection(ref, client)
            if (
                saved.get("workspace_id") != ref.workspace_id
                or saved.get("id") != output.owner_id
                or saved.get("version", 0) <= output.owner_version
                or any(saved.get(key) != value for key, value in fields.items())
                or self._metadata(saved) != self._metadata(row)
            ):
                raise ValueError("The Server did not confirm the updated revision")

    async def export_output(self, output: WorkspaceOutputRef, format: str) -> bytes:
        if output.owner_kind is StudioOwnerKind.SERVER_DATA_TABLE:
            ref = output.workspace
            client, _ = self._connection(ref)
            if not await self.resolve_output(output):
                raise ValueError("Open the current native Data Table revision before exporting")
            tables = self._tables(ref, client)
            data = await tables.export(output.owner_id, format)
            detail = await tables.detail(output.owner_id)
            self._check_connection(ref, client)
            if not detail or detail.table.version != output.owner_version:
                raise ValueError("The native Data Table changed while exporting")
            return data
        if format != "markdown":
            raise ValueError("Choose Markdown for this workspace artifact")
        ref = output.workspace
        client, _ = self._connection(ref)
        row = await self.resolve_output(output)
        self._check_connection(ref, client)
        if (
            not row
            or row.get("version") != output.owner_version
            or row.get("review_state") != "accepted"
        ):
            raise ValueError("Review and accept the current version before exporting")
        version_id = validate_chat_id(
            row.get("artifact_version_id"), "artifact version ID"
        )
        result = await export_accepted_workspace_markdown(
            client, ref.workspace_id, output.owner_id, version_id
        )
        self._check_connection(ref, client)
        return result

    async def save(
        self,
        request: ResearchStudioRequest,
        generated: StudioGeneratedOutput,
        *,
        replace_output: WorkspaceOutputRef | None = None,
        supersedes: WorkspaceOutputRef | None = None,
    ) -> WorkspaceOutputRef:
        ref = request.workspace
        client, service = self._connection(ref)
        mapping = STUDIO_OUTPUT_MAPPINGS[ref.data_source, request.output_kind]
        if generated.status != "complete":
            return WorkspaceOutputRef(
                ref,
                request.output_kind,
                mapping.owner_kind,
                "",
                None,
                request.source_snapshot,
                request.generation.fingerprint,
                generated.status,
            )
        if any(
            citation.source not in request.retrieval_sources
            for citation in generated.citations
        ):
            raise ValueError(
                "An output citation belongs to a different source snapshot"
            )
        target = replace_output or supersedes
        if (replace_output and supersedes) or (
            target
            and (
                target.workspace != ref or target.output_kind is not request.output_kind
            )
        ):
            raise ValueError(
                "The selected output belongs to a different workspace or output kind"
            )
        if mapping.owner_kind is StudioOwnerKind.SERVER_DATA_TABLE:
            if replace_output:
                raise ValueError("Regenerate the native Data Table as a new version")
            return await self._save_native_table(ref, client, service, request, generated, supersedes)
        if mapping.owner_kind is not StudioOwnerKind.SERVER_ARTIFACT:
            if replace_output:
                raise ValueError(
                    "Create a new version or edit the item in Server Study"
                )
            payload = study_payload(request, generated, supersedes)
            row = await self._study(ref, client).save(payload)
            self._check_connection(ref, client)
            expected = generation_metadata(request, supersedes)["research_generation"]
            return study_result(ref, row, expected=expected)[0]
        if (
            not generated.text.strip()
            or len(generated.text.encode("utf-8")) > MAX_ANSWER_BYTES
        ):
            raise ValueError(
                "The generated text is empty or exceeds the output size limit"
            )
        metadata = generation_metadata(request, supersedes)
        content = generated.text
        if request.output_kind in SERVER_DIAGRAM_TYPES:
            content = validate_diagram_text(request.output_kind, content)
        citations = citation_references(
            generated, {citation.marker for citation in generated.citations}
        )
        producer = {**metadata, "research_citations": citations}
        artifact_id = (
            replace_output.owner_id if replace_output else self._artifact_id(request)
        )
        async with self._mutations:
            records = await self._records(ref, client, service)
            row = next(
                (
                    candidate
                    for candidate in records
                    if candidate.get("id") == artifact_id
                ),
                None,
            )
            if row is not None and self._metadata(row) == metadata:
                return self._reference(ref, row, metadata)
            if replace_output:
                if (
                    row is None
                    or self._metadata(row) is None
                    or row.get("version") != replace_output.owner_version
                ):
                    raise ValueError(
                        "The output changed. Reload it before replacing it"
                    )
                if (
                    self._reference(ref, row, self._metadata(row)).output_kind
                    is not request.output_kind
                ):
                    raise ValueError(
                        "The output kind changed. Reload it before replacing it"
                    )
            elif row is not None:
                raise ValueError(
                    "The Server did not confirm this operation's provenance"
                )
            if supersedes and not any(
                candidate.get("id") == supersedes.owner_id and self._metadata(candidate)
                for candidate in records
            ):
                raise ValueError("The previous Server output is unavailable")

            try:
                row = await service.save_workspace_artifact(
                    workspace_id=ref.workspace_id,
                    artifact_id=artifact_id,
                    artifact_type=mapping.artifact_type,
                    title=request.generation.title
                    or request.output_kind.value.replace("_", " ").title(),
                    status="complete",
                    content=content,
                    producer_metadata=producer,
                    version=replace_output.owner_version if replace_output else None,
                    **({"review_state": "draft"} if replace_output else {}),
                )
            except APIResponseError as exc:
                if exc.status_code == 409:
                    records = await self._records(ref, client, service)
                    replay = next(
                        (
                            candidate
                            for candidate in records
                            if candidate.get("id") == artifact_id
                        ),
                        None,
                    )
                    if replay is not None and self._metadata(replay) == metadata:
                        return self._reference(ref, replay, metadata)
                    raise ValueError(
                        "The output changed. Reload it before retrying"
                    ) from exc
                raise
            self._check_connection(ref, client)
            if (
                row.get("id") != artifact_id
                or row.get("workspace_id") != ref.workspace_id
                or self._metadata(row) != metadata
                or row["producer_metadata"].get("research_citations") != citations
                or row.get("content") != content
            ):
                raise ValueError(
                    "The Server did not confirm the saved artifact and its provenance"
                )
            return self._reference(ref, row, metadata)

    async def _save_native_table(self, ref, client, service, request, generated, supersedes):
        if generated.native_owner is None:
            raise ValueError("The native Data Table generation has no completed owner")
        table_id, version = generated.native_owner
        table_id = canonical_table_uuid(table_id)
        artifact_id = self._artifact_id(request)
        metadata = generation_metadata(request, supersedes)
        producer = {
            **metadata,
            "research_native_owner": {
                "owner_kind": StudioOwnerKind.SERVER_DATA_TABLE.value,
                "owner_id": table_id,
                "owner_version": version,
            },
        }

        def validated(row):
            if (
                row.get("id") != artifact_id or row.get("workspace_id") != ref.workspace_id
                or row.get("producer_metadata") != producer or row.get("content")
            ):
                raise ValueError("The Server did not confirm the native Data Table workspace link")
            return self._reference(ref, row, metadata)

        async with self._mutations:
            rows = await self._records(ref, client, service)
            for row in rows:
                if row.get("id") == artifact_id:
                    return validated(row)
            if supersedes and not any(
                row.get("artifact_type") == "data_table" and (saved := self._metadata(row))
                and self._reference(ref, row, saved) == supersedes for row in rows
            ):
                raise ValueError("The previous Workspace Data Table reference changed")
            detail = await self._tables(ref, client).detail(table_id)
            self._check_connection(ref, client)
            if not detail or detail.table.version != version or detail.table.status != "ready":
                raise ValueError("The native Data Table changed before linking")
            try:
                row = await service.save_workspace_artifact(
                    workspace_id=ref.workspace_id, artifact_id=artifact_id, artifact_type="data_table",
                    title=detail.table.name, status="complete", producer_metadata=producer,
                )
            except APIResponseError as exc:
                if exc.status_code != 409:
                    raise
                rows = await self._records(ref, client, service)
                row = next((candidate for candidate in rows if candidate.get("id") == artifact_id), None)
                if row is None:
                    raise
            self._check_connection(ref, client)
            return validated(row)
