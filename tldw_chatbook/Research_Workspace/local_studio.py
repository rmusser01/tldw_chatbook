"""Persist Local Studio results in Chatbooks, Study decks, and quizzes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import asdict, replace
from hashlib import sha256

from tldw_chatbook.Backup_Recovery.participants import run_finite_local_worker
from tldw_chatbook.Chatbooks.local_chatbook_service import ChatbookRevisionConflict
from tldw_chatbook.DB.ChaChaNotes_DB import ConflictError
from tldw_chatbook.Workspaces.output_membership import output_membership_metadata
from tldw_chatbook.Workspaces.registry_service import WorkspaceMembershipConflict

from .chat_models import ResearchSourceVersionRef
from .chatbook_artifact_payload import build_research_chatbook_payload
from .chat_service import MAX_ANSWER_BYTES
from .contracts import BoundedPageResult, QualifiedWorkspaceRef, WorkspaceDataSource
from .contracts import ResearchCapability
from .studio_generation import StudioGeneratedOutput
from .studio_mapping import PRIMARY_OUTPUT_MAPPINGS, StudioOwnerMapping
from .studio_models import (
    ResearchStudioRequest,
    StudioOwnerKind,
    WorkspaceOutputRef,
    validate_output_page,
)


class StudioOutputConflict(ValueError):
    """A saved result changed after it was selected."""


class StudioOutputPersistenceError(RuntimeError):
    """A failed owner write or an output that requires recovery."""

    def __init__(self, owner_id: str = "") -> None:
        self.owner_id = owner_id
        super().__init__(
            "The saved output needs recovery."
            if owner_id
            else "The output could not be saved."
        )


class LocalStudioAdapter:
    def __init__(self, registry, chatbooks, study, quizzes) -> None:
        self.registry = registry
        self.chatbooks = chatbooks
        self.study = study
        self.quizzes = quizzes
        self._mutations = asyncio.Lock()

    @staticmethod
    def _owner(ref: QualifiedWorkspaceRef, kind) -> StudioOwnerMapping:
        if ref.data_source is not WorkspaceDataSource.LOCAL:
            raise ValueError("This Studio owner requires a Local workspace")
        return PRIMARY_OUTPUT_MAPPINGS[ref.data_source, kind]

    def capabilities(self) -> dict[str, ResearchCapability]:
        backends = {
            StudioOwnerKind.LOCAL_CHATBOOK: self.chatbooks,
            StudioOwnerKind.LOCAL_STUDY_DECK: self.study,
            StudioOwnerKind.LOCAL_QUIZ: self.quizzes,
        }
        result = {}
        for (authority, _kind), mapping in PRIMARY_OUTPUT_MAPPINGS.items():
            if authority is not WorkspaceDataSource.LOCAL:
                continue
            available = backends[mapping.owner_kind] is not None
            for action in (
                mapping.create_capability,
                mapping.replace_capability,
                mapping.delete_capability,
            ):
                if action is not None:
                    result[action] = ResearchCapability(
                        available,
                        "available" if available else "studio_owner_unavailable",
                        f"{mapping.owner_label} is ready."
                        if available
                        else f"{mapping.owner_label} is unavailable.",
                        mapping.owner_kind,
                        "" if available else "Open the owner and verify its storage.",
                    )
        return result

    @staticmethod
    async def _off_loop(function, *args, **kwargs):
        return await asyncio.to_thread(
            run_finite_local_worker, function, *args, **kwargs
        )

    async def _study_call(self, service, method, *args, **kwargs):
        function = getattr(service, method)
        if service.db.is_memory_db:
            return function(*args, **kwargs)
        return await self._off_loop(function, *args, **kwargs)

    async def _read_owner(self, owner: StudioOwnerMapping, owner_id: str):
        if owner.owner_kind is StudioOwnerKind.LOCAL_CHATBOOK:
            if self.chatbooks is None:
                return None
            try:
                return await self.chatbooks.get_chatbook(owner_id)
            except KeyError:
                return None
        if owner.owner_kind is StudioOwnerKind.LOCAL_STUDY_DECK:
            return (
                await self._study_call(self.study, "get_deck", owner_id)
                if self.study
                else None
            )
        return (
            await self._study_call(self.quizzes, "get_quiz", owner_id)
            if self.quizzes
            else None
        )

    async def _related_link(self, output: WorkspaceOutputRef):
        owner = self._owner(output.workspace, output.output_kind)
        memberships = await self._off_loop(
            self.registry.get_item_memberships,
            owner.membership_item_type,
            output.owner_id,
        )
        return next(
            (
                row
                for row in memberships
                if row.workspace_id == output.workspace.workspace_id
                and row.role == "output"
                and row.metadata.get("research_generation", {}).get("output_kind")
                == output.output_kind
            ),
            None,
        )

    async def resolve_output(self, output: WorkspaceOutputRef) -> Mapping | None:
        if not await self._related_link(output):
            return None
        return await self._read_owner(
            self._owner(output.workspace, output.output_kind), output.owner_id
        )

    async def list_outputs(
        self, ref: QualifiedWorkspaceRef, *, limit=20, offset=0
    ) -> BoundedPageResult[WorkspaceOutputRef]:
        self._owner(ref, "summary")
        rows, total = await self._off_loop(
            self.registry.list_workspace_output_memberships,
            ref.workspace_id,
            limit=limit,
            offset=offset,
        )
        outputs = []
        for row in rows:
            details = row.metadata["research_generation"]
            owner = self._owner(ref, details["output_kind"])
            if row.item_type != owner.membership_item_type:
                raise ValueError(
                    "The Studio output link does not match its canonical owner"
                )
            record = await self._read_owner(owner, row.item_id)
            outputs.append(self._as_ref(ref, row.item_id, details, owner, record))
        page = BoundedPageResult(
            tuple(outputs), limit, offset, total, offset + len(rows) < total
        )
        return validate_output_page(ref, page)

    @staticmethod
    def _as_ref(ref, owner_id, metadata, owner, record=None) -> WorkspaceOutputRef:
        return WorkspaceOutputRef(
            ref,
            metadata["output_kind"],
            owner.owner_kind,
            owner_id,
            record.get(owner.version_field)
            if record
            else metadata.get("owner_version"),
            tuple(
                ResearchSourceVersionRef(ref=ref, **source)
                for source in metadata["source_snapshot"]
            ),
            metadata["settings_fingerprint"],
            metadata["status"],
        )

    @staticmethod
    def _provenance(request: ResearchStudioRequest, status, supersedes=None):
        sources = []
        for source in request.source_snapshot:
            source_fields = asdict(source)
            del source_fields["ref"]
            sources.append(source_fields)
        route = json.dumps(asdict(request.processing_route), sort_keys=True)
        return {
            "operation_id": request.operation_id,
            "output_kind": request.output_kind.value,
            "source_snapshot": sources,
            "settings_fingerprint": request.generation.fingerprint,
            "processing_route_fingerprint": sha256(route.encode()).hexdigest(),
            "provider": request.processing_route.provider,
            "model": request.processing_route.model,
            "status": status,
            "supersedes_owner_id": supersedes.owner_id if supersedes else None,
        }

    async def save(
        self,
        request: ResearchStudioRequest,
        generated: StudioGeneratedOutput,
        *,
        replace_output: WorkspaceOutputRef | None = None,
        supersedes: WorkspaceOutputRef | None = None,
    ) -> WorkspaceOutputRef:
        """Make one canonical owner write and link its bounded generation provenance."""
        owner = self._owner(request.workspace, request.output_kind)
        if generated.status != "complete":
            return WorkspaceOutputRef(
                request.workspace,
                request.output_kind,
                owner.owner_kind,
                "",
                None,
                request.source_snapshot,
                request.generation.fingerprint,
                generated.status,
            )
        if replace_output is not None and supersedes is not None:
            raise ValueError("Choose either replacement or a new output version")
        target = replace_output or supersedes
        if target and (
            target.workspace != request.workspace
            or target.output_kind is not request.output_kind
        ):
            raise ValueError(
                "The selected output belongs to a different workspace or output kind"
            )
        if any(
            citation.source not in request.retrieval_sources
            for citation in generated.citations
        ):
            raise ValueError(
                "An output citation belongs to a different source snapshot"
            )
        task = asyncio.create_task(
            self._save_locked(request, generated, owner, replace_output, supersedes)
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    async def _save_locked(
        self, request, generated, owner, replace_output, supersedes
    ) -> WorkspaceOutputRef:
        async with self._mutations:
            ref = request.workspace
            workspace = await self._off_loop(
                self.registry.get_workspace, ref.workspace_id
            )
            if workspace is None or workspace.archived:
                raise ValueError("Choose an active workspace before saving an output")
            details = self._provenance(request, generated.status, supersedes)
            metadata = output_membership_metadata(
                {"research_generation": details}, "output"
            )
            old, count = await self._off_loop(
                self.registry.list_workspace_output_memberships,
                ref.workspace_id,
                limit=1,
                operation_id=request.operation_id,
            )
            if count:
                previous = old[0]
                saved_details = previous.metadata["research_generation"]
                original_details = metadata["research_generation"]
                if count != 1 or any(
                    saved_details.get(key) != value
                    for key, value in original_details.items()
                ):
                    raise StudioOutputConflict(
                        "This generation operation already belongs to a different output"
                    )
                record = await self._read_owner(owner, previous.item_id)
                if record is None:
                    raise StudioOutputConflict(
                        "The canonical output is missing from its owner"
                    )
                return self._as_ref(ref, previous.item_id, saved_details, owner, record)

            if supersedes is not None and await self.resolve_output(supersedes) is None:
                raise StudioOutputConflict("The previous output is unavailable")
            if replace_output is not None:
                return await self._replace(
                    request, generated, owner, replace_output, metadata
                )

            try:
                record = await self._create_owner(request, generated, owner)
            except (TypeError, ValueError):
                raise
            except Exception as exc:
                raise StudioOutputPersistenceError() from exc
            owner_id = str(record[owner.id_field])
            try:
                link = await self._off_loop(
                    self.registry.link_membership,
                    ref.workspace_id,
                    item_type=owner.membership_item_type,
                    item_id=owner_id,
                    role="output",
                    title=str(record.get("name") or request.output_kind.value),
                )
                return await self._link_metadata(ref, owner, record, link, metadata)
            except Exception as exc:
                try:
                    if not await self._delete_owner(
                        owner, owner_id, record.get(owner.version_field)
                    ):
                        raise StudioOutputPersistenceError(owner_id)
                    await self._unlink(ref, owner, owner_id)
                except Exception as cleanup:
                    raise StudioOutputPersistenceError(owner_id) from cleanup
                raise StudioOutputPersistenceError() from exc

    async def _link_metadata(
        self, ref, owner, record, link, metadata
    ) -> WorkspaceOutputRef:
        metadata["research_generation"]["owner_version"] = record.get(
            owner.version_field
        )
        saved = await self._off_loop(
            self.registry.upsert_membership_metadata,
            replace(link, metadata=metadata),
            expected_updated_at=link.updated_at,
        )
        return self._as_ref(
            ref, link.item_id, saved.metadata["research_generation"], owner, record
        )

    async def _replace(
        self, request, generated, owner, output, metadata
    ) -> WorkspaceOutputRef:
        link = await self._related_link(output)
        if not link or output.owner_version is None:
            raise StudioOutputConflict("Reload the output before replacing it")
        if (
            owner.owner_kind is not StudioOwnerKind.LOCAL_CHATBOOK
            or self.chatbooks is None
        ):
            raise ValueError("This Local owner supports saving a new version")
        payload = self._text_payload(request, generated)
        try:
            record = await self.chatbooks.update_chatbook(
                output.owner_id, expected_revision=output.owner_version, **payload
            )
        except (KeyError, ChatbookRevisionConflict) as exc:
            raise StudioOutputConflict(
                "The output changed. Reload it before replacing it"
            ) from exc
        try:
            return await self._link_metadata(
                request.workspace, owner, record, link, metadata
            )
        except Exception as exc:
            raise StudioOutputPersistenceError(output.owner_id) from exc

    @staticmethod
    def _text_payload(request, generated):
        if (
            not generated.text.strip()
            or len(generated.text.encode("utf-8")) > MAX_ANSWER_BYTES
        ):
            raise ValueError(
                "The generated text is empty or exceeds the output size limit"
            )
        return build_research_chatbook_payload(request, generated)

    async def _create_owner(self, request, generated, owner):
        if owner.owner_kind is StudioOwnerKind.LOCAL_CHATBOOK and self.chatbooks:
            return await self.chatbooks.create_chatbook(
                **self._text_payload(request, generated)
            )
        if owner.owner_kind is StudioOwnerKind.LOCAL_STUDY_DECK and self.study:
            return await self._save_study_items(
                self.study, self._write_deck, request, generated
            )
        if owner.owner_kind is StudioOwnerKind.LOCAL_QUIZ and self.quizzes:
            return await self._save_study_items(
                self.quizzes, self._write_quiz, request, generated
            )
        raise ValueError("The Local owner for this output is unavailable")

    async def _save_study_items(self, service, write, request, generated):
        if service.db.is_memory_db:
            return write(request, generated)
        return await self._off_loop(write, request, generated)

    def _write_deck(self, request, generated):
        if not generated.cards or len(generated.cards) > request.generation.item_count:
            raise ValueError("A Study deck needs a bounded set of valid flashcards")
        with self.study.db.transaction(immediate=True):
            deck = self.study.create_deck(
                name=request.generation.title or "Research flashcards"
            )
            items = [
                {
                    "deck_id": deck["id"],
                    "front": card.front,
                    "back": card.back,
                    "extra": json.dumps(
                        {"source_citations": self._citations(generated, card.citations)}
                    ),
                }
                for card in generated.cards
            ]
            if self.study.create_flashcards_bulk(items)["count"] != len(items):
                raise StudioOutputPersistenceError()
            return self.study.get_deck(deck["id"])

    def _write_quiz(self, request, generated):
        if (
            not generated.questions
            or len(generated.questions) > request.generation.item_count
        ):
            raise ValueError("A quiz needs a bounded set of valid questions")
        with self.quizzes.db.transaction(immediate=True):
            quiz = self.quizzes.create_quiz(
                name=request.generation.title or "Research quiz"
            )
            for order, item in enumerate(generated.questions):
                self.quizzes.create_question(
                    quiz["id"],
                    question_type="multiple_choice",
                    question_text=item.question,
                    options=list(item.options),
                    correct_answer=item.correct_index,
                    explanation=item.explanation,
                    order_index=order,
                    source_citations=self._citations(generated, item.citations),
                )
            return self.quizzes.get_quiz(quiz["id"])

    @staticmethod
    def _citations(generated, markers):
        return [
            {
                "marker": citation.marker,
                "source_id": citation.source.source_id,
                "catalog_item_id": citation.source.catalog_item_id,
            }
            for citation in generated.citations
            if citation.marker in markers
        ]

    async def _delete_owner(self, owner, owner_id, version):
        if type(version) is not int:
            raise StudioOutputConflict("Reload the output before deleting it")
        if owner.owner_kind is StudioOwnerKind.LOCAL_CHATBOOK:
            return await self.chatbooks.delete_chatbook(
                owner_id, expected_revision=version
            )
        if owner.owner_kind is StudioOwnerKind.LOCAL_STUDY_DECK:
            return await self._study_call(
                self.study, "delete_deck", owner_id, expected_version=version
            )
        return await self._study_call(
            self.quizzes, "delete_quiz", owner_id, expected_version=version
        )

    async def _unlink(self, ref, owner, owner_id):
        await self._off_loop(
            self.registry.unlink_membership,
            ref.workspace_id,
            item_type=owner.membership_item_type,
            item_id=owner_id,
            role="output",
        )

    async def delete(self, output: WorkspaceOutputRef) -> None:
        """Delete the exact canonical version before unlinking it from the workspace."""
        owner = self._owner(output.workspace, output.output_kind)
        async with self._mutations:
            if not await self._related_link(output):
                raise StudioOutputConflict(
                    "The output is unavailable in this workspace"
                )
            try:
                deleted = await self._delete_owner(
                    owner, output.owner_id, output.owner_version
                )
            except (
                KeyError,
                ChatbookRevisionConflict,
                ConflictError,
                WorkspaceMembershipConflict,
            ) as exc:
                raise StudioOutputConflict(
                    "The output changed. Reload it before deleting it"
                ) from exc
            if not deleted:
                raise StudioOutputConflict("The canonical output was not deleted")
            await self._unlink(output.workspace, owner, output.owner_id)

    async def export_output(self, output: WorkspaceOutputRef, format: str) -> bytes:
        owner = self._owner(output.workspace, output.output_kind)
        if (
            format != "markdown"
            or owner.owner_kind is not StudioOwnerKind.LOCAL_CHATBOOK
        ):
            raise ValueError("This Local owner does not support that output format")
        record = await self.resolve_output(output)
        if not record or record.get(owner.version_field) != output.owner_version:
            raise StudioOutputConflict("Reload the output before exporting it")
        return str(record["metadata"]["content"]).encode("utf-8")
