"""Persist Local Studio results in Chatbooks, Study decks, and quizzes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import replace
from hashlib import sha256

from tldw_chatbook.Backup_Recovery.participants import run_finite_local_worker
from tldw_chatbook.Chatbooks.local_chatbook_service import ChatbookRevisionConflict
from tldw_chatbook.DB.ChaChaNotes_DB import ConflictError
from tldw_chatbook.Workspaces.registry_service import WorkspaceMembershipConflict

from .chatbook_artifact_payload import build_research_chatbook_payload
from .chat_service import MAX_ANSWER_BYTES
from .contracts import BoundedPageResult, QualifiedWorkspaceRef, WorkspaceDataSource
from .contracts import ResearchCapability
from .studio_generation import StudioGeneratedOutput
from .studio_mapping import STUDIO_OUTPUT_MAPPINGS, StudioOwnerMapping
from .studio_models import (
    COPYABLE_STUDIO_TEXT_KINDS,
    ResearchOutputKind,
    ResearchStudioRequest,
    StudioOwnerKind,
    WorkspaceOutputRef,
    validate_output_page,
)
from .studio_provenance import (
    citation_references,
    copied_output_reference,
    generation_metadata,
    output_reference,
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
    def __init__(self, registry, chatbooks, study, quizzes, *, audio=None) -> None:
        from .local_studio_audio import LocalStudioAudio

        self.registry = registry
        self.chatbooks = chatbooks
        self.study = study
        self.quizzes = quizzes
        self.audio = LocalStudioAudio(audio)
        self._mutations = asyncio.Lock()

    @staticmethod
    def _owner(ref: QualifiedWorkspaceRef, kind) -> StudioOwnerMapping:
        if ref.data_source is not WorkspaceDataSource.LOCAL:
            raise ValueError("This Studio owner requires a Local workspace")
        return STUDIO_OUTPUT_MAPPINGS[ref.data_source, kind]

    def capabilities(self) -> dict[str, ResearchCapability]:
        backends = {
            StudioOwnerKind.LOCAL_CHATBOOK: self.chatbooks,
            StudioOwnerKind.LOCAL_STUDY_DECK: self.study,
            StudioOwnerKind.LOCAL_QUIZ: self.quizzes,
            StudioOwnerKind.LOCAL_AUDIO: self.audio if self.audio.persistent_history else None,
        }
        result = {}
        for (authority, _kind), mapping in STUDIO_OUTPUT_MAPPINGS.items():
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

    async def speech_choices(self):
        return await self.audio.choices()

    async def prepare_output(self, request, generated, *, cancel_event):
        if request.output_kind is not ResearchOutputKind.AUDIO_SUMMARY:
            return None
        return await self.audio.prepare(request, generated, cancel_event)

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
        if owner.owner_kind is StudioOwnerKind.LOCAL_AUDIO:
            return await self.audio.detail(owner_id)
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
        link = next(
            (
                row
                for row in memberships
                if row.workspace_id == output.workspace.workspace_id
                and row.role == "output"
            ),
            None,
        )
        if link is None:
            return None
        if output.copy_provenance is None:
            return (
                link
                if link.metadata.get("research_generation", {}).get("output_kind")
                == output.output_kind
                else None
            )
        proofs = await self._off_loop(
            self.registry.list_workspace_artifact_copy_proofs,
            output.workspace.workspace_id,
            owner_id=output.owner_id,
        )
        record = await self._read_owner(owner, output.owner_id)
        copied = self._copied_ref(output.workspace, record, proofs)
        if copied and (copied.output_kind, copied.copy_provenance) == (
            output.output_kind,
            output.copy_provenance,
        ):
            return link
        return None

    @staticmethod
    def _copied_ref(ref, record, proofs):
        if not record or not isinstance(metadata := record.get("metadata"), Mapping):
            return None
        kind = metadata.get("artifact_kind")
        key = metadata.get("research_copy_replacement") or record.get("research_copy_key")
        if (
            metadata.get("artifact_source") != "research"
            or not isinstance(kind, str)
            or kind not in COPYABLE_STUDIO_TEXT_KINDS
            or not isinstance(key, str)
        ):
            return None
        indexed = proofs.get((record["id"], key))
        version = record.get("artifact_revision")
        if indexed and type(version) is int and version >= indexed[0]:
            return copied_output_reference(
                ref, record["id"], version, kind, indexed[1]
            )
        return None

    async def _list_copied_refs(self, ref):
        if self.chatbooks is None:
            return {}
        proofs = await self._off_loop(
            self.registry.list_workspace_artifact_copy_proofs, ref.workspace_id
        )
        if not proofs:
            return {}
        owner_ids = {owner_id for owner_id, _ in proofs}
        snapshot = await self._off_loop(
            self.chatbooks.artifact_read_snapshot, chatbook_ids=owner_ids
        )
        outputs = {}
        for owner_id in owner_ids:
            if owner_id.isascii() and owner_id.isdecimal():
                record = snapshot.get_record(int(owner_id))
                if copied := self._copied_ref(ref, record, proofs):
                    outputs[owner_id] = copied
        return outputs

    async def resolve_output(self, output: WorkspaceOutputRef) -> Mapping | None:
        link = await self._related_link(output)
        if not link:
            return None
        row = await self._read_owner(
            self._owner(output.workspace, output.output_kind), output.owner_id
        )
        if row and output.owner_kind is StudioOwnerKind.LOCAL_AUDIO:
            return {**row, "title": link.title}
        return row

    async def get_imported_report(
        self, request: ResearchStudioRequest
    ) -> WorkspaceOutputRef | None:
        if request.research_run_ref is None:
            raise ValueError("Choose a verified Research run to import")
        ref = request.workspace
        rows, count = await self._off_loop(
            self.registry.list_workspace_output_memberships,
            ref.workspace_id,
            limit=1,
            operation_id=request.operation_id,
        )
        if not count:
            return None
        link = rows[0]
        details = link.metadata["research_generation"]
        owner = self._owner(ref, request.output_kind)
        if (
            count != 1
            or details.get("research_run_ref") != request.research_run_ref.provenance()
        ):
            raise StudioOutputConflict(
                "The import already belongs to a different Research run"
            )
        record = await self._read_owner(owner, link.item_id)
        if record is None:
            raise StudioOutputConflict("The imported Report is missing from its owner")
        return self._as_ref(ref, link.item_id, details, owner, record)

    async def list_outputs(
        self, ref: QualifiedWorkspaceRef, *, limit=20, offset=0
    ) -> BoundedPageResult[WorkspaceOutputRef]:
        self._owner(ref, "summary")
        copies = await self._list_copied_refs(ref)
        rows, total = await self._off_loop(
            self.registry.list_workspace_output_memberships,
            ref.workspace_id,
            limit=limit,
            offset=offset,
            copied_owner_ids=tuple(copies),
        )
        outputs = []
        for row in rows:
            if not row.metadata:
                outputs.append(copies[row.item_id])
                continue
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
        version = (
            record.get(owner.version_field) if record else metadata.get("owner_version")
        )
        return output_reference(ref, owner_id, owner.owner_kind, version, metadata)

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
            metadata = generation_metadata(request, supersedes)
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
                    transfer_policy=(
                        "copy"
                        if owner.owner_kind is StudioOwnerKind.LOCAL_CHATBOOK
                        else "reference"
                    ),
                    title=str(
                        request.generation.title or "Audio Summary"
                        if owner.owner_kind is StudioOwnerKind.LOCAL_AUDIO
                        else record.get("name") or request.output_kind.value
                    ),
                )
                return await self._link_metadata(ref, owner, record, link, metadata)
            except Exception as exc:
                if owner.owner_kind is StudioOwnerKind.LOCAL_AUDIO:
                    raise StudioOutputPersistenceError(owner_id) from exc
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
        payload["metadata"]["research_operation_id"] = request.operation_id
        record = await self._read_owner(owner, output.owner_id)
        already_committed = (
            record is not None
            and record.get(owner.version_field) == output.owner_version + 1
            and record.get("metadata") == payload["metadata"]
            and record.get("name") == payload["name"]
        )
        if not already_committed:
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
        if owner.owner_kind is StudioOwnerKind.LOCAL_AUDIO:
            if generated.native_owner:
                owner_id, version = generated.native_owner
                saved = await self.audio.detail(owner_id)
                if saved and saved["version"] == version:
                    return saved
            raise ValueError("The saved Local speech history entry is unavailable")
        if owner.owner_kind is StudioOwnerKind.LOCAL_CHATBOOK and self.chatbooks:
            options = {}
            if request.research_run_ref:
                key = f"research-report:{request.workspace.workspace_id}:{request.operation_id}"
                options["copy_operation_key"] = sha256(key.encode("utf-8")).hexdigest()
            return await self.chatbooks.create_chatbook(
                **self._text_payload(request, generated),
                **options,
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
                        {
                            "source_citations": citation_references(
                                generated, card.citations
                            )
                        }
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
                    source_citations=citation_references(generated, item.citations),
                )
            return self.quizzes.get_quiz(quiz["id"])

    async def _delete_owner(self, owner, owner_id, version):
        if type(version) is not int:
            raise StudioOutputConflict("Reload the output before deleting it")
        if owner.owner_kind is StudioOwnerKind.LOCAL_CHATBOOK:
            return await self.chatbooks.delete_chatbook(
                owner_id, expected_revision=version
            )
        if owner.owner_kind is StudioOwnerKind.LOCAL_AUDIO:
            saved = await self.audio.detail(owner_id)
            if not saved or saved["version"] != version:
                return False
            deleted = await self.audio.scope.delete_tts_history_entry(mode="local", history_id=saved["id"])
            return deleted.get("id") == saved["id"] and deleted.get("deleted") is True
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

    async def unlink_missing(self, output: WorkspaceOutputRef) -> None:
        owner = self._owner(output.workspace, output.output_kind)
        async with self._mutations:
            if not await self._related_link(output) or await self._read_owner(
                owner, output.owner_id
            ):
                raise StudioOutputConflict(
                    "Reload the history before removing a missing link"
                )
            await self._unlink(output.workspace, owner, output.owner_id)

    async def edit_output(self, output: WorkspaceOutputRef, body: str) -> None:
        owner = self._owner(output.workspace, output.output_kind)
        if (
            owner.owner_kind is not StudioOwnerKind.LOCAL_CHATBOOK
            or self.chatbooks is None
        ):
            raise ValueError("Open this output in its canonical editor")
        if not body.strip() or len(body.encode("utf-8")) > MAX_ANSWER_BYTES:
            raise ValueError("Enter a non-empty output within the size limit")
        async with self._mutations:
            row = await self.resolve_output(output)
            if not row or output.owner_version is None:
                raise StudioOutputConflict("Reload the output before editing it")
            try:
                await self.chatbooks.update_chatbook(
                    output.owner_id,
                    expected_revision=output.owner_version,
                    metadata={**row["metadata"], "content": body},
                )
            except (KeyError, ChatbookRevisionConflict) as exc:
                raise StudioOutputConflict(
                    "The output changed. Reload it before editing"
                ) from exc

    async def export_output(self, output: WorkspaceOutputRef, format: str) -> bytes:
        owner = self._owner(output.workspace, output.output_kind)
        if owner.owner_kind is StudioOwnerKind.LOCAL_AUDIO:
            if await self._related_link(output):
                detail = await self.audio.detail(output.owner_id, include_content=True)
                if detail and detail["format"] == format and detail["version"] == output.owner_version:
                    return detail["content"]
            raise StudioOutputConflict("Reload the Local audio history entry before downloading it")
        if (
            format != "markdown"
            or owner.owner_kind is not StudioOwnerKind.LOCAL_CHATBOOK
        ):
            raise ValueError("This Local owner does not support that output format")
        record = await self.resolve_output(output)
        if not record or record.get(owner.version_field) != output.owner_version:
            raise StudioOutputConflict("Reload the output before exporting it")
        return str(record["metadata"]["content"]).encode("utf-8")
