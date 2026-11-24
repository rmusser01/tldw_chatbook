"""Server-owned Research Chat, including resumable canonical message append."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any

from tldw_chatbook.Chat.chat_conversation_scope_service import (
    ChatConversationScopeService,
)
from tldw_chatbook.Chat.server_chat_conversation_service import (
    ServerChatConversationService,
)
from tldw_chatbook.tldw_api.chat_completion_schemas import (
    ResearchChatCompletionRequest,
    ResearchChatRagCitation,
    ResearchChatRagContext,
    ResearchChatRagDocument,
    ResearchChatRagSearch,
    ResearchChatRagSettings,
    ResearchChatRagSource,
)

from .chat_models import (
    MAX_CHAT_APPEND_RECEIPTS,
    RESEARCH_CHAT_TOPIC,
    ResearchChatAppendReceipt,
    ResearchChatMode,
    ResearchChatSession,
    ResearchChatSessionRequest,
    ResearchChatTurnOutcome,
    ResearchChatTurnRequest,
    ResearchCitationRef,
    ResearchSourceVersionRef,
    validate_chat_draft,
    validate_chat_id,
)
from .chat_service import collect_answer
from .contracts import (
    BoundedPageResult,
    ProcessingRoute,
    QualifiedWorkspaceRef,
    RetrievalMode,
    WorkspaceDataSource,
)
from .layout_state import ResearchPanePreferences
from .overlay_store import OverlayConflictError, ResearchPresentationOverlayStore

_EXCERPT_CHARS = 12_000
_HISTORY_CHARS = 64_000
_STAGES = {"pending": 0, "user_committed": 1, "assistant_committed": 2, "complete": 3}
ReadSources = Callable[
    [QualifiedWorkspaceRef], Awaitable[tuple[ResearchSourceVersionRef, ...]]
]


@dataclass(frozen=True, slots=True)
class _PendingAnswer:
    request: ResearchChatTurnRequest
    outcome: ResearchChatTurnOutcome
    text: str = field(repr=False)
    parent_id: str | None
    context: ResearchChatRagContext = field(repr=False)


class ServerResearchChat:
    def __init__(
        self,
        client_for: Callable[[QualifiedWorkspaceRef], Any],
        overlay: ResearchPresentationOverlayStore,
        *,
        policy_enforcer=None,
    ) -> None:
        self.client_for = client_for
        self.overlay = overlay
        self.policy_enforcer = policy_enforcer
        self._pending: dict[tuple[QualifiedWorkspaceRef, str], _PendingAnswer] = {}

    def _connection(self, ref: QualifiedWorkspaceRef):
        if ref.data_source is not WorkspaceDataSource.SERVER:
            raise ValueError("Server Research Chat requires a Server workspace")
        client = self.client_for(ref)
        service = ServerChatConversationService(
            client, policy_enforcer=self.policy_enforcer
        )
        return client, service

    @staticmethod
    def _session(
        ref: QualifiedWorkspaceRef, row: Mapping[str, Any]
    ) -> ResearchChatSession:
        if (
            row.get("scope_type") != "workspace"
            or row.get("workspace_id") != ref.workspace_id
            or row.get("topic_label") != RESEARCH_CHAT_TOPIC
        ):
            raise ValueError(
                "Server conversation does not belong to this Research workspace"
            )
        return ResearchChatSession(
            ref,
            validate_chat_id(row.get("id"), "conversation ID"),
            str(row.get("title") or "Research Chat"),
        )

    async def list_sessions(
        self, ref: QualifiedWorkspaceRef, *, limit: int = 20, offset: int = 0
    ):
        _, service = self._connection(ref)
        page = await service.list_conversations(
            scope_type="workspace",
            workspace_id=ref.workspace_id,
            topic_label=RESEARCH_CHAT_TOPIC,
            limit=limit,
            offset=offset,
        )
        pagination = page["pagination"]
        return BoundedPageResult(
            tuple(self._session(ref, row) for row in page["items"]),
            limit,
            offset,
            pagination["total"],
            pagination["has_more"],
        )

    async def create_session(
        self, ref: QualifiedWorkspaceRef, request: ResearchChatSessionRequest
    ) -> ResearchChatSession:
        _, service = self._connection(ref)
        scope = ChatConversationScopeService(local_service=None, server_service=service)
        row = await scope.create_conversation(
            mode="server",
            scope_type="workspace",
            workspace_id=ref.workspace_id,
            workspace_assistant_selection="none",
            topic_label=RESEARCH_CHAT_TOPIC,
            title=request.title,
            idempotency_key=_key(request.operation_id, "session"),
        )
        return self._session(ref, row)

    async def _read_messages(self, service, ref, conversation_id, limit):
        row = await service.get_conversation(
            conversation_id, scope_type="workspace", workspace_id=ref.workspace_id
        )
        session = self._session(ref, row)
        if session.conversation_id != conversation_id:
            raise ValueError("Server returned a different conversation")
        count = row.get("message_count")
        offset = max(0, count - limit) if type(count) is int else 0
        rows = await service.get_messages_with_context(
            conversation_id,
            scope_type="workspace",
            workspace_id=ref.workspace_id,
            limit=limit,
            offset=offset,
            include_rag_context=True,
        )
        if not isinstance(rows, list) or any(
            row.get("conversation_id") != conversation_id for row in rows
        ):
            raise ValueError("Server returned messages from a different conversation")
        return tuple(rows)

    async def load_messages(
        self, ref: QualifiedWorkspaceRef, conversation_id: str, *, limit: int = 100
    ):
        _, service = self._connection(ref)
        return await self._read_messages(service, ref, conversation_id, limit)

    async def _check_selected_message(self, ref, conversation_id, selected):
        client, service = self._connection(ref)
        scope = {"scope_type": "workspace", "workspace_id": ref.workspace_id}
        conversation = await service.get_conversation(conversation_id, **scope)
        if self._session(ref, conversation).conversation_id != conversation_id:
            raise ValueError("Server returned a different conversation")
        message_id = validate_chat_id(selected.get("id"), "message ID")
        version = selected.get("version")
        if type(version) is not int or version < 1:
            raise ValueError(
                "The selected message has no server version. Reload the conversation"
            )
        row = await client.get_character_message(message_id, **scope)
        if (
            row.id != message_id
            or row.conversation_id != conversation_id
            or row.version != version
        ):
            raise ValueError(
                "The selected message changed. Reload the conversation and try again"
            )
        if self.client_for(ref) is not client:
            raise ValueError("The selected server changed. Reopen the workspace")
        return service, row, {**scope, "expected_version": version}

    async def edit(self, ref, conversation_id, selected, body):
        if not validate_chat_draft(body).strip():
            raise ValueError("Write a message before saving")
        service, row, scope = await self._check_selected_message(
            ref, conversation_id, selected
        )
        if row.sender != "user":
            raise ValueError("Only questions are editable on the selected server")
        await service.edit_message(row.id, body, **scope)

    async def delete(self, ref, conversation_id, selected):
        service, row, scope = await self._check_selected_message(
            ref, conversation_id, selected
        )
        await service.delete_message(row.id, **scope)

    def resolve_route(
        self, ref: QualifiedWorkspaceRef, provider: str, model: str
    ) -> ProcessingRoute:
        client, _ = self._connection(ref)
        return self._route(client, provider, model)

    @staticmethod
    def _route(client, provider, model):
        return ProcessingRoute(
            WorkspaceDataSource.SERVER,
            "server-chat",
            provider=provider,
            model=model or "auto",
            remote=True,
            endpoint_class="server",
            endpoint_fingerprint=sha256(client.base_url.encode()).hexdigest(),
        )

    async def answer(
        self,
        request: ResearchChatTurnRequest,
        on_chunk: Callable[[str], None],
        cancel: asyncio.Event,
        read_sources: ReadSources,
    ) -> ResearchChatTurnOutcome:
        ref = request.workspace
        client, service = self._connection(ref)
        route = request.processing_route
        if self._route(client, route.provider, route.model) != route:
            raise ValueError(
                "Server processing route changed. Review it before sending"
            )
        pending_key = (ref, request.turn_id)
        overlay = await asyncio.to_thread(self.overlay.load, ref)
        if request.retry_user_message_id and not (
            overlay
            and any(
                row.stage == "user_committed"
                and row.user_message_id == request.retry_user_message_id
                and row.conversation_id == request.conversation_id
                for row in overlay.chat_append_receipts
            )
        ):
            raise ValueError("No interrupted Server answer matches this saved question")
        receipt = (
            next(
                (
                    row
                    for row in overlay.chat_append_receipts
                    if row.turn_id == request.turn_id
                ),
                None,
            )
            if overlay
            else None
        )
        pending = self._pending.get(pending_key)
        if receipt is not None and pending is None:
            raise ValueError(
                "This answer was interrupted. Review the saved conversation to resume"
            )
        if pending is not None and pending.request != request:
            raise ValueError(
                "Retry must use the same conversation, draft, and source snapshot"
            )
        if pending is None:
            if len(self._pending) >= MAX_CHAT_APPEND_RECEIPTS:
                raise ValueError("Finish saving an interrupted answer first")
            pending = await self._generate(
                request, client, service, on_chunk, cancel, read_sources
            )
            if isinstance(pending, ResearchChatTurnOutcome):
                return pending
            self._pending[pending_key] = pending
        outcome = await self._append(pending, service, receipt)
        del self._pending[pending_key]
        return outcome

    async def _generate(self, request, client, service, on_chunk, cancel, read_sources):
        ref = request.workspace
        history = await self._read_messages(
            service,
            ref,
            request.conversation_id,
            max(1, request.retrieval.history_limit),
        )
        if request.retry_user_message_id:
            if (
                not history
                or history[-1]["id"] != request.retry_user_message_id
                or history[-1].get("content") != request.user_text
            ):
                raise ValueError(
                    "The interrupted question is no longer the conversation's latest message"
                )
            history = history[:-1]
        documents = ()
        sources = {
            source.catalog_item_id: source for source in request.retrieval_sources
        }
        mode = _server_mode(request.effective_retrieval)
        if sources:
            await _check_sources(ref, sources, read_sources)
            result = await client.search_research_chat(
                ResearchChatRagSearch(
                    query=request.user_text,
                    workspace_id=ref.workspace_id,
                    include_media_ids=tuple(int(media_id) for media_id in sources),
                    search_mode=mode,
                    top_k=request.retrieval.top_k,
                    min_score=request.retrieval.threshold or 0,
                    enable_reranking=request.retrieval.rerank,
                )
            )
            documents = _read_documents(result, sources, request.retrieval.top_k)
            await _check_sources(ref, sources, read_sources)
        retrieved = bool(documents)
        if cancel.is_set():
            return ResearchChatTurnOutcome.from_request(
                request, status="cancelled", retrieved=retrieved
            )
        if request.mode is ResearchChatMode.RAG and not retrieved:
            raise ValueError(
                "No matching evidence was found in the selected Server sources"
            )

        # Check the active identity immediately before emitting the prompt. The
        # request uses the client captured before retrieval and never retargets.
        if self.client_for(ref) is not client:
            raise ValueError("Server identity changed before sending")
        citations = (
            tuple(
                ResearchChatRagCitation(
                    id=f"server-{request.turn_id[:32]}-{i}",
                    media_id=document.id,
                    marker=f"[S{i}]",
                    title=document.title,
                )
                for i, document in enumerate(documents, 1)
            )
            if request.retrieval.citations
            else ()
        )
        messages = _history_rows(history, request.retrieval.history_limit)
        if documents:
            evidence = "\n\n".join(
                f"[S{i}] {item.title}\n{item.excerpt}"
                for i, item in enumerate(documents, 1)
            )
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": "Use this quoted evidence. Cite the supplied [S#] markers.\n\n"
                    + evidence,
                },
            )
        messages.append(
            {"role": "user", "content": _literal_question(request.user_text)}
        )
        route = request.processing_route
        completion = ResearchChatCompletionRequest(
            conversation_id=request.conversation_id,
            messages=messages,
            model=route.model,
            api_provider=route.provider or None,
            stream=True,
            save_to_db=False,
        )
        answer = await collect_answer(
            client.stream_chat_completion(completion, workspace_id=ref.workspace_id),
            on_chunk,
            cancel,
        )
        references = tuple(
            ResearchCitationRef(item.id, sources[item.media_id])
            for item in citations
            if item.marker in answer.text
        )
        outcome = ResearchChatTurnOutcome.from_request(
            request, status=answer.status, retrieved=retrieved, citations=references
        )
        if answer.status == "failed" or not answer.text.strip():
            return outcome

        context = ResearchChatRagContext(
            search_query=request.user_text,
            search_mode=mode or "general",
            generated_answer=answer.text,
            retrieved_documents=documents,
            citations=tuple(item for item in citations if item.marker in answer.text),
            settings_snapshot=ResearchChatRagSettings(
                requested_mode=request.mode.value,
                requested_retrieval=_server_mode(request.retrieval.mode),
                effective_retrieval=mode,
                retrieved=retrieved,
                generation_state=answer.status,
                provider=route.provider,
                model=route.model,
                top_k=request.retrieval.top_k,
                threshold=request.retrieval.threshold,
                rerank=request.retrieval.rerank,
                processing_route_fingerprint=outcome.processing_route_fingerprint,
                sources=tuple(
                    ResearchChatRagSource(
                        source_id=item.source_id,
                        catalog_item_id=item.catalog_item_id,
                        association_version=item.association_version,
                        catalog_item_version=item.catalog_item_version,
                    )
                    for item in request.source_snapshot
                ),
            ),
        )
        return _PendingAnswer(
            request,
            outcome,
            answer.text,
            history[-1]["id"] if history else None,
            context,
        )

    async def _append(self, pending: _PendingAnswer, service, receipt):
        request = pending.request
        ref = request.workspace
        if receipt is None:
            if request.retry_user_message_id:
                tail = await self._read_messages(
                    service, ref, request.conversation_id, 1
                )
                if not tail or tail[-1]["id"] != request.retry_user_message_id:
                    raise ValueError(
                        "The conversation advanced while regenerating the answer"
                    )
            receipt = ResearchChatAppendReceipt(
                request.turn_id,
                request.conversation_id,
                "user_committed" if request.retry_user_message_id else "pending",
                user_message_id=request.retry_user_message_id,
            )
            await asyncio.to_thread(self._save_receipt, request, receipt)
        if receipt.conversation_id != request.conversation_id:
            raise ValueError("Interrupted answer belongs to a different conversation")
        if receipt.stage == "pending":
            self._check_client(ref, service.client)
            user = await service.create_message(
                request.conversation_id,
                workspace_id=ref.workspace_id,
                role="user",
                content=request.user_text,
                parent_message_id=pending.parent_id,
                idempotency_key=_key(request.turn_id, "user"),
            )
            _validate_append(user, request.conversation_id, pending.parent_id)
            receipt = replace(receipt, stage="user_committed", user_message_id=user.id)
            await asyncio.to_thread(self._save_receipt, request, receipt)
        if receipt.stage == "user_committed":
            self._check_client(ref, service.client)
            assistant = await service.create_message(
                request.conversation_id,
                workspace_id=ref.workspace_id,
                role="assistant",
                content=pending.text,
                parent_message_id=receipt.user_message_id,
                idempotency_key=_key(request.turn_id, "assistant"),
            )
            _validate_append(
                assistant, request.conversation_id, receipt.user_message_id
            )
            receipt = replace(
                receipt, stage="assistant_committed", assistant_message_id=assistant.id
            )
            await asyncio.to_thread(self._save_receipt, request, receipt)
        if receipt.stage == "assistant_committed":
            self._check_client(ref, service.client)
            await service.persist_rag_context(
                receipt.assistant_message_id,
                pending.context,
                workspace_id=ref.workspace_id,
            )
            receipt = replace(receipt, stage="complete")
            await asyncio.to_thread(self._save_receipt, request, receipt)
        return replace(
            pending.outcome,
            user_message_id=receipt.user_message_id,
            assistant_message_id=receipt.assistant_message_id,
        )

    def _check_client(self, ref: QualifiedWorkspaceRef, client: Any) -> None:
        if self.client_for(ref) is not client:
            raise ValueError("Server identity changed while saving the answer")

    def _save_receipt(
        self, request: ResearchChatTurnRequest, receipt: ResearchChatAppendReceipt
    ) -> None:
        for _ in range(3):
            current = self.overlay.load(request.workspace)
            rows = list(current.chat_append_receipts) if current else []
            previous = next(
                (row for row in rows if row.turn_id == receipt.turn_id), None
            )
            if previous is not None:
                if (
                    previous.conversation_id != receipt.conversation_id
                    or _STAGES[previous.stage] > _STAGES[receipt.stage]
                ):
                    raise OverlayConflictError("Interrupted answer receipt changed")
                rows.remove(previous)
            elif len(rows) == MAX_CHAT_APPEND_RECEIPTS:
                completed = next((row for row in rows if row.stage == "complete"), None)
                if completed is None:
                    raise OverlayConflictError(
                        "Finish saving an interrupted answer first"
                    )
                rows.remove(completed)
            rows.append(receipt)
            if receipt.stage == "complete" and request.retry_user_message_id:
                rows = [
                    row
                    for row in rows
                    if not (
                        row.stage == "user_committed"
                        and row.conversation_id == request.conversation_id
                        and row.user_message_id == request.retry_user_message_id
                    )
                ]
            clear = (
                receipt.stage == "complete"
                and current is not None
                and current.chat_conversation_id in {"", request.conversation_id}
                and current.chat_draft == request.user_text
            )
            try:
                self.overlay.save(
                    request.workspace,
                    current.preferences if current else ResearchPanePreferences(),
                    expected_revision=current.revision if current else 0,
                    chat_append_receipts=tuple(rows),
                    chat_draft="" if clear else None,
                )
                return
            except OverlayConflictError:
                continue
        raise OverlayConflictError(
            "Workspace state changed while saving the answer receipt"
        )


def _key(operation_id: str, stage: str) -> str:
    return f"research-{sha256(operation_id.encode()).hexdigest()}:{stage}"


def _server_mode(mode: RetrievalMode | None):
    return "vector" if mode is RetrievalMode.SEMANTIC else mode.value if mode else None


async def _check_sources(ref, sources, read_sources):
    current = await read_sources(ref)
    if any(source not in current for source in sources.values()):
        raise ValueError("Selected Server sources changed. Review and retry")


def _read_documents(result, sources, top_k):
    documents, remaining = [], _EXCERPT_CHARS
    for row in result.get("documents", ()):
        metadata = row.get("metadata") or {}
        media_id = str(
            metadata.get("media_id")
            or metadata.get("source_id")
            or row.get("media_id")
            or row.get("id")
            or ""
        )
        source_type = (
            row.get("source") or row.get("source_type") or metadata.get("source_type")
        )
        content = row.get("content") or row.get("text") or row.get("excerpt")
        if (
            media_id not in sources
            or source_type not in {"media", "media_db"}
            or not isinstance(content, str)
            or not content.strip()
        ):
            continue
        title = str(row.get("title") or metadata.get("title") or "Source")[:300]
        excerpt = content.strip()[: min(remaining, 6000)]
        score = row.get("score")
        documents.append(
            ResearchChatRagDocument(
                id=media_id, title=title, excerpt=excerpt, score=score
            )
        )
        remaining -= len(excerpt)
        if remaining == 0 or len(documents) == top_k:
            break
    return tuple(documents)


def _history_rows(history, limit):
    messages, remaining = [], _HISTORY_CHARS
    for row in reversed(history if limit else ()):
        role, text = row.get("role") or row.get("sender"), row.get("content")
        if role not in {"user", "assistant"} or not isinstance(text, str) or not text:
            continue
        if len(text) > remaining:
            break
        remaining -= len(text)
        messages.append(
            {
                "role": role,
                "content": _literal_question(text) if role == "user" else text,
            }
        )
    return list(reversed(messages))


def _literal_question(text: str) -> str:
    # The legacy completion route dispatches leading slash commands before inference.
    return f"Question:\n{text}" if text.lstrip().startswith("/") else text


def _validate_append(row, conversation_id, parent_id):
    if row.conversation_id != conversation_id or row.parent_message_id != parent_id:
        raise ValueError("Server did not confirm the requested conversation and parent")
