"""Content-minimizing scripted provider for Console Library qualification."""

from __future__ import annotations

import asyncio
import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from tldw_chatbook.Agents.agent_models import ToolCall
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleResolvedDestination,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderResolution,
    ProviderToolCalls,
    ProviderTurnMetadata,
)
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationRound,
    ProviderContinuationCheckpoint,
)
from tldw_chatbook.Library.library_rag_state import LibraryRagResultRow


@dataclass(frozen=True, slots=True)
class RecordedProviderCall:
    """One provider crossing with content-free request metadata."""

    ordinal: int
    kind: str
    destination: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RetrievalScript:
    """One deterministic automatic/manual retrieval outcome."""

    outcome: Literal["success", "zero", "failure", "timeout"]

    @classmethod
    def success(cls) -> "RetrievalScript":
        return cls("success")

    @classmethod
    def zero(cls) -> "RetrievalScript":
        return cls("zero")

    @classmethod
    def failure(cls) -> "RetrievalScript":
        return cls("failure")

    @classmethod
    def timeout(cls) -> "RetrievalScript":
        return cls("timeout")


@dataclass(frozen=True, slots=True)
class StreamScript:
    """One deterministic token, failure, timeout, or terminal stream."""

    outcome: Literal["tokens", "failure", "timeout", "terminal"]
    chunks: tuple[str, ...] = ()

    @classmethod
    def tokens(cls, *chunks: str) -> "StreamScript":
        return cls("tokens", tuple(chunks))

    @classmethod
    def failure(cls) -> "StreamScript":
        return cls("failure")

    @classmethod
    def timeout(cls) -> "StreamScript":
        return cls("timeout")

    @classmethod
    def terminal(cls, text: str) -> "StreamScript":
        return cls("terminal", (text,))


@dataclass(frozen=True, slots=True)
class ToolBatchScript:
    """One native tool-call batch, optionally carrying continuation state."""

    calls: tuple[ToolCall, ...]
    continuation: ProviderContinuationCheckpoint | None = field(
        default=None,
        repr=False,
    )

    @classmethod
    def library_search_then_continue(cls) -> "ToolBatchScript":
        call = ToolCall(
            name="search_library_rag",
            args={"query": "qualification"},
            call_id="call-library-1",
            raw_arguments='{"query":"qualification"}',
        )
        continuation = ProviderContinuationCheckpoint(
            schema_version=1,
            checkpoint_revision=1,
            provider="deepseek",
            protocol="chat_completions",
            model="qualification-model",
            api_base_url="https://provider.example.invalid/v1",
            state="active",
            rounds=(
                ContinuationRound(
                    assistant_content="",
                    reasoning_blocks=(),
                    calls=(
                        ContinuationCall(
                            call_id=call.call_id,
                            name=call.name,
                            arguments=call.raw_arguments,
                            state="pending",
                        ),
                    ),
                ),
            ),
        )
        return cls((call,), continuation)


class RecordingConsoleProvider:
    """Script provider, retrieval service, and direct Library service.

    Recorded rows retain only counts, names, routing metadata, and hashes or
    lengths. Message content, retrieval excerpts, and Library bodies may cross
    the return boundary to production code but never enter ``calls``.
    """

    def __init__(
        self,
        *,
        ready: bool = True,
        egress: ConsoleEgressClass = ConsoleEgressClass.ON_DEVICE,
        retrieval_scripts: list[RetrievalScript] | None = None,
        stream_scripts: list[StreamScript] | None = None,
        model_scripts: list[ToolBatchScript | StreamScript] | None = None,
        stream_gates: list[asyncio.Event] | None = None,
        fixed_provider: str | None = None,
        fixed_model: str | None = None,
    ) -> None:
        self.ready = ready
        self.egress = egress
        self.retrieval_scripts = deque(retrieval_scripts or [])
        self.stream_scripts = deque(stream_scripts or [])
        self.model_scripts = deque(model_scripts or [])
        self.stream_gates = tuple(stream_gates or ())
        self.stream_started = tuple(asyncio.Event() for _ in self.stream_gates)
        self.fixed_provider = fixed_provider
        self.fixed_model = fixed_model
        self.calls: list[RecordedProviderCall] = []
        self.activity_events: list[Any] = []

    def _record(self, kind: str, destination: str, **metadata: Any) -> None:
        self.calls.append(
            RecordedProviderCall(len(self.calls) + 1, kind, destination, metadata)
        )

    def calls_of(self, kind: str) -> list[RecordedProviderCall]:
        """Return calls of one kind in boundary-crossing order."""
        return [call for call in self.calls if call.kind == kind]

    async def resolve_for_send(self, selection: Any) -> ConsoleProviderResolution:
        """Return scripted readiness and record only routing metadata."""
        provider = self.fixed_provider or str(
            getattr(selection, "provider", None) or "llama_cpp"
        )
        model = str(
            self.fixed_model
            or getattr(selection, "explicit_model", None)
            or "qualification-model"
        )
        endpoint = (
            "http://127.0.0.1:9099"
            if self.egress is ConsoleEgressClass.ON_DEVICE
            else "https://provider.example.invalid/v1"
        )
        destination = ConsoleResolvedDestination(
            provider=provider,
            model=model,
            endpoint_identity=endpoint,
            egress_class=self.egress,
        )
        self._record(
            "readiness",
            self.egress.value,
            provider=provider,
            model=model,
            ready=self.ready,
        )
        return ConsoleProviderResolution(
            provider=provider,
            base_url=endpoint,
            model=model,
            ready=self.ready,
            visible_copy="" if self.ready else "Provider is not ready.",
            readiness_key=provider,
            execution_key=provider,
            continuation_protocol="chat_completions",
            resolved_destination=destination if self.ready else None,
        )

    async def stream_chat(
        self,
        resolution: ConsoleProviderResolution,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ):
        """Yield the next scripted stream without recording message bodies."""
        tools = kwargs.get("tools") or ()
        tool_names = tuple(
            str(
                (tool.get("function") or {}).get("name")
                if isinstance(tool, dict)
                else getattr(tool, "name", "")
            )
            for tool in tools
        )
        destination = (
            resolution.resolved_destination.egress_class.value
            if resolution.resolved_destination is not None
            else "unresolved"
        )
        stream_index = len(self.calls_of("stream"))
        self._record(
            "stream",
            destination,
            message_count=len(messages),
            roles=tuple(str(message.get("role", "")) for message in messages),
            tool_names=tool_names,
        )
        if stream_index < len(self.stream_gates):
            self.stream_started[stream_index].set()
            await self.stream_gates[stream_index].wait()
        if self.model_scripts:
            script = self.model_scripts.popleft()
        elif self.stream_scripts:
            script = self.stream_scripts.popleft()
        else:
            script = StreamScript.terminal("ok")
        if isinstance(script, ToolBatchScript):
            yield ProviderToolCalls(
                tuple(
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.raw_arguments,
                        },
                    }
                    for call in script.calls
                ),
                metadata=ProviderTurnMetadata(
                    finish_reason="tool_calls",
                    provider_continuation=script.continuation,
                ),
            )
            return
        if script.outcome == "failure":
            raise RuntimeError("scripted provider failure")
        if script.outcome == "timeout":
            await asyncio.sleep(60)
            return
        for chunk in script.chunks:
            yield chunk

    async def search(
        self,
        query: str,
        source_types: tuple[str, ...] | list[str],
        mode: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return scripted RAG data and record no query or result body."""
        scope = kwargs.get("scope")
        self._record(
            "retrieval",
            "local_library",
            query_bytes=len(query.encode("utf-8")),
            query_sha256=hashlib.sha256(query.encode("utf-8")).hexdigest(),
            source_types=tuple(source_types),
            mode=mode,
            top_k=kwargs.get("top_k"),
            scoped=getattr(scope, "state", "all") != "all",
        )
        script = (
            self.retrieval_scripts.popleft()
            if self.retrieval_scripts
            else RetrievalScript.zero()
        )
        if script.outcome == "failure":
            raise RuntimeError("scripted retrieval failure")
        if script.outcome == "timeout":
            await asyncio.sleep(60)
        if script.outcome in {"zero", "timeout"}:
            return {"runtime_backend": "local", "results": []}
        row = LibraryRagResultRow.from_result(
            {
                "source_id": "note-qualification",
                "chunk_id": "chunk-qualification",
                "title": "Qualification note",
                "content": "PRIVATE LIBRARY BODY",
                "score": 0.9,
                "runtime_backend": "local",
                "source_type": "notes",
            }
        )
        return {"runtime_backend": "local", "results": [row]}

    def invoke(self, name: str, arguments: object) -> dict[str, Any]:
        """Serve bounded direct-tool data while recording no result body."""
        argument_map = arguments if isinstance(arguments, dict) else {}
        query = argument_map.get("query")
        tool_name = name.rsplit(":", 1)[-1]
        metadata: dict[str, Any] = {
            "name": tool_name,
            "argument_keys": tuple(sorted(str(key) for key in argument_map)),
            "has_query": isinstance(query, str) and bool(query),
        }
        if tool_name == "library_list_notes":
            limit = argument_map.get("limit")
            metadata["limit"] = limit if type(limit) is int else None
        elif tool_name == "library_get_note":
            note_id = argument_map.get("note_id")
            encoded_note_id = (
                note_id.encode("utf-8") if isinstance(note_id, str) else b""
            )
            metadata["note_id_bytes"] = len(encoded_note_id)
            metadata["note_id_sha256"] = hashlib.sha256(encoded_note_id).hexdigest()
        self._record(
            "direct_tool",
            "local_library",
            **metadata,
        )
        return {
            "items": [{"id": "note:qualification", "title": "Qualification"}],
            "total": 1,
        }
