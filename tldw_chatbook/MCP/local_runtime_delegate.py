from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from tldw_chatbook.config import get_chachanotes_db_lazy, get_media_db_lazy
from tldw_chatbook.Library.library_tool_contract import LIBRARY_TOOL_DESCRIPTORS

from .server import MCP_AVAILABLE, describe_local_mcp_capabilities

# Fix Round A (PR-T3 whole-branch review), Item 2. Task 6 (PR-T3) refused a
# raw `tools/call` in `UnifiedMCPControlPlaneService.run_action()`
# (`_refuse_raw_tool_call`), one layer above this delegate -- but that
# refusal sits one layer above the seam it protects: `request()`'s
# `tools/call` branch below calls `self.execute_tool()` directly, so any
# caller that reaches `request()`/`batch()` WITHOUT going through
# `run_action()` (a direct call on this delegate, `batch()`'s own loop over
# `request()`, or `LocalMCPControlService.run_runtime_request()`/
# `run_runtime_batch()` -- both public methods reachable off
# `app.local_mcp_control_service`) bypassed the gate and the audit-log row
# entirely.
#
# This module-level constant is shared verbatim with
# `unified_control_plane_service.py`'s own refusal so the two independent
# enforcement points can never show the user different copy for the same
# refusal. Each site carries a comment naming which job it does:
#   - `run_action()`'s pre-dispatch scan (control-plane layer): preserves
#     `runtime.batch`'s all-or-nothing property -- checked before ANY item
#     dispatches, since the batch runs serially and a per-item refusal here
#     alone would only stop the offending item, not the ones before it that
#     had already executed.
#   - This delegate's `request()` (this layer): the durable backstop that
#     catches every caller, including ones that never went through that
#     scan.
RAW_TOOL_CALL_REFUSED_MESSAGE = (
    "Tool calls run through the Execute Local Tool action, which applies your "
    "Permissions settings and records the run."
)

# Fix Round G, Item 7 (PR-T3 review of Fix Round F). Independent surfaces
# state the identical "the permission RESOLVER itself failed, not a
# genuine verdict" condition (`EffectiveToolState(state="deny",
# origin="gate_error")`) in their own sentence shape:
#   - `unified_control_plane_service._ADVANCED_EXECUTE_GATE_ERROR_MESSAGE`:
#     a bare, capitalized sentence -- the Advanced hatch already carries
#     "Blocked · not run" on its own heading line (`MCPInspector.
#     _ADVANCED_BLOCKED_HEADING`), so the body under it is just the claim.
#   - `mcp_workbench._TOOL_TEST_BLOCKED_UNKNOWN_TEXT`: `"Blocked —
#     <clause>."`, deliberately parallel to its own genuine-deny sibling
#     `_TOOL_TEST_BLOCKED_TEXT` ("Blocked — this tool is set to Off in
#     Permissions.") -- the Test Tool panel has no separate "Blocked"
#     heading of its own, so each of its blocked bodies carries the prefix.
#   - `mcp_inspector._UNKNOWN_ORIGIN_SENTENCE` (Fix Round I, Item 4): a
#     bare, capitalized sentence like the Advanced hatch's -- the
#     Permissions-explanation block's fallback for an `EffectiveToolState.
#     origin` this UI doesn't otherwise recognize (`_ORIGIN_SENTENCES.get
#     (effective.origin, _UNKNOWN_ORIGIN_SENTENCE)`), reachable whenever a
#     Tools-mode tool selection's own `gate_tool_test()` call raises
#     (`MCPWorkbench._effective_for_display()`'s single-tool fallback
#     path). A review found this one was STILL an independently-maintained
#     literal after the first two converged -- brought in here rather than
#     left as the "majority phrasing" a comment nearby used to justify the
#     other two's convergence while not actually including it.
#
# Before the first fix these were independently maintained literals that
# happened to read close ("could not be resolved" vs. "could not be
# determined") -- exactly the drifted-duplicate shape this whole PR exists
# to close. One clause here, lowercase and without terminal punctuation so
# any sentence SHAPE can compose around it (a leading capital for a
# bare-sentence surface, a lowercase mid-sentence fragment after a
# "Blocked — " em-dash for another) -- this is what all three surfaces
# above are now DERIVED from, not merely equal to: a reword here changes
# all three, or they go visibly out of sync at the assertion syntax level
# (a stale call site left unedited), not just at the reader's eye. Homed
# here rather than in `unified_control_plane_service.py`/`mcp_workbench.
# py`/`mcp_inspector.py`, mirroring `RAW_TOOL_CALL_REFUSED_MESSAGE` just
# above: this module imports from none of them, and all three already
# import from this module, so each can depend on this one without a cycle.
PERMISSION_STATE_UNRESOLVED_CLAUSE = "permission state could not be resolved"


def capitalize_first(text: str) -> str:
    """Uppercase only the first character; every other character is left
    exactly as given.

    Fix Round I, Item 3. `str.capitalize()` does NOT do this -- it
    additionally LOWERCASES every character after the first, silently
    mangling any acronym, proper noun, or server name a future clause
    might contain. Proven live at `unified_control_plane_service.
    _ADVANCED_EXECUTE_GATE_ERROR_MESSAGE`'s old
    `f"{PERMISSION_STATE_UNRESOLVED_CLAUSE.capitalize()}."`: a clause
    reading "MUTATED permission state is unknown" rendered as "Mutated
    permission state is unknown." -- silently downcasing MUTATED with no
    signal that anything had changed. `PERMISSION_STATE_UNRESOLVED_CLAUSE`
    itself has no acronym today, so the bug was latent, not yet visible in
    the shipped copy; the fragility was in the FORMULA, reachable by any
    future reword of the clause, and would have reintroduced a
    cross-surface divergence through the very mechanism (`capitalize_
    first`, replacing three separate `.capitalize()`/reimplementation call
    sites) added to prevent one.

    Every surface that turns `PERMISSION_STATE_UNRESOLVED_CLAUSE` (or any
    future shared clause homed here) into a leading-capital sentence must
    call this, not `str.capitalize()` and not its own hand-rolled
    `text[0].upper() + text[1:]` -- one implementation, so a future
    behavior change (e.g. Unicode titlecasing edge cases) updates every
    caller identically instead of risking a fourth reimplementation
    drifting from the other three.

    Args:
        text: The clause to sentence-case; may be empty.

    Returns:
        ``text`` with only its first character uppercased -- every other
        character byte-identical to the input; ``""`` for ``""``.
    """
    return text[:1].upper() + text[1:]


class RawToolCallRefusedError(PermissionError):
    """A raw ``tools/call`` was refused -- run it through Execute Local Tool.

    Item 2 (PR-T3 fix round D). Raised at BOTH enforcement points that
    share :data:`RAW_TOOL_CALL_REFUSED_MESSAGE` above -- this delegate's
    own ``request()`` (the durable backstop, below) and
    ``unified_control_plane_service.UnifiedMCPControlPlaneService.
    _refuse_raw_tool_call()`` (the control-plane pre-dispatch scan, one
    layer up) -- so the two independent refusals of the identical event
    share one type the same way they already share one message. Defined
    here, not in ``unified_control_plane_service.py``, because this
    module is the dependency-safe common ground: ``local_control_
    service.py`` (home of the sibling ``MCPGovernanceDenied``) and
    ``unified_control_plane_service.py`` both already import from this
    module, and this module imports from neither of them.

    Subclasses ``PermissionError`` so any existing ``except
    PermissionError`` handler upstream keeps working unchanged.
    `UI/MCP_Modules/mcp_inspector.py`'s Advanced runner narrows its own
    handler to this type (among others) instead of the bare base class,
    so a tool's own body raising an unrelated ``PermissionError`` renders
    as a failure, not a refusal that never reached the tool.
    """

    def __init__(self, message: str = RAW_TOOL_CALL_REFUSED_MESSAGE) -> None:
        super().__init__(message)


class LocalMCPRuntimeDelegate:
    """Direct in-process runtime kept separate from the standalone gateway."""

    _PROTOCOL_VERSION = "2025-03-26"
    _REQUEST_METHODS = (
        "initialize",
        "status/get",
        "tools/list",
        "resources/list",
        "prompts/list",
        "tools/call",
        "resources/read",
        "prompts/get",
    )
    _UNAVAILABLE_DIRECT_TOOLS = {"chat_with_llm"}
    #: Fix Round C (PR-T3 review), Item 1. `request()`'s `tools/call` branch
    #: below refuses unconditionally (`RAW_TOOL_CALL_REFUSED_MESSAGE`), but
    #: `tools/call` stayed listed as an ordinary, fully-supported entry in
    #: `get_protocol_diagnostics()`'s `methods` list -- the exact surface an
    #: agent reads before planning a `runtime.request` call (reachable via
    #: `run_action("runtime.protocol.inspect")`). An agent that saw
    #: `{"name": "tools/call", "supported": true}` there had no way to learn
    #: the call would be refused short of trying it. Mirrors
    #: `_UNAVAILABLE_DIRECT_TOOLS` above in NAME only: `tools/call` stays a
    #: *recognized* method (still enumerated in `_REQUEST_METHODS`, since
    #: `request()` genuinely understands it well enough to refuse it by name
    #: rather than raising "unsupported method") and is reported
    #: `supported: False` in diagnostics -- but that is a two-way flag, not
    #: the tools bucket's three-way `implemented`/`unavailable`/`missing`
    #: split (Fix Round E, Item 5: the shapes differ). `supported: False`
    #: here can ONLY mean a policy refusal, though: every entry in
    #: `get_protocol_diagnostics()`'s `methods` list is built by iterating
    #: `_REQUEST_METHODS` -- this class's own fixed roster of methods it
    #: recognizes -- so a method that is simply not implemented never gets
    #: a `methods` entry at all; it is ABSENT, not present with `supported:
    #: False`. Fix Round G (review of Fix Round E): the prior wording here
    #: -- "a reader of THIS surface alone cannot tell a policy refusal like
    #: this one from a method that is simply not implemented" -- overstated
    #: the gap. The two cases ARE distinguishable, by presence vs. absence
    #: in the list; what the two-way/three-way shape difference actually
    #: means is narrower: absence is not itself LABELED "unimplemented" the
    #: way the tools bucket's own `missing` key spells it out. The flag
    #: itself is honest and load-bearing either way; only that framing was
    #: not quite right. See `get_protocol_diagnostics()` for where this is
    #: read.
    _UNAVAILABLE_DIRECT_METHODS = frozenset({"tools/call"})
    _RESOURCE_URI_PREFIXES = (
        "conversation://",
        "note://",
        "character://",
        "media://",
        "rag-chunk://",
    )
    _PROMPT_NAMES = (
        "summarize_conversation",
        "generate_document",
        "analyze_media",
        "search_and_synthesize",
        "character_writing",
    )

    def __init__(
        self,
        *,
        manifest_provider: Callable[[], dict[str, Any]] | None = None,
        library_service: Any | None = None,
        policy_enforcer: Any | None = None,
    ) -> None:
        self._manifest_provider = manifest_provider or describe_local_mcp_capabilities
        # task-1337 (plan Task 9): shared synchronous LocalLibraryToolService
        # (duck-typed ``invoke``). Injected by tests/hosts; lazily composed
        # from the process-local databases on first Library dispatch.
        self._library_service = library_service
        # chunking-agent-tools (Task 5, spec §6): threaded into the lazily
        # composed shared Library service so the WRITING chunk tools are
        # service-level gated on the local MCP surface (the Console
        # construction site passes the same app handle).
        self._policy_enforcer = policy_enforcer
        self._tools: Any | None = None
        self._resources: Any | None = None
        self._prompts: Any | None = None
        self._initialized_at = datetime.now(timezone.utc)

    def get_status(self) -> dict[str, Any]:
        manifest = self._get_manifest()
        return {
            "server_id": manifest.get("server_id", "local:tldw_chatbook"),
            "server_label": manifest.get("server_label", "tldw_chatbook local MCP"),
            "mcp_sdk_available": MCP_AVAILABLE,
            "tool_count": len(list(manifest.get("tools", []))),
            "resource_count": len(list(manifest.get("resources", []))),
            "prompt_count": len(list(manifest.get("prompts", []))),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_protocol_capabilities(self) -> dict[str, Any]:
        """The methods `request()` recognizes and will dispatch on by name.

        NOTE: "recognized" is not "will succeed" -- `tools/call` is listed
        in `request_methods` because `request()` genuinely understands that
        method name (it is not an "unsupported method" `KeyError`), but
        every call is unconditionally refused (see `request()`'s
        `tools/call` branch and `RAW_TOOL_CALL_REFUSED_MESSAGE`).

        Item 4 (PR-T3 fix round D), closing the scope Fix Round C
        deliberately left open (see that round's own commit message):
        `request_methods` listed `tools/call` with nothing on THIS
        method's own return value to say it is refused -- that signal
        lived only on `get_protocol_diagnostics()`'s per-method
        `supported` flag. `request_methods` itself is UNCHANGED (Fix
        Round C's own reasoning still holds: dropping `tools/call` from
        it would itself be inaccurate, since `request()` recognizes the
        method by name rather than raising the generic "unsupported
        method" `KeyError`). Instead, `unavailable_request_methods` is a
        NEW field, using the same `_UNAVAILABLE_DIRECT_METHODS`
        vocabulary `get_protocol_diagnostics()` already uses -- so the
        "recognized but refused" distinction is visible on THIS method's
        own payload, not only the sibling one. `get_protocol_diagnostics()`
        remains the surface with full per-method detail (a real
        `supported` flag on every entry, not just the unavailable ones).

        Item 3 (PR-T3 fix round F): the paragraph above used to justify
        this by describing "an agent reading ONLY this method, never
        cross-referencing `get_protocol_diagnostics()`" -- overclaiming a
        reader that does not exist. This method has exactly ONE
        production consumer, `LocalMCPControlService.get_advanced()`
        (`local_control_service.py:329`), which already returns this
        method's output and `get_protocol_diagnostics()`'s output
        together, as sibling keys (`"protocol"` / `"protocol_diagnostics"`)
        of the SAME payload -- nothing in this codebase ever sees one
        without the other. The field is still worth having: a caller that
        renders or forwards only the `"protocol"` slice of that payload
        (or any future consumer of this method taken in isolation) would
        otherwise have to reconstruct "which methods are unavailable"
        from the diagnostics' full per-method list instead of reading it
        directly here. The change was never in question, only its
        justification -- say what this actually does.
        """
        return {
            "adapter": "direct_in_process",
            "supports_batch": True,
            "request_methods": list(self._REQUEST_METHODS),
            "unavailable_request_methods": [
                method
                for method in self._REQUEST_METHODS
                if method in self._UNAVAILABLE_DIRECT_METHODS
            ],
        }

    def get_protocol_diagnostics(self) -> dict[str, Any]:
        manifest = self._get_manifest()
        tools = list(manifest.get("tools", []))
        resources = list(manifest.get("resources", []))
        prompts = list(manifest.get("prompts", []))
        tool_names = self._entry_names(tools, "name")
        prompt_names = self._entry_names(prompts, "name")
        resource_prefixes = self._resource_prefixes(resources)

        return {
            "adapter": "direct_in_process",
            "protocol_version": self._PROTOCOL_VERSION,
            "transport": "in_process",
            "mcp_sdk_available": MCP_AVAILABLE,
            "supports_batch": True,
            "methods": [
                {
                    "name": method,
                    "supported": method not in self._UNAVAILABLE_DIRECT_METHODS,
                }
                for method in self._REQUEST_METHODS
            ],
            "manifest": {
                "tools": len(tools),
                "resources": len(resources),
                "prompts": len(prompts),
            },
            "implementation": {
                "tools": {
                    "implemented": [
                        name
                        for name in tool_names
                        if name not in self._UNAVAILABLE_DIRECT_TOOLS
                        and (
                            name in LIBRARY_TOOL_DESCRIPTORS
                            or hasattr(self, f"_tool_{name}")
                        )
                    ],
                    "unavailable": [
                        name
                        for name in tool_names
                        if name in self._UNAVAILABLE_DIRECT_TOOLS
                    ],
                    "missing": [
                        name
                        for name in tool_names
                        if name not in self._UNAVAILABLE_DIRECT_TOOLS
                        and name not in LIBRARY_TOOL_DESCRIPTORS
                        and not hasattr(self, f"_tool_{name}")
                    ],
                },
                "resources": {
                    "supported_uri_prefixes": [
                        prefix
                        for prefix in self._RESOURCE_URI_PREFIXES
                        if prefix in resource_prefixes
                    ],
                },
                "prompts": {
                    "implemented": [
                        name for name in prompt_names if name in self._PROMPT_NAMES
                    ],
                    "missing": [
                        name for name in prompt_names if name not in self._PROMPT_NAMES
                    ],
                },
            },
        }

    def get_runtime_health(self) -> dict[str, Any]:
        manifest = self._get_manifest()
        now = datetime.now(timezone.utc)
        tools = list(manifest.get("tools", []))
        resources = list(manifest.get("resources", []))
        prompts = list(manifest.get("prompts", []))
        issues: list[str] = []
        if not tools and not resources and not prompts:
            issues.append("Local MCP manifest is empty.")
        return {
            "state": "ready" if not issues else "degraded",
            "adapter": "direct_in_process",
            "transport": "in_process",
            "mcp_sdk_available": MCP_AVAILABLE,
            "initialized_at": self._initialized_at.isoformat(),
            "uptime_seconds": max(0.0, (now - self._initialized_at).total_seconds()),
            "manifest": {
                "loaded": True,
                "tools": len(tools),
                "resources": len(resources),
                "prompts": len(prompts),
            },
            "component_cache": {
                "tools_loaded": self._tools is not None,
                "resources_loaded": self._resources is not None,
                "prompts_loaded": self._prompts is not None,
            },
            "issues": issues,
        }

    async def execute_tool(
        self, tool_name: str, arguments: Mapping[str, Any] | None = None
    ) -> Any:
        normalized_name = str(tool_name or "").strip()
        payload = dict(arguments or {})
        if normalized_name in LIBRARY_TOOL_DESCRIPTORS:
            # task-1337 (plan Task 9): descriptor-backed Library tools dispatch
            # to the shared synchronous service, always off the event loop,
            # and the service payload returns unchanged (structured errors
            # included -- they are data, not exceptions).
            service = self._get_library_service()
            return await asyncio.to_thread(
                service.invoke, normalized_name, payload
            )
        handler = getattr(self, f"_tool_{normalized_name}", None)
        if handler is None:
            raise KeyError(f"Unsupported local MCP tool: {normalized_name}")
        return await handler(payload)

    async def request(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        normalized_method = str(method or "").strip()
        payload = dict(params or {})
        manifest = self._get_manifest()

        if normalized_method == "initialize":
            return {
                "protocolVersion": self._PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"listChanged": False},
                    "prompts": {"listChanged": False},
                },
                "serverInfo": {
                    "name": manifest.get("server_id", "local:tldw_chatbook"),
                    "label": manifest.get("server_label", "tldw_chatbook local MCP"),
                },
            }
        if normalized_method == "status/get":
            return self.get_status()
        if normalized_method == "tools/list":
            return {"tools": list(manifest.get("tools", []))}
        if normalized_method == "resources/list":
            return {"resources": list(manifest.get("resources", []))}
        if normalized_method == "prompts/list":
            return {"prompts": list(manifest.get("prompts", []))}
        if normalized_method == "tools/call":
            # Fix Round A, Item 2: the durable backstop -- refuse here too,
            # not just at the control-plane pre-dispatch scan (see the
            # module-level `RAW_TOOL_CALL_REFUSED_MESSAGE` comment for why
            # both layers are needed). Tool execution keeps exactly one
            # door through this delegate: `execute_tool()`, called directly
            # by `LocalMCPControlService.execute_tool()` for the gated,
            # logged `tool.execute` action -- never through this raw
            # protocol branch. Fix Round C, Item 1: this refusal is now also
            # advertised, not just enforced -- `get_protocol_diagnostics()`
            # reports `tools/call` as `supported: False`
            # (`_UNAVAILABLE_DIRECT_METHODS`), so an agent inspecting the
            # protocol before planning a call sees the same "no" this raise
            # produces, instead of discovering it by trying.
            # Item 2 (PR-T3 fix round D): typed, not a bare `PermissionError`
            # -- see `RawToolCallRefusedError`'s own docstring for why.
            raise RawToolCallRefusedError(RAW_TOOL_CALL_REFUSED_MESSAGE)
        if normalized_method == "resources/read":
            return {
                "resource_uri": self._require_payload_field(
                    payload, "uri", aliases=("resource_uri",)
                ),
                "result": await self.read_resource(
                    self._require_payload_field(
                        payload, "uri", aliases=("resource_uri",)
                    )
                ),
            }
        if normalized_method == "prompts/get":
            arguments = payload.get("arguments")
            prompt_name = self._require_payload_field(
                payload, "name", aliases=("prompt_name",)
            )
            normalized_arguments = arguments if isinstance(arguments, Mapping) else {}
            return {
                "prompt_name": prompt_name,
                "arguments": dict(normalized_arguments),
                "messages": await self.get_prompt(prompt_name, normalized_arguments),
            }
        raise KeyError(f"Unsupported local MCP runtime method: {normalized_method}")

    async def batch(
        self, requests: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for index, request in enumerate(requests):
            method = (
                str(request.get("method") or "").strip()
                if isinstance(request, Mapping)
                else ""
            )
            params = request.get("params") if isinstance(request, Mapping) else None
            try:
                await self.request(
                    method, params if isinstance(params, Mapping) else {}
                )
                results.append(
                    {
                        "index": index,
                        "method": method,
                        "ok": True,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "index": index,
                        "method": method,
                        "ok": False,
                        "error": str(exc),
                    }
                )
        return results

    async def read_resource(self, resource_uri: str) -> dict[str, Any]:
        normalized_uri = str(resource_uri or "").strip()
        resources = self._get_resources()
        if normalized_uri.startswith("conversation://"):
            return await resources.get_conversation_resource(
                normalized_uri.removeprefix("conversation://")
            )
        if normalized_uri.startswith("note://"):
            return await resources.get_note_resource(
                normalized_uri.removeprefix("note://")
            )
        if normalized_uri.startswith("character://"):
            return await resources.get_character_resource(
                normalized_uri.removeprefix("character://")
            )
        if normalized_uri.startswith("media://"):
            return await resources.get_media_resource(
                normalized_uri.removeprefix("media://")
            )
        if normalized_uri.startswith("rag-chunk://"):
            return await resources.get_rag_chunk_resource(
                normalized_uri.removeprefix("rag-chunk://")
            )
        raise KeyError(f"Unsupported local MCP resource URI: {normalized_uri}")

    async def get_prompt(
        self, prompt_name: str, arguments: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        normalized_name = str(prompt_name or "").strip()
        payload = dict(arguments or {})
        prompts = self._get_prompts()
        if normalized_name == "summarize_conversation":
            return await prompts.summarize_conversation_prompt(**payload)
        if normalized_name == "generate_document":
            return await prompts.generate_document_prompt(**payload)
        if normalized_name == "analyze_media":
            return await prompts.analyze_media_prompt(**payload)
        if normalized_name == "search_and_synthesize":
            return await prompts.search_and_synthesize_prompt(**payload)
        if normalized_name == "character_writing":
            return await prompts.character_writing_prompt(**payload)
        raise KeyError(f"Unsupported local MCP prompt: {normalized_name}")

    async def _tool_chat_with_llm(self, arguments: dict[str, Any]) -> dict[str, Any]:
        del arguments
        raise RuntimeError(
            "Local MCP tool 'chat_with_llm' is not available through the direct local runtime delegate yet."
        )

    async def _tool_chat_with_character(self, arguments: dict[str, Any]) -> Any:
        return await self._get_tools().chat_with_character(
            message=str(arguments.get("message") or ""),
            character_id=int(arguments.get("character_id")),
            provider=str(arguments.get("provider") or "openai"),
            model=arguments.get("model"),
            temperature=float(arguments.get("temperature", 0.7)),
            max_tokens=int(arguments.get("max_tokens", 4096)),
            conversation_id=arguments.get("conversation_id"),
        )

    async def _tool_search_rag(self, arguments: dict[str, Any]) -> Any:
        media_types = arguments.get("media_types")
        return await self._get_tools().perform_rag_search(
            query=str(arguments.get("query") or ""),
            limit=int(arguments.get("limit", 10)),
            media_types=list(media_types) if isinstance(media_types, list) else None,
            use_semantic=bool(arguments.get("use_semantic", True)),
        )

    async def _tool_search_conversations(self, arguments: dict[str, Any]) -> Any:
        return await self._get_tools().search_conversations(
            query=str(arguments.get("query") or ""),
            limit=int(arguments.get("limit", 10)),
            character_id=arguments.get("character_id"),
        )

    async def _tool_create_note(self, arguments: dict[str, Any]) -> dict[str, Any]:
        db = self._require_chachanotes_db()
        note_id = db.add_note(
            title=str(arguments.get("title") or ""),
            content=str(arguments.get("content") or ""),
            note_id=arguments.get("note_id"),
        )
        result: dict[str, Any] = {
            "id": note_id,
            "title": str(arguments.get("title") or ""),
            "created": datetime.now(timezone.utc).isoformat(),
        }
        if arguments.get("tags") or arguments.get("template"):
            result["warning"] = (
                "Local MCP create_note currently persists title/content only."
            )
        return result

    async def _tool_search_notes(
        self, arguments: dict[str, Any]
    ) -> list[dict[str, Any]]:
        db = self._require_chachanotes_db()
        results = db.search_notes(
            search_term=str(arguments.get("query") or ""),
            limit=int(arguments.get("limit", 10)),
        )
        return [
            {
                "id": item["id"],
                "title": item["title"],
                "preview": item["content"][:200] + "..."
                if len(item["content"]) > 200
                else item["content"],
                "created": item.get("created_at"),
                "modified": item.get("last_modified"),
            }
            for item in results
        ]

    async def _tool_list_characters(self, arguments: dict[str, Any]) -> Any:
        del arguments
        return await self._get_tools().list_available_characters()

    async def _tool_get_conversation_history(self, arguments: dict[str, Any]) -> Any:
        return await self._get_tools().get_conversation_history(
            conversation_id=int(arguments.get("conversation_id")),
            limit=arguments.get("limit"),
        )

    async def _tool_export_conversation(self, arguments: dict[str, Any]) -> Any:
        return await self._get_tools().export_conversation(
            conversation_id=int(arguments.get("conversation_id")),
            format=str(arguments.get("format") or "markdown"),
        )

    def _require_chachanotes_db(self):
        db = get_chachanotes_db_lazy()
        if db is None:
            raise RuntimeError("Local ChaChaNotes database is unavailable.")
        return db

    def _require_media_db(self):
        db = get_media_db_lazy()
        if db is None:
            raise RuntimeError("Local media database is unavailable.")
        return db

    def _get_manifest(self) -> dict[str, Any]:
        return self._manifest_provider() or {}

    @staticmethod
    def _entry_names(entries: list[Any], field_name: str) -> list[str]:
        names: list[str] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            name = str(entry.get(field_name) or "").strip()
            if name:
                names.append(name)
        return names

    @staticmethod
    def _resource_prefixes(entries: list[Any]) -> set[str]:
        prefixes: set[str] = set()
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            uri = str(entry.get("uri") or "").strip()
            if "://" not in uri:
                continue
            scheme, _, remainder = uri.partition("://")
            prefixes.add(f"{scheme}://")
            if remainder and "{" not in remainder:
                prefixes.add(uri)
        return prefixes

    @staticmethod
    def _require_payload_field(
        payload: Mapping[str, Any],
        field_name: str,
        *,
        aliases: tuple[str, ...] = (),
    ) -> str:
        for candidate in (field_name, *aliases):
            value = payload.get(candidate)
            if value not in (None, ""):
                return str(value)
        raise KeyError(f"Missing required field: {field_name}")

    def _get_tools(self):
        if self._tools is None:
            from tldw_chatbook.MCP.tools import MCPTools

            self._tools = MCPTools(
                self._require_chachanotes_db(), self._require_media_db()
            )
        return self._tools

    def _get_library_service(self):
        """Lazily compose the shared Library tool service (built once, cached).

        Uses the server module's single service construction site
        (``server.build_local_library_tool_service``) so direct Library
        execution and standalone-server backend wiring cannot drift.
        """
        if self._library_service is None:
            from .server import build_local_library_tool_service

            self._library_service = build_local_library_tool_service(
                chachanotes_db=self._require_chachanotes_db(),
                media_db=self._require_media_db(),
                policy_enforcer=self._policy_enforcer,
            )
        return self._library_service

    def _get_resources(self):
        if self._resources is None:
            from tldw_chatbook.MCP.resources import MCPResources

            self._resources = MCPResources(
                self._require_chachanotes_db(), self._require_media_db()
            )
        return self._resources

    def _get_prompts(self):
        if self._prompts is None:
            from tldw_chatbook.MCP.prompts import MCPPrompts

            self._prompts = MCPPrompts(
                self._require_chachanotes_db(), self._require_media_db()
            )
        return self._prompts
