from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from tldw_chatbook.config import get_chachanotes_db_lazy, get_media_db_lazy

from .server import MCP_AVAILABLE, describe_local_mcp_capabilities


class LocalMCPRuntimeDelegate:
    """Direct local MCP runtime adapter that avoids loopback FastMCP dependency."""

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

    def __init__(self, *, manifest_provider: Callable[[], dict[str, Any]] | None = None) -> None:
        self._manifest_provider = manifest_provider or describe_local_mcp_capabilities
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
        return {
            "adapter": "direct_in_process",
            "supports_batch": True,
            "request_methods": list(self._REQUEST_METHODS),
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
                {"name": method, "supported": True}
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
                        if name not in self._UNAVAILABLE_DIRECT_TOOLS and hasattr(self, f"_tool_{name}")
                    ],
                    "unavailable": [
                        name
                        for name in tool_names
                        if name in self._UNAVAILABLE_DIRECT_TOOLS
                    ],
                    "missing": [
                        name
                        for name in tool_names
                        if name not in self._UNAVAILABLE_DIRECT_TOOLS and not hasattr(self, f"_tool_{name}")
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
                        name
                        for name in prompt_names
                        if name in self._PROMPT_NAMES
                    ],
                    "missing": [
                        name
                        for name in prompt_names
                        if name not in self._PROMPT_NAMES
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

    async def execute_tool(self, tool_name: str, arguments: Mapping[str, Any] | None = None) -> Any:
        normalized_name = str(tool_name or "").strip()
        payload = dict(arguments or {})
        handler = getattr(self, f"_tool_{normalized_name}", None)
        if handler is None:
            raise KeyError(f"Unsupported local MCP tool: {normalized_name}")
        return await handler(payload)

    async def request(self, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
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
            arguments = payload.get("arguments")
            return {
                "tool_name": self._require_payload_field(payload, "name", aliases=("tool_name",)),
                "result": await self.execute_tool(
                    self._require_payload_field(payload, "name", aliases=("tool_name",)),
                    arguments if isinstance(arguments, Mapping) else {},
                ),
            }
        if normalized_method == "resources/read":
            return {
                "resource_uri": self._require_payload_field(payload, "uri", aliases=("resource_uri",)),
                "result": await self.read_resource(
                    self._require_payload_field(payload, "uri", aliases=("resource_uri",))
                ),
            }
        if normalized_method == "prompts/get":
            arguments = payload.get("arguments")
            prompt_name = self._require_payload_field(payload, "name", aliases=("prompt_name",))
            normalized_arguments = arguments if isinstance(arguments, Mapping) else {}
            return {
                "prompt_name": prompt_name,
                "arguments": dict(normalized_arguments),
                "messages": await self.get_prompt(prompt_name, normalized_arguments),
            }
        raise KeyError(f"Unsupported local MCP runtime method: {normalized_method}")

    async def batch(self, requests: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for index, request in enumerate(requests):
            method = str(request.get("method") or "").strip() if isinstance(request, Mapping) else ""
            params = request.get("params") if isinstance(request, Mapping) else None
            try:
                await self.request(method, params if isinstance(params, Mapping) else {})
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
            return await resources.get_conversation_resource(normalized_uri.removeprefix("conversation://"))
        if normalized_uri.startswith("note://"):
            return await resources.get_note_resource(normalized_uri.removeprefix("note://"))
        if normalized_uri.startswith("character://"):
            return await resources.get_character_resource(normalized_uri.removeprefix("character://"))
        if normalized_uri.startswith("media://"):
            return await resources.get_media_resource(normalized_uri.removeprefix("media://"))
        if normalized_uri.startswith("rag-chunk://"):
            return await resources.get_rag_chunk_resource(normalized_uri.removeprefix("rag-chunk://"))
        raise KeyError(f"Unsupported local MCP resource URI: {normalized_uri}")

    async def get_prompt(self, prompt_name: str, arguments: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
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
            result["warning"] = "Local MCP create_note currently persists title/content only."
        return result

    async def _tool_search_notes(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        db = self._require_chachanotes_db()
        results = db.search_notes(
            search_term=str(arguments.get("query") or ""),
            limit=int(arguments.get("limit", 10)),
        )
        return [
            {
                "id": item["id"],
                "title": item["title"],
                "preview": item["content"][:200] + "..." if len(item["content"]) > 200 else item["content"],
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

    async def _tool_ingest_media(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not arguments.get("url") and not arguments.get("file_path"):
            return {"error": "Either url or file_path must be provided"}
        return {
            "status": "queued",
            "media_id": "placeholder_id",
            "message": "Media ingestion queued",
        }

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

            self._tools = MCPTools(self._require_chachanotes_db(), self._require_media_db())
        return self._tools

    def _get_resources(self):
        if self._resources is None:
            from tldw_chatbook.MCP.resources import MCPResources

            self._resources = MCPResources(self._require_chachanotes_db(), self._require_media_db())
        return self._resources

    def _get_prompts(self):
        if self._prompts is None:
            from tldw_chatbook.MCP.prompts import MCPPrompts

            self._prompts = MCPPrompts(self._require_chachanotes_db(), self._require_media_db())
        return self._prompts
