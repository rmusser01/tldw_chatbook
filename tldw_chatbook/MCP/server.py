from __future__ import annotations

"""
MCP Server implementation for tldw_chatbook

This module provides the main MCP server that exposes tldw_chatbook's functionality
through the Model Context Protocol.

## Exposed local agent tools (opt-in)

When `[mcp] expose_local_tools = true` is set in config.toml, the server also
exposes workspace, web, and Watchlists agent tools (`fs_*`, `fs_patch`,
`git_*`, `web_fetch`, `web_search`, `web_crawl`, `web_deep_search`,
`watchlists_search_items`, and `watchlists_get_item`) to external MCP clients
-- `web_deep_search` is opt-in (see below). Invocation is routed through
`Agents/local_tool_provider.LocalToolProvider`'s permission gate
(`MCP/local_server_tools.py`) — never by wrapping the tool cores directly.
`web_deep_search` needs its OWN gate on top of this one: it is absent from
the provider's spec list entirely unless `[tools]
web_deep_search_enabled = true` (app restart required) — see
`Agents/local_tool_provider.py`'s `_default_specs`.

Permission model for external callers: there is no approval card outside the
Console, so tools in the `ask` state fail closed with an external-appropriate
refusal. An operator grants a tool externally by approving it "Always allow"
in a Console session (which persists `allow` + definition hash to the shared
permission store under `local:__local__`) or by editing
`<user_data_dir>/mcp_permissions.json` directly. `mutates`-tagged tools
(writes, edits, patches) are therefore effectively denied to external clients
by default. The kill switch and `deny` states are honored identically to the
Console path. The standalone server supplies no Console `SessionTodoStore`, so
`todo_create`, `todo_update`, `todo_get`, and `todo_list` are not registered;
the retired `todo_write` tool is also absent.

## Exposed local Library tools (task-1337)

The 23 descriptor-backed `library_*` tools (media/notes/prompts/skills/
conversations/collections list+get+search, plus the chunking-agent-tools
siblings: structure/chunk/spec-list/spec-save/re-chunk) are part of the
local MCP surface: they are locally served and contract-governed by
`Library/library_tool_contract.py`. The capability manifest appends them from
the descriptor table (`_describe_local_library_tools`), and the in-app direct
runtime (`local_runtime_delegate.LocalMCPRuntimeDelegate`) dispatches them to
one shared `LocalLibraryToolService` (composed by
`build_local_library_tool_service`) via `asyncio.to_thread`. The WRITING
chunk tools are service-level policy-gated: the factory threads the
runtime-policy enforcer into the chunk tool service (chunking-agent-tools
Task 5, spec §6), on top of the always-on MCP action mapping. The standalone
`TldwMCPServer` below uses `mcp-unified` and deliberately does not publish
these in-process Library tools. The Console-only
`[console].direct_library_tools` retrieval-mode toggle has no effect on this
surface.
"""

import asyncio  # noqa: E402
import ast  # noqa: E402
import copy  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Any  # noqa: E402
from datetime import datetime  # noqa: E402

# Import the standalone MCP runtime conditionally.
try:
    from mcp_unified.gateway import serve_stdio

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    serve_stdio = None  # type: ignore[assignment]

from loguru import logger  # noqa: E402

from tldw_chatbook.Library.library_tool_contract import (  # noqa: E402
    LIBRARY_TOOL_DESCRIPTORS,
)


#: Matches ``Tools/note_management_tools.py``'s ``_DEFAULT_USER_ID`` /
#: ``config.py``'s ``default_users_name_fallback``, so an unconfigured user
#: sees the same attribution the rest of the app uses.
_DEFAULT_NOTES_USER_ID = "default_user"


def _resolve_notes_user_id() -> str:
    """Return the id notes created/searched via MCP should be attributed to.

    This is an ATTRIBUTION value, not a visibility partition: the ``notes``
    table has no user column, and ``NotesInteropService`` documents that the
    ``user_id`` it's given is used as the underlying ``CharactersRAGDB``
    instance's ``client_id`` -- the column sync and conflict resolution key
    off, not a per-user filter. Mirrors
    ``Tools/note_management_tools.py::_resolve_user_id`` (same source, same
    fallback), duplicated locally rather than imported so this module keeps
    no reach-through into an unrelated tool-catalog module for one helper.

    Returns:
        The configured user id, or ``"default_user"`` if settings cannot be
        read.
    """
    from ..config import load_settings

    try:
        return load_settings().get("USERS_NAME") or _DEFAULT_NOTES_USER_ID
    except Exception as e:  # noqa: BLE001 - a tool must not crash on config
        logger.warning(f"Could not resolve USERS_NAME, using default: {e}")
        return _DEFAULT_NOTES_USER_ID


def _first_doc_line(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().splitlines()[0].strip()


def _load_server_module_ast() -> ast.Module:
    return ast.parse(Path(__file__).read_text(encoding="utf-8"))


_AST_SIMPLE_TYPES = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}


def _annotation_to_property(node: ast.expr | None) -> dict:
    """Best-effort JSON-schema fragment for one annotation AST node.

    Returns {} for anything unrecognised so the form layer falls back to
    raw JSON for that tool instead of rendering a wrong field.
    """
    if isinstance(node, ast.Name) and node.id in _AST_SIMPLE_TYPES:
        return {"type": _AST_SIMPLE_TYPES[node.id]}
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        inner = node.slice
        if node.value.id == "Optional":
            base = _annotation_to_property(inner)
            if isinstance(base.get("type"), str):
                return {**base, "type": [base["type"], "null"]}
            return {}
        if node.value.id in ("List", "list"):
            items = _annotation_to_property(inner)
            if items:
                return {"type": "array", "items": items}
    return {}


def _signature_to_input_schema(fn: ast.AsyncFunctionDef | ast.FunctionDef) -> dict:
    """Synthesize a JSON-schema ``inputSchema`` fragment from a tool function's AST signature.

    Keyword-only args (``fn.args.kwonlyargs``/``kw_defaults``) are
    intentionally unhandled: none of the nine built-in ``@self.mcp.tool()``
    registrations in this module use them (verified by reading
    ``server.py``'s ``_register_tools`` body), so there is nothing to map
    yet -- add support here if a future tool introduces one.
    """
    properties: dict = {}
    required: list[str] = []
    args = fn.args.args
    defaults: list = fn.args.defaults
    first_default_index = len(args) - len(defaults)
    for index, arg in enumerate(args):
        if arg.arg in ("self", "cls"):
            continue
        prop = _annotation_to_property(arg.annotation)
        if index >= first_default_index:
            default_node = defaults[index - first_default_index]
            try:
                default_value = ast.literal_eval(default_node)
            except (ValueError, SyntaxError):
                default_value = None
            if default_value is not None and prop:
                prop = {**prop, "default": default_value}
        else:
            required.append(arg.arg)
        properties[arg.arg] = prop
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _signature_to_prompt_arguments(
    fn: ast.AsyncFunctionDef | ast.FunctionDef,
) -> list[dict[str, Any]]:
    """Describe prompt arguments without exposing Python annotations."""
    positional = [*fn.args.posonlyargs, *fn.args.args]
    first_default_index = len(positional) - len(fn.args.defaults)
    arguments = [
        {"name": arg.arg, "required": index < first_default_index}
        for index, arg in enumerate(positional)
        if arg.arg not in ("self", "cls")
    ]
    arguments.extend(
        {"name": arg.arg, "required": default is None}
        for arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults)
    )
    return arguments


def _extract_registered_entries(
    method_name: str, decorator_name: str
) -> list[dict[str, Any]]:
    module_node = _load_server_module_ast()
    for node in module_node.body:
        if not isinstance(node, ast.ClassDef) or node.name != "TldwMCPServer":
            continue
        for child in node.body:
            if not isinstance(child, ast.FunctionDef) or child.name != method_name:
                continue
            entries: list[dict[str, Any]] = []
            for nested in child.body:
                if not isinstance(nested, ast.AsyncFunctionDef):
                    continue
                for decorator in nested.decorator_list:
                    if not isinstance(decorator, ast.Call):
                        continue
                    func = decorator.func
                    if (
                        not isinstance(func, ast.Attribute)
                        or func.attr != decorator_name
                    ):
                        continue
                    entry: dict[str, Any] = {
                        "name": nested.name,
                        "description": _first_doc_line(ast.get_docstring(nested)),
                    }
                    if decorator_name == "resource" and decorator.args:
                        first_arg = decorator.args[0]
                        if isinstance(first_arg, ast.Constant) and isinstance(
                            first_arg.value, str
                        ):
                            entry["uri"] = first_arg.value
                    if decorator_name == "tool":
                        entry["inputSchema"] = _signature_to_input_schema(nested)
                    elif decorator_name == "prompt":
                        entry["arguments"] = _signature_to_prompt_arguments(nested)
                    entries.append(entry)
                    break
            return entries
    return []


def _describe_local_resources() -> list[dict[str, Any]]:
    return _extract_registered_entries("_register_resources", "resource")


def _describe_local_prompts() -> list[dict[str, Any]]:
    return _extract_registered_entries("_register_prompts", "prompt")


def _describe_local_tools() -> list[dict[str, Any]]:
    return _extract_registered_entries("_register_tools", "tool")


def _describe_local_library_tools() -> list[dict[str, Any]]:
    """Manifest entries for the descriptor-backed Library tools (task-1337).

    Derived from ``LIBRARY_TOOL_DESCRIPTORS`` -- never hand-maintained here --
    so the local MCP capability manifest can never drift from the contract the
    Console provider exposes. ``inputSchema`` is deep-copied so manifest
    consumers can never mutate the shared descriptor table.
    """
    return [
        {
            "name": descriptor.name,
            "description": descriptor.description,
            "inputSchema": copy.deepcopy(descriptor.input_schema),
        }
        for descriptor in LIBRARY_TOOL_DESCRIPTORS.values()
    ]


def describe_local_mcp_capabilities() -> dict[str, Any]:
    """Return a stable local MCP capability manifest without opening a loopback connection."""
    return {
        "server_id": "local:tldw_chatbook",
        "server_label": "tldw_chatbook local MCP",
        "tools": _describe_local_tools() + _describe_local_library_tools(),
        "resources": _describe_local_resources(),
        "prompts": _describe_local_prompts(),
    }


def build_local_library_tool_service(
    *,
    chachanotes_db: Any,
    media_db: Any,
    notes_service: Any = None,
    notes_scope_service: Any = None,
    policy_enforcer: Any = None,
) -> Any:
    """Compose the six local Library backends into one shared synchronous service.

    Single construction site for ``LocalLibraryToolService`` on the local MCP
    surface (task-1337, plan Task 9): ``LocalMCPRuntimeDelegate`` calls this
    lazily on first Library dispatch, so every in-process consumer of the
    direct runtime shares identical backend wiring. The ``mcp-unified``
    standalone ``TldwMCPServer`` deliberately does NOT use this; Library
    tools remain available only through the in-process direct runtime.

    Every backend is best-effort: a construction failure degrades that item
    type's tools to the service's structured ``feature_unavailable`` payload
    instead of sinking the MCP surface. The skills backend is built WITHOUT a
    trust service, which fails closed by design: skill list/search expose
    safe metadata only and skill bodies/supporting files stay restricted.

    Args:
        chachanotes_db: Open ``CharactersRAGDB`` handle (conversations, and
            the notes backend's ``global_db_to_use``).
        media_db: Open ``MediaDatabase`` handle.
        notes_service: Optional pre-built ``NotesInteropService``; when
            omitted, one is constructed with the canonical signature off
            ``get_chachanotes_db_path()`` and ``chachanotes_db``.
        notes_scope_service: Optional pre-built ``NotesScopeService``
            (student-workflow spec §4.3): the note-save tool's folder seam.
            When omitted, one is composed over ``chachanotes_db`` with the
            app builder's own shape (shared local folder repository); a
            construction failure degrades folder requests to
            ``feature_unavailable`` rather than sinking the surface.
        policy_enforcer: Optional runtime-policy enforcer
            (``require_allowed(action_id=...)`` seam) threaded into the
            media chunk tool service and the note-save path, whose WRITING
            tools (``library_save_chunk_spec``, ``library_rechunk_media``,
            ``library_save_note``) are service-level gated
            (chunking-agent-tools Tasks 4-5 + student-workflow Task 1,
            spec §6); ``None`` leaves the always-on MCP action mapping as
            the outer gate.

    Returns:
        The shared ``LocalLibraryToolService``.
    """
    from ..config import (
        CLI_APP_CLIENT_ID,
        get_chachanotes_db_path,
        get_library_collections_db_path,
        get_user_data_dir,
    )
    from ..Library.local_library_tool_service import LocalLibraryToolService

    backends: dict[str, Any] = {}

    def _build(key: str, builder) -> None:
        try:
            backends[key] = builder()
        except Exception:  # noqa: BLE001 - degrade, never sink the surface
            logger.exception(
                f"Local Library {key} backend unavailable; "
                "its tools will report feature_unavailable"
            )
            backends[key] = None

    if notes_service is not None:
        backends["note"] = notes_service
    else:

        def _build_notes():
            from ..Notes.Notes_Library import NotesInteropService

            return NotesInteropService(
                base_db_directory=get_chachanotes_db_path().parent,
                api_client_id=CLI_APP_CLIENT_ID,
                global_db_to_use=chachanotes_db,
            )

        _build("note", _build_notes)

    def _build_media():
        from ..Media.local_media_reading_service import LocalMediaReadingService

        return LocalMediaReadingService(media_db)

    _build("media", _build_media)

    def _build_prompts():
        from ..Prompt_Management.local_prompt_service import LocalPromptService

        return LocalPromptService()

    _build("prompt", _build_prompts)

    def _build_skills():
        from ..Skills_Interop.local_skills_service import (
            LocalSkillsService,
            default_local_skills_store_dir,
        )

        return LocalSkillsService(
            store_dir=default_local_skills_store_dir(get_user_data_dir())
        )

    _build("skill", _build_skills)

    def _build_conversations():
        from ..Chat.chat_conversation_service import ChatConversationService

        return ChatConversationService(chachanotes_db)

    _build("conversation", _build_conversations)

    def _build_collections():
        from ..DB.Library_Collections_DB import LibraryCollectionsDB
        from ..Library.library_collections_service import (
            LocalLibraryCollectionsService,
        )

        return LocalLibraryCollectionsService(
            LibraryCollectionsDB(get_library_collections_db_path(), CLI_APP_CLIENT_ID)
        )

    _build("collection", _build_collections)

    def _build_media_chunk():
        from ..Chunking.chunking_interop_library import get_chunking_service
        from ..Library.local_media_chunk_tool_service import (
            LocalMediaChunkToolService,
        )

        return LocalMediaChunkToolService(
            media_db,
            backends["media"],
            template_interop=get_chunking_service(media_db),
            # chunking-agent-tools (Task 5, spec §6): the writing chunk
            # tools are service-level gated here too -- the delegate
            # threads the runtime-policy enforcer through (the Console
            # construction site passes the same app handle).
            policy_enforcer=policy_enforcer,
        )

    _build("media_chunk", _build_media_chunk)

    if notes_scope_service is not None:
        backends["notes_scope"] = notes_scope_service
    else:

        def _build_notes_scope():
            # student-workflow (spec §4.3): the note-save folder seam, built
            # with the app builder's own shape -- the scope facade over one
            # shared local folder repository (the notes UI's own scope, so
            # folders saved here are visible there).
            from ..Notes.note_folder_repository import LocalNoteFolderRepository
            from ..Notes.notes_scope_service import NotesScopeService

            return NotesScopeService(
                local_notes_service=backends["note"],
                server_service=None,
                folder_repository=LocalNoteFolderRepository(chachanotes_db),
            )

        _build("notes_scope", _build_notes_scope)

    return LocalLibraryToolService(
        media_service=backends["media"],
        notes_service=backends["note"],
        prompt_service=backends["prompt"],
        skills_service=backends["skill"],
        conversation_service=backends["conversation"],
        collections_service=backends["collection"],
        media_chunk_service=backends["media_chunk"],
        # student-workflow (spec §4.3/§6): the note-save folder seam and the
        # writing note tool's service-level gate (the chunk-tools pattern).
        notes_scope_service=backends.get("notes_scope"),
        policy_enforcer=policy_enforcer,
    )


class TldwMCPServer:
    """MCP Server for tldw_chatbook"""

    def __init__(self, name: str = "tldw_chatbook", version: str = "0.1.0"):
        """Initialize the MCP server."""
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP dependencies not available. Install with: pip install tldw-chatbook[mcp]"
            )

        self.name = name
        self.version = version
        # Defer this import: local-tool provider modules refer back to server helpers.
        from .gateway_runtime import ChatbookGatewayRuntime

        self.mcp = ChatbookGatewayRuntime(
            name=name,
            version=version,
            tool_descriptors=_describe_local_tools(),
        )

        # Initialize databases
        self._init_databases()

        # Initialize MCP components
        from .tools import MCPTools
        from .resources import MCPResources
        from .prompts import MCPPrompts

        self.tools = MCPTools(self.chachanotes_db, self.media_db)
        self.resources = MCPResources(self.chachanotes_db, self.media_db)
        self.prompts = MCPPrompts(self.chachanotes_db, self.media_db)

        # Register tools, resources, and prompts
        self._register_tools()
        self._register_resources()
        self._register_prompts()
        self._register_local_agent_tools()
        self.mcp.finalize()

        logger.info(f"MCP Server '{name}' initialized")

    def _init_databases(self):
        """Initialize database connections."""
        try:
            from ..config import (
                get_chachanotes_db_path,
                get_media_db_path,
                CLI_APP_CLIENT_ID,
            )
            from ..DB.ChaChaNotes_DB import CharactersRAGDB
            from ..DB.Client_Media_DB_v2 import MediaDatabase
            from ..Notes.Notes_Library import NotesInteropService

            # Initialize character/chat/notes database
            self.chachanotes_db = CharactersRAGDB(
                db_path=get_chachanotes_db_path(), client_id=CLI_APP_CLIENT_ID
            )

            # Initialize media database. Uses the same resolver the rest of
            # the app opens the media DB through -- ``get_cli_setting("database",
            # "media_db", ...)`` used to read a config key ("media_db") that
            # was never declared anywhere; the real key is "media_db_path"
            # and get_media_db_path() is its accessor (see TASK-854).
            media_db_path = get_media_db_path()
            self.media_db = MediaDatabase(
                db_path=media_db_path, client_id=CLI_APP_CLIENT_ID
            )

            # Initialize services. There is deliberately no
            # ``self.character_service`` here: this used to construct a
            # ``CharacterInteropService`` that does not exist anywhere in the
            # codebase (Character_Chat_Lib.py is a free-function module, not
            # a service class), so the server could never actually be
            # constructed (see TASK-968). The attribute was never read by
            # anything else in this module either -- the character-related
            # tools below (``chat_with_character``, ``list_characters``)
            # already go through ``self.tools``, which reads character rows
            # directly off ``self.chachanotes_db``
            # (``get_character_card_by_id`` / ``list_character_cards``) --
            # so the dead reference was removed rather than resolved to a
            # real service.
            # ``NotesInteropService.__init__`` takes (base_db_directory,
            # api_client_id, global_db_to_use=None) -- it used to be
            # constructed here with a single positional arg
            # (``self.chachanotes_db``, a ``CharactersRAGDB``, bound to the
            # ``base_db_directory: Union[str, Path]`` parameter), so every
            # construction raised before the server could open a single
            # database connection (TASK-983). Mirrors the already-working
            # construction in ``Tools/note_management_tools.py`` and
            # ``app.py``: the *parent directory* of the unified DB file as
            # ``base_db_directory`` (used only to verify the directory is
            # trusted), this server's own app-wide client id as
            # ``api_client_id``, and the already-open ``self.chachanotes_db``
            # handle as ``global_db_to_use`` so the service reuses this
            # process's connection instead of opening a second one.
            self.notes_service = NotesInteropService(
                base_db_directory=get_chachanotes_db_path().parent,
                api_client_id=CLI_APP_CLIENT_ID,
                global_db_to_use=self.chachanotes_db,
            )

            logger.info("Databases initialized successfully")
        except Exception:
            logger.bind(operation="initialize_standalone_mcp_databases").error(
                "Standalone MCP database initialization failed."
            )
            raise

    def _register_tools(self):
        """Register MCP tools."""
        from ..config import get_api_key
        from ..Chat.Chat_Functions import chat_api_call, extract_response_content

        # Basic chat tool
        @self.mcp.tool()
        async def chat_with_llm(
            message: str,
            provider: str = "openai",
            model: Optional[str] = None,
            system_prompt: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 4096,
            conversation_id: Optional[int] = None,
        ) -> Dict[str, Any]:
            """Send a message to an LLM and get a response."""
            # For basic chat, we'll implement directly here
            try:
                # get_api_key() is the declared accessor: it checks the
                # newer api_settings.<provider> structure, then the legacy
                # [API] section, then a bare env var -- a direct
                # get_cli_setting("API", ...) lookup only covers the middle
                # tier and silently misses a key configured either of the
                # other two ways.
                api_key = get_api_key(provider)
                if not api_key:
                    return {"error": f"No API key configured for {provider}"}

                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": message})

                raw = await asyncio.to_thread(
                    chat_api_call,
                    api_endpoint=provider,
                    messages_payload=messages,
                    api_key=api_key,
                    model=model,
                    temp=temperature,
                    max_tokens=max_tokens,
                    streaming=False,
                )
                response = extract_response_content(raw)

                return {
                    "response": response,
                    "conversation_id": conversation_id or "new_conversation",
                }
            except Exception as e:
                logger.error(f"Error in chat_with_llm: {e}")
                return {"error": str(e)}

        # Character chat tool
        @self.mcp.tool()
        async def chat_with_character(
            message: str,
            character_id: int,
            provider: str = "openai",
            model: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 4096,
            conversation_id: Optional[int] = None,
        ) -> Dict[str, Any]:
            """Chat with a specific character."""
            return await self.tools.chat_with_character(
                message=message,
                character_id=character_id,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                conversation_id=conversation_id,
            )

        # RAG search tool
        @self.mcp.tool()
        async def search_rag(
            query: str,
            limit: int = 10,
            media_types: Optional[List[str]] = None,
            use_semantic: bool = True,
        ) -> List[Dict[str, Any]]:
            """Search the RAG database for relevant content."""
            return await self.tools.perform_rag_search(
                query=query,
                limit=limit,
                media_types=media_types,
                use_semantic=use_semantic,
            )

        # Search conversations tool
        @self.mcp.tool()
        async def search_conversations(
            query: str, limit: int = 10, character_id: Optional[int] = None
        ) -> List[Dict[str, Any]]:
            """Search conversations by message content.

            Each result's ``preview`` is the best-matching message's text in
            that conversation (there is no conversation-level content field
            to preview from), truncated to 200 characters -- see
            ``MCPTools.search_conversations`` for the full rationale
            (TASK-985).
            """
            return await self.tools.search_conversations(
                query=query, limit=limit, character_id=character_id
            )

        # Note creation tool
        @self.mcp.tool()
        async def create_note(
            title: str,
            content: str,
        ) -> Dict[str, Any]:
            """Create a new note.

            ``tags`` and ``template`` were dropped from this tool's
            signature (TASK-983): ``NotesInteropService.add_note`` -- the
            real method, ``create_note`` does not exist on the class -- has
            no such parameters, and the ``notes`` table it wraps has no
            tags column and no template concept at all. Keyword/tag linking
            is a separate, unrelated API (``link_note_to_keyword``) this
            tool never actually called even before the fix.
            """
            try:
                note_id = await asyncio.to_thread(
                    self.notes_service.add_note,
                    user_id=_resolve_notes_user_id(),
                    title=title,
                    content=content,
                )
                return {
                    "id": note_id,
                    "title": title,
                    "created": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.error(f"Error in create_note: {e}")
                return {"error": str(e)}

        # Note search tool
        @self.mcp.tool()
        async def search_notes(query: str, limit: int = 10) -> List[Dict[str, Any]]:
            """Search notes by content or title."""
            try:
                # NotesInteropService.search_notes(user_id, search_term, limit)
                # returns a list of plain dicts (the notes table's own
                # columns: id/title/content/created_at/last_modified), not
                # objects with attribute access -- this used to call
                # ``query=``/``limit=`` with no ``user_id`` at all (a
                # TypeError on the real signature) and then read
                # ``note.id``/``note.updated_at`` off the result, an
                # AttributeError on a dict either way. ``last_modified`` is
                # the real column name; the notes table has no
                # ``updated_at`` column (TASK-983).
                results = await asyncio.to_thread(
                    self.notes_service.search_notes,
                    user_id=_resolve_notes_user_id(),
                    search_term=query,
                    limit=limit,
                )
                return [
                    {
                        "id": note.get("id"),
                        "title": note.get("title"),
                        "preview": note.get("content", "")[:200] + "..."
                        if len(note.get("content", "")) > 200
                        else note.get("content", ""),
                        "created": note.get("created_at"),
                        "modified": note.get("last_modified"),
                    }
                    for note in results
                ]
            except Exception as e:
                logger.error(f"Error in search_notes: {e}")
                return [{"error": str(e)}]

        # List characters tool
        @self.mcp.tool()
        async def list_characters() -> List[Dict[str, Any]]:
            """List all available characters."""
            return await self.tools.list_available_characters()

        # Get conversation history tool
        @self.mcp.tool()
        async def get_conversation_history(
            conversation_id: int, limit: Optional[int] = None
        ) -> Dict[str, Any]:
            """Get conversation history."""
            return await self.tools.get_conversation_history(
                conversation_id=conversation_id, limit=limit
            )

        # Export conversation tool
        @self.mcp.tool()
        async def export_conversation(
            conversation_id: int, format: str = "markdown"
        ) -> Dict[str, Any]:
            """Export a conversation in various formats."""
            return await self.tools.export_conversation(
                conversation_id=conversation_id, format=format
            )

    def _register_resources(self):
        """Register MCP resources."""
        from urllib.parse import quote

        def json_metadata(
            resource: Dict[str, Any], scheme: str, identifier: str
        ) -> Dict[str, Any]:
            """Normalize trusted legacy URI spelling and SQLite metadata."""

            def normalize(value: Any) -> Any:
                if isinstance(value, datetime):
                    return value.isoformat()
                if isinstance(value, dict):
                    return {key: normalize(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [normalize(item) for item in value]
                return value

            expected_legacy_uri = f"{scheme}://{identifier}"
            if resource.get("uri") == expected_legacy_uri:
                canonical_identifier = quote(
                    identifier,
                    safe="-._~",
                )
                resource = {
                    **resource,
                    "uri": f"{scheme}://{canonical_identifier}",
                }
            metadata = resource.get("metadata")
            return (
                {**resource, "metadata": normalize(metadata)}
                if isinstance(metadata, dict)
                else resource
            )

        @self.mcp.resource("conversation://{conversation_id}")
        async def get_conversation(conversation_id: str) -> Dict[str, Any]:
            """Get a conversation by ID."""
            return json_metadata(
                await self.resources.get_conversation_resource(conversation_id),
                "conversation",
                conversation_id,
            )

        @self.mcp.resource("note://{note_id}")
        async def get_note(note_id: str) -> Dict[str, Any]:
            """Get a note by ID."""
            return json_metadata(
                await self.resources.get_note_resource(note_id), "note", note_id
            )

        @self.mcp.resource("character://{character_id}")
        async def get_character(character_id: str) -> Dict[str, Any]:
            """Get a character profile by ID."""
            return json_metadata(
                await self.resources.get_character_resource(character_id),
                "character",
                character_id,
            )

        @self.mcp.resource("media://{media_id}")
        async def get_media(media_id: str) -> Dict[str, Any]:
            """Get media content by ID."""
            return json_metadata(
                await self.resources.get_media_resource(media_id), "media", media_id
            )

        @self.mcp.resource("rag-chunk://{chunk_uuid}")
        async def get_rag_chunk(chunk_uuid: str) -> Dict[str, Any]:
            """Get a RAG chunk by UUID.

            Keyed on the chunk's UUID (`UnvectorizedMediaChunks.uuid`), not
            an integer id -- see `MCPResources.get_rag_chunk_resource` for
            why (TASK-985).
            """
            return json_metadata(
                await self.resources.get_rag_chunk_resource(chunk_uuid),
                "rag-chunk",
                chunk_uuid,
            )

        # List resources
        @self.mcp.list_resources()
        async def list_resources() -> List[Dict[str, Any]]:
            """List available resources."""
            resources = []

            # Add recent conversations
            recent_convs = await self.resources.list_recent_conversations(limit=5)
            resources.extend(recent_convs)

            # Add recent notes
            recent_notes = await self.resources.list_recent_notes(limit=5)
            resources.extend(recent_notes)

            return resources

    def _register_prompts(self):
        """Register MCP prompts."""

        @self.mcp.prompt()
        async def summarize_conversation(
            conversation_id: int, style: str = "concise", focus: Optional[str] = None
        ) -> List[Dict[str, str]]:
            """Generate a prompt to summarize a conversation."""
            return await self.prompts.summarize_conversation_prompt(
                conversation_id=conversation_id, style=style, focus=focus
            )

        @self.mcp.prompt()
        async def generate_document(
            conversation_id: int, doc_type: str = "summary", format: str = "markdown"
        ) -> List[Dict[str, str]]:
            """Generate a prompt to create a document from a conversation."""
            return await self.prompts.generate_document_prompt(
                conversation_id=conversation_id, doc_type=doc_type, format=format
            )

        @self.mcp.prompt()
        async def analyze_media(
            media_id: int, analysis_type: str = "summary", detail_level: str = "medium"
        ) -> List[Dict[str, str]]:
            """Generate a prompt to analyze ingested media."""
            return await self.prompts.analyze_media_prompt(
                media_id=media_id,
                analysis_type=analysis_type,
                detail_level=detail_level,
            )

        @self.mcp.prompt()
        async def search_and_synthesize(
            query: str, num_sources: int = 5, synthesis_type: str = "overview"
        ) -> List[Dict[str, str]]:
            """Generate a prompt to search RAG and synthesize results."""
            return await self.prompts.search_and_synthesize_prompt(
                query=query, num_sources=num_sources, synthesis_type=synthesis_type
            )

        @self.mcp.prompt()
        async def character_writing(
            character_id: int,
            writing_type: str = "response",
            context: Optional[str] = None,
            style_notes: Optional[str] = None,
        ) -> List[Dict[str, str]]:
            """Generate a prompt for character-based writing."""
            return await self.prompts.character_writing_prompt(
                character_id=character_id,
                writing_type=writing_type,
                context=context,
                style_notes=style_notes,
            )

    def _register_local_agent_tools(self):
        """Register workspace, web, and Watchlists agent tools when enabled.

        Gated behind ``[mcp] expose_local_tools`` (default false); a no-op
        when the flag is off. Called from ``__init__`` -- deliberately NOT
        part of ``_register_tools`` so the AST-walking
        ``_extract_registered_entries`` capability catalog stays unaffected.

        Every call is routed through the composed ``LocalToolProvider``'s
        permission gate (fresh ``MCPPermissionStore`` state per call, kill
        switch honored, ask fails closed -- external clients cannot
        approve). The store path is pinned to
        ``get_user_data_dir() / "mcp_permissions.json"`` so Console
        "Always allow" grants apply here.

        The provider's exact JSON schemas and ``ToolResult`` handlers are
        staged together and published only after the full set validates.
        """
        from ..config import get_user_data_dir
        from .local_server_tools import (
            _local_agent_tool_registrations,
            build_server_local_provider,
            local_tools_exposure_enabled,
            resolve_server_workspace_root,
        )
        from .permission_store import MCPPermissionStore

        # The gate lives in local_server_tools (server.py never calls
        # get_cli_setting directly — an AST test pins that).
        if not local_tools_exposure_enabled():
            return

        # Guard the whole flag-on body: a failure here must never cost the
        # operator the built-in tools — log and start without local tools.
        try:
            workspace_root = resolve_server_workspace_root()
            store = MCPPermissionStore(get_user_data_dir() / "mcp_permissions.json")
            provider = build_server_local_provider(workspace_root, store)

            registrations = _local_agent_tool_registrations(provider)
            self.mcp.register_local_tools(registrations)
        except Exception:  # noqa: BLE001 — never sink the whole server for this
            import sys

            print(
                "Local MCP tools unavailable; continuing with built-in tools.",
                file=sys.stderr,
            )

    async def run(self, transport: str = "stdio") -> int:
        """Run the MCP server.

        Args:
            transport: Transport name; only ``stdio`` is supported.
        """
        if transport != "stdio":
            raise NotImplementedError("Only stdio transport is supported")
        if serve_stdio is None:
            raise RuntimeError("MCP stdio runtime is unavailable")
        return await serve_stdio(self.mcp)


async def main() -> int:
    """Main entry point for running the MCP server."""
    return await TldwMCPServer().run("stdio")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
