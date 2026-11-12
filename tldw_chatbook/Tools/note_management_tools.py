"""
Note Management Tools for LLM function calling.

These tools allow LLMs to create, search, update, and manage notes.
"""

import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from loguru import logger

from . import Tool
from ..Notes.Notes_Library import NotesInteropService
from ..config import load_settings


#: Cached ``NotesInteropService`` for the tool path (task-692).
#:
#: Each note tool used to build its own service per ``execute()``. That is
#: not just object churn: the service's ``_db_instances`` cache is an
#: INSTANCE attribute, so a per-call service always missed it and
#: ``_get_db`` constructed a fresh ``CharactersRAGDB`` -- a real DB open
#: plus schema-version check (~1.8 ms measured on an existing DB) -- on
#: every single call, and the instance was then discarded. ``TASK-545 P2``
#: put these tools on the agent worker thread, where one run can invoke
#: them many times.
#:
#: Keyed on the supplied DB (path and identity) so a config change (or a test that
#: repoints the data dir) still rebuilds rather than serving a service
#: bound to the previous database.
_SERVICE_LOCK = threading.Lock()
# ponytail: holds (db, service) strongly -- at most ONE DB, the last one
# supplied, stays alive until a different DB is supplied. CharactersRAGDB has
# no cheap closed check, so a closed DB is not detected. Upgrade path: an
# is_closed flag on the DB (then rebuild on it), or a per-DB cache.
_SERVICE_CACHE: Optional[Tuple[Any, NotesInteropService]] = None


def _default_db_provider() -> Any:
    """Self-initializing ChaChaNotes getter, resolved at call time."""
    from ..config import get_chachanotes_db_lazy

    return get_chachanotes_db_lazy()


def _notes_service(db_provider: Callable[[], Any]) -> NotesInteropService:
    """Return the shared notes service for the tool path.

    Built lazily on first use and reused afterwards, so the service's own
    per-user ``CharactersRAGDB`` cache survives across tool calls instead
    of being rebuilt for each one. The database comes from ``db_provider``
    (supplied by whoever registered the tool), never from a config global.

    Args:
        db_provider: Zero-argument callable returning the ChaChaNotes DB.

    Returns:
        A ``NotesInteropService`` bound to that DB.
    """
    global _SERVICE_CACHE

    db = db_provider()
    if db is None:
        raise RuntimeError("The notes database is unavailable.")

    def _fresh(cached) -> bool:
        return cached is not None and cached[0] is db

    cached = _SERVICE_CACHE
    if _fresh(cached):
        return cached[1]

    with _SERVICE_LOCK:
        cached = _SERVICE_CACHE
        if _fresh(cached):
            return cached[1]
        service = NotesInteropService(
            base_db_directory=Path(db.db_path_str).parent,
            api_client_id="tool_executor",
            global_db_to_use=db,
        )
        _SERVICE_CACHE = (db, service)
        return service


def _reset_notes_service_cache() -> None:
    """Drop the cached service (tests, and any deliberate re-point)."""
    global _SERVICE_CACHE
    with _SERVICE_LOCK:
        _SERVICE_CACHE = None


#: Matches config.py's own `default_users_name_fallback`, so an
#: unconfigured user sees no change from the previously hardcoded value.
_DEFAULT_USER_ID = "default_user"


def _resolve_user_id() -> str:
    """Return the id notes should be attributed to.

    This is an ATTRIBUTION value, not a visibility partition: the
    ``notes`` table has no user column, and ``NotesInteropService.add_note``
    documents that "the user_id will be used as the client_id" -- the
    column sync and conflict resolution key off. Notes written under the
    wrong id are still visible; they are misattributed.

    Reads ``load_settings()["USERS_NAME"]`` -- the SAME source
    ``app.notes_user_id`` comes from (``app.py:3140``). Deliberately not
    ``get_cli_setting("general", "users_name", ...)``: the real value is
    ``os.getenv("USERS_NAME", <toml value>)`` resolved inside
    ``load_settings`` (``config.py:826``), so a direct TOML read would
    diverge from the app whenever the env var is set and stamp a third
    distinct client_id.

    Resolved per call rather than at construction time: a
    ``BuiltinToolProvider`` is built from four sites, two of which have no
    app access, and the tool classes take no constructor arguments. Note
    this means a mid-session change to ``users_name`` takes effect here
    while ``app.notes_user_id`` (bound once at init) keeps the old value.

    Returns:
        The configured user id, or ``"default_user"`` if settings cannot
        be read.
    """
    try:
        return load_settings().get("USERS_NAME") or _DEFAULT_USER_ID
    except Exception as e:  # noqa: BLE001 — a tool must not crash on config
        logger.warning(f"Could not resolve USERS_NAME, using default: {e}")
        return _DEFAULT_USER_ID


class _NoteTool(Tool):
    """Base for the note tools: holds the host-supplied database provider."""

    #: Read by ``build_gateable_tool`` to decide whether to pass a provider.
    accepts_notes_db_provider = True

    def __init__(self, db_provider: Optional[Callable[[], Any]] = None) -> None:
        """Bind the tool to a database source.

        Args:
            db_provider: Zero-argument callable returning the ChaChaNotes DB.
                Defaults to the self-initializing config getter.
        """
        self._db_provider = db_provider or _default_db_provider


class CreateNoteTool(_NoteTool):
    """Tool for creating new notes."""

    @property
    def name(self) -> str:
        return "create_note"

    @property
    def description(self) -> str:
        return "Create a new note with a title and content. Returns the note ID."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The title of the note"},
                "content": {"type": "string", "description": "The content of the note"},
            },
            "required": ["title", "content"],
        }

    @property
    def risk_tags(self) -> tuple[str, ...]:
        """Inserts a note row into the user's database."""
        return ("mutates",)

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Create a new note.

        Args:
            title: Note title
            content: Note content

        Returns:
            Dictionary with note ID or error
        """
        title = kwargs.get("title")
        content = kwargs.get("content")

        if not title:
            return {"error": "No title provided"}
        if not content:
            return {"error": "No content provided"}

        try:
            notes_service = _notes_service(self._db_provider)

            note_id = notes_service.add_note(
                user_id=_resolve_user_id(),
                title=title,
                content=content,
            )

            logger.info(f"Created note: {note_id}")

            return {
                "note_id": note_id,
                "title": title,
                "message": "Note created successfully",
            }

        except Exception as e:
            logger.error(f"Error creating note: {e}")
            return {"error": f"Failed to create note: {str(e)}"}


class SearchNotesTool(_NoteTool):
    """Tool for searching notes."""

    @property
    def name(self) -> str:
        return "search_notes"

    @property
    def description(self) -> str:
        return "Search for notes by keyword. Returns matching notes with their titles and content snippets."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50,
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Search notes.

        Args:
            query: Search query
            limit: Max results

        Returns:
            Dictionary with search results or error
        """
        query = kwargs.get("query")
        if not query:
            return {"error": "No search query provided"}

        limit = kwargs.get("limit", 10)
        if not isinstance(limit, int) or limit < 1 or limit > 50:
            limit = 10

        try:
            # Get the notes service
            notes_service = _notes_service(self._db_provider)

            # Search notes
            results = notes_service.search_notes(
                user_id="default_user", search_term=query, limit=limit
            )

            # Format results
            formatted_results = []
            for i, note in enumerate(results):
                # Truncate content for display
                content_snippet = note.get("content", "")[:200]
                if len(note.get("content", "")) > 200:
                    content_snippet += "..."

                formatted_results.append(
                    {
                        "position": i + 1,
                        "note_id": note.get("id"),
                        "title": note.get("title", "Untitled"),
                        "content_snippet": content_snippet,
                        "created_at": note.get("created_at"),
                        "updated_at": note.get("updated_at"),
                    }
                )

            return {
                "query": query,
                "result_count": len(formatted_results),
                "results": formatted_results,
            }

        except Exception as e:
            logger.error(f"Error searching notes: {e}")
            return {"query": query, "error": f"Search failed: {str(e)}"}


class UpdateNoteTool(_NoteTool):
    """Tool for updating existing notes."""

    @property
    def name(self) -> str:
        return "update_note"

    @property
    def description(self) -> str:
        return "Update an existing note's title or content. Requires the note ID and current version."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "note_id": {
                    "type": "string",
                    "description": "The ID of the note to update",
                },
                "title": {"type": "string", "description": "New title (optional)"},
                "content": {"type": "string", "description": "New content (optional)"},
                "expected_version": {
                    "type": "integer",
                    "description": "Expected version for optimistic locking (default: 1)",
                    "default": 1,
                },
            },
            "required": ["note_id"],
        }

    @property
    def risk_tags(self) -> tuple[str, ...]:
        """Mutates an existing note the user owns."""
        return ("mutates",)

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Update a note.

        Args:
            note_id: ID of the note
            title: New title (optional)
            content: New content (optional)
            expected_version: Version for optimistic locking

        Returns:
            Dictionary with success status or error
        """
        note_id = kwargs.get("note_id")
        if not note_id:
            return {"error": "No note ID provided"}

        title = kwargs.get("title")
        content = kwargs.get("content")
        expected_version = kwargs.get("expected_version", 1)

        if not title and not content:
            return {"error": "No updates provided (need title or content)"}

        try:
            notes_service = _notes_service(self._db_provider)

            user_id = _resolve_user_id()

            # First, get the current note to check it exists
            current_note = notes_service.get_note_by_id(
                user_id=user_id, note_id=note_id
            )

            if not current_note:
                return {"note_id": note_id, "error": "Note not found"}

            # Build update data
            update_data = {}
            if title is not None:
                update_data["title"] = title
            if content is not None:
                update_data["content"] = content

            # Update the note
            success = notes_service.update_note(
                user_id=user_id,
                note_id=note_id,
                update_data=update_data,
                expected_version=expected_version,
            )

            if success:
                return {
                    "note_id": note_id,
                    "message": "Note updated successfully",
                    "updated_fields": list(update_data.keys()),
                }
            else:
                return {
                    "note_id": note_id,
                    "error": "Failed to update note (version conflict or other error)",
                }

        except Exception as e:
            logger.error(f"Error updating note {note_id}: {e}")
            return {"note_id": note_id, "error": f"Update failed: {str(e)}"}
