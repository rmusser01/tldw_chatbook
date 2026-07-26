"""
Note Management Tools for LLM function calling.

These tools allow LLMs to create, search, update, and manage notes.
"""

from typing import Dict, Any
from loguru import logger

from . import Tool
from ..Notes.Notes_Library import NotesInteropService
from ..config import get_chachanotes_db_path, load_settings

# Base directory for the notes DB, mirroring the app's own NotesInteropService
# wiring in app.py (`get_chachanotes_db_path().parent`). The previous
# `from ..config import USER_DB_BASE_DIR` imported a name that only exists
# inside the CONFIG_TOML_CONTENT string literal, not as a real module-level
# constant, so this module could never actually be imported.
USER_DB_BASE_DIR = get_chachanotes_db_path().parent

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


class CreateNoteTool(Tool):
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
            from ..config import chachanotes_db

            notes_service = NotesInteropService(
                base_db_directory=USER_DB_BASE_DIR,
                api_client_id="tool_executor",
                global_db_to_use=chachanotes_db,
            )

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


class SearchNotesTool(Tool):
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
            from ..config import chachanotes_db

            notes_service = NotesInteropService(
                base_db_directory=USER_DB_BASE_DIR,
                api_client_id="tool_executor",
                global_db_to_use=chachanotes_db,
            )

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


class UpdateNoteTool(Tool):
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
            from ..config import chachanotes_db

            notes_service = NotesInteropService(
                base_db_directory=USER_DB_BASE_DIR,
                api_client_id="tool_executor",
                global_db_to_use=chachanotes_db,
            )

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
