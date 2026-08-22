# chunking_interop_library.py
# Description: This module provides a service layer for managing chunking templates and configurations
#
# Imports
import json
import logging
import uuid as uuid_module
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

# Local Imports
from ..DB.Client_Media_DB_v2 import MediaDatabase, InputError
from ..Metrics.metrics_logger import log_counter
from .auto_selection import AUTO_SENTINEL

#######################################################################################################################
#
# Functions:

logger = logging.getLogger(__name__)


class ChunkingTemplateError(Exception):
    """Base exception for chunking template related errors."""

    pass


class TemplateNotFoundError(ChunkingTemplateError):
    """Exception raised when a template is not found."""

    pass


class BuiltinTemplateError(ChunkingTemplateError):
    """Exception raised when trying to modify or delete a builtin template."""

    pass


class InvalidTemplateError(ChunkingTemplateError):
    """Exception raised when a template body fails server-parity validation.

    Raised by ``create_template``/``update_template`` (validate-on-write,
    spec §7/AC 24): the Task-6 validator itself never raises — it returns a
    ``{"valid": False, ...}`` verdict and this error is how the CRUD layer
    refuses the write, carrying the validator's field/message summary.

    Also raised for the RESERVED sentinel name (auto-selection spec §4.3,
    ruling 8.7/AC 14): ``"auto"`` is the picker's Auto choice riding the
    ``chunk_template`` slot, so no user template may take the name —
    create and rename both refuse it with this error. A legacy row that
    already holds the name (minted before this gate) is never deleted:
    the listing decoration flags it ``name_reserved`` and auto tier 1
    skips it by name.
    """

    pass


class ChunkingInteropService:
    """Service layer for managing chunking templates and per-document
    configurations against Media DB schema v7 (task-8, spec §5.2.1).

    Column contract (v7): ``uuid`` (NOT NULL UNIQUE), ``tags`` (JSON list
    column), ``is_builtin``, ``version``, ``deleted``. Every read filters
    ``deleted = 0``; every write supplies a fresh ``uuid4`` and goes through
    ``MediaDatabase.transaction()`` — never a raw ``get_connection()`` +
    ``commit()``.

    Division of labor with ``Chunking.template_runtime`` (AC 8): THIS layer
    is full-record CRUD (fetch/mutate rows); template name→body RESOLUTION
    for runtime use lives solely in
    ``template_runtime.resolve_template``. The name-keyed queries here are
    record fetches shaped ``WHERE deleted = 0 AND name = ?`` — the
    ``WHERE name``-first fingerprint stays exclusive to genuine resolution
    sites (pinned by the resolver guard in
    ``Tests/Chunking/test_template_runtime.py``).

    Validate-on-write (AC 24): ``create_template`` and ``update_template``
    run the server-parity validator (``RAG_Admin.template_validation``) on
    the body being written and refuse it with :class:`InvalidTemplateError`.
    Stored-invalid rows (minted by the v6→v7 conversion or predating this
    gate) stay editable: update validates the NEW body only. The validator
    import is lazy per-call — module scope would be circular
    (``RAG_Admin`` → ``local_rag_admin_service`` → this module).
    """

    def __init__(self, media_db: MediaDatabase):
        """
        Initialize the chunking interop service.

        Args:
            media_db: MediaDatabase instance to use for operations
        """
        self.media_db = media_db
        logger.info("ChunkingInteropService initialized")

    # --- Template Management Methods ---

    def get_all_templates(self, include_builtin: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve all live (non-deleted) chunking templates.

        Args:
            include_builtin: Whether to include builtin templates

        Returns:
            List of template dictionaries
        """
        try:
            query = "SELECT * FROM ChunkingTemplates WHERE deleted = 0"
            if not include_builtin:
                query += " AND is_builtin = 0"
            query += " ORDER BY is_builtin DESC, name ASC"

            conn = self.media_db.get_connection()
            cursor = conn.execute(query)

            templates = [self._row_to_template_dict(row) for row in cursor]

            log_counter("chunking_templates_fetched", len(templates))
            return templates

        except Exception as e:
            logger.error(f"Error fetching templates: {e}")
            raise ChunkingTemplateError(f"Failed to fetch templates: {str(e)}")

    def get_template_by_id(self, template_id: int) -> Dict[str, Any]:
        """
        Retrieve a specific live template by ID.

        Args:
            template_id: Template ID

        Returns:
            Template dictionary

        Raises:
            TemplateNotFoundError: If no live template has this ID
        """
        try:
            conn = self.media_db.get_connection()
            row = conn.execute(
                "SELECT * FROM ChunkingTemplates WHERE deleted = 0 AND id = ?",
                (template_id,),
            ).fetchone()
            if not row:
                raise TemplateNotFoundError(f"Template with ID {template_id} not found")
            return self._row_to_template_dict(row)

        except TemplateNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error fetching template {template_id}: {e}")
            raise ChunkingTemplateError(f"Failed to fetch template: {str(e)}")

    def get_template_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a live template record by name.

        A full-record CRUD fetch, not runtime resolution — body
        interpretation lives in ``template_runtime.resolve_template``
        (see the class docstring).

        Args:
            name: Template name

        Returns:
            Template dictionary or None if no live template has this name
        """
        try:
            conn = self.media_db.get_connection()
            row = conn.execute(
                "SELECT * FROM ChunkingTemplates WHERE deleted = 0 AND name = ?",
                (name,),
            ).fetchone()
            if not row:
                return None
            return self._row_to_template_dict(row)

        except Exception as e:
            logger.error(f"Error fetching template by name '{name}': {e}")
            raise ChunkingTemplateError(f"Failed to fetch template: {str(e)}")

    def create_template(
        self,
        name: str,
        description: str,
        template_json: Union[str, Dict[str, Any]],
        tags: Optional[Sequence[str]] = None,
        is_builtin: bool = False,
    ) -> int:
        """
        Create a new chunking template (validate-on-write, AC 24).

        Args:
            name: Template name (unique among live rows — the partial
                unique index frees soft-deleted names)
            description: Template description
            template_json: Template configuration as JSON string or dict
            tags: Tags for the ``tags`` JSON column; when ``None``, a
                top-level ``tags`` list (else ``metadata.tags``) is moved
                out of the body into the column, matching the v6→v7
                conversion's placement
            is_builtin: Whether this is a builtin template

        Returns:
            ID of created template

        Raises:
            InputError: If input validation fails or the name is taken
            InvalidTemplateError: If the body fails server-parity
                validation, or the name is the reserved sentinel "auto"
                (auto-selection spec §4.3/AC 14)
            ChunkingTemplateError: If creation fails
        """
        if not name or not name.strip():
            raise InputError("Template name cannot be empty")

        if name.strip() == AUTO_SENTINEL:
            raise InvalidTemplateError(
                f"Template name '{AUTO_SENTINEL}' is reserved for the "
                "auto-selection sentinel (the picker's 'Auto' choice) and "
                "cannot be used by a template."
            )

        if not description or not description.strip():
            raise InputError("Template description cannot be empty")

        body = self._parse_template_body(template_json)
        body_tags = self._pop_body_tags(body)
        column_tags = list(tags) if tags is not None else body_tags
        # §7.1 carve-out: name/description/tags never enter the validated
        # body — validation sees only the chunking configuration.
        self._validate_body(name, body)
        template_json_str = json.dumps(body)

        try:
            with self.media_db.transaction() as conn:
                # Name liveness under the partial unique index: a
                # soft-deleted row with the same name does not block.
                existing = conn.execute(
                    "SELECT id FROM ChunkingTemplates WHERE deleted = 0 AND name = ?",
                    (name.strip(),),
                ).fetchone()
                if existing:
                    raise InputError(f"Template with name '{name}' already exists")

                cursor = conn.execute(
                    """
                    INSERT INTO ChunkingTemplates
                    (uuid, name, description, template_json, tags, is_builtin)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        str(uuid_module.uuid4()),
                        name.strip(),
                        description.strip(),
                        template_json_str,
                        json.dumps(column_tags) if column_tags is not None else None,
                        int(is_builtin),
                    ),
                )
                template_id = cursor.lastrowid

            log_counter("chunking_template_created", 1)
            logger.info(f"Created chunking template '{name}' with ID {template_id}")

            return template_id

        except InputError:
            raise
        except Exception as e:
            logger.error(f"Error creating template: {e}")
            raise ChunkingTemplateError(f"Failed to create template: {str(e)}")

    def update_template(
        self,
        template_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        template_json: Optional[Union[str, Dict[str, Any]]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Update an existing live template (validates the NEW body only, AC 24).

        A stored-invalid row stays editable: only fields supplied here are
        checked, and only the ``template_json`` being written is validated —
        the stored body is never re-validated. ``version`` increments on
        every update (AC 25); ``updated_at`` is maintained by the
        ``update_chunking_templates_timestamp`` trigger.

        Args:
            template_id: Template ID
            name: New name (optional)
            description: New description (optional)
            template_json: New template JSON (optional; validated on write)
            tags: New tags for the JSON column (optional; also refreshed
                from the new body's tags when the body carries them)

        Raises:
            TemplateNotFoundError: If no live template has this ID
            BuiltinTemplateError: If trying to modify a builtin template
            InputError: If validation fails or the new name is taken
            InvalidTemplateError: If the NEW body fails validation, or the
                new name is the reserved sentinel "auto" (auto-selection
                spec §4.3/AC 14)
        """
        template = self.get_template_by_id(template_id)

        if template["is_builtin"]:
            raise BuiltinTemplateError("Cannot modify builtin templates")

        updates = []
        params = []

        if name is not None:
            if not name.strip():
                raise InputError("Template name cannot be empty")
            if name.strip() == AUTO_SENTINEL:
                raise InvalidTemplateError(
                    f"Template name '{AUTO_SENTINEL}' is reserved for the "
                    "auto-selection sentinel (the picker's 'Auto' choice) "
                    "and cannot be used by a template."
                )
            updates.append("name = ?")
            params.append(name.strip())

        if description is not None:
            if not description.strip():
                raise InputError("Template description cannot be empty")
            updates.append("description = ?")
            params.append(description.strip())

        column_tags: Optional[List[str]] = None
        if template_json is not None:
            body = self._parse_template_body(template_json)
            body_tags = self._pop_body_tags(body)
            self._validate_body(template["name"], body)
            updates.append("template_json = ?")
            params.append(json.dumps(body))
            if body_tags is not None:
                # An explicit empty list CLEARS the column (Qodo on PR
                # #1938); None means the body carries no tags key and the
                # column is left untouched.
                column_tags = body_tags
        if tags is not None:
            column_tags = list(tags)
        if column_tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(column_tags))

        if not updates:
            return  # Nothing to update

        updates.append("version = version + 1")

        try:
            query = (
                f"UPDATE ChunkingTemplates SET {', '.join(updates)} "
                "WHERE id = ? AND deleted = 0"
            )
            params.append(template_id)

            with self.media_db.transaction() as conn:
                if name is not None:
                    # Name liveness excluding this row itself.
                    existing = conn.execute(
                        "SELECT id FROM ChunkingTemplates "
                        "WHERE deleted = 0 AND name = ?",
                        (name.strip(),),
                    ).fetchone()
                    if existing and existing["id"] != template_id:
                        raise InputError(
                            f"Template with name '{name}' already exists"
                        )
                conn.execute(query, params)

            log_counter("chunking_template_updated", 1)
            logger.info(f"Updated chunking template ID {template_id}")

        except InputError:
            raise
        except Exception as e:
            logger.error(f"Error updating template: {e}")
            raise ChunkingTemplateError(f"Failed to update template: {str(e)}")

    def delete_template(self, template_id: int) -> None:
        """
        Soft-delete a template (AC 25).

        The row stays in the table (``deleted = 1``) but leaves every
        listing and lookup, and its name becomes reusable through the
        partial unique index.

        Args:
            template_id: Template ID

        Raises:
            TemplateNotFoundError: If no live template has this ID
            BuiltinTemplateError: If trying to delete a builtin template
        """
        template = self.get_template_by_id(template_id)

        if template["is_builtin"]:
            raise BuiltinTemplateError("Cannot delete builtin templates")

        try:
            with self.media_db.transaction() as conn:
                conn.execute(
                    "UPDATE ChunkingTemplates SET deleted = 1 "
                    "WHERE id = ? AND deleted = 0",
                    (template_id,),
                )

            log_counter("chunking_template_deleted", 1)
            logger.info(f"Soft-deleted chunking template ID {template_id}")

        except Exception as e:
            logger.error(f"Error deleting template: {e}")
            raise ChunkingTemplateError(f"Failed to delete template: {str(e)}")

    def duplicate_template(
        self, template_id: int, new_name: str, new_description: Optional[str] = None
    ) -> int:
        """
        Duplicate an existing live template as a custom row.

        Args:
            template_id: Source template ID
            new_name: Name for the duplicate
            new_description: Description for duplicate (optional)

        Returns:
            ID of created duplicate (fresh uuid via ``create_template``)

        Raises:
            InvalidTemplateError: If the source body fails validation —
                duplicates go through the same validate-on-write gate as
                any other create
        """
        source = self.get_template_by_id(template_id)

        if new_description is None:
            new_description = f"Copy of {source['description']}"

        return self.create_template(
            name=new_name,
            description=new_description,
            template_json=source["template_json"],
            tags=source["tags"],
            is_builtin=False,
        )

    # --- Document Configuration Methods ---

    def get_document_config(self, media_id: int) -> Optional[Dict[str, Any]]:
        """
        Get chunking configuration for a specific document.

        Args:
            media_id: Media document ID

        Returns:
            Configuration dict or None if not configured
        """
        try:
            conn = self.media_db.get_connection()
            cursor = conn.execute(
                "SELECT chunking_config FROM Media WHERE id = ?", (media_id,)
            )

            row = cursor.fetchone()
            if not row or not row["chunking_config"]:
                return None

            return json.loads(row["chunking_config"])

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in chunking_config for media {media_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching document config: {e}")
            raise ChunkingTemplateError(f"Failed to fetch document config: {str(e)}")

    def set_document_config(self, media_id: int, config: Dict[str, Any]) -> None:
        """
        Set chunking configuration for a document.

        Args:
            media_id: Media document ID
            config: Configuration dictionary
        """
        try:
            config_json = json.dumps(config)

            with self.media_db.transaction() as conn:
                conn.execute(
                    "UPDATE Media SET chunking_config = ? WHERE id = ?",
                    (config_json, media_id),
                )

            log_counter("document_chunking_config_set", 1)
            logger.info(f"Set chunking config for media {media_id}")

        except Exception as e:
            logger.error(f"Error setting document config: {e}")
            raise ChunkingTemplateError(f"Failed to set document config: {str(e)}")

    def clear_document_config(self, media_id: int) -> None:
        """
        Clear chunking configuration for a document.

        Args:
            media_id: Media document ID
        """
        try:
            with self.media_db.transaction() as conn:
                conn.execute(
                    "UPDATE Media SET chunking_config = NULL WHERE id = ?",
                    (media_id,),
                )

            log_counter("document_chunking_config_cleared", 1)
            logger.info(f"Cleared chunking config for media {media_id}")

        except Exception as e:
            logger.error(f"Error clearing document config: {e}")
            raise ChunkingTemplateError(f"Failed to clear document config: {str(e)}")

    def get_documents_using_template(self, template_name: str) -> List[Dict[str, Any]]:
        """
        Get all documents using a specific template.

        Args:
            template_name: Template name to search for

        Returns:
            List of media items using the template
        """
        try:
            conn = self.media_db.get_connection()
            cursor = conn.execute(
                """
                SELECT id, title, author, type, chunking_config
                FROM Media
                WHERE chunking_config LIKE ?
                AND deleted = 0
            """,
                (f'%"template": "{template_name}"%',),
            )

            documents = []
            for row in cursor:
                doc = {
                    "id": row["id"],
                    "title": row["title"],
                    "author": row["author"],
                    "type": row["type"],
                    "config": json.loads(row["chunking_config"])
                    if row["chunking_config"]
                    else None,
                }
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Error fetching documents using template: {e}")
            raise ChunkingTemplateError(f"Failed to fetch documents: {str(e)}")

    # --- Import/Export Methods ---

    def export_template(self, template_id: int) -> Dict[str, Any]:
        """
        Export a template for sharing.

        Args:
            template_id: Template ID

        Returns:
            Export dictionary with template data
        """
        template = self.get_template_by_id(template_id)

        # Parse template JSON
        template_data = json.loads(template["template_json"])

        export_data = {
            "name": template["name"],
            "description": template["description"],
            "template_json": template_data,
            "exported_at": datetime.now().isoformat(),
            "version": "1.0",
            "source": "tldw_chatbook",
        }

        return export_data

    def import_template(
        self, import_data: Dict[str, Any], name_suffix: str = " (Imported)"
    ) -> int:
        """
        Import a template from export data (validate-on-write applies).

        Args:
            import_data: Template export data
            name_suffix: Suffix to add if name conflicts

        Returns:
            ID of imported template
        """
        # Validate required fields
        required_fields = ["name", "description", "template_json"]
        for field in required_fields:
            if field not in import_data:
                raise InputError(f"Missing required field: {field}")

        name = import_data["name"]

        # Check for name conflict
        existing = self.get_template_by_name(name)
        if existing:
            name = f"{name}{name_suffix}"

        # Create template
        return self.create_template(
            name=name,
            description=import_data["description"],
            template_json=import_data["template_json"],
        )

    # --- Statistics Methods ---

    def get_template_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about template usage (live rows only).

        Returns:
            Dictionary with statistics
        """
        try:
            conn = self.media_db.get_connection()

            # Count live templates by builtin flag
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN is_builtin = 1 THEN 1 ELSE 0 END) as builtin_count,
                    SUM(CASE WHEN is_builtin = 0 THEN 1 ELSE 0 END) as custom_count
                FROM ChunkingTemplates
                WHERE deleted = 0
            """)

            template_stats = cursor.fetchone()

            # Count configured documents
            cursor = conn.execute("""
                SELECT COUNT(*) as configured_docs
                FROM Media
                WHERE chunking_config IS NOT NULL
                AND deleted = 0
            """)

            doc_stats = cursor.fetchone()

            # Get most used templates
            cursor = conn.execute("""
                SELECT
                    json_extract(chunking_config, '$.template') as template_name,
                    COUNT(*) as usage_count
                FROM Media
                WHERE chunking_config IS NOT NULL
                AND deleted = 0
                GROUP BY template_name
                ORDER BY usage_count DESC
                LIMIT 5
            """)

            most_used = []
            for row in cursor:
                if row["template_name"]:
                    most_used.append(
                        {"template": row["template_name"], "count": row["usage_count"]}
                    )

            return {
                "total_templates": template_stats["total"],
                "builtin_templates": template_stats["builtin_count"],
                "custom_templates": template_stats["custom_count"],
                "configured_documents": doc_stats["configured_docs"],
                "most_used_templates": most_used,
            }

        except Exception as e:
            logger.error(f"Error getting template statistics: {e}")
            return {
                "total_templates": 0,
                "builtin_templates": 0,
                "custom_templates": 0,
                "configured_documents": 0,
                "most_used_templates": [],
            }

    # --- Helper Methods ---

    def _row_to_template_dict(self, row: Any) -> Dict[str, Any]:
        """Convert a v7 database row to a template dictionary."""
        tags_raw = row["tags"]
        tags: List[str] = []
        if tags_raw:
            try:
                parsed = json.loads(tags_raw)
                if isinstance(parsed, list):
                    tags = [str(tag) for tag in parsed]
            except (TypeError, ValueError):
                logger.warning(
                    "Chunking template %s has a corrupt tags column; "
                    "treating tags as empty",
                    row["name"],
                )
        return {
            "id": row["id"],
            "uuid": row["uuid"],
            "name": row["name"],
            "description": row["description"],
            "template_json": row["template_json"],
            "tags": tags,
            "is_builtin": bool(row["is_builtin"]),
            "version": int(row["version"]),
            "deleted": bool(row["deleted"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _parse_template_body(template_json: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Parse a template body to a dict, raising ``InputError`` on bad JSON.

        Args:
            template_json: Template configuration as JSON string or dict

        Returns:
            The parsed body dict (never the original object — callers may
            mutate it to move tags into the column).

        Raises:
            InputError: If the value is not a dict or not valid JSON, or
                does not parse to an object.
        """
        if isinstance(template_json, dict):
            return dict(template_json)
        try:
            parsed = json.loads(template_json)
        except (TypeError, ValueError) as e:
            raise InputError(f"Invalid template JSON: {str(e)}")
        if not isinstance(parsed, dict):
            raise InputError("Template JSON must be an object")
        return parsed

    @staticmethod
    def _pop_body_tags(body: Dict[str, Any]) -> Optional[List[str]]:
        """Move ``tags`` out of the body (top-level, else ``metadata.tags``).

        Mirrors the v6→v7 conversion's placement: tags live in the column,
        not the JSON body. Mutates ``body`` only when tags are found.

        Returns:
            The extracted tags, or None when the body carries none.
        """
        tags: Optional[List[str]] = None
        if isinstance(body.get("tags"), list):
            tags = [str(tag) for tag in body.pop("tags")]
        metadata = body.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("tags"), list):
            if tags is None:
                tags = [str(tag) for tag in metadata.pop("tags")]
            else:
                del metadata["tags"]
            if not metadata:
                body.pop("metadata", None)
        return tags

    @staticmethod
    def _validate_body(name: str, body: Dict[str, Any]) -> None:
        """Run the server-parity validator on a body being written.

        §7.1 carve-out: ``name``/``description`` never enter the validated
        body — the validator sees only the chunking configuration.

        Args:
            name: Template name (for the error message only)
            body: The body dict (tags already moved to the column).

        Raises:
            InvalidTemplateError: When the validator's verdict is
                ``valid: False`` — the validator itself never raises
                (AC 24: a NAMED refusal, not a leaked exception).
        """
        # Lazy: module scope would be circular (RAG_Admin imports this
        # module through local_rag_admin_service).
        from ..RAG_Admin.template_validation import validate_template

        validated = {
            key: value
            for key, value in body.items()
            if key not in ("name", "description")
        }
        result = validate_template(validated)
        if not result["valid"]:
            summary = "; ".join(
                f"{issue['field']}: {issue['message']}"
                for issue in result["errors"][:3]
            )
            raise InvalidTemplateError(
                f"Template '{name}' failed validation and was refused: {summary}"
            )


# --- Convenience Functions ---


def get_chunking_service(media_db: MediaDatabase) -> ChunkingInteropService:
    """
    Get a ChunkingInteropService instance.

    Args:
        media_db: MediaDatabase instance

    Returns:
        ChunkingInteropService instance
    """
    return ChunkingInteropService(media_db)


# End of chunking_interop_library.py
#######################################################################################################################
