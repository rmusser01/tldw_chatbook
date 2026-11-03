"""Media database error values, shared through the existing DB API."""


from .store_errors import StoreConflictError, StoreInputError, StoreSchemaError


class DatabaseError(Exception):
    """Base exception for database related errors."""

    __module__ = "tldw_chatbook.DB.Client_Media_DB_v2"


class SchemaError(DatabaseError, StoreSchemaError):
    """Exception for schema version mismatches or migration failures."""

    __module__ = "tldw_chatbook.DB.Client_Media_DB_v2"


class InputError(StoreInputError, ValueError):
    """Custom exception for input validation errors."""

    __module__ = "tldw_chatbook.DB.Client_Media_DB_v2"


class ConflictError(DatabaseError, StoreConflictError):
    """Indicates a conflict due to concurrent modification (version mismatch)."""

    __module__ = "tldw_chatbook.DB.Client_Media_DB_v2"

    def __init__(
        self,
        message="Conflict detected: Record modified concurrently.",
        entity=None,
        identifier=None,
    ):
        super().__init__(message)
        self.entity = entity
        self.identifier = identifier  # Can be id or uuid

    def __str__(self):
        base = super().__str__()
        details = []
        if self.entity:
            details.append(f"Entity: {self.entity}")
        if self.identifier:
            details.append(f"ID: {self.identifier}")
        return f"{base} ({', '.join(details)})" if details else base
