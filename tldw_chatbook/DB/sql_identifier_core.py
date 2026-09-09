"""Exact SQL identifier grammar and escaping without application logging."""

import re

# This pattern is designed to be safe while supporting non-English identifiers
SQL_IDENTIFIER_PATTERN = re.compile(r"^[\w\u0080-\uFFFF]+$", re.UNICODE)

# Reserved SQL keywords that should not be used as identifiers
SQL_RESERVED_KEYWORDS = {
    "SELECT",
    "FROM",
    "WHERE",
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "CREATE",
    "TABLE",
    "INDEX",
    "VIEW",
    "UNION",
    "JOIN",
    "LEFT",
    "RIGHT",
    "INNER",
    "OUTER",
    "ORDER",
    "BY",
    "GROUP",
    "HAVING",
    "LIMIT",
    "OFFSET",
    "AS",
    "ON",
    "AND",
    "OR",
    "NOT",
    "NULL",
    "PRIMARY",
    "KEY",
    "FOREIGN",
    "REFERENCES",
    "CASCADE",
    "SET",
    "VALUES",
    "INTO",
    "EXISTS",
    "BETWEEN",
    "LIKE",
    "IN",
    "IS",
    "DISTINCT",
    "ALL",
}


def identifier_error(identifier: str) -> str | None:
    """Return the fixed reason used by the public logging wrapper."""
    if not identifier:
        return "empty"
    if len(identifier) > 64:
        return "length"
    if not SQL_IDENTIFIER_PATTERN.match(identifier):
        return "characters"
    if identifier.upper() in SQL_RESERVED_KEYWORDS:
        return "reserved"
    return None


def validate_identifier(identifier: str, identifier_type: str = "identifier") -> bool:
    """Validate the existing grammar; identifier_type is compatibility context."""
    return identifier_error(identifier) is None


def escape_identifier(identifier: str) -> str:
    """
    Escapes a SQL identifier by wrapping it in double quotes.
    Note: This should only be used after validation, not as a replacement for validation.

    Args:
        identifier: The identifier to escape

    Returns:
        str: The escaped identifier
    """
    # Replace any existing double quotes with two double quotes (SQL escaping)
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'
