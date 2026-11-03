# text.py
# Description: This file contains utility functions for text processing, including color formatting and text manipulation.
#
# Imports
#
# 3rd-party Libraries
#
# Local Imports
#
######################################################################################################################
#
# Functions:
import re



def sanitize_filename(filename):
    """
    Sanitizes the filename by:
      1) Removing forbidden characters entirely (rather than replacing them with '-')
      2) Collapsing consecutive whitespace into a single space
      3) Collapsing consecutive dashes into a single dash
    """
    # 1) Remove forbidden characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "", filename)
    # 2) Replace runs of whitespace with a single space
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    # 3) Replace consecutive dashes with a single dash
    sanitized = re.sub(r"-{2,}", "-", sanitized)
    return sanitized






def slugify(text: str) -> str:
    """Simple slugify function, robust for empty or non-string."""
    if not isinstance(text, str) or not text:
        return "unknown_type"  # Default slug for unexpected types
    return (
        text.lower()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("&", "and")
        .replace("(", "")
        .replace(")", "")
        .replace(":", "")
        .replace(",", "")
    )


#
# End of text.py
######################################################################################################################


MAX_WIRE_INTEGER = (1 << 53) - 1


class CanvasLimitError(ValueError):
    """Raised when a Canvas value exceeds a closed runtime boundary."""


def _validate_non_negative_integer(value: "object", *, field_name: "str") -> "int":
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CanvasLimitError(f"{field_name} must be a non-negative integer")
    return value


def utf8_byte_length(value: "str") -> "int":
    """Return the strict UTF-8 byte length of a string without coercion.

    Args:
        value: Text to encode as strict UTF-8.

    Returns:
        The encoded byte length.

    Raises:
        CanvasLimitError: If ``value`` is not a string or contains invalid
            Unicode.
    """
    if not isinstance(value, str):
        raise CanvasLimitError("value must be a string")
    try:
        return len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as exc:
        raise CanvasLimitError("value must contain valid Unicode") from exc


def validate_utf8_text(value: "str", *, limit: "int", field_name: "str") -> "int":
    """Validate a strict UTF-8 string and return its encoded byte length.

    Args:
        value: Text to validate.
        limit: Maximum accepted UTF-8 byte length.
        field_name: Safe field label used in failures.

    Returns:
        The encoded byte length.

    Raises:
        CanvasLimitError: If the value is not valid strict UTF-8 text, the
            limit is invalid, or the encoded value exceeds the limit.
    """
    byte_count = utf8_byte_length(value)
    _validate_non_negative_integer(limit, field_name=f"{field_name} limit")
    if limit > MAX_WIRE_INTEGER:
        raise CanvasLimitError(
            f"{field_name} limit exceeds the supported integer range"
        )
    if byte_count > limit:
        raise CanvasLimitError(f"{field_name} exceeds {limit} UTF-8 bytes")
    return byte_count


def validate_opaque_identifier(
    identifier: "str", *, field_name: "str" = "identifier"
) -> "str":
    """Validate an application-issued opaque identifier without interpreting it.

    Args:
        identifier: Non-empty identifier to validate.
        field_name: Safe field label used in failures.

    Returns:
        The unchanged validated identifier.

    Raises:
        CanvasLimitError: If the identifier is not a non-empty strict UTF-8
            string within the identifier byte limit.
    """
    if not isinstance(identifier, str) or not identifier:
        raise CanvasLimitError(f"{field_name} must be a non-empty string")
    validate_utf8_text(identifier, limit=256, field_name=field_name)
    return identifier
