"""Finite float32 vectors and bounded exact cosine ranking for project search."""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence

MAX_DIMENSION = 16384


def encode_vector(values: Sequence[float]) -> bytes:
    """Encode a nonempty finite vector as little-endian float32 bytes."""
    if not 0 < len(values) <= MAX_DIMENSION:
        raise ValueError("invalid_semantic_dimension")
    try:
        numbers = tuple(float(value) for value in values)
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("invalid_semantic_vector")
        encoded = struct.pack(f"<{len(numbers)}f", *numbers)
    except (OverflowError, TypeError, struct.error) as error:
        raise ValueError("invalid_semantic_vector") from error
    if not all(math.isfinite(value) for value in struct.unpack(f"<{len(numbers)}f", encoded)):
        raise ValueError("invalid_semantic_vector")
    return encoded


def decode_vector(encoded: bytes, dimension: int) -> tuple[float, ...]:
    """Decode only an exact, finite float32 vector of the expected dimension."""
    if (
        type(encoded) is not bytes  # noqa: E721 - reject bytearray
        or type(dimension) is not int  # noqa: E721 - reject bool
        or not 0 < dimension <= MAX_DIMENSION
        or len(encoded) != 4 * dimension
    ):
        raise ValueError("invalid_semantic_vector")
    values = struct.unpack(f"<{dimension}f", encoded)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("invalid_semantic_vector")
    return values
