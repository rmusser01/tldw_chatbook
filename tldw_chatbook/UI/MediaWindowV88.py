# MediaWindowV88.py
# Compatibility exports for older media UI imports.

from .MediaWindow_v2 import MediaWindow as MediaWindowV88
from ..Widgets.Media import MediaItemSelectedEvent as MediaItemSelectedEventV88
from ..Widgets.Media import MediaSearchEvent as MediaSearchEventV88
from ..Widgets.Media.media_navigation_panel import (
    MediaTypeSelectedEvent as MediaTypeSelectedEventV88,
)

__all__ = [
    "MediaWindowV88",
    "MediaItemSelectedEventV88",
    "MediaSearchEventV88",
    "MediaTypeSelectedEventV88",
]
