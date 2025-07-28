"""
Constants and shared configuration for RAG Search components
"""

from typing import Dict

# Source type icons and colors
SOURCE_ICONS: Dict[str, str] = {
    "media": "🎬",
    "conversations": "💬", 
    "notes": "📝"
}

SOURCE_COLORS: Dict[str, str] = {
    "media": "cyan",
    "conversations": "green",
    "notes": "yellow"
}

# Default configuration values
DEFAULT_TOP_K = 10
DEFAULT_TEMPERATURE = 0.1
DEFAULT_PARENT_SIZE = 512
MAX_CONCURRENT_SEARCHES = 5

# Search modes
SEARCH_MODES = {
    "plain": "Plain Search",
    "contextual": "Contextual Search", 
    "hybrid": "Hybrid Search"
}

# Parent retrieval strategies
PARENT_STRATEGIES = [
    ("full", "Full Parent"),
    ("sentence_window", "Sentence Window"),
    ("auto_merging", "Auto Merging")
]