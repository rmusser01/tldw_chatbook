"""
Shared version information for packaging
"""

# Version info - should match pyproject.toml
VERSION = "0.1.6.2"
VERSION_TUPLE = (0, 1, 6, 2)

# Company/Product info
COMPANY_NAME = "TLDW Project"
PRODUCT_NAME = "tldw chatbook"
COPYRIGHT = "Copyright (c) 2024 Robert Musser. Licensed under AGPL-3.0-or-later"

# Build configuration
DEFAULT_BUILD_MODE = "standard"

# Feature sets for different build modes
BUILD_FEATURES = {
    "minimal": {
        "description": "Core features only",
        "extras": [],
    },
    "standard": {
        "description": "Core + web server + common features",
        "extras": ["web", "audio", "pdf"],
    },
    "full": {
        "description": "All features including ML models",
        "extras": ["web",
                   "embeddings_rag",
                   "chunker",
                   "websearch",
                   "audio",
                   "video",
                   "pdf",
                   "ebook",
                   "local_tts",
                   "mcp"
                   ],
    }
}