"""Shared version information for packaging."""

import re
import tomllib
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_project_version() -> str:
    with (_PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        return str(tomllib.load(stream)["project"]["version"])


def _version_tuple(version: str) -> tuple[int, ...]:
    release = re.match(r"^\d+(?:\.\d+)*", version)
    if release is None:
        raise ValueError(
            f"Project version must start with a numeric release: {version!r}"
        )
    return tuple(int(part) for part in release.group(0).split("."))


VERSION = _read_project_version()
VERSION_TUPLE = _version_tuple(VERSION)

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
