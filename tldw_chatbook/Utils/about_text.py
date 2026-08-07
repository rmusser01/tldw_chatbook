"""Application About text and version lookup (TASK-2775).

The About content lived only in the deprecated ToolsSettingsWindow, which has
been unrouted dead UI since TASK-1346 — no reachable surface showed the
project description, license, or links. The canonical home is now the F9
Settings screen's About category; the legacy window re-imports from here.
"""

from importlib import metadata

#: Real markdown (TASK-1995 converted it from Rich console markup, which the
#: Markdown widget rendered literally).
ABOUT_MARKDOWN = """\
**tldw-chatbook** is a sophisticated Terminal User Interface (TUI) application for interacting with various Large Language Model APIs.

*Features:*

- Multi-provider LLM support (OpenAI, Anthropic, Google, and many more)
- Advanced conversation management with branching
- Character-based conversations with personality cards
- Comprehensive note-taking with bidirectional file sync
- Media ingestion and analysis (video, audio, documents, PDFs, e-books)
- RAG (Retrieval-Augmented Generation) for intelligent search
- Local LLM server management
- Extensive customization options

*License:* AGPLv3+

*Links:*

- GitHub: <https://github.com/rmusser01/tldw>
- Documentation: <https://github.com/rmusser01/tldw/wiki>
- Issues: <https://github.com/rmusser01/tldw/issues>

*Created by:* rmusser01 and contributors

Thank you for using tldw-chatbook! 🎉
"""


def get_app_version() -> str:
    """Return the installed tldw_chatbook version, or a source-checkout marker.

    Returns:
        The distribution version string, or ``"unknown (source checkout)"``
        when the package metadata is unavailable (e.g. running from a
        checkout that was never pip-installed).
    """
    try:
        return metadata.version("tldw_chatbook")
    except metadata.PackageNotFoundError:
        return "unknown (source checkout)"
