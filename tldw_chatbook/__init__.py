"""
tldw_chatbook - A Textual TUI for chatting with LLMs

A sophisticated Terminal User Interface (TUI) application built with the Textual 
framework for interacting with various Large Language Model APIs. Provides a 
complete ecosystem for AI-powered interactions including conversation management, 
character/persona chat, notes with bidirectional file sync, media ingestion, 
and advanced RAG (Retrieval-Augmented Generation) capabilities.
"""

__version__ = "0.1.0"
__author__ = "Robert Musser"
__email__ = "contact@rmusser.net"
__license__ = "AGPLv3+"

# Version tuple for programmatic comparison
VERSION_TUPLE = (0, 1, 0)

# Export key components when package is imported
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "VERSION_TUPLE",
]