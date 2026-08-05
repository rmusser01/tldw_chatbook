"""
RAG Search handoff helpers.

This package now only holds ``search_handoff.py`` (Library RAG result ->
Chat/Console live-work handoff payload builders). It is import-only: those
helpers are imported by full module path from ``UI/Screens/library_screen.py``
and ``UI/Screens/chat_screen.py``, not re-exported from here. The standalone
Search screen and its ``SearchResult`` widget were retired in PR #1258; the
widget and its ``constants.py`` were removed in RAG UX v2 PR-2 Task 3.
"""
