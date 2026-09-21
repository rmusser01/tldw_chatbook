"""task-32804.2: the Settings screen caches the active-profile RAG defaults
for a category visit instead of re-loading them ~7x per keystroke.

Gate-free: builds a bare screen via ``__new__`` and stubs the load function, so
it isolates the memoise/invalidate contract without mounting (which would trip
the ADR-126 storage gate).
"""

from __future__ import annotations

import tldw_chatbook.UI.Screens.settings_screen as ss


def test_library_rag_loaded_defaults_cached_and_invalidatable(monkeypatch):
    calls = {"n": 0}
    sentinel = object()

    def _load():
        calls["n"] += 1
        return sentinel

    monkeypatch.setattr(ss, "load_rag_defaults_from_active_profile", _load)

    screen = ss.SettingsScreen.__new__(ss.SettingsScreen)
    screen._library_rag_loaded_defaults_cache = None

    assert screen._library_rag_loaded_defaults() is sentinel
    assert screen._library_rag_loaded_defaults() is sentinel
    assert calls["n"] == 1  # loaded once, then cached across reads

    screen._invalidate_library_rag_loaded_cache()
    assert screen._library_rag_loaded_defaults() is sentinel
    assert calls["n"] == 2  # a fresh load only after invalidation
