"""Existing-session draft refresh must not build discarded new-chat defaults."""

import pytest

from Tests.UI.test_console_mcp_approval import (
    _build_ready_production_console_app,
    _wait_for_production_console_ready,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def test_active_draft_refresh_does_not_derive_ignored_defaults(monkeypatch):
    app = _build_ready_production_console_app()
    async with app.run_test(size=(120, 50)) as pilot:
        screen = await _wait_for_production_console_ready(app, pilot)
        store = screen._console_chat_store
        session_id = store.active_session_id
        session = next(s for s in store.sessions() if s.id == session_id)
        settings = session.settings
        baseline = session.canonical_settings_baseline
        generation = session.new_chat_default_generation
        composer = screen._console_composer_or_none()
        text = composer.draft_text()
        derived = []
        native = screen._session._blank_console_session_settings

        def build():
            value = native()
            derived.append(value)
            return value

        monkeypatch.setattr(screen._session, "_blank_console_session_settings", build)
        for _ in range(10):
            screen._session._sync_console_session_draft()
        assert derived == []
        assert store.active_session_id == session_id
        assert session.settings is settings
        assert session.canonical_settings_baseline is baseline
        assert session.new_chat_default_generation == generation
        assert composer.draft_text() == store.session_draft(session_id) == text
        assert screen._console_visible_draft_session_id == session_id
