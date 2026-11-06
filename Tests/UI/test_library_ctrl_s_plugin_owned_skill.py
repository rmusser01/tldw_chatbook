"""Ctrl+S on a plugin-owned skill says it is read-only (PR #3423 review m-6).

Dev's plugin-owned Skill view (``library_skills_canvas``, ``editor_state.
plugin_owned``) is read-only package content: no Save, no editable field --
only "Manage in Plugins…" and Back. ``_library_editor_save_target`` still
answered ``"skill"`` while focus sat in ``library-skill-work-pane``, so
Ctrl+S there reached ``_save_library_skill_from_key``, which painted "Saved —
no changes." on the skill's status line: wording for an editor that could
have saved. The key now answers with the read-only refusal, in the Ctrl+S
refusal grammar ("Save unavailable — …", ADR-299 rule (c): said at the
editor, no focus moved), and still offers no "ctrl+s" chip.

The production ``TldwCli`` booted into Library over a real
``LocalSkillsService`` whose plugin service is the real ``PluginService``
with the native fixture package installed (the Plugins suite's own stack,
qualified on 64-bit Darwin only). The owned skill is opened by name, through
the entry the Skills header's "review" action uses: the Skills list cannot
compose an owned row yet (its row id embeds the ``alias:skill`` name and
Textual refuses the ``:`` -- a separate defect, reported with this fix).
Then a real click on its Edit tab and a real Ctrl+S; the claim is painted
text. The refusal arm on a fake, which the Linux PR lane runs, is in
``test_library_ctrl_s_refusals.py``.

Base (``bd01c0e46b``): the status line reads "Saved — no changes.".
"""

from __future__ import annotations

import asyncio

import pytest

from Tests.app_module_patches import patch_app_global
from Tests.Plugins import platform_support
from Tests.Plugins.conftest import native_package as _native_package
from Tests.UI.library_quit_guard_support import (
    SIZE,
    _library,
    _library_app,
    _settings_without_splash,
)
from Tests.UI.test_library_ctrl_s_saves_in_every_editor import (
    NO_CHANGES,
    _focused_id,
    _footer,
    _painted,
    _press_processed,
    _record_resolved_keys,
)
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

native_package = _native_package

#: The refusal the user reads: a literal, so the base fails on what it
#: paints, not on a missing import.
READ_ONLY = "Save unavailable — this skill is read-only plugin content."

#: A behaviour wait: long enough for a detail read under load, short enough
#: that a RED arm names its reason in seconds.
WAIT = 10.0


async def _until(pilot, predicate, what) -> None:
    """Pump until ``predicate()`` holds; ``what`` may be read at failure."""
    try:
        async with asyncio.timeout(WAIT):
            while not predicate():
                await pilot.pause(0.02)
    except TimeoutError:
        raise AssertionError(
            f"timed out waiting for {what() if callable(what) else what}"
        ) from None


async def test_ctrl_s_on_a_plugin_owned_skill_says_it_is_read_only(
    tmp_path, monkeypatch, native_package
) -> None:
    platform_support.require_managed_runtime()
    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore
    from tldw_chatbook.Plugins.service import PluginService

    service = PluginService(
        tmp_path / "plugin-profile",
        workspace_lookup=lambda _: None,
        marker_store_factory=lambda _: FilePluginMarkerStore(tmp_path / "marker"),
        accept_reduced_protection=True,
    )
    try:
        await service.bootstrap("test passphrase")
        review = await service.review_install(
            native_package(), selection=("skill:review",), workspace_id=None
        )
        assert (await service.commit(review, review.operation_id)).committed
        (owned,) = service.list_skills()
        owned_name = owned["name"]
        assert owned["plugin_owned"] and ":" in owned_name
        local = LocalSkillsService(
            store_dir=tmp_path / "skills",
            plugin_service=service,
            allow_untrusted_without_trust_service=True,
        )
        app, _profile = _library_app(
            tmp_path, monkeypatch, [], lambda _p: None, notes=False
        )
        app.skills_scope_service = SkillsScopeService(
            local_service=local, server_service=None, policy_enforcer=None
        )
        resolved = _record_resolved_keys(monkeypatch, app)
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=SIZE) as pilot:
                screen = await _library(app, pilot)
                # Opened by name, the entry the Skills header's "review"
                # action uses: the Skills LIST cannot compose an owned row
                # yet (its row id embeds the ``alias:skill`` name, and
                # Textual refuses the ``:``), so it is never painted here.
                await screen._open_library_skill_editor_for_review(owned_name)

                def _diagnosis() -> str:
                    return (
                        f"view {screen._skills_state.view!r}, owned "
                        f"{getattr(screen._skills_state.editor_state, 'plugin_owned', None)!r}"
                        f", focus #{_focused_id(screen)}: {_painted(app, app.screen)}"
                    )

                await _until(
                    pilot,
                    lambda: screen._skills_state.view == "editor"
                    and getattr(screen._skills_state.editor_state, "plugin_owned", False)
                    and bool(screen.query("#library-skill-mode-edit")),
                    lambda: f"the owned skill's editor: {_diagnosis()}",
                )
                # Its Edit tab is the read-only package view: no Save, no
                # editable field -- Manage in Plugins… and Back.
                await pilot.pause(0.2)
                await pilot.click("#library-skill-mode-edit")
                await _until(
                    pilot,
                    lambda: bool(screen.query("#library-skill-manage-plugin"))
                    and bool(screen.query("#library-skill-save-status"))
                    and _focused_id(screen) == "library-skill-mode-edit",
                    lambda: f"the read-only view, focus on its Edit tab: {_diagnosis()}",
                )
                assert not screen.query("#library-skill-save")
                assert not screen.query("#library-skill-body")
                origin = _focused_id(screen)

                def _status() -> str:
                    return _painted(app, screen.query_one("#library-skill-save-status"))

                assert "ctrl+s" not in _footer(app), _footer(app)
                await _press_processed(pilot, resolved, "ctrl+s")
                await _until(
                    pilot,
                    lambda: READ_ONLY in _status(),
                    lambda: f"the read-only refusal on the status line: {_status()!r}",
                )
                assert NO_CHANGES not in _status()
                # Said at the editor: focus never moved, and no chip promised
                # a save.
                assert _focused_id(screen) == origin
                assert screen._skills_state.view == "editor"
                assert "ctrl+s" not in _footer(app), _footer(app)
    finally:
        await service.aclose()
