"""Master shell destination wrapper tests."""

import asyncio
import inspect
import logging
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App
from textual.widgets import Button, Checkbox, Select, Static, TextArea

from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_unified_mcp_panel import FakeUnifiedMCPService
from tldw_chatbook.ACP_Interop.runtime_session import ACPRuntimeSessionState
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Home.dashboard_state import HomeActiveWorkItem, HomeDashboardInput
from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
from tldw_chatbook.MCP.unified_control_models import ConfiguredServerTarget
from tldw_chatbook.runtime_policy.types import PolicyDeniedError
from tldw_chatbook.UI.MCP_Modules.unified_mcp_panel import UnifiedMCPPanel
from tldw_chatbook.UI.Screens.artifacts_screen import ArtifactsScreen
from tldw_chatbook.UI.Screens.acp_screen import ACPScreen
from tldw_chatbook.UI.Screens.destination_recovery import DestinationRecoveryState
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.UI.Screens.mcp_screen import MCPScreen
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.UI.Screens.schedules_screen import SchedulesScreen
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.UI.Screens import settings_screen as settings_screen_module
from tldw_chatbook.UI.Screens.skills_screen import SkillsScreen
from tldw_chatbook.UI.Screens.watchlists_collections_screen import WatchlistsCollectionsScreen
from tldw_chatbook.UI.Screens.workflows_screen import WorkflowsScreen
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.UI.Screens import personas_screen as personas_screen_module
from tldw_chatbook.UI.Screens import skills_screen as skills_screen_module
from tldw_chatbook.UI.Screens import watchlists_collections_screen as wc_screen_module


SCREEN_BY_ROUTE = {
    "library": LibraryScreen,
    "artifacts": ArtifactsScreen,
    "personas": PersonasScreen,
    "watchlists_collections": WatchlistsCollectionsScreen,
    "schedules": SchedulesScreen,
    "workflows": WorkflowsScreen,
    "mcp": MCPScreen,
    "tools_settings": MCPScreen,
    "acp": ACPScreen,
    "skills": SkillsScreen,
    "settings": SettingsScreen,
}

PHASE4_MCP_ADOPTION_EVIDENCE = Path(
    "Docs/superpowers/qa/unified-shell/phase-4/2026-05-04-mcp-destination-service-adoption.md"
)
PHASE4_SKILLS_ADOPTION_EVIDENCE = Path(
    "Docs/superpowers/qa/unified-shell/phase-4/2026-05-04-skills-destination-service-adoption.md"
)
PHASE4_LIBRARY_ADOPTION_EVIDENCE = Path(
    "Docs/superpowers/qa/unified-shell/phase-4/2026-05-04-library-source-service-adoption.md"
)
PHASE4_PERSONAS_ADOPTION_EVIDENCE = Path(
    "Docs/superpowers/qa/unified-shell/phase-4/2026-05-04-personas-service-adoption.md"
)
PHASE4_WC_ADOPTION_EVIDENCE = Path(
    "Docs/superpowers/qa/unified-shell/phase-4/2026-05-04-wc-service-adoption.md"
)


class StaticPersonasScopeService:
    def __init__(self, *, characters=(), profiles=()):
        self.characters = tuple(characters)
        self.profiles = tuple(profiles)
        self.character_calls = []
        self.profile_calls = []

    async def list_characters(self, **kwargs):
        self.character_calls.append(kwargs)
        return list(self.characters)

    async def list_persona_profiles(self, **kwargs):
        self.profile_calls.append(kwargs)
        return list(self.profiles)


class RaisingPersonasScopeService:
    async def list_characters(self, **kwargs):
        raise RuntimeError("characters unavailable")

    async def list_persona_profiles(self, **kwargs):
        return []


class SlowPersonasScopeService(StaticPersonasScopeService):
    async def list_characters(self, **kwargs):
        await asyncio.sleep(0.35)
        return await super().list_characters(**kwargs)


class HangingPersonasScopeService(StaticPersonasScopeService):
    async def list_characters(self, **kwargs):
        await asyncio.sleep(999)
        return await super().list_characters(**kwargs)


class BlockingPersonasScopeService(StaticPersonasScopeService):
    def __init__(self, *, characters=(), profiles=(), sleep_seconds=0.3):
        super().__init__(characters=characters, profiles=profiles)
        self.sleep_seconds = sleep_seconds

    def list_characters(self, **kwargs):
        self.character_calls.append(kwargs)
        time.sleep(self.sleep_seconds)
        return list(self.characters)

    def list_persona_profiles(self, **kwargs):
        self.profile_calls.append(kwargs)
        return list(self.profiles)


class StaticLibraryNotesScopeService:
    def __init__(self, notes):
        self.notes = tuple(notes)
        self.calls = []

    async def list_notes(self, **kwargs):
        self.calls.append(kwargs)
        return {"items": list(self.notes), "pagination": {"total": len(self.notes)}}


class StaticLibraryNotesListScopeService:
    def __init__(self, notes):
        self.notes = tuple(notes)
        self.calls = []

    async def list_notes(self, **kwargs):
        self.calls.append(kwargs)
        limit = kwargs.get("limit")
        if isinstance(limit, int):
            return list(self.notes[:limit])
        return list(self.notes)


class StaticLibraryMediaScopeService:
    # Mirrors LocalMediaReadingService._SUPPORTED_METADATA_FIELDS in
    # tldw_chatbook/Media/local_media_reading_service.py -- keep in sync so
    # this fake fails the same way the real local backend does when a
    # caller sends an unsupported field (e.g. ``version``).
    _SUPPORTED_METADATA_FIELDS = {"title", "media_type", "author", "url", "keywords"}

    def __init__(self, media_items, highlights=None):
        self.media_items = tuple(media_items)
        self.calls = []
        self.detail_calls = []
        self.update_calls = []
        self.delete_calls = []
        # item_id (str) -> list of highlight dicts, mirroring the field
        # names LocalMediaReadingService._highlight_row_to_dict returns
        # ("id"/"item_id"/"quote"/"note"/"color"/...), keyed the same way
        # ``media_id``/``item_id`` is normalized elsewhere in this fake.
        self.highlights_by_item_id: dict = {}
        self._next_highlight_id = 1
        self.list_highlights_calls = []
        self.create_highlight_calls = []
        self.delete_highlight_calls = []
        self.read_it_later_calls = []
        self.analysis_calls = []
        for item_id, quote, note, color in highlights or ():
            self._seed_highlight(item_id, quote=quote, note=note, color=color)

    def _seed_highlight(self, item_id, *, quote, note=None, color=None):
        highlight = {
            "id": self._next_highlight_id,
            "item_id": str(item_id),
            "quote": quote,
            "note": note,
            "color": color,
        }
        self._next_highlight_id += 1
        self.highlights_by_item_id.setdefault(str(item_id), []).append(highlight)
        return highlight

    async def list_highlights(self, *, item_id, mode=None):
        """Return the stored highlights for ``item_id``, matching the real
        ``media_reading_scope_service.list_highlights`` -> ``LocalMediaReadingService.list_highlights``
        chain (``mode`` accepted and discarded).
        """
        self.list_highlights_calls.append({"item_id": item_id})
        return [dict(highlight) for highlight in self.highlights_by_item_id.get(str(item_id), [])]

    async def create_highlight(self, *, item_id, quote, mode=None, note=None, color=None, **kwargs):
        """Record and store a new highlight, matching the real
        ``media_reading_scope_service.create_highlight`` -> ``LocalMediaReadingService.create_highlight``
        chain (``mode`` accepted and discarded).
        """
        self.create_highlight_calls.append(
            {"item_id": item_id, "quote": quote, "note": note, "color": color}
        )
        return self._seed_highlight(item_id, quote=quote, note=note, color=color)

    async def delete_highlight(self, *, highlight_id, mode=None):
        """Remove a highlight by id, matching the real
        ``media_reading_scope_service.delete_highlight`` -> ``LocalMediaReadingService.delete_highlight``
        chain (``mode`` accepted and discarded).
        """
        self.delete_highlight_calls.append({"highlight_id": highlight_id})
        removed = False
        for item_id, highlights in list(self.highlights_by_item_id.items()):
            remaining = [h for h in highlights if str(h["id"]) != str(highlight_id)]
            if len(remaining) != len(highlights):
                removed = True
            self.highlights_by_item_id[item_id] = remaining
        return {"success": removed}

    async def list_media_items(self, **kwargs):
        self.calls.append(kwargs)
        return {"items": list(self.media_items), "pagination": {"total_items": len(self.media_items)}}

    async def get_media_item(self, *, media_id, **kwargs):
        self.detail_calls.append({"media_id": media_id, **kwargs})
        target_id = str(media_id)
        for item in self.media_items:
            item_id = str(item.get("id") or item.get("media_id") or "")
            if item_id == target_id:
                return dict(item)
        return None

    async def update_media_item(self, *, media_id, mode=None, **fields):
        """Record the update call and apply it to the stored item in place.

        Mirrors the real ``media_reading_scope_service.update_media_item`` ->
        ``LocalMediaReadingService.update_media_item`` chain: the scope-level
        ``mode`` kwarg is stripped (never forwarded to the local backend),
        and the remaining fields are checked against the local backend's
        metadata allowlist. Any other field (e.g. a leftover ``version``)
        raises ``ValueError`` just like the real
        ``update_media_metadata`` does, so tests can't go green by sending
        fields the production local backend would reject.

        Args:
            media_id: The media item id to update.
            mode: Scope-level backend selector (e.g. ``"local"``); accepted
                and discarded, matching ``MediaReadingScopeService``.
            **fields: Field values to merge onto the stored item (e.g.
                ``title``, ``author``, ``url``, ``keywords``).

        Returns:
            The updated item mapping, or None when ``media_id`` is unknown.

        Raises:
            ValueError: If any field is outside the local metadata
                allowlist.
        """
        unsupported = sorted(
            key for key, value in fields.items()
            if value is not None and key not in self._SUPPORTED_METADATA_FIELDS
        )
        if unsupported:
            unsupported_text = ", ".join(unsupported)
            raise ValueError(f"Unsupported local media metadata fields: {unsupported_text}")

        self.update_calls.append({"media_id": media_id, **fields})
        target_id = str(media_id)
        updated_items = []
        updated = None
        for item in self.media_items:
            item_id = str(item.get("id") or item.get("media_id") or "")
            if item_id == target_id:
                merged = dict(item)
                merged.update(fields)
                updated = merged
                updated_items.append(merged)
            else:
                updated_items.append(item)
        self.media_items = tuple(updated_items)
        return updated

    async def delete_media_item(self, *, media_id, mode=None):
        """Record the delete call and remove the item from storage in place.

        Mirrors the real ``media_reading_scope_service.delete_media_item`` ->
        ``LocalMediaReadingService.delete_media_item`` chain, which moves the
        item to trash: the scope-level ``mode`` kwarg is stripped, and the
        item is dropped from ``media_items`` so a subsequent
        ``list_media_items`` call omits it, matching the real backend's
        normal (non-trash) paginated list no longer returning trashed items.

        Args:
            media_id: The media item id to delete.
            mode: Scope-level backend selector (e.g. ``"local"``); accepted
                and discarded, matching ``MediaReadingScopeService``.

        Returns:
            A ``{"ok": True, "media_id": media_id}`` mapping, matching the
            real local backend's response shape.
        """
        self.delete_calls.append({"media_id": media_id})
        target_id = str(media_id)
        self.media_items = tuple(
            item
            for item in self.media_items
            if str(item.get("id") or item.get("media_id") or "") != target_id
        )
        return {"ok": True, "media_id": media_id}

    async def save_to_read_it_later(self, *, media_id, mode=None):
        """Mark a media item saved for read-it-later, matching the real
        ``media_reading_scope_service.save_to_read_it_later`` ->
        ``LocalMediaReadingService.save_to_read_it_later`` chain (``mode``
        accepted and discarded).
        """
        self.read_it_later_calls.append({"action": "save", "media_id": media_id})
        return self._set_read_it_later_state(media_id, saved=True)

    async def remove_from_read_it_later(self, *, media_id, mode=None):
        """Clear a media item's read-it-later state, matching the real
        ``media_reading_scope_service.remove_from_read_it_later`` ->
        ``LocalMediaReadingService.remove_from_read_it_later`` chain
        (``mode`` accepted and discarded).
        """
        self.read_it_later_calls.append({"action": "remove", "media_id": media_id})
        return self._set_read_it_later_state(media_id, saved=False)

    def _set_read_it_later_state(self, media_id, *, saved):
        """Apply the read-it-later flag to the stored item in place.

        On removal, the ``is_read_it_later``/``saved_at`` keys are dropped
        entirely rather than set to ``False``/``None`` -- mirroring the
        real local backend, whose ``MediaReadItLaterState`` row is deleted
        outright on removal, so a re-fetched detail has neither key present
        (see ``LocalMediaReadingService._enrich_with_read_it_later_state``).
        """
        target_id = str(media_id)
        updated_items = []
        saved_at = None
        for item in self.media_items:
            item_id = str(item.get("id") or item.get("media_id") or "")
            if item_id == target_id:
                merged = dict(item)
                if saved:
                    saved_at = merged.get("saved_at") or "2026-01-01T00:00:00Z"
                    merged["is_read_it_later"] = True
                    merged["saved_at"] = saved_at
                else:
                    merged.pop("is_read_it_later", None)
                    merged.pop("saved_at", None)
                updated_items.append(merged)
            else:
                updated_items.append(item)
        self.media_items = tuple(updated_items)
        return {"media_id": media_id, "is_read_it_later": saved, "saved_at": saved_at}

    async def save_analysis_version(self, *, media_id, content, analysis_content, mode=None, prompt=None):
        """Create a new document version carrying ``analysis_content``, matching
        the real ``media_reading_scope_service.save_analysis_version`` ->
        ``LocalMediaReadingService.save_analysis_version`` ->
        ``Client_Media_DB_v2.create_document_version`` chain (``mode``
        accepted and discarded). New versions are prepended (newest first),
        matching the real backend's ``get_all_document_versions`` ordering
        (``ORDER BY version_number DESC``).
        """
        self.analysis_calls.append(
            {
                "media_id": media_id,
                "content": content,
                "analysis_content": analysis_content,
                "prompt": prompt,
            }
        )
        return self._append_document_version(
            media_id, content=content, analysis_content=analysis_content, prompt=prompt
        )

    async def overwrite_analysis_version(self, *, media_id, content, analysis_content, mode=None, prompt=None):
        """Alias for ``save_analysis_version``, matching the real backend
        (``LocalMediaReadingService.overwrite_analysis_version`` simply
        delegates to ``save_analysis_version``).
        """
        return await self.save_analysis_version(
            media_id=media_id, content=content, analysis_content=analysis_content, mode=mode, prompt=prompt
        )

    def _append_document_version(self, media_id, *, content, analysis_content, prompt):
        target_id = str(media_id)
        updated_items = []
        new_version = None
        for item in self.media_items:
            item_id = str(item.get("id") or item.get("media_id") or "")
            if item_id == target_id:
                merged = dict(item)
                existing_versions = list(merged.get("versions") or [])
                next_version_number = (
                    max((v.get("version_number", 0) for v in existing_versions), default=0) + 1
                )
                new_version = {
                    "id": next_version_number,
                    "uuid": f"version-{next_version_number}",
                    "media_id": media_id,
                    "version_number": next_version_number,
                    "analysis_content": analysis_content,
                    "prompt": prompt,
                    "content": content,
                }
                merged["versions"] = [new_version, *existing_versions]
                merged["content"] = content
                updated_items.append(merged)
            else:
                updated_items.append(item)
        self.media_items = tuple(updated_items)
        return new_version


class StaticLibraryConversationScopeService:
    def __init__(self, conversations):
        self.conversations = tuple(conversations)
        self.calls = []

    async def list_conversations(self, **kwargs):
        self.calls.append(kwargs)
        return {"items": list(self.conversations), "pagination": {"total": len(self.conversations)}}


def _link_library_items_to_active_workspace(app, items) -> None:
    active_workspace = app.workspace_registry_service.get_active_workspace()
    if active_workspace is None:
        return
    for item_type, item_id, title in items:
        app.workspace_registry_service.link_membership(
            active_workspace.workspace_id,
            item_type=item_type,
            item_id=item_id,
            title=title,
        )


class RaisingLibraryNotesScopeService:
    async def list_notes(self, **kwargs):
        raise RuntimeError("notes unavailable")


class PolicyDeniedLibraryNotesScopeService:
    def __init__(
        self,
        *,
        reason_code="wrong_source",
        user_message="Server Library sources require server mode.",
        effective_source="local",
        authority_owner="active server",
    ):
        self.reason_code = reason_code
        self.user_message = user_message
        self.effective_source = effective_source
        self.authority_owner = authority_owner

    async def list_notes(self, **kwargs):
        raise PolicyDeniedError(
            action_id="library.sources.list",
            reason_code=self.reason_code,
            user_message=self.user_message,
            effective_source=self.effective_source,
            authority_owner=self.authority_owner,
        )


class StaticWatchlistsScopeService:
    def __init__(self, watch_items):
        self.watch_items = tuple(watch_items)
        self.calls = []

    async def list_watch_items(self, **kwargs):
        self.calls.append(kwargs)
        return list(self.watch_items)


class RaisingWatchlistsScopeService:
    async def list_watch_items(self, **kwargs):
        raise RuntimeError("watchlists unavailable")


class PolicyDeniedWatchlistsScopeService:
    def __init__(
        self,
        *,
        reason_code="server_session_invalid",
        user_message="The Watchlists server session has expired.",
        effective_source="server",
        authority_owner="active server",
    ):
        self.reason_code = reason_code
        self.user_message = user_message
        self.effective_source = effective_source
        self.authority_owner = authority_owner

    async def list_watch_items(self, **kwargs):
        raise PolicyDeniedError(
            action_id="wc.watchlists.list",
            reason_code=self.reason_code,
            user_message=self.user_message,
            effective_source=self.effective_source,
            authority_owner=self.authority_owner,
        )


class HangingWatchlistsScopeService:
    async def list_watch_items(self, **kwargs):
        await asyncio.sleep(10)
        return []


class StaticReadItLaterScopeService:
    def __init__(self, items):
        self.items = tuple(items)
        self.calls = []

    async def list_read_it_later(self, **kwargs):
        self.calls.append(kwargs)
        return {"items": list(self.items), "total": len(self.items)}


class StaticSkillsScopeService:
    def __init__(self, skills):
        self.skills = tuple(skills)
        self.calls = []

    def set_skills(self, skills) -> None:
        self.skills = tuple(skills)

    async def list_skills(self, **kwargs):
        self.calls.append(kwargs)
        return {"skills": list(self.skills), "total": len(self.skills)}


class RecordingSkillTrustService:
    def __init__(self, review_id="review-id", on_capture_review=None):
        self.review_id = review_id
        self.bootstrap_calls = 0
        self.bootstrap_passphrases = []
        self.unlock_calls = 0
        self.unlock_passphrases = []
        self.reviewed_skill = None
        self.trusted_review_id = None
        self.trust_reviewed_calls = 0
        self.on_capture_review = on_capture_review

    def bootstrap_trust(self, passphrase, *args, **kwargs):
        self.bootstrap_calls += 1
        self.bootstrap_passphrases.append(passphrase)

    def unlock_with_passphrase(self, passphrase, *args, **kwargs):
        self.unlock_calls += 1
        self.unlock_passphrases.append(passphrase)

    def capture_review(self, skill_name):
        self.reviewed_skill = skill_name
        if self.on_capture_review is not None:
            self.on_capture_review(skill_name)
        return {
            "review_id": self.review_id,
            "skill_name": skill_name,
            "changed_files": ["SKILL.md"],
        }

    def trust_reviewed_snapshot(self, review_id):
        self.trust_reviewed_calls += 1
        self.trusted_review_id = review_id


class RaisingSkillsScopeService:
    async def list_skills(self, **kwargs):
        raise RuntimeError("skills registry unavailable")


class PolicyDeniedSkillsScopeService:
    def __init__(
        self,
        *,
        reason_code="authority_denied",
        user_message="Local Skills are disabled by workspace policy.",
        effective_source="local",
        authority_owner="local",
    ):
        self.reason_code = reason_code
        self.user_message = user_message
        self.effective_source = effective_source
        self.authority_owner = authority_owner

    async def list_skills(self, **kwargs):
        raise PolicyDeniedError(
            action_id="skills.list.local",
            reason_code=self.reason_code,
            user_message=self.user_message,
            effective_source=self.effective_source,
            authority_owner=self.authority_owner,
        )


class PolicyDeniedPersonasScopeService:
    def __init__(
        self,
        *,
        reason_code="server_auth_required",
        user_message="Server Personas require sign-in.",
        effective_source="server",
        authority_owner="active server",
    ):
        self.reason_code = reason_code
        self.user_message = user_message
        self.effective_source = effective_source
        self.authority_owner = authority_owner

    async def list_characters(self, **kwargs):
        raise PolicyDeniedError(
            action_id="personas.characters.list",
            reason_code=self.reason_code,
            user_message=self.user_message,
            effective_source=self.effective_source,
            authority_owner=self.authority_owner,
        )

    async def list_persona_profiles(self, **kwargs):
        return []


class StaticHomeActiveWorkAdapter:
    def __init__(self, *items: HomeActiveWorkItem):
        self.items = tuple(items)
        self.calls = []

    def build_dashboard_input(self, *, providers_models, has_recent_work):
        self.calls.append(
            {
                "providers_models": providers_models,
                "has_recent_work": has_recent_work,
            }
        )
        return HomeDashboardInput(active_work_items=self.items)


class DestinationHarness(App):
    def __init__(self, app_instance, route, seen_routes=None, restored_state=None):
        super().__init__()
        self.app_instance = app_instance
        self.route = route
        self.seen_routes = seen_routes if seen_routes is not None else []
        self.restored_state = restored_state

    async def on_mount(self) -> None:
        screen = SCREEN_BY_ROUTE[self.route](self.app_instance)
        if self.restored_state is not None:
            screen.restore_state(self.restored_state)
        await self.push_screen(screen)

    def on_navigate_to_screen(self, message) -> None:
        self.seen_routes.append(message.screen_name)


def _active_destination_screen(host: DestinationHarness):
    return host.screen_stack[-1]


def _static_text(widget: Static) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


def _visible_text(screen) -> str:
    return " ".join(
        [
            *(
                _static_text(widget)
                for widget in screen.query(Static)
                if widget.display and hasattr(widget, "renderable")
            ),
            *(
                str(button.label)
                for button in screen.query(Button)
                if button.display and button.label is not None
            ),
        ]
    )


async def _wait_for_personas_snapshot(screen, pilot, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    terminal_selectors = (
        "#personas-service-error",
        "#personas-empty-state",
        "#personas-characters-summary",
        "#personas-profiles-summary",
        "#personas-readiness-console",
    )
    while time.monotonic() < deadline:
        terminal_state_visible = any(screen.query(selector) for selector in terminal_selectors)
        buttons = list(screen.query("#personas-attach-to-console"))
        if buttons:
            button = buttons[0]
            if button.disabled is False:
                await pilot.pause()
                return
            if terminal_state_visible:
                await pilot.pause()
                return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for Personas snapshot. Visible text: {_visible_text(screen)}")


async def _wait_for_wc_snapshot(screen, pilot, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    terminal_selectors = (
        "#wc-service-error",
        "#wc-empty-state",
        "#wc-watchlists-summary",
    )
    while time.monotonic() < deadline:
        if any(screen.query(selector) for selector in terminal_selectors):
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for Watchlists snapshot. Visible text: {_visible_text(screen)}")


async def _wait_for_library_snapshot(screen, pilot, *, timeout: float = 2.0) -> None:
    """Wait for the Library rail shell to finish its async source-count load.

    Mirrors ``Tests/UI/test_library_shell.py::_wait_for_library_shell`` for
    suites (like this one) that mount Library through the generic
    ``DestinationHarness`` instead of the dedicated ``LibraryHarness``. The
    retired 3-pane hub's terminal selectors (``#library-source-error``,
    ``#library-source-empty``, the per-source summary ids) never mount under
    the rail + canvas shell, so readiness is keyed off ``_library_loaded``
    and the rail itself.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if getattr(screen, "_library_loaded", False) and screen.query("#library-rail"):
            await pilot.pause()
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError(f"Library shell never loaded. Visible text: {_visible_text(screen)}")


async def _wait_for_skills_snapshot(screen, pilot, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    terminal_selectors = (
        "#skills-service-error",
        "#skills-empty-state",
        "#skills-local-summary",
    )
    while time.monotonic() < deadline:
        if any(screen.query(selector) for selector in terminal_selectors):
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for Skills snapshot. Visible text: {_visible_text(screen)}")


async def _wait_for_selector(screen, pilot, selector: str, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if screen.query(selector):
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for {selector}. Visible text: {_visible_text(screen)}")


async def _wait_for_visible_text(screen, pilot, expected_text: str, *, timeout: float = 4.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if expected_text in _visible_text(screen):
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for {expected_text!r}. Visible text: {_visible_text(screen)}")


async def _wait_for_mock_call(mock: Mock, pilot, *, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if mock.call_count:
            return
        await pilot.pause()
    raise AssertionError("Timed out waiting for mock call")


async def _wait_for_skills_list_calls(
    service: StaticSkillsScopeService,
    count: int,
    pilot,
    *,
    timeout: float = 1.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(service.calls) >= count:
            return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for {count} Skills list calls; saw {len(service.calls)}")


async def _wait_for_route(seen_routes: list[str], target_route: str, pilot, *, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if seen_routes and seen_routes[-1] == target_route:
            return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for route {target_route!r}; seen routes: {seen_routes!r}")


def _assert_policy_recovery_copy(
    *,
    visible_text: str,
    button: Button,
    status_label: str,
    unavailable_what: str,
    why: str,
    next_action: str,
    recovery_action: str,
    authority_owner: str,
) -> None:
    assert status_label in visible_text
    assert f"Unavailable: {unavailable_what}." in visible_text
    assert f"Why: {why}." in visible_text
    assert f"Next: {next_action}" in visible_text
    assert f"Recovery: {recovery_action}." in visible_text
    assert f"Owner: {authority_owner}." in visible_text
    assert button.disabled is True
    assert why in str(button.tooltip)
    assert next_action in str(button.tooltip)


def _custom_policy_recovery_state(exc, *, unavailable_what, stable_selector, policy_message=None):
    return DestinationRecoveryState(
        status_label="Custom policy state",
        unavailable_what=unavailable_what,
        why=policy_message or exc.user_message,
        next_action="Use the custom recovery target.",
        recovery_action="Custom recovery",
        authority_owner=exc.authority_owner,
        stable_selector=f"custom-{stable_selector}",
        disabled_tooltip="Custom policy tooltip.",
    )


@pytest.mark.parametrize(
    ("route", "title_id", "purpose_text"),
    [
        # Library's Console-rail redesign replaced the shared
        # title+purpose destination header with its own header line
        # (#library-header-line); the landing canvas still carries a
        # `.destination-purpose` line, just with Library's own copy instead
        # of the generic "source material" phrasing artifacts/personas use.
        ("library", "#library-header-line", "content type"),
        ("artifacts", "#artifacts-title", "generated"),
        ("personas", "#personas-title", "behavior"),
    ],
)
@pytest.mark.asyncio
async def test_primary_destination_wrappers_mount(route, title_id, purpose_text):
    app = _build_test_app()
    host = DestinationHarness(app, route)

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_destination_screen(host)

        title = screen.query_one(title_id)
        assert title
        if route != "library":
            assert title.has_class("ds-destination-header")
        assert purpose_text in _static_text(screen.query_one(".destination-purpose", Static)).lower()
        assert screen.query_one(".ds-panel")


@pytest.mark.asyncio
async def test_watchlists_collections_uses_compact_title_and_clear_sections():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_destination_screen(host)

        assert (
            _static_text(screen.query_one("#watchlists-collections-title", Static))
            == "Watchlists | Monitored sources, runs, alerts, recovery | Mixed | Local/Server"
        )
        visible_text = _visible_text(screen)
        assert "Watchlists" in visible_text
        assert "Collections" not in visible_text


@pytest.mark.asyncio
async def test_watchlists_collections_lists_local_snapshot_from_services():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService(
        [
            {"title": "Research feeds", "id": 1, "record_id": "local:subscription:1"},
            {"name": "Vendor changelogs", "id": 2, "record_id": "local:subscription:2"},
        ]
    )
    app.media_reading_scope_service = StaticReadItLaterScopeService(
        [
            {"title": "Saved article", "id": 10, "record_id": "local:media:10"},
        ]
    )
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_wc_snapshot(screen, pilot)
        text = _visible_text(screen)
        button = screen.query_one("#wc-attach-to-console", Button)

        assert "Local Watchlists snapshot" in text
        assert "Watchlists (showing up to 5): 2" in text
        assert "Research feeds" in text
        assert "Vendor changelogs" in text
        assert "Saved article" not in text
        assert button.disabled is False

    assert app.watchlist_scope_service.calls[0] == {
        "runtime_backend": "local",
        "limit": getattr(wc_screen_module, "WC_LOCAL_PAGE_SIZE", None),
        "offset": 0,
    }
    assert app.media_reading_scope_service.calls == []


@pytest.mark.asyncio
async def test_watchlists_collections_empty_state_disables_console_attach():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    app.media_reading_scope_service = StaticReadItLaterScopeService([])
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_wc_snapshot(screen, pilot)
        button = screen.query_one("#wc-attach-to-console", Button)

        assert "No local Watchlists are available yet." in _visible_text(screen)
        assert button.disabled is True
        assert "Stage local Watchlists context" in str(button.tooltip)


@pytest.mark.asyncio
async def test_watchlists_collections_service_failure_uses_recovery_copy():
    app = _build_test_app()
    app.watchlist_scope_service = RaisingWatchlistsScopeService()
    app.media_reading_scope_service = StaticReadItLaterScopeService([])
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_wc_snapshot(screen, pilot)
        button = screen.query_one("#wc-attach-to-console", Button)

        assert "Watchlists services unavailable; retry Watchlists later." in _visible_text(screen)
        assert button.disabled is True
        assert "Watchlists services are unavailable" in str(button.tooltip)


@pytest.mark.asyncio
async def test_watchlists_collections_loading_times_out_to_recovery_copy():
    app = _build_test_app()
    app.watchlist_scope_service = HangingWatchlistsScopeService()
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#wc-service-error", timeout=3.0)
        button = screen.query_one("#wc-attach-to-console", Button)

        assert "Watchlists services unavailable; retry Watchlists later." in _visible_text(screen)
        assert button.disabled is True


@pytest.mark.asyncio
async def test_watchlists_collections_initial_load_uses_distinct_loading_copy(monkeypatch):
    monkeypatch.setattr(wc_screen_module, "WC_SNAPSHOT_TIMEOUT_SECONDS", 30.0)
    app = _build_test_app()
    app.watchlist_scope_service = HangingWatchlistsScopeService()
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#wc-loading-state", timeout=2.0)

        text = _visible_text(screen)
        assert "Loading local Watchlists snapshot..." in text
        assert "Watchlists services unavailable; retry Watchlists later." not in text


@pytest.mark.asyncio
async def test_watchlists_collections_policy_denial_uses_runtime_recovery_taxonomy():
    app = _build_test_app()
    app.watchlist_scope_service = PolicyDeniedWatchlistsScopeService()
    app.media_reading_scope_service = StaticReadItLaterScopeService([])
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_wc_snapshot(screen, pilot)
        error = screen.query_one("#wc-service-error", Static)
        button = screen.query_one("#wc-attach-to-console", Button)

        _assert_policy_recovery_copy(
            visible_text=_static_text(error),
            button=button,
            status_label="Server session expired",
            unavailable_what="Stage Watchlists context in Console",
            why="The Watchlists server session has expired",
            next_action="Re-authenticate the active server profile before retrying.",
            recovery_action="Settings",
            authority_owner="active server",
        )


@pytest.mark.asyncio
async def test_watchlists_collections_policy_denial_uses_recovery_state_selector(monkeypatch):
    monkeypatch.setattr(
        wc_screen_module,
        "policy_denied_recovery_state",
        _custom_policy_recovery_state,
    )
    app = _build_test_app()
    app.watchlist_scope_service = PolicyDeniedWatchlistsScopeService()
    app.media_reading_scope_service = StaticReadItLaterScopeService([])
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#custom-wc-service-error")
        button = screen.query_one("#wc-attach-to-console", Button)

        assert "Custom policy state" in _visible_text(screen)
        assert button.tooltip == "Custom policy tooltip."


@pytest.mark.asyncio
async def test_watchlists_collections_attach_to_console_uses_listed_context():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService(
        [{"title": "Research feeds", "id": 1, "record_id": "local:subscription:1"}]
    )
    app.media_reading_scope_service = StaticReadItLaterScopeService(
        [{"title": "Saved article", "id": 10, "record_id": "local:media:10"}]
    )
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_wc_snapshot(screen, pilot)
        await pilot.click("#wc-attach-to-console")
        await _wait_for_mock_call(app.open_chat_with_handoff, pilot)

    payload = app.open_chat_with_handoff.call_args.args[0]
    assert isinstance(payload, ChatHandoffPayload)
    assert payload.source == "watchlists_collections"
    assert payload.item_type == "wc-context"
    assert payload.title == "Local Watchlists snapshot"
    assert "Research feeds" in payload.body
    assert "Saved article" not in payload.body
    assert payload.metadata["watchlist_count"] == 1
    assert "collection_count" not in payload.metadata


@pytest.mark.asyncio
async def test_watchlists_collections_preserves_safe_comparison_titles_and_rejects_dangerous_text():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService(
        [{"title": "Model A < Model B > Baseline", "id": 1}]
    )
    app.media_reading_scope_service = StaticReadItLaterScopeService(
        [{"title": "javascript:alert(1)", "id": 10}]
    )
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_wc_snapshot(screen, pilot)
        text = _visible_text(screen)

        assert "Model A < Model B > Baseline" in text
        assert "javascript:alert(1)" not in text
        assert "alert(1)" not in text

        await pilot.click("#wc-attach-to-console")
        await _wait_for_mock_call(app.open_chat_with_handoff, pilot)

    payload = app.open_chat_with_handoff.call_args.args[0]
    assert "Model A < Model B > Baseline" in payload.body
    assert "javascript:alert(1)" not in payload.body
    assert "alert(1)" not in payload.body


# The legacy thin-shell Personas tests that lived here were retired with the
# snapshot shell; the workbench contract lives in Tests/UI/test_personas_workbench.py.
# The adoption-tracking evidence test was retired alongside dev's QA evidence archive.


@pytest.mark.asyncio
async def test_library_exposes_source_sections_and_import_export_boundary():
    app = _build_test_app()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(160, 40)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        # The retired hub's per-source open buttons and mode-search chip are
        # dead; the rail rows are the surviving reachable surface for the
        # same source sections and Search/RAG mode.
        for selector in [
            "#library-row-browse-notes",
            "#library-row-browse-media",
            "#library-row-browse-conversations",
            "#library-row-ingest-import-export",
            "#library-row-browse-search",
        ]:
            assert screen.query_one(selector)

        # The Import/Export ownership-boundary copy survives on the mode
        # canvas reached from the Ingest ▸ Import/Export row.
        screen.query_one("#library-row-ingest-import-export", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert (
            "Library owns source acquisition framing; Ingest and Media own deeper file handling."
            in _visible_text(screen)
        )


@pytest.mark.asyncio
async def test_library_destination_lists_local_source_snapshot_from_services():
    app = _build_test_app()
    app.notes_user_id = "unit-user"
    app.notes_scope_service = StaticLibraryNotesScopeService(
        [
            {"title": "Research Note", "id": "note-1"},
            {"title": "Meeting Note", "id": "note-2"},
        ]
    )
    app.media_reading_scope_service = StaticLibraryMediaScopeService(
        [{"title": "Transcript A", "id": "media-1"}]
    )
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(
        [{"title": "Planning Chat", "id": "chat-1"}]
    )
    _link_library_items_to_active_workspace(
        app,
        (
            ("note", "note-1", "Research Note"),
            ("note", "note-2", "Meeting Note"),
            ("media", "media-1", "Transcript A"),
            ("conversation", "chat-1", "Planning Chat"),
        ),
    )
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        await _wait_for_visible_text(screen, pilot, "Notes (2)")
        text = _visible_text(screen)
        button = screen.query_one("#library-use-in-console", Button)

        # The retired "Library snapshot" hub copy has no successor; the rail
        # row counts are the surviving surface for "services feed real
        # counts" and the linked-workspace items keep Console handoff live.
        assert "Notes (2)" in text
        assert "Media (1)" in text
        assert "Conversations (1)" in text
        assert button.disabled is False

    assert app.notes_scope_service.calls[0] == {
        "scope": "local_note",
        "limit": getattr(library_screen_module, "LIBRARY_SOURCE_PAGE_SIZE", None),
        "offset": 0,
        "user_id": "unit-user",
    }
    assert app.media_reading_scope_service.calls[0] == {
        "mode": "local",
        "page": 1,
        "results_per_page": getattr(library_screen_module, "LIBRARY_SOURCE_PAGE_SIZE", None),
        "include_keywords": False,
    }
    assert app.chat_conversation_scope_service.calls[0] == {
        "mode": "local",
        "limit": getattr(library_screen_module, "LIBRARY_SOURCE_PAGE_SIZE", None),
        "offset": 0,
    }


@pytest.mark.asyncio
async def test_library_destination_empty_state_disables_console_handoff():
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        button = screen.query_one("#library-use-in-console", Button)
        text = _visible_text(screen)

        # The retired "No local Library content yet." hub copy has no
        # successor as standalone canvas text; the landing canvas purpose
        # line plus the zero-state rail row counts carry the same "there is
        # nothing here yet" signal, and Console handoff stays disabled.
        assert "Search, pick a content type, or ingest something new." in text
        assert "Notes (0)" in text
        assert "Media (0)" in text
        assert "Conversations (0)" in text
        assert screen.query_one("#library-canvas-landing")
        assert button.disabled is True
        assert "Stage Library source context" in str(button.tooltip)


@pytest.mark.asyncio
async def test_library_destination_labels_plain_list_notes_as_sample_snapshot():
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService(
        [{"title": f"Research Note {index}", "id": f"note-{index}"} for index in range(1, 7)]
    )
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.open_chat_with_handoff = Mock()
    _link_library_items_to_active_workspace(
        app,
        tuple(
            ("note", f"note-{index}", f"Research Note {index}")
            for index in range(1, 6)
        ),
    )
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        await _wait_for_visible_text(screen, pilot, "Notes (5+)")
        text = _visible_text(screen)

        # The retired hub's "Notes: 5+" line has no successor as standalone
        # canvas text; the rail row count suffix is the surviving surface
        # that distinguishes a capped sample ("(5+)") from an exact count
        # ("(n)"). The sample-vs-exact contract itself is asserted below via
        # the Console handoff payload, which still carries the exact titles.
        assert "Notes (5+)" in text
        assert "Notes (5)" not in text

        screen.query_one("#library-use-in-console", Button).press()
        await pilot.pause(0.1)

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]
    assert "Notes (showing up to 5): 5" in payload.body
    assert "Notes: 5" not in payload.body
    assert payload.metadata["notes_sample_count"] == 5
    assert payload.metadata["notes_total_count"] is None
    assert "notes_count" not in payload.metadata
    assert payload.metadata["note_titles"] == [
        "Research Note 1",
        "Research Note 2",
        "Research Note 3",
        "Research Note 4",
        "Research Note 5",
    ]


@pytest.mark.asyncio
async def test_library_destination_service_failure_uses_recovery_copy():
    app = _build_test_app()
    app.notes_scope_service = RaisingLibraryNotesScopeService()
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = _active_destination_screen(host)
        button = screen.query_one("#library-use-in-console", Button)

        assert "Library source services unavailable; retry Library later." in _visible_text(screen)
        assert button.disabled is True
        assert "Library source services are unavailable" in str(button.tooltip)


@pytest.mark.asyncio
async def test_library_policy_denial_uses_runtime_recovery_taxonomy():
    app = _build_test_app()
    app.notes_scope_service = PolicyDeniedLibraryNotesScopeService()
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        # The retired #library-source-error Static never mounts under the
        # rail shell; the same lookup-error taxonomy copy now renders on the
        # Details rail body instead (see test_library_status_row_preserves_
        # policy_recovery_status in test_product_maturity_phase3_library_
        # contract_layout.py for the sibling re-anchor of this surface).
        screen.query_one("#console-rail-section-toggle-library-details", Button).press()
        await pilot.pause()
        await pilot.pause()
        details_body = screen.query_one("#library-details-body", Static)
        button = screen.query_one("#library-use-in-console", Button)

        _assert_policy_recovery_copy(
            visible_text=_static_text(details_body),
            button=button,
            status_label="Wrong source",
            unavailable_what="Use Library sources in Console",
            why="Server Library sources require server mode",
            next_action="Switch to the required source, then retry this workflow.",
            recovery_action="Source switch or Settings",
            authority_owner="active server",
        )


@pytest.mark.asyncio
async def test_library_policy_denial_uses_recovery_state_selector(monkeypatch):
    monkeypatch.setattr(
        library_screen_module,
        "policy_denied_recovery_state",
        _custom_policy_recovery_state,
    )
    app = _build_test_app()
    app.notes_scope_service = PolicyDeniedLibraryNotesScopeService()
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        # The retired #custom-library-source-error id (a stable_selector-
        # derived dynamic id) never mounts under the rail shell; the custom
        # recovery state's copy renders on the surviving Details rail body
        # surface instead, and the disabled tooltip still reflects it.
        screen.query_one("#console-rail-section-toggle-library-details", Button).press()
        await pilot.pause()
        await pilot.pause()
        details_body = screen.query_one("#library-details-body", Static)
        button = screen.query_one("#library-use-in-console", Button)

        assert "Custom policy state" in _static_text(details_body)
        assert button.tooltip == "Custom policy tooltip."


@pytest.mark.asyncio
async def test_library_use_in_console_uses_source_snapshot_context():
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService(
        [{"title": "Research Note", "id": "note-1"}]
    )
    app.media_reading_scope_service = StaticLibraryMediaScopeService(
        [{"title": "Transcript A", "id": "media-1"}]
    )
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(
        [{"title": "Planning Chat", "id": "chat-1"}]
    )
    app.open_chat_with_handoff = Mock()
    _link_library_items_to_active_workspace(
        app,
        (
            ("note", "note-1", "Research Note"),
            ("media", "media-1", "Transcript A"),
            ("conversation", "chat-1", "Planning Chat"),
        ),
    )
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        # #library-use-in-console now lives in the Workspaces body under the
        # Details rail section, which starts collapsed and, once open, can
        # scroll the button below the visible viewport; press it directly
        # rather than simulating a pointer click at an on-screen offset.
        screen.query_one("#console-rail-section-toggle-library-details", Button).press()
        await pilot.pause()
        await pilot.pause()
        screen.query_one("#library-use-in-console", Button).press()
        await pilot.pause(0.1)

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]
    assert isinstance(payload, ChatHandoffPayload)
    assert payload.source == "library"
    assert payload.item_type == "library-source-snapshot"
    assert payload.title == "Local Library Sources"
    assert "Notes: 1" in payload.body
    assert "Media: 1" in payload.body
    assert "Conversations: 1" in payload.body
    assert "Research Note" in payload.body
    assert "Transcript A" in payload.body
    assert "Planning Chat" in payload.body
    assert "Stage Library source material" not in payload.body
    assert payload.metadata["notes_count"] == 1
    assert payload.metadata["media_count"] == 1
    assert payload.metadata["conversations_count"] == 1


@pytest.mark.parametrize(
    ("route", "selector", "target_route"),
    [
        # The retired hub's #library-open-notes button is dead; the Browse ▸
        # Notes rail row is the surviving surface that still emits the same
        # NavigateToScreen compatibility route. Browse ▸ Media is no longer
        # a screen-route row -- it is a canvas row like Conversations now
        # (see test_library_media_action_switches_to_native_mode_without_route_handoff
        # below); the media SCREEN route lives behind the canvas's own
        # "Open in Media" action.
        ("library", "#library-row-browse-notes", "notes"),
        ("artifacts", "#artifacts-open-chatbooks", "chatbooks"),
        # The Personas destination is now a native workbench and no longer
        # hands off to the legacy CCP route via #personas-open-profiles.
        ("watchlists_collections", "#wc-open-watchlists", "subscriptions"),
    ],
)
@pytest.mark.asyncio
async def test_destination_action_buttons_emit_compatibility_routes(route, selector, target_route):
    app = _build_test_app()
    seen_routes = []
    host = DestinationHarness(app, route, seen_routes)

    async with host.run_test(size=(160, 40)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, selector)
        screen.query_one(selector, Button).press()
        await _wait_for_route(seen_routes, target_route, pilot)

    assert seen_routes[-1] == target_route


@pytest.mark.asyncio
async def test_library_media_action_switches_to_native_mode_without_route_handoff():
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    seen_routes = []
    host = DestinationHarness(app, "library", seen_routes)

    async with host.run_test(size=(160, 40)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        # The retired hub's #library-open-media button is dead; the Browse ▸
        # Media rail row is the surviving trigger, and the media canvas
        # mounting is the successor signal that the native (non-route) mode
        # switch happened. The media SCREEN route now lives behind the
        # canvas's own "Open in Media" action.
        screen.query_one("#library-row-browse-media", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-canvas")

        assert getattr(screen, "_active_mode") == "media"
        assert screen.query_one("#library-media-canvas")

    assert seen_routes == []


@pytest.mark.asyncio
async def test_library_conversations_action_switches_to_native_mode_without_route_handoff():
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    seen_routes = []
    host = DestinationHarness(app, "library", seen_routes)

    async with host.run_test(size=(160, 40)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        # The retired #library-open-conversations button and its
        # #library-conversations-browser-title / "Conversations mode" copy
        # are dead; the Browse ▸ Conversations rail row is the surviving
        # trigger, and the conversations canvas mounting is the successor
        # signal that the native (non-route) mode switch happened.
        screen.query_one("#library-row-browse-conversations", Button).press()
        await _wait_for_selector(screen, pilot, "#library-conversations-canvas")

        assert getattr(screen, "_active_mode") == "conversations"
        assert screen.query_one("#library-conversations-canvas")

    assert seen_routes == []


@pytest.mark.asyncio
async def test_library_import_export_action_switches_to_native_mode_without_route_handoff():
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    seen_routes = []
    host = DestinationHarness(app, "library", seen_routes)

    async with host.run_test(size=(160, 40)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        # The retired #library-open-import-export button is dead; the
        # Ingest ▸ Import/Export rail row is the surviving trigger for the
        # same native mode switch.
        screen.query_one("#library-row-ingest-import-export", Button).press()
        await _wait_for_selector(screen, pilot, "#library-import-export-workflow-title")

        assert getattr(screen, "_active_mode") == "import-export"
        assert "Import/Export mode" in _visible_text(screen)

    assert seen_routes == []


@pytest.mark.asyncio
async def test_library_import_export_dedicated_import_action_emits_ingest_route():
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    seen_routes = []
    host = DestinationHarness(app, "library", seen_routes)

    async with host.run_test(size=(160, 40)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        # The in-canvas #library-import-export-open-ingest button lived only
        # in the never-mounted #library-action-region and is dead; the
        # always-reachable Ingest ▸ Import media rail row is the surviving
        # trigger for the same Ingest route (no Import/Export mode entry
        # required first -- the rail row is visible regardless of mode).
        screen.query_one("#library-row-ingest-import-media", Button).press()
        await _wait_for_route(seen_routes, "ingest", pilot)

    assert seen_routes[-1] == "ingest"


@pytest.mark.asyncio
async def test_library_search_action_switches_to_search_mode_without_route_handoff():
    app = _build_test_app()
    seen_routes = []
    host = DestinationHarness(app, "library", seen_routes)

    async with host.run_test(size=(160, 40)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        # The retired #library-mode-search chip and its "Library |
        # Search/RAG |" title text are dead (no mode-chip strip is rendered
        # at all); the Browse ▸ Search/RAG rail row is the surviving
        # trigger, and the Search/RAG panel mounting in the canvas is the
        # successor signal for the native (non-route) mode switch.
        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        assert getattr(screen, "_active_mode") == "search"

    assert seen_routes == []


@pytest.mark.asyncio
async def test_acp_missing_runtime_explains_acp_owned_setup_recovery():
    app = _build_test_app()
    host = DestinationHarness(app, "acp")

    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        screen = _active_destination_screen(host)
        visible_text = _visible_text(screen)
        # The recovery copy must not route users to Settings. The shared top
        # nav legitimately carries a Settings destination button, so exclude
        # nav chrome before asserting the absence.
        non_nav_text = " ".join(
            [
                *(
                    _static_text(widget)
                    for widget in screen.query(Static)
                    if widget.display and hasattr(widget, "renderable")
                ),
                *(
                    str(button.label)
                    for button in screen.query(Button)
                    if button.display
                    and button.label is not None
                    and not (button.id or "").startswith("nav-")
                ),
            ]
        )
        follow_button = screen.query_one("#acp-follow-in-console", Button)
        launch_button = screen.query_one("#acp-launch-agent", Button)

    assert "Runtime not configured" in visible_text
    assert "Configure ACP runtime setup in ACP before launch." in visible_text
    assert "Runtime owner: ACP" in visible_text
    assert "Settings" not in non_nav_text
    assert follow_button.disabled is True
    assert launch_button.disabled is True


def test_acp_runtime_session_state_preserves_numeric_zero_fields():
    state = ACPRuntimeSessionState.from_any(
        {
            "runtime_id": 0,
            "runtime_label": 0,
            "runtime_version": 0,
            "session_id": 0,
            "session_title": 0,
            "session_status": 0,
            "session_payload": {"ok": True},
        }
    )

    assert state.runtime_id == "0"
    assert state.runtime_label == "0"
    assert state.runtime_version == "0"
    assert state.session_id == "0"
    assert state.session_title == "0"
    assert state.session_status == "0"


def test_app_acp_runtime_session_state_helper_normalizes_manager_snapshot():
    app = _build_test_app()
    app.acp_runtime_session_state = None
    app.acp_runtime_process_manager = Mock()
    app.acp_runtime_process_manager.snapshot.return_value = {
        "runtime_id": "codex-local",
        "runtime_label": "Codex local ACP",
        "runtime_version": "0.1",
        "session_id": "session-1",
        "session_title": "Research agent",
        "session_status": "running",
        "session_payload": {"pid": 1234},
    }

    state = app.get_acp_runtime_session_state()

    assert isinstance(state, ACPRuntimeSessionState)
    assert state.session_id == "session-1"
    assert state.session_payload == {"pid": 1234}


def test_acp_runtime_button_handlers_schedule_workers_without_blocking_manager():
    app = _build_test_app()
    app.acp_runtime_process_manager = Mock()
    screen = ACPScreen(app)
    screen._launch_acp_runtime_worker = Mock()
    screen._stop_acp_runtime_worker = Mock()
    event = Mock()

    screen.launch_acp_runtime(event)
    screen.stop_acp_runtime(event)

    assert event.stop.call_count == 2
    screen._launch_acp_runtime_worker.assert_called_once_with("ACP agent session")
    screen._stop_acp_runtime_worker.assert_called_once_with()
    app.acp_runtime_process_manager.start_session.assert_not_called()
    app.acp_runtime_process_manager.stop.assert_not_called()


@pytest.mark.asyncio
async def test_acp_configured_runtime_without_session_disables_console_follow():
    app = _build_test_app()
    app.acp_runtime_session_state = {
        "runtime_id": "codex-local",
        "runtime_label": "Codex local ACP",
        "runtime_version": "0.1",
        "session_id": None,
    }
    host = DestinationHarness(app, "acp")

    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        screen = _active_destination_screen(host)
        visible_text = _visible_text(screen)
        follow_button = screen.query_one("#acp-follow-in-console", Button)

    assert "Runtime configured" in visible_text
    assert "Runtime: Codex local ACP" in visible_text
    assert "ACP version: 0.1" in visible_text
    assert "Session: none" in visible_text
    assert "Console follow disabled: no ACP session payload" in visible_text
    assert "Start or resume an ACP session in ACP before following it in Console." in visible_text
    assert "Runtime owner: ACP" in visible_text
    assert follow_button.disabled is True


@pytest.mark.asyncio
async def test_acp_configured_runtime_process_enables_launch_and_creates_session_payload():
    app = _build_test_app()
    app.acp_runtime_process_manager = Mock()
    app.acp_runtime_process_manager.snapshot.return_value = {
        "status": "configured",
        "runtime_id": "codex-local",
        "runtime_label": "Codex local ACP",
        "runtime_version": "0.1",
        "launch_available": True,
        "recovery": "Launch ACP runtime.",
    }
    app.acp_runtime_process_manager.start_session.return_value = Mock(
        status="running",
        recovery="ACP runtime is running.",
        session_state=ACPRuntimeSessionState(
            runtime_id="codex-local",
            runtime_label="Codex local ACP",
            runtime_version="0.1",
            session_id="session-1",
            session_title="Research agent",
            session_status="running",
            session_payload={"pid": 1234, "command": "acp-runtime"},
        ),
    )
    host = DestinationHarness(app, "acp")

    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        screen = _active_destination_screen(host)
        launch_button = screen.query_one("#acp-launch-agent", Button)
        assert launch_button.disabled is False

        await pilot.click("#acp-launch-agent")
        await _wait_for_mock_call(app.acp_runtime_process_manager.start_session, pilot)
        await pilot.pause()

    app.acp_runtime_process_manager.start_session.assert_called_once()
    assert app.acp_runtime_session_state.session_id == "session-1"
    assert app.acp_runtime_session_state.has_console_session_payload is True


@pytest.mark.asyncio
async def test_acp_failed_runtime_process_surfaces_recovery_and_restart_action():
    app = _build_test_app()
    app.acp_runtime_process_manager = Mock()
    app.acp_runtime_process_manager.snapshot.return_value = {
        "status": "failed",
        "runtime_id": "codex-local",
        "runtime_label": "Codex local ACP",
        "runtime_version": "0.1",
        "launch_available": True,
        "recovery": "ACP runtime exited before it became ready.",
    }
    host = DestinationHarness(app, "acp")

    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        screen = _active_destination_screen(host)
        visible_text = _visible_text(screen)
        restart_button = screen.query_one("#acp-restart-runtime", Button)

    assert "Runtime state: failed" in visible_text
    assert "ACP runtime exited before it became ready." in visible_text
    assert restart_button.disabled is False


@pytest.mark.asyncio
async def test_acp_runtime_and_session_labels_are_markup_escaped():
    app = _build_test_app()
    app.acp_runtime_session_state = {
        "runtime_id": "runtime-1",
        "runtime_label": "[bold]Runtime[/bold]",
        "runtime_version": "1",
        "session_id": "session-1",
        "session_title": "[red]Session[/red]",
        "session_payload": {"thread_id": "thread-1"},
    }
    host = DestinationHarness(app, "acp")

    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        screen = _active_destination_screen(host)

        assert len(screen.query("#acp-agent-codex-local")) == 0
        assert len(screen.query("#acp-no-sessions")) == 0
        assert len(screen.query("#acp-runtime-blocked")) == 0
        assert screen.query_one("#acp-session-list-row", Static).renderable == (
            "> \\[red]Session\\[/red] (pending) (console-ready)"
        )
        assert screen.query_one("#acp-runtime-display", Static).renderable == (
            "  Runtime: \\[bold]Runtime\\[/bold]"
        )
        assert screen.query_one("#acp-session-status", Static).renderable == "  Session: \\[red]Session\\[/red]"
        assert screen.query_one("#acp-runtime-status", Static).renderable == (
            "  Runtime configured: \\[bold]Runtime\\[/bold]"
        )
        assert screen.query_one("#acp-runtime-summary", Static).renderable == (
            "Runtime: \\[bold]Runtime\\[/bold]"
        )
        assert screen.query_one("#acp-session-summary", Static).renderable == (
            "Session: \\[red]Session\\[/red]"
        )
        assert screen.query_one("#acp-session-ready", Static).renderable == (
            "Session ready: \\[red]Session\\[/red]"
        )


@pytest.mark.asyncio
async def test_acp_session_payload_enables_console_follow_live_work_handoff():
    app = _build_test_app()
    app.acp_runtime_session_state = {
        "runtime_id": "codex-local",
        "runtime_label": "Codex local ACP",
        "runtime_version": "0.1",
        "session_id": "session-1",
        "session_title": "Research agent",
        "session_status": "running",
        "session_payload": {
            "thread_id": "thread-1",
            "workspace": "docs",
        },
    }
    app.open_console_for_live_work = Mock()
    host = DestinationHarness(app, "acp")

    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        screen = _active_destination_screen(host)
        visible_text = _visible_text(screen)
        follow_button = screen.query_one("#acp-follow-in-console", Button)
        await pilot.click("#acp-follow-in-console")
        await _wait_for_mock_call(app.open_console_for_live_work, pilot)

    assert "Session ready: Research agent" in visible_text
    assert "Console follow ready: session payload available" in visible_text
    assert follow_button.disabled is False
    app.open_console_for_live_work.assert_called_once()
    kwargs = app.open_console_for_live_work.call_args.kwargs
    assert kwargs["source"] == "ACP"
    assert kwargs["title"] == "Research agent"
    assert kwargs["status"] == "running"
    assert kwargs["action_label"] == "Follow ACP session"
    assert kwargs["payload"]["target_id"] == "local:acp_session:session-1"
    assert kwargs["payload"]["session_id"] == "session-1"
    assert kwargs["payload"]["runtime_id"] == "codex-local"
    assert kwargs["payload"]["session_payload"] == {
        "thread_id": "thread-1",
        "workspace": "docs",
    }


@pytest.mark.asyncio
async def test_acp_running_runtime_presents_actionable_hierarchy_without_dead_actions():
    app = _build_test_app()
    app.acp_runtime_session_state = {
        "runtime_id": "codex-local",
        "runtime_label": "Codex local ACP",
        "runtime_version": "0.1",
        "session_id": "session-1",
        "session_title": "Research agent",
        "session_status": "running",
        "session_payload": {
            "pid": 1234,
            "started_at": "2026-05-22T12:00:00",
        },
    }
    app.acp_runtime_process_manager = Mock()
    app.acp_runtime_process_manager.snapshot.return_value = {
        "status": "running",
        "runtime_id": "codex-local",
        "runtime_label": "Codex local ACP",
        "runtime_version": "0.1",
        "launch_available": False,
        "stop_available": True,
        "recovery": "ACP runtime is running.",
    }
    host = DestinationHarness(app, "acp")

    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        screen = _active_destination_screen(host)
        visible_text = _visible_text(screen)
        follow_button = screen.query_one("#acp-follow-in-console", Button)
        stop_button = screen.query_one("#acp-stop-runtime", Button)
        list_pane = screen.query_one("#acp-list-pane")
        detail_pane = screen.query_one("#acp-detail-pane")
        inspector_pane = screen.query_one("#acp-inspector-pane")
        session_row = screen.query_one("#acp-session-list-row")

    assert "State: Running · Console-ready" in visible_text
    assert "Active Session" in visible_text
    assert "> Research agent (running) (console-ready)" in visible_text
    assert "Diffs: not supported by current runtime payload" in visible_text
    assert "Terminal: no terminal stream attached" in visible_text
    assert "Primary action" in visible_text
    assert "Runtime controls" in visible_text
    assert "Process: pid 1234" in visible_text
    assert "Started: 2026-05-22T12:00:00" in visible_text
    assert "Handoff ID: session-1" in visible_text
    assert "Console target:" not in visible_text
    assert list_pane.has_class("acp-framed-pane")
    assert detail_pane.has_class("acp-framed-pane")
    assert inspector_pane.has_class("acp-framed-pane")
    assert session_row.has_class("acp-selected-session-row")
    assert len(screen.query("#acp-launch-agent")) == 0
    assert len(screen.query("#acp-restart-runtime")) == 0
    assert follow_button.disabled is False
    assert stop_button.disabled is False


@pytest.mark.parametrize("route", SCREEN_BY_ROUTE)
@pytest.mark.asyncio
async def test_destination_action_buttons_explain_their_outcome(route):
    if route == "personas":
        pytest.skip(
            "Several ds-native Personas workbench buttons (card Edit, editor "
            "Save/Cancel, preview controls) do not carry tooltips yet; re-enable "
            "this audit when tooltips are added (tracked as a UX follow-up)."
        )
    app = _build_test_app()
    host = DestinationHarness(app, route)

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_destination_screen(host)

        for button in screen.query(Button):
            tooltip = getattr(button, "tooltip", None)
            assert tooltip is not None, button.id
            assert str(tooltip).strip().lower() not in {"", "none"}, button.id


@pytest.mark.parametrize(
    ("route", "expected_sections"),
    [
        ("watchlists_collections", ["Watchlists", "Monitored sources"]),
        ("schedules", ["Next Run", "Paused", "Failed"]),
        ("workflows", ["Recipes", "Dry Run", "Console launch unavailable"]),
    ],
)
@pytest.mark.asyncio
async def test_automation_destination_wrappers_explain_ownership(route, expected_sections):
    app = _build_test_app()
    host = DestinationHarness(app, route)

    async with host.run_test(size=(180, 40)) as pilot:
        screen = _active_destination_screen(host)
        for section in expected_sections:
            await _wait_for_visible_text(screen, pilot, section)


@pytest.mark.asyncio
async def test_schedules_failed_run_exposes_consistent_retry_control_state():
    app = _build_test_app()
    app.home_active_work_adapter = StaticHomeActiveWorkAdapter(
        HomeActiveWorkItem(
            item_id="local:schedule_run:7",
            title="Morning digest",
            source="Schedules",
            status="failed",
            detail_route="schedules",
            console_available=True,
        )
    )
    app.open_active_home_item_in_console = Mock()
    host = DestinationHarness(app, "schedules")

    async with host.run_test(size=(180, 45)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#schedules-console-available")
        visible_text = _visible_text(screen)
        follow_button = screen.query_one("#schedules-follow-in-console", Button)
        retry_button = screen.query_one("#schedules-retry-run", Button)
        pause_button = screen.query_one("#schedules-pause-run", Button)

    assert "Morning digest" in visible_text
    assert "Status: failed" in visible_text
    assert "State: failed" in visible_text
    assert "Retry/backoff: retry available from Schedules" in visible_text
    assert "Run control: retry available" in visible_text
    assert "Next action: retry or open in Console" in visible_text
    assert follow_button.disabled is False
    assert retry_button.disabled is True
    assert str(retry_button.tooltip) == "Retry this schedule run from Schedules when run-control services are available."
    assert pause_button.disabled is True


@pytest.mark.asyncio
async def test_schedules_pending_run_uses_shared_approval_status_taxonomy():
    app = _build_test_app()
    app.home_active_work_adapter = StaticHomeActiveWorkAdapter(
        HomeActiveWorkItem(
            item_id="local:schedule_run:8",
            title="Digest needs approval",
            source="Schedules",
            status="pending",
            detail_route="schedules",
            console_available=True,
        )
    )
    app.open_active_home_item_in_console = Mock()
    host = DestinationHarness(app, "schedules")

    async with host.run_test(size=(180, 45)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#schedules-console-available")
        visible_text = _visible_text(screen)
        approval_button = screen.query_one("#schedules-review-approval", Button)

    assert "Digest needs approval" in visible_text
    assert "Status: pending" in visible_text
    assert "Run control: approval required" in visible_text
    assert "Next action: review approval before Console follow" in visible_text
    assert "Approval review controls are not wired yet" in visible_text
    assert approval_button.disabled is True


@pytest.mark.asyncio
async def test_workflows_approval_pending_run_exposes_review_before_console_state():
    app = _build_test_app()
    app.home_active_work_adapter = StaticHomeActiveWorkAdapter(
        HomeActiveWorkItem(
            item_id="local:workflow_run:9",
            title="Publish newsletter",
            source="Workflows",
            status="pending_approval",
            detail_route="workflows",
            console_available=True,
        )
    )
    app.open_active_home_item_in_console = Mock()
    host = DestinationHarness(app, "workflows")

    async with host.run_test(size=(180, 45)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#workflows-console-available")
        visible_text = _visible_text(screen)
        launch_button = screen.query_one("#workflows-launch-in-console", Button)
        approval_button = screen.query_one("#workflows-review-approval", Button)
        retry_button = screen.query_one("#workflows-retry-run", Button)

    assert "Publish newsletter" in visible_text
    assert "Status: pending_approval" in visible_text
    assert "State: pending_approval" in visible_text
    assert "Approvals: pending" in visible_text
    assert "Run control: approval required" in visible_text
    assert "Next action: review approval before Console follow" in visible_text
    assert launch_button.disabled is False
    assert approval_button.disabled is True
    assert str(approval_button.tooltip) == "Review this workflow approval from Workflows when approval services are available."
    assert retry_button.disabled is True


@pytest.mark.asyncio
async def test_schedules_empty_state_reads_as_live_queue_with_recovery_path():
    app = _build_test_app()
    app.home_active_work_adapter = StaticHomeActiveWorkAdapter()
    host = DestinationHarness(app, "schedules")

    async with host.run_test(size=(180, 45)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#schedules-console-unavailable")
        visible_text = _visible_text(screen)
        list_pane = screen.query_one("#schedules-list-pane")
        detail_pane = screen.query_one("#schedules-detail-pane")
        inspector_pane = screen.query_one("#schedules-inspector-pane")
        control_label = screen.query_one("#schedules-action-state-label", Static)

    for expected in (
        "Next Run 0",
        "Paused 0",
        "Failed 0",
        "Retry 0",
        "History 0",
        "No active schedule run selected",
        "Next action: start or select a schedule run",
        "Recovery controls require an active schedule run",
    ):
        assert expected in visible_text
    assert "destination-workbench-pane" in list_pane.classes
    assert "destination-workbench-pane" in detail_pane.classes
    assert "destination-workbench-pane" in inspector_pane.classes
    assert str(control_label.renderable) == "Recovery controls require an active schedule run"


@pytest.mark.asyncio
async def test_workflows_empty_state_reads_as_live_queue_with_recovery_path():
    app = _build_test_app()
    app.home_active_work_adapter = StaticHomeActiveWorkAdapter()
    host = DestinationHarness(app, "workflows")

    async with host.run_test(size=(180, 45)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#workflows-console-unavailable")
        visible_text = _visible_text(screen)
        list_pane = screen.query_one("#workflows-list-pane")
        detail_pane = screen.query_one("#workflows-detail-pane")
        inspector_pane = screen.query_one("#workflows-inspector-pane")
        control_label = screen.query_one("#workflows-action-state-label", Static)

    for expected in (
        "Recipes 0",
        "Inputs 0",
        "Steps 0",
        "Dry Run 0",
        "Approvals 0",
        "Outputs 0",
        "No active workflow run selected",
        "Next action: start or select a workflow run",
        "Recovery controls require an active workflow run",
    ):
        assert expected in visible_text
    assert "destination-workbench-pane" in list_pane.classes
    assert "destination-workbench-pane" in detail_pane.classes
    assert "destination-workbench-pane" in inspector_pane.classes
    assert str(control_label.renderable) == "Recovery controls require an active workflow run"


@pytest.mark.parametrize(
    ("route", "expected_text"),
    [
        ("mcp", "scoped tools"),
        ("acp", "Agent Client Protocol"),
        ("skills", "SKILL.md"),
        ("settings", "Global preferences"),
    ],
)
@pytest.mark.asyncio
async def test_protocol_and_settings_wrappers_have_distinct_boundaries(route, expected_text):
    app = _build_test_app()
    host = DestinationHarness(app, route)

    async with host.run_test(size=(180, 40)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_visible_text(screen, pilot, expected_text)


@pytest.mark.parametrize(
    ("route", "selector", "copy"),
    [
        ("skills", "#skills-import-skill", "Skill import is not wired in this shell yet."),
    ],
)
@pytest.mark.asyncio
async def test_unwired_destination_actions_are_disabled_with_honest_copy(route, selector, copy):
    app = _build_test_app()
    host = DestinationHarness(app, route)

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_destination_screen(host)

        button = screen.query_one(selector, Button)
        assert button.disabled is True
        assert copy in _visible_text(screen)


@pytest.mark.asyncio
async def test_mcp_destination_embeds_unified_mcp_management_panel():
    app = _build_test_app()
    host = DestinationHarness(app, "mcp")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = _active_destination_screen(host)

        assert screen.query_one(UnifiedMCPPanel)
        assert screen.query_one("#unified-mcp-source", Select)
        assert screen.query_one("#unified-mcp-server-target", Select)
        assert screen.query_one("#unified-mcp-scope", Select)
        assert screen.query_one("#unified-mcp-section", Select)
        assert screen.query_one("#unified-mcp-action", Select)
        assert screen.query_one("#unified-mcp-action-payload", TextArea)
        assert screen.query_one("#unified-mcp-action-run", Button)
        assert not screen.query("#mcp-open-management")
        assert "Unified MCP management is not embedded in this shell yet." not in _visible_text(screen)


@pytest.mark.asyncio
async def test_mcp_destination_labels_server_first_workbench_columns():
    app = _build_test_app()
    host = DestinationHarness(app, "mcp")

    async with host.run_test(size=(180, 50)) as pilot:
        await _wait_for_selector(_active_destination_screen(host), pilot, "#unified-mcp-action-readiness")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "Servers + Scope" in text
        assert "Server Detail" in text
        assert "Readiness + Actions" in text
        assert "Manage MCP servers, scoped tools, permissions, and audit readiness." in text
        assert "Section" in text
        assert "try Inventory" not in text
        assert "Column 1:" not in text
        assert "Column 2:" not in text
        assert "Column 3:" not in text
        assert screen.query_one("#mcp-column-divider-left").has_class("mcp-column-resize-handle")
        assert screen.query_one("#mcp-column-divider-right").has_class("mcp-column-resize-handle")
        assert screen.query_one("#mcp-column-divider-left").tooltip == "Resize columns"
        assert screen.query_one("#mcp-column-divider-right").tooltip == "Resize columns"
        assert "Blocked" in text
        assert "Select Section: Inventory" in text
        assert "Run Action disabled" in text
        assert "Payload (JSON)" not in text


@pytest.mark.asyncio
async def test_mcp_destination_restores_unified_mcp_view_state_after_mount(tmp_path):
    target_store = ConfiguredServerTargetStore(tmp_path / "targets.json")
    target_store.save_targets(
        [ConfiguredServerTarget(server_id="server-a", label="Server A", base_url="https://a.example/api", is_default=True)]
    )
    app = _build_test_app()
    app.unified_mcp_service = FakeUnifiedMCPService(target_store)
    host = DestinationHarness(
        app,
        "mcp",
        restored_state={
            "unified_mcp_view_state": {
                "selected_source": "server",
                "selected_active_server_id": "server-a",
                "selected_scope": "team",
                "selected_scope_ref": "21",
                "selected_section": "inventory",
            }
        },
    )

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        panel = _active_destination_screen(host).query_one(UnifiedMCPPanel)

        assert panel.context.selected_source == "server"
        assert panel.context.selected_active_server_id == "server-a"
        assert panel.context.selected_scope == "team"
        assert panel.context.selected_scope_ref == "21"
        assert panel.context.selected_section == "inventory"


@pytest.mark.asyncio
async def test_mcp_destination_runtime_refresh_uses_exclusive_worker(monkeypatch):
    app = _build_test_app()
    screen = MCPScreen(app)
    scheduled = {}

    class FakePanel:
        async def load_context(self):
            return None

    def capture_worker(coro, **kwargs):
        scheduled["kwargs"] = kwargs
        coro.close()

    screen.mcp_panel = FakePanel()
    monkeypatch.setattr(screen, "run_worker", capture_worker)

    await screen.handle_runtime_backend_changed("server")

    assert scheduled["kwargs"]["name"] == "mcp-screen-runtime-refresh"
    assert scheduled["kwargs"]["group"] == "mcp-screen-runtime-refresh"
    assert scheduled["kwargs"]["exclusive"] is True


def test_skills_screen_public_initializer_is_typed():
    signature = inspect.signature(SkillsScreen.__init__)

    assert signature.parameters["app_instance"].annotation is not inspect.Parameter.empty
    assert signature.parameters["kwargs"].annotation is not inspect.Parameter.empty
    assert signature.return_annotation in {None, "None"}


@pytest.mark.asyncio
async def test_skills_destination_lists_local_skills_from_scope_service():
    app = _build_test_app()
    app.skills_scope_service = StaticSkillsScopeService(
        [
            {
                "name": "summarize-notes",
                "description": "Summarize note collections",
                "argument_hint": "note id",
                "record_id": "local:skill:summarize-notes",
            },
            {
                "name": "code-review",
                "description": "Review code changes",
                "record_id": "local:skill:code-review",
            },
        ]
    )
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_skills_snapshot(screen, pilot)
        text = _visible_text(screen)
        button = screen.query_one("#skills-attach-to-console", Button)

        assert "Installed local skills: 2" in text
        assert "summarize-notes" in text
        assert "Summarize note collections" in text
        assert "code-review" in text
        assert button.disabled is False

    assert app.skills_scope_service.calls[0]["mode"] == "local"
    assert app.skills_scope_service.calls[0]["limit"] == getattr(
        skills_screen_module,
        "SKILLS_LOCAL_PAGE_SIZE",
        None,
    )


@pytest.mark.asyncio
async def test_skills_destination_distinguishes_valid_and_invalid_skill_readiness():
    app = _build_test_app()
    app.skills_scope_service = StaticSkillsScopeService(
        [
            {
                "name": "summarize-notes",
                "description": "Summarize note collections",
                "record_id": "local:skill:summarize-notes",
                "validation_status": "valid",
                "validation_errors": [],
            },
            {
                "name": "broken-skill",
                "description": "Missing valid frontmatter",
                "record_id": "local:skill:broken-skill",
                "validation_status": "invalid",
                "validation_errors": ["description is required"],
            },
        ]
    )
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_skills_snapshot(screen, pilot)
        text = _visible_text(screen)
        attach_button = screen.query_one("#skills-attach-to-console", Button)

        assert "Ready: valid SKILL.md" in text
        assert "Blocked: invalid SKILL.md" in text
        assert "description is required" in text
        assert "Selected: summarize-notes" in text
        assert "Runtime target: local:skill:summarize-notes" in text
        assert "Execution: ready to stage in Console" in text
        assert attach_button.disabled is False

        await pilot.click("#skills-select-local-1")
        await pilot.pause()
        text = _visible_text(screen)

        assert "Selected: broken-skill" in text
        assert "Execution: blocked" in text
        assert "Reason: description is required" in text
        assert attach_button.disabled is True


@pytest.mark.asyncio
async def test_skills_destination_blocks_metadata_valid_trust_blocked_skill():
    app = _build_test_app()
    app.skills_scope_service = StaticSkillsScopeService(
        [
            {
                "name": "summarize-notes",
                "description": "Summarize note collections",
                "record_id": "local:skill:summarize-notes",
                "validation_status": "valid",
                "validation_errors": [],
                "trust_status": "quarantined_modified",
                "trust_reason_code": "skill_modified",
                "trust_blocked": True,
                "trust_changed_files": ["SKILL.md"],
            },
        ]
    )
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_skills_snapshot(screen, pilot)
        text = _visible_text(screen)
        button = screen.query_one("#skills-attach-to-console", Button)

        assert "Trust: changed since trusted baseline" in text
        assert "Reason code: skill_modified" in text
        assert "Changed files: SKILL.md" in text
        assert "Execution: blocked" in text
        assert (
            "Reason: local skill files changed since the trusted baseline. "
            "Next: Review Diff, then Trust Reviewed Version."
        ) in text
        assert "Reason: trust blocked" not in text
        assert button.disabled is True


@pytest.mark.asyncio
async def test_skills_destination_shows_uninitialized_bootstrap_action():
    app = _build_test_app()
    app.skills_scope_service = StaticSkillsScopeService(
        [
            {
                "name": "new-skill",
                "description": "New local skill",
                "record_id": "local:skill:new-skill",
                "validation_status": "valid",
                "validation_errors": [],
                "trust_status": "trust_uninitialized",
                "trust_reason_code": "trust_uninitialized",
                "trust_blocked": True,
            },
        ]
    )
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_skills_snapshot(screen, pilot)
        text = _visible_text(screen)

        assert "Trust: not initialized" in text
        assert screen.query_one("#skills-bootstrap-trust", Button).disabled is False


@pytest.mark.asyncio
async def test_skills_destination_bootstrap_action_calls_trust_service():
    app = _build_test_app()
    app.local_skill_trust_service = RecordingSkillTrustService()
    app.skills_scope_service = StaticSkillsScopeService(
        [
            {
                "name": "new-skill",
                "description": "New local skill",
                "record_id": "local:skill:new-skill",
                "validation_status": "valid",
                "validation_errors": [],
                "trust_status": "trust_uninitialized",
                "trust_reason_code": "trust_uninitialized",
                "trust_blocked": True,
            },
        ]
    )
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_skills_snapshot(screen, pilot)
        initial_list_calls = len(app.skills_scope_service.calls)
        screen._request_skill_trust_passphrase = AsyncMock(return_value="passphrase")
        await pilot.click("#skills-bootstrap-trust")
        await _wait_for_skills_list_calls(app.skills_scope_service, initial_list_calls + 1, pilot)

        assert app.local_skill_trust_service.bootstrap_calls == 1
        assert app.local_skill_trust_service.bootstrap_passphrases == ["passphrase"]


@pytest.mark.asyncio
async def test_skills_destination_unlock_action_calls_trust_service():
    app = _build_test_app()
    app.local_skill_trust_service = RecordingSkillTrustService()
    app.skills_scope_service = StaticSkillsScopeService(
        [
            {
                "name": "locked-skill",
                "description": "Locked local skill",
                "record_id": "local:skill:locked-skill",
                "validation_status": "valid",
                "validation_errors": [],
                "trust_status": "trust_locked",
                "trust_reason_code": "trust_locked",
                "trust_blocked": True,
            },
        ]
    )
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_skills_snapshot(screen, pilot)
        initial_list_calls = len(app.skills_scope_service.calls)
        screen._request_skill_trust_passphrase = AsyncMock(return_value="passphrase")

        unlock_button = screen.query_one("#skills-unlock-trust", Button)
        assert unlock_button.disabled is False

        await pilot.click("#skills-unlock-trust")
        await _wait_for_skills_list_calls(app.skills_scope_service, initial_list_calls + 1, pilot)

        assert app.local_skill_trust_service.unlock_calls == 1
        assert app.local_skill_trust_service.unlock_passphrases == ["passphrase"]


@pytest.mark.asyncio
async def test_skills_destination_review_action_enables_trust_reviewed_version():
    app = _build_test_app()
    app.local_skill_trust_service = RecordingSkillTrustService(review_id="review-1")
    app.skills_scope_service = StaticSkillsScopeService(
        [
            {
                "name": "summarize-notes",
                "description": "Summarize note collections",
                "record_id": "local:skill:summarize-notes",
                "validation_status": "valid",
                "validation_errors": [],
                "trust_status": "quarantined_modified",
                "trust_reason_code": "skill_modified",
                "trust_blocked": True,
                "trust_changed_files": ["SKILL.md"],
            },
        ]
    )
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_skills_snapshot(screen, pilot)
        initial_list_calls = len(app.skills_scope_service.calls)
        await pilot.click("#skills-review-diff")
        await _wait_for_skills_list_calls(app.skills_scope_service, initial_list_calls + 1, pilot)
        text = _visible_text(screen)

        assert app.local_skill_trust_service.reviewed_skill == "summarize-notes"
        assert "Review captured: SKILL.md." in text
        assert "Confirm these current files should become the trusted baseline." in text
        assert "Reviewed skill: summarize-notes" in text
        assert "/SKILL.md" not in text
        assert screen.query_one("#skills-trust-reviewed-version", Button).disabled is False

        list_calls_after_review = len(app.skills_scope_service.calls)
        await pilot.click("#skills-trust-reviewed-version")
        await _wait_for_skills_list_calls(app.skills_scope_service, list_calls_after_review + 1, pilot)
        assert app.local_skill_trust_service.trusted_review_id == "review-1"


@pytest.mark.asyncio
async def test_skills_destination_clears_stale_review_after_refresh():
    app = _build_test_app()
    trusted_skill = {
        "name": "summarize-notes",
        "description": "Summarize note collections",
        "record_id": "local:skill:summarize-notes",
        "validation_status": "valid",
        "validation_errors": [],
        "trust_status": "trusted",
        "trust_reason_code": None,
        "trust_blocked": False,
        "trust_changed_files": [],
    }
    app.skills_scope_service = StaticSkillsScopeService(
        [
            {
                "name": "summarize-notes",
                "description": "Summarize note collections",
                "record_id": "local:skill:summarize-notes",
                "validation_status": "valid",
                "validation_errors": [],
                "trust_status": "quarantined_modified",
                "trust_reason_code": "skill_modified",
                "trust_blocked": True,
                "trust_changed_files": ["SKILL.md"],
            },
        ]
    )
    app.local_skill_trust_service = RecordingSkillTrustService(
        review_id="review-1",
        on_capture_review=lambda _skill_name: app.skills_scope_service.set_skills([trusted_skill]),
    )
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_skills_snapshot(screen, pilot)
        initial_list_calls = len(app.skills_scope_service.calls)

        await pilot.click("#skills-review-diff")
        await _wait_for_skills_list_calls(app.skills_scope_service, initial_list_calls + 1, pilot)
        await _wait_for_visible_text(screen, pilot, "Trust: trusted baseline")

        approve_button = screen.query_one("#skills-trust-reviewed-version", Button)
        assert approve_button.disabled is True
        assert "Review captured:" not in _visible_text(screen)

        await pilot.click("#skills-trust-reviewed-version")
        await pilot.pause()
        assert app.local_skill_trust_service.trust_reviewed_calls == 0


@pytest.mark.asyncio
async def test_skills_destination_hides_review_summary_when_selecting_another_skill():
    app = _build_test_app()
    app.local_skill_trust_service = RecordingSkillTrustService(review_id="review-1")
    app.skills_scope_service = StaticSkillsScopeService(
        [
            {
                "name": "summarize-notes",
                "description": "Summarize note collections",
                "record_id": "local:skill:summarize-notes",
                "validation_status": "valid",
                "validation_errors": [],
                "trust_status": "quarantined_modified",
                "trust_reason_code": "skill_modified",
                "trust_blocked": True,
                "trust_changed_files": ["SKILL.md"],
            },
            {
                "name": "code-review",
                "description": "Review code changes",
                "record_id": "local:skill:code-review",
                "validation_status": "valid",
                "validation_errors": [],
                "trust_status": "quarantined_modified",
                "trust_reason_code": "skill_modified",
                "trust_blocked": True,
                "trust_changed_files": ["SKILL.md"],
            },
        ]
    )
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_skills_snapshot(screen, pilot)
        initial_list_calls = len(app.skills_scope_service.calls)
        await pilot.click("#skills-review-diff")
        await _wait_for_skills_list_calls(app.skills_scope_service, initial_list_calls + 1, pilot)

        assert "Reviewed skill: summarize-notes" in _visible_text(screen)
        assert "Review captured: SKILL.md." in _visible_text(screen)
        assert screen.query_one("#skills-trust-reviewed-version", Button).disabled is False

        await pilot.click("#skills-select-local-1")
        await pilot.pause()
        text = _visible_text(screen)

        assert "Selected: code-review" in text
        assert screen.query_one("#skills-trust-reviewed-version", Button).disabled is True
        assert "Reviewed skill: summarize-notes" not in text
        assert "Review captured: SKILL.md." not in text


@pytest.mark.asyncio
async def test_skills_destination_escapes_selected_skill_metadata_in_inspector():
    app = _build_test_app()
    app.skills_scope_service = StaticSkillsScopeService(
        [
            {
                "name": "[red]summarize-notes[/red]",
                "description": "Summarize note collections",
                "record_id": "local:skill:[red]summarize-notes[/red]",
                "validation_status": "valid",
                "validation_errors": [],
            }
        ]
    )
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_skills_snapshot(screen, pilot)
        text = _visible_text(screen)

        assert "Selected: [red]summarize-notes[/red]" in text
        assert "Runtime target: local:skill:[red]summarize-notes[/red]" in text


@pytest.mark.asyncio
async def test_skills_destination_empty_state_disables_console_attach():
    app = _build_test_app()
    app.skills_scope_service = StaticSkillsScopeService([])
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_skills_snapshot(screen, pilot)
        button = screen.query_one("#skills-attach-to-console", Button)

        assert "No local Agent Skills are installed yet." in _visible_text(screen)
        assert button.disabled is True
        assert "Stage local skill context" in str(button.tooltip)


@pytest.mark.asyncio
async def test_skills_destination_uses_three_column_workbench_contract():
    app = _build_test_app()
    app.skills_scope_service = StaticSkillsScopeService([])
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_skills_snapshot(screen, pilot)
        text = _visible_text(screen)

        assert "Skills | Agent Skills packs, validation, Console attachments | Local" in text
        assert "Mode: Installed / Validate / Attach | Source: local SKILL.md directories" in text
        assert "Skill Library" in text
        assert "Skill Detail" in text
        assert "Skill Inspector" in text
        assert "Column 1:" not in text
        assert "Column 2:" not in text
        assert "Column 3:" not in text
        assert screen.query_one("#skills-workbench").region.height >= 20
        assert screen.query_one("#skills-list-detail-divider")
        assert screen.query_one("#skills-detail-inspector-divider")


@pytest.mark.asyncio
async def test_skills_destination_service_failure_uses_recovery_copy():
    app = _build_test_app()
    app.skills_scope_service = RaisingSkillsScopeService()
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_skills_snapshot(screen, pilot)
        button = screen.query_one("#skills-attach-to-console", Button)

        assert "Skills service unavailable; retry Skills later." in _visible_text(screen)
        assert button.disabled is True
        assert "Skills service is unavailable" in str(button.tooltip)


@pytest.mark.asyncio
async def test_skills_destination_missing_service_uses_unavailable_state():
    app = _build_test_app()
    app.skills_scope_service = object()
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_skills_snapshot(screen, pilot)
        button = screen.query_one("#skills-attach-to-console", Button)

        assert "Skills service is unavailable in this runtime." in _visible_text(screen)
        assert button.disabled is True
        assert "Skills service is unavailable" in str(button.tooltip)


@pytest.mark.asyncio
async def test_skills_destination_policy_denied_surfaces_policy_message():
    app = _build_test_app()
    app.skills_scope_service = PolicyDeniedSkillsScopeService()
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_skills_snapshot(screen, pilot)
        button = screen.query_one("#skills-attach-to-console", Button)

        _assert_policy_recovery_copy(
            visible_text=_visible_text(screen),
            button=button,
            status_label="Policy denied",
            unavailable_what="Attach local Skills to Console",
            why="Local Skills are disabled by workspace policy",
            next_action="Review workspace policy or ask the authority owner to allow this action.",
            recovery_action="Workspace policy",
            authority_owner="local",
        )
        assert "Skills service unavailable; retry Skills later." not in _visible_text(screen)


@pytest.mark.asyncio
async def test_skills_policy_denial_uses_recovery_state_selector(monkeypatch):
    monkeypatch.setattr(
        skills_screen_module,
        "policy_denied_recovery_state",
        _custom_policy_recovery_state,
    )
    app = _build_test_app()
    app.skills_scope_service = PolicyDeniedSkillsScopeService()
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#custom-skills-service-error")
        button = screen.query_one("#skills-attach-to-console", Button)

        assert "Custom policy state" in _visible_text(screen)
        assert button.tooltip == "Custom policy tooltip."


@pytest.mark.asyncio
async def test_skills_attach_to_console_uses_listed_skill_context():
    app = _build_test_app()
    app.skills_scope_service = StaticSkillsScopeService(
        [
            {
                "name": "summarize-notes",
                "description": "Summarize note collections",
                "argument_hint": "note id",
                "record_id": "local:skill:summarize-notes",
            }
        ]
    )
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_skills_snapshot(screen, pilot)
        await pilot.click("#skills-attach-to-console")
        await pilot.pause(0.1)

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]
    assert isinstance(payload, ChatHandoffPayload)
    assert payload.source == "skills"
    assert payload.item_type == "skills-context"
    assert payload.title == "Local Agent Skill: summarize-notes"
    assert "summarize-notes" in payload.body
    assert "Summarize note collections" in payload.body
    assert "argument hint: note id" in payload.body
    assert "Stage installed skills, SKILL.md instructions" not in payload.body
    assert payload.metadata["skill_count"] == 1
    assert payload.metadata["skill_names"] == ["summarize-notes"]
    assert payload.metadata["selected_skill_name"] == "summarize-notes"
    assert payload.metadata["selected_target_id"] == "local:skill:summarize-notes"


@pytest.mark.asyncio
async def test_skills_attach_to_console_uses_selected_valid_skill_context_only():
    app = _build_test_app()
    app.skills_scope_service = StaticSkillsScopeService(
        [
            {
                "name": "summarize-notes",
                "description": "Summarize note collections",
                "argument_hint": "note id",
                "record_id": "local:skill:summarize-notes",
                "backend": "local",
                "validation_status": "valid",
                "validation_errors": [],
            },
            {
                "name": "broken-skill",
                "description": "Broken metadata",
                "record_id": "local:skill:broken-skill",
                "validation_status": "invalid",
                "validation_errors": ["description is required"],
            },
        ]
    )
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_skills_snapshot(screen, pilot)
        await pilot.click("#skills-attach-to-console")
        await _wait_for_mock_call(app.open_chat_with_handoff, pilot)

    payload = app.open_chat_with_handoff.call_args.args[0]
    assert payload.title == "Local Agent Skill: summarize-notes"
    assert "summarize-notes" in payload.body
    assert "broken-skill" not in payload.body
    assert payload.metadata["skill_count"] == 1
    assert payload.metadata["skill_names"] == ["summarize-notes"]
    assert payload.metadata["selected_skill_name"] == "summarize-notes"
    assert payload.metadata["selected_record_id"] == "summarize-notes"
    assert payload.metadata["selected_target_id"] == "local:skill:summarize-notes"
    assert payload.metadata["validation_status"] == "valid"


@pytest.mark.asyncio
async def test_skills_attach_to_console_sanitizes_listed_skill_text():
    app = _build_test_app()
    app.skills_scope_service = StaticSkillsScopeService(
        [
            {
                "name": "unsafe-skill",
                "description": "Summarize <script>alert(1)</script> notes",
                "argument_hint": "note id onclick=steal",
                "record_id": "local:skill:unsafe-skill\x00",
            }
        ]
    )
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "skills")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_skills_snapshot(screen, pilot)
        visible_text = _visible_text(screen).lower()
        await pilot.click("#skills-attach-to-console")
        await pilot.pause(0.1)

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]

    assert "unsafe-skill" in payload.body
    assert "<script" not in payload.body.lower()
    assert "onclick=" not in payload.body.lower()
    assert "\x00" not in payload.body
    assert "<script" not in visible_text
    assert "onclick=" not in visible_text
    assert payload.metadata["skill_names"] == ["unsafe-skill"]


@pytest.mark.asyncio
async def test_settings_destination_uses_three_column_workbench_contract():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_visible_text(
            screen,
            pilot,
            "Settings | Global preferences, appearance, accounts, storage | Local",
        )
        text = _visible_text(screen)

        assert "Settings | Global preferences, appearance, accounts, storage | Local" in text
        assert "Mode: Overview | Runtime controls stay in MCP and ACP" in text
        assert "Settings Sections" in text
        assert "Preference Detail" in text
        assert "Scope Inspector" in text
        assert "Overview" in text
        assert "Provider readiness" in text
        assert "Storage" in text
        assert "Privacy" in text
        assert "Console paste collapse" in text
        assert "Mutation replay: disabled" in text
        assert "runtime MCP, ACP, and tool control stay in their own destinations" in text
        assert "Column 1:" not in text
        assert "Column 2:" not in text
        assert "Column 3:" not in text
        assert screen.query_one("#settings-workbench").region.height >= 20
        assert screen.query_one("#settings-overview-card").region.height >= 6
        category_pane = screen.query_one("#settings-category-pane")
        detail_pane = screen.query_one("#settings-detail-pane")
        impact_pane = screen.query_one("#settings-impact-pane")
        assert category_pane.region.height > 0
        assert detail_pane.region.height > 0
        assert impact_pane.region.height > 0
        assert screen.query_one("#settings-category-detail-divider")
        assert screen.query_one("#settings-detail-impact-divider")


def test_settings_sync_safety_state_failure_logs_context(caplog):
    class BrokenSyncScopeService:
        def list_write_sync_promotion_states(self, **_kwargs):
            raise RuntimeError("secret-token-123")

    app = _build_test_app()
    app.sync_scope_service = BrokenSyncScopeService()
    screen = SettingsScreen(app)

    with caplog.at_level(logging.WARNING, logger=settings_screen_module.__name__):
        states = screen._sync_safety_states()

    assert [state.domain for state in states] == ["library_collections", "workspaces"]
    assert "Failed to load Settings sync safety states" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "secret-token-123" not in caplog.text


@pytest.mark.asyncio
async def test_settings_appearance_action_routes_to_customize_surface():
    app = _build_test_app()
    seen_routes = []
    host = DestinationHarness(app, "settings", seen_routes)

    async with host.run_test(size=(180, 40)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#settings-open-appearance")
        await pilot.click("#settings-open-appearance")
        await pilot.pause(0.1)

    assert seen_routes[-1] == "customize"


@pytest.mark.parametrize(
    ("initial_value", "expected_label", "expected_saved_value"),
    [
        (True, "Enabled", False),
        ("false", "Disabled", True),
    ],
)
@pytest.mark.asyncio
async def test_settings_console_paste_collapse_toggle_reflects_and_persists_config(
    monkeypatch,
    initial_value,
    expected_label,
    expected_saved_value,
):
    app = _build_test_app()
    app.app_config["console"] = {"collapse_large_pastes": initial_value}
    saved_sections = []

    class FakeAdapter:
        def save_sections(self, section_values):
            saved_sections.append(section_values)
            return True

    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#settings-category-console-behavior")
        await pilot.click("#settings-category-console-behavior")
        await _wait_for_selector(screen, pilot, "#settings-console-collapse-large-pastes-toggle")
        toggle = screen.query_one(
            "#settings-console-collapse-large-pastes-toggle",
            Button,
        )

        assert expected_label in str(toggle.label)

        await pilot.click("#settings-console-collapse-large-pastes-toggle")
        await pilot.pause(0.1)

        assert app.app_config["console"]["collapse_large_pastes"] == initial_value
        assert saved_sections == []

        await pilot.click("#settings-save-category")
        await _wait_for_visible_text(screen, pilot, "Console behavior settings saved.")

    assert app.app_config["console"]["collapse_large_pastes"] is expected_saved_value
    assert saved_sections == [{"console": {"collapse_large_pastes": expected_saved_value}}]


@pytest.mark.asyncio
async def test_legacy_tools_settings_route_opens_mcp_not_global_settings():
    app = _build_test_app()
    screen_name, current_tab, screen_class = app._resolve_screen_navigation_target("tools_settings")

    assert screen_name == "tools_settings"
    assert current_tab == "mcp"
    assert screen_class is MCPScreen

    host = DestinationHarness(app, "tools_settings")

    async with host.run_test(size=(180, 40)) as pilot:
        await pilot.pause(0.1)
        screen = _active_destination_screen(host)

        visible_text = " ".join(
            _static_text(widget)
            for widget in screen.query(Static)
            if hasattr(widget, "renderable")
        )
        assert "MCP servers" in visible_text
        assert "scoped tools" in visible_text
        assert "global preferences" not in visible_text
