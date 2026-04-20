"""Handler for persona-related operations in the CCP screen."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger

from .ccp_messages import PersonaMessage, ViewChangeMessage
from ...tldw_api import PersonaProfileCreate, PersonaProfileUpdate

if TYPE_CHECKING:
    from ..Screens.ccp_screen import CCPScreen

logger = logger.bind(module="CCPPersonaHandler")


class CCPPersonaHandler:
    """Handles first-class persona profile operations for the CCP screen."""

    def __init__(self, window: "CCPScreen"):
        self.window = window
        self.app_instance = window.app_instance
        self.current_persona_id: Optional[str] = None
        self.current_persona_data: Dict[str, Any] = {}
        self.persona_list: List[Dict[str, Any]] = []

    def _current_mode(self) -> str:
        """Resolve the currently selected backend mode, defaulting to local."""
        candidates = (
            getattr(getattr(self.window, "state", None), "runtime_backend", None),
            getattr(self.window, "runtime_backend", None),
            getattr(self.app_instance, "runtime_backend", None),
            getattr(self.app_instance, "current_runtime_backend", None),
        )
        for candidate in candidates:
            if candidate in {"local", "server"}:
                return candidate
        return "local"

    def _notify(self, message: str, severity: str = "warning") -> None:
        """Surface a CCP notification when a user action cannot complete."""
        notifier = getattr(self.window, "notify", None)
        if callable(notifier):
            notifier(message, severity=severity)

    def _current_chat_id(self, chat_id: str | None = None) -> str:
        """Resolve the active chat identifier for chat-scoped execution helpers."""
        candidates = (
            chat_id,
            getattr(getattr(self.window, "state", None), "selected_conversation_id", None),
            getattr(getattr(self.window, "conversation_handler", None), "current_conversation_id", None),
        )
        for candidate in candidates:
            if candidate not in {None, ""}:
                return str(candidate)
        raise ValueError("Select a conversation first.")

    def _normalize_persona_record(self, record: Any) -> Dict[str, Any]:
        """Normalize raw backend responses into a dict the CCP widgets can render."""
        if hasattr(record, "model_dump"):
            record = record.model_dump(mode="json")
        if not isinstance(record, dict):
            return {}

        normalized = dict(record)
        normalized["id"] = str(normalized.get("id", "") or "")
        normalized["name"] = str(normalized.get("name", "") or "")
        normalized["mode"] = normalized.get("mode") or "session_scoped"
        normalized["system_prompt"] = str(
            normalized.get("system_prompt")
            or normalized.get("description")
            or ""
        )
        return normalized

    def _normalize_persona_list(self, payload: Any) -> List[Dict[str, Any]]:
        """Normalize persona collection responses across local and server backends."""
        items = payload.get("items", []) if isinstance(payload, dict) else payload
        normalized_items = [self._normalize_persona_record(item) for item in list(items or [])]
        return [item for item in normalized_items if item.get("id")]

    def _load_editor(self, persona_data: Dict[str, Any]) -> None:
        """Push persona data into the editor widget when present."""
        try:
            editor = self.window.query_one("#ccp-persona-editor-view")
            if hasattr(editor, "load_persona"):
                editor.load_persona(persona_data)
        except Exception:
            logger.debug("Persona editor widget unavailable during load", exc_info=True)

    async def refresh_persona_list(self) -> List[Dict[str, Any]]:
        """Refresh the available persona profile list from the shared scope service."""
        service = getattr(self.app_instance, "character_persona_scope_service", None)
        if service is None or not hasattr(service, "list_persona_profiles"):
            logger.debug("Persona scope service unavailable; returning empty persona list")
            self.persona_list = []
            return self.persona_list

        try:
            personas = await service.list_persona_profiles(mode=self._current_mode())
        except ValueError:
            logger.debug("Persona list unavailable in current mode", exc_info=True)
            self.persona_list = []
            return self.persona_list

        self.persona_list = self._normalize_persona_list(personas)

        sidebar = getattr(self.window, "query_one", None)
        if callable(sidebar):
            try:
                widget = self.window.query_one("#ccp-sidebar")
                if hasattr(widget, "update_persona_list"):
                    widget.update_persona_list(self.persona_list)
            except Exception:
                logger.debug("Sidebar persona list update skipped", exc_info=True)

        return self.persona_list

    async def load_persona(self, persona_id: str) -> None:
        """Load a persona profile by identifier via the mode-aware scope service."""
        service = getattr(self.app_instance, "character_persona_scope_service", None)
        if service is None or not hasattr(service, "get_persona_profile"):
            logger.warning("Persona scope service unavailable; cannot load persona {}", persona_id)
            self._notify("Persona profiles are not available in the current backend.")
            return

        try:
            persona_data = await service.get_persona_profile(
                persona_id,
                mode=self._current_mode(),
            )
        except ValueError as exc:
            logger.warning("Persona {} unavailable in current mode: {}", persona_id, exc)
            self._notify(str(exc))
            return
        except Exception:
            logger.error("Error loading persona {}", persona_id, exc_info=True)
            self._notify("Failed to load persona profile.", severity="error")
            return

        normalized = self._normalize_persona_record(persona_data)
        if not normalized:
            logger.warning("Persona {} not found", persona_id)
            self._notify("Persona profile could not be loaded.", severity="warning")
            return

        self.current_persona_id = normalized["id"] or persona_id
        self.current_persona_data = normalized
        self._load_editor(normalized)
        self.window.post_message(
            PersonaMessage.Loaded(self.current_persona_id, normalized),
        )
        self.window.post_message(
            ViewChangeMessage.Requested(
                "persona_profiles",
                {"persona_id": self.current_persona_id},
            ),
        )

    async def handle_create_persona(self) -> None:
        """Start a new persona profile flow with a clean editor state."""
        self.current_persona_id = None
        self.current_persona_data = {
            "name": "",
            "mode": "session_scoped",
            "system_prompt": "",
            "is_active": True,
        }
        self._load_editor(self.current_persona_data)
        self.window.post_message(ViewChangeMessage.Requested("persona_editor"))

    async def handle_edit_persona(self, persona_id: str | None = None) -> None:
        """Switch to the editor for the currently selected persona."""
        if persona_id and persona_id != self.current_persona_id:
            await self.load_persona(persona_id)
        if not self.current_persona_data:
            self._notify("Load a persona profile before editing it.")
            return
        self._load_editor(self.current_persona_data)
        self.window.post_message(
            ViewChangeMessage.Requested(
                "persona_editor",
                {"persona_id": self.current_persona_id},
            )
        )

    async def save_persona(self, persona_data: Dict[str, Any]) -> None:
        """Create or update a persona profile through the shared scope service."""
        service = getattr(self.app_instance, "character_persona_scope_service", None)
        if service is None:
            self._notify("Persona profiles are not available in the current backend.")
            return

        name = str(persona_data.get("name", "") or "").strip()
        if not name:
            self._notify("Persona name is required.")
            return

        mode = persona_data.get("mode") or "session_scoped"
        current_mode = self._current_mode()

        try:
            if self.current_persona_id:
                request_data = PersonaProfileUpdate(
                    name=name,
                    mode=mode,
                    system_prompt=persona_data.get("system_prompt"),
                )
                result = await service.update_persona_profile(
                    self.current_persona_id,
                    request_data,
                    expected_version=persona_data.get("version"),
                    mode=current_mode,
                )
            else:
                request_data = PersonaProfileCreate(
                    id=persona_data.get("id"),
                    name=name,
                    mode=mode,
                    system_prompt=persona_data.get("system_prompt"),
                    is_active=bool(persona_data.get("is_active", True)),
                    setup=persona_data.get("setup") or {},
                    voice_defaults=persona_data.get("voice_defaults") or {},
                )
                result = await service.create_persona_profile(
                    request_data,
                    mode=current_mode,
                )
        except ValueError as exc:
            logger.warning("Persona save unavailable in {} mode: {}", current_mode, exc)
            self._notify(str(exc))
            return
        except Exception:
            logger.error("Error saving persona profile", exc_info=True)
            self._notify("Failed to save persona profile.", severity="error")
            return

        normalized = self._normalize_persona_record(result)
        if not normalized:
            normalized = self._normalize_persona_record(persona_data)
            normalized.setdefault("id", self.current_persona_id or str(persona_data.get("id", "") or ""))

        self.current_persona_id = normalized.get("id") or self.current_persona_id
        self.current_persona_data = normalized
        await self.refresh_persona_list()
        self._load_editor(normalized)
        self.window.post_message(
            PersonaMessage.Loaded(self.current_persona_id or "", normalized),
        )
        self.window.post_message(
            ViewChangeMessage.Requested(
                "persona_profiles",
                {"persona_id": self.current_persona_id},
            )
        )

    async def list_chat_greetings(self, chat_id: str | None = None) -> Dict[str, Any]:
        """List server chat greetings for the active CCP conversation."""
        service = getattr(self.app_instance, "character_persona_scope_service", None)
        if service is None or not hasattr(service, "list_chat_greetings"):
            self._notify("Chat greeting execution support is not available in the current backend.")
            return {}

        try:
            return await service.list_chat_greetings(
                self._current_chat_id(chat_id),
                mode=self._current_mode(),
            )
        except ValueError as exc:
            logger.warning("Chat greetings unavailable in {} mode: {}", self._current_mode(), exc)
            self._notify(str(exc))
            return {}
        except Exception:
            logger.error("Error loading chat greetings", exc_info=True)
            self._notify("Failed to load chat greetings.", severity="error")
            return {}

    async def select_chat_greeting(self, index: int, chat_id: str | None = None) -> Dict[str, Any]:
        """Select a greeting for the active CCP conversation."""
        service = getattr(self.app_instance, "character_persona_scope_service", None)
        if service is None or not hasattr(service, "select_chat_greeting"):
            self._notify("Chat greeting execution support is not available in the current backend.")
            return {}

        try:
            return await service.select_chat_greeting(
                self._current_chat_id(chat_id),
                index,
                mode=self._current_mode(),
            )
        except ValueError as exc:
            logger.warning("Chat greeting selection unavailable in {} mode: {}", self._current_mode(), exc)
            self._notify(str(exc))
            return {}
        except Exception:
            logger.error("Error selecting chat greeting", exc_info=True)
            self._notify("Failed to select chat greeting.", severity="error")
            return {}

    async def list_chat_presets(self) -> Dict[str, Any]:
        """List server chat prompt presets available to CCP execution."""
        service = getattr(self.app_instance, "character_persona_scope_service", None)
        if service is None or not hasattr(service, "list_chat_presets"):
            self._notify("Chat preset execution support is not available in the current backend.")
            return {}

        try:
            return await service.list_chat_presets(mode=self._current_mode())
        except ValueError as exc:
            logger.warning("Chat presets unavailable in {} mode: {}", self._current_mode(), exc)
            self._notify(str(exc))
            return {}
        except Exception:
            logger.error("Error loading chat presets", exc_info=True)
            self._notify("Failed to load chat presets.", severity="error")
            return {}

    async def create_chat_preset(self, request_data: Any) -> Dict[str, Any]:
        """Create a server chat prompt preset through the shared scope service."""
        service = getattr(self.app_instance, "character_persona_scope_service", None)
        if service is None or not hasattr(service, "create_chat_preset"):
            self._notify("Chat preset execution support is not available in the current backend.")
            return {}

        try:
            return await service.create_chat_preset(request_data, mode=self._current_mode())
        except ValueError as exc:
            logger.warning("Chat preset creation unavailable in {} mode: {}", self._current_mode(), exc)
            self._notify(str(exc))
            return {}
        except Exception:
            logger.error("Error creating chat preset", exc_info=True)
            self._notify("Failed to create chat preset.", severity="error")
            return {}

    async def update_chat_preset(self, preset_id: str, request_data: Any) -> Dict[str, Any]:
        """Update a server chat prompt preset through the shared scope service."""
        service = getattr(self.app_instance, "character_persona_scope_service", None)
        if service is None or not hasattr(service, "update_chat_preset"):
            self._notify("Chat preset execution support is not available in the current backend.")
            return {}

        try:
            return await service.update_chat_preset(
                preset_id,
                request_data,
                mode=self._current_mode(),
            )
        except ValueError as exc:
            logger.warning("Chat preset update unavailable in {} mode: {}", self._current_mode(), exc)
            self._notify(str(exc))
            return {}
        except Exception:
            logger.error("Error updating chat preset", exc_info=True)
            self._notify("Failed to update chat preset.", severity="error")
            return {}

    async def delete_chat_preset(self, preset_id: str) -> Dict[str, Any]:
        """Delete a server chat prompt preset through the shared scope service."""
        service = getattr(self.app_instance, "character_persona_scope_service", None)
        if service is None or not hasattr(service, "delete_chat_preset"):
            self._notify("Chat preset execution support is not available in the current backend.")
            return {}

        try:
            return await service.delete_chat_preset(preset_id, mode=self._current_mode())
        except ValueError as exc:
            logger.warning("Chat preset delete unavailable in {} mode: {}", self._current_mode(), exc)
            self._notify(str(exc))
            return {}
        except Exception:
            logger.error("Error deleting chat preset", exc_info=True)
            self._notify("Failed to delete chat preset.", severity="error")
            return {}
