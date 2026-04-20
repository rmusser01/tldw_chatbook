"""Study screen implementation with scope-aware navigation state."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger
from textual.app import ComposeResult
from textual.reactive import reactive

from ..Navigation.base_app_screen import BaseAppScreen
from ..Study_Window import StudyWindow
from .notes_scope_models import WorkspaceSubview
from .study_scope_models import StudyScopeContext, StudyScopeState, StudyScopeType


class StudyScreen(BaseAppScreen):
    """Screen wrapper for Study functionality."""
    
    # Screen-specific state
    current_study_session: reactive[Optional[Dict[str, Any]]] = reactive(None)
    study_materials: reactive[List[str]] = reactive([])
    is_studying: reactive[bool] = reactive(False)
    current_topic: reactive[str] = reactive("")
    scope_state: reactive[StudyScopeState] = reactive(StudyScopeState)

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "study", **kwargs)
        self.scope_state = StudyScopeState(backend=self._runtime_backend())
        self._effective_scope_key: tuple[str, Optional[str], str, bool] = self._scope_key(self.scope_state)

    @property
    def current_scope(self) -> StudyScopeState:
        return self.scope_state

    def compose_content(self) -> ComposeResult:
        """Compose the Study screen with the Study window."""
        yield StudyWindow(self.app_instance)

    def _runtime_backend(self) -> str:
        candidates = (
            getattr(self.app_instance, "current_runtime_backend", None),
            getattr(self.app_instance, "runtime_backend", None),
        )
        for candidate in candidates:
            normalized = str(candidate or "").strip().lower()
            if normalized in {"local", "server"}:
                return normalized
        return "local"

    @staticmethod
    def _scope_key(scope_state: StudyScopeState) -> tuple[str, Optional[str], str, bool]:
        return (
            scope_state.scope_type.value,
            scope_state.workspace_id,
            scope_state.backend,
            scope_state.workspace_scope_available,
        )

    def _derive_scope_state(self, scope_context: StudyScopeContext) -> StudyScopeState:
        backend = self._runtime_backend()
        workspace_scope_available = backend == "server"
        error_message: Optional[str] = None

        if scope_context.scope_type == StudyScopeType.WORKSPACE:
            if not scope_context.workspace_id:
                error_message = "Workspace study requires a workspace id."
            elif not workspace_scope_available:
                error_message = "Workspace study is only available in server mode."

        return StudyScopeState(
            scope_type=scope_context.scope_type,
            workspace_id=scope_context.workspace_id,
            workspace_name=scope_context.workspace_name,
            return_hint=scope_context.return_hint,
            backend=backend,
            workspace_scope_available=workspace_scope_available,
            error_message=error_message,
        )

    def _consume_pending_scope_context(self) -> Optional[StudyScopeContext]:
        pending = getattr(self.app_instance, "pending_study_scope_context", None)
        if pending is None:
            return None
        self.app_instance.pending_study_scope_context = None
        return pending

    def _current_scope_context(self) -> StudyScopeContext:
        return self.scope_state.as_context()

    def _hard_reset_controller(self, controller: Any) -> None:
        for attribute, value in (
            ("current_review_card", None),
            ("current_review_session_id", None),
            ("current_decks", []),
            ("current_cards", []),
            ("selected_deck_record", None),
            ("selected_card_record", None),
            ("has_decks", False),
            ("current_attempt_id", None),
            ("current_attempt_questions", []),
            ("current_attempt_answers", []),
            ("current_question_index", 0),
            ("current_quiz_questions", []),
            ("current_attempt_history", []),
            ("has_quizzes", False),
        ):
            if hasattr(controller, attribute):
                setattr(controller, attribute, value)

    def _hard_reset_study_controllers(self, study_window: Any) -> None:
        for controller_name in ("flashcards_controller", "quizzes_controller"):
            controller = getattr(study_window, controller_name, None)
            if controller is not None:
                self._hard_reset_controller(controller)

    async def _reload_scoped_study_data(self, study_window: Any) -> None:
        for scheduler_name in ("_schedule_flashcards_refresh", "_schedule_quizzes_refresh"):
            scheduler = getattr(study_window, scheduler_name, None)
            if callable(scheduler):
                scheduler()

    async def _apply_scope_context(self, scope_context: StudyScopeContext, *, study_window: Any) -> None:
        next_state = self._derive_scope_state(scope_context)
        previous_key = self._effective_scope_key
        next_key = self._scope_key(next_state)
        backend_changed = previous_key[2:] != next_key[2:]

        self.scope_state = next_state
        self._effective_scope_key = next_key

        if previous_key == next_key:
            return

        if backend_changed:
            self._hard_reset_study_controllers(study_window)

        for controller_name in ("flashcards_controller", "quizzes_controller"):
            controller = getattr(study_window, controller_name, None)
            handler = getattr(controller, "handle_scope_changed", None)
            if callable(handler):
                handler()

        await self._reload_scoped_study_data(study_window)

    async def on_mount(self) -> None:
        """Initialize Study features when screen is mounted."""
        super().on_mount()
        logger.info("Study screen mounted")

        study_window = self.query_one(StudyWindow)

        if hasattr(study_window, 'load_saved_sessions'):
            await study_window.load_saved_sessions()

        if hasattr(study_window, 'initialize'):
            await study_window.initialize()

        scope_context = self._consume_pending_scope_context() or self._current_scope_context()
        await self._apply_scope_context(scope_context, study_window=study_window)

    async def on_screen_suspend(self) -> None:
        """Save state when screen is suspended (navigated away)."""
        logger.debug("Study screen suspended")
        
        # Save current study session if active
        if self.is_studying and self.current_study_session:
            study_window = self.query_one(StudyWindow)
            if hasattr(study_window, 'save_session'):
                await study_window.save_session(self.current_study_session)
        
        self.is_studying = False
    
    async def on_screen_resume(self) -> None:
        """Restore state when screen is resumed."""
        logger.debug("Study screen resumed")

        study_window = self.query_one(StudyWindow)
        scope_context = self._consume_pending_scope_context() or self._current_scope_context()
        await self._apply_scope_context(scope_context, study_window=study_window)

        if self.current_study_session:
            if hasattr(study_window, 'restore_session'):
                await study_window.restore_session(self.current_study_session)

    def save_state(self) -> dict[str, Any]:
        state = super().save_state()
        state.update(
            {
                "study_scope": {
                    "scope_type": self.scope_state.scope_type.value,
                    "workspace_id": self.scope_state.workspace_id,
                    "workspace_name": self.scope_state.workspace_name,
                    "return_hint": self.scope_state.return_hint,
                }
            }
        )
        return state

    def restore_state(self, state: dict[str, Any]) -> None:
        super().restore_state(state)
        saved_scope = state.get("study_scope")
        if not saved_scope:
            return
        self.scope_state = self._derive_scope_state(
            StudyScopeContext(
                scope_type=StudyScopeType(saved_scope.get("scope_type", StudyScopeType.GLOBAL.value)),
                workspace_id=saved_scope.get("workspace_id"),
                workspace_name=saved_scope.get("workspace_name"),
                return_hint=saved_scope.get("return_hint"),
            )
        )
        self._effective_scope_key = self._scope_key(self.scope_state)

    def update_study_materials(self, materials: List[str]) -> None:
        """Update the list of study materials."""
        self.study_materials = materials
        logger.debug(f"Updated study materials: {len(materials)} items")
    
    def start_study_session(self, topic: str) -> None:
        """Start a new study session."""
        self.current_topic = topic
        self.is_studying = True
        self.current_study_session = {
            "topic": topic,
            "start_time": None,  # Will be set by StudyWindow
            "materials": self.study_materials
        }
        logger.info(f"Started study session for topic: {topic}")

    def switch_to_global_scope(self) -> None:
        scope_context = StudyScopeContext(scope_type=StudyScopeType.GLOBAL)
        if self.is_mounted:
            try:
                study_window = self.query_one(StudyWindow)
            except Exception:
                study_window = None
            if study_window is not None:
                self.run_worker(self._apply_scope_context(scope_context, study_window=study_window), exclusive=True)
                return
        open_study = getattr(self.app_instance, "open_study_screen", None)
        if callable(open_study):
            open_study(scope_context)

    def return_to_workspace(self) -> None:
        if not self.scope_state.workspace_id:
            return
        open_workspace = getattr(self.app_instance, "open_notes_workspace", None)
        if callable(open_workspace):
            open_workspace(
                self.scope_state.workspace_id,
                subview=WorkspaceSubview.DETAILS,
            )

    def enter_workspace_scope(self, workspace_id: str, workspace_name: Optional[str] = None) -> None:
        scope_context = StudyScopeContext(
            scope_type=StudyScopeType.WORKSPACE,
            workspace_id=workspace_id,
            workspace_name=workspace_name,
        )
        if self.is_mounted:
            try:
                study_window = self.query_one(StudyWindow)
            except Exception:
                study_window = None
            if study_window is not None:
                self.run_worker(self._apply_scope_context(scope_context, study_window=study_window), exclusive=True)
                return
        open_study = getattr(self.app_instance, "open_study_screen", None)
        if callable(open_study):
            open_study(scope_context)
