"""Study screen implementation with scope-aware navigation state."""

from __future__ import annotations

from inspect import isawaitable
from typing import Any, Dict, List, Optional

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button

from ..Navigation.base_app_screen import BaseAppScreen
from ..Study_Window import StudyWindow
from ...Widgets.Study import QuizSessionWidget, StudyDashboard
from .notes_scope_models import WorkspaceSubview
from .study_scope_models import StudyScopeContext, StudyScopeState, StudyScopeType


class StudyScreen(BaseAppScreen):
    """Screen wrapper for Study functionality."""
    
    # Screen-specific state
    current_section: reactive[str] = reactive("dashboard")
    current_study_session: reactive[Optional[Dict[str, Any]]] = reactive(None)
    study_materials: reactive[List[str]] = reactive([])
    is_studying: reactive[bool] = reactive(False)
    current_topic: reactive[str] = reactive("")
    scope_state: reactive[StudyScopeState] = reactive(StudyScopeState)
    _SECTION_TO_VIEW = {
        "paths": "structured_learning",
        "flashcards": "flashcards",
        "quizzes": "quizzes",
        "mindmaps": "mindmaps",
        "course": "course_creation",
        "guides": "study_guide",
        "learning_map": "learning_map",
    }

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "study", **kwargs)
        pending_scope = getattr(app_instance, "pending_study_scope_context", None)
        if isinstance(pending_scope, StudyScopeContext):
            self.scope_state = self._derive_scope_state(pending_scope)
        else:
            self.scope_state = StudyScopeState(backend=self._runtime_backend())
        self._effective_scope_key: tuple[str, Optional[str], str, bool] = self._scope_key(self.scope_state)
        self.study_dashboard: Optional[StudyDashboard] = None
        self.quiz_session_widget: Optional[QuizSessionWidget] = None
        self.study_window_widget: Optional[StudyWindow] = None
        self._dashboard_due_count: int = 0
        self._recent_deck_titles: list[str] = []
        self._recent_quiz_titles: list[str] = []

    @property
    def current_scope(self) -> StudyScopeState:
        return self.scope_state

    def compose_content(self) -> ComposeResult:
        """Compose the Study screen with a shell-level dashboard and study surface."""
        self.study_dashboard = StudyDashboard()
        self.quiz_session_widget = QuizSessionWidget()
        self.study_window_widget = StudyWindow(self.app_instance, show_sidebar=False)
        self.quiz_session_widget.display = False
        self.study_window_widget.display = False

        with Vertical(id="study-shell"):
            with Horizontal(id="study-section-bar"):
                yield Button("Dashboard", id="view-dashboard-btn", variant="primary")
                yield Button("Paths", id="view-structured-btn")
                yield Button("Flashcards", id="view-flashcards-btn")
                yield Button("Quizzes", id="view-quizzes-btn")
                yield Button("Guides", id="view-study-guide-btn")
                yield Button("Mindmaps", id="view-mindmaps-btn")
                yield Button("Course", id="view-course-btn")
                yield Button("Map", id="view-learning-map-btn")
            yield self.study_dashboard
            yield self.quiz_session_widget
            yield self.study_window_widget

    def _runtime_backend(self) -> str:
        getter = getattr(self.app_instance, "get_authoritative_runtime_source", None)
        if callable(getter):
            normalized = str(getter() or "").strip().lower()
            if normalized in {"local", "server"}:
                return normalized
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

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if isawaitable(value):
            return await value
        return value

    def _scope_arguments(self) -> dict[str, Any]:
        scope_type = self.scope_state.scope_type.value
        return {
            "scope_type": scope_type,
            "workspace_id": self.scope_state.workspace_id if scope_type == StudyScopeType.WORKSPACE.value else None,
        }

    @staticmethod
    def _normalize_records(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            items = payload.get("items")
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
            return []
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    def _scope_summary_text(self) -> str:
        if self.scope_state.scope_type == StudyScopeType.WORKSPACE:
            workspace_name = self.scope_state.workspace_name or self.scope_state.workspace_id or "Workspace"
            backend = self.scope_state.backend
            if self.scope_state.error_message:
                return f"Workspace: {workspace_name} | {self.scope_state.error_message}"
            return f"Workspace: {workspace_name} | Backend: {backend}"
        return "Global study"

    def _apply_section_layout(self) -> None:
        dashboard = self.study_dashboard
        quiz_session = self.quiz_session_widget
        study_window = self.study_window_widget

        if dashboard is not None and dashboard.is_mounted:
            dashboard.display = self.current_section == "dashboard"
        if quiz_session is not None and quiz_session.is_mounted:
            quiz_session.display = self.current_section == "quizzes"
        if study_window is not None and study_window.is_mounted:
            show_window = self.current_section != "dashboard"
            study_window.display = show_window
            target_view = self._SECTION_TO_VIEW.get(self.current_section)
            if show_window and target_view and study_window.current_view != target_view:
                study_window.current_view = target_view
        self._update_section_button_states()

    def _update_section_button_states(self) -> None:
        button_map = {
            "dashboard": "view-dashboard-btn",
            "paths": "view-structured-btn",
            "flashcards": "view-flashcards-btn",
            "quizzes": "view-quizzes-btn",
            "guides": "view-study-guide-btn",
            "mindmaps": "view-mindmaps-btn",
            "course": "view-course-btn",
            "learning_map": "view-learning-map-btn",
        }
        for section, button_id in button_map.items():
            try:
                button = self.query_one(f"#{button_id}", Button)
            except Exception:
                continue
            button.variant = "primary" if section == self.current_section else "default"

    def watch_current_section(self, old_section: str, new_section: str) -> None:
        del old_section, new_section
        self._apply_section_layout()

    def watch_current_study_session(
        self,
        old_session: Optional[Dict[str, Any]],
        new_session: Optional[Dict[str, Any]],
    ) -> None:
        del old_session, new_session
        self.sync_shell_from_window()

    def activate_section(self, section: str) -> None:
        if section not in {"dashboard", "paths", "flashcards", "quizzes", "guides", "mindmaps", "course", "learning_map"}:
            return
        self.current_section = section

    def _record_study_session(self, *, section: str, topic: str) -> None:
        self.current_topic = topic
        self.is_studying = True
        self.current_study_session = {
            "section": section,
            "topic": topic,
            "start_time": None,
            "materials": list(self.study_materials),
        }

    def _sync_dashboard_widgets(self) -> None:
        if self.study_dashboard is None or not self.study_dashboard.is_mounted:
            return
        self.study_dashboard.update_scope_summary(self._scope_summary_text())
        self.study_dashboard.update_due_today(self._dashboard_due_count)
        self.study_dashboard.update_recent_decks(self._recent_deck_titles)
        self.study_dashboard.update_recent_quizzes(self._recent_quiz_titles)

        summary = None
        if self.current_study_session:
            section = str(self.current_study_session.get("section") or "study").replace("_", " ")
            topic = str(self.current_study_session.get("topic") or "session").strip()
            summary = f"{section}: {topic}"
        self.study_dashboard.update_resume_action(summary)

    def _status_text_from_window(self, widget_id: str) -> str:
        study_window = self.study_window_widget
        if study_window is None or not study_window.is_mounted:
            return ""
        try:
            widget = study_window.query_one(widget_id)
        except Exception:
            return ""
        render = getattr(widget, "render", None)
        if callable(render):
            return str(render()).strip()
        return str(getattr(widget, "renderable", "") or "").strip()

    def _sync_quiz_session_widget(self) -> None:
        if self.quiz_session_widget is None or not self.quiz_session_widget.is_mounted:
            return

        self.quiz_session_widget.update_scope_summary(self._scope_summary_text())
        controller = getattr(self.study_window_widget, "quizzes_controller", None)
        if controller is None:
            self.quiz_session_widget.update_session_summary("Select a quiz to begin.")
            self.quiz_session_widget.update_status("")
            self.quiz_session_widget.set_start_enabled(False)
            return

        quiz_name = None
        label_getter = getattr(controller, "selected_quiz_label", None)
        if callable(label_getter):
            quiz_name = label_getter()

        if quiz_name:
            self.quiz_session_widget.update_session_summary(f"Selected quiz: {quiz_name}")
        elif getattr(controller, "has_quizzes", False):
            self.quiz_session_widget.update_session_summary("Select a quiz to begin.")
        else:
            self.quiz_session_widget.update_session_summary("No quizzes available yet.")

        status = self._status_text_from_window("#quiz-attempt-status")
        self.quiz_session_widget.update_status(status)

        scope_available = True
        scope_checker = getattr(controller, "_scope_is_available", None)
        if callable(scope_checker):
            scope_available = bool(scope_checker())

        attempt_active = bool(getattr(controller, "current_attempt_id", None)) and bool(
            getattr(controller, "current_attempt_questions", None)
        )
        self.quiz_session_widget.set_start_enabled(bool(quiz_name) and scope_available and not attempt_active)

    def sync_shell_from_window(self) -> None:
        self._sync_dashboard_widgets()
        self._sync_quiz_session_widget()

    async def _refresh_dashboard_snapshot(self) -> None:
        if self.scope_state.scope_type == StudyScopeType.WORKSPACE and self.scope_state.error_message:
            self._dashboard_due_count = 0
            self._recent_deck_titles = []
            self._recent_quiz_titles = []
            self._sync_dashboard_widgets()
            return

        scope_args = self._scope_arguments()
        mode = self._runtime_backend()
        study_service = getattr(self.app_instance, "study_scope_service", None)
        quiz_service = getattr(self.app_instance, "study_quiz_scope_service", None)
        db = getattr(self.app_instance, "chachanotes_db", None)

        due_count = 0
        try:
            due_loader = getattr(study_service, "get_due_flashcards", None)
            if callable(due_loader):
                due_records = await self._maybe_await(
                    due_loader(mode=mode, **scope_args, limit=25)
                )
                due_count = len(self._normalize_records(due_records) or list(due_records or []))
            elif db is not None and hasattr(db, "get_due_flashcards"):
                due_count = len(list(db.get_due_flashcards(limit=25) or []))
        except Exception:
            logger.debug("Failed to load Study due counts", exc_info=True)

        recent_decks: list[str] = []
        try:
            deck_loader = getattr(study_service, "list_decks", None)
            if callable(deck_loader):
                decks = await self._maybe_await(deck_loader(mode=mode, **scope_args, limit=3, offset=0))
                recent_decks = [
                    str(deck.get("name") or "Untitled deck")
                    for deck in self._normalize_records(decks)[:3]
                ]
            elif db is not None and hasattr(db, "list_decks"):
                recent_decks = [
                    str(deck.get("name") or "Untitled deck")
                    for deck in list(db.list_decks(limit=3, offset=0) or [])[:3]
                ]
        except Exception:
            logger.debug("Failed to load Study deck recents", exc_info=True)

        recent_quizzes: list[str] = []
        try:
            quiz_loader = getattr(quiz_service, "list_quizzes", None)
            if callable(quiz_loader):
                quizzes = await self._maybe_await(quiz_loader(mode=mode, **scope_args, q=None, limit=3, offset=0))
                recent_quizzes = [
                    str(quiz.get("name") or "Untitled quiz")
                    for quiz in self._normalize_records(quizzes)[:3]
                ]
            elif db is not None and hasattr(db, "list_quizzes"):
                quizzes = db.list_quizzes(q=None, workspace_id=self.scope_state.workspace_id, limit=3, offset=0)
                recent_quizzes = [
                    str(quiz.get("name") or "Untitled quiz")
                    for quiz in self._normalize_records(quizzes)[:3]
                ]
        except Exception:
            logger.debug("Failed to load Study quiz recents", exc_info=True)

        self._dashboard_due_count = due_count
        self._recent_deck_titles = recent_decks
        self._recent_quiz_titles = recent_quizzes
        self._sync_dashboard_widgets()

    def _reset_controller_state(self, controller: Any) -> None:
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

    def _ensure_scope_change_handler(self, controller: Any) -> Optional[Any]:
        if controller is None:
            return None

        handler = getattr(controller, "handle_scope_changed", None)
        if callable(handler):
            return handler

        def _fallback_handler() -> None:
            self._reset_controller_state(controller)

        controller.handle_scope_changed = _fallback_handler
        return controller.handle_scope_changed

    async def _reload_scoped_study_data(self, study_window: Any) -> None:
        current_view = str(getattr(study_window, "current_view", "") or "")
        if current_view == "flashcards":
            scheduler = getattr(study_window, "_schedule_flashcards_refresh", None)
            if callable(scheduler):
                scheduler()
        elif current_view == "quizzes":
            scheduler = getattr(study_window, "_schedule_quizzes_refresh", None)
            if callable(scheduler):
                scheduler()

    async def _apply_scope_context(
        self,
        scope_context: StudyScopeContext,
        *,
        study_window: Any,
        force_controller_notify: bool = False,
    ) -> None:
        next_state = self._derive_scope_state(scope_context)
        previous_key = self._effective_scope_key
        next_key = self._scope_key(next_state)

        self.scope_state = next_state
        self._effective_scope_key = next_key

        if previous_key == next_key and not force_controller_notify:
            return

        for controller_name in ("flashcards_controller", "quizzes_controller"):
            controller = getattr(study_window, controller_name, None)
            if controller_name == "flashcards_controller":
                end_review_session = getattr(controller, "end_review_session_if_needed", None)
                if callable(end_review_session):
                    await end_review_session()
            handler = self._ensure_scope_change_handler(controller)
            if callable(handler):
                handler()

        await self._reload_scoped_study_data(study_window)

    async def _apply_scope_context_and_refresh(
        self,
        scope_context: StudyScopeContext,
        *,
        study_window: Any,
        force_controller_notify: bool = False,
    ) -> None:
        await self._apply_scope_context(
            scope_context,
            study_window=study_window,
            force_controller_notify=force_controller_notify,
        )
        sync_scope_banner = getattr(study_window, "_sync_scope_banner", None)
        if callable(sync_scope_banner):
            sync_scope_banner()
        await self._refresh_dashboard_snapshot()
        self.sync_shell_from_window()

    async def on_mount(self) -> None:
        """Initialize Study features when screen is mounted."""
        super().on_mount()
        logger.info("Study screen mounted")

        study_window = self.query_one(StudyWindow)

        pending_scope_context = self._consume_pending_scope_context()
        scope_context = pending_scope_context or self._current_scope_context()
        await self._apply_scope_context_and_refresh(
            scope_context,
            study_window=study_window,
            force_controller_notify=pending_scope_context is not None,
        )

        if hasattr(study_window, 'load_saved_sessions'):
            await study_window.load_saved_sessions()

        if hasattr(study_window, 'initialize'):
            await study_window.initialize()
        self._apply_section_layout()
        self.sync_shell_from_window()

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
        await self._apply_scope_context_and_refresh(scope_context, study_window=study_window)

        if self.current_study_session:
            if hasattr(study_window, 'restore_session'):
                await study_window.restore_session(self.current_study_session)
        self._apply_section_layout()
        self.sync_shell_from_window()

    async def handle_runtime_backend_changed(self, runtime_backend: str) -> None:
        normalized_backend = str(runtime_backend or "").strip().lower()
        if normalized_backend in {"local", "server"}:
            self.app_instance.current_runtime_backend = normalized_backend
        study_window = self.query_one(StudyWindow)
        await self._apply_scope_context_and_refresh(self._current_scope_context(), study_window=study_window)

    def save_state(self) -> dict[str, Any]:
        state = super().save_state()
        state.update(
            {
                "study_scope": {
                    "scope_type": self.scope_state.scope_type.value,
                    "workspace_id": self.scope_state.workspace_id,
                    "workspace_name": self.scope_state.workspace_name,
                    "return_hint": self.scope_state.return_hint,
                },
                "study_section": self.current_section,
                "current_study_session": self.current_study_session,
            }
        )
        return state

    def restore_state(self, state: dict[str, Any]) -> None:
        super().restore_state(state)
        saved_scope = state.get("study_scope")
        if saved_scope:
            self.scope_state = self._derive_scope_state(
                StudyScopeContext(
                    scope_type=StudyScopeType(saved_scope.get("scope_type", StudyScopeType.GLOBAL.value)),
                    workspace_id=saved_scope.get("workspace_id"),
                    workspace_name=saved_scope.get("workspace_name"),
                    return_hint=saved_scope.get("return_hint"),
                )
            )
            self._effective_scope_key = self._scope_key(self.scope_state)
        self.current_section = str(state.get("study_section") or "dashboard")
        self.current_study_session = state.get("current_study_session")

    def update_study_materials(self, materials: List[str]) -> None:
        """Update the list of study materials."""
        self.study_materials = materials
        logger.debug(f"Updated study materials: {len(materials)} items")
    
    def start_study_session(self, topic: str, *, section: str = "dashboard") -> None:
        """Start a new study session."""
        self._record_study_session(section=section, topic=topic)
        logger.info(f"Started study session for topic: {topic}")

    def switch_to_global_scope(self) -> None:
        scope_context = StudyScopeContext(scope_type=StudyScopeType.GLOBAL)
        if self.is_mounted:
            try:
                study_window = self.query_one(StudyWindow)
            except Exception:
                study_window = None
            if study_window is not None:
                self.run_worker(self._apply_scope_context_and_refresh(scope_context, study_window=study_window), exclusive=True)
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
                self.run_worker(self._apply_scope_context_and_refresh(scope_context, study_window=study_window), exclusive=True)
                return
        open_study = getattr(self.app_instance, "open_study_screen", None)
        if callable(open_study):
            open_study(scope_context)

    @on(Button.Pressed, "#view-dashboard-btn")
    def handle_dashboard_section(self) -> None:
        self.activate_section("dashboard")

    @on(Button.Pressed, "#view-structured-btn")
    def handle_paths_section(self) -> None:
        self.activate_section("paths")

    @on(Button.Pressed, "#view-flashcards-btn")
    def handle_flashcards_section(self) -> None:
        self.activate_section("flashcards")

    @on(Button.Pressed, "#view-quizzes-btn")
    def handle_quizzes_section(self) -> None:
        self.activate_section("quizzes")

    @on(Button.Pressed, "#view-study-guide-btn")
    def handle_guides_section(self) -> None:
        self.activate_section("guides")

    @on(Button.Pressed, "#view-mindmaps-btn")
    def handle_mindmaps_section(self) -> None:
        self.activate_section("mindmaps")

    @on(Button.Pressed, "#view-course-btn")
    def handle_course_section(self) -> None:
        self.activate_section("course")

    @on(Button.Pressed, "#view-learning-map-btn")
    def handle_learning_map_section(self) -> None:
        self.activate_section("learning_map")

    @on(Button.Pressed, "#study-open-flashcards")
    def handle_dashboard_open_flashcards(self) -> None:
        self.activate_section("flashcards")

    @on(Button.Pressed, "#study-open-quizzes")
    def handle_dashboard_open_quizzes(self) -> None:
        self.activate_section("quizzes")

    @on(Button.Pressed, "#study-resume-last")
    def handle_resume_last_session(self) -> None:
        if not self.current_study_session:
            return
        section = str(self.current_study_session.get("section") or "dashboard")
        self.activate_section(section)

    @on(Button.Pressed, "#quiz-start")
    async def handle_quiz_start(self) -> None:
        study_window = self.study_window_widget
        if study_window is None:
            return
        self.activate_section("quizzes")
        await study_window.quizzes_controller.start_attempt()
        quiz_name = study_window.quizzes_controller.selected_quiz_label() or "Quiz session"
        self._record_study_session(section="quizzes", topic=quiz_name)
        self.sync_shell_from_window()
