"""Study screen implementation with scope-aware navigation state."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import hashlib
from inspect import isawaitable
import re
from typing import Any, Dict, List, Optional

from loguru import logger
from rich.markup import escape as escape_markup
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button

from ...Chat.chat_handoff_models import ChatHandoffPayload
from ...Utils.input_validation import sanitize_string, validate_text_input
from ..Navigation.base_app_screen import BaseAppScreen
from ..Study_Window import StudyWindow
from ...Widgets.Study import QuizSessionWidget, StudyDashboard
from ...Widgets.Study.quiz_session_widget import (
    QUIZ_REVIEW_ENABLED_TOOLTIP,
    QUIZ_REVIEW_HANDOFF_UNAVAILABLE_TOOLTIP,
    QUIZ_REVIEW_SELECT_TOOLTIP,
    QUIZ_START_ATTEMPT_ACTIVE_TOOLTIP,
    QUIZ_START_ENABLED_TOOLTIP,
    QUIZ_START_NO_QUIZZES_TOOLTIP,
    QUIZ_START_SCOPE_UNAVAILABLE_TOOLTIP,
    QUIZ_START_SELECT_TOOLTIP,
)
from .notes_scope_models import WorkspaceSubview
from .study_scope_models import (
    STUDY_MATERIAL_SUMMARY_LENGTH_LIMIT,
    STUDY_MATERIAL_TITLE_LENGTH_LIMIT,
    STUDY_MATERIAL_TITLES_LIMIT,
    STUDY_SOURCE_ID_LENGTH_LIMIT,
    STUDY_SOURCE_ITEMS_LIMIT,
    StudyScopeContext,
    StudyScopeState,
    StudyScopeType,
    StudySourceItem,
)


ScopeKey = tuple[str, Optional[str], str, bool, Optional[str]]
_HTML_TAG_RE = re.compile(r"<[^>]*>")
_DANGEROUS_TEXT_RE = re.compile(r"javascript\s*:|\bon(?:click|error)\s*=", re.IGNORECASE)
SOURCE_STUDY_PACK_STATUS_CHECKS = 8
SOURCE_STUDY_PACK_STATUS_DELAY_SECONDS = 0.25
SOURCE_STUDY_PACK_ERROR_LENGTH_LIMIT = 240
SOURCE_STUDY_PACK_ID_LENGTH_LIMIT = 64
SOURCE_STUDY_PACK_JOB_STATUSES = frozenset({"queued", "running", "completed", "failed", "cancelled"})


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
    _SECTION_TOOLTIPS = {
        "dashboard": "Review due cards, recent decks, quizzes, and resume study sessions.",
        "paths": "Build or follow structured learning paths.",
        "flashcards": "Review decks and spaced-repetition cards.",
        "quizzes": "Create, start, and review quizzes.",
        "guides": "Generate or open study guides from your material.",
        "mindmaps": "Explore topics as visual knowledge maps.",
        "course": "Create course outlines and study sequences.",
        "learning_map": "Open the learning map for relationships across study material.",
    }
    _VALID_INITIAL_SECTIONS = frozenset({"dashboard", *_SECTION_TO_VIEW.keys()})

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "study", **kwargs)
        pending_scope = getattr(app_instance, "pending_study_scope_context", None)
        if isinstance(pending_scope, StudyScopeContext):
            self.scope_state = self._derive_scope_state(pending_scope)
        else:
            self.scope_state = StudyScopeState(backend=self._runtime_backend())
        self._effective_scope_key: ScopeKey = self._scope_key(self.scope_state)
        self.study_dashboard: Optional[StudyDashboard] = None
        self.quiz_session_widget: Optional[QuizSessionWidget] = None
        self.study_window_widget: Optional[StudyWindow] = None
        self._dashboard_due_count: int = 0
        self._recent_deck_titles: list[str] = []
        self._recent_quiz_titles: list[str] = []
        self._latest_source_study_pack: dict[str, Any] | None = None
        self._pending_initial_section = self._consume_pending_initial_section()

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
                yield Button(
                    "Dashboard",
                    id="view-dashboard-btn",
                    variant="primary",
                    tooltip=self._SECTION_TOOLTIPS["dashboard"],
                )
                yield Button("Paths", id="view-structured-btn", tooltip=self._SECTION_TOOLTIPS["paths"])
                yield Button(
                    "Flashcards",
                    id="view-flashcards-btn",
                    tooltip=self._SECTION_TOOLTIPS["flashcards"],
                )
                yield Button("Quizzes", id="view-quizzes-btn", tooltip=self._SECTION_TOOLTIPS["quizzes"])
                yield Button(
                    "Guides",
                    id="view-study-guide-btn",
                    tooltip=self._SECTION_TOOLTIPS["guides"],
                )
                yield Button(
                    "Mindmaps",
                    id="view-mindmaps-btn",
                    tooltip=self._SECTION_TOOLTIPS["mindmaps"],
                )
                yield Button("Course", id="view-course-btn", tooltip=self._SECTION_TOOLTIPS["course"])
                yield Button(
                    "Map",
                    id="view-learning-map-btn",
                    tooltip=self._SECTION_TOOLTIPS["learning_map"],
                )
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
    def _scope_key(scope_state: StudyScopeState) -> ScopeKey:
        return (
            scope_state.scope_type.value,
            scope_state.workspace_id,
            scope_state.backend,
            scope_state.workspace_scope_available,
            StudyScreen._material_context_fingerprint(scope_state),
        )

    @staticmethod
    def _material_context_fingerprint(scope_state: StudyScopeState) -> Optional[str]:
        if not any(
            (
                scope_state.material_source,
                scope_state.material_title,
                scope_state.material_summary,
                scope_state.material_titles,
                scope_state.source_items,
            )
        ):
            return None

        digest = hashlib.sha256()
        parts = (
            scope_state.material_source or "",
            scope_state.material_title or "",
            scope_state.material_summary or "",
            str(len(scope_state.material_titles)),
            *scope_state.material_titles[:STUDY_MATERIAL_TITLES_LIMIT],
            str(len(scope_state.source_items)),
            *(
                f"{item.source_type}:{item.source_id}:{item.label or ''}"
                for item in scope_state.source_items[:STUDY_SOURCE_ITEMS_LIMIT]
            ),
        )
        for part in parts:
            digest.update(part.encode("utf-8", errors="ignore"))
            digest.update(b"\0")
        source = scope_state.material_source or "material"
        return f"{source}:{len(scope_state.material_titles)}:{digest.hexdigest()[:12]}"

    @staticmethod
    def _clean_material_text(value: Any, *, max_length: int) -> str:
        text = sanitize_string(str(value or ""), max_length=max_length).strip()
        if not text:
            return ""
        text = _HTML_TAG_RE.sub("", text)
        text = _DANGEROUS_TEXT_RE.sub("", text).strip()
        if not validate_text_input(text, max_length=max_length, allow_html=False):
            return ""
        return text

    def _clean_material_titles(self, material_titles: tuple[str, ...]) -> tuple[str, ...]:
        cleaned: list[str] = []
        for title in material_titles:
            clean_title = self._clean_material_text(
                title,
                max_length=STUDY_MATERIAL_TITLE_LENGTH_LIMIT,
            )
            if clean_title:
                cleaned.append(clean_title)
            if len(cleaned) >= STUDY_MATERIAL_TITLES_LIMIT:
                break
        return tuple(cleaned)

    def _clean_source_items(self, source_items: tuple[StudySourceItem, ...]) -> tuple[StudySourceItem, ...]:
        cleaned: list[StudySourceItem] = []
        for item in source_items:
            if not isinstance(item, StudySourceItem):
                continue
            source_type = self._clean_material_text(item.source_type, max_length=32)
            if source_type not in {"note", "media", "message"}:
                continue
            source_id = self._clean_material_text(
                item.source_id,
                max_length=STUDY_SOURCE_ID_LENGTH_LIMIT,
            )
            if not source_id:
                continue
            label = self._clean_material_text(
                item.label,
                max_length=STUDY_MATERIAL_TITLE_LENGTH_LIMIT,
            ) or None
            excerpt_text = self._clean_material_text(
                item.excerpt_text,
                max_length=STUDY_MATERIAL_SUMMARY_LENGTH_LIMIT,
            ) or None
            locator: dict[str, Any] = {}
            if isinstance(item.locator, Mapping):
                for key, value in item.locator.items():
                    clean_key = self._clean_material_text(key, max_length=64)
                    clean_value = self._clean_material_text(value, max_length=160)
                    if clean_key and clean_value:
                        locator[clean_key] = clean_value
            cleaned.append(
                StudySourceItem(
                    source_type=source_type,
                    source_id=source_id,
                    label=label,
                    excerpt_text=excerpt_text,
                    locator=locator,
                )
            )
            if len(cleaned) >= STUDY_SOURCE_ITEMS_LIMIT:
                break
        return tuple(cleaned)

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
            material_source=self._clean_material_text(
                scope_context.material_source,
                max_length=64,
            )
            or None,
            material_title=self._clean_material_text(
                scope_context.material_title,
                max_length=STUDY_MATERIAL_TITLE_LENGTH_LIMIT,
            )
            or None,
            material_summary=self._clean_material_text(
                scope_context.material_summary,
                max_length=STUDY_MATERIAL_SUMMARY_LENGTH_LIMIT,
            )
            or None,
            material_titles=self._clean_material_titles(tuple(scope_context.material_titles or ())),
            source_items=self._clean_source_items(tuple(scope_context.source_items or ())),
        )

    def _consume_pending_scope_context(self) -> Optional[StudyScopeContext]:
        pending = getattr(self.app_instance, "pending_study_scope_context", None)
        if pending is None:
            return None
        self.app_instance.pending_study_scope_context = None
        return pending

    def _consume_pending_initial_section(self) -> Optional[str]:
        pending = getattr(self.app_instance, "pending_study_initial_section", None)
        if pending is None:
            return None
        self.app_instance.pending_study_initial_section = None
        normalized = str(pending or "").strip()
        if normalized in self._VALID_INITIAL_SECTIONS:
            return normalized
        return None

    def _apply_pending_initial_section(self) -> None:
        if self._pending_initial_section is None:
            return
        self.current_section = self._pending_initial_section
        self._pending_initial_section = None

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
        material_summary = self._material_context_summary_text()
        if self.scope_state.scope_type == StudyScopeType.WORKSPACE:
            workspace_name = self.scope_state.workspace_name or self.scope_state.workspace_id or "Workspace"
            backend = self.scope_state.backend
            if self.scope_state.error_message:
                return f"Workspace: {workspace_name} | {self.scope_state.error_message}"
            base = f"Workspace: {workspace_name} | Backend: {backend}"
            return f"{base} | {material_summary}" if material_summary else base
        return f"Global study | {material_summary}" if material_summary else "Global study"

    def _material_context_summary_text(self) -> str | None:
        if not self.scope_state.material_source and not self.scope_state.material_title:
            return None
        title = escape_markup(
            self._clean_material_text(
                self.scope_state.material_title,
                max_length=STUDY_MATERIAL_TITLE_LENGTH_LIMIT,
            )
            or "Study material"
        )
        sample_titles: list[str] = []
        for raw_title in self.scope_state.material_titles:
            clean_title = self._clean_material_text(
                raw_title,
                max_length=STUDY_MATERIAL_TITLE_LENGTH_LIMIT,
            )
            if clean_title:
                sample_titles.append(escape_markup(clean_title))
        if sample_titles:
            sample_text = ", ".join(sample_titles[:3])
            if len(sample_titles) > 3:
                sample_text = f"{sample_text}, +{len(sample_titles) - 3} more"
            return f"{title}: {sample_text}"
        summary = self._clean_material_text(
            self.scope_state.material_summary,
            max_length=STUDY_MATERIAL_SUMMARY_LENGTH_LIMIT,
        )
        if summary:
            first_line = escape_markup(summary.splitlines()[0].strip())
            return f"{title}: {first_line}"
        return title

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
        self.study_dashboard.update_source_generation_action(**self._source_generation_dashboard_state())

    def _source_generation_dashboard_state(self) -> dict[str, Any]:
        if not self.scope_state.source_items:
            return {
                "enabled": False,
                "status": "Source generation is unavailable until Study has selected source items.",
                "tooltip": "Open Study from selected Library sources in server mode to generate a study pack.",
            }
        if self.scope_state.error_message:
            return {
                "enabled": False,
                "status": self.scope_state.error_message,
                "tooltip": self.scope_state.error_message,
            }
        if self._runtime_backend() != "server":
            return {
                "enabled": False,
                "status": "Source generation requires server mode.",
                "tooltip": "Switch to server mode to generate a study pack from selected Library sources.",
            }
        if self._latest_source_study_pack:
            return {
                "enabled": True,
                "status": self._source_study_pack_ready_status(self._latest_source_study_pack),
                "tooltip": "Generate another server study pack from the selected Library sources.",
            }
        return {
            "enabled": True,
            "status": f"{len(self.scope_state.source_items)} selected source items ready for study-pack generation.",
            "tooltip": "Generate a server study pack from the selected Library sources.",
        }

    def _source_study_pack_title(self) -> str:
        return self.scope_state.material_title or "Selected Source Study Pack"

    def _source_items_payload(self) -> list[dict[str, Any]]:
        return [item.as_payload() for item in self.scope_state.source_items]

    def _source_study_pack_title_from_payload(self, study_pack: Mapping[str, Any]) -> str:
        return (
            self._clean_material_text(
                study_pack.get("title"),
                max_length=STUDY_MATERIAL_TITLE_LENGTH_LIMIT,
            )
            or "Generated study pack"
        )

    @staticmethod
    def _has_source_pack_value(value: Any) -> bool:
        return value is not None and str(value).strip() != ""

    def _source_study_pack_ready_status(self, study_pack: Mapping[str, Any]) -> str:
        title = self._source_study_pack_title_from_payload(study_pack)
        details: list[str] = []
        pack_id = self._source_pack_value_text(study_pack.get("id"))
        deck_id = self._source_pack_value_text(study_pack.get("deck_id"))
        if pack_id:
            details.append(f"pack {pack_id}")
        if deck_id:
            details.append(f"deck {deck_id}")
        detail_text = f" ({', '.join(details)})" if details else ""
        return f"Study pack ready: {title}{detail_text}. Open flashcards or resume it from the dashboard."

    def _record_source_study_pack_ready(self, study_pack: Mapping[str, Any]) -> None:
        self._latest_source_study_pack = dict(study_pack)
        title = self._source_study_pack_title_from_payload(study_pack)
        if title:
            self._recent_deck_titles = [
                title,
                *[deck_title for deck_title in self._recent_deck_titles if deck_title != title],
            ][:3]
        section = "flashcards" if self._has_source_pack_value(study_pack.get("deck_id")) else "dashboard"
        self._record_study_session(section=section, topic=title)

    def _source_pack_job_status_text(self, status: str, job_id: Any) -> str:
        clean_status = self._source_pack_status_name(status)
        clean_job_id = self._source_pack_value_text(job_id)
        if clean_job_id:
            return f"Study pack generation {clean_status}: job {clean_job_id}."
        return f"Study pack generation {clean_status}."

    def _source_pack_value_text(
        self,
        value: Any,
        *,
        max_length: int = SOURCE_STUDY_PACK_ID_LENGTH_LIMIT,
    ) -> str:
        return self._clean_material_text(value, max_length=max_length)

    def _source_pack_error_text(self, value: Any) -> str:
        return self._clean_material_text(value, max_length=SOURCE_STUDY_PACK_ERROR_LENGTH_LIMIT)

    def _source_pack_status_name(self, status: Any, *, fallback: str = "queued") -> str:
        clean_status = self._clean_material_text(status, max_length=32).lower()
        if clean_status in SOURCE_STUDY_PACK_JOB_STATUSES:
            return clean_status
        return fallback if fallback in SOURCE_STUDY_PACK_JOB_STATUSES else "queued"

    @staticmethod
    def _source_pack_job_id(value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        text = str(value).strip()
        if text.isdecimal():
            return int(text)
        return None

    def _update_source_generation_dashboard(self, *, enabled: bool, status: str, tooltip: str) -> None:
        if self.study_dashboard is not None and self.study_dashboard.is_mounted:
            self.study_dashboard.update_source_generation_action(
                enabled=enabled,
                status=status,
                tooltip=tooltip,
            )

    async def _observe_source_study_pack_job(self, study_service: Any, job_id: Any) -> bool:
        status_loader = getattr(study_service, "get_study_pack_job_status", None)
        if not callable(status_loader):
            return False

        normalized_job_id = self._source_pack_job_id(job_id)
        if normalized_job_id is None:
            status = "Study pack generation was queued, but the server returned an invalid job id."
            self._update_source_generation_dashboard(
                enabled=True,
                status=status,
                tooltip="Retry source study-pack generation or check server jobs.",
            )
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(status, severity="warning")
            return True

        last_status = "queued"
        mode = self._runtime_backend()
        for attempt in range(SOURCE_STUDY_PACK_STATUS_CHECKS):
            try:
                result = await self._maybe_await(status_loader(mode=mode, job_id=normalized_job_id))
            except Exception:
                logger.exception("Failed to observe source study-pack generation")
                status = "Study pack generation was queued, but status refresh failed."
                self._update_source_generation_dashboard(
                    enabled=True,
                    status=status,
                    tooltip="Retry source study-pack generation or check server jobs.",
                )
                notify = getattr(self.app_instance, "notify", None)
                if callable(notify):
                    notify(status, severity="warning")
                return True

            payload = result if isinstance(result, Mapping) else {}
            job = payload.get("job") if isinstance(payload.get("job"), Mapping) else {}
            last_status = self._source_pack_status_name(job.get("status"), fallback=last_status)
            study_pack = payload.get("study_pack")
            if last_status == "completed" and isinstance(study_pack, Mapping):
                self._record_source_study_pack_ready(study_pack)
                self._update_source_generation_dashboard(
                    enabled=True,
                    status=self._source_study_pack_ready_status(study_pack),
                    tooltip="Generate another server study pack from the selected Library sources.",
                )
                notify = getattr(self.app_instance, "notify", None)
                if callable(notify):
                    notify("Study pack ready.", severity="information")
                return True
            if last_status in {"failed", "cancelled"}:
                error = self._source_pack_error_text(payload.get("error"))
                status = f"Study pack generation {last_status}."
                if error:
                    status = f"{status} {error}"
                self._update_source_generation_dashboard(
                    enabled=True,
                    status=status,
                    tooltip="Retry source study-pack generation.",
                )
                notify = getattr(self.app_instance, "notify", None)
                if callable(notify):
                    notify(status, severity="error" if last_status == "failed" else "warning")
                return True
            if attempt + 1 < SOURCE_STUDY_PACK_STATUS_CHECKS:
                await asyncio.sleep(SOURCE_STUDY_PACK_STATUS_DELAY_SECONDS)

        self._update_source_generation_dashboard(
            enabled=True,
            status=self._source_pack_job_status_text(last_status, job_id),
            tooltip="Generate another server study pack from the selected Library sources.",
        )
        return True

    async def _generate_source_study_pack(self) -> None:
        self._update_source_generation_dashboard(
            enabled=False,
            status="Queuing source study-pack generation...",
            tooltip="Source study-pack generation is already queued.",
        )

        state = self._source_generation_dashboard_state()
        if not state["enabled"]:
            self._update_source_generation_dashboard(**state)
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(state["status"], severity="warning")
            return

        study_service = getattr(self.app_instance, "study_scope_service", None)
        create_job = getattr(study_service, "create_study_pack_job", None)
        if not callable(create_job):
            status = "Study pack generation is unavailable in this runtime."
            self._update_source_generation_dashboard(
                enabled=False,
                status=status,
                tooltip=status,
            )
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(status, severity="warning")
            return

        try:
            result = await self._maybe_await(
                create_job(
                    mode=self._runtime_backend(),
                    title=self._source_study_pack_title(),
                    workspace_id=(
                        self.scope_state.workspace_id
                        if self.scope_state.scope_type == StudyScopeType.WORKSPACE
                        else None
                    ),
                    source_items=self._source_items_payload(),
                )
            )
        except Exception:
            logger.exception("Failed to queue source study-pack generation")
            status = "Failed to queue source study-pack generation."
            self._update_source_generation_dashboard(
                enabled=True,
                status=status,
                tooltip="Retry source study-pack generation.",
            )
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(status, severity="error")
            return

        job = result.get("job") if isinstance(result, Mapping) else None
        job_id = job.get("id") if isinstance(job, Mapping) else None
        normalized_job_id = self._source_pack_job_id(job_id)
        job_status = self._source_pack_status_name(job.get("status") if isinstance(job, Mapping) else None)
        observing_job = (
            normalized_job_id is not None
            and callable(getattr(study_service, "get_study_pack_job_status", None))
        )
        self._update_source_generation_dashboard(
            enabled=not observing_job,
            status=self._source_pack_job_status_text(
                job_status,
                normalized_job_id if normalized_job_id is not None else job_id,
            ),
            tooltip=(
                "Source study-pack generation is queued; observing server completion."
                if observing_job
                else "Generate another server study pack from the selected Library sources."
            ),
        )
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify("Study pack generation queued.", severity="information")
        if normalized_job_id is not None and await self._observe_source_study_pack_job(
            study_service,
            normalized_job_id,
        ):
            return

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
            self.quiz_session_widget.set_start_enabled(False, QUIZ_START_SELECT_TOOLTIP)
            self.quiz_session_widget.set_review_in_chat_enabled(False, QUIZ_REVIEW_SELECT_TOOLTIP)
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
        start_enabled = bool(quiz_name) and scope_available and not attempt_active
        if start_enabled:
            tooltip = QUIZ_START_ENABLED_TOOLTIP
        elif quiz_name and not scope_available:
            tooltip = QUIZ_START_SCOPE_UNAVAILABLE_TOOLTIP
        elif quiz_name and attempt_active:
            tooltip = QUIZ_START_ATTEMPT_ACTIVE_TOOLTIP
        elif getattr(controller, "has_quizzes", False):
            tooltip = QUIZ_START_SELECT_TOOLTIP
        else:
            tooltip = QUIZ_START_NO_QUIZZES_TOOLTIP
        self.quiz_session_widget.set_start_enabled(start_enabled, tooltip)

        open_chat = getattr(self.app_instance, "open_chat_with_handoff", None)
        if not quiz_name:
            self.quiz_session_widget.set_review_in_chat_enabled(False, QUIZ_REVIEW_SELECT_TOOLTIP)
        elif not callable(open_chat):
            self.quiz_session_widget.set_review_in_chat_enabled(
                False,
                QUIZ_REVIEW_HANDOFF_UNAVAILABLE_TOOLTIP,
            )
        else:
            self.quiz_session_widget.set_review_in_chat_enabled(True, QUIZ_REVIEW_ENABLED_TOOLTIP)

    def _build_quiz_chat_handoff_payload(self) -> ChatHandoffPayload | None:
        controller = getattr(self.study_window_widget, "quizzes_controller", None)
        if controller is None:
            return None

        label_getter = getattr(controller, "selected_quiz_label", None)
        quiz_name = label_getter() if callable(label_getter) else None
        if not quiz_name:
            return None

        id_getter = getattr(controller, "_selected_quiz_id", None)
        quiz_id = id_getter() if callable(id_getter) else None
        status = self._status_text_from_window("#quiz-attempt-status") or "No active attempt."
        question = self._status_text_from_window("#quiz-attempt-question")
        attempt_questions = list(getattr(controller, "current_attempt_questions", None) or [])

        body_lines = [
            f"Quiz: {quiz_name}",
            f"Scope: {self._scope_summary_text()}",
            f"Quiz status: {status}",
        ]
        if attempt_questions:
            body_lines.append(f"Active attempt questions: {len(attempt_questions)}")
        if question:
            body_lines.append(f"Current question: {question}")

        runtime_backend = self._runtime_backend()
        scope_type = self.scope_state.scope_type.value
        workspace_id = (
            self.scope_state.workspace_id
            if scope_type == StudyScopeType.WORKSPACE.value
            else None
        )
        source_owner = "workspace" if workspace_id else runtime_backend
        backend_contracts = (
            {"workspace_isolation": {"workspace_scope_id": workspace_id}}
            if workspace_id
            else {}
        )
        return ChatHandoffPayload.from_source_content(
            source="study",
            item_type="quiz",
            title=str(quiz_name),
            body="\n".join(body_lines),
            source_id=str(quiz_id) if quiz_id else None,
            display_summary=f"Study quiz: {quiz_name}",
            suggested_prompt="Help me review this quiz and identify what to study next.",
            runtime_backend=runtime_backend,
            source_owner=source_owner,
            source_selector_state=source_owner,
            scope_type=scope_type,
            workspace_id=workspace_id,
            backend_contracts=backend_contracts,
            metadata={"scope_summary": self._scope_summary_text()},
        )

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
        self.study_materials = list(next_state.material_titles)

        if previous_key == next_key and not force_controller_notify:
            return

        self._latest_source_study_pack = None

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
        self._apply_pending_initial_section()
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
                    "material_source": self.scope_state.material_source,
                    "material_title": self.scope_state.material_title,
                    "material_summary": self.scope_state.material_summary,
                    "material_titles": list(self.scope_state.material_titles),
                    "source_items": [item.as_payload() for item in self.scope_state.source_items],
                },
                "study_section": self.current_section,
                "current_study_session": self.current_study_session,
            }
        )
        return state

    @staticmethod
    def _restored_source_item_locator(item: dict[str, Any]) -> dict[str, Any]:
        locator = item.get("locator")
        if isinstance(locator, Mapping):
            return dict(locator)
        return {}

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
                    material_source=saved_scope.get("material_source"),
                    material_title=saved_scope.get("material_title"),
                    material_summary=saved_scope.get("material_summary"),
                    material_titles=tuple(saved_scope.get("material_titles") or ()),
                    source_items=tuple(
                        StudySourceItem(
                            source_type=str(item.get("source_type") or ""),
                            source_id=str(item.get("source_id") or ""),
                            label=item.get("label"),
                            excerpt_text=item.get("excerpt_text"),
                            locator=self._restored_source_item_locator(item),
                        )
                        for item in list(saved_scope.get("source_items") or [])
                        if isinstance(item, dict)
                    ),
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

    @on(Button.Pressed, "#study-generate-source-pack")
    def handle_generate_source_pack(self) -> None:
        self.run_worker(self._generate_source_study_pack(), exclusive=True)

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

    @on(Button.Pressed, "#quiz-open-in-chat")
    def handle_quiz_review_in_chat(self) -> None:
        open_chat = getattr(self.app_instance, "open_chat_with_handoff", None)
        payload = self._build_quiz_chat_handoff_payload()
        if not callable(open_chat) or payload is None:
            return
        open_chat(payload)
