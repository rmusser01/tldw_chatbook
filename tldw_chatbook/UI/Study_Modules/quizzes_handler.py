"""Screen-local quizzes controller for the Study window."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger
from textual.widgets import Input, Label, ListItem, ListView, Select, Static, TextArea

from ...Study_Interop import LocalQuizService, QuizScopeService, ServerQuizService
from ..Screens.study_scope_models import StudyScopeType

if TYPE_CHECKING:
    from ..Study_Window import StudyWindow


class StudyQuizzesController:
    """Own quiz CRUD and attempt interactions inside the Study screen."""

    def __init__(self, window: "StudyWindow"):
        self.window = window
        self.app_instance = window.app_instance
        self.current_attempt_id: Optional[str] = None
        self.current_attempt_questions: list[dict[str, Any]] = []
        self.current_attempt_answers: list[dict[str, Any]] = []
        self.current_question_index: int = 0
        self.current_quiz_questions: list[dict[str, Any]] = []
        self.current_attempt_history: list[dict[str, Any]] = []
        self._scope_service_cache: Optional[QuizScopeService] = None
        self.has_quizzes: bool = False
        self._suppress_quiz_change_events: bool = False

    def _current_mode(self) -> str:
        getter = getattr(self.app_instance, "get_authoritative_runtime_source", None)
        if callable(getter):
            normalized = str(getter() or "").strip().lower()
            if normalized in {"local", "server"}:
                return normalized
        candidates = (
            getattr(self.window, "runtime_backend", None),
            getattr(self.app_instance, "runtime_backend", None),
            getattr(self.app_instance, "current_runtime_backend", None),
        )
        for candidate in candidates:
            normalized = str(candidate or "").strip().lower()
            if normalized in {"local", "server"}:
                return normalized
        return "local"

    def _notify(self, message: str, severity: str = "warning") -> None:
        notifier = getattr(self.window, "notify", None)
        if callable(notifier):
            notifier(message, severity=severity)
            return
        notifier = getattr(self.app_instance, "notify", None)
        if callable(notifier):
            notifier(message, severity=severity)

    def _scope_state(self) -> Any:
        scope_state = getattr(self.window, "current_scope_state", None)
        if scope_state is not None:
            return scope_state
        screen = getattr(self.window, "screen", None)
        return getattr(screen, "current_scope", None)

    @staticmethod
    def _scope_type_value(scope_state: Any) -> str:
        scope_type = getattr(scope_state, "scope_type", None)
        scope_value = getattr(scope_type, "value", scope_type)
        return str(scope_value or "global").strip().lower()

    def _scope_type(self) -> str:
        return self._scope_type_value(self._scope_state())

    def _workspace_id(self) -> Optional[str]:
        scope_state = self._scope_state()
        if self._scope_type_value(scope_state) != StudyScopeType.WORKSPACE.value:
            return None
        workspace_id = getattr(scope_state, "workspace_id", None)
        return str(workspace_id or "").strip() or None

    def _scope_is_available(self) -> bool:
        scope_state = self._scope_state()
        if scope_state is None:
            return True
        if self._scope_type_value(scope_state) != StudyScopeType.WORKSPACE.value:
            return True
        return bool(getattr(scope_state, "workspace_scope_available", False)) and not bool(
            str(getattr(scope_state, "error_message", "") or "").strip()
        )

    def _scope_unavailable_message(self) -> str:
        scope_state = self._scope_state()
        message = str(getattr(scope_state, "error_message", "") or "").strip()
        if message:
            return message
        if self._scope_type() == StudyScopeType.WORKSPACE.value:
            backend = str(getattr(scope_state, "backend", "") or "unknown")
            return f"Workspace study is unavailable on {backend}."
        return "Study quizzes backend is unavailable."

    def _scope_empty_message(self) -> str:
        if self._scope_type() == StudyScopeType.WORKSPACE.value:
            return (
                "No quizzes in this workspace yet. Create a workspace quiz, "
                "or switch to Global Study to practice existing quizzes."
            )
        return "No quizzes yet. Create a quiz, add questions, then start an attempt."

    def _scope_arguments(self) -> dict[str, Any]:
        scope_type = self._scope_type()
        return {
            "scope_type": scope_type,
            "workspace_id": self._workspace_id() if scope_type == StudyScopeType.WORKSPACE.value else None,
        }

    def _workspace_create_arguments(self) -> dict[str, Any]:
        if self._current_mode() != "server" or self._scope_type() != StudyScopeType.WORKSPACE.value:
            return {"scope_type": self._scope_type(), "workspace_id": None}
        return self._scope_arguments()

    def _policy_action_allowed(self, action_id: str) -> bool:
        checker = getattr(self.app_instance, "require_ui_action_allowed", None)
        if not callable(checker):
            return True
        decision = checker(
            action_id=action_id,
            scope_type=self._scope_type(),
        )
        return bool(getattr(decision, "allowed", False))

    def _sync_quiz_controls(self) -> None:
        configure_controls = getattr(self.window, "_configure_quizzes_lifecycle_controls", None)
        if callable(configure_controls):
            configure_controls()

    def _notify_shell_state_changed(self) -> None:
        notifier = getattr(self.window, "_notify_shell_state_changed", None)
        if callable(notifier):
            notifier()

    @staticmethod
    def _is_blank_select_value(value: Any) -> bool:
        return value in {None, "", False, Select.BLANK} or str(value).startswith("Select.")

    def _attempt_active(self) -> bool:
        return self.current_attempt_id is not None and bool(self.current_attempt_questions)

    def reset_quiz_panel(self, message: str) -> None:
        self.current_attempt_id = None
        self.current_attempt_questions = []
        self.current_attempt_answers = []
        self.current_question_index = 0
        self.current_quiz_questions = []
        self.current_attempt_history = []
        self.has_quizzes = False

        try:
            quiz_select = self.window.query_one("#quiz-select", Select)
            self._suppress_quiz_change_events = True
            quiz_select.set_options([("No quizzes available", Select.BLANK)])
            quiz_select.clear()
        except Exception:
            pass
        finally:
            self._suppress_quiz_change_events = False

        try:
            question_list = self.window.query_one("#quiz-question-list", ListView)
            question_list.remove_children()
        except Exception:
            pass

        try:
            history_select = self.window.query_one("#quiz-attempt-history-select", Select)
            history_select.set_options([("No attempt history", Select.BLANK)])
            history_select.clear()
        except Exception:
            pass

        try:
            self.window.query_one("#quiz-question-text", TextArea).text = ""
            self.window.query_one("#quiz-correct-answer-input", Input).value = ""
            self.window.query_one("#quiz-answer-input", Input).value = ""
        except Exception:
            pass

        self._set_attempt_question("")
        self._set_attempt_history_summary("")
        self._set_attempt_status(message)
        self._sync_quiz_controls()
        self._notify_shell_state_changed()

    def _scope_service(self) -> Optional[QuizScopeService]:
        service = getattr(self.app_instance, "study_quiz_scope_service", None)
        if service is not None:
            return service
        if self._scope_service_cache is not None:
            return self._scope_service_cache

        local_service = None
        db = getattr(self.app_instance, "chachanotes_db", None)
        if db is not None:
            local_service = LocalQuizService(db=db)

        try:
            server_service = ServerQuizService.from_config(getattr(self.app_instance, "app_config", {}) or {})
        except ValueError:
            server_service = ServerQuizService(client=None)

        if local_service is None and getattr(server_service, "client", None) is None:
            return None

        self._scope_service_cache = QuizScopeService(
            local_service=local_service,
            server_service=server_service,
        )
        return self._scope_service_cache

    def _selected_quiz_id(self) -> Optional[str]:
        try:
            quiz_select = self.window.query_one("#quiz-select", Select)
        except Exception:
            return None
        value = getattr(quiz_select, "value", None)
        if self._is_blank_select_value(value):
            return None
        return str(value)

    def selected_quiz_label(self) -> Optional[str]:
        selected_quiz_id = self._selected_quiz_id()
        if selected_quiz_id is None:
            return None
        try:
            quiz_select = self.window.query_one("#quiz-select", Select)
        except Exception:
            return None
        for option in getattr(quiz_select, "_options", []):
            if len(option) >= 2 and str(option[1]) == selected_quiz_id:
                return str(option[0] or "").strip() or None
        return None

    def _set_attempt_status(self, message: str) -> None:
        try:
            self.window.query_one("#quiz-attempt-status", Static).update(message)
        except Exception:
            pass

    def _set_attempt_question(self, question_text: str = "") -> None:
        try:
            self.window.query_one("#quiz-attempt-question", Static).update(question_text)
        except Exception:
            pass

    def _set_attempt_history_summary(self, message: str = "") -> None:
        try:
            self.window.query_one("#quiz-attempt-history-summary", Static).update(message)
        except Exception:
            pass

    def _selected_attempt_id(self) -> Optional[str]:
        try:
            history_select = self.window.query_one("#quiz-attempt-history-select", Select)
        except Exception:
            return None
        value = getattr(history_select, "value", None)
        if self._is_blank_select_value(value):
            return None
        return str(value)

    def _selected_question(self) -> Optional[dict[str, Any]]:
        if not self.current_quiz_questions:
            return None
        try:
            question_list = self.window.query_one("#quiz-question-list", ListView)
        except Exception:
            return None
        index = getattr(question_list, "index", None)
        try:
            normalized_index = int(index)
        except (TypeError, ValueError):
            normalized_index = 0
        if normalized_index < 0 or normalized_index >= len(self.current_quiz_questions):
            return None
        return self.current_quiz_questions[normalized_index]

    @staticmethod
    def _format_attempt_option_label(attempt: dict[str, Any]) -> str:
        completed_at = str(attempt.get("completed_at") or attempt.get("started_at") or "Attempt")
        completed_at = completed_at.replace("T", " ").replace("Z", "")
        score = attempt.get("score")
        total_possible = attempt.get("total_possible")
        if score is not None and total_possible is not None:
            return f"{completed_at} | {score}/{total_possible}"
        return completed_at

    def reset_attempt_panel(self, message: str) -> None:
        self.current_attempt_id = None
        self.current_attempt_questions = []
        self.current_attempt_answers = []
        self.current_question_index = 0
        self.current_quiz_questions = []
        self.window.query_one("#quiz-answer-input", Input).value = ""
        self._set_attempt_question("")
        self._set_attempt_status(message)
        self._set_attempt_history_summary("")
        self._sync_quiz_controls()
        self._notify_shell_state_changed()

    async def _wait_for_widgets(self) -> bool:
        for _ in range(25):
            try:
                quiz_select = self.window.query_one("#quiz-select", Select)
            except Exception:
                await asyncio.sleep(0.01)
                continue

            if getattr(quiz_select, "is_mounted", False):
                return True
            await asyncio.sleep(0.01)
        return False

    async def initialize_view(self) -> None:
        service = self._scope_service()
        if service is None:
            self.reset_quiz_panel("Study quizzes backend is unavailable.")
            return
        if not self._scope_is_available():
            self.reset_quiz_panel(self._scope_unavailable_message())
            return
        if not await self._wait_for_widgets():
            logger.warning("Study quizzes UI did not finish mounting before initialization")
            self.reset_quiz_panel("Study quizzes UI is still loading.")
            return
        await self.refresh_quizzes()

    async def refresh_quizzes(
        self,
        *,
        preserve_selection: bool = True,
        preferred_selection: Optional[str] = None,
    ) -> None:
        service = self._scope_service()
        if service is None:
            self.reset_quiz_panel("Study quizzes backend is unavailable.")
            return
        if not self._scope_is_available():
            self.reset_quiz_panel(self._scope_unavailable_message())
            return

        selected_before = self._selected_quiz_id() if preserve_selection else None
        quizzes = await service.list_quizzes(
            mode=self._current_mode(),
            **self._scope_arguments(),
            q=None,
            limit=100,
            offset=0,
        )

        quiz_select = self.window.query_one("#quiz-select", Select)
        options = [
            (str(quiz.get("name") or "Unnamed quiz"), str(quiz.get("backing_id")))
            for quiz in quizzes
            if quiz.get("backing_id") not in {None, ""}
        ]
        self.has_quizzes = bool(options)
        if not options:
            if self._scope_type() == StudyScopeType.WORKSPACE.value:
                self.reset_quiz_panel(self._scope_empty_message())
                return
            options = [("No quizzes available", Select.BLANK)]
        quiz_select.set_options(options)

        available_values = [option[1] for option in options]
        try:
            self._suppress_quiz_change_events = True
            if preferred_selection in available_values:
                quiz_select.value = str(preferred_selection)
            elif selected_before in available_values:
                quiz_select.value = str(selected_before)
            elif self.has_quizzes:
                quiz_select.value = str(available_values[0])
            else:
                quiz_select.clear()
        finally:
            self._suppress_quiz_change_events = False

        await self.refresh_questions()
        await self.refresh_attempt_history()
        self._sync_quiz_controls()
        self._notify_shell_state_changed()

    async def refresh_questions(self) -> None:
        service = self._scope_service()
        if service is None:
            self.reset_quiz_panel("Study quizzes backend is unavailable.")
            return
        if not self._scope_is_available():
            self.reset_quiz_panel(self._scope_unavailable_message())
            return

        list_view = self.window.query_one("#quiz-question-list", ListView)
        await list_view.clear()

        quiz_id = self._selected_quiz_id()
        if quiz_id is None:
            if not self.has_quizzes:
                self.reset_quiz_panel(self._scope_empty_message())
            else:
                self.reset_quiz_panel("Select a quiz to manage its questions.")
            return

        questions = await service.list_questions(
            mode=self._current_mode(),
            quiz_id=quiz_id,
            q=None,
            include_answers=True,
            limit=100,
            offset=0,
        )
        self.current_quiz_questions = list(questions)

        if not questions:
            await list_view.append(ListItem(Label("No questions in this quiz.")))
        else:
            for question in questions:
                label = str(question.get("question_text") or "Untitled question")
                await list_view.append(ListItem(Label(label)))
            list_view.index = 0

        if self.current_attempt_id is None:
            self._set_attempt_question("")
            self.window.query_one("#quiz-answer-input", Input).value = ""
            self._set_attempt_status("Ready to manage selected quiz.")
        self._notify_shell_state_changed()

    async def refresh_attempt_history(
        self,
        *,
        preserve_selection: bool = True,
        preferred_selection: Optional[str] = None,
    ) -> None:
        service = self._scope_service()
        if service is None:
            self._set_attempt_history_summary("")
            return
        if not self._scope_is_available():
            self._set_attempt_history_summary("")
            return

        history_select = self.window.query_one("#quiz-attempt-history-select", Select)
        selected_before = self._selected_attempt_id() if preserve_selection else None
        self._set_attempt_history_summary("")
        quiz_id = self._selected_quiz_id()
        if quiz_id is None:
            self.current_attempt_history = []
            history_select.set_options([("No attempt history", Select.BLANK)])
            history_select.clear()
            self._set_attempt_history_summary("")
            return

        attempts = await service.list_attempts(
            mode=self._current_mode(),
            **self._scope_arguments(),
            quiz_id=quiz_id,
            limit=100,
            offset=0,
        )
        self.current_attempt_history = list(attempts or [])
        options = [
            (
                self._format_attempt_option_label(attempt),
                str(attempt.get("backing_id")),
            )
            for attempt in self.current_attempt_history
            if attempt.get("backing_id") not in {None, ""}
        ]
        if not options:
            history_select.set_options([("No attempt history", Select.BLANK)])
            history_select.clear()
            self._set_attempt_history_summary("")
            self._notify_shell_state_changed()
            return

        history_select.set_options(options)
        available_values = {option[1] for option in options}
        if preferred_selection in available_values:
            history_select.value = str(preferred_selection)
        elif selected_before in available_values:
            history_select.value = str(selected_before)
        else:
            history_select.value = str(options[0][1])
        self._notify_shell_state_changed()

    async def delete_quiz(self) -> None:
        service = self._scope_service()
        if service is None:
            self._notify("Study quizzes backend is unavailable.")
            return
        if self._attempt_active():
            self._notify("Finish the current attempt before changing quizzes.")
            return
        if not self._scope_is_available():
            self._notify(self._scope_unavailable_message())
            return

        quiz_id = self._selected_quiz_id()
        if quiz_id is None:
            self._notify("Select a quiz before deleting it.")
            return

        try:
            deleted = await service.delete_quiz(
                mode=self._current_mode(),
                quiz_id=quiz_id,
                expected_version=None,
                hard_delete=False,
            )
        except Exception:
            logger.error("Failed to delete quiz", exc_info=True)
            self._notify("Failed to delete quiz.", severity="error")
            return

        if not deleted:
            self._notify("Quiz could not be deleted.", severity="error")
            return

        await self.refresh_quizzes(preserve_selection=False)
        if self.has_quizzes:
            self._set_attempt_status("Quiz deleted.")
        self._notify_shell_state_changed()

    async def delete_question(self) -> None:
        service = self._scope_service()
        if service is None:
            self._notify("Study quizzes backend is unavailable.")
            return
        if self._attempt_active():
            self._notify("Finish the current attempt before changing quizzes.")
            return
        if not self._scope_is_available():
            self._notify(self._scope_unavailable_message())
            return

        quiz_id = self._selected_quiz_id()
        selected_question = self._selected_question()
        if quiz_id is None:
            self._notify("Select a quiz before deleting a question.")
            return
        if selected_question is None:
            self._notify("Select a question before deleting it.")
            return

        question_id = str(selected_question.get("backing_id") or selected_question.get("id") or "")
        if not question_id:
            self._notify("Selected question is missing an identifier.")
            return

        try:
            deleted = await service.delete_question(
                mode=self._current_mode(),
                quiz_id=quiz_id,
                question_id=question_id,
                expected_version=None,
                hard_delete=False,
            )
        except Exception:
            logger.error("Failed to delete quiz question", exc_info=True)
            self._notify("Failed to delete question.", severity="error")
            return

        if not deleted:
            self._notify("Question could not be deleted.", severity="error")
            return

        await self.refresh_quizzes(preserve_selection=True, preferred_selection=quiz_id)
        self._set_attempt_status("Question deleted.")
        self._notify_shell_state_changed()

    async def create_quiz(self) -> None:
        service = self._scope_service()
        if service is None:
            self._notify("Study quizzes backend is unavailable.")
            return
        if self._attempt_active():
            self._notify("Finish the current attempt before changing quizzes.")
            return
        if not self._scope_is_available():
            self._notify(self._scope_unavailable_message())
            return

        name_input = self.window.query_one("#new-quiz-name-input", Input)
        description_input = self.window.query_one("#new-quiz-description-input", Input)
        name = str(name_input.value or "").strip()
        description = str(description_input.value or "").strip() or None
        if not name:
            self._notify("Quiz name is required.")
            return

        try:
            created = await service.create_quiz(
                mode=self._current_mode(),
                **self._workspace_create_arguments(),
                name=name,
                description=description,
                time_limit_seconds=None,
                passing_score=None,
            )
        except Exception:
            logger.error("Failed to create quiz", exc_info=True)
            self._notify("Failed to create quiz.", severity="error")
            return

        name_input.value = ""
        description_input.value = ""
        created_id = str(created.get("backing_id") or created.get("record_id") or Select.BLANK)
        await self.refresh_quizzes(preserve_selection=False, preferred_selection=created_id)
        self._set_attempt_status(f"Quiz '{created.get('name', name)}' created.")
        self._notify_shell_state_changed()

    async def create_question(self) -> None:
        service = self._scope_service()
        if service is None:
            self._notify("Study quizzes backend is unavailable.")
            return
        if self._attempt_active():
            self._notify("Finish the current attempt before changing quizzes.")
            return
        if not self._scope_is_available():
            self._notify(self._scope_unavailable_message())
            return

        quiz_id = self._selected_quiz_id()
        if quiz_id is None:
            self._notify("Select a quiz before creating a question.")
            return

        question_text_widget = self.window.query_one("#quiz-question-text", TextArea)
        correct_answer_widget = self.window.query_one("#quiz-correct-answer-input", Input)
        question_text = str(question_text_widget.text or "").strip()
        correct_answer = str(correct_answer_widget.value or "").strip()
        if not question_text:
            self._notify("Question text is required.")
            return
        if not correct_answer:
            self._notify("Correct answer is required.")
            return

        try:
            await service.create_question(
                mode=self._current_mode(),
                quiz_id=quiz_id,
                question_type="fill_blank",
                question_text=question_text,
                correct_answer=correct_answer,
                options=None,
                explanation=None,
                hint=None,
                hint_penalty_points=0,
                points=1,
                order_index=0,
                tags=None,
                source_citations=None,
            )
        except Exception:
            logger.error("Failed to create quiz question", exc_info=True)
            self._notify("Failed to create quiz question.", severity="error")
            return

        question_text_widget.text = ""
        correct_answer_widget.value = ""
        await self.refresh_questions()
        self._set_attempt_status("Question created.")
        self._notify_shell_state_changed()

    async def start_attempt(self) -> None:
        service = self._scope_service()
        if service is None:
            self._notify("Study quizzes backend is unavailable.")
            return
        if self._attempt_active():
            self._notify("Finish the current attempt before starting a new one.")
            return
        if not self._scope_is_available():
            self._notify(self._scope_unavailable_message())
            return

        quiz_id = self._selected_quiz_id()
        if quiz_id is None:
            self._notify("Select a quiz before starting an attempt.")
            return

        if not self._policy_action_allowed(f"quiz.attempt.create.{self._current_mode()}"):
            return
        try:
            attempt = await service.start_attempt(
                mode=self._current_mode(),
                **self._scope_arguments(),
                quiz_id=quiz_id,
            )
        except Exception:
            logger.error("Failed to start quiz attempt", exc_info=True)
            self._notify("Failed to start quiz attempt.", severity="error")
            return

        self.current_attempt_id = str(attempt.get("backing_id") or attempt.get("id") or "")
        self.current_attempt_questions = list(attempt.get("questions") or [])
        self.current_attempt_answers = []
        self.current_question_index = 0
        self.window.query_one("#quiz-answer-input", Input).value = ""

        if not self.current_attempt_questions:
            self.reset_attempt_panel("This quiz has no questions yet.")
            return

        self._show_current_question()
        self._sync_quiz_controls()
        self._notify_shell_state_changed()

    def _show_current_question(self) -> None:
        if not self.current_attempt_questions:
            self.reset_attempt_panel("No quiz questions are available.")
            return

        current_question = self.current_attempt_questions[self.current_question_index]
        question_text = str(current_question.get("question_text") or "")
        self._set_attempt_question(question_text)
        self._set_attempt_status(
            f"Question {self.current_question_index + 1} of {len(self.current_attempt_questions)}."
        )

    async def submit_current_answer(self) -> None:
        service = self._scope_service()
        if service is None:
            self._notify("Study quizzes backend is unavailable.")
            return
        if not self._scope_is_available():
            self._notify(self._scope_unavailable_message())
            return
        if self.current_attempt_id is None or not self.current_attempt_questions:
            self._notify("Start a quiz attempt before submitting answers.")
            return

        answer_widget = self.window.query_one("#quiz-answer-input", Input)
        user_answer = str(answer_widget.value or "").strip()
        if not user_answer:
            self._notify("Answer is required.")
            return

        current_question = self.current_attempt_questions[self.current_question_index]
        question_id = str(current_question.get("backing_id") or current_question.get("id") or "")
        self.current_attempt_answers.append(
            {
                "question_id": question_id,
                "user_answer": user_answer,
            }
        )

        if self.current_question_index + 1 < len(self.current_attempt_questions):
            self.current_question_index += 1
            answer_widget.value = ""
            self._show_current_question()
            self._sync_quiz_controls()
            self._notify_shell_state_changed()
            return

        try:
            submitted = await service.submit_attempt(
                mode=self._current_mode(),
                **self._scope_arguments(),
                attempt_id=self.current_attempt_id,
                answers=list(self.current_attempt_answers),
            )
        except Exception:
            logger.error("Failed to submit quiz attempt", exc_info=True)
            self._notify("Failed to submit quiz attempt.", severity="error")
            return

        score = submitted.get("score")
        total_possible = submitted.get("total_possible")
        submitted_attempt_id = str(submitted.get("backing_id") or submitted.get("id") or "")
        answer_widget.value = ""
        self._set_attempt_question("Attempt complete.")
        self._set_attempt_status(f"Score: {score} / {total_possible}")
        self.current_attempt_id = None
        self.current_attempt_questions = []
        self.current_attempt_answers = []
        self.current_question_index = 0
        await self.refresh_attempt_history(
            preserve_selection=False,
            preferred_selection=submitted_attempt_id or None,
        )
        self._sync_quiz_controls()
        self._notify_shell_state_changed()

    async def load_selected_attempt(self) -> None:
        service = self._scope_service()
        if service is None:
            self._notify("Study quizzes backend is unavailable.")
            return
        if self._attempt_active():
            self._notify("Finish the current attempt before loading another attempt.")
            return
        if not self._scope_is_available():
            self._notify(self._scope_unavailable_message())
            return

        attempt_id = self._selected_attempt_id()
        if attempt_id is None:
            self._notify("Select a past attempt before loading it.")
            return

        try:
            attempt = await service.get_attempt(
                mode=self._current_mode(),
                **self._scope_arguments(),
                attempt_id=attempt_id,
                include_questions=True,
                include_answers=True,
            )
        except Exception:
            logger.error("Failed to load quiz attempt history", exc_info=True)
            self._notify("Failed to load attempt history.", severity="error")
            return

        self.current_attempt_id = None
        self.current_attempt_questions = []
        self.current_attempt_answers = []
        self.current_question_index = 0
        self.window.query_one("#quiz-answer-input", Input).value = ""
        self._set_attempt_question("Loaded completed attempt.")
        self._set_attempt_status(f"Score: {attempt.get('score')} / {attempt.get('total_possible')}")

        answers_by_question_id = {
            str(answer.get("question_id")): answer
            for answer in list(attempt.get("answers") or [])
            if answer.get("question_id") is not None
        }
        summary_lines: list[str] = []
        for index, question in enumerate(list(attempt.get("questions") or []), start=1):
            question_id = str(question.get("backing_id") or question.get("id") or "")
            answer = answers_by_question_id.get(question_id, {})
            summary_lines.append(f"{index}. {question.get('question_text') or 'Untitled question'}")
            if answer:
                summary_lines.append(f"Answer: {answer.get('user_answer') or ''}")
                if answer.get("correct_answer") is not None:
                    summary_lines.append(f"Correct: {answer.get('correct_answer')}")
                points_awarded = answer.get("points_awarded")
                if points_awarded is not None:
                    summary_lines.append(f"Points: {points_awarded}")
        self._set_attempt_history_summary("\n".join(summary_lines))
        self._sync_quiz_controls()
        self._notify_shell_state_changed()

    async def handle_quiz_changed(self) -> None:
        if self._suppress_quiz_change_events:
            return
        self.current_attempt_id = None
        self.current_attempt_questions = []
        self.current_attempt_answers = []
        self.current_question_index = 0
        await self.refresh_questions()
        await self.refresh_attempt_history()
        self._sync_quiz_controls()
        self._notify_shell_state_changed()

    def handle_scope_changed(self) -> None:
        """Reset controller-local state before scoped study data reloads."""
        self.reset_quiz_panel(
            self._scope_unavailable_message() if not self._scope_is_available() else "Select a quiz to manage its questions."
        )
