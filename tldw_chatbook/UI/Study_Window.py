# Study_Window.py
# Description: Study tab with Structured Learning, Anki/Flashcards, and Mindmaps
#
# Imports
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from pathlib import Path
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer, Container
from textual.widgets import Label, Button, TextArea, Select, Input, Static, ListView, ListItem, Tree, Switch
from textual.widget import Widget
from textual.reactive import reactive
from textual.message import Message
from textual import work
from textual.binding import Binding
from loguru import logger

# Local imports
from tldw_chatbook.config import get_cli_setting
from tldw_chatbook.Utils.input_validation import validate_text_input
from tldw_chatbook.UI.Study_Modules import StudyFlashcardsController, StudyQuizzesController
# StudyDB import removed - using ChaChaNotes_DB instead

# Type checking imports
if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli

#######################################################################################################################
#
# Classes:

class StructuredLearningWidget(Widget):
    """Widget for structured learning paths and topics"""
    
    DEFAULT_CSS = """
    StructuredLearningWidget {
        height: 100%;
        width: 100%;
    }
    
    .structured-learning-container {
        padding: 1;
        height: 100%;
    }
    
    .topic-tree {
        height: 1fr;
        width: 50%;
        border: round $surface;
        margin-right: 1;
    }
    
    .topic-content {
        height: 1fr;
        width: 50%;
        border: round $surface;
        padding: 1;
    }
    
    .section-title {
        text-style: bold;
        margin-bottom: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Compose the Structured Learning UI"""
        with ScrollableContainer(classes="structured-learning-container"):
            yield Label("📚 Structured Learning", classes="section-title")
            
            with Horizontal():
                # Topic tree on the left
                with Vertical(classes="topic-tree-container"):
                    yield Label("Learning Topics:", classes="subsection-title")
                    yield Tree("Learning Paths", id="topic-tree", classes="topic-tree")
                
                # Content display on the right
                with Vertical(classes="topic-content-container"):
                    yield Label("Topic Content:", classes="subsection-title")
                    yield TextArea(
                        "Select a topic from the tree to view content...",
                        id="topic-content",
                        classes="topic-content",
                        disabled=True
                    )
            
            # Add new topic section
            yield Label("Add New Topic:", classes="subsection-title")
            with Horizontal(classes="form-row"):
                yield Input(placeholder="Topic title...", id="new-topic-title")
                yield Button("Add Topic", id="add-topic-btn", variant="primary")


class AnkiFlashcardsWidget(Widget):
    """Widget for Anki-compatible flashcards with spaced repetition"""
    
    DEFAULT_CSS = """
    AnkiFlashcardsWidget {
        height: 100%;
        width: 100%;
    }
    
    .flashcards-container {
        padding: 1;
        height: 100%;
    }
    
    .card-editor {
        border: round $surface;
        padding: 1;
        margin-bottom: 1;
    }

    .deck-controls {
        height: auto;
        margin-bottom: 1;
    }

    .deck-lifecycle-controls {
        height: auto;
        margin-bottom: 1;
    }

    .deck-warning {
        margin-bottom: 1;
        color: $text-muted;
    }

    .search-row {
        height: auto;
        margin-bottom: 1;
    }
    
    .card-list {
        height: 10;
        border: round $surface;
        margin-bottom: 1;
    }
    
    .review-area {
        border: round $surface;
        padding: 1;
        height: auto;
    }
    
    .form-row {
        height: auto;
        margin-bottom: 1;
    }

    .review-actions {
        height: auto;
        margin-top: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Compose the Anki/Flashcards UI"""
        with ScrollableContainer(classes="flashcards-container"):
            yield Label("🗂️ Anki/Flashcards", classes="section-title")
            
            # Card creation section
            with Vertical(classes="card-editor"):
                yield Label("Decks:", classes="subsection-title")
                
                with Horizontal(classes="deck-controls"):
                    yield Label("Deck:", classes="form-label")
                    yield Select(
                        options=[("No decks available", Select.BLANK)],
                        allow_blank=True,
                        prompt="Select deck...",
                        id="deck-select"
                    )
                with Horizontal(classes="deck-controls"):
                    yield Input(placeholder="New deck name...", id="new-deck-name-input")
                    yield Button("Create Deck", id="create-deck-button", variant="primary")

                with Vertical(classes="deck-lifecycle-controls"):
                    yield Label("Deck Actions:", classes="subsection-title")
                    delete_note = Static(
                        "In server mode, deck delete is disabled in the flashcards pane.",
                        id="delete-deck-note",
                        classes="deck-warning",
                    )
                    delete_note.display = False
                    yield delete_note
                    with Horizontal(classes="deck-controls"):
                        yield Select(
                            options=[("No target decks available", Select.BLANK)],
                            allow_blank=True,
                            prompt="Select target deck...",
                            id="move-card-target-select",
                        )
                        yield Button("Move Selected Card", id="move-selected-card-button")
                        yield Button("Delete Selected Card", id="delete-selected-card-button", variant="error")
                    yield Button(
                        "Delete Deck",
                        id="delete-deck-button",
                        variant="error",
                        disabled=False,
                    )

                yield Label("Search Cards:", classes="subsection-title")
                with Horizontal(classes="search-row"):
                    yield Input(placeholder="Search selected deck...", id="flashcard-search-input")
                    yield Button("Refresh", id="flashcard-refresh-button")

                yield Label("Create New Card:", classes="subsection-title")
                
                yield Label("Front (Question):")
                yield TextArea(
                    "",
                    id="card-front",
                    classes="card-input"
                )
                
                yield Label("Back (Answer):")
                yield TextArea(
                    "",
                    id="card-back",
                    classes="card-input"
                )
                
                with Horizontal(classes="form-row"):
                    yield Label("Tags:", classes="form-label")
                    yield Input(placeholder="space-separated tags", id="card-tags")
                
                yield Button("Create Card", id="create-card-btn", variant="primary")
            
            # Card list
            yield Label("Your Cards:", classes="subsection-title")
            yield ListView(id="card-list", classes="card-list")
            
            # Review section
            with Vertical(classes="review-area"):
                yield Label("Review Cards:", classes="subsection-title")
                yield Static("Create a deck to begin studying.", id="review-status")
                yield Static("", id="review-front")
                review_back = Static("", id="review-back")
                review_back.display = False
                yield review_back
                yield Static("", id="review-next-intervals")
                yield Button("Show Answer", id="show-answer-button")
                yield Button("Start Review", id="start-review-btn", variant="success")
                with Horizontal(classes="review-actions"):
                    for rating in range(6):
                        yield Button(str(rating), id=f"review-rating-{rating}")


class QuizzesWidget(Widget):
    """Widget for local/server-compatible quiz authoring and attempts."""

    DEFAULT_CSS = """
    QuizzesWidget {
        height: 100%;
        width: 100%;
    }

    .quizzes-container {
        padding: 1;
        height: 100%;
    }

    .quiz-editor {
        border: round $surface;
        padding: 1;
        margin-bottom: 1;
    }

    .quiz-list {
        height: 10;
        border: round $surface;
        margin-bottom: 1;
    }

    .quiz-attempt-area {
        border: round $surface;
        padding: 1;
        height: auto;
    }

    .quiz-actions {
        height: auto;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the quizzes UI."""
        with ScrollableContainer(classes="quizzes-container"):
            yield Label("📝 Quizzes", classes="section-title")

            with Vertical(classes="quiz-editor"):
                yield Label("Quiz Selection:", classes="subsection-title")
                with Horizontal(classes="form-row"):
                    yield Label("Quiz:", classes="form-label")
                    yield Select(
                        options=[("No quizzes available", Select.BLANK)],
                        allow_blank=True,
                        prompt="Select quiz...",
                        id="quiz-select",
                    )

                yield Label("Create New Quiz:", classes="subsection-title")
                with Horizontal(classes="form-row"):
                    yield Input(placeholder="Quiz name...", id="new-quiz-name-input")
                    yield Input(placeholder="Description...", id="new-quiz-description-input")
                with Horizontal(classes="quiz-actions"):
                    yield Button("Create Quiz", id="create-quiz-button", variant="primary")
                    yield Button("Delete Quiz", id="delete-quiz-button", variant="error")

                yield Label("Add Fill Blank Question:", classes="subsection-title")
                yield Label("Question Text:")
                yield TextArea("", id="quiz-question-text", classes="card-input")
                with Horizontal(classes="form-row"):
                    yield Label("Correct Answer:", classes="form-label")
                    yield Input(placeholder="Correct answer...", id="quiz-correct-answer-input")
                with Horizontal(classes="quiz-actions"):
                    yield Button("Add Question", id="create-quiz-question-button", variant="primary")
                    yield Button("Delete Selected Question", id="delete-quiz-question-button", variant="error")

            yield Label("Quiz Questions:", classes="subsection-title")
            yield ListView(id="quiz-question-list", classes="quiz-list")

            with Vertical(classes="quiz-attempt-area"):
                yield Label("Attempt Quiz:", classes="subsection-title")
                yield Static("Create a quiz to begin practicing.", id="quiz-attempt-status")
                yield Static("", id="quiz-attempt-question")
                yield Label("Attempt History:", classes="subsection-title")
                with Horizontal(classes="form-row"):
                    yield Select(
                        options=[("No attempt history", Select.BLANK)],
                        allow_blank=True,
                        prompt="Select attempt...",
                        id="quiz-attempt-history-select",
                    )
                    yield Button("Load Attempt", id="load-quiz-attempt-history-button")
                yield Static("", id="quiz-attempt-history-summary")
                with Horizontal(classes="form-row"):
                    yield Label("Your Answer:", classes="form-label")
                    yield Input(placeholder="Enter your answer...", id="quiz-answer-input")
                with Horizontal(classes="quiz-actions"):
                    yield Button("Start Attempt", id="start-quiz-attempt-button", variant="primary")
                    yield Button("Submit Answer", id="submit-quiz-answer-button", variant="success")


class MindmapsWidget(Widget):
    """Widget for creating and viewing mindmaps"""
    
    DEFAULT_CSS = """
    MindmapsWidget {
        height: 100%;
        width: 100%;
    }
    
    .mindmaps-container {
        padding: 1;
        height: 100%;
    }
    
    .mindmap-tree {
        height: 1fr;
        width: 70%;
        border: round $surface;
        margin-right: 1;
    }
    
    .mindmap-controls {
        width: 30%;
        padding: 1;
    }
    
    .form-row {
        height: 3;
        margin-bottom: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Compose the Mindmaps UI"""
        with ScrollableContainer(classes="mindmaps-container"):
            yield Label("🧠 Mindmaps", classes="section-title")
            
            with Horizontal():
                # Mindmap display
                yield Tree("Root Topic", id="mindmap-tree", classes="mindmap-tree")
                
                # Controls
                with Vertical(classes="mindmap-controls"):
                    yield Label("Add Node:", classes="subsection-title")
                    yield Input(placeholder="Node text...", id="node-text")
                    yield Button("Add Child", id="add-child-btn", variant="primary")
                    yield Button("Add Sibling", id="add-sibling-btn", variant="default")
                    
                    yield Label("Actions:", classes="subsection-title")
                    yield Button("Delete Node", id="delete-node-btn", variant="error")
                    yield Button("Edit Node", id="edit-node-btn", variant="default")
                    
                    yield Label("Import/Export:", classes="subsection-title")
                    yield Button("Import from Notes", id="import-notes-btn")
                    yield Button("Export to Markdown", id="export-md-btn")
                    yield Button("Generate from LLM", id="generate-mindmap-btn", variant="success")


class CourseCreationWidget(Widget):
    """Widget for creating and managing courses"""
    
    DEFAULT_CSS = """
    CourseCreationWidget {
        height: 100%;
        width: 100%;
    }
    
    .course-creation-container {
        padding: 1;
        height: 100%;
    }
    
    .course-form {
        border: round $surface;
        padding: 1;
        margin-bottom: 1;
    }
    
    .module-list {
        height: 15;
        border: round $surface;
        margin-bottom: 1;
    }
    
    .form-row {
        height: 3;
        margin-bottom: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Compose the Course Creation UI"""
        with ScrollableContainer(classes="course-creation-container"):
            yield Label("📖 Course Creation", classes="section-title")
            
            # Course details form
            with Vertical(classes="course-form"):
                yield Label("Course Details:", classes="subsection-title")
                
                yield Label("Course Title:")
                yield Input(placeholder="Enter course title...", id="course-title")
                
                yield Label("Description:")
                yield TextArea(
                    "Enter course description...",
                    id="course-description",
                    classes="course-description"
                )
                
                with Horizontal(classes="form-row"):
                    yield Label("Level:", classes="form-label")
                    yield Select(
                        options=[
                            ("beginner", "Beginner"),
                            ("intermediate", "Intermediate"),
                            ("advanced", "Advanced")
                        ],
                        id="course-level"
                    )
                
                yield Label("Prerequisites:")
                yield Input(placeholder="Enter prerequisites...", id="course-prerequisites")
                
                yield Button("Create Course", id="create-course-btn", variant="primary")
            
            # Module management
            yield Label("Course Modules:", classes="subsection-title")
            yield ListView(id="module-list", classes="module-list")
            
            with Horizontal(classes="form-row"):
                yield Input(placeholder="Module name...", id="module-name")
                yield Button("Add Module", id="add-module-btn")
            
            # Export options
            yield Label("Export Options:", classes="subsection-title")
            with Horizontal(classes="form-row"):
                yield Button("Export to PDF", id="export-pdf-btn")
                yield Button("Export to Markdown", id="export-md-btn")
                yield Button("Export to SCORM", id="export-scorm-btn")


class StudyGuideWidget(Widget):
    """Widget for creating and managing study guides"""
    
    DEFAULT_CSS = """
    StudyGuideWidget {
        height: 100%;
        width: 100%;
    }
    
    .study-guide-container {
        padding: 1;
        height: 100%;
    }
    
    .guide-content {
        border: round $surface;
        padding: 1;
        height: 20;
        margin-bottom: 1;
    }
    
    .key-concepts {
        height: 10;
        border: round $surface;
        margin-bottom: 1;
    }
    
    .practice-questions {
        height: 10;
        border: round $surface;
        margin-bottom: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Compose the Study Guide UI"""
        with ScrollableContainer(classes="study-guide-container"):
            yield Label("📋 Study Guide", classes="section-title")
            
            # Topic selection
            with Horizontal(classes="form-row"):
                yield Label("Topic:", classes="form-label")
                yield Select(
                    options=[("new", "New Topic")],
                    id="guide-topic-select"
                )
            
            yield Label("Guide Title:")
            yield Input(placeholder="Enter guide title...", id="guide-title")
            
            # Guide content
            yield Label("Guide Content:")
            yield TextArea(
                "Enter or generate study guide content...",
                id="guide-content",
                classes="guide-content"
            )
            
            # Key concepts section
            yield Label("Key Concepts:", classes="subsection-title")
            yield ListView(id="key-concepts-list", classes="key-concepts")
            
            with Horizontal(classes="form-row"):
                yield Input(placeholder="Add key concept...", id="concept-input")
                yield Button("Add", id="add-concept-btn")
            
            # Practice questions
            yield Label("Practice Questions:", classes="subsection-title")
            yield ListView(id="practice-questions-list", classes="practice-questions")
            
            # Action buttons
            with Horizontal(classes="form-row"):
                yield Button("Generate from Topic", id="generate-guide-btn", variant="success")
                yield Button("Generate Questions", id="generate-questions-btn")
                yield Button("Save Guide", id="save-guide-btn", variant="primary")


class LearningMapWidget(Widget):
    """Widget for visualizing and managing learning paths"""
    
    DEFAULT_CSS = """
    LearningMapWidget {
        height: 100%;
        width: 100%;
    }
    
    .learning-map-container {
        padding: 1;
        height: 100%;
    }
    
    .map-tree {
        height: 1fr;
        width: 70%;
        border: round $surface;
        margin-right: 1;
    }
    
    .map-controls {
        width: 30%;
        padding: 1;
    }
    
    .progress-display {
        border: round $surface;
        padding: 1;
        margin-bottom: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        """Compose the Learning Map UI"""
        with ScrollableContainer(classes="learning-map-container"):
            yield Label("🗺️ Learning Map", classes="section-title")
            
            with Horizontal():
                # Learning path visualization
                yield Tree("Learning Path", id="learning-map-tree", classes="map-tree")
                
                # Controls and info
                with Vertical(classes="map-controls"):
                    # Progress display
                    with Vertical(classes="progress-display"):
                        yield Label("Overall Progress:", classes="subsection-title")
                        yield Static("0% Complete", id="overall-progress")
                        yield Label("Current Topic:", classes="subsection-title")
                        yield Static("None selected", id="current-topic")
                    
                    yield Label("Path Actions:", classes="subsection-title")
                    yield Button("Add Milestone", id="add-milestone-btn", variant="primary")
                    yield Button("Mark Complete", id="mark-complete-btn", variant="success")
                    yield Button("Set Dependencies", id="set-dependencies-btn")
                    
                    yield Label("Import/Export:", classes="subsection-title")
                    yield Button("Import from Course", id="import-course-btn")
                    yield Button("Export Path", id="export-path-btn")
                    yield Button("Generate Suggestions", id="generate-suggestions-btn", variant="success")


class StudyWindow(Container):
    """Main Study window containing all sub-windows"""
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.flashcards_controller = StudyFlashcardsController(self)
        self.quizzes_controller = StudyQuizzesController(self)
    
    DEFAULT_CSS = """
    StudyWindow {
        layout: horizontal;
        height: 100%;
    }
    
    .study-sidebar {
        width: 30;
        border-right: solid $primary;
        padding: 1;
    }
    
    .study-content {
        width: 1fr;
    }
    
    .section-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .subsection-title {
        text-style: bold italic;
        margin: 1 0;
    }
    
    .sidebar-button {
        width: 100%;
        margin-bottom: 1;
    }
    
    .form-label {
        width: 15;
    }
    
    .card-input {
        height: 5;
        margin-bottom: 1;
    }
    
    .course-description {
        height: 5;
        margin-bottom: 1;
    }
    """
    
    # Reactive property to track current view
    current_view = reactive("structured_learning")
    
    def compose(self) -> ComposeResult:
        """Compose the Study window"""
        # Sidebar
        with Vertical(classes="study-sidebar"):
            yield Label("Study Menu", classes="section-title")
            yield Button("📚 Structured Learning", id="view-structured-btn", classes="sidebar-button", variant="primary")
            yield Button("🗂️ Anki/Flashcards", id="view-flashcards-btn", classes="sidebar-button")
            yield Button("📝 Quizzes", id="view-quizzes-btn", classes="sidebar-button")
            yield Button("🧠 Mindmaps", id="view-mindmaps-btn", classes="sidebar-button")
            yield Button("📖 Course Creation", id="view-course-btn", classes="sidebar-button")
            yield Button("📋 Study Guide", id="view-study-guide-btn", classes="sidebar-button")
            yield Button("🗺️ Learning Map", id="view-learning-map-btn", classes="sidebar-button")
        
        # Content area
        with Container(classes="study-content"):
            # Show structured learning by default
            yield StructuredLearningWidget()
    
    def watch_current_view(self, old_view: str, new_view: str) -> None:
        """Handle view changes"""
        if old_view == "flashcards":
            self.run_worker(self.flashcards_controller.end_review_session_if_needed(), exclusive=True)

        # Remove old content
        content_container = self.query_one(".study-content", Container)
        
        # Clear existing content
        content_container.remove_children()
        
        # Add new content based on view
        if new_view == "structured_learning":
            content_container.mount(StructuredLearningWidget())
        elif new_view == "flashcards":
            content_container.mount(AnkiFlashcardsWidget())
        elif new_view == "quizzes":
            content_container.mount(QuizzesWidget())
        elif new_view == "mindmaps":
            content_container.mount(MindmapsWidget())
        elif new_view == "course_creation":
            content_container.mount(CourseCreationWidget())
        elif new_view == "study_guide":
            content_container.mount(StudyGuideWidget())
        elif new_view == "learning_map":
            content_container.mount(LearningMapWidget())
        
        # Update button states
        self.update_button_states(new_view)

        if new_view == "flashcards":
            self.call_after_refresh(self._schedule_flashcards_refresh)
        elif new_view == "quizzes":
            self.call_after_refresh(self._schedule_quizzes_refresh)
    
    def update_button_states(self, active_view: str) -> None:
        """Update sidebar button variants based on active view"""
        buttons = {
            "structured_learning": "#view-structured-btn",
            "flashcards": "#view-flashcards-btn",
            "quizzes": "#view-quizzes-btn",
            "mindmaps": "#view-mindmaps-btn",
            "course_creation": "#view-course-btn",
            "study_guide": "#view-study-guide-btn",
            "learning_map": "#view-learning-map-btn"
        }
        
        for view, button_id in buttons.items():
            button = self.query_one(button_id, Button)
            button.variant = "primary" if view == active_view else "default"
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle sidebar button presses"""
        button_id = event.button.id
        if button_id is None or not button_id.startswith("view-"):
            return
        
        if button_id == "view-structured-btn":
            self.current_view = "structured_learning"
        elif button_id == "view-flashcards-btn":
            self.current_view = "flashcards"
        elif button_id == "view-quizzes-btn":
            self.current_view = "quizzes"
        elif button_id == "view-mindmaps-btn":
            self.current_view = "mindmaps"
        elif button_id == "view-course-btn":
            self.current_view = "course_creation"
        elif button_id == "view-study-guide-btn":
            self.current_view = "study_guide"
        elif button_id == "view-learning-map-btn":
            self.current_view = "learning_map"
    
    def on_mount(self) -> None:
        """Initialize the window"""
        # Set up initial state
        self.update_button_states("structured_learning")
        
        # Note: Study functionality now uses ChaChaNotes_DB from the app instance
    
    def _is_server_mode(self) -> bool:
        return getattr(self.app_instance, "current_runtime_backend", None) == "server"

    def _configure_flashcards_lifecycle_controls(self) -> None:
        try:
            delete_deck_button = self.query_one("#delete-deck-button", Button)
            delete_deck_note = self.query_one("#delete-deck-note", Static)
        except Exception:
            return

        server_mode = self._is_server_mode()
        delete_deck_button.disabled = server_mode
        delete_deck_note.display = server_mode

    def _schedule_flashcards_refresh(self) -> None:
        self.run_worker(self.flashcards_controller.initialize_view(), exclusive=True)
        self.call_after_refresh(self._configure_flashcards_lifecycle_controls)

    def _schedule_quizzes_refresh(self) -> None:
        self.run_worker(self.quizzes_controller.initialize_view(), exclusive=True)

    @on(Button.Pressed, "#create-deck-button")
    def handle_create_deck(self) -> None:
        self.run_worker(self.flashcards_controller.create_deck(), exclusive=True)

    @on(Button.Pressed, "#flashcard-refresh-button")
    def handle_refresh_cards(self) -> None:
        self.run_worker(self.flashcards_controller.refresh_cards(), exclusive=True)

    @on(Button.Pressed, "#create-card-btn")
    def handle_create_card(self) -> None:
        self.run_worker(self.flashcards_controller.create_card(), exclusive=True)

    @on(Button.Pressed, "#delete-deck-button")
    def handle_delete_deck(self) -> None:
        pass

    @on(Select.Changed, "#move-card-target-select")
    def handle_move_card_target_changed(self, event: Select.Changed) -> None:
        pass

    @on(Button.Pressed, "#move-selected-card-button")
    def handle_move_selected_card(self) -> None:
        pass

    @on(Button.Pressed, "#delete-selected-card-button")
    def handle_delete_selected_card(self) -> None:
        pass

    @on(Button.Pressed, "#start-review-btn")
    def handle_start_review(self) -> None:
        self.run_worker(self.flashcards_controller.start_review(), exclusive=True)

    @on(Button.Pressed, "#show-answer-button")
    def handle_show_answer(self) -> None:
        self.flashcards_controller.show_answer()

    @on(Button.Pressed, "#review-rating-0")
    @on(Button.Pressed, "#review-rating-1")
    @on(Button.Pressed, "#review-rating-2")
    @on(Button.Pressed, "#review-rating-3")
    @on(Button.Pressed, "#review-rating-4")
    @on(Button.Pressed, "#review-rating-5")
    def handle_review_rating(self, event: Button.Pressed) -> None:
        rating = int(str(event.button.id).rsplit("-", 1)[-1])
        self.run_worker(self.flashcards_controller.submit_rating(rating), exclusive=True)

    @on(Select.Changed, "#deck-select")
    def handle_deck_select_changed(self, event: Select.Changed) -> None:
        self.run_worker(self.flashcards_controller.handle_deck_changed())

    @on(Button.Pressed, "#create-quiz-button")
    def handle_create_quiz(self) -> None:
        self.run_worker(self.quizzes_controller.create_quiz(), exclusive=True)

    @on(Button.Pressed, "#delete-quiz-button")
    def handle_delete_quiz(self) -> None:
        self.run_worker(self.quizzes_controller.delete_quiz(), exclusive=True)

    @on(Button.Pressed, "#create-quiz-question-button")
    def handle_create_quiz_question(self) -> None:
        self.run_worker(self.quizzes_controller.create_question(), exclusive=True)

    @on(Button.Pressed, "#delete-quiz-question-button")
    def handle_delete_quiz_question(self) -> None:
        self.run_worker(self.quizzes_controller.delete_question(), exclusive=True)

    @on(Button.Pressed, "#start-quiz-attempt-button")
    def handle_start_quiz_attempt(self) -> None:
        self.run_worker(self.quizzes_controller.start_attempt(), exclusive=True)

    @on(Button.Pressed, "#submit-quiz-answer-button")
    def handle_submit_quiz_answer(self) -> None:
        self.run_worker(self.quizzes_controller.submit_current_answer(), exclusive=True)

    @on(Button.Pressed, "#load-quiz-attempt-history-button")
    def handle_load_quiz_attempt_history(self) -> None:
        self.run_worker(self.quizzes_controller.load_selected_attempt(), exclusive=True)

    @on(Select.Changed, "#quiz-select")
    async def handle_quiz_select_changed(self, event: Select.Changed) -> None:
        await self.quizzes_controller.handle_quiz_changed()
