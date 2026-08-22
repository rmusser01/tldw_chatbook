# Study_Window.py
# Description: Study tab with Structured Learning, Anki/Flashcards, and Mindmaps
#
# Imports
from typing import Any, TYPE_CHECKING
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer, Container
from textual.widgets import (
    Label,
    Button,
    TextArea,
    Select,
    Input,
    Static,
    ListView,
)
from textual.widget import Widget
from textual.reactive import reactive

# Local imports
from tldw_chatbook.UI.Study_Modules import (
    StudyFlashcardsController,
    StudyQuizzesController,
)
from .Screens.study_scope_models import StudyScopeType
# StudyDB import removed - using ChaChaNotes_DB instead

QUIZ_SCOPE_UNAVAILABLE_TOOLTIP = "Workspace Study requires server mode. Switch to server mode or use Global Study to edit quizzes."
QUIZ_ATTEMPT_ACTIVE_TOOLTIP = (
    "Submit the active quiz attempt before editing quizzes or starting another attempt."
)
QUIZ_SUBMIT_INACTIVE_TOOLTIP = "Start a quiz attempt before submitting an answer."

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

    .structured-learning-empty-state {
        border: round $surface;
        padding: 1;
        color: $text-muted;
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

            # task-16195 removed the dead "Add New Topic" row (Input
            # #new-topic-title + Button #add-topic-btn) here; its only
            # handler lived in the legacy Event_Handlers/Study_Events table,
            # which nothing dispatched since the Study rebuild, and no code
            # in the app reads the topics table back.
            #
            # task-16845: the residual chrome left behind -- a #topic-tree
            # that could only ever show its static "Learning Paths" root
            # (nothing populates it, and no Tree.NodeSelected handler exists
            # anywhere since task-16196 deleted the legacy module that held
            # one) next to a disabled #topic-content TextArea whose
            # placeholder promised "Select a topic from the tree to view
            # content..." -- read as broken rather than intentionally empty.
            # ChaChaNotes_DB's create_learning_path/create_topic are
            # write-only (no list/get method exists for learning_paths or
            # topics anywhere), so a real browsing UI would be a feature
            # build, not a wiring fix (same call task-16195 made for the
            # add-topic write path). Replaced with an honest notice instead
            # of building that feature or leaving the misleading chrome.
            yield Static(
                "Structured Learning does not have a browsing UI yet in "
                "this build. There is currently no way to list or view "
                "learning paths and topics from this screen.",
                id="structured-learning-empty-state",
                classes="structured-learning-empty-state",
            )


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
                        id="deck-select",
                    )
                with Horizontal(classes="deck-controls"):
                    yield Input(
                        placeholder="New deck name...", id="new-deck-name-input"
                    )
                    yield Button(
                        "Create Deck", id="create-deck-button", variant="primary"
                    )

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
                        yield Button(
                            "Move Selected Card", id="move-selected-card-button"
                        )
                        yield Button(
                            "Delete Selected Card",
                            id="delete-selected-card-button",
                            variant="error",
                        )
                    yield Button(
                        "Delete Deck",
                        id="delete-deck-button",
                        variant="error",
                        disabled=False,
                    )

                yield Label("Search Cards:", classes="subsection-title")
                with Horizontal(classes="search-row"):
                    yield Input(
                        placeholder="Search selected deck...",
                        id="flashcard-search-input",
                    )
                    yield Button("Refresh", id="flashcard-refresh-button")

                yield Label("Create New Card:", classes="subsection-title")

                yield Label("Front (Question):")
                yield TextArea("", id="card-front", classes="card-input")

                yield Label("Back (Answer):")
                yield TextArea("", id="card-back", classes="card-input")

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
                    yield Input(
                        placeholder="Description...", id="new-quiz-description-input"
                    )
                with Horizontal(classes="quiz-actions"):
                    yield Button(
                        "Create Quiz", id="create-quiz-button", variant="primary"
                    )
                    yield Button(
                        "Delete Quiz", id="delete-quiz-button", variant="error"
                    )

                yield Label("Add Fill Blank Question:", classes="subsection-title")
                yield Label("Question Text:")
                yield TextArea("", id="quiz-question-text", classes="card-input")
                with Horizontal(classes="form-row"):
                    yield Label("Correct Answer:", classes="form-label")
                    yield Input(
                        placeholder="Correct answer...", id="quiz-correct-answer-input"
                    )
                with Horizontal(classes="quiz-actions"):
                    yield Button(
                        "Add Question",
                        id="create-quiz-question-button",
                        variant="primary",
                    )
                    yield Button(
                        "Delete Selected Question",
                        id="delete-quiz-question-button",
                        variant="error",
                    )

            yield Label("Quiz Questions:", classes="subsection-title")
            yield ListView(id="quiz-question-list", classes="quiz-list")

            with Vertical(classes="quiz-attempt-area"):
                yield Label("Attempt Quiz:", classes="subsection-title")
                yield Static(
                    "Create a quiz to begin practicing.", id="quiz-attempt-status"
                )
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
                    yield Input(
                        placeholder="Enter your answer...", id="quiz-answer-input"
                    )
                with Horizontal(classes="quiz-actions"):
                    yield Button(
                        "Start Attempt",
                        id="start-quiz-attempt-button",
                        variant="primary",
                    )
                    yield Button(
                        "Submit Answer",
                        id="submit-quiz-answer-button",
                        variant="success",
                    )


class MindmapsWidget(Widget):
    """Placeholder pane: mindmaps have no working editor in this build."""

    DEFAULT_CSS = """
    MindmapsWidget {
        height: 100%;
        width: 100%;
    }

    .mindmaps-container {
        padding: 1;
        height: 100%;
    }

    .mindmaps-empty-state {
        border: round $surface;
        padding: 1;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the Mindmaps UI"""
        with ScrollableContainer(classes="mindmaps-container"):
            yield Label("🧠 Mindmaps", classes="section-title")

            # task-16845 removed #add-child-btn; task-19041 removed the rest
            # of the pane's chrome for the same reasons: every remaining
            # button (#add-sibling-btn, #delete-node-btn, #edit-node-btn,
            # #import-notes-btn, #export-md-btn, #generate-mindmap-btn) had
            # no dispatcher anywhere (StudyWindow.on_button_pressed
            # early-returned, so each press was a silent no-op),
            # ChaChaNotes_DB's create_mindmap/add_mindmap_node are
            # write-only (no read/list method exists anywhere), and the
            # #mindmap-tree was a static "Root Topic" skeleton with no
            # population code -- nothing added, edited, imported, or
            # generated could ever be displayed or exported. #node-text fed
            # only the dead add buttons and went with them. Replaced with an
            # honest notice. (The orphaned mindmap SUBSYSTEM -- Tools/
            # Mind_Map, MindmapViewer -- is task-19042's scope; this pane
            # never composed it.)
            yield Static(
                "Mindmaps do not have a working editor yet in this build. "
                "There is currently no way to create, edit, or export a "
                "mindmap from this screen.",
                id="mindmaps-empty-state",
                classes="mindmaps-empty-state",
            )


class CourseCreationWidget(Widget):
    """Placeholder pane: course creation is not available in this build."""

    DEFAULT_CSS = """
    CourseCreationWidget {
        height: 100%;
        width: 100%;
    }

    .course-creation-container {
        padding: 1;
        height: 100%;
    }

    .course-creation-empty-state {
        border: round $surface;
        padding: 1;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the Course Creation UI"""
        with ScrollableContainer(classes="course-creation-container"):
            yield Label("📖 Course Creation", classes="section-title")

            # task-16845 removed the "Course Details" form and
            # #create-course-btn (no `course`/`courses` table exists
            # anywhere in ChaChaNotes_DB); task-19041 removed the rest of
            # the pane for the same reasons: #add-module-btn had no
            # dispatcher anywhere and no module concept exists in any
            # schema (#module-list was never populated, #module-name fed
            # only that dead button), and the export row (#export-pdf-btn,
            # #export-md-btn -- which also duplicated the Mindmaps pane's
            # button id -- and #export-scorm-btn) had no exportable course
            # to act on and no SCORM code exists anywhere in the tree.
            # Replaced with an honest notice.
            yield Static(
                "Course creation is not available yet in this build. "
                "There is currently no way to build, import, or export a "
                "course from this screen.",
                id="course-creation-empty-state",
                classes="course-creation-empty-state",
            )


class StudyGuideWidget(Widget):
    """Placeholder pane: study guides are not available in this build."""

    DEFAULT_CSS = """
    StudyGuideWidget {
        height: 100%;
        width: 100%;
    }

    .study-guide-container {
        padding: 1;
        height: 100%;
    }

    .study-guide-empty-state {
        border: round $surface;
        padding: 1;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the Study Guide UI"""
        with ScrollableContainer(classes="study-guide-container"):
            yield Label("📋 Study Guide", classes="section-title")

            # task-16845 removed #generate-guide-btn (no topic to generate
            # from: the topics table is write-only); task-19041 removed the
            # rest of the pane for the same reasons: no guide or concept
            # schema exists anywhere in ChaChaNotes_DB or Study_Interop, so
            # #save-guide-btn had no destination, #add-concept-btn could
            # only feed an in-session #key-concepts-list nothing persists
            # or reads, and #generate-questions-btn had no generation
            # service (#practice-questions-list was never populated). The
            # dead-end form around them (#guide-topic-select -- one static
            # option, its .value consumer-less per the TASK-16841 sweep --
            # #guide-title, #guide-content, #concept-input) existed solely
            # to feed those buttons and went with them. Replaced with an
            # honest notice. (Chat/document_generator.py's study-guide
            # export is a different, conversation-scoped feature -- see
            # task-16845's evidence.)
            yield Static(
                "Study guides are not available yet in this build. "
                "There is currently no way to create, generate, or save a "
                "study guide from this screen.",
                id="study-guide-empty-state",
                classes="study-guide-empty-state",
            )


class LearningMapWidget(Widget):
    """Placeholder pane: the learning map is not available in this build."""

    DEFAULT_CSS = """
    LearningMapWidget {
        height: 100%;
        width: 100%;
    }

    .learning-map-container {
        padding: 1;
        height: 100%;
    }

    .learning-map-empty-state {
        border: round $surface;
        padding: 1;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the Learning Map UI"""
        with ScrollableContainer(classes="learning-map-container"):
            yield Label("🗺️ Learning Map", classes="section-title")

            # task-16845 removed #add-milestone-btn (no milestone concept
            # exists in any schema); task-19041 removed the rest of the
            # pane for the same reasons: every remaining button had no
            # dispatcher anywhere -- #mark-complete-btn's only conceivable
            # sink is the write-only update_topic_progress (nothing reads
            # topic progress back), #set-dependencies-btn and
            # #generate-suggestions-btn have no backing concept or service,
            # #import-course-btn has no course to import (no course table
            # exists), and #export-path-btn has nothing to export
            # (learning_paths is write-only). #learning-map-tree was a
            # static "Learning Path" skeleton nothing populates, and
            # #overall-progress/#current-topic were hard-coded statics
            # nothing ever updated. Replaced with an honest notice.
            yield Static(
                "The learning map is not available yet in this build. "
                "There is currently no way to view a learning path or "
                "track progress from this screen.",
                id="learning-map-empty-state",
                classes="learning-map-empty-state",
            )


class StudyWindow(Container):
    """Main Study window containing all sub-windows"""

    def __init__(self, app_instance: "TldwCli", *, show_sidebar: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.show_sidebar = show_sidebar
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
        height: 100%;
    }

    .study-view-container {
        height: 1fr;
    }

    .study-scope-banner {
        border: round $surface;
        padding: 1;
        margin-bottom: 1;
        width: 100%;
    }

    .study-scope-row {
        height: auto;
        margin-bottom: 1;
    }

    .study-scope-actions {
        height: auto;
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
    """

    # Reactive property to track current view
    current_view = reactive("structured_learning")

    def compose(self) -> ComposeResult:
        """Compose the Study window"""
        if self.show_sidebar:
            with Vertical(classes="study-sidebar"):
                yield Label("Study Menu", classes="section-title")
                yield Button(
                    "📚 Structured Learning",
                    id="view-structured-btn",
                    classes="sidebar-button",
                    variant="primary",
                )
                yield Button(
                    "🗂️ Anki/Flashcards",
                    id="view-flashcards-btn",
                    classes="sidebar-button",
                )
                yield Button(
                    "📝 Quizzes", id="view-quizzes-btn", classes="sidebar-button"
                )
                yield Button(
                    "🧠 Mindmaps", id="view-mindmaps-btn", classes="sidebar-button"
                )
                yield Button(
                    "📖 Course Creation", id="view-course-btn", classes="sidebar-button"
                )
                yield Button(
                    "📋 Study Guide",
                    id="view-study-guide-btn",
                    classes="sidebar-button",
                )
                yield Button(
                    "🗺️ Learning Map",
                    id="view-learning-map-btn",
                    classes="sidebar-button",
                )

        # Content area
        with Vertical(classes="study-content"):
            scope_banner = Container(
                id="study-scope-banner", classes="study-scope-banner"
            )
            scope_banner.display = False
            with scope_banner:
                yield Static(
                    "Workspace Study", id="study-scope-title", classes="section-title"
                )
                yield Static(
                    "", id="study-scope-workspace-name", classes="study-scope-row"
                )
                yield Static(
                    "", id="study-scope-backend-status", classes="study-scope-row"
                )
                with Horizontal(classes="study-scope-actions"):
                    yield Button(
                        "Back to Workspace", id="study-back-to-workspace-button"
                    )
                    yield Button(
                        "Switch To Global Study",
                        id="study-switch-global-button",
                        variant="primary",
                    )

            with Container(id="study-view-container", classes="study-view-container"):
                # Show structured learning by default
                yield StructuredLearningWidget()

    def _current_scope_state(self) -> Any:
        screen = getattr(self, "screen", None)
        return getattr(screen, "current_scope", None)

    @property
    def current_scope_state(self) -> Any:
        return self._current_scope_state()

    def _sync_scope_banner(self) -> None:
        if not self.is_mounted:
            return
        try:
            banner = self.query_one("#study-scope-banner", Container)
            scope_state = self._current_scope_state()
            is_workspace_scope = (
                getattr(scope_state, "scope_type", None) == StudyScopeType.WORKSPACE
            )
            banner.display = is_workspace_scope
            if not is_workspace_scope:
                return

            workspace_name = (
                getattr(scope_state, "workspace_name", None)
                or getattr(scope_state, "workspace_id", None)
                or "Workspace"
            )
            backend = str(getattr(scope_state, "backend", "") or "unknown")
            available = bool(getattr(scope_state, "workspace_scope_available", False))

            self.query_one("#study-scope-title", Static).update("Workspace Study")
            self.query_one("#study-scope-workspace-name", Static).update(
                f"Workspace: {workspace_name}"
            )
            self.query_one("#study-scope-backend-status", Static).update(
                f"Backend availability: {'available' if available else 'unavailable'} on {backend}"
            )
        except Exception:
            return

    def _hide_scope_banner(self) -> None:
        if not self.is_mounted:
            return
        try:
            banner = self.query_one("#study-scope-banner", Container)
            banner.display = False
        except Exception:
            return

    def _notify_shell_state_changed(self) -> None:
        screen = getattr(self, "screen", None)
        notifier = getattr(screen, "sync_shell_from_window", None)
        if callable(notifier):
            notifier()

    def watch_current_view(self, old_view: str, new_view: str) -> None:
        """Handle view changes"""
        if old_view == "flashcards":
            self.run_worker(
                self.flashcards_controller.end_review_session_if_needed(),
                exclusive=True,
                group="study-end-review-session",
            )

        # Remove old content
        content_container = self.query_one("#study-view-container", Container)

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
        self._sync_scope_banner()

        if new_view == "flashcards":
            self.call_after_refresh(self._schedule_flashcards_refresh)
        elif new_view == "quizzes":
            self.call_after_refresh(self._schedule_quizzes_refresh)
        else:
            self.call_after_refresh(self._notify_shell_state_changed)

    def update_button_states(self, active_view: str) -> None:
        """Update sidebar button variants based on active view"""
        if not self.show_sidebar:
            return
        buttons = {
            "structured_learning": "#view-structured-btn",
            "flashcards": "#view-flashcards-btn",
            "quizzes": "#view-quizzes-btn",
            "mindmaps": "#view-mindmaps-btn",
            "course_creation": "#view-course-btn",
            "study_guide": "#view-study-guide-btn",
            "learning_map": "#view-learning-map-btn",
        }

        for view, button_id in buttons.items():
            button = self.query_one(button_id, Button)
            button.variant = "primary" if view == active_view else "default"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle sidebar button presses"""
        button_id = event.button.id
        if button_id == "study-back-to-workspace-button":
            screen = getattr(self, "screen", None)
            return_to_workspace = getattr(screen, "return_to_workspace", None)
            if callable(return_to_workspace):
                return_to_workspace()
            return
        if button_id == "study-switch-global-button":
            self._hide_scope_banner()
            screen = getattr(self, "screen", None)
            switch_to_global_scope = getattr(screen, "switch_to_global_scope", None)
            if callable(switch_to_global_scope):
                switch_to_global_scope()
            return
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
        if self.show_sidebar:
            self.update_button_states("structured_learning")
        self._sync_scope_banner()
        self._notify_shell_state_changed()

        # Note: Study functionality now uses ChaChaNotes_DB from the app instance

    def on_show(self) -> None:
        self._sync_scope_banner()
        self._notify_shell_state_changed()

    def _is_server_mode(self) -> bool:
        candidates = (
            getattr(self, "runtime_backend", None),
            getattr(self.app_instance, "runtime_backend", None),
            getattr(self.app_instance, "current_runtime_backend", None),
        )
        for candidate in candidates:
            normalized = str(candidate or "").strip().lower()
            if normalized in {"local", "server"}:
                return normalized == "server"
        return False

    def _configure_flashcards_lifecycle_controls(self) -> None:
        try:
            self.query_one("#delete-deck-button", Button)
            delete_deck_note = self.query_one("#delete-deck-note", Static)
        except Exception:
            return

        server_mode = self._is_server_mode()
        controller = getattr(self, "flashcards_controller", None)
        scope_checker = getattr(controller, "_scope_is_available", None)
        if callable(scope_checker):
            bool(scope_checker())

        delete_deck_note.display = server_mode
        delete_deck_note.display = server_mode

    def _schedule_flashcards_refresh(self) -> None:
        self.run_worker(
            self.flashcards_controller.initialize_view(),
            exclusive=True,
            group="study-flashcards-initialize-view",
        )
        self.call_after_refresh(self._configure_flashcards_lifecycle_controls)
        self.call_after_refresh(self._notify_shell_state_changed)

    def _schedule_quizzes_refresh(self) -> None:
        self.run_worker(
            self.quizzes_controller.initialize_view(),
            exclusive=True,
            group="study-quizzes-initialize-view",
        )
        self.call_after_refresh(self._configure_quizzes_lifecycle_controls)
        self.call_after_refresh(self._notify_shell_state_changed)

    def _configure_quizzes_lifecycle_controls(self) -> None:
        try:
            quiz_select = self.query_one("#quiz-select", Select)
            quiz_question_list = self.query_one("#quiz-question-list", ListView)
            quiz_name_input = self.query_one("#new-quiz-name-input", Input)
            quiz_description_input = self.query_one(
                "#new-quiz-description-input", Input
            )
            quiz_question_text = self.query_one("#quiz-question-text", TextArea)
            quiz_correct_answer_input = self.query_one(
                "#quiz-correct-answer-input", Input
            )
            quiz_answer_input = self.query_one("#quiz-answer-input", Input)
            create_quiz_button = self.query_one("#create-quiz-button", Button)
            delete_quiz_button = self.query_one("#delete-quiz-button", Button)
            create_question_button = self.query_one(
                "#create-quiz-question-button", Button
            )
            delete_question_button = self.query_one(
                "#delete-quiz-question-button", Button
            )
            start_attempt_button = self.query_one("#start-quiz-attempt-button", Button)
            submit_answer_button = self.query_one("#submit-quiz-answer-button", Button)
            load_attempt_button = self.query_one(
                "#load-quiz-attempt-history-button", Button
            )
            history_select = self.query_one("#quiz-attempt-history-select", Select)
        except Exception:
            return

        scope_enabled = True
        controller = getattr(self, "quizzes_controller", None)
        scope_checker = getattr(controller, "_scope_is_available", None)
        if callable(scope_checker):
            scope_enabled = bool(scope_checker())
        attempt_active = bool(getattr(controller, "current_attempt_id", None)) and bool(
            getattr(controller, "current_attempt_questions", None)
        )

        quiz_select.disabled = not scope_enabled or attempt_active
        quiz_question_list.disabled = not scope_enabled or attempt_active
        quiz_name_input.disabled = not scope_enabled or attempt_active
        quiz_description_input.disabled = not scope_enabled or attempt_active
        quiz_question_text.disabled = not scope_enabled or attempt_active
        quiz_correct_answer_input.disabled = not scope_enabled or attempt_active
        quiz_answer_input.disabled = not scope_enabled or not attempt_active
        create_quiz_button.disabled = not scope_enabled or attempt_active
        delete_quiz_button.disabled = not scope_enabled or attempt_active
        create_question_button.disabled = not scope_enabled or attempt_active
        delete_question_button.disabled = not scope_enabled or attempt_active
        start_attempt_button.disabled = not scope_enabled or attempt_active
        submit_answer_button.disabled = not scope_enabled or not attempt_active
        load_attempt_button.disabled = not scope_enabled or attempt_active
        history_select.disabled = not scope_enabled or attempt_active

        if not scope_enabled:
            for button in (
                create_quiz_button,
                delete_quiz_button,
                create_question_button,
                delete_question_button,
                start_attempt_button,
                submit_answer_button,
                load_attempt_button,
            ):
                button.tooltip = QUIZ_SCOPE_UNAVAILABLE_TOOLTIP
            return

        mutation_buttons = (
            create_quiz_button,
            delete_quiz_button,
            create_question_button,
            delete_question_button,
            start_attempt_button,
            load_attempt_button,
        )
        for button in mutation_buttons:
            button.tooltip = QUIZ_ATTEMPT_ACTIVE_TOOLTIP if attempt_active else None
        submit_answer_button.tooltip = (
            None if attempt_active else QUIZ_SUBMIT_INACTIVE_TOOLTIP
        )

    @on(Button.Pressed, "#create-deck-button")
    def handle_create_deck(self) -> None:
        self.run_worker(
            self.flashcards_controller.create_deck(),
            exclusive=True,
            group="study-create-deck",
        )

    @on(Button.Pressed, "#flashcard-refresh-button")
    def handle_refresh_cards(self) -> None:
        self.run_worker(
            self.flashcards_controller.refresh_cards(),
            exclusive=True,
            group="study-refresh-cards",
        )

    @on(Button.Pressed, "#create-card-btn")
    def handle_create_card(self) -> None:
        self.run_worker(
            self.flashcards_controller.create_card(),
            exclusive=True,
            group="study-create-card",
        )

    @on(Button.Pressed, "#delete-deck-button")
    async def handle_delete_deck(self) -> None:
        await self.flashcards_controller.delete_selected_deck()

    @on(Select.Changed, "#move-card-target-select")
    def handle_move_card_target_changed(self, event: Select.Changed) -> None:
        self.flashcards_controller.handle_move_target_changed()

    @on(Button.Pressed, "#move-selected-card-button")
    async def handle_move_selected_card(self) -> None:
        await self.flashcards_controller.move_selected_card()

    @on(Button.Pressed, "#delete-selected-card-button")
    async def handle_delete_selected_card(self) -> None:
        await self.flashcards_controller.delete_selected_card()

    @on(ListView.Selected, "#card-list")
    async def handle_card_selected(self, event: ListView.Selected) -> None:
        await self.flashcards_controller.handle_card_selected(event)

    @on(Button.Pressed, "#start-review-btn")
    def handle_start_review(self) -> None:
        self.run_worker(
            self.flashcards_controller.start_review(),
            exclusive=True,
            group="study-start-review",
        )

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
        # TASK-19559: a spaced-repetition rating is a DURABLE WRITE, so it is
        # deliberately NOT `exclusive=True`. Exclusivity cancels the previous
        # worker in the same group, and `submit_rating` awaits the review save
        # -- a second fast press would kill the first press's save before it
        # reached the database, and `CancelledError` is a `BaseException` that
        # the handler's `except Exception:` cannot even observe. Ratings are
        # instead serialised inside `StudyFlashcardsController.submit_rating`
        # (an `asyncio.Lock`), so presses queue and every one persists.
        # The explicit group also keeps these workers out of `"default"`, where
        # any other Study worker would cancel them.
        self.run_worker(
            self.flashcards_controller.submit_rating(rating),
            group="study-flashcard-rating",
        )

    @on(Select.Changed, "#deck-select")
    def handle_deck_select_changed(self, event: Select.Changed) -> None:
        self.run_worker(self.flashcards_controller.handle_deck_changed())

    @on(Button.Pressed, "#create-quiz-button")
    def handle_create_quiz(self) -> None:
        self.run_worker(
            self.quizzes_controller.create_quiz(),
            exclusive=True,
            group="study-create-quiz",
        )

    @on(Button.Pressed, "#delete-quiz-button")
    def handle_delete_quiz(self) -> None:
        self.run_worker(
            self.quizzes_controller.delete_quiz(),
            exclusive=True,
            group="study-delete-quiz",
        )

    @on(Button.Pressed, "#create-quiz-question-button")
    def handle_create_quiz_question(self) -> None:
        self.run_worker(
            self.quizzes_controller.create_question(),
            exclusive=True,
            group="study-create-quiz-question",
        )

    @on(Button.Pressed, "#delete-quiz-question-button")
    def handle_delete_quiz_question(self) -> None:
        self.run_worker(
            self.quizzes_controller.delete_question(),
            exclusive=True,
            group="study-delete-quiz-question",
        )

    @on(Button.Pressed, "#start-quiz-attempt-button")
    def handle_start_quiz_attempt(self) -> None:
        self.run_worker(
            self.quizzes_controller.start_attempt(),
            exclusive=True,
            group="study-start-quiz-attempt",
        )

    @on(Button.Pressed, "#submit-quiz-answer-button")
    def handle_submit_quiz_answer(self) -> None:
        self.run_worker(
            self.quizzes_controller.submit_current_answer(),
            exclusive=True,
            group="study-submit-quiz-answer",
        )

    @on(Button.Pressed, "#load-quiz-attempt-history-button")
    def handle_load_quiz_attempt_history(self) -> None:
        self.run_worker(
            self.quizzes_controller.load_selected_attempt(),
            exclusive=True,
            group="study-load-quiz-attempt-history",
        )

    @on(Select.Changed, "#quiz-select")
    async def handle_quiz_select_changed(self, event: Select.Changed) -> None:
        await self.quizzes_controller.handle_quiz_changed()
