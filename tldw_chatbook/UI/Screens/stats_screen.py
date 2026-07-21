# Stats_Screen.py
#
# Description: Screen for displaying user statistics.
#
# Imports
import logging
from typing import TYPE_CHECKING, Dict, Any, Optional

from textual.app import ComposeResult
from textual.widgets import Label, LoadingIndicator, Button
from textual.containers import VerticalScroll, Horizontal, Container, Grid
from textual.reactive import reactive
from textual import work

# Local imports
from tldw_chatbook.Stats.user_statistics import UserStatistics
from ..Navigation.base_app_screen import BaseAppScreen
from ..Workbench.workbench_state import WorkbenchHeaderState, WorkbenchStatus
from ..Workbench.workbench_widgets import DestinationHeader

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli

########################################################################################################################

logger = logging.getLogger(__name__)


class StatCard(Container):
    """A card widget for displaying a statistic."""

    def __init__(
        self, title: str, value: str, subtitle: str = "", icon: str = "", **kwargs
    ):
        super().__init__(**kwargs)
        self.title = title
        self.value = value
        self.subtitle = subtitle
        self.icon = icon

    def compose(self) -> ComposeResult:
        """Compose the stat card."""
        with Container(classes="stat-card-inner"):
            if self.icon:
                yield Label(self.icon, classes="stat-card-icon")
            yield Label(self.title, classes="stat-card-title")
            yield Label(self.value, classes="stat-card-value")
            if self.subtitle:
                yield Label(self.subtitle, classes="stat-card-subtitle")


class TopicBar(Container):
    """A horizontal bar representing a topic's frequency."""

    def __init__(self, topic: str, count: int, max_count: int, **kwargs):
        super().__init__(**kwargs)
        self.topic = topic
        self.count = count
        self.max_count = max_count

    def compose(self) -> ComposeResult:
        """Compose the topic bar."""
        (self.count / self.max_count * 100) if self.max_count > 0 else 0

        with Horizontal(classes="topic-bar-container"):
            yield Label(f"{self.topic} ({self.count})", classes="topic-label")
            with Container(classes="topic-bar-bg"):
                yield Container(classes="topic-bar-fill", id=f"bar-{self.topic}")

    def on_mount(self) -> None:
        """Set the width of the bar after mounting."""
        percentage = (self.count / self.max_count * 100) if self.max_count > 0 else 0
        try:
            bar = self.query_one(f"#bar-{self.topic}")
            bar.styles.width = f"{percentage}%"
        except Exception:
            pass


class StatsScreen(BaseAppScreen):
    """
    A screen to display dynamic user statistics.
    """

    # Reactive attributes
    stats_data: reactive[Optional[Dict[str, Any]]] = reactive(None)
    is_loading: reactive[bool] = reactive(False)  # Start as False
    error_message: reactive[Optional[str]] = reactive(None)

    def __init__(self, app_instance: "TldwCli", **kwargs):
        super().__init__(app_instance, "stats", **kwargs)

    def on_mount(self) -> None:
        """Load statistics when the screen is mounted."""
        super().on_mount()
        # Verify we have the app instance
        if not self.app_instance:
            # Try to get from ancestry as fallback
            from ..app import TldwCli

            self.app_instance = self.app
            if not isinstance(self.app_instance, TldwCli):
                logger.error(f"App instance is not TldwCli: {type(self.app_instance)}")
                self.error_message = "Unable to access application instance"
                return
        logger.info("StatsScreen mounted, loading statistics...")
        self._start_statistics_load()

    def _start_statistics_load(self) -> None:
        """Begin loading statistics and render the loading state on the UI thread."""
        self.stats_data = None
        self.error_message = None
        self.is_loading = True
        self._sync_destination_header("loading")
        if self.is_mounted:
            self.call_after_refresh(self.refresh_stats_display)
        self.load_statistics()

    def _apply_statistics_result(
        self,
        stats_data: Optional[Dict[str, Any]],
        error_message: Optional[str],
    ) -> None:
        """Apply worker results on the main thread."""
        self.stats_data = stats_data
        self.error_message = error_message
        self.is_loading = False
        if error_message:
            self._sync_destination_header("error")
        elif stats_data:
            self._sync_destination_header("ready")
        else:
            self._sync_destination_header("empty")

    def _sync_destination_header(self, status: WorkbenchStatus) -> None:
        """Refresh the destination header status badge; tolerate teardown races."""
        try:
            header = self.query_one("#stats-destination-header", DestinationHeader)
        except Exception:
            return
        header.sync_state(
            WorkbenchHeaderState(
                title="Stats",
                subtitle="Usage statistics from your local data.",
                status=status,
            )
        )

    @staticmethod
    def _resolve_stats_db(app_instance: Any) -> Any:
        """Return the ChaChaNotes DB handle via the canonical access path.

        Prefer ``app_instance.chachanotes_db``, fall back to
        ``notes_service.db`` — the same resolution the Library screen uses.
        """
        notes_service = getattr(app_instance, "notes_service", None)
        return getattr(app_instance, "chachanotes_db", None) or getattr(
            notes_service, "db", None
        )

    @work(thread=True)
    def load_statistics(self) -> None:
        """Load user statistics in a background thread."""
        stats_data: Optional[Dict[str, Any]] = None
        error_message: Optional[str] = None

        try:
            logger.info("Starting statistics load...")

            db = self._resolve_stats_db(self.app_instance)
            if not db:
                notes_service = getattr(self.app_instance, "notes_service", None)
                logger.error(
                    "Stats load aborted: no ChaChaNotes DB handle "
                    "(chachanotes_db is "
                    f"{'set' if getattr(self.app_instance, 'chachanotes_db', None) else 'unset'}, "
                    "notes_service.db is "
                    f"{'set' if getattr(notes_service, 'db', None) else 'unset'})"
                )
                error_message = "Database not available"
                return

            logger.info(f"Database instance obtained: {type(db)}")

            # Calculate statistics
            stats_calculator = UserStatistics(db)
            stats = stats_calculator.get_all_stats()

            logger.info(f"Statistics calculated: {len(stats)} items")

            stats_data = stats

        except Exception as e:
            logger.error(f"Error loading user statistics: {e}", exc_info=True)
            error_message = f"Error loading statistics: {str(e)}"
        finally:
            self.app.call_from_thread(
                self._apply_statistics_result, stats_data, error_message
            )
            logger.info("Statistics loading complete")

    def compose_content(self) -> ComposeResult:
        """Create the statistics display with its destination header."""
        yield DestinationHeader(
            WorkbenchHeaderState(
                title="Stats",
                subtitle="Usage statistics from your local data.",
                status="ready",
            ),
            id="stats-destination-header",
        )
        stats_container = Container(
            VerticalScroll(
                Container(
                    # Initial content - will be replaced by refresh_stats_display
                    Container(
                        LoadingIndicator(),
                        Label("Initializing statistics...", classes="loading-text"),
                        classes="loading-container",
                    ),
                    id="stats-content",
                ),
                id="stats-scroll",
            ),
            id="stats-container",
        )
        # Leave room for the destination header above the stats content.
        stats_container.styles.height = "1fr"
        yield stats_container

    def watch_is_loading(self, is_loading: bool) -> None:
        """React to loading state changes."""
        del is_loading
        if self.is_mounted:
            self.call_after_refresh(self.refresh_stats_display)

    def watch_stats_data(self, stats_data: Optional[Dict[str, Any]]) -> None:
        """React to stats data changes."""
        del stats_data
        if self.is_mounted:
            self.call_after_refresh(self.refresh_stats_display)

    def watch_error_message(self, error_message: Optional[str]) -> None:
        """React to error message changes."""
        del error_message
        if self.is_mounted:
            self.call_after_refresh(self.refresh_stats_display)

    def refresh_stats_display(self) -> None:
        """Refresh the statistics display based on current state."""
        try:
            logger.debug(
                f"Refreshing stats display - loading: {self.is_loading}, error: {self.error_message}, has_data: {self.stats_data is not None}"
            )

            content = self.query_one("#stats-content", Container)
            content.remove_children()

            if self.is_loading:
                logger.debug("Showing loading state")
                content.mount(
                    Container(
                        LoadingIndicator(),
                        Label("Loading your statistics...", classes="loading-text"),
                        classes="loading-container",
                    )
                )
            elif self.error_message:
                logger.debug(f"Showing error state: {self.error_message}")
                content.mount(
                    Container(
                        Label(f"❌ {self.error_message}", classes="error-message"),
                        Button(
                            "Retry", id="retry-stats-button", classes="retry-button"
                        ),
                        classes="error-container",
                    )
                )
            elif self.stats_data:
                logger.debug(f"Showing stats data with {len(self.stats_data)} items")
                # Display all statistics
                self._mount_overview_section(content)
                self._mount_activity_section(content)
                self._mount_preferences_section(content)
                self._mount_topics_section(content)
                self._mount_fun_stats_section(content)
                self._mount_character_stats_section(content)
            else:
                logger.debug("No data available, showing empty state")
                content.mount(
                    Label(
                        "No statistics available yet. Start chatting to see your stats!",
                        classes="no-data-message",
                    )
                )

        except Exception as e:
            logger.error(f"Error refreshing stats display: {e}", exc_info=True)

    def _mount_overview_section(self, parent: Container) -> None:
        """Mount the overview statistics section."""
        stats = self.stats_data

        parent.mount(Label("📈 Overview", classes="section-header"))

        # Children must ride the constructor: mounting into a detached
        # container raises MountError on current Textual.
        history = stats.get("data_history_length", {})
        overview_grid = Grid(
            StatCard(
                "Total Conversations",
                str(stats.get("total_conversations", 0)),
                icon="💬",
            ),
            StatCard("Total Messages", str(stats.get("total_messages", 0)), icon="✉️"),
            StatCard(
                "Avg Messages/Conversation",
                f"{stats.get('avg_messages_per_conversation', 0):.1f}",
                icon="📊",
            ),
            StatCard(
                "Data History",
                history.get("formatted", "No data"),
                subtitle=f"Since {history.get('earliest_date', 'N/A')}",
                icon="📅",
            ),
            classes="stats-grid overview-grid",
        )

        parent.mount(overview_grid)

    def _mount_activity_section(self, parent: Container) -> None:
        """Mount the activity statistics section."""
        stats = self.stats_data

        parent.mount(Label("⚡ Activity Patterns", classes="section-header"))

        activity_30 = stats.get("activity_last_30_days", {})
        streaks = stats.get("conversation_streaks", {})
        activity_grid = Grid(
            StatCard(
                "Last 30 Days",
                f"{activity_30.get('messages', 0)} messages",
                subtitle=f"{activity_30.get('daily_average', 0):.1f} per day",
                icon="📆",
            ),
            StatCard(
                "Most Active Time", stats.get("most_active_time", "Unknown"), icon="🕐"
            ),
            StatCard(
                "Most Active Day", stats.get("most_active_day", "Unknown"), icon="📅"
            ),
            StatCard(
                "Current Streak",
                f"{streaks.get('current_streak', 0)} days",
                subtitle=f"Longest: {streaks.get('longest_streak', 0)} days",
                icon="🔥",
            ),
            classes="stats-grid activity-grid",
        )

        parent.mount(activity_grid)

    def _mount_preferences_section(self, parent: Container) -> None:
        """Mount the user preferences section."""
        stats = self.stats_data

        parent.mount(Label("👤 User Profile", classes="section-header"))

        cards = [
            StatCard("Preferred Name", stats.get("preferred_name", "User"), icon="👋"),
            StatCard(
                "Preferred Device", stats.get("preferred_device", "Unknown"), icon="💻"
            ),
            StatCard(
                "Avg Message Length",
                f"{stats.get('avg_message_length', 0):.0f} chars",
                icon="📝",
            ),
        ]

        satisfaction = stats.get("satisfaction_rate")
        if satisfaction is not None:
            cards.append(
                StatCard(
                    "Satisfaction Rate",
                    f"{satisfaction}%",
                    subtitle="Based on ratings",
                    icon="⭐",
                )
            )

        parent.mount(Grid(*cards, classes="stats-grid prefs-grid"))

    def _mount_topics_section(self, parent: Container) -> None:
        """Mount the topics analysis section."""
        stats = self.stats_data

        parent.mount(Label("🏷️ Topic Analysis", classes="section-header"))

        # Main topics
        main_topics = stats.get("main_topics", [])
        if main_topics:
            parent.mount(Label("Main Discussion Topics", classes="subsection-header"))

            max_count = main_topics[0][1] if main_topics else 1
            parent.mount(
                Container(
                    *(
                        TopicBar(topic.capitalize(), count, max_count)
                        for topic, count in main_topics[:5]
                    ),
                    classes="topics-container",
                )
            )

        # Topics by message count ranges
        topics_by_count = stats.get("top_topics_by_message_count", {})
        if topics_by_count:
            parent.mount(Label("Recent Topics Trends", classes="subsection-header"))

            range_cards = [
                StatCard(
                    range_name.replace("_", " ").title(),
                    ", ".join([t[0] for t in topics[:3]]) or "No data",
                    classes="topic-range-card",
                )
                for range_name, topics in topics_by_count.items()
                if topics
            ]
            parent.mount(Grid(*range_cards, classes="stats-grid ranges-grid"))

    def _mount_fun_stats_section(self, parent: Container) -> None:
        """Mount the fun statistics section."""
        stats = self.stats_data

        parent.mount(Label("🎉 Fun Facts", classes="section-header"))

        emoji_stats = stats.get("emoji_usage", {})
        top_emojis = emoji_stats.get("top_emojis", [])
        emoji_display = (
            " ".join([e[0] for e in top_emojis[:3]]) if top_emojis else "None"
        )
        question_stats = stats.get("question_ratio", {})
        vocab_stats = stats.get("vocabulary_diversity", {})
        longest_conv = stats.get("longest_conversation", {})

        fun_grid = Grid(
            StatCard(
                "Emoji Usage",
                f"{emoji_stats.get('usage_rate', 0)}%",
                subtitle=f"Favorites: {emoji_display}",
                icon="😊",
            ),
            StatCard(
                "Curiosity Level",
                question_stats.get("curiosity_level", "Unknown"),
                subtitle=f"{question_stats.get('question_percentage', 0)}% questions",
                icon="❓",
            ),
            StatCard(
                "Vocabulary Score",
                f"{vocab_stats.get('score', 0)}/100",
                subtitle=vocab_stats.get("level", "Unknown"),
                icon="📚",
            ),
            StatCard(
                "Longest Chat",
                f"{longest_conv.get('message_count', 0)} messages",
                subtitle=longest_conv.get("title", "Unknown"),
                icon="🏆",
            ),
            classes="stats-grid fun-grid",
        )

        parent.mount(fun_grid)

    def _mount_character_stats_section(self, parent: Container) -> None:
        """Mount the character chat statistics section."""
        stats = self.stats_data
        char_stats = stats.get("character_chat_stats", {})

        if char_stats.get("total_characters", 0) > 0:
            parent.mount(Label("🎭 Character Chats", classes="section-header"))

            children = [
                Label(
                    f"Total Characters: {char_stats.get('total_characters', 0)}",
                    classes="character-total",
                )
            ]

            top_chars = char_stats.get("top_characters", [])
            if top_chars:
                children.append(
                    Label("Most Chatted With:", classes="subsection-header")
                )
                children.extend(
                    Container(
                        Label(f"🎭 {char['name']}", classes="character-name"),
                        Label(
                            f"{char['messages']} messages in {char['conversations']} chats",
                            classes="character-stats",
                        ),
                        classes="character-item",
                    )
                    for char in top_chars[:3]
                )

            parent.mount(
                Container(*children, classes="character-stats-container")
            )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "retry-stats-button":
            self._start_statistics_load()


#
#
# End of Metrics_Screen.py
########################################################################################################################
