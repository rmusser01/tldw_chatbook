# Stats_Screen.py
#
# Description: Screen for displaying user statistics.
#
# Imports
import logging
from typing import TYPE_CHECKING, Dict, Any, Optional
from datetime import datetime

from textual.app import ComposeResult
from textual.widgets import Static, Label, LoadingIndicator, ProgressBar, Placeholder, Button
from textual.containers import VerticalScroll, Horizontal, Container, Grid
from textual.reactive import reactive
from textual import work

# Local imports
from tldw_chatbook.Stats.user_statistics import UserStatistics
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli

########################################################################################################################

logger = logging.getLogger(__name__)


class StatCard(Container):
    """A card widget for displaying a statistic."""
    
    def __init__(self, title: str, value: str, subtitle: str = "", icon: str = "", **kwargs):
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
        percentage = (self.count / self.max_count * 100) if self.max_count > 0 else 0
        
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
        except:
            pass


class StatsScreen(Container):
    """
    A screen to display dynamic user statistics.
    """
    
    # Reactive attributes
    stats_data: reactive[Optional[Dict[str, Any]]] = reactive(None)
    is_loading: reactive[bool] = reactive(False)  # Start as False
    error_message: reactive[Optional[str]] = reactive(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.app_instance: Optional['TldwCli'] = None
        
    def on_mount(self) -> None:
        """Load statistics when the screen is mounted."""
        # Get the app instance from the ancestry
        from ..app import TldwCli
        self.app_instance = self.app
        if not isinstance(self.app_instance, TldwCli):
            logger.error(f"App instance is not TldwCli: {type(self.app_instance)}")
            self.error_message = "Unable to access application instance"
            return
        logger.info("StatsScreen mounted, loading statistics...")
        # Set loading state and trigger initial display
        self.is_loading = True
        self.load_statistics()
        
    @work(thread=True)
    def load_statistics(self) -> None:
        """Load user statistics in a background thread."""
        try:
            logger.info("Starting statistics load...")
            self.is_loading = True
            self.error_message = None
            
            # Get database instance
            if not hasattr(self.app_instance, 'characters_rag_db'):
                logger.error("App instance does not have characters_rag_db attribute")
                self.error_message = "Database connection not initialized"
                return
                
            db = self.app_instance.characters_rag_db
            if not db:
                logger.error("Database instance is None")
                self.error_message = "Database not available"
                return
            
            logger.info(f"Database instance obtained: {type(db)}")
                
            # Calculate statistics
            stats_calculator = UserStatistics(db)
            stats = stats_calculator.get_all_stats()
            
            logger.info(f"Statistics calculated: {len(stats)} items")
            
            # Update reactive attribute
            self.stats_data = stats
            
        except Exception as e:
            logger.error(f"Error loading user statistics: {e}", exc_info=True)
            self.error_message = f"Error loading statistics: {str(e)}"
        finally:
            self.is_loading = False
            logger.info("Statistics loading complete")
    
    def compose(self) -> ComposeResult:
        """Create the statistics display."""
        with Container(id="stats-container"):
            yield Label("📊 User Statistics Dashboard", classes="stats-header")
            with VerticalScroll(id="stats-scroll"):
                with Container(id="stats-content"):
                    # Initial content - will be replaced by refresh_stats_display
                    yield Container(
                        LoadingIndicator(),
                        Label("Initializing statistics...", classes="loading-text"),
                        classes="loading-container"
                    )
    
    def watch_is_loading(self, is_loading: bool) -> None:
        """React to loading state changes."""
        self.refresh_stats_display()
    
    def watch_stats_data(self, stats_data: Optional[Dict[str, Any]]) -> None:
        """React to stats data changes."""
        self.refresh_stats_display()
    
    def watch_error_message(self, error_message: Optional[str]) -> None:
        """React to error message changes."""
        self.refresh_stats_display()
    
    def refresh_stats_display(self) -> None:
        """Refresh the statistics display based on current state."""
        try:
            logger.debug(f"Refreshing stats display - loading: {self.is_loading}, error: {self.error_message}, has_data: {self.stats_data is not None}")
            
            content = self.query_one("#stats-content", Container)
            content.remove_children()
            
            if self.is_loading:
                logger.debug("Showing loading state")
                content.mount(
                    Container(
                        LoadingIndicator(),
                        Label("Loading your statistics...", classes="loading-text"),
                        classes="loading-container"
                    )
                )
            elif self.error_message:
                logger.debug(f"Showing error state: {self.error_message}")
                content.mount(
                    Container(
                        Label(f"❌ {self.error_message}", classes="error-message"),
                        Button("Retry", id="retry-stats-button", classes="retry-button"),
                        classes="error-container"
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
                content.mount(Label("No statistics available yet. Start chatting to see your stats!", 
                                  classes="no-data-message"))
                
        except Exception as e:
            logger.error(f"Error refreshing stats display: {e}", exc_info=True)
    
    def _mount_overview_section(self, parent: Container) -> None:
        """Mount the overview statistics section."""
        stats = self.stats_data
        
        parent.mount(Label("📈 Overview", classes="section-header"))
        
        overview_grid = Grid(classes="stats-grid overview-grid")
        
        # Total conversations
        overview_grid.mount(StatCard(
            "Total Conversations",
            str(stats.get('total_conversations', 0)),
            icon="💬"
        ))
        
        # Total messages
        overview_grid.mount(StatCard(
            "Total Messages",
            str(stats.get('total_messages', 0)),
            icon="✉️"
        ))
        
        # Average messages per conversation
        overview_grid.mount(StatCard(
            "Avg Messages/Conversation",
            f"{stats.get('avg_messages_per_conversation', 0):.1f}",
            icon="📊"
        ))
        
        # Data history
        history = stats.get('data_history_length', {})
        overview_grid.mount(StatCard(
            "Data History",
            history.get('formatted', 'No data'),
            subtitle=f"Since {history.get('earliest_date', 'N/A')}",
            icon="📅"
        ))
        
        parent.mount(overview_grid)
    
    def _mount_activity_section(self, parent: Container) -> None:
        """Mount the activity statistics section."""
        stats = self.stats_data
        
        parent.mount(Label("⚡ Activity Patterns", classes="section-header"))
        
        activity_grid = Grid(classes="stats-grid activity-grid")
        
        # Last 30 days activity
        activity_30 = stats.get('activity_last_30_days', {})
        activity_grid.mount(StatCard(
            "Last 30 Days",
            f"{activity_30.get('messages', 0)} messages",
            subtitle=f"{activity_30.get('daily_average', 0):.1f} per day",
            icon="📆"
        ))
        
        # Most active time
        activity_grid.mount(StatCard(
            "Most Active Time",
            stats.get('most_active_time', 'Unknown'),
            icon="🕐"
        ))
        
        # Most active day
        activity_grid.mount(StatCard(
            "Most Active Day",
            stats.get('most_active_day', 'Unknown'),
            icon="📅"
        ))
        
        # Conversation streaks
        streaks = stats.get('conversation_streaks', {})
        activity_grid.mount(StatCard(
            "Current Streak",
            f"{streaks.get('current_streak', 0)} days",
            subtitle=f"Longest: {streaks.get('longest_streak', 0)} days",
            icon="🔥"
        ))
        
        parent.mount(activity_grid)
    
    def _mount_preferences_section(self, parent: Container) -> None:
        """Mount the user preferences section."""
        stats = self.stats_data
        
        parent.mount(Label("👤 User Profile", classes="section-header"))
        
        prefs_grid = Grid(classes="stats-grid prefs-grid")
        
        # Preferred name
        prefs_grid.mount(StatCard(
            "Preferred Name",
            stats.get('preferred_name', 'User'),
            icon="👋"
        ))
        
        # Preferred device
        prefs_grid.mount(StatCard(
            "Preferred Device",
            stats.get('preferred_device', 'Unknown'),
            icon="💻"
        ))
        
        # Average message length
        prefs_grid.mount(StatCard(
            "Avg Message Length",
            f"{stats.get('avg_message_length', 0):.0f} chars",
            icon="📝"
        ))
        
        # Satisfaction rate
        satisfaction = stats.get('satisfaction_rate')
        if satisfaction is not None:
            prefs_grid.mount(StatCard(
                "Satisfaction Rate",
                f"{satisfaction}%",
                subtitle="Based on ratings",
                icon="⭐"
            ))
        
        parent.mount(prefs_grid)
    
    def _mount_topics_section(self, parent: Container) -> None:
        """Mount the topics analysis section."""
        stats = self.stats_data
        
        parent.mount(Label("🏷️ Topic Analysis", classes="section-header"))
        
        # Main topics
        main_topics = stats.get('main_topics', [])
        if main_topics:
            parent.mount(Label("Main Discussion Topics", classes="subsection-header"))
            topics_container = Container(classes="topics-container")
            
            max_count = main_topics[0][1] if main_topics else 1
            for topic, count in main_topics[:5]:
                topics_container.mount(TopicBar(topic.capitalize(), count, max_count))
            
            parent.mount(topics_container)
        
        # Topics by message count ranges
        topics_by_count = stats.get('top_topics_by_message_count', {})
        if topics_by_count:
            parent.mount(Label("Recent Topics Trends", classes="subsection-header"))
            
            ranges_grid = Grid(classes="stats-grid ranges-grid")
            
            for range_name, topics in topics_by_count.items():
                if topics:
                    label = range_name.replace('_', ' ').title()
                    top_words = ", ".join([t[0] for t in topics[:3]])
                    ranges_grid.mount(StatCard(
                        label,
                        top_words if top_words else "No data",
                        classes="topic-range-card"
                    ))
            
            parent.mount(ranges_grid)
    
    def _mount_fun_stats_section(self, parent: Container) -> None:
        """Mount the fun statistics section."""
        stats = self.stats_data
        
        parent.mount(Label("🎉 Fun Facts", classes="section-header"))
        
        fun_grid = Grid(classes="stats-grid fun-grid")
        
        # Emoji usage
        emoji_stats = stats.get('emoji_usage', {})
        top_emojis = emoji_stats.get('top_emojis', [])
        emoji_display = " ".join([e[0] for e in top_emojis[:3]]) if top_emojis else "None"
        fun_grid.mount(StatCard(
            "Emoji Usage",
            f"{emoji_stats.get('usage_rate', 0)}%",
            subtitle=f"Favorites: {emoji_display}",
            icon="😊"
        ))
        
        # Question ratio
        question_stats = stats.get('question_ratio', {})
        fun_grid.mount(StatCard(
            "Curiosity Level",
            question_stats.get('curiosity_level', 'Unknown'),
            subtitle=f"{question_stats.get('question_percentage', 0)}% questions",
            icon="❓"
        ))
        
        # Vocabulary diversity
        vocab_stats = stats.get('vocabulary_diversity', {})
        fun_grid.mount(StatCard(
            "Vocabulary Score",
            f"{vocab_stats.get('score', 0)}/100",
            subtitle=vocab_stats.get('level', 'Unknown'),
            icon="📚"
        ))
        
        # Longest conversation
        longest_conv = stats.get('longest_conversation', {})
        fun_grid.mount(StatCard(
            "Longest Chat",
            f"{longest_conv.get('message_count', 0)} messages",
            subtitle=longest_conv.get('title', 'Unknown'),
            icon="🏆"
        ))
        
        parent.mount(fun_grid)
    
    def _mount_character_stats_section(self, parent: Container) -> None:
        """Mount the character chat statistics section."""
        stats = self.stats_data
        char_stats = stats.get('character_chat_stats', {})
        
        if char_stats.get('total_characters', 0) > 0:
            parent.mount(Label("🎭 Character Chats", classes="section-header"))
            
            char_container = Container(classes="character-stats-container")
            
            # Total characters
            char_container.mount(Label(
                f"Total Characters: {char_stats.get('total_characters', 0)}",
                classes="character-total"
            ))
            
            # Top characters
            top_chars = char_stats.get('top_characters', [])
            if top_chars:
                char_container.mount(Label("Most Chatted With:", classes="subsection-header"))
                
                for char in top_chars[:3]:
                    char_info = Container(
                        Label(f"🎭 {char['name']}", classes="character-name"),
                        Label(f"{char['messages']} messages in {char['conversations']} chats", 
                              classes="character-stats"),
                        classes="character-item"
                    )
                    char_container.mount(char_info)
            
            parent.mount(char_container)
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "retry-stats-button":
            self.load_statistics()

#
#
# End of Metrics_Screen.py
########################################################################################################################
