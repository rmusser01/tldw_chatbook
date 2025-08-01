# dictation_performance_widget.py
"""
Widget for displaying dictation performance metrics and analytics.
"""

from typing import Dict, Any
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, Container, Grid
from textual.widgets import Label, Static, ProgressBar, Button, Rule
from textual.widget import Widget
from textual.reactive import reactive
from textual import work
from loguru import logger

from ..Audio.dictation_metrics import get_performance_monitor


class DictationPerformanceWidget(Widget):
    """
    Dashboard widget showing dictation performance metrics.
    """
    
    DEFAULT_CSS = """
    DictationPerformanceWidget {
        height: 100%;
        padding: 1;
    }
    
    .metrics-grid {
        height: auto;
        grid-size: 3 2;
        grid-gutter: 1;
        margin-bottom: 2;
    }
    
    .metric-card {
        height: 6;
        padding: 1;
        border: round $surface;
        background: $boost;
    }
    
    .metric-value {
        text-style: bold;
        text-align: center;
        color: $primary;
    }
    
    .metric-label {
        text-align: center;
        color: $text-muted;
    }
    
    .provider-comparison {
        height: 15;
        border: round $surface;
        padding: 1;
        margin-bottom: 2;
    }
    
    .comparison-row {
        layout: horizontal;
        height: 3;
        margin-bottom: 1;
    }
    
    .provider-name {
        width: 20;
    }
    
    .provider-bar {
        width: 1fr;
        margin: 0 1;
    }
    
    .recent-sessions {
        height: 20;
        border: round $surface;
        padding: 1;
    }
    
    .session-item {
        margin-bottom: 0.5;
        color: $text-muted;
    }
    
    .refresh-button {
        dock: top;
        align: right top;
        width: auto;
    }
    
    .performance-good {
        color: $success;
    }
    
    .performance-warning {
        color: $warning;
    }
    
    .performance-poor {
        color: $error;
    }
    """
    
    # Reactive data
    metrics_data = reactive({})
    
    def compose(self) -> ComposeResult:
        """Compose performance dashboard."""
        with Container():
            # Header with refresh
            with Horizontal():
                yield Label("📊 Dictation Performance", classes="section-title")
                yield Button("🔄 Refresh", id="refresh-btn", classes="refresh-button")
            
            # Summary metrics grid
            with Grid(classes="metrics-grid"):
                # Total Sessions
                with Container(classes="metric-card"):
                    yield Static("0", id="total-sessions", classes="metric-value")
                    yield Label("Total Sessions", classes="metric-label")
                
                # Total Words
                with Container(classes="metric-card"):
                    yield Static("0", id="total-words", classes="metric-value")
                    yield Label("Words Transcribed", classes="metric-label")
                
                # Average WPM
                with Container(classes="metric-card"):
                    yield Static("0", id="avg-wpm", classes="metric-value")
                    yield Label("Avg Words/Min", classes="metric-label")
                
                # Average Latency
                with Container(classes="metric-card"):
                    yield Static("0ms", id="avg-latency", classes="metric-value")
                    yield Label("Avg Latency", classes="metric-label")
                
                # Efficiency
                with Container(classes="metric-card"):
                    yield Static("0%", id="efficiency", classes="metric-value")
                    yield Label("Efficiency", classes="metric-label")
                
                # Active Time
                with Container(classes="metric-card"):
                    yield Static("0h 0m", id="active-time", classes="metric-value")
                    yield Label("Active Time", classes="metric-label")
            
            # Provider comparison
            with Container(classes="provider-comparison"):
                yield Label("Provider Performance", classes="section-title")
                yield Container(id="provider-stats")
            
            # Recent sessions
            with Container(classes="recent-sessions"):
                yield Label("Recent Sessions", classes="section-title")
                yield Container(id="recent-list")
    
    def on_mount(self):
        """Load metrics on mount."""
        self.refresh_metrics()
    
    @work(exclusive=True)
    async def refresh_metrics(self):
        """Refresh performance metrics."""
        monitor = get_performance_monitor()
        
        # Get summary
        summary = await self.run_worker(monitor.get_session_summary).wait()
        
        # Get provider comparison
        comparison = await self.run_worker(monitor.get_provider_comparison).wait()
        
        # Update UI
        self.metrics_data = {
            'summary': summary,
            'comparison': comparison
        }
    
    def watch_metrics_data(self, old_data: Dict, new_data: Dict):
        """Update UI when metrics change."""
        if not new_data:
            return
        
        summary = new_data.get('summary', {})
        comparison = new_data.get('comparison', {})
        
        # Update summary cards
        self.query_one("#total-sessions", Static).update(
            str(summary.get('total_sessions', 0))
        )
        
        self.query_one("#total-words", Static).update(
            f"{summary.get('total_words', 0):,}"
        )
        
        avg_wpm = summary.get('average_wpm', 0)
        wpm_widget = self.query_one("#avg-wpm", Static)
        wpm_widget.update(f"{avg_wpm:.0f}")
        self._apply_performance_class(wpm_widget, avg_wpm, 100, 150)
        
        avg_latency = summary.get('average_latency', 0)
        latency_widget = self.query_one("#avg-latency", Static)
        latency_widget.update(f"{avg_latency:.0f}ms")
        self._apply_performance_class(latency_widget, avg_latency, 200, 100, inverse=True)
        
        efficiency = summary.get('average_efficiency', 0) * 100
        eff_widget = self.query_one("#efficiency", Static)
        eff_widget.update(f"{efficiency:.0f}%")
        self._apply_performance_class(eff_widget, efficiency, 70, 85)
        
        # Format active time
        total_duration = summary.get('total_duration', 0)
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        self.query_one("#active-time", Static).update(f"{hours}h {minutes}m")
        
        # Update provider comparison
        self._update_provider_comparison(comparison)
        
        # Update recent sessions
        self._update_recent_sessions(summary.get('recent_sessions', []))
    
    def _apply_performance_class(
        self, 
        widget: Static, 
        value: float, 
        warning_threshold: float,
        good_threshold: float,
        inverse: bool = False
    ):
        """Apply performance color class based on value."""
        widget.remove_class("performance-good", "performance-warning", "performance-poor")
        
        if inverse:
            # Lower is better
            if value <= good_threshold:
                widget.add_class("performance-good")
            elif value <= warning_threshold:
                widget.add_class("performance-warning")
            else:
                widget.add_class("performance-poor")
        else:
            # Higher is better
            if value >= good_threshold:
                widget.add_class("performance-good")
            elif value >= warning_threshold:
                widget.add_class("performance-warning")
            else:
                widget.add_class("performance-poor")
    
    def _update_provider_comparison(self, comparison: Dict[str, Dict[str, float]]):
        """Update provider comparison display."""
        container = self.query_one("#provider-stats", Container)
        container.remove_children()
        
        if not comparison:
            container.mount(Static("No provider data available", classes="help-text"))
            return
        
        # Find max values for scaling
        max_wpm = max(p['average_wpm'] for p in comparison.values())
        
        for provider, stats in comparison.items():
            with container:
                with Horizontal(classes="comparison-row"):
                    # Provider name
                    container.mount(
                        Label(provider.title(), classes="provider-name")
                    )
                    
                    # WPM bar
                    wpm_percent = (stats['average_wpm'] / max_wpm * 100) if max_wpm > 0 else 0
                    bar = ProgressBar(total=100, show_eta=False)
                    bar.advance(wpm_percent)
                    container.mount(bar)
                    
                    # Stats text
                    container.mount(
                        Static(
                            f"{stats['average_wpm']:.0f} WPM | "
                            f"{stats['average_latency']:.0f}ms | "
                            f"{stats['session_count']} sessions"
                        )
                    )
    
    def _update_recent_sessions(self, sessions: list):
        """Update recent sessions list."""
        container = self.query_one("#recent-list", Container)
        container.remove_children()
        
        if not sessions:
            container.mount(Static("No recent sessions", classes="help-text"))
            return
        
        for session in sessions[:10]:  # Show last 10
            timestamp = session['timestamp'][:16]  # Just date and time
            duration = session['duration']
            words = session['words']
            wpm = session['wpm']
            provider = session['provider']
            
            # Format duration
            dur_min = int(duration // 60)
            dur_sec = int(duration % 60)
            
            text = f"{timestamp} - {dur_min}:{dur_sec:02d} - "
            text += f"{words} words ({wpm:.0f} WPM) - {provider}"
            
            container.mount(Static(text, classes="session-item"))
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "refresh-btn":
            self.run_worker(self.refresh_metrics())