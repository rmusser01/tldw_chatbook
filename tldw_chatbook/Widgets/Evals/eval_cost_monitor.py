# eval_cost_monitor.py
# Description: Cost monitoring widget for evaluation dashboard
#
"""
Evaluation Cost Monitor Widget
------------------------------

Provides real-time cost tracking and budget warnings:
- Daily/weekly/monthly cost tracking
- Budget limit warnings
- Cost breakdown by model/task
- Smart suggestions for cost optimization
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, Grid
from textual.widgets import Static, ProgressBar, Button, Label
from textual.reactive import reactive
from textual.message import Message
from loguru import logger

@dataclass
class CostLimit:
    """Budget limit configuration."""
    daily: float = 10.0
    weekly: float = 50.0
    monthly: float = 200.0
    warning_threshold: float = 0.8  # Warn at 80% of limit

@dataclass
class CostBreakdown:
    """Cost breakdown by category."""
    by_model: Dict[str, float]
    by_task: Dict[str, float]
    by_date: Dict[str, float]
    total: float

class CostWarning(Message):
    """Message emitted when approaching budget limit."""
    def __init__(self, limit_type: str, current: float, limit: float, suggestions: List[str]):
        super().__init__()
        self.limit_type = limit_type
        self.current = current
        self.limit = limit
        self.suggestions = suggestions

class CostMonitor(Container):
    """
    Cost monitoring widget with budget tracking and warnings.
    
    Features:
    - Real-time cost tracking
    - Budget progress bars
    - Warning notifications
    - Cost optimization suggestions
    """
    
    # Reactive state
    current_cost: reactive[float] = reactive(0.0)
    daily_cost: reactive[float] = reactive(0.0)
    weekly_cost: reactive[float] = reactive(0.0)
    monthly_cost: reactive[float] = reactive(0.0)
    
    # Configuration
    limits: reactive[CostLimit] = reactive(CostLimit())
    
    def __init__(self, cost_history: Optional[List[Dict[str, Any]]] = None, **kwargs):
        super().__init__(**kwargs)
        self.cost_history = cost_history or []
        self._warning_shown = {
            'daily': False,
            'weekly': False,
            'monthly': False
        }
    
    def compose(self) -> ComposeResult:
        """Compose the cost monitor widget."""
        with Vertical(classes="cost-monitor"):
            yield Static("💰 Cost Monitor", classes="monitor-title")
            
            # Current run cost
            with Horizontal(classes="current-cost-row"):
                yield Static("Current Run:", classes="cost-label")
                yield Static("$0.00", id="current-cost", classes="cost-value")
            
            # Budget progress bars
            with Grid(classes="budget-grid"):
                # Daily budget
                with Vertical(classes="budget-item"):
                    yield Label("Daily Budget", classes="budget-label")
                    yield ProgressBar(
                        total=100,
                        show_eta=False,
                        id="daily-progress",
                        classes="budget-progress"
                    )
                    yield Static("$0.00 / $10.00", id="daily-text", classes="budget-text")
                
                # Weekly budget
                with Vertical(classes="budget-item"):
                    yield Label("Weekly Budget", classes="budget-label")
                    yield ProgressBar(
                        total=100,
                        show_eta=False,
                        id="weekly-progress",
                        classes="budget-progress"
                    )
                    yield Static("$0.00 / $50.00", id="weekly-text", classes="budget-text")
                
                # Monthly budget
                with Vertical(classes="budget-item"):
                    yield Label("Monthly Budget", classes="budget-label")
                    yield ProgressBar(
                        total=100,
                        show_eta=False,
                        id="monthly-progress",
                        classes="budget-progress"
                    )
                    yield Static("$0.00 / $200.00", id="monthly-text", classes="budget-text")
            
            # Cost optimization suggestions
            with Container(id="cost-suggestions", classes="suggestions-container"):
                yield Static("💡 Cost Optimization Tips", classes="suggestions-title")
                yield Static("", id="suggestions-text", classes="suggestions-text")
            
            # Actions
            with Horizontal(classes="monitor-actions"):
                yield Button("View Details", id="view-details", variant="default")
                yield Button("Set Limits", id="set-limits", variant="default")
    
    def on_mount(self) -> None:
        """Initialize cost tracking on mount."""
        self._calculate_period_costs()
        self._update_displays()
    
    def watch_current_cost(self, old_cost: float, new_cost: float) -> None:
        """React to current cost changes."""
        self.query_one("#current-cost").update(f"${new_cost:.2f}")
        
    def watch_daily_cost(self, old_cost: float, new_cost: float) -> None:
        """React to daily cost changes."""
        self._update_budget_display("daily", new_cost, self.limits.daily)
        self._check_budget_warning("daily", new_cost, self.limits.daily)
        
    def watch_weekly_cost(self, old_cost: float, new_cost: float) -> None:
        """React to weekly cost changes."""
        self._update_budget_display("weekly", new_cost, self.limits.weekly)
        self._check_budget_warning("weekly", new_cost, self.limits.weekly)
        
    def watch_monthly_cost(self, old_cost: float, new_cost: float) -> None:
        """React to monthly cost changes."""
        self._update_budget_display("monthly", new_cost, self.limits.monthly)
        self._check_budget_warning("monthly", new_cost, self.limits.monthly)
    
    def add_cost(self, amount: float, model: str, task: str) -> None:
        """Add a cost entry."""
        entry = {
            'amount': amount,
            'model': model,
            'task': task,
            'timestamp': datetime.now()
        }
        self.cost_history.append(entry)
        
        # Update current cost
        self.current_cost += amount
        
        # Recalculate period costs
        self._calculate_period_costs()
        
        # Update suggestions
        self._update_suggestions()
    
    def reset_current_cost(self) -> None:
        """Reset current run cost."""
        self.current_cost = 0.0
    
    def _calculate_period_costs(self) -> None:
        """Calculate costs for different time periods."""
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        daily_total = 0.0
        weekly_total = 0.0
        monthly_total = 0.0
        
        for entry in self.cost_history:
            timestamp = entry['timestamp']
            amount = entry['amount']
            
            if timestamp >= month_start:
                monthly_total += amount
                if timestamp >= week_start:
                    weekly_total += amount
                    if timestamp >= today_start:
                        daily_total += amount
        
        self.daily_cost = daily_total
        self.weekly_cost = weekly_total
        self.monthly_cost = monthly_total
    
    def _update_budget_display(self, period: str, current: float, limit: float) -> None:
        """Update budget progress bar and text."""
        percentage = min(100, (current / limit * 100) if limit > 0 else 0)
        
        progress_bar = self.query_one(f"#{period}-progress", ProgressBar)
        progress_bar.update(progress=percentage)
        
        # Change color based on percentage
        if percentage >= 100:
            progress_bar.add_class("over-budget")
        elif percentage >= 80:
            progress_bar.add_class("warning-budget")
        else:
            progress_bar.remove_class("over-budget", "warning-budget")
        
        # Update text
        text_widget = self.query_one(f"#{period}-text")
        text_widget.update(f"${current:.2f} / ${limit:.2f}")
    
    def _check_budget_warning(self, period: str, current: float, limit: float) -> None:
        """Check if we should emit a budget warning."""
        percentage = (current / limit) if limit > 0 else 0
        
        if percentage >= self.limits.warning_threshold and not self._warning_shown[period]:
            self._warning_shown[period] = True
            
            # Generate suggestions based on usage
            suggestions = self._generate_cost_suggestions(period, current, limit)
            
            # Emit warning message
            self.post_message(CostWarning(period, current, limit, suggestions))
    
    def _generate_cost_suggestions(self, period: str, current: float, limit: float) -> List[str]:
        """Generate cost optimization suggestions."""
        suggestions = []
        
        # Analyze cost breakdown
        breakdown = self.get_cost_breakdown()
        
        # Find most expensive model
        if breakdown.by_model:
            expensive_model = max(breakdown.by_model.items(), key=lambda x: x[1])
            if expensive_model[1] > current * 0.5:  # If one model is >50% of cost
                suggestions.append(f"Consider using a cheaper model than {expensive_model[0]}")
        
        # General suggestions
        remaining = limit - current
        if remaining > 0:
            samples_remaining = int(remaining / 0.001)  # Rough estimate
            suggestions.append(f"Budget remaining: ${remaining:.2f} (~{samples_remaining} samples)")
        
        suggestions.extend([
            "Use GPT-3.5 instead of GPT-4 (-95% cost)",
            "Reduce sample count for initial testing",
            "Enable result caching to avoid re-runs",
            "Use batch processing for better rates"
        ])
        
        return suggestions[:4]  # Limit to 4 suggestions
    
    def _update_suggestions(self) -> None:
        """Update cost optimization suggestions display."""
        suggestions = self._generate_cost_suggestions(
            "current", 
            self.current_cost, 
            self.limits.daily - self.daily_cost + self.current_cost
        )
        
        suggestions_text = "\n".join(f"• {s}" for s in suggestions)
        self.query_one("#suggestions-text").update(suggestions_text)
    
    def _update_displays(self) -> None:
        """Update all display elements."""
        self._update_budget_display("daily", self.daily_cost, self.limits.daily)
        self._update_budget_display("weekly", self.weekly_cost, self.limits.weekly)
        self._update_budget_display("monthly", self.monthly_cost, self.limits.monthly)
        self._update_suggestions()
    
    def get_cost_breakdown(self) -> CostBreakdown:
        """Get detailed cost breakdown."""
        by_model = {}
        by_task = {}
        by_date = {}
        total = 0.0
        
        for entry in self.cost_history:
            amount = entry['amount']
            model = entry.get('model', 'unknown')
            task = entry.get('task', 'unknown')
            date = entry['timestamp'].date().isoformat()
            
            by_model[model] = by_model.get(model, 0) + amount
            by_task[task] = by_task.get(task, 0) + amount
            by_date[date] = by_date.get(date, 0) + amount
            total += amount
        
        return CostBreakdown(
            by_model=by_model,
            by_task=by_task,
            by_date=by_date,
            total=total
        )
    
    def set_limits(self, limits: CostLimit) -> None:
        """Update budget limits."""
        self.limits = limits
        self._warning_shown = {k: False for k in self._warning_shown}  # Reset warnings
        self._update_displays()