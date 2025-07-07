# cost_estimation_widget.py
# Description: Widget for displaying cost estimation in evaluations
#
"""
Cost Estimation Widget
---------------------

Provides real-time cost estimation and tracking display:
- Pre-run cost estimates
- Live cost tracking during evaluation
- Historical cost analysis
- Budget warnings
"""

from typing import Optional, Dict, Any
from textual import on
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import Static, Button, Label, ProgressBar
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.css.query import NoMatches
from loguru import logger

from ..Evals.cost_estimator import CostEstimator, estimate_tokens

class CostEstimationWidget(Container):
    """Widget for displaying cost estimation and tracking."""
    
    # Reactive attributes
    current_cost = reactive(0.0)
    estimated_total = reactive(0.0)
    is_tracking = reactive(False)
    provider = reactive("")
    model = reactive("")
    
    def __init__(self, cost_estimator: Optional[CostEstimator] = None, **kwargs):
        super().__init__(**kwargs)
        self.cost_estimator = cost_estimator or CostEstimator()
        self.run_id: Optional[str] = None
        self.num_samples = 0
        self.completed_samples = 0
    
    def compose(self) -> ComposeResult:
        with Vertical(classes="cost-widget"):
            yield Static("💰 Cost Estimation", classes="widget-title")
            
            # Pre-run estimation
            with Container(classes="cost-section", id="estimation-section"):
                yield Static("Estimated Cost", classes="section-label")
                yield Static("$0.00", id="estimated-cost", classes="cost-display large")
                yield Static("", id="cost-breakdown", classes="cost-details")
            
            # Live tracking (hidden initially)
            with Container(classes="cost-section hidden", id="tracking-section"):
                yield Static("Current Cost", classes="section-label")
                
                with Horizontal(classes="cost-progress"):
                    yield Static("$0.00", id="current-cost", classes="cost-display")
                    yield Static("/", classes="separator")
                    yield Static("$0.00", id="total-estimate", classes="cost-display muted")
                
                yield ProgressBar(id="cost-progress-bar", show_percentage=False)
                yield Static("0 / 0 samples", id="sample-progress", classes="progress-text")
            
            # Cost breakdown
            with Container(classes="cost-section", id="details-section"):
                yield Static("Details", classes="section-label collapsible")
                
                with Container(classes="details-content collapsed", id="details-content"):
                    yield Static("", id="provider-info", classes="info-row")
                    yield Static("", id="pricing-info", classes="info-row")
                    yield Static("", id="token-info", classes="info-row")
                    yield Static("", id="quality-tier", classes="info-row")
            
            # Budget warning (hidden initially)
            with Container(classes="warning-section hidden", id="budget-warning"):
                yield Static("⚠️ Approaching budget limit", classes="warning-text")
                yield Button("View Alternatives", id="view-alternatives", variant="warning")
    
    def estimate_cost(
        self, 
        provider: str, 
        model_id: str, 
        num_samples: int,
        avg_input_length: int = 2000,  # Characters
        avg_output_length: int = 800
    ) -> None:
        """Display cost estimation for upcoming run."""
        self.provider = provider
        self.model = model_id
        self.num_samples = num_samples
        
        # Estimate tokens from character counts
        avg_input_tokens = estimate_tokens(str(avg_input_length))
        avg_output_tokens = estimate_tokens(str(avg_output_length))
        
        # Get estimation
        estimation = self.cost_estimator.estimate_run_cost(
            provider, model_id, num_samples,
            avg_input_tokens, avg_output_tokens
        )
        
        self.estimated_total = estimation['estimated_cost']
        
        # Update display
        self._update_estimation_display(estimation)
    
    def start_tracking(self, run_id: str) -> None:
        """Start live cost tracking."""
        self.run_id = run_id
        self.is_tracking = True
        self.current_cost = 0.0
        self.completed_samples = 0
        
        self.cost_estimator.start_tracking(run_id)
        
        # Show tracking section
        try:
            tracking = self.query_one("#tracking-section")
            tracking.remove_class("hidden")
            
            # Update displays
            self.query_one("#total-estimate").update(
                self.cost_estimator.format_cost_display(self.estimated_total)
            )
        except NoMatches:
            pass
    
    def update_sample_cost(
        self, 
        input_tokens: int, 
        output_tokens: int,
        sample_index: int
    ) -> None:
        """Update cost for completed sample."""
        if not self.is_tracking or not self.run_id:
            return
        
        # Add sample cost
        sample_cost = self.cost_estimator.add_sample_cost(
            self.run_id, input_tokens, output_tokens,
            self.provider, self.model
        )
        
        self.current_cost = self.cost_estimator.get_run_cost(self.run_id)
        self.completed_samples = sample_index + 1
        
        # Update displays
        self._update_tracking_display()
        
        # Check budget warnings
        if self.estimated_total > 0:
            progress_ratio = self.current_cost / self.estimated_total
            if progress_ratio > 1.2:  # 20% over estimate
                self._show_budget_warning()
    
    def finalize_tracking(self) -> Dict[str, Any]:
        """Finalize cost tracking and show summary."""
        if not self.run_id:
            return {}
        
        result = self.cost_estimator.finalize_run(
            self.run_id,
            {
                "provider": self.provider,
                "model": self.model,
                "num_samples": self.completed_samples
            }
        )
        
        self.is_tracking = False
        self._show_final_summary(result)
        
        return result
    
    def _update_estimation_display(self, estimation: Dict[str, Any]) -> None:
        """Update the estimation display."""
        try:
            # Main cost display
            cost_display = self.query_one("#estimated-cost")
            cost_display.update(
                self.cost_estimator.format_cost_display(estimation['estimated_cost'])
            )
            
            if estimation['is_free']:
                cost_display.add_class("free")
            else:
                cost_display.remove_class("free")
            
            # Breakdown
            breakdown = self.query_one("#cost-breakdown")
            if not estimation['is_free']:
                breakdown_text = (
                    f"{estimation['num_samples']} samples × "
                    f"${estimation['breakdown']['cost_per_sample']:.4f}/sample"
                )
                breakdown.update(breakdown_text)
            else:
                breakdown.update(estimation.get('message', 'Free model'))
            
            # Details
            self.query_one("#provider-info").update(
                f"Provider: {estimation['provider']} / {estimation['model_id']}"
            )
            
            if 'pricing' in estimation:
                pricing = estimation['pricing']
                self.query_one("#pricing-info").update(
                    f"Pricing: ${pricing['input_price_per_1k']:.4f} / "
                    f"${pricing['output_price_per_1k']:.4f} per 1k tokens"
                )
            
            if 'breakdown' in estimation:
                tokens = estimation['breakdown']
                self.query_one("#token-info").update(
                    f"Tokens: ~{tokens['input_tokens']:,} input, "
                    f"~{tokens['output_tokens']:,} output"
                )
            
        except NoMatches:
            logger.warning("Cost estimation widgets not found")
    
    def _update_tracking_display(self) -> None:
        """Update live tracking display."""
        try:
            # Current cost
            self.query_one("#current-cost").update(
                self.cost_estimator.format_cost_display(self.current_cost)
            )
            
            # Progress bar
            if self.num_samples > 0:
                progress = self.completed_samples / self.num_samples
                progress_bar = self.query_one("#cost-progress-bar", ProgressBar)
                progress_bar.update(progress=progress)
            
            # Sample progress
            self.query_one("#sample-progress").update(
                f"{self.completed_samples} / {self.num_samples} samples"
            )
            
            # Color coding based on estimate accuracy
            if self.estimated_total > 0:
                ratio = self.current_cost / self.estimated_total
                cost_elem = self.query_one("#current-cost")
                
                if ratio > 1.2:
                    cost_elem.add_class("over-budget")
                elif ratio > 1.0:
                    cost_elem.add_class("near-budget")
                else:
                    cost_elem.remove_class("over-budget", "near-budget")
                    
        except NoMatches:
            logger.warning("Tracking widgets not found")
    
    def _show_budget_warning(self) -> None:
        """Show budget warning."""
        try:
            warning = self.query_one("#budget-warning")
            warning.remove_class("hidden")
        except NoMatches:
            pass
    
    def _show_final_summary(self, result: Dict[str, Any]) -> None:
        """Show final cost summary."""
        try:
            # Update final displays
            final_cost = result['total_cost']
            
            # Show accuracy
            if self.estimated_total > 0:
                accuracy = abs(final_cost - self.estimated_total) / self.estimated_total
                accuracy_text = f"Estimate accuracy: {(1 - accuracy) * 100:.1f}%"
            else:
                accuracy_text = ""
            
            breakdown = self.query_one("#cost-breakdown")
            breakdown.update(
                f"Final cost: {self.cost_estimator.format_cost_display(final_cost)} "
                f"{accuracy_text}"
            )
            
        except NoMatches:
            pass
    
    def on_click(self, event) -> None:
        """Handle click events on the widget."""
        # Check if the clicked element has the collapsible class
        if event.target and hasattr(event.target, 'has_class') and event.target.has_class("collapsible"):
            self.toggle_details()
    
    def toggle_details(self) -> None:
        """Toggle details section."""
        try:
            content = self.query_one("#details-content")
            if content.has_class("collapsed"):
                content.remove_class("collapsed")
            else:
                content.add_class("collapsed")
        except NoMatches:
            pass
    
    @on(Button.Pressed, "#view-alternatives")
    def show_alternatives(self) -> None:
        """Show alternative model recommendations."""
        # This would open a dialog with cheaper alternatives
        budget = self.estimated_total * 0.8  # 80% of current estimate
        recommendations = self.cost_estimator.get_provider_recommendations(
            "simple_qa",  # Could be dynamic based on task
            budget
        )
        
        logger.info(f"Alternative recommendations: {recommendations}")
        # TODO: Show in a dialog

class CostSummaryWidget(Container):
    """Widget for displaying historical cost summary."""
    
    def __init__(self, cost_estimator: CostEstimator, **kwargs):
        super().__init__(**kwargs)
        self.cost_estimator = cost_estimator
    
    def compose(self) -> ComposeResult:
        with Vertical(classes="cost-summary-widget"):
            yield Static("📊 Cost Summary (Last 30 Days)", classes="widget-title")
            
            with Grid(classes="summary-grid"):
                yield Static("Total Spent:", classes="label")
                yield Static("$0.00", id="total-spent", classes="value")
                
                yield Static("Runs:", classes="label")
                yield Static("0", id="num-runs", classes="value")
                
                yield Static("Avg per Run:", classes="label")
                yield Static("$0.00", id="avg-cost", classes="value")
                
                yield Static("Most Used:", classes="label")
                yield Static("N/A", id="most-used", classes="value")
            
            yield Button("View Detailed Report", id="view-report", variant="primary")
    
    def on_mount(self) -> None:
        """Update summary on mount."""
        self.update_summary()
    
    def update_summary(self) -> None:
        """Update the cost summary display."""
        summary = self.cost_estimator.get_cost_summary(30)
        
        try:
            self.query_one("#total-spent").update(
                self.cost_estimator.format_cost_display(summary['total_cost'])
            )
            self.query_one("#num-runs").update(str(summary['num_runs']))
            self.query_one("#avg-cost").update(
                self.cost_estimator.format_cost_display(summary['average_cost_per_run'])
            )
            
            # Most used provider
            if summary['provider_breakdown']:
                most_used = max(summary['provider_breakdown'].items(), 
                              key=lambda x: x[1])
                self.query_one("#most-used").update(most_used[0])
            
        except NoMatches:
            logger.warning("Summary widgets not found")