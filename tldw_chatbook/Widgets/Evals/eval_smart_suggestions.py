# eval_smart_suggestions.py
# Description: Smart suggestions engine for evaluation configuration
#
"""
Evaluation Smart Suggestions Widget
-----------------------------------

Provides intelligent suggestions for evaluation configuration:
- Model recommendations based on task type
- Sample size optimization based on budget
- Learning from usage patterns
- Quick access to frequent configurations
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from dataclasses import dataclass
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Static, Button, Label
from textual.reactive import reactive
from textual.message import Message
from loguru import logger
import json
from pathlib import Path

@dataclass
class ModelRecommendation:
    """Model recommendation with reasoning."""
    model_id: str
    provider: str
    reason: str
    confidence: float
    estimated_cost: float
    estimated_time: int  # minutes

@dataclass
class ConfigSuggestion:
    """Configuration suggestion."""
    task: str
    model: str
    samples: int
    config: Dict[str, Any]
    reason: str
    based_on: str  # "history", "task_type", "budget", etc.

class SuggestionSelected(Message):
    """Message emitted when a suggestion is selected."""
    def __init__(self, suggestion: ConfigSuggestion):
        super().__init__()
        self.suggestion = suggestion

class SmartSuggestions(Container):
    """
    Smart suggestions engine for evaluation configuration.
    
    Features:
    - Task-based model recommendations
    - Budget-aware sample sizing
    - Usage pattern learning
    - Quick configuration templates
    """
    
    # Usage history for pattern learning
    usage_history: reactive[List[Dict[str, Any]]] = reactive([])
    
    # Current context
    current_task: reactive[Optional[str]] = reactive(None)
    current_budget: reactive[float] = reactive(10.0)
    
    def __init__(self, history_file: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        self.history_file = history_file
        self._load_history()
        
        # Task-model affinity matrix (based on best practices)
        self.task_model_affinity = {
            'mmlu': {
                'gpt-4': 0.9,
                'claude-3-opus': 0.9,
                'gpt-3.5-turbo': 0.7,
                'gemini-pro': 0.8,
            },
            'gsm8k': {
                'gpt-4': 0.95,
                'claude-3-opus': 0.9,
                'gpt-3.5-turbo': 0.6,
                'gemini-pro': 0.85,
            },
            'humaneval': {
                'gpt-4': 0.95,
                'claude-3-opus': 0.95,
                'gpt-3.5-turbo': 0.7,
                'gemini-pro': 0.8,
            },
            'truthfulqa': {
                'gpt-4': 0.85,
                'claude-3-opus': 0.9,
                'gpt-3.5-turbo': 0.6,
                'gemini-pro': 0.8,
            }
        }
        
        # Model characteristics
        self.model_characteristics = {
            'gpt-4': {'cost': 'high', 'quality': 'excellent', 'speed': 'slow'},
            'gpt-3.5-turbo': {'cost': 'low', 'quality': 'good', 'speed': 'fast'},
            'claude-3-opus': {'cost': 'high', 'quality': 'excellent', 'speed': 'medium'},
            'claude-3-sonnet': {'cost': 'medium', 'quality': 'very_good', 'speed': 'fast'},
            'gemini-pro': {'cost': 'low', 'quality': 'very_good', 'speed': 'fast'},
            'mixtral-8x7b': {'cost': 'very_low', 'quality': 'good', 'speed': 'very_fast'},
        }
    
    def compose(self) -> ComposeResult:
        """Compose the smart suggestions widget."""
        with Vertical(classes="smart-suggestions"):
            yield Static("🧠 Smart Suggestions", classes="suggestions-title")
            
            # Model recommendations
            with Container(classes="recommendations-section"):
                yield Label("Recommended Models", classes="section-label")
                yield Container(id="model-recommendations", classes="recommendations-list")
            
            # Configuration suggestions
            with Container(classes="suggestions-section"):
                yield Label("Quick Configurations", classes="section-label")
                yield Container(id="config-suggestions", classes="suggestions-list")
            
            # Usage insights
            with Container(classes="insights-section"):
                yield Label("Usage Insights", classes="section-label")
                yield Static("", id="usage-insights", classes="insights-text")
    
    def on_mount(self) -> None:
        """Initialize suggestions on mount."""
        self._update_suggestions()
        self._update_insights()
    
    def watch_current_task(self, old_task: Optional[str], new_task: Optional[str]) -> None:
        """Update suggestions when task changes."""
        if new_task:
            self._update_suggestions()
    
    def watch_current_budget(self, old_budget: float, new_budget: float) -> None:
        """Update suggestions when budget changes."""
        self._update_suggestions()
    
    def set_context(self, task: Optional[str], budget: float) -> None:
        """Update current context."""
        self.current_task = task
        self.current_budget = budget
    
    def record_usage(self, config: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Record usage for pattern learning."""
        usage_entry = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'result': result,
            'success': result.get('accuracy', 0) > 0.8  # Simple success metric
        }
        self.usage_history.append(usage_entry)
        
        # Keep last 1000 entries
        if len(self.usage_history) > 1000:
            self.usage_history = self.usage_history[-1000:]
        
        # Save history
        self._save_history()
        
        # Update insights
        self._update_insights()
    
    def _update_suggestions(self) -> None:
        """Update model and configuration suggestions."""
        # Update model recommendations
        if self.current_task:
            recommendations = self._get_model_recommendations(
                self.current_task, 
                self.current_budget
            )
            self._display_model_recommendations(recommendations)
        
        # Update configuration suggestions
        suggestions = self._get_config_suggestions()
        self._display_config_suggestions(suggestions)
    
    def _get_model_recommendations(
        self, 
        task: str, 
        budget: float
    ) -> List[ModelRecommendation]:
        """Get model recommendations for a task."""
        recommendations = []
        
        # Get task affinities
        affinities = self.task_model_affinity.get(task, {})
        
        for model_id, affinity in affinities.items():
            # Get model characteristics
            chars = self.model_characteristics.get(model_id, {})
            
            # Calculate recommendation score
            score = affinity
            
            # Adjust for budget
            if budget < 5 and chars.get('cost') == 'high':
                score *= 0.5  # Penalize expensive models on low budget
                reason = f"Good for {task}, but consider budget"
            elif budget > 20 and chars.get('cost') == 'low':
                reason = f"Cost-effective for {task} at scale"
            else:
                reason = f"Excellent for {task} tasks"
            
            # Learn from history
            history_score = self._get_history_score(task, model_id)
            score = score * 0.7 + history_score * 0.3
            
            # Estimate cost (simplified)
            cost_per_sample = {
                'high': 0.01,
                'medium': 0.005,
                'low': 0.001,
                'very_low': 0.0001
            }.get(chars.get('cost', 'medium'), 0.005)
            
            estimated_cost = cost_per_sample * 1000  # For 1000 samples
            
            recommendations.append(ModelRecommendation(
                model_id=model_id,
                provider=self._get_provider_for_model(model_id),
                reason=reason,
                confidence=score,
                estimated_cost=estimated_cost,
                estimated_time=self._estimate_time(model_id, 1000)
            ))
        
        # Sort by score
        recommendations.sort(key=lambda r: r.confidence, reverse=True)
        
        return recommendations[:3]  # Top 3
    
    def _get_config_suggestions(self) -> List[ConfigSuggestion]:
        """Get configuration suggestions based on patterns."""
        suggestions = []
        
        # Frequent configurations
        frequent_configs = self._get_frequent_configurations()
        for config, count in frequent_configs[:2]:
            suggestions.append(ConfigSuggestion(
                task=config['task'],
                model=config['model'],
                samples=config.get('samples', 1000),
                config=config.get('config', {}),
                reason=f"Used {count} times successfully",
                based_on="history"
            ))
        
        # Budget-optimized suggestion
        if self.current_task and self.current_budget:
            budget_config = self._get_budget_optimized_config(
                self.current_task, 
                self.current_budget
            )
            if budget_config:
                suggestions.append(budget_config)
        
        # Task-specific best practice
        if self.current_task:
            best_practice = self._get_best_practice_config(self.current_task)
            if best_practice:
                suggestions.append(best_practice)
        
        return suggestions[:4]  # Limit to 4
    
    def _get_frequent_configurations(self) -> List[Tuple[Dict[str, Any], int]]:
        """Get frequently used configurations."""
        config_counter = Counter()
        
        for entry in self.usage_history:
            if entry.get('success', False):
                # Serialize config for counting
                config_key = json.dumps(entry['config'], sort_keys=True)
                config_counter[config_key] += 1
        
        # Get top configurations
        top_configs = []
        for config_str, count in config_counter.most_common(5):
            config = json.loads(config_str)
            top_configs.append((config, count))
        
        return top_configs
    
    def _get_budget_optimized_config(
        self, 
        task: str, 
        budget: float
    ) -> Optional[ConfigSuggestion]:
        """Get configuration optimized for budget."""
        # Calculate optimal sample size for budget
        # Assume average cost per sample
        avg_cost_per_sample = 0.003
        max_samples = int(budget / avg_cost_per_sample)
        
        # Find cheapest viable model
        cheap_models = [
            ('gpt-3.5-turbo', 'openai', 0.001),
            ('mixtral-8x7b', 'groq', 0.0001),
            ('gemini-pro', 'google', 0.0005),
        ]
        
        for model_id, provider, cost_per_sample in cheap_models:
            if self.task_model_affinity.get(task, {}).get(model_id, 0) > 0.6:
                samples = min(max_samples, int(budget / cost_per_sample))
                return ConfigSuggestion(
                    task=task,
                    model=f"{provider}/{model_id}",
                    samples=samples,
                    config={'temperature': 0.3, 'max_tokens': 512},
                    reason=f"Maximizes samples ({samples}) within ${budget:.2f} budget",
                    based_on="budget"
                )
        
        return None
    
    def _get_best_practice_config(self, task: str) -> Optional[ConfigSuggestion]:
        """Get best practice configuration for a task."""
        best_practices = {
            'mmlu': {
                'model': 'openai/gpt-4',
                'samples': 1000,
                'config': {'temperature': 0.0, 'max_tokens': 1024},
                'reason': 'Standard MMLU evaluation setup'
            },
            'gsm8k': {
                'model': 'openai/gpt-4',
                'samples': 500,
                'config': {'temperature': 0.0, 'max_tokens': 1024},
                'reason': 'Math tasks benefit from deterministic output'
            },
            'humaneval': {
                'model': 'openai/gpt-4',
                'samples': 164,  # Full dataset
                'config': {'temperature': 0.2, 'max_tokens': 1024},
                'reason': 'Complete HumanEval benchmark'
            },
            'truthfulqa': {
                'model': 'anthropic/claude-3-opus',
                'samples': 817,  # Full dataset
                'config': {'temperature': 0.3, 'max_tokens': 512},
                'reason': 'Claude excels at truthfulness evaluation'
            }
        }
        
        if task in best_practices:
            bp = best_practices[task]
            return ConfigSuggestion(
                task=task,
                model=bp['model'],
                samples=bp['samples'],
                config=bp['config'],
                reason=bp['reason'],
                based_on="best_practice"
            )
        
        return None
    
    def _get_history_score(self, task: str, model: str) -> float:
        """Get historical performance score for task-model pair."""
        successes = 0
        total = 0
        
        for entry in self.usage_history:
            if (entry['config'].get('task') == task and 
                model in entry['config'].get('model', '')):
                total += 1
                if entry.get('success', False):
                    successes += 1
        
        if total > 0:
            return successes / total
        return 0.5  # Neutral if no history
    
    def _get_provider_for_model(self, model_id: str) -> str:
        """Get provider for a model ID."""
        provider_map = {
            'gpt': 'openai',
            'claude': 'anthropic',
            'gemini': 'google',
            'mixtral': 'groq',
        }
        
        for key, provider in provider_map.items():
            if key in model_id:
                return provider
        
        return 'unknown'
    
    def _estimate_time(self, model_id: str, samples: int) -> int:
        """Estimate evaluation time in minutes."""
        # Rough estimates based on model speed
        speed_map = {
            'very_fast': 0.5,  # seconds per sample
            'fast': 1.0,
            'medium': 2.0,
            'slow': 3.0
        }
        
        chars = self.model_characteristics.get(model_id, {})
        speed = speed_map.get(chars.get('speed', 'medium'), 2.0)
        
        total_seconds = samples * speed
        return max(1, int(total_seconds / 60))
    
    def _display_model_recommendations(
        self, 
        recommendations: List[ModelRecommendation]
    ) -> None:
        """Display model recommendations in UI."""
        container = self.query_one("#model-recommendations")
        container.remove_children()
        
        for rec in recommendations:
            with container:
                with Horizontal(classes="recommendation-item"):
                    # Model info
                    with Vertical(classes="model-info"):
                        yield Static(
                            f"{rec.provider}/{rec.model_id}",
                            classes="model-name"
                        )
                        yield Static(
                            f"{rec.reason} (confidence: {rec.confidence:.0%})",
                            classes="model-reason"
                        )
                    
                    # Cost and time
                    with Vertical(classes="model-stats"):
                        yield Static(
                            f"~${rec.estimated_cost:.2f}",
                            classes="model-cost"
                        )
                        yield Static(
                            f"~{rec.estimated_time}min",
                            classes="model-time"
                        )
                    
                    # Use button
                    yield Button(
                        "Use",
                        classes="use-model-btn",
                        id=f"use-{rec.model_id}"
                    )
    
    def _display_config_suggestions(
        self, 
        suggestions: List[ConfigSuggestion]
    ) -> None:
        """Display configuration suggestions in UI."""
        container = self.query_one("#config-suggestions")
        container.remove_children()
        
        for i, suggestion in enumerate(suggestions):
            with container:
                with Horizontal(classes="suggestion-item"):
                    with Vertical(classes="suggestion-info"):
                        yield Static(
                            f"{suggestion.task} on {suggestion.model.split('/')[-1]}",
                            classes="suggestion-name"
                        )
                        yield Static(
                            f"{suggestion.samples} samples • {suggestion.reason}",
                            classes="suggestion-details"
                        )
                    
                    yield Button(
                        "Apply",
                        classes="apply-suggestion-btn",
                        id=f"apply-suggestion-{i}"
                    )
    
    def _update_insights(self) -> None:
        """Update usage insights display."""
        insights = []
        
        # Most successful model
        model_success = defaultdict(lambda: {'success': 0, 'total': 0})
        for entry in self.usage_history:
            model = entry['config'].get('model', 'unknown')
            model_success[model]['total'] += 1
            if entry.get('success', False):
                model_success[model]['success'] += 1
        
        if model_success:
            best_model = max(
                model_success.items(),
                key=lambda x: x[1]['success'] / x[1]['total'] if x[1]['total'] > 0 else 0
            )
            success_rate = best_model[1]['success'] / best_model[1]['total'] * 100
            insights.append(
                f"Best performer: {best_model[0].split('/')[-1]} "
                f"({success_rate:.0f}% success rate)"
            )
        
        # Average cost per run
        if self.usage_history:
            avg_cost = sum(
                entry['result'].get('cost', 0) 
                for entry in self.usage_history
            ) / len(self.usage_history)
            insights.append(f"Average cost per run: ${avg_cost:.2f}")
        
        # Peak usage time
        if self.usage_history:
            hours = [
                datetime.fromisoformat(entry['timestamp']).hour 
                for entry in self.usage_history
            ]
            peak_hour = Counter(hours).most_common(1)[0][0]
            insights.append(f"Peak usage: {peak_hour}:00-{peak_hour+1}:00")
        
        # Update display
        insights_text = "\n".join(f"• {insight}" for insight in insights)
        self.query_one("#usage-insights").update(insights_text or "No usage data yet")
    
    def _load_history(self) -> None:
        """Load usage history from file."""
        if self.history_file and self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    self.usage_history = json.load(f)
                logger.info(f"Loaded {len(self.usage_history)} usage entries")
            except Exception as e:
                logger.error(f"Error loading history: {e}")
    
    def _save_history(self) -> None:
        """Save usage history to file."""
        if self.history_file:
            try:
                self.history_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.history_file, 'w') as f:
                    json.dump(self.usage_history, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving history: {e}")