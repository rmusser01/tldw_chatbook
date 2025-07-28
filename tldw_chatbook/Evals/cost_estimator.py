# cost_estimator.py
# Description: Cost estimation for evaluation runs
#
"""
Cost Estimator for LLM Evaluations
----------------------------------

Provides cost estimation and tracking for evaluation runs:
- Per-provider pricing models
- Token counting and estimation
- Real-time cost tracking during runs
- Historical cost analysis
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from loguru import logger
from ..Metrics.metrics_logger import log_counter, log_histogram

@dataclass
class ModelPricing:
    """Pricing information for a model."""
    provider: str
    model_id: str
    input_price_per_1k: float  # USD per 1k input tokens
    output_price_per_1k: float  # USD per 1k output tokens
    updated_at: datetime = field(default_factory=datetime.now)
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token counts."""
        input_cost = (input_tokens / 1000) * self.input_price_per_1k
        output_cost = (output_tokens / 1000) * self.output_price_per_1k
        return round(input_cost + output_cost, 6)

class CostEstimator:
    """Estimates and tracks costs for evaluation runs."""
    
    # Default pricing (as of 2024) - prices in USD per 1k tokens
    DEFAULT_PRICING = {
        # OpenAI
        "openai/gpt-4-turbo": ModelPricing("openai", "gpt-4-turbo", 0.01, 0.03),
        "openai/gpt-4": ModelPricing("openai", "gpt-4", 0.03, 0.06),
        "openai/gpt-3.5-turbo": ModelPricing("openai", "gpt-3.5-turbo", 0.0005, 0.0015),
        
        # Anthropic
        "anthropic/claude-3-opus": ModelPricing("anthropic", "claude-3-opus", 0.015, 0.075),
        "anthropic/claude-3-sonnet": ModelPricing("anthropic", "claude-3-sonnet", 0.003, 0.015),
        "anthropic/claude-3-haiku": ModelPricing("anthropic", "claude-3-haiku", 0.00025, 0.00125),
        
        # Cohere
        "cohere/command-r": ModelPricing("cohere", "command-r", 0.0005, 0.0015),
        "cohere/command-r-plus": ModelPricing("cohere", "command-r-plus", 0.003, 0.015),
        
        # Google
        "google/gemini-pro": ModelPricing("google", "gemini-pro", 0.00025, 0.0005),
        "google/gemini-pro-vision": ModelPricing("google", "gemini-pro-vision", 0.00025, 0.0005),
        
        # Groq (very competitive pricing)
        "groq/llama3-8b": ModelPricing("groq", "llama3-8b", 0.00005, 0.00008),
        "groq/llama3-70b": ModelPricing("groq", "llama3-70b", 0.00059, 0.00079),
        "groq/mixtral-8x7b": ModelPricing("groq", "mixtral-8x7b", 0.00027, 0.00027),
    }
    
    def __init__(self, custom_pricing_path: Optional[Path] = None):
        """Initialize cost estimator with optional custom pricing."""
        self.pricing = self.DEFAULT_PRICING.copy()
        
        # Load custom pricing if available
        if custom_pricing_path and custom_pricing_path.exists():
            self._load_custom_pricing(custom_pricing_path)
        
        # Track costs during runs
        self.current_run_costs: Dict[str, float] = {}
        self.run_history: List[Dict[str, Any]] = []
    
    def _load_custom_pricing(self, path: Path) -> None:
        """Load custom pricing from JSON file."""
        try:
            with open(path, 'r') as f:
                custom_data = json.load(f)
            
            for key, pricing_data in custom_data.items():
                self.pricing[key] = ModelPricing(
                    provider=pricing_data['provider'],
                    model_id=pricing_data['model_id'],
                    input_price_per_1k=pricing_data['input_price_per_1k'],
                    output_price_per_1k=pricing_data['output_price_per_1k']
                )
            
            logger.info(f"Loaded custom pricing for {len(custom_data)} models")
        except Exception as e:
            logger.error(f"Failed to load custom pricing: {e}")
    
    def estimate_run_cost(
        self, 
        provider: str,
        model_id: str,
        num_samples: int,
        avg_input_tokens: int = 500,
        avg_output_tokens: int = 200
    ) -> Dict[str, Any]:
        """
        Estimate cost for an evaluation run.
        
        Args:
            provider: LLM provider name
            model_id: Model identifier
            num_samples: Number of evaluation samples
            avg_input_tokens: Average input tokens per sample
            avg_output_tokens: Average output tokens per sample
            
        Returns:
            Cost estimation with breakdown
        """
        # Log cost estimation request
        log_counter("eval_cost_estimation_requested", labels={
            "provider": provider,
            "model_id": model_id
        })
        
        pricing_key = f"{provider}/{model_id}"
        
        # Try exact match first
        if pricing_key not in self.pricing:
            # Try provider defaults
            if provider == "openai" and "turbo" in model_id:
                pricing_key = "openai/gpt-3.5-turbo"
            elif provider == "anthropic" and "claude" in model_id:
                pricing_key = "anthropic/claude-3-haiku"  # Conservative estimate
            else:
                # Return zero cost for unknown models (e.g., local models)
                # Log free/unknown model
                log_counter("eval_cost_free_model", labels={
                    "provider": provider,
                    "model_id": model_id
                })
                
                return {
                    "estimated_cost": 0.0,
                    "is_free": True,
                    "provider": provider,
                    "model_id": model_id,
                    "num_samples": num_samples,
                    "message": "Local or unknown model - no cost"
                }
        
        pricing = self.pricing[pricing_key]
        
        # Calculate totals
        total_input_tokens = num_samples * avg_input_tokens
        total_output_tokens = num_samples * avg_output_tokens
        total_cost = pricing.calculate_cost(total_input_tokens, total_output_tokens)
        
        # Log cost estimation metrics
        log_histogram("eval_cost_estimated_total", total_cost, labels={
            "provider": provider,
            "model_id": model_id
        })
        log_histogram("eval_cost_estimated_per_sample", total_cost / num_samples if num_samples > 0 else 0, labels={
            "provider": provider,
            "model_id": model_id
        })
        log_histogram("eval_cost_estimated_input_tokens", total_input_tokens, labels={
            "provider": provider
        })
        log_histogram("eval_cost_estimated_output_tokens", total_output_tokens, labels={
            "provider": provider
        })
        
        return {
            "estimated_cost": total_cost,
            "is_free": False,
            "provider": provider,
            "model_id": model_id,
            "num_samples": num_samples,
            "breakdown": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "input_cost": (total_input_tokens / 1000) * pricing.input_price_per_1k,
                "output_cost": (total_output_tokens / 1000) * pricing.output_price_per_1k,
                "cost_per_sample": total_cost / num_samples if num_samples > 0 else 0
            },
            "pricing": {
                "input_price_per_1k": pricing.input_price_per_1k,
                "output_price_per_1k": pricing.output_price_per_1k
            }
        }
    
    def start_tracking(self, run_id: str) -> None:
        """Start tracking costs for a run."""
        self.current_run_costs[run_id] = 0.0
        
        # Log tracking start
        log_counter("eval_cost_tracking_started", labels={
            "run_id": run_id
        })
        
        logger.info(f"Started cost tracking for run {run_id}")
    
    def add_sample_cost(
        self, 
        run_id: str,
        input_tokens: int,
        output_tokens: int,
        provider: str,
        model_id: str
    ) -> float:
        """Add cost for a single sample."""
        if run_id not in self.current_run_costs:
            self.start_tracking(run_id)
        
        pricing_key = f"{provider}/{model_id}"
        if pricing_key not in self.pricing:
            log_counter("eval_cost_unknown_model_sample", labels={
                "provider": provider,
                "model_id": model_id
            })
            return 0.0  # No cost for unknown models
        
        pricing = self.pricing[pricing_key]
        cost = pricing.calculate_cost(input_tokens, output_tokens)
        self.current_run_costs[run_id] += cost
        
        # Log sample cost metrics
        log_histogram("eval_cost_sample_cost", cost, labels={
            "provider": provider,
            "model_id": model_id
        })
        log_histogram("eval_cost_sample_input_tokens", input_tokens, labels={
            "provider": provider
        })
        log_histogram("eval_cost_sample_output_tokens", output_tokens, labels={
            "provider": provider
        })
        log_counter("eval_cost_sample_added", labels={
            "provider": provider,
            "model_id": model_id
        })
        
        return cost
    
    def get_run_cost(self, run_id: str) -> float:
        """Get current cost for a run."""
        return self.current_run_costs.get(run_id, 0.0)
    
    def finalize_run(self, run_id: str, run_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize cost tracking for a run."""
        total_cost = self.current_run_costs.pop(run_id, 0.0)
        
        cost_record = {
            "run_id": run_id,
            "total_cost": total_cost,
            "timestamp": datetime.now().isoformat(),
            **run_metadata
        }
        
        self.run_history.append(cost_record)
        
        # Log finalized cost metrics
        log_histogram("eval_cost_run_total", total_cost, labels={
            "provider": run_metadata.get('provider', 'unknown'),
            "model_id": run_metadata.get('model_id', 'unknown')
        })
        log_counter("eval_cost_run_finalized", labels={
            "provider": run_metadata.get('provider', 'unknown'),
            "model_id": run_metadata.get('model_id', 'unknown'),
            "cost_tier": self._get_cost_tier(total_cost)
        })
        
        logger.info(f"Finalized run {run_id} with total cost: ${total_cost:.4f}")
        
        return cost_record
    
    def get_cost_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get cost summary for recent runs."""
        cutoff_date = datetime.now().timestamp() - (days * 24 * 3600)
        recent_runs = [
            run for run in self.run_history
            if datetime.fromisoformat(run['timestamp']).timestamp() > cutoff_date
        ]
        
        total_cost = sum(run['total_cost'] for run in recent_runs)
        
        # Log cost summary request
        log_counter("eval_cost_summary_requested", labels={
            "period_days": str(days)
        })
        log_histogram("eval_cost_summary_total", total_cost, labels={
            "period_days": str(days)
        })
        log_histogram("eval_cost_summary_runs", len(recent_runs), labels={
            "period_days": str(days)
        })
        
        # Group by provider
        provider_costs = {}
        for run in recent_runs:
            provider = run.get('provider', 'unknown')
            provider_costs[provider] = provider_costs.get(provider, 0.0) + run['total_cost']
        
        return {
            "period_days": days,
            "total_cost": total_cost,
            "num_runs": len(recent_runs),
            "average_cost_per_run": total_cost / len(recent_runs) if recent_runs else 0,
            "provider_breakdown": provider_costs,
            "most_expensive_run": max(recent_runs, key=lambda r: r['total_cost']) if recent_runs else None,
            "cheapest_run": min(recent_runs, key=lambda r: r['total_cost']) if recent_runs else None
        }
    
    def format_cost_display(self, cost: float) -> str:
        """Format cost for display."""
        if cost == 0:
            return "Free"
        elif cost < 0.01:
            return f"<$0.01"
        elif cost < 1:
            return f"${cost:.3f}"
        else:
            return f"${cost:.2f}"
    
    def get_provider_recommendations(self, task_type: str, budget: float) -> List[Dict[str, Any]]:
        """Get provider recommendations based on task type and budget."""
        recommendations = []
        
        # Define task-specific recommendations
        task_profiles = {
            "simple_qa": ["groq/llama3-8b", "openai/gpt-3.5-turbo", "anthropic/claude-3-haiku"],
            "complex_reasoning": ["openai/gpt-4-turbo", "anthropic/claude-3-opus", "google/gemini-pro"],
            "code_generation": ["openai/gpt-4", "anthropic/claude-3-sonnet", "groq/mixtral-8x7b"],
            "safety_evaluation": ["anthropic/claude-3-opus", "openai/gpt-4", "cohere/command-r-plus"]
        }
        
        recommended_models = task_profiles.get(task_type, task_profiles["simple_qa"])
        
        for model_key in recommended_models:
            if model_key in self.pricing:
                pricing = self.pricing[model_key]
                
                # Estimate how many samples can be run with budget
                avg_cost_per_sample = pricing.calculate_cost(500, 200)  # Average tokens
                max_samples = int(budget / avg_cost_per_sample) if avg_cost_per_sample > 0 else float('inf')
                
                recommendations.append({
                    "provider": pricing.provider,
                    "model_id": pricing.model_id,
                    "estimated_samples_in_budget": max_samples,
                    "cost_per_sample": avg_cost_per_sample,
                    "quality_tier": self._get_quality_tier(model_key)
                })
        
        return sorted(recommendations, key=lambda x: x['cost_per_sample'])
    
    def _get_quality_tier(self, model_key: str) -> str:
        """Get quality tier for a model."""
        if any(term in model_key for term in ["gpt-4", "opus", "command-r-plus"]):
            return "premium"
        elif any(term in model_key for term in ["gpt-3.5", "claude-3-sonnet", "gemini-pro"]):
            return "balanced"
        else:
            return "efficient"
    
    def _get_cost_tier(self, cost: float) -> str:
        """Get cost tier for a run."""
        if cost == 0:
            return "free"
        elif cost < 0.01:
            return "minimal"
        elif cost < 0.1:
            return "low"
        elif cost < 1.0:
            return "medium"
        else:
            return "high"

# Utility functions for token counting (approximate)
def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars per token average)."""
    return len(text) // 4

def count_messages_tokens(messages: List[Dict[str, str]]) -> int:
    """Count tokens in a message list."""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get('content', ''))
        total += 4  # Role tokens
    return total