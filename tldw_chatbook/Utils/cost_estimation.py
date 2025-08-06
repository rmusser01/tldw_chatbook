# tldw_chatbook/Utils/cost_estimation.py
"""
Cost estimation utilities for evaluation runs
"""

# Model pricing per 1K tokens (input + output averaged)
MODEL_COSTS = {
    # OpenAI
    "gpt-4": 0.03,
    "gpt-4-turbo": 0.01,
    "gpt-3.5-turbo": 0.002,
    "gpt-3.5-turbo-16k": 0.003,
    
    # Anthropic
    "claude-3-opus": 0.015,
    "claude-3-sonnet": 0.003,
    "claude-3-haiku": 0.00025,
    "claude-2.1": 0.008,
    "claude-2": 0.008,
    "claude-instant": 0.0008,
    
    # Google
    "gemini-pro": 0.001,
    "gemini-pro-vision": 0.002,
    "palm-2": 0.002,
    
    # Cohere
    "command": 0.002,
    "command-light": 0.0006,
    
    # Default for unknown models
    "default": 0.01
}


def estimate_evaluation_cost(model: str, sample_count: int, avg_tokens_per_sample: int = 150) -> float:
    """
    Estimate the cost of running an evaluation
    
    Args:
        model: Model name
        sample_count: Number of samples to evaluate
        avg_tokens_per_sample: Average tokens per sample (input + output)
    
    Returns:
        Estimated cost in USD
    """
    # Get cost per 1K tokens
    cost_per_1k = MODEL_COSTS.get(model, MODEL_COSTS["default"])
    
    # Calculate total tokens
    total_tokens = sample_count * avg_tokens_per_sample
    
    # Calculate cost
    cost = (total_tokens / 1000) * cost_per_1k
    
    return round(cost, 2)


def get_model_cost_per_1k_tokens(model: str) -> float:
    """Get the cost per 1K tokens for a model"""
    return MODEL_COSTS.get(model, MODEL_COSTS["default"])


def calculate_token_cost(model: str, token_count: int) -> float:
    """Calculate cost for a specific number of tokens"""
    cost_per_1k = get_model_cost_per_1k_tokens(model)
    return (token_count / 1000) * cost_per_1k