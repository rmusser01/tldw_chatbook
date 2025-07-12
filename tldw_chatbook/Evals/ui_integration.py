# ui_integration.py
# Description: UI Integration helpers for evaluation system
#
"""
UI Integration helpers for evaluation system
===========================================

Provides helper functions for connecting the evaluation backend to the UI:
- Format conversion for display
- Progress tracking utilities
- State management helpers
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta

def format_model_item(model: Dict[str, Any]) -> str:
    """Format model data for display."""
    return f"{model['name']} ({model['provider']})"

def format_run_data(run: Dict[str, Any]) -> tuple:
    """Format run data for table display."""
    return (
        run['name'],
        run['model_name'],
        run['task_name'],
        f"{run.get('score', 0):.2%}",
        run['status'],
        format_duration(run.get('duration', 0))
    )

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def create_progress_message(completed: int, total: int, rate: float) -> str:
    """Create progress status message."""
    percent = (completed / total * 100) if total > 0 else 0
    eta = ((total - completed) / rate) if rate > 0 else 0
    return f"{percent:.1f}% complete | {completed}/{total} samples | ETA: {format_duration(eta)}"

def format_task_item(task: Dict[str, Any]) -> str:
    """Format task data for display."""
    return f"{task['name']} - {task.get('description', 'No description')}"

def format_dataset_item(dataset: Dict[str, Any]) -> str:
    """Format dataset data for display."""
    sample_count = dataset.get('sample_count', '?')
    return f"{dataset['name']} ({dataset.get('format', 'unknown')}) - {sample_count} samples"

def calculate_success_rate(results: list) -> float:
    """Calculate success rate from results."""
    if not results:
        return 0.0
    successful = sum(1 for r in results if not r.get('error_info'))
    return (successful / len(results)) * 100

def format_error_message(error_info: Dict[str, Any]) -> str:
    """Format error information for display."""
    error_category = error_info.get('error_category', 'unknown')
    error_message = error_info.get('error_message', 'Unknown error')
    suggested_action = error_info.get('suggested_action', 'Check logs for details')
    
    return f"Error ({error_category}): {error_message}\nSuggestion: {suggested_action}"

def format_metrics_display(metrics: Dict[str, float]) -> list:
    """Format metrics dictionary for display in a list."""
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            if 0 <= value <= 1:
                # Assume it's a percentage
                formatted.append(f"{key}: {value:.2%}")
            else:
                # Regular float
                formatted.append(f"{key}: {value:.2f}")
        else:
            formatted.append(f"{key}: {value}")
    return formatted

def get_status_emoji(status: str) -> str:
    """Get emoji for status display."""
    status_emojis = {
        'completed': '✅',
        'failed': '❌',
        'running': '⏳',
        'cancelled': '🚫',
        'pending': '⏸️'
    }
    return status_emojis.get(status.lower(), '❓')

def format_run_summary(run: Dict[str, Any]) -> str:
    """Format a run summary for display."""
    status_emoji = get_status_emoji(run['status'])
    name = run['name']
    status = run['status']
    task_name = run.get('task_name', 'Unknown')
    model_name = run.get('model_name', 'Unknown')
    samples = run.get('completed_samples', 0)
    
    summary = f"{status_emoji} {name} - {status}\n"
    summary += f"    Task: {task_name}\n"
    summary += f"    Model: {model_name}\n"
    summary += f"    Samples: {samples}"
    
    if run.get('error_info'):
        summary += f"\n    Error: {run['error_info'].get('error_message', 'Unknown error')}"
    
    return summary

def group_runs_by_status(runs: list) -> Dict[str, list]:
    """Group runs by their status."""
    grouped = {
        'completed': [],
        'failed': [],
        'running': [],
        'cancelled': [],
        'pending': []
    }
    
    for run in runs:
        status = run.get('status', 'unknown').lower()
        if status in grouped:
            grouped[status].append(run)
        else:
            grouped.setdefault('other', []).append(run)
    
    return grouped

def calculate_aggregate_stats(runs: list) -> Dict[str, Any]:
    """Calculate aggregate statistics from runs."""
    if not runs:
        return {
            'total_runs': 0,
            'completed': 0,
            'failed': 0,
            'average_success_rate': 0.0,
            'total_samples': 0,
            'total_duration': 0.0
        }
    
    grouped = group_runs_by_status(runs)
    
    total_samples = sum(run.get('completed_samples', 0) for run in runs)
    total_duration = sum(run.get('duration', 0) for run in runs)
    
    # Calculate average success rate for completed runs
    completed_runs = grouped['completed']
    if completed_runs:
        success_rates = [run.get('success_rate', 0) for run in completed_runs]
        avg_success_rate = sum(success_rates) / len(success_rates)
    else:
        avg_success_rate = 0.0
    
    return {
        'total_runs': len(runs),
        'completed': len(grouped['completed']),
        'failed': len(grouped['failed']),
        'running': len(grouped['running']),
        'average_success_rate': avg_success_rate,
        'total_samples': total_samples,
        'total_duration': total_duration
    }