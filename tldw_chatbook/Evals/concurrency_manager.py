# concurrency_manager.py
# Description: Manages concurrent evaluation runs to prevent conflicts
#
"""
Concurrency Manager for Evaluations
-----------------------------------

Manages concurrent evaluation runs to ensure no conflicts
and proper resource utilization.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from loguru import logger

from .eval_errors import ValidationError, ErrorContext, ErrorCategory, ErrorSeverity


class ConcurrentRunManager:
    """Manages concurrent evaluation runs to prevent conflicts."""
    
    def __init__(self):
        self._active_runs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def register_run(self, run_id: str, task_id: str, model_id: str) -> bool:
        """Register a new run, checking for conflicts."""
        async with self._lock:
            # Check for existing runs with same task/model
            for existing_id, run_info in self._active_runs.items():
                if (run_info['task_id'] == task_id and 
                    run_info['model_id'] == model_id and
                    existing_id != run_id):
                    raise ValidationError(ErrorContext(
                        category=ErrorCategory.VALIDATION,
                        severity=ErrorSeverity.WARNING,
                        message=f"An evaluation is already running for this task and model combination",
                        details=f"Existing run: {existing_id}",
                        suggestion="Wait for the existing run to complete or cancel it",
                        is_retryable=True
                    ))
            
            # Register the new run
            self._active_runs[run_id] = {
                'task_id': task_id,
                'model_id': model_id,
                'start_time': datetime.now(timezone.utc),
                'status': 'running'
            }
            logger.info(f"Registered run {run_id} for task {task_id} with model {model_id}")
            return True
    
    async def unregister_run(self, run_id: str):
        """Remove a run from active tracking."""
        async with self._lock:
            if run_id in self._active_runs:
                del self._active_runs[run_id]
                logger.info(f"Unregistered run {run_id}")
    
    async def get_active_runs(self) -> List[Dict[str, Any]]:
        """Get list of active runs."""
        async with self._lock:
            return [
                {'run_id': run_id, **info}
                for run_id, info in self._active_runs.items()
            ]
    
    async def is_task_running(self, task_id: str) -> bool:
        """Check if a specific task is currently running."""
        async with self._lock:
            return any(
                run['task_id'] == task_id 
                for run in self._active_runs.values()
            )
    
    async def cancel_run(self, run_id: str) -> bool:
        """Mark a run as cancelled."""
        async with self._lock:
            if run_id in self._active_runs:
                self._active_runs[run_id]['status'] = 'cancelled'
                return True
            return False