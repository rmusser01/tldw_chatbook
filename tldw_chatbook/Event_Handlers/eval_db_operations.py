# eval_db_operations.py
# Description: Database operations for the evaluation system
#
"""
Evaluation Database Operations
-----------------------------

Helper functions for interacting with the evaluation database.
Provides async wrappers and data formatting for UI consumption.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

from ..DB.Evals_DB import EvalsDB, EvalsDBError


class EvalDBOperations:
    """Handles database operations for the evaluation system."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize with database path."""
        if db_path is None:
            # Use default evals database location
            db_path = Path.home() / ".config" / "tldw_cli" / "evals.db"
        
        self.db = EvalsDB(db_path)
    
    async def get_recent_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent evaluation runs."""
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            runs = await loop.run_in_executor(
                None, 
                lambda: self.db.list_runs(limit=limit)
            )
            
            # Format for UI display
            formatted_runs = []
            for run in runs:
                formatted_runs.append({
                    "id": run['run_id'],
                    "model": run.get('model_id', 'Unknown'),
                    "task": run.get('task_id', 'Unknown'),
                    "status": run.get('status', 'unknown'),
                    "cost": f"${run.get('total_cost', 0):.2f}",
                    "started": run.get('started_at', '')
                })
            
            return formatted_runs
            
        except Exception as e:
            logger.error(f"Failed to get recent runs: {e}")
            return []
    
    async def get_datasets(self) -> List[Dict[str, Any]]:
        """Get available datasets."""
        try:
            loop = asyncio.get_event_loop()
            datasets = await loop.run_in_executor(
                None,
                lambda: self.db.list_datasets(limit=100)
            )
            
            # Format for UI display
            formatted_datasets = []
            for dataset in datasets:
                formatted_datasets.append({
                    "id": dataset['dataset_id'],
                    "name": dataset.get('name', 'Unknown'),
                    "type": dataset.get('task_type', 'Unknown'),
                    "size": dataset.get('num_samples', 0),
                    "format": dataset.get('format', 'json'),
                    "created": dataset.get('created_at', '')
                })
            
            return formatted_datasets
            
        except Exception as e:
            logger.error(f"Failed to get datasets: {e}")
            return []
    
    async def get_models(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get available models, optionally filtered by provider."""
        try:
            loop = asyncio.get_event_loop()
            models = await loop.run_in_executor(
                None,
                lambda: self.db.list_models(provider=provider, limit=100)
            )
            
            # Format for UI display
            formatted_models = []
            for model in models:
                formatted_models.append({
                    "id": model['model_id'],
                    "name": model.get('name', model['model_id']),
                    "provider": model.get('provider', 'Unknown'),
                    "context_size": model.get('context_window', 0),
                    "input_cost": model.get('input_cost_per_1k', 0),
                    "output_cost": model.get('output_cost_per_1k', 0),
                    "capabilities": model.get('capabilities', {})
                })
            
            return formatted_models
            
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return []
    
    async def get_run_details(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific run."""
        try:
            loop = asyncio.get_event_loop()
            
            # Get run info
            run = await loop.run_in_executor(
                None,
                lambda: self.db.get_run(run_id)
            )
            
            if not run:
                return None
            
            # Get metrics
            metrics = await loop.run_in_executor(
                None,
                lambda: self.db.get_run_metrics(run_id)
            )
            
            # Format for UI display
            return {
                "run_id": run['run_id'],
                "model": run.get('model_id', 'Unknown'),
                "task": run.get('task_id', 'Unknown'),
                "dataset": run.get('dataset_id', 'Unknown'),
                "status": run.get('status', 'unknown'),
                "started_at": run.get('started_at', ''),
                "completed_at": run.get('completed_at', ''),
                "duration": self._calculate_duration(
                    run.get('started_at'),
                    run.get('completed_at')
                ),
                "total_cost": run.get('total_cost', 0),
                "total_samples": run.get('total_samples', 0),
                "completed_samples": run.get('completed_samples', 0),
                "metrics": metrics,
                "config": run.get('config', {}),
                "error": run.get('error_message', None)
            }
            
        except Exception as e:
            logger.error(f"Failed to get run details: {e}")
            return None
    
    async def create_run(
        self,
        task_id: str,
        model_id: str,
        dataset_id: str,
        config: Dict[str, Any]
    ) -> str:
        """Create a new evaluation run."""
        try:
            loop = asyncio.get_event_loop()
            run_id = await loop.run_in_executor(
                None,
                lambda: self.db.create_run(
                    task_id=task_id,
                    model_id=model_id,
                    dataset_id=dataset_id,
                    config=config
                )
            )
            
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to create run: {e}")
            raise
    
    async def update_run_status(
        self,
        run_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> None:
        """Update the status of a run."""
        try:
            loop = asyncio.get_event_loop()
            
            update_data = {"status": status}
            if error_message:
                update_data["error_message"] = error_message
            if status == "completed":
                update_data["completed_at"] = datetime.now(timezone.utc).isoformat()
            
            await loop.run_in_executor(
                None,
                lambda: self.db.update_run(run_id, **update_data)
            )
            
        except Exception as e:
            logger.error(f"Failed to update run status: {e}")
            raise
    
    async def get_task_types(self) -> List[str]:
        """Get available task types."""
        try:
            loop = asyncio.get_event_loop()
            tasks = await loop.run_in_executor(
                None,
                lambda: self.db.list_tasks(limit=100)
            )
            
            # Extract unique task types
            task_types = list(set(
                task.get('task_type', 'custom')
                for task in tasks
            ))
            
            return sorted(task_types)
            
        except Exception as e:
            logger.error(f"Failed to get task types: {e}")
            return ["simple_qa", "complex_qa", "coding", "summarization", "translation", "custom"]
    
    async def create_dataset(
        self,
        name: str,
        task_type: str,
        data_path: str,
        format: str = "json",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new dataset entry."""
        try:
            loop = asyncio.get_event_loop()
            dataset_id = await loop.run_in_executor(
                None,
                lambda: self.db.create_dataset(
                    name=name,
                    task_type=task_type,
                    data_path=data_path,
                    format=format,
                    metadata=metadata or {}
                )
            )
            
            return dataset_id
            
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            raise
    
    async def search_runs(
        self,
        query: str,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search evaluation runs."""
        try:
            loop = asyncio.get_event_loop()
            
            # Simple search implementation
            # In a real implementation, this would use FTS or more sophisticated search
            all_runs = await loop.run_in_executor(
                None,
                lambda: self.db.list_runs(status=status, limit=limit)
            )
            
            # Filter by query
            query_lower = query.lower()
            filtered_runs = []
            
            for run in all_runs:
                if (query_lower in run.get('run_id', '').lower() or
                    query_lower in run.get('model_id', '').lower() or
                    query_lower in run.get('task_id', '').lower()):
                    filtered_runs.append({
                        "id": run['run_id'],
                        "model": run.get('model_id', 'Unknown'),
                        "task": run.get('task_id', 'Unknown'),
                        "status": run.get('status', 'unknown'),
                        "cost": f"${run.get('total_cost', 0):.2f}",
                        "started": run.get('started_at', '')
                    })
            
            return filtered_runs
            
        except Exception as e:
            logger.error(f"Failed to search runs: {e}")
            return []
    
    def _calculate_duration(self, started_at: str, completed_at: str) -> str:
        """Calculate duration between two timestamps."""
        try:
            if not started_at or not completed_at:
                return "N/A"
            
            start = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
            end = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
            duration = end - start
            
            # Format duration
            total_seconds = int(duration.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            
            if hours > 0:
                return f"{hours}h {minutes}m"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"
                
        except Exception:
            return "N/A"


# Singleton instance for easy access
_db_ops = None

def get_eval_db_ops() -> EvalDBOperations:
    """Get singleton instance of evaluation database operations."""
    global _db_ops
    if _db_ops is None:
        _db_ops = EvalDBOperations()
    return _db_ops