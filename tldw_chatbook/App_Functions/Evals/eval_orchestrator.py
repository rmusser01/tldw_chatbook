# eval_orchestrator.py
# Description: Orchestrates evaluation runs and manages results storage
#
"""
Evaluation Orchestrator
----------------------

Coordinates the complete evaluation pipeline:
1. Task and model configuration validation
2. Evaluation execution with progress tracking
3. Results storage and aggregation
4. Error handling and recovery

This is the main entry point for running evaluations.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path

from loguru import logger

from .task_loader import TaskLoader, TaskConfig
from .eval_runner import EvalRunner, EvalResult
from ...DB.Evals_DB import EvalsDB

class EvaluationOrchestrator:
    """Orchestrates evaluation runs from start to finish."""
    
    def __init__(self, db_path: str = None, client_id: str = "eval_orchestrator"):
        """
        Initialize the evaluation orchestrator.
        
        Args:
            db_path: Path to evaluation database (defaults to user data directory)
            client_id: Client identifier for audit trail
        """
        if db_path is None:
            # Use default path in user data directory
            from ...config import load_settings
            settings = load_settings()
            user_data_dir = Path(settings.get('user_data_dir', '~/.local/share/tldw_cli')).expanduser()
            db_path = user_data_dir / 'evals.db'
        
        self.db = EvalsDB(db_path=str(db_path), client_id=client_id)
        self.task_loader = TaskLoader()
        
        logger.info(f"EvaluationOrchestrator initialized with DB: {db_path}")
    
    async def create_task_from_file(self, task_file_path: str, format_type: str = 'auto') -> str:
        """
        Create a task from a file and store it in the database.
        
        Args:
            task_file_path: Path to task configuration file
            format_type: Format type ('eleuther', 'custom', 'auto')
            
        Returns:
            Task ID
        """
        logger.info(f"Creating task from file: {task_file_path}")
        
        # Load task configuration
        task_config = self.task_loader.load_task(task_file_path, format_type)
        
        # Validate task
        validation_issues = self.task_loader.validate_task(task_config)
        if validation_issues:
            raise ValueError(f"Task validation failed: {validation_issues}")
        
        # Store in database
        task_id = self.db.create_task(
            name=task_config.name,
            description=task_config.description,
            task_type=task_config.task_type,
            config_format=task_config.metadata.get('format', 'custom'),
            config_data=task_config.__dict__,
            dataset_id=None  # Will be handled separately if needed
        )
        
        logger.info(f"Created task: {task_config.name} ({task_id})")
        return task_id
    
    def create_model_config(self, name: str, provider: str, model_id: str, 
                          config: Dict[str, Any] = None) -> str:
        """
        Create a model configuration and store it in the database.
        
        Args:
            name: Human-readable name for the model
            provider: Provider name (openai, anthropic, etc.)
            model_id: Model identifier
            config: Additional configuration parameters
            
        Returns:
            Model configuration ID
        """
        logger.info(f"Creating model config: {name} ({provider}/{model_id})")
        
        model_config_id = self.db.create_model(
            name=name,
            provider=provider,
            model_id=model_id,
            config=config or {}
        )
        
        logger.info(f"Created model config: {name} ({model_config_id})")
        return model_config_id
    
    async def run_evaluation(self, 
                           task_id: str, 
                           model_id: str, 
                           run_name: str = None,
                           max_samples: int = None,
                           config_overrides: Dict[str, Any] = None,
                           progress_callback: Callable = None) -> str:
        """
        Run a complete evaluation.
        
        Args:
            task_id: ID of the task to run
            model_id: ID of the model configuration to use
            run_name: Name for this evaluation run
            max_samples: Maximum number of samples to evaluate (None for all)
            config_overrides: Override task configuration parameters
            progress_callback: Function to call with progress updates
            
        Returns:
            Evaluation run ID
        """
        # Load task and model configurations
        task_data = self.db.get_task(task_id)
        model_data = self.db.get_model(model_id)
        
        if not task_data:
            raise ValueError(f"Task not found: {task_id}")
        if not model_data:
            raise ValueError(f"Model not found: {model_id}")
        
        # Create TaskConfig from database data
        task_config_dict = task_data['config_data'].copy()
        if config_overrides:
            task_config_dict.update(config_overrides)
        
        task_config = TaskConfig(**task_config_dict)
        
        # Generate run name if not provided
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{task_data['name']}_{model_data['name']}_{timestamp}"
        
        logger.info(f"Starting evaluation run: {run_name}")
        
        # Create evaluation run record
        run_id = self.db.create_run(
            name=run_name,
            task_id=task_id,
            model_id=model_id,
            config_overrides=config_overrides or {}
        )
        
        try:
            # Update run status to running
            self.db.update_run_status(run_id, 'running')
            
            # Create evaluation runner
            eval_runner = EvalRunner(task_config, model_data)
            
            # Create progress callback wrapper
            def progress_wrapper(completed: int, total: int, result: EvalResult):
                # Store individual result
                self.db.store_result(
                    run_id=run_id,
                    sample_id=result.sample_id,
                    input_data={'input': result.input_text},
                    actual_output=result.actual_output,
                    expected_output=result.expected_output,
                    logprobs=result.logprobs,
                    metrics=result.metrics,
                    metadata=result.metadata
                )
                
                # Call user progress callback if provided
                if progress_callback:
                    progress_callback(completed, total, result)
                
                # Log progress
                if completed % 10 == 0 or completed == total:
                    logger.info(f"Evaluation progress: {completed}/{total} samples completed")
            
            # Run evaluation
            results = await eval_runner.run_evaluation(
                max_samples=max_samples,
                progress_callback=progress_wrapper
            )
            
            # Calculate and store aggregate metrics
            aggregate_metrics = eval_runner.calculate_aggregate_metrics(results)
            
            # Convert to required format for database
            db_metrics = {}
            for metric_name, value in aggregate_metrics.items():
                metric_type = 'custom'
                if 'accuracy' in metric_name:
                    metric_type = 'accuracy'
                elif 'f1' in metric_name:
                    metric_type = 'f1'
                elif 'bleu' in metric_name:
                    metric_type = 'bleu'
                
                db_metrics[metric_name] = (value, metric_type)
            
            self.db.store_run_metrics(run_id, db_metrics)
            
            # Update run status to completed
            self.db.update_run_status(run_id, 'completed')
            
            logger.info(f"Evaluation completed successfully: {run_name} ({run_id})")
            logger.info(f"Results summary: {aggregate_metrics}")
            
            return run_id
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            
            # Update run status to failed
            self.db.update_run_status(run_id, 'failed', str(e))
            
            raise
    
    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """Get a comprehensive summary of an evaluation run."""
        run_data = self.db.get_run(run_id)
        if not run_data:
            raise ValueError(f"Run not found: {run_id}")
        
        metrics = self.db.get_run_metrics(run_id)
        
        summary = {
            'run_info': run_data,
            'metrics': metrics,
            'sample_count': run_data['completed_samples'],
            'status': run_data['status']
        }
        
        # Calculate duration if completed
        if run_data['start_time'] and run_data['end_time']:
            start = datetime.fromisoformat(run_data['start_time'])
            end = datetime.fromisoformat(run_data['end_time'])
            summary['duration_seconds'] = (end - start).total_seconds()
        
        return summary
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare metrics across multiple evaluation runs."""
        return self.db.compare_runs(run_ids)
    
    def list_tasks(self, **kwargs) -> List[Dict[str, Any]]:
        """List available tasks."""
        return self.db.list_tasks(**kwargs)
    
    def list_models(self, **kwargs) -> List[Dict[str, Any]]:
        """List available model configurations."""
        return self.db.list_models(**kwargs)
    
    def list_runs(self, **kwargs) -> List[Dict[str, Any]]:
        """List evaluation runs."""
        return self.db.list_runs(**kwargs)
    
    def search_tasks(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search tasks using full-text search."""
        return self.db.search_tasks(query, **kwargs)
    
    def get_task_details(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a task."""
        return self.db.get_task(task_id)
    
    def get_model_details(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model configuration."""
        return self.db.get_model(model_id)
    
    def get_run_results(self, run_id: str, **kwargs) -> List[Dict[str, Any]]:
        """Get detailed results for a run."""
        return self.db.get_run_results(run_id, **kwargs)
    
    def export_results(self, run_id: str, output_path: str, format_type: str = 'json'):
        """Export evaluation results to file."""
        run_summary = self.get_run_summary(run_id)
        results = self.get_run_results(run_id)
        
        export_data = {
            'run_summary': run_summary,
            'results': results,
            'export_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        output_path = Path(output_path)
        
        if format_type.lower() == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format_type.lower() == 'csv':
            import csv
            
            # Export results as CSV
            with open(output_path, 'w', newline='') as f:
                if not results:
                    return
                
                fieldnames = ['sample_id', 'input_text', 'expected_output', 'actual_output'] + \
                           list(results[0]['metrics'].keys())
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    row = {
                        'sample_id': result['sample_id'],
                        'input_text': result['input_data'].get('input', ''),
                        'expected_output': result['expected_output'],
                        'actual_output': result['actual_output']
                    }
                    row.update(result['metrics'])
                    writer.writerow(row)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
        
        logger.info(f"Results exported to: {output_path}")
    
    def create_dataset_from_file(self, name: str, file_path: str, description: str = None) -> str:
        """Create a dataset record from a local file."""
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"Dataset file not found: {file_path}")
        
        # Determine format from file extension
        format_type = 'json'
        if path.suffix.lower() in ['.csv', '.tsv']:
            format_type = 'csv'
        elif path.suffix.lower() in ['.yaml', '.yml']:
            format_type = 'yaml'
        
        dataset_id = self.db.create_dataset(
            name=name,
            description=description or f"Dataset from {path.name}",
            format=format_type,
            source_path=str(path),
            metadata={'file_size': path.stat().st_size, 'created_from': 'local_file'}
        )
        
        logger.info(f"Created dataset: {name} ({dataset_id})")
        return dataset_id
    
    def create_task_from_template(self, template_name: str, 
                                output_dir: str = None, **kwargs) -> Tuple[str, str]:
        """
        Create a task and sample dataset from a template.
        
        Args:
            template_name: Name of the evaluation template
            output_dir: Directory to save template files (optional)
            **kwargs: Override parameters for the template
            
        Returns:
            Tuple of (task_id, dataset_id)
        """
        logger.info(f"Creating task from template: {template_name}")
        
        # Load template and create task config
        task_config = self.task_loader.create_task_from_template(template_name, **kwargs)
        
        # Create task in database
        task_id = self.db.create_task(
            name=task_config.name,
            description=task_config.description,
            task_type=task_config.task_type,
            config_format='template',
            config_data=task_config.__dict__
        )
        
        # Create sample dataset if template has sample problems
        dataset_id = None
        try:
            from .eval_templates import get_eval_templates
            template_manager = get_eval_templates()
            
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Export sample dataset
                dataset_path = output_dir / f"{template_name}_samples.json"
                num_samples = template_manager.create_sample_dataset(
                    template_name, str(dataset_path), num_samples=20
                )
                
                # Create dataset record
                dataset_id = self.create_dataset_from_file(
                    name=f"{task_config.name} Samples",
                    file_path=str(dataset_path),
                    description=f"Sample dataset for {task_config.name} with {num_samples} examples"
                )
                
                logger.info(f"Created sample dataset with {num_samples} examples: {dataset_path}")
            
        except Exception as e:
            logger.warning(f"Could not create sample dataset for template {template_name}: {e}")
        
        logger.info(f"Created task from template: {task_config.name} ({task_id})")
        return task_id, dataset_id
    
    def list_available_templates(self) -> List[Dict[str, Any]]:
        """List all available evaluation templates."""
        return self.task_loader.list_available_templates()
    
    def get_templates_by_category(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get evaluation templates organized by category."""
        templates = self.list_available_templates()
        categories = {}
        
        for template in templates:
            category = template.get('category', 'general')
            if category not in categories:
                categories[category] = []
            categories[category].append(template)
        
        return categories
    
    def close(self):
        """Close database connections."""
        self.db.close()

# Convenience functions for common operations

async def quick_eval(task_file: str, provider: str, model: str, 
                    max_samples: int = 100, output_dir: str = None) -> Dict[str, Any]:
    """
    Quick evaluation helper function.
    
    Args:
        task_file: Path to task configuration file
        provider: LLM provider name
        model: Model identifier
        max_samples: Maximum samples to evaluate
        output_dir: Directory to save results (optional)
        
    Returns:
        Evaluation summary
    """
    orchestrator = EvaluationOrchestrator()
    
    try:
        # Create task
        task_id = await orchestrator.create_task_from_file(task_file)
        
        # Create model config
        model_id = orchestrator.create_model_config(
            name=f"{provider}_{model}",
            provider=provider,
            model_id=model
        )
        
        # Run evaluation
        run_name = f"quick_eval_{int(time.time())}"
        run_id = await orchestrator.run_evaluation(
            task_id=task_id,
            model_id=model_id,
            run_name=run_name,
            max_samples=max_samples
        )
        
        # Get summary
        summary = orchestrator.get_run_summary(run_id)
        
        # Export results if output directory specified
        if output_dir:
            output_path = Path(output_dir) / f"{run_name}_results.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            orchestrator.export_results(run_id, str(output_path))
            summary['exported_to'] = str(output_path)
        
        return summary
        
    finally:
        orchestrator.close()

def create_task_template(task_type: str, name: str, output_path: str):
    """Create a task template file."""
    loader = TaskLoader()
    
    task_config = loader.create_task_from_template(
        template_name=task_type,
        name=name,
        description=f"Template for {task_type} evaluation"
    )
    
    loader.export_task(task_config, output_path, format_type='custom')
    logger.info(f"Created task template: {output_path}")