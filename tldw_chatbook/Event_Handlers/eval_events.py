# eval_events.py
# Description: Event handlers for evaluation functionality
#
"""
Evaluation Event Handlers
-------------------------

Handles all evaluation-related events:
- Task upload and creation
- Model configuration
- Evaluation execution
- Results management
- Export functionality
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from loguru import logger

from textual.widgets import Button, Input
from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram

if TYPE_CHECKING:
    from ..app import TldwCli

# Import evaluation components
from tldw_chatbook.Evals import EvaluationOrchestrator
from tldw_chatbook.Evals import TaskLoader, TaskLoadError
from ..Widgets.eval_config_dialogs import ModelConfigDialog, TaskConfigDialog, RunConfigDialog

# File picker dialogs
from ..Widgets.file_picker_dialog import TaskFilePickerDialog, DatasetFilePickerDialog, ExportFilePickerDialog

# Template selector - might not exist
try:
    from ..Widgets.template_selector import TemplateSelectorDialog
except ImportError:
    TemplateSelectorDialog = None

# Global orchestrator instance (will be initialized on first use)
_orchestrator: Optional[EvaluationOrchestrator] = None

def get_orchestrator() -> EvaluationOrchestrator:
    """Get or create the global evaluation orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        # Initialize with proper database path
        from pathlib import Path
        from tldw_chatbook.config import load_settings
        
        settings = load_settings()
        user_data_dir = Path(settings.get('user_data_dir', '~/.local/share/tldw_cli')).expanduser()
        
        # Use user ID from config instead of hardcoded 'default_user'
        user_id = settings.get('user_id', settings.get('username', 'default_user'))
        db_path = user_data_dir / user_id / 'evals.db'
        
        # Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        _orchestrator = EvaluationOrchestrator(db_path=str(db_path))
    return _orchestrator

# === TASK MANAGEMENT EVENTS ===

async def handle_upload_task(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle task file upload."""
    logger.info("Starting task file upload")
    
    # Log UI interaction
    log_counter("eval_ui_task_upload_clicked")
    
    def on_file_selected(file_path: Optional[str]):
        if file_path:
            asyncio.create_task(_process_task_upload(app, file_path))
    
    # Open file picker dialog
    dialog = TaskFilePickerDialog(callback=on_file_selected)
    app.push_screen(dialog)

async def _process_task_upload(app: 'TldwCli', file_path: str):
    """Process the uploaded task file."""
    start_time = time.time()
    try:
        app.notify(f"Processing task file: {Path(file_path).name}", severity="information")
        
        orchestrator = get_orchestrator()
        task_id = await orchestrator.create_task_from_file(file_path)
        
        # Log successful task upload
        duration = time.time() - start_time
        log_histogram("eval_ui_task_upload_duration", duration, labels={
            "status": "success"
        })
        log_counter("eval_ui_task_upload_success")
        
        app.notify(f"Task created successfully: {task_id}", severity="information")
        
        # Update UI
        await refresh_tasks_list(app)
        
    except TaskLoadError as e:
        duration = time.time() - start_time
        logger.error(f"Task load error: {e}")
        
        # Log task load error
        log_histogram("eval_ui_task_upload_duration", duration, labels={
            "status": "error"
        })
        log_counter("eval_ui_task_upload_error", labels={
            "error_type": "TaskLoadError"
        })
        
        app.notify(f"Failed to load task: {str(e)}", severity="error")
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error processing task upload: {e}")
        
        # Log generic error
        log_histogram("eval_ui_task_upload_duration", duration, labels={
            "status": "error"
        })
        log_counter("eval_ui_task_upload_error", labels={
            "error_type": type(e).__name__
        })
        
        app.notify(f"Error processing task: {str(e)}", severity="error")

async def handle_create_task(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle new task creation."""
    logger.info("Creating new task")
    
    # Log UI interaction
    log_counter("eval_ui_create_task_clicked")
    
    def on_config_ready(config: Optional[Dict[str, Any]]):
        if config:
            asyncio.create_task(_create_custom_task(app, config))
    
    # Open task configuration dialog
    dialog = TaskConfigDialog(callback=on_config_ready)
    app.push_screen(dialog)

async def _create_custom_task(app: 'TldwCli', config: Dict[str, Any]):
    """Create a custom task from configuration."""
    start_time = time.time()
    try:
        app.notify("Creating custom task...", severity="information")
        
        orchestrator = get_orchestrator()
        
        # Create task using orchestrator
        task_id = orchestrator.db.create_task(
            name=config['name'],
            description=config.get('description', ''),
            task_type=config['task_type'],
            config_format='custom',
            config_data=config
        )
        
        # Log successful task creation
        duration = time.time() - start_time
        log_histogram("eval_ui_custom_task_creation_duration", duration, labels={
            "task_type": config['task_type'],
            "status": "success"
        })
        log_counter("eval_ui_custom_task_created", labels={
            "task_type": config['task_type']
        })
        
        app.notify(f"Task created successfully: {config['name']}", severity="information")
        
        # Update UI
        await refresh_tasks_list(app)
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error creating custom task: {e}")
        
        # Log error
        log_histogram("eval_ui_custom_task_creation_duration", duration, labels={
            "status": "error"
        })
        log_counter("eval_ui_custom_task_error", labels={
            "error_type": type(e).__name__
        })
        
        app.notify(f"Error creating task: {str(e)}", severity="error")

async def handle_template_selection(app: 'TldwCli', template_name: str) -> None:
    """Handle template selection for task creation."""
    logger.info(f"Creating task from template: {template_name}")
    
    # Log template selection
    log_counter("eval_ui_template_selected", labels={
        "template_name": template_name
    })
    
    try:
        app.notify(f"Creating task from template: {template_name}", severity="information")
        
        orchestrator = get_orchestrator()
        
        # Create task from template
        task_id, dataset_id = orchestrator.create_task_from_template(
            template_name=template_name,
            output_dir=None  # Don't save files by default
        )
        
        # Log successful template usage
        log_counter("eval_ui_template_task_created", labels={
            "template_name": template_name
        })
        
        app.notify(f"Task created from template: {template_name}", severity="information")
        
        # Update UI
        await refresh_tasks_list(app)
        
    except Exception as e:
        logger.error(f"Error creating task from template: {e}")
        
        # Log error
        log_counter("eval_ui_template_task_error", labels={
            "template_name": template_name,
            "error_type": type(e).__name__
        })
        
        app.notify(f"Error creating template task: {str(e)}", severity="error")

# === MODEL MANAGEMENT EVENTS ===

async def handle_add_model(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle adding a new model configuration."""
    logger.info("Adding new model configuration")
    
    # Log UI interaction
    log_counter("eval_ui_add_model_clicked")
    
    def on_config_ready(config: Optional[Dict[str, Any]]):
        if config:
            asyncio.create_task(_save_model_config(app, config))
    
    # Open model configuration dialog
    dialog = ModelConfigDialog(callback=on_config_ready)
    app.push_screen(dialog)

async def _save_model_config(app: 'TldwCli', config: Dict[str, Any]):
    """Save model configuration."""
    try:
        app.notify("Saving model configuration...", severity="information")
        
        orchestrator = get_orchestrator()
        
        # Save model configuration
        model_id = orchestrator.create_model_config(
            name=config['name'],
            provider=config['provider'],
            model_id=config['model_id'],
            config=config
        )
        
        # Log successful model config
        log_counter("eval_ui_model_configured", labels={
            "provider": config['provider'],
            "model_id": config['model_id']
        })
        
        app.notify(f"Model configured: {config['name']}", severity="information")
        
        # Update UI
        await refresh_models_list(app)
        
    except Exception as e:
        logger.error(f"Error saving model config: {e}")
        
        # Log error
        log_counter("eval_ui_model_config_error", labels={
            "error_type": type(e).__name__
        })
        
        app.notify(f"Error saving model: {str(e)}", severity="error")

async def handle_provider_setup(app: 'TldwCli', provider: str) -> None:
    """Handle quick provider setup."""
    logger.info(f"Setting up provider: {provider}")
    
    # Log provider setup
    log_counter("eval_ui_provider_setup_clicked", labels={
        "provider": provider
    })
    
    # Pre-fill configuration based on provider
    provider_configs = {
        'openai': {
            'provider': 'openai',
            'model_id': 'gpt-3.5-turbo',
            'name': 'OpenAI GPT-3.5 Turbo'
        },
        'anthropic': {
            'provider': 'anthropic',
            'model_id': 'claude-3-haiku-20240307',
            'name': 'Anthropic Claude 3 Haiku'
        },
        'cohere': {
            'provider': 'cohere',
            'model_id': 'command-r',
            'name': 'Cohere Command R'
        },
        'groq': {
            'provider': 'groq',
            'model_id': 'llama3-8b-8192',
            'name': 'Groq Llama 3 8B'
        }
    }
    
    config = provider_configs.get(provider.lower(), {})
    
    def on_config_ready(final_config: Optional[Dict[str, Any]]):
        if final_config:
            asyncio.create_task(_save_model_config(app, final_config))
    
    # Open configuration dialog with pre-filled values
    dialog = ModelConfigDialog(callback=on_config_ready, existing_config=config)
    app.push_screen(dialog)

# === EVALUATION EXECUTION EVENTS ===

async def handle_start_evaluation(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle starting an evaluation run."""
    logger.info("Starting evaluation run")
    
    # Log UI interaction
    log_counter("eval_ui_start_evaluation_clicked")
    
    # Get available tasks and models
    orchestrator = get_orchestrator()
    tasks = orchestrator.list_tasks()
    models = orchestrator.list_models()
    
    if not tasks:
        log_counter("eval_ui_start_evaluation_blocked", labels={
            "reason": "no_tasks"
        })
        app.notify("No tasks available. Create a task first.", severity="error")
        return
    
    if not models:
        log_counter("eval_ui_start_evaluation_blocked", labels={
            "reason": "no_models"
        })
        app.notify("No models configured. Add a model first.", severity="error")
        return
    
    def on_config_ready(config: Optional[Dict[str, Any]]):
        if config:
            asyncio.create_task(_execute_evaluation(app, config))
    
    # Open run configuration dialog
    dialog = RunConfigDialog(
        callback=on_config_ready,
        available_tasks=tasks,
        available_models=models
    )
    app.push_screen(dialog)

async def _execute_evaluation(app: 'TldwCli', config: Dict[str, Any]):
    """Execute the evaluation run."""
    eval_start_time = time.time()
    try:
        app.notify("Starting evaluation...", severity="information")
        
        # Log evaluation start
        log_counter("eval_ui_evaluation_started", labels={
            "task_id": config.get('task_id', 'unknown'),
            "model_id": config.get('model_id', 'unknown')
        })
        
        orchestrator = get_orchestrator()
        
        # Update progress tracker if available
        progress_tracker = None
        cost_estimator = None
        try:
            # Try to find progress tracker and cost estimator in current evals view
            evals_window = app.query_one("EvalsWindow")
            progress_tracker = evals_window.query_one("#progress-tracker")
            cost_estimator = evals_window.query_one("#cost-estimator")
        except:
            pass
        
        def progress_callback(completed: int, total: int, result):
            """Progress callback for real-time updates."""
            # Try to update via EvalsWindow method for more detailed updates
            try:
                def update_ui():
                    # Convert result to dict format if needed
                    result_dict = None
                    if hasattr(result, '__dict__'):
                        result_dict = {
                            'error_info': getattr(result, 'error_info', None),
                            'sample_id': getattr(result, 'sample_id', None),
                            'metrics': getattr(result, 'metrics', {})
                        }
                    
                    # Use EvalsWindow's update method if available
                    if hasattr(evals_window, 'update_evaluation_progress'):
                        evals_window.update_evaluation_progress(
                            run_id=config.get('run_id', 'current'),
                            completed=completed,
                            total=total,
                            current_result=result_dict
                        )
                    elif progress_tracker:
                        # Fallback to direct progress tracker update
                        progress_tracker.current_progress = completed
                        progress_tracker.total_samples = total
                        
                        # Update status message
                        if completed == total:
                            progress_tracker.complete_evaluation()
                        else:
                            progress_tracker.status_message = f"Processing sample {completed}/{total}"
                
                app.call_from_thread(update_ui)
            except Exception as e:
                logger.error(f"Error updating progress: {e}")
        
        # Get model details for cost estimation
        model_info = orchestrator.db.get_model(config['model_id'])
        if cost_estimator and model_info:
            # Show cost estimation
            app.call_from_thread(
                cost_estimator.estimate_cost,
                model_info['provider'],
                model_info['model_id'],
                config.get('max_samples', 100)
            )
        
        # Start evaluation
        if progress_tracker:
            app.call_from_thread(progress_tracker.start_evaluation, config.get('max_samples', 100))
        
        # Generate run ID for tracking
        from uuid import uuid4
        run_id = str(uuid4())
        
        # Start cost tracking
        if cost_estimator:
            app.call_from_thread(cost_estimator.start_tracking, run_id)
        
        # Modified progress callback to include cost tracking
        def enhanced_progress_callback(completed: int, total: int, result):
            # Call original progress callback
            progress_callback(completed, total, result)
            
            # Update cost if we have token counts
            if cost_estimator and hasattr(result, 'token_usage'):
                app.call_from_thread(
                    cost_estimator.update_sample_cost,
                    result.token_usage.get('input_tokens', 0),
                    result.token_usage.get('output_tokens', 0),
                    completed - 1  # 0-based index
                )
        
        actual_run_id = await orchestrator.run_evaluation(
            task_id=config['task_id'],
            model_id=config['model_id'],
            run_name=config['name'],
            max_samples=config.get('max_samples'),
            progress_callback=enhanced_progress_callback if cost_estimator else progress_callback
        )
        
        # Log successful evaluation
        eval_duration = time.time() - eval_start_time
        log_histogram("eval_ui_evaluation_duration", eval_duration, labels={
            "status": "success",
            "task_id": config.get('task_id', 'unknown'),
            "model_id": config.get('model_id', 'unknown')
        })
        log_counter("eval_ui_evaluation_completed", labels={
            "task_id": config.get('task_id', 'unknown'),
            "model_id": config.get('model_id', 'unknown')
        })
        
        app.notify(f"Evaluation completed: {config['name']}", severity="information")
        
        # Finalize cost tracking
        if cost_estimator:
            cost_result = app.call_from_thread(cost_estimator.finalize_tracking)
            if cost_result:
                app.notify(
                    f"Total cost: {cost_estimator.cost_estimator.format_cost_display(cost_result['total_cost'])}", 
                    severity="information"
                )
        
        # Update UI
        await refresh_results_list(app)
        
        # Update the results table with the completed run
        await update_results_table(app, actual_run_id)
        
        # Auto-export if requested
        if config.get('auto_export'):
            await _auto_export_results(app, actual_run_id)
        
    except Exception as e:
        eval_duration = time.time() - eval_start_time
        logger.error(f"Error executing evaluation: {e}")
        
        # Log evaluation error
        log_histogram("eval_ui_evaluation_duration", eval_duration, labels={
            "status": "error",
            "task_id": config.get('task_id', 'unknown'),
            "model_id": config.get('model_id', 'unknown')
        })
        log_counter("eval_ui_evaluation_error", labels={
            "error_type": type(e).__name__,
            "task_id": config.get('task_id', 'unknown'),
            "model_id": config.get('model_id', 'unknown')
        })
        
        app.notify(f"Evaluation failed: {str(e)}", severity="error")
        
        # Update progress tracker with error
        if progress_tracker:
            app.call_from_thread(progress_tracker.error_evaluation, str(e))

# === RESULTS MANAGEMENT EVENTS ===

async def handle_refresh_results(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle refreshing results list."""
    logger.info("Refreshing results list")
    
    # Log UI interaction
    log_counter("eval_ui_refresh_results_clicked")
    
    await refresh_results_list(app)

async def refresh_results_list(app: 'TldwCli'):
    """Refresh the results display."""
    start_time = time.time()
    try:
        orchestrator = get_orchestrator()
        runs = orchestrator.list_runs()
        
        # Log refresh metrics
        duration = time.time() - start_time
        log_histogram("eval_ui_refresh_results_duration", duration)
        log_histogram("eval_ui_results_count", len(runs))
        
        # Update results display
        try:
            evals_window = app.query_one("EvalsWindow")
            results_list = evals_window.query_one("#results-list")
            
            if runs:
                content = "**Recent Evaluations:**\n\n"
                for run in runs[:10]:  # Show last 10 runs
                    status_emoji = "✅" if run['status'] == 'completed' else "❌" if run['status'] == 'failed' else "⏳"
                    content += f"{status_emoji} {run['name']} - {run['status']}\n"
                    content += f"    Task: {run.get('task_name', 'Unknown')}\n"
                    content += f"    Model: {run.get('model_name', 'Unknown')}\n"
                    content += f"    Samples: {run.get('completed_samples', 0)}\n\n"
                
                results_list.update(content)
                
                # Also update the ResultsTable if we have a completed run
                if runs and runs[0]['status'] == 'completed':
                    await update_results_table(app, runs[0]['id'])
            else:
                results_list.update("No evaluation runs found")
                
        except Exception as e:
            logger.error(f"Error updating results display: {e}")
        
    except Exception as e:
        logger.error(f"Error refreshing results: {e}")

async def update_results_table(app: 'TldwCli', run_id: str):
    """Update the ResultsTable widget with detailed results from a run."""
    try:
        orchestrator = get_orchestrator()
        
        # Get detailed results for the run
        results = orchestrator.get_run_results(run_id)
        
        if results:
            # Process results to match the format expected by ResultsTable
            processed_results = []
            for result in results:
                processed_result = {
                    'sample_id': result.get('sample_id', 'Unknown'),
                    'input_text': result.get('input_data', {}).get('input', ''),
                    'expected_output': result.get('expected_output', ''),
                    'actual_output': result.get('actual_output', ''),
                    'metrics': result.get('metrics', {}),
                    'metadata': result.get('metadata', {})
                }
                processed_results.append(processed_result)
            
            # Try to find and update the ResultsTable widget
            try:
                evals_window = app.query_one("EvalsWindow")
                from ..Widgets.eval_results_widgets import ResultsTable
                results_table = evals_window.query_one(ResultsTable)
                
                # Update the table on the main thread
                def update_table():
                    results_table.update_results(processed_results)
                
                app.call_from_thread(update_table)
                
                # Also update the metrics display with the run summary
                run_summary = orchestrator.get_run_summary(run_id)
                if run_summary and run_summary.get('metrics'):
                    try:
                        from ..Widgets.eval_results_widgets import MetricsDisplay
                        metrics_display = evals_window.query_one(MetricsDisplay)
                        
                        def update_metrics():
                            metrics_display.update_metrics(run_summary['metrics'])
                        
                        app.call_from_thread(update_metrics)
                    except Exception as e:
                        logger.warning(f"Could not update metrics display: {e}")
                
            except Exception as e:
                logger.error(f"Error finding or updating ResultsTable: {e}")
        
    except Exception as e:
        logger.error(f"Error loading run results: {e}")
        app.notify(f"Error refreshing results: {str(e)}", severity="error")

async def handle_view_detailed_results(app: 'TldwCli', event) -> None:
    """Handle viewing detailed results for the most recent run."""
    logger.info("Viewing detailed results")
    
    # Log UI interaction
    log_counter("eval_ui_view_detailed_results_clicked")
    
    try:
        orchestrator = get_orchestrator()
        
        # Get the most recent completed run
        runs = orchestrator.list_runs(status='completed', limit=1)
        
        if runs:
            run_id = runs[0]['id']
            await update_results_table(app, run_id)
            
            # Log successful result viewing
            log_counter("eval_ui_detailed_results_viewed", labels={
                "run_id": run_id
            })
            
            app.notify(f"Loaded results for: {runs[0]['name']}", severity="information")
        else:
            log_counter("eval_ui_detailed_results_not_found")
            app.notify("No completed evaluation runs found", severity="warning")
            
    except Exception as e:
        logger.error(f"Error viewing detailed results: {e}")
        app.notify(f"Error loading results: {str(e)}", severity="error")

async def handle_export_results(app: 'TldwCli', format_type: str) -> None:
    """Handle exporting results."""
    logger.info(f"Exporting results in {format_type} format")
    
    # Log export request
    log_counter("eval_ui_export_results_clicked", labels={
        "format_type": format_type
    })
    
    # Get list of completed runs
    orchestrator = get_orchestrator()
    runs = orchestrator.list_runs(status='completed')
    
    if not runs:
        log_counter("eval_ui_export_blocked", labels={
            "reason": "no_completed_runs"
        })
        app.notify("No completed runs to export", severity="error")
        return
    
    # For now, export the most recent run
    run = runs[0]
    
    def on_file_selected(file_path: Optional[str]):
        if file_path:
            asyncio.create_task(_export_run_results(app, run['id'], file_path, format_type))
    
    # Open export file picker
    dialog = ExportFilePickerDialog(callback=on_file_selected)
    app.push_screen(dialog)

async def _export_run_results(app: 'TldwCli', run_id: str, file_path: str, format_type: str):
    """Export results to file."""
    start_time = time.time()
    try:
        app.notify("Exporting results...", severity="information")
        
        orchestrator = get_orchestrator()
        
        # Ensure file has correct extension
        file_path = Path(file_path)
        if format_type == 'csv' and not file_path.suffix.lower() == '.csv':
            file_path = file_path.with_suffix('.csv')
        elif format_type == 'json' and not file_path.suffix.lower() == '.json':
            file_path = file_path.with_suffix('.json')
        
        # Create parent directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export based on format
        if format_type == 'csv':
            # Custom CSV export for better formatting
            await _export_to_csv(orchestrator, run_id, str(file_path))
        else:
            # Use built-in JSON export
            orchestrator.export_results(run_id, str(file_path), format_type)
        
        # Log successful export
        duration = time.time() - start_time
        log_histogram("eval_ui_export_duration", duration, labels={
            "format_type": format_type,
            "status": "success"
        })
        log_counter("eval_ui_results_exported", labels={
            "format_type": format_type,
            "run_id": run_id
        })
        
        app.notify(f"Results exported to: {file_path.name}", severity="information")
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error exporting results: {e}")
        
        # Log export error
        log_histogram("eval_ui_export_duration", duration, labels={
            "format_type": format_type,
            "status": "error"
        })
        log_counter("eval_ui_export_error", labels={
            "format_type": format_type,
            "error_type": type(e).__name__
        })
        
        app.notify(f"Export failed: {str(e)}", severity="error")

async def _export_to_csv(orchestrator, run_id: str, file_path: str):
    """Export results to CSV format with proper formatting."""
    import csv
    
    # Get run summary and results
    run_summary = orchestrator.get_run_summary(run_id)
    results = orchestrator.get_run_results(run_id)
    
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        # Write run summary as header comments
        csvfile.write(f"# Evaluation Run: {run_summary['run_info']['name']}\n")
        csvfile.write(f"# Task: {run_summary['run_info'].get('task_name', 'Unknown')}\n")
        csvfile.write(f"# Model: {run_summary['run_info'].get('model_name', 'Unknown')}\n")
        csvfile.write(f"# Status: {run_summary['run_info']['status']}\n")
        csvfile.write(f"# Samples: {run_summary['run_info']['completed_samples']}\n")
        
        # Write metrics summary
        if run_summary.get('metrics'):
            csvfile.write("# Metrics:\n")
            for metric, value in run_summary['metrics'].items():
                if isinstance(value, dict):
                    value = value.get('value', 'N/A')
                csvfile.write(f"#   {metric}: {value}\n")
        csvfile.write("\n")
        
        # Write results table
        if results:
            # Determine columns from first result
            fieldnames = ['sample_id', 'input', 'expected_output', 'actual_output']
            
            # Add metric columns
            if results[0].get('metrics'):
                metric_names = list(results[0]['metrics'].keys())
                fieldnames.extend(metric_names)
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {
                    'sample_id': result.get('sample_id', ''),
                    'input': result.get('input_data', {}).get('input', ''),
                    'expected_output': result.get('expected_output', ''),
                    'actual_output': result.get('actual_output', '')
                }
                
                # Add metrics
                if result.get('metrics'):
                    for metric_name, metric_value in result['metrics'].items():
                        if isinstance(metric_value, dict):
                            metric_value = metric_value.get('value', '')
                        row[metric_name] = metric_value
                
                writer.writerow(row)

async def _auto_export_results(app: 'TldwCli', run_id: str):
    """Auto-export results after evaluation."""
    try:
        # Create default export path
        export_dir = Path.home() / "eval_results"
        export_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = export_dir / f"eval_results_{timestamp}.json"
        
        orchestrator = get_orchestrator()
        orchestrator.export_results(run_id, str(export_path), 'json')
        
        app.notify(f"Results auto-exported to: {export_path.name}", severity="information")
        
    except Exception as e:
        logger.error(f"Error auto-exporting results: {e}")

# === DATASET MANAGEMENT EVENTS ===

async def handle_upload_dataset(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle dataset file upload."""
    logger.info("Starting dataset upload")
    
    # Log UI interaction
    log_counter("eval_ui_upload_dataset_clicked")
    
    def on_file_selected(file_path: Optional[str]):
        if file_path:
            asyncio.create_task(_process_dataset_upload(app, file_path))
    
    # Open dataset file picker
    dialog = DatasetFilePickerDialog(callback=on_file_selected)
    app.push_screen(dialog)

async def _process_dataset_upload(app: 'TldwCli', file_path: str):
    """Process the uploaded dataset file."""
    try:
        app.notify(f"Processing dataset: {Path(file_path).name}", severity="information")
        
        orchestrator = get_orchestrator()
        dataset_name = Path(file_path).stem
        
        dataset_id = orchestrator.create_dataset_from_file(
            name=dataset_name,
            file_path=file_path,
            description=f"Uploaded dataset from {Path(file_path).name}"
        )
        
        # Log successful dataset upload
        log_counter("eval_ui_dataset_uploaded", labels={
            "file_extension": Path(file_path).suffix
        })
        
        app.notify(f"Dataset uploaded: {dataset_name}", severity="information")
        
        # Update UI
        await refresh_datasets_list(app)
        
    except Exception as e:
        logger.error(f"Error processing dataset upload: {e}")
        
        # Log dataset upload error
        log_counter("eval_ui_dataset_upload_error", labels={
            "error_type": type(e).__name__
        })
        
        app.notify(f"Dataset upload failed: {str(e)}", severity="error")

async def handle_refresh_datasets(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle refreshing datasets list."""
    logger.info("Refreshing datasets list")
    
    # Log UI interaction
    log_counter("eval_ui_refresh_datasets_clicked")
    
    await refresh_datasets_list(app)

# === UI UPDATE HELPERS ===

async def refresh_tasks_list(app: 'TldwCli'):
    """Refresh the tasks display."""
    start_time = time.time()
    try:
        orchestrator = get_orchestrator()
        tasks = orchestrator.list_tasks()
        
        # Log refresh metrics
        duration = time.time() - start_time
        log_histogram("eval_ui_refresh_tasks_duration", duration)
        log_histogram("eval_ui_tasks_count", len(tasks))
        
        # Update task status display
        try:
            evals_window = app.query_one("EvalsWindow")
            task_status = evals_window.query_one("#task-status")
            
            if tasks:
                task_status.update(f"✅ {len(tasks)} tasks available")
            else:
                task_status.update("No tasks configured")
                
        except Exception as e:
            logger.error(f"Error updating task status: {e}")
        
    except Exception as e:
        logger.error(f"Error refreshing tasks: {e}")

async def refresh_models_list(app: 'TldwCli'):
    """Refresh the models display."""
    start_time = time.time()
    try:
        orchestrator = get_orchestrator()
        models = orchestrator.list_models()
        
        # Log refresh metrics
        duration = time.time() - start_time
        log_histogram("eval_ui_refresh_models_duration", duration)
        log_histogram("eval_ui_models_count", len(models))
        
        # Update models list display
        try:
            evals_window = app.query_one("EvalsWindow")
            models_list = evals_window.query_one("#models-list")
            
            if models:
                content = "**Configured Models:**\n\n"
                for model in models:
                    content += f"• {model['name']} ({model['provider']})\n"
                    content += f"  Model: {model['model_id']}\n\n"
                
                models_list.update(content)
            else:
                models_list.update("No models configured")
                
        except Exception as e:
            logger.error(f"Error updating models display: {e}")
        
    except Exception as e:
        logger.error(f"Error refreshing models: {e}")

async def refresh_datasets_list(app: 'TldwCli'):
    """Refresh the datasets display."""
    start_time = time.time()
    try:
        orchestrator = get_orchestrator()
        datasets = orchestrator.db.list_datasets()
        
        # Log refresh metrics
        duration = time.time() - start_time
        log_histogram("eval_ui_refresh_datasets_duration", duration)
        log_histogram("eval_ui_datasets_count", len(datasets))
        
        # Update datasets display
        try:
            evals_window = app.query_one("EvalsWindow")
            datasets_list = evals_window.query_one("#datasets-list")
            
            if datasets:
                content = "**Available Datasets:**\n\n"
                for dataset in datasets:
                    content += f"• {dataset['name']}\n"
                    content += f"  Format: {dataset['format']}\n"
                    content += f"  Source: {Path(dataset['source_path']).name}\n\n"
                
                datasets_list.update(content)
            else:
                datasets_list.update("No datasets found")
                
        except Exception as e:
            logger.error(f"Error updating datasets display: {e}")
        
    except Exception as e:
        logger.error(f"Error refreshing datasets: {e}")

# === TEMPLATE EVENTS ===

async def handle_template_button(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle template button press."""
    button_id = event.button.id
    
    # Log template button click
    log_counter("eval_ui_template_button_clicked", labels={
        "button_id": button_id
    })
    
    # Map button IDs to template names
    template_mapping = {
        'template-gsm8k-btn': 'gsm8k_math',
        'template-logic-btn': 'logical_reasoning',
        'template-cot-btn': 'chain_of_thought',
        'template-harm-btn': 'harmfulness_detection',
        'template-bias-btn': 'bias_evaluation',
        'template-truth-btn': 'truthfulness_qa',
        'template-humaneval-btn': 'humaneval_coding',
        'template-bugs-btn': 'bug_detection',
        'template-sql-btn': 'sql_generation',
        'template-medical-btn': 'medical_qa',
        'template-legal-btn': 'legal_reasoning',
        'template-science-btn': 'scientific_reasoning',
        'template-creative-btn': 'creative_writing',
        'template-story-btn': 'story_completion',
        'template-summary-btn': 'summarization_quality'
    }
    
    template_name = template_mapping.get(button_id)
    if template_name:
        await handle_template_selection(app, template_name)
    else:
        logger.warning(f"Unknown template button: {button_id}")

# === COMPARISON EVENTS ===

async def handle_compare_runs(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle run comparison."""
    logger.info("Starting run comparison")
    
    # Log UI interaction
    log_counter("eval_ui_compare_runs_clicked")
    
    orchestrator = get_orchestrator()
    runs = orchestrator.list_runs(status='completed', limit=10)
    
    if len(runs) < 2:
        log_counter("eval_ui_compare_runs_blocked", labels={
            "reason": "insufficient_runs",
            "available_runs": str(len(runs))
        })
        app.notify("Need at least 2 completed runs to compare", severity="error")
        return
    
    # Show run selection dialog if more than 2 runs available
    if len(runs) > 2:
        # Import here to avoid circular imports
        try:
            from ..Widgets.eval_additional_dialogs import RunSelectionDialog
            
            def on_runs_selected(selected_runs):
                if selected_runs and len(selected_runs) >= 2:
                    asyncio.create_task(_perform_comparison(app, selected_runs))
            
            dialog = RunSelectionDialog(
                available_runs=runs,
                callback=on_runs_selected,
                min_selection=2,
                max_selection=4
            )
            app.push_screen(dialog)
            return
        except ImportError:
            # Fallback to comparing two most recent
            pass
    
    # Compare the two most recent runs
    await _perform_comparison(app, [runs[0], runs[1]])

async def _perform_comparison(app: 'TldwCli', runs_to_compare: List[Dict[str, Any]]):
    """Perform the actual comparison of runs."""
    try:
        orchestrator = get_orchestrator()
        run_ids = [run['id'] for run in runs_to_compare]
        
        # Get detailed summaries for each run
        comparison_data = {}
        for run in runs_to_compare:
            run_id = run['id']
            summary = orchestrator.get_run_summary(run_id)
            comparison_data[run['name']] = summary
        
        # Log successful comparison
        log_counter("eval_ui_runs_compared", labels={
            "num_runs": str(len(run_ids))
        })
        
        # Update comparison display
        try:
            evals_window = app.query_one("EvalsWindow")
            comparison_results = evals_window.query_one("#comparison-results")
            
            content = "**Run Comparison:**\n\n"
            
            # Create comparison table
            metric_names = set()
            for run_data in comparison_data.values():
                if run_data.get('metrics'):
                    metric_names.update(run_data['metrics'].keys())
            
            # Header
            content += "| Metric | " + " | ".join(comparison_data.keys()) + " |\n"
            content += "|--------|" + "|".join(["---------"] * len(comparison_data)) + "|\n"
            
            # Metrics rows
            for metric in sorted(metric_names):
                row = f"| {metric} |"
                for run_name, run_data in comparison_data.items():
                    value = run_data.get('metrics', {}).get(metric, 'N/A')
                    if isinstance(value, dict):
                        value = value.get('value', 'N/A')
                    if isinstance(value, float):
                        row += f" {value:.3f} |"
                    else:
                        row += f" {value} |"
                content += row + "\n"
            
            # Add run metadata
            content += "\n**Run Details:**\n\n"
            for run_name, run_data in comparison_data.items():
                run_info = run_data.get('run_info', {})
                content += f"**{run_name}:**\n"
                content += f"  - Task: {run_info.get('task_name', 'Unknown')}\n"
                content += f"  - Model: {run_info.get('model_name', 'Unknown')}\n"
                content += f"  - Samples: {run_info.get('completed_samples', 0)}\n"
                content += f"  - Status: {run_info.get('status', 'Unknown')}\n"
                if run_data.get('duration'):
                    content += f"  - Duration: {run_data['duration']:.1f}s\n"
                content += "\n"
            
            comparison_results.update(content)
            
        except Exception as e:
            logger.error(f"Error updating comparison display: {e}")
            app.notify(f"Error displaying comparison: {str(e)}", severity="error")
            
    except Exception as e:
        logger.error(f"Error performing comparison: {e}")
        app.notify(f"Error comparing runs: {str(e)}", severity="error")

# === INITIALIZATION ===

async def initialize_evals_system(app: 'TldwCli') -> None:
    """Initialize the evaluation system."""
    logger.info("Initializing evaluation system")
    
    # Log system initialization
    log_counter("eval_ui_system_initialized")
    
    try:
        # Initialize orchestrator
        orchestrator = get_orchestrator()
        
        # Check if we have any data to show
        tasks = orchestrator.list_tasks()
        models = orchestrator.list_models()
        
        # Refresh all displays
        await refresh_tasks_list(app)
        await refresh_models_list(app)
        await refresh_datasets_list(app)
        await refresh_results_list(app)
        
        # Log successful initialization
        log_counter("eval_ui_system_initialization_success")
        
        # More informative notification
        if not tasks:
            app.notify("Evaluation system ready. Create a task to get started!", severity="information")
        elif not models:
            app.notify("Evaluation system ready. Add a model to start evaluating!", severity="information")
        else:
            app.notify(f"Evaluation system ready. {len(tasks)} tasks, {len(models)} models available.", severity="information")
        
    except Exception as e:
        logger.error(f"Error initializing evaluation system: {e}")
        
        # Log initialization error
        log_counter("eval_ui_system_initialization_error", labels={
            "error_type": type(e).__name__
        })
        
        app.notify(f"Failed to initialize evaluation system: {str(e)}", severity="error")

# === HELPER FUNCTIONS FOR UX ENHANCEMENTS ===

def get_recent_evaluations(app: 'TldwCli', limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent evaluation runs for dashboard display."""
    try:
        orchestrator = get_orchestrator()
        runs = orchestrator.list_runs(limit=limit)
        
        # Enhance with metrics summary
        enhanced_runs = []
        for run in runs:
            try:
                summary = orchestrator.get_run_summary(run['id'])
                run['metrics'] = summary.get('error_statistics', {})
                enhanced_runs.append(run)
            except Exception as e:
                logger.warning(f"Could not get summary for run {run['id']}: {e}")
                enhanced_runs.append(run)
        
        return enhanced_runs
        
    except Exception as e:
        logger.error(f"Error getting recent evaluations: {e}")
        return []

def get_available_models(app: 'TldwCli', limit: int = 50) -> List[Dict[str, Any]]:
    """Get available model configurations."""
    try:
        orchestrator = get_orchestrator()
        return orchestrator.list_models(limit=limit)
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        return []

def get_available_tasks(app: 'TldwCli', limit: int = 50) -> List[Dict[str, Any]]:
    """Get available task configurations."""
    try:
        orchestrator = get_orchestrator()
        return orchestrator.list_tasks(limit=limit)
    except Exception as e:
        logger.error(f"Error getting available tasks: {e}")
        return []

def get_available_datasets(app: 'TldwCli', limit: int = 50) -> List[Dict[str, Any]]:
    """Get available datasets."""
    try:
        orchestrator = get_orchestrator()
        return orchestrator.list_datasets(limit=limit)
    except Exception as e:
        logger.error(f"Error getting available datasets: {e}")
        return []

async def get_dataset_info(app: 'TldwCli', dataset_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific dataset."""
    try:
        orchestrator = get_orchestrator()
        datasets = orchestrator.list_datasets()
        
        # Find the dataset by name
        for dataset in datasets:
            if dataset.get('name') == dataset_name:
                # Add additional computed fields
                dataset['created'] = dataset.get('created_at', 'Unknown')
                dataset['format'] = Path(dataset.get('source_path', '')).suffix.lstrip('.') or 'Unknown'
                dataset['task'] = dataset.get('task_type', 'General')
                return dataset
        
        # Dataset not found
        return {
            'name': dataset_name,
            'type': 'Unknown',
            'size': 0,
            'created': 'Unknown',
            'format': 'Unknown',
            'task': 'Unknown'
        }
    except Exception as e:
        logger.error(f"Error getting dataset info: {e}")
        return {}

async def validate_dataset(app: 'TldwCli', file_path: str) -> Dict[str, Any]:
    """Validate a dataset file."""
    try:
        path = Path(file_path)
        if not path.exists():
            return {'valid': False, 'error': 'File not found'}
        
        # Check file size
        file_size = path.stat().st_size
        if file_size == 0:
            return {'valid': False, 'error': 'File is empty'}
        
        # Try to load and count samples based on file type
        samples_count = 0
        try:
            if path.suffix.lower() == '.json':
                import json
                with open(path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples_count = len(data)
                    else:
                        samples_count = 1
            elif path.suffix.lower() == '.jsonl':
                with open(path, 'r') as f:
                    samples_count = sum(1 for line in f if line.strip())
            elif path.suffix.lower() in ['.csv', '.tsv']:
                import csv
                with open(path, 'r') as f:
                    reader = csv.reader(f)
                    samples_count = sum(1 for _ in reader) - 1  # Subtract header
            else:
                return {'valid': False, 'error': f'Unsupported file format: {path.suffix}'}
            
            return {
                'valid': True,
                'samples': samples_count,
                'format': path.suffix.lstrip('.'),
                'size_mb': round(file_size / 1024 / 1024, 2)
            }
            
        except Exception as e:
            return {'valid': False, 'error': f'Failed to parse file: {str(e)}'}
            
    except Exception as e:
        logger.error(f"Error validating dataset: {e}")
        return {'valid': False, 'error': str(e)}

async def upload_dataset(app: 'TldwCli', source: str, source_type: str = "file") -> Dict[str, Any]:
    """Upload a dataset from file or URL."""
    try:
        orchestrator = get_orchestrator()
        
        if source_type == "file":
            # Validate file first
            validation = await validate_dataset(app, source)
            if not validation['valid']:
                return {'success': False, 'error': validation['error']}
            
            dataset_name = Path(source).stem
            dataset_id = orchestrator.create_dataset_from_file(
                name=dataset_name,
                file_path=source,
                description=f"Uploaded from {Path(source).name}"
            )
            
            return {
                'success': True,
                'dataset_name': dataset_name,
                'dataset_id': dataset_id
            }
            
        elif source_type == "url":
            # TODO: Implement URL download
            return {'success': False, 'error': 'URL upload not yet implemented'}
            
        else:
            return {'success': False, 'error': f'Unknown source type: {source_type}'}
            
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        return {'success': False, 'error': str(e)}

async def delete_dataset(app: 'TldwCli', dataset_name: str) -> Dict[str, Any]:
    """Delete a dataset."""
    try:
        orchestrator = get_orchestrator()
        
        # Find dataset ID by name
        datasets = orchestrator.list_datasets()
        dataset_id = None
        for dataset in datasets:
            if dataset.get('name') == dataset_name:
                dataset_id = dataset.get('id')
                break
        
        if not dataset_id:
            return {'success': False, 'error': 'Dataset not found'}
        
        # Delete the dataset
        orchestrator.db.delete_dataset(dataset_id)
        
        return {'success': True}
        
    except Exception as e:
        logger.error(f"Error deleting dataset: {e}")
        return {'success': False, 'error': str(e)}

async def export_dataset(app: 'TldwCli', dataset_name: str, export_path: str, format: str = 'json') -> Dict[str, Any]:
    """Export a dataset to file."""
    try:
        # TODO: Implement dataset export
        return {'success': False, 'error': 'Export not yet implemented'}
        
    except Exception as e:
        logger.error(f"Error exporting dataset: {e}")
        return {'success': False, 'error': str(e)}

async def handle_cancel_evaluation(app: 'TldwCli', run_id: Optional[str]) -> None:
    """Handle cancelling an active evaluation."""
    log_counter("eval_ui_cancel_evaluation_clicked")
    
    if not run_id:
        log_counter("eval_ui_cancel_evaluation_blocked", labels={
            "reason": "no_active_evaluation"
        })
        app.notify("No active evaluation to cancel", severity="warning")
        return
    
    try:
        orchestrator = get_orchestrator()
        success = orchestrator.cancel_evaluation(run_id)
        
        if success:
            log_counter("eval_ui_evaluation_cancelled", labels={
                "run_id": run_id
            })
            app.notify(f"Evaluation {run_id} cancelled", severity="information")
        else:
            log_counter("eval_ui_evaluation_cancel_failed", labels={
                "run_id": run_id
            })
            app.notify(f"Could not cancel evaluation {run_id}", severity="warning")
            
    except Exception as e:
        logger.error(f"Error cancelling evaluation: {e}")
        
        # Log cancellation error
        log_counter("eval_ui_evaluation_cancel_error", labels={
            "error_type": type(e).__name__
        })
        
        app.notify(f"Error cancelling evaluation: {str(e)}", severity="error")