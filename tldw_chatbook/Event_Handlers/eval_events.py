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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from loguru import logger

from textual.widgets import Button, Input

if TYPE_CHECKING:
    from ..app import TldwCli

# Import evaluation components
from ..App_Functions.Evals.eval_orchestrator import EvaluationOrchestrator
from ..App_Functions.Evals.task_loader import TaskLoader, TaskLoadError
from ..Widgets.file_picker_dialog import TaskFilePickerDialog, DatasetFilePickerDialog, ExportFilePickerDialog
from ..Widgets.eval_config_dialogs import ModelConfigDialog, TaskConfigDialog, RunConfigDialog
from ..Widgets.template_selector import TemplateSelectorDialog

# Global orchestrator instance (will be initialized on first use)
_orchestrator: Optional[EvaluationOrchestrator] = None

def get_orchestrator() -> EvaluationOrchestrator:
    """Get or create the global evaluation orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = EvaluationOrchestrator()
    return _orchestrator

# === TASK MANAGEMENT EVENTS ===

async def handle_upload_task(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle task file upload."""
    logger.info("Starting task file upload")
    
    def on_file_selected(file_path: Optional[str]):
        if file_path:
            asyncio.create_task(_process_task_upload(app, file_path))
    
    # Open file picker dialog
    dialog = TaskFilePickerDialog(callback=on_file_selected)
    app.push_screen(dialog)

async def _process_task_upload(app: 'TldwCli', file_path: str):
    """Process the uploaded task file."""
    try:
        app.notify(f"Processing task file: {Path(file_path).name}", severity="information")
        
        orchestrator = get_orchestrator()
        task_id = await orchestrator.create_task_from_file(file_path)
        
        app.notify(f"Task created successfully: {task_id}", severity="information")
        
        # Update UI
        await refresh_tasks_list(app)
        
    except TaskLoadError as e:
        logger.error(f"Task load error: {e}")
        app.notify(f"Failed to load task: {str(e)}", severity="error")
    except Exception as e:
        logger.error(f"Error processing task upload: {e}")
        app.notify(f"Error processing task: {str(e)}", severity="error")

async def handle_create_task(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle new task creation."""
    logger.info("Creating new task")
    
    def on_config_ready(config: Optional[Dict[str, Any]]):
        if config:
            asyncio.create_task(_create_custom_task(app, config))
    
    # Open task configuration dialog
    dialog = TaskConfigDialog(callback=on_config_ready)
    app.push_screen(dialog)

async def _create_custom_task(app: 'TldwCli', config: Dict[str, Any]):
    """Create a custom task from configuration."""
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
        
        app.notify(f"Task created successfully: {config['name']}", severity="information")
        
        # Update UI
        await refresh_tasks_list(app)
        
    except Exception as e:
        logger.error(f"Error creating custom task: {e}")
        app.notify(f"Error creating task: {str(e)}", severity="error")

async def handle_template_selection(app: 'TldwCli', template_name: str) -> None:
    """Handle template selection for task creation."""
    logger.info(f"Creating task from template: {template_name}")
    
    try:
        app.notify(f"Creating task from template: {template_name}", severity="information")
        
        orchestrator = get_orchestrator()
        
        # Create task from template
        task_id, dataset_id = orchestrator.create_task_from_template(
            template_name=template_name,
            output_dir=None  # Don't save files by default
        )
        
        app.notify(f"Task created from template: {template_name}", severity="information")
        
        # Update UI
        await refresh_tasks_list(app)
        
    except Exception as e:
        logger.error(f"Error creating task from template: {e}")
        app.notify(f"Error creating template task: {str(e)}", severity="error")

# === MODEL MANAGEMENT EVENTS ===

async def handle_add_model(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle adding a new model configuration."""
    logger.info("Adding new model configuration")
    
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
        
        app.notify(f"Model configured: {config['name']}", severity="information")
        
        # Update UI
        await refresh_models_list(app)
        
    except Exception as e:
        logger.error(f"Error saving model config: {e}")
        app.notify(f"Error saving model: {str(e)}", severity="error")

async def handle_provider_setup(app: 'TldwCli', provider: str) -> None:
    """Handle quick provider setup."""
    logger.info(f"Setting up provider: {provider}")
    
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
    
    # Get available tasks and models
    orchestrator = get_orchestrator()
    tasks = orchestrator.list_tasks()
    models = orchestrator.list_models()
    
    if not tasks:
        app.notify("No tasks available. Create a task first.", severity="error")
        return
    
    if not models:
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
    try:
        app.notify("Starting evaluation...", severity="information")
        
        orchestrator = get_orchestrator()
        
        # Update progress tracker if available
        progress_tracker = None
        try:
            # Try to find progress tracker in current evals view
            evals_window = app.query_one("EvalsWindow")
            progress_tracker = evals_window.query_one("#progress-tracker")
        except:
            pass
        
        def progress_callback(completed: int, total: int, result):
            """Progress callback for real-time updates."""
            if progress_tracker:
                progress_tracker.current_progress = completed
                progress_tracker.total_samples = total
                
                # Update status message
                if completed == total:
                    progress_tracker.complete_evaluation()
                else:
                    progress_tracker.status_message = f"Processing sample {completed}/{total}"
        
        # Start evaluation
        if progress_tracker:
            progress_tracker.start_evaluation(config.get('max_samples', 100))
        
        run_id = await orchestrator.run_evaluation(
            task_id=config['task_id'],
            model_id=config['model_id'],
            run_name=config['name'],
            max_samples=config.get('max_samples'),
            progress_callback=progress_callback
        )
        
        app.notify(f"Evaluation completed: {config['name']}", severity="information")
        
        # Update UI
        await refresh_results_list(app)
        
        # Auto-export if requested
        if config.get('auto_export'):
            await _auto_export_results(app, run_id)
        
    except Exception as e:
        logger.error(f"Error executing evaluation: {e}")
        app.notify(f"Evaluation failed: {str(e)}", severity="error")
        
        # Update progress tracker with error
        if progress_tracker:
            progress_tracker.error_evaluation(str(e))

# === RESULTS MANAGEMENT EVENTS ===

async def handle_refresh_results(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle refreshing results list."""
    logger.info("Refreshing results list")
    await refresh_results_list(app)

async def refresh_results_list(app: 'TldwCli'):
    """Refresh the results display."""
    try:
        orchestrator = get_orchestrator()
        runs = orchestrator.list_runs()
        
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
            else:
                results_list.update("No evaluation runs found")
                
        except Exception as e:
            logger.error(f"Error updating results display: {e}")
        
    except Exception as e:
        logger.error(f"Error refreshing results: {e}")
        app.notify(f"Error refreshing results: {str(e)}", severity="error")

async def handle_export_results(app: 'TldwCli', format_type: str) -> None:
    """Handle exporting results."""
    logger.info(f"Exporting results in {format_type} format")
    
    # Get list of completed runs
    orchestrator = get_orchestrator()
    runs = orchestrator.list_runs(status='completed')
    
    if not runs:
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
    try:
        app.notify("Exporting results...", severity="information")
        
        orchestrator = get_orchestrator()
        orchestrator.export_results(run_id, file_path, format_type)
        
        app.notify(f"Results exported to: {Path(file_path).name}", severity="information")
        
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        app.notify(f"Export failed: {str(e)}", severity="error")

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
        
        app.notify(f"Dataset uploaded: {dataset_name}", severity="information")
        
        # Update UI
        await refresh_datasets_list(app)
        
    except Exception as e:
        logger.error(f"Error processing dataset upload: {e}")
        app.notify(f"Dataset upload failed: {str(e)}", severity="error")

async def handle_refresh_datasets(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handle refreshing datasets list."""
    logger.info("Refreshing datasets list")
    await refresh_datasets_list(app)

# === UI UPDATE HELPERS ===

async def refresh_tasks_list(app: 'TldwCli'):
    """Refresh the tasks display."""
    try:
        orchestrator = get_orchestrator()
        tasks = orchestrator.list_tasks()
        
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
    try:
        orchestrator = get_orchestrator()
        models = orchestrator.list_models()
        
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
    try:
        orchestrator = get_orchestrator()
        datasets = orchestrator.db.list_datasets()
        
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
    
    orchestrator = get_orchestrator()
    runs = orchestrator.list_runs(status='completed')
    
    if len(runs) < 2:
        app.notify("Need at least 2 completed runs to compare", severity="error")
        return
    
    # For now, compare the two most recent runs
    run_ids = [runs[0]['id'], runs[1]['id']]
    comparison = orchestrator.compare_runs(run_ids)
    
    # Update comparison display
    try:
        evals_window = app.query_one("EvalsWindow")
        comparison_results = evals_window.query_one("#comparison-results")
        
        content = "**Run Comparison:**\n\n"
        for run_name, run_data in comparison.items():
            content += f"**{run_name}:**\n"
            for metric, data in run_data.get('metrics', {}).items():
                if isinstance(data, dict):
                    value = data.get('value', 'N/A')
                else:
                    value = data
                content += f"  {metric}: {value}\n"
            content += "\n"
        
        comparison_results.update(content)
        
    except Exception as e:
        logger.error(f"Error updating comparison display: {e}")

# === INITIALIZATION ===

async def initialize_evals_system(app: 'TldwCli') -> None:
    """Initialize the evaluation system."""
    logger.info("Initializing evaluation system")
    
    try:
        # Initialize orchestrator
        get_orchestrator()
        
        # Refresh all displays
        await refresh_tasks_list(app)
        await refresh_models_list(app)
        await refresh_datasets_list(app)
        await refresh_results_list(app)
        
        app.notify("Evaluation system initialized", severity="information")
        
    except Exception as e:
        logger.error(f"Error initializing evaluation system: {e}")
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
        # This would need to be implemented in the orchestrator
        # For now, return empty list
        return []
    except Exception as e:
        logger.error(f"Error getting available datasets: {e}")
        return []

async def handle_cancel_evaluation(app: 'TldwCli', run_id: Optional[str]) -> None:
    """Handle cancelling an active evaluation."""
    if not run_id:
        app.notify("No active evaluation to cancel", severity="warning")
        return
    
    try:
        orchestrator = get_orchestrator()
        success = orchestrator.cancel_evaluation(run_id)
        
        if success:
            app.notify(f"Evaluation {run_id} cancelled", severity="information")
        else:
            app.notify(f"Could not cancel evaluation {run_id}", severity="warning")
            
    except Exception as e:
        logger.error(f"Error cancelling evaluation: {e}")
        app.notify(f"Error cancelling evaluation: {str(e)}", severity="error")