# FINISHING-EVALS.md

## Completing the tldw_chatbook Evaluation System Implementation

**Document Created**: 2025-07-06  
**Estimated Completion Time**: 2-3 weeks  
**Overall Progress**: 80% Complete (Backend 100%, UI 60%, Integration 20%)

## Executive Summary

The evaluation system backend is **fully implemented and tested** with comprehensive functionality for evaluating LLMs across 27+ task types using 30+ providers. All core components (database, task loading, runners, metrics) are production-ready.

**What remains is UI integration work** - connecting the existing UI widgets to the backend through event handlers. No architectural changes are needed; only integration code to wire existing components together.

## Current State Assessment

### ✅ Complete Components
1. **Backend (100%)**
   - Database schema and operations
   - Task loading from multiple formats
   - Evaluation runners (base + specialized)
   - LLM interface for 30+ providers
   - Metrics calculation
   - Results storage and export

2. **UI Widgets (90%)**
   - Configuration dialogs exist
   - Results display widgets exist
   - Progress tracking widgets exist
   - File picker dialogs exist

3. **Event Handler Structure (70%)**
   - Handler functions defined
   - Imports configured
   - Basic routing implemented

### ❌ Missing Integration Points
1. **UI → Event Handler Connections (20%)**
   - Progress callbacks not wired
   - Results refresh not automated
   - Status updates not propagated

2. **Event Handler → Backend Calls (40%)**
   - Some handlers have placeholder code
   - Progress tracking not implemented
   - Error handling incomplete

3. **Backend → UI Updates (10%)**
   - No real-time progress updates
   - Results not automatically displayed
   - Status changes not reflected

## Detailed Gap Analysis

### 1. Progress Tracking Integration
**Location**: `Event_Handlers/eval_events.py` → `handle_start_evaluation()`
**Issue**: Progress callback not implemented
**Fix Required**:
```python
async def handle_start_evaluation(app: 'TldwCli', event):
    # MISSING: Progress callback implementation
    def progress_callback(completed: int, total: int, current_result: Dict[str, Any]):
        # Update UI progress tracker
        app.call_from_thread(update_progress_ui, app, completed, total, current_result)
    
    # Pass callback to orchestrator
    run_id = await orchestrator.run_evaluation(
        task_id=task_id,
        model_configs=model_configs,
        progress_callback=progress_callback  # <- Not currently implemented
    )
```

### 2. Results Display Integration
**Location**: `UI/Evals_Window.py` → Results view
**Issue**: Results table not populated from database
**Fix Required**:
```python
async def refresh_results_table(self):
    # MISSING: Fetch results from orchestrator
    orchestrator = get_orchestrator()
    recent_runs = orchestrator.db.get_recent_runs(limit=20)
    
    # MISSING: Populate DataTable widget
    results_table = self.query_one("#results-table")
    results_table.clear()
    for run in recent_runs:
        results_table.add_row(*format_run_data(run))
```

### 3. Model Configuration Persistence
**Location**: `Event_Handlers/eval_events.py` → `handle_add_model()`
**Issue**: Model configs not saved to database
**Fix Required**:
```python
async def handle_add_model(app: 'TldwCli', event):
    def on_config_ready(config: Optional[Dict[str, Any]]):
        if config:
            # MISSING: Save to database
            orchestrator = get_orchestrator()
            model_id = orchestrator.create_model_config(
                name=config['name'],
                provider=config['provider'],
                model_id=config['model_id'],
                config=config['parameters']
            )
            # MISSING: Update UI model list
            await refresh_models_list(app)
```

### 4. Real-time Status Updates
**Location**: `UI/Evals_Window.py` → `update_evaluation_progress()`
**Issue**: Method exists but not called during evaluation
**Fix Required**: Wire progress callbacks to UI update method

### 5. Export Functionality
**Location**: `Event_Handlers/eval_events.py` → `handle_export_results()`
**Issue**: Export dialog not connected to orchestrator export
**Fix Required**: Implement file save dialog and call orchestrator.export_results()

## Implementation Roadmap

### Phase 1: Core Integration (Week 1)
**Goal**: Get basic evaluation flow working end-to-end

#### Day 1-2: Progress Tracking
- [ ] Implement progress callback in `handle_start_evaluation`
- [ ] Wire callback to `update_evaluation_progress` in UI
- [ ] Test with simple evaluation task

#### Day 3-4: Results Display
- [ ] Implement `refresh_results_table` function
- [ ] Connect to refresh button handler
- [ ] Add auto-refresh after evaluation completes

#### Day 5: Model Management
- [ ] Complete `handle_add_model` implementation
- [ ] Add model list refresh functionality
- [ ] Test model CRUD operations

### Phase 2: Enhanced Features (Week 2)
**Goal**: Complete all UI functionality

#### Day 1-2: Export Functionality
- [ ] Implement export file dialog
- [ ] Connect to orchestrator export methods
- [ ] Add format selection (CSV/JSON)

#### Day 3-4: Task Management
- [ ] Complete task upload flow
- [ ] Add task validation feedback
- [ ] Implement template selection

#### Day 5: Error Handling
- [ ] Add comprehensive error notifications
- [ ] Implement retry mechanisms
- [ ] Add validation for all inputs

### Phase 3: Polish & Testing (Week 3)
**Goal**: Production-ready system

#### Day 1-2: UI Polish
- [ ] Add loading states
- [ ] Implement smooth transitions
- [ ] Add keyboard shortcuts

#### Day 3-4: Integration Testing
- [ ] End-to-end evaluation tests
- [ ] Multi-provider testing
- [ ] Large dataset testing

#### Day 5: Documentation
- [ ] Update user guide
- [ ] Create video tutorials
- [ ] Update API documentation

## Specific Code Changes Required

### 1. Event_Handlers/eval_events.py

```python
# Add these missing implementations:

async def update_progress_ui(app: 'TldwCli', completed: int, total: int, result: Dict):
    """Update UI with evaluation progress."""
    evals_window = app.query_one(EvalsWindow)
    evals_window.update_evaluation_progress(
        run_id=app.active_run_id,
        completed=completed,
        total=total,
        current_result=result
    )

async def refresh_models_list(app: 'TldwCli'):
    """Refresh the models list in UI."""
    orchestrator = get_orchestrator()
    models = orchestrator.db.list_models()
    
    # Update models list widget
    models_list = app.query_one("#models-list")
    models_list.clear()
    for model in models:
        models_list.append(format_model_item(model))

async def refresh_tasks_list(app: 'TldwCli'):
    """Refresh the tasks list in UI."""
    orchestrator = get_orchestrator()
    tasks = orchestrator.db.list_tasks()
    
    # Update tasks dropdown
    task_select = app.query_one("#task-select")
    task_select.set_options([(t['id'], t['name']) for t in tasks])
```

### 2. UI/Evals_Window.py

```python
# Add these methods to EvalsWindow class:

@work(exclusive=True)
async def load_evaluation_results(self):
    """Load and display evaluation results."""
    orchestrator = get_orchestrator()
    runs = orchestrator.db.get_recent_runs(limit=50)
    
    await self.populate_results_table(runs)

async def populate_results_table(self, runs: List[Dict]):
    """Populate the results data table."""
    table = self.query_one("#results-table", DataTable)
    table.clear()
    
    # Add columns if not already added
    if not table.columns:
        table.add_columns("Run Name", "Model", "Task", "Score", "Status", "Duration")
    
    # Add rows
    for run in runs:
        table.add_row(
            run['name'],
            run['model_name'],
            run['task_name'],
            f"{run.get('score', 0):.2%}",
            run['status'],
            format_duration(run.get('duration', 0))
        )
```

### 3. Integration Helper Functions

Create `tldw_chatbook/Evals/ui_integration.py`:

```python
"""UI Integration helpers for evaluation system."""

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
```

## Quick Wins (Can Be Done Immediately)

### 1. Enable Basic Evaluation Flow (2 hours)
```python
# In eval_events.py, update handle_start_evaluation:
async def handle_start_evaluation(app: 'TldwCli', event):
    """Minimal working implementation."""
    try:
        # Get selected task and model
        task_id = app.query_one("#task-select").value
        model_id = app.query_one("#model-select").value
        
        if not task_id or not model_id:
            app.notify("Please select a task and model", severity="warning")
            return
        
        # Start evaluation
        orchestrator = get_orchestrator()
        run_id = await orchestrator.run_evaluation(
            task_id=task_id,
            model_id=model_id,
            max_samples=10  # Start small for testing
        )
        
        app.notify(f"Evaluation started: {run_id}", severity="information")
        
        # Refresh results after completion
        await refresh_results_list(app)
        
    except Exception as e:
        app.notify(f"Evaluation failed: {str(e)}", severity="error")
```

### 2. Display Results (1 hour)
```python
# In eval_events.py:
async def handle_refresh_results(app: 'TldwCli', event):
    """Display evaluation results."""
    orchestrator = get_orchestrator()
    runs = orchestrator.db.get_recent_runs(limit=10)
    
    # Simple display in status area
    results_text = "\n".join([
        f"{run['name']}: {run.get('score', 0):.2%}"
        for run in runs
    ])
    
    status_area = app.query_one("#results-status")
    status_area.update(results_text or "No results yet")
```

### 3. Enable Model Addition (1 hour)
```python
# Complete the model dialog callback:
async def _create_model_config(app: 'TldwCli', config: Dict[str, Any]):
    """Save model configuration."""
    orchestrator = get_orchestrator()
    model_id = orchestrator.create_model_config(**config)
    
    app.notify(f"Model added: {config['name']}", severity="success")
    
    # Update dropdown
    await refresh_model_dropdown(app)
```

## Success Criteria

### Functional Requirements
- [ ] Can upload and create evaluation tasks through UI
- [ ] Can configure and save model settings
- [ ] Can start evaluations with progress tracking
- [ ] Can view results in table format
- [ ] Can export results to CSV/JSON
- [ ] Can compare multiple evaluation runs

### Performance Requirements
- [ ] Progress updates at least every second
- [ ] Results load within 2 seconds
- [ ] Export completes within 5 seconds for 1000 results

### User Experience Requirements
- [ ] Clear error messages for all failure cases
- [ ] Loading states for all async operations
- [ ] Keyboard shortcuts for common actions
- [ ] Responsive UI during long operations

## Testing Strategy

### Integration Tests
1. **End-to-end Evaluation**
   - Upload task file
   - Configure model
   - Run evaluation
   - Verify results displayed

2. **Progress Tracking**
   - Start long evaluation
   - Verify progress updates
   - Test cancellation

3. **Export Functionality**
   - Run evaluation
   - Export to CSV
   - Verify file contents

### Manual Testing Checklist
- [ ] All buttons trigger appropriate actions
- [ ] Forms validate input correctly
- [ ] Progress bar updates smoothly
- [ ] Results refresh automatically
- [ ] Exports contain correct data
- [ ] Error messages are helpful

## Conclusion

The evaluation system is remarkably close to completion. The backend is fully functional and well-tested. The UI components exist and are well-designed. What remains is approximately 2-3 weeks of integration work to connect these pieces together.

The highest priority is establishing the basic evaluation flow (task selection → evaluation → results display). Once this works, the remaining features can be added incrementally.

With the backend complexity already handled, completing the UI integration is a straightforward engineering task that will unlock a powerful evaluation system for tldw_chatbook users.