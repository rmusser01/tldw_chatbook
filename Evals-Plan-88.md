# Evals UI Rebuild Plan - Following Textual Best Practices

## Executive Summary
Complete architectural rebuild of the Evaluation UI system using modern Textual patterns, focusing on maintainability, performance, and user experience.

## Current Issues Identified
1. **Monolithic Structure**: Single 441-line file mixing UI, business logic, and event handling
2. **Poor Separation of Concerns**: UI components directly manipulating state without proper abstraction
3. **Inconsistent Event Handling**: Mixed patterns of event handling and direct method calls
4. **Limited Reusability**: Components tightly coupled, difficult to test or reuse
5. **No Proper State Management**: Reactive attributes used inconsistently, no centralized state
6. **CSS Embedded in Python**: Styling mixed with logic, making maintenance difficult

## Proposed Architecture

### 1. Screen Architecture
```
tldw_chatbook/UI/Evals/
├── __init__.py
├── evals_screen.py          # Main screen extending Screen
├── evals_state.py           # Centralized state management
├── evals_messages.py        # Custom message definitions
└── evals_workers.py         # Background operation handlers
```

### 2. Widget Components
```
tldw_chatbook/Widgets/Evals/
├── __init__.py
├── task_configuration.py    # TaskConfigWidget - task setup
├── model_selector.py        # ModelSelectorWidget - model management
├── dataset_manager.py       # DatasetManagerWidget - dataset handling
├── evaluation_runner.py     # EvaluationRunnerWidget - execution display
├── results_dashboard.py     # ResultsDashboardWidget - results view
├── cost_estimator.py        # CostEstimatorWidget - cost tracking
└── progress_tracker.py      # ProgressTrackerWidget - progress display
```

### 3. State Management System
```python
class EvaluationState:
    """Centralized reactive state using Textual patterns"""
    current_config = reactive({})
    evaluation_status = reactive("idle")
    selected_model = reactive(None)
    selected_dataset = reactive(None)
    results = reactive([])
    progress = reactive(0.0)
```

### 4. Message System
```python
# Custom messages for component communication
class TaskSelected(Message)
class EvaluationStarted(Message)
class ProgressUpdate(Message)
class ResultsReady(Message)
class ErrorOccurred(Message)
```

### 5. Worker Patterns
```python
@work(exclusive=True, thread=True)
async def run_evaluation(self, config):
    """Background evaluation execution"""
    
@work(exclusive=True)
async def load_dataset(self, path):
    """Async dataset loading"""
```

## Implementation Steps

### Phase 1: Foundation (Week 1)
1. **Create base screen structure**
   - `EvaluationScreen(Screen)` with proper composition
   - Implement `compose()` method with container hierarchy
   - Set up basic layout with VerticalScroll

2. **Implement state management**
   - Create `EvaluationState` class with reactive attributes
   - Implement watchers for state changes
   - Add compute methods for derived state

3. **Define message system**
   - Create custom message classes
   - Implement message handlers
   - Set up bubbling patterns

### Phase 2: Core Components (Week 2)
4. **Build reusable widgets**
   - TaskConfigWidget with form validation
   - ModelSelectorWidget with provider support
   - DatasetManagerWidget with upload/import
   - ResultsDashboardWidget with metrics display

5. **Implement workers**
   - Evaluation runner with progress tracking
   - Dataset loader with validation
   - Result exporter with format options
   - Cost calculator with real-time updates

### Phase 3: Integration (Week 3)
6. **Connect components**
   - Wire up message passing between widgets
   - Implement state synchronization
   - Add error boundaries and recovery

7. **Form validation system**
   - Input validators for each field
   - Real-time validation feedback
   - Submit button state management

### Phase 4: Polish (Week 4)
8. **Modular CSS**
   - Extract styles to separate TCSS files
   - Implement theme variables
   - Add responsive design patterns

9. **Testing suite**
   - Unit tests for each widget
   - Integration tests for workflows
   - Property-based testing for state management

10. **Documentation**
    - API documentation for each component
    - Usage examples and patterns
    - Migration guide from old system

## Key Design Patterns

### 1. Composition Over Inheritance
```python
class EvaluationScreen(Screen):
    def compose(self) -> ComposeResult:
        with Container():
            yield TaskConfigWidget()
            yield ModelSelectorWidget()
            yield EvaluationRunnerWidget()
```

### 2. Reactive State Management
```python
class TaskConfigWidget(Widget):
    task_name = reactive("")
    
    def watch_task_name(self, old: str, new: str):
        self.post_message(TaskNameChanged(new))
```

### 3. Message-Driven Communication
```python
@on(TaskSelected)
def handle_task_selection(self, message: TaskSelected):
    self.state.current_task = message.task
    self.update_cost_estimation()
```

### 4. Worker-Based Concurrency
```python
@work(exclusive=True)
async def load_results(self):
    async with self.db.get_connection() as conn:
        results = await conn.fetch_results()
        self.call_from_thread(self.update_display, results)
```

## File Structure
```
tldw_chatbook/
├── UI/
│   └── Evals/
│       ├── evals_screen.py (250 lines)
│       ├── evals_state.py (100 lines)
│       ├── evals_messages.py (50 lines)
│       └── evals_workers.py (150 lines)
├── Widgets/
│   └── Evals/
│       ├── task_configuration.py (200 lines)
│       ├── model_selector.py (150 lines)
│       ├── dataset_manager.py (200 lines)
│       ├── evaluation_runner.py (250 lines)
│       ├── results_dashboard.py (300 lines)
│       ├── cost_estimator.py (100 lines)
│       └── progress_tracker.py (100 lines)
├── css/
│   └── features/
│       ├── _evaluation_screen.tcss (200 lines)
│       ├── _evaluation_widgets.tcss (300 lines)
│       └── _evaluation_responsive.tcss (100 lines)
└── Tests/
    └── Evals/
        ├── test_evals_screen.py
        ├── test_evals_widgets.py
        ├── test_evals_state.py
        └── test_evals_integration.py
```

## Success Metrics
- **Code Quality**: 90%+ test coverage, no cyclomatic complexity > 10
- **Performance**: UI updates < 16ms, background operations non-blocking
- **Maintainability**: Clear separation of concerns, documented APIs
- **User Experience**: Responsive UI, clear progress indication, error recovery

## Migration Strategy
1. Build new system in parallel (no breaking changes)
2. Feature flag to switch between old/new UI
3. Gradual rollout with user feedback
4. Deprecate old system after validation

## Benefits of New Architecture

### 1. Maintainability
- Clear separation of concerns
- Single responsibility principle for each component
- Easy to locate and fix issues
- Modular structure enables parallel development

### 2. Testability
- Isolated components are easier to unit test
- Mock message passing for integration tests
- State management can be tested independently
- Worker patterns enable async testing

### 3. Performance
- Reactive attributes minimize unnecessary updates
- Workers prevent UI blocking
- Efficient message passing reduces overhead
- CSS separation improves rendering

### 4. Developer Experience
- Clear file organization
- Consistent patterns throughout
- Self-documenting code structure
- Easy onboarding for new developers

### 5. User Experience
- Responsive UI during long operations
- Clear progress indication
- Graceful error handling
- Consistent visual design

## Technical Decisions

### Why Separate Screen from Widgets?
- **Screen** manages overall layout and navigation
- **Widgets** handle specific functionality
- Enables widget reuse across different screens
- Simplifies testing and maintenance

### Why Centralized State?
- Single source of truth for application state
- Predictable state updates
- Easy debugging with state inspection
- Enables time-travel debugging if needed

### Why Custom Messages?
- Type-safe communication between components
- Clear contracts for component interaction
- Enables loose coupling
- Supports async message handling

### Why Workers for Operations?
- Prevents UI freezing during long operations
- Enables progress reporting
- Supports cancellation
- Improves perceived performance

## Implementation Priority

### High Priority (Must Have)
1. Base screen structure
2. State management
3. Core widgets (task, model, dataset)
4. Basic evaluation runner
5. Essential CSS styling

### Medium Priority (Should Have)
6. Advanced workers
7. Cost estimation
8. Progress tracking
9. Results dashboard
10. Form validation

### Low Priority (Nice to Have)
11. Export functionality
12. Template system
13. Comparison features
14. Advanced metrics
15. Responsive design

## Risk Mitigation

### Technical Risks
- **Risk**: Complex state synchronization
- **Mitigation**: Use proven reactive patterns, extensive testing

- **Risk**: Performance issues with large datasets
- **Mitigation**: Implement pagination, virtual scrolling

- **Risk**: Worker deadlocks
- **Mitigation**: Use exclusive workers, timeout mechanisms

### Process Risks
- **Risk**: Scope creep
- **Mitigation**: Strict adherence to phases, regular reviews

- **Risk**: Breaking existing functionality
- **Mitigation**: Feature flags, parallel implementation

## Conclusion

This architectural rebuild will transform the Evals UI from a monolithic, tightly-coupled system into a modern, maintainable, and extensible application. By following Textual best practices and established design patterns, we create a foundation that can evolve with future requirements while maintaining code quality and developer productivity.

The phased approach ensures we can deliver value incrementally while maintaining system stability. Each phase builds upon the previous one, creating a robust evaluation system that serves both current needs and future growth.