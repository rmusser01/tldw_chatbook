# Evaluation System Implementation Scratch Pad

## Todo List
- [ ] Complete event handler implementations
- [ ] Wire up database queries
- [ ] Implement widget classes
- [ ] Add dialog implementations
- [ ] Connect orchestrator to UI
- [ ] Add real-time progress updates
- [ ] Implement cost tracking
- [ ] Add export functionality

## Code Snippets

### Event Handler Template
```python
def get_available_providers(app_instance) -> List[str]:
    """Get list of configured LLM providers."""
    providers = []
    api_settings = app_instance.app_config.get("api_settings", {})
    
    # Check each provider
    if api_settings.get("openai", {}).get("api_key"):
        providers.append("openai")
    if api_settings.get("anthropic", {}).get("api_key"):
        providers.append("anthropic")
    # ... etc
    
    return providers
```

### Widget Structure Template
```python
class MetricsDisplay(Container):
    """Display evaluation metrics in a grid."""
    
    def __init__(self, metrics: Dict[str, float], **kwargs):
        super().__init__(**kwargs)
        self.metrics = metrics
    
    def compose(self) -> ComposeResult:
        with Grid(classes="metrics-grid"):
            for name, value in self.metrics.items():
                yield MetricCard(name, value)
```

### Database Query Examples
```python
# Get evaluation runs
def get_evaluation_runs(db_path: str, limit: int = 50) -> List[Dict]:
    with EvalsDB(db_path) as db:
        runs = db.get_runs(limit=limit)
        return [run.to_dict() for run in runs]

# Get run results
def get_run_results(db_path: str, run_id: str) -> Dict:
    with EvalsDB(db_path) as db:
        results = db.get_run_results(run_id)
        return results
```

### Progress Update Pattern
```python
# In eval runner
async def run_evaluation_with_progress(self, task, model, callback):
    total = len(task.samples)
    for i, sample in enumerate(task.samples):
        # Process sample
        result = await self.process_sample(sample, model)
        
        # Send progress update
        await callback({
            'current': i + 1,
            'total': total,
            'percent': (i + 1) / total * 100,
            'current_sample': sample.id,
            'status': 'running'
        })
```

### Reactive Pattern for View Updates
```python
# In EvalsWindow
evaluation_results = reactive([])

def watch_evaluation_results(self, results: List[Dict]):
    """Update UI when results change."""
    results_table = self.query_one("#results-table", ResultsTable)
    results_table.update_data(results)
```

## CSS Classes Needed

### Grid Layouts
```css
.eval-setup-grid {
    grid-size: 2;
    grid-gutter: 2;
}

.metrics-grid {
    grid-size: 3;
    grid-gutter: 1;
}

.model-grid {
    grid-size: 4;
    grid-rows: 3;
}
```

### Status Indicators
```css
.status-running { color: $warning; }
.status-completed { color: $success; }
.status-error { color: $error; }
.status-cancelled { color: $text-muted; }
```

## Integration Points

### From UI to Backend
1. `on_run_evaluation` → `eval_orchestrator.start_evaluation()`
2. Progress callback → `post_message(EvaluationProgress)`
3. Completion → `post_message(EvaluationCompleted)`

### From Backend to UI
1. Database changes → Reactive updates
2. Cost calculations → Cost widget updates
3. Results available → Results table populates

## Error Cases to Handle
1. API key not configured
2. Model not available
3. Dataset format invalid
4. Network timeout
5. Insufficient quota/credits
6. Database write failure
7. User cancellation

## Testing Checklist
- [ ] Can start evaluation
- [ ] Progress updates in real-time
- [ ] Results display correctly
- [ ] Cost tracking accurate
- [ ] Error handling works
- [ ] Export functions work
- [ ] Navigation between views
- [ ] Responsive on different screen sizes

## Performance Notes
- Use `work` decorator for heavy operations
- Implement pagination for large result sets
- Cache provider/model lists
- Debounce cost calculations
- Use virtual scrolling for long lists

## Accessibility Checklist
- [ ] All buttons have labels
- [ ] Status updates announced
- [ ] Keyboard navigation works
- [ ] Focus management correct
- [ ] Color not sole indicator
- [ ] Loading states clear

## Next Steps
1. Implement basic event handlers
2. Create simple widget implementations
3. Test basic flow (select → run → view)
4. Add advanced features incrementally
5. Polish UI/UX based on testing