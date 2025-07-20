# Evaluation System UX Design Plan

## Overview
The evaluation system provides a comprehensive interface for benchmarking and testing LLM models. The design follows established patterns from the MediaWindow and other tabs, ensuring consistency across the application.

## Design Principles
1. **Progressive Disclosure**: Start with simple options, reveal advanced features as needed
2. **Real-time Feedback**: Show progress, costs, and results as they happen
3. **Error Prevention**: Validate inputs before expensive operations
4. **Consistency**: Use existing UI patterns and components
5. **Efficiency**: Minimize clicks for common tasks

## Navigation Structure

### Left Navigation Pane
- **Evaluation Setup** (Default view) ✅
- **Results Dashboard** ✅
- **Model Management** ✅
- **Dataset Management** ✅

## Current Implementation Status

### ✅ Completed Features

1. **Navigation System**
   - Sidebar navigation with collapsible pane
   - View switching without screen pushing
   - Active view highlighting
   - Responsive layout

2. **Evaluation Setup View**
   - Provider/model cascade selection
   - Task type selection (Q&A, coding, etc.)
   - Dataset selection dropdown
   - Live cost estimation widget
   - Progress tracker (hidden until evaluation starts)
   - Recent runs table with DataTable widget

3. **Results Dashboard View**
   - Two-column layout (history + details)
   - Run history list with search and filter
   - Metrics display with dynamic cards
   - Results table placeholder
   - Export action buttons

4. **Model Management View**
   - Provider list (ListView)
   - Model details panel
   - Connection testing buttons
   - Model configuration actions

5. **Dataset Management View**
   - Dataset list with search
   - Dataset preview area
   - Upload/import buttons
   - Validation actions

6. **Core Widgets Implemented**
   - `CostEstimationWidget` - Real-time cost tracking
   - `ProgressTracker` - Evaluation progress with ETA
   - `MetricsDisplay` - Grid of metric cards
   - `MetricCard` - Individual metric visualization

7. **Event Handling**
   - Provider → Model population
   - Selection → Cost updates
   - Run selection → Details display
   - Progress updates during evaluation

### 🔄 Partially Implemented

1. **Evaluation Execution**
   - Orchestrator connected but falls back to simulation
   - Progress callbacks wired up
   - Cost tracking updates in real-time

2. **Data Loading**
   - Sample data used instead of real database
   - Hardcoded provider/model lists
   - Mock evaluation results

### ❌ Not Yet Implemented

1. **Dialogs**
   - File picker for dataset upload
   - Advanced configuration dialog
   - Template selector dialog
   - Export format selection
   - Model configuration dialog

2. **Database Integration**
   - Load providers from actual config
   - Query real evaluation history
   - Save evaluation results
   - Dataset persistence

3. **Export Functionality**
   - CSV export of results
   - JSON export for analysis
   - PDF report generation

4. **Comparison Features**
   - Multi-run selection
   - Side-by-side metrics
   - Diff visualization

5. **Advanced Features**
   - Batch evaluation support
   - Custom metric definitions
   - Evaluation scheduling
   - Budget limits and warnings

## Work That Needs to Be Done

### High Priority Tasks

1. **Complete Database Integration**
   ```python
   # Replace sample data with real queries
   - Load providers from app_config
   - Query Evals_DB for run history
   - Save new evaluation runs
   - Load datasets from database
   ```

2. **Implement File Dialogs**
   ```python
   # Create dialog classes
   - DatasetFilePickerDialog
   - AdvancedConfigDialog
   - TemplateSelectorDialog
   - ExportDialog
   ```

3. **Wire Up Real Orchestrator**
   ```python
   # Connect to actual evaluation engine
   - Test with real LLM providers
   - Handle API errors gracefully
   - Implement retry logic
   ```

4. **Add Export Functionality**
   ```python
   # Implement export handlers
   - CSV: Results table → CSV file
   - JSON: Full run data → JSON
   - PDF: Generate formatted report
   ```

### Medium Priority Tasks

1. **Enhance Error Handling**
   - API connection failures
   - Invalid dataset formats
   - Token limit exceeded
   - Budget exceeded warnings

2. **Implement Comparison View**
   - Multi-select in run history
   - Comparison dialog/view
   - Metric diff calculations
   - Visual comparison charts

3. **Add Dataset Validation**
   - Format checking
   - Sample preview
   - Statistics display
   - Error reporting

### Low Priority Tasks

1. **Performance Optimizations**
   - Virtual scrolling for large datasets
   - Pagination for run history
   - Caching for provider/model lists

2. **Advanced Features**
   - Evaluation templates
   - Custom scoring functions
   - Webhook notifications
   - Team collaboration

## Technical Debt to Address

1. **Hardcoded Values**
   - Sample count (100) should be configurable
   - Token estimates need real calculation
   - Provider filtering needs fixing

2. **Error Handling**
   - Many try/except blocks catch all exceptions
   - Need specific error types
   - Better error recovery

3. **Async/Await Patterns**
   - Some async methods not properly awaited
   - Worker thread usage needs review
   - Progress callback threading

## User Experience Improvements

1. **Visual Feedback**
   - Loading states for data fetching
   - Skeleton screens while loading
   - Success/error animations

2. **Keyboard Shortcuts**
   - Quick navigation between views
   - Start/stop evaluation
   - Export shortcuts

3. **Tooltips and Help**
   - Hover help for settings
   - First-time user guide
   - Inline documentation

## Testing Requirements

1. **Unit Tests**
   - Widget rendering
   - Event handler logic
   - Cost calculations

2. **Integration Tests**
   - Full evaluation flow
   - Database operations
   - Export functionality

3. **User Acceptance Tests**
   - Common workflows
   - Error scenarios
   - Performance benchmarks

## View Designs

### 1. Evaluation Setup View
**Purpose**: Configure and launch evaluations

**Layout**: Two-column with actions
```
┌─────────────────────────────────────────────┐
│ 🚀 Evaluation Setup                         │
├─────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────────────────┐ │
│ │ Quick Setup │ │ Configuration Details   │ │
│ │             │ │                         │ │
│ │ - Provider  │ │ [Selected Config Info]  │ │
│ │ - Model     │ │                         │ │
│ │ - Dataset   │ │ Cost Estimate: $X.XX    │ │
│ │ - Task Type │ │                         │ │
│ │             │ │ [Advanced Options]      │ │
│ │ [Templates] │ │                         │ │
│ │             │ │ [Run Button]            │ │
│ └─────────────┘ └─────────────────────────┘ │
│                                             │
│ ┌─────────────────────────────────────────┐ │
│ │ Active Evaluations (Progress Tracker)   │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

**Key Components**:
- **Quick Setup Panel**: Select dropdowns for provider, model, dataset, task
- **Template Buttons**: Pre-configured evaluation setups
- **Configuration Details**: Shows selected options, cost estimate
- **Advanced Options**: Collapsible section for power users
- **Run Controls**: Start/pause/stop buttons with validation
- **Progress Tracker**: Real-time status of running evaluations

### 2. Results Dashboard View
**Purpose**: View and analyze evaluation results

**Layout**: Master-detail with filters
```
┌─────────────────────────────────────────────┐
│ 📊 Results Dashboard                        │
├─────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────────────────┐ │
│ │ Run History │ │ Run Details             │ │
│ │             │ │                         │ │
│ │ [Filter]    │ │ Summary Stats           │
│ │             │ │ - Accuracy: XX%         │ │
│ │ Run #1      │ │ - Cost: $X.XX           │ │
│ │ Run #2 ←    │ │ - Duration: XX min      │ │
│ │ Run #3      │ │                         │ │
│ │             │ │ [Metrics Chart]         │ │
│ │             │ │                         │ │
│ │             │ │ [Results Table]         │ │
│ │             │ │                         │ │
│ │ [Compare]   │ │ [Export] [Share]        │ │
│ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────┘
```

**Key Components**:
- **Run History List**: Sortable/filterable list of evaluation runs
- **Run Summary**: Key metrics at a glance
- **Metrics Visualization**: Charts for visual analysis
- **Results Table**: Detailed results with sorting/filtering
- **Action Buttons**: Compare runs, export data, share results

### 3. Model Management View
**Purpose**: Configure LLM providers and models

**Layout**: Provider list with model grid
```
┌─────────────────────────────────────────────┐
│ 🤖 Model Management                         │
├─────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────────────────┐ │
│ │ Providers   │ │ Models for [Provider]   │ │
│ │             │ │                         │ │
│ │ ○ OpenAI    │ │ ┌─────┐ ┌─────┐ ┌─────┐│ │
│ │ ● Anthropic │ │ │GPT-4│ │GPT-3│ │Ada  ││ │
│ │ ○ Google    │ │ └─────┘ └─────┘ └─────┘│ │
│ │ ○ Local     │ │                         │ │
│ │             │ │ Model Details:          │ │
│ │ [+ Add]     │ │ - Context: 128k         │ │
│ │             │ │ - Cost: $X/1k tokens   │ │
│ │             │ │ - Status: ✓ Connected   │ │
│ │             │ │                         │ │
│ │             │ │ [Test] [Configure]      │ │
│ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────┘
```

**Key Components**:
- **Provider List**: Radio selection of configured providers
- **Model Grid**: Visual cards for available models
- **Model Details**: Selected model specifications
- **Action Buttons**: Test connection, configure settings
- **Add Provider**: Button to add new provider configuration

### 4. Dataset Management View
**Purpose**: Upload and manage evaluation datasets

**Layout**: Dataset list with preview
```
┌─────────────────────────────────────────────┐
│ 📚 Dataset Management                       │
├─────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────────────────┐ │
│ │ Datasets    │ │ Dataset Preview         │ │
│ │             │ │                         │ │
│ │ [Search]    │ │ Name: MMLU-Physics      │ │
│ │             │ │ Type: Multiple Choice   │ │
│ │ MMLU        │ │ Size: 1,234 samples     │ │
│ │ HumanEval   │ │                         │ │
│ │ Custom-1 ←  │ │ Sample Preview:         │ │
│ │             │ │ ┌─────────────────────┐ │ │
│ │ [Upload]    │ │ │ Q: What is...       │ │ │
│ │ [Import]    │ │ │ A) Option 1         │ │ │
│ │             │ │ │ B) Option 2         │ │ │
│ │             │ │ └─────────────────────┘ │ │
│ │             │ │                         │ │
│ │             │ │ [Validate] [Edit]       │ │
│ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────┘
```

**Key Components**:
- **Dataset List**: Searchable list of available datasets
- **Dataset Info**: Metadata about selected dataset
- **Sample Preview**: Shows example questions/tasks
- **Upload Button**: Add new datasets
- **Import Button**: Import from standard sources
- **Action Buttons**: Validate format, edit dataset

## Widget Specifications

### Progress Tracker Widget
- Shows active evaluation progress
- Real-time updates via WebSocket/events
- Displays: progress bar, ETA, current sample, cost tracking
- Actions: pause/resume, stop, view details

### Cost Estimation Widget
- Live cost calculation based on selections
- Shows: estimated total, cost per sample, budget warnings
- Updates as configuration changes
- Color coding: green (under budget), yellow (near limit), red (over budget)

### Results Table Widget
- Sortable columns: metric, value, sample count
- Filterable by metric type, threshold
- Expandable rows for detailed breakdowns
- Export functionality (CSV, JSON)

### Metrics Chart Widget
- Line/bar charts for metrics over samples
- Interactive: hover for details, zoom
- Multiple metrics on same chart
- Export as image

## Interaction Patterns

### Starting an Evaluation
1. User selects provider → Models populate
2. User selects model → Cost estimates update
3. User selects dataset → Compatibility check
4. User clicks "Run" → Validation → Confirmation → Start

### Viewing Results
1. Click run in history → Details load
2. Automatic refresh if run is active
3. Click metric → Detailed breakdown
4. Click "Compare" → Select other runs → Comparison view

### Error Handling
- Inline validation messages
- Toast notifications for async errors
- Detailed error logs accessible
- Suggested fixes for common issues

## Responsive Behavior
- Navigation pane collapses on narrow screens
- Tables become cards on mobile
- Charts resize appropriately
- Touch-friendly controls

## Accessibility
- Keyboard navigation throughout
- Screen reader announcements for updates
- High contrast mode support
- Focus indicators on all interactive elements

## Performance Considerations
- Lazy load large result sets
- Virtual scrolling for long lists
- Debounced search inputs
- Cached API responses
- Progressive enhancement

## Future Enhancements
1. **Batch Operations**: Run multiple evaluations
2. **Scheduling**: Schedule evaluations for later
3. **Collaboration**: Share results with team
4. **Custom Metrics**: Define custom evaluation metrics
5. **A/B Testing**: Compare model versions
6. **Report Builder**: Custom report generation