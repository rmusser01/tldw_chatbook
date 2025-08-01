# Code-Gen-1: GitHub Repository File Selector Feature

## Table of Contents
1. [Feature Overview](#feature-overview)
2. [Core Functionality](#core-functionality)
3. [User Interface Design](#user-interface-design)
4. [Advanced Features](#advanced-features)
5. [Use Cases](#use-cases)
6. [Textual Implementation Plan](#textual-implementation-plan)

---

## Feature Overview

The GitHub Repository File Selector is a tool that allows users to browse, preview, and selectively download files from GitHub repositories without cloning the entire repository. Users can navigate through a visual tree structure, select specific files or folders, and export their selection in various formats.

### Key Benefits
- No need to clone entire repositories
- Visual file tree navigation
- Selective file extraction
- Multiple export formats
- Direct integration with embeddings and chat features

---

## Core Functionality

### 1. Repository Input Methods
- **Direct URL Input**: Accept GitHub URLs (e.g., `https://github.com/user/repo`)
- **GitHub Search**: Built-in repository search
- **Authentication**: Support for private repos via personal access tokens
- **Multi-Platform Support**: Eventually support GitLab, Bitbucket, and generic git URLs

### 2. Tree View Interface

```
📁 awesome-project
├── ☑️ 📁 src/
│   ├── ☑️ 📁 components/
│   │   ├── ☑️ Button.tsx
│   │   ├── ☐ Button.test.tsx
│   │   └── ☑️ Button.css
│   ├── ☑️ 📁 utils/
│   │   └── ☑️ helpers.ts
│   └── ☑️ index.ts
├── ☐ 📁 tests/
├── ☑️ README.md
├── ☑️ package.json
└── ☐ .gitignore
```

### 3. Smart Selection Features
- **Cascade Selection**: Select/deselect folders applies to all children
- **Quick Filters**: Pre-defined filters for common file types
- **Pattern Matching**: Regex-based file selection
- **File Type Filters**: Filter by extension (.py, .js, .md, etc.)
- **Size Filters**: Exclude files above certain size thresholds
- **.gitignore Awareness**: Option to respect or ignore .gitignore rules

### 4. Preview Capabilities
- **Quick Preview**: View file contents without downloading
- **Syntax Highlighting**: Language-aware code highlighting
- **Metadata Display**: File size, last modified, commit info
- **Binary File Handling**: Show appropriate placeholders for images/binaries

### 5. Export Options
- **ZIP Download**: Create a ZIP with selected files
- **Clipboard Copy**: Concatenated content with file markers
- **Directory Export**: Maintain folder structure locally
- **Markdown Export**: Single markdown file with all content
- **Direct to Embeddings**: Create embeddings from selection

---

## User Interface Design

### Main Interface Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ GitHub Repository File Selector                                 │
├─────────────────────────────────────────────────────────────────┤
│ Repository: [https://github.com/user/repo____] [Load] [Search] │
│ Branch: [main ▼] | 📊 23 files selected (1.2 MB)              │
├─────────────────────────────────────────────────────────────────┤
│ Filters: [🔍 Search...] [Type: All ▼] [Size: Any ▼]           │
│ Quick: [📝 Docs] [💻 Code] [⚙️ Config] [Select All] [Clear]    │
├──────────────────────────────────┬──────────────────────────────┤
│ 📁 Repository Files              │ 📄 Preview                   │
│ ├── ☑️ 📁 src/                   │ File: src/index.ts           │
│ │   ├── ☑️ index.ts              │ Size: 2.3 KB                 │
│ │   ├── ☑️ 📁 components/        │ Modified: 2 days ago         │
│ │   └── ☑️ 📁 utils/             │ ─────────────────────────    │
│ ├── ☑️ README.md                 │ import express from 'express'│
│ ├── ☑️ package.json              │ import { router } from './ro │
│ └── ☐ .gitignore                 │                              │
│                                  │ const app = express()        │
│                                  │ const PORT = 3000            │
├──────────────────────────────────┴──────────────────────────────┤
│ [Cancel] [Export as ZIP] [Copy to Clipboard] [Create Embeddings]│
└─────────────────────────────────────────────────────────────────┘
```

### Selection Summary Bar

```
Selected: 23 files (1.2 MB) | ~45,000 tokens | Export size: 890 KB
[Invert Selection] [Save Selection Profile] [Load Profile]
```

---

## Advanced Features

### 1. Smart Defaults
- **Auto-Selection**: Intelligently select important files based on project type
- **Exclusion Rules**: Auto-exclude build artifacts, dependencies, temp files
- **Learning**: Remember user preferences per language/framework

### 2. Version Control Integration
- **Branch Selection**: Choose any branch or tag
- **Commit Pinning**: Select files from specific commit
- **Diff View**: See changes between branches
- **History**: View file history and select versions

### 3. Batch Operations
- **Multi-Repository**: Select files from multiple repositories
- **Merge Operations**: Combine similar files across repos
- **Bulk Export**: Process multiple selections at once

### 4. Selection Profiles
- **Save Profiles**: Save selection patterns for reuse
- **Share Profiles**: Export/import selection configurations
- **Template Library**: Pre-built profiles for common scenarios

### 5. Integration Features
- **Direct Chat Import**: Add selected files to chat context
- **Embedding Creation**: Generate embeddings from selection
- **Documentation Generation**: Auto-generate docs from code
- **Code Analysis**: Run analysis on selected files

---

## Use Cases

### 1. Documentation Extraction
- Select all README and documentation files
- Extract API documentation from code comments
- Gather configuration examples

### 2. Code Review & Learning
- Extract specific modules for review
- Get all test files for a feature
- Study implementation patterns

### 3. Project Setup
- Extract boilerplate from existing projects
- Cherry-pick features from multiple repos
- Create custom project templates

### 4. Research & Analysis
- Gather similar implementations across projects
- Extract configuration files for comparison
- Build datasets for analysis

---

## Textual Implementation Plan

After reviewing Textual's layout constraints and capabilities, here's a detailed implementation plan for building this feature within the TUI framework.

### Architecture Overview

```
GitHubFileSelectorScreen
├── Header (docked top)
│   ├── RepositoryInput
│   ├── BranchSelector
│   └── SelectionSummary
├── FilterBar (docked below header)
│   ├── SearchInput
│   ├── TypeFilter
│   └── QuickFilterButtons
├── MainContent (grid layout: 2 columns)
│   ├── TreeContainer (column 1)
│   │   └── ScrollableContainer
│   │       └── TreeView (custom widget)
│   └── PreviewContainer (column 2)
│       └── ScrollableContainer
│           └── CodePreview
└── ActionBar (docked bottom)
    └── ExportButtons
```

### Key Implementation Components

#### 1. Custom Tree Widget

Since Textual doesn't have a built-in tree widget, we'll create one using nested containers:

```python
class TreeNode(Widget):
    """Single node in the tree view"""
    
    def __init__(self, path: str, is_directory: bool, level: int = 0):
        super().__init__()
        self.path = path
        self.is_directory = is_directory
        self.level = level
        self.expanded = reactive(False)
        self.selected = reactive(False)
        self.children_loaded = False
    
    def compose(self) -> ComposeResult:
        # Use grid for precise alignment
        with Container(classes="tree-node-row"):
            # Grid automatically places these in order
            # Column 1: Indentation
            yield Static(" " * (self.level * 2), classes="tree-indent")
            
            # Column 2: Expand/collapse button
            if self.is_directory:
                yield Button(
                    "▶" if not self.expanded else "▼",
                    classes="tree-expand-btn",
                    id=f"expand-{self.path}"
                )
            else:
                yield Static("", classes="tree-expand-spacer")
            
            # Column 3: Checkbox
            yield Checkbox(
                value=self.selected,
                id=f"select-{self.path}",
                classes="tree-checkbox"
            )
            
            # Column 4: Icon and name
            icon = "📁" if self.is_directory else self._get_file_icon()
            yield Static(
                f"{icon} {os.path.basename(self.path)}", 
                classes="tree-content"
            )
```

#### 2. Tree Container Implementation

```python
class TreeView(VerticalScroll):
    """Scrollable tree view container"""
    
    def __init__(self, repo_url: str):
        super().__init__()
        self.repo_url = repo_url
        self.nodes = {}  # path -> TreeNode mapping
        self.selection = reactive(set())  # Selected paths
    
    def compose(self) -> ComposeResult:
        # Root container for all nodes
        with Container(id="tree-nodes"):
            yield Static("Loading repository structure...")
    
    async def load_repository_structure(self):
        """Fetch and display repository tree"""
        # This would use GitHub API to get tree structure
        tree_data = await self.fetch_github_tree()
        await self.build_tree_view(tree_data)
```

#### 3. Layout Configuration

```css
/* Main screen layout */
GitHubFileSelectorScreen {
    layout: vertical;
}

/* Header section using grid for better alignment */
.repo-header {
    dock: top;
    height: 5;
    background: $boost;
    padding: 1;
}

.repo-header-content {
    layout: grid;
    grid-size: 4 2;  /* 4 columns, 2 rows */
    grid-columns: 3fr 1fr 1fr 1fr;  /* URL input gets most space */
    grid-rows: 3 2;  /* Input row taller than summary */
    grid-gutter: 1 1;
}

/* Filter bar using grid for precise control */
.filter-bar {
    dock: top;
    height: 3;
    background: $panel;
    layout: grid;
    grid-size: 6 1;  /* 6 columns for all filter elements */
    grid-columns: 2fr 1fr 20 20 20 1fr;  /* Search, dropdown, 3 buttons, spacer */
    grid-gutter: 0 1;
    padding: 0 1;
}

/* Main content area using grid layout */
.main-content {
    layout: grid;
    grid-size: 5 1;  /* 5 columns for golden ratio split */
    grid-columns: 2fr 3fr;  /* 40% tree, 60% preview */
    height: 1fr;
}

/* Tree container spans first 2 columns */
.tree-container {
    column-span: 2;
    border-right: solid $primary;
    overflow: hidden;
}

/* Preview container spans last 3 columns */
.preview-container {
    column-span: 3;
    padding: 1;
    overflow: hidden;
}

/* Tree node styling with grid for alignment */
.tree-node-row {
    layout: grid;
    grid-size: 4 1;  /* Indent, expand, checkbox, content */
    grid-columns: auto 3 3 1fr;
    height: 3;
    align: left middle;
}

.tree-expand-btn {
    width: 100%;
    height: 3;
    background: transparent;
    border: none;
}

/* Action bar with evenly spaced buttons */
.action-bar {
    dock: bottom;
    height: 3;
    background: $panel;
    layout: grid;
    grid-size: 7 1;  /* 7 columns for centering */
    grid-columns: 1fr auto auto auto auto auto 1fr;
    grid-gutter: 0 1;
    align: center middle;
    padding: 0 2;
}

/* Selection summary bar */
.selection-summary {
    grid-column-span: 4;  /* Spans all 4 columns in header */
    layout: grid;
    grid-size: 3 1;
    grid-columns: 2fr 1fr 1fr;
    align: left middle;
}
```

#### 4. Core Functionality Implementation

```python
class GitHubFileSelectorScreen(Screen):
    """Main screen for GitHub file selection"""
    
    BINDINGS = [
        ("ctrl+a", "select_all", "Select All"),
        ("ctrl+i", "invert_selection", "Invert"),
        ("ctrl+e", "export_files", "Export"),
        ("escape", "close_screen", "Close"),
    ]
    
    def compose(self) -> ComposeResult:
        # Header with grid layout
        with Container(classes="repo-header"):
            with Container(classes="repo-header-content"):
                # First row: input and controls
                yield Input(
                    placeholder="https://github.com/user/repo",
                    id="repo-url-input"
                )
                yield Button("Load", id="load-repo-btn")
                yield Select(
                    options=[("main", "main")],
                    id="branch-selector"
                )
                yield Button("⚙️", id="repo-settings-btn")
                
                # Second row: selection summary spans all columns
                with Container(classes="selection-summary"):
                    yield Static("No files selected", id="selection-count")
                    yield Static("0 KB", id="selection-size")
                    yield Static("~0 tokens", id="selection-tokens")
        
        # Filter bar - grid layout for precise alignment
        with Container(classes="filter-bar"):
            yield Input(placeholder="Search files...", id="file-search")
            yield Select(
                options=[
                    ("all", "All Files"),
                    ("code", "Code Only"),
                    ("docs", "Documentation"),
                    ("config", "Config Files"),
                ],
                id="file-type-filter"
            )
            yield Button("📝", id="filter-docs", classes="filter-quick-btn")
            yield Button("💻", id="filter-code", classes="filter-quick-btn")
            yield Button("⚙️", id="filter-config", classes="filter-quick-btn")
            yield Static()  # Spacer
        
        # Main content with grid layout
        with Container(classes="main-content"):
            # Tree view container
            with Container(classes="tree-container"):
                yield TreeView(id="repo-tree")
            
            # Preview pane container
            with Container(classes="preview-container"):
                with ScrollableContainer():
                    yield Static(
                        "Select a file to preview",
                        id="file-preview"
                    )
        
        # Action bar with grid for button spacing
        with Container(classes="action-bar"):
            yield Static()  # Left spacer
            yield Button("Cancel", variant="error", id="cancel-btn")
            yield Button("Export ZIP", id="export-zip-btn")
            yield Button("Copy to Clipboard", id="copy-btn")
            yield Button("Create Embeddings", variant="primary", id="embed-btn")
            yield Button("Save Profile", id="save-profile-btn")
            yield Static()  # Right spacer
```

#### 5. Event Handling

```python
@on(Button.Pressed, "#load-repo-btn")
async def load_repository(self, event: Button.Pressed) -> None:
    """Load repository structure from GitHub"""
    url_input = self.query_one("#repo-url-input", Input)
    repo_url = url_input.value
    
    if not self.validate_github_url(repo_url):
        self.notify("Invalid GitHub URL", severity="error")
        return
    
    tree_view = self.query_one("#repo-tree", TreeView)
    await tree_view.load_repository_structure()

@on(Checkbox.Changed)
async def handle_selection(self, event: Checkbox.Changed) -> None:
    """Handle file/folder selection"""
    # Extract path from checkbox ID
    path = event.checkbox.id.replace("select-", "")
    
    if event.value:
        await self.add_to_selection(path)
    else:
        await self.remove_from_selection(path)
    
    # Update summary
    self.update_selection_summary()

@on(Button.Pressed, ".tree-expand-btn")
async def toggle_expand(self, event: Button.Pressed) -> None:
    """Expand/collapse tree nodes"""
    path = event.button.id.replace("expand-", "")
    node = self.get_tree_node(path)
    
    if node:
        node.expanded = not node.expanded
        if node.expanded and not node.children_loaded:
            await self.load_node_children(node)
```

#### 6. Performance Optimizations

1. **Lazy Loading**: Load tree nodes on-demand when expanded
2. **Virtual Scrolling**: For large repositories, implement virtual scrolling
3. **Caching**: Cache API responses to avoid repeated fetches
4. **Batch Operations**: Group API calls when possible

```python
class VirtualTreeView(ScrollableContainer):
    """Optimized tree view with virtual scrolling"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.visible_nodes = []
        self.total_nodes = 0
        self.node_height = 3
        
    def on_scroll(self, event: events.Scroll) -> None:
        """Update visible nodes based on scroll position"""
        viewport_height = self.size.height
        scroll_offset = self.scroll_y
        
        start_index = scroll_offset // self.node_height
        end_index = start_index + (viewport_height // self.node_height) + 1
        
        self.render_visible_nodes(start_index, end_index)
```

#### 7. API Integration

```python
class GitHubAPIClient:
    """Handle GitHub API interactions"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token
        self.base_url = "https://api.github.com"
        self.session = httpx.AsyncClient()
    
    async def get_repository_tree(
        self, 
        owner: str, 
        repo: str, 
        branch: str = "main"
    ) -> dict:
        """Fetch repository tree structure"""
        url = f"{self.base_url}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        headers = {"Authorization": f"token {self.token}"} if self.token else {}
        
        response = await self.session.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    
    async def get_file_content(
        self, 
        owner: str, 
        repo: str, 
        path: str
    ) -> str:
        """Fetch file content for preview"""
        url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
        headers = {"Authorization": f"token {self.token}"} if self.token else {}
        
        response = await self.session.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        # Decode base64 content
        import base64
        content = base64.b64decode(data["content"]).decode("utf-8")
        return content
```

### Implementation Challenges & Solutions

1. **Tree Rendering Without Native Support**
   - Solution: Create custom tree widget using grid layout for precise alignment
   - Each tree node uses a 4-column grid (indent, expand, checkbox, content)
   - Use reactive attributes for expand/collapse state
   - Implement virtual scrolling for large trees

2. **Leveraging Grid Layout**
   - Solution: Use grid extensively for better control
   - Header: 4x2 grid for input and summary
   - Filter bar: 6x1 grid for consistent spacing
   - Main content: 5x1 grid with column-span for golden ratio
   - Action bar: 7x1 grid with spacers for centered buttons

3. **No Individual Cell Positioning**
   - Solution: Widgets placed in order, use column-span for larger elements
   - Empty Static widgets as spacers where needed
   - Careful widget ordering in compose method

4. **Performance with Large Repositories**
   - Solution: Implement lazy loading
   - Use pagination for API calls
   - Cache frequently accessed data
   - Show loading states during operations

### Testing Strategy

1. **Unit Tests**: Test individual components (TreeNode, API client)
2. **Integration Tests**: Test full workflow with mock API
3. **Performance Tests**: Test with large repositories
4. **UI Tests**: Use Textual's snapshot testing

### Future Enhancements

1. **Multi-Repository Support**: Browse multiple repos simultaneously
2. **Diff View**: Compare files across branches
3. **Search Integration**: Full-text search within files
4. **Template System**: Save and share selection patterns
5. **Plugin Architecture**: Allow custom processors/exporters

---

### Grid Layout Benefits

By leveraging Textual's grid layout system, we achieve:

1. **Precise Alignment**: Each UI element has its exact place
2. **Responsive Design**: Grid columns can use `fr` units for flexible sizing
3. **Consistent Spacing**: Grid gutters provide uniform spacing
4. **Simplified Code**: Less manual positioning and sizing calculations
5. **Better Performance**: Grid layout is optimized in Textual

Example of advanced grid usage for file info display:

```css
/* File info grid - responsive layout */
.file-info-grid {
    layout: grid;
    grid-size: 2 4;  /* 2 columns, 4 rows */
    grid-columns: auto 1fr;  /* Label column sizes to content */
    grid-gutter: 1 1;
}

/* Preview pane with metadata sidebar */
.preview-with-metadata {
    layout: grid;
    grid-size: 3 1;
    grid-columns: 1fr 1fr 1fr;  /* Even split */
}

.preview-content {
    column-span: 2;  /* Takes 2/3 of space */
}

.file-metadata {
    column-span: 1;  /* Takes 1/3 of space */
    border-left: solid $primary;
}
```

## Summary

This implementation plan provides a robust foundation for building a GitHub Repository File Selector within Textual's framework. By properly utilizing Textual's grid layout system, we achieve:

- **Precise UI control** through grid-based layouts
- **User-friendly tree navigation** with custom tree widget
- **Efficient handling** of large repositories  
- **Clear visual feedback** with consistent alignment
- **Multiple export options** for flexibility
- **Seamless integration** with existing tldw_chatbook features

The extensive use of grid layouts throughout the interface ensures consistent spacing, proper alignment, and responsive behavior while working within Textual's constraints. The modular architecture allows for incremental development and testing.