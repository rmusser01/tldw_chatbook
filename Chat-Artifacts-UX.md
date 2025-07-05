# Chat Artifacts UX Design Document

## Overview

This document outlines the design and implementation plan for adding a Claude-style Artifacts feature to the `tldw_chatbook` application's chat interface. The feature will allow AI-generated code, documents, and other structured content to be displayed in dedicated tabs alongside the conversation, providing better visibility, interaction, and export capabilities.

## Goals

1. **Enhanced Content Visibility**: Display code and documents in a dedicated area with proper formatting and syntax highlighting
2. **Persistent Reference**: Keep artifacts visible across the conversation for easy reference
3. **Better Interaction**: Allow users to copy, save, and work with generated content easily
4. **Improved UX**: Separate "content to work on" from "conversation about that content"
5. **Extensibility**: Support multiple artifact types (code, markdown, HTML, SVG, etc.)

## Current Architecture Analysis

### Chat Window Structure
- **Main Container**: `Chat_Window_Enhanced.py` implements the chat UI
- **Layout**: Uses a vertical layout with:
  - Left sidebar (settings)
  - Main content area (chat log + input area)
  - Right sidebar (character details)
- **Message Display**: `ChatMessageEnhanced` widget handles individual messages with:
  - Image support
  - Action buttons (edit, copy, speak, etc.)
  - Role-based styling

### Key Components
1. **Message Widget** (`chat_message_enhanced.py`):
   - Already supports rich content display
   - Has action buttons for message interaction
   - Supports image attachments

2. **Streaming Support** (`chat_streaming_events.py`):
   - Handles real-time text streaming from LLMs
   - Updates message widgets during generation
   - Manages completion states

3. **Event System**:
   - Uses Textual's event system for decoupled communication
   - Custom events for chat operations
   - Worker threads for async operations

### Existing Tab Usage
- The app uses `TabbedContent` widget in:
  - `IngestTldwApiTabbedWindow.py` for media ingestion forms
  - Main app navigation (different feature tabs)
- Pattern established for tab-based interfaces

## Proposed Implementation

### 1. Artifacts Container Widget

Create a new widget that manages multiple artifacts using Textual's `TabbedContent`:

```python
# tldw_chatbook/Widgets/artifacts_container.py

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import TabbedContent, TabPane, Button, Static
from textual.reactive import reactive
from typing import Dict, List, Optional

class ArtifactsContainer(Container):
    """Container for displaying multiple artifacts in tabs."""
    
    # Track artifacts by ID
    artifacts: reactive[Dict[str, 'Artifact']] = reactive({})
    visible: reactive[bool] = reactive(False)
    
    def compose(self) -> ComposeResult:
        with Container(id="artifacts-header"):
            yield Static("Artifacts")
            yield Button("×", id="close-artifacts")
        
        yield TabbedContent(id="artifacts-tabs")
    
    def add_artifact(self, artifact_id: str, title: str, content: str, 
                    artifact_type: str, language: Optional[str] = None) -> None:
        """Add a new artifact or update existing one."""
        # Implementation details...
    
    def remove_artifact(self, artifact_id: str) -> None:
        """Remove an artifact."""
        # Implementation details...
```

### 2. Modified Chat Layout

Update the chat window to include a collapsible artifacts panel:

```python
# Modified compose method in Chat_Window_Enhanced.py

def compose(self) -> ComposeResult:
    # Settings Sidebar (Left)
    yield from create_settings_sidebar(TAB_CHAT, self.app_instance.app_config)
    
    # Main Chat Area with Artifacts
    with Container(id="chat-content-wrapper"):
        # Chat conversation area
        with Container(id="chat-conversation-area"):
            yield VerticalScroll(id="chat-log")
            # ... input area components ...
        
        # Artifacts panel (initially hidden)
        yield ArtifactsContainer(
            id="chat-artifacts-panel",
            classes="collapsed"
        )
    
    # Character Details Sidebar (Right)
    yield from create_chat_right_sidebar(...)
```

### 3. Artifact Detection System

Create a service to detect and extract artifacts from AI responses:

```python
# tldw_chatbook/Chat/artifact_detector.py

import re
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

@dataclass
class DetectedArtifact:
    content: str
    artifact_type: str  # 'code', 'markdown', 'html', etc.
    language: Optional[str] = None
    title: Optional[str] = None
    start_pos: int = 0
    end_pos: int = 0

class ArtifactDetector:
    """Detects and extracts artifacts from AI responses."""
    
    # Patterns for different artifact types
    CODE_BLOCK_PATTERN = re.compile(
        r'```(\w+)?\n(.*?)\n```', 
        re.DOTALL | re.MULTILINE
    )
    
    # Future: Support for explicit artifact markers
    ARTIFACT_MARKER_PATTERN = re.compile(
        r'<artifact type="(\w+)"(?:\s+title="([^"]+)")?>(.*?)</artifact>',
        re.DOTALL
    )
    
    @classmethod
    def detect_artifacts(cls, text: str) -> List[DetectedArtifact]:
        """Detect all artifacts in the given text."""
        artifacts = []
        
        # Detect code blocks
        for match in cls.CODE_BLOCK_PATTERN.finditer(text):
            language = match.group(1) or 'text'
            content = match.group(2)
            
            # Only consider substantial code blocks as artifacts
            if len(content.strip().split('\n')) >= 10:
                artifacts.append(DetectedArtifact(
                    content=content,
                    artifact_type='code',
                    language=language,
                    title=cls._generate_title(content, language),
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        # Future: Detect explicit artifact markers
        # ...
        
        return artifacts
    
    @staticmethod
    def _generate_title(content: str, language: str) -> str:
        """Generate a title for an artifact based on its content."""
        # Implementation to extract meaningful title
        # (e.g., function name, class name, first comment, etc.)
        lines = content.strip().split('\n')
        for line in lines:
            if line.strip() and not line.strip().startswith('#'):
                return f"{language}: {line[:30]}..."
        return f"{language} code"
```

### 4. Enhanced Streaming Events

Modify the streaming event handlers to detect and route artifacts:

```python
# Enhanced chat_streaming_events.py

from ...Chat.artifact_detector import ArtifactDetector

async def handle_stream_done(self, event: StreamDone) -> None:
    """Enhanced handler that detects artifacts in completed streams."""
    # ... existing code ...
    
    if not event.error and event.full_text:
        # Detect artifacts in the completed response
        artifacts = ArtifactDetector.detect_artifacts(event.full_text)
        
        if artifacts:
            # Get or create artifacts container
            artifacts_container = self.query_one("#chat-artifacts-panel", ArtifactsContainer)
            
            # Add each detected artifact
            for i, artifact in enumerate(artifacts):
                artifact_id = f"msg-{ai_widget.message_id_internal}-artifact-{i}"
                artifacts_container.add_artifact(
                    artifact_id=artifact_id,
                    title=artifact.title,
                    content=artifact.content,
                    artifact_type=artifact.artifact_type,
                    language=artifact.language
                )
            
            # Show artifacts panel if hidden
            if artifacts and not artifacts_container.visible:
                artifacts_container.show()
                self.notify(f"Created {len(artifacts)} artifact(s)")
            
            # Add visual indicator to message
            ai_widget.add_class("has-artifacts")
            ai_widget.artifact_count = len(artifacts)
    
    # ... rest of existing code ...
```

### 5. Artifact Tab Widget

Create individual artifact display widgets:

```python
# tldw_chatbook/Widgets/artifact_tab.py

from rich.syntax import Syntax
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Static, Button, TextArea
from textual.reactive import reactive

class ArtifactTab(Container):
    """Widget for displaying a single artifact."""
    
    content: reactive[str] = reactive("")
    artifact_type: reactive[str] = reactive("text")
    language: reactive[Optional[str]] = reactive(None)
    
    def compose(self) -> ComposeResult:
        # Toolbar
        with Container(id="artifact-toolbar"):
            yield Button("📋 Copy", id="copy-artifact")
            yield Button("💾 Save", id="save-artifact")
            yield Button("🔄 Refresh", id="refresh-artifact")
        
        # Content area
        with VerticalScroll(id="artifact-content"):
            if self.artifact_type == "code" and self.language:
                # Syntax highlighted code
                yield Static(
                    Syntax(self.content, self.language, theme="monokai"),
                    id="artifact-display"
                )
            else:
                # Plain text or other content
                yield TextArea(
                    self.content,
                    id="artifact-display",
                    read_only=True
                )
```

### 6. Message Enhancement

Add artifact indicators and controls to chat messages:

```python
# Enhanced ChatMessageEnhanced widget

class ChatMessageEnhanced(Widget):
    # ... existing code ...
    
    artifact_count: reactive[int] = reactive(0)
    
    def compose(self) -> ComposeResult:
        # ... existing message composition ...
        
        # Add artifact indicator if present
        if self.artifact_count > 0:
            with Container(classes="artifact-indicator"):
                yield Static(f"📎 {self.artifact_count} artifact(s)")
                yield Button("View", id="view-artifacts", classes="mini-button")
```

### 7. CSS Styling

Add styles for the artifacts panel:

```css
/* artifacts.tcss */

#chat-content-wrapper {
    layout: horizontal;
    width: 100%;
    height: 100%;
}

#chat-conversation-area {
    width: 100%;
    height: 100%;
}

#chat-artifacts-panel {
    width: 50%;
    height: 100%;
    border-left: solid $primary;
    background: $surface;
    display: none;
}

#chat-artifacts-panel.visible {
    display: block;
}

/* Smooth transitions */
#chat-conversation-area {
    transition: width 200ms ease-out;
}

#chat-artifacts-panel.visible ~ #chat-conversation-area {
    width: 50%;
}

/* Artifact tabs styling */
#artifacts-tabs {
    height: 100%;
}

.artifact-indicator {
    background: $surface-lighten-1;
    padding: 0 1;
    margin: 1 0;
    height: 3;
}

/* Code display */
#artifact-display {
    padding: 1;
    background: $surface-darken-1;
}
```

## User Interaction Flow

### Creating Artifacts

1. User sends a message requesting code/document generation
2. AI responds with content
3. System automatically detects artifacts (code blocks, documents)
4. Artifacts appear in tabs on the right side
5. Message shows artifact indicator

### Viewing Artifacts

1. Click on artifact tab to switch between artifacts
2. Artifacts remain visible across conversation
3. Toggle button to show/hide artifacts panel
4. Visual indicators on messages with artifacts

### Working with Artifacts

1. **Copy**: Click copy button to copy artifact content
2. **Save**: Save artifact to file with appropriate extension
3. **Edit**: Future - allow in-place editing
4. **Export**: Export all artifacts from conversation

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- [ ] Create `ArtifactsContainer` widget
- [ ] Create `ArtifactTab` widget  
- [ ] Modify chat window layout
- [ ] Add CSS styling

### Phase 2: Detection & Display (Week 2)
- [ ] Implement `ArtifactDetector` service
- [ ] Integrate with streaming events
- [ ] Add artifact indicators to messages
- [ ] Test with various content types

### Phase 3: User Actions (Week 3)
- [ ] Implement copy functionality
- [ ] Implement save to file
- [ ] Add keyboard shortcuts
- [ ] Add artifact management UI

### Phase 4: Persistence & Enhancement (Week 4)
- [ ] Optional: Add artifacts table to database
- [ ] Implement artifact versioning
- [ ] Add export functionality
- [ ] Performance optimization

## Technical Considerations

### Performance
- Lazy loading for large artifacts
- Virtualization for many tabs
- Efficient syntax highlighting
- Debounced artifact detection during streaming

### Accessibility
- Keyboard navigation between artifacts
- Screen reader support
- High contrast mode support
- Clear visual indicators

### Edge Cases
- Very large artifacts (>1000 lines)
- Multiple artifacts in single message
- Artifacts in edited messages
- Handling artifacts during message regeneration

## Future Enhancements

1. **Rich Artifact Types**:
   - Interactive HTML/CSS/JS previews
   - SVG rendering
   - Markdown preview with live editing
   - JSON/YAML viewers with folding

2. **Collaboration Features**:
   - Share artifacts via link
   - Version control integration
   - Diff view for artifact changes

3. **AI Integration**:
   - "Improve this code" actions
   - Artifact-aware responses
   - Code execution (sandboxed)

4. **Advanced Management**:
   - Artifact library across conversations
   - Search within artifacts
   - Artifact templates

## Conclusion

This implementation plan provides a robust foundation for adding Claude-style Artifacts to the tldw_chatbook application. The tab-based approach integrates well with the existing UI patterns while providing a powerful new feature for working with AI-generated content. The phased implementation allows for iterative development and testing, ensuring a high-quality user experience.