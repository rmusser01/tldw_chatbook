"""
Search Result Component

Enhanced container for displaying individual search results
"""

from typing import Dict, Any
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Static, Button

from .constants import SOURCE_ICONS, SOURCE_COLORS


class SearchResult(Container):
    """Enhanced container for displaying a single search result with better visual design"""
    
    def __init__(self, result: Dict[str, Any], index: int):
        super().__init__(id=f"result-{index}", classes="search-result-card-enhanced")
        self.result = result
        self.index = index
        self.expanded = False
        
    def compose(self) -> ComposeResult:
        """Create the enhanced result display"""
        source = self.result.get('source', 'unknown')
        source_icon = SOURCE_ICONS.get(source, "📄")
        source_color = SOURCE_COLORS.get(source, "white")
        
        with Container(classes="result-card-wrapper"):
            # Left side - Source indicator
            with Vertical(classes="result-source-column"):
                yield Static(source_icon, classes=f"source-icon source-{source}")
                yield Static(source.upper(), classes=f"source-label source-{source}")
            
            # Main content area
            with Vertical(classes="result-content-column"):
                # Header with title and score
                with Horizontal(classes="result-header-enhanced"):
                    yield Static(
                        f"[bold]{self.result['title']}[/bold]",
                        classes="result-title-enhanced"
                    )
                    # Score visualization
                    score = self.result.get('score', 0)
                    yield Container(
                        Static(f"{score:.1%}", classes="score-text"),
                        classes=f"score-indicator score-{'high' if score > 0.7 else 'medium' if score > 0.4 else 'low'}"
                    )
                
                # Content preview with better formatting
                content = self.result.get('content', '')
                content_preview = content[:250] + "..." if len(content) > 250 else content
                yield Static(content_preview, classes="result-preview-enhanced")
                
                # Metadata pills
                if self.result.get('metadata'):
                    with Horizontal(classes="metadata-pills"):
                        for key, value in list(self.result['metadata'].items())[:3]:
                            if value and str(value).strip():
                                yield Static(
                                    f"{key}: {str(value)[:30]}{'...' if len(str(value)) > 30 else ''}",
                                    classes="metadata-pill"
                                )
                        if len(self.result['metadata']) > 3:
                            yield Static(
                                f"+{len(self.result['metadata']) - 3} more",
                                classes="metadata-pill more"
                            )
                
                # Expanded content (initially hidden)
                with Container(id=f"expanded-{self.index}", classes="result-expanded-content hidden"):
                    # Full content
                    yield Static("[bold]Full Content:[/bold]", classes="expanded-section-title")
                    yield Static(content, classes="result-full-content")
                    
                    # Citations if available
                    if self.result.get('citations'):
                        yield Static("[bold]Sources:[/bold]", classes="expanded-section-title")
                        citations = self.result['citations']
                        
                        # Group citations by document
                        citations_by_doc = {}
                        for citation in citations:
                            doc_title = citation.get('document_title', 'Unknown')
                            if doc_title not in citations_by_doc:
                                citations_by_doc[doc_title] = []
                            citations_by_doc[doc_title].append(citation)
                        
                        # Display citations grouped by document
                        for doc_title, doc_citations in citations_by_doc.items():
                            yield Static(f"📄 {doc_title}", classes="citation-doc-title")
                            for citation in doc_citations:
                                match_type = citation.get('match_type', 'unknown')
                                confidence = citation.get('confidence', 0)
                                text_snippet = citation.get('text', '')[:100] + '...' if len(citation.get('text', '')) > 100 else citation.get('text', '')
                                
                                citation_text = f"  • [{match_type}] ({confidence:.0%}) {text_snippet}"
                                yield Static(citation_text, classes=f"citation-item citation-{match_type}")
                    
                    # Full metadata
                    if self.result.get('metadata'):
                        yield Static("[bold]All Metadata:[/bold]", classes="expanded-section-title")
                        for key, value in self.result['metadata'].items():
                            yield Static(f"• {key}: {value}", classes="metadata-full-item")
                
                # Action bar
                with Horizontal(classes="result-actions-enhanced"):
                    yield Button(
                        "🔽 View" if not self.expanded else "🔼 Hide",
                        id=f"expand-{self.index}",
                        classes="result-button view-button"
                    )
                    yield Button("📋 Copy", id=f"copy-{self.index}", classes="result-button")
                    yield Button("📝 Note", id=f"add-note-{self.index}", classes="result-button")
                    yield Button("📤 Export", id=f"export-{self.index}", classes="result-button")