# tldw_chatbook/Widgets/multi_item_review_window.py
"""
MultiItemReviewWindow widget for reviewing multiple media items and generating batch analyses.
Allows users to select multiple items, generate analyses with custom prompts, and save results.
"""

from typing import TYPE_CHECKING, List, Dict, Any, Optional
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Static, Button, Label, Input, ListView, ListItem, Markdown, Checkbox, TextArea, ProgressBar
from textual.reactive import reactive
from textual.worker import Worker, WorkerState
from textual.message import Message
from loguru import logger
import asyncio
from datetime import datetime

if TYPE_CHECKING:
    from ..app import TldwCli


class AnalysisGenerationEvent(Message):
    """Event for analysis generation progress updates."""
    
    def __init__(self, media_id: int, status: str, content: Optional[str] = None) -> None:
        super().__init__()
        self.media_id = media_id
        self.status = status  # "started", "completed", "error"
        self.content = content


class MultiItemReviewWindow(Container):
    """
    Window for reviewing multiple media items and generating batch analyses.
    """
    
    selected_items: reactive[List[Dict[str, Any]]] = reactive([])
    save_analyses: reactive[bool] = reactive(False)
    analysis_in_progress: reactive[bool] = reactive(False)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.analysis_results: Dict[int, str] = {}  # media_id -> analysis content
        self.generation_worker: Optional[Worker] = None
        
    def compose(self) -> ComposeResult:
        """Compose the UI structure."""
        with VerticalScroll(classes="multi-review-container"):
            # Top section - Search and selection
            with Container(classes="selection-section"):
                yield Label("Multi-Item Review", classes="section-title")
                
                # Search controls
                with Horizontal(classes="search-controls"):
                    yield Input(
                        placeholder="Search by title, content, or tags (comma-separated)...",
                        id="multi-review-search",
                        classes="search-input"
                    )
                    yield Button("Search", id="perform-search", variant="primary")
                    
                # Tag filter section
                with Horizontal(classes="tag-filter-section"):
                    yield Label("Filter by tags:", classes="filter-label")
                    yield Input(
                        placeholder="Enter tags separated by commas",
                        id="tag-filter-input",
                        classes="tag-input"
                    )
                    
                # Selection controls
                with Horizontal(classes="selection-controls"):
                    yield Button("Select All", id="select-all-items", classes="small-button")
                    yield Button("Clear Selection", id="clear-all-items", classes="small-button")
                    yield Static("0 items selected", id="selection-count", classes="selection-info")
                    
            # Middle section - Item list
            with Container(classes="items-section"):
                yield Label("Available Items", classes="section-title")
                yield ListView(id="review-items-list", classes="review-items-list")
                
            # Bottom section - Analysis controls and results
            with Container(classes="analysis-section"):
                yield Label("Analysis Generation", classes="section-title")
                
                # Analysis prompt
                yield Label("Analysis Prompt:", classes="prompt-label")
                yield TextArea(
                    "Please provide a comprehensive summary of this content, highlighting key points and insights.",
                    id="analysis-prompt",
                    classes="analysis-prompt"
                )
                
                # Analysis controls
                with Horizontal(classes="analysis-controls"):
                    yield Checkbox("Save analyses permanently", id="save-analyses-checkbox", value=False)
                    yield Button("Generate Analyses", id="generate-analyses", variant="success", disabled=True)
                    yield Button("Cancel", id="cancel-generation", variant="error", disabled=True)
                    
                # Progress indicator
                yield ProgressBar(id="analysis-progress", classes="analysis-progress hidden", show_eta=False)
                yield Static("", id="progress-label", classes="progress-label hidden")
                
                # Results area
                yield Label("Analysis Results", classes="section-title")
                with VerticalScroll(id="results-scroll", classes="results-container"):
                    yield Container(id="analysis-results", classes="analysis-results")
                    
    def on_mount(self) -> None:
        """Initialize the window when mounted."""
        # Set initial focus to search input
        self.query_one("#multi-review-search").focus()
        
    @on(Input.Submitted, "#multi-review-search")
    async def handle_search_submit(self, event: Input.Submitted) -> None:
        """Handle search input submission."""
        await self.perform_search()
        
    @on(Button.Pressed, "#perform-search")
    async def handle_search_button(self) -> None:
        """Handle search button press."""
        await self.perform_search()
        
    async def perform_search(self) -> None:
        """Perform search and populate the item list."""
        try:
            if not self.app_instance.media_db:
                logger.error("Media DB not available")
                return
                
            search_input = self.query_one("#multi-review-search", Input).value.strip()
            tag_filter = self.query_one("#tag-filter-input", Input).value.strip()
            
            # Parse tags if provided
            tags = [tag.strip() for tag in tag_filter.split(',') if tag.strip()] if tag_filter else None
            
            # Perform search
            results, total = self.app_instance.media_db.search_media_db(
                search_query=search_input if search_input else None,
                must_have_keywords=tags,
                search_fields=['title', 'content', 'author'],
                sort_by="ingestion_date_desc",
                page=1,
                results_per_page=100,  # Get more results for multi-item review
                include_trash=False,
                include_deleted=False
            )
            
            # Clear existing items
            list_view = self.query_one("#review-items-list", ListView)
            await list_view.clear()
            self.selected_items = []
            
            # Populate list with results
            for item in results:
                # Create rich list item
                title = item.get('title', 'Untitled')
                media_type = item.get('type', 'Unknown')
                date = item.get('ingestion_date', 'N/A')
                if isinstance(date, datetime):
                    date = date.strftime('%Y-%m-%d')
                elif isinstance(date, str) and 'T' in date:
                    date = date.split('T')[0]
                    
                # Get keywords for this item
                keywords = []
                if item.get('id'):
                    try:
                        from ..DB.Client_Media_DB_v2 import fetch_keywords_for_media
                        keywords = fetch_keywords_for_media(self.app_instance.media_db, item['id'])
                    except Exception:
                        pass
                        
                keywords_str = f" | Tags: {', '.join(keywords)}" if keywords else ""
                
                list_item = ListItem(
                    Vertical(
                        Label(f"[bold]{title}[/bold]", classes="item-title"),
                        Static(f"{media_type} | {date}{keywords_str}", classes="item-meta")
                    )
                )
                list_item.media_data = item
                list_item.add_class("review-item")
                
                await list_view.append(list_item)
                
            # Update count
            self.update_selection_count()
            
            if not results:
                await list_view.append(ListItem(Label("No items found matching your search criteria.")))
                
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            self.app_instance.notify(f"Error searching: {str(e)}", severity="error")
            
    @on(ListView.Selected, "#review-items-list")
    def handle_item_selection(self, event: ListView.Selected) -> None:
        """Handle item selection from list."""
        if hasattr(event.item, 'media_data'):
            item = event.item.media_data
            
            # Toggle selection
            if any(i['id'] == item['id'] for i in self.selected_items):
                self.selected_items = [i for i in self.selected_items if i['id'] != item['id']]
                event.item.remove_class("selected")
            else:
                self.selected_items.append(item)
                event.item.add_class("selected")
                
            self.update_selection_count()
            self.update_generate_button()
            
    @on(Button.Pressed, "#select-all-items")
    async def select_all_items(self) -> None:
        """Select all items in the list."""
        list_view = self.query_one("#review-items-list", ListView)
        self.selected_items = []
        
        for item in list_view.children:
            if isinstance(item, ListItem) and hasattr(item, 'media_data'):
                self.selected_items.append(item.media_data)
                item.add_class("selected")
                
        self.update_selection_count()
        self.update_generate_button()
        
    @on(Button.Pressed, "#clear-all-items")
    async def clear_all_items(self) -> None:
        """Clear all item selections."""
        list_view = self.query_one("#review-items-list", ListView)
        self.selected_items = []
        
        for item in list_view.children:
            if isinstance(item, ListItem):
                item.remove_class("selected")
                
        self.update_selection_count()
        self.update_generate_button()
        
    @on(Checkbox.Changed, "#save-analyses-checkbox")
    def handle_save_checkbox(self, event: Checkbox.Changed) -> None:
        """Handle save analyses checkbox change."""
        self.save_analyses = event.value
        
    def update_selection_count(self) -> None:
        """Update the selection count display."""
        count = len(self.selected_items)
        self.query_one("#selection-count", Static).update(f"{count} items selected")
        
    def update_generate_button(self) -> None:
        """Enable/disable generate button based on selection."""
        button = self.query_one("#generate-analyses", Button)
        button.disabled = len(self.selected_items) == 0 or self.analysis_in_progress
        
    @on(Button.Pressed, "#generate-analyses")
    async def handle_generate_analyses(self) -> None:
        """Start generating analyses for selected items."""
        if not self.selected_items or self.analysis_in_progress:
            return
            
        self.analysis_in_progress = True
        self.analysis_results.clear()
        
        # Update UI
        self.query_one("#generate-analyses", Button).disabled = True
        self.query_one("#cancel-generation", Button).disabled = False
        self.query_one("#analysis-progress", ProgressBar).remove_class("hidden")
        self.query_one("#progress-label", Static).remove_class("hidden")
        
        # Clear previous results
        results_container = self.query_one("#analysis-results", Container)
        await results_container.remove_children()
        
        # Get analysis prompt
        prompt = self.query_one("#analysis-prompt", TextArea).text
        
        # Start generation worker
        self.generation_worker = self.run_worker(
            self._generate_analyses_worker,
            prompt=prompt,
            items=self.selected_items.copy(),
            save=self.save_analyses
        )
        
    @work(thread=True)
    async def _generate_analyses_worker(self, prompt: str, items: List[Dict[str, Any]], save: bool) -> None:
        """Worker to generate analyses for multiple items."""
        total_items = len(items)
        progress_bar = self.query_one("#analysis-progress", ProgressBar)
        progress_label = self.query_one("#progress-label", Static)
        
        # Set up progress bar
        progress_bar.total = total_items
        
        for index, item in enumerate(items):
            if self.generation_worker and self.generation_worker.state == WorkerState.CANCELLED:
                break
                
            media_id = item['id']
            title = item.get('title', 'Untitled')
            
            # Update progress
            self.call_from_thread(progress_bar.update, completed=index)
            self.call_from_thread(
                progress_label.update,
                f"Analyzing {index + 1}/{total_items}: {title[:50]}..."
            )
            
            # Post start event
            self.post_message(AnalysisGenerationEvent(media_id, "started"))
            
            try:
                # Generate analysis using LLM
                analysis_content = await self._generate_single_analysis(item, prompt)
                
                if analysis_content:
                    # Store result
                    self.analysis_results[media_id] = analysis_content
                    
                    # Save to database if requested
                    if save and self.app_instance.media_db:
                        try:
                            # Update the analysis_content field in the database
                            update_query = """
                                UPDATE Media 
                                SET analysis_content = ?, last_modified = ?
                                WHERE id = ?
                            """
                            self.app_instance.media_db.execute_query(
                                update_query,
                                (analysis_content, datetime.now().isoformat(), media_id)
                            )
                            self.app_instance.media_db.commit()
                        except Exception as e:
                            logger.error(f"Error saving analysis for media {media_id}: {e}")
                            
                    # Post completion event
                    self.post_message(AnalysisGenerationEvent(media_id, "completed", analysis_content))
                else:
                    self.post_message(AnalysisGenerationEvent(media_id, "error", "Failed to generate analysis"))
                    
            except Exception as e:
                logger.error(f"Error generating analysis for media {media_id}: {e}")
                self.post_message(AnalysisGenerationEvent(media_id, "error", str(e)))
                
            # Small delay between requests to avoid rate limiting
            await asyncio.sleep(0.5)
            
        # Update final progress
        self.call_from_thread(progress_bar.update, completed=total_items)
        self.call_from_thread(progress_label.update, "Analysis generation complete!")
        
        # Reset UI state
        self.call_from_thread(self._reset_generation_ui)
        
    async def _generate_single_analysis(self, item: Dict[str, Any], prompt: str) -> Optional[str]:
        """Generate analysis for a single media item."""
        # Use the event handler's implementation
        from ..Event_Handlers.multi_item_review_events import generate_single_analysis
        return await generate_single_analysis(self.app_instance, item, prompt)
            
    def _reset_generation_ui(self) -> None:
        """Reset the UI after generation completes."""
        self.analysis_in_progress = False
        self.query_one("#generate-analyses", Button).disabled = len(self.selected_items) == 0
        self.query_one("#cancel-generation", Button).disabled = True
        self.query_one("#analysis-progress", ProgressBar).add_class("hidden")
        self.query_one("#progress-label", Static).add_class("hidden")
        
    @on(Button.Pressed, "#cancel-generation")
    def handle_cancel_generation(self) -> None:
        """Cancel the ongoing analysis generation."""
        if self.generation_worker:
            self.generation_worker.cancel()
            self.app_instance.notify("Analysis generation cancelled", severity="warning")
            
    @on(AnalysisGenerationEvent)
    async def handle_analysis_event(self, event: AnalysisGenerationEvent) -> None:
        """Handle analysis generation events."""
        if event.status == "completed" and event.content:
            # Find the media item
            item = next((i for i in self.selected_items if i['id'] == event.media_id), None)
            if item:
                # Add result to the results container
                results_container = self.query_one("#analysis-results", Container)
                
                # Create a result card
                with Container(classes="analysis-result-card") as card:
                    # Header with title and actions
                    with Horizontal(classes="result-header"):
                        title = item.get('title', 'Untitled')
                        card.mount(Label(f"[bold]{title}[/bold]", classes="result-title"))
                        
                    # Metadata
                    meta_info = f"Type: {item.get('type', 'Unknown')} | ID: {item['id']}"
                    if self.save_analyses:
                        meta_info += " | [green]Saved[/green]"
                    card.mount(Static(meta_info, classes="result-meta"))
                    
                    # Analysis content
                    card.mount(Markdown(event.content, classes="result-content"))
                    
                    # Separator
                    card.mount(Static("─" * 80, classes="result-separator"))
                    
                await results_container.mount(card)
                
                # Scroll to show the new result
                scroll_container = self.query_one("#results-scroll", VerticalScroll)
                scroll_container.scroll_end(animate=True)