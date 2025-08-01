# ScraperBuilderWindow.py
# Description: Visual scraper builder UI for creating custom scraping rules
#
# This window provides an interactive interface for building custom scrapers
# without writing code. Users can:
# - Test CSS selectors in real-time
# - Preview extracted content
# - Configure extraction rules
# - Save and load scraper configurations
#
# Imports
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse
#
# Third-Party Imports
from textual import on, work
from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Horizontal, Vertical, Container, ScrollableContainer, Center
from textual.widgets import (
    Static, Button, Input, Select, TextArea, Label, 
    DataTable, TabbedContent, TabPane, ListView, ListItem,
    LoadingIndicator, Collapsible
)
from textual.reactive import reactive
from textual.worker import Worker, WorkerState
from textual.binding import Binding
from textual.events import Mount
from rich.syntax import Syntax
from rich.table import Table
from rich.panel import Panel
from rich import box
from loguru import logger
#
# Local Imports
from ..Widgets.form_components import FormBuilder, FormField, FormFieldSet
from ..Subscriptions.scrapers.custom_scraper import CustomScrapingPipeline
from ..Subscriptions.web_scraping_pipelines import ScrapingConfig
from ..Web_Scraping.Article_Extractor_Lib import get_page_content
from ..Constants import Colors
#
########################################################################################################################
#
# Scraper Builder Window
#
########################################################################################################################

class ScraperBuilderWindow(Screen):
    """Visual scraper builder for creating custom extraction rules."""
    
    CSS = """
    ScraperBuilderWindow {
        background: $background;
    }
    
    /* Layout */
    .builder-container {
        height: 100%;
        layout: grid;
        grid-size: 2 1;
        grid-columns: 1fr 1fr;
        grid-rows: 1fr;
        padding: 1;
    }
    
    .left-panel {
        height: 100%;
        border: solid $primary;
        padding: 1;
        overflow: auto;
    }
    
    .right-panel {
        height: 100%;
        border: solid $secondary;
        padding: 1;
        margin-left: 1;
        overflow: auto;
    }
    
    /* Header styling */
    .builder-header {
        height: 5;
        background: $surface;
        padding: 1;
        margin-bottom: 1;
    }
    
    .url-input-container {
        height: 3;
        margin-bottom: 1;
    }
    
    /* Selector testing */
    .selector-test {
        margin: 1 0;
        padding: 1;
        border: tall $accent;
    }
    
    .selector-input {
        width: 100%;
        margin-bottom: 1;
    }
    
    .test-result {
        height: 10;
        padding: 1;
        background: $surface;
        border: solid $primary;
        overflow: auto;
    }
    
    /* Preview panel */
    .preview-container {
        height: 100%;
        overflow: auto;
    }
    
    .preview-header {
        height: 3;
        background: $surface;
        padding: 0 1;
        margin-bottom: 1;
    }
    
    .html-preview {
        height: 50%;
        overflow: auto;
        margin-bottom: 1;
    }
    
    .extracted-preview {
        height: 50%;
        overflow: auto;
    }
    
    /* Action buttons */
    .action-buttons {
        dock: bottom;
        height: 3;
        padding: 1 0;
    }
    
    /* Status indicator */
    .status-indicator {
        height: 3;
        padding: 1;
        background: $surface;
        text-align: center;
    }
    
    /* Rule builder */
    .rule-item {
        margin: 1 0;
        padding: 1;
        border: tall $primary;
    }
    
    .rule-actions {
        dock: right;
        width: auto;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+t", "test_selector", "Test Selector"),
        Binding("ctrl+s", "save_config", "Save Config"),
        Binding("ctrl+l", "load_config", "Load Config"),
        Binding("ctrl+f", "fetch_page", "Fetch Page"),
        Binding("escape", "close", "Close"),
    ]
    
    # Reactive attributes
    current_url = reactive("")
    page_content = reactive("")
    is_loading = reactive(False)
    
    def __init__(self, url: Optional[str] = None):
        """
        Initialize scraper builder.
        
        Args:
            url: Optional URL to start with
        """
        super().__init__()
        self.url = url or ""
        self.extraction_rules = []
        self.form_builder = FormBuilder()
        self.page_html = ""
        self.page_text = ""
    
    def compose(self) -> ComposeResult:
        """Compose the scraper builder UI."""
        with Container(classes="builder-container"):
            # Left panel - Configuration
            with ScrollableContainer(classes="left-panel"):
                yield Static("Scraper Configuration Builder", classes="builder-header")
                
                # URL input
                with Container(classes="url-input-container"):
                    yield Label("Test URL:")
                    with Horizontal():
                        yield Input(
                            placeholder="https://example.com/article",
                            value=self.url,
                            id="test-url-input"
                        )
                        yield Button("Fetch", variant="primary", id="fetch-btn")
                
                # Status
                yield Container(id="status-container", classes="status-indicator")
                
                with TabbedContent():
                    # Selector Builder Tab
                    with TabPane("Selectors", id="selectors-tab"):
                        yield from self._compose_selector_builder()
                    
                    # Rules Tab
                    with TabPane("Extraction Rules", id="rules-tab"):
                        yield from self._compose_rules_builder()
                    
                    # Options Tab
                    with TabPane("Options", id="options-tab"):
                        yield from self._compose_options()
                
                # Action buttons
                with Horizontal(classes="action-buttons"):
                    yield Button("Save Configuration", variant="success", id="save-config-btn")
                    yield Button("Load Configuration", variant="primary", id="load-config-btn")
                    yield Button("Export as Code", variant="default", id="export-code-btn")
            
            # Right panel - Preview
            with Container(classes="right-panel"):
                yield Static("Live Preview", classes="preview-header")
                
                with TabbedContent():
                    # HTML Preview
                    with TabPane("HTML", id="html-preview-tab"):
                        yield TextArea(
                            id="html-preview",
                            language="html",
                            theme="monokai",
                            read_only=True,
                            show_line_numbers=True
                        )
                    
                    # Text Preview
                    with TabPane("Text", id="text-preview-tab"):
                        yield TextArea(
                            id="text-preview",
                            read_only=True
                        )
                    
                    # Extracted Data Preview
                    with TabPane("Extracted", id="extracted-preview-tab"):
                        yield Container(id="extracted-data-container")
    
    def _compose_selector_builder(self) -> ComposeResult:
        """Compose selector builder section."""
        # Quick selector templates
        with Collapsible("Common Selectors", collapsed=False):
            yield from self._create_selector_templates()
        
        # Custom selector testing
        with Container(classes="selector-test"):
            yield Label("Test CSS Selector:")
            yield Input(
                placeholder="article h1, .post-title, #title",
                id="test-selector-input",
                classes="selector-input"
            )
            yield Button("Test Selector", id="test-selector-btn")
            
            yield Label("Results:")
            yield Container(id="selector-test-results", classes="test-result")
        
        # Add to rules
        with Container():
            yield Label("Add as extraction rule:")
            yield Select(
                options=[
                    ("title", "Title"),
                    ("content", "Content"),
                    ("author", "Author"),
                    ("date", "Published Date"),
                    ("image", "Main Image"),
                    ("custom", "Custom Field")
                ],
                id="rule-type-select"
            )
            yield Input(
                placeholder="Custom field name",
                id="custom-field-name",
                classes="hidden"
            )
            yield Button("Add Rule", variant="primary", id="add-rule-btn")
    
    def _compose_rules_builder(self) -> ComposeResult:
        """Compose extraction rules section."""
        yield Label("Active Extraction Rules:")
        yield Container(id="rules-container")
        
        with Horizontal():
            yield Button("Clear All", variant="error", id="clear-rules-btn")
            yield Button("Test All Rules", variant="primary", id="test-all-rules-btn")
    
    def _compose_options(self) -> ComposeResult:
        """Compose options section."""
        with FormFieldSet("Page Processing"):
            yield FormField(
                "Remove Scripts",
                self.form_builder.create_switch("remove-scripts", default=True)
            )
            
            yield FormField(
                "Remove Styles", 
                self.form_builder.create_switch("remove-styles", default=True)
            )
            
            yield FormField(
                "Preserve Links",
                self.form_builder.create_switch("preserve-links", default=True)
            )
        
        with FormFieldSet("Content Filters"):
            yield Label("Exclude Selectors (one per line):")
            yield TextArea(
                placeholder=".advertisement\n.sidebar\n#comments",
                id="exclude-selectors",
                tab_behavior="indent"
            )
            
            yield Label("Text Processing:")
            yield Select(
                options=[
                    ("none", "No processing"),
                    ("clean", "Clean whitespace"),
                    ("markdown", "Convert to Markdown"),
                ],
                value="clean",
                id="text-processing"
            )
        
        with FormFieldSet("Advanced"):
            yield FormField(
                "Wait for JavaScript",
                self.form_builder.create_switch("wait-javascript", default=False)
            )
            
            yield FormField(
                "Wait Selector",
                Input(placeholder=".content-loaded", id="wait-selector")
            )
            
            yield FormField(
                "Custom Headers (JSON)",
                TextArea(
                    placeholder='{"User-Agent": "Custom Bot"}',
                    id="custom-headers",
                    tab_behavior="indent"
                )
            )
    
    def _create_selector_templates(self) -> ComposeResult:
        """Create common selector templates."""
        templates = [
            ("Article Title", "h1, article h1, .article-title, .post-title, .entry-title"),
            ("Article Content", "article, .article-content, .post-content, .entry-content, main"),
            ("Author", ".author, .by-author, .post-author, [rel='author']"),
            ("Date", "time, .date, .published, .post-date, [datetime]"),
            ("Images", "img, picture img, figure img"),
            ("Links", "a[href]"),
            ("Paragraphs", "p, article p"),
            ("Headers", "h1, h2, h3, h4, h5, h6"),
        ]
        
        for name, selector in templates:
            with Horizontal():
                yield Button(
                    name,
                    id=f"template-{name.lower().replace(' ', '-')}",
                    classes="template-btn"
                )
                yield Static(f"`{selector}`", classes="template-selector")
    
    async def on_mount(self):
        """Initialize when mounted."""
        self.update_status("Ready")
        
        if self.url:
            input_field = self.query_one("#test-url-input", Input)
            input_field.value = self.url
    
    def update_status(self, message: str, severity: str = "info"):
        """Update status indicator."""
        container = self.query_one("#status-container", Container)
        container.remove_children()
        
        # Choose style based on severity
        styles = {
            "info": "blue",
            "success": "green",
            "warning": "yellow",
            "error": "red"
        }
        style = styles.get(severity, "white")
        
        container.mount(Static(f"[{style}]{message}[/{style}]"))
    
    @on(Button.Pressed, "#fetch-btn")
    async def on_fetch_page(self):
        """Fetch and analyze the test page."""
        url_input = self.query_one("#test-url-input", Input)
        url = url_input.value.strip()
        
        if not url:
            self.update_status("Please enter a URL", "warning")
            return
        
        self.current_url = url
        self.run_worker(self.fetch_page_content(url))
    
    @work(thread=True)
    def fetch_page_content(self, url: str):
        """Fetch page content in background."""
        try:
            self.call_from_thread(self.update_status, "Fetching page...", "info")
            self.is_loading = True
            
            # Use existing article extractor
            result = get_page_content(url)
            
            if result and result.get('extraction_successful'):
                self.page_html = result.get('raw_html', '')
                self.page_text = result.get('extracted_text', '')
                
                self.call_from_thread(self.display_page_content)
                self.call_from_thread(
                    self.update_status, 
                    f"Page loaded successfully ({len(self.page_html)} bytes)", 
                    "success"
                )
            else:
                error = result.get('error', 'Unknown error')
                self.call_from_thread(
                    self.update_status,
                    f"Failed to fetch page: {error}",
                    "error"
                )
                
        except Exception as e:
            logger.error(f"Error fetching page: {str(e)}")
            self.call_from_thread(
                self.update_status,
                f"Error: {str(e)}",
                "error"
            )
        finally:
            self.is_loading = False
    
    def display_page_content(self):
        """Display fetched page content in preview."""
        # Update HTML preview
        html_preview = self.query_one("#html-preview", TextArea)
        html_preview.load_text(self.page_html[:50000])  # Limit size
        
        # Update text preview
        text_preview = self.query_one("#text-preview", TextArea)
        text_preview.load_text(self.page_text[:10000])  # Limit size
    
    @on(Button.Pressed, "#test-selector-btn")
    async def on_test_selector(self):
        """Test CSS selector on the page."""
        if not self.page_html:
            self.update_status("Please fetch a page first", "warning")
            return
        
        selector_input = self.query_one("#test-selector-input", Input)
        selector = selector_input.value.strip()
        
        if not selector:
            return
        
        self.run_worker(self.test_selector(selector))
    
    @work(thread=True)
    def test_selector(self, selector: str):
        """Test selector in background."""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(self.page_html, 'html.parser')
            elements = soup.select(selector)
            
            results = []
            for i, elem in enumerate(elements[:10]):  # Limit to 10 results
                # Get element info
                tag_name = elem.name
                attrs = ' '.join([f'{k}="{v}"' for k, v in elem.attrs.items()][:3])
                text = elem.get_text(strip=True)[:100]
                
                results.append({
                    'index': i + 1,
                    'tag': tag_name,
                    'attrs': attrs,
                    'text': text
                })
            
            self.call_from_thread(self.display_selector_results, results, len(elements))
            
        except Exception as e:
            logger.error(f"Error testing selector: {str(e)}")
            self.call_from_thread(
                self.update_status,
                f"Invalid selector: {str(e)}",
                "error"
            )
    
    def display_selector_results(self, results: List[Dict], total_count: int):
        """Display selector test results."""
        container = self.query_one("#selector-test-results", Container)
        container.remove_children()
        
        if not results:
            container.mount(Static("[yellow]No elements found[/yellow]"))
            return
        
        # Create results table
        table = Table(box=box.ROUNDED)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Tag", style="green")
        table.add_column("Attributes", style="blue")
        table.add_column("Text", style="white")
        
        for result in results:
            table.add_row(
                str(result['index']),
                result['tag'],
                result['attrs'],
                result['text'] + "..." if len(result['text']) == 100 else result['text']
            )
        
        # Add summary
        summary = f"\n[green]Found {total_count} elements[/green]"
        if total_count > 10:
            summary += f" [dim](showing first 10)[/dim]"
        
        container.mount(Static(table))
        container.mount(Static(summary))
    
    @on(Button.Pressed, "#add-rule-btn")
    async def on_add_rule(self):
        """Add extraction rule."""
        selector_input = self.query_one("#test-selector-input", Input)
        selector = selector_input.value.strip()
        
        if not selector:
            self.update_status("Please enter a selector", "warning")
            return
        
        rule_type = self.query_one("#rule-type-select", Select).value
        
        # Get custom field name if needed
        field_name = rule_type
        if rule_type == "custom":
            custom_name_input = self.query_one("#custom-field-name", Input)
            field_name = custom_name_input.value.strip()
            if not field_name:
                self.update_status("Please enter a custom field name", "warning")
                return
        
        # Add rule
        rule = {
            'field': field_name,
            'selector': selector,
            'type': rule_type,
            'extract': 'text',  # Default to text extraction
            'multiple': False
        }
        
        self.extraction_rules.append(rule)
        self.refresh_rules_display()
        
        # Clear inputs
        selector_input.value = ""
        if rule_type == "custom":
            self.query_one("#custom-field-name", Input).value = ""
    
    def refresh_rules_display(self):
        """Refresh the rules display."""
        container = self.query_one("#rules-container", Container)
        container.remove_children()
        
        for i, rule in enumerate(self.extraction_rules):
            with container:
                rule_widget = self.create_rule_widget(rule, i)
                container.mount(rule_widget)
    
    def create_rule_widget(self, rule: Dict, index: int) -> Container:
        """Create a widget for displaying a rule."""
        rule_container = Container(classes="rule-item")
        
        with rule_container:
            with Horizontal():
                with Vertical():
                    rule_container.mount(
                        Static(f"[bold]{rule['field']}[/bold]")
                    )
                    rule_container.mount(
                        Static(f"Selector: [cyan]{rule['selector']}[/cyan]")
                    )
                    rule_container.mount(
                        Static(f"Extract: {rule['extract']} | Multiple: {rule['multiple']}")
                    )
                
                with Horizontal(classes="rule-actions"):
                    rule_container.mount(
                        Button("Edit", id=f"edit-rule-{index}", classes="small")
                    )
                    rule_container.mount(
                        Button("Delete", id=f"delete-rule-{index}", classes="small", variant="error")
                    )
        
        return rule_container
    
    @on(Button.Pressed, "#test-all-rules-btn")
    async def on_test_all_rules(self):
        """Test all extraction rules."""
        if not self.page_html:
            self.update_status("Please fetch a page first", "warning")
            return
        
        if not self.extraction_rules:
            self.update_status("No rules to test", "warning")
            return
        
        self.run_worker(self.test_all_rules())
    
    @work(thread=True)
    def test_all_rules(self):
        """Test all rules and display results."""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(self.page_html, 'html.parser')
            results = {}
            
            for rule in self.extraction_rules:
                selector = rule['selector']
                field = rule['field']
                
                try:
                    elements = soup.select(selector)
                    
                    if not elements:
                        results[field] = None
                    elif rule['multiple']:
                        # Extract from all matching elements
                        if rule['extract'] == 'text':
                            results[field] = [elem.get_text(strip=True) for elem in elements]
                        elif rule['extract'] == 'href':
                            results[field] = [elem.get('href') for elem in elements if elem.get('href')]
                        elif rule['extract'] == 'src':
                            results[field] = [elem.get('src') for elem in elements if elem.get('src')]
                        else:
                            results[field] = [elem.get(rule['extract']) for elem in elements]
                    else:
                        # Extract from first matching element
                        elem = elements[0]
                        if rule['extract'] == 'text':
                            results[field] = elem.get_text(strip=True)
                        else:
                            results[field] = elem.get(rule['extract'])
                
                except Exception as e:
                    results[field] = f"Error: {str(e)}"
            
            self.call_from_thread(self.display_extraction_results, results)
            
        except Exception as e:
            logger.error(f"Error testing rules: {str(e)}")
            self.call_from_thread(
                self.update_status,
                f"Error testing rules: {str(e)}",
                "error"
            )
    
    def display_extraction_results(self, results: Dict[str, Any]):
        """Display extraction test results."""
        container = self.query_one("#extracted-data-container", Container)
        container.remove_children()
        
        # Create results panel
        content = ""
        for field, value in results.items():
            if value is None:
                content += f"[yellow]{field}:[/yellow] [dim]No data found[/dim]\n"
            elif isinstance(value, list):
                content += f"[yellow]{field}:[/yellow] [green]{len(value)} items[/green]\n"
                for i, item in enumerate(value[:3]):  # Show first 3
                    content += f"  - {item[:100]}{'...' if len(str(item)) > 100 else ''}\n"
                if len(value) > 3:
                    content += f"  [dim]... and {len(value) - 3} more[/dim]\n"
            else:
                content += f"[yellow]{field}:[/yellow] {str(value)[:200]}{'...' if len(str(value)) > 200 else ''}\n"
            content += "\n"
        
        panel = Panel(
            content.strip(),
            title="Extracted Data",
            border_style="green"
        )
        
        container.mount(Static(panel))
    
    @on(Button.Pressed, "#save-config-btn")
    async def on_save_config(self):
        """Save scraper configuration."""
        if not self.extraction_rules:
            self.update_status("No rules to save", "warning")
            return
        
        # Build configuration
        config = {
            'url': self.current_url,
            'rules': self.extraction_rules,
            'options': {
                'remove_scripts': self.query_one("#remove-scripts").value,
                'remove_styles': self.query_one("#remove-styles").value,
                'preserve_links': self.query_one("#preserve-links").value,
                'exclude_selectors': self.query_one("#exclude-selectors", TextArea).text.strip().split('\n'),
                'text_processing': self.query_one("#text-processing", Select).value,
                'wait_javascript': self.query_one("#wait-javascript").value,
                'wait_selector': self.query_one("#wait-selector", Input).value
            },
            'created_at': datetime.now().isoformat()
        }
        
        # Custom headers
        headers_text = self.query_one("#custom-headers", TextArea).text.strip()
        if headers_text:
            try:
                config['options']['custom_headers'] = json.loads(headers_text)
            except json.JSONDecodeError:
                self.update_status("Invalid JSON in custom headers", "error")
                return
        
        # Save to file
        domain = urlparse(self.current_url).netloc if self.current_url else "custom"
        filename = f"scraper_{domain.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # TODO: Show file save dialog
        # For now, save to a default location
        save_path = Path.home() / "Downloads" / filename
        save_path.write_text(json.dumps(config, indent=2))
        
        self.update_status(f"Configuration saved to {filename}", "success")
        self.notify(f"Scraper configuration saved to:\n{save_path}")
    
    @on(Select.Changed, "#rule-type-select")
    def on_rule_type_changed(self, event: Select.Changed):
        """Show/hide custom field name input."""
        custom_input = self.query_one("#custom-field-name", Input)
        if event.value == "custom":
            custom_input.remove_class("hidden")
        else:
            custom_input.add_class("hidden")
    
    @on(Button.Pressed, ".template-btn")
    async def on_template_clicked(self, event: Button.Pressed):
        """Insert template selector."""
        # Find the corresponding selector
        template_name = event.button.id.replace("template-", "").replace("-", " ").title()
        
        templates = {
            "Article Title": "h1, article h1, .article-title, .post-title, .entry-title",
            "Article Content": "article, .article-content, .post-content, .entry-content, main",
            "Author": ".author, .by-author, .post-author, [rel='author']",
            "Date": "time, .date, .published, .post-date, [datetime]",
            "Images": "img, picture img, figure img",
            "Links": "a[href]",
            "Paragraphs": "p, article p",
            "Headers": "h1, h2, h3, h4, h5, h6",
        }
        
        selector = templates.get(template_name, "")
        if selector:
            selector_input = self.query_one("#test-selector-input", Input)
            selector_input.value = selector
    
    @on(Button.Pressed, "#export-code-btn")
    async def on_export_code(self):
        """Export configuration as Python code."""
        if not self.extraction_rules:
            self.update_status("No rules to export", "warning")
            return
        
        # Generate Python code
        code = self.generate_python_code()
        
        # Save to file
        filename = f"scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        save_path = Path.home() / "Downloads" / filename
        save_path.write_text(code)
        
        self.update_status(f"Code exported to {filename}", "success")
        self.notify(f"Python code exported to:\n{save_path}")
    
    def generate_python_code(self) -> str:
        """Generate Python code from configuration."""
        code = '''# Generated by Scraper Builder
from tldw_chatbook.Subscriptions.web_scraping_pipelines import ScrapingConfig
from tldw_chatbook.Subscriptions.scrapers.custom_scraper import CustomScrapingPipeline

# Scraper configuration
config = ScrapingConfig(
    url="{url}",
    pipeline_type="custom",
    selectors={{
{selectors}
    }},
    options={{
{options}
    }}
)

# Create and run scraper
scraper = CustomScrapingPipeline(config)
items = await scraper.scrape()

# Process results
for item in items:
    print(f"Title: {{item.title}}")
    print(f"URL: {{item.url}}")
    print(f"Content: {{item.content[:200]}}...")
    print("---")
'''.format(
            url=self.current_url,
            selectors=self._format_selectors_code(),
            options=self._format_options_code()
        )
        
        return code
    
    def _format_selectors_code(self) -> str:
        """Format selectors for code generation."""
        lines = []
        for rule in self.extraction_rules:
            lines.append(f'        "{rule["field"]}": {{')
            lines.append(f'            "selector": "{rule["selector"]}",')
            lines.append(f'            "extract": "{rule["extract"]}",')
            lines.append(f'            "multiple": {rule["multiple"]}')
            lines.append('        },')
        return '\n'.join(lines).rstrip(',')
    
    def _format_options_code(self) -> str:
        """Format options for code generation."""
        options = {
            'remove_scripts': self.query_one("#remove-scripts").value,
            'remove_styles': self.query_one("#remove-styles").value,
            'preserve_links': self.query_one("#preserve-links").value,
            'text_processing': self.query_one("#text-processing", Select).value,
        }
        
        lines = []
        for key, value in options.items():
            if isinstance(value, bool):
                lines.append(f'        "{key}": {value},')
            else:
                lines.append(f'        "{key}": "{value}",')
        
        return '\n'.join(lines).rstrip(',')
    
    def action_close(self):
        """Close the scraper builder."""
        self.app.pop_screen()


# End of ScraperBuilderWindow.py