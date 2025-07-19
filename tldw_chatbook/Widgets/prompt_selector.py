# tldw_chatbook/Widgets/prompt_selector.py
"""
Reusable prompt selector widget for analysis prompts with dropdown and text area.
"""

from typing import TYPE_CHECKING, Optional, List, Tuple, Dict, Any
from textual.app import ComposeResult
from textual.containers import Vertical, Container
from textual.widgets import Label, Select, TextArea, Static
from textual import on
from loguru import logger

if TYPE_CHECKING:
    from ..app import TldwCli

class PromptSelector(Container):
    """
    A reusable widget for selecting and editing analysis prompts.
    
    Provides dropdowns for selecting pre-configured prompts and text areas
    for viewing/editing the selected prompts.
    """
    
    DEFAULT_CSS = """
    PromptSelector {
        layout: vertical;
        height: auto;
        margin: 1 0;
    }
    
    PromptSelector .prompt-select {
        margin: 0 0 1 0;
        width: 100%;
    }
    
    PromptSelector .prompt-textarea {
        height: 8;
        margin: 0 0 1 0;
    }
    
    PromptSelector .prompt-label {
        margin: 0 0 0 0;
    }
    """
    
    def __init__(
        self, 
        app_instance: 'TldwCli',
        system_prompt_id: str,
        user_prompt_id: str,
        media_type: str = "general",
        **kwargs
    ):
        """
        Initialize the prompt selector.
        
        Args:
            app_instance: Reference to the main app
            system_prompt_id: ID for the system prompt textarea
            user_prompt_id: ID for the user prompt textarea
            media_type: Type of media for filtering prompts (audio, video, pdf, etc.)
        """
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.system_prompt_id = system_prompt_id
        self.user_prompt_id = user_prompt_id
        self.media_type = media_type
        self.prompts_cache: Dict[str, Dict[str, str]] = {}
        logger.debug(f"PromptSelector initialized for {media_type}")
    
    def compose(self) -> ComposeResult:
        """Compose the prompt selector UI."""
        # System Prompt Section
        yield Label("System Prompt (for analysis):", classes="prompt-label")
        yield Select(
            [("Custom", "custom"), ("Loading...", Select.BLANK)],
            id=f"{self.system_prompt_id}-select",
            classes="prompt-select",
            prompt="Select a system prompt template...",
            value="custom"
        )
        yield TextArea(
            id=self.system_prompt_id,
            classes="ingest-textarea-medium prompt-textarea"
        )
        
        # User Prompt Section
        yield Label("Custom Prompt (for analysis):", classes="prompt-label")
        yield Select(
            [("Custom", "custom"), ("Loading...", Select.BLANK)],
            id=f"{self.user_prompt_id}-select",
            classes="prompt-select",
            prompt="Select a user prompt template...",
            value="custom"
        )
        yield TextArea(
            id=self.user_prompt_id,
            classes="ingest-textarea-medium prompt-textarea"
        )
    
    def on_mount(self) -> None:
        """Load available prompts when mounted."""
        self.load_available_prompts()
    
    def load_available_prompts(self) -> None:
        """Load available prompts from the database."""
        try:
            # Check if prompts database is available
            if not hasattr(self.app_instance, 'prompts_db') or not self.app_instance.prompts_db:
                logger.warning("Prompts database not available")
                self._set_default_prompts()
                return
            
            # Query prompts from database based on media type
            try:
                if self.media_type == "general":
                    # For general, get all prompts
                    all_prompts = self.app_instance.prompts_db.get_all_prompts(limit=500)
                    # Add keywords to each prompt
                    for prompt in all_prompts:
                        prompt['keywords'] = self.app_instance.prompts_db.fetch_keywords_for_prompt(
                            prompt['id'], include_deleted=False
                        )
                else:
                    # For specific media types, search by keyword
                    media_prompts = self.app_instance.prompts_db.search_prompts_by_keyword(
                        self.media_type, include_deleted=False
                    )
                    
                    # Also check for related keywords (e.g., "document" for "pdf")
                    related_prompts = []
                    if self.media_type == "pdf":
                        related_prompts = self.app_instance.prompts_db.search_prompts_by_keyword(
                            "document", include_deleted=False
                        )
                    elif self.media_type == "document":
                        related_prompts = self.app_instance.prompts_db.search_prompts_by_keyword(
                            "pdf", include_deleted=False
                        )
                    
                    # Combine and deduplicate
                    prompt_dict = {p['id']: p for p in media_prompts}
                    for prompt in related_prompts:
                        if prompt['id'] not in prompt_dict:
                            prompt_dict[prompt['id']] = prompt
                    
                    all_prompts = list(prompt_dict.values())
                    
                    # Also get general prompts that might be useful
                    general_prompts = self.app_instance.prompts_db.search_prompts_by_keyword(
                        "general", include_deleted=False
                    )
                    for prompt in general_prompts:
                        if prompt['id'] not in prompt_dict:
                            all_prompts.append(prompt)
                
                if all_prompts:
                    logger.info(f"Loaded {len(all_prompts)} prompts for media type '{self.media_type}'")
                    self._load_prompts_from_db(all_prompts)
                else:
                    logger.warning(f"No prompts found for media type '{self.media_type}', using defaults")
                    self._set_default_prompts()
                    
            except Exception as db_error:
                logger.error(f"Error querying prompts database: {db_error}")
                self._set_default_prompts()
            
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            self._set_default_prompts()
    
    def _load_prompts_from_db(self, prompts: List[Dict[str, Any]]) -> None:
        """Load prompts from database records and organize them for the UI."""
        # Separate system and user prompts
        system_prompts = [("Custom", "custom")]
        user_prompts = [("Custom", "custom")]
        
        # Clear prompts cache
        self.prompts_cache = {}
        
        # Process each prompt
        for prompt in prompts:
            try:
                prompt_name = prompt.get('name', '')
                prompt_id = str(prompt.get('id', ''))
                system_prompt = prompt.get('system_prompt', '')
                user_prompt = prompt.get('user_prompt', '')
                
                # Skip if no name
                if not prompt_name:
                    continue
                
                # Store prompt in cache
                cache_key = f"db_{prompt_id}"
                self.prompts_cache[cache_key] = {
                    "system": system_prompt,
                    "user": user_prompt
                }
                
                # Add to appropriate list
                if system_prompt and not user_prompt:
                    # System prompt only
                    system_prompts.append((prompt_name, cache_key))
                elif user_prompt and not system_prompt:
                    # User prompt only
                    user_prompts.append((prompt_name, cache_key))
                elif system_prompt and user_prompt:
                    # Both - add to both lists
                    system_prompts.append((prompt_name, cache_key))
                    user_prompts.append((prompt_name, cache_key))
                    
            except Exception as e:
                logger.error(f"Error processing prompt: {e}")
                continue
        
        # Sort prompts alphabetically (except Custom which stays first)
        system_prompts = system_prompts[:1] + sorted(system_prompts[1:], key=lambda x: x[0])
        user_prompts = user_prompts[:1] + sorted(user_prompts[1:], key=lambda x: x[0])
        
        # Update the select widgets
        try:
            system_select = self.query_one(f"#{self.system_prompt_id}-select", Select)
            system_select.set_options(system_prompts)
            logger.debug(f"Set {len(system_prompts)} system prompt options for {self.media_type}")
        except Exception as e:
            logger.error(f"Error updating system prompt select: {e}")
        
        try:
            user_select = self.query_one(f"#{self.user_prompt_id}-select", Select)
            user_select.set_options(user_prompts)
            logger.debug(f"Set {len(user_prompts)} user prompt options for {self.media_type}")
        except Exception as e:
            logger.error(f"Error updating user prompt select: {e}")
    
    def _set_default_prompts(self) -> None:
        """Set default prompt options based on media type."""
        # Define default prompts for different media types
        default_prompts = {
            "audio": {
                "system_prompts": [
                    ("Custom", "custom"),
                    ("Audio Analysis Assistant", "audio_analysis"),
                    ("Transcription Summarizer", "transcription_summary"),
                    ("Meeting Notes Generator", "meeting_notes"),
                    ("Podcast Summary", "podcast_summary")
                ],
                "user_prompts": [
                    ("Custom", "custom"),
                    ("Summarize Key Points", "summarize_key"),
                    ("Extract Action Items", "extract_actions"),
                    ("Identify Speakers", "identify_speakers"),
                    ("Create Detailed Summary", "detailed_summary")
                ]
            },
            "video": {
                "system_prompts": [
                    ("Custom", "custom"),
                    ("Video Content Analyzer", "video_analysis"),
                    ("Educational Content Summarizer", "edu_summary"),
                    ("Tutorial Breakdown", "tutorial_breakdown")
                ],
                "user_prompts": [
                    ("Custom", "custom"),
                    ("Summarize Video Content", "summarize_video"),
                    ("Extract Key Concepts", "extract_concepts"),
                    ("Create Study Notes", "study_notes")
                ]
            },
            "pdf": {
                "system_prompts": [
                    ("Custom", "custom"),
                    ("Document Analyzer", "doc_analyzer"),
                    ("Research Paper Summarizer", "research_summary"),
                    ("Technical Document Parser", "tech_parser")
                ],
                "user_prompts": [
                    ("Custom", "custom"),
                    ("Extract Main Points", "extract_main"),
                    ("Summarize Findings", "summarize_findings"),
                    ("Create Executive Summary", "exec_summary")
                ]
            },
            "document": {
                "system_prompts": [
                    ("Custom", "custom"),
                    ("Document Assistant", "doc_assistant"),
                    ("Content Analyzer", "content_analyzer"),
                    ("Report Summarizer", "report_summary")
                ],
                "user_prompts": [
                    ("Custom", "custom"),
                    ("Summarize Document", "summarize_doc"),
                    ("Extract Key Information", "extract_info"),
                    ("Create Outline", "create_outline")
                ]
            },
            "general": {
                "system_prompts": [
                    ("Custom", "custom"),
                    ("General Analysis Assistant", "general_assistant"),
                    ("Content Summarizer", "content_summary")
                ],
                "user_prompts": [
                    ("Custom", "custom"),
                    ("Provide Summary", "provide_summary"),
                    ("Extract Key Points", "extract_key_points")
                ]
            }
        }
        
        # Store prompt templates
        self.prompts_cache = {
            # System prompts
            "audio_analysis": {
                "system": "You are an expert audio content analyst. Analyze the transcribed audio and provide insights on the content, speakers, and key discussions.",
                "user": ""
            },
            "transcription_summary": {
                "system": "You are a professional transcription summarizer. Create clear, concise summaries of transcribed content while preserving key information.",
                "user": ""
            },
            "meeting_notes": {
                "system": "You are a meeting notes specialist. Extract and organize key points, decisions, and action items from meeting transcriptions.",
                "user": ""
            },
            "podcast_summary": {
                "system": "You are a podcast content analyst. Summarize podcast episodes, highlighting key topics, guest insights, and memorable quotes.",
                "user": ""
            },
            "video_analysis": {
                "system": "You are a video content analyst. Analyze video transcripts and provide comprehensive summaries of visual and audio content.",
                "user": ""
            },
            "edu_summary": {
                "system": "You are an educational content specialist. Create study-friendly summaries of educational videos, highlighting key concepts and learning objectives.",
                "user": ""
            },
            "tutorial_breakdown": {
                "system": "You are a tutorial content organizer. Break down tutorial videos into clear, step-by-step instructions with key points.",
                "user": ""
            },
            "doc_analyzer": {
                "system": "You are a document analysis expert. Analyze documents thoroughly and provide structured summaries of content, findings, and implications.",
                "user": ""
            },
            "research_summary": {
                "system": "You are a research paper analyst. Summarize academic papers, highlighting methodology, findings, and conclusions.",
                "user": ""
            },
            "tech_parser": {
                "system": "You are a technical documentation specialist. Parse and summarize technical documents, focusing on specifications, procedures, and key technical details.",
                "user": ""
            },
            "doc_assistant": {
                "system": "You are a document processing assistant. Help users understand and extract information from various document types.",
                "user": ""
            },
            "content_analyzer": {
                "system": "You are a content analysis expert. Analyze text content and provide insights on structure, themes, and key information.",
                "user": ""
            },
            "report_summary": {
                "system": "You are a report summarization specialist. Create executive summaries of reports, highlighting key findings and recommendations.",
                "user": ""
            },
            "general_assistant": {
                "system": "You are a helpful analysis assistant. Provide clear and insightful analysis of the provided content.",
                "user": ""
            },
            "content_summary": {
                "system": "You are a content summarization expert. Create concise yet comprehensive summaries of various content types.",
                "user": ""
            },
            # User prompts
            "summarize_key": {
                "system": "",
                "user": "Please summarize the key points discussed in this audio. Include main topics, important decisions, and any action items mentioned."
            },
            "extract_actions": {
                "system": "",
                "user": "Extract all action items, tasks, and commitments mentioned in this transcription. Organize them by speaker if possible."
            },
            "identify_speakers": {
                "system": "",
                "user": "Identify the different speakers in this transcription and summarize what each person discussed or contributed."
            },
            "detailed_summary": {
                "system": "",
                "user": "Provide a detailed summary of this audio content, including main topics, subtopics, key points, and any conclusions or next steps."
            },
            "summarize_video": {
                "system": "",
                "user": "Summarize this video content, including main topics covered, key visual elements mentioned, and important takeaways."
            },
            "extract_concepts": {
                "system": "",
                "user": "Extract and explain the key concepts presented in this video. Organize them in a logical learning sequence."
            },
            "study_notes": {
                "system": "",
                "user": "Create comprehensive study notes from this video content, including main topics, definitions, examples, and key takeaways."
            },
            "extract_main": {
                "system": "",
                "user": "Extract the main points and arguments from this document. Organize them in a clear, hierarchical structure."
            },
            "summarize_findings": {
                "system": "",
                "user": "Summarize the key findings, results, or conclusions presented in this document."
            },
            "exec_summary": {
                "system": "",
                "user": "Create an executive summary of this document, highlighting the most important information for decision-makers."
            },
            "summarize_doc": {
                "system": "",
                "user": "Provide a comprehensive summary of this document, including its purpose, main content, and conclusions."
            },
            "extract_info": {
                "system": "",
                "user": "Extract and organize the key information from this document, categorizing it by topic or importance."
            },
            "create_outline": {
                "system": "",
                "user": "Create a detailed outline of this document's content, showing the hierarchical structure of topics and subtopics."
            },
            "provide_summary": {
                "system": "",
                "user": "Please provide a clear and concise summary of this content."
            },
            "extract_key_points": {
                "system": "",
                "user": "Extract and list the key points from this content."
            }
        }
        
        # Get prompts for media type
        prompts = default_prompts.get(self.media_type, default_prompts["general"])
        
        # Update system prompt select
        try:
            system_select = self.query_one(f"#{self.system_prompt_id}-select", Select)
            system_select.set_options(prompts["system_prompts"])
        except Exception as e:
            logger.error(f"Error updating system prompt select: {e}")
        
        # Update user prompt select
        try:
            user_select = self.query_one(f"#{self.user_prompt_id}-select", Select)
            user_select.set_options(prompts["user_prompts"])
        except Exception as e:
            logger.error(f"Error updating user prompt select: {e}")
    
    @on(Select.Changed)
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle prompt selection changes."""
        if not event.value or event.value == Select.BLANK:
            return
        
        select_id = event.control.id
        if not select_id:
            return
        
        # Determine which prompt was changed
        if select_id == f"{self.system_prompt_id}-select":
            textarea_id = self.system_prompt_id
            prompt_type = "system"
        elif select_id == f"{self.user_prompt_id}-select":
            textarea_id = self.user_prompt_id
            prompt_type = "user"
        else:
            return
        
        # Get the textarea
        try:
            textarea = self.query_one(f"#{textarea_id}", TextArea)
        except Exception as e:
            logger.error(f"Could not find textarea {textarea_id}: {e}")
            return
        
        # Handle selection
        if event.value == "custom":
            # Don't clear for custom - let user type their own
            logger.debug(f"Custom {prompt_type} prompt selected")
        else:
            # Load the pre-configured prompt
            prompt_data = self.prompts_cache.get(str(event.value), {})
            prompt_text = prompt_data.get(prompt_type, "")
            if prompt_text:
                textarea.load_text(prompt_text)
                logger.debug(f"Loaded {prompt_type} prompt: {event.value}")
            else:
                # If no text for this prompt type, check if there's text for the other type
                # This handles cases where a prompt might only have system or user text
                other_type = "user" if prompt_type == "system" else "system"
                other_text = prompt_data.get(other_type, "")
                if other_text and prompt_type == "user":
                    # If we're looking for user prompt but only system exists, leave empty
                    textarea.load_text("")
                    logger.debug(f"No user prompt for {event.value}, leaving empty")
                elif other_text and prompt_type == "system":
                    # If we're looking for system prompt but only user exists, leave empty
                    textarea.load_text("")
                    logger.debug(f"No system prompt for {event.value}, leaving empty")
                else:
                    logger.warning(f"No {prompt_type} prompt text found for: {event.value}")
    
    def get_prompts(self) -> Tuple[str, str]:
        """
        Get the current system and user prompts.
        
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        try:
            system_textarea = self.query_one(f"#{self.system_prompt_id}", TextArea)
            user_textarea = self.query_one(f"#{self.user_prompt_id}", TextArea)
            return (system_textarea.text.strip(), user_textarea.text.strip())
        except Exception as e:
            logger.error(f"Error getting prompts: {e}")
            return ("", "")