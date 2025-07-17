"""
MCP Prompts implementation for tldw_chatbook

This module provides reusable prompt templates for common workflows.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from loguru import logger

# Import tldw_chatbook components
from ..DB.ChaChaNotes_DB import ChaChaNotes_DB
from ..DB.Client_Media_DB_v2 import MediaDatabase
from ..Character_Chat.Character_Chat_Lib import get_character_by_id


class MCPPrompts:
    """Container for MCP prompt implementations."""
    
    def __init__(self, chachanotes_db: ChaChaNotes_DB, media_db: MediaDatabase):
        """Initialize prompts with database connections."""
        self.chachanotes_db = chachanotes_db
        self.media_db = media_db
    
    async def summarize_conversation_prompt(
        self,
        conversation_id: int,
        style: str = "concise",
        focus: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Generate a prompt to summarize a conversation.
        
        Args:
            conversation_id: ID of conversation to summarize
            style: Summary style (concise, detailed, bullet_points, executive)
            focus: Optional focus area (action_items, decisions, technical_details)
        
        Returns:
            List of prompt messages
        """
        try:
            # Get conversation
            conv = self.chachanotes_db.get_conversation_by_id(conversation_id)
            if not conv:
                return [{
                    "role": "user",
                    "content": f"Error: Conversation {conversation_id} not found"
                }]
            
            # Get messages
            messages = self.chachanotes_db.get_conversation_messages(conversation_id)
            
            # Build conversation text
            conversation_text = f"Title: {conv['title']}\n\n"
            for msg in messages:
                conversation_text += f"{msg['role'].upper()}: {msg['content']}\n\n"
            
            # Build prompt based on style
            prompt = "Please analyze the following conversation and provide a "
            
            if style == "concise":
                prompt += "concise summary (3-5 sentences) highlighting the key points and outcomes."
            elif style == "detailed":
                prompt += "detailed summary including all major topics discussed, decisions made, and action items."
            elif style == "bullet_points":
                prompt += "summary in bullet point format, organizing information by topic."
            elif style == "executive":
                prompt += "executive summary suitable for senior leadership, focusing on decisions and strategic implications."
            else:
                prompt += f"{style} summary."
            
            if focus:
                if focus == "action_items":
                    prompt += "\n\nPay special attention to any action items, tasks, or commitments made."
                elif focus == "decisions":
                    prompt += "\n\nHighlight all decisions made and their rationale."
                elif focus == "technical_details":
                    prompt += "\n\nFocus on technical details, specifications, and implementation aspects."
            
            prompt += f"\n\n---\n\n{conversation_text}"
            
            return [{
                "role": "user",
                "content": prompt
            }]
            
        except Exception as e:
            logger.error(f"Error creating summarize_conversation prompt: {e}")
            return [{
                "role": "user",
                "content": f"Error creating prompt: {str(e)}"
            }]
    
    async def generate_document_prompt(
        self,
        conversation_id: int,
        doc_type: str = "summary",
        format: str = "markdown"
    ) -> List[Dict[str, str]]:
        """Generate a prompt to create a document from a conversation.
        
        Args:
            conversation_id: ID of conversation
            doc_type: Type of document (summary, report, timeline, study_guide, briefing)
            format: Output format (markdown, html, plain_text)
        
        Returns:
            List of prompt messages
        """
        try:
            # Get conversation
            conv = self.chachanotes_db.get_conversation_by_id(conversation_id)
            if not conv:
                return [{
                    "role": "user",
                    "content": f"Error: Conversation {conversation_id} not found"
                }]
            
            # Get messages
            messages = self.chachanotes_db.get_conversation_messages(conversation_id)
            
            # Build conversation text
            conversation_text = ""
            for msg in messages:
                conversation_text += f"{msg['role'].upper()}: {msg['content']}\n\n"
            
            # Build prompt based on document type
            prompt = f"Based on the following conversation, please create a {doc_type}"
            
            if doc_type == "summary":
                prompt += " that captures the key points, decisions, and outcomes."
            elif doc_type == "report":
                prompt += " with sections for: Executive Summary, Key Discussion Points, Decisions Made, Action Items, and Next Steps."
            elif doc_type == "timeline":
                prompt += " showing the chronological flow of topics and decisions."
            elif doc_type == "study_guide":
                prompt += " with key concepts, definitions, examples, and potential questions."
            elif doc_type == "briefing":
                prompt += " suitable for briefing someone who wasn't present, including context, key points, and outcomes."
            
            prompt += f"\n\nPlease format the output as {format}."
            prompt += f"\n\nConversation Title: {conv['title']}"
            prompt += f"\n\n---\n\n{conversation_text}"
            
            return [{
                "role": "user",
                "content": prompt
            }]
            
        except Exception as e:
            logger.error(f"Error creating generate_document prompt: {e}")
            return [{
                "role": "user",
                "content": f"Error creating prompt: {str(e)}"
            }]
    
    async def analyze_media_prompt(
        self,
        media_id: int,
        analysis_type: str = "summary",
        detail_level: str = "medium"
    ) -> List[Dict[str, str]]:
        """Generate a prompt to analyze ingested media.
        
        Args:
            media_id: ID of media to analyze
            analysis_type: Type of analysis (summary, transcript, key_points, themes, sentiment)
            detail_level: Level of detail (brief, medium, comprehensive)
        
        Returns:
            List of prompt messages
        """
        try:
            # Get media
            media = self.media_db.get_media_by_id(media_id)
            if not media:
                return [{
                    "role": "user",
                    "content": f"Error: Media {media_id} not found"
                }]
            
            # Get transcript or content
            content = self.media_db.get_media_transcript(media_id) or media.get('content', '')
            
            # Build prompt based on analysis type
            prompt = f"Please analyze the following {media['media_type']} content"
            
            if analysis_type == "summary":
                prompt += " and provide a summary"
            elif analysis_type == "transcript":
                prompt += " and create a cleaned, formatted transcript"
            elif analysis_type == "key_points":
                prompt += " and extract the key points and main ideas"
            elif analysis_type == "themes":
                prompt += " and identify the major themes and topics discussed"
            elif analysis_type == "sentiment":
                prompt += " and analyze the sentiment and emotional tone"
            
            # Add detail level
            if detail_level == "brief":
                prompt += " (keep it brief, 3-5 sentences)."
            elif detail_level == "comprehensive":
                prompt += " (provide a comprehensive analysis with examples and quotes)."
            else:
                prompt += "."
            
            prompt += f"\n\nTitle: {media['title']}"
            if media.get('author'):
                prompt += f"\nAuthor: {media['author']}"
            if media.get('url'):
                prompt += f"\nSource: {media['url']}"
            
            prompt += f"\n\n---\n\n{content}"
            
            return [{
                "role": "user",
                "content": prompt
            }]
            
        except Exception as e:
            logger.error(f"Error creating analyze_media prompt: {e}")
            return [{
                "role": "user",
                "content": f"Error creating prompt: {str(e)}"
            }]
    
    async def search_and_synthesize_prompt(
        self,
        query: str,
        num_sources: int = 5,
        synthesis_type: str = "overview"
    ) -> List[Dict[str, str]]:
        """Generate a prompt to search RAG and synthesize results.
        
        Args:
            query: Search query
            num_sources: Number of sources to include
            synthesis_type: Type of synthesis (overview, comparison, deep_dive, answer)
        
        Returns:
            List of prompt messages
        """
        try:
            # Perform RAG search
            from ..RAG_Search.simplified.search_service import SimplifiedRAGSearchService
            rag_service = SimplifiedRAGSearchService(self.media_db)
            
            results = rag_service.keyword_search(query, limit=num_sources)
            
            if not results:
                return [{
                    "role": "user",
                    "content": f"No results found for query: {query}"
                }]
            
            # Build prompt
            prompt = f"Based on the following search results for '{query}', please "
            
            if synthesis_type == "overview":
                prompt += "provide a comprehensive overview of the topic, synthesizing information from all sources."
            elif synthesis_type == "comparison":
                prompt += "compare and contrast the different perspectives or information presented in these sources."
            elif synthesis_type == "deep_dive":
                prompt += "create a detailed analysis, exploring the topic in depth using all available information."
            elif synthesis_type == "answer":
                prompt += f"answer the question: {query}"
            
            prompt += "\n\nSearch Results:\n\n"
            
            for i, result in enumerate(results, 1):
                prompt += f"Source {i}: {result.get('title', 'Untitled')}\n"
                if result.get('media_type'):
                    prompt += f"Type: {result['media_type']}\n"
                prompt += f"Content: {result.get('content', '')}\n"
                prompt += "-" * 50 + "\n\n"
            
            prompt += "\nPlease cite sources by number when referencing specific information."
            
            return [{
                "role": "user",
                "content": prompt
            }]
            
        except Exception as e:
            logger.error(f"Error creating search_and_synthesize prompt: {e}")
            return [{
                "role": "user",
                "content": f"Error creating prompt: {str(e)}"
            }]
    
    async def character_writing_prompt(
        self,
        character_id: int,
        writing_type: str = "response",
        context: Optional[str] = None,
        style_notes: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Generate a prompt for character-based writing.
        
        Args:
            character_id: ID of the character
            writing_type: Type of writing (response, story, dialogue, monologue)
            context: Optional context or scenario
            style_notes: Optional style guidelines
        
        Returns:
            List of prompt messages
        """
        try:
            # Get character
            character = get_character_by_id(self.chachanotes_db, character_id)
            if not character:
                return [{
                    "role": "user",
                    "content": f"Error: Character {character_id} not found"
                }]
            
            # Build system message with character info
            system_content = f"You are {character['name']}."
            if character.get('description'):
                system_content += f" {character['description']}"
            if character.get('personality'):
                system_content += f"\n\nPersonality: {character['personality']}"
            
            messages = [{"role": "system", "content": system_content}]
            
            # Build user prompt
            if writing_type == "response":
                prompt = "Please respond to the following in character"
                if context:
                    prompt += f":\n\n{context}"
                else:
                    prompt += ". What would you like to talk about?"
            
            elif writing_type == "story":
                prompt = "Write a short story featuring your character"
                if context:
                    prompt += f" in the following scenario: {context}"
            
            elif writing_type == "dialogue":
                prompt = "Write a dialogue between your character and someone else"
                if context:
                    prompt += f" about: {context}"
            
            elif writing_type == "monologue":
                prompt = "Write an internal monologue for your character"
                if context:
                    prompt += f" while: {context}"
            
            if style_notes:
                prompt += f"\n\nStyle notes: {style_notes}"
            
            messages.append({"role": "user", "content": prompt})
            
            return messages
            
        except Exception as e:
            logger.error(f"Error creating character_writing prompt: {e}")
            return [{
                "role": "user",
                "content": f"Error creating prompt: {str(e)}"
            }]