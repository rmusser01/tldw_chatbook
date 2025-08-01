# mindmap_integration.py
# Description: Integration with tldw_chatbook content
#
"""
Mindmap Integration
------------------

Integrates mindmap functionality with existing tldw_chatbook features:
- Create mindmaps from conversations, notes, and media
- Import content from SmartContentTree selections
- Sync with database content
"""

from typing import List, Dict, Optional, Any
from anytree import Node
from loguru import logger

from ...UI.Widgets.SmartContentTree import SmartContentTree, ContentNodeData
from ...Chatbooks.chatbook_models import ContentType
from ..Mind_Map.mermaid_parser import NodeShape


class MindmapIntegration:
    """Integrate mindmap with existing tldw_chatbook features"""
    
    def __init__(self, app_instance):
        """Initialize integration with app instance
        
        Args:
            app_instance: Main application instance with database access
        """
        self.app = app_instance
        self.db = app_instance.chachanotes_db
        self.media_db = getattr(app_instance, 'client_media_db_v2', None)
        self.prompts_db = getattr(app_instance, 'prompts_db', None)
    
    def create_from_conversation(self, conversation_id: str, 
                                include_messages: bool = True,
                                max_messages: int = 50) -> Node:
        """Create mindmap from conversation messages
        
        Args:
            conversation_id: ID of the conversation
            include_messages: Whether to include individual messages
            max_messages: Maximum number of messages to include
            
        Returns:
            Root node of the mindmap
        """
        # Get conversation details
        conversation = self.db.get_conversation_by_id(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        # Create root node
        root = Node(
            f"conv_{conversation_id}",
            text=conversation['title'],
            shape=NodeShape.DOUBLE_CIRCLE,
            metadata={'type': 'conversation', 'id': conversation_id}
        )
        
        # Add metadata nodes
        meta_node = Node(
            f"meta_{conversation_id}",
            parent=root,
            text="Metadata",
            shape=NodeShape.SQUARE
        )
        
        Node(
            f"created_{conversation_id}",
            parent=meta_node,
            text=f"Created: {conversation['created_at'][:10]}",
            shape=NodeShape.DEFAULT
        )
        
        if conversation.get('character_id'):
            Node(
                f"character_{conversation_id}",
                parent=meta_node,
                text=f"Character: {conversation.get('character_name', 'Unknown')}",
                shape=NodeShape.DEFAULT
            )
        
        # Add messages if requested
        if include_messages:
            messages = self.db.get_messages_for_conversation(conversation_id, limit=max_messages)
            
            if messages:
                messages_node = Node(
                    f"messages_{conversation_id}",
                    parent=root,
                    text=f"Messages ({len(messages)})",
                    shape=NodeShape.SQUARE
                )
                
                for i, msg in enumerate(messages):
                    # Truncate long messages
                    content = msg['content']
                    if len(content) > 100:
                        content = content[:97] + "..."
                    
                    msg_node = Node(
                        f"msg_{msg['id']}",
                        parent=messages_node,
                        text=f"{msg['sender']}: {content}",
                        shape=NodeShape.ROUNDED,
                        metadata={'message_id': msg['id'], 'timestamp': msg['timestamp']}
                    )
                    
                    # Add key points if we can extract them
                    if len(msg['content']) > 200:
                        self._add_key_points(msg_node, msg['content'])
        
        return root
    
    def create_from_notes(self, note_ids: List[str], 
                         include_headers: bool = True) -> Node:
        """Create mindmap from notes
        
        Args:
            note_ids: List of note IDs to include
            include_headers: Whether to parse and include headers
            
        Returns:
            Root node of the mindmap
        """
        if not note_ids:
            raise ValueError("No note IDs provided")
        
        # Create root node
        root = Node(
            "notes_root",
            text=f"Notes Collection ({len(note_ids)} notes)",
            shape=NodeShape.DOUBLE_CIRCLE,
            metadata={'type': 'notes_collection'}
        )
        
        for note_id in note_ids:
            note = self.db.get_note_by_id(note_id)
            if not note:
                logger.warning(f"Note {note_id} not found")
                continue
            
            # Create note node
            note_node = Node(
                f"note_{note_id}",
                parent=root,
                text=note['title'],
                shape=NodeShape.SQUARE,
                metadata={'type': 'note', 'id': note_id}
            )
            
            # Add metadata
            if note.get('tags'):
                tags_text = ", ".join(note['tags']) if isinstance(note['tags'], list) else note['tags']
                Node(
                    f"tags_{note_id}",
                    parent=note_node,
                    text=f"Tags: {tags_text}",
                    shape=NodeShape.DEFAULT
                )
            
            # Parse headers if requested
            if include_headers and note.get('content'):
                headers = self._extract_headers(note['content'])
                for i, header in enumerate(headers):
                    Node(
                        f"header_{note_id}_{i}",
                        parent=note_node,
                        text=header['text'],
                        shape=NodeShape.ROUNDED,
                        metadata={'level': header['level']}
                    )
        
        return root
    
    def create_from_media(self, media_ids: List[str]) -> Node:
        """Create mindmap from media items
        
        Args:
            media_ids: List of media IDs to include
            
        Returns:
            Root node of the mindmap
        """
        if not self.media_db:
            raise ValueError("Media database not available")
        
        if not media_ids:
            raise ValueError("No media IDs provided")
        
        # Create root node
        root = Node(
            "media_root",
            text=f"Media Collection ({len(media_ids)} items)",
            shape=NodeShape.DOUBLE_CIRCLE,
            metadata={'type': 'media_collection'}
        )
        
        for media_id in media_ids:
            media = self.media_db.get_media_item(media_id)
            if not media:
                logger.warning(f"Media {media_id} not found")
                continue
            
            # Create media node
            media_node = Node(
                f"media_{media_id}",
                parent=root,
                text=media['title'],
                shape=NodeShape.HEXAGON,
                metadata={'type': 'media', 'id': media_id}
            )
            
            # Add media type
            Node(
                f"type_{media_id}",
                parent=media_node,
                text=f"Type: {media['type']}",
                shape=NodeShape.DEFAULT
            )
            
            # Add author if available
            if media.get('author'):
                Node(
                    f"author_{media_id}",
                    parent=media_node,
                    text=f"Author: {media['author']}",
                    shape=NodeShape.DEFAULT
                )
            
            # Add summary if available
            if media.get('summary'):
                summary = media['summary']
                if len(summary) > 100:
                    summary = summary[:97] + "..."
                
                Node(
                    f"summary_{media_id}",
                    parent=media_node,
                    text=f"Summary: {summary}",
                    shape=NodeShape.ROUNDED
                )
        
        return root
    
    def create_from_smart_tree_selection(self, tree_widget: SmartContentTree) -> Node:
        """Create mindmap from SmartContentTree selections
        
        Args:
            tree_widget: SmartContentTree widget with selections
            
        Returns:
            Root node of the mindmap
        """
        selections = tree_widget.get_selections()
        
        # Count total selections
        total_items = sum(len(items) for items in selections.values())
        if total_items == 0:
            raise ValueError("No items selected")
        
        # Create root node
        root = Node(
            "selected_content",
            text=f"Selected Content ({total_items} items)",
            shape=NodeShape.DOUBLE_CIRCLE,
            metadata={'type': 'selection'}
        )
        
        # Process each content type
        for content_type, item_ids in selections.items():
            if not item_ids:
                continue
            
            # Create category node
            type_node = Node(
                f"type_{content_type.value}",
                parent=root,
                text=f"{content_type.value.title()} ({len(item_ids)} items)",
                shape=NodeShape.SQUARE,
                metadata={'content_type': content_type.value}
            )
            
            # Add individual items based on type
            if content_type == ContentType.CONVERSATION:
                for conv_id in item_ids:
                    conv = self.db.get_conversation_by_id(conv_id)
                    if conv:
                        Node(
                            f"conv_{conv_id}",
                            parent=type_node,
                            text=conv['title'],
                            shape=NodeShape.ROUNDED,
                            metadata={'type': 'conversation', 'id': conv_id}
                        )
            
            elif content_type == ContentType.NOTE:
                for note_id in item_ids:
                    note = self.db.get_note_by_id(note_id)
                    if note:
                        Node(
                            f"note_{note_id}",
                            parent=type_node,
                            text=note['title'],
                            shape=NodeShape.ROUNDED,
                            metadata={'type': 'note', 'id': note_id}
                        )
            
            elif content_type == ContentType.CHARACTER:
                for char_id in item_ids:
                    character = self.db.get_character_by_id(char_id)
                    if character:
                        Node(
                            f"char_{char_id}",
                            parent=type_node,
                            text=character['name'],
                            shape=NodeShape.ROUNDED,
                            metadata={'type': 'character', 'id': char_id}
                        )
            
            elif content_type == ContentType.MEDIA:
                if self.media_db:
                    for media_id in item_ids:
                        media = self.media_db.get_media_item(media_id)
                        if media:
                            Node(
                                f"media_{media_id}",
                                parent=type_node,
                                text=media['title'],
                                shape=NodeShape.ROUNDED,
                                metadata={'type': 'media', 'id': media_id}
                            )
            
            elif content_type == ContentType.PROMPT:
                if self.prompts_db:
                    for prompt_id in item_ids:
                        prompt = self.prompts_db.get_prompt_by_id(prompt_id)
                        if prompt:
                            Node(
                                f"prompt_{prompt_id}",
                                parent=type_node,
                                text=prompt['name'],
                                shape=NodeShape.ROUNDED,
                                metadata={'type': 'prompt', 'id': prompt_id}
                            )
        
        return root
    
    def create_from_search_results(self, query: str, results: Dict[str, List[Any]]) -> Node:
        """Create mindmap from search results
        
        Args:
            query: Search query used
            results: Dictionary of results by content type
            
        Returns:
            Root node of the mindmap
        """
        # Count total results
        total_results = sum(len(items) for items in results.values())
        
        # Create root node
        root = Node(
            "search_results",
            text=f'Search: "{query}" ({total_results} results)',
            shape=NodeShape.DOUBLE_CIRCLE,
            metadata={'type': 'search', 'query': query}
        )
        
        # Add results by type
        for content_type, items in results.items():
            if not items:
                continue
            
            type_node = Node(
                f"results_{content_type}",
                parent=root,
                text=f"{content_type.title()} ({len(items)} results)",
                shape=NodeShape.SQUARE
            )
            
            for i, item in enumerate(items[:20]):  # Limit to 20 per type
                # Extract title based on item structure
                if isinstance(item, dict):
                    title = item.get('title', item.get('name', f"Item {i+1}"))
                else:
                    title = str(item)
                
                if len(title) > 50:
                    title = title[:47] + "..."
                
                Node(
                    f"result_{content_type}_{i}",
                    parent=type_node,
                    text=title,
                    shape=NodeShape.ROUNDED,
                    metadata={'item': item}
                )
        
        return root
    
    def _extract_headers(self, content: str) -> List[Dict[str, Any]]:
        """Extract markdown headers from content
        
        Args:
            content: Markdown content
            
        Returns:
            List of header dictionaries with text and level
        """
        import re
        headers = []
        
        # Match markdown headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
        for line in content.split('\n'):
            match = re.match(header_pattern, line.strip())
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                headers.append({
                    'level': level,
                    'text': text
                })
        
        return headers
    
    def _add_key_points(self, parent_node: Node, content: str, max_points: int = 3):
        """Extract and add key points from content
        
        Args:
            parent_node: Parent node to attach points to
            content: Content to analyze
            max_points: Maximum number of points to extract
        """
        # Simple extraction: Look for sentences with certain keywords
        # In a real implementation, this could use NLP or LLM
        import re
        
        sentences = re.split(r'[.!?]\s+', content)
        key_sentences = []
        
        # Keywords that might indicate important points
        keywords = ['important', 'key', 'main', 'crucial', 'significant', 
                   'remember', 'note', 'summary', 'conclusion', 'therefore']
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                key_sentences.append(sentence.strip())
        
        # Also add first and last sentences if not already included
        if sentences and sentences[0].strip() not in key_sentences:
            key_sentences.insert(0, sentences[0].strip())
        
        if len(sentences) > 1 and sentences[-1].strip() not in key_sentences:
            key_sentences.append(sentences[-1].strip())
        
        # Create nodes for key points
        for i, point in enumerate(key_sentences[:max_points]):
            if len(point) > 60:
                point = point[:57] + "..."
            
            Node(
                f"point_{parent_node.name}_{i}",
                parent=parent_node,
                text=f"• {point}",
                shape=NodeShape.DEFAULT,
                metadata={'type': 'key_point'}
            )
    
    def create_study_mindmap(self, topic: str, note_ids: List[str]) -> Node:
        """Create a study-focused mindmap from notes
        
        Args:
            topic: Study topic
            note_ids: Related note IDs
            
        Returns:
            Root node of study mindmap
        """
        # Create root with study topic
        root = Node(
            "study_root",
            text=f"Study: {topic}",
            shape=NodeShape.DOUBLE_CIRCLE,
            icon="📚",
            metadata={'type': 'study', 'topic': topic}
        )
        
        # Create main branches
        concepts_node = Node(
            "concepts",
            parent=root,
            text="Key Concepts",
            shape=NodeShape.SQUARE,
            icon="💡"
        )
        
        questions_node = Node(
            "questions",
            parent=root,
            text="Review Questions",
            shape=NodeShape.SQUARE,
            icon="❓"
        )
        
        resources_node = Node(
            "resources",
            parent=root,
            text="Resources",
            shape=NodeShape.SQUARE,
            icon="📖"
        )
        
        # Process notes to extract study content
        for note_id in note_ids:
            note = self.db.get_note_by_id(note_id)
            if not note:
                continue
            
            # Add to resources
            Node(
                f"resource_{note_id}",
                parent=resources_node,
                text=note['title'],
                shape=NodeShape.ROUNDED,
                metadata={'note_id': note_id}
            )
            
            # Extract concepts and questions from content
            if note.get('content'):
                self._extract_study_content(note['content'], concepts_node, questions_node)
        
        return root
    
    def _extract_study_content(self, content: str, concepts_node: Node, questions_node: Node):
        """Extract study-relevant content
        
        Args:
            content: Note content to analyze
            concepts_node: Node to attach concepts to
            questions_node: Node to attach questions to
        """
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for question patterns
            if line.endswith('?') or line.lower().startswith(('what', 'why', 'how', 'when', 'where', 'who')):
                if len(line) > 10:  # Filter out very short questions
                    Node(
                        f"q_{len(questions_node.children)}",
                        parent=questions_node,
                        text=line[:100],
                        shape=NodeShape.ROUNDED
                    )
            
            # Look for definition patterns
            elif ':' in line and len(line) > 20:
                parts = line.split(':', 1)
                if len(parts[0]) < 50:  # Likely a term definition
                    Node(
                        f"c_{len(concepts_node.children)}",
                        parent=concepts_node,
                        text=line[:100],
                        shape=NodeShape.ROUNDED
                    )