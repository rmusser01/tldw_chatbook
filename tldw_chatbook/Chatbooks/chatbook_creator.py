# chatbook_creator.py
# Description: Service for creating chatbooks/knowledge packs
#
"""
Chatbook Creator
----------------

Handles the creation and packaging of chatbooks from database content.
"""

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from loguru import logger

from .chatbook_models import (
    Chatbook, ChatbookManifest, ChatbookContent, 
    ContentItem, ContentType, ChatbookVersion, Relationship
)
from ..DB.ChaChaNotes_DB import CharactersRAGDB
from ..DB.Client_Media_DB_v2 import MediaDatabase
from ..DB.Prompts_DB import PromptsDatabase
from ..DB.RAG_Indexing_DB import RAGIndexingDB
from ..DB.Evals_DB import EvalsDB
from ..DB.Subscriptions_DB import SubscriptionsDB


class ChatbookCreator:
    """Service for creating chatbooks from database content."""
    
    def __init__(self, db_paths: Dict[str, str]):
        """
        Initialize the chatbook creator.
        
        Args:
            db_paths: Dictionary mapping database names to their paths
        """
        self.db_paths = db_paths
        self.temp_dir = Path.home() / ".local" / "share" / "tldw_cli" / "temp" / "chatbooks"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    def create_chatbook(
        self,
        name: str,
        description: str,
        content_selections: Dict[ContentType, List[str]],
        output_path: Path,
        author: Optional[str] = None,
        include_media: bool = False,
        media_quality: str = "thumbnail",
        include_embeddings: bool = False,
        tags: List[str] = None,
        categories: List[str] = None
    ) -> Tuple[bool, str]:
        """
        Create a chatbook from selected content.
        
        Args:
            name: Name of the chatbook
            description: Description of the chatbook
            content_selections: Dictionary mapping content types to lists of IDs
            output_path: Path where the chatbook should be saved
            author: Author name (optional)
            include_media: Whether to include media files
            media_quality: Quality of media to include (thumbnail/compressed/original)
            include_embeddings: Whether to include embeddings
            tags: List of tags for the chatbook
            categories: List of categories for the chatbook
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Create temporary working directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            work_dir = self.temp_dir / f"chatbook_{timestamp}"
            work_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize manifest
            manifest = ChatbookManifest(
                version=ChatbookVersion.V1,
                name=name,
                description=description,
                author=author,
                include_media=include_media,
                include_embeddings=include_embeddings,
                media_quality=media_quality,
                tags=tags or [],
                categories=categories or []
            )
            
            # Initialize content container
            content = ChatbookContent()
            
            # Process each content type
            logger.info(f"Creating chatbook '{name}' with selections: {content_selections}")
            
            # Collect conversations
            if ContentType.CONVERSATION in content_selections:
                self._collect_conversations(
                    content_selections[ContentType.CONVERSATION],
                    work_dir,
                    manifest,
                    content
                )
            
            # Collect notes
            if ContentType.NOTE in content_selections:
                self._collect_notes(
                    content_selections[ContentType.NOTE],
                    work_dir,
                    manifest,
                    content
                )
            
            # Collect characters
            if ContentType.CHARACTER in content_selections:
                self._collect_characters(
                    content_selections[ContentType.CHARACTER],
                    work_dir,
                    manifest,
                    content
                )
            
            # Collect media (if enabled)
            if include_media and ContentType.MEDIA in content_selections:
                self._collect_media(
                    content_selections[ContentType.MEDIA],
                    work_dir,
                    manifest,
                    content,
                    media_quality
                )
            
            # Collect prompts
            if ContentType.PROMPT in content_selections:
                self._collect_prompts(
                    content_selections[ContentType.PROMPT],
                    work_dir,
                    manifest,
                    content
                )
            
            # Auto-discover relationships
            self._discover_relationships(manifest, content)
            
            # Update statistics
            manifest.total_conversations = len(content.conversations)
            manifest.total_notes = len(content.notes)
            manifest.total_characters = len(content.characters)
            manifest.total_media_items = len(content.media_items)
            
            # Write manifest
            manifest_path = work_dir / "manifest.json"
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)
            
            # Create README
            self._create_readme(work_dir, manifest)
            
            # Package into archive
            if output_path.suffix == '.zip':
                self._create_zip_archive(work_dir, output_path)
            else:
                # Default to ZIP if no extension specified
                output_path = output_path.with_suffix('.zip')
                self._create_zip_archive(work_dir, output_path)
            
            # Calculate final size
            manifest.total_size_bytes = output_path.stat().st_size
            
            # Cleanup temp directory
            shutil.rmtree(work_dir)
            
            return True, f"Chatbook created successfully at {output_path}"
            
        except Exception as e:
            logger.error(f"Error creating chatbook: {e}")
            return False, f"Error creating chatbook: {str(e)}"
    
    def _collect_conversations(
        self,
        conversation_ids: List[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ) -> None:
        """Collect conversations and their messages."""
        db_path = self.db_paths.get("chachanotes")
        if not db_path:
            return
            
        db = CharactersRAGDB(db_path, "chatbook_creator")
        conv_dir = work_dir / "content" / "conversations"
        conv_dir.mkdir(parents=True, exist_ok=True)
        
        for conv_id in conversation_ids:
            try:
                # Get conversation details
                conv = db.get_conversation_by_id(conv_id)
                if not conv:
                    continue
                
                # Get all messages
                messages = db.get_messages_for_conversation(conv_id)
                
                # Create conversation data
                conv_data = {
                    "id": conv['id'],
                    "name": conv['conversation_name'],
                    "created_at": conv['created_at'],
                    "updated_at": conv['updated_at'],
                    "character_id": conv.get('character_id'),
                    "messages": [
                        {
                            "id": msg['id'],
                            "role": msg['sender'],
                            "content": msg['message'],
                            "timestamp": msg['timestamp']
                        }
                        for msg in messages
                    ]
                }
                
                # Write conversation file
                conv_file = conv_dir / f"conversation_{conv_id}.json"
                with open(conv_file, 'w', encoding='utf-8') as f:
                    json.dump(conv_data, f, indent=2, ensure_ascii=False)
                
                # Add to content
                content.conversations.append(conv_data)
                
                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=conv_id,
                    type=ContentType.CONVERSATION,
                    title=conv['conversation_name'],
                    created_at=datetime.fromisoformat(conv['created_at']),
                    updated_at=datetime.fromisoformat(conv['updated_at']),
                    file_path=f"content/conversations/conversation_{conv_id}.json"
                ))
                
                # Track character dependency if present
                if conv.get('character_id'):
                    self._add_character_dependency(conv['character_id'], manifest)
                    
            except Exception as e:
                logger.error(f"Error collecting conversation {conv_id}: {e}")
    
    def _collect_notes(
        self,
        note_ids: List[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ) -> None:
        """Collect notes and export as markdown."""
        db_path = self.db_paths.get("chachanotes")
        if not db_path:
            return
            
        db = CharactersRAGDB(db_path, "chatbook_creator")
        notes_dir = work_dir / "content" / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        
        for note_id in note_ids:
            try:
                # Get note details
                note = db.get_note_by_id(note_id)
                if not note:
                    continue
                
                # Create note metadata
                note_data = {
                    "id": note['id'],
                    "title": note['title'],
                    "content": note['content'],
                    "created_at": note['created_at'],
                    "updated_at": note['updated_at'],
                    "tags": note.get('keywords', '').split(',') if note.get('keywords') else []
                }
                
                # Write markdown file
                note_file = notes_dir / f"{note['title'].replace('/', '_')}.md"
                with open(note_file, 'w', encoding='utf-8') as f:
                    # Write frontmatter
                    f.write("---\n")
                    f.write(f"id: {note['id']}\n")
                    f.write(f"title: {note['title']}\n")
                    f.write(f"created_at: {note['created_at']}\n")
                    f.write(f"updated_at: {note['updated_at']}\n")
                    if note_data['tags']:
                        f.write(f"tags: {', '.join(note_data['tags'])}\n")
                    f.write("---\n\n")
                    
                    # Write content
                    f.write(note['content'])
                
                # Add to content
                content.notes.append(note_data)
                
                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=note_id,
                    type=ContentType.NOTE,
                    title=note['title'],
                    created_at=datetime.fromisoformat(note['created_at']),
                    updated_at=datetime.fromisoformat(note['updated_at']),
                    tags=note_data['tags'],
                    file_path=f"content/notes/{note_file.name}"
                ))
                
            except Exception as e:
                logger.error(f"Error collecting note {note_id}: {e}")
    
    def _collect_characters(
        self,
        character_ids: List[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ) -> None:
        """Collect character cards."""
        db_path = self.db_paths.get("chachanotes")
        if not db_path:
            return
            
        db = CharactersRAGDB(db_path, "chatbook_creator")
        chars_dir = work_dir / "content" / "characters"
        chars_dir.mkdir(parents=True, exist_ok=True)
        
        for char_id in character_ids:
            try:
                # Get character details
                char = db.get_character_details(char_id)
                if not char:
                    continue
                
                # Get character card
                card = db.get_character_card_details(char_id)
                
                # Create character data
                char_data = {
                    "id": char['id'],
                    "name": char['name'],
                    "description": char.get('description', ''),
                    "personality": char.get('personality', ''),
                    "created_at": char['created_at'],
                    "updated_at": char['updated_at'],
                    "avatar_path": char.get('avatar_path'),
                    "card": card
                }
                
                # Write character file
                char_file = chars_dir / f"character_{char_id}.json"
                with open(char_file, 'w', encoding='utf-8') as f:
                    json.dump(char_data, f, indent=2, ensure_ascii=False)
                
                # Add to content
                content.characters.append(char_data)
                
                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=char_id,
                    type=ContentType.CHARACTER,
                    title=char['name'],
                    description=char.get('description'),
                    created_at=datetime.fromisoformat(char['created_at']),
                    updated_at=datetime.fromisoformat(char['updated_at']),
                    file_path=f"content/characters/character_{char_id}.json"
                ))
                
            except Exception as e:
                logger.error(f"Error collecting character {char_id}: {e}")
    
    def _collect_media(
        self,
        media_ids: List[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent,
        quality: str
    ) -> None:
        """Collect media items."""
        # TODO: Implement media collection
        # This would involve copying media files and creating metadata
        pass
    
    def _collect_prompts(
        self,
        prompt_ids: List[str],
        work_dir: Path,
        manifest: ChatbookManifest,
        content: ChatbookContent
    ) -> None:
        """Collect prompts."""
        db_path = self.db_paths.get("prompts")
        if not db_path:
            return
            
        db = PromptsDatabase(db_path, "chatbook_creator")
        prompts_dir = work_dir / "content" / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        
        for prompt_id in prompt_ids:
            try:
                # Get prompt details
                prompt = db.get_prompt_by_id(int(prompt_id))
                if not prompt:
                    continue
                
                # Create prompt data
                prompt_data = {
                    "id": prompt['id'],
                    "name": prompt['name'],
                    "description": prompt.get('details', ''),
                    "content": prompt.get('system_prompt', '') or prompt.get('user_prompt', ''),
                    "created_at": prompt.get('created_at', datetime.now().isoformat()),
                    "updated_at": prompt.get('updated_at', datetime.now().isoformat())
                }
                
                # Write prompt file
                prompt_file = prompts_dir / f"prompt_{prompt_id}.json"
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    json.dump(prompt_data, f, indent=2, ensure_ascii=False)
                
                # Add to content
                content.prompts.append(prompt_data)
                
                # Add to manifest
                manifest.content_items.append(ContentItem(
                    id=prompt_id,
                    type=ContentType.PROMPT,
                    title=prompt['name'],
                    description=prompt.get('description'),
                    created_at=datetime.fromisoformat(prompt['created_at']),
                    updated_at=datetime.fromisoformat(prompt['updated_at']),
                    file_path=f"content/prompts/prompt_{prompt_id}.json"
                ))
                
            except Exception as e:
                logger.error(f"Error collecting prompt {prompt_id}: {e}")
    
    def _add_character_dependency(self, character_id: int, manifest: ChatbookManifest) -> None:
        """Add a character as a dependency if not already included."""
        # Check if character already in manifest
        for item in manifest.content_items:
            if item.type == ContentType.CHARACTER and item.id == str(character_id):
                return
        
        # TODO: Auto-include the character
        # For now, just log a warning
        logger.warning(f"Character {character_id} is referenced but not included in chatbook")
    
    def _discover_relationships(self, manifest: ChatbookManifest, content: ChatbookContent) -> None:
        """Discover relationships between content items."""
        # Find conversation-character relationships
        for conv in content.conversations:
            if conv.get('character_id'):
                # Find if character is in manifest
                char_id = str(conv['character_id'])
                for item in manifest.content_items:
                    if item.type == ContentType.CHARACTER and item.id == char_id:
                        manifest.relationships.append(Relationship(
                            source_id=conv['id'],
                            target_id=char_id,
                            relationship_type="uses_character"
                        ))
                        break
        
        # TODO: Add more relationship discovery logic
        # - Notes mentioning conversations
        # - Media used in conversations
        # - etc.
    
    def _create_readme(self, work_dir: Path, manifest: ChatbookManifest) -> None:
        """Create a README file for the chatbook."""
        readme_path = work_dir / "README.md"
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"# {manifest.name}\n\n")
            f.write(f"{manifest.description}\n\n")
            
            if manifest.author:
                f.write(f"**Author:** {manifest.author}\n\n")
            
            f.write(f"**Created:** {manifest.created_at.strftime('%Y-%m-%d %H:%M')}\n\n")
            
            f.write("## Contents\n\n")
            
            if manifest.total_conversations > 0:
                f.write(f"- **Conversations:** {manifest.total_conversations}\n")
            if manifest.total_notes > 0:
                f.write(f"- **Notes:** {manifest.total_notes}\n")
            if manifest.total_characters > 0:
                f.write(f"- **Characters:** {manifest.total_characters}\n")
            if manifest.total_media_items > 0:
                f.write(f"- **Media Items:** {manifest.total_media_items}\n")
            
            if manifest.tags:
                f.write(f"\n## Tags\n\n")
                f.write(", ".join(manifest.tags))
                f.write("\n")
            
            f.write("\n## Structure\n\n")
            f.write("```\n")
            f.write("chatbook/\n")
            f.write("├── manifest.json     # Chatbook metadata\n")
            f.write("├── README.md        # This file\n")
            f.write("└── content/         # Content files\n")
            if manifest.total_conversations > 0:
                f.write("    ├── conversations/   # Chat conversations\n")
            if manifest.total_notes > 0:
                f.write("    ├── notes/          # Markdown notes\n")
            if manifest.total_characters > 0:
                f.write("    ├── characters/     # Character cards\n")
            if manifest.total_media_items > 0:
                f.write("    └── media/          # Media files\n")
            f.write("```\n")
            
            f.write("\n## License\n\n")
            if manifest.license:
                f.write(manifest.license)
            else:
                f.write("See individual content files for licensing information.")
    
    def _create_zip_archive(self, work_dir: Path, output_path: Path) -> None:
        """Create a ZIP archive of the chatbook."""
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in work_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(work_dir)
                    zf.write(file_path, arcname)