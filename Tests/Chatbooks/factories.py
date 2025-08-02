# factories.py
# Description: Test data factories for chatbook tests
#
"""
Test Data Factories
-------------------

Factory functions for creating consistent test data.
"""

import uuid
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

from tldw_chatbook.Chatbooks.chatbook_models import (
    ContentType, ContentItem, Relationship, ChatbookManifest, ChatbookVersion
)


class CharacterFactory:
    """Factory for creating test character data."""
    
    @staticmethod
    def create(
        name: Optional[str] = None,
        description: Optional[str] = None,
        personality: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a test character."""
        base_name = name or f"Test Character {uuid.uuid4().hex[:8]}"
        
        return {
            "name": base_name,
            "description": description or f"Description for {base_name}",
            "personality": personality or random.choice([
                "Friendly and helpful",
                "Curious and analytical", 
                "Wise and patient",
                "Energetic and enthusiastic"
            ]),
            "scenario": kwargs.get("scenario", "General conversation"),
            "first_message": kwargs.get("first_message", f"Hello! I'm {base_name}."),
            "example_messages": kwargs.get("example_messages", ""),
            "system_prompt": kwargs.get("system_prompt", "You are a helpful assistant."),
            "post_history_instructions": kwargs.get("post_history_instructions", ""),
            "alternate_greetings": kwargs.get("alternate_greetings", []),
            "tags": tags or ["test", "factory"],
            "creator": kwargs.get("creator", "Test Factory"),
            "character_version": kwargs.get("character_version", "1.0"),
            "extensions": kwargs.get("extensions", {}),
            **kwargs
        }
    
    @staticmethod
    def create_batch(count: int, **kwargs) -> List[Dict[str, Any]]:
        """Create multiple test characters."""
        return [CharacterFactory.create(**kwargs) for _ in range(count)]


class ConversationFactory:
    """Factory for creating test conversation data."""
    
    @staticmethod
    def create(
        name: Optional[str] = None,
        character_id: Optional[int] = None,
        message_count: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a test conversation."""
        conv_id = str(uuid.uuid4())
        base_name = name or f"Test Conversation {conv_id[:8]}"
        
        messages = ConversationFactory.create_messages(message_count)
        
        return {
            "id": conv_id,
            "conversation_name": base_name,
            "name": base_name,  # For chatbook format
            "character_id": character_id,
            "created_at": kwargs.get("created_at", datetime.now().isoformat()),
            "updated_at": kwargs.get("updated_at", datetime.now().isoformat()),
            "messages": messages,
            **kwargs
        }
    
    @staticmethod
    def create_messages(count: int) -> List[Dict[str, Any]]:
        """Create test messages for a conversation."""
        messages = []
        timestamp = datetime.now()
        
        for i in range(count):
            role = "user" if i % 2 == 0 else "assistant"
            
            if role == "user":
                content = random.choice([
                    "Tell me about yourself.",
                    "What do you think about AI?",
                    "Can you help me with something?",
                    "What's your favorite topic?",
                    "How are you today?"
                ])
            else:
                content = random.choice([
                    "I'd be happy to help you with that!",
                    "That's an interesting question. Let me think...",
                    "I'm doing well, thank you for asking!",
                    "I find that topic quite fascinating.",
                    "Of course! What would you like to know?"
                ])
            
            messages.append({
                "id": f"msg_{i}_{uuid.uuid4().hex[:8]}",
                "role": role,
                "content": content,
                "timestamp": (timestamp + timedelta(minutes=i)).isoformat()
            })
        
        return messages


class NoteFactory:
    """Factory for creating test note data."""
    
    @staticmethod
    def create(
        title: Optional[str] = None,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a test note."""
        note_id = str(uuid.uuid4())
        base_title = title or f"Test Note {note_id[:8]}"
        
        default_content = f"""# {base_title}

This is a test note created by the factory.

## Section 1

Some content here with **bold** and *italic* text.

## Section 2

- List item 1
- List item 2
- List item 3

## Code Example

```python
def hello_world():
    print("Hello from test note!")
```

Tags: {', '.join(tags or ['test', 'factory'])}
"""
        
        return {
            "id": note_id,
            "title": base_title,
            "content": content or default_content,
            "created_at": kwargs.get("created_at", datetime.now().isoformat()),
            "updated_at": kwargs.get("updated_at", datetime.now().isoformat()),
            "tags": tags or ["test", "factory"],
            **kwargs
        }


class MediaFactory:
    """Factory for creating test media data."""
    
    @staticmethod
    def create(
        title: Optional[str] = None,
        media_type: Optional[str] = None,
        url: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a test media item."""
        media_id = str(uuid.uuid4())
        base_title = title or f"Test Media {media_id[:8]}"
        media_type = media_type or random.choice(["video", "audio", "document", "image"])
        
        return {
            "id": media_id,
            "title": base_title,
            "media_type": media_type,
            "url": url or f"https://example.com/{media_type}/{media_id}.{media_type[:3]}",
            "author": kwargs.get("author", "Test Factory"),
            "content": kwargs.get("content", f"Transcript/content for {base_title}"),
            "created_at": kwargs.get("created_at", datetime.now().isoformat()),
            "updated_at": kwargs.get("updated_at", datetime.now().isoformat()),
            "metadata": {
                "ingestion_date": kwargs.get("ingestion_date", datetime.now().isoformat()),
                "media_keywords": kwargs.get("media_keywords", "test, factory, media"),
                "summary": kwargs.get("summary", f"Summary of {base_title}"),
                "transcription_model": kwargs.get("transcription_model", "test_model"),
                **kwargs.get("metadata", {})
            }
        }


class PromptFactory:
    """Factory for creating test prompt data."""
    
    @staticmethod
    def create(
        name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a test prompt."""
        prompt_id = str(uuid.uuid4())
        base_name = name or f"Test Prompt {prompt_id[:8]}"
        
        return {
            "id": prompt_id,
            "name": base_name,
            "description": kwargs.get("description", f"Description for {base_name}"),
            "content": kwargs.get("content", kwargs.get("system_prompt", 
                "You are a helpful AI assistant created for testing.")),
            "author": kwargs.get("author", "Test Factory"),
            "created_at": kwargs.get("created_at", datetime.now().isoformat()),
            "updated_at": kwargs.get("updated_at", datetime.now().isoformat()),
            **kwargs
        }


class ContentItemFactory:
    """Factory for creating ContentItem objects."""
    
    @staticmethod
    def create(
        content_type: ContentType,
        item_id: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> ContentItem:
        """Create a ContentItem."""
        item_id = item_id or str(uuid.uuid4())
        title = title or f"Test {content_type.value.title()} {item_id[:8]}"
        
        return ContentItem(
            id=item_id,
            type=content_type,
            title=title,
            description=kwargs.get("description", f"Description for {title}"),
            created_at=kwargs.get("created_at", datetime.now()),
            updated_at=kwargs.get("updated_at", datetime.now()),
            tags=kwargs.get("tags", ["test"]),
            metadata=kwargs.get("metadata", {}),
            file_path=kwargs.get("file_path", 
                f"content/{content_type.value}s/{content_type.value}_{item_id}.json")
        )


class ManifestFactory:
    """Factory for creating ChatbookManifest objects."""
    
    @staticmethod
    def create(
        name: Optional[str] = None,
        content_items: Optional[List[ContentItem]] = None,
        **kwargs
    ) -> ChatbookManifest:
        """Create a ChatbookManifest."""
        manifest_name = name or f"Test Chatbook {uuid.uuid4().hex[:8]}"
        
        manifest = ChatbookManifest(
            version=kwargs.get("version", ChatbookVersion.V1),
            name=manifest_name,
            description=kwargs.get("description", f"Test chatbook: {manifest_name}"),
            author=kwargs.get("author", "Test Factory"),
            tags=kwargs.get("tags", ["test", "factory"]),
            categories=kwargs.get("categories", ["testing"]),
            include_media=kwargs.get("include_media", True),
            include_embeddings=kwargs.get("include_embeddings", False),
            media_quality=kwargs.get("media_quality", "thumbnail")
        )
        
        if content_items:
            manifest.content_items = content_items
            
            # Update statistics
            for item in content_items:
                if item.type == ContentType.CONVERSATION:
                    manifest.total_conversations += 1
                elif item.type == ContentType.NOTE:
                    manifest.total_notes += 1
                elif item.type == ContentType.CHARACTER:
                    manifest.total_characters += 1
                elif item.type == ContentType.MEDIA:
                    manifest.total_media_items += 1
        
        return manifest


class ChatbookDataFactory:
    """Factory for creating complete chatbook test data."""
    
    @staticmethod
    def create_simple_chatbook() -> Dict[str, Any]:
        """Create a simple chatbook with minimal content."""
        character = CharacterFactory.create(name="Simple Character")
        conversation = ConversationFactory.create(
            name="Simple Conversation",
            character_id=1,
            message_count=3
        )
        note = NoteFactory.create(title="Simple Note")
        
        content_items = [
            ContentItemFactory.create(ContentType.CHARACTER, "char1", character["name"]),
            ContentItemFactory.create(ContentType.CONVERSATION, "conv1", conversation["name"]),
            ContentItemFactory.create(ContentType.NOTE, "note1", note["title"])
        ]
        
        manifest = ManifestFactory.create(
            name="Simple Test Chatbook",
            content_items=content_items
        )
        
        return {
            "manifest": manifest,
            "characters": [character],
            "conversations": [conversation],
            "notes": [note],
            "media_items": [],
            "prompts": []
        }
    
    @staticmethod
    def create_complex_chatbook() -> Dict[str, Any]:
        """Create a complex chatbook with multiple content types."""
        # Create multiple characters
        characters = CharacterFactory.create_batch(3)
        
        # Create conversations with different characters
        conversations = []
        for i, char in enumerate(characters):
            for j in range(2):  # 2 conversations per character
                conv = ConversationFactory.create(
                    name=f"Conversation {i}-{j} with {char['name']}",
                    character_id=i+1,
                    message_count=10
                )
                conversations.append(conv)
        
        # Create various notes
        notes = [
            NoteFactory.create(title="Research Notes", tags=["research", "important"]),
            NoteFactory.create(title="Meeting Minutes", tags=["meeting", "work"]),
            NoteFactory.create(title="Ideas", tags=["brainstorm", "creative"])
        ]
        
        # Create media items
        media_items = [
            MediaFactory.create(title="Tutorial Video", media_type="video"),
            MediaFactory.create(title="Podcast Episode", media_type="audio"),
            MediaFactory.create(title="Reference Document", media_type="document")
        ]
        
        # Create prompts
        prompts = [
            PromptFactory.create(name="Analysis Prompt"),
            PromptFactory.create(name="Creative Writing Prompt")
        ]
        
        # Create content items
        content_items = []
        
        for i, char in enumerate(characters):
            content_items.append(
                ContentItemFactory.create(ContentType.CHARACTER, f"char{i+1}", char["name"])
            )
        
        for i, conv in enumerate(conversations):
            content_items.append(
                ContentItemFactory.create(ContentType.CONVERSATION, f"conv{i+1}", conv["name"])
            )
        
        for i, note in enumerate(notes):
            content_items.append(
                ContentItemFactory.create(ContentType.NOTE, f"note{i+1}", note["title"])
            )
        
        for i, media in enumerate(media_items):
            content_items.append(
                ContentItemFactory.create(ContentType.MEDIA, f"media{i+1}", media["title"])
            )
        
        for i, prompt in enumerate(prompts):
            content_items.append(
                ContentItemFactory.create(ContentType.PROMPT, f"prompt{i+1}", prompt["name"])
            )
        
        manifest = ManifestFactory.create(
            name="Complex Test Chatbook",
            content_items=content_items,
            description="A complex chatbook with multiple content types for testing"
        )
        
        return {
            "manifest": manifest,
            "characters": characters,
            "conversations": conversations,
            "notes": notes,
            "media_items": media_items,
            "prompts": prompts
        }