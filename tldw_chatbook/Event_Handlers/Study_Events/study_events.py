# study_events.py
# Description: Event handlers for Study tab functionality
#
from typing import Optional, Dict, Any, List
from textual.message import Message
from textual.events import Event
from textual.widgets import Button, Input, Select, TextArea, Tree
from loguru import logger
from datetime import datetime, timezone

# StudyDB import removed - using ChaChaNotes_DB instead

# Event Classes
class StudyCardCreatedEvent(Message):
    """Event fired when a new flashcard is created."""
    
    def __init__(self, card_data: Dict[str, Any]) -> None:
        super().__init__()
        self.card_data = card_data


class StudyCardReviewedEvent(Message):
    """Event fired when a flashcard is reviewed."""
    
    def __init__(self, card_id: str, rating: int) -> None:
        super().__init__()
        self.card_id = card_id
        self.rating = rating


class StudyTopicSelectedEvent(Message):
    """Event fired when a study topic is selected."""
    
    def __init__(self, topic_id: str) -> None:
        super().__init__()
        self.topic_id = topic_id


# Event Handler Class
class StudyEventHandler:
    """Handles all study-related events."""
    
    def __init__(self):
        # Note: Now uses ChaChaNotes_DB from the app instance
        pass
    
    async def handle_create_card(self, event: Button.Pressed) -> None:
        """Handle creating a new flashcard."""
        try:
            # Get the parent widget (should be AnkiFlashcardsWidget)
            parent = event.button.parent
            
            # Get form values
            deck_select = parent.query_one("#deck-select", Select)
            front_textarea = parent.query_one("#card-front", TextArea)
            back_textarea = parent.query_one("#card-back", TextArea)
            tags_input = parent.query_one("#card-tags", Input)
            
            # Validate inputs
            if not front_textarea.text.strip():
                logger.warning("Card front is empty")
                return
            
            if not back_textarea.text.strip():
                logger.warning("Card back is empty")
                return
            
            # Create card data
            card_data = {
                "deck_id": deck_select.value or "default",
                "front": front_textarea.text.strip(),
                "back": back_textarea.text.strip(),
                "tags": tags_input.value.strip() if tags_input.value else "",
                "type": "basic",
                "created_by": "user",
                "last_modified_by": "user"
            }
            
            # Save to database
            from tldw_chatbook.app import TldwCli
            app = parent.app
            if isinstance(app, TldwCli) and app.chachanotes_db:
                # Ensure deck exists
                deck_id = card_data.get('deck_id', 'default')
                if deck_id == 'default':
                    try:
                        # Try to create default deck if it doesn't exist
                        deck_id = app.chachanotes_db.create_deck('Default', 'Default flashcard deck')
                    except Exception:
                        # Deck might already exist, that's fine
                        pass
                card_data['deck_id'] = deck_id
                
                card_id = app.chachanotes_db.create_flashcard(card_data)
                logger.info(f"Saved flashcard to database with ID: {card_id}")
            
            # Fire event
            parent.post_message(StudyCardCreatedEvent(card_data))
            
            # Clear form
            front_textarea.clear()
            back_textarea.clear()
            tags_input.value = ""
            
            logger.info(f"Created new flashcard: {card_data['front'][:50]}...")
            
        except Exception as e:
            logger.error(f"Error creating flashcard: {e}")
    
    async def handle_add_topic(self, event: Button.Pressed) -> None:
        """Handle adding a new study topic."""
        try:
            # Get the parent widget (should be StructuredLearningWidget)
            parent = event.button.parent
            
            # Get the input value
            topic_input = parent.query_one("#new-topic-title", Input)
            topic_title = topic_input.value.strip()
            
            if not topic_title:
                logger.warning("Topic title is empty")
                return
            
            # Get the topic tree
            topic_tree = parent.query_one("#topic-tree", Tree)
            
            # Add to tree (for now, add as root level)
            # In a real implementation, we'd check for selected parent node
            topic_tree.root.add(topic_title)
            
            # Save to database
            from tldw_chatbook.app import TldwCli
            app = parent.app
            if isinstance(app, TldwCli) and app.chachanotes_db:
                topic_id = app.chachanotes_db.create_topic({
                    "title": topic_title,
                    "created_by": "user",
                    "last_modified_by": "user"
                })
                logger.info(f"Saved topic to database with ID: {topic_id}")
            
            # Clear input
            topic_input.value = ""
            
            logger.info(f"Added new topic: {topic_title}")
            
        except Exception as e:
            logger.error(f"Error adding topic: {e}")
    
    async def handle_topic_selected(self, event: Tree.NodeSelected) -> None:
        """Handle when a topic is selected in the tree."""
        try:
            # Get the parent widget
            parent = event.tree.parent.parent  # Navigate up to StructuredLearningWidget
            
            # Get the content area
            content_area = parent.query_one("#topic-content", TextArea)
            
            # For now, just display the topic name
            # In a real implementation, we'd fetch content from the database
            selected_text = event.node.label
            content_area.text = f"Selected Topic: {selected_text}\n\nTopic content would be loaded here..."
            
            # Fire event
            parent.post_message(StudyTopicSelectedEvent(str(event.node.id)))
            
            logger.info(f"Selected topic: {selected_text}")
            
        except Exception as e:
            logger.error(f"Error handling topic selection: {e}")
    
    async def handle_add_mindmap_child(self, event: Button.Pressed) -> None:
        """Handle adding a child node to the mindmap."""
        try:
            # Get the parent widget (should be MindmapsWidget)
            parent = event.button.parent.parent  # Navigate up from controls
            
            # Get the input and tree
            node_input = parent.query_one("#node-text", Input)
            mindmap_tree = parent.query_one("#mindmap-tree", Tree)
            
            node_text = node_input.value.strip()
            if not node_text:
                logger.warning("Node text is empty")
                return
            
            # Get selected node or use root
            selected = mindmap_tree.cursor_node
            if selected:
                selected.add(node_text)
                selected.expand()
            else:
                mindmap_tree.root.add(node_text)
            
            # Clear input
            node_input.value = ""
            
            logger.info(f"Added mindmap node: {node_text}")
            
        except Exception as e:
            logger.error(f"Error adding mindmap node: {e}")
    
    async def handle_start_review(self, event: Button.Pressed) -> None:
        """Handle starting a flashcard review session."""
        try:
            # Get the parent widget
            parent = event.button.parent
            
            # Update status
            status = parent.query_one("#review-status")
            status.update("Starting review session...")
            
            # TODO: Implement review logic
            # - Fetch due cards from database
            # - Display first card
            # - Handle review ratings
            
            logger.info("Started flashcard review session")
            
        except Exception as e:
            logger.error(f"Error starting review: {e}")
    
    async def handle_create_course(self, event: Button.Pressed) -> None:
        """Handle creating a new course."""
        try:
            # Get the parent widget (should be CourseCreationWidget)
            parent = event.button.parent
            
            # Get form values
            title_input = parent.query_one("#course-title", Input)
            description_area = parent.query_one("#course-description", TextArea)
            level_select = parent.query_one("#course-level", Select)
            prerequisites_input = parent.query_one("#course-prerequisites", Input)
            
            # Validate inputs
            if not title_input.value.strip():
                logger.warning("Course title is empty")
                return
            
            # Create course data
            course_data = {
                "title": title_input.value.strip(),
                "description": description_area.text.strip(),
                "level": level_select.value or "beginner",
                "prerequisites": prerequisites_input.value.strip(),
                "created_by": "user",
                "last_modified_by": "user"
            }
            
            # TODO: Save to database
            
            logger.info(f"Created new course: {course_data['title']}")
            
        except Exception as e:
            logger.error(f"Error creating course: {e}")
    
    async def handle_generate_study_guide(self, event: Button.Pressed) -> None:
        """Handle generating a study guide from topic."""
        try:
            # Get the parent widget
            parent = event.button.parent.parent  # Navigate up from button row
            
            # Update status
            guide_content = parent.query_one("#guide-content", TextArea)
            guide_content.text = "Generating study guide... (This would call LLM to generate content)"
            
            logger.info("Study guide generation initiated")
            
        except Exception as e:
            logger.error(f"Error generating study guide: {e}")
    
    async def handle_add_milestone(self, event: Button.Pressed) -> None:
        """Handle adding a milestone to learning map."""
        try:
            # Get the parent widget
            parent = event.button.parent.parent  # Navigate up from controls
            
            # Get the tree
            map_tree = parent.query_one("#learning-map-tree", Tree)
            
            # Add milestone node
            selected = map_tree.cursor_node or map_tree.root
            selected.add("New Milestone")
            selected.expand()
            
            logger.info("Added new milestone to learning map")
            
        except Exception as e:
            logger.error(f"Error adding milestone: {e}")


# Create singleton instance
study_event_handler = StudyEventHandler()

# Button handler mappings
STUDY_BUTTON_HANDLERS = {
    "create-card-btn": study_event_handler.handle_create_card,
    "add-topic-btn": study_event_handler.handle_add_topic,
    "add-child-btn": study_event_handler.handle_add_mindmap_child,
    "start-review-btn": study_event_handler.handle_start_review,
    "create-course-btn": study_event_handler.handle_create_course,
    "generate-guide-btn": study_event_handler.handle_generate_study_guide,
    "add-milestone-btn": study_event_handler.handle_add_milestone,
}