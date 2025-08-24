# swarmui_events.py
# Description: Event handlers and messages for SwarmUI image generation

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from textual.message import Message
from textual import on, work
from loguru import logger

from ...Media_Creation import ImageGenerationService, GenerationResult


@dataclass
class SwarmUIGenerateRequest(Message):
    """Request to generate an image."""
    prompt: str
    negative_prompt: str = ""
    template_id: Optional[str] = None
    parameters: Dict[str, Any] = None
    context: Optional[Dict[str, Any]] = None
    use_conversation_context: bool = False
    conversation_id: Optional[int] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class SwarmUIGenerationComplete(Message):
    """Image generation completed successfully."""
    result: GenerationResult
    conversation_id: Optional[int] = None


@dataclass
class SwarmUIGenerationError(Message):
    """Image generation failed."""
    error: str
    prompt: str
    conversation_id: Optional[int] = None


@dataclass 
class SwarmUIStatusUpdate(Message):
    """Status update for SwarmUI service."""
    status: str  # "online", "offline", "generating", "idle"
    message: Optional[str] = None
    progress: Optional[float] = None  # 0.0 to 1.0


class SwarmUIEventHandler:
    """Mixin class for handling SwarmUI events."""
    
    def __init__(self):
        """Initialize the event handler."""
        self._generation_service = None
        self._generation_in_progress = False
        
    def get_generation_service(self) -> ImageGenerationService:
        """Get or create the generation service.
        
        Returns:
            ImageGenerationService instance
        """
        if not self._generation_service:
            self._generation_service = ImageGenerationService()
        return self._generation_service
    
    @on(SwarmUIGenerateRequest)
    async def handle_generation_request(self, event: SwarmUIGenerateRequest) -> None:
        """Handle image generation request.
        
        Args:
            event: Generation request event
        """
        if self._generation_in_progress:
            logger.warning("Generation already in progress, ignoring request")
            return
        
        self._generation_in_progress = True
        
        try:
            # Post status update
            self.post_message(SwarmUIStatusUpdate(
                status="generating",
                message=f"Generating: {event.prompt[:50]}..."
            ))
            
            service = self.get_generation_service()
            
            # Initialize service if needed
            if not await service.initialize():
                raise ConnectionError("SwarmUI service not available")
            
            # Generate based on request type
            if event.template_id:
                # Use template
                result = await service.generate_from_template(
                    template_id=event.template_id,
                    custom_params=event.parameters,
                    context=event.context
                )
            elif event.use_conversation_context and event.conversation_id:
                # Use conversation context
                # TODO: Get conversation messages from database
                messages = []  # This would be fetched from DB
                result = await service.generate_from_conversation(
                    conversation_messages=messages,
                    base_prompt=event.prompt,
                    **event.parameters
                )
            else:
                # Custom generation
                result = await service.generate_custom(
                    prompt=event.prompt,
                    negative_prompt=event.negative_prompt,
                    **event.parameters
                )
            
            # Post result
            if result.success:
                self.post_message(SwarmUIGenerationComplete(
                    result=result,
                    conversation_id=event.conversation_id
                ))
                self.post_message(SwarmUIStatusUpdate(
                    status="idle",
                    message=f"Generated in {result.generation_time:.1f}s"
                ))
            else:
                self.post_message(SwarmUIGenerationError(
                    error=result.error or "Unknown error",
                    prompt=event.prompt,
                    conversation_id=event.conversation_id
                ))
                self.post_message(SwarmUIStatusUpdate(
                    status="idle",
                    message=f"Generation failed: {result.error}"
                ))
                
        except Exception as e:
            logger.error(f"Generation request failed: {e}")
            self.post_message(SwarmUIGenerationError(
                error=str(e),
                prompt=event.prompt,
                conversation_id=event.conversation_id
            ))
            self.post_message(SwarmUIStatusUpdate(
                status="idle",
                message=f"Error: {str(e)}"
            ))
            
        finally:
            self._generation_in_progress = False
    
    @on(SwarmUIGenerationComplete)
    async def handle_generation_complete(self, event: SwarmUIGenerationComplete) -> None:
        """Handle generation completion.
        
        Args:
            event: Generation complete event
        """
        logger.info(f"Image generation complete: {len(event.result.images)} images")
        
        # Save to database if associated with conversation
        if event.conversation_id:
            await self.save_generation_to_db(event.result, event.conversation_id)
    
    @on(SwarmUIGenerationError)
    async def handle_generation_error(self, event: SwarmUIGenerationError) -> None:
        """Handle generation error.
        
        Args:
            event: Generation error event
        """
        logger.error(f"Image generation error: {event.error}")
        
        # Could show notification or update UI
    
    async def save_generation_to_db(self, result: GenerationResult, conversation_id: int) -> None:
        """Save generation result to database.
        
        Args:
            result: Generation result
            conversation_id: Associated conversation ID
        """
        # TODO: Implement database saving
        # This would save to the media_generations table
        logger.info(f"Would save generation to DB for conversation {conversation_id}")
    
    async def cleanup_generation_service(self) -> None:
        """Cleanup generation service resources."""
        if self._generation_service:
            await self._generation_service.cleanup()
            self._generation_service = None


def integrate_swarmui_events(app_or_widget):
    """Decorator to integrate SwarmUI event handling into a class.
    
    Usage:
        @integrate_swarmui_events
        class MyApp(App):
            pass
    """
    # Mix in the event handler
    original_bases = app_or_widget.__bases__
    app_or_widget.__bases__ = (SwarmUIEventHandler,) + original_bases
    
    # Wrap __init__ to initialize handler
    original_init = app_or_widget.__init__
    
    def new_init(self, *args, **kwargs):
        SwarmUIEventHandler.__init__(self)
        original_init(self, *args, **kwargs)
    
    app_or_widget.__init__ = new_init
    
    return app_or_widget