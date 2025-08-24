# image_generation_service.py
# Description: High-level service for image generation with context integration

import os
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import aiofiles
from loguru import logger

from .swarmui_client import SwarmUIClient
from .generation_templates import GenerationTemplate, get_template, BUILTIN_TEMPLATES
from ..config import load_settings
from ..Utils.paths import get_user_data_dir


@dataclass
class GenerationResult:
    """Result of an image generation request."""
    success: bool
    images: List[str]  # Paths to generated images
    prompt: str
    negative_prompt: str
    parameters: Dict[str, Any]
    error: Optional[str] = None
    generation_time: Optional[float] = None
    template_used: Optional[str] = None


class ImageGenerationService:
    """Service for managing image generation with templates and context."""
    
    def __init__(self):
        """Initialize the image generation service."""
        self.client: Optional[SwarmUIClient] = None
        self.output_dir = self._setup_output_directory()
        self._generation_cache: Dict[str, GenerationResult] = {}
        logger.info("Image generation service initialized")
    
    def _setup_output_directory(self) -> Path:
        """Setup directory for storing generated images.
        
        Returns:
            Path to output directory
        """
        user_data = get_user_data_dir()
        output_dir = user_data / "generated_images"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        (output_dir / "temp").mkdir(exist_ok=True)
        (output_dir / "saved").mkdir(exist_ok=True)
        
        logger.debug(f"Output directory setup at: {output_dir}")
        return output_dir
    
    async def initialize(self) -> bool:
        """Initialize the service and check SwarmUI availability.
        
        Returns:
            True if SwarmUI is available, False otherwise
        """
        try:
            self.client = SwarmUIClient()
            await self.client.connect()
            
            # Check if server is available
            is_healthy = await self.client.health_check()
            if is_healthy:
                logger.info("SwarmUI server is available")
            else:
                logger.warning("SwarmUI server is not responding")
            
            return is_healthy
            
        except Exception as e:
            logger.error(f"Failed to initialize image generation service: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.client:
            await self.client.disconnect()
            self.client = None
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from SwarmUI.
        
        Returns:
            List of model information
        """
        if not self.client:
            await self.initialize()
        
        if self.client:
            return await self.client.get_models()
        return []
    
    def extract_context_from_messages(self, messages: List[Dict[str, Any]], 
                                     max_context_length: int = 500) -> Dict[str, Any]:
        """Extract relevant context from chat messages for image generation.
        
        Args:
            messages: List of chat messages
            max_context_length: Maximum length of context to extract
            
        Returns:
            Dictionary with extracted context elements
        """
        context = {
            'last_message': '',
            'mentioned_characters': [],
            'mentioned_settings': [],
            'mood': '',
            'style_hints': []
        }
        
        if not messages:
            return context
        
        # Get last user message
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                context['last_message'] = msg.get('content', '')[:max_context_length]
                break
        
        # Extract potential visual elements from recent messages
        recent_messages = messages[-5:] if len(messages) > 5 else messages
        combined_text = ' '.join([m.get('content', '') for m in recent_messages])
        
        # Simple keyword extraction (could be enhanced with NLP)
        visual_keywords = ['looks like', 'appears', 'wearing', 'standing', 'sitting', 
                          'background', 'scene', 'environment', 'style', 'color']
        
        for keyword in visual_keywords:
            if keyword in combined_text.lower():
                # Extract sentence containing keyword
                sentences = combined_text.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        context['style_hints'].append(sentence.strip()[:100])
        
        # Detect mood words
        mood_words = {
            'happy': ['joyful', 'cheerful', 'bright', 'sunny'],
            'dark': ['gloomy', 'shadow', 'night', 'mysterious'],
            'epic': ['grand', 'majestic', 'powerful', 'dramatic'],
            'calm': ['peaceful', 'serene', 'tranquil', 'quiet']
        }
        
        for mood, words in mood_words.items():
            if any(word in combined_text.lower() for word in words):
                context['mood'] = mood
                break
        
        logger.debug(f"Extracted context: {context}")
        return context
    
    async def generate_from_template(self, 
                                    template_id: str,
                                    custom_params: Optional[Dict[str, Any]] = None,
                                    context: Optional[Dict[str, Any]] = None) -> GenerationResult:
        """Generate image using a template.
        
        Args:
            template_id: ID of template to use
            custom_params: Optional parameter overrides
            context: Optional context from conversation
            
        Returns:
            GenerationResult with outcome
        """
        start_time = datetime.now()
        
        try:
            # Get template
            template = get_template(template_id)
            if not template:
                return GenerationResult(
                    success=False,
                    images=[],
                    prompt="",
                    negative_prompt="",
                    parameters={},
                    error=f"Template not found: {template_id}"
                )
            
            # Apply template
            prompt = template.base_prompt
            negative_prompt = template.negative_prompt
            params = template.default_params.copy()
            
            # Apply context if provided
            if context and template.context_mappings:
                for key, mapping in template.context_mappings.items():
                    if key in context and context[key]:
                        prompt = prompt.replace(f"{{{{{key}}}}}", context[key])
            
            # Apply custom parameter overrides
            if custom_params:
                params.update(custom_params)
                if 'prompt' in custom_params:
                    prompt = custom_params['prompt']
                if 'negative_prompt' in custom_params:
                    negative_prompt = custom_params['negative_prompt']
            
            # Generate image
            result = await self.generate_custom(prompt, negative_prompt, **params)
            result.template_used = template_id
            
            return result
            
        except Exception as e:
            logger.error(f"Template generation failed: {e}")
            return GenerationResult(
                success=False,
                images=[],
                prompt="",
                negative_prompt="",
                parameters={},
                error=str(e)
            )
    
    async def generate_from_conversation(self,
                                        conversation_messages: List[Dict[str, Any]],
                                        base_prompt: Optional[str] = None,
                                        **kwargs) -> GenerationResult:
        """Generate image based on conversation context.
        
        Args:
            conversation_messages: List of conversation messages
            base_prompt: Optional base prompt to enhance with context
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult with outcome
        """
        # Extract context from conversation
        context = self.extract_context_from_messages(conversation_messages)
        
        # Build prompt from context
        if base_prompt:
            prompt = base_prompt
        else:
            prompt = context.get('last_message', '')
        
        # Enhance prompt with context
        if context.get('mood'):
            prompt = f"{prompt}, {context['mood']} mood"
        
        if context.get('style_hints'):
            hints = ', '.join(context['style_hints'][:2])  # Use first 2 hints
            prompt = f"{prompt}, {hints}"
        
        # Use a default negative prompt if not provided
        negative_prompt = kwargs.pop('negative_prompt', 'blurry, low quality, bad anatomy')
        
        return await self.generate_custom(prompt, negative_prompt, **kwargs)
    
    async def generate_custom(self,
                            prompt: str,
                            negative_prompt: str = "",
                            **kwargs) -> GenerationResult:
        """Generate image with custom parameters.
        
        Args:
            prompt: Image description
            negative_prompt: Things to avoid
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult with outcome
        """
        start_time = datetime.now()
        
        try:
            if not self.client:
                await self.initialize()
            
            if not self.client:
                return GenerationResult(
                    success=False,
                    images=[],
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    parameters=kwargs,
                    error="SwarmUI service not available"
                )
            
            # Generate image
            logger.info(f"Generating image: {prompt[:50]}...")
            result = await self.client.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                **kwargs
            )
            
            if result['success']:
                # Save images locally
                saved_paths = []
                for image_path in result['images']:
                    try:
                        # Download image data
                        image_data = await self.client.get_image(image_path)
                        
                        # Save to local directory
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"generated_{timestamp}_{len(saved_paths)}.png"
                        local_path = self.output_dir / "temp" / filename
                        
                        async with aiofiles.open(local_path, 'wb') as f:
                            await f.write(image_data)
                        
                        saved_paths.append(str(local_path))
                        logger.debug(f"Saved image to: {local_path}")
                        
                    except Exception as e:
                        logger.error(f"Failed to save image {image_path}: {e}")
                
                generation_time = (datetime.now() - start_time).total_seconds()
                
                return GenerationResult(
                    success=True,
                    images=saved_paths,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    parameters=kwargs,
                    generation_time=generation_time
                )
            else:
                return GenerationResult(
                    success=False,
                    images=[],
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    parameters=kwargs,
                    error=result.get('error', 'Unknown error')
                )
                
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return GenerationResult(
                success=False,
                images=[],
                prompt=prompt,
                negative_prompt=negative_prompt,
                parameters=kwargs,
                error=str(e)
            )
    
    async def save_generation(self, result: GenerationResult, name: Optional[str] = None) -> List[str]:
        """Save a generation result permanently.
        
        Args:
            result: GenerationResult to save
            name: Optional name for the saved files
            
        Returns:
            List of paths to saved files
        """
        saved_paths = []
        
        try:
            for i, temp_path in enumerate(result.images):
                temp_file = Path(temp_path)
                if temp_file.exists():
                    # Generate filename
                    if name:
                        filename = f"{name}_{i}.png" if len(result.images) > 1 else f"{name}.png"
                    else:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"saved_{timestamp}_{i}.png"
                    
                    # Move from temp to saved
                    saved_path = self.output_dir / "saved" / filename
                    temp_file.rename(saved_path)
                    saved_paths.append(str(saved_path))
                    
                    logger.info(f"Saved generation to: {saved_path}")
                    
        except Exception as e:
            logger.error(f"Failed to save generation: {e}")
        
        return saved_paths
    
    def cleanup_temp_images(self, older_than_hours: int = 24):
        """Clean up temporary images older than specified hours.
        
        Args:
            older_than_hours: Age threshold in hours
        """
        try:
            temp_dir = self.output_dir / "temp"
            cutoff_time = datetime.now().timestamp() - (older_than_hours * 3600)
            
            for file in temp_dir.glob("*.png"):
                if file.stat().st_mtime < cutoff_time:
                    file.unlink()
                    logger.debug(f"Cleaned up old temp image: {file}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up temp images: {e}")