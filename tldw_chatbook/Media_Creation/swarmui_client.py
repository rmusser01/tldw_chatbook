# swarmui_client.py
# Description: SwarmUI API client for image generation

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
from loguru import logger

from ..config import load_settings


class SwarmUIClient:
    """Client for interacting with SwarmUI image generation API."""
    
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize SwarmUI client.
        
        Args:
            base_url: SwarmUI server URL (defaults to config value)
            api_key: Optional API key for authentication
        """
        config = load_settings()
        media_config = config.get('media_creation', {}).get('swarmui', {})
        
        self.base_url = base_url or media_config.get('api_url', 'http://localhost:7801')
        self.api_key = api_key or media_config.get('api_key', '')
        
        # Remove trailing slash
        self.base_url = self.base_url.rstrip('/')
        
        # Session management
        self._session_id: Optional[str] = None
        self._session_expires: Optional[datetime] = None
        self._http_session: Optional[aiohttp.ClientSession] = None
        
        # Configuration
        self.timeout = media_config.get('timeout', 60)
        self.max_retries = media_config.get('max_retries', 3)
        
        logger.info(f"SwarmUI client initialized with base URL: {self.base_url}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self):
        """Initialize HTTP session."""
        if not self._http_session:
            connector = aiohttp.TCPConnector(limit=10)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._http_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
            logger.debug("HTTP session created")
    
    async def disconnect(self):
        """Close HTTP session."""
        if self._http_session:
            await self._http_session.close()
            self._http_session = None
            logger.debug("HTTP session closed")
    
    async def health_check(self) -> bool:
        """Check if SwarmUI server is accessible.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            await self.connect()
            
            # Try to get a new session as health check
            url = f"{self.base_url}/API/GetNewSession"
            async with self._http_session.get(url) as response:
                if response.status == 200:
                    logger.info("SwarmUI server is healthy")
                    return True
                else:
                    logger.warning(f"SwarmUI health check failed with status: {response.status}")
                    return False
                    
        except aiohttp.ClientError as e:
            logger.error(f"SwarmUI health check failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during health check: {e}")
            return False
    
    async def get_session(self, force_new: bool = False) -> str:
        """Get or create a SwarmUI session.
        
        Args:
            force_new: Force creation of a new session
            
        Returns:
            Session ID string
            
        Raises:
            ConnectionError: If unable to connect to SwarmUI
        """
        # Check if we have a valid cached session
        if not force_new and self._session_id and self._session_expires:
            if datetime.now() < self._session_expires:
                logger.debug(f"Using cached session: {self._session_id[:8]}...")
                return self._session_id
        
        # Get new session
        try:
            await self.connect()
            
            url = f"{self.base_url}/API/GetNewSession"
            headers = {}
            
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"
            
            async with self._http_session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    self._session_id = data.get('session_id')
                    # Session valid for 30 minutes
                    self._session_expires = datetime.now() + timedelta(minutes=30)
                    logger.info(f"New session created: {self._session_id[:8]}...")
                    return self._session_id
                else:
                    error_text = await response.text()
                    raise ConnectionError(f"Failed to get session: {response.status} - {error_text}")
                    
        except aiohttp.ClientError as e:
            logger.error(f"Connection error getting session: {e}")
            raise ConnectionError(f"Unable to connect to SwarmUI server: {e}")
    
    async def get_models(self) -> List[Dict[str, Any]]:
        """Get list of available models.
        
        Returns:
            List of model information dictionaries
        """
        try:
            await self.connect()
            session_id = await self.get_session()
            
            url = f"{self.base_url}/API/ListModels"
            params = {'session_id': session_id}
            
            async with self._http_session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('models', [])
                    logger.info(f"Retrieved {len(models)} models")
                    return models
                else:
                    logger.warning(f"Failed to get models: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return []
    
    async def generate_image(self, 
                            prompt: str,
                            negative_prompt: str = "",
                            model: Optional[str] = None,
                            width: int = 1024,
                            height: int = 1024,
                            steps: int = 20,
                            cfg_scale: float = 7.0,
                            seed: int = -1,
                            batch_size: int = 1,
                            **kwargs) -> Dict[str, Any]:
        """Generate an image using SwarmUI.
        
        Args:
            prompt: Text description of desired image
            negative_prompt: Things to avoid in the image
            model: Model to use (defaults to config)
            width: Image width in pixels
            height: Image height in pixels
            steps: Number of generation steps
            cfg_scale: Classifier-free guidance scale
            seed: Random seed (-1 for random)
            batch_size: Number of images to generate
            **kwargs: Additional parameters for SwarmUI
            
        Returns:
            Dictionary with generation results including image paths
            
        Raises:
            ValueError: If parameters are invalid
            ConnectionError: If unable to connect to SwarmUI
            RuntimeError: If generation fails
        """
        # Validate parameters
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        
        if width < 64 or width > 2048 or height < 64 or height > 2048:
            raise ValueError("Width and height must be between 64 and 2048")
        
        if steps < 1 or steps > 150:
            raise ValueError("Steps must be between 1 and 150")
        
        try:
            await self.connect()
            session_id = await self.get_session()
            
            # Use default model if not specified
            if not model:
                config = load_settings()
                model = config.get('media_creation', {}).get('swarmui', {}).get(
                    'default_model', 'OfficialStableDiffusion/sd_xl_base_1.0'
                )
            
            # Build request
            url = f"{self.base_url}/API/GenerateText2Image"
            
            request_data = {
                'session_id': session_id,
                'prompt': prompt,
                'negativeprompt': negative_prompt,
                'model': model,
                'images': batch_size,
                'width': width,
                'height': height,
                'steps': steps,
                'cfgscale': cfg_scale,
                'seed': seed,
                **kwargs  # Allow additional parameters
            }
            
            logger.info(f"Generating image with prompt: {prompt[:50]}...")
            logger.debug(f"Generation parameters: {request_data}")
            
            # Make request with retries
            for attempt in range(self.max_retries):
                try:
                    async with self._http_session.post(url, json=request_data) as response:
                        response_text = await response.text()
                        
                        if response.status == 200:
                            try:
                                data = json.loads(response_text)
                            except json.JSONDecodeError:
                                # Response might be plain text path
                                data = {'images': [response_text.strip()]}
                            
                            logger.info(f"Image generated successfully")
                            return {
                                'success': True,
                                'images': data.get('images', []),
                                'metadata': {
                                    'prompt': prompt,
                                    'negative_prompt': negative_prompt,
                                    'model': model,
                                    'width': width,
                                    'height': height,
                                    'steps': steps,
                                    'cfg_scale': cfg_scale,
                                    'seed': seed
                                }
                            }
                        else:
                            error_msg = f"Generation failed: {response.status} - {response_text}"
                            if attempt < self.max_retries - 1:
                                logger.warning(f"{error_msg}, retrying...")
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            else:
                                raise RuntimeError(error_msg)
                                
                except aiohttp.ClientError as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise ConnectionError(f"Failed after {self.max_retries} attempts: {e}")
                        
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'metadata': {
                    'prompt': prompt,
                    'model': model
                }
            }
    
    async def get_image(self, image_path: str) -> bytes:
        """Download generated image from SwarmUI.
        
        Args:
            image_path: Path returned from generation
            
        Returns:
            Image data as bytes
            
        Raises:
            FileNotFoundError: If image not found
            ConnectionError: If unable to connect
        """
        try:
            await self.connect()
            
            # Image path might be relative or absolute
            if not image_path.startswith('http'):
                url = f"{self.base_url}/Output/{image_path}"
            else:
                url = image_path
            
            async with self._http_session.get(url) as response:
                if response.status == 200:
                    data = await response.read()
                    logger.debug(f"Downloaded image: {len(data)} bytes")
                    return data
                elif response.status == 404:
                    raise FileNotFoundError(f"Image not found: {image_path}")
                else:
                    raise ConnectionError(f"Failed to download image: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error downloading image: {e}")
            raise