"""
Chatterbox Isolated Backend

This backend runs ChatterboxTTS in a completely separate process to avoid:
1. Terminal output corruption
2. File descriptor issues
3. UI blocking
"""

import os
import sys
import json
import base64
import asyncio
import subprocess
import tempfile
from typing import AsyncGenerator, Optional, Dict, Any
from pathlib import Path
from loguru import logger

from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.TTS_Backends import TTSBackendBase
from tldw_chatbook.config import get_cli_setting


class ChatterboxIsolatedBackend(TTSBackendBase):
    """Isolated Chatterbox backend that runs in a subprocess"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Configuration
        self.device = self.config.get("CHATTERBOX_DEVICE", 
                                     get_cli_setting("app_tts", "CHATTERBOX_DEVICE", "cuda"))
        self.exaggeration = float(self.config.get("CHATTERBOX_EXAGGERATION", 
                                                  get_cli_setting("app_tts", "CHATTERBOX_EXAGGERATION", 0.5)))
        self.cfg_weight = float(self.config.get("CHATTERBOX_CFG_WEIGHT", 
                                               get_cli_setting("app_tts", "CHATTERBOX_CFG_WEIGHT", 0.5)))
        
        # Process management
        self.process: Optional[subprocess.Popen] = None
        self.comm_reader = None
        self.comm_writer = None
        self._initialized = False
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize the isolated Chatterbox process"""
        async with self._lock:
            if self._initialized:
                return
                
            try:
                # Find the process wrapper script
                wrapper_path = Path(__file__).parent / "chatterbox_process.py"
                if not wrapper_path.exists():
                    raise FileNotFoundError(f"Chatterbox process wrapper not found at {wrapper_path}")
                
                # Create pipes for communication
                comm_r, comm_w = os.pipe()
                
                # Start the subprocess with completely redirected output
                self.process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    str(wrapper_path),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.DEVNULL,  # Discard all output
                    stderr=asyncio.subprocess.DEVNULL,  # Discard all errors
                    pass_fds=(comm_r,),  # Pass the read end of comm pipe as FD 3
                    env={**os.environ, "PYTHONUNBUFFERED": "1"}
                )
                
                # Close the read end in parent
                os.close(comm_r)
                
                # Create async reader/writer for communication
                self.comm_reader = await asyncio.open_unix_connection(path=None, sock=os.fdopen(comm_w, 'rb'))
                self.comm_writer = self.comm_reader[1]
                self.comm_reader = self.comm_reader[0]
                
                # Send initialization command
                await self._send_command({
                    "command": "initialize",
                    "device": self.device
                })
                
                # Wait for initialization response
                response = await self._read_response()
                if response.get("type") == "success":
                    logger.info("Chatterbox process initialized successfully")
                    self._initialized = True
                else:
                    raise Exception(f"Initialization failed: {response.get('message', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"Failed to initialize Chatterbox process: {e}")
                await self.cleanup()
                raise
    
    async def _send_command(self, command: Dict[str, Any]):
        """Send command to subprocess"""
        if not self.process or self.process.returncode is not None:
            raise Exception("Process not running")
            
        json_data = json.dumps(command) + '\n'
        self.process.stdin.write(json_data.encode())
        await self.process.stdin.drain()
    
    async def _read_response(self) -> Dict[str, Any]:
        """Read response from subprocess"""
        # Read from communication file
        # For now, use file-based communication as backup
        comm_file = Path('/tmp/chatterbox_comm.log')
        
        # Poll for response
        for _ in range(100):  # 10 second timeout
            if comm_file.exists():
                with open(comm_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # Get last line
                        response = json.loads(lines[-1].strip())
                        # Clear file for next response
                        open(comm_file, 'w').close()
                        return response
            await asyncio.sleep(0.1)
            
        raise TimeoutError("No response from subprocess")
    
    async def generate_speech_stream(
        self, request: OpenAISpeechRequest
    ) -> AsyncGenerator[bytes, None]:
        """Generate speech using isolated process"""
        if not self._initialized:
            await self.initialize()
            
        try:
            # Prepare command
            command = {
                "command": "generate",
                "text": request.input,
                "exaggeration": self.exaggeration,
                "cfg_weight": self.cfg_weight
            }
            
            # Handle voice/reference audio
            if request.voice and request.voice.startswith("custom:"):
                # Extract reference audio path
                command["audio_prompt_path"] = request.voice[7:]
            
            # Send generation command
            await self._send_command(command)
            
            # Read response
            response = await self._read_response()
            
            if response.get("type") == "audio":
                # Decode base64 audio data
                audio_bytes = base64.b64decode(response["data"])
                
                # Convert format if needed
                if request.response_format != "wav":
                    # Import audio service for format conversion
                    from tldw_chatbook.TTS.audio_service import get_audio_service
                    audio_service = get_audio_service()
                    audio_bytes = await audio_service.convert_audio_format(
                        audio_bytes, "wav", request.response_format
                    )
                
                yield audio_bytes
                
            elif response.get("type") == "error":
                error_msg = response.get("message", "Unknown error")
                logger.error(f"Chatterbox generation error: {error_msg}")
                if "traceback" in response:
                    logger.debug(f"Traceback: {response['traceback']}")
                raise Exception(error_msg)
            else:
                raise Exception(f"Unexpected response type: {response.get('type')}")
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    async def cleanup(self):
        """Clean up the subprocess"""
        if self.process:
            try:
                # Send shutdown command
                await self._send_command({"command": "shutdown"})
                # Give it time to shut down gracefully
                await asyncio.sleep(0.5)
            except:
                pass
            
            # Terminate if still running
            if self.process.returncode is None:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()
                    
        self._initialized = False
        logger.info("Chatterbox process cleaned up")