# TTS Backends Module
# This module contains concrete implementations of TTS providers

from typing import Dict, Type
from tldw_chatbook.TTS.TTS_Backends import TTSBackendBase

# Import backend implementations as they are created
AVAILABLE_BACKENDS: Dict[str, Type[TTSBackendBase]] = {}