"""Native WebRTC AEC3 processing for tldw_chatbook."""

from ._native import AecProcessor as AecProcessor
from ._native import DUPLEX_ABI_VERSION as DUPLEX_ABI_VERSION
from ._native import NativeDuplexBridge as NativeDuplexBridge

__all__ = ["AecProcessor", "DUPLEX_ABI_VERSION", "NativeDuplexBridge"]
