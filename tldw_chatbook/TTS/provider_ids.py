"""Canonical identifiers for Chatbook's built-in TTS providers."""

from __future__ import annotations


BUILT_IN_TTS_PROVIDER_IDS: tuple[str, ...] = (
    "audio_cpp",
    "openai",
    "elevenlabs",
    "kokoro",
    "chatterbox",
    "higgs",
    "alltalk",
)
"""The bounded provider IDs shared by TTS domain and Speech UI contracts."""


__all__ = ["BUILT_IN_TTS_PROVIDER_IDS"]
