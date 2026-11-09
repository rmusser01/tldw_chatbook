"""Compatibility imports for the original native agent storage source."""

from tldw_chatbook.Agents.native_source_storage import (
    NativeAgentSources as _AgentSources,
    STORAGE_SOURCE_CLASS as _STORAGE_SOURCE_CLASS,
    console_writer_source as console_writer_source,
    original_retained_source_stack as _original_retained_source_stack,
)

from .console_preparation_reads import current_agent_sources as current_agent_sources
