"""Compatibility imports for the original native agent storage source."""

from tldw_chatbook.Agents.native_source_storage import (
    NativeAgentSources as _AgentSources,  # noqa: F401 - original source type import
    console_writer_source as console_writer_source,
    original_retained_source_stack as _original_retained_source_stack,  # noqa: F401 - original storage lifetime import
)

from .console_preparation_reads import current_agent_sources as current_agent_sources
