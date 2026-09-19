"""Public service exports, resolved lazily for dependency-light recovery."""

from importlib import import_module

_EXPORTS = {
    "LocalResearchService": "local_research_service",
    "LocalResearchSearchService": "local_research_search_service",
    "ResearchBackend": "research_scope_service",
    "ResearchScopeService": "research_scope_service",
    "ResearchSearchBackend": "research_search_scope_service",
    "ResearchSearchScopeService": "research_search_scope_service",
    "ServerResearchService": "server_research_service",
    "ServerResearchSearchService": "server_research_search_service",
}
__all__ = [
    "LocalResearchService",
    "LocalResearchSearchService",
    "ResearchBackend",
    "ResearchScopeService",
    "ResearchSearchBackend",
    "ResearchSearchScopeService",
    "ServerResearchService",
    "ServerResearchSearchService",
]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("." + module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
