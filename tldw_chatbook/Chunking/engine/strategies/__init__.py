"""
Chunking strategies for different text processing methods.
Each strategy implements the ChunkingStrategy protocol from base.py.
"""

from ..base import ChunkingStrategy
from .fixed_size import FixedSizeChunkingStrategy
from .json_xml import JSONChunkingStrategy, XMLChunkingStrategy
from .semantic import SemanticChunkingStrategy
from .sentences import SentenceChunkingStrategy
from .structure_aware import StructureAwareChunkingStrategy
from .tokens import TokenChunkingStrategy

# Import strategies as they are implemented
from .words import WordChunkingStrategy

# Strategy registry will be populated as strategies are implemented
STRATEGY_REGISTRY: dict[str, type[ChunkingStrategy]] = {
    'words': WordChunkingStrategy,
    'sentences': SentenceChunkingStrategy,
    'tokens': TokenChunkingStrategy,
    'structure_aware': StructureAwareChunkingStrategy,
    'semantic': SemanticChunkingStrategy,
    'json': JSONChunkingStrategy,
    'xml': XMLChunkingStrategy,
    'fixed_size': FixedSizeChunkingStrategy,
}

def get_strategy(name: str) -> type[ChunkingStrategy]:
    """
    Get a chunking strategy by name.

    Args:
        name: Strategy name

    Returns:
        Strategy class

    Raises:
        ValueError: If strategy not found
    """
    if name not in STRATEGY_REGISTRY:
        available = ', '.join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy: {name}. Available: {available}")
    return STRATEGY_REGISTRY[name]

__all__ = ['STRATEGY_REGISTRY', 'get_strategy']
