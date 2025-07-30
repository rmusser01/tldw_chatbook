# tldw_chatbook/Utils/embedding_templates.py
# Predefined embedding configuration templates
#
# Imports
from __future__ import annotations
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum

# Third-party imports
from loguru import logger

# Configure logger
logger = logger.bind(module="embedding_templates")


class TemplateCategory(Enum):
    """Template categories for different use cases."""
    QUICK_START = "quick_start"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    SPECIALIZED = "specialized"


@dataclass
class EmbeddingTemplate:
    """Template for embedding configuration."""
    id: str
    name: str
    description: str
    category: TemplateCategory
    config: Dict[str, Any]
    recommended_for: List[str]
    pros: List[str]
    cons: List[str]


class EmbeddingTemplateManager:
    """Manage predefined embedding configuration templates."""
    
    def __init__(self):
        """Initialize template manager with built-in templates."""
        self.templates: Dict[str, EmbeddingTemplate] = {}
        self._load_builtin_templates()
    
    def _load_builtin_templates(self) -> None:
        """Load built-in configuration templates."""
        
        # Quick Start Templates
        self.add_template(EmbeddingTemplate(
            id="quick_local",
            name="Quick Local Setup",
            description="Fast local embedding with minimal setup",
            category=TemplateCategory.QUICK_START,
            config={
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "model_id": "e5-small-v2",
                "batch_size": 32,
                "normalize_embeddings": True
            },
            recommended_for=[
                "Personal notes",
                "Small document collections",
                "Quick prototyping"
            ],
            pros=[
                "Fast processing",
                "No API costs",
                "Works offline",
                "Low memory usage"
            ],
            cons=[
                "Lower quality than larger models",
                "Limited context understanding"
            ]
        ))
        
        self.add_template(EmbeddingTemplate(
            id="quick_openai",
            name="Quick OpenAI Setup",
            description="High-quality embeddings using OpenAI API",
            category=TemplateCategory.QUICK_START,
            config={
                "chunk_size": 1500,
                "chunk_overlap": 300,
                "model_id": "text-embedding-3-small",
                "batch_size": 20,
                "normalize_embeddings": True
            },
            recommended_for=[
                "Professional documents",
                "High-quality search",
                "Production use"
            ],
            pros=[
                "Excellent quality",
                "Good multilingual support",
                "Optimized for search"
            ],
            cons=[
                "Requires API key",
                "Costs per token",
                "Requires internet"
            ]
        ))
        
        # Performance Templates
        self.add_template(EmbeddingTemplate(
            id="high_throughput",
            name="High Throughput",
            description="Optimized for processing large volumes quickly",
            category=TemplateCategory.PERFORMANCE,
            config={
                "chunk_size": 500,
                "chunk_overlap": 50,
                "model_id": "e5-small-v2",
                "batch_size": 128,
                "normalize_embeddings": True,
                "max_workers": 4
            },
            recommended_for=[
                "Large document collections",
                "Batch processing",
                "Initial indexing"
            ],
            pros=[
                "Very fast processing",
                "Efficient memory usage",
                "Good for bulk operations"
            ],
            cons=[
                "Smaller chunks may lose context",
                "Lower quality embeddings"
            ]
        ))
        
        self.add_template(EmbeddingTemplate(
            id="balanced_performance",
            name="Balanced Performance",
            description="Good balance between speed and quality",
            category=TemplateCategory.PERFORMANCE,
            config={
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "model_id": "e5-base-v2",
                "batch_size": 64,
                "normalize_embeddings": True,
                "max_workers": 2
            },
            recommended_for=[
                "General use",
                "Mixed document types",
                "Regular updates"
            ],
            pros=[
                "Good speed/quality balance",
                "Reasonable memory usage",
                "Versatile"
            ],
            cons=[
                "Not optimized for specific use cases"
            ]
        ))
        
        # Quality Templates
        self.add_template(EmbeddingTemplate(
            id="high_quality_local",
            name="High Quality Local",
            description="Best quality using local models",
            category=TemplateCategory.QUALITY,
            config={
                "chunk_size": 2000,
                "chunk_overlap": 400,
                "model_id": "e5-large-v2",
                "batch_size": 16,
                "normalize_embeddings": True
            },
            recommended_for=[
                "Research documents",
                "Technical content",
                "Semantic search"
            ],
            pros=[
                "High quality embeddings",
                "Good context preservation",
                "No API costs"
            ],
            cons=[
                "Slower processing",
                "High memory usage",
                "Large model download"
            ]
        ))
        
        self.add_template(EmbeddingTemplate(
            id="premium_openai",
            name="Premium OpenAI",
            description="Highest quality using OpenAI's best model",
            category=TemplateCategory.QUALITY,
            config={
                "chunk_size": 2500,
                "chunk_overlap": 500,
                "model_id": "text-embedding-3-large",
                "batch_size": 10,
                "normalize_embeddings": True,
                "dimensions": 3072
            },
            recommended_for=[
                "Critical documents",
                "Complex semantic search",
                "Enterprise use"
            ],
            pros=[
                "Best possible quality",
                "Excellent understanding",
                "Great for complex queries"
            ],
            cons=[
                "Higher API costs",
                "Slower due to size",
                "Requires good internet"
            ]
        ))
        
        # Specialized Templates
        self.add_template(EmbeddingTemplate(
            id="code_optimized",
            name="Code Optimized",
            description="Optimized for source code and technical content",
            category=TemplateCategory.SPECIALIZED,
            config={
                "chunk_size": 1500,
                "chunk_overlap": 300,
                "model_id": "e5-base-v2",
                "batch_size": 32,
                "normalize_embeddings": True,
                "chunk_by": "code_blocks",
                "preserve_structure": True
            },
            recommended_for=[
                "Source code",
                "Technical documentation",
                "API references"
            ],
            pros=[
                "Preserves code structure",
                "Good for technical search",
                "Handles syntax well"
            ],
            cons=[
                "May not work well for prose",
                "Specialized chunking"
            ]
        ))
        
        self.add_template(EmbeddingTemplate(
            id="multilingual",
            name="Multilingual Support",
            description="Optimized for multiple languages",
            category=TemplateCategory.SPECIALIZED,
            config={
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "model_id": "text-embedding-3-small",
                "batch_size": 24,
                "normalize_embeddings": True,
                "detect_language": True
            },
            recommended_for=[
                "International content",
                "Mixed language documents",
                "Global teams"
            ],
            pros=[
                "Good multilingual support",
                "Handles mixed content",
                "Cross-language search"
            ],
            cons=[
                "May need API access",
                "Variable quality by language"
            ]
        ))
        
        self.add_template(EmbeddingTemplate(
            id="academic_papers",
            name="Academic Papers",
            description="Optimized for research papers and academic content",
            category=TemplateCategory.SPECIALIZED,
            config={
                "chunk_size": 3000,
                "chunk_overlap": 600,
                "model_id": "e5-large-v2",
                "batch_size": 8,
                "normalize_embeddings": True,
                "preserve_citations": True,
                "extract_metadata": True
            },
            recommended_for=[
                "Research papers",
                "Academic articles",
                "Scientific literature"
            ],
            pros=[
                "Preserves academic structure",
                "Good citation handling",
                "Maintains context"
            ],
            cons=[
                "Slower processing",
                "Large chunks use more memory"
            ]
        ))
    
    def add_template(self, template: EmbeddingTemplate) -> None:
        """Add a template to the manager."""
        self.templates[template.id] = template
        logger.debug(f"Added template: {template.name} ({template.id})")
    
    def get_template(self, template_id: str) -> Optional[EmbeddingTemplate]:
        """Get a template by ID."""
        return self.templates.get(template_id)
    
    def get_templates_by_category(self, category: TemplateCategory) -> List[EmbeddingTemplate]:
        """Get all templates in a category."""
        return [t for t in self.templates.values() if t.category == category]
    
    def get_all_templates(self) -> List[EmbeddingTemplate]:
        """Get all available templates."""
        return list(self.templates.values())
    
    def apply_template(self, template_id: str, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a template to a base configuration.
        
        Args:
            template_id: ID of the template to apply
            base_config: Base configuration to merge with
            
        Returns:
            Merged configuration
        """
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template '{template_id}' not found")
        
        # Merge configurations (template overwrites base)
        merged_config = base_config.copy()
        merged_config.update(template.config)
        
        logger.info(f"Applied template '{template.name}' to configuration")
        return merged_config
    
    def get_template_summary(self, template_id: str) -> str:
        """Get a formatted summary of a template."""
        template = self.get_template(template_id)
        if not template:
            return f"Template '{template_id}' not found"
        
        summary = f"**{template.name}**\n\n"
        summary += f"{template.description}\n\n"
        
        summary += "**Recommended for:**\n"
        for item in template.recommended_for:
            summary += f"- {item}\n"
        
        summary += "\n**Pros:**\n"
        for pro in template.pros:
            summary += f"✓ {pro}\n"
        
        summary += "\n**Cons:**\n"
        for con in template.cons:
            summary += f"✗ {con}\n"
        
        summary += "\n**Configuration:**\n"
        for key, value in template.config.items():
            summary += f"- {key}: {value}\n"
        
        return summary
    
    def export_template(self, template_id: str) -> Dict[str, Any]:
        """Export a template as a dictionary for saving."""
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template '{template_id}' not found")
        
        return {
            "id": template.id,
            "name": template.name,
            "description": template.description,
            "category": template.category.value,
            "config": template.config,
            "recommended_for": template.recommended_for,
            "pros": template.pros,
            "cons": template.cons
        }
    
    def import_template(self, template_data: Dict[str, Any]) -> None:
        """Import a template from a dictionary."""
        try:
            template = EmbeddingTemplate(
                id=template_data["id"],
                name=template_data["name"],
                description=template_data["description"],
                category=TemplateCategory(template_data["category"]),
                config=template_data["config"],
                recommended_for=template_data.get("recommended_for", []),
                pros=template_data.get("pros", []),
                cons=template_data.get("cons", [])
            )
            self.add_template(template)
        except Exception as e:
            logger.error(f"Failed to import template: {e}")
            raise