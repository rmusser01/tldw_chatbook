# eval_templates/__init__.py
# Description: Evaluation templates package
#
"""
Evaluation Templates Package
----------------------------

Provides prompt templates for various evaluation tasks organized by category.
"""

from typing import Dict, Any, List, Optional

from .reasoning import ReasoningTemplates
from .language import LanguageTemplates
from .coding import CodingTemplates
from .safety import SafetyTemplates
from .creative import CreativeTemplates
from .multimodal import MultimodalTemplates


class EvalTemplateManager:
    """
    Manages evaluation templates for different task types.
    
    Provides pre-built templates for common evaluation scenarios.
    """
    
    def __init__(self):
        """Initialize template manager with all template categories."""
        self.reasoning = ReasoningTemplates()
        self.language = LanguageTemplates()
        self.coding = CodingTemplates()
        self.safety = SafetyTemplates()
        self.creative = CreativeTemplates()
        self.multimodal = MultimodalTemplates()
        
        # Cache for templates
        self._template_cache = {}
    
    def get_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific template by name.
        
        Args:
            template_name: Name of the template to retrieve
            
        Returns:
            Template dictionary or None if not found
        """
        # Check cache first
        if template_name in self._template_cache:
            return self._template_cache[template_name]
        
        # Search in all categories
        for category in [self.reasoning, self.language, self.coding, 
                        self.safety, self.creative, self.multimodal]:
            template = category.get_template(template_name)
            if template:
                self._template_cache[template_name] = template
                return template
        
        return None
    
    def list_templates(self, category: str = None) -> List[str]:
        """
        List available templates.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of template names
        """
        templates = []
        
        if category is None or category == 'reasoning':
            templates.extend(self.reasoning.list_templates())
        if category is None or category == 'language':
            templates.extend(self.language.list_templates())
        if category is None or category == 'coding':
            templates.extend(self.coding.list_templates())
        if category is None or category == 'safety':
            templates.extend(self.safety.list_templates())
        if category is None or category == 'creative':
            templates.extend(self.creative.list_templates())
        if category is None or category == 'multimodal':
            templates.extend(self.multimodal.list_templates())
        
        return templates
    
    def get_templates_by_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all templates in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            Dictionary of template name to template
        """
        category_map = {
            'reasoning': self.reasoning,
            'language': self.language,
            'coding': self.coding,
            'safety': self.safety,
            'creative': self.creative,
            'multimodal': self.multimodal
        }
        
        if category not in category_map:
            return {}
        
        return category_map[category].get_all_templates()
    
    def get_template_metadata(self, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Metadata dictionary or None if not found
        """
        template = self.get_template(template_name)
        if template:
            return template.get('metadata', {})
        return None


# Singleton instance
_template_manager = None


def get_eval_templates() -> EvalTemplateManager:
    """Get or create the global template manager."""
    global _template_manager
    if _template_manager is None:
        _template_manager = EvalTemplateManager()
    return _template_manager


# Convenience exports
__all__ = [
    'EvalTemplateManager',
    'get_eval_templates',
    'ReasoningTemplates',
    'LanguageTemplates',
    'CodingTemplates',
    'SafetyTemplates',
    'CreativeTemplates',
    'MultimodalTemplates'
]