# eval_templates/base.py
# Description: Base class for evaluation templates
#
"""
Base Template Class
-------------------

Abstract base class for all template categories.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseTemplates(ABC):
    """Abstract base class for template categories."""
    
    def __init__(self):
        """Initialize template category."""
        self._templates = {}
        self._initialize_templates()
    
    @abstractmethod
    def _initialize_templates(self):
        """Initialize all templates for this category."""
        pass
    
    def get_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific template by name.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Template dictionary or None if not found
        """
        # Try exact match first
        if template_name in self._templates:
            return self._templates[template_name]
        
        # Try case-insensitive match
        for name, template in self._templates.items():
            if name.lower() == template_name.lower():
                return template
        
        return None
    
    def list_templates(self) -> List[str]:
        """
        List all available template names.
        
        Returns:
            List of template names
        """
        return list(self._templates.keys())
    
    def get_all_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all templates in this category.
        
        Returns:
            Dictionary of template name to template
        """
        return self._templates.copy()
    
    def add_template(self, name: str, template: Dict[str, Any]):
        """
        Add a new template to this category.
        
        Args:
            name: Template name
            template: Template dictionary
        """
        self._templates[name] = template
    
    @staticmethod
    def _create_base_template(
        name: str,
        description: str,
        task_type: str,
        metric: str,
        category: str,
        subcategory: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a base template with common fields.
        
        Args:
            name: Template name
            description: Template description
            task_type: Type of task (question_answer, classification, etc.)
            metric: Evaluation metric to use
            category: Main category
            subcategory: Optional subcategory
            **kwargs: Additional template fields
            
        Returns:
            Base template dictionary
        """
        template = {
            'name': name,
            'description': description,
            'task_type': task_type,
            'metric': metric,
            'metadata': {
                'category': category,
                'subcategory': subcategory
            }
        }
        
        # Add any additional fields
        template.update(kwargs)
        
        return template