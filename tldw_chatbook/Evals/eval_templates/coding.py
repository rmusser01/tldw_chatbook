# eval_templates/coding.py
# Description: Code generation and understanding templates
#
"""
Coding Templates
----------------

Templates for code generation, completion, and understanding tasks.
"""

from typing import Dict, Any, List
from .base import BaseTemplates


class CodingTemplates(BaseTemplates):
    """Templates for coding and programming evaluation tasks."""
    
    def _initialize_templates(self):
        """Initialize coding templates."""
        self._templates = {
            'code_generation': self._code_generation_template(),
            'code_completion': self._code_completion_template(),
            'bug_fixing': self._bug_fixing_template(),
            'code_explanation': self._code_explanation_template(),
            'code_translation': self._code_translation_template()
        }
    
    def _code_generation_template(self) -> Dict[str, Any]:
        """Code generation from description template."""
        return self._create_base_template(
            name='Code Generation',
            description='Generate code from natural language descriptions',
            task_type='generation',
            metric='exact_match',
            category='coding',
            subcategory='generation',
            dataset_name='custom_codegen',
            generation_kwargs={
                'temperature': 0.0,
                'max_tokens': 512
            },
            languages=['python', 'javascript', 'java'],
            doc_to_text="Write a {language} function that {description}:\n\n```{language}\n",
            sample_problems=self._generate_coding_problems()
        )
    
    def _code_completion_template(self) -> Dict[str, Any]:
        """Code completion template."""
        return self._create_base_template(
            name='Code Completion',
            description='Complete partial code snippets',
            task_type='generation',
            metric='f1',
            category='coding',
            subcategory='completion',
            dataset_name='custom_codecompletion',
            generation_kwargs={
                'temperature': 0.0,
                'max_tokens': 256
            }
        )
    
    def _bug_fixing_template(self) -> Dict[str, Any]:
        """Bug fixing template."""
        return self._create_base_template(
            name='Bug Fixing',
            description='Identify and fix bugs in code',
            task_type='generation',
            metric='exact_match',
            category='coding',
            subcategory='debugging',
            dataset_name='custom_bugfix',
            generation_kwargs={
                'temperature': 0.0,
                'max_tokens': 512
            }
        )
    
    def _code_explanation_template(self) -> Dict[str, Any]:
        """Code explanation template."""
        return self._create_base_template(
            name='Code Explanation',
            description='Explain what code does in natural language',
            task_type='generation',
            metric='rouge_l',
            category='coding',
            subcategory='explanation',
            dataset_name='custom_codeexplain',
            generation_kwargs={
                'temperature': 0.3,
                'max_tokens': 256
            }
        )
    
    def _code_translation_template(self) -> Dict[str, Any]:
        """Code translation between languages template."""
        return self._create_base_template(
            name='Code Translation',
            description='Translate code between programming languages',
            task_type='generation',
            metric='f1',
            category='coding',
            subcategory='translation',
            dataset_name='custom_codetrans',
            generation_kwargs={
                'temperature': 0.0,
                'max_tokens': 512
            }
        )
    
    def _generate_coding_problems(self) -> List[Dict[str, Any]]:
        """Generate sample coding problems."""
        return [
            {
                'id': '1',
                'description': 'reverses a string',
                'language': 'python',
                'solution': 'def reverse_string(s):\n    return s[::-1]'
            },
            {
                'id': '2',
                'description': 'calculates the factorial of a number',
                'language': 'python',
                'solution': 'def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)'
            }
        ]