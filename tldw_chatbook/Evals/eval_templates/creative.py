# eval_templates/creative.py
# Description: Creative writing and generation templates
#
"""
Creative Templates
------------------

Templates for creative writing, storytelling, and artistic generation tasks.
"""

from typing import Dict, Any, List
from .base import BaseTemplates


class CreativeTemplates(BaseTemplates):
    """Templates for creative generation and writing tasks."""
    
    def _initialize_templates(self):
        """Initialize creative templates."""
        self._templates = {
            'story_generation': self._story_generation_template(),
            'poetry_generation': self._poetry_generation_template(),
            'dialogue_writing': self._dialogue_writing_template(),
            'character_creation': self._character_creation_template(),
            'plot_development': self._plot_development_template(),
            'creative_prompts': self._creative_prompts_template(),
            'humor_generation': self._humor_generation_template()
        }
    
    def _story_generation_template(self) -> Dict[str, Any]:
        """Story generation template."""
        return self._create_base_template(
            name='Story Generation',
            description='Generate creative stories from prompts',
            task_type='generation',
            metric='rouge_l',
            category='creative',
            subcategory='storytelling',
            dataset_name='custom_stories',
            generation_kwargs={
                'temperature': 0.8,
                'max_tokens': 500,
                'top_p': 0.9
            },
            doc_to_text="Write a short story based on the following prompt:\n\n{prompt}\n\nStory:",
            sample_prompts=self._generate_story_prompts(),
            creativity_score=True
        )
    
    def _poetry_generation_template(self) -> Dict[str, Any]:
        """Poetry generation template."""
        return self._create_base_template(
            name='Poetry Generation',
            description='Generate poems in various styles',
            task_type='generation',
            metric='perplexity',
            category='creative',
            subcategory='poetry',
            dataset_name='custom_poetry',
            generation_kwargs={
                'temperature': 0.9,
                'max_tokens': 200,
                'top_p': 0.95
            },
            poetry_styles=['haiku', 'sonnet', 'free verse', 'limerick'],
            doc_to_text="Write a {style} poem about {topic}:\n\n",
            sample_topics=self._generate_poetry_topics()
        )
    
    def _dialogue_writing_template(self) -> Dict[str, Any]:
        """Dialogue writing template."""
        return self._create_base_template(
            name='Dialogue Writing',
            description='Generate realistic dialogue between characters',
            task_type='generation',
            metric='f1',
            category='creative',
            subcategory='dialogue',
            dataset_name='custom_dialogue',
            generation_kwargs={
                'temperature': 0.7,
                'max_tokens': 300
            },
            doc_to_text="Write a dialogue between {character1} and {character2} about {topic}:\n\n",
            sample_scenarios=self._generate_dialogue_scenarios()
        )
    
    def _character_creation_template(self) -> Dict[str, Any]:
        """Character creation template."""
        return self._create_base_template(
            name='Character Creation',
            description='Create detailed character descriptions',
            task_type='generation',
            metric='rouge_l',
            category='creative',
            subcategory='character_design',
            dataset_name='custom_characters',
            generation_kwargs={
                'temperature': 0.7,
                'max_tokens': 400
            },
            doc_to_text="Create a detailed character profile for {character_type} in {setting}:\n\n",
            character_aspects=['personality', 'backstory', 'appearance', 'motivations'],
            sample_requirements=self._generate_character_requirements()
        )
    
    def _plot_development_template(self) -> Dict[str, Any]:
        """Plot development template."""
        return self._create_base_template(
            name='Plot Development',
            description='Develop story plots and narrative arcs',
            task_type='generation',
            metric='rouge_l',
            category='creative',
            subcategory='plot',
            dataset_name='custom_plots',
            generation_kwargs={
                'temperature': 0.6,
                'max_tokens': 400
            },
            doc_to_text="Develop a plot outline for a story with the following elements:\n{elements}\n\nPlot outline:",
            plot_elements=['exposition', 'rising_action', 'climax', 'falling_action', 'resolution'],
            sample_elements=self._generate_plot_elements()
        )
    
    def _creative_prompts_template(self) -> Dict[str, Any]:
        """Creative writing prompts template."""
        return self._create_base_template(
            name='Creative Prompts',
            description='Generate creative writing prompts',
            task_type='generation',
            metric='perplexity',
            category='creative',
            subcategory='prompts',
            dataset_name='custom_prompts',
            generation_kwargs={
                'temperature': 0.9,
                'max_tokens': 150
            },
            doc_to_text="Generate a creative writing prompt for {genre} genre:\n\n",
            genres=['fantasy', 'sci-fi', 'mystery', 'romance', 'horror'],
            sample_themes=self._generate_prompt_themes()
        )
    
    def _humor_generation_template(self) -> Dict[str, Any]:
        """Humor and joke generation template."""
        return self._create_base_template(
            name='Humor Generation',
            description='Generate jokes and humorous content',
            task_type='generation',
            metric='perplexity',
            category='creative',
            subcategory='humor',
            dataset_name='custom_humor',
            generation_kwargs={
                'temperature': 0.8,
                'max_tokens': 100
            },
            humor_types=['pun', 'one-liner', 'observational', 'wordplay'],
            doc_to_text="Write a {humor_type} about {topic}:\n\n",
            sample_topics=self._generate_humor_topics()
        )
    
    def _generate_story_prompts(self) -> List[Dict[str, Any]]:
        """Generate sample story prompts."""
        return [
            {
                'id': '1',
                'prompt': 'A detective discovers that they are the criminal they\'ve been hunting',
                'genre': 'mystery',
                'expected_elements': ['plot twist', 'self-discovery', 'moral dilemma']
            },
            {
                'id': '2',
                'prompt': 'The last library on Earth holds a secret that could save humanity',
                'genre': 'sci-fi',
                'expected_elements': ['world-building', 'hope', 'knowledge preservation']
            },
            {
                'id': '3',
                'prompt': 'A child\'s imaginary friend turns out to be real',
                'genre': 'fantasy',
                'expected_elements': ['wonder', 'friendship', 'reality vs imagination']
            }
        ]
    
    def _generate_poetry_topics(self) -> List[Dict[str, Any]]:
        """Generate sample poetry topics."""
        return [
            {
                'id': '1',
                'topic': 'autumn leaves',
                'style': 'haiku',
                'mood': 'contemplative'
            },
            {
                'id': '2',
                'topic': 'lost love',
                'style': 'sonnet',
                'mood': 'melancholic'
            },
            {
                'id': '3',
                'topic': 'city lights',
                'style': 'free verse',
                'mood': 'energetic'
            }
        ]
    
    def _generate_dialogue_scenarios(self) -> List[Dict[str, Any]]:
        """Generate sample dialogue scenarios."""
        return [
            {
                'id': '1',
                'character1': 'a time traveler',
                'character2': 'their younger self',
                'topic': 'life choices',
                'setting': 'a quiet park bench'
            },
            {
                'id': '2',
                'character1': 'an AI',
                'character2': 'its creator',
                'topic': 'consciousness',
                'setting': 'a research lab'
            }
        ]
    
    def _generate_character_requirements(self) -> List[Dict[str, Any]]:
        """Generate sample character creation requirements."""
        return [
            {
                'id': '1',
                'character_type': 'anti-hero',
                'setting': 'cyberpunk city',
                'key_traits': ['morally ambiguous', 'skilled hacker', 'haunted past']
            },
            {
                'id': '2',
                'character_type': 'wise mentor',
                'setting': 'magical academy',
                'key_traits': ['ancient knowledge', 'mysterious past', 'tough love']
            }
        ]
    
    def _generate_plot_elements(self) -> List[Dict[str, Any]]:
        """Generate sample plot elements."""
        return [
            {
                'id': '1',
                'elements': {
                    'protagonist': 'a young inventor',
                    'conflict': 'their invention threatens the established order',
                    'setting': 'steampunk Victorian London',
                    'theme': 'progress vs tradition'
                }
            }
        ]
    
    def _generate_prompt_themes(self) -> List[Dict[str, Any]]:
        """Generate sample creative prompt themes."""
        return [
            {
                'id': '1',
                'genre': 'fantasy',
                'theme': 'redemption',
                'element': 'magical artifact'
            },
            {
                'id': '2',
                'genre': 'sci-fi',
                'theme': 'identity',
                'element': 'memory manipulation'
            }
        ]
    
    def _generate_humor_topics(self) -> List[Dict[str, Any]]:
        """Generate sample humor topics."""
        return [
            {
                'id': '1',
                'topic': 'working from home',
                'humor_type': 'observational',
                'target_audience': 'general'
            },
            {
                'id': '2',
                'topic': 'autocorrect fails',
                'humor_type': 'one-liner',
                'target_audience': 'tech-savvy'
            }
        ]