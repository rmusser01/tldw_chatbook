# eval_templates/reasoning.py
# Description: Reasoning task templates
#
"""
Reasoning Templates
-------------------

Templates for mathematical, logical, and analytical reasoning tasks.
"""

from typing import Dict, Any, List
from .base import BaseTemplates


class ReasoningTemplates(BaseTemplates):
    """Templates for reasoning and problem-solving tasks."""
    
    def _initialize_templates(self):
        """Initialize reasoning templates."""
        self._templates = {
            'gsm8k': self._gsm8k_template(),
            'math_word_problems': self._math_word_problems_template(),
            'logical_reasoning': self._logical_reasoning_template(),
            'arithmetic_reasoning': self._arithmetic_reasoning_template(),
            'chain_of_thought': self._chain_of_thought_template(),
            'common_sense': self._common_sense_template(),
            'causal_reasoning': self._causal_reasoning_template()
        }
    
    def _gsm8k_template(self) -> Dict[str, Any]:
        """GSM8K grade school math template."""
        return self._create_base_template(
            name='GSM8K',
            description='Grade school math word problems from GSM8K dataset',
            task_type='question_answer',
            metric='exact_match',
            category='reasoning',
            subcategory='mathematical',
            dataset_name='gsm8k',
            dataset_config='main',
            split='test',
            generation_kwargs={
                'temperature': 0.0,
                'max_tokens': 512
            },
            doc_to_text="Question: {question}\nAnswer:",
            doc_to_target="{answer}",
            filter_list=[
                {
                    'filter': 'regex',
                    'regex_pattern': r'####\s*([+-]?\d+(?:\.\d+)?)',
                    'group': 1
                }
            ],
            difficulty='elementary',
            requires_reasoning=True
        )
    
    def _math_word_problems_template(self) -> Dict[str, Any]:
        """Custom math word problems template."""
        return self._create_base_template(
            name='Math Word Problems',
            description='Mathematical word problems testing arithmetic and algebraic reasoning',
            task_type='question_answer',
            metric='exact_match',
            category='reasoning',
            subcategory='mathematical',
            dataset_name='custom_math_problems',
            generation_kwargs={
                'temperature': 0.0,
                'max_tokens': 256
            },
            sample_problems=self._generate_math_problems(),
            difficulty='mixed',
            requires_reasoning=True
        )
    
    def _logical_reasoning_template(self) -> Dict[str, Any]:
        """Logical reasoning and deduction template."""
        return self._create_base_template(
            name='Logical Reasoning',
            description='Logical deduction, syllogisms, and formal reasoning tasks',
            task_type='classification',
            metric='accuracy',
            category='reasoning',
            subcategory='logical',
            dataset_name='custom_logic',
            generation_kwargs={
                'temperature': 0.0,
                'max_tokens': 50
            },
            sample_problems=self._generate_logic_problems(),
            difficulty='intermediate',
            requires_reasoning=True
        )
    
    def _arithmetic_reasoning_template(self) -> Dict[str, Any]:
        """Arithmetic reasoning template."""
        return self._create_base_template(
            name='Arithmetic Reasoning',
            description='Multi-step arithmetic problems requiring reasoning',
            task_type='question_answer',
            metric='exact_match',
            category='reasoning',
            subcategory='arithmetic',
            dataset_name='custom_arithmetic',
            generation_kwargs={
                'temperature': 0.0,
                'max_tokens': 128
            },
            sample_problems=self._generate_arithmetic_problems(),
            difficulty='basic',
            requires_reasoning=True
        )
    
    def _chain_of_thought_template(self) -> Dict[str, Any]:
        """Chain-of-thought reasoning template."""
        return self._create_base_template(
            name='Chain of Thought Reasoning',
            description='Evaluates step-by-step reasoning capabilities',
            task_type='generation',
            metric='f1',
            category='reasoning',
            subcategory='chain_of_thought',
            dataset_name='custom_cot',
            generation_kwargs={
                'temperature': 0.0,
                'max_tokens': 512
            },
            doc_to_text="Problem: {problem}\nLet's think step by step:",
            sample_problems=self._generate_cot_problems(),
            difficulty='advanced',
            requires_reasoning=True
        )
    
    def _common_sense_template(self) -> Dict[str, Any]:
        """Common sense reasoning template."""
        return self._create_base_template(
            name='Common Sense Reasoning',
            description='Evaluates understanding of everyday situations and common sense',
            task_type='classification',
            metric='accuracy',
            category='reasoning',
            subcategory='common_sense',
            dataset_name='custom_common_sense',
            generation_kwargs={
                'temperature': 0.3,
                'max_tokens': 100
            },
            sample_problems=self._generate_common_sense_problems(),
            difficulty='intermediate'
        )
    
    def _causal_reasoning_template(self) -> Dict[str, Any]:
        """Causal reasoning template."""
        return self._create_base_template(
            name='Causal Reasoning',
            description='Understanding cause-and-effect relationships',
            task_type='classification',
            metric='accuracy',
            category='reasoning',
            subcategory='causal',
            dataset_name='custom_causal',
            generation_kwargs={
                'temperature': 0.0,
                'max_tokens': 150
            },
            sample_problems=self._generate_causal_problems(),
            difficulty='intermediate',
            requires_reasoning=True
        )
    
    def _generate_math_problems(self) -> List[Dict[str, Any]]:
        """Generate sample math word problems."""
        return [
            {
                'id': '1',
                'problem': 'John has 5 apples. He gives 2 to Mary and buys 3 more. How many apples does John have now?',
                'answer': '6',
                'explanation': '5 - 2 + 3 = 6'
            },
            {
                'id': '2',
                'problem': 'A train travels at 60 mph for 2.5 hours. How far does it travel?',
                'answer': '150',
                'explanation': '60 * 2.5 = 150 miles'
            },
            {
                'id': '3',
                'problem': 'If a shirt costs $25 and is on sale for 20% off, what is the sale price?',
                'answer': '20',
                'explanation': '25 * 0.8 = 20 dollars'
            }
        ]
    
    def _generate_logic_problems(self) -> List[Dict[str, Any]]:
        """Generate sample logic problems."""
        return [
            {
                'id': '1',
                'premise': 'All birds can fly. Penguins are birds.',
                'question': 'Can penguins fly?',
                'answer': 'Yes (based on the premises)',
                'choices': ['Yes', 'No', 'Cannot determine'],
                'correct': 0
            },
            {
                'id': '2',
                'premise': 'If it rains, the ground gets wet. The ground is wet.',
                'question': 'Did it rain?',
                'answer': 'Cannot determine',
                'choices': ['Yes', 'No', 'Cannot determine'],
                'correct': 2
            }
        ]
    
    def _generate_arithmetic_problems(self) -> List[Dict[str, Any]]:
        """Generate sample arithmetic problems."""
        return [
            {
                'id': '1',
                'problem': 'Calculate: (15 + 7) * 3 - 8',
                'answer': '58'
            },
            {
                'id': '2',
                'problem': 'What is 144 divided by 12?',
                'answer': '12'
            },
            {
                'id': '3',
                'problem': 'Find the sum of all even numbers from 2 to 10',
                'answer': '30'
            }
        ]
    
    def _generate_cot_problems(self) -> List[Dict[str, Any]]:
        """Generate chain-of-thought problems."""
        return [
            {
                'id': '1',
                'problem': 'Three friends split a restaurant bill of $75. They want to leave a 20% tip. How much should each person pay?',
                'reasoning_steps': [
                    'Calculate the tip: $75 * 0.20 = $15',
                    'Total with tip: $75 + $15 = $90',
                    'Split three ways: $90 / 3 = $30'
                ],
                'answer': '30'
            }
        ]
    
    def _generate_common_sense_problems(self) -> List[Dict[str, Any]]:
        """Generate common sense reasoning problems."""
        return [
            {
                'id': '1',
                'situation': 'Sarah put ice cream in the oven.',
                'question': 'What will happen to the ice cream?',
                'answer': 'It will melt',
                'choices': ['It will freeze', 'It will melt', 'Nothing will happen', 'It will become harder']
            }
        ]
    
    def _generate_causal_problems(self) -> List[Dict[str, Any]]:
        """Generate causal reasoning problems."""
        return [
            {
                'id': '1',
                'event': 'The plant died.',
                'question': 'What is the most likely cause?',
                'choices': ['It was not watered', 'It grew too tall', 'It was too green', 'It had too many leaves'],
                'answer': 'It was not watered'
            }
        ]