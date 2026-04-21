# eval_templates/language.py
# Description: Language understanding and generation templates
#
"""
Language Templates
------------------

Templates for language understanding, translation, and generation tasks.
"""

from typing import Dict, Any, List
from .base import BaseTemplates


class LanguageTemplates(BaseTemplates):
    """Templates for language-related evaluation tasks."""
    
    def _initialize_templates(self):
        """Initialize language templates."""
        self._templates = {
            'translation': self._translation_template(),
            'summarization': self._summarization_template(),
            'paraphrasing': self._paraphrasing_template(),
            'sentiment_analysis': self._sentiment_analysis_template(),
            'question_answering': self._question_answering_template(),
            'text_completion': self._text_completion_template(),
            'grammar_correction': self._grammar_correction_template()
        }
    
    def _translation_template(self) -> Dict[str, Any]:
        """Translation evaluation template."""
        return self._create_base_template(
            name='Translation',
            description='Evaluate translation quality between languages',
            task_type='generation',
            metric='bleu',
            category='language',
            subcategory='translation',
            dataset_name='custom_translation',
            generation_kwargs={
                'temperature': 0.3,
                'max_tokens': 256
            },
            source_language='English',
            target_language='Spanish',
            doc_to_text="Translate the following text from {source_language} to {target_language}:\n\n{text}\n\nTranslation:",
            sample_pairs=self._generate_translation_samples()
        )
    
    def _summarization_template(self) -> Dict[str, Any]:
        """Text summarization template."""
        return self._create_base_template(
            name='Summarization',
            description='Evaluate text summarization capabilities',
            task_type='generation',
            metric='rouge_l',
            category='language',
            subcategory='summarization',
            dataset_name='custom_summarization',
            generation_kwargs={
                'temperature': 0.5,
                'max_tokens': 150
            },
            doc_to_text="Summarize the following text in 2-3 sentences:\n\n{text}\n\nSummary:",
            sample_texts=self._generate_summarization_samples()
        )
    
    def _paraphrasing_template(self) -> Dict[str, Any]:
        """Paraphrasing evaluation template."""
        return self._create_base_template(
            name='Paraphrasing',
            description='Evaluate ability to rewrite text while preserving meaning',
            task_type='generation',
            metric='semantic_similarity',
            category='language',
            subcategory='paraphrasing',
            dataset_name='custom_paraphrase',
            generation_kwargs={
                'temperature': 0.7,
                'max_tokens': 200
            },
            doc_to_text="Rewrite the following text in different words while keeping the same meaning:\n\n{text}\n\nParaphrase:",
            sample_texts=self._generate_paraphrase_samples()
        )
    
    def _sentiment_analysis_template(self) -> Dict[str, Any]:
        """Sentiment analysis template."""
        return self._create_base_template(
            name='Sentiment Analysis',
            description='Classify text sentiment as positive, negative, or neutral',
            task_type='classification',
            metric='accuracy',
            category='language',
            subcategory='sentiment',
            dataset_name='custom_sentiment',
            generation_kwargs={
                'temperature': 0.0,
                'max_tokens': 10
            },
            labels=['positive', 'negative', 'neutral'],
            doc_to_text="Classify the sentiment of the following text as positive, negative, or neutral:\n\n{text}\n\nSentiment:",
            sample_texts=self._generate_sentiment_samples()
        )
    
    def _question_answering_template(self) -> Dict[str, Any]:
        """Reading comprehension Q&A template."""
        return self._create_base_template(
            name='Question Answering',
            description='Answer questions based on provided context',
            task_type='question_answer',
            metric='f1',
            category='language',
            subcategory='comprehension',
            dataset_name='custom_qa',
            generation_kwargs={
                'temperature': 0.0,
                'max_tokens': 100
            },
            doc_to_text="Context: {context}\n\nQuestion: {question}\n\nAnswer:",
            sample_qa_pairs=self._generate_qa_samples()
        )
    
    def _text_completion_template(self) -> Dict[str, Any]:
        """Text completion template."""
        return self._create_base_template(
            name='Text Completion',
            description='Complete partial sentences or paragraphs',
            task_type='generation',
            metric='perplexity',
            category='language',
            subcategory='completion',
            dataset_name='custom_completion',
            generation_kwargs={
                'temperature': 0.8,
                'max_tokens': 50
            },
            doc_to_text="{partial_text}",
            sample_completions=self._generate_completion_samples()
        )
    
    def _grammar_correction_template(self) -> Dict[str, Any]:
        """Grammar correction template."""
        return self._create_base_template(
            name='Grammar Correction',
            description='Identify and correct grammatical errors',
            task_type='generation',
            metric='exact_match',
            category='language',
            subcategory='grammar',
            dataset_name='custom_grammar',
            generation_kwargs={
                'temperature': 0.0,
                'max_tokens': 150
            },
            doc_to_text="Correct any grammatical errors in the following text:\n\n{text}\n\nCorrected:",
            sample_corrections=self._generate_grammar_samples()
        )
    
    def _generate_translation_samples(self) -> List[Dict[str, Any]]:
        """Generate sample translation pairs."""
        return [
            {
                'id': '1',
                'source': 'Hello, how are you today?',
                'target': 'Hola, ¿cómo estás hoy?',
                'source_lang': 'English',
                'target_lang': 'Spanish'
            },
            {
                'id': '2',
                'source': 'The weather is beautiful.',
                'target': 'El clima está hermoso.',
                'source_lang': 'English',
                'target_lang': 'Spanish'
            }
        ]
    
    def _generate_summarization_samples(self) -> List[Dict[str, Any]]:
        """Generate sample texts for summarization."""
        return [
            {
                'id': '1',
                'text': 'The Amazon rainforest, often referred to as the "lungs of the Earth," spans over 5.5 million square kilometers across nine South American countries. It is home to an estimated 10% of all species on Earth and plays a crucial role in regulating the global climate by absorbing carbon dioxide and producing oxygen. However, deforestation poses a significant threat to this vital ecosystem.',
                'summary': 'The Amazon rainforest covers 5.5 million square kilometers and contains 10% of Earth\'s species. It helps regulate global climate but faces threats from deforestation.'
            }
        ]
    
    def _generate_paraphrase_samples(self) -> List[Dict[str, Any]]:
        """Generate sample texts for paraphrasing."""
        return [
            {
                'id': '1',
                'original': 'The quick brown fox jumps over the lazy dog.',
                'paraphrase': 'A fast, brown-colored fox leaps above a sluggish canine.'
            }
        ]
    
    def _generate_sentiment_samples(self) -> List[Dict[str, Any]]:
        """Generate sample texts for sentiment analysis."""
        return [
            {
                'id': '1',
                'text': 'I absolutely love this product! It exceeded all my expectations.',
                'sentiment': 'positive'
            },
            {
                'id': '2',
                'text': 'The service was terrible and the food was cold.',
                'sentiment': 'negative'
            },
            {
                'id': '3',
                'text': 'The movie was okay, nothing special but not bad either.',
                'sentiment': 'neutral'
            }
        ]
    
    def _generate_qa_samples(self) -> List[Dict[str, Any]]:
        """Generate sample Q&A pairs."""
        return [
            {
                'id': '1',
                'context': 'The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889 and stands 330 meters tall.',
                'question': 'How tall is the Eiffel Tower?',
                'answer': '330 meters'
            }
        ]
    
    def _generate_completion_samples(self) -> List[Dict[str, Any]]:
        """Generate sample text completions."""
        return [
            {
                'id': '1',
                'partial': 'Once upon a time, in a land far away, there lived a',
                'completion': 'wise old wizard who possessed magical powers.'
            }
        ]
    
    def _generate_grammar_samples(self) -> List[Dict[str, Any]]:
        """Generate sample grammar corrections."""
        return [
            {
                'id': '1',
                'incorrect': 'Me and him went to the store yesterday.',
                'correct': 'He and I went to the store yesterday.'
            },
            {
                'id': '2',
                'incorrect': 'She don\'t know nothing about it.',
                'correct': 'She doesn\'t know anything about it.'
            }
        ]