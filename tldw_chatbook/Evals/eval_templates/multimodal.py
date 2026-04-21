# eval_templates/multimodal.py
# Description: Multimodal evaluation templates
#
"""
Multimodal Templates
--------------------

Templates for multimodal tasks including vision, audio, and cross-modal understanding.
"""

from typing import Dict, Any, List
from .base import BaseTemplates


class MultimodalTemplates(BaseTemplates):
    """Templates for multimodal evaluation tasks."""
    
    def _initialize_templates(self):
        """Initialize multimodal templates."""
        self._templates = {
            'image_captioning': self._image_captioning_template(),
            'visual_qa': self._visual_qa_template(),
            'image_classification': self._image_classification_template(),
            'ocr_evaluation': self._ocr_evaluation_template(),
            'chart_understanding': self._chart_understanding_template(),
            'diagram_reasoning': self._diagram_reasoning_template(),
            'cross_modal_retrieval': self._cross_modal_retrieval_template()
        }
    
    def _image_captioning_template(self) -> Dict[str, Any]:
        """Image captioning evaluation template."""
        return self._create_base_template(
            name='Image Captioning',
            description='Generate descriptive captions for images',
            task_type='generation',
            metric='bleu',
            category='multimodal',
            subcategory='vision_to_text',
            dataset_name='custom_caption',
            generation_kwargs={
                'temperature': 0.5,
                'max_tokens': 100
            },
            doc_to_text="Describe this image in one sentence:\n[IMAGE]\n\nCaption:",
            requires_vision=True,
            sample_images=self._generate_caption_samples()
        )
    
    def _visual_qa_template(self) -> Dict[str, Any]:
        """Visual question answering template."""
        return self._create_base_template(
            name='Visual Question Answering',
            description='Answer questions about images',
            task_type='question_answer',
            metric='exact_match',
            category='multimodal',
            subcategory='visual_qa',
            dataset_name='custom_vqa',
            generation_kwargs={
                'temperature': 0.0,
                'max_tokens': 50
            },
            doc_to_text="[IMAGE]\n\nQuestion: {question}\nAnswer:",
            requires_vision=True,
            sample_qa_pairs=self._generate_vqa_samples()
        )
    
    def _image_classification_template(self) -> Dict[str, Any]:
        """Image classification template."""
        return self._create_base_template(
            name='Image Classification',
            description='Classify images into categories',
            task_type='classification',
            metric='accuracy',
            category='multimodal',
            subcategory='image_classification',
            dataset_name='custom_image_class',
            generation_kwargs={
                'temperature': 0.0,
                'max_tokens': 20
            },
            doc_to_text="Classify this image into one of the following categories: {categories}\n[IMAGE]\n\nCategory:",
            requires_vision=True,
            categories=['animal', 'vehicle', 'food', 'landscape', 'person', 'object'],
            sample_images=self._generate_classification_samples()
        )
    
    def _ocr_evaluation_template(self) -> Dict[str, Any]:
        """OCR (Optical Character Recognition) evaluation template."""
        return self._create_base_template(
            name='OCR Evaluation',
            description='Extract and transcribe text from images',
            task_type='generation',
            metric='exact_match',
            category='multimodal',
            subcategory='ocr',
            dataset_name='custom_ocr',
            generation_kwargs={
                'temperature': 0.0,
                'max_tokens': 200
            },
            doc_to_text="Extract all text from this image:\n[IMAGE]\n\nText:",
            requires_vision=True,
            sample_documents=self._generate_ocr_samples()
        )
    
    def _chart_understanding_template(self) -> Dict[str, Any]:
        """Chart and graph understanding template."""
        return self._create_base_template(
            name='Chart Understanding',
            description='Interpret and answer questions about charts and graphs',
            task_type='question_answer',
            metric='f1',
            category='multimodal',
            subcategory='chart_understanding',
            dataset_name='custom_charts',
            generation_kwargs={
                'temperature': 0.0,
                'max_tokens': 150
            },
            doc_to_text="[CHART IMAGE]\n\n{question}\n\nAnswer:",
            requires_vision=True,
            chart_types=['bar', 'line', 'pie', 'scatter'],
            sample_charts=self._generate_chart_samples()
        )
    
    def _diagram_reasoning_template(self) -> Dict[str, Any]:
        """Diagram reasoning and understanding template."""
        return self._create_base_template(
            name='Diagram Reasoning',
            description='Reason about relationships and processes in diagrams',
            task_type='generation',
            metric='rouge_l',
            category='multimodal',
            subcategory='diagram_reasoning',
            dataset_name='custom_diagrams',
            generation_kwargs={
                'temperature': 0.0,
                'max_tokens': 200
            },
            doc_to_text="Explain the process shown in this diagram:\n[DIAGRAM IMAGE]\n\nExplanation:",
            requires_vision=True,
            diagram_types=['flowchart', 'network', 'venn', 'sequence'],
            sample_diagrams=self._generate_diagram_samples()
        )
    
    def _cross_modal_retrieval_template(self) -> Dict[str, Any]:
        """Cross-modal retrieval template."""
        return self._create_base_template(
            name='Cross-Modal Retrieval',
            description='Match images with text descriptions or vice versa',
            task_type='classification',
            metric='accuracy',
            category='multimodal',
            subcategory='retrieval',
            dataset_name='custom_retrieval',
            generation_kwargs={
                'temperature': 0.0,
                'max_tokens': 10
            },
            doc_to_text="Does this text describe the image? Text: {text}\n[IMAGE]\n\nAnswer (yes/no):",
            requires_vision=True,
            sample_pairs=self._generate_retrieval_samples()
        )
    
    def _generate_caption_samples(self) -> List[Dict[str, Any]]:
        """Generate sample image captioning tasks."""
        return [
            {
                'id': '1',
                'image_description': 'A golden retriever playing with a ball in a park',
                'expected_caption': 'A golden retriever dog plays with a red ball on green grass',
                'key_elements': ['dog', 'ball', 'outdoor']
            },
            {
                'id': '2',
                'image_description': 'A busy city street at night with neon signs',
                'expected_caption': 'A crowded urban street illuminated by colorful neon signs at night',
                'key_elements': ['city', 'night', 'lights']
            }
        ]
    
    def _generate_vqa_samples(self) -> List[Dict[str, Any]]:
        """Generate sample visual QA tasks."""
        return [
            {
                'id': '1',
                'image_description': 'Three apples on a table',
                'question': 'How many apples are there?',
                'answer': 'Three',
                'answer_type': 'counting'
            },
            {
                'id': '2',
                'image_description': 'A red car parked next to a blue car',
                'question': 'What color is the car on the left?',
                'answer': 'Red',
                'answer_type': 'attribute'
            }
        ]
    
    def _generate_classification_samples(self) -> List[Dict[str, Any]]:
        """Generate sample image classification tasks."""
        return [
            {
                'id': '1',
                'image_description': 'A close-up photo of a tiger',
                'category': 'animal',
                'subcategory': 'wild_animal'
            },
            {
                'id': '2',
                'image_description': 'A plate of spaghetti with tomato sauce',
                'category': 'food',
                'subcategory': 'pasta'
            }
        ]
    
    def _generate_ocr_samples(self) -> List[Dict[str, Any]]:
        """Generate sample OCR tasks."""
        return [
            {
                'id': '1',
                'document_type': 'receipt',
                'expected_text': 'Store Name\nDate: 2024-01-15\nTotal: $25.99',
                'difficulty': 'easy'
            },
            {
                'id': '2',
                'document_type': 'handwritten_note',
                'expected_text': 'Meeting at 3pm tomorrow',
                'difficulty': 'medium'
            }
        ]
    
    def _generate_chart_samples(self) -> List[Dict[str, Any]]:
        """Generate sample chart understanding tasks."""
        return [
            {
                'id': '1',
                'chart_type': 'bar',
                'description': 'Monthly sales data for Q1',
                'question': 'Which month had the highest sales?',
                'answer': 'March',
                'data_points': ['January: 100', 'February: 150', 'March: 200']
            },
            {
                'id': '2',
                'chart_type': 'pie',
                'description': 'Market share distribution',
                'question': 'What percentage does Company A hold?',
                'answer': '35%',
                'segments': ['Company A: 35%', 'Company B: 25%', 'Others: 40%']
            }
        ]
    
    def _generate_diagram_samples(self) -> List[Dict[str, Any]]:
        """Generate sample diagram reasoning tasks."""
        return [
            {
                'id': '1',
                'diagram_type': 'flowchart',
                'description': 'User login process',
                'key_steps': ['Enter credentials', 'Validate', 'Grant access or show error'],
                'expected_explanation': 'The flowchart shows a user authentication process...'
            },
            {
                'id': '2',
                'diagram_type': 'venn',
                'description': 'Overlap between three skill sets',
                'sets': ['Programming', 'Design', 'Marketing'],
                'intersection': 'Product Management'
            }
        ]
    
    def _generate_retrieval_samples(self) -> List[Dict[str, Any]]:
        """Generate sample cross-modal retrieval tasks."""
        return [
            {
                'id': '1',
                'text': 'A sunset over the ocean with orange and pink clouds',
                'image_description': 'Beach sunset with colorful sky',
                'match': True
            },
            {
                'id': '2',
                'text': 'A snowy mountain peak',
                'image_description': 'Tropical beach with palm trees',
                'match': False
            }
        ]