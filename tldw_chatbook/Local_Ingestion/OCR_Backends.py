# OCR_Backends.py
"""
Pluggable OCR backend architecture for visual document processing.

This module provides a unified interface for different OCR engines,
allowing users to choose the best backend for their use case.
"""

import abc
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from loguru import logger

# Import optional dependencies
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

try:
    import paddle
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    import docext
    DOCEXT_AVAILABLE = True
except ImportError:
    DOCEXT_AVAILABLE = False

try:
    from gradio_client import Client, handle_file
    GRADIO_CLIENT_AVAILABLE = True
except ImportError:
    GRADIO_CLIENT_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OCRBackendType(Enum):
    """Supported OCR backend types."""
    DOCLING = "docling"
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"
    DOCEXT = "docext"  # NanoNets docext backend
    # Future backends can be added here
    # AZURE_CV = "azure_cv"
    # AWS_TEXTRACT = "aws_textract"
    # GOOGLE_VISION = "google_vision"


@dataclass
class OCRResult:
    """Standardized OCR result container."""
    text: str
    confidence: float  # Overall confidence score (0-1)
    language: str  # Detected or specified language
    backend: str  # Backend used
    
    # Optional detailed results
    words: Optional[List[Dict[str, Any]]] = None  # Word-level results
    lines: Optional[List[Dict[str, Any]]] = None  # Line-level results
    blocks: Optional[List[Dict[str, Any]]] = None  # Block/paragraph-level results
    layout_info: Optional[Dict[str, Any]] = None  # Document layout information
    
    # Metadata
    processing_time: Optional[float] = None  # Time taken in seconds
    image_size: Optional[Tuple[int, int]] = None  # (width, height)
    warnings: Optional[List[str]] = None


class OCRBackend(abc.ABC):
    """Abstract base class for OCR backends."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OCR backend.
        
        Args:
            config: Backend-specific configuration
        """
        self.config = config or {}
        self._initialized = False
    
    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available and properly configured."""
        pass
    
    @abc.abstractmethod
    def initialize(self) -> None:
        """Initialize the backend (load models, etc.)."""
        pass
    
    @abc.abstractmethod
    def process_image(self, 
                     image_path: Union[str, Path], 
                     language: str = "en",
                     **kwargs) -> OCRResult:
        """
        Process an image and extract text.
        
        Args:
            image_path: Path to the image file
            language: Language code for OCR
            **kwargs: Backend-specific options
            
        Returns:
            OCRResult with extracted text and metadata
        """
        pass
    
    @abc.abstractmethod
    def process_pdf(self,
                   pdf_path: Union[str, Path],
                   language: str = "en",
                   **kwargs) -> List[OCRResult]:
        """
        Process a PDF and extract text from all pages.
        
        Args:
            pdf_path: Path to the PDF file
            language: Language code for OCR
            **kwargs: Backend-specific options
            
        Returns:
            List of OCRResult, one per page
        """
        pass
    
    @abc.abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        pass
    
    def cleanup(self) -> None:
        """Cleanup resources (models, memory, etc.)."""
        pass


class DoclingOCRBackend(OCRBackend):
    """OCR backend using IBM's Docling library."""
    
    def is_available(self) -> bool:
        """Check if Docling is available."""
        return DOCLING_AVAILABLE
    
    def initialize(self) -> None:
        """Initialize Docling converter."""
        if not self._initialized:
            try:
                self.converter = DocumentConverter()
                self._initialized = True
                logger.info("Docling OCR backend initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Docling: {e}")
                raise
    
    def process_image(self, 
                     image_path: Union[str, Path], 
                     language: str = "en",
                     **kwargs) -> OCRResult:
        """Process image with Docling (converts to PDF internally if needed)."""
        # Docling primarily works with PDFs, so for images we might need to convert
        # For now, we'll raise NotImplementedError
        raise NotImplementedError("Docling backend currently only supports PDF processing")
    
    def process_pdf(self,
                   pdf_path: Union[str, Path],
                   language: str = "en",
                   **kwargs) -> List[OCRResult]:
        """Process PDF with Docling OCR."""
        if not self._initialized:
            self.initialize()
        
        try:
            # Configure pipeline options
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = kwargs.get('extract_tables', True)
            
            if hasattr(pipeline_options, 'ocr_lang'):
                pipeline_options.ocr_lang = language
            
            # Convert document
            result = self.converter.convert(str(pdf_path), pipeline_options=pipeline_options)
            
            # Extract text and create OCR results
            ocr_results = []
            
            # Process each page
            if hasattr(result.document, 'pages'):
                for page_idx, page in enumerate(result.document.pages):
                    page_text = page.export_to_markdown() if hasattr(page, 'export_to_markdown') else str(page)
                    
                    ocr_result = OCRResult(
                        text=page_text,
                        confidence=0.95,  # Docling doesn't provide confidence scores
                        language=language,
                        backend="docling",
                        layout_info={"page_number": page_idx + 1}
                    )
                    ocr_results.append(ocr_result)
            else:
                # Single result for entire document
                full_text = result.document.export_to_markdown()
                ocr_result = OCRResult(
                    text=full_text,
                    confidence=0.95,
                    language=language,
                    backend="docling"
                )
                ocr_results.append(ocr_result)
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"Docling OCR failed: {e}")
            raise
    
    def get_supported_languages(self) -> List[str]:
        """Get supported languages (Docling supports many languages)."""
        # This is a subset; Docling supports many more
        return ["en", "de", "fr", "es", "it", "pt", "ru", "zh", "ja", "ko", "ar"]


class TesseractOCRBackend(OCRBackend):
    """OCR backend using Tesseract."""
    
    def is_available(self) -> bool:
        """Check if Tesseract is available."""
        if not TESSERACT_AVAILABLE:
            return False
        
        # Check if tesseract binary is available
        try:
            pytesseract.get_tesseract_version()
            return True
        except:
            return False
    
    def initialize(self) -> None:
        """Initialize Tesseract (check for language data)."""
        if not self._initialized:
            try:
                # Get available languages
                self.available_langs = pytesseract.get_languages()
                self._initialized = True
                logger.info(f"Tesseract OCR backend initialized with languages: {self.available_langs}")
            except Exception as e:
                logger.error(f"Failed to initialize Tesseract: {e}")
                raise
    
    def process_image(self, 
                     image_path: Union[str, Path], 
                     language: str = "en",
                     **kwargs) -> OCRResult:
        """Process image with Tesseract."""
        if not self._initialized:
            self.initialize()
        
        try:
            # Map common language codes to Tesseract codes
            lang_map = {
                "en": "eng",
                "de": "deu",
                "fr": "fra",
                "es": "spa",
                "it": "ita",
                "pt": "por",
                "ru": "rus",
                "zh": "chi_sim",
                "ja": "jpn",
                "ko": "kor",
                "ar": "ara"
            }
            tesseract_lang = lang_map.get(language, language)
            
            # Get OCR data
            import time
            start_time = time.time()
            
            # Get text with confidence scores
            data = pytesseract.image_to_data(
                str(image_path),
                lang=tesseract_lang,
                output_type=pytesseract.Output.DICT,
                config=kwargs.get('config', '')
            )
            
            # Extract text and calculate confidence
            text_parts = []
            confidences = []
            words_data = []
            
            for i in range(len(data['text'])):
                txt = data['text'][i].strip()
                if txt:
                    text_parts.append(txt)
                    conf = float(data['conf'][i])
                    if conf > 0:  # Tesseract uses -1 for no confidence
                        confidences.append(conf / 100.0)  # Convert to 0-1 range
                    
                    words_data.append({
                        'text': txt,
                        'confidence': conf / 100.0 if conf > 0 else 0,
                        'bbox': {
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i]
                        }
                    })
            
            full_text = ' '.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Get image size
            if PIL_AVAILABLE:
                with Image.open(image_path) as img:
                    image_size = img.size
            else:
                image_size = None
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                language=language,
                backend="tesseract",
                words=words_data,
                processing_time=processing_time,
                image_size=image_size
            )
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            raise
    
    def process_pdf(self,
                   pdf_path: Union[str, Path],
                   language: str = "en",
                   **kwargs) -> List[OCRResult]:
        """Process PDF by converting pages to images first."""
        # This would require pdf2image or similar
        raise NotImplementedError("PDF processing not implemented for Tesseract backend. Convert to images first.")
    
    def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        if not self._initialized:
            self.initialize()
        
        # Map Tesseract language codes to standard codes
        lang_map = {
            "eng": "en",
            "deu": "de",
            "fra": "fr",
            "spa": "es",
            "ita": "it",
            "por": "pt",
            "rus": "ru",
            "chi_sim": "zh",
            "jpn": "ja",
            "kor": "ko",
            "ara": "ar"
        }
        
        return [lang_map.get(lang, lang) for lang in self.available_langs]


class EasyOCRBackend(OCRBackend):
    """OCR backend using EasyOCR."""
    
    def is_available(self) -> bool:
        """Check if EasyOCR is available."""
        return EASYOCR_AVAILABLE
    
    def initialize(self) -> None:
        """Initialize EasyOCR reader."""
        if not self._initialized:
            try:
                # Default to English if no language specified
                default_langs = self.config.get('languages', ['en'])
                self.reader = easyocr.Reader(
                    default_langs,
                    gpu=self.config.get('use_gpu', True)
                )
                self._initialized = True
                logger.info(f"EasyOCR backend initialized with languages: {default_langs}")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                raise
    
    def process_image(self, 
                     image_path: Union[str, Path], 
                     language: str = "en",
                     **kwargs) -> OCRResult:
        """Process image with EasyOCR."""
        if not self._initialized or language not in self.reader.lang_list:
            # Reinitialize with requested language
            self.reader = easyocr.Reader([language], gpu=self.config.get('use_gpu', True))
            self._initialized = True
        
        try:
            import time
            start_time = time.time()
            
            # Perform OCR
            results = self.reader.readtext(
                str(image_path),
                detail=1,  # Get bounding boxes and confidence
                paragraph=kwargs.get('paragraph', True)
            )
            
            # Extract text and metadata
            text_parts = []
            confidences = []
            lines_data = []
            
            for bbox, text, confidence in results:
                text_parts.append(text)
                confidences.append(confidence)
                
                lines_data.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': {
                        'points': bbox  # EasyOCR returns polygon points
                    }
                })
            
            full_text = ' '.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Get image size
            if PIL_AVAILABLE:
                with Image.open(image_path) as img:
                    image_size = img.size
            else:
                image_size = None
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                language=language,
                backend="easyocr",
                lines=lines_data,
                processing_time=processing_time,
                image_size=image_size
            )
            
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            raise
    
    def process_pdf(self,
                   pdf_path: Union[str, Path],
                   language: str = "en",
                   **kwargs) -> List[OCRResult]:
        """Process PDF by converting pages to images first."""
        raise NotImplementedError("PDF processing not implemented for EasyOCR backend. Convert to images first.")
    
    def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        # EasyOCR supports 80+ languages
        return ['en', 'zh', 'ja', 'ko', 'de', 'fr', 'es', 'pt', 'it', 'ru', 'ar', 'hi', 'th', 'vi']


class PaddleOCRBackend(OCRBackend):
    """OCR backend using PaddleOCR."""
    
    def is_available(self) -> bool:
        """Check if PaddleOCR is available."""
        return PADDLEOCR_AVAILABLE
    
    def initialize(self) -> None:
        """Initialize PaddleOCR."""
        if not self._initialized:
            try:
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.config.get('lang', 'en'),
                    use_gpu=self.config.get('use_gpu', True)
                )
                self._initialized = True
                logger.info("PaddleOCR backend initialized")
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR: {e}")
                raise
    
    def process_image(self, 
                     image_path: Union[str, Path], 
                     language: str = "en",
                     **kwargs) -> OCRResult:
        """Process image with PaddleOCR."""
        if not self._initialized or self.ocr.lang != language:
            # Reinitialize with requested language
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=language,
                use_gpu=self.config.get('use_gpu', True)
            )
            self._initialized = True
        
        try:
            import time
            start_time = time.time()
            
            # Perform OCR
            result = self.ocr.ocr(str(image_path), cls=True)
            
            # Extract text and metadata
            text_parts = []
            confidences = []
            lines_data = []
            
            if result and result[0]:
                for line in result[0]:
                    bbox, (text, confidence) = line
                    text_parts.append(text)
                    confidences.append(confidence)
                    
                    lines_data.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': {
                            'points': bbox
                        }
                    })
            
            full_text = ' '.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Get image size
            if PIL_AVAILABLE:
                with Image.open(image_path) as img:
                    image_size = img.size
            else:
                image_size = None
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                language=language,
                backend="paddleocr",
                lines=lines_data,
                processing_time=processing_time,
                image_size=image_size
            )
            
        except Exception as e:
            logger.error(f"PaddleOCR failed: {e}")
            raise
    
    def process_pdf(self,
                   pdf_path: Union[str, Path],
                   language: str = "en",
                   **kwargs) -> List[OCRResult]:
        """Process PDF by converting pages to images first."""
        raise NotImplementedError("PDF processing not implemented for PaddleOCR backend. Convert to images first.")
    
    def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        # PaddleOCR supports many languages
        return ['en', 'ch', 'ja', 'ko', 'de', 'fr', 'es', 'pt', 'it', 'ru', 'ar']


class DocextOCRBackend(OCRBackend):
    """OCR backend using NanoNets docext - Vision-Language Model based document understanding."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize docext backend with configuration."""
        super().__init__(config)
        self.mode = self.config.get('mode', 'api')  # 'api', 'model', or 'openai'
        self.api_url = self.config.get('api_url', 'http://localhost:7860')
        self.model_name = self.config.get('model_name', 'nanonets/Nanonets-OCR-s')
        # Get credentials from config or environment, no defaults for security
        self.username = self.config.get('username') or os.environ.get('DOCEXT_USERNAME')
        self.password = self.config.get('password') or os.environ.get('DOCEXT_PASSWORD')
        self.max_new_tokens = self.config.get('max_new_tokens', 4096)
        self.client = None
        self.model = None
        self.processor = None
        self.tokenizer = None
    
    def is_available(self) -> bool:
        """Check if docext backend is available based on mode."""
        if self.mode == 'api':
            return DOCEXT_AVAILABLE and GRADIO_CLIENT_AVAILABLE
        elif self.mode == 'model':
            return DOCEXT_AVAILABLE and TRANSFORMERS_AVAILABLE
        elif self.mode == 'openai':
            return DOCEXT_AVAILABLE and OPENAI_AVAILABLE
        return False
    
    def initialize(self) -> None:
        """Initialize the backend based on selected mode."""
        if not self._initialized:
            try:
                if self.mode == 'api':
                    # Initialize Gradio client
                    # Note: Authentication is optional - only use if credentials are provided
                    auth = None
                    if self.username and self.password:
                        auth = (self.username, self.password)
                        logger.info("Using authentication for Docext API")
                    elif self.username or self.password:
                        logger.warning("Both username and password must be provided for authentication")
                    
                    self.client = Client(self.api_url, auth=auth)
                    logger.info(f"Docext API client initialized at {self.api_url}")
                    
                elif self.mode == 'model':
                    # Initialize transformers model directly
                    if not TRANSFORMERS_AVAILABLE:
                        raise ImportError("transformers library not available for model mode")
                    
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_name,
                        torch_dtype="auto",
                        device_map="auto",
                        attn_implementation="flash_attention_2"
                    )
                    self.model.eval()
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self.processor = AutoProcessor.from_pretrained(self.model_name)
                    logger.info(f"Docext model loaded: {self.model_name}")
                    
                elif self.mode == 'openai':
                    # Initialize OpenAI-compatible client
                    if not OPENAI_AVAILABLE:
                        raise ImportError("openai library not available for OpenAI mode")
                    
                    base_url = self.config.get('openai_base_url', 'http://localhost:8000/v1')
                    api_key = self.config.get('openai_api_key', '123')
                    self.client = OpenAI(api_key=api_key, base_url=base_url)
                    logger.info(f"Docext OpenAI client initialized at {base_url}")
                
                self._initialized = True
                
            except Exception as e:
                logger.error(f"Failed to initialize docext backend: {e}")
                raise
    
    def _get_docext_prompt(self) -> str:
        """Get the standard docext prompt for document extraction."""
        return """Extract the text from the above document as if you were reading it naturally.
Return the tables in html format.
Return the equations in LaTeX representation.
If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>.
Watermarks should be wrapped in brackets.
Ex: <page_number>14</page_number> or <page_number>9/22</page_number>.
Prefer using ☐ and ☑ for check boxes."""
    
    def process_image(self, 
                     image_path: Union[str, Path], 
                     language: str = "en",
                     **kwargs) -> OCRResult:
        """Process image with docext."""
        if not self._initialized:
            self.initialize()
        
        try:
            import time
            start_time = time.time()
            
            if self.mode == 'api':
                # Use Gradio API
                result = self.client.predict(
                    images=[{"image": handle_file(str(image_path))}],
                    api_name="/process_markdown_streaming"
                )
                
                # Extract text from result
                if isinstance(result, list) and result:
                    text = result[0]
                else:
                    text = str(result)
                
            elif self.mode == 'model':
                # Use transformers directly
                prompt = kwargs.get('prompt', self._get_docext_prompt())
                
                image = Image.open(image_path)
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": prompt},
                    ]},
                ]
                
                text_input = self.processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[text_input], 
                    images=[image], 
                    padding=True, 
                    return_tensors="pt"
                )
                inputs = inputs.to(self.model.device)
                
                output_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=self.max_new_tokens, 
                    do_sample=False
                )
                generated_ids = [
                    output_ids[len(input_ids):] 
                    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ]
                text = self.processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True
                )[0]
                
            elif self.mode == 'openai':
                # Use OpenAI-compatible API
                import base64
                
                with open(image_path, "rb") as image_file:
                    img_base64 = base64.b64encode(image_file.read()).decode("utf-8")
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                                },
                                {
                                    "type": "text",
                                    "text": kwargs.get('prompt', self._get_docext_prompt())
                                }
                            ]
                        }
                    ],
                    max_tokens=self.max_new_tokens
                )
                text = response.choices[0].message.content
            
            processing_time = time.time() - start_time
            
            # Parse structured content from markdown
            lines_data = []
            blocks_data = []
            layout_info = {}
            
            # Extract special tags
            import re
            
            # Extract tables
            tables = re.findall(r'<table>.*?</table>', text, re.DOTALL)
            if tables:
                layout_info['tables'] = tables
            
            # Extract equations
            equations = re.findall(r'\$\$.*?\$\$|\$.*?\$', text, re.DOTALL)
            if equations:
                layout_info['equations'] = equations
            
            # Extract page numbers
            page_nums = re.findall(r'<page_number>(.*?)</page_number>', text)
            if page_nums:
                layout_info['page_numbers'] = page_nums
            
            # Extract images
            images = re.findall(r'<img>(.*?)</img>', text, re.DOTALL)
            if images:
                layout_info['image_descriptions'] = images
            
            # Get image size if available
            image_size = None
            if PIL_AVAILABLE:
                try:
                    with Image.open(image_path) as img:
                        image_size = img.size
                except:
                    pass
            
            return OCRResult(
                text=text,
                confidence=0.95,  # Docext doesn't provide confidence scores
                language=language,
                backend="docext",
                lines=lines_data if lines_data else None,
                blocks=blocks_data if blocks_data else None,
                layout_info=layout_info if layout_info else None,
                processing_time=processing_time,
                image_size=image_size
            )
            
        except Exception as e:
            logger.error(f"Docext processing failed: {e}")
            raise
    
    def process_pdf(self,
                   pdf_path: Union[str, Path],
                   language: str = "en",
                   **kwargs) -> List[OCRResult]:
        """Process PDF with docext by converting pages to images."""
        # For PDFs, we need to extract pages as images first
        try:
            import pymupdf
            
            results = []
            doc = pymupdf.open(str(pdf_path))
            
            for page_num, page in enumerate(doc):
                # Convert page to image
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # 2x scaling for better quality
                img_data = pix.tobytes("png")
                
                # Save to temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp.write(img_data)
                    tmp_path = tmp.name
                
                try:
                    # Process with docext
                    result = self.process_image(tmp_path, language=language, **kwargs)
                    result.layout_info = result.layout_info or {}
                    result.layout_info['page_number'] = page_num + 1
                    results.append(result)
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
            
            doc.close()
            return results
            
        except ImportError:
            raise NotImplementedError("PDF processing requires pymupdf. Install with: pip install pymupdf")
        except Exception as e:
            logger.error(f"Docext PDF processing failed: {e}")
            raise
    
    def get_supported_languages(self) -> List[str]:
        """Get supported languages (docext is multilingual)."""
        # Docext uses vision-language models so it supports many languages implicitly
        return ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko', 'ar', 'hi', 'multi']
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.mode == 'model' and self.model is not None:
            # Clear model from memory
            del self.model
            self.model = None
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._initialized = False


class OCRManager:
    """Manager class for handling multiple OCR backends."""
    
    def __init__(self, default_backend: Optional[str] = None):
        """
        Initialize OCR manager.
        
        Args:
            default_backend: Default backend to use
        """
        self.backends: Dict[str, OCRBackend] = {}
        self.default_backend = default_backend
        self._register_backends()
    
    def _register_backends(self):
        """Register available backends."""
        backend_classes = {
            OCRBackendType.DOCLING.value: DoclingOCRBackend,
            OCRBackendType.TESSERACT.value: TesseractOCRBackend,
            OCRBackendType.EASYOCR.value: EasyOCRBackend,
            OCRBackendType.PADDLEOCR.value: PaddleOCRBackend,
            OCRBackendType.DOCEXT.value: DocextOCRBackend,
        }
        
        for name, backend_class in backend_classes.items():
            try:
                backend = backend_class()
                if backend.is_available():
                    self.backends[name] = backend
                    logger.info(f"Registered OCR backend: {name}")
            except Exception as e:
                logger.debug(f"Failed to register {name} backend: {e}")
        
        # Set default backend if not specified
        if not self.default_backend and self.backends:
            # Priority order: docext, docling, tesseract, easyocr, paddleocr
            priority = [OCRBackendType.DOCEXT.value, OCRBackendType.DOCLING.value, 
                       OCRBackendType.TESSERACT.value, OCRBackendType.EASYOCR.value, 
                       OCRBackendType.PADDLEOCR.value]
            for backend in priority:
                if backend in self.backends:
                    self.default_backend = backend
                    break
    
    def get_available_backends(self) -> List[str]:
        """Get list of available backend names."""
        return list(self.backends.keys())
    
    def get_backend(self, name: Optional[str] = None) -> OCRBackend:
        """
        Get an OCR backend by name.
        
        Args:
            name: Backend name, or None to use default
            
        Returns:
            OCR backend instance
            
        Raises:
            ValueError: If backend not found
        """
        backend_name = name or self.default_backend
        if not backend_name:
            raise ValueError("No OCR backend available")
        
        if backend_name not in self.backends:
            raise ValueError(f"OCR backend '{backend_name}' not available. "
                           f"Available backends: {self.get_available_backends()}")
        
        return self.backends[backend_name]
    
    def process_image(self,
                     image_path: Union[str, Path],
                     language: str = "en",
                     backend: Optional[str] = None,
                     **kwargs) -> OCRResult:
        """
        Process an image using specified or default backend.
        
        Args:
            image_path: Path to image
            language: Language code
            backend: Backend name (optional)
            **kwargs: Backend-specific options
            
        Returns:
            OCR result
        """
        ocr_backend = self.get_backend(backend)
        return ocr_backend.process_image(image_path, language, **kwargs)
    
    def process_pdf(self,
                   pdf_path: Union[str, Path],
                   language: str = "en",
                   backend: Optional[str] = None,
                   **kwargs) -> List[OCRResult]:
        """
        Process a PDF using specified or default backend.
        
        Args:
            pdf_path: Path to PDF
            language: Language code
            backend: Backend name (optional)
            **kwargs: Backend-specific options
            
        Returns:
            List of OCR results (one per page)
        """
        ocr_backend = self.get_backend(backend)
        return ocr_backend.process_pdf(pdf_path, language, **kwargs)


# Create a global OCR manager instance
ocr_manager = OCRManager()


def get_ocr_manager() -> OCRManager:
    """Get the global OCR manager instance."""
    return ocr_manager