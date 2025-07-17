# Image_Processing_Lib.py
#########################################
# Library to hold functions for ingesting image files and visual documents.
# Supports: PNG, JPG, JPEG, GIF, BMP, TIFF, WebP, and other image formats
#
####################
# Function List
#
# 1. process_image(file_path, title, author, keywords, enable_ocr, ocr_backend, ocr_language, chunk_options, perform_analysis)
# 2. extract_text_from_image(image_path, ocr_backend, language)
# 3. extract_visual_features(image_path)
# 4. process_image_batch(image_paths, **kwargs)
#
####################
# Import necessary libraries
import os
import gc
import json
import tempfile
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import numpy as np

# Import External Libs
try:
    from PIL import Image, ImageFilter, ImageEnhance
    import pillow_heif  # For HEIF/HEIC support
    pillow_heif.register_heif_opener()
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Import Local
from ..config import get_cli_setting, get_media_ingestion_defaults
from ..LLM_Calls.Summarization_General_Lib import analyze
from ..Metrics.metrics_logger import log_counter, log_histogram
from ..Utils.optional_deps import get_safe_import
from .OCR_Backends import ocr_manager, OCRResult
from loguru import logger

# Constants
SUPPORTED_IMAGE_FORMATS = {
    '.png': 'PNG Image',
    '.jpg': 'JPEG Image',
    '.jpeg': 'JPEG Image',
    '.gif': 'GIF Image',
    '.bmp': 'Bitmap Image',
    '.tiff': 'TIFF Image',
    '.tif': 'TIFF Image',
    '.webp': 'WebP Image',
    '.svg': 'SVG Image',
    '.ico': 'Icon File',
    '.heic': 'HEIC Image',
    '.heif': 'HEIF Image',
}

# Get media processing config from CLI settings
def get_image_config():
    """Get image processing configuration."""
    return get_media_ingestion_defaults('image') if get_media_ingestion_defaults else {
        'chunk_method': 'visual_blocks',
        'chunk_size': 1000,
        'chunk_overlap': 100,
        'enable_ocr': True,
        'ocr_backend': 'auto',
        'ocr_language': 'en',
        'extract_visual_features': True,
        'visual_feature_model': 'basic',
        'image_preprocessing': True,
        'max_image_size': 4096,  # Max dimension in pixels
    }

#######################################################################################################################
# Function Definitions
#

def preprocess_image_for_ocr(image_path: Union[str, Path]) -> Optional[Path]:
    """
    Preprocess image to improve OCR accuracy.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        Path to preprocessed image, or None if preprocessing fails
    """
    if not PIL_AVAILABLE:
        logger.warning("PIL not available, skipping image preprocessing")
        return Path(image_path)
    
    try:
        # Open image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if too large
        max_size = get_image_config().get('max_image_size', 4096)
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Apply preprocessing steps
        # 1. Increase contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        
        # 2. Convert to grayscale
        img = img.convert('L')
        
        # 3. Apply sharpening
        img = img.filter(ImageFilter.SHARPEN)
        
        # 4. Binarization (optional, can help with some OCR engines)
        # threshold = 128
        # img = img.point(lambda p: p > threshold and 255)
        
        # Save preprocessed image
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img.save(temp_file.name, 'PNG')
        temp_file.close()
        
        return Path(temp_file.name)
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return Path(image_path)


def extract_visual_features(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract visual features from an image for similarity search.
    
    Args:
        image_path: Path to the image
        
    Returns:
        Dictionary containing visual features
    """
    features = {
        'has_visual_features': False,
        'error': None
    }
    
    try:
        if PIL_AVAILABLE:
            with Image.open(image_path) as img:
                # Basic image properties
                features['width'] = img.width
                features['height'] = img.height
                features['aspect_ratio'] = img.width / img.height if img.height > 0 else 0
                features['mode'] = img.mode
                features['format'] = img.format
                
                # Color statistics
                if img.mode in ['RGB', 'RGBA']:
                    # Convert to RGB if RGBA
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    
                    # Get color histogram
                    histogram = img.histogram()
                    
                    # Calculate dominant colors (simplified)
                    pixels = list(img.getdata())
                    if pixels:
                        # Simple average color
                        avg_color = tuple(sum(col) // len(pixels) for col in zip(*pixels))
                        features['dominant_color'] = {
                            'r': avg_color[0],
                            'g': avg_color[1],
                            'b': avg_color[2]
                        }
                
                # Image fingerprint (perceptual hash)
                # This can be used for duplicate detection
                img_small = img.resize((8, 8), Image.Resampling.LANCZOS).convert('L')
                pixels = list(img_small.getdata())
                avg = sum(pixels) / len(pixels)
                hash_bits = ''.join('1' if pixel > avg else '0' for pixel in pixels)
                features['perceptual_hash'] = hex(int(hash_bits, 2))[2:]
                
                features['has_visual_features'] = True
                
        # Advanced features with OpenCV (if available)
        if CV2_AVAILABLE:
            img_cv = cv2.imread(str(image_path))
            if img_cv is not None:
                # Detect edges
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                features['edge_density'] = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])
                
                # Detect keypoints (corners, blobs, etc.)
                orb = cv2.ORB_create(nfeatures=100)
                keypoints = orb.detect(gray, None)
                features['num_keypoints'] = len(keypoints)
        
    except Exception as e:
        logger.error(f"Error extracting visual features: {e}")
        features['error'] = str(e)
    
    return features


def extract_text_from_image(
    image_path: Union[str, Path],
    ocr_backend: str = "auto",
    language: str = "en",
    preprocess: bool = True
) -> Optional[OCRResult]:
    """
    Extract text from an image using OCR.
    
    Args:
        image_path: Path to the image
        ocr_backend: OCR backend to use ('auto', 'tesseract', 'easyocr', etc.)
        language: Language code for OCR
        preprocess: Whether to preprocess image before OCR
        
    Returns:
        OCRResult or None if extraction fails
    """
    try:
        # Preprocess image if requested
        if preprocess:
            processed_path = preprocess_image_for_ocr(image_path)
        else:
            processed_path = Path(image_path)
        
        # Use OCR manager to extract text
        if ocr_backend == "auto":
            result = ocr_manager.process_image(processed_path, language=language)
        else:
            result = ocr_manager.process_image(processed_path, language=language, backend=ocr_backend)
        
        # Clean up preprocessed image if created
        if preprocess and processed_path != Path(image_path):
            try:
                os.unlink(processed_path)
            except:
                pass
        
        return result
        
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        return None


def process_image(
    file_path: Union[str, Path],
    title_override: Optional[str] = None,
    author_override: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    enable_ocr: bool = True,
    ocr_backend: str = "auto",
    ocr_language: str = "en",
    extract_features: bool = True,
    chunk_options: Optional[Dict[str, Any]] = None,
    perform_analysis: bool = False,
    api_name: Optional[str] = None,
    api_key: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process an image file, extract text via OCR and visual features.
    
    Args:
        file_path: Path to the image file
        title_override: Optional title override
        author_override: Optional author override
        keywords: List of keywords
        enable_ocr: Whether to perform OCR
        ocr_backend: OCR backend to use
        ocr_language: Language for OCR
        extract_features: Whether to extract visual features
        chunk_options: Options for text chunking (if OCR is performed)
        perform_analysis: Whether to analyze/summarize extracted text
        api_name: LLM API for analysis
        api_key: API key for LLM
        custom_prompt: Custom prompt for analysis
        system_prompt: System prompt for analysis
        
    Returns:
        Dictionary containing processing results
    """
    logger.info(f"Processing image: {file_path}")
    log_counter("image_processing_attempt", labels={"file_path": str(file_path)})
    
    start_time = datetime.now()
    
    # Initialize result dictionary
    result = {
        "status": "Pending",
        "input_ref": str(file_path),
        "media_type": "image",
        "content": None,
        "metadata": {},
        "visual_features": None,
        "ocr_result": None,
        "chunks": None,
        "analysis": None,
        "keywords": keywords or [],
        "error": None,
        "warnings": []
    }
    
    file_path = Path(file_path)
    file_ext = file_path.suffix.lower()
    
    # Check if format is supported
    if file_ext not in SUPPORTED_IMAGE_FORMATS:
        result["status"] = "Error"
        result["error"] = f"Unsupported image format: {file_ext}"
        return result
    
    try:
        # Extract basic metadata
        metadata = {
            "filename": file_path.name,
            "file_size": file_path.stat().st_size,
            "file_format": SUPPORTED_IMAGE_FORMATS.get(file_ext, "Unknown"),
            "processing_method": "image_processing"
        }
        
        # Get image properties with PIL
        if PIL_AVAILABLE:
            try:
                with Image.open(file_path) as img:
                    metadata["width"] = img.width
                    metadata["height"] = img.height
                    metadata["mode"] = img.mode
                    metadata["format"] = img.format or file_ext[1:].upper()
                    
                    # Extract EXIF data if available
                    if hasattr(img, '_getexif') and img._getexif():
                        exif = img._getexif()
                        # Store selected EXIF tags
                        if exif:
                            metadata["exif"] = {
                                "make": exif.get(271),  # Camera make
                                "model": exif.get(272),  # Camera model
                                "datetime": exif.get(306),  # DateTime
                                "orientation": exif.get(274),  # Orientation
                            }
            except Exception as e:
                logger.warning(f"Error reading image metadata: {e}")
                result["warnings"].append(f"Could not read image metadata: {str(e)}")
        
        # Extract visual features if requested
        visual_features = None
        if extract_features:
            logger.info("Extracting visual features")
            visual_features = extract_visual_features(file_path)
            result["visual_features"] = visual_features
            
            if visual_features.get('error'):
                result["warnings"].append(f"Visual feature extraction warning: {visual_features['error']}")
        
        # Perform OCR if requested
        ocr_text = ""
        ocr_result = None
        if enable_ocr:
            logger.info(f"Performing OCR with backend: {ocr_backend}")
            ocr_result = extract_text_from_image(
                file_path,
                ocr_backend=ocr_backend,
                language=ocr_language,
                preprocess=True
            )
            
            if ocr_result:
                ocr_text = ocr_result.text
                result["ocr_result"] = {
                    "text": ocr_result.text,
                    "confidence": ocr_result.confidence,
                    "language": ocr_result.language,
                    "backend": ocr_result.backend,
                    "processing_time": ocr_result.processing_time,
                    "num_words": len(ocr_result.text.split()) if ocr_result.text else 0
                }
                
                # Add OCR metadata
                metadata["ocr_performed"] = True
                metadata["ocr_backend"] = ocr_result.backend
                metadata["ocr_confidence"] = ocr_result.confidence
                
                log_histogram("image_ocr_confidence", ocr_result.confidence)
                log_counter("image_ocr_success", labels={"backend": ocr_result.backend})
            else:
                result["warnings"].append("OCR failed or returned no text")
                metadata["ocr_performed"] = False
                log_counter("image_ocr_failure")
        
        # Set content (OCR text or empty)
        result["content"] = ocr_text
        
        # Chunk text if OCR produced content
        if ocr_text and chunk_options:
            try:
                from ..RAG_Search.chunking_service import improved_chunking_process
                chunks = improved_chunking_process(ocr_text, chunk_options)
                result["chunks"] = chunks
                log_histogram("image_chunks_created", len(chunks))
            except Exception as e:
                logger.error(f"Chunking failed: {e}")
                result["warnings"].append(f"Chunking failed: {str(e)}")
                # Create single chunk as fallback
                result["chunks"] = [{
                    'text': ocr_text,
                    'metadata': {'chunk_num': 0}
                }]
        elif ocr_text:
            # Single chunk if no chunking options
            result["chunks"] = [{
                'text': ocr_text,
                'metadata': {'chunk_num': 0}
            }]
        
        # Perform analysis if requested and we have text
        if perform_analysis and api_name and api_key and ocr_text:
            logger.info("Performing content analysis")
            try:
                analysis_prompt = custom_prompt or f"Analyze this text extracted from an image: {ocr_text[:500]}..."
                
                analysis = analyze(
                    input_data=ocr_text,
                    custom_prompt_arg=analysis_prompt,
                    api_name=api_name,
                    api_key=api_key,
                    system_message=system_prompt
                )
                result["analysis"] = analysis
                log_counter("image_analysis_success")
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                result["warnings"].append(f"Analysis failed: {str(e)}")
                log_counter("image_analysis_failure")
        
        # Set final metadata
        result["metadata"] = metadata
        result["title"] = title_override or file_path.stem
        result["author"] = author_override or "Unknown"
        
        # Determine final status
        if result["error"]:
            result["status"] = "Error"
        elif result["warnings"]:
            result["status"] = "Warning"
        else:
            result["status"] = "Success"
        
        # Log completion
        processing_time = (datetime.now() - start_time).total_seconds()
        log_histogram("image_processing_duration", processing_time)
        log_counter("image_processing_success" if result["status"] != "Error" else "image_processing_error")
        
        logger.info(f"Image processing completed with status: {result['status']}")
        
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        result["status"] = "Error"
        result["error"] = str(e)
        log_counter("image_processing_error", labels={"error": type(e).__name__})
    
    return result


def process_image_batch(
    image_paths: List[Union[str, Path]],
    parallel: bool = True,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Process multiple images in batch.
    
    Args:
        image_paths: List of image file paths
        parallel: Whether to process in parallel (not implemented yet)
        **kwargs: Arguments passed to process_image
        
    Returns:
        List of processing results
    """
    results = []
    total = len(image_paths)
    
    for idx, image_path in enumerate(image_paths):
        logger.info(f"Processing image {idx + 1}/{total}: {image_path}")
        
        try:
            result = process_image(image_path, **kwargs)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            results.append({
                "status": "Error",
                "input_ref": str(image_path),
                "error": str(e)
            })
    
    return results


def extract_images_from_pdf(pdf_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Extract embedded images from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of dictionaries containing image data and metadata
    """
    extracted_images = []
    
    try:
        import pymupdf
        
        doc = pymupdf.open(pdf_path)
        
        for page_num, page in enumerate(doc):
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Extract image
                    xref = img[0]
                    pix = pymupdf.Pixmap(doc, xref)
                    
                    # Convert to PIL Image
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                    else:  # CMYK
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                        img_data = pix.tobytes("png")
                    
                    # Save to temp file
                    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    temp_file.write(img_data)
                    temp_file.close()
                    
                    extracted_images.append({
                        'path': temp_file.name,
                        'page': page_num + 1,
                        'index': img_index,
                        'width': pix.width,
                        'height': pix.height,
                        'size': len(img_data)
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
        
        doc.close()
        
    except Exception as e:
        logger.error(f"Error extracting images from PDF: {e}")
    
    return extracted_images


# Convenience function for simple OCR
def simple_ocr(image_path: Union[str, Path], language: str = "en") -> str:
    """
    Simple OCR function that just returns extracted text.
    
    Args:
        image_path: Path to image
        language: Language code
        
    Returns:
        Extracted text or empty string
    """
    result = extract_text_from_image(image_path, language=language)
    return result.text if result else ""


#
# End of Image_Processing_Lib.py
#######################################################################################################################