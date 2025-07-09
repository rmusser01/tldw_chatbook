# tldw_chatbook/Local_Ingestion/local_file_ingestion.py
"""
Programmatic interface for ingesting local files into the Media database.

This module provides functions to process and store various file types (PDFs, documents, 
e-books, etc.) without going through the UI, leveraging existing processing capabilities.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from loguru import logger

# Import processing libraries
from .PDF_Processing_Lib import process_pdf
from .Document_Processing_Lib import process_document
from .Book_Ingestion_Lib import process_ebook
from .audio_processing import LocalAudioProcessor
from .video_processing import LocalVideoProcessor

# Import database
from ..DB.Client_Media_DB_v2 import MediaDatabase


class FileIngestionError(Exception):
    """Raised when file ingestion fails."""
    pass


def detect_file_type(file_path: Union[str, Path]) -> str:
    """
    Detect the type of file based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File type as string: 'pdf', 'document', 'ebook', 'xml', 'plaintext', 'html'
        
    Raises:
        FileIngestionError: If file type is not supported
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    # PDF files
    if extension == '.pdf':
        return 'pdf'
    
    # Document files
    elif extension in ['.doc', '.docx', '.odt', '.rtf']:
        return 'document'
    
    # E-book files
    elif extension in ['.epub', '.mobi', '.azw', '.azw3', '.fb2']:
        return 'ebook'
    
    # XML files
    elif extension in ['.xml', '.rss', '.atom']:
        return 'xml'
    
    # HTML files (web articles)
    elif extension in ['.html', '.htm']:
        return 'html'
    
    # Plain text files
    elif extension in ['.txt', '.md', '.markdown', '.rst', '.log', '.csv']:
        return 'plaintext'
    
    # Audio files
    elif extension in ['.mp3', '.m4a', '.wav', '.flac', '.ogg', '.aac', '.wma', '.opus']:
        return 'audio'
    
    # Video files
    elif extension in ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv', '.m4v', '.mpg', '.mpeg']:
        return 'video'
    
    else:
        raise FileIngestionError(
            f"Unsupported file type: {extension}. "
            f"Supported types: PDF, DOCX, ODT, RTF, EPUB, MOBI, AZW, FB2, XML, HTML, TXT, MD, "
            f"MP3, M4A, WAV, FLAC, OGG, AAC, MP4, AVI, MKV, MOV, WEBM"
        )


def get_supported_extensions() -> Dict[str, List[str]]:
    """
    Get all supported file extensions organized by media type.
    
    Returns:
        Dictionary mapping media types to their supported extensions
    """
    return {
        'pdf': ['.pdf'],
        'document': ['.doc', '.docx', '.odt', '.rtf'],
        'ebook': ['.epub', '.mobi', '.azw', '.azw3', '.fb2'],
        'xml': ['.xml', '.rss', '.atom'],
        'html': ['.html', '.htm'],
        'plaintext': ['.txt', '.md', '.markdown', '.rst', '.log', '.csv'],
        'audio': ['.mp3', '.m4a', '.wav', '.flac', '.ogg', '.aac', '.wma', '.opus'],
        'video': ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv', '.m4v', '.mpg', '.mpeg']
    }


def ingest_local_file(
    file_path: Union[str, Path],
    media_db: MediaDatabase,
    title: Optional[str] = None,
    author: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    custom_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    perform_analysis: bool = False,
    api_name: Optional[str] = None,
    api_key: Optional[str] = None,
    chunk_options: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Ingest a local file into the Media database.
    
    Args:
        file_path: Path to the file to ingest
        media_db: MediaDatabase instance to store the content
        title: Optional title override (defaults to filename)
        author: Optional author name
        keywords: Optional list of keywords
        custom_prompt: Optional custom prompt for analysis
        system_prompt: Optional system prompt for analysis
        perform_analysis: Whether to perform analysis (summarization)
        api_name: API provider name for analysis (if enabled)
        api_key: API key for analysis provider (if needed)
        chunk_options: Dictionary with chunking options:
            - method: 'semantic', 'tokens', 'paragraphs', 'sentences', 'words', 'ebook_chapters'
            - size: chunk size (default varies by method)
            - overlap: chunk overlap (default varies by method)
            - adaptive: use adaptive chunking (bool)
            - multi_level: use multi-level chunking (bool)
            - language: language code for semantic chunking
        metadata: Additional metadata to store with the media
        
    Returns:
        Dictionary with ingestion results:
            - media_id: ID of the created media entry
            - title: Title of the media
            - author: Author of the media
            - content_length: Length of extracted content
            - chunks_created: Number of chunks created
            - keywords: Keywords associated with the media
            - analysis: Analysis results (if performed)
            - error: Error message (if any)
            
    Raises:
        FileIngestionError: If ingestion fails
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    
    # Validate file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Detect file type
    try:
        file_type = detect_file_type(file_path)
    except FileIngestionError as e:
        logger.error(f"Unsupported file type: {file_path} - {e}")
        raise
    
    # Set default values
    if title is None:
        title = file_path.stem
    if keywords is None:
        keywords = []
    if chunk_options is None:
        chunk_options = {}
    
    # Prepare common parameters
    common_params = {
        'title': title,
        'author': author or 'Unknown',
        'keywords': ', '.join(keywords) if keywords else '',
        'custom_prompt': custom_prompt,
        'system_prompt': system_prompt,
        'summary': perform_analysis,
        'api_name': api_name,
        'api_key': api_key,
    }
    
    # Get media-specific defaults from config
    from ..config import get_media_ingestion_defaults
    
    # Map file types to media types used in config
    file_type_to_media_type = {
        'pdf': 'pdf',
        'document': 'document',
        'ebook': 'ebook',
        'xml': 'xml',
        'plaintext': 'plaintext',
        'html': 'web_article',  # HTML files map to web_article config
        'audio': 'audio',
        'video': 'video'
    }
    
    media_type = file_type_to_media_type.get(file_type, file_type)
    config_defaults = get_media_ingestion_defaults(media_type)
    
    # Extract defaults with proper key mapping
    defaults = {
        'method': config_defaults.get('chunk_method', 'paragraphs'),
        'size': config_defaults.get('chunk_size', 500),
        'overlap': config_defaults.get('chunk_overlap', 200)
    }
    common_params.update({
        'chunk_method': chunk_options.get('method', defaults.get('method', 'paragraphs')),
        'chunk_size': chunk_options.get('size', defaults.get('size', 500)),
        'chunk_overlap': chunk_options.get('overlap', defaults.get('overlap', 200)),
        'use_adaptive_chunking': chunk_options.get('adaptive', False),
        'use_multi_level_chunking': chunk_options.get('multi_level', False),
        'chunk_language': chunk_options.get('language', ''),
    })
    
    try:
        logger.info(f"Ingesting {file_type} file: {file_path}")
        
        # Process based on file type
        if file_type == 'pdf':
            result = process_pdf(
                file_input=str(file_path),
                filename=file_path.name,
                title_override=title,
                author_override=author,
                keywords=keywords,
                perform_chunking=True,
                chunk_options=chunk_options,
                perform_analysis=perform_analysis,
                api_name=api_name,
                api_key=api_key,
                custom_prompt=custom_prompt,
                system_prompt=system_prompt
            )
            
        elif file_type == 'document':
            result = process_document(
                file_path=str(file_path),
                title_override=title,
                author_override=author,
                keywords=keywords,
                custom_prompt=custom_prompt,
                system_prompt=system_prompt,
                auto_summarize=perform_analysis,
                api_name=api_name,
                api_key=api_key,
                chunk_options=chunk_options
            )
            
        elif file_type == 'ebook':
            result = process_ebook(
                file_path=str(file_path),
                title_override=title,
                author_override=author,
                keywords=keywords,
                custom_prompt=custom_prompt,
                system_prompt=system_prompt,
                perform_chunking=True,
                chunk_options=chunk_options,
                perform_analysis=perform_analysis,
                api_name=api_name,
                api_key=api_key
            )
            
        elif file_type == 'audio':
            # Initialize audio processor
            audio_processor = LocalAudioProcessor(media_db)
            
            # Process single audio file
            results = audio_processor.process_audio_files(
                inputs=[str(file_path)],
                transcription_model=chunk_options.get('transcription_model', 'base'),
                transcription_language=chunk_options.get('transcription_language', 'en'),
                perform_chunking=True,
                chunk_method=chunk_options.get('method', 'sentences'),
                max_chunk_size=chunk_options.get('size', 500),
                chunk_overlap=chunk_options.get('overlap', 200),
                use_adaptive_chunking=chunk_options.get('adaptive', False),
                use_multi_level_chunking=chunk_options.get('multi_level', False),
                chunk_language=chunk_options.get('language', 'en'),
                diarize=chunk_options.get('diarize', False),
                vad_use=chunk_options.get('vad_filter', False),
                timestamp_option=True,
                perform_analysis=perform_analysis,
                api_name=api_name,
                api_key=api_key,
                custom_prompt=custom_prompt,
                system_prompt=system_prompt,
                summarize_recursively=chunk_options.get('recursive_summary', False),
                custom_title=title,
                author=author
            )
            
            # Extract first (and only) result
            if results['results']:
                result_data = results['results'][0]
                if result_data['status'] == 'Error':
                    raise FileIngestionError(f"Audio processing failed: {result_data.get('error', 'Unknown error')}")
                    
                result = {
                    'content': result_data.get('content', ''),
                    'title': result_data.get('metadata', {}).get('title', title),
                    'author': result_data.get('metadata', {}).get('author', author or 'Unknown'),
                    'keywords': keywords,
                    'chunks': result_data.get('chunks', []),
                    'analysis': result_data.get('analysis', ''),
                    'metadata': result_data.get('metadata', {})
                }
            else:
                raise FileIngestionError("Audio processing returned no results")
                
        elif file_type == 'video':
            # Initialize video processor
            video_processor = LocalVideoProcessor(media_db)
            
            # Process single video file
            results = video_processor.process_videos(
                inputs=[str(file_path)],
                download_video_flag=False,  # Extract audio only for transcription
                transcription_model=chunk_options.get('transcription_model', 'base'),
                transcription_language=chunk_options.get('transcription_language', 'en'),
                perform_chunking=True,
                chunk_method=chunk_options.get('method', 'sentences'),
                max_chunk_size=chunk_options.get('size', 500),
                chunk_overlap=chunk_options.get('overlap', 200),
                use_adaptive_chunking=chunk_options.get('adaptive', False),
                use_multi_level_chunking=chunk_options.get('multi_level', False),
                chunk_language=chunk_options.get('language', 'en'),
                diarize=chunk_options.get('diarize', False),
                vad_use=chunk_options.get('vad_filter', False),
                timestamp_option=True,
                perform_analysis=perform_analysis,
                api_name=api_name,
                api_key=api_key,
                custom_prompt=custom_prompt,
                system_prompt=system_prompt,
                summarize_recursively=chunk_options.get('recursive_summary', False),
                custom_title=title,
                author=author
            )
            
            # Extract first (and only) result
            if results['results']:
                result_data = results['results'][0]
                if result_data['status'] == 'Error':
                    raise FileIngestionError(f"Video processing failed: {result_data.get('error', 'Unknown error')}")
                    
                result = {
                    'content': result_data.get('content', ''),
                    'title': result_data.get('metadata', {}).get('title', title),
                    'author': result_data.get('metadata', {}).get('author', author or 'Unknown'),
                    'keywords': keywords,
                    'chunks': result_data.get('chunks', []),
                    'analysis': result_data.get('analysis', ''),
                    'metadata': result_data.get('metadata', {})
                }
            else:
                raise FileIngestionError("Video processing returned no results")
            
        elif file_type == 'xml':
            # XML processing not yet implemented in the expected format
            raise FileIngestionError(f"XML file processing is not yet implemented")
            
        elif file_type == 'plaintext':
            # For plaintext files, we'll process them directly
            content = file_path.read_text(encoding='utf-8', errors='replace')
            
            # Simple result structure for plaintext
            result = {
                'content': content,
                'title': title,
                'author': author or 'Unknown',
                'keywords': keywords,
                'chunks': [],  # Chunking will be handled by the database
                'analysis': ''
            }
            
            # If chunking is requested, we'll let the database handle it
            # The MediaDatabase.add_media_with_keywords method will handle chunking
            
        elif file_type == 'html':
            # For HTML files, we'll extract text content
            from bs4 import BeautifulSoup
            
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Parse HTML and extract text
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title if present
            title_tag = soup.find('title')
            if title_tag and not title:
                title = title_tag.text.strip()
            
            # Extract text content
            content = soup.get_text(separator='\n', strip=True)
            
            result = {
                'content': content,
                'title': title or file_path.stem,
                'author': author or 'Unknown',
                'keywords': keywords,
                'chunks': [],  # Chunking will be handled by the database
                'analysis': ''
            }
        
        # Check if processing was successful
        if not result or 'error' in result:
            error_msg = result.get('error', 'Unknown error') if result else 'Processing returned no result'
            raise FileIngestionError(f"Failed to process {file_type} file: {error_msg}")
        
        # Extract content and metadata
        content = result.get('content', '')
        extracted_title = result.get('title', title)
        extracted_author = result.get('author', author or 'Unknown')
        extracted_keywords = result.get('keywords', [])
        chunks = result.get('chunks', [])
        analysis = result.get('analysis', '')
        
        # Combine keywords
        all_keywords = list(set(keywords + extracted_keywords))
        
        # Add custom metadata
        media_metadata = result.get('metadata', {})
        if metadata:
            media_metadata.update(metadata)
        media_metadata['ingestion_method'] = 'local_file'
        media_metadata['file_path'] = str(file_path)
        media_metadata['file_type'] = file_type
        
        # Store in database
        logger.debug(f"Storing {file_type} content in database...")
        # Note: add_media_with_keywords returns tuple: (media_id, media_uuid, message)
        media_id, media_uuid, message = media_db.add_media_with_keywords(
            title=extracted_title,
            media_type=file_type,
            content=content,
            keywords=all_keywords,
            url=f"file://{file_path.absolute()}",
            analysis_content=analysis,
            author=extracted_author,
            ingestion_date=datetime.now().strftime('%Y-%m-%d'),
            chunks=chunks if chunks else None,
            chunk_options=chunk_options if chunk_options else None
        )
        
        logger.info(f"Successfully ingested {file_type} file with media_id: {media_id}")
        
        # Return results
        return {
            'media_id': media_id,
            'title': extracted_title,
            'author': extracted_author,
            'content_length': len(content),
            'chunks_created': len(chunks),
            'keywords': all_keywords,
            'analysis': analysis,
            'file_type': file_type,
            'file_path': str(file_path)
        }
        
    except Exception as e:
        logger.error(f"Error ingesting {file_type} file {file_path}: {e}")
        raise FileIngestionError(f"Failed to ingest {file_type} file: {str(e)}")


def batch_ingest_files(
    file_paths: List[Union[str, Path]],
    media_db: MediaDatabase,
    common_keywords: Optional[List[str]] = None,
    perform_analysis: bool = False,
    api_name: Optional[str] = None,
    api_key: Optional[str] = None,
    chunk_options: Optional[Dict[str, Any]] = None,
    stop_on_error: bool = False
) -> List[Dict[str, Any]]:
    """
    Ingest multiple files in batch.
    
    Args:
        file_paths: List of file paths to ingest
        media_db: MediaDatabase instance
        common_keywords: Keywords to apply to all files
        perform_analysis: Whether to perform analysis on all files
        api_name: API provider for analysis
        api_key: API key for analysis
        chunk_options: Chunking options for all files
        stop_on_error: Whether to stop on first error or continue
        
    Returns:
        List of ingestion results (one per file)
        
    Raises:
        FileIngestionError: If stop_on_error is True and an error occurs
    """
    results = []
    
    for file_path in file_paths:
        try:
            result = ingest_local_file(
                file_path=file_path,
                media_db=media_db,
                keywords=common_keywords,
                perform_analysis=perform_analysis,
                api_name=api_name,
                api_key=api_key,
                chunk_options=chunk_options
            )
            results.append(result)
            
        except Exception as e:
            error_result = {
                'file_path': str(file_path),
                'error': str(e),
                'success': False
            }
            results.append(error_result)
            
            if stop_on_error:
                raise FileIngestionError(f"Batch ingestion stopped: {e}")
            else:
                logger.error(f"Error ingesting {file_path}, continuing with next file: {e}")
    
    return results


def ingest_directory(
    directory_path: Union[str, Path],
    media_db: MediaDatabase,
    recursive: bool = False,
    file_types: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Ingest all supported files in a directory.
    
    Args:
        directory_path: Path to directory
        media_db: MediaDatabase instance
        recursive: Whether to process subdirectories
        file_types: List of file types to process (e.g., ['pdf', 'document'])
                   If None, process all supported types
        exclude_patterns: List of filename patterns to exclude (e.g., ['*.tmp', 'draft_*'])
        **kwargs: Additional arguments passed to ingest_local_file
        
    Returns:
        List of ingestion results
    """
    directory_path = Path(directory_path)
    
    if not directory_path.is_dir():
        raise FileIngestionError(f"Not a directory: {directory_path}")
    
    # Get all files
    if recursive:
        files = list(directory_path.rglob('*'))
    else:
        files = list(directory_path.glob('*'))
    
    # Filter to only files (not directories)
    files = [f for f in files if f.is_file()]
    
    # Filter by file type if specified
    if file_types:
        filtered_files = []
        for file_path in files:
            try:
                if detect_file_type(file_path) in file_types:
                    filtered_files.append(file_path)
            except FileIngestionError:
                # Skip unsupported file types
                pass
        files = filtered_files
    else:
        # Filter to only supported file types
        supported_files = []
        for file_path in files:
            try:
                detect_file_type(file_path)
                supported_files.append(file_path)
            except FileIngestionError:
                # Skip unsupported file types
                pass
        files = supported_files
    
    # Apply exclude patterns
    if exclude_patterns:
        import fnmatch
        filtered_files = []
        for file_path in files:
            excluded = False
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(file_path.name, pattern):
                    excluded = True
                    break
            if not excluded:
                filtered_files.append(file_path)
        files = filtered_files
    
    logger.info(f"Found {len(files)} files to ingest in {directory_path}")
    
    # Ingest files
    return batch_ingest_files(
        file_paths=files,
        media_db=media_db,
        **kwargs
    )


# Convenience function for use in scripts
def quick_ingest(file_path: Union[str, Path], db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick ingestion function for scripts and notebooks.
    
    Args:
        file_path: Path to file to ingest
        db_path: Optional path to media database (uses default if not provided)
        
    Returns:
        Ingestion result dictionary
    """
    from ..config import get_cli_setting
    
    if db_path is None:
        db_config = get_cli_setting("database", {})
        db_path = db_config.get("media_db_path", "~/.local/share/tldw_cli/tldw_cli_media_v2.db")
        db_path = Path(db_path).expanduser()
    
    # Initialize database
    media_db = MediaDatabase(str(db_path), client_id="quick_ingest")
    
    try:
        result = ingest_local_file(file_path, media_db)
        return result
    finally:
        media_db.close_connection()