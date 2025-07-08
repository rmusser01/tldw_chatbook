#!/usr/bin/env python3
"""
Example script demonstrating programmatic local file ingestion.

This script shows how to ingest local files into the tldw_chatbook Media database
without using the UI.
"""

import sys
from pathlib import Path
from loguru import logger

# Add parent directory to path if running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from tldw_chatbook.Local_Ingestion import (
    ingest_local_file,
    batch_ingest_files,
    ingest_directory,
    get_supported_extensions
)
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.config import get_project_root, get_database_dir


def example_single_file_ingestion():
    """Example: Ingest a single file."""
    print("\n=== Single File Ingestion Example ===")
    
    # Initialize database
    db_path = get_database_dir() / "media_db.sqlite"
    media_db = MediaDatabase(str(db_path), client_id="example_script")
    
    # Example file path (adjust to your actual file)
    file_path = Path("~/Documents/example.pdf").expanduser()
    
    if not file_path.exists():
        print(f"Example file not found: {file_path}")
        print("Please update the file_path variable to point to an actual file.")
        return
    
    # Simple ingestion
    result = ingest_local_file(
        file_path=file_path,
        media_db=media_db,
        keywords=["example", "test"]
    )
    
    print(f"Ingestion {'successful' if result['success'] else 'failed'}")
    print(f"Message: {result['message']}")
    if result['success']:
        print(f"Media ID: {result['media_id']}")
        print(f"Title: {result['title']}")


def example_advanced_ingestion():
    """Example: Ingest with analysis and custom options."""
    print("\n=== Advanced Ingestion Example ===")
    
    # Initialize database
    db_path = get_database_dir() / "media_db.sqlite"
    media_db = MediaDatabase(str(db_path), client_id="example_script")
    
    # Example file path
    file_path = Path("~/Documents/research_paper.pdf").expanduser()
    
    if not file_path.exists():
        print(f"Example file not found: {file_path}")
        return
    
    # Advanced ingestion with analysis
    result = ingest_local_file(
        file_path=file_path,
        media_db=media_db,
        title="Important Research Paper",
        author="Dr. Smith",
        keywords=["research", "science", "2024"],
        perform_analysis=True,
        api_name="openai",  # or your preferred API
        custom_prompt="Summarize the key findings and methodology of this research paper",
        perform_chunking=True,
        chunk_options={
            "method": "semantic",
            "chunk_size": 1000,
            "overlap": 200
        }
    )
    
    print(f"Ingestion {'successful' if result['success'] else 'failed'}")
    print(f"Processing details available: {bool(result['processing_details'])}")


def example_batch_ingestion():
    """Example: Ingest multiple files at once."""
    print("\n=== Batch Ingestion Example ===")
    
    # Initialize database
    db_path = get_database_dir() / "media_db.sqlite"
    media_db = MediaDatabase(str(db_path), client_id="example_script")
    
    # Example files
    file_paths = [
        Path("~/Documents/file1.pdf").expanduser(),
        Path("~/Documents/file2.docx").expanduser(),
        Path("~/Documents/file3.epub").expanduser(),
    ]
    
    # Filter to existing files
    existing_files = [f for f in file_paths if f.exists()]
    
    if not existing_files:
        print("No example files found. Please update file_paths.")
        return
    
    # Batch process with common options
    results = batch_ingest_files(
        file_paths=existing_files,
        media_db=media_db,
        common_keywords=["batch", "import"],
        perform_chunking=True,
        chunk_options={"method": "tokens", "chunk_size": 500}
    )
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"Processed {len(results)} files: {successful} successful")
    
    for i, result in enumerate(results):
        print(f"\nFile {i+1}: {result['file_path']}")
        print(f"  Status: {'Success' if result['success'] else 'Failed'}")
        print(f"  Message: {result['message']}")


def example_directory_ingestion():
    """Example: Ingest all supported files in a directory."""
    print("\n=== Directory Ingestion Example ===")
    
    # Initialize database
    db_path = get_database_dir() / "media_db.sqlite"
    media_db = MediaDatabase(str(db_path), client_id="example_script")
    
    # Example directory
    directory = Path("~/Documents/Books").expanduser()
    
    if not directory.exists():
        print(f"Example directory not found: {directory}")
        return
    
    # Process all PDFs and EPUBs in directory
    results = ingest_directory(
        directory_path=directory,
        media_db=media_db,
        recursive=True,  # Include subdirectories
        file_extensions=['.pdf', '.epub'],
        keywords=["books", "library"]
    )
    
    print(f"Found and processed {len(results)} files")
    successful = sum(1 for r in results if r['success'])
    print(f"Successful: {successful}/{len(results)}")


def show_supported_formats():
    """Show all supported file formats."""
    print("\n=== Supported File Formats ===")
    
    extensions = get_supported_extensions()
    print(f"Total supported extensions: {len(extensions)}")
    
    # Group by category
    pdf_exts = [e for e in extensions if 'pdf' in e]
    doc_exts = [e for e in extensions if e in ['.docx', '.doc', '.odt', '.rtf', '.pptx', '.xlsx', '.ods', '.odp']]
    ebook_exts = [e for e in extensions if e in ['.epub', '.mobi', '.azw', '.azw3', '.fb2']]
    text_exts = [e for e in extensions if e in ['.txt', '.md', '.markdown', '.rst', '.text']]
    xml_exts = [e for e in extensions if e in ['.xml', '.opml']]
    
    print("\nPDF:", ', '.join(pdf_exts))
    print("Documents:", ', '.join(doc_exts))
    print("E-books:", ', '.join(ebook_exts))
    print("Text:", ', '.join(text_exts))
    print("XML:", ', '.join(xml_exts))


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    print("=== tldw_chatbook Local File Ingestion Examples ===")
    
    # Show supported formats
    show_supported_formats()
    
    # Run examples (comment out any you don't want to run)
    try:
        # example_single_file_ingestion()
        # example_advanced_ingestion()
        # example_batch_ingestion()
        # example_directory_ingestion()
        
        print("\n\nTo run the examples, uncomment the function calls above and")
        print("update the file paths to point to actual files on your system.")
        
    except Exception as e:
        logger.error(f"Error in example: {e}", exc_info=True)