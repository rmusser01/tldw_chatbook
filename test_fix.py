#!/usr/bin/env python3
"""
Test script for ebook ingestion pipeline fixes
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tldw_chatbook.Local_Ingestion.Book_Ingestion_Lib import process_ebook

def test_process_ebook():
    """Test the process_ebook function with a simple example"""
    
    # Create a test EPUB file path (you'll need to provide an actual EPUB file)
    test_file = "/path/to/test.epub"  # Replace with actual test file
    
    print("Testing ebook processing...")
    
    # Test basic processing
    result = process_ebook(
        file_path=test_file,
        title_override="Test Book",
        author_override="Test Author",
        keywords=["test", "ebook"],
        perform_chunking=True,
        chunk_options={'method': 'ebook_chapters', 'max_size': 1000, 'overlap': 200},
        perform_analysis=False,
        extraction_method='filtered'
    )
    
    print(f"Status: {result.get('status')}")
    print(f"Title: {result.get('metadata', {}).get('title')}")
    print(f"Author: {result.get('metadata', {}).get('author')}")
    print(f"Content length: {len(result.get('content', ''))}")
    print(f"Number of chunks: {len(result.get('chunks', []))}")
    
    if result.get('warnings'):
        print(f"Warnings: {result['warnings']}")
    
    if result.get('error'):
        print(f"Error: {result['error']}")
    
    return result

def test_supported_formats():
    """Test that all supported formats are handled"""
    formats = ['.epub', '.mobi', '.azw', '.azw3', '.fb2']
    
    for fmt in formats:
        test_file = f"/path/to/test{fmt}"  # Replace with actual test files
        print(f"\nTesting format: {fmt}")
        
        # This will fail with file not found, but we can check the routing works
        result = process_ebook(file_path=test_file)
        print(f"  Routed correctly: {'Error' in result.get('status', '')}")

if __name__ == "__main__":
    print("Ebook Ingestion Pipeline Test")
    print("=" * 50)
    
    # You can uncomment these when you have test files
    # test_process_ebook()
    # test_supported_formats()
    
    print("\nTo fully test the pipeline:")
    print("1. Place test ebook files (EPUB, MOBI, FB2) in a test directory")
    print("2. Update the file paths in this script")
    print("3. Run the script to verify processing works")
    print("4. Check that content extraction, chunking, and metadata work correctly")