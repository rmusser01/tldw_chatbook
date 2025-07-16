#!/usr/bin/env python3
"""Test if the imports work correctly."""

try:
    from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
        ingest_local_file,
        batch_ingest_files,
        ingest_directory,
        get_supported_extensions,
        detect_file_type,
        FileIngestionError
    )
    print("✓ All imports successful!")
    
    # Test get_supported_extensions
    extensions = get_supported_extensions()
    print(f"✓ get_supported_extensions returns: {type(extensions)}")
    print(f"  Contains {len(extensions)} media types")
    
    # Test detect_file_type
    try:
        file_type = detect_file_type("test.pdf")
        print(f"✓ detect_file_type('test.pdf') returns: '{file_type}'")
    except Exception as e:
        print(f"✗ detect_file_type failed: {e}")
        
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")