#!/usr/bin/env python3
"""
Quick test to verify imports work correctly.
"""

print("Testing imports...")

try:
    print("1. Testing enhanced chunking service...")
    from tldw_chatbook.RAG_Search.enhanced_chunking_service import EnhancedChunkingService
    print("   ✓ EnhancedChunkingService imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import EnhancedChunkingService: {e}")

try:
    print("\n2. Testing table serializer...")
    from tldw_chatbook.RAG_Search.table_serializer import TableProcessor, serialize_table
    print("   ✓ TableProcessor and serialize_table imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import table serializer: {e}")

try:
    print("\n3. Testing enhanced RAG service...")
    from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service import EnhancedRAGService
    print("   ✓ EnhancedRAGService imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import EnhancedRAGService: {e}")

try:
    print("\n4. Testing Literal import...")
    try:
        from typing import Literal
        print("   ✓ Literal imported from typing")
    except ImportError:
        from typing_extensions import Literal
        print("   ✓ Literal imported from typing_extensions")
except Exception as e:
    print(f"   ✗ Failed to import Literal: {e}")

print("\n✅ All imports successful!")