#!/usr/bin/env python3
"""Test script to verify AutoModel initialization issue"""

import sys
import os

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test the issue
print("Testing AutoModel initialization issue...")

# First, let's check if transformers is available
try:
    import transformers
    print("✅ transformers is available")
    from transformers import AutoModel, AutoTokenizer
    print("✅ AutoModel and AutoTokenizer imported successfully from transformers")
except ImportError as e:
    print(f"❌ transformers is not available: {e}")

# Now let's test the Embeddings_Lib module
print("\nTesting Embeddings_Lib module...")
try:
    from tldw_chatbook.Embeddings.Embeddings_Lib import (
        EmbeddingFactory, 
        EmbeddingConfigSchema,
        AutoModel as EmbeddingsAutoModel,
        AutoTokenizer as EmbeddingsAutoTokenizer
    )
    print("✅ Successfully imported from Embeddings_Lib")
    
    # Check what AutoModel actually is
    print(f"\nAutoModel type from Embeddings_Lib: {type(EmbeddingsAutoModel)}")
    print(f"AutoModel value: {EmbeddingsAutoModel}")
    
    # Try to use it (this should fail if transformers is not available)
    if callable(EmbeddingsAutoModel):
        print("\n⚠️  AutoModel is a callable (function), not a class")
        try:
            # This will raise an error if it's the placeholder
            result = EmbeddingsAutoModel()
        except Exception as e:
            print(f"❌ Calling AutoModel raised: {type(e).__name__}: {e}")
    
except ImportError as e:
    print(f"❌ Failed to import from Embeddings_Lib: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {type(e).__name__}: {e}")

# Test the optional_deps module
print("\n\nTesting optional_deps module...")
try:
    from tldw_chatbook.Utils.optional_deps import (
        DEPENDENCIES_AVAILABLE,
        create_unavailable_feature_handler
    )
    print("✅ Successfully imported optional_deps")
    
    print(f"\nDependencies available:")
    for dep in ['torch', 'transformers', 'numpy', 'embeddings_rag']:
        available = DEPENDENCIES_AVAILABLE.get(dep, False)
        status = "✅" if available else "❌"
        print(f"  {status} {dep}: {available}")
    
    # Test the placeholder function behavior
    print("\n\nTesting placeholder function behavior...")
    test_placeholder = create_unavailable_feature_handler('test_feature', 'pip install test')
    print(f"Placeholder type: {type(test_placeholder)}")
    print(f"Placeholder callable: {callable(test_placeholder)}")
    
    try:
        test_placeholder()
    except ImportError as e:
        print(f"✅ Placeholder correctly raised ImportError: {e}")
    
except Exception as e:
    print(f"❌ Error testing optional_deps: {type(e).__name__}: {e}")

print("\n\nSummary:")
print("=" * 50)
print("The issue is that AutoModel is being assigned a placeholder function")
print("that raises ImportError when called, instead of being a proper class.")
print("This happens when transformers is not installed.")
print("\nThe code should check for transformers availability before trying")
print("to use AutoModel, which it does in the _build method.")
print("\nThe placeholder is only problematic if code tries to use AutoModel")
print("without first checking if the dependencies are available.")