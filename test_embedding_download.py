#!/usr/bin/env python3
"""Test embedding model download directly."""

import os
import sys

# Set environment variables to avoid subprocess issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Test different models
models_to_test = [
    ("mixedbread-ai/mxbai-embed-large-v1", "mxbai-embed-large-v1"),
    ("intfloat/e5-small-v2", "e5-small-v2"), 
    ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2")
]

print("Testing embedding model downloads...\n")

for model_path, model_name in models_to_test:
    print(f"Testing {model_name} ({model_path})...")
    try:
        from transformers import AutoModel, AutoTokenizer
        
        # Try to load tokenizer
        print(f"  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"  ✓ Tokenizer loaded successfully")
        
        # Try to load model
        print(f"  Loading model...")
        model = AutoModel.from_pretrained(model_path)
        print(f"  ✓ Model loaded successfully")
        print(f"  Model type: {type(model).__name__}")
        print()
        
    except Exception as e:
        print(f"  ✗ Failed: {type(e).__name__}: {str(e)}")
        print()

print("\nTest complete. If all models failed, check your internet connection.")
print("If only mxbai-embed-large-v1 failed, the model might not be publicly available.")