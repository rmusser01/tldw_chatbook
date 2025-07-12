#!/usr/bin/env python3
"""
Test script to verify optional dependency handling.
Run this to test that the optional dependency system is working correctly.
"""

import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run the optional dependency tests."""
    test_file = Path(__file__).parent / "Tests" / "Utils" / "test_optional_deps.py"
    
    if not test_file.exists():
        print(f"Error: Test file not found at {test_file}")
        return 1
    
    print("Running optional dependency tests...")
    print("-" * 60)
    
    # Run pytest with verbose output
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
        capture_output=False
    )
    
    return result.returncode

def check_dependencies():
    """Check and report on optional dependencies."""
    print("\nChecking optional dependencies...")
    print("-" * 60)
    
    try:
        from tldw_chatbook.Utils.optional_deps import DEPENDENCIES_AVAILABLE, initialize_dependency_checks
        
        # Initialize checks
        initialize_dependency_checks()
        
        # Group dependencies by category
        categories = {
            'Core': ['embeddings_rag', 'web_scraping'],
            'Document Processing': ['pdf_processing', 'ebook_processing'],
            'Local LLM': ['local_llm', 'vllm', 'onnxruntime', 'mlx_lm'],
            'Media': ['tts', 'stt', 'ocr', 'image_processing'],
            'Other': ['mcp']
        }
        
        for category, deps in categories.items():
            print(f"\n{category}:")
            for dep in deps:
                if dep in DEPENDENCIES_AVAILABLE:
                    status = "✅ Available" if DEPENDENCIES_AVAILABLE[dep] else "❌ Not available"
                    print(f"  {dep}: {status}")
        
        # Show individual library status for key features
        print("\nDetailed Library Status:")
        key_libs = {
            'PDF': ['pymupdf', 'pymupdf4llm', 'docling'],
            'E-book': ['ebooklib', 'html2text', 'defusedxml'],
            'Web Scraping': ['beautifulsoup4', 'playwright', 'trafilatura', 'lxml', 'pandas'],
            'Embeddings': ['torch', 'transformers', 'numpy', 'chromadb'],
        }
        
        for feature, libs in key_libs.items():
            print(f"\n  {feature}:")
            for lib in libs:
                if lib in DEPENDENCIES_AVAILABLE:
                    status = "✅" if DEPENDENCIES_AVAILABLE[lib] else "❌"
                    print(f"    {lib}: {status}")
                    
    except Exception as e:
        print(f"Error checking dependencies: {e}")
        return 1
    
    return 0

def main():
    """Main entry point."""
    print("=" * 60)
    print("Optional Dependency System Test")
    print("=" * 60)
    
    # First check dependencies
    dep_result = check_dependencies()
    
    # Then run tests
    test_result = run_tests()
    
    # Summary
    print("\n" + "=" * 60)
    if test_result == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    print("\nTo install missing dependencies:")
    print("  - For PDF processing: pip install tldw_chatbook[pdf]")
    print("  - For e-book processing: pip install tldw_chatbook[ebook]")
    print("  - For web scraping: pip install tldw_chatbook[websearch]")
    print("  - For embeddings/RAG: pip install tldw_chatbook[embeddings_rag]")
    print("  - For all features: pip install tldw_chatbook[all]")
    print("=" * 60)
    
    return max(dep_result, test_result)

if __name__ == "__main__":
    sys.exit(main())