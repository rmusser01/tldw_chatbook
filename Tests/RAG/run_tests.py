#!/usr/bin/env python3
# run_tests.py
# Script to run RAG tests with proper configuration

import subprocess
import sys
from pathlib import Path

# Test categories
TEST_CATEGORIES = {
    'unit': [
        'test_cache_service.py',
        'test_chunking_service.py',
        'test_embeddings_service.py',
        'test_indexing_service.py'
    ],
    'integration': [
        'test_rag_integration.py'
    ],
    'property': [
        'test_rag_properties.py'
    ]
}

def run_tests(category=None, verbose=False, timeout=None):
    """Run RAG tests"""
    cmd = ['pytest']
    
    if verbose:
        cmd.append('-v')
    
    # Add timeout if specified
    if timeout:
        cmd.extend(['--timeout', str(timeout)])
    
    # Add coverage if available
    try:
        import pytest_cov
        cmd.extend(['--cov=tldw_chatbook.RAG_Search', '--cov-report=term-missing'])
    except ImportError:
        pass
    
    if category and category in TEST_CATEGORIES:
        # Run specific category
        for test_file in TEST_CATEGORIES[category]:
            cmd.append(test_file)
        print(f"Running {category} tests...")
    else:
        # Run all tests
        cmd.append('.')
        print("Running all RAG tests...")
    
    # Change to the test directory
    test_dir = Path(__file__).parent
    result = subprocess.run(cmd, cwd=test_dir)
    
    return result.returncode

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run RAG tests')
    parser.add_argument(
        'category',
        nargs='?',
        choices=['all', 'unit', 'integration', 'property'],
        default='all',
        help='Test category to run'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '-t', '--timeout',
        type=int,
        default=60,
        help='Timeout for each test in seconds (default: 60)'
    )
    
    args = parser.parse_args()
    
    if args.category == 'all':
        category = None
    else:
        category = args.category
    
    exit_code = run_tests(category, args.verbose, args.timeout)
    sys.exit(exit_code)

if __name__ == '__main__':
    main()