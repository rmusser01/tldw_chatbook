#!/usr/bin/env python3
"""
run_embeddings_tests.py
Run the comprehensive embeddings test suite with various configurations
"""

import sys
import subprocess
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd, description):
    """Run a command and report results"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"❌ Failed with return code: {result.returncode}")
    else:
        print(f"✅ Passed")
    
    return result.returncode == 0


def main():
    """Run all embeddings tests with different configurations"""
    
    # Change to project root
    os.chdir(project_root)
    
    test_dir = "Tests/RAG_Search"
    all_passed = True
    
    # Test configurations
    test_configs = [
        {
            "name": "Unit Tests",
            "cmd": ["pytest", f"{test_dir}/test_embeddings_unit.py", "-v", "--tb=short"],
            "description": "Testing individual components in isolation"
        },
        {
            "name": "Integration Tests", 
            "cmd": ["pytest", f"{test_dir}/test_embeddings_integration.py", "-v", "-m", "integration", "--tb=short"],
            "description": "Testing component interactions"
        },
        {
            "name": "Property-Based Tests",
            "cmd": ["pytest", f"{test_dir}/test_embeddings_properties.py", "-v", "--tb=short", "--hypothesis-show-statistics"],
            "description": "Testing invariants with Hypothesis"
        },
        {
            "name": "Performance Tests",
            "cmd": ["pytest", f"{test_dir}/test_embeddings_performance.py", "-v", "-m", "performance", "--tb=short"],
            "description": "Benchmarking performance characteristics"
        },
        {
            "name": "Compatibility Tests",
            "cmd": ["pytest", f"{test_dir}/test_embeddings_compatibility.py", "-v", "-m", "compatibility", "--tb=short"],
            "description": "Testing backward compatibility"
        },
        {
            "name": "Original Service Tests",
            "cmd": ["pytest", f"{test_dir}/test_embeddings_service.py", "-v", "--tb=short"],
            "description": "Testing the original test suite still works"
        }
    ]
    
    # Run each test configuration
    for config in test_configs:
        passed = run_command(config["cmd"], f"{config['name']}: {config['description']}")
        if not passed:
            all_passed = False
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    if all_passed:
        print("✅ All test suites passed!")
    else:
        print("❌ Some tests failed. Please review the output above.")
    
    # Run coverage report if all tests passed
    if all_passed:
        print(f"\n{'='*60}")
        print("Running coverage analysis...")
        print('='*60)
        
        coverage_cmd = [
            "pytest",
            f"{test_dir}/test_embeddings_unit.py",
            f"{test_dir}/test_embeddings_integration.py", 
            f"{test_dir}/test_embeddings_properties.py",
            f"{test_dir}/test_embeddings_compatibility.py",
            "--cov=tldw_chatbook.RAG_Search.Services.embeddings_service",
            "--cov=tldw_chatbook.RAG_Search.Services.embeddings_compat",
            "--cov-report=term-missing",
            "--tb=short"
        ]
        
        run_command(coverage_cmd, "Code coverage analysis")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())