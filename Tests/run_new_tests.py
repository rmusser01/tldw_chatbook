#!/usr/bin/env python3
"""
Run the newly created tests for validation and pagination features.
"""

import subprocess
import sys
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent

# Add project to Python path
sys.path.insert(0, str(project_root))

# Define test modules to run
test_modules = [
    "Tests.Utils.test_path_validation",
    "Tests.Utils.test_path_validation_properties",
    "Tests.DB.test_sql_validation",
    "Tests.DB.test_pagination",
    "Tests.integration.test_file_operations_with_validation",
]

def run_tests():
    """Run all the new tests."""
    print("Running new validation and pagination tests...\n")
    
    failed_modules = []
    
    for module in test_modules:
        print(f"\n{'='*60}")
        print(f"Running tests in: {module}")
        print('='*60)
        
        # Run pytest for this module
        cmd = [sys.executable, "-m", "pytest", "-v", "-s", f"{module.replace('.', '/')}.py"]
        
        result = subprocess.run(cmd, cwd=project_root)
        
        if result.returncode != 0:
            failed_modules.append(module)
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    if failed_modules:
        print(f"\n❌ {len(failed_modules)} module(s) had failing tests:")
        for module in failed_modules:
            print(f"   - {module}")
        return 1
    else:
        print("\n✅ All tests passed!")
        return 0

if __name__ == "__main__":
    # Install test dependencies if needed
    print("Checking test dependencies...")
    try:
        import pytest
        import hypothesis
    except ImportError:
        print("Installing test dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "hypothesis"])
    
    # Run the tests
    exit_code = run_tests()
    sys.exit(exit_code)