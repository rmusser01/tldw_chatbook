#!/usr/bin/env python3
# run_tests.py
# Description: Test runner for the refactored Evals module
#
"""
Evals Module Test Runner
------------------------

Executes all unit and integration tests for the refactored Evals module.
"""

import sys
import pytest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_all_tests():
    """Run all Evals module tests."""
    test_dir = Path(__file__).parent
    
    print("=" * 70)
    print("Running Evals Module Tests")
    print("=" * 70)
    
    # Test categories
    test_suites = [
        ("Unit Tests - Orchestrator", ["test_eval_orchestrator.py::TestEvaluationOrchestrator"]),
        ("Unit Tests - Error Handling", ["test_eval_errors.py"]),
        ("Unit Tests - Exporters", ["test_exporters.py"]),
        ("Integration Tests", ["test_integration.py"]),
    ]
    
    results = {}
    
    for suite_name, test_files in test_suites:
        print(f"\n{'=' * 40}")
        print(f"Running: {suite_name}")
        print(f"{'=' * 40}")
        
        args = ["-v", "--tb=short"] + [str(test_dir / f) for f in test_files]
        result = pytest.main(args)
        
        results[suite_name] = result
        
        if result == 0:
            print(f"✅ {suite_name}: PASSED")
        else:
            print(f"❌ {suite_name}: FAILED")
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    total_suites = len(results)
    passed_suites = sum(1 for r in results.values() if r == 0)
    
    for suite_name, result in results.items():
        status = "✅ PASSED" if result == 0 else "❌ FAILED"
        print(f"{suite_name}: {status}")
    
    print(f"\nTotal: {passed_suites}/{total_suites} suites passed")
    
    return 0 if all(r == 0 for r in results.values()) else 1


def run_specific_test(test_name):
    """Run a specific test suite."""
    test_dir = Path(__file__).parent
    
    test_map = {
        'orchestrator': 'test_eval_orchestrator.py',
        'errors': 'test_eval_errors.py',
        'exporters': 'test_exporters.py',
        'integration': 'test_integration.py',
    }
    
    if test_name not in test_map:
        print(f"Unknown test suite: {test_name}")
        print(f"Available: {', '.join(test_map.keys())}")
        return 1
    
    test_file = test_dir / test_map[test_name]
    args = ["-v", "--tb=short", str(test_file)]
    
    print(f"Running {test_name} tests...")
    return pytest.main(args)


def run_coverage():
    """Run tests with coverage report."""
    test_dir = Path(__file__).parent
    evals_dir = Path(__file__).parent.parent.parent / "tldw_chatbook" / "Evals"
    
    args = [
        "--cov=" + str(evals_dir),
        "--cov-report=term-missing",
        "--cov-report=html",
        "-v",
        str(test_dir)
    ]
    
    print("Running tests with coverage...")
    result = pytest.main(args)
    
    if result == 0:
        print("\n✅ Coverage report generated in htmlcov/index.html")
    
    return result


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Evals module tests")
    parser.add_argument(
        'command',
        nargs='?',
        default='all',
        choices=['all', 'orchestrator', 'errors', 'exporters', 'integration', 'coverage'],
        help='Test suite to run (default: all)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    if args.command == 'all':
        return run_all_tests()
    elif args.command == 'coverage':
        return run_coverage()
    else:
        return run_specific_test(args.command)


if __name__ == "__main__":
    sys.exit(main())