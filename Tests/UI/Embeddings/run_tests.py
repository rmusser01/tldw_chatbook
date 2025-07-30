#!/usr/bin/env python3
"""
Test runner for Embeddings UI tests.
Provides convenient options for running different test suites.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Embeddings UI tests")
    parser.add_argument(
        "suite",
        nargs="?",
        default="all",
        choices=[
            "all", "unit", "integration", "widgets", "windows",
            "toast", "progress", "preferences", "templates", 
            "activity", "performance", "coverage"
        ],
        help="Test suite to run"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-x", "--exitfirst", action="store_true", help="Exit on first failure")
    parser.add_argument("-k", "--keyword", help="Run tests matching keyword")
    parser.add_argument("--pdb", action="store_true", help="Drop to debugger on failures")
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["pytest"]
    
    # Add common options
    if args.verbose:
        cmd.append("-v")
    if args.exitfirst:
        cmd.append("-x")
    if args.pdb:
        cmd.append("--pdb")
    if args.keyword:
        cmd.extend(["-k", args.keyword])
    
    # Determine which tests to run
    test_dir = Path(__file__).parent
    
    if args.suite == "all":
        cmd.append(str(test_dir))
    elif args.suite == "unit":
        cmd.extend([str(test_dir), "-m", "not integration"])
    elif args.suite == "integration":
        cmd.append(str(test_dir / "test_integration.py"))
    elif args.suite == "widgets":
        # Run all widget tests
        cmd.extend([
            str(test_dir / "test_toast_notifications.py"),
            str(test_dir / "test_detailed_progress.py"),
            str(test_dir / "test_activity_log.py"),
            str(test_dir / "test_performance_metrics.py"),
        ])
    elif args.suite == "windows":
        # Run window tests
        cmd.extend([
            str(test_dir / "test_model_preferences.py"),
            str(test_dir / "test_embedding_templates.py"),
            str(test_dir / "test_integration.py"),
        ])
    elif args.suite == "toast":
        cmd.append(str(test_dir / "test_toast_notifications.py"))
    elif args.suite == "progress":
        cmd.append(str(test_dir / "test_detailed_progress.py"))
    elif args.suite == "preferences":
        cmd.append(str(test_dir / "test_model_preferences.py"))
    elif args.suite == "templates":
        cmd.append(str(test_dir / "test_embedding_templates.py"))
    elif args.suite == "activity":
        cmd.append(str(test_dir / "test_activity_log.py"))
    elif args.suite == "performance":
        cmd.append(str(test_dir / "test_performance_metrics.py"))
    elif args.suite == "coverage":
        # Run with coverage
        cmd.extend([
            "--cov=tldw_chatbook.UI",
            "--cov=tldw_chatbook.Widgets",
            "--cov=tldw_chatbook.Utils.model_preferences",
            "--cov=tldw_chatbook.Utils.embedding_templates",
            "--cov-report=html",
            "--cov-report=term-missing",
            str(test_dir)
        ])
        print("\nRunning tests with coverage analysis...")
        print("HTML report will be generated in htmlcov/")
    
    # Run the tests
    return_code = run_command(cmd)
    
    if return_code == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    return return_code


if __name__ == "__main__":
    sys.exit(main())