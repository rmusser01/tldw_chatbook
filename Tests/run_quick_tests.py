#!/usr/bin/env python3
"""
Quick test runner for common test scenarios
Provides shortcuts for frequently used test combinations
"""

import sys
import subprocess
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

def run_command(cmd: list, description: str):
    """Run a command with description"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0

def main():
    """Main entry point"""
    
    scenarios = {
        "smoke": {
            "description": "Run smoke tests only (quick validation)",
            "args": ["-m", "smoke", "--format", "console"]
        },
        "unit": {
            "description": "Run all unit tests",
            "args": ["-m", "unit", "--format", "console", "json"]
        },
        "integration": {
            "description": "Run integration tests",
            "args": ["-m", "integration", "--format", "console", "json"]
        },
        "chat": {
            "description": "Run Chat module tests only",
            "args": ["--modules", "Chat", "--format", "all"]
        },
        "db": {
            "description": "Run all database tests",
            "args": ["--modules", "Database", "--format", "all"]
        },
        "ui": {
            "description": "Run UI tests",
            "args": ["--modules", "UI", "Widgets", "--format", "console", "html"]
        },
        "rag": {
            "description": "Run RAG tests",
            "args": ["--modules", "RAG", "--format", "all"]
        },
        "quick": {
            "description": "Quick test run (unit tests, no slow tests)",
            "args": ["-m", "unit", "-m", "not slow", "-n", "4", "--format", "console"]
        },
        "full": {
            "description": "Full test suite with all reports",
            "args": ["--format", "all", "-v"]
        },
        "parallel": {
            "description": "Run all tests in parallel",
            "args": ["-n", "auto", "--format", "console", "json"]
        },
        "failed": {
            "description": "Re-run only failed tests from last run",
            "args": ["--lf", "--format", "console"]  # pytest's --last-failed option
        }
    }
    
    if len(sys.argv) < 2 or sys.argv[1] not in scenarios:
        print("Quick Test Runner - Common Test Scenarios")
        print("\nUsage: python run_quick_tests.py <scenario>")
        print("\nAvailable scenarios:")
        for name, config in scenarios.items():
            print(f"  {name:<12} - {config['description']}")
        print("\nExamples:")
        print("  python run_quick_tests.py smoke")
        print("  python run_quick_tests.py unit")
        print("  python run_quick_tests.py chat")
        sys.exit(1)
    
    scenario = sys.argv[1]
    config = scenarios[scenario]
    
    # Build command
    cmd = [sys.executable, "run_all_tests_with_report.py"] + config["args"]
    
    # Run the test scenario
    success = run_command(cmd, config["description"])
    
    if success:
        print(f"\n✅ {scenario} tests completed successfully!")
    else:
        print(f"\n❌ {scenario} tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()