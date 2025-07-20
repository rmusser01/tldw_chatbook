#!/usr/bin/env python3
"""
Convenience script to run all RAG tests with detailed reporting
"""

import sys
import subprocess
import os

def main():
    """Run all RAG tests with detailed report"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Running all RAG tests with detailed reporting...")
    print("=" * 60)
    
    # Build command to run the main test runner with RAG modules
    cmd = [
        sys.executable,
        os.path.join(script_dir, "run_all_tests_with_report.py"),
        "--modules", "RAG_Simplified", "RAG_Legacy", "RAG_Other",
        "--rag-detailed",
        "--format", "console", "markdown"
    ]
    
    # Add any additional arguments passed to this script
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    # Run the command
    try:
        result = subprocess.run(cmd, cwd=script_dir)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nTest run interrupted!")
        sys.exit(1)
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()