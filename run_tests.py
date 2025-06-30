#!/usr/bin/env python3

import subprocess
import sys
import os

def run_tests():
    """Run pytest on specified test modules"""
    # Change to the project directory
    project_dir = "/Users/appledev/Working/tldw_chatbook_dev"
    os.chdir(project_dir)
    
    # Run pytest command
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "Tests/Character_Chat/",
        "Tests/Chat/",
        "Tests/Notes/",
        "Tests/Prompts_DB/",
        "-v",
        "--tb=short"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nReturn code: {result.returncode}")

if __name__ == "__main__":
    run_tests()