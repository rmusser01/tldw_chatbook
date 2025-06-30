#!/usr/bin/env python3

import subprocess
import sys
import os
import time
from datetime import datetime

def run_tests():
    """Run pytest on specified test modules and capture results"""
    start_time = time.time()
    
    # Change to the project directory
    project_dir = "/Users/appledev/Working/tldw_chatbook_dev"
    os.chdir(project_dir)
    
    # Test modules to run
    test_modules = [
        "Tests/Character_Chat/",
        "Tests/Chat/",
        "Tests/Notes/",
        "Tests/Prompts_DB/"
    ]
    
    print(f"Starting test execution at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print("-" * 80)
    
    # Run pytest command
    cmd = [
        sys.executable,
        "-m",
        "pytest"
    ] + test_modules + [
        "-v",
        "--tb=short",
        "--no-header",
        "-q"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)#, timeout=300) requires pytest-timeout
        
        # Process output
        output_lines = result.stdout.splitlines()
        error_lines = result.stderr.splitlines()
        
        # Parse test results
        total_tests = 0
        passed = 0
        failed = 0
        skipped = 0
        errors = 0
        
        # Module-specific counts
        module_results = {
            "Character_Chat": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0},
            "Chat": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0},
            "Notes": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0},
            "Prompts_DB": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0}
        }
        
        # Parse individual test results
        for line in output_lines:
            if " PASSED" in line or " FAILED" in line or " SKIPPED" in line or " ERROR" in line:
                # Determine module
                for module in module_results:
                    if f"Tests/{module}/" in line:
                        module_results[module]["total"] += 1
                        if " PASSED" in line:
                            module_results[module]["passed"] += 1
                            passed += 1
                        elif " FAILED" in line:
                            module_results[module]["failed"] += 1
                            failed += 1
                        elif " SKIPPED" in line:
                            module_results[module]["skipped"] += 1
                            skipped += 1
                        elif " ERROR" in line:
                            module_results[module]["errors"] += 1
                            errors += 1
                        break
                total_tests += 1
        
        # Extract summary from output
        summary_line = None
        for line in output_lines:
            if "passed" in line and ("failed" in line or "error" in line or "skipped" in line):
                summary_line = line
                break
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Print results
        print("\n" + "=" * 80)
        print("TEST EXECUTION REPORT")
        print("=" * 80)
        print(f"\nExecution Time: {execution_time:.2f} seconds")
        print(f"Python Version: {sys.version.split()[0]}")
        print(f"Return Code: {result.returncode}")
        
        print("\n" + "-" * 80)
        print("MODULE-WISE RESULTS:")
        print("-" * 80)
        print(f"{'Module':<20} {'Total':<10} {'Passed':<10} {'Failed':<10} {'Skipped':<10} {'Errors':<10}")
        print("-" * 80)
        
        for module, counts in module_results.items():
            print(f"{module:<20} {counts['total']:<10} {counts['passed']:<10} {counts['failed']:<10} {counts['skipped']:<10} {counts['errors']:<10}")
        
        print("-" * 80)
        print(f"{'TOTAL':<20} {total_tests:<10} {passed:<10} {failed:<10} {skipped:<10} {errors:<10}")
        
        print("\n" + "-" * 80)
        print("KEY FAILURE PATTERNS:")
        print("-" * 80)
        
        # Extract failure patterns
        failures = []
        current_failure = None
        for line in output_lines:
            if " FAILED " in line:
                if current_failure:
                    failures.append(current_failure)
                current_failure = {"test": line.strip(), "reason": ""}
            elif current_failure and "E   " in line:
                current_failure["reason"] = line.strip()
        
        if current_failure:
            failures.append(current_failure)
        
        # Group failures by pattern
        failure_patterns = {}
        for failure in failures:
            reason = failure["reason"]
            if reason:
                # Extract key part of error
                if "ImportError" in reason:
                    pattern = "ImportError"
                elif "AttributeError" in reason:
                    pattern = "AttributeError"
                elif "TypeError" in reason:
                    pattern = "TypeError"
                elif "AssertionError" in reason:
                    pattern = "AssertionError"
                elif "fixture" in reason and "not found" in reason:
                    pattern = "Missing Fixture"
                else:
                    pattern = "Other"
                
                if pattern not in failure_patterns:
                    failure_patterns[pattern] = []
                failure_patterns[pattern].append(failure["test"])
        
        if failure_patterns:
            for pattern, tests in failure_patterns.items():
                print(f"\n{pattern} ({len(tests)} occurrences):")
                for test in tests[:3]:  # Show first 3 examples
                    print(f"  - {test}")
                if len(tests) > 3:
                    print(f"  ... and {len(tests) - 3} more")
        else:
            print("No failures detected!")
        
        print("\n" + "-" * 80)
        print("SUMMARY:")
        print("-" * 80)
        if summary_line:
            print(summary_line)
        
        # Print raw output for debugging
        if result.returncode != 0:
            print("\n" + "-" * 80)
            print("RAW OUTPUT (last 50 lines):")
            print("-" * 80)
            for line in output_lines[-50:]:
                print(line)
        
        if error_lines:
            print("\n" + "-" * 80)
            print("STDERR OUTPUT:")
            print("-" * 80)
            for line in error_lines[:20]:  # First 20 lines of stderr
                print(line)
            
    except subprocess.TimeoutExpired:
        print("ERROR: Test execution timed out after 5 minutes")
    except Exception as e:
        print(f"ERROR: Failed to run tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_tests()