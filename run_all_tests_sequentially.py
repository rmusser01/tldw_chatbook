#!/usr/bin/env python3
"""
Sequential Test Runner for tldw_chatbook
Runs all tests directory by directory to avoid context overflow
"""

import os
import sys
import time
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

class TestResult:
    def __init__(self):
        self.directory = ""
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = 0
        self.duration = 0.0
        self.failures = []
        self.output = ""

class SequentialTestRunner:
    def __init__(self, base_path: Path, verbose: bool = False):
        self.base_path = base_path
        self.test_dir = base_path / "Tests"
        self.verbose = verbose
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    def get_test_directories(self) -> List[Tuple[str, Path]]:
        """Get all test directories in execution order"""
        # Define execution order based on dependencies
        ordered_dirs = [
            # Core/Utility Tests (foundational)
            "Utils",
            "DB",
            "Web_Scraping",
            
            # Database Layer Tests
            "ChaChaNotesDB",
            "Media_DB",
            "Prompts_DB",
            
            # Business Logic Tests
            "Character_Chat",
            "Chat",
            "Notes",
            "LLM_Management",
            "Evals",
            
            # Event Handling Tests
            "Event_Handlers",
            
            # UI/Widget Tests
            "Widgets",
            "UI",
            
            # Integration Tests
            "integration",
            "unit",
            "RAG_Search",
            "RAG",
        ]
        
        # Find actual directories
        test_dirs = []
        for dir_name in ordered_dirs:
            dir_path = self.test_dir / dir_name
            if dir_path.exists() and dir_path.is_dir():
                # Check if directory contains test files
                test_files = list(dir_path.glob("test_*.py"))
                if test_files:
                    test_dirs.append((dir_name, dir_path))
        
        # Add any directories not in ordered list
        for item in self.test_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                if item.name not in ordered_dirs:
                    test_files = list(item.glob("test_*.py"))
                    if test_files:
                        test_dirs.append((item.name, item))
                        
        return test_dirs
    
    def run_tests_for_directory(self, dir_name: str, dir_path: Path) -> TestResult:
        """Run tests for a single directory"""
        result = TestResult()
        result.directory = dir_name
        
        print(f"\n{'='*60}")
        print(f"Running tests for: {dir_name}")
        print(f"Path: {dir_path}")
        print(f"{'='*60}")
        
        start = time.time()
        
        # Construct pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            str(dir_path),
            "-v" if self.verbose else "-q",
            "--tb=short",
            "--no-header",
            f"--json-report-file={self.base_path}/test_report_{dir_name}.json",
            "--json-report"
        ]
        
        try:
            # Run pytest
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.base_path)
            )
            
            result.output = process.stdout + process.stderr
            result.duration = time.time() - start
            
            # Parse results from output
            self._parse_pytest_output(result, process.stdout, process.stderr)
            
            # Try to get detailed results from json report if available
            json_report_file = self.base_path / f"test_report_{dir_name}.json"
            if json_report_file.exists():
                try:
                    with open(json_report_file, 'r') as f:
                        json_report = json.load(f)
                        self._parse_json_report(result, json_report)
                    # Clean up json report
                    json_report_file.unlink()
                except Exception as e:
                    print(f"Warning: Could not parse JSON report: {e}")
            
            # Print summary for this directory
            self._print_directory_summary(result)
            
        except Exception as e:
            print(f"Error running tests for {dir_name}: {e}")
            result.errors = 1
            result.duration = time.time() - start
            
        return result
    
    def _parse_pytest_output(self, result: TestResult, stdout: str, stderr: str):
        """Parse pytest output to extract test counts"""
        # Look for summary line like "5 passed, 2 failed, 1 skipped"
        import re
        
        summary_pattern = r'(\d+)\s+passed|(\d+)\s+failed|(\d+)\s+skipped|(\d+)\s+error'
        
        for line in stdout.split('\n'):
            if 'passed' in line or 'failed' in line or 'skipped' in line:
                matches = re.findall(summary_pattern, line)
                for match in matches:
                    if match[0]:  # passed
                        result.passed = int(match[0])
                    elif match[1]:  # failed
                        result.failed = int(match[1])
                    elif match[2]:  # skipped
                        result.skipped = int(match[2])
                    elif match[3]:  # error
                        result.errors = int(match[3])
        
        # Extract failure details
        if result.failed > 0 or result.errors > 0:
            in_failure = False
            failure_text = []
            for line in stdout.split('\n'):
                if 'FAILED' in line or 'ERROR' in line:
                    if failure_text:
                        result.failures.append('\n'.join(failure_text))
                    failure_text = [line]
                    in_failure = True
                elif in_failure and line.strip():
                    failure_text.append(line)
                elif in_failure and not line.strip():
                    in_failure = False
            if failure_text:
                result.failures.append('\n'.join(failure_text))
    
    def _parse_json_report(self, result: TestResult, report: Dict):
        """Parse pytest-json-report output for detailed results"""
        if 'summary' in report:
            summary = report['summary']
            result.passed = summary.get('passed', 0)
            result.failed = summary.get('failed', 0)
            result.skipped = summary.get('skipped', 0)
            result.errors = summary.get('error', 0)
            
        # Extract failure details
        if 'tests' in report:
            for test in report['tests']:
                if test['outcome'] in ['failed', 'error']:
                    failure_info = f"{test['nodeid']}: {test.get('call', {}).get('longrepr', 'No details available')}"
                    result.failures.append(failure_info)
    
    def _print_directory_summary(self, result: TestResult):
        """Print summary for a single directory"""
        total = result.passed + result.failed + result.skipped + result.errors
        print(f"\nSummary for {result.directory}:")
        print(f"  Total tests: {total}")
        print(f"  Passed: {result.passed}")
        print(f"  Failed: {result.failed}")
        print(f"  Skipped: {result.skipped}")
        print(f"  Errors: {result.errors}")
        print(f"  Duration: {result.duration:.2f}s")
        
        if result.failures and self.verbose:
            print(f"\n  Failures:")
            for i, failure in enumerate(result.failures[:5]):  # Show first 5 failures
                print(f"    {i+1}. {failure.split(chr(10))[0]}")  # First line only
            if len(result.failures) > 5:
                print(f"    ... and {len(result.failures) - 5} more")
    
    def run_all_tests(self):
        """Run all tests sequentially"""
        self.start_time = datetime.now()
        print(f"Starting sequential test run at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Base path: {self.base_path}")
        print(f"Test directory: {self.test_dir}")
        
        test_dirs = self.get_test_directories()
        print(f"\nFound {len(test_dirs)} test directories to process")
        
        for dir_name, dir_path in test_dirs:
            result = self.run_tests_for_directory(dir_name, dir_path)
            self.results[dir_name] = result
            
            # Small delay between directories to allow cleanup
            time.sleep(0.5)
        
        self.end_time = datetime.now()
        self._generate_final_report()
    
    def _generate_final_report(self):
        """Generate comprehensive final report"""
        print(f"\n{'='*80}")
        print("FINAL TEST REPORT")
        print(f"{'='*80}")
        
        print(f"\nTest run completed at {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {(self.end_time - self.start_time).total_seconds():.2f}s")
        
        # Calculate totals
        total_passed = sum(r.passed for r in self.results.values())
        total_failed = sum(r.failed for r in self.results.values())
        total_skipped = sum(r.skipped for r in self.results.values())
        total_errors = sum(r.errors for r in self.results.values())
        total_tests = total_passed + total_failed + total_skipped + total_errors
        
        print(f"\nOVERALL SUMMARY:")
        print(f"  Total tests run: {total_tests}")
        print(f"  Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)" if total_tests > 0 else "  Passed: 0")
        print(f"  Failed: {total_failed}")
        print(f"  Skipped: {total_skipped}")
        print(f"  Errors: {total_errors}")
        
        # Directory breakdown
        print(f"\nDIRECTORY BREAKDOWN:")
        print(f"{'Directory':<20} {'Tests':<8} {'Pass':<8} {'Fail':<8} {'Skip':<8} {'Error':<8} {'Time':<10}")
        print("-" * 80)
        
        for dir_name in self.results:
            r = self.results[dir_name]
            total = r.passed + r.failed + r.skipped + r.errors
            print(f"{dir_name:<20} {total:<8} {r.passed:<8} {r.failed:<8} {r.skipped:<8} {r.errors:<8} {r.duration:<10.2f}s")
        
        # Failed directories
        failed_dirs = [d for d, r in self.results.items() if r.failed > 0 or r.errors > 0]
        if failed_dirs:
            print(f"\nDIRECTORIES WITH FAILURES:")
            for dir_name in failed_dirs:
                r = self.results[dir_name]
                print(f"\n  {dir_name}: {r.failed} failed, {r.errors} errors")
                if r.failures:
                    print("    Sample failures:")
                    for failure in r.failures[:3]:
                        lines = failure.split('\n')
                        print(f"      - {lines[0]}")
        
        # Write detailed report to file
        self._write_detailed_report()
        
        print(f"\nDetailed report written to: test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        # Exit code
        if total_failed > 0 or total_errors > 0:
            print(f"\nTEST RUN FAILED: {total_failed + total_errors} tests did not pass")
            return 1
        else:
            print(f"\nTEST RUN SUCCESSFUL: All {total_passed} tests passed!")
            return 0
    
    def _write_detailed_report(self):
        """Write detailed report to file"""
        report_file = self.base_path / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w') as f:
            f.write("TLDW_CHATBOOK TEST RESULTS\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total duration: {(self.end_time - self.start_time).total_seconds():.2f}s\n")
            f.write("="*80 + "\n\n")
            
            # Summary
            total_passed = sum(r.passed for r in self.results.values())
            total_failed = sum(r.failed for r in self.results.values())
            total_skipped = sum(r.skipped for r in self.results.values())
            total_errors = sum(r.errors for r in self.results.values())
            
            f.write("OVERALL SUMMARY\n")
            f.write(f"Total tests: {total_passed + total_failed + total_skipped + total_errors}\n")
            f.write(f"Passed: {total_passed}\n")
            f.write(f"Failed: {total_failed}\n")
            f.write(f"Skipped: {total_skipped}\n")
            f.write(f"Errors: {total_errors}\n\n")
            
            # Detailed results per directory
            f.write("DETAILED RESULTS BY DIRECTORY\n")
            f.write("="*80 + "\n\n")
            
            for dir_name, result in self.results.items():
                f.write(f"Directory: {dir_name}\n")
                f.write(f"Duration: {result.duration:.2f}s\n")
                f.write(f"Passed: {result.passed}, Failed: {result.failed}, Skipped: {result.skipped}, Errors: {result.errors}\n")
                
                if result.failures:
                    f.write(f"\nFailures:\n")
                    for i, failure in enumerate(result.failures):
                        f.write(f"\n{i+1}. {failure}\n")
                        f.write("-"*40 + "\n")
                
                f.write("\n" + "="*80 + "\n\n")
            
            # Analysis section
            f.write("ANALYSIS AND RECOMMENDATIONS\n")
            f.write("="*80 + "\n\n")
            
            # Common failure patterns
            if total_failed > 0 or total_errors > 0:
                f.write("Common Failure Patterns:\n")
                failure_keywords = {}
                for result in self.results.values():
                    for failure in result.failures:
                        # Extract common keywords
                        if "ImportError" in failure:
                            failure_keywords["Import Errors"] = failure_keywords.get("Import Errors", 0) + 1
                        if "AssertionError" in failure:
                            failure_keywords["Assertion Failures"] = failure_keywords.get("Assertion Failures", 0) + 1
                        if "AttributeError" in failure:
                            failure_keywords["Attribute Errors"] = failure_keywords.get("Attribute Errors", 0) + 1
                        if "KeyError" in failure:
                            failure_keywords["Key Errors"] = failure_keywords.get("Key Errors", 0) + 1
                        if "Database" in failure or "sqlite" in failure:
                            failure_keywords["Database Issues"] = failure_keywords.get("Database Issues", 0) + 1
                
                for pattern, count in sorted(failure_keywords.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  - {pattern}: {count} occurrences\n")
                
                f.write("\nRecommendations:\n")
                if "Import Errors" in failure_keywords:
                    f.write("  - Check that all dependencies are installed correctly\n")
                    f.write("  - Verify PYTHONPATH includes the project root\n")
                if "Database Issues" in failure_keywords:
                    f.write("  - Ensure test databases are properly initialized\n")
                    f.write("  - Check for database migration issues\n")

def main():
    parser = argparse.ArgumentParser(description="Run tldw_chatbook tests sequentially")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--path", type=Path, default=Path.cwd(), help="Base path of the project")
    
    args = parser.parse_args()
    
    runner = SequentialTestRunner(args.path, args.verbose)
    exit_code = runner.run_all_tests()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()