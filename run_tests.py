#!/usr/bin/env python3
"""
Local Test Runner for tldw_chatbook
Provides CI/CD-like functionality for running tests locally with reporting and tracking.

Usage:
    python run_tests.py [options]
    
Options:
    --mode: Test mode (quick, full, coverage, smoke) [default: quick]
    --parallel: Run tests in parallel [default: False]
    --html: Generate HTML report [default: True]
    --track: Track test results over time [default: True]
    --groups: Comma-separated test groups to run [default: all]
    --verbose: Verbose output [default: False]
"""

import argparse
import json
import os
import subprocess
import sys
import time
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import platform
import shutil

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class TestRunner:
    """Local test runner with CI/CD-like capabilities."""
    
    # Test groups matching the structure from Test-Review-22.md
    TEST_GROUPS = {
        "database": {
            "name": "Database Tests",
            "paths": ["Tests/ChaChaNotesDB/", "Tests/DB/", "Tests/Prompts_DB/", "Tests/Media_DB/"],
            "critical": True
        },
        "core": {
            "name": "Core Feature Tests", 
            "paths": ["Tests/Character_Chat/", "Tests/Chat/", "Tests/Notes/", "Tests/Evals/"],
            "critical": True
        },
        "integration": {
            "name": "Integration Tests",
            "paths": ["Tests/Event_Handlers/", "Tests/LLM_Management/", "Tests/RAG/", "Tests/RAG_Search/"],
            "critical": True
        },
        "ui": {
            "name": "UI and Widget Tests",
            "paths": ["Tests/UI/", "Tests/Widgets/"],
            "critical": False
        },
        "utility": {
            "name": "Utility Tests",
            "paths": ["Tests/Utils/", "Tests/Web_Scraping/", "Tests/tldw_api/", "Tests/integration/", "Tests/unit/"],
            "critical": False
        }
    }
    
    # Quick smoke tests for critical functionality
    SMOKE_TESTS = [
        "Tests/ChaChaNotesDB/test_chachanotes_db.py::TestChaChaNotesDatabaseOperations::test_db_creation_and_schema_version",
        "Tests/Chat/test_chat_functions.py::test_add_conversation",
        "Tests/Character_Chat/test_character_chat.py::TestCharacterOperations::test_add_character_basic",
        "Tests/Notes/test_notes_library_unit.py::TestNotesLibraryMocked::test_create_note",
        "Tests/DB/test_sql_validation.py::test_valid_identifiers",
    ]
    
    def __init__(self, mode: str = "quick", parallel: bool = False, 
                 html_report: bool = True, track_history: bool = True,
                 groups: Optional[List[str]] = None, verbose: bool = False):
        self.mode = mode
        self.parallel = parallel
        self.html_report = html_report
        self.track_history = track_history
        self.groups = groups or ["all"]
        self.verbose = verbose
        
        # Set up directories
        self.reports_dir = Path("test_reports")
        self.reports_dir.mkdir(exist_ok=True)
        self.history_file = self.reports_dir / "test_history.json"
        self.current_report_dir = self.reports_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_report_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.results: Dict[str, Dict] = {}
        self.start_time = time.time()
        
    def run(self) -> int:
        """Main entry point for test runner."""
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"{Colors.HEADER}tldw_chatbook Local Test Runner{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"\nMode: {Colors.OKCYAN}{self.mode}{Colors.ENDC}")
        print(f"Parallel: {Colors.OKCYAN}{self.parallel}{Colors.ENDC}")
        print(f"Groups: {Colors.OKCYAN}{', '.join(self.groups)}{Colors.ENDC}")
        print(f"Report Directory: {Colors.OKCYAN}{self.current_report_dir}{Colors.ENDC}\n")
        
        # Determine which tests to run
        if self.mode == "smoke":
            test_paths = self.SMOKE_TESTS
            print(f"Running {Colors.OKBLUE}smoke tests{Colors.ENDC} ({len(test_paths)} tests)...\n")
            self._run_smoke_tests(test_paths)
        else:
            groups_to_run = self._get_groups_to_run()
            print(f"Running {Colors.OKBLUE}{len(groups_to_run)} test groups{Colors.ENDC}...\n")
            
            if self.parallel and len(groups_to_run) > 1:
                self._run_parallel(groups_to_run)
            else:
                self._run_sequential(groups_to_run)
        
        # Generate reports
        self._generate_summary()
        
        if self.html_report:
            self._generate_html_report()
            
        if self.track_history:
            self._update_history()
            
        # Return exit code based on failures
        total_failed = sum(r.get("failed", 0) + r.get("errors", 0) for r in self.results.values())
        return 1 if total_failed > 0 else 0
    
    def _get_groups_to_run(self) -> Dict[str, Dict]:
        """Determine which test groups to run based on mode and groups filter."""
        if "all" in self.groups:
            groups = self.TEST_GROUPS
        else:
            groups = {k: v for k, v in self.TEST_GROUPS.items() if k in self.groups}
            
        if self.mode == "quick":
            # Quick mode only runs critical groups
            groups = {k: v for k, v in groups.items() if v.get("critical", False)}
            
        return groups
    
    def _run_smoke_tests(self, test_paths: List[str]):
        """Run smoke tests."""
        cmd = ["pytest"] + test_paths + ["-v", "--tb=short"]
        if self.mode == "coverage":
            cmd.extend(["--cov=tldw_chatbook", "--cov-report=html", "--cov-report=term"])
            
        result = self._run_pytest_command(cmd, "smoke")
        self.results["smoke"] = result
    
    def _run_sequential(self, groups: Dict[str, Dict]):
        """Run test groups sequentially."""
        for group_id, group_info in groups.items():
            print(f"\n{Colors.HEADER}Running {group_info['name']}...{Colors.ENDC}")
            result = self._run_test_group(group_id, group_info)
            self.results[group_id] = result
            self._print_group_result(group_id, result)
    
    def _run_parallel(self, groups: Dict[str, Dict]):
        """Run test groups in parallel."""
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(groups)) as executor:
            future_to_group = {
                executor.submit(self._run_test_group, group_id, group_info): (group_id, group_info)
                for group_id, group_info in groups.items()
            }
            
            for future in concurrent.futures.as_completed(future_to_group):
                group_id, group_info = future_to_group[future]
                try:
                    result = future.result()
                    self.results[group_id] = result
                    self._print_group_result(group_id, result)
                except Exception as exc:
                    print(f"{Colors.FAIL}Group {group_id} generated an exception: {exc}{Colors.ENDC}")
                    self.results[group_id] = {"error": str(exc)}
    
    def _run_test_group(self, group_id: str, group_info: Dict) -> Dict:
        """Run a single test group."""
        start_time = time.time()
        
        # Build pytest command
        cmd = ["pytest"] + group_info["paths"]
        
        if self.verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
            
        cmd.append("--tb=short")
        
        if self.mode == "coverage":
            cmd.extend([
                f"--cov=tldw_chatbook",
                f"--cov-report=html:{self.current_report_dir}/coverage_{group_id}",
                "--cov-report=term"
            ])
        
        # Add JSON report for parsing
        json_report = self.current_report_dir / f"{group_id}_report.json"
        cmd.extend(["--json-report", f"--json-report-file={json_report}"])
        
        result = self._run_pytest_command(cmd, group_id)
        result["duration"] = time.time() - start_time
        
        # Parse JSON report if available
        if json_report.exists():
            try:
                with open(json_report) as f:
                    json_data = json.load(f)
                    result.update(self._parse_json_report(json_data))
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not parse JSON report for {group_id}: {e}")
        
        return result
    
    def _run_pytest_command(self, cmd: List[str], group_id: str) -> Dict:
        """Run a pytest command and capture results."""
        # Capture output to file
        output_file = self.current_report_dir / f"{group_id}_output.txt"
        
        with open(output_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            output_lines = []
            for line in iter(process.stdout.readline, ""):
                f.write(line)
                output_lines.append(line.strip())
                if self.verbose:
                    print(line, end="")
            
            process.wait()
        
        # Parse output for results
        result = self._parse_pytest_output(output_lines)
        result["exit_code"] = process.returncode
        result["output_file"] = str(output_file)
        
        return result
    
    def _parse_pytest_output(self, lines: List[str]) -> Dict:
        """Parse pytest output to extract test results."""
        result = {
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "total": 0,
            "failures": [],
            "errors_list": []
        }
        
        for line in lines:
            # Look for summary line
            if "passed" in line or "failed" in line or "error" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed" and i > 0:
                        result["passed"] = int(parts[i-1])
                    elif part == "failed" and i > 0:
                        result["failed"] = int(parts[i-1])
                    elif part == "error" and i > 0:
                        result["errors"] = int(parts[i-1])
                    elif part == "skipped" and i > 0:
                        result["skipped"] = int(parts[i-1])
            
            # Capture failed test names
            if "FAILED" in line and "::" in line:
                test_name = line.split("FAILED")[1].strip().split()[0]
                result["failures"].append(test_name)
                
            # Capture error test names  
            if "ERROR" in line and "::" in line:
                test_name = line.split("ERROR")[1].strip().split()[0]
                result["errors_list"].append(test_name)
        
        result["total"] = result["passed"] + result["failed"] + result["errors"] + result["skipped"]
        return result
    
    def _parse_json_report(self, json_data: Dict) -> Dict:
        """Parse pytest JSON report for detailed information."""
        summary = json_data.get("summary", {})
        tests = json_data.get("tests", [])
        
        result = {
            "duration_detailed": summary.get("duration", 0),
            "detailed_tests": []
        }
        
        for test in tests:
            test_info = {
                "nodeid": test.get("nodeid"),
                "outcome": test.get("outcome"),
                "duration": test.get("duration", 0)
            }
            if test.get("call", {}).get("longrepr"):
                test_info["error"] = test["call"]["longrepr"]
            result["detailed_tests"].append(test_info)
            
        return result
    
    def _print_group_result(self, group_id: str, result: Dict):
        """Print results for a test group."""
        if "error" in result:
            print(f"{Colors.FAIL}✗ {group_id}: ERROR - {result['error']}{Colors.ENDC}")
            return
            
        total = result.get("total", 0)
        passed = result.get("passed", 0)
        failed = result.get("failed", 0)
        errors = result.get("errors", 0)
        skipped = result.get("skipped", 0)
        
        if failed > 0 or errors > 0:
            status_color = Colors.FAIL
            status_symbol = "✗"
        elif skipped == total:
            status_color = Colors.WARNING
            status_symbol = "⚠"
        else:
            status_color = Colors.OKGREEN
            status_symbol = "✓"
            
        print(f"{status_color}{status_symbol} {group_id}: "
              f"{passed}/{total} passed, "
              f"{failed} failed, "
              f"{errors} errors, "
              f"{skipped} skipped{Colors.ENDC}")
    
    def _generate_summary(self):
        """Generate and print test summary."""
        total_duration = time.time() - self.start_time
        
        # Calculate totals
        total_tests = sum(r.get("total", 0) for r in self.results.values())
        total_passed = sum(r.get("passed", 0) for r in self.results.values())
        total_failed = sum(r.get("failed", 0) for r in self.results.values())
        total_errors = sum(r.get("errors", 0) for r in self.results.values())
        total_skipped = sum(r.get("skipped", 0) for r in self.results.values())
        
        print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"{Colors.HEADER}Test Summary{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
        
        print(f"\nTotal Tests: {Colors.BOLD}{total_tests}{Colors.ENDC}")
        print(f"Passed: {Colors.OKGREEN}{total_passed}{Colors.ENDC}")
        print(f"Failed: {Colors.FAIL}{total_failed}{Colors.ENDC}")
        print(f"Errors: {Colors.FAIL}{total_errors}{Colors.ENDC}")
        print(f"Skipped: {Colors.WARNING}{total_skipped}{Colors.ENDC}")
        
        if total_tests > 0:
            pass_rate = (total_passed / (total_tests - total_skipped)) * 100 if (total_tests - total_skipped) > 0 else 0
            print(f"\nPass Rate: {Colors.BOLD}{pass_rate:.1f}%{Colors.ENDC}")
        
        print(f"Duration: {Colors.BOLD}{total_duration:.1f}s{Colors.ENDC}")
        
        # Print failures and errors
        if total_failed > 0 or total_errors > 0:
            print(f"\n{Colors.FAIL}Failed Tests:{Colors.ENDC}")
            for group_id, result in self.results.items():
                for failure in result.get("failures", []):
                    print(f"  - {failure}")
                for error in result.get("errors_list", []):
                    print(f"  - {error} (ERROR)")
        
        # Save summary to file
        summary_file = self.current_report_dir / "summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Test Run Summary\n")
            f.write(f"================\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Mode: {self.mode}\n")
            f.write(f"Total Tests: {total_tests}\n")
            f.write(f"Passed: {total_passed}\n")
            f.write(f"Failed: {total_failed}\n")
            f.write(f"Errors: {total_errors}\n")
            f.write(f"Skipped: {total_skipped}\n")
            f.write(f"Pass Rate: {pass_rate:.1f}%\n" if total_tests > 0 else "Pass Rate: N/A\n")
            f.write(f"Duration: {total_duration:.1f}s\n")
    
    def _generate_html_report(self):
        """Generate HTML report with test results."""
        html_file = self.current_report_dir / "report.html"
        
        # Calculate statistics
        total_tests = sum(r.get("total", 0) for r in self.results.values())
        total_passed = sum(r.get("passed", 0) for r in self.results.values())
        total_failed = sum(r.get("failed", 0) for r in self.results.values())
        total_errors = sum(r.get("errors", 0) for r in self.results.values())
        total_skipped = sum(r.get("skipped", 0) for r in self.results.values())
        pass_rate = (total_passed / (total_tests - total_skipped)) * 100 if (total_tests - total_skipped) > 0 else 0
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background-color: #333; color: white; padding: 20px; border-radius: 5px; }}
        .summary {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .group {{ background-color: white; padding: 15px; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .passed {{ color: #28a745; font-weight: bold; }}
        .failed {{ color: #dc3545; font-weight: bold; }}
        .skipped {{ color: #ffc107; font-weight: bold; }}
        .error {{ color: #dc3545; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        .progress-bar {{ width: 100%; height: 20px; background-color: #e0e0e0; border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background-color: #28a745; float: left; }}
        .progress-fail {{ height: 100%; background-color: #dc3545; float: left; }}
        .progress-skip {{ height: 100%; background-color: #ffc107; float: left; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>tldw_chatbook Test Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Mode: {self.mode} | Platform: {platform.system()} {platform.release()}</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {(total_passed/total_tests*100) if total_tests > 0 else 0:.1f}%"></div>
            <div class="progress-fail" style="width: {((total_failed+total_errors)/total_tests*100) if total_tests > 0 else 0:.1f}%"></div>
            <div class="progress-skip" style="width: {(total_skipped/total_tests*100) if total_tests > 0 else 0:.1f}%"></div>
        </div>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr><td>Total Tests</td><td>{total_tests}</td></tr>
            <tr><td>Passed</td><td class="passed">{total_passed}</td></tr>
            <tr><td>Failed</td><td class="failed">{total_failed}</td></tr>
            <tr><td>Errors</td><td class="error">{total_errors}</td></tr>
            <tr><td>Skipped</td><td class="skipped">{total_skipped}</td></tr>
            <tr><td>Pass Rate</td><td><strong>{pass_rate:.1f}%</strong></td></tr>
            <tr><td>Duration</td><td>{time.time() - self.start_time:.1f}s</td></tr>
        </table>
    </div>
    
    <h2>Test Groups</h2>
"""
        
        # Add results for each group
        for group_id, result in self.results.items():
            if "error" in result:
                status_class = "error"
                status_text = "ERROR"
            elif result.get("failed", 0) > 0 or result.get("errors", 0) > 0:
                status_class = "failed"
                status_text = "FAILED"
            else:
                status_class = "passed"
                status_text = "PASSED"
                
            group_name = self.TEST_GROUPS.get(group_id, {}).get("name", group_id)
            
            html_content += f"""
    <div class="group">
        <h3>{group_name} - <span class="{status_class}">{status_text}</span></h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr><td>Total</td><td>{result.get('total', 0)}</td></tr>
            <tr><td>Passed</td><td class="passed">{result.get('passed', 0)}</td></tr>
            <tr><td>Failed</td><td class="failed">{result.get('failed', 0)}</td></tr>
            <tr><td>Errors</td><td class="error">{result.get('errors', 0)}</td></tr>
            <tr><td>Skipped</td><td class="skipped">{result.get('skipped', 0)}</td></tr>
            <tr><td>Duration</td><td>{result.get('duration', 0):.1f}s</td></tr>
        </table>
"""
            
            # Add failure details
            failures = result.get("failures", []) + result.get("errors_list", [])
            if failures:
                html_content += "<h4>Failed Tests:</h4><ul>"
                for failure in failures:
                    html_content += f"<li><code>{failure}</code></li>"
                html_content += "</ul>"
                
            html_content += "</div>"
        
        html_content += """
</body>
</html>
"""
        
        with open(html_file, "w") as f:
            f.write(html_content)
            
        print(f"\n{Colors.OKGREEN}HTML report generated: {html_file}{Colors.ENDC}")
    
    def _update_history(self):
        """Update test history tracking."""
        # Load existing history
        history = []
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    history = json.load(f)
            except:
                pass
        
        # Add current run
        total_tests = sum(r.get("total", 0) for r in self.results.values())
        total_passed = sum(r.get("passed", 0) for r in self.results.values())
        total_failed = sum(r.get("failed", 0) for r in self.results.values())
        total_errors = sum(r.get("errors", 0) for r in self.results.values())
        total_skipped = sum(r.get("skipped", 0) for r in self.results.values())
        
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "mode": self.mode,
            "total": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "errors": total_errors,
            "skipped": total_skipped,
            "pass_rate": (total_passed / (total_tests - total_skipped)) * 100 if (total_tests - total_skipped) > 0 else 0,
            "duration": time.time() - self.start_time,
            "groups": list(self.results.keys())
        }
        
        history.append(history_entry)
        
        # Keep only last 100 entries
        history = history[-100:]
        
        # Save history
        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2)
            
        # Generate trend report if we have enough history
        if len(history) >= 5:
            self._generate_trend_report(history)
    
    def _generate_trend_report(self, history: List[Dict]):
        """Generate a trend report showing test health over time."""
        print(f"\n{Colors.HEADER}Test Trend (Last 5 Runs){Colors.ENDC}")
        print(f"{'Date':<20} {'Mode':<10} {'Pass Rate':<10} {'Duration':<10}")
        print("-" * 50)
        
        for entry in history[-5:]:
            date = datetime.fromisoformat(entry["timestamp"]).strftime("%Y-%m-%d %H:%M")
            mode = entry["mode"]
            pass_rate = entry["pass_rate"]
            duration = entry["duration"]
            
            # Color code pass rate
            if pass_rate >= 90:
                rate_color = Colors.OKGREEN
            elif pass_rate >= 70:
                rate_color = Colors.WARNING
            else:
                rate_color = Colors.FAIL
                
            print(f"{date:<20} {mode:<10} {rate_color}{pass_rate:>6.1f}%{Colors.ENDC}   {duration:>6.1f}s")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Local test runner for tldw_chatbook")
    parser.add_argument("--mode", choices=["quick", "full", "coverage", "smoke"], 
                        default="quick", help="Test mode")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML report generation")
    parser.add_argument("--no-track", action="store_true", help="Skip history tracking")
    parser.add_argument("--groups", help="Comma-separated test groups to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Parse groups
    groups = ["all"]
    if args.groups:
        groups = [g.strip() for g in args.groups.split(",")]
    
    # Check if pytest is installed
    if shutil.which("pytest") is None:
        print(f"{Colors.FAIL}Error: pytest is not installed. Please run: pip install pytest pytest-json-report{Colors.ENDC}")
        sys.exit(1)
    
    # Run tests
    runner = TestRunner(
        mode=args.mode,
        parallel=args.parallel,
        html_report=not args.no_html,
        track_history=not args.no_track,
        groups=groups,
        verbose=args.verbose
    )
    
    exit_code = runner.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()