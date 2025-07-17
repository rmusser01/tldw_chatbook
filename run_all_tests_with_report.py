#!/usr/bin/env python3
"""
Comprehensive Test Runner with Detailed Reporting
Runs all tests in the project and generates detailed reports by module/test group
"""

import sys
import subprocess
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import xml.etree.ElementTree as ET
import os

# Try to import optional dependencies
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: 'rich' library not available. Install with: pip install rich")

# Project configuration
PROJECT_ROOT = Path(__file__).parent
TESTS_DIR = PROJECT_ROOT / "Tests"
REPORTS_DIR = TESTS_DIR / "test_reports"

# Test module categories - map display names to actual test directories
TEST_MODULES = {
    "Core": ["test_smoke.py", "unit/"],
    "Chat": ["Chat/"],
    "Character_Chat": ["Character_Chat/"],
    "Database": ["DB/", "ChaChaNotesDB/", "Media_DB/", "Prompts_DB/"],
    "UI": ["UI/", "Widgets/"],
    "RAG": ["RAG/", "RAG_Search/"],
    "Notes": ["Notes/"],
    "Event_Handlers": ["Event_Handlers/"],
    "Evals": ["Evals/"],
    "LLM_Management": ["LLM_Management/"],
    "Local_Ingestion": ["Local_Ingestion/"],
    "Transcription": ["Transcription/"],
    "Web_Scraping": ["Web_Scraping/"],
    "Utils": ["Utils/"],
    "Chunking": ["Chunking/"],
    "TTS": ["TTS/"],
    "API": ["tldw_api/"],
    "Integration": ["integration/"],
}


class TestRunner:
    """Main test runner class"""
    
    def __init__(self, verbose=False, parallel=None, markers=None, modules=None):
        self.verbose = verbose
        self.parallel = parallel
        self.markers = markers or []
        self.modules = modules or []
        self.console = Console() if RICH_AVAILABLE else None
        
    def run_pytest_with_junit(self, test_paths: List[Path], report_file: Path, extra_args: List[str] = None) -> Dict[str, Any]:
        """Run pytest with junit XML output for reliable parsing"""
        # Build pytest command
        cmd = [sys.executable, "-m", "pytest"]
        
        # Add junit XML output
        cmd.extend(["--junit-xml", str(report_file)])
        
        # Add standard options
        cmd.extend(["-v", "--tb=short"])
        
        # Add extra arguments
        if extra_args:
            cmd.extend(extra_args)
        
        # Add markers if specified
        if self.markers:
            marker_expr = " or ".join(self.markers)
            cmd.extend(["-m", marker_expr])
        
        # Add parallel execution if specified
        if self.parallel:
            cmd.extend(["-n", str(self.parallel)])
        
        # Add test paths
        for path in test_paths:
            cmd.append(str(path))
        
        if self.verbose:
            print(f"\nCommand: {' '.join(cmd)}")
        
        # Run pytest
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        duration = time.time() - start_time
        
        # Parse results from XML
        test_results = self.parse_junit_xml(report_file)
        test_results["duration"] = duration
        test_results["stdout"] = result.stdout
        test_results["stderr"] = result.stderr
        test_results["returncode"] = result.returncode
        
        return test_results
    
    def parse_junit_xml(self, xml_file: Path) -> Dict[str, Any]:
        """Parse JUnit XML to get detailed test results"""
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "tests": [],
            "failures": []
        }
        
        if not xml_file.exists():
            return results
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get overall statistics from testsuite
            for testsuite in root.findall(".//testsuite"):
                results["total"] += int(testsuite.get("tests", 0))
                results["errors"] += int(testsuite.get("errors", 0))
                results["failed"] += int(testsuite.get("failures", 0))
                results["skipped"] += int(testsuite.get("skipped", 0))
            
            # Calculate passed (total - failed - errors - skipped)
            results["passed"] = results["total"] - results["failed"] - results["errors"] - results["skipped"]
            
            # Get individual test details
            for testcase in root.findall(".//testcase"):
                test_info = {
                    "classname": testcase.get("classname", ""),
                    "name": testcase.get("name", ""),
                    "time": float(testcase.get("time", 0)),
                    "file": testcase.get("file", ""),
                    "line": testcase.get("line", ""),
                    "status": "passed"
                }
                
                # Check for failures
                failure = testcase.find("failure")
                if failure is not None:
                    test_info["status"] = "failed"
                    test_info["failure_message"] = failure.get("message", "")
                    test_info["failure_text"] = failure.text or ""
                    results["failures"].append({
                        "test": f"{test_info['classname']}::{test_info['name']}",
                        "message": test_info["failure_message"],
                        "details": test_info["failure_text"]
                    })
                
                # Check for errors
                error = testcase.find("error")
                if error is not None:
                    test_info["status"] = "error"
                    test_info["error_message"] = error.get("message", "")
                    test_info["error_text"] = error.text or ""
                
                # Check for skips
                skipped = testcase.find("skipped")
                if skipped is not None:
                    test_info["status"] = "skipped"
                    test_info["skip_message"] = skipped.get("message", "")
                
                results["tests"].append(test_info)
                
        except Exception as e:
            if self.verbose:
                print(f"Error parsing XML: {e}")
        
        return results
    
    def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run all tests organized by module"""
        all_results = {}
        
        # Create reports directory
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Filter modules if specified
        modules_to_run = TEST_MODULES
        if self.modules:
            modules_to_run = {k: v for k, v in TEST_MODULES.items() if k in self.modules}
        
        # Progress display
        if RICH_AVAILABLE and self.console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("Running tests...", total=len(modules_to_run))
                
                for module_name, test_patterns in modules_to_run.items():
                    progress.update(task, description=f"Testing {module_name}...")
                    all_results[module_name] = self.run_module_tests(module_name, test_patterns)
                    progress.advance(task)
        else:
            for module_name, test_patterns in modules_to_run.items():
                print(f"\nTesting {module_name}...")
                all_results[module_name] = self.run_module_tests(module_name, test_patterns)
        
        return all_results
    
    def run_module_tests(self, module_name: str, test_patterns: List[str]) -> Dict[str, Any]:
        """Run tests for a specific module"""
        # Collect test paths
        test_paths = []
        for pattern in test_patterns:
            test_path = TESTS_DIR / pattern
            if test_path.exists():
                test_paths.append(test_path)
        
        if not test_paths:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
                "duration": 0,
                "tests": [],
                "failures": []
            }
        
        # Run tests with XML output
        xml_file = REPORTS_DIR / f"{module_name}_junit.xml"
        results = self.run_pytest_with_junit(test_paths, xml_file)
        
        # Clean up XML file
        try:
            xml_file.unlink()
        except:
            pass
        
        return results


class ReportGenerator:
    """Generate test reports in various formats"""
    
    def __init__(self, results: Dict[str, Dict[str, Any]]):
        self.results = results
        self.timestamp = datetime.now()
        
    def generate_summary(self) -> Dict[str, Any]:
        """Generate overall summary statistics"""
        summary = {
            "timestamp": self.timestamp.isoformat(),
            "total_modules": len(self.results),
            "total_tests": 0,
            "total_passed": 0,
            "total_failed": 0,
            "total_skipped": 0,
            "total_errors": 0,
            "total_duration": 0.0,
            "success_rate": 0.0,
            "modules": {}
        }
        
        for module_name, result in self.results.items():
            summary["total_tests"] += result["total"]
            summary["total_passed"] += result["passed"]
            summary["total_failed"] += result["failed"]
            summary["total_skipped"] += result["skipped"]
            summary["total_errors"] += result["errors"]
            summary["total_duration"] += result.get("duration", 0)
            
            summary["modules"][module_name] = {
                "total": result["total"],
                "passed": result["passed"],
                "failed": result["failed"],
                "skipped": result["skipped"],
                "errors": result["errors"],
                "duration": round(result.get("duration", 0), 2),
                "success_rate": round(result["passed"] / result["total"] * 100, 2) if result["total"] > 0 else 0
            }
        
        if summary["total_tests"] > 0:
            summary["success_rate"] = round(summary["total_passed"] / summary["total_tests"] * 100, 2)
        
        summary["total_duration"] = round(summary["total_duration"], 2)
        
        return summary
    
    def print_console_report(self):
        """Print a formatted console report"""
        summary = self.generate_summary()
        
        if RICH_AVAILABLE:
            console = Console()
            
            # Header
            console.print("\n[bold blue]Test Execution Report[/bold blue]")
            console.print(f"Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            console.print("")
            
            # Summary statistics
            if summary["total_tests"] == 0:
                console.print("[yellow]No tests were run![/yellow]")
                return
            
            # Summary table
            summary_table = Table(title="Overall Summary", show_header=True)
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="white")
            
            summary_table.add_row("Total Tests", str(summary["total_tests"]))
            summary_table.add_row("Passed", f"[green]{summary['total_passed']}[/green]")
            summary_table.add_row("Failed", f"[red]{summary['total_failed']}[/red]")
            summary_table.add_row("Skipped", f"[yellow]{summary['total_skipped']}[/yellow]")
            summary_table.add_row("Errors", f"[red]{summary['total_errors']}[/red]")
            summary_table.add_row("Success Rate", f"{summary['success_rate']}%")
            summary_table.add_row("Total Duration", f"{summary['total_duration']}s")
            
            console.print(summary_table)
            console.print("")
            
            # Module details table
            module_table = Table(title="Module Breakdown", show_header=True)
            module_table.add_column("Module", style="cyan")
            module_table.add_column("Total", justify="right")
            module_table.add_column("Passed", justify="right", style="green")
            module_table.add_column("Failed", justify="right", style="red")
            module_table.add_column("Skipped", justify="right", style="yellow")
            module_table.add_column("Errors", justify="right", style="red")
            module_table.add_column("Success Rate", justify="right")
            module_table.add_column("Duration", justify="right")
            
            for module_name, module_data in summary["modules"].items():
                module_table.add_row(
                    module_name,
                    str(module_data["total"]),
                    str(module_data["passed"]),
                    str(module_data["failed"]),
                    str(module_data["skipped"]),
                    str(module_data["errors"]),
                    f"{module_data['success_rate']}%",
                    f"{module_data['duration']}s"
                )
            
            console.print(module_table)
            
            # Failed tests details
            if summary["total_failed"] > 0 or summary["total_errors"] > 0:
                console.print("\n[bold red]Failed Tests:[/bold red]")
                for module_name, result in self.results.items():
                    failures = result.get("failures", [])
                    if failures:
                        console.print(f"\n[yellow]{module_name}:[/yellow]")
                        for failure in failures[:10]:  # Show first 10 failures
                            console.print(f"  • {failure['test']}")
                            if failure.get('message'):
                                console.print(f"    [dim]{failure['message']}[/dim]")
        else:
            # Fallback to basic printing
            print("\n" + "="*80)
            print("TEST EXECUTION REPORT")
            print(f"Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            
            if summary["total_tests"] == 0:
                print("\nNo tests were run!")
                return
            
            print("\nOVERALL SUMMARY:")
            print(f"Total Tests: {summary['total_tests']}")
            print(f"Passed: {summary['total_passed']}")
            print(f"Failed: {summary['total_failed']}")
            print(f"Skipped: {summary['total_skipped']}")
            print(f"Errors: {summary['total_errors']}")
            print(f"Success Rate: {summary['success_rate']}%")
            print(f"Total Duration: {summary['total_duration']}s")
            
            print("\nMODULE BREAKDOWN:")
            print(f"{'Module':<20} {'Total':>8} {'Passed':>8} {'Failed':>8} {'Skipped':>8} {'Errors':>8} {'Rate':>8} {'Time':>8}")
            print("-"*90)
            
            for module_name, module_data in summary["modules"].items():
                print(f"{module_name:<20} {module_data['total']:>8} {module_data['passed']:>8} "
                      f"{module_data['failed']:>8} {module_data['skipped']:>8} {module_data['errors']:>8} "
                      f"{module_data['success_rate']:>7}% {module_data['duration']:>7}s")
    
    def save_json_report(self) -> Path:
        """Save detailed JSON report"""
        report_data = {
            "summary": self.generate_summary(),
            "modules": self.results
        }
        
        report_file = REPORTS_DIR / f"test_report_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\nJSON report saved to: {report_file}")
        return report_file
    
    def save_markdown_report(self) -> Path:
        """Save report in Markdown format"""
        summary = self.generate_summary()
        
        report_file = REPORTS_DIR / f"test_report_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Test Execution Report\n\n")
            f.write(f"Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if summary["total_tests"] == 0:
                f.write("**No tests were run!**\n")
                return report_file
            
            f.write("## Overall Summary\n\n")
            f.write(f"- **Total Tests**: {summary['total_tests']}\n")
            f.write(f"- **Passed**: {summary['total_passed']}\n")
            f.write(f"- **Failed**: {summary['total_failed']}\n")
            f.write(f"- **Skipped**: {summary['total_skipped']}\n")
            f.write(f"- **Errors**: {summary['total_errors']}\n")
            f.write(f"- **Success Rate**: {summary['success_rate']}%\n")
            f.write(f"- **Total Duration**: {summary['total_duration']}s\n\n")
            
            f.write("## Module Breakdown\n\n")
            f.write("| Module | Total | Passed | Failed | Skipped | Errors | Success Rate | Duration |\n")
            f.write("|--------|-------|--------|--------|---------|--------|--------------|----------|\n")
            
            for module_name, module_data in summary["modules"].items():
                f.write(f"| {module_name} | {module_data['total']} | {module_data['passed']} | "
                       f"{module_data['failed']} | {module_data['skipped']} | {module_data['errors']} | "
                       f"{module_data['success_rate']}% | {module_data['duration']}s |\n")
            
            # Failed tests
            if summary["total_failed"] > 0 or summary["total_errors"] > 0:
                f.write("\n## Failed Tests\n\n")
                for module_name, result in self.results.items():
                    failures = result.get("failures", [])
                    if failures:
                        f.write(f"### {module_name}\n\n")
                        for failure in failures:
                            f.write(f"- **{failure['test']}**\n")
                            if failure.get('message'):
                                f.write(f"  - {failure['message']}\n")
                            f.write("\n")
        
        print(f"Markdown report saved to: {report_file}")
        return report_file


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run all tests with comprehensive reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python run_all_tests_with_report.py
  
  # Run only unit tests
  python run_all_tests_with_report.py -m unit
  
  # Run specific modules
  python run_all_tests_with_report.py --modules Chat DB UI
  
  # Run TTS and Transcription tests
  python run_all_tests_with_report.py --modules TTS Transcription
  
  # Run with parallel execution
  python run_all_tests_with_report.py -n 4
  
  # Generate only specific report formats
  python run_all_tests_with_report.py --format console json
"""
    )
    
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("-n", "--parallel", type=int, metavar="N",
                       help="Run tests in parallel with N workers")
    parser.add_argument("-m", "--markers", nargs="+",
                       help="Run only tests with specific markers (e.g., unit, integration)")
    parser.add_argument("--modules", nargs="+", choices=list(TEST_MODULES.keys()),
                       help="Run only specific test modules")
    parser.add_argument("--format", nargs="+", 
                       choices=["console", "json", "markdown", "all"],
                       default=["console"],
                       help="Output format(s) for the report (default: console)")
    
    args = parser.parse_args()
    
    # Ensure we're in the project root
    os.chdir(PROJECT_ROOT)
    
    # Check if pytest is available
    try:
        subprocess.run([sys.executable, "-m", "pytest", "--version"], 
                      capture_output=True, check=True)
    except subprocess.CalledProcessError:
        print("Error: pytest is not installed!")
        print("Install with: pip install pytest pytest-xdist")
        sys.exit(1)
    
    # Create test runner
    runner = TestRunner(
        verbose=args.verbose,
        parallel=args.parallel,
        markers=args.markers,
        modules=args.modules
    )
    
    print("Starting test run...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Tests directory: {TESTS_DIR}")
    
    if args.markers:
        print(f"Markers: {', '.join(args.markers)}")
    if args.modules:
        print(f"Modules: {', '.join(args.modules)}")
    if args.parallel:
        print(f"Parallel workers: {args.parallel}")
    
    # Run tests
    start_time = time.time()
    try:
        results = runner.run_all_tests()
    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user!")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during test run: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    total_time = time.time() - start_time
    
    # Generate reports
    generator = ReportGenerator(results)
    
    formats = args.format
    if "all" in formats:
        formats = ["console", "json", "markdown"]
    
    try:
        if "console" in formats:
            generator.print_console_report()
        
        if "json" in formats:
            generator.save_json_report()
        
        if "markdown" in formats:
            generator.save_markdown_report()
    except Exception as e:
        print(f"\nError generating reports: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
    
    # Calculate overall success
    summary = generator.generate_summary()
    success = summary["total_failed"] == 0 and summary["total_errors"] == 0
    
    print(f"\n{'='*60}")
    print(f"Test run completed in {total_time:.2f} seconds")
    print(f"Overall result: {'PASSED' if success else 'FAILED'}")
    print(f"{'='*60}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()