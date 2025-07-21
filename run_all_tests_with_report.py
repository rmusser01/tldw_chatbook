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
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import xml.etree.ElementTree as ET
import os
import importlib.util

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

# Required test dependencies
REQUIRED_DEPENDENCIES = {
    "pytest": "Core test framework",
    "pytest-asyncio": "Async test support", 
    "pytest-xdist": "Parallel test execution",
    "pytest-timeout": "Test timeout handling",
    "pytest-mock": "Mocking support"
}

# Optional but recommended dependencies
OPTIONAL_DEPENDENCIES = {
    "hypothesis": "Property-based testing",
    "rich": "Better console output",
    "pytest-cov": "Coverage reporting"
}


def check_dependency(package_name: str) -> bool:
    """Check if a Python package is installed."""
    # Handle special cases for package imports
    import_name = package_name
    
    # Special mappings for packages where import name differs from package name
    special_mappings = {
        "pytest-asyncio": "pytest_asyncio",
        "pytest-xdist": "xdist",
        "pytest-timeout": "pytest_timeout",
        "pytest-mock": "pytest_mock",
        "pytest-cov": "pytest_cov"
    }
    
    if package_name in special_mappings:
        import_name = special_mappings[package_name]
    else:
        # Default: replace dashes with underscores
        import_name = package_name.replace("-", "_")
    
    try:
        spec = importlib.util.find_spec(import_name)
        return spec is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        # Some packages might not be directly importable, try pytest plugins
        if package_name.startswith("pytest-"):
            try:
                import pytest
                # Check if plugin is registered
                return package_name in str(pytest)
            except:
                pass
        return False


def check_dependencies() -> Tuple[Dict[str, bool], Dict[str, bool]]:
    """Check which dependencies are installed."""
    required_status = {}
    optional_status = {}
    
    for pkg, desc in REQUIRED_DEPENDENCIES.items():
        required_status[pkg] = check_dependency(pkg)
    
    for pkg, desc in OPTIONAL_DEPENDENCIES.items():
        optional_status[pkg] = check_dependency(pkg)
    
    return required_status, optional_status


def install_dependencies(packages: List[str], verbose: bool = False) -> bool:
    """Install Python packages using pip."""
    if not packages:
        return True
    
    cmd = [sys.executable, "-m", "pip", "install"] + packages
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=not verbose, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error installing packages: {e}")
        return False


def print_dependency_status(required_status: Dict[str, bool], optional_status: Dict[str, bool]):
    """Print dependency status in a formatted way."""
    if RICH_AVAILABLE:
        console = Console()
        
        # Required dependencies table
        req_table = Table(title="Required Dependencies", show_header=True)
        req_table.add_column("Package", style="cyan")
        req_table.add_column("Description", style="white")
        req_table.add_column("Status", style="white")
        
        for pkg, desc in REQUIRED_DEPENDENCIES.items():
            status = "[green]✓ Installed[/green]" if required_status[pkg] else "[red]✗ Missing[/red]"
            req_table.add_row(pkg, desc, status)
        
        console.print(req_table)
        console.print()
        
        # Optional dependencies table
        opt_table = Table(title="Optional Dependencies", show_header=True)
        opt_table.add_column("Package", style="cyan")
        opt_table.add_column("Description", style="white")
        opt_table.add_column("Status", style="white")
        
        for pkg, desc in OPTIONAL_DEPENDENCIES.items():
            status = "[green]✓ Installed[/green]" if optional_status[pkg] else "[yellow]○ Not installed[/yellow]"
            opt_table.add_row(pkg, desc, status)
        
        console.print(opt_table)
    else:
        print("\n=== Required Dependencies ===")
        for pkg, desc in REQUIRED_DEPENDENCIES.items():
            status = "✓ Installed" if required_status[pkg] else "✗ Missing"
            print(f"{pkg:<20} {desc:<30} {status}")
        
        print("\n=== Optional Dependencies ===")
        for pkg, desc in OPTIONAL_DEPENDENCIES.items():
            status = "✓ Installed" if optional_status[pkg] else "○ Not installed"
            print(f"{pkg:<20} {desc:<30} {status}")


def validate_environment() -> bool:
    """Validate the test environment and return True if ready to run tests."""
    print("Checking test environment...")
    
    # Check Python version
    if sys.version_info < (3, 11):
        print(f"Error: Python 3.11+ required, but you have {sys.version}")
        return False
    
    # Check if we're in the project directory
    if not TESTS_DIR.exists():
        print(f"Error: Tests directory not found at {TESTS_DIR}")
        print("Please run this script from the project root directory.")
        return False
    
    # Check dependencies
    required_status, optional_status = check_dependencies()
    
    # Check if all required dependencies are installed
    missing_required = [pkg for pkg, installed in required_status.items() if not installed]
    
    if missing_required:
        print("\n⚠️  Missing required dependencies:")
        for pkg in missing_required:
            print(f"  - {pkg}: {REQUIRED_DEPENDENCIES[pkg]}")
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing_required)}")
        print("\nOr run this script with --install-deps to install automatically.")
        return False
    
    return True


# Test module categories - map display names to actual test directories
# Updated to match actual directory structure
TEST_MODULES = {
    "Core": ["test_smoke.py", "unit/", "test_fts5_pattern.py", "test_model_capabilities.py"],
    "Chat": ["Chat/"],
    "Character_Chat": ["Character_Chat/"],
    "Database": ["DB/", "ChaChaNotesDB/", "Media_DB/", "Prompts_DB/"],
    "UI": ["UI/", "Widgets/"],
    "RAG_Simplified": ["RAG/simplified/"],
    "RAG_Legacy": ["RAG_Search/"],
    "RAG_Other": ["RAG/test_rag_dependencies.py", "RAG/test_rag_ui_integration.py", "test_enhanced_rag.py"],
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
    "Misc": ["test_autosave.py", "test_embeddings_datatable_fix.py", "test_enhanced_filepicker.py", 
            "test_minimal_world_book.py", "test_splash_fullscreen.py"],
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
        cmd.extend(["-v", "--tb=short", "--continue-on-collection-errors"])
        
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
        
        # Parse collection errors from stderr
        if result.stderr:
            collection_errors = self.parse_collection_errors(result.stderr)
            test_results["collection_errors"] = collection_errors
        
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
            "failures": [],
            "collection_errors": []
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
    
    def parse_collection_errors(self, stderr: str) -> List[Dict[str, str]]:
        """Parse collection errors from pytest stderr output"""
        errors = []
        lines = stderr.split('\n')
        
        # Look for import errors and collection failures
        import_error_pattern = r"ImportError: (.+)"
        module_not_found_pattern = r"ModuleNotFoundError: (.+)"
        collection_error_pattern = r"ERROR collecting (.+)"
        
        current_file = None
        current_error = []
        
        for line in lines:
            # Check for collection error start
            if "ERROR collecting" in line:
                # Save previous error if exists
                if current_file and current_error:
                    errors.append({
                        "file": current_file,
                        "error": "\n".join(current_error).strip()
                    })
                # Start new error
                match = re.search(collection_error_pattern, line)
                if match:
                    current_file = match.group(1).strip()
                    current_error = []
            # Check for import/module errors
            elif any(pattern in line for pattern in ["ImportError:", "ModuleNotFoundError:"]):
                current_error.append(line.strip())
            # Continue collecting error details
            elif current_file and line.strip() and not line.startswith("="):
                current_error.append(line.strip())
        
        # Don't forget the last error
        if current_file and current_error:
            errors.append({
                "file": current_file,
                "error": "\n".join(current_error).strip()
            })
        
        return errors
    
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
        missing_paths = []
        
        for pattern in test_patterns:
            test_path = TESTS_DIR / pattern
            if test_path.exists():
                test_paths.append(test_path)
            else:
                missing_paths.append(pattern)
        
        if missing_paths and self.verbose:
            print(f"  Warning: Missing test paths for {module_name}: {', '.join(missing_paths)}")
        
        if not test_paths:
            if self.verbose:
                print(f"  Skipping {module_name}: No test files found")
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
                "duration": 0,
                "tests": [],
                "failures": [],
                "skipped_reason": "No test files found"
            }
        
        # Run tests with XML output
        xml_file = REPORTS_DIR / f"{module_name}_junit.xml"
        
        try:
            results = self.run_pytest_with_junit(test_paths, xml_file)
        except Exception as e:
            if self.verbose:
                print(f"  Error running tests for {module_name}: {e}")
            results = {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 1,
                "duration": 0,
                "tests": [],
                "failures": [{"test": module_name, "message": str(e)}],
                "error_message": str(e)
            }
        
        # Clean up XML file
        try:
            if xml_file.exists():
                xml_file.unlink()
        except:
            pass
        
        return results


class ReportGenerator:
    """Generate test reports in various formats"""
    
    def __init__(self, results: Dict[str, Dict[str, Any]], rag_detailed: bool = False):
        self.results = results
        self.timestamp = datetime.now()
        self.rag_detailed = rag_detailed
        
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
            
            # Collection errors
            collection_error_count = 0
            for module_name, result in self.results.items():
                collection_errors = result.get("collection_errors", [])
                collection_error_count += len(collection_errors)
            
            if collection_error_count > 0:
                console.print("\n[bold yellow]Collection Errors:[/bold yellow]")
                console.print(f"[yellow]{collection_error_count} file(s) could not be imported[/yellow]")
                
                for module_name, result in self.results.items():
                    collection_errors = result.get("collection_errors", [])
                    if collection_errors:
                        console.print(f"\n[yellow]{module_name}:[/yellow]")
                        for error in collection_errors[:5]:  # Show first 5 errors per module
                            console.print(f"  • {error['file']}")
                            error_msg = error['error'].split('\n')[0] if '\n' in error['error'] else error['error']
                            console.print(f"    [dim]{error_msg}[/dim]")
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
    
    def print_rag_detailed_report(self):
        """Print detailed RAG test report"""
        if not RICH_AVAILABLE:
            print("\nDetailed RAG report requires 'rich' library")
            return
            
        console = Console()
        console.print("\n[bold blue]Detailed RAG Test Report[/bold blue]")
        console.print(f"Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        console.print("")
        
        # Collect RAG-specific results
        rag_modules = ['RAG_Simplified', 'RAG_Legacy', 'RAG_Other']
        rag_results = {k: v for k, v in self.results.items() if k in rag_modules}
        
        if not rag_results:
            console.print("[yellow]No RAG tests were found in the results[/yellow]")
            return
        
        # RAG Summary Table
        rag_table = Table(title="RAG Test Categories", show_header=True)
        rag_table.add_column("Category", style="cyan")
        rag_table.add_column("Description", style="white")
        rag_table.add_column("Total", justify="right")
        rag_table.add_column("Passed", justify="right", style="green")
        rag_table.add_column("Failed", justify="right", style="red")
        rag_table.add_column("Success Rate", justify="right")
        
        descriptions = {
            "RAG_Simplified": "New simplified RAG implementation (V2)",
            "RAG_Legacy": "Legacy RAG_Search tests",
            "RAG_Other": "Other RAG tests (enhanced, UI integration)"
        }
        
        total_rag_tests = 0
        total_rag_passed = 0
        total_rag_failed = 0
        
        for module_name in rag_modules:
            if module_name in rag_results:
                result = rag_results[module_name]
                total = result["total"]
                passed = result["passed"]
                failed = result["failed"] + result["errors"]
                success_rate = round(passed / total * 100, 2) if total > 0 else 0
                
                total_rag_tests += total
                total_rag_passed += passed
                total_rag_failed += failed
                
                rag_table.add_row(
                    module_name,
                    descriptions.get(module_name, ""),
                    str(total),
                    str(passed),
                    str(failed),
                    f"{success_rate}%"
                )
        
        console.print(rag_table)
        
        # Overall RAG statistics
        overall_success_rate = round(total_rag_passed / total_rag_tests * 100, 2) if total_rag_tests > 0 else 0
        console.print(f"\n[bold]Overall RAG Statistics:[/bold]")
        console.print(f"Total RAG Tests: {total_rag_tests}")
        console.print(f"Total Passed: [green]{total_rag_passed}[/green]")
        console.print(f"Total Failed: [red]{total_rag_failed}[/red]")
        console.print(f"Overall Success Rate: {overall_success_rate}%")
        
        # Show specific test file breakdown for simplified RAG
        if "RAG_Simplified" in rag_results:
            console.print("\n[bold]Simplified RAG Test Breakdown:[/bold]")
            simplified_tests = rag_results["RAG_Simplified"].get("tests", [])
            
            # Group by test file
            test_files = defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0})
            for test in simplified_tests:
                file_name = test.get("file", "unknown").split("/")[-1]
                status = test.get("status", "unknown")
                if status == "passed":
                    test_files[file_name]["passed"] += 1
                elif status == "failed" or status == "error":
                    test_files[file_name]["failed"] += 1
                elif status == "skipped":
                    test_files[file_name]["skipped"] += 1
            
            file_table = Table(show_header=True)
            file_table.add_column("Test File", style="cyan")
            file_table.add_column("Passed", justify="right", style="green")
            file_table.add_column("Failed", justify="right", style="red")
            file_table.add_column("Skipped", justify="right", style="yellow")
            
            for file_name, stats in sorted(test_files.items()):
                file_table.add_row(
                    file_name,
                    str(stats["passed"]),
                    str(stats["failed"]),
                    str(stats["skipped"])
                )
            
            console.print(file_table)
        
        # Show RAG-specific failures
        rag_failures = []
        for module_name in rag_modules:
            if module_name in rag_results:
                failures = rag_results[module_name].get("failures", [])
                for failure in failures:
                    failure["module"] = module_name
                    rag_failures.append(failure)
        
        if rag_failures:
            console.print("\n[bold red]RAG Test Failures:[/bold red]")
            
            # Group failures by error type
            error_types = defaultdict(list)
            for failure in rag_failures:
                message = failure.get("message", "")
                if "persist_directory" in message:
                    error_types["ChromaDB Configuration"].append(failure)
                elif "import" in message.lower():
                    error_types["Import Errors"].append(failure)
                elif "api" in message.lower():
                    error_types["API Changes"].append(failure)
                else:
                    error_types["Other"].append(failure)
            
            for error_type, failures in error_types.items():
                if failures:
                    console.print(f"\n[yellow]{error_type}:[/yellow]")
                    for failure in failures[:5]:  # Show first 5 of each type
                        console.print(f"  • [{failure['module']}] {failure['test']}")
                        if failure.get('message'):
                            console.print(f"    [dim]{failure['message'][:100]}...[/dim]")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run all tests with comprehensive reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check test environment without running tests
  python run_all_tests_with_report.py --check
  
  # Install missing dependencies and run tests
  python run_all_tests_with_report.py --install-deps
  
  # Setup complete test environment
  python run_all_tests_with_report.py --setup
  
  # Run all tests
  python run_all_tests_with_report.py
  
  # Run only unit tests
  python run_all_tests_with_report.py -m unit
  
  # Run specific modules
  python run_all_tests_with_report.py --modules Chat DB UI
  
  # Run all RAG tests with detailed report
  python run_all_tests_with_report.py --modules RAG_Simplified RAG_Legacy RAG_Other --rag-detailed
  
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
    parser.add_argument("--rag-detailed", action="store_true",
                       help="Generate detailed RAG test report with subcategories")
    parser.add_argument("--check", action="store_true",
                       help="Check test environment without running tests")
    parser.add_argument("--install-deps", action="store_true",
                       help="Automatically install missing test dependencies")
    parser.add_argument("--setup", action="store_true",
                       help="Setup complete test environment (install all dependencies)")
    
    args = parser.parse_args()
    
    # Ensure we're in the project root
    os.chdir(PROJECT_ROOT)
    
    # Handle --check command
    if args.check:
        print("Checking test environment...\n")
        required_status, optional_status = check_dependencies()
        print_dependency_status(required_status, optional_status)
        
        # Check if all required are installed
        missing_required = [pkg for pkg, installed in required_status.items() if not installed]
        if missing_required:
            print("\n❌ Test environment is not ready.")
            print(f"Missing required dependencies: {', '.join(missing_required)}")
            sys.exit(1)
        else:
            print("\n✅ Test environment is ready!")
            sys.exit(0)
    
    # Handle --setup command
    if args.setup:
        print("Setting up test environment...\n")
        
        # Install all required and optional dependencies
        all_deps = list(REQUIRED_DEPENDENCIES.keys()) + list(OPTIONAL_DEPENDENCIES.keys())
        print(f"Installing {len(all_deps)} packages: {', '.join(all_deps)}")
        
        if install_dependencies(all_deps, verbose=args.verbose):
            print("\n✅ Test environment setup complete!")
            sys.exit(0)
        else:
            print("\n❌ Failed to setup test environment.")
            sys.exit(1)
    
    # Handle --install-deps command
    if args.install_deps:
        print("Checking and installing missing dependencies...\n")
        required_status, optional_status = check_dependencies()
        
        # Find missing required dependencies
        missing_required = [pkg for pkg, installed in required_status.items() if not installed]
        
        if missing_required:
            print(f"Installing missing required dependencies: {', '.join(missing_required)}")
            if not install_dependencies(missing_required, verbose=args.verbose):
                print("\n❌ Failed to install dependencies.")
                sys.exit(1)
        
        # Optionally install recommended dependencies
        missing_optional = [pkg for pkg, installed in optional_status.items() if not installed]
        if missing_optional and not args.verbose:
            print(f"\nOptional dependencies not installed: {', '.join(missing_optional)}")
            print("Use --setup to install all dependencies including optional ones.")
    
    # Validate environment before running tests
    if not validate_environment():
        print("\n❌ Test environment validation failed.")
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
            if args.rag_detailed and any(m in results for m in ["RAG_Simplified", "RAG_Legacy", "RAG_Other"]):
                generator.print_rag_detailed_report()
        
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