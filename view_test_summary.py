#!/usr/bin/env python3
"""
View test summary from the latest test report
Provides a quick overview of test results without re-running tests
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Try to import rich for better formatting
try:
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

PROJECT_ROOT = Path(__file__).parent
REPORTS_DIR = PROJECT_ROOT / "Tests" / "test_reports"


def find_latest_report(format_type: str = "json") -> Optional[Path]:
    """Find the most recent test report file"""
    pattern = f"test_report_*{format_type}"
    reports = list(REPORTS_DIR.glob(pattern))
    
    if not reports:
        return None
    
    # Sort by modification time, newest first
    reports.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return reports[0]


def display_json_report(report_path: Path):
    """Display a JSON test report"""
    with open(report_path) as f:
        data = json.load(f)
    
    summary = data.get("summary", {})
    modules = data.get("modules", {})
    
    if RICH_AVAILABLE:
        console = Console()
        
        # Header
        console.print(f"\n[bold blue]Test Report Summary[/bold blue]")
        console.print(f"Report: {report_path.name}")
        console.print(f"Generated: {summary.get('timestamp', 'Unknown')}")
        console.print("")
        
        # Overall summary
        summary_table = Table(title="Overall Results")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")
        
        summary_table.add_row("Total Tests", str(summary.get('total_tests', 0)))
        summary_table.add_row("Passed", f"[green]{summary.get('total_passed', 0)}[/green]")
        summary_table.add_row("Failed", f"[red]{summary.get('total_failed', 0)}[/red]")
        summary_table.add_row("Skipped", f"[yellow]{summary.get('total_skipped', 0)}[/yellow]")
        summary_table.add_row("Success Rate", f"{summary.get('success_rate', 0)}%")
        summary_table.add_row("Duration", f"{summary.get('total_duration', 0)}s")
        
        console.print(summary_table)
        console.print("")
        
        # Module breakdown
        if summary.get('modules'):
            module_table = Table(title="Module Results")
            module_table.add_column("Module", style="cyan")
            module_table.add_column("Total", justify="right")
            module_table.add_column("Passed", justify="right", style="green")
            module_table.add_column("Failed", justify="right", style="red")
            module_table.add_column("Rate", justify="right")
            
            for module_name, module_data in summary['modules'].items():
                module_table.add_row(
                    module_name,
                    str(module_data.get('total', 0)),
                    str(module_data.get('passed', 0)),
                    str(module_data.get('failed', 0)),
                    f"{module_data.get('success_rate', 0)}%"
                )
            
            console.print(module_table)
        
        # Failed tests summary
        total_failed = summary.get('total_failed', 0)
        if total_failed > 0:
            console.print(f"\n[bold red]Failed Tests Summary:[/bold red]")
            failed_count = 0
            for module_name, module_data in modules.items():
                failed_tests = module_data.get('failed_tests', [])
                if failed_tests:
                    console.print(f"\n[yellow]{module_name}:[/yellow]")
                    for test in failed_tests[:5]:  # Show first 5 failed tests
                        console.print(f"  • {test.get('test', 'Unknown test')}")
                        failed_count += 1
            
            if failed_count < total_failed:
                console.print(f"\n... and {total_failed - failed_count} more failed tests")
    else:
        # Fallback to basic printing
        print(f"\nTest Report Summary")
        print(f"Report: {report_path.name}")
        print(f"Generated: {summary.get('timestamp', 'Unknown')}")
        print("="*60)
        
        print(f"\nOverall Results:")
        print(f"  Total Tests: {summary.get('total_tests', 0)}")
        print(f"  Passed: {summary.get('total_passed', 0)}")
        print(f"  Failed: {summary.get('total_failed', 0)}")
        print(f"  Skipped: {summary.get('total_skipped', 0)}")
        print(f"  Success Rate: {summary.get('success_rate', 0)}%")
        print(f"  Duration: {summary.get('total_duration', 0)}s")
        
        if summary.get('modules'):
            print(f"\nModule Results:")
            print(f"{'Module':<20} {'Total':>8} {'Passed':>8} {'Failed':>8} {'Rate':>8}")
            print("-"*60)
            
            for module_name, module_data in summary['modules'].items():
                print(f"{module_name:<20} {module_data.get('total', 0):>8} "
                      f"{module_data.get('passed', 0):>8} {module_data.get('failed', 0):>8} "
                      f"{module_data.get('success_rate', 0):>7}%")


def list_available_reports():
    """List all available test reports"""
    reports = list(REPORTS_DIR.glob("test_report_*"))
    
    if not reports:
        print("No test reports found.")
        print(f"Run tests first with: python run_all_tests_with_report.py")
        return
    
    # Sort by modification time, newest first
    reports.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if RICH_AVAILABLE:
        console = Console()
        table = Table(title="Available Test Reports")
        table.add_column("Report", style="cyan")
        table.add_column("Format", style="yellow")
        table.add_column("Modified", style="white")
        table.add_column("Size", justify="right")
        
        for report in reports[:20]:  # Show last 20 reports
            format_type = report.suffix[1:]  # Remove the dot
            modified = datetime.fromtimestamp(report.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            size = f"{report.stat().st_size / 1024:.1f} KB"
            table.add_row(report.name, format_type, modified, size)
        
        console.print(table)
    else:
        print("\nAvailable Test Reports:")
        print(f"{'Report':<50} {'Format':<10} {'Modified':<20}")
        print("-"*80)
        
        for report in reports[:20]:
            format_type = report.suffix[1:]
            modified = datetime.fromtimestamp(report.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"{report.name:<50} {format_type:<10} {modified:<20}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="View test report summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View latest test report
  python view_test_summary.py
  
  # List all available reports
  python view_test_summary.py --list
  
  # View a specific report
  python view_test_summary.py --report test_report_20231225_143022.json
"""
    )
    
    parser.add_argument("--list", "-l", action="store_true",
                       help="List all available test reports")
    parser.add_argument("--report", "-r", type=str,
                       help="View a specific report file")
    parser.add_argument("--format", "-f", choices=["json", "html", "md"],
                       default="json",
                       help="Report format to look for (default: json)")
    
    args = parser.parse_args()
    
    # Ensure reports directory exists
    if not REPORTS_DIR.exists():
        print(f"No test reports directory found at: {REPORTS_DIR}")
        print("Run tests first with: python run_all_tests_with_report.py")
        sys.exit(1)
    
    if args.list:
        list_available_reports()
    elif args.report:
        report_path = REPORTS_DIR / args.report
        if not report_path.exists():
            print(f"Report not found: {report_path}")
            print("Use --list to see available reports")
            sys.exit(1)
        
        if report_path.suffix == ".json":
            display_json_report(report_path)
        elif report_path.suffix == ".html":
            print(f"HTML report: {report_path}")
            print("Open this file in a web browser to view")
        elif report_path.suffix == ".md":
            print(f"Markdown report: {report_path}")
            with open(report_path) as f:
                print(f.read())
        else:
            print(f"Unknown report format: {report_path.suffix}")
    else:
        # Find and display latest report
        latest_report = find_latest_report(args.format)
        if latest_report:
            if args.format == "json":
                display_json_report(latest_report)
            else:
                print(f"Latest {args.format} report: {latest_report}")
                if args.format == "html":
                    print("Open this file in a web browser to view")
                elif args.format == "md":
                    with open(latest_report) as f:
                        print(f.read())
        else:
            print(f"No {args.format} test reports found.")
            print("Run tests first with: python run_all_tests_with_report.py")


if __name__ == "__main__":
    main()