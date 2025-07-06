#!/usr/bin/env python3
"""
Test Coverage Report Generator for tldw_chatbook
Generates comprehensive coverage reports with multiple output formats.

Usage:
    python generate_coverage.py [options]
    
Options:
    --format: Output format (html, xml, json, term) [default: html,term]
    --threshold: Minimum coverage threshold percentage [default: 70]
    --module: Specific module to analyze [default: all]
    --exclude: Comma-separated patterns to exclude
"""

import argparse
import subprocess
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

class CoverageReporter:
    """Generate and analyze test coverage reports."""
    
    # Modules to analyze
    MODULES = {
        "db": "tldw_chatbook/DB",
        "chat": "tldw_chatbook/Chat",
        "character": "tldw_chatbook/Character_Chat",
        "notes": "tldw_chatbook/Notes",
        "llm": "tldw_chatbook/LLM_Calls",
        "ui": "tldw_chatbook/UI",
        "widgets": "tldw_chatbook/Widgets",
        "rag": "tldw_chatbook/RAG_Search",
        "utils": "tldw_chatbook/Utils",
        "services": "tldw_chatbook/Services",
        "evals": "tldw_chatbook/Evals",
    }
    
    # Files to exclude from coverage
    DEFAULT_EXCLUDES = [
        "*/tests/*",
        "*/test_*",
        "*_test.py",
        "*/migrations/*",
        "*/__pycache__/*",
        "*/venv/*",
        "*/.venv/*",
        "setup.py",
        "*/Third_Party/*",
    ]
    
    def __init__(self, formats: List[str], threshold: float, 
                 module: Optional[str] = None, excludes: Optional[List[str]] = None):
        self.formats = formats
        self.threshold = threshold
        self.module = module
        self.excludes = self.DEFAULT_EXCLUDES + (excludes or [])
        
        # Set up output directory
        self.coverage_dir = Path("coverage_reports")
        self.coverage_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run(self) -> int:
        """Run coverage analysis."""
        print("=" * 80)
        print("tldw_chatbook Test Coverage Report")
        print("=" * 80)
        print(f"Timestamp: {self.timestamp}")
        print(f"Formats: {', '.join(self.formats)}")
        print(f"Threshold: {self.threshold}%")
        print(f"Module: {self.module or 'all'}")
        print()
        
        # Clean previous coverage data
        self._clean_coverage_data()
        
        # Determine what to test
        if self.module and self.module != "all":
            if self.module in self.MODULES:
                source_paths = [self.MODULES[self.module]]
                test_paths = [f"Tests/{self.module.title()}/"]
            else:
                print(f"Error: Unknown module '{self.module}'")
                print(f"Available modules: {', '.join(self.MODULES.keys())}")
                return 1
        else:
            source_paths = list(self.MODULES.values())
            test_paths = ["Tests/"]
        
        # Build coverage command
        cmd = self._build_coverage_command(source_paths, test_paths)
        
        print("Running tests with coverage...")
        print("-" * 80)
        
        # Run coverage
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Error running tests:")
            print(result.stderr)
            return result.returncode
        
        # Generate reports
        coverage_data = self._generate_reports()
        
        # Analyze results
        self._analyze_coverage(coverage_data)
        
        # Check threshold
        if coverage_data:
            total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
            if total_coverage < self.threshold:
                print(f"\n❌ Coverage {total_coverage:.1f}% is below threshold {self.threshold}%")
                return 1
            else:
                print(f"\n✅ Coverage {total_coverage:.1f}% meets threshold {self.threshold}%")
                return 0
        
        return 0
    
    def _clean_coverage_data(self):
        """Clean previous coverage data."""
        coverage_files = [".coverage", ".coverage.*"]
        for pattern in coverage_files:
            for file in Path(".").glob(pattern):
                file.unlink()
    
    def _build_coverage_command(self, source_paths: List[str], test_paths: List[str]) -> List[str]:
        """Build the coverage command."""
        cmd = [
            "coverage", "run",
            "--source", ",".join(source_paths),
            "--omit", ",".join(self.excludes),
            "-m", "pytest"
        ]
        
        cmd.extend(test_paths)
        cmd.extend(["-v", "--tb=short"])
        
        return cmd
    
    def _generate_reports(self) -> Dict:
        """Generate coverage reports in requested formats."""
        coverage_data = {}
        
        for fmt in self.formats:
            print(f"\nGenerating {fmt} report...")
            
            if fmt == "html":
                output_dir = self.coverage_dir / f"html_{self.timestamp}"
                subprocess.run([
                    "coverage", "html",
                    "-d", str(output_dir),
                    "--omit", ",".join(self.excludes)
                ])
                print(f"HTML report: {output_dir}/index.html")
                
            elif fmt == "xml":
                output_file = self.coverage_dir / f"coverage_{self.timestamp}.xml"
                subprocess.run([
                    "coverage", "xml",
                    "-o", str(output_file),
                    "--omit", ",".join(self.excludes)
                ])
                print(f"XML report: {output_file}")
                
            elif fmt == "json":
                output_file = self.coverage_dir / f"coverage_{self.timestamp}.json"
                subprocess.run([
                    "coverage", "json",
                    "-o", str(output_file),
                    "--omit", ",".join(self.excludes)
                ])
                print(f"JSON report: {output_file}")
                
                # Load JSON data for analysis
                with open(output_file) as f:
                    coverage_data = json.load(f)
                    
            elif fmt == "term":
                print("\nTerminal Report:")
                print("-" * 80)
                result = subprocess.run([
                    "coverage", "report",
                    "--omit", ",".join(self.excludes),
                    "--skip-covered",
                    "--show-missing"
                ], capture_output=True, text=True)
                print(result.stdout)
        
        # If no JSON report was generated, generate one for analysis
        if not coverage_data and "json" not in self.formats:
            temp_json = self.coverage_dir / "temp_coverage.json"
            subprocess.run([
                "coverage", "json",
                "-o", str(temp_json),
                "--omit", ",".join(self.excludes)
            ], capture_output=True)
            
            if temp_json.exists():
                with open(temp_json) as f:
                    coverage_data = json.load(f)
                temp_json.unlink()
        
        return coverage_data
    
    def _analyze_coverage(self, coverage_data: Dict):
        """Analyze coverage data and generate insights."""
        if not coverage_data:
            return
        
        print("\n" + "=" * 80)
        print("Coverage Analysis")
        print("=" * 80)
        
        # Overall statistics
        totals = coverage_data.get("totals", {})
        total_statements = totals.get("num_statements", 0)
        covered_statements = totals.get("covered_lines", 0)
        missing_statements = totals.get("missing_lines", 0)
        total_coverage = totals.get("percent_covered", 0)
        
        print(f"\nOverall Coverage: {total_coverage:.1f}%")
        print(f"Statements: {covered_statements}/{total_statements}")
        print(f"Missing: {missing_statements}")
        
        # Module breakdown
        print("\nModule Coverage:")
        print("-" * 60)
        print(f"{'Module':<40} {'Coverage':>10} {'Missing':>10}")
        print("-" * 60)
        
        files = coverage_data.get("files", {})
        module_stats = {}
        
        for file_path, file_data in files.items():
            # Determine module
            module_name = self._get_module_name(file_path)
            if module_name not in module_stats:
                module_stats[module_name] = {
                    "statements": 0,
                    "covered": 0,
                    "missing": 0,
                    "files": 0
                }
            
            summary = file_data.get("summary", {})
            module_stats[module_name]["statements"] += summary.get("num_statements", 0)
            module_stats[module_name]["covered"] += summary.get("covered_lines", 0)
            module_stats[module_name]["missing"] += summary.get("missing_lines", 0)
            module_stats[module_name]["files"] += 1
        
        # Sort and display module stats
        for module, stats in sorted(module_stats.items()):
            if stats["statements"] > 0:
                coverage = (stats["covered"] / stats["statements"]) * 100
                print(f"{module:<40} {coverage:>9.1f}% {stats['missing']:>10}")
        
        # Find least covered files
        print("\nLeast Covered Files (< 50%):")
        print("-" * 80)
        
        low_coverage_files = []
        for file_path, file_data in files.items():
            summary = file_data.get("summary", {})
            percent = summary.get("percent_covered", 0)
            if percent < 50 and summary.get("num_statements", 0) > 10:
                low_coverage_files.append((file_path, percent, summary.get("missing_lines", 0)))
        
        for file_path, percent, missing in sorted(low_coverage_files, key=lambda x: x[1])[:10]:
            short_path = self._shorten_path(file_path)
            print(f"{short_path:<60} {percent:>6.1f}% ({missing} missing)")
        
        # Generate coverage badge
        self._generate_badge(total_coverage)
    
    def _get_module_name(self, file_path: str) -> str:
        """Extract module name from file path."""
        parts = file_path.split("/")
        if "tldw_chatbook" in parts:
            idx = parts.index("tldw_chatbook")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return "other"
    
    def _shorten_path(self, file_path: str) -> str:
        """Shorten file path for display."""
        if "tldw_chatbook" in file_path:
            return "..." + file_path.split("tldw_chatbook")[-1]
        return file_path
    
    def _generate_badge(self, coverage: float):
        """Generate a coverage badge."""
        badge_file = self.coverage_dir / "coverage_badge.svg"
        
        # Determine color based on coverage
        if coverage >= 80:
            color = "#4c1"  # Green
        elif coverage >= 60:
            color = "#dfb317"  # Yellow
        else:
            color = "#e05d44"  # Red
        
        # Simple SVG badge
        badge_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="104" height="20">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <mask id="a">
    <rect width="104" height="20" rx="3" fill="#fff"/>
  </mask>
  <g mask="url(#a)">
    <path fill="#555" d="M0 0h63v20H0z"/>
    <path fill="{color}" d="M63 0h41v20H63z"/>
    <path fill="url(#b)" d="M0 0h104v20H0z"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="31.5" y="15" fill="#010101" fill-opacity=".3">coverage</text>
    <text x="31.5" y="14">coverage</text>
    <text x="82.5" y="15" fill="#010101" fill-opacity=".3">{coverage:.0f}%</text>
    <text x="82.5" y="14">{coverage:.0f}%</text>
  </g>
</svg>"""
        
        with open(badge_file, "w") as f:
            f.write(badge_svg)
        
        print(f"\nCoverage badge: {badge_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate test coverage reports")
    parser.add_argument("--format", default="html,term",
                        help="Output formats (comma-separated: html,xml,json,term)")
    parser.add_argument("--threshold", type=float, default=70.0,
                        help="Minimum coverage threshold percentage")
    parser.add_argument("--module", help="Specific module to analyze")
    parser.add_argument("--exclude", help="Comma-separated patterns to exclude")
    
    args = parser.parse_args()
    
    # Parse formats
    formats = [f.strip() for f in args.format.split(",")]
    valid_formats = ["html", "xml", "json", "term"]
    for fmt in formats:
        if fmt not in valid_formats:
            print(f"Error: Invalid format '{fmt}'. Valid formats: {', '.join(valid_formats)}")
            sys.exit(1)
    
    # Parse excludes
    excludes = []
    if args.exclude:
        excludes = [e.strip() for e in args.exclude.split(",")]
    
    # Check if coverage is installed
    if subprocess.run(["which", "coverage"], capture_output=True).returncode != 0:
        print("Error: coverage not installed. Please run: pip install coverage")
        sys.exit(1)
    
    # Run coverage analysis
    reporter = CoverageReporter(
        formats=formats,
        threshold=args.threshold,
        module=args.module,
        excludes=excludes
    )
    
    exit_code = reporter.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()