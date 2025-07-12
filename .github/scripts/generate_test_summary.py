#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a test summary from pytest JSON reports.
"""

import json
import glob
import os
from pathlib import Path
from typing import Dict, List, Any


def load_json_reports() -> List[Dict[str, Any]]:
    """Load all JSON test reports."""
    reports = []
    for json_file in glob.glob("*-test-results*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                reports.append({
                    'name': json_file,
                    'data': data
                })
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    return reports


def generate_summary(reports: List[Dict[str, Any]]) -> str:
    """Generate a markdown summary from test reports."""
    summary = ["# Test Results Summary\n"]
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    total_duration = 0.0
    
    for report in reports:
        name = report['name']
        data = report['data']
        summary_data = data.get('summary', {})
        
        tests = summary_data.get('total', 0)
        passed = summary_data.get('passed', 0)
        failed = summary_data.get('failed', 0)
        skipped = summary_data.get('skipped', 0)
        duration = data.get('duration', 0.0)
        
        total_tests += tests
        total_passed += passed
        total_failed += failed
        total_skipped += skipped
        total_duration += duration
        
        # Extract test type from filename
        test_type = "Unknown"
        if "unit-test" in name:
            test_type = "Unit Tests"
        elif "integration-test" in name:
            test_type = "Integration Tests"
        elif "ui-test" in name:
            test_type = "UI Tests"
        elif "all-test" in name:
            test_type = "All Tests"
        
        # Extract platform/python version
        parts = name.replace(".json", "").split("-")
        details = " - ".join(parts[3:]) if len(parts) > 3 else ""
        
        status_emoji = "✅" if failed == 0 else "❌"
        
        summary.append(f"\n## {status_emoji} {test_type} {details}")
        summary.append(f"- **Total**: {tests}")
        summary.append(f"- **Passed**: {passed} ({passed/tests*100:.1f}%)" if tests > 0 else "- **Passed**: 0")
        summary.append(f"- **Failed**: {failed}")
        summary.append(f"- **Skipped**: {skipped}")
        summary.append(f"- **Duration**: {duration:.2f}s")
        
        # Add failure details if any
        if failed > 0 and 'tests' in data:
            summary.append("\n### Failed Tests:")
            for test in data['tests']:
                if test.get('outcome') == 'failed':
                    test_name = test.get('nodeid', 'Unknown test')
                    summary.append(f"- `{test_name}`")
                    if 'call' in test and 'longrepr' in test['call']:
                        # Add first line of error
                        error = test['call']['longrepr']
                        if isinstance(error, str):
                            first_line = error.split('\n')[0]
                            summary.append(f"  - {first_line}")
    
    # Overall summary
    summary.insert(1, f"\n## Overall Summary")
    summary.insert(2, f"- **Total Tests**: {total_tests}")
    summary.insert(3, f"- **Passed**: {total_passed} ({total_passed/total_tests*100:.1f}%)" if total_tests > 0 else "- **Passed**: 0")
    summary.insert(4, f"- **Failed**: {total_failed}")
    summary.insert(5, f"- **Skipped**: {total_skipped}")
    summary.insert(6, f"- **Total Duration**: {total_duration:.2f}s")
    
    # Add pass/fail badge
    if total_failed == 0:
        summary.insert(7, f"\n**Status**: ✅ All tests passed!")
    else:
        summary.insert(7, f"\n**Status**: ❌ {total_failed} tests failed")
    
    return "\n".join(summary)


def main():
    """Main function."""
    reports = load_json_reports()
    
    if not reports:
        print("No test reports found!")
        summary = "# Test Results Summary\n\nNo test reports were found."
    else:
        summary = generate_summary(reports)
    
    # Write summary
    with open("test-summary.md", "w") as f:
        f.write(summary)
    
    print("Test summary generated: test-summary.md")
    print(summary)


if __name__ == "__main__":
    main()