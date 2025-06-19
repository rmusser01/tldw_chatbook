#!/usr/bin/env python3
"""Script to analyze test failures in the tldw_chatbook project."""

import subprocess
import json
import sys
from collections import defaultdict

def run_pytest_with_json():
    """Run pytest and capture results in JSON format."""
    cmd = [
        sys.executable, '-m', 'pytest', 
        'Tests/', 
        '--tb=short',
        '--json-report', 
        '--json-report-file=test_report.json',
        '--maxfail=999',
        '-q'
    ]
    
    print("Running pytest... This may take a few minutes.")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(f"\nPytest exit code: {result.returncode}")
    if result.stderr:
        print(f"Stderr output:\n{result.stderr[:1000]}")
    
    return result

def analyze_json_report():
    """Analyze the JSON test report."""
    try:
        with open('test_report.json', 'r') as f:
            data = json.load(f)
        
        summary = data.get('summary', {})
        tests = data.get('tests', [])
        
        # Count test outcomes
        outcomes = defaultdict(int)
        failed_tests = []
        error_tests = []
        
        for test in tests:
            outcome = test.get('outcome', 'unknown')
            outcomes[outcome] += 1
            
            if outcome == 'failed':
                failed_tests.append({
                    'nodeid': test.get('nodeid', 'Unknown'),
                    'call': test.get('call', {})
                })
            elif outcome == 'error':
                error_tests.append({
                    'nodeid': test.get('nodeid', 'Unknown'),
                    'setup': test.get('setup', {}),
                    'call': test.get('call', {})
                })
        
        # Print summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total tests collected: {summary.get('collected', 0)}")
        print(f"Duration: {summary.get('duration', 0):.2f} seconds")
        print("\nTest Outcomes:")
        for outcome, count in sorted(outcomes.items()):
            print(f"  {outcome}: {count}")
        
        # Analyze failure patterns
        print("\n" + "="*80)
        print("FAILURE ANALYSIS")
        print("="*80)
        
        # Group failures by test file
        failures_by_file = defaultdict(list)
        for test in failed_tests + error_tests:
            nodeid = test['nodeid']
            file_path = nodeid.split('::')[0]
            failures_by_file[file_path].append(test)
        
        print(f"\nFiles with failures: {len(failures_by_file)}")
        for file_path, tests in sorted(failures_by_file.items()):
            print(f"\n{file_path}: {len(tests)} failures")
            for test in tests[:3]:  # Show first 3 failures per file
                print(f"  - {test['nodeid'].split('::')[-1]}")
        
        # Common error patterns
        print("\n" + "="*80)
        print("COMMON ERROR PATTERNS")
        print("="*80)
        
        error_patterns = defaultdict(int)
        for test in failed_tests:
            if 'longrepr' in test.get('call', {}):
                error_msg = str(test['call']['longrepr'])
                if 'TypeError' in error_msg:
                    if 'missing 1 required positional argument' in error_msg:
                        error_patterns['Missing required argument'] += 1
                    else:
                        error_patterns['TypeError'] += 1
                elif 'AssertionError' in error_msg:
                    error_patterns['AssertionError'] += 1
                elif 'AttributeError' in error_msg:
                    error_patterns['AttributeError'] += 1
                elif 'ImportError' in error_msg:
                    error_patterns['ImportError'] += 1
                else:
                    error_patterns['Other'] += 1
        
        for pattern, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pattern}: {count}")
        
        return data
        
    except FileNotFoundError:
        print("test_report.json not found. Running tests without JSON report.")
        return None
    except json.JSONDecodeError:
        print("Failed to parse test_report.json")
        return None

def main():
    """Main function."""
    # First try to analyze existing report
    data = analyze_json_report()
    
    if not data:
        # Run pytest if no report exists
        result = run_pytest_with_json()
        
        # Try to analyze again
        data = analyze_json_report()
    
    if data:
        # Write detailed failure list
        with open('test_failures_detailed.txt', 'w') as f:
            f.write("DETAILED TEST FAILURES\n")
            f.write("=" * 80 + "\n\n")
            
            tests = data.get('tests', [])
            failed_count = 0
            
            for test in tests:
                if test.get('outcome') in ['failed', 'error']:
                    failed_count += 1
                    f.write(f"\n{failed_count}. {test.get('nodeid', 'Unknown')}\n")
                    f.write("-" * 40 + "\n")
                    
                    if 'call' in test and 'longrepr' in test['call']:
                        f.write(str(test['call']['longrepr'])[:500] + "\n")
                    elif 'setup' in test and 'longrepr' in test['setup']:
                        f.write(str(test['setup']['longrepr'])[:500] + "\n")
        
        print(f"\nDetailed failure report written to: test_failures_detailed.txt")

if __name__ == "__main__":
    main()