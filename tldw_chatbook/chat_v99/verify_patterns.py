"""Verification script to ensure all Textual patterns are correctly followed.

This script checks the implementation against the patterns specified in
the rebuild strategy document.
"""

import ast
import os
from pathlib import Path
from typing import List, Tuple, Dict


class TextualPatternChecker(ast.NodeVisitor):
    """AST visitor to check for Textual pattern violations."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.violations = []
        self.reactive_attrs = []
        self.css_usage = []
        self.worker_methods = []
        self.direct_manipulations = []
    
    def visit_Assign(self, node):
        """Check for reactive attribute definitions."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Check for reactive attributes
                if isinstance(node.value, ast.Call):
                    if hasattr(node.value.func, 'id') and node.value.func.id == 'reactive':
                        # Check if it has type annotation
                        has_type = False
                        if isinstance(target, ast.Name) and target.id in self.get_annotations():
                            has_type = True
                        
                        self.reactive_attrs.append({
                            'name': target.id if isinstance(target, ast.Name) else str(target),
                            'has_type': has_type,
                            'line': node.lineno
                        })
                        
                        if not has_type:
                            self.violations.append(
                                f"Line {node.lineno}: Reactive attribute '{target.id}' missing type hint"
                            )
        
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Check function definitions for patterns."""
        # Check for watch methods
        if node.name.startswith('watch_'):
            # Watch methods should have 2 parameters (self, old, new)
            if len(node.args.args) != 3:
                self.violations.append(
                    f"Line {node.lineno}: Watch method '{node.name}' should have (self, old, new) parameters"
                )
        
        # Check for worker methods
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == 'work':
                self.worker_methods.append(node.name)
                # Check for return statements in workers
                for child in ast.walk(node):
                    if isinstance(child, ast.Return):
                        if child.value is not None:
                            self.violations.append(
                                f"Line {node.lineno}: Worker method '{node.name}' should not return values"
                            )
        
        # Check for direct widget manipulation
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute):
                if isinstance(child.value, ast.Call):
                    if hasattr(child.value.func, 'attr') and child.value.func.attr == 'query_one':
                        # Found widget query, check for direct manipulation
                        parent = self.get_parent_node(child)
                        if isinstance(parent, ast.Assign):
                            self.direct_manipulations.append({
                                'line': child.lineno,
                                'method': node.name
                            })
        
        self.generic_visit(node)
    
    def get_annotations(self):
        """Get all type annotations in the file."""
        # This is a simplified version - in production would parse properly
        return []
    
    def get_parent_node(self, node):
        """Get parent node (simplified)."""
        return None


def check_file(filepath: Path) -> Dict:
    """Check a single file for pattern violations.
    
    Args:
        filepath: Path to the Python file
        
    Returns:
        Dictionary with checking results
    """
    with open(filepath, 'r') as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError as e:
            return {'error': str(e), 'violations': []}
    
    checker = TextualPatternChecker(str(filepath))
    checker.visit(tree)
    
    return {
        'filepath': str(filepath),
        'violations': checker.violations,
        'reactive_attrs': checker.reactive_attrs,
        'worker_methods': checker.worker_methods,
        'direct_manipulations': checker.direct_manipulations
    }


def check_css_patterns(directory: Path) -> List[str]:
    """Check CSS patterns in the directory.
    
    Args:
        directory: Directory to check
        
    Returns:
        List of violations
    """
    violations = []
    
    # Check for CSS files in subdirectories (violation)
    for css_file in directory.rglob('*.tcss'):
        # CSS files should be in same directory as Python files or inline
        parent = css_file.parent
        if parent != directory and 'styles' in str(parent):
            violations.append(
                f"CSS file in subdirectory (violation): {css_file}"
            )
    
    # Check Python files for CSS usage
    for py_file in directory.rglob('*.py'):
        with open(py_file, 'r') as f:
            content = f.read()
            
            # Check for CSS_PATH with subdirectories
            if 'CSS_PATH' in content:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'CSS_PATH' in line and '/' in line:
                        violations.append(
                            f"{py_file}:{i+1} - CSS_PATH references subdirectory"
                        )
            
            # Check for inline CSS (good pattern)
            if 'CSS = """' in content or "CSS = '''" in content:
                # This is good - inline CSS
                pass
    
    return violations


def verify_all_patterns() -> Dict:
    """Verify all Textual patterns in the chat_v99 implementation.
    
    Returns:
        Dictionary with all verification results
    """
    base_dir = Path(__file__).parent
    results = {
        'files_checked': [],
        'total_violations': 0,
        'pattern_violations': [],
        'css_violations': [],
        'summary': {}
    }
    
    # Check all Python files
    for py_file in base_dir.rglob('*.py'):
        if '__pycache__' not in str(py_file) and 'verify_patterns.py' not in str(py_file):
            file_results = check_file(py_file)
            results['files_checked'].append(file_results)
            results['total_violations'] += len(file_results.get('violations', []))
            results['pattern_violations'].extend(file_results.get('violations', []))
    
    # Check CSS patterns
    css_violations = check_css_patterns(base_dir)
    results['css_violations'] = css_violations
    results['total_violations'] += len(css_violations)
    
    # Additional specific checks
    specific_checks = perform_specific_checks(base_dir)
    results['specific_checks'] = specific_checks
    
    # Generate summary
    results['summary'] = {
        'total_files': len(results['files_checked']),
        'files_with_violations': len([f for f in results['files_checked'] if f.get('violations')]),
        'total_violations': results['total_violations'],
        'css_violations': len(results['css_violations']),
        'passed': results['total_violations'] == 0
    }
    
    return results


def perform_specific_checks(directory: Path) -> Dict:
    """Perform specific pattern checks.
    
    Args:
        directory: Directory to check
        
    Returns:
        Dictionary with specific check results
    """
    checks = {
        'app_pushes_screen': False,
        'reactive_types_used': False,
        'workers_use_callbacks': False,
        'no_direct_manipulation': True,
        'css_inline_or_same_dir': True
    }
    
    # Check app.py
    app_file = directory / 'app.py'
    if app_file.exists():
        with open(app_file, 'r') as f:
            content = f.read()
            
            # Check for push_screen in on_mount
            if 'push_screen' in content and 'on_mount' in content:
                checks['app_pushes_screen'] = True
            
            # Check for reactive with type hints
            if 'reactive[' in content:
                checks['reactive_types_used'] = True
    
    # Check for worker callbacks
    for py_file in directory.rglob('*.py'):
        with open(py_file, 'r') as f:
            content = f.read()
            if 'call_from_thread' in content and '@work' in content:
                checks['workers_use_callbacks'] = True
                break
    
    # Check for direct manipulation (violation)
    for py_file in directory.rglob('*.py'):
        with open(py_file, 'r') as f:
            content = f.read()
            # Look for patterns like widget.property = value after query
            if '.query' in content:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if '.query' in line and i + 1 < len(lines):
                        next_line = lines[i + 1]
                        # Simplified check - in production would be more sophisticated
                        if '.' in next_line and '=' in next_line and 'reactive' not in next_line:
                            checks['no_direct_manipulation'] = False
                            break
    
    return checks


def print_verification_report(results: Dict):
    """Print a formatted verification report.
    
    Args:
        results: Verification results dictionary
    """
    print("\n" + "="*60)
    print("TEXTUAL PATTERNS VERIFICATION REPORT")
    print("="*60 + "\n")
    
    # Summary
    summary = results['summary']
    status = "✅ PASSED" if summary['passed'] else "❌ FAILED"
    print(f"Status: {status}")
    print(f"Files Checked: {summary['total_files']}")
    print(f"Total Violations: {summary['total_violations']}")
    
    # Specific checks
    print("\n" + "-"*40)
    print("SPECIFIC PATTERN CHECKS:")
    print("-"*40)
    
    checks = results.get('specific_checks', {})
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check.replace('_', ' ').title()}")
    
    # Violations detail
    if results['pattern_violations']:
        print("\n" + "-"*40)
        print("PATTERN VIOLATIONS:")
        print("-"*40)
        for violation in results['pattern_violations']:
            print(f"  • {violation}")
    
    if results['css_violations']:
        print("\n" + "-"*40)
        print("CSS VIOLATIONS:")
        print("-"*40)
        for violation in results['css_violations']:
            print(f"  • {violation}")
    
    # Recommendations
    if not summary['passed']:
        print("\n" + "-"*40)
        print("RECOMMENDATIONS:")
        print("-"*40)
        print("  1. Review all reactive attributes have type hints")
        print("  2. Ensure workers use call_from_thread, not return values")
        print("  3. Move CSS to inline strings or same directory")
        print("  4. Avoid direct widget property manipulation")
        print("  5. Use reactive patterns for all state updates")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    print("Verifying Textual patterns in chat_v99 implementation...")
    results = verify_all_patterns()
    print_verification_report(results)
    
    # Return exit code based on results
    import sys
    sys.exit(0 if results['summary']['passed'] else 1)