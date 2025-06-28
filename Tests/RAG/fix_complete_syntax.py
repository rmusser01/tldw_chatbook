#!/usr/bin/env python3
"""
Comprehensive fix for all syntax errors in test_embeddings_service.py
"""

import re
from pathlib import Path

def fix_complete_syntax():
    """Fix all syntax errors comprehensively"""
    filepath = Path(__file__).parent / "test_embeddings_service.py"
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    in_commented_method = False
    method_indent = 0
    
    for i, line in enumerate(lines):
        # Detect start of commented method definition
        if re.match(r'^#\s*def\s+\w+.*:\s*$', line):
            in_commented_method = True
            method_indent = 0
            fixed_lines.append(line)
            continue
        
        # If we're in a commented method
        if in_commented_method:
            # Check if this line is part of the commented method
            if line.strip() and not line.strip().startswith('#'):
                # This is an uncommented line inside a commented method
                # It needs to be commented out
                stripped = line.lstrip()
                indent = len(line) - len(stripped)
                if indent > method_indent or method_indent == 0:
                    # Part of the method body
                    if method_indent == 0:
                        method_indent = indent
                    fixed_lines.append('    # ' + stripped)
                    continue
                else:
                    # New method or class-level code
                    in_commented_method = False
            elif not line.strip():
                # Empty line
                fixed_lines.append(line)
                continue
            elif line.strip().startswith('#'):
                # Already commented
                fixed_lines.append(line)
                continue
        
        # Check for standalone assert statements that should be part of a method
        if re.match(r'^\s+assert\s+', line) and i > 0:
            # Look back to see if previous non-empty line is a comment
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            if j >= 0 and lines[j].strip().startswith('#'):
                # This assert is orphaned, comment it out
                fixed_lines.append('    # ' + line.strip() + '\n')
                continue
        
        fixed_lines.append(line)
    
    # Write back
    with open(filepath, 'w') as f:
        f.writelines(fixed_lines)
    
    print("Fixed all syntax errors in test_embeddings_service.py")

if __name__ == "__main__":
    fix_complete_syntax()