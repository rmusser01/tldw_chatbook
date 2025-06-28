#!/usr/bin/env python3
"""
Fix missing closing parentheses in test_embeddings_service.py
"""

import re
from pathlib import Path

def fix_missing_parens():
    """Fix all missing closing parentheses"""
    filepath = Path(__file__).parent / "test_embeddings_service.py"
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Look for lines that end with "# )" and should actually be ")"
        if line.strip() == '# )' and i > 0:
            # Check if previous line looks like end of a method call
            prev_line = lines[i-1].strip()
            if (prev_line.endswith('=2') or 
                prev_line.endswith('where_filter') or 
                prev_line.endswith('ids') or
                prev_line.endswith(']')):
                # This should be a real closing paren, not a comment
                fixed_lines.append(line.replace('# )', ')'))
                continue
        
        fixed_lines.append(line)
    
    # Write back
    with open(filepath, 'w') as f:
        f.writelines(fixed_lines)
    
    print("Fixed missing parentheses")

if __name__ == "__main__":
    fix_missing_parens()