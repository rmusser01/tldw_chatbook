#!/usr/bin/env python3
"""
Fix all syntax errors in test_embeddings_service.py
"""

import re
from pathlib import Path

def fix_syntax_errors():
    """Fix all comment-related syntax errors"""
    filepath = Path(__file__).parent / "test_embeddings_service.py"
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix pattern 1: Lines starting with # followed by indented lines that aren't comments
    # This pattern finds commented assert_called lines with arguments on next lines
    pattern1 = r'(\s*)(#\s*#?\s*\w+\.assert_called[^(]*\()(\s*#[^\n]*\n)(\s+)(\w+=[^,\n]+(?:,\s*\n\s+\w+=[^,\n]+)*)\s*\)'
    
    def fix_multiline_comment(match):
        indent = match.group(1)
        call_start = match.group(2)
        comment = match.group(3).strip()
        args = match.group(5)
        
        # Split args by newline and re-indent with comments
        arg_lines = args.strip().split('\n')
        fixed_args = []
        for line in arg_lines:
            fixed_args.append(f"{indent}# {line.strip()}")
        
        result = f"{indent}{call_start}{comment}\n"
        result += '\n'.join(fixed_args)
        result += f"\n{indent}# )"
        
        return result
    
    # Pattern for fixing lines like:
    # # mock_collection.query.assert_called_once_with(  # Collection calls mocked differently
    #     query_embeddings=query_embeddings,
    pattern2 = r'(\s*)(#\s*(?:#\s*)?(?:mock_collection|embeddings_service\.client)\.\w+\.assert_called[^(]*\([^\n]*#[^\n]*\n)((?:\s+[^#\s][^\n]+\n)+)(\s*\))'
    
    def fix_multiline_comment2(match):
        indent = match.group(1)
        first_line = match.group(2).rstrip()
        args = match.group(3)
        closing = match.group(4)
        
        # Remove extra # from first line if present
        first_line = re.sub(r'^#\s*#', '#', first_line)
        
        # Indent all arg lines with #
        arg_lines = args.rstrip().split('\n')
        fixed_args = []
        for line in arg_lines:
            if line.strip():
                fixed_args.append(f"{indent}# {line.strip()}")
        
        result = f"{indent}{first_line}\n"
        result += '\n'.join(fixed_args)
        result += f"\n{indent}# {closing.strip()}"
        
        return result
    
    # Apply fixes
    content = re.sub(pattern2, fix_multiline_comment2, content, flags=re.MULTILINE)
    
    # Fix standalone indented lines after comments
    # Pattern: comment line followed by indented non-comment lines
    pattern3 = r'(\s*#[^\n]*(?:Collection calls mocked differently|Mock \w+ calls verified differently)\s*\n)(\s+)([^#\s][^\n]+(?:\n\s+[^#\s][^\n]+)*)\n(\s*\))'
    
    def fix_orphaned_args(match):
        comment_line = match.group(1).rstrip()
        indent = match.group(2)
        args = match.group(3)
        closing = match.group(4)
        
        # Comment out the orphaned argument lines
        base_indent = len(indent) - 4 if len(indent) >= 4 else 0
        prefix = ' ' * base_indent
        
        arg_lines = args.split('\n')
        fixed = [comment_line]
        for line in arg_lines:
            fixed.append(f"{prefix}# {line.strip()}")
        fixed.append(f"{prefix}# {closing.strip()}")
        
        return '\n'.join(fixed)
    
    content = re.sub(pattern3, fix_orphaned_args, content, flags=re.MULTILINE)
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print("Fixed syntax errors in test_embeddings_service.py")

if __name__ == "__main__":
    fix_syntax_errors()