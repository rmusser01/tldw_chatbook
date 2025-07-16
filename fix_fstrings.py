#!/usr/bin/env python3
"""Fix f-strings with backslashes in splash_animations.py"""

import re

# Read the file
with open('tldw_chatbook/Utils/splash_animations.py', 'r') as f:
    content = f.read()

# Find all f-strings with .replace() containing backslashes
# This pattern looks for f"...{something.replace(...\...)}..."
pattern = r'([ \t]*)(.*?)f"([^"]*)\{([^}]+\.replace\([^)]*\\\\[^)]*\))\}([^"]*)"'

def fix_fstring(match):
    indent = match.group(1)
    prefix = match.group(2)
    before_expr = match.group(3)
    expr = match.group(4)
    after_expr = match.group(5)
    
    # Extract the variable being replaced
    var_match = re.match(r'(\w+)\.replace\(', expr)
    if var_match:
        var_name = var_match.group(1)
        # Create a unique escaped variable name
        escaped_var = f"escaped_{var_name}"
    else:
        escaped_var = "escaped_text"
    
    # Return the fixed version with the replace outside the f-string
    return f'{indent}{prefix}{escaped_var} = {expr}\n{indent}{prefix}f"{before_expr}{{{escaped_var}}}{after_expr}"'

# Apply the fix
fixed_content = re.sub(pattern, fix_fstring, content, flags=re.MULTILINE)

# Count how many replacements were made
original_count = len(re.findall(pattern, content, flags=re.MULTILINE))
print(f"Found {original_count} f-strings with backslashes to fix")

# Write the fixed content
with open('tldw_chatbook/Utils/splash_animations.py', 'w') as f:
    f.write(fixed_content)

print("File has been fixed!")

# Verify by trying to compile it
try:
    import ast
    ast.parse(fixed_content)
    print("File compiles successfully!")
except SyntaxError as e:
    print(f"Still has syntax errors: {e}")
    print(f"Line {e.lineno}: {e.text}")