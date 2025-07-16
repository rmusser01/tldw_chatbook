#!/usr/bin/env python3
"""Fix f-string issues in splash_animations.py by moving replace operations outside f-strings."""

import re

# Read the file
with open('tldw_chatbook/Utils/splash_animations.py', 'r') as f:
    content = f.read()

# Split into lines for easier processing
lines = content.split('\n')
fixed_lines = []
i = 0

while i < len(lines):
    line = lines[i]
    
    # Check if this line contains an f-string with a replace operation inside
    if 'f"' in line or "f'" in line:
        # Check if there's a .replace inside the f-string
        if '.replace(' in line and '{' in line and '}' in line:
            # Find the pattern f"...{expr.replace(...)}..."
            # This is complex, so let's check if the line contains problematic patterns
            
            # Pattern 1: f"[{style}]{char.replace('[', r'\[')}[/{style}]" or similar
            match = re.search(r'(\s*)(.*)f(["\'])([^{]*)(\{)([^}]+\.replace\([^}]+\))(\})([^"\']*)\3', line)
            if match:
                indent = match.group(1)
                prefix = match.group(2)
                quote = match.group(3)
                before_expr = match.group(4)
                open_brace = match.group(5)
                expr = match.group(6)
                close_brace = match.group(7)
                after_expr = match.group(8)
                
                # Extract the variable name from the expression
                var_match = re.match(r'(\w+)\.replace\(', expr)
                if var_match:
                    var_name = var_match.group(1)
                    # Create escaped variable name
                    escaped_var = f"escaped_{var_name}"
                    
                    # Check if we need to use ESCAPED_OPEN_BRACKET constant
                    if ".replace('[', '\\\\['" in expr or '.replace("[", "\\\\["' in expr:
                        new_expr = f"{var_name}.replace('[', ESCAPED_OPEN_BRACKET)"
                    elif ".replace(']', '\\\\]'" in expr or '.replace("]", "\\\\]"' in expr:
                        new_expr = f"{var_name}.replace(']', ESCAPED_CLOSE_BRACKET)"
                    else:
                        new_expr = expr
                    
                    # Add the escaped variable line before the f-string
                    fixed_lines.append(f"{indent}{escaped_var} = {new_expr}")
                    # Replace the f-string to use the escaped variable
                    fixed_lines.append(f"{indent}{prefix}f{quote}{before_expr}{{{escaped_var}}}{after_expr}{quote}")
                    i += 1
                    continue
    
    fixed_lines.append(line)
    i += 1

# Write the fixed content back
with open('tldw_chatbook/Utils/splash_animations.py', 'w') as f:
    f.write('\n'.join(fixed_lines))

print("Fixed f-string issues in splash_animations.py")

# Try to compile it to check for syntax errors
try:
    import ast
    ast.parse('\n'.join(fixed_lines))
    print("File compiles successfully!")
except SyntaxError as e:
    print(f"Still has syntax errors: {e}")
    print(f"Line {e.lineno}: {e.text}")