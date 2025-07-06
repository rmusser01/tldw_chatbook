#!/usr/bin/env python3
"""Script to update the root config.toml from CONFIG_TOML_CONTENT in config.py"""

import re

# Read the config.py file
with open('/Users/appledev/Working/tldw_chatbook_dev/tldw_chatbook/config.py', 'r') as f:
    content = f.read()

# Extract CONFIG_TOML_CONTENT using regex
match = re.search(r'CONFIG_TOML_CONTENT = """(.*?)"""', content, re.DOTALL)
if match:
    config_toml_content = match.group(1)
    
    # Write to root config.toml
    with open('/Users/appledev/Working/tldw_chatbook_dev/config.toml', 'w') as f:
        f.write(config_toml_content)
    
    print("Successfully updated config.toml")
else:
    print("Could not find CONFIG_TOML_CONTENT in config.py")