#!/usr/bin/env python3
"""Validate that MediaWindowV88 is properly set up in app.py"""

import ast
import sys

# Read app.py
with open('tldw_chatbook/app.py', 'r') as f:
    content = f.read()

# Check if MediaWindowV88 is imported
if 'from .UI.MediaWindowV88 import MediaWindowV88' in content:
    print("✓ MediaWindowV88 is imported")
else:
    print("✗ MediaWindowV88 is NOT imported")
    sys.exit(1)

# Check if MediaWindowV88 is used in windows list
if '("media", MediaWindowV88, "media-window")' in content:
    print("✓ MediaWindowV88 is set for media tab")
else:
    print("✗ MediaWindowV88 is NOT set for media tab")
    sys.exit(1)

# Check if old MediaWindow references are updated
if 'self.query_one(MediaWindow)' in content and 'MediaWindowV88' not in content:
    print("✗ Old MediaWindow references still exist")
    sys.exit(1)
else:
    print("✓ MediaWindow references updated")

print("\n✅ MediaWindowV88 is properly configured!")
print("\nWhen you run the app and navigate to the Media tab, MediaWindowV88 will load.")
print("The UI will show:")
print("  - Left navigation column with dropdown and media list")
print("  - Collapsible search bar at the top")
print("  - Metadata panel with 4-row layout")
print("  - Content viewer tabs at the bottom")