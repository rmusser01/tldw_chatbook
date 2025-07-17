#!/usr/bin/env python3
import re

# Read the file
with open('../tldw_chatbook/UI/Tools_Settings_Window.py', 'r') as f:
    lines = f.readlines()

# Find all occurrences of call_from_thread
for i, line in enumerate(lines):
    if 'call_from_thread' in line:
        # Print context (5 lines before and after)
        start = max(0, i - 5)
        end = min(len(lines), i + 6)
        
        print(f"\n--- Occurrence at line {i + 1} ---")
        for j in range(start, end):
            if j == i:
                print(f">>> {j + 1}: {lines[j].rstrip()}")
            else:
                print(f"    {j + 1}: {lines[j].rstrip()}")