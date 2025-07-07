#!/usr/bin/env python3
"""Debug embeddings button issue."""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check the actual error
print("Checking embeddings button handler issue...")
print("-" * 50)

# Read the file and analyze
with open("tldw_chatbook/UI/Embeddings_Management_Window.py", "r") as f:
    lines = f.readlines()

# Find the button handler
for i, line in enumerate(lines):
    if '@on(Button.Pressed, "#embeddings-download-model")' in line:
        print(f"Found handler decorator at line {i+1}")
        # Print the next few lines to see the method definition
        for j in range(i, min(i+10, len(lines))):
            print(f"{j+1}: {lines[j].rstrip()}")
        break

print("\n" + "-" * 50)
print("Button definition in compose:")

# Find where the button is yielded
for i, line in enumerate(lines):
    if 'yield Button("Download", id="embeddings-download-model"' in line:
        print(f"Found button at line {i+1}")
        # Print context
        for j in range(max(0, i-2), min(i+3, len(lines))):
            print(f"{j+1}: {lines[j].rstrip()}")
        break

print("\n" + "-" * 50)
print("Worker method definition:")

# Find the worker method
for i, line in enumerate(lines):
    if 'def _download_model_worker(self)' in line:
        print(f"Found worker at line {i+1}")
        # Print first few lines
        for j in range(i, min(i+5, len(lines))):
            print(f"{j+1}: {lines[j].rstrip()}")
        break

print("\n" + "-" * 50)
print("Summary:")
print("1. The button handler should NOT be async")
print("2. The worker should be called directly without wrapper")
print("3. Message handlers should be present for UI updates")
print("4. Check that EmbeddingsManagementWindow is properly imported and used")