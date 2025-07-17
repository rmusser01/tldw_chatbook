#!/usr/bin/env python3
"""Clean Python cache files to force recompilation."""

import os
import shutil

def clean_pycache(root_dir):
    """Remove all __pycache__ directories."""
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '__pycache__' in dirnames:
            pycache_path = os.path.join(dirpath, '__pycache__')
            try:
                shutil.rmtree(pycache_path)
                print(f"Removed: {pycache_path}")
                count += 1
            except Exception as e:
                print(f"Failed to remove {pycache_path}: {e}")
    
    # Also remove .pyc files
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.pyc'):
                pyc_path = os.path.join(dirpath, filename)
                try:
                    os.remove(pyc_path)
                    print(f"Removed: {pyc_path}")
                    count += 1
                except Exception as e:
                    print(f"Failed to remove {pyc_path}: {e}")
    
    return count

if __name__ == "__main__":
    import sys
    
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"Cleaning Python cache in: {root}")
    
    count = clean_pycache(root)
    print(f"\nCleaned {count} cache files/directories")