#!/usr/bin/env python3
"""
Check that all expected files are included in the distribution.
Run this after building to verify MANIFEST.in is working correctly.
"""

import tarfile
import zipfile
import sys
from pathlib import Path

def check_distribution():
    """Check the built distribution for expected files."""
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("Error: dist/ directory not found. Run build_dist.sh first.")
        return False
    
    # Find the source distribution
    sdist_files = list(dist_dir.glob("*.tar.gz"))
    if not sdist_files:
        print("Error: No source distribution (*.tar.gz) found in dist/")
        return False
    
    sdist_file = sdist_files[0]
    print(f"Checking source distribution: {sdist_file}")
    
    expected_patterns = [
        "LICENSE",
        "README.md",
        "MANIFEST.in",
        "pyproject.toml",
        "requirements.txt",
        "tldw_chatbook/__init__.py",
        "tldw_chatbook/app.py",
        "tldw_chatbook/css/*.tcss",
        "tldw_chatbook/css/Themes/*.tcss",
        "tldw_chatbook/Config_Files/*.json",
    ]
    
    found_files = set()
    missing_patterns = []
    
    with tarfile.open(sdist_file, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.isfile():
                # Remove the top-level directory from the path
                parts = member.name.split('/', 1)
                if len(parts) > 1:
                    found_files.add(parts[1])
    
    # Check each pattern
    for pattern in expected_patterns:
        if '*' in pattern:
            # It's a wildcard pattern
            prefix, suffix = pattern.split('*')
            matching = [f for f in found_files if f.startswith(prefix) and f.endswith(suffix)]
            if not matching:
                missing_patterns.append(pattern)
            else:
                print(f"✓ Found {len(matching)} files matching {pattern}")
        else:
            # It's a specific file
            if pattern in found_files:
                print(f"✓ Found {pattern}")
            else:
                missing_patterns.append(pattern)
                print(f"✗ Missing {pattern}")
    
    # Check wheel distribution
    wheel_files = list(dist_dir.glob("*.whl"))
    if wheel_files:
        print(f"\nChecking wheel distribution: {wheel_files[0]}")
        with zipfile.ZipFile(wheel_files[0], 'r') as whl:
            wheel_files = whl.namelist()
            # Check for data files in wheel
            data_files = [f for f in wheel_files if '.dist-info' not in f and not f.endswith('.py')]
            print(f"Found {len(data_files)} non-Python files in wheel")
    
    if missing_patterns:
        print(f"\nWarning: {len(missing_patterns)} expected patterns not found!")
        return False
    else:
        print("\nAll expected files found in distribution! ✓")
        return True

if __name__ == "__main__":
    success = check_distribution()
    sys.exit(0 if success else 1)