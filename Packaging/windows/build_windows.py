#!/usr/bin/env python3
"""
Complete build automation script for Windows packaging
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse
import zipfile

# Add parent directory to path to import version
sys.path.insert(0, str(Path(__file__).parent.parent / "common"))
from version import VERSION, COMPANY_NAME

def check_requirements():
    """Check if all required tools are installed"""
    tools = {
        "nuitka": "pip install nuitka",
        "NSIS": "Download from https://nsis.sourceforge.io/Download"
    }
    
    # Check Python version
    if sys.version_info < (3, 11):
        print("ERROR: Python 3.11+ is required")
        return False
    
    # Check Nuitka
    try:
        subprocess.run([sys.executable, "-m", "nuitka", "--version"], 
                      capture_output=True, check=True)
        print("✓ Nuitka found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"✗ Nuitka not found. Install with: {tools['nuitka']}")
        return False
    
    # Check NSIS
    nsis_path = shutil.which("makensis")
    if not nsis_path:
        # Try common locations
        common_paths = [
            r"C:\Program Files\NSIS\makensis.exe",
            r"C:\Program Files (x86)\NSIS\makensis.exe"
        ]
        for path in common_paths:
            if Path(path).exists():
                nsis_path = path
                break
    
    if nsis_path:
        print(f"✓ NSIS found at: {nsis_path}")
    else:
        print(f"✗ NSIS not found. {tools['NSIS']}")
        return False
    
    return True

def create_icon():
    """Create a basic icon if it doesn't exist"""
    icon_path = Path(__file__).parent / "assets" / "icon.ico"
    if not icon_path.exists():
        print("WARNING: icon.ico not found. Creating placeholder...")
        icon_path.parent.mkdir(exist_ok=True)
        # For now, we'll skip icon creation - in production, you'd convert a PNG
        return False
    return True

def create_license():
    """Copy license file"""
    license_src = Path(__file__).parent.parent.parent / "LICENSE"
    license_dst = Path(__file__).parent / "assets" / "license.txt"
    
    if license_src.exists():
        license_dst.parent.mkdir(exist_ok=True)
        shutil.copy2(license_src, license_dst)
        print("✓ License file copied")
    else:
        print("WARNING: LICENSE file not found")
        # Create a placeholder
        license_dst.parent.mkdir(exist_ok=True)
        license_dst.write_text("See https://github.com/rmusser01/tldw_chatbook for license information.")

def build_executables(mode):
    """Build the executables using Nuitka"""
    print(f"\nBuilding executables in {mode} mode...")
    
    build_script = Path(__file__).parent / "build_exe.py"
    result = subprocess.run([
        sys.executable, str(build_script), "--mode", mode
    ])
    
    return result.returncode == 0

def build_installer(nsis_path=None):
    """Build the NSIS installer"""
    print("\nBuilding NSIS installer...")
    
    nsi_script = Path(__file__).parent / "installer.nsi"
    
    # Find makensis
    if not nsis_path:
        nsis_path = shutil.which("makensis")
        if not nsis_path:
            # Try common locations
            common_paths = [
                r"C:\Program Files\NSIS\makensis.exe",
                r"C:\Program Files (x86)\NSIS\makensis.exe"
            ]
            for path in common_paths:
                if Path(path).exists():
                    nsis_path = path
                    break
    
    if not nsis_path:
        print("ERROR: makensis not found")
        return False
    
    # Run NSIS with version and publisher defines
    cmd = [
        nsis_path,
        f"/DPRODUCT_VERSION={VERSION}",
        f"/DPRODUCT_PUBLISHER={COMPANY_NAME}",
        str(nsi_script)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    return result.returncode == 0

def create_portable_zip(mode):
    """Create a portable ZIP distribution"""
    print("\nCreating portable ZIP distribution...")
    
    dist_dir = Path(__file__).parent / "dist"
    zip_name = f"tldw-chatbook-portable-{mode}.zip"
    zip_path = Path(__file__).parent / zip_name
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add all files from dist directory
        for file_path in dist_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(dist_dir)
                zf.write(file_path, arcname)
    
    print(f"✓ Created portable distribution: {zip_name}")
    return True

def clean_build():
    """Clean build artifacts"""
    print("\nCleaning build artifacts...")
    dirs_to_clean = ["build", "dist", "__pycache__"]
    
    for dir_name in dirs_to_clean:
        dir_path = Path(__file__).parent / dir_name
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"✓ Removed {dir_name}/")

def main():
    parser = argparse.ArgumentParser(description="Build tldw_chatbook for Windows")
    parser.add_argument(
        "--mode",
        choices=["minimal", "standard", "full"],
        default="standard",
        help="Build mode"
    )
    parser.add_argument(
        "--portable",
        action="store_true",
        help="Create portable ZIP instead of installer"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build artifacts and exit"
    )
    parser.add_argument(
        "--nsis-path",
        help="Path to makensis.exe"
    )
    
    args = parser.parse_args()
    
    if args.clean:
        clean_build()
        return
    
    print(f"tldw_chatbook Windows Build Script")
    print("=" * 40)
    print(f"Build mode: {args.mode}")
    print(f"Output type: {'Portable ZIP' if args.portable else 'NSIS Installer'}")
    print()
    
    # Check requirements
    if not check_requirements():
        print("\nBuild failed: Missing requirements")
        sys.exit(1)
    
    # Prepare assets
    if not create_icon():
        print("\nERROR: Failed to create/find icon file. Build cannot continue.")
        print("Please ensure icon.ico exists in the assets directory.")
        sys.exit(1)
    create_license()
    
    # Build executables
    if not build_executables(args.mode):
        print("\nBuild failed: Executable compilation error")
        sys.exit(1)
    
    # Create distribution
    if args.portable:
        if not create_portable_zip(args.mode):
            print("\nBuild failed: ZIP creation error")
            sys.exit(1)
    else:
        if not build_installer(args.nsis_path):
            print("\nBuild failed: Installer creation error")
            sys.exit(1)
    
    print("\n✓ Build completed successfully!")
    
    # List output files
    print("\nOutput files:")
    output_dir = Path(__file__).parent
    for pattern in ["*.exe", "*.zip"]:
        for file in output_dir.glob(pattern):
            if file.stem != "build_windows":
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()