#!/usr/bin/env python3
"""
Build script for macOS .app bundle
Can use either py2app or Nuitka
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import plistlib
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.version import VERSION, COMPANY_NAME, PRODUCT_NAME, COPYRIGHT

class MacOSBuilder:
    def __init__(self, build_mode="standard", use_nuitka=False):
        self.project_root = Path(__file__).parent.parent.parent
        self.build_dir = Path(__file__).parent / "build"
        self.dist_dir = Path(__file__).parent / "dist"
        self.build_mode = build_mode
        self.use_nuitka = use_nuitka
        self.app_name = "tldw chatbook"
        
    def clean_build_dirs(self):
        """Clean previous build directories"""
        print("Cleaning previous builds...")
        for dir_path in [self.build_dir, self.dist_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(exist_ok=True)
    
    def create_app_icon(self):
        """Create .icns file from PNG if available"""
        icon_path = Path(__file__).parent / "assets" / "icon.icns"
        if not icon_path.exists():
            print("WARNING: icon.icns not found")
            # In production, you'd convert from PNG using iconutil
            return None
        return icon_path
    
    def build_with_nuitka(self):
        """Build using Nuitka for macOS"""
        print("Building with Nuitka...")
        
        entry_point = self.project_root / "tldw_chatbook" / "app.py"
        
        args = [
            sys.executable, "-m", "nuitka",
            "--standalone",
            "--macos-create-app-bundle",
            "--assume-yes-for-downloads",
            f"--output-dir={self.dist_dir}",
            f"--macos-app-name={self.app_name}",
            "--enable-console",  # Keep console for TUI
            
            # App metadata
            f"--macos-app-version={VERSION}",
            f"--company-name={COMPANY_NAME}",
            f"--product-name={PRODUCT_NAME}",
            f"--copyright={COPYRIGHT}",
            
            # Optimizations
            "--follow-imports",
            "--show-progress",
        ]
        
        # Icon
        icon_path = self.create_app_icon()
        if icon_path:
            args.append(f"--macos-app-icon={icon_path}")
        
        # Plugins based on build mode
        if self.build_mode == "minimal":
            plugins = ["anti-bloat", "dataclasses", "multiprocessing"]
        elif self.build_mode == "standard":
            plugins = ["anti-bloat", "dataclasses", "multiprocessing", "numpy"]
        else:  # full
            plugins = ["anti-bloat", "dataclasses", "multiprocessing", "numpy", "torch", "transformers"]
        
        for plugin in plugins:
            args.append(f"--enable-plugin={plugin}")
        
        # Include packages
        include_packages = [
            "tldw_chatbook",
            "textual",
            "rich",
            "httpx",
            "pydantic",
        ]
        
        if self.build_mode in ["standard", "full"]:
            include_packages.extend(["textual_serve", "aiohttp"])
        
        for package in include_packages:
            args.append(f"--include-package={package}")
        
        # Entry point
        args.append(str(entry_point))
        
        result = subprocess.run(args, cwd=self.project_root)
        return result.returncode == 0
    
    def build_with_py2app(self):
        """Build using py2app"""
        print("Building with py2app...")
        
        # Create setup.py for py2app
        setup_py = self.build_dir / "setup.py"
        setup_content = f"""
from setuptools import setup

APP = ['{self.project_root}/tldw_chatbook/app.py']
DATA_FILES = [
    ('css', glob.glob(f"{self.project_root}/tldw_chatbook/css/*")),
    ('Config_Files', glob.glob(f"{self.project_root}/tldw_chatbook/Config_Files/*")),
]
]

OPTIONS = {{
    'argv_emulation': False,
    'packages': ['tldw_chatbook', 'textual', 'rich', 'httpx', 'pydantic'],
    'iconfile': '../assets/icon.icns',
    'plist': {{
        'CFBundleName': '{self.app_name}',
        'CFBundleDisplayName': '{PRODUCT_NAME}',
        'CFBundleVersion': '{VERSION}',
        'CFBundleShortVersionString': '{VERSION}',
        'NSHumanReadableCopyright': '{COPYRIGHT}',
        'LSMinimumSystemVersion': '11.0',
        'LSUIElement': False,
    }}
}}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={{'py2app': OPTIONS}},
    setup_requires=['py2app'],
)
"""
        setup_py.write_text(setup_content)
        
        # Run py2app
        result = subprocess.run([
            sys.executable, str(setup_py), "py2app", 
            f"--dist-dir={self.dist_dir}"
        ], cwd=self.build_dir)
        
        return result.returncode == 0
    
    def process_info_plist_template(self):
        """Process Info.plist.template to create Info.plist with actual values"""
        template_path = Path(__file__).parent / "Info.plist.template"
        output_path = self.build_dir / "Info.plist"
        
        if template_path.exists():
            content = template_path.read_text()
            content = content.replace("__VERSION__", VERSION)
            content = content.replace("__COPYRIGHT__", COPYRIGHT)
            output_path.write_text(content)
            print(f"Created Info.plist from template with version {VERSION}")
    
    def create_launcher_script(self):
        """Create launcher script for terminal"""
        # First, rename the original executable
        app_path = self.dist_dir / f"{self.app_name}.app"
        macos_dir = app_path / "Contents" / "MacOS"
        original_exec = macos_dir / "tldw_chatbook"
        renamed_exec = macos_dir / "tldw_chatbook_exec"
        
        if original_exec.exists():
            original_exec.rename(renamed_exec)
        
        launcher_content = '''#!/bin/bash
# Launcher for tldw chatbook

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if running in Terminal.app
if [[ "$TERM_PROGRAM" != "Apple_Terminal" && "$TERM_PROGRAM" != "iTerm.app" ]]; then
    # Not in a proper terminal, launch in Terminal.app
    osascript -e "tell application \\"Terminal\\" to do script \\"'$DIR/tldw_chatbook' $@\\""
else
    # Already in terminal, just run the renamed executable
    cd "$DIR"
    ./tldw_chatbook_exec "$@"
fi
'''
        
        launcher_path = macos_dir / "tldw_chatbook"
        launcher_path.write_text(launcher_content)
        os.chmod(launcher_path, 0o755)
    
    def create_info_plist_additions(self):
        """Add custom Info.plist entries"""
        app_path = self.dist_dir / f"{self.app_name}.app"
        plist_path = app_path / "Contents" / "Info.plist"
        
        if plist_path.exists():
            with open(plist_path, 'rb') as f:
                plist_data = plistlib.load(f)
            
            # Add URL scheme for web server
            plist_data['CFBundleURLTypes'] = [{
                'CFBundleURLName': 'tldw-chatbook',
                'CFBundleURLSchemes': ['tldw-chatbook']
            }]
            
            # Add document types if needed
            plist_data['CFBundleDocumentTypes'] = [{
                'CFBundleTypeName': 'Text Document',
                'CFBundleTypeRole': 'Editor',
                'LSItemContentTypes': ['public.plain-text', 'public.text'],
            }]
            
            with open(plist_path, 'wb') as f:
                plistlib.dump(plist_data, f)
    
    def build(self):
        """Run the complete build process"""
        print(f"Starting macOS build process (mode: {self.build_mode})...")
        self.clean_build_dirs()
        
        if self.use_nuitka:
            success = self.build_with_nuitka()
        else:
            success = self.build_with_py2app()
        
        if success:
            self.create_launcher_script()
            self.create_info_plist_additions()
            print(f"\nBuild complete! App bundle in: {self.dist_dir}")
        else:
            print("\nBuild failed!")
            return False
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Build tldw_chatbook macOS app")
    parser.add_argument(
        "--mode",
        choices=["minimal", "standard", "full"],
        default="standard",
        help="Build mode"
    )
    parser.add_argument(
        "--builder",
        choices=["py2app", "nuitka"],
        default="py2app",
        help="Build system to use"
    )
    
    args = parser.parse_args()
    
    # Check for builder
    if args.builder == "nuitka":
        try:
            subprocess.run([sys.executable, "-m", "nuitka", "--version"], 
                          capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("ERROR: Nuitka not found. Install with: pip install nuitka")
            sys.exit(1)
    else:
        try:
            import py2app
        except ImportError:
            print("ERROR: py2app not found. Install with: pip install py2app")
            sys.exit(1)
    
    builder = MacOSBuilder(
        build_mode=args.mode,
        use_nuitka=(args.builder == "nuitka")
    )
    
    if not builder.build():
        sys.exit(1)


if __name__ == "__main__":
    main()