#!/usr/bin/env python3
"""
Nuitka build script for tldw_chatbook Windows executable
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse

# Add parent directory to path to import version info
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.version import VERSION, COMPANY_NAME, PRODUCT_NAME, COPYRIGHT

class NuitkaBuilder:
    def __init__(self, build_mode="standard"):
        self.project_root = Path(__file__).parent.parent.parent
        self.build_dir = Path(__file__).parent / "build"
        self.dist_dir = Path(__file__).parent / "dist"
        self.build_mode = build_mode
        
    def clean_build_dirs(self):
        """Clean previous build directories"""
        print("Cleaning previous builds...")
        for dir_path in [self.build_dir, self.dist_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(exist_ok=True)
    
    def get_nuitka_args(self, entry_point, output_name=None):
        """Get Nuitka compilation arguments"""
        args = [
            sys.executable, "-m", "nuitka",
            "--standalone",
            "--assume-yes-for-downloads",
            f"--output-dir={self.dist_dir}",
            "--enable-console",
            "--windows-console-mode=force",
            
            # Company info
            f"--company-name={COMPANY_NAME}",
            f"--product-name={PRODUCT_NAME}",
            f"--file-version={VERSION}",
            f"--product-version={VERSION}",
            f"--file-description={PRODUCT_NAME}",
            f"--copyright={COPYRIGHT}",
            
            # Icon
            "--windows-icon-from-ico=packaging/windows/assets/icon.ico",
            
            # Optimizations
            "--follow-imports",
            "--show-progress",
            "--show-memory",
        ]
        
        # Plugins based on build mode
        if self.build_mode == "minimal":
            plugins = ["anti-bloat", "dataclasses", "multiprocessing"]
        elif self.build_mode == "standard":
            plugins = ["anti-bloat", "dataclasses", "multiprocessing", "numpy", "pkg-resources"]
        else:  # full
            plugins = ["anti-bloat", "dataclasses", "multiprocessing", "numpy", "torch", "transformers", "pkg-resources"]
        
        for plugin in plugins:
            args.append(f"--enable-plugin={plugin}")
        
        # Include packages
        include_packages = [
            "tldw_chatbook",
            "textual",
            "rich",
            "httpx",
            "pydantic",
            "loguru",
            "aiofiles",
            "jinja2",
            "toml",
            "yaml",
        ]
        
        # Add optional packages based on build mode
        if self.build_mode in ["standard", "full"]:
            include_packages.extend(["textual_serve", "aiohttp", "aiohttp_jinja2"])
        
        if self.build_mode == "full":
            include_packages.extend([
                "torch", "transformers", "numpy", "chromadb",
                "sentence_transformers", "nltk", "langdetect"
            ])
        
        for package in include_packages:
            args.append(f"--include-package={package}")
        
        # Include data files
        args.extend([
            "--include-package-data=tldw_chatbook",
            "--include-data-files=tldw_chatbook/css=tldw_chatbook/css",
            "--include-data-files=tldw_chatbook/Config_Files=tldw_chatbook/Config_Files",
        ])
        
        # Entry point
        args.append(str(entry_point))
        
        return args
    
    def build_cli_exe(self):
        """Build the main CLI executable"""
        print("Building tldw-cli.exe...")
        entry_point = self.project_root / "tldw_chatbook" / "app.py"
        args = self.get_nuitka_args(entry_point)
        
        result = subprocess.run(args, cwd=self.project_root)
        if result.returncode != 0:
            print("ERROR: Nuitka compilation failed for CLI")
            sys.exit(1)
            
        # Nuitka creates app.exe and app.exe.dist - rename them
        original_exe = self.dist_dir / "app.exe"
        original_dist = self.dist_dir / "app.exe.dist"
        target_exe = self.dist_dir / "tldw-cli.exe"
        target_dist = self.dist_dir / "tldw-cli.exe.dist"
        
        if original_exe.exists():
            original_exe.rename(target_exe)
        if original_dist.exists():
            original_dist.rename(target_dist)
            
        print("Successfully built tldw-cli.exe")
    
    def build_serve_exe(self):
        """Build the web server executable"""
        if self.build_mode == "minimal":
            print("Skipping tldw-serve.exe (not included in minimal build)")
            return
            
        print("Building tldw-serve.exe...")
        # Create a simple entry point for the server
        serve_entry = self.build_dir / "serve_main.py"
        serve_entry.write_text("""
import sys
from tldw_chatbook.Web_Server.serve import main

if __name__ == "__main__":
    main()
""")
        
        args = self.get_nuitka_args(serve_entry)
        
        result = subprocess.run(args, cwd=self.project_root)
        if result.returncode != 0:
            print("ERROR: Nuitka compilation failed for serve")
            sys.exit(1)
            
        # Nuitka creates serve_main.exe and serve_main.exe.dist - rename them
        original_exe = self.dist_dir / "serve_main.exe"
        original_dist = self.dist_dir / "serve_main.exe.dist"
        target_exe = self.dist_dir / "tldw-serve.exe"
        target_dist = self.dist_dir / "tldw-serve.exe.dist"
        
        if original_exe.exists():
            original_exe.rename(target_exe)
        if original_dist.exists():
            original_dist.rename(target_dist)
            
        print("Successfully built tldw-serve.exe")
    
    def copy_assets(self):
        """Copy additional assets to distribution"""
        print("Copying assets...")
        
        # Copy batch files
        scripts_dir = Path(__file__).parent / "scripts"
        for script in scripts_dir.glob("*.bat"):
            shutil.copy2(script, self.dist_dir)
        
        # Copy license
        license_file = self.project_root / "LICENSE"
        if license_file.exists():
            shutil.copy2(license_file, self.dist_dir / "LICENSE.txt")
        
        # Create README for distribution
        readme_content = f"""
{PRODUCT_NAME} v{VERSION}
{'=' * (len(PRODUCT_NAME) + len(VERSION) + 3)}

Thank you for using {PRODUCT_NAME}!

RUNNING THE APPLICATION:
- Double-click 'tldw-cli.bat' to run in terminal mode
- Double-click 'tldw-serve.bat' to run in web browser mode

FIRST TIME SETUP:
- The application will create a configuration file on first run
- Default location: %APPDATA%\\tldw_cli\\config.toml

For more information, visit:
https://github.com/rmusser01/tldw_chatbook

Build mode: {self.build_mode}
"""
        (self.dist_dir / "README.txt").write_text(readme_content)
        
    def build(self):
        """Run the complete build process"""
        print(f"Starting Nuitka build process (mode: {self.build_mode})...")
        self.clean_build_dirs()
        self.build_cli_exe()
        self.build_serve_exe()
        self.copy_assets()
        print(f"\nBuild complete! Output in: {self.dist_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build tldw_chatbook Windows executable")
    parser.add_argument(
        "--mode",
        choices=["minimal", "standard", "full"],
        default="standard",
        help="Build mode: minimal (core only), standard (common features), full (all features)"
    )
    
    args = parser.parse_args()
    
    # Check for Nuitka
    try:
        subprocess.run([sys.executable, "-m", "nuitka", "--version"], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: Nuitka not found. Install with: pip install nuitka")
        sys.exit(1)
    
    builder = NuitkaBuilder(build_mode=args.mode)
    builder.build()


if __name__ == "__main__":
    main()
