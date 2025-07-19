#!/usr/bin/env python3
"""
Helper script to copy rag_pipelines.toml to user's config directory.
This ensures users have their own copy that they can modify.
"""

import sys
import shutil
from pathlib import Path

def copy_rag_pipelines_to_user_config():
    """Copy the default rag_pipelines.toml to user's config directory."""
    
    # Source file location
    source_file = Path(__file__).parent.parent / "tldw_chatbook" / "Config_Files" / "rag_pipelines.toml"
    
    # Destination directory and file
    user_config_dir = Path.home() / ".config" / "tldw_cli"
    dest_file = user_config_dir / "rag_pipelines.toml"
    
    # Check if source exists
    if not source_file.exists():
        print(f"Error: Source file not found at {source_file}")
        return False
    
    # Create user config directory if it doesn't exist
    user_config_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if destination already exists
    if dest_file.exists():
        response = input(f"File already exists at {dest_file}\nOverwrite? (y/N): ")
        if response.lower() != 'y':
            print("Skipping copy.")
            return False
    
    # Copy the file
    try:
        shutil.copy2(source_file, dest_file)
        print(f"Successfully copied rag_pipelines.toml to {dest_file}")
        print("\nYou can now edit this file to customize your RAG pipelines.")
        print("The application will use this file instead of the default one.")
        return True
    except Exception as e:
        print(f"Error copying file: {e}")
        return False

if __name__ == "__main__":
    success = copy_rag_pipelines_to_user_config()
    sys.exit(0 if success else 1)