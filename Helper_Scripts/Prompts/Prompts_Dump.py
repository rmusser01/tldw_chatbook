#!/usr/bin/env python3
"""
Prompts_Dump.py - Export all prompts from the user's prompt database to markdown files.

This script exports all prompts from a user's prompt database to individual markdown files,
following the standard format used in the Helper_Scripts/Prompts/ directory.

Usage:
    python3 Prompts_Dump.py [--user USERNAME] [--output-dir PATH] [--verbose]
"""

import argparse
import logging
import sys
from pathlib import Path
import re
from typing import Optional, List, Dict, Any
from datetime import datetime

# Add project root to path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

try:
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
    from tldw_chatbook.config import BASE_DATA_DIR_CLI
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print(f"Please ensure you're running this script from the tldw_chatbook project directory")
    sys.exit(1)

# Configure logging
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # File handler
    log_file = Path("prompts_dump.log")
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file.absolute()}")


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be used as a filename.
    
    Args:
        name: The original name
        
    Returns:
        A sanitized filename safe for use on most filesystems
    """
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Remove any non-printable characters
    sanitized = ''.join(char for char in sanitized if char.isprintable())
    # Limit length to 200 characters (leaving room for extension and potential numbering)
    sanitized = sanitized[:200]
    # Remove trailing periods and spaces (Windows compatibility)
    sanitized = sanitized.rstrip('. ')
    # If empty after sanitization, use a default name
    if not sanitized:
        sanitized = "unnamed_prompt"
    return sanitized


def get_user_db_path(username: str) -> Path:
    """
    Get the path to the prompts database for a specific user.
    
    Args:
        username: The username (folder name) for the user
        
    Returns:
        Path to the prompts database file
    """
    user_dir = BASE_DATA_DIR_CLI / username
    # Try the new database name first
    prompts_db = user_dir / "prompts.db"
    if prompts_db.exists():
        return prompts_db
    # Fall back to old name if new doesn't exist
    old_prompts_db = user_dir / "tldw_chatbook_prompts.db"
    if old_prompts_db.exists():
        return old_prompts_db
    # Return new name if neither exists (will create new)
    return prompts_db


def export_prompt_to_markdown(prompt: Dict[str, Any], output_path: Path) -> None:
    """
    Export a single prompt to a markdown file.
    
    Args:
        prompt: Dictionary containing prompt data
        output_path: Path where the markdown file should be saved
    """
    # Extract prompt data with defaults
    title = prompt.get('name', 'Untitled Prompt')
    author = prompt.get('author', 'Unknown')
    system_prompt = prompt.get('system_prompt', '')
    user_prompt = prompt.get('user_prompt', '')
    keywords = prompt.get('keywords', [])
    
    # Convert keywords list to comma-separated string
    if isinstance(keywords, list):
        keywords_str = ', '.join(keywords)
    else:
        keywords_str = str(keywords) if keywords else ''
    
    # Build the markdown content following the template format
    content = f"""### TITLE ###
{title}

### AUTHOR ###
{author}

### SYSTEM ###
{system_prompt}

### USER ###
{user_prompt}

### KEYWORDS ###
{keywords_str}
"""
    
    # Write to file
    try:
        output_path.write_text(content, encoding='utf-8')
        logger.debug(f"Successfully exported prompt '{title}' to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write prompt '{title}' to {output_path}: {e}")
        raise


def export_prompts(username: str, output_dir: Path, verbose: bool = False) -> int:
    """
    Export all prompts from the user's database to markdown files.
    
    Args:
        username: The username to export prompts for
        output_dir: Directory where prompt files should be saved
        verbose: Enable verbose logging
        
    Returns:
        Number of prompts successfully exported
    """
    # Get database path
    db_path = get_user_db_path(username)
    
    if not db_path.exists():
        logger.error(f"Database not found for user '{username}' at {db_path}")
        return 0
    
    logger.info(f"Connecting to database: {db_path}")
    
    try:
        # Initialize database connection
        db = PromptsDatabase(str(db_path), client_id="prompts_dump_script")
        logger.info("Successfully connected to database")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return 0
    
    try:
        # Fetch all prompts (including soft-deleted if needed)
        logger.info("Fetching all prompts from database...")
        all_prompts = db.get_all_prompts(include_deleted=False, limit=10000)
        
        if not all_prompts:
            logger.warning("No prompts found in database")
            return 0
        
        logger.info(f"Found {len(all_prompts)} prompts to export")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir.absolute()}")
        
        # Track exported prompts and handle duplicates
        exported_count = 0
        filename_counter = {}
        
        for i, prompt in enumerate(all_prompts, 1):
            try:
                # Get prompt name for filename
                prompt_name = prompt.get('name', f'prompt_{i}')
                base_filename = sanitize_filename(prompt_name)
                
                # Handle duplicate filenames
                if base_filename in filename_counter:
                    filename_counter[base_filename] += 1
                    filename = f"{base_filename}_{filename_counter[base_filename]}.md"
                else:
                    filename_counter[base_filename] = 0
                    filename = f"{base_filename}.md"
                
                output_path = output_dir / filename
                
                # Fetch keywords for this prompt
                prompt_id = prompt.get('id')
                if prompt_id:
                    keywords = db.fetch_keywords_for_prompt(prompt_id, include_deleted=False)
                    prompt['keywords'] = keywords
                
                # Export to markdown
                export_prompt_to_markdown(prompt, output_path)
                exported_count += 1
                
                if verbose or (i % 10 == 0):
                    logger.info(f"Progress: {i}/{len(all_prompts)} prompts exported")
                    
            except Exception as e:
                logger.error(f"Failed to export prompt {i} ('{prompt.get('name', 'Unknown')}'): {e}")
                continue
        
        logger.info(f"Successfully exported {exported_count} out of {len(all_prompts)} prompts")
        return exported_count
        
    finally:
        # Clean up database connection
        try:
            db.close()
            logger.debug("Database connection closed")
        except:
            pass


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Export all prompts from a user's prompt database to markdown files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export prompts for default user
  python3 Prompts_Dump.py
  
  # Export prompts for specific user
  python3 Prompts_Dump.py --user john_doe
  
  # Export to custom directory with verbose output
  python3 Prompts_Dump.py --output-dir /path/to/exports --verbose
        """
    )
    
    parser.add_argument(
        '--user',
        type=str,
        default='default_user',
        help='Username (folder name) to export prompts from (default: default_user)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='Exported_Prompts',
        help='Directory to export prompts to (default: Exported_Prompts)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Log script start
    logger.info("="*60)
    logger.info(f"Prompts Dump Script Started - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"User: {args.user}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info("="*60)
    
    # Convert output directory to Path
    output_dir = Path(args.output_dir)
    
    try:
        # Export prompts
        exported_count = export_prompts(args.user, output_dir, args.verbose)
        
        # Print summary
        if exported_count > 0:
            print(f"\n✅ Successfully exported {exported_count} prompts to {output_dir.absolute()}")
            logger.info(f"Export completed successfully. {exported_count} prompts exported.")
        else:
            print(f"\n❌ No prompts were exported.")
            logger.warning("Export completed with no prompts exported.")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Export interrupted by user")
        logger.warning("Export interrupted by user (KeyboardInterrupt)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        logger.error(f"Export failed with unexpected error: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Script completed")


if __name__ == "__main__":
    main()