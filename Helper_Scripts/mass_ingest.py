#!/usr/bin/env python3
"""
Mass Media Ingestion Script for tldw_chatbook

This script allows bulk ingestion of media files into the tldw_chatbook database
without needing to use the UI. It supports all file types that the application
can handle: PDFs, documents (DOCX, ODT, RTF), e-books (EPUB, MOBI, AZW), 
plain text files, and HTML files.

Usage Examples:
    # Ingest a single directory (non-recursive)
    python mass_ingest.py /path/to/media/folder

    # Ingest a directory recursively
    python mass_ingest.py /path/to/media/folder --recursive

    # Ingest only specific file types
    python mass_ingest.py /path/to/media/folder --types pdf document

    # Ingest with analysis/summarization (requires API key)
    python mass_ingest.py /path/to/media/folder --analyze --api-name openai --api-key YOUR_KEY

    # Add common keywords to all ingested files
    python mass_ingest.py /path/to/media/folder --keywords "research" "2024" "important"

    # Exclude certain file patterns
    python mass_ingest.py /path/to/media/folder --exclude "*.tmp" "draft_*" "~*"

    # Use custom chunking options
    python mass_ingest.py /path/to/media/folder --chunk-method semantic --chunk-size 1000

    # Continue on errors (don't stop on first failure)
    python mass_ingest.py /path/to/media/folder --continue-on-error

    # Use a custom database path
    python mass_ingest.py /path/to/media/folder --db-path /custom/path/to/media.db
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add the project root to Python path if running as a script
if __name__ == "__main__":
    project_root = Path(__file__).parent
    if project_root not in sys.path:
        sys.path.insert(0, str(project_root))

try:
    from tldw_chatbook.Local_Ingestion import (
        ingest_local_file,
        batch_ingest_files,
        ingest_directory,
        get_supported_extensions,
        FileIngestionError
    )
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.config import get_cli_setting
except ImportError as e:
    print(f"Error importing tldw_chatbook modules: {e}")
    print("Make sure tldw_chatbook is installed or run this script from the project directory")
    sys.exit(1)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def print_supported_types():
    """Print all supported file types."""
    extensions = get_supported_extensions()
    print("\nSupported file types:")
    for media_type, exts in extensions.items():
        print(f"  {media_type}: {', '.join(exts)}")
    print()


def scan_directory(directory: Path, recursive: bool, file_types: Optional[List[str]], 
                  exclude_patterns: Optional[List[str]]) -> List[Path]:
    """Scan directory and return list of files to process."""
    import fnmatch
    
    # Get all files
    if recursive:
        all_files = list(directory.rglob('*'))
    else:
        all_files = list(directory.glob('*'))
    
    # Filter to only files
    files = [f for f in all_files if f.is_file()]
    
    # Get supported extensions
    supported_extensions = get_supported_extensions()
    
    # Build extension filter
    valid_extensions = set()
    if file_types:
        for ft in file_types:
            if ft in supported_extensions:
                valid_extensions.update(supported_extensions[ft])
    else:
        for exts in supported_extensions.values():
            valid_extensions.update(exts)
    
    # Filter by extension
    filtered_files = [f for f in files if f.suffix.lower() in valid_extensions]
    
    # Apply exclude patterns
    if exclude_patterns:
        final_files = []
        for file_path in filtered_files:
            excluded = False
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(file_path.name, pattern):
                    excluded = True
                    break
            if not excluded:
                final_files.append(file_path)
        filtered_files = final_files
    
    return sorted(filtered_files)


def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """Create a simple progress bar."""
    if total == 0:
        return "[" + "=" * width + "]"
    
    progress = current / total
    filled = int(width * progress)
    bar = "[" + "=" * filled + "-" * (width - filled) + "]"
    percentage = f"{progress * 100:.1f}%"
    return f"{bar} {percentage} ({current}/{total})"


def main():
    parser = argparse.ArgumentParser(
        description="Mass ingest media files into tldw_chatbook database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing media files to ingest"
    )
    
    # Optional arguments
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively process subdirectories"
    )
    
    parser.add_argument(
        "-t", "--types",
        nargs="+",
        choices=['pdf', 'document', 'ebook', 'xml', 'html', 'plaintext'],
        help="Only process specific file types"
    )
    
    parser.add_argument(
        "-k", "--keywords",
        nargs="+",
        help="Keywords to add to all ingested files"
    )
    
    parser.add_argument(
        "-e", "--exclude",
        nargs="+",
        help="File patterns to exclude (e.g., '*.tmp', 'draft_*')"
    )
    
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Perform analysis/summarization on ingested content"
    )
    
    parser.add_argument(
        "--api-name",
        help="API provider for analysis (e.g., openai, anthropic)"
    )
    
    parser.add_argument(
        "--api-key",
        help="API key for analysis provider"
    )
    
    parser.add_argument(
        "--chunk-method",
        choices=['semantic', 'tokens', 'paragraphs', 'sentences', 'words'],
        help="Chunking method to use"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Size of chunks"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        help="Overlap between chunks"
    )
    
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing if a file fails"
    )
    
    parser.add_argument(
        "--db-path",
        type=Path,
        help="Custom path to media database"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be ingested without actually doing it"
    )
    
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="List all supported file types and exit"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    # Handle --list-types before full parsing (since it doesn't need directory)
    if '--list-types' in sys.argv:
        print_supported_types()
        return 0
    
    args = parser.parse_args()
    
    # Validate directory
    if not args.directory.exists():
        print(f"Error: Directory '{args.directory}' does not exist")
        return 1
    
    if not args.directory.is_dir():
        print(f"Error: '{args.directory}' is not a directory")
        return 1
    
    # Scan for files
    print(f"\nScanning directory: {args.directory}")
    if args.recursive:
        print("Mode: Recursive")
    
    files = scan_directory(args.directory, args.recursive, args.types, args.exclude)
    
    if not files:
        print("No supported files found to ingest")
        return 0
    
    # Display found files
    total_size = sum(f.stat().st_size for f in files)
    print(f"\nFound {len(files)} files to process ({format_file_size(total_size)} total)")
    
    if args.verbose or args.dry_run:
        print("\nFiles to process:")
        for f in files:
            size = format_file_size(f.stat().st_size)
            print(f"  - {f.name} ({size})")
    
    if args.dry_run:
        print("\nDry run complete. No files were ingested.")
        return 0
    
    # Confirm before proceeding
    if not args.continue_on_error and len(files) > 10:
        response = input(f"\nProceed with ingesting {len(files)} files? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0
    
    # Get database path
    if args.db_path:
        db_path = args.db_path
    else:
        db_config = get_cli_setting("database", {})
        db_path = Path(db_config.get("media_db_path", "~/.local/share/tldw_cli/tldw_cli_media_v2.db")).expanduser()
    
    print(f"\nUsing database: {db_path}")
    
    # Initialize database
    try:
        media_db = MediaDatabase(str(db_path), client_id="mass_ingest")
    except Exception as e:
        print(f"Error initializing database: {e}")
        return 1
    
    # Prepare chunk options
    chunk_options = {}
    if args.chunk_method:
        chunk_options['method'] = args.chunk_method
    if args.chunk_size:
        chunk_options['size'] = args.chunk_size
    if args.chunk_overlap:
        chunk_options['overlap'] = args.chunk_overlap
    
    # Start ingestion
    print("\nStarting ingestion...")
    print("-" * 60)
    
    successful = 0
    failed = 0
    results = []
    
    start_time = datetime.now()
    
    try:
        for i, file_path in enumerate(files, 1):
            # Progress indicator
            print(f"\n{create_progress_bar(i, len(files))}")
            print(f"Processing: {file_path.name}")
            
            try:
                result = ingest_local_file(
                    file_path=file_path,
                    media_db=media_db,
                    keywords=args.keywords,
                    perform_analysis=args.analyze,
                    api_name=args.api_name,
                    api_key=args.api_key,
                    chunk_options=chunk_options if chunk_options else None
                )
                
                successful += 1
                results.append({
                    'file': str(file_path),
                    'success': True,
                    'media_id': result.get('media_id'),
                    'title': result.get('title'),
                    'chunks': result.get('chunks_created', 0)
                })
                
                if args.verbose:
                    print(f"  ✓ Success: ID={result.get('media_id')}, "
                          f"Title='{result.get('title')}', "
                          f"Chunks={result.get('chunks_created', 0)}")
                else:
                    print(f"  ✓ Success")
                
            except Exception as e:
                failed += 1
                error_msg = str(e)
                results.append({
                    'file': str(file_path),
                    'success': False,
                    'error': error_msg
                })
                
                print(f"  ✗ Failed: {error_msg}")
                
                if not args.continue_on_error:
                    print("\nStopping due to error. Use --continue-on-error to continue.")
                    break
    
    finally:
        media_db.close_connection()
    
    # Print summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {successful + failed}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Time taken: {duration:.1f} seconds")
    if successful > 0:
        print(f"Average time per file: {duration/successful:.1f} seconds")
    
    # Save detailed results to JSON file
    if args.verbose:
        results_file = Path(f"ingestion_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump({
                'summary': {
                    'directory': str(args.directory),
                    'recursive': args.recursive,
                    'total_files': len(files),
                    'successful': successful,
                    'failed': failed,
                    'duration_seconds': duration,
                    'timestamp': datetime.now().isoformat()
                },
                'results': results
            }, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)