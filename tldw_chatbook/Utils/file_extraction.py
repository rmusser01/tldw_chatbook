"""
File extraction utilities for extracting code blocks and files from LLM responses.
"""
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFile:
    """Represents an extracted file from LLM response."""
    filename: str
    content: str
    language: str
    start_pos: int
    end_pos: int


class FileExtractor:
    """Extract files from LLM responses with code blocks."""
    
    # Pattern to match code blocks with optional language
    CODE_BLOCK_PATTERN = r'```(?P<lang>\w+)?\n(?P<content>.*?)```'
    
    # Patterns to detect filename hints in context
    FILENAME_HINTS = [
        r'(?:file|filename|save as|create):\s*([^\n]+)',
        r'#\s*([^\n]+\.\w+)',  # Comments with filenames
        r'<!--\s*([^\n]+\.\w+)\s*-->',  # HTML comments
        r'//\s*([^\n]+\.\w+)',  # C-style comments
    ]
    
    # Map languages to file extensions
    LANGUAGE_EXTENSIONS = {
        'python': '.py',
        'py': '.py',
        'javascript': '.js',
        'js': '.js',
        'typescript': '.ts',
        'ts': '.ts',
        'html': '.html',
        'css': '.css',
        'json': '.json',
        'yaml': '.yaml',
        'yml': '.yaml',
        'csv': '.csv',
        'svg': '.svg',
        'xml': '.xml',
        'sql': '.sql',
        'bash': '.sh',
        'sh': '.sh',
        'shell': '.sh',
        'markdown': '.md',
        'md': '.md',
        'toml': '.toml',
        'ini': '.ini',
        'env': '.env',
        'txt': '.txt',
        'text': '.txt',
        'jsx': '.jsx',
        'tsx': '.tsx',
        'cpp': '.cpp',
        'c': '.c',
        'h': '.h',
        'hpp': '.hpp',
        'java': '.java',
        'go': '.go',
        'rust': '.rs',
        'rs': '.rs',
        'swift': '.swift',
        'kotlin': '.kt',
        'kt': '.kt',
        'r': '.r',
        'R': '.R',
        'matlab': '.m',
        'lua': '.lua',
        'dockerfile': '',
        'makefile': '',
    }
    
    def extract_files(self, text: str) -> List[ExtractedFile]:
        """
        Extract all file-like content from text.
        
        Args:
            text: The text containing code blocks
            
        Returns:
            List of ExtractedFile objects
        """
        files = []
        
        # Find all code blocks
        for match in re.finditer(self.CODE_BLOCK_PATTERN, text, re.DOTALL):
            lang = match.group('lang') or 'text'
            content = match.group('content')
            
            # Skip empty content
            if not content.strip():
                continue
            
            # Try to find filename hint before the code block
            filename = self._find_filename_hint(text, match.start())
            
            if not filename:
                # Generate filename from language and index
                ext = self.LANGUAGE_EXTENSIONS.get(lang.lower(), '.txt')
                # Special handling for dockerfile/makefile
                if lang.lower() == 'dockerfile':
                    filename = 'Dockerfile'
                elif lang.lower() == 'makefile':
                    filename = 'Makefile'
                else:
                    filename = f"extracted_{len(files)+1}{ext}"
            
            files.append(ExtractedFile(
                filename=filename,
                content=content.rstrip('\n'),  # Remove trailing newline from code block
                language=lang,
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        return files
    
    def _find_filename_hint(self, text: str, code_block_start: int, search_lines: int = 5) -> Optional[str]:
        """
        Look for filename hints in the text before a code block.
        
        Args:
            text: Full text
            code_block_start: Starting position of the code block
            search_lines: Number of lines to search backwards
            
        Returns:
            Filename if found, None otherwise
        """
        # Get text before the code block
        before_text = text[:code_block_start]
        lines = before_text.split('\n')
        
        # Search the last few lines before the code block
        search_text = '\n'.join(lines[-search_lines:]) if len(lines) >= search_lines else before_text
        
        # Try each pattern
        for pattern in self.FILENAME_HINTS:
            matches = re.findall(pattern, search_text)
            if matches:
                # Get the last match (closest to code block)
                potential_filename = matches[-1].strip()
                # Basic validation
                if self._is_valid_filename(potential_filename):
                    return potential_filename
        
        return None
    
    def _is_valid_filename(self, filename: str) -> bool:
        """
        Check if a string is a valid filename.
        
        Args:
            filename: Potential filename
            
        Returns:
            True if valid filename
        """
        if not filename or len(filename) > 255:
            return False
        
        # Must have an extension or be a known extensionless file
        if '.' not in filename and filename.lower() not in ['dockerfile', 'makefile']:
            return False
        
        # Check for invalid characters
        invalid_chars = '<>:"|?*'
        if any(char in filename for char in invalid_chars):
            return False
        
        # Don't allow path separators
        if '/' in filename or '\\' in filename:
            return False
        
        return True
    
    def validate_content(self, file: ExtractedFile, max_size: int = 10 * 1024 * 1024) -> Optional[str]:
        """
        Validate extracted file content.
        
        Args:
            file: ExtractedFile to validate
            max_size: Maximum allowed file size in bytes
            
        Returns:
            Error message if validation fails, None if valid
        """
        # Check size
        if len(file.content.encode('utf-8')) > max_size:
            return f"File too large: {len(file.content)} bytes (max: {max_size})"
        
        # Validate specific file types
        ext = Path(file.filename).suffix.lower()
        
        if ext in ['.json']:
            try:
                json.loads(file.content)
            except json.JSONDecodeError as e:
                return f"Invalid JSON: {str(e)}"
        
        elif ext in ['.yaml', '.yml']:
            try:
                yaml.safe_load(file.content)
            except yaml.YAMLError as e:
                return f"Invalid YAML: {str(e)}"
        
        elif ext == '.csv':
            # Basic CSV validation - check if it has consistent columns
            lines = file.content.strip().split('\n')
            if lines:
                first_line_cols = len(lines[0].split(','))
                for i, line in enumerate(lines[1:], 2):
                    if line.strip() and len(line.split(',')) != first_line_cols:
                        logger.warning(f"CSV row {i} has inconsistent column count")
        
        return None