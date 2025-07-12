# tldw_chatbook/Utils/note_importers.py
# Note Import Handlers for multiple file formats

import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
from loguru import logger

@dataclass
class ParsedNote:
    """Represents a parsed note with all possible fields."""
    title: str
    content: str
    template: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_template: bool = False  # If True, import as template not note
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            'title': self.title,
            'content': self.content,
            'template': self.template,
            'keywords': self.keywords,
            'metadata': self.metadata,
            'is_template': self.is_template
        }

class NoteImporter(ABC):
    """Base class for note importers."""
    
    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        """Check if this importer can handle the given file."""
        pass
    
    @abstractmethod
    def parse_file(self, file_path: Path) -> List[ParsedNote]:
        """Parse file and return list of notes."""
        pass
    
    def validate_note(self, note: ParsedNote) -> Tuple[bool, Optional[str]]:
        """Validate a parsed note. Returns (is_valid, error_message)."""
        if not note.title or not note.title.strip():
            return False, "Note must have a non-empty title"
        if not note.content or not note.content.strip():
            return False, "Note must have non-empty content"
        return True, None

class PlainTextImporter(NoteImporter):
    """Import plain text files as notes."""
    
    SUPPORTED_EXTENSIONS = {'.txt', '.text', '.md', '.markdown', '.rst'}
    
    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def parse_file(self, file_path: Path) -> List[ParsedNote]:
        """Parse plain text file. Title is filename, content is file content."""
        try:
            content = file_path.read_text(encoding='utf-8')
            title = file_path.stem  # Filename without extension
            
            # For markdown files, try to extract title from first heading
            if file_path.suffix.lower() in {'.md', '.markdown'}:
                lines = content.split('\n')
                for line in lines[:10]:  # Check first 10 lines
                    if line.startswith('# '):
                        title = line[2:].strip()
                        break
            
            return [ParsedNote(
                title=title,
                content=content,
                created_at=datetime.fromtimestamp(file_path.stat().st_ctime),
                updated_at=datetime.fromtimestamp(file_path.stat().st_mtime)
            )]
        except Exception as e:
            logger.error(f"Error parsing plain text file {file_path}: {e}")
            raise ValueError(f"Failed to parse file: {e}")

class JSONImporter(NoteImporter):
    """Import JSON files containing notes or templates."""
    
    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.json'
    
    def parse_file(self, file_path: Path) -> List[ParsedNote]:
        """Parse JSON file containing note(s) or template(s)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            notes = []
            
            # Handle single note/template object
            if isinstance(data, dict):
                notes.append(self._parse_note_dict(data))
            
            # Handle array of notes/templates
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        notes.append(self._parse_note_dict(item))
            
            else:
                raise ValueError("JSON must contain an object or array of objects")
            
            return notes
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {file_path}: {e}")
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            logger.error(f"Error parsing JSON file {file_path}: {e}")
            raise ValueError(f"Failed to parse file: {e}")
    
    def _parse_note_dict(self, data: Dict[str, Any]) -> ParsedNote:
        """Parse a single note/template dictionary."""
        # Check if this is a template definition
        is_template = data.get('is_template', False) or data.get('type') == 'template'
        
        # Extract fields with defaults
        title = data.get('title', data.get('name', 'Untitled'))
        content = data.get('content', data.get('body', ''))
        
        # For templates, content might be in 'template' field
        if is_template and 'template' in data and not content:
            content = data['template']
        
        # Extract optional fields
        template = data.get('template') if not is_template else None
        keywords = data.get('keywords', data.get('tags', []))
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(',')]
        
        # Extract timestamps
        created_at = self._parse_timestamp(data.get('created_at', data.get('created')))
        updated_at = self._parse_timestamp(data.get('updated_at', data.get('updated')))
        
        # Store any extra fields in metadata
        standard_fields = {'title', 'name', 'content', 'body', 'template', 'keywords', 
                          'tags', 'created_at', 'created', 'updated_at', 'updated', 
                          'is_template', 'type'}
        metadata = {k: v for k, v in data.items() if k not in standard_fields}
        
        return ParsedNote(
            title=title,
            content=content,
            template=template,
            keywords=keywords,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
            is_template=is_template
        )
    
    def _parse_timestamp(self, value: Any) -> Optional[datetime]:
        """Parse timestamp from various formats."""
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except:
                pass
        return None

class YAMLImporter(NoteImporter):
    """Import YAML files containing notes or templates."""
    
    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {'.yaml', '.yml'}
    
    def parse_file(self, file_path: Path) -> List[ParsedNote]:
        """Parse YAML file containing note(s) or template(s)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Reuse JSON importer logic for parsing
            json_importer = JSONImporter()
            notes = []
            
            if isinstance(data, dict):
                notes.append(json_importer._parse_note_dict(data))
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        notes.append(json_importer._parse_note_dict(item))
            else:
                raise ValueError("YAML must contain an object or array of objects")
            
            return notes
            
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in file {file_path}: {e}")
            raise ValueError(f"Invalid YAML format: {e}")
        except Exception as e:
            logger.error(f"Error parsing YAML file {file_path}: {e}")
            raise ValueError(f"Failed to parse file: {e}")

class CSVImporter(NoteImporter):
    """Import CSV files as notes."""
    
    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.csv'
    
    def parse_file(self, file_path: Path) -> List[ParsedNote]:
        """Parse CSV file. Each row becomes a note."""
        import csv
        
        try:
            notes = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try to map common column names
                    title = row.get('title', row.get('Title', row.get('name', row.get('Name', ''))))
                    content = row.get('content', row.get('Content', row.get('body', row.get('Body', ''))))
                    
                    # If no title/content columns, use first two columns
                    if not title and not content and reader.fieldnames:
                        values = list(row.values())
                        title = values[0] if len(values) > 0 else ''
                        content = values[1] if len(values) > 1 else ''
                    
                    if title or content:
                        notes.append(ParsedNote(
                            title=title or 'Untitled',
                            content=content or '',
                            metadata=row  # Store all columns as metadata
                        ))
            
            return notes
            
        except Exception as e:
            logger.error(f"Error parsing CSV file {file_path}: {e}")
            raise ValueError(f"Failed to parse CSV file: {e}")

class NoteImporterRegistry:
    """Registry for note importers."""
    
    def __init__(self):
        self.importers = [
            JSONImporter(),
            YAMLImporter(),
            PlainTextImporter(),
            CSVImporter(),
        ]
    
    def parse_file(self, file_path: Union[str, Path], import_as_template: bool = False) -> List[ParsedNote]:
        """Parse a file using the appropriate importer."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        for importer in self.importers:
            if importer.can_handle(file_path):
                notes = importer.parse_file(file_path)
                
                # Override is_template if specified
                if import_as_template:
                    for note in notes:
                        note.is_template = True
                
                # Validate all notes
                validated_notes = []
                for note in notes:
                    is_valid, error = importer.validate_note(note)
                    if is_valid:
                        validated_notes.append(note)
                    else:
                        logger.warning(f"Skipping invalid note '{note.title}': {error}")
                
                return validated_notes
        
        raise ValueError(f"No importer available for file type: {file_path.suffix}")
    
    def get_supported_extensions(self) -> List[str]:
        """Get all supported file extensions."""
        extensions = set()
        for importer in self.importers:
            if hasattr(importer, 'SUPPORTED_EXTENSIONS'):
                extensions.update(importer.SUPPORTED_EXTENSIONS)
            elif isinstance(importer, JSONImporter):
                extensions.add('.json')
            elif isinstance(importer, YAMLImporter):
                extensions.update({'.yaml', '.yml'})
            elif isinstance(importer, CSVImporter):
                extensions.add('.csv')
        return sorted(list(extensions))

# Global registry instance
note_importer_registry = NoteImporterRegistry()