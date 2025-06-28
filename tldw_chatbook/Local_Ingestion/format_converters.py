# format_converters.py
"""
Format converters for various file types.
Provides extensible conversion system with graceful fallbacks.
"""

import csv
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from loguru import logger

# Import optional dependencies with fallback
from ..Utils.optional_deps import get_safe_import, check_dependency

# Required dependencies (in base requirements)
import yaml
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

# Optional dependencies
defusedxml = get_safe_import('defusedxml')
if defusedxml:
    from defusedxml.ElementTree import parse as safe_parse_xml
else:
    safe_parse_xml = ET.parse


class FormatConverter(ABC):
    """Base class for format converters."""
    
    @abstractmethod
    def convert(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        Convert file to text and extract metadata.
        
        Returns:
            Tuple of (converted_text, metadata_dict)
        """
        pass
    
    @abstractmethod
    def can_convert(self, file_path: Path) -> bool:
        """Check if this converter can handle the file."""
        pass
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """List of supported file extensions."""
        pass


class JSONConverter(FormatConverter):
    """Convert JSON files to readable text."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.json', '.jsonl']
    
    def can_convert(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.supported_extensions
    
    def convert(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Convert JSON to formatted text."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() == '.jsonl':
                    # Handle JSON Lines format
                    lines = []
                    for i, line in enumerate(f):
                        try:
                            data = json.loads(line.strip())
                            lines.append(json.dumps(data, indent=2, ensure_ascii=False))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON line {i+1}: {e}")
                    
                    text = '\n---\n'.join(lines)
                    metadata = {
                        'format': 'jsonl',
                        'line_count': len(lines)
                    }
                else:
                    # Regular JSON
                    data = json.load(f)
                    text = json.dumps(data, indent=2, ensure_ascii=False)
                    
                    metadata = {
                        'format': 'json',
                        'keys': list(data.keys()) if isinstance(data, dict) else None,
                        'items': len(data) if isinstance(data, (list, dict)) else None,
                        'type': type(data).__name__
                    }
            
            return text, metadata
            
        except Exception as e:
            logger.error(f"Failed to convert JSON file {file_path}: {e}")
            raise ValueError(f"JSON conversion failed: {str(e)}")


class CSVConverter(FormatConverter):
    """Convert CSV files to markdown tables."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.csv', '.tsv']
    
    def can_convert(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.supported_extensions
    
    def convert(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Convert CSV to markdown table."""
        try:
            delimiter = '\t' if file_path.suffix.lower() == '.tsv' else ','
            
            rows = []
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f, delimiter=delimiter)
                all_rows = list(reader)
            
            if not all_rows:
                return "Empty CSV file", {'format': 'csv', 'row_count': 0}
            
            # Get headers
            headers = all_rows[0] if all_rows else []
            
            # Create markdown table
            if headers:
                # Escape pipe characters in cells
                headers = [cell.replace('|', '\\|') for cell in headers]
                rows.append('| ' + ' | '.join(headers) + ' |')
                rows.append('|' + '---|' * len(headers))
                
                # Add data rows
                for row in all_rows[1:]:
                    # Ensure row has same number of columns as headers
                    while len(row) < len(headers):
                        row.append('')
                    row = row[:len(headers)]  # Truncate if too many columns
                    
                    # Escape pipe characters
                    row = [cell.replace('|', '\\|') for cell in row]
                    rows.append('| ' + ' | '.join(row) + ' |')
            
            text = '\n'.join(rows)
            
            metadata = {
                'format': 'csv' if delimiter == ',' else 'tsv',
                'delimiter': delimiter,
                'columns': headers,
                'column_count': len(headers),
                'row_count': len(all_rows) - 1  # Excluding header
            }
            
            return text, metadata
            
        except Exception as e:
            logger.error(f"Failed to convert CSV file {file_path}: {e}")
            raise ValueError(f"CSV conversion failed: {str(e)}")


class YAMLConverter(FormatConverter):
    """Convert YAML files to readable format."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.yaml', '.yml']
    
    def can_convert(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.supported_extensions
    
    def convert(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Convert YAML to formatted text."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Convert back to YAML with nice formatting
            text = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            metadata = {
                'format': 'yaml',
                'keys': list(data.keys()) if isinstance(data, dict) else None,
                'type': type(data).__name__,
                'document_count': 1  # Could be extended for multi-document YAML
            }
            
            return text, metadata
            
        except Exception as e:
            logger.error(f"Failed to convert YAML file {file_path}: {e}")
            raise ValueError(f"YAML conversion failed: {str(e)}")


class XMLEnhancedConverter(FormatConverter):
    """Enhanced XML converter with better formatting."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.xml']
    
    def can_convert(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.supported_extensions
    
    def convert(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Convert XML to structured text."""
        try:
            # Use defusedxml if available for security
            if defusedxml:
                tree = safe_parse_xml(str(file_path))
            else:
                tree = ET.parse(str(file_path))
            
            root = tree.getroot()
            
            # Convert to hierarchical text representation
            text_lines = []
            self._element_to_text(root, text_lines, level=0)
            text = '\n'.join(text_lines)
            
            # Extract metadata
            metadata = {
                'format': 'xml',
                'root_tag': root.tag,
                'root_attributes': dict(root.attrib),
                'encoding': tree.docinfo.encoding if hasattr(tree, 'docinfo') else 'utf-8',
                'element_count': len(root.findall('.//*'))
            }
            
            return text, metadata
            
        except Exception as e:
            logger.error(f"Failed to convert XML file {file_path}: {e}")
            raise ValueError(f"XML conversion failed: {str(e)}")
    
    def _element_to_text(self, element: ET.Element, lines: List[str], level: int = 0):
        """Recursively convert XML element to text lines."""
        indent = '  ' * level
        
        # Start tag with attributes
        tag_parts = [element.tag]
        if element.attrib:
            attrs = ' '.join(f'{k}="{v}"' for k, v in element.attrib.items())
            tag_parts.append(f"[{attrs}]")
        
        lines.append(f"{indent}{' '.join(tag_parts)}:")
        
        # Element text
        if element.text and element.text.strip():
            lines.append(f"{indent}  {element.text.strip()}")
        
        # Child elements
        for child in element:
            self._element_to_text(child, lines, level + 1)
        
        # Tail text
        if element.tail and element.tail.strip():
            lines.append(f"{indent}{element.tail.strip()}")


class FormatRegistry:
    """Registry for format converters with priority and fallback support."""
    
    def __init__(self):
        self.converters: Dict[str, List[Tuple[FormatConverter, int]]] = {}
        self._register_default_converters()
    
    def _register_default_converters(self):
        """Register built-in converters."""
        # Core converters (always available)
        self.register_converter(JSONConverter(), priority=100)
        self.register_converter(CSVConverter(), priority=100)
        self.register_converter(YAMLConverter(), priority=100)
        self.register_converter(XMLEnhancedConverter(), priority=90)
        
        # Optional converters (check dependencies)
        self._register_optional_converters()
    
    def _register_optional_converters(self):
        """Register converters that depend on optional packages."""
        # Excel support
        if check_dependency('openpyxl'):
            from .excel_converter import ExcelConverter
            self.register_converter(ExcelConverter(), priority=90)
        
        # PowerPoint support
        if check_dependency('python-pptx'):
            from .pptx_converter import PowerPointConverter
            self.register_converter(PowerPointConverter(), priority=90)
        
        # Enhanced PDF support
        if check_dependency('pdfplumber'):
            from .pdf_converter import PDFPlumberConverter
            self.register_converter(PDFPlumberConverter(), priority=80)
    
    def register_converter(self, converter: FormatConverter, priority: int = 50):
        """Register a converter with priority."""
        for ext in converter.supported_extensions:
            if ext not in self.converters:
                self.converters[ext] = []
            
            self.converters[ext].append((converter, priority))
            # Sort by priority (higher first)
            self.converters[ext].sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Registered {converter.__class__.__name__} for extensions: {converter.supported_extensions}")
    
    def can_convert(self, file_path: Path) -> bool:
        """Check if any converter can handle this file."""
        extension = file_path.suffix.lower()
        return extension in self.converters
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get all supported formats grouped by converter."""
        supported = {}
        for ext, converters in self.converters.items():
            for converter, priority in converters:
                converter_name = converter.__class__.__name__
                if converter_name not in supported:
                    supported[converter_name] = []
                supported[converter_name].append(ext)
        return supported
    
    def convert(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Convert file using appropriate converter."""
        extension = file_path.suffix.lower()
        
        if extension not in self.converters:
            raise ValueError(f"No converter registered for {extension} files")
        
        errors = []
        for converter, priority in self.converters[extension]:
            if converter.can_convert(file_path):
                try:
                    logger.debug(f"Trying {converter.__class__.__name__} for {file_path.name}")
                    return converter.convert(file_path)
                except Exception as e:
                    error_msg = f"{converter.__class__.__name__}: {str(e)}"
                    errors.append(error_msg)
                    logger.warning(f"Converter failed: {error_msg}")
        
        # All converters failed
        raise ValueError(f"All converters failed for {file_path.name}:\n" + '\n'.join(errors))


# Global registry instance
_global_registry = None


def get_format_registry() -> FormatRegistry:
    """Get or create the global format registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = FormatRegistry()
    return _global_registry


def convert_file(file_path: Path) -> Tuple[str, Dict[str, Any]]:
    """Convenience function to convert a file using the global registry."""
    return get_format_registry().convert(file_path)


def get_supported_formats() -> Dict[str, List[str]]:
    """Get all supported file formats."""
    return get_format_registry().get_supported_formats()


def can_convert_file(file_path: Path) -> bool:
    """Check if a file can be converted."""
    return get_format_registry().can_convert(file_path)