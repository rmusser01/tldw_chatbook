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
    CODE_BLOCK_PATTERN = r'```(?P<lang>[\w-]+)?\n(?P<content>.*?)```'
    
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
        'tsv': '.tsv',
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
        'vcf': '.vcf',
        'vcard': '.vcf',
        'ics': '.ics',
        'ical': '.ics',
        'calendar': '.ics',
        'gpx': '.gpx',
        'kml': '.kml',
        'dot': '.dot',
        'graphviz': '.dot',
        'puml': '.puml',
        'plantuml': '.puml',
        'mmd': '.mmd',
        'mermaid': '.mmd',
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
        # Jupyter notebooks
        'ipynb': '.ipynb',
        'jupyter': '.ipynb',
        # Infrastructure as Code
        'terraform': '.tf',
        'tf': '.tf',
        'hcl': '.tf',
        'tfvars': '.tfvars',
        # Container orchestration
        'docker-compose': '.yml',
        'compose': '.yml',
        # CI/CD
        'github-actions': '.yml',
        'gitlab-ci': '.yml',
        'jenkinsfile': '',
        'travis': '.yml',
        # Dependencies/Package files
        'requirements': '.txt',
        'pipfile': '',
        'gemfile': '',
        'cargo': '.toml',
        'package': '.json',
        'composer': '.json',
        # Configuration files
        'gitignore': '',
        'htaccess': '',
        'service': '.service',
        'systemd': '.service',
        # API definitions
        'proto': '.proto',
        'protobuf': '.proto',
        'graphql': '.graphql',
        'gql': '.graphql',
        'openapi': '.yaml',
        'swagger': '.yaml',
        # Data formats
        'ndjson': '.ndjson',
        'jsonl': '.jsonl',
        'parquet': '.parquet',
        'avro': '.avro',
        'jsonld': '.jsonld',
        'geojson': '.geojson',
        'rdf': '.rdf',
        'ttl': '.ttl',
        'xsd': '.xsd',
        
        # Configuration files
        'conf': '.conf',
        'config': '.conf',
        'cfg': '.cfg',
        'properties': '.properties',
        'gradle': '.gradle',
        'sbt': '.sbt',
        'cmake': '.cmake',
        'pri': '.pri',
        'pro': '.pro',
        
        # Template files
        'hbs': '.hbs',
        'handlebars': '.handlebars',
        'ejs': '.ejs',
        'pug': '.pug',
        'jade': '.jade',
        'liquid': '.liquid',
        'mustache': '.mustache',
        'njk': '.njk',
        'nunjucks': '.njk',
        'jinja': '.j2',
        'jinja2': '.j2',
        'j2': '.j2',
        
        # Script files
        'psm1': '.psm1',
        'psd1': '.psd1',
        'ps1': '.ps1',
        'bat': '.bat',
        'cmd': '.cmd',
        'awk': '.awk',
        'sed': '.sed',
        'vim': '.vim',
        'vimrc': '.vimrc',
        'el': '.el',
        'lisp': '.lisp',
        'scm': '.scm',
        'rkt': '.rkt',
        
        # Programming languages
        'dart': '.dart',
        'scala': '.scala',
        'sc': '.scala',
        'clj': '.clj',
        'cljs': '.cljs',
        'cljc': '.cljc',
        'ex': '.ex',
        'exs': '.exs',
        'erl': '.erl',
        'hrl': '.hrl',
        'nim': '.nim',
        'nims': '.nims',
        'zig': '.zig',
        'v': '.v',
        'vsh': '.vsh',
        'jl': '.jl',
        'julia': '.jl',
        'pas': '.pas',
        'pp': '.pp',
        'inc': '.inc',
        'hs': '.hs',
        'lhs': '.lhs',
        'elm': '.elm',
        'purs': '.purs',
        'idr': '.idr',
        'agda': '.agda',
        'lean': '.lean',
        'coq': '.v',
        'ml': '.ml',
        'mli': '.mli',
        'fs': '.fs',
        'fsx': '.fsx',
        'fsi': '.fsi',
        'fsscript': '.fsx',
        
        # Web Assembly & Low Level
        'wat': '.wat',
        'wasm': '.wasm',
        'll': '.ll',
        'llvm': '.ll',
        's': '.s',
        'asm': '.asm',
        'nasm': '.nasm',
        'masm': '.masm',
        
        # Documentation formats
        'texi': '.texi',
        'texinfo': '.texinfo',
        'man': '.man',
        'rdoc': '.rdoc',
        'pod': '.pod',
        'adoc': '.adoc',
        'asciidoc': '.adoc',
        'org': '.org',
        
        # Build & Project files
        'proj': '.proj',
        'csproj': '.csproj',
        'vbproj': '.vbproj',
        'fsproj': '.fsproj',
        'vcxproj': '.vcxproj',
        'vcproj': '.vcproj',
        'sln': '.sln',
        'cabal': '.cabal',
        'mix': '.mix',
        'rebar': '.config',
        'bazel': '.bazel',
        'bzl': '.bzl',
        'buck': '.buck',
        'pants': '.pants',
        
        # API & Testing
        'http': '.http',
        'rest': '.rest',
        'feature': '.feature',
        'spec': '.spec',
        
        # Other development files
        'env': '.env',
        'example': '.example',
        'sample': '.sample',
        'tmpl': '.tmpl',
        'tpl': '.tpl',
        'in': '.in',
        'ac': '.ac',
        'am': '.am',
        'm4': '.m4',
        'mk': '.mk',
        'mak': '.mak',
        'rakefile': '',
        'guardfile': '',
        'capfile': '',
        'vagrantfile': '',
        'berksfile': '',
        'appfile': '',
        'deliverfile': '',
        'fastfile': '',
        'scanfile': '',
        'snapfile': '',
        'gymfile': '',
        'matchfile': '',
        'podfile': '',
        'cartfile': '',
        'jazzy': '.yaml',
        'mintfile': '',
        'brewfile': '',
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
        
        # First, extract markdown tables as CSV files
        table_files = self._extract_markdown_tables(text)
        files.extend(table_files)
        
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
                # Special handling for files without extensions
                if lang.lower() == 'dockerfile':
                    filename = 'Dockerfile'
                elif lang.lower() == 'makefile':
                    filename = 'Makefile'
                elif lang.lower() == 'jenkinsfile':
                    filename = 'Jenkinsfile'
                elif lang.lower() == 'pipfile':
                    filename = 'Pipfile'
                elif lang.lower() == 'gemfile':
                    filename = 'Gemfile'
                elif lang.lower() == 'gitignore':
                    filename = '.gitignore'
                elif lang.lower() == 'htaccess':
                    filename = '.htaccess'
                # Special handling for specific config files
                elif lang.lower() == 'docker-compose':
                    filename = 'docker-compose.yml'
                elif lang.lower() == 'github-actions':
                    filename = 'workflow.yml'
                elif lang.lower() == 'gitlab-ci':
                    filename = '.gitlab-ci.yml'
                elif lang.lower() == 'travis':
                    filename = '.travis.yml'
                elif lang.lower() == 'requirements':
                    filename = 'requirements.txt'
                elif lang.lower() == 'package':
                    filename = 'package.json'
                elif lang.lower() == 'composer':
                    filename = 'composer.json'
                elif lang.lower() == 'cargo':
                    filename = 'Cargo.toml'
                # Additional special cases for files without extensions
                elif lang.lower() == 'rakefile':
                    filename = 'Rakefile'
                elif lang.lower() == 'guardfile':
                    filename = 'Guardfile'
                elif lang.lower() == 'capfile':
                    filename = 'Capfile'
                elif lang.lower() == 'vagrantfile':
                    filename = 'Vagrantfile'
                elif lang.lower() == 'berksfile':
                    filename = 'Berksfile'
                elif lang.lower() == 'appfile':
                    filename = 'Appfile'
                elif lang.lower() == 'deliverfile':
                    filename = 'Deliverfile'
                elif lang.lower() == 'fastfile':
                    filename = 'Fastfile'
                elif lang.lower() == 'scanfile':
                    filename = 'Scanfile'
                elif lang.lower() == 'snapfile':
                    filename = 'Snapfile'
                elif lang.lower() == 'gymfile':
                    filename = 'Gymfile'
                elif lang.lower() == 'matchfile':
                    filename = 'Matchfile'
                elif lang.lower() == 'podfile':
                    filename = 'Podfile'
                elif lang.lower() == 'cartfile':
                    filename = 'Cartfile'
                elif lang.lower() == 'mintfile':
                    filename = 'Mintfile'
                elif lang.lower() == 'brewfile':
                    filename = 'Brewfile'
                # Special config files with specific names
                elif lang.lower() == 'nginx':
                    filename = 'nginx.conf'
                elif lang.lower() == 'apache':
                    filename = 'httpd.conf'
                elif lang.lower() == 'redis':
                    filename = 'redis.conf'
                elif lang.lower() == 'mysql':
                    filename = 'my.cnf'
                elif lang.lower() == 'postgresql':
                    filename = 'postgresql.conf'
                elif lang.lower() == 'ssh':
                    filename = 'ssh_config'
                elif lang.lower() == 'eslint':
                    filename = '.eslintrc.json'
                elif lang.lower() == 'prettier':
                    filename = '.prettierrc'
                elif lang.lower() == 'babel':
                    filename = '.babelrc'
                elif lang.lower() == 'editorconfig':
                    filename = '.editorconfig'
                elif lang.lower() == 'jest':
                    filename = 'jest.config.js'
                elif lang.lower() == 'webpack':
                    filename = 'webpack.config.js'
                elif lang.lower() == 'rollup':
                    filename = 'rollup.config.js'
                elif lang.lower() == 'vite':
                    filename = 'vite.config.js'
                else:
                    filename = f"extracted_{len(files)+1}{ext}"
            
            # Clean up content based on file type
            if lang in ['csv', 'tsv']:
                # For CSV/TSV, strip each line but preserve internal structure
                lines = content.split('\n')
                cleaned_lines = [line.strip() for line in lines if line.strip()]
                content = '\n'.join(cleaned_lines)
            else:
                content = content.rstrip('\n')  # Remove trailing newline from code block
            
            files.append(ExtractedFile(
                filename=filename,
                content=content,
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
    
    def _extract_markdown_tables(self, text: str) -> List[ExtractedFile]:
        """
        Extract markdown tables and convert them to CSV files.
        
        Args:
            text: The text potentially containing markdown tables
            
        Returns:
            List of ExtractedFile objects for each table found
        """
        files = []
        
        # Pattern to match markdown tables
        # This matches tables with header separator (---|---|---)
        table_pattern = r'(\|[^\n]+\|\s*\n\s*\|[\s\-:|]+\|\s*\n(?:\s*\|[^\n]+\|\s*\n)*)'
        
        for i, match in enumerate(re.finditer(table_pattern, text, re.MULTILINE)):
            table_text = match.group(0)
            
            # Convert markdown table to CSV
            csv_content = self._markdown_table_to_csv(table_text)
            
            if csv_content:
                # Try to find a title/caption near the table
                before_text = text[max(0, match.start()-200):match.start()]
                filename_hint = self._find_table_title(before_text)
                
                if not filename_hint:
                    filename_hint = f"table_{i+1}.csv"
                elif not filename_hint.endswith('.csv'):
                    filename_hint = filename_hint.replace(' ', '_') + '.csv'
                
                files.append(ExtractedFile(
                    filename=filename_hint,
                    content=csv_content,
                    language='csv',
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        return files
    
    def _markdown_table_to_csv(self, table_text: str) -> str:
        """
        Convert a markdown table to CSV format.
        
        Args:
            table_text: The markdown table text
            
        Returns:
            CSV formatted string
        """
        import csv
        import io
        
        lines = table_text.strip().split('\n')
        if len(lines) < 3:  # Need at least header, separator, and one data row
            return ""
        
        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer)
        
        # Process each row
        for i, line in enumerate(lines):
            if i == 1:  # Skip separator line
                continue
                
            # Extract cells from markdown table row
            cells = []
            # Remove leading/trailing pipes and split
            row = line.strip()
            if row.startswith('|'):
                row = row[1:]
            if row.endswith('|'):
                row = row[:-1]
                
            # Split by pipe and clean each cell
            for cell in row.split('|'):
                cells.append(cell.strip())
            
            csv_writer.writerow(cells)
        
        return csv_buffer.getvalue()
    
    def _find_table_title(self, text: str) -> Optional[str]:
        """
        Try to find a title or caption for a table in the preceding text.
        
        Args:
            text: Text before the table
            
        Returns:
            Potential filename or None
        """
        lines = text.strip().split('\n')
        
        # Look for patterns like "Table: Title" or "## Title" near the end
        for line in reversed(lines[-5:]):  # Check last 5 lines
            line = line.strip()
            
            # Check for explicit table labels
            if line.lower().startswith(('table:', 'table -', 'table.')):
                title = line.split(':', 1)[-1].strip()
                return self._sanitize_filename(title)
            
            # Check for headings that might be table titles
            if line.startswith('#') and len(line) > 2:
                title = line.lstrip('#').strip()
                # Only use if it looks like a table title
                if any(word in title.lower() for word in ['table', 'data', 'results', 'summary', 'report']):
                    return self._sanitize_filename(title)
        
        return None
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a string to be a valid filename.
        
        Args:
            filename: Raw filename string
            
        Returns:
            Sanitized filename
        """
        # Remove invalid characters
        invalid_chars = '<>:"|?*\\/\n\r\t'
        for char in invalid_chars:
            filename = filename.replace(char, '')
        
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        
        # Limit length
        if len(filename) > 100:
            filename = filename[:100]
        
        return filename
    
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
        
        # Check for specific filenames first before generic extension checks
        if file.filename.lower() == 'package.json':
            # package.json validation
            try:
                pkg_data = json.loads(file.content)
                if 'name' not in pkg_data:
                    return "Invalid package.json: Missing 'name' field"
                if 'version' not in pkg_data:
                    return "Invalid package.json: Missing 'version' field"
            except json.JSONDecodeError as e:
                return f"Invalid package.json: {str(e)}"
        
        elif file.filename.lower() == 'composer.json':
            # composer.json validation
            try:
                composer_data = json.loads(file.content)
                if 'require' not in composer_data and 'require-dev' not in composer_data:
                    logger.warning("composer.json has no dependencies defined")
            except json.JSONDecodeError as e:
                return f"Invalid composer.json: {str(e)}"
        
        elif ext in ['.json']:
            try:
                json.loads(file.content)
            except json.JSONDecodeError as e:
                return f"Invalid JSON: {str(e)}"
        
        elif ext in ['.yaml', '.yml']:
            try:
                yaml.safe_load(file.content)
            except yaml.YAMLError as e:
                return f"Invalid YAML: {str(e)}"
        
        elif ext in ['.csv', '.tsv']:
            # Enhanced CSV/TSV validation
            import csv
            import io
            
            delimiter = '\t' if ext == '.tsv' else ','
            try:
                reader = csv.reader(io.StringIO(file.content), delimiter=delimiter)
                rows = list(reader)
                
                if not rows:
                    return f"Empty {ext.upper()} file"
                
                # Check for consistent column count
                header_cols = len(rows[0])
                for i, row in enumerate(rows[1:], 2):
                    if len(row) != header_cols:
                        return f"{ext.upper()} row {i} has {len(row)} columns, expected {header_cols}"
                        
            except csv.Error as e:
                return f"Invalid {ext.upper()} format: {str(e)}"
        
        elif ext == '.sql':
            # Basic SQL validation - check for dangerous operations
            sql_lower = file.content.lower()
            dangerous_keywords = ['drop database', 'drop schema', 'delete from', 'truncate']
            for keyword in dangerous_keywords:
                if keyword in sql_lower:
                    logger.warning(f"SQL file contains potentially dangerous operation: {keyword}")
        
        elif ext == '.xml':
            # Basic XML validation
            try:
                import xml.etree.ElementTree as ET
                ET.fromstring(file.content)
            except ET.ParseError as e:
                return f"Invalid XML: {str(e)}"
        
        elif ext in ['.vcf', '.vcard']:
            # Basic vCard validation
            if not file.content.strip().startswith('BEGIN:VCARD'):
                return "Invalid vCard: Must start with BEGIN:VCARD"
            if not file.content.strip().endswith('END:VCARD'):
                return "Invalid vCard: Must end with END:VCARD"
        
        elif ext in ['.ics', '.ical']:
            # Basic iCalendar validation
            if not file.content.strip().startswith('BEGIN:VCALENDAR'):
                return "Invalid iCalendar: Must start with BEGIN:VCALENDAR"
            if not file.content.strip().endswith('END:VCALENDAR'):
                return "Invalid iCalendar: Must end with END:VCALENDAR"
        
        elif ext == '.dot':
            # Basic GraphViz validation
            if not any(keyword in file.content for keyword in ['digraph', 'graph', 'subgraph']):
                return "Invalid GraphViz: Must contain graph definition"
        
        elif ext in ['.puml', '.plantuml']:
            # Basic PlantUML validation
            if not file.content.strip().startswith('@startuml'):
                return "Invalid PlantUML: Must start with @startuml"
            if not file.content.strip().endswith('@enduml'):
                return "Invalid PlantUML: Must end with @enduml"
        
        elif ext in ['.mmd', '.mermaid']:
            # Basic Mermaid validation
            valid_diagrams = ['graph', 'sequenceDiagram', 'classDiagram', 'stateDiagram', 
                            'erDiagram', 'gantt', 'pie', 'flowchart', 'gitGraph']
            if not any(file.content.strip().startswith(diagram) for diagram in valid_diagrams):
                return f"Invalid Mermaid: Must start with a valid diagram type ({', '.join(valid_diagrams)})"
        
        elif ext == '.ipynb':
            # Jupyter notebook validation
            try:
                notebook_data = json.loads(file.content)
                if 'cells' not in notebook_data:
                    return "Invalid Jupyter notebook: Missing 'cells' array"
                if 'metadata' not in notebook_data:
                    return "Invalid Jupyter notebook: Missing 'metadata'"
            except json.JSONDecodeError as e:
                return f"Invalid Jupyter notebook: {str(e)}"
        
        elif ext == '.tf':
            # Basic Terraform validation
            tf_keywords = ['resource', 'variable', 'output', 'provider', 'module', 'data', 'locals']
            if not any(keyword in file.content for keyword in tf_keywords):
                return "Invalid Terraform file: Must contain Terraform configuration blocks"
        
        elif ext == '.tfvars':
            # Terraform variables file - should be valid HCL syntax
            # Basic check for key = value pairs
            lines = file.content.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' not in line and '{' not in line:
                        return f"Invalid tfvars syntax: Line should contain assignment: {line}"
        
        elif ext == '.proto':
            # Protocol Buffers validation
            if not any(keyword in file.content for keyword in ['syntax', 'package', 'message', 'service']):
                return "Invalid Protocol Buffers: Must contain proto definitions"
            if file.content.strip() and not file.content.strip().startswith('syntax'):
                logger.warning("Protocol Buffers file should start with syntax declaration")
        
        elif ext == '.graphql' or ext == '.gql':
            # GraphQL schema validation
            graphql_keywords = ['type', 'interface', 'enum', 'scalar', 'union', 'schema', 'query', 'mutation']
            if not any(keyword in file.content.lower() for keyword in graphql_keywords):
                return "Invalid GraphQL: Must contain GraphQL type definitions"
        
        elif ext in ['.ndjson', '.jsonl']:
            # Newline-delimited JSON validation
            lines = file.content.strip().split('\n')
            for i, line in enumerate(lines, 1):
                if line.strip():  # Skip empty lines
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as e:
                        return f"Invalid NDJSON at line {i}: {str(e)}"
        
        elif ext == '.service':
            # systemd service file validation
            if not file.content.strip().startswith('['):
                return "Invalid systemd service: Must start with a section like [Unit]"
            required_sections = ['[Unit]', '[Service]']
            for section in required_sections:
                if section not in file.content:
                    return f"Invalid systemd service: Missing required section {section}"
        
        elif file.filename.lower() == '.gitignore':
            # .gitignore validation - very permissive, just check it's not empty
            if not file.content.strip():
                return "Empty .gitignore file"
        
        elif file.filename.lower() == '.htaccess':
            # Basic .htaccess validation
            # Check for common directives
            if file.content.strip() and not any(directive in file.content for directive in 
                ['RewriteEngine', 'RewriteRule', 'Redirect', 'Order', 'Allow', 'Deny', 
                 'ErrorDocument', 'Options', 'DirectoryIndex', 'AddType']):
                logger.warning(".htaccess file doesn't contain common Apache directives")
        
        elif file.filename.lower() in ['requirements.txt', 'requirements.in']:
            # Python requirements file validation
            lines = file.content.strip().split('\n')
            for i, line in enumerate(lines, 1):
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                # Basic package name validation
                if not any(c.isalnum() or c in '-_.' for c in line.split('==')[0].split('>=')[0].split('<=')[0]):
                    return f"Invalid requirements.txt at line {i}: Invalid package name"
        
        elif file.filename.lower() in ['pipfile', 'pipfile.lock']:
            # Pipfile validation (TOML format)
            try:
                import toml
                toml.loads(file.content)
            except Exception as e:
                return f"Invalid Pipfile (TOML format): {str(e)}"
        
        
        elif file.filename.lower() == 'cargo.toml':
            # Cargo.toml validation
            try:
                import toml
                cargo_data = toml.loads(file.content)
                if '[package]' not in file.content:
                    return "Invalid Cargo.toml: Missing [package] section"
            except Exception as e:
                return f"Invalid Cargo.toml: {str(e)}"
        
        elif file.filename.lower() in ['docker-compose.yml', 'docker-compose.yaml']:
            # Docker Compose validation
            try:
                compose_data = yaml.safe_load(file.content)
                if not isinstance(compose_data, dict):
                    return "Invalid docker-compose.yml: Root must be a mapping"
                # Check for version or services
                if 'version' not in compose_data and 'services' not in compose_data:
                    return "Invalid docker-compose.yml: Must define version and/or services"
            except yaml.YAMLError as e:
                return f"Invalid docker-compose.yml: {str(e)}"
        
        return None