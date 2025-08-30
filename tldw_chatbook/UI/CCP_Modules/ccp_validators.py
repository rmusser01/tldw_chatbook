"""Pydantic validators for CCP input data validation."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator, ValidationError
from pathlib import Path
import re


class ConversationInput(BaseModel):
    """Validation model for conversation input data."""
    
    title: str = Field(..., min_length=1, max_length=255, description="Conversation title")
    keywords: Optional[str] = Field(None, max_length=1000, description="Keywords/tags")
    character_id: Optional[int] = Field(None, gt=0, description="Associated character ID")
    
    @validator('title')
    def validate_title(cls, v):
        """Ensure title doesn't contain invalid characters."""
        if not v.strip():
            raise ValueError("Title cannot be empty or whitespace only")
        # Remove any potential SQL injection attempts
        if any(char in v for char in [';', '--', '/*', '*/', 'DROP', 'DELETE']):
            raise ValueError("Title contains invalid characters")
        return v.strip()
    
    @validator('keywords')
    def validate_keywords(cls, v):
        """Validate and clean keywords."""
        if v:
            # Split by commas and clean each keyword
            keywords = [k.strip() for k in v.split(',') if k.strip()]
            return ', '.join(keywords)
        return v


class CharacterCardInput(BaseModel):
    """Validation model for character card input data."""
    
    name: str = Field(..., min_length=1, max_length=100, description="Character name")
    description: Optional[str] = Field(None, max_length=5000, description="Character description")
    personality: Optional[str] = Field(None, max_length=5000, description="Character personality")
    scenario: Optional[str] = Field(None, max_length=5000, description="Scenario/setting")
    first_message: Optional[str] = Field(None, max_length=5000, description="First message")
    keywords: Optional[str] = Field(None, max_length=1000, description="Keywords/tags")
    system_prompt: Optional[str] = Field(None, max_length=10000, description="System prompt")
    post_history_instructions: Optional[str] = Field(None, max_length=5000)
    alternate_greetings: Optional[List[str]] = Field(default_factory=list)
    tags: Optional[str] = Field(None, max_length=500)
    creator: Optional[str] = Field(None, max_length=100)
    version: Optional[str] = Field(None, max_length=20)
    image_path: Optional[Path] = None
    avatar_url: Optional[str] = None
    
    @validator('name')
    def validate_name(cls, v):
        """Ensure character name is valid."""
        if not v.strip():
            raise ValueError("Character name cannot be empty")
        # Basic sanitization
        if any(char in v for char in ['<', '>', '"', "'"]):
            raise ValueError("Character name contains invalid HTML characters")
        return v.strip()
    
    @validator('avatar_url')
    def validate_avatar_url(cls, v):
        """Validate avatar URL if provided."""
        if v:
            # Basic URL validation
            url_pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
                r'localhost|'  # localhost...
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            if not url_pattern.match(v):
                raise ValueError("Invalid avatar URL format")
        return v
    
    @validator('version')
    def validate_version(cls, v):
        """Validate version format."""
        if v:
            # Allow formats like "1.0", "2.1.3", "v1.0"
            version_pattern = re.compile(r'^v?\d+(\.\d+)*$')
            if not version_pattern.match(v):
                raise ValueError("Invalid version format. Use format like '1.0' or 'v2.1.3'")
        return v


class PromptInput(BaseModel):
    """Validation model for prompt input data."""
    
    name: str = Field(..., min_length=1, max_length=100, description="Prompt name")
    author: Optional[str] = Field(None, max_length=100, description="Prompt author")
    description: Optional[str] = Field(None, max_length=2000, description="Prompt description")
    system_prompt: Optional[str] = Field(None, max_length=10000, description="System prompt")
    user_prompt: Optional[str] = Field(None, max_length=10000, description="User prompt template")
    keywords: Optional[str] = Field(None, max_length=500, description="Keywords")
    
    @validator('name')
    def validate_name(cls, v):
        """Ensure prompt name is unique and valid."""
        if not v.strip():
            raise ValueError("Prompt name cannot be empty")
        # Remove any special characters that might cause issues
        if any(char in v for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']):
            raise ValueError("Prompt name contains invalid file system characters")
        return v.strip()
    
    @validator('system_prompt', 'user_prompt')
    def validate_prompt_content(cls, v):
        """Validate prompt content."""
        if v:
            # Check for potential injection patterns
            if '{{' in v and '}}' in v:
                # Allow template variables but validate them
                template_pattern = re.compile(r'\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}')
                templates = re.findall(r'\{\{.*?\}\}', v)
                for template in templates:
                    if not template_pattern.match(template):
                        raise ValueError(f"Invalid template variable: {template}")
        return v


class DictionaryInput(BaseModel):
    """Validation model for dictionary/world book input data."""
    
    name: str = Field(..., min_length=1, max_length=100, description="Dictionary name")
    description: Optional[str] = Field(None, max_length=2000, description="Description")
    strategy: str = Field("sorted_evenly", description="Replacement strategy")
    max_tokens: int = Field(1000, gt=0, le=100000, description="Maximum tokens")
    entries: List[Dict[str, Any]] = Field(default_factory=list, description="Dictionary entries")
    
    @validator('strategy')
    def validate_strategy(cls, v):
        """Validate replacement strategy."""
        valid_strategies = ["sorted_evenly", "character_lore_first", "global_lore_first"]
        if v not in valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of: {', '.join(valid_strategies)}")
        return v
    
    @validator('entries')
    def validate_entries(cls, v):
        """Validate dictionary entries."""
        for entry in v:
            if not isinstance(entry, dict):
                raise ValueError("Each entry must be a dictionary")
            if 'key' not in entry or 'value' not in entry:
                raise ValueError("Each entry must have 'key' and 'value' fields")
            # Validate regex patterns if present
            key = entry.get('key', '')
            if key.startswith('/') and key.endswith('/'):
                # It's a regex pattern
                try:
                    re.compile(key[1:-1])
                except re.error as e:
                    raise ValueError(f"Invalid regex pattern in entry: {e}")
            # Validate probability
            if 'probability' in entry:
                prob = entry['probability']
                if not isinstance(prob, (int, float)) or prob < 0 or prob > 100:
                    raise ValueError("Probability must be between 0 and 100")
        return v


class SearchInput(BaseModel):
    """Validation model for search operations."""
    
    search_term: str = Field(..., min_length=1, max_length=500, description="Search term")
    search_type: str = Field("title", description="Type of search")
    include_character_chats: bool = Field(True, description="Include character chats")
    all_characters: bool = Field(True, description="Search all characters")
    tags: Optional[List[str]] = Field(default_factory=list, description="Filter tags")
    
    @validator('search_type')
    def validate_search_type(cls, v):
        """Validate search type."""
        valid_types = ["title", "content", "tags", "keywords"]
        if v not in valid_types:
            raise ValueError(f"Invalid search type. Must be one of: {', '.join(valid_types)}")
        return v
    
    @validator('search_term')
    def sanitize_search_term(cls, v):
        """Sanitize search term to prevent injection."""
        # Remove SQL wildcards and special characters
        v = v.replace('%', '').replace('_', '').replace('*', '')
        # Remove potential SQL injection attempts
        if any(keyword in v.upper() for keyword in ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER']):
            raise ValueError("Search term contains invalid SQL keywords")
        return v.strip()


class FileImportInput(BaseModel):
    """Validation model for file import operations."""
    
    file_path: Path = Field(..., description="Path to file to import")
    file_type: str = Field(..., description="Type of file being imported")
    overwrite_existing: bool = Field(False, description="Whether to overwrite existing entries")
    
    @validator('file_path')
    def validate_file_path(cls, v):
        """Validate file path exists and is readable."""
        if not v.exists():
            raise ValueError(f"File does not exist: {v}")
        if not v.is_file():
            raise ValueError(f"Path is not a file: {v}")
        if not v.suffix in ['.json', '.yaml', '.yml', '.txt', '.png', '.jpg', '.jpeg', '.gif', '.webp']:
            raise ValueError(f"Unsupported file type: {v.suffix}")
        return v
    
    @validator('file_type')
    def validate_file_type(cls, v):
        """Validate file type."""
        valid_types = ["character_card", "conversation", "prompt", "dictionary", "world_book", "image"]
        if v not in valid_types:
            raise ValueError(f"Invalid file type. Must be one of: {', '.join(valid_types)}")
        return v


def validate_with_model(model_class: BaseModel, data: Dict[str, Any]) -> tuple[bool, Optional[BaseModel], Optional[str]]:
    """
    Generic validation function using Pydantic models.
    
    Args:
        model_class: The Pydantic model class to use for validation
        data: The data to validate
        
    Returns:
        Tuple of (is_valid, validated_data, error_message)
    """
    try:
        validated = model_class(**data)
        return True, validated, None
    except ValidationError as e:
        # Format error messages nicely
        errors = []
        for error in e.errors():
            field = ' -> '.join(str(loc) for loc in error['loc'])
            msg = error['msg']
            errors.append(f"{field}: {msg}")
        error_message = "Validation failed:\n" + "\n".join(errors)
        return False, None, error_message
    except Exception as e:
        return False, None, f"Unexpected validation error: {str(e)}"