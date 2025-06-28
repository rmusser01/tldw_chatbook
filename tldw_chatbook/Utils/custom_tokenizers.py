# custom_tokenizers.py
# Description: Support for custom tokenizer.json files for accurate token counting
#
# Imports
import json
import os
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
#
# 3rd-Party Imports
from loguru import logger
#
# Check for optional tokenizers library
try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    logger.warning("tokenizers library not available. Install with: pip install tokenizers")
#
########################################################################################################################
#
# Custom Tokenizer Manager

class CustomTokenizerManager:
    """Manages custom tokenizer.json files for accurate token counting."""
    
    def __init__(self, tokenizers_dir: Optional[str] = None):
        """
        Initialize the tokenizer manager.
        
        Args:
            tokenizers_dir: Directory containing tokenizer.json files
        """
        self.tokenizers_dir = tokenizers_dir or os.path.expanduser("~/.config/tldw_cli/tokenizers")
        self._tokenizers: Dict[str, Any] = {}
        self._model_mappings: Dict[str, str] = {}
        
        # Create directory if it doesn't exist
        os.makedirs(self.tokenizers_dir, exist_ok=True)
        
        # Load tokenizer mappings if available
        self._load_mappings()
        
    def _load_mappings(self) -> None:
        """Load model to tokenizer mappings from config file."""
        mappings_file = os.path.join(self.tokenizers_dir, "mappings.json")
        if os.path.exists(mappings_file):
            try:
                with open(mappings_file, 'r') as f:
                    self._model_mappings = json.load(f)
                logger.info(f"Loaded {len(self._model_mappings)} tokenizer mappings")
            except Exception as e:
                logger.error(f"Failed to load tokenizer mappings: {e}")
    
    def save_mappings(self) -> None:
        """Save current model to tokenizer mappings."""
        mappings_file = os.path.join(self.tokenizers_dir, "mappings.json")
        try:
            with open(mappings_file, 'w') as f:
                json.dump(self._model_mappings, f, indent=2)
            logger.debug("Saved tokenizer mappings")
        except Exception as e:
            logger.error(f"Failed to save tokenizer mappings: {e}")
    
    def add_mapping(self, model_name: str, tokenizer_name: str) -> None:
        """
        Add a mapping from model name to tokenizer name.
        
        Args:
            model_name: The model identifier
            tokenizer_name: The tokenizer file name (without .json)
        """
        self._model_mappings[model_name] = tokenizer_name
        self.save_mappings()
    
    def load_tokenizer(self, name: str) -> Optional[Any]:
        """
        Load a tokenizer by name.
        
        Args:
            name: Tokenizer name (without .json extension)
            
        Returns:
            Loaded tokenizer or None if not available
        """
        if not TOKENIZERS_AVAILABLE:
            return None
            
        # Check cache first
        if name in self._tokenizers:
            return self._tokenizers[name]
        
        # Try to load from file
        tokenizer_path = os.path.join(self.tokenizers_dir, f"{name}.json")
        if os.path.exists(tokenizer_path):
            try:
                tokenizer = Tokenizer.from_file(tokenizer_path)
                self._tokenizers[name] = tokenizer
                logger.info(f"Loaded tokenizer: {name}")
                return tokenizer
            except Exception as e:
                logger.error(f"Failed to load tokenizer {name}: {e}")
        
        return None
    
    def get_tokenizer_for_model(self, model: str, provider: str) -> Optional[Any]:
        """
        Get the appropriate tokenizer for a model.
        
        Args:
            model: Model name
            provider: Provider name
            
        Returns:
            Tokenizer instance or None
        """
        # Check direct model mapping
        if model in self._model_mappings:
            return self.load_tokenizer(self._model_mappings[model])
        
        # Check provider-based patterns
        provider_patterns = {
            "anthropic": ["claude", "anthropic"],
            "google": ["gemini", "google", "palm"],
            "mistral": ["mistral", "mixtral"],
            "openai": ["gpt", "openai"],
            "meta": ["llama", "meta"],
        }
        
        # Try to match by provider patterns
        if provider in provider_patterns:
            for pattern in provider_patterns[provider]:
                if pattern in model.lower():
                    # Try to find a tokenizer with this pattern
                    for tokenizer_name in os.listdir(self.tokenizers_dir):
                        if tokenizer_name.endswith('.json') and pattern in tokenizer_name.lower():
                            name = tokenizer_name[:-5]  # Remove .json
                            return self.load_tokenizer(name)
        
        # Try to find by model name patterns
        model_lower = model.lower()
        for tokenizer_file in os.listdir(self.tokenizers_dir):
            if tokenizer_file.endswith('.json'):
                name = tokenizer_file[:-5]  # Remove .json
                if any(part in model_lower for part in name.lower().split('-')):
                    return self.load_tokenizer(name)
        
        return None
    
    def count_tokens(self, text: str, model: str, provider: str) -> Optional[int]:
        """
        Count tokens using a custom tokenizer if available.
        
        Args:
            text: Text to tokenize
            model: Model name
            provider: Provider name
            
        Returns:
            Token count or None if no suitable tokenizer
        """
        tokenizer = self.get_tokenizer_for_model(model, provider)
        if tokenizer:
            try:
                encoding = tokenizer.encode(text)
                return len(encoding.ids)
            except Exception as e:
                logger.error(f"Error counting tokens with custom tokenizer: {e}")
        
        return None
    
    def count_messages_tokens(self, messages: List[Dict[str, str]], model: str, provider: str) -> Optional[int]:
        """
        Count tokens for a list of messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name
            provider: Provider name
            
        Returns:
            Total token count or None
        """
        tokenizer = self.get_tokenizer_for_model(model, provider)
        if not tokenizer:
            return None
        
        total_tokens = 0
        
        # Different providers format messages differently
        if provider == "anthropic":
            # Claude format: Human: ... Assistant: ...
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role == "user":
                    formatted = f"Human: {content}"
                elif role == "assistant":
                    formatted = f"Assistant: {content}"
                elif role == "system":
                    formatted = f"System: {content}"
                else:
                    formatted = content
                
                count = self.count_tokens(formatted, model, provider)
                if count:
                    total_tokens += count
                    # Add some tokens for message separators
                    total_tokens += 4
        
        elif provider == "openai":
            # OpenAI format includes special tokens for roles
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                # Role tokens (approximate)
                total_tokens += 4  # <|im_start|>role
                
                # Content tokens
                count = self.count_tokens(content, model, provider)
                if count:
                    total_tokens += count
                
                # End token
                total_tokens += 2  # <|im_end|>
        
        else:
            # Generic format
            for msg in messages:
                content = msg.get("content", "")
                count = self.count_tokens(content, model, provider)
                if count:
                    total_tokens += count
                    # Add some overhead for formatting
                    total_tokens += 3
        
        return total_tokens
    
    def list_available_tokenizers(self) -> List[str]:
        """List all available tokenizer names."""
        tokenizers = []
        for file in os.listdir(self.tokenizers_dir):
            if file.endswith('.json'):
                tokenizers.append(file[:-5])  # Remove .json
        return sorted(tokenizers)
    
    def install_tokenizer(self, source_path: str, name: Optional[str] = None) -> bool:
        """
        Install a tokenizer from a file path.
        
        Args:
            source_path: Path to the tokenizer.json file
            name: Optional name for the tokenizer (defaults to filename)
            
        Returns:
            Success status
        """
        try:
            source = Path(source_path)
            if not source.exists():
                logger.error(f"Tokenizer file not found: {source_path}")
                return False
            
            if name is None:
                name = source.stem  # Filename without extension
            
            dest_path = os.path.join(self.tokenizers_dir, f"{name}.json")
            
            # Validate it's a valid tokenizer file
            if TOKENIZERS_AVAILABLE:
                try:
                    test_tokenizer = Tokenizer.from_file(str(source))
                except Exception as e:
                    logger.error(f"Invalid tokenizer file: {e}")
                    return False
            
            # Copy the file
            import shutil
            shutil.copy2(source, dest_path)
            logger.info(f"Installed tokenizer: {name}")
            
            # Clear cache to reload
            if name in self._tokenizers:
                del self._tokenizers[name]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to install tokenizer: {e}")
            return False


# Global instance
_tokenizer_manager: Optional[CustomTokenizerManager] = None

def get_tokenizer_manager() -> CustomTokenizerManager:
    """Get or create the global tokenizer manager instance."""
    global _tokenizer_manager
    if _tokenizer_manager is None:
        _tokenizer_manager = CustomTokenizerManager()
    return _tokenizer_manager


def count_tokens_with_custom(text: str, model: str, provider: str) -> Optional[int]:
    """
    Count tokens using custom tokenizer if available.
    
    Args:
        text: Text to tokenize
        model: Model name
        provider: Provider name
        
    Returns:
        Token count or None if no custom tokenizer
    """
    manager = get_tokenizer_manager()
    return manager.count_tokens(text, model, provider)


def count_messages_with_custom(messages: List[Dict[str, str]], model: str, provider: str) -> Optional[int]:
    """
    Count tokens for messages using custom tokenizer if available.
    
    Args:
        messages: List of message dicts
        model: Model name
        provider: Provider name
        
    Returns:
        Token count or None if no custom tokenizer
    """
    manager = get_tokenizer_manager()
    return manager.count_messages_tokens(messages, model, provider)


#
# End of custom_tokenizers.py
########################################################################################################################