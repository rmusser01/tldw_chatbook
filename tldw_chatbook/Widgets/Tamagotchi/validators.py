"""
Validation utilities for Tamagotchi module.

Provides input validation and sanitization for user-provided parameters.
"""

import re
from typing import Any, Optional, Union


class ValidationError(ValueError):
    """Custom exception for validation errors."""
    pass


class TamagotchiValidator:
    """Validator for tamagotchi parameters."""
    
    # Constraints
    MIN_NAME_LENGTH = 1
    MAX_NAME_LENGTH = 20
    MIN_UPDATE_INTERVAL = 1.0  # seconds
    MAX_UPDATE_INTERVAL = 3600.0  # 1 hour
    VALID_SIZES = {'normal', 'compact', 'minimal'}
    VALID_THEMES = {'emoji', 'ascii', 'custom'}
    
    # Regex for safe names (alphanumeric, spaces, hyphens, underscores)
    NAME_PATTERN = re.compile(r'^[\w\s\-]+$')
    
    @classmethod
    def validate_name(cls, name: str) -> str:
        """
        Validate and sanitize pet name.
        
        Args:
            name: Pet name to validate
            
        Returns:
            Sanitized name
            
        Raises:
            ValidationError: If name is invalid
        """
        if not name:
            raise ValidationError("Pet name cannot be empty")
        
        # Strip whitespace
        name = name.strip()
        
        # Check length
        if len(name) < cls.MIN_NAME_LENGTH:
            raise ValidationError(f"Pet name must be at least {cls.MIN_NAME_LENGTH} character(s)")
        if len(name) > cls.MAX_NAME_LENGTH:
            raise ValidationError(f"Pet name cannot exceed {cls.MAX_NAME_LENGTH} characters")
        
        # Check pattern
        if not cls.NAME_PATTERN.match(name):
            raise ValidationError(
                "Pet name can only contain letters, numbers, spaces, hyphens, and underscores"
            )
        
        return name
    
    @classmethod
    def validate_update_interval(cls, interval: float) -> float:
        """
        Validate update interval.
        
        Args:
            interval: Update interval in seconds
            
        Returns:
            Validated interval
            
        Raises:
            ValidationError: If interval is invalid
        """
        try:
            interval = float(interval)
        except (TypeError, ValueError):
            raise ValidationError("Update interval must be a number")
        
        if interval < cls.MIN_UPDATE_INTERVAL:
            raise ValidationError(
                f"Update interval must be at least {cls.MIN_UPDATE_INTERVAL} seconds"
            )
        if interval > cls.MAX_UPDATE_INTERVAL:
            raise ValidationError(
                f"Update interval cannot exceed {cls.MAX_UPDATE_INTERVAL} seconds"
            )
        
        return interval
    
    @classmethod
    def validate_personality(cls, personality: str, available_personalities: dict) -> str:
        """
        Validate personality type.
        
        Args:
            personality: Personality type name
            available_personalities: Dictionary of available personalities
            
        Returns:
            Validated personality name
            
        Raises:
            ValidationError: If personality is invalid
        """
        if not personality:
            return 'balanced'  # Default
        
        personality = personality.lower().strip()
        
        if personality not in available_personalities:
            available = ', '.join(sorted(available_personalities.keys()))
            raise ValidationError(
                f"Invalid personality '{personality}'. Available: {available}"
            )
        
        return personality
    
    @classmethod
    def validate_size(cls, size: str) -> str:
        """
        Validate widget size.
        
        Args:
            size: Size setting
            
        Returns:
            Validated size
            
        Raises:
            ValidationError: If size is invalid
        """
        if not size:
            return 'normal'  # Default
        
        size = size.lower().strip()
        
        if size not in cls.VALID_SIZES:
            available = ', '.join(sorted(cls.VALID_SIZES))
            raise ValidationError(
                f"Invalid size '{size}'. Available: {available}"
            )
        
        return size
    
    @classmethod
    def validate_sprite_theme(cls, theme: str) -> str:
        """
        Validate sprite theme.
        
        Args:
            theme: Theme name
            
        Returns:
            Validated theme
            
        Raises:
            ValidationError: If theme is invalid
        """
        if not theme:
            return 'emoji'  # Default
        
        theme = theme.lower().strip()
        
        if theme not in cls.VALID_THEMES:
            available = ', '.join(sorted(cls.VALID_THEMES))
            raise ValidationError(
                f"Invalid theme '{theme}'. Available: {available}"
            )
        
        return theme
    
    @classmethod
    def validate_stat(cls, value: float, stat_name: str) -> float:
        """
        Validate a stat value (0-100 range).
        
        Args:
            value: Stat value
            stat_name: Name of the stat for error messages
            
        Returns:
            Clamped value between 0 and 100
        """
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{stat_name} must be a number")
        
        # Clamp to valid range
        return max(0.0, min(100.0, value))
    
    @classmethod
    def validate_age(cls, age: float) -> float:
        """
        Validate age value.
        
        Args:
            age: Age in hours
            
        Returns:
            Validated age
            
        Raises:
            ValidationError: If age is invalid
        """
        try:
            age = float(age)
        except (TypeError, ValueError):
            raise ValidationError("Age must be a number")
        
        if age < 0:
            raise ValidationError("Age cannot be negative")
        
        # Cap at reasonable maximum (1 year in hours)
        max_age = 365 * 24
        if age > max_age:
            age = max_age
        
        return age


class StateValidator:
    """Validator for saved state data."""
    
    REQUIRED_FIELDS = {'name', 'happiness', 'hunger', 'energy', 'health', 'age'}
    
    @classmethod
    def validate_state(cls, state: dict) -> tuple[bool, Optional[str]]:
        """
        Validate saved state for corruption.
        
        Args:
            state: State dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not state:
            return False, "State is empty"
        
        if not isinstance(state, dict):
            return False, "State is not a dictionary"
        
        # Check required fields
        missing_fields = cls.REQUIRED_FIELDS - set(state.keys())
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"
        
        # Validate field types and ranges
        try:
            # Name should be a string
            if not isinstance(state['name'], str) or not state['name']:
                return False, "Invalid name"
            
            # Stats should be numbers in valid range
            for stat in ['happiness', 'hunger', 'energy', 'health']:
                value = state.get(stat)
                if not isinstance(value, (int, float)):
                    return False, f"{stat} is not a number"
                if not 0 <= value <= 100:
                    return False, f"{stat} is out of range (0-100)"
            
            # Age should be non-negative
            age = state.get('age')
            if not isinstance(age, (int, float)):
                return False, "Age is not a number"
            if age < 0:
                return False, "Age is negative"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
        
        return True, None
    
    @classmethod
    def create_default_state(cls, name: str = "Pet") -> dict:
        """
        Create a default valid state.
        
        Args:
            name: Pet name
            
        Returns:
            Default state dictionary
        """
        return {
            'name': name,
            'happiness': 50.0,
            'hunger': 50.0,
            'energy': 50.0,
            'health': 100.0,
            'age': 0.0,
            'personality': 'balanced',
            'is_alive': True,
            'total_interactions': 0
        }
    
    @classmethod
    def repair_state(cls, state: dict, name: str = "Pet") -> dict:
        """
        Attempt to repair a corrupted state.
        
        Args:
            state: Potentially corrupted state
            name: Default name if missing
            
        Returns:
            Repaired state dictionary
        """
        if not isinstance(state, dict):
            return cls.create_default_state(name)
        
        # Start with default
        repaired = cls.create_default_state(name)
        
        # Try to salvage valid fields
        for field in state:
            if field in repaired:
                try:
                    value = state[field]
                    
                    # Validate and copy field
                    if field == 'name' and isinstance(value, str) and value:
                        repaired[field] = value
                    elif field in ['happiness', 'hunger', 'energy', 'health']:
                        if isinstance(value, (int, float)):
                            repaired[field] = max(0.0, min(100.0, float(value)))
                    elif field == 'age' and isinstance(value, (int, float)) and value >= 0:
                        repaired[field] = float(value)
                    elif field == 'personality' and isinstance(value, str):
                        repaired[field] = value
                    elif field == 'is_alive' and isinstance(value, bool):
                        repaired[field] = value
                    elif field == 'total_interactions' and isinstance(value, int) and value >= 0:
                        repaired[field] = value
                except Exception:
                    # Skip corrupted field
                    pass
        
        return repaired


class RateLimiter:
    """Rate limiting for interactions."""
    
    def __init__(
        self,
        global_cooldown: float = 0.5,
        action_cooldowns: Optional[dict[str, float]] = None
    ):
        """
        Initialize rate limiter.
        
        Args:
            global_cooldown: Minimum time between any interactions
            action_cooldowns: Per-action cooldown times
        """
        self.global_cooldown = global_cooldown
        self.action_cooldowns = action_cooldowns or {}
        self.last_interaction = 0.0
        self.last_action_times: dict[str, float] = {}
    
    def can_interact(self, current_time: float, action: Optional[str] = None) -> tuple[bool, float]:
        """
        Check if interaction is allowed.
        
        Args:
            current_time: Current timestamp
            action: Optional action name for per-action limiting
            
        Returns:
            Tuple of (allowed, remaining_cooldown)
        """
        # Check global cooldown
        global_remaining = max(0, self.global_cooldown - (current_time - self.last_interaction))
        if global_remaining > 0:
            return False, global_remaining
        
        # Check action-specific cooldown
        if action and action in self.action_cooldowns:
            last_time = self.last_action_times.get(action, 0)
            cooldown = self.action_cooldowns[action]
            action_remaining = max(0, cooldown - (current_time - last_time))
            if action_remaining > 0:
                return False, action_remaining
        
        return True, 0
    
    def record_interaction(self, current_time: float, action: Optional[str] = None) -> None:
        """
        Record that an interaction occurred.
        
        Args:
            current_time: Current timestamp
            action: Optional action name
        """
        self.last_interaction = current_time
        if action:
            self.last_action_times[action] = current_time