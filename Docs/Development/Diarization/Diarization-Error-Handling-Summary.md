# Enhanced Error Handling for VAD Utilities in Diarization Service

## Overview

Added extensive error handling and comments to the `_vad_utils` usage in the diarization service to address its brittleness. The Silero VAD utilities are particularly fragile because they:

1. Return as a tuple in a specific order that can change between versions
2. Have implicit dependencies on parameter names (e.g., `sampling_rate` vs `sample_rate`)
3. Mix functions and classes in the utility tuple
4. May not be available if PyTorch hub loading fails

## Key Improvements

### 1. Enhanced `_lazy_import_silero_vad()` Function

```python
def _lazy_import_silero_vad():
    """Lazy import Silero VAD model.
    
    This function loads the Silero VAD model from torch hub.
    The model returns a tuple of (model, utils) where utils is a tuple of functions.
    
    Returns:
        tuple: (model, utils) or (None, None) if loading fails
        
    Note:
        The utils tuple order is critical and can break between versions:
        - utils[0]: get_speech_timestamps - Main VAD function
        - utils[1]: save_audio - Save audio to file
        - utils[2]: read_audio - Read audio from file  
        - utils[3]: VADIterator - Class for streaming VAD
        - utils[4]: collect_chunks - Collect speech chunks
    """
```

Key improvements:
- Detailed documentation of utility order
- Validation of result format before unpacking
- Explicit hub directory handling
- Comprehensive error messages with type information
- Graceful failure with (None, None) return

### 2. Robust `_load_vad_model()` Method

```python
def _load_vad_model(self):
    """Load the VAD model (lazy loading).
    
    This method loads the Silero VAD model and its utility functions.
    The VAD utilities are particularly brittle as they return as a tuple
    in a specific order that can change between versions.
    """
```

Key improvements:
- Validates utils format before mapping
- Checks each utility is callable (except VADIterator)
- Provides detailed error messages about format mismatches
- Safely handles indexing errors

### 3. Safe Utility Getter `_get_vad_utility()`

```python
def _get_vad_utility(self, name: str) -> Callable:
    """Safely get a VAD utility function with validation.
    
    Args:
        name: Name of the utility ('get_speech_timestamps', 'read_audio', etc.)
        
    Returns:
        The utility function
        
    Raises:
        DiarizationError: If utility is not available or not callable
    """
```

This method:
- Validates utilities are loaded
- Checks utility exists with helpful error showing available utilities
- Handles special case for VADIterator (class not function)
- Validates callability

### 4. Enhanced `_detect_speech()` Method

Key improvements:
- Uses safe getter instead of direct dictionary access
- Detailed parameter documentation warning about 'sampling_rate' vs 'sample_rate'
- Validates output format (list of dicts with start/end fields)
- Comprehensive error messages

### 5. Improved `_load_audio()` Fallback

The fallback to Silero's read_audio now includes:
- Detailed logging of fallback reason
- Safe utility access with validation
- Clear error messages about which loading methods failed
- Parameter name documentation (sampling_rate)

## Error Message Examples

The enhanced error handling provides clear, actionable error messages:

```
"Unexpected Silero VAD utils format. Expected tuple/list with 5+ items, got dict with 3 items"

"VAD utility 'get_speech_timestamps' not found. Available: ['read_audio', 'save_audio']"

"Failed to map Silero VAD utilities. The utility order may have changed. Error: list index out of range"

"VAD utility 'read_audio' is not callable. Got type: NoneType"
```

## Testing

All 19 tests pass, confirming the error handling doesn't break normal operation while providing better diagnostics when things go wrong.

## Future Considerations

1. **Version Detection**: Could add Silero VAD version detection to handle different utility orders
2. **Utility Discovery**: Could introspect utilities instead of hardcoding positions
3. **Fallback Strategies**: Could implement alternative VAD methods if Silero fails
4. **Configuration**: Could make utility mapping configurable for different Silero versions

## Usage Safety

The service now:
- Fails fast with clear errors instead of cryptic exceptions
- Provides debugging information about what went wrong
- Maintains backward compatibility
- Handles edge cases gracefully