# Fix for "bad value(s) in fds_to_keep" Error

## Root Cause
The error occurs because the app's logging system (RichLogHandler) wraps stdout/stderr, which creates invalid file descriptors when HuggingFace's transformers library tries to spawn subprocesses for model downloads.

## Evidence Found
1. The app uses a RichLogHandler that captures stdout/stderr for the UI log display
2. When transformers calls subprocess operations internally, it inherits these wrapped file descriptors
3. On macOS, subprocess is strict about file descriptor validation, causing the error

## Solution Implemented

### 1. Created a Context Manager (`protect_file_descriptors`)
- Temporarily replaces wrapped stdout/stderr with real file descriptors
- Sets environment variables to disable parallel processing
- Restores original state after model loading

### 2. Applied Protection During Model Loading
- Modified `_HuggingFaceEmbedder.__init__` to use the context manager
- This ensures clean file descriptors during `from_pretrained` calls

### 3. Simplified Download Worker
- Removed redundant environment variable manipulation
- The protection is now centralized in the EmbeddingFactory

## How It Works

```python
@contextmanager
def protect_file_descriptors():
    # Save wrapped file descriptors
    original_stdout = sys.stdout
    
    # Create real file descriptors for subprocess
    if wrapped:
        sys.stdout = open(os.devnull, 'w')
    
    # Disable parallelism to avoid more subprocesses
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    yield
    
    # Restore everything
    sys.stdout = original_stdout
```

## Result
The model downloads should now work without the file descriptor error because:
1. Subprocesses get valid file descriptors instead of wrapped objects
2. Parallel processing is disabled to reduce subprocess spawning
3. The original logging system is preserved for the rest of the app