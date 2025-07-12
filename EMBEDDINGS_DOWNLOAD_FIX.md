# Embeddings Model Download Fix

## Issue
The "bad value(s) in fds_to_keep" error occurs when downloading embedding models due to subprocess file descriptor issues on macOS.

## Root Cause
- The HuggingFace transformers library uses subprocess calls during model downloads
- macOS has stricter file descriptor handling which causes the error
- The error occurs in `AutoTokenizer.from_pretrained()` and `AutoModel.from_pretrained()`

## Fixes Applied

### 1. Environment Variables
Added environment variables to prevent subprocess issues:
```python
os.environ['PYTHONNOUSERSITE'] = '1'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'  # macOS specific
```

### 2. Async Download Method
Added `_async_download_model()` that:
- Uses `asyncio.to_thread()` to run the download in a separate thread
- Avoids blocking the UI
- Provides better error handling

### 3. Fallback Mechanism
The download button now:
1. First tries the async download approach
2. Falls back to the worker thread method if async fails
3. Uses `huggingface_hub.snapshot_download()` when available

### 4. Improved Error Handling
- Added detailed logging with `exc_info=True`
- Better error messages to the user
- Proper cleanup in finally blocks

## Testing
To test the fix:
1. Navigate to Embeddings tab
2. Click "Manage Embeddings"
3. Select a model (e.g., "e5-small-v2")
4. Click "Download"
5. The model should download without the "bad value(s) in fds_to_keep" error

## Alternative Solutions
If the error persists:
1. Pre-download models using the command line:
   ```bash
   python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('intfloat/e5-small-v2'); AutoTokenizer.from_pretrained('intfloat/e5-small-v2')"
   ```

2. Set HF_HOME environment variable to control cache location:
   ```bash
   export HF_HOME=~/.cache/huggingface
   ```

3. Update dependencies:
   ```bash
   pip install --upgrade transformers huggingface-hub
   ```