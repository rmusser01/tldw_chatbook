# Embeddings Download Button Fix Summary

## Issues Fixed

1. **Duplicate `_download_model_worker` methods** - Removed the async version that was causing conflicts
2. **Button handler not being triggered** - Made sure the handler is synchronous (not async)
3. **Worker call simplified** - Removed unnecessary wrapper function
4. **Added view switching logic** - Added `embeddings_active_view` reactive and watch method to handle view transitions
5. **Fixed undefined `_update_status` calls** - Replaced with `notify()` calls
6. **Set initial view visibility** - Ensured views are properly shown/hidden on mount

## Key Changes Made

### In `Embeddings_Management_Window.py`:
- Removed duplicate async `_download_model_worker` method
- Kept the synchronous worker method that uses message passing
- Fixed the button handler to be synchronous:
  ```python
  @on(Button.Pressed, "#embeddings-download-model")
  def on_download_model(self) -> None:
  ```
- Simplified worker call to avoid wrapper function

### In `Embeddings_Window.py`:
- Added `embeddings_active_view` reactive attribute
- Added `watch_embeddings_active_view` method to handle view switching
- Fixed initial view visibility in `on_mount`
- Replaced `_update_status` with `notify`

## How It Works

1. When user clicks "Manage Embeddings" navigation button, the app's generic navigation handler sets `embeddings_active_view` to "embeddings-view-manage"
2. The watch method detects this change and shows/hides the appropriate views
3. When user clicks "Download" button in the management view, the button handler executes
4. The handler starts a worker thread to download the model
5. The worker uses message passing to update the UI safely from the thread

## Testing

To test the fix:
1. Run the app
2. Go to Embeddings tab
3. Click "Manage Embeddings" in the navigation
4. Select a HuggingFace model from the list
5. Click the "Download" button
6. The download should start without "unhandled button press" error

## Notes

- The navigation between views is handled by the app's generic navigation system
- The button handlers in child widgets (EmbeddingsManagementWindow) are properly registered
- Thread-safe UI updates use Textual's message passing system