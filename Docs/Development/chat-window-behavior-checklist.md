# ChatWindowEnhanced Behavior Checklist

## Purpose
This document captures all current behaviors of ChatWindowEnhanced that must be preserved during architectural refactoring. Use this as a manual test checklist before and after refactoring.

## Core Message Flow

### Sending Messages
- [ ] User can type text in chat input area (TextArea with ID "chat-input")
- [ ] Send button shows "Send" icon when not streaming
- [ ] Pressing Send button sends the message
- [ ] Enter key sends message (if configured)
- [ ] Chat input clears after sending
- [ ] Message appears in chat log
- [ ] Send button changes to "Stop" during streaming
- [ ] Stop button actually stops generation when clicked
- [ ] Button returns to "Send" after streaming completes
- [ ] Button click debouncing (300ms) prevents rapid duplicate sends
- [ ] Worker marked as exclusive prevents concurrent message sends

### Message Display
- [ ] User messages appear aligned right (if styled)
- [ ] Assistant messages appear aligned left (if styled)
- [ ] Messages support markdown rendering
- [ ] Code blocks are properly formatted
- [ ] Long messages scroll properly
- [ ] Chat log auto-scrolls to newest message
- [ ] Messages mount in VerticalScroll container with ID "chat-log"

## File Attachments

### Attachment UI
- [ ] Attach button (📎) visible when enabled in config (`chat.images.show_attach_button`)
- [ ] Attach button hidden when disabled in config
- [ ] Clicking attach button opens file picker dialog
- [ ] File picker shows appropriate file type filters
- [ ] File picker supports custom filter patterns
- [ ] Test mode shows file path input field for direct entry

### Image Attachments
- [ ] Can select image files (PNG, JPG, GIF, WEBP, BMP, ICO, SVG)
- [ ] Attachment indicator shows with image filename
- [ ] Attach button changes to "📎✓" when file attached
- [ ] Image data included in message when sent
- [ ] Warning shown if model doesn't support vision
- [ ] Can clear attachment with clear button (ID "clear-image")
- [ ] Indicator hides when attachment cleared
- [ ] Image processed as base64 encoded data
- [ ] MIME type correctly detected for images

### Document Attachments
- [ ] Can select text files (TXT, MD, RST, LOG)
- [ ] Can select code files (PY, JS, TS, CPP, C, H, JAVA, GO, RS, etc.)
- [ ] Can select data files (JSON, CSV, XML, YAML, TOML, INI)
- [ ] File content inserted inline into chat input
- [ ] Cursor positioned at end of inserted content
- [ ] Notification shows file was inserted
- [ ] Content properly formatted with filename header
- [ ] Large files handled without freezing UI (worker processing)

### Attachment State
- [ ] Attachment persists if message send fails
- [ ] Attachment clears after successful send
- [ ] Multiple attachments handled correctly (if supported)
- [ ] Invalid file paths show error message
- [ ] Permission errors handled gracefully
- [ ] Out of memory errors caught and reported
- [ ] File not found errors show user-friendly message
- [ ] Worker cancellation handled cleanly

## Voice Input

### Voice UI
- [ ] Mic button (🎤) visible when enabled in config (`chat.voice.show_mic_button`)
- [ ] Mic button hidden when disabled in config
- [ ] Ctrl+M keyboard shortcut toggles voice input
- [ ] Button ID is "mic-button"

### Recording Flow
- [ ] Click mic button starts recording
- [ ] Mic button changes to stop icon (🛑) during recording
- [ ] Button variant changes to "error" (red) during recording
- [ ] Notification shows "🎤 Listening..."
- [ ] Click stop button ends recording
- [ ] Button returns to mic icon after recording
- [ ] VoiceInputWidget created dynamically when needed
- [ ] Worker thread handles recording (exclusive mode)

### Transcription
- [ ] Transcribed text appears in chat input
- [ ] Existing text preserved (space added between)
- [ ] Chat input receives focus after transcription
- [ ] Empty transcription shows "No speech detected"
- [ ] Partial transcripts update in real-time (if supported)
- [ ] Uses configured transcription provider from settings
- [ ] Uses configured transcription model from settings
- [ ] Uses configured language from settings

### Voice Errors
- [ ] Microphone permission denied shows helpful message
- [ ] Audio initialization errors show user-friendly message  
- [ ] Missing dependencies handled gracefully
- [ ] Button resets to default state on error
- [ ] ImportError for missing sounddevice module handled
- [ ] RuntimeError for audio stream issues handled
- [ ] PermissionError for microphone access handled
- [ ] Generic exceptions caught with fallback message

## Sidebar Integration

### Left Sidebar (Settings)
- [ ] Toggle button shows/hides settings sidebar
- [ ] Settings changes apply immediately
- [ ] Provider/model selection cascades properly
- [ ] Basic/Advanced mode toggle works
- [ ] Search filters settings in advanced mode
- [ ] RAG settings panel prominent in basic mode

### Right Sidebar (Character/Context)
- [ ] Toggle button shows/hides character sidebar
- [ ] Character selection loads character card
- [ ] Character context included in messages
- [ ] Clear character button removes selection
- [ ] Conversation title and keywords editable
- [ ] Save/Clone chat buttons functional

## Tab Support (when enabled)

### Tab Management
- [ ] Multiple chat sessions supported via config setting
- [ ] Tab switching preserves session state
- [ ] Each tab has independent message history
- [ ] Attachments are tab-specific
- [ ] Button states are tab-specific
- [ ] Each tab gets unique session ID
- [ ] Widgets properly namespaced per tab

## Keyboard Shortcuts

- [ ] Ctrl+E - Edit focused message
- [ ] Ctrl+M - Toggle voice input
- [ ] Ctrl+Shift+Left - Shrink sidebar
- [ ] Ctrl+Shift+Right - Expand sidebar
- [ ] Enter - Send message (if configured)

## Error Handling

### User-Facing Errors
- [ ] File not found shows notification
- [ ] Permission denied shows notification
- [ ] Invalid file type shows notification
- [ ] Network errors show notification
- [ ] All errors show appropriate severity (info/warning/error)

### Recovery
- [ ] UI remains responsive after errors
- [ ] Buttons return to correct state after errors
- [ ] Attachments cleared on file errors
- [ ] Voice recording stops cleanly on error

## Performance

### Responsiveness
- [ ] No lag when typing in chat input
- [ ] File picker opens immediately
- [ ] Buttons respond immediately to clicks
- [ ] No UI freezing during file processing
- [ ] Smooth scrolling in chat log

### Large Files
- [ ] Large text files process without freezing UI
- [ ] Progress indication for long operations (if implemented)
- [ ] Can cancel long-running operations (if implemented)

## Configuration

### Settings Respected
- [ ] `show_attach_button` - Controls attach button visibility
- [ ] `show_mic_button` - Controls mic button visibility
- [ ] `enable_tabs` - Controls tab container vs single session
- [ ] Theme variables applied correctly
- [ ] Custom CSS classes work

## State Management

### Reactive Properties
- [ ] `is_send_button` - Updates button label/tooltip/style
- [ ] `pending_image` - Triggers attachment UI updates
- [ ] Watchers fire correctly on state changes

### Persistence
- [ ] Conversation state preserved during session
- [ ] Attachments cleared appropriately
- [ ] Character selection persists

## Edge Cases

### Rapid Actions
- [ ] Rapid send/stop clicking handled (debounced)
- [ ] Can't send while already sending
- [ ] Can't start recording while recording
- [ ] Double-click attach doesn't open two pickers

### Missing Elements
- [ ] Handles missing widgets gracefully
- [ ] Works without optional features
- [ ] Degrades gracefully with missing dependencies

### Concurrent Operations
- [ ] Can type while file processing
- [ ] Can't attach during send
- [ ] Workers marked exclusive prevent conflicts

## Visual Indicators

### Loading States
- [ ] Send button disabled during send
- [ ] Stop button shows during streaming
- [ ] Attachment indicator visible when file attached

### Status Feedback
- [ ] Notifications appear for important events
- [ ] Errors shown with appropriate styling
- [ ] Success messages confirm actions

## Manual Test Procedure

### Basic Flow Test (5 min)
1. Start application
2. Send a text message
3. Verify message appears
4. Attach an image file
5. Verify indicator shows
6. Send message with attachment
7. Verify attachment clears
8. Start voice recording
9. Speak test phrase
10. Verify transcription appears
11. Send voice message
12. Toggle sidebars
13. Test keyboard shortcuts

### Error Test (3 min)
1. Try to attach non-existent file
2. Try to attach file without permissions
3. Start/stop recording rapidly
4. Send empty message
5. Click stop when not streaming

### Performance Test (2 min)
1. Type long message quickly
2. Attach large text file
3. Scroll through long chat history
4. Toggle sidebars rapidly
5. Switch tabs (if enabled)

## Widget Caching & Performance

### Widget References
- [ ] Widgets cached on mount for performance
- [ ] Cache includes: send button, chat input, mic button, attach button, etc.
- [ ] Cached references used instead of repeated queries
- [ ] NoMatches exceptions handled gracefully
- [ ] Cache updated if widgets recreated

### Performance Optimizations
- [ ] Batch DOM updates used where applicable
- [ ] Workers prevent UI blocking for file processing
- [ ] Reactive properties minimize recomposition
- [ ] CSS extracted to external files

## Button Routing

### Button Handler Logic
- [ ] Core buttons handled (send, stop, etc.)
- [ ] Sidebar toggle buttons work
- [ ] Attachment buttons functional (attach, clear)
- [ ] Notes expand button toggles size
- [ ] Unknown button IDs logged but don't crash
- [ ] Button handlers return proper stop/continue signals

## Reactive Properties

### State Management
- [ ] `is_send_button` reactive updates button label/icon
- [ ] `pending_image` reactive triggers attachment UI
- [ ] Watchers fire on state changes
- [ ] No duplicate state (reactive vs instance variables)
- [ ] Reactive properties don't conflict

## Worker Operations

### Background Processing
- [ ] File processing uses thread workers
- [ ] Workers are synchronous (not async)
- [ ] Workers marked exclusive to prevent conflicts
- [ ] Worker cancellation handled gracefully
- [ ] UI updates via call_from_thread
- [ ] Progress feedback during long operations

## Notes for Refactoring

### Must Preserve
- All user-facing behaviors above
- Public API (methods other components rely on)
- Event handling signatures
- Configuration keys
- Worker patterns (sync with @work(thread=True))

### Can Change
- Internal method organization
- Private method names
- File structure
- Internal state management
- Class hierarchy (as long as public API preserved)

### Risk Areas
- Widget caching - ensure cached refs updated if widgets recreated
- Worker threads - must remain synchronous
- Event bubbling - ensure proper propagation
- Tab support - complex interaction with sessions
- Reactive properties - avoid reading during compose()

---

**Last Updated**: 2025-08-18
**Total Behaviors**: ~150 checkpoints
**Estimated Manual Test Time**: 15-20 minutes for full checklist