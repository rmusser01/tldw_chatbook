# File Picker Enhancements

This document describes the enhancements made to the textual-fspicker module to improve user experience and functionality.

## New Features

### 1. Enhanced Keyboard Shortcuts
- **Ctrl+H** - Toggle hidden files (in addition to existing '.')
- **Ctrl+L** - Toggle path input field for direct path entry (shows/hides a text field where you can type absolute or relative paths)
- **Ctrl+R** - Toggle recent locations panel
- **Ctrl+F** - Toggle search mode and focus search input
- **F5** - Refresh current directory
- **Ctrl+D** - Bookmark current directory (placeholder for future implementation)

### 2. Recent Locations Panel
- Shows recently accessed files and directories
- Toggle visibility with Ctrl+R
- Click to quickly navigate to recent locations
- Placeholder for persistent storage (needs implementation)

### 3. Breadcrumb Navigation
- Visual path display with clickable components
- Click any path segment to navigate directly
- Better understanding of current location in filesystem hierarchy

### 4. Search Within Directory
- Real-time search filtering of directory contents
- Toggle with Ctrl+F
- Filters files and folders by name
- Clear button to reset search

### 5. Direct Path Input
- Toggle with Ctrl+L to show/hide a path input field
- Enter absolute paths (e.g., `/home/user/documents`) or relative paths (e.g., `../folder`)
- Supports home directory expansion (e.g., `~/Documents`)
- Press Enter or click "Go" to navigate
- Automatically navigates to parent directory if a file path is entered
- Shows error notification if path doesn't exist

### 5. Improved Visual Feedback
- Notifications for keyboard actions
- Better error messages
- Search active indicator

## Implementation Details

### Modified Files

1. **base_dialog.py**
   - Added new keyboard bindings
   - Added reactive properties for UI state
   - Implemented breadcrumb navigation
   - Added recent locations panel
   - Added search container
   - Implemented all keyboard action handlers

2. **parts/directory_navigation.py**
   - Added `search_filter` reactive variable
   - Modified `hide()` method to respect search filter
   - Added `_watch_search_filter()` method
   - Modified `_repopulate_display()` to handle search state

### Code Structure

The enhancements maintain backward compatibility while adding new optional features. The UI components are hidden by default and can be toggled via keyboard shortcuts.

### CSS Additions

New CSS rules were added for:
- Breadcrumb navigation styling
- Recent locations panel
- Search container
- Visibility toggles

## Usage Example

```python
from textual_fspicker import FileOpen

# Use the enhanced file picker
file_dialog = FileOpen(
    title="Select a file",
    filters=Filters(
        ("Python Files", "*.py"),
        ("All Files", "*.*")
    )
)

# All new features are available via keyboard shortcuts
```

## Future Improvements

1. **Persistent Storage for Recent Locations**
   - Save to user config directory
   - Load on startup
   - Configurable max items

2. **Bookmarks System**
   - Save frequently used directories
   - Manage bookmarks UI
   - Persistent storage

3. **Advanced Search**
   - Regex support
   - Case sensitivity toggle
   - Search in subdirectories option

4. **File Preview**
   - Preview pane for text files
   - Image thumbnails
   - File metadata display

## Testing

The enhancements have been tested with:
- Various directory structures
- Hidden files toggle
- Search functionality
- Breadcrumb navigation
- Keyboard shortcuts

### 6. Usable Filename Input for Save Dialogs (task-1479)

Live UAT of a keyboard-only export flow (Evals results-grid export, at a
235x52 terminal) found the `FileSave`/`FileOpen` input bar unusable:

- The filename `Input`'s rendered width could collapse to a handful of
  columns because the file-type filter `Select` next to it was set to
  `width: 1fr` -- flexible, not fixed, so it competed with the Input for
  space and, once the app's own `Select { width: 100%; }` bundle rule
  (`components/_dialogs.tcss` documents this in full) wins the CSS-origin
  battle against this package's `DEFAULT_CSS` regardless of source order,
  the Select claimed the entire row. The filter `Select` now gets a fixed
  `width: 24` in `file_dialog.py`'s `DEFAULT_CSS`, and the app bundle pins
  the same width with a selector specific enough to win there too.
- `FileSave` (not `FileOpen`) now focuses its filename `Input` on mount
  instead of the directory listing (`FileSystemPickerScreen._focus_initial_widget`,
  overridden in `file_save.py`), so a keyboard user can press Enter right
  away to confirm the seeded default filename, instead of Enter activating
  the highlighted directory row (usually `..`).
- `Select` posts its own `Changed` message as a side effect of mounting
  with an explicit initial `value=` -- not from a user picking a filter.
  `BaseFileDialog._change_filter` used to unconditionally move focus back
  to the directory listing on every `Select.Changed`, including that first,
  synthetic one, which raced with (and usually beat) the new mount-time
  focus above. It now ignores focus-stealing for the first event only,
  tracked via `BaseFileDialog._filter_select_changed_by_user`.
- `_select_file`/`_confirm_file` (`file_dialog.py`) read the filename back
  via `self.query_one(Input)`, which is ambiguous: the screen also carries
  a hidden `#path-input` (Ctrl+L) and `#search-input` (Ctrl+F), both
  mounted before the input bar's own filename `Input`. An unscoped
  `query_one(Input)` silently grabbed one of those instead, so even once
  focus and width were fixed, pressing Enter on the (correctly focused,
  correctly filled) filename field read back an empty, unrelated Input and
  rejected with "A file must be chosen". Both call sites now query through
  `InputBar` first (`self.query_one(InputBar).query_one(Input)`), which is
  unambiguous since InputBar's own children are exactly one `Input` (the
  filename) and, if filters were supplied, one `Select`.

None of this touches a plain `FileOpen`'s own default focus behaviour (still
the directory listing) or the separate `EnhancedFileDialog` picker in
`Widgets/enhanced_file_picker.py`, which composes its own, differently-`id`d
Input/Select and is unaffected. (`FileOpen(offer_select_folder=True)`,
`SelectDirectory` and `EnhancedSelectDirectory` DO focus the field now -- see
section 8 below.)

### 7. The Filename Field Is Also the Path Field (task-32122, task-32229)

Both changes extend the ONE input the bar already has, rather than adding a
second field beside it:

- **Label follows the value** (task-32122): `#file-name-label` reads
  "Folder path:" instead of "File name:" once the typed text resolves to an
  existing directory, and only for `FileOpen(offer_select_folder=True)`,
  whose "Select folder" button reads that same field.
- **Typing a rooted path moves the listing** (task-32229): on
  `Input.Changed`, a value that is absolute or starts with `~` goes through
  `resolve_typed_directory` (the shared validator "Select folder" already
  used) and `DirectoryNavigation.location` follows it. Relative values are
  deliberately ignored -- they are file names, and one of them is the
  basename `_select_file` pre-fills on a click for "Select folder" to read.
  The placeholder says "File name or path"; the pre-existing hidden Ctrl+L
  path bar is untouched.
- **Ctrl+A selects the field** (task-32229): Textual's `Input` binds
  `home,ctrl+a` to "go to start", and a focused widget's own bindings beat
  the screen's, so the bar's Input is now a three-line `FileNameInput`
  subclass whose only content is `Binding("ctrl+a", "select_all")`. `Home`
  is unchanged.

`SelectDirectory` is not a `BaseFileDialog` and none of this reaches it. (It
does share the mount-time focus rule in section 8 -- that one lives on the
common base precisely so a second directory dialog cannot miss it.)

### 8. Folder-Offering Open Dialogs Open on the Field (task-32540, task-32554)

- **Initial focus** (task-32540, generalised by task-32606): a dialog whose
  result is a DIRECTORY opens on the input bar's field, with any pre-filled
  value selected, instead of on the listing. Reproduced live at 235x52 before
  the change: Library ▸ Notes ▸ Add from files… ▸ Import once, then typing
  `/Users` -- every character went into the directory listing's type-ahead and
  Enter opened `..`.

  task-32540 shipped this as a `FileOpen._focus_initial_widget` override, and
  critique #4 found the sibling it never reached: Library ▸ Notes ▸ Folder
  files pushes the vendored `SelectDirectory`, which has no override at all
  (nor does `EnhancedSelectDirectory`), so that door stayed mouse-only. The
  behaviour now lives ONCE on `FileSystemPickerScreen._focus_initial_widget`,
  keyed on a single declarative fact -- the `RETURNS_A_FOLDER` class
  attribute, which `SelectDirectory` and `EnhancedSelectDirectory` set and
  `FileOpen` answers per instance through its existing
  `_offer_select_folder`. `FileOpen` no longer overrides the method. The
  field is read generically as the input bar's one `Input`, the way
  `_resolve_select_folder_target` already reads it, so the three differing ids
  (`#path_input`, `#dir-path-input`, the anonymous `FileNameInput`) need no
  special-casing.

  `FileSave` keeps its own override: it is not folder-returning, and its
  reason (section 6) is different. A plain file-only `FileOpen` (character
  import, skill folders, TTS model directories) is deliberately untouched,
  since there Enter on a listing row is the natural first keystroke. Both
  branches are pinned, the second as an explicit negative control, in
  `Tests/UI/test_library_notes_w4_import_keyboard.py` and
  `Tests/UI/test_library_notes_w5_picker_keyboard.py`.
- **The dialog's own footer** (task-32606): `compose` yields a `Footer` as a
  direct child of the SCREEN (not of `Dialog`). A `ModalScreen` is
  translucent, so the host screen's footer shows straight through it --
  critique #4 pressed Tab three times inside this dialog and kept reading
  Library's `/ focus search | F6 next pane | esc notes`. An opaque footer on
  the bottom row replaces those chips for as long as the modal is up; mounted
  inside `Dialog` it would instead add a second key row eight lines above a
  contradicting one. `EnhancedFileDialog` mirrors the base layout by hand
  rather than calling it, so the base's `Footer` does not reach it and
  `EnhancedSelectDirectory` still shows the HOST screen's chips through the
  modal. That is a known, deliberate gap, not an oversight: a screen-docked
  footer takes the bottom terminal row, and that dialog is `height: 95%`
  against this one's 80%, so at the 60x24 its pickers are pinned at, adding
  it pushed the character-import picker's selection marker off the bottom
  and turned three existing size pins red. Giving that family the chips
  needs a layout answer for that row, not one more `yield`.

  Three consequences of putting it at screen level, all handled:

  1. It docks OUTSIDE `SAFE_MODAL_CONTENT`, so `SafeModalDismissMixin`
     classified a chip click as a backdrop click and cancelled the dialog --
     on `FileSave`, discarding a typed filename. `modal_dismissal.py` now
     exempts a click landing inside any mounted `Footer` on the screen
     (`target_is_modal_chrome`). Tested by POINT, not by the target's
     ancestry: `FooterKey` fires its key from `on_mouse_down`, and a key
     that moves focus changes the active bindings, so `Footer` recomposes
     and the chip is already detached when the `Click` arrives.
  2. `Footer` lays chips out in binding order and scrolls the overflow off
     the right edge, so `BINDINGS` order IS the narrow-width priority:
     `escape` leads, the path actions follow. At 100 columns `esc Cancel`
     was otherwise off-screen entirely.
  3. `Footer` renders every `show=True` binding and marks a
     `check_action`-vetoed one dim rather than dropping it, so `^s Select
     this folder` still appears (dimmed) on dialogs that do not offer it.
     It is therefore ordered LAST, where a narrow terminal scrolls it off
     first instead of spending 23 of 60 columns on it.

  `Footer` is `can_focus=False, can_focus_children=False`, so no picker's
  Tab order changes.
- **A shape-based focus cue** (task-32540): the bar's buttons told a keyboard
  user which one Enter would press by colour and a label underline only. The
  rule that fixes it cannot live here -- the app's own
  `components/_buttons.tcss` sets `Button:focus { outline: none }`, and
  Textual ranks every CSS_PATH rule above any widget `DEFAULT_CSS` whatever
  its specificity -- so `FileSystemPickerScreen Button:focus` lives in the
  app bundle (`components/_dialogs.tcss`) and paints heavy left/right
  outlines. Vendored code owns none of it; only this note.
- **Breadcrumb collapse** (task-32554): `_update_breadcrumbs` rendered every
  segment, so a 100+ character location ran off the dialog's right edge and
  clipped exactly the crumbs that say where you are. It now shows root + `…`
  + the last few (`_MAX_VISIBLE_BREADCRUMBS`, matching
  `EnhancedFileDialog`'s own ceiling), each crumb keeping its absolute path
  as its tooltip.

### 9. Folder-Returning Dialogs Are One Picker, Not Three (task-32611, task-32643)

Critique #4 opened the three Library ▸ Notes folder doors in one sitting and
found three dialogs for one decision. The fix keeps reading the SAME
declarative fact section 8 introduced, `RETURNS_A_FOLDER`, rather than adding
flags beside it.

- **The third door was using the wrong one of the two.** "Keep a folder
  synced" pushed `FileOpen(offer_select_folder=True)` -- the files-AND-folder
  dialog -- while its callback dropped anything that was not a directory. So
  its field was labelled "File name", its placeholder read "File name or
  path", and every file in the folder was listed as though pickable, under
  the title "Choose a folder to keep synced". It now pushes `SelectDirectory`,
  the folder-only mode of the same family. Nothing in the picker changed for
  it; the caller was asking the wrong question.
- **One hint, one button name.** `FileOpen._hint_text` and
  `SelectDirectory._hint_text` are both gone: `FileSystemPickerScreen.
  _hint_text` returns the one sentence whenever `RETURNS_A_FOLDER`, and
  `FOLDER_CONFIRM_LABEL` is the single literal behind both that sentence and
  the "Select folder" button `compose` adds on a files-and-folder dialog.
  `SELECT_BUTTON_DEFAULT` (a class attribute, `"Select folder"` on
  `SelectDirectory`) is the folder-only half, so the button that commits a
  folder is called the same thing on all three doors. A caller passing an
  explicit `select_button` still wins.
- **`_default_listing_sort`.** A new `"folders"` sort key (folders first,
  name-ascending; `Descending` reverses the NAMES and keeps folders on top,
  via a second stable sort on `not is_directory` rather than folding the flag
  into the key) is the DEFAULT for a folder-returning dialog and only for one.
  A file picker keeps "Discovery order", where rows arriving in disk order is
  the point, and "Discovery order" stays on the menu for both. The Select's
  initial `value=` and the navigation's `sort_key` are set from the one
  method, the latter in `on_mount`, which Textual dispatches to every class in
  the MRO -- so `EnhancedFileDialog`, which builds its own navigation in its
  own `compose`, is covered without a second copy of the two lines.
- **The folder badge is bounded, and off by default.** `FileRecord` gained
  `note_count` / `is_vault` / `folder_summary_loaded`, rendered through
  `size_text` (the column a directory has always left blank) and
  `display_name` -- so both the vendored row and `EnhancedFileDialog`'s
  responsive one show them with no renderer change. The count is one
  `os.scandir` of that folder, no recursion, capped at `NOTE_COUNT_CEILING`
  (500) entries READ, not just displayed; it rides the existing
  visible-rows-only metadata hydration worker. `show_folder_notes` is off
  unless the caller passed a `notes_context`, because "12 notes" is a useful
  badge when choosing where notes live and noise in a model-file or
  character-card picker. `_wants_hydration` asks the badge question
  independently of `metadata_loaded`: a metadata sort stats every record
  without counting notes, so keying the queue on `metadata_loaded` alone left
  folders that first became visible under "Size" permanently badge-less.
- **The vault predicate is CALLED, not copied.** `read_folder_summary` imports
  `Notes.note_import_discovery.folder_is_obsidian_vault` -- the body lifted
  out of `library_notes_sync_controller._carries_obsidian_marker`, which now
  delegates to it -- so the listing and the sync setup cannot drift on what a
  vault is. The import is lazy: this package is deliberately absent from the
  app's boot import closure (`Tests/Packaging/test_app_import_diet_closure`)
  and so is that module.
- **Recent roots reuse the store that already exists.** `_get_recent_paths`
  was a stub returning `[]`, which is why ctrl+r opened a permanently empty
  box on every picker. With a `notes_context` it now reads
  `RecentLocations(context=...)` -- the enhanced family's own
  `[filepicker] recent_<context>` store -- written by
  `Library.library_browse_location.remember_browse_directory`, which all three
  doors already call on a worker, folded into the config write it was already
  making. `watch_show_recent` focuses the list when it has entries, and
  `_on_recent_selected` DISMISSES with the root on a folder-returning dialog
  (navigating there and then requiring Select would leave it two keystrokes
  away, not one). Two deliberate non-changes: ctrl+r still opens an empty
  panel, because `test_fspicker_keyboard_save` pins the Escape-peel order by
  opening all three transients on an empty picker; and `_add_to_recent` stays
  a no-op here, because it fires on every `DirectoryNavigation.Changed`, i.e.
  on merely passing through a folder.

## Contributing Upstream

These enhancements are designed to be contributed back to the original textual-fspicker project. They:
- Maintain backward compatibility
- Follow the existing code style
- Add optional features that don't change default behavior
- Include proper documentation

To contribute:
1. Fork the original repository
2. Apply these changes
3. Add tests for new features
4. Submit a pull request with this documentation