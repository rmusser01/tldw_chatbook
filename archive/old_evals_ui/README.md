# Archived Eval UI Files

## Archive Date: 2025-01-13

## Reason for Archiving
These files represent old or experimental versions of the Evaluation UI that are no longer in use. The current active version is `tldw_chatbook/UI/evals_window_v2.py`.

## Archived Files

### 1. `2025-01-13_Evals_Window_v3_unified.py`
- **Original Location**: `tldw_chatbook/UI/Evals_Window_v3_unified.py`
- **Description**: Version 3 unified implementation, not currently in use
- **Status**: Experimental/abandoned

### 2. `2025-01-13_evals_window_v2_debug.py`
- **Original Location**: `tldw_chatbook/UI/evals_window_v2_debug.py`
- **Description**: Debug version of v2 implementation
- **Status**: Debug/development version

### 3. `2025-01-13_evals_screen_v2.py`
- **Original Location**: `tldw_chatbook/UI/evals_screen_v2.py`
- **Description**: Alternative screen-based implementation
- **Status**: Alternative approach, not adopted

### 4. `2025-01-13_EvaluationSetupWindow.py`
- **Original Location**: `tldw_chatbook/UI/EvaluationSetupWindow.py`
- **Description**: Older setup window implementation
- **Status**: Replaced by v2

### 5. `2025-01-13_Evals_directory/`
- **Original Location**: `tldw_chatbook/UI/Evals/`
- **Contents**:
  - `evals_screen.py` - Screen implementation
  - `evals_state.py` - State management
  - `evals_messages.py` - Message definitions
  - `__init__.py` - Package init
- **Description**: Modular approach that was not fully adopted
- **Status**: Experimental/abandoned

## Current Active Version
The current production version is:
- **File**: `tldw_chatbook/UI/evals_window_v2.py`
- **Description**: Fixed and fully functional implementation with Collapsible widgets
- **Last Updated**: 2025-01-13 (fixed CSS overflow issues, layout problems, and dropdown crashes)

## Notes
These files are archived for reference and potential future use. They should not be imported or used in production code. If you need to restore any of these files, move them back to their original locations and update the imports in `app.py`.