# Chat Window Button CSS Customization Guide

## CSS File Locations

Based on the project structure and git status, the CSS files for button styling are located at:

1. **General Button Styles**: `tldw_chatbook/css/components/_buttons.tcss`
   - Contains base button styling for all buttons in the application
   - This is where you'll find the default outline styles

2. **Chat-Specific Styles**: `tldw_chatbook/css/features/_chat.tcss`
   - Contains styles specific to the Chat window
   - May override or extend button styles for the Chat window

3. **Main CSS File**: `tldw_chatbook/css/tldw_cli_modular.tcss`
   - The main modular CSS file that likely imports the component files

## Button Outline Properties

In Textual CSS (.tcss files), button outlines are typically controlled by:

### 1. Border Properties
```css
Button {
    border: solid $primary;      /* Current: likely a themed color variable */
    border: solid #336699;       /* Solid color example */
    border: heavy #003366;       /* Darker, heavier border */
}
```

### 2. Outline Properties
```css
Button {
    outline: solid $primary;     /* Outline separate from border */
    outline: dashed $secondary;  /* Different outline styles */
}
```

### 3. Focused State
```css
Button:focus {
    border: heavy $accent;       /* Different style when focused */
}
```

## How to Modify Button Outlines

### Step 1: Locate the Current Style
Look in `_buttons.tcss` for the base Button class:
```css
Button {
    border: [current_style];
}
```

### Step 2: Change to Solid/Darker Color
Replace with one of these options:

**Option A - Solid Dark Color:**
```css
Button {
    border: solid #2c2c2c;       /* Dark gray */
    /* or */
    border: solid rgb(44, 44, 44);
}
```

**Option B - Heavy Border:**
```css
Button {
    border: heavy $primary;       /* Uses theme color but heavier */
}
```

**Option C - Custom Dark Theme Colors:**
```css
Button {
    border: solid $primary-darken-3;  /* If theme supports color variants */
}
```

### Step 3: Chat Window Specific Overrides
In `_chat.tcss`, you can override for just the Chat window:
```css
ChatWindowEnhanced Button {
    border: solid #1a1a1a;        /* Even darker for chat */
}

/* Or target specific buttons */
#send-button {
    border: heavy #0066cc;        /* Specific button styling */
}
```

## Color Experimentation Guide

### 1. Color Formats Supported
- Hex: `#336699`
- RGB: `rgb(51, 102, 153)`
- Named colors: `darkblue`, `darkgray`
- Theme variables: `$primary`, `$secondary`, `$accent`

### 2. Border Styles
- `solid` - Standard solid line
- `heavy` - Thicker line
- `double` - Double line
- `dashed` - Dashed line
- `round` - Rounded corners

### 3. Quick Color Options to Try

**Dark Professional:**
```css
border: solid #2c3e50;    /* Dark blue-gray */
border: solid #34495e;    /* Darker blue-gray */
```

**Dark with Accent:**
```css
border: solid #1e3a5f;    /* Dark blue */
border: solid #2c5530;    /* Dark green */
```

**Monochrome:**
```css
border: solid #333333;    /* Dark gray */
border: solid #1a1a1a;    /* Very dark gray */
border: solid #000000;    /* Black */
```

### 4. Testing Different Colors

1. Edit the CSS file
2. Save the changes
3. Restart the application (Textual apps need restart for CSS changes)
4. Test the buttons in the Chat window

### 5. Advanced: Dynamic Theming

If you want different colors for different states:
```css
Button {
    border: solid #333333;
}

Button:hover {
    border: solid #555555;    /* Lighter on hover */
}

Button:focus {
    border: heavy #0066cc;    /* Blue when focused */
}

Button.-active {
    border: solid #006600;    /* Green when active */
}
```

## Note

Since the files are in `/Users/appledev/Working/tldw_chatbook/` (not in the current working directory), you'll need to edit the files there directly. The CSS changes will apply to all instances of the application once saved and the app is restarted.