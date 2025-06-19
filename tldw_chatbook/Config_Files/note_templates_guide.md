# Note Templates Guide

## Overview

Note templates in tldw_chatbook allow you to quickly create new notes with predefined structures. Templates support placeholders that are automatically filled with current date/time information.

## Template Locations

Templates are loaded from the following locations (in order of priority):

1. **User Config Directory**: `~/.config/tldw_cli/note_templates.json`
2. **App Config Directory**: `<app_dir>/Config_Files/note_templates.json`
3. **Hardcoded Defaults**: Built into the application

## Template Structure

Each template is defined in JSON format with the following fields:

```json
{
  "templates": {
    "template_key": {
      "title": "Template Title - {date}",
      "content": "Template content with {placeholders}",
      "keywords": "comma, separated, keywords",
      "description": "Human-readable description shown in UI"
    }
  }
}
```

### Available Placeholders

- `{date}` - Current date in YYYY-MM-DD format
- `{time}` - Current time in HH:MM format
- `{datetime}` - Current date and time in YYYY-MM-DD HH:MM format

## Creating Custom Templates

### Method 1: Copy to User Config Directory

1. Create the config directory if it doesn't exist:
   ```bash
   mkdir -p ~/.config/tldw_cli
   ```

2. Copy the default templates file:
   ```bash
   cp <app_dir>/Config_Files/note_templates.json ~/.config/tldw_cli/
   ```

3. Edit the file with your custom templates:
   ```bash
   nano ~/.config/tldw_cli/note_templates.json
   ```

### Method 2: Edit App Config File

Edit the file directly in the application directory:
```
<app_dir>/Config_Files/note_templates.json
```

**Note**: Changes to the app config file may be overwritten during updates.

## Example Custom Templates

### Stand-up Meeting Template
```json
"standup": {
  "title": "Stand-up - {date}",
  "content": "## Daily Stand-up - {date}\n\n### Yesterday\n- \n\n### Today\n- \n\n### Blockers\n- \n",
  "keywords": "standup, daily, scrum",
  "description": "Daily stand-up meeting"
}
```

### Book Notes Template
```json
"book_notes": {
  "title": "Book: ",
  "content": "## Book Notes\n\n**Title:** \n**Author:** \n**Started:** {date}\n\n### Key Themes\n- \n\n### Important Quotes\n> \n\n### Chapter Notes\n\n#### Chapter 1\n- \n\n### Overall Thoughts\n",
  "keywords": "book, reading, notes",
  "description": "Book reading notes"
}
```

### Recipe Template
```json
"recipe": {
  "title": "Recipe: ",
  "content": "## Recipe\n\n**Name:** \n**Cuisine:** \n**Prep Time:** \n**Cook Time:** \n**Servings:** \n\n### Ingredients\n- \n- \n\n### Instructions\n1. \n2. \n\n### Notes\n- \n\n**Date Added:** {date}",
  "keywords": "recipe, cooking, food",
  "description": "Recipe template"
}
```

## Tips

1. Keep template keys short and descriptive (they're used internally)
2. Use the `description` field for UI-friendly names
3. Templates are loaded once at startup - restart the app after making changes
4. Use markdown formatting in content for better rendering
5. Leave empty lines with just `-` or `1.` for easy cursor placement

## Troubleshooting

If your templates aren't loading:

1. Check the JSON syntax - must be valid JSON
2. Ensure file permissions allow reading
3. Check the app logs for template loading messages
4. Verify the file path is correct

## Sharing Templates

You can share your `note_templates.json` file with others. They can place it in their user config directory to use your templates.