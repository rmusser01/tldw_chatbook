#!/usr/bin/env python3

"""
Helper script to create custom note templates for tldw_chatbook.
This will create/update the user's personal note_templates.json file.
"""

import json
import os
from pathlib import Path

def create_custom_template():
    # User config path
    user_config_dir = Path.home() / ".config" / "tldw_cli"
    user_templates_path = user_config_dir / "note_templates.json"
    
    # Create directory if needed
    user_config_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing templates if any
    templates = {}
    if user_templates_path.exists():
        try:
            with open(user_templates_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                templates = data.get('templates', data)
            print(f"Loaded {len(templates)} existing templates from {user_templates_path}")
        except Exception as e:
            print(f"Error loading existing templates: {e}")
            templates = {}
    
    print("\n=== Create Custom Note Template ===")
    
    # Get template details
    key = input("Template key (e.g., 'weekly_review'): ").strip().lower().replace(' ', '_')
    if not key:
        print("Template key is required!")
        return
    
    title = input("Title template (use {date}, {time}, {datetime} for placeholders): ").strip()
    if not title:
        title = key.replace('_', ' ').title()
    
    description = input("Description (shown in UI): ").strip()
    if not description:
        description = title
    
    keywords = input("Keywords (comma-separated): ").strip()
    
    print("\nEnter content (multiline, type 'END' on a new line to finish):")
    content_lines = []
    while True:
        line = input()
        if line.strip().upper() == 'END':
            break
        content_lines.append(line)
    content = '\n'.join(content_lines)
    
    # Create template
    templates[key] = {
        "title": title,
        "content": content,
        "keywords": keywords,
        "description": description
    }
    
    # Save templates
    output = {"templates": templates}
    
    with open(user_templates_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Template '{key}' saved to {user_templates_path}")
    print(f"Total templates: {len(templates)}")
    print("\nRestart tldw_chatbook to use your new template!")

if __name__ == "__main__":
    create_custom_template()