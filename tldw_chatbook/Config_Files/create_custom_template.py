#!/usr/bin/env python3

"""
Helper script to create custom note templates for tldw_chatbook.
This will create/update the user's personal note_templates.json file.
"""

from tldw_chatbook.config import _get_effective_config_path


def create_custom_template():
    # User config path -- honors TLDW_CONFIG_PATH so this writes to the same
    # file the running app's profile actually reads (TASK-865).
    user_config_dir = _get_effective_config_path().parent
    user_templates_path = user_config_dir / "note_templates.json"

    print("\n=== Create Custom Note Template ===")

    # Get template details
    key = (
        input("Template key (e.g., 'weekly_review'): ")
        .strip()
        .lower()
        .replace(" ", "_")
    )
    if not key:
        print("Template key is required!")
        return

    title = input(
        "Title template (use {date}, {time}, {datetime} for placeholders): "
    ).strip()
    if not title:
        title = key.replace("_", " ").title()

    description = input("Description (shown in UI): ").strip()
    if not description:
        description = title

    keywords = input("Keywords (comma-separated): ").strip()

    print("\nEnter content (multiline, type 'END' on a new line to finish):")
    content_lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        content_lines.append(line)
    content = "\n".join(content_lines)

    # Create template
    template = {
        "title": title,
        "content": content,
        "keywords": keywords,
        "description": description,
    }

    from tldw_chatbook.Notes.template_store import merge_templates

    _, count = merge_templates([(key, template)], selected=user_templates_path)

    print(f"\n✓ Template '{key}' saved to {user_templates_path}")
    print(f"Total templates: {count}")
    print("\nRestart tldw_chatbook to use your new template!")


if __name__ == "__main__":
    create_custom_template()
