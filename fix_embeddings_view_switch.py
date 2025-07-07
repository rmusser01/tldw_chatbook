#!/usr/bin/env python3
"""Add view switching logic to EmbeddingsWindow."""

import re

# Read the file
with open("tldw_chatbook/UI/Embeddings_Window.py", "r") as f:
    content = f.read()

# Check if we have the navigation reactive attribute
if "embeddings_active_view" not in content:
    print("Need to add embeddings_active_view reactive attribute...")
    
    # Find where to add it (after other reactive attributes)
    pattern = r'(selected_db_items: reactive\[set\] = reactive\(set\(\)\).*?\n)(.*?specific_item_ids:)'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        # Add the reactive attribute
        insert_pos = match.end(1)
        new_line = '    embeddings_active_view: reactive[str] = reactive("embeddings-view-create")  # Track active view\n    '
        content = content[:insert_pos] + new_line + content[insert_pos:]
        print("Added embeddings_active_view reactive attribute")

# Check if we have a watch method for the reactive
if "def watch_embeddings_active_view" not in content:
    print("Need to add watch_embeddings_active_view method...")
    
    # Find a good place to add it (after other methods)
    pattern = r'(def _get_chunk_methods\(self\).*?\n.*?return.*?\n)(.*?def)'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        insert_pos = match.end(1)
        watch_method = '''
    def watch_embeddings_active_view(self, old: str, new: str) -> None:
        """React to view changes by showing/hiding containers."""
        logger.debug(f"Switching from view {old} to {new}")
        
        # Hide all views first
        for view_id in EMBEDDINGS_VIEW_IDS:
            try:
                view = self.query_one(f"#{view_id}")
                view.styles.display = "none"
            except Exception:
                pass
        
        # Show the selected view
        try:
            active_view = self.query_one(f"#{new}")
            active_view.styles.display = "block"
        except Exception as e:
            logger.error(f"Failed to show view {new}: {e}")
    
'''
        content = content[:insert_pos] + watch_method + '    ' + content[insert_pos:]
        print("Added watch_embeddings_active_view method")

# Write the updated file
with open("tldw_chatbook/UI/Embeddings_Window.py", "w") as f:
    f.write(content)

print("Done! The view switching should now work properly.")