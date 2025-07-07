#!/usr/bin/env python3
"""Debug script to check ListView ID construction for ebook file selection"""

def get_listview_id(media_type):
    """Mimics the ListView ID construction logic"""
    if media_type.startswith("local_"):
        # For local media ingestion, extract the actual media type
        actual_media_type = media_type.replace("local_", "")
        list_view_id = f"#local-selected-local-files-list-{actual_media_type}"
    else:
        # For tldw API ingestion
        list_view_id = f"#tldw-api-selected-local-files-list-{media_type}"
    
    return list_view_id

# Test with the media type from the log
media_type = "local_ebook"
expected_id = get_listview_id(media_type)
print(f"Media type: {media_type}")
print(f"Expected ListView ID: {expected_id}")
print()

# What the error shows
error_id = "#ingest-local-ebook-files-list"
print(f"Error shows ID: {error_id}")
print(f"IDs match: {expected_id == error_id}")
print()

# Check the actual UI widget ID
ui_widget_id = "#local-selected-local-files-list-ebook"
print(f"UI widget has ID: {ui_widget_id}")
print(f"Expected matches UI: {expected_id == ui_widget_id}")