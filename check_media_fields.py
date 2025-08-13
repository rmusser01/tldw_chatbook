#!/usr/bin/env python3
"""Check what fields are in the media database."""

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.config import get_media_db_path

# Initialize database
media_db = MediaDatabase(
    db_path=get_media_db_path(),
    client_id="cli-debug",
    check_integrity_on_startup=False
)

# Get a media item
media = media_db.get_media_by_id(7, include_trash=True)

if media:
    print(f"\n=== Media Item {media.get('id')} ===")
    print(f"Title: {media.get('title')}")
    print("\nAvailable fields:")
    
    # Check all fields
    for key in sorted(media.keys()):
        value = media[key]
        if value:
            if isinstance(value, str):
                if len(value) > 100:
                    print(f"  {key}: STRING ({len(value)} chars)")
                else:
                    print(f"  {key}: {repr(value[:50])}")
            else:
                print(f"  {key}: {type(value).__name__} = {value}")
        else:
            print(f"  {key}: None/Empty")
    
    # Check content fields specifically
    print("\n=== Content Fields ===")
    content_fields = ['content', 'transcription', 'summary', 'text', 'description']
    for field in content_fields:
        value = media.get(field)
        if value:
            print(f"{field}: {len(str(value))} chars - First 100 chars:")
            print(f"  {str(value)[:100]}...")
        else:
            print(f"{field}: None/Empty")
else:
    print("No media found with ID 7")