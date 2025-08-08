#!/usr/bin/env python3
"""
Test script to verify the Ingest UI factory integration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from tldw_chatbook.config import get_ingest_ui_style
from tldw_chatbook.Widgets.Media_Ingest.IngestUIFactory import create_ingest_ui

# Check what UI style is configured
ui_style = get_ingest_ui_style()
print(f"Current UI style from config: {ui_style}")

# Create a mock app instance
class MockApp:
    def __init__(self):
        self.app_config = {
            "api_settings": {
                "openai": {"api_key": "test"},
                "anthropic": {"api_key": "test"}
            }
        }
        self.selected_local_files = {}
    
    def notify(self, message: str, severity: str = "information"):
        print(f"[{severity.upper()}] {message}")

# Test creating the UI
app = MockApp()
ui_widget = create_ingest_ui(app, media_type="video")

print(f"Created UI widget: {ui_widget.__class__.__name__}")

# Expected: IngestGridWindow since config is set to "grid"
if ui_style == "grid":
    expected = "IngestGridWindow"
elif ui_style == "wizard":
    expected = "IngestWizardWindow"
elif ui_style == "split":
    expected = "IngestSplitPaneWindow"
else:
    expected = "IngestLocalVideoWindowSimplified"

if ui_widget.__class__.__name__ == expected:
    print(f"✅ SUCCESS: Factory correctly created {expected} for style '{ui_style}'")
else:
    print(f"❌ FAIL: Expected {expected} but got {ui_widget.__class__.__name__}")
    sys.exit(1)