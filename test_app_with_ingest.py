#!/usr/bin/env python3
"""Test the main app with focus on the new Media Ingest tab."""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    from tldw_chatbook.app import TldwCli
    
    # Start the app and let user navigate to ingest tab
    app = TldwCli()
    app.run()