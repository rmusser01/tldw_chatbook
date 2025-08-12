#!/usr/bin/env python
"""Standalone runner for Chat v99.

This script allows running the chat interface independently
for testing and development.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def main():
    """Run the Chat v99 application."""
    from tldw_chatbook.chat_v99.app import ChatV99App
    
    print("Starting Chat v99...")
    print("Press Ctrl+C to exit")
    print("-" * 40)
    
    app = ChatV99App()
    app.run()

if __name__ == "__main__":
    main()