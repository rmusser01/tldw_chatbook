"""Main entry point for running Chat v99 as a module.

This allows running with: python -m tldw_chatbook.chat_v99
"""

from .app import ChatV99App

def main():
    """Run the Chat v99 application."""
    app = ChatV99App()
    app.run()

if __name__ == "__main__":
    main()