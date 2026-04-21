import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    print("Importing get_app...")
    from tldw_chatbook.app import get_app
    print("Calling get_app()...")
    app = get_app()
    print("App loaded successfully!")
except Exception as e:
    print(f"Failed to load app: {e}")
    import traceback
    traceback.print_exc()
