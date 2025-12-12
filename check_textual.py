import sys
print(f"Python version: {sys.version}")
try:
    import textual
    print(f"Textual version: {textual.__version__}")
    from textual.app import App
    print("Textual App imported successfully")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
