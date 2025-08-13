"""Check if pyperclip is available."""

from tldw_chatbook.chat_v99.widgets.message_item_enhanced import HAS_PYPERCLIP, pyperclip

print(f"HAS_PYPERCLIP: {HAS_PYPERCLIP}")
print(f"pyperclip module: {pyperclip}")

if HAS_PYPERCLIP:
    # Try to use pyperclip
    try:
        pyperclip.copy("test")
        content = pyperclip.paste()
        print(f"Pyperclip works! Pasted: {content}")
    except Exception as e:
        print(f"Pyperclip error: {e}")
else:
    print("Pyperclip not available")