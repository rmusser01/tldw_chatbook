#!/usr/bin/env python3
import os

def find_button_handlers():
    with open('tldw_chatbook/app.py', 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if 'on_button_pressed' in line or 'nav_handlers' in line or 'button_handlers' in line or 'Unhandled button press' in line:
            print(f"{i+1:4d}: {line.rstrip()}")

if __name__ == "__main__":
    find_button_handlers()