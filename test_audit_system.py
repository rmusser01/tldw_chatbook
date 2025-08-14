#!/usr/bin/env python3
"""Test file to verify the audit system is working correctly"""

import os

# This should trigger a sensitive data warning
API_KEY = "sk-test123456789"
SECRET_TOKEN = "github_pat_testtoken"

# Debug statements that should be detected
print("Debug: Starting application")
print(f"Using API key: {API_KEY}")

def process_data(data):
    """Process some data"""
    password = "admin123"  # Should trigger warning
    print(f"Processing with password: {password}")
    return data

if __name__ == "__main__":
    print("Audit test file executed")
    process_data("test")
    # Modified to test audit system
    # Test completed at 19:20