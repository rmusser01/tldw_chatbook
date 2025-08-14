#!/usr/bin/env python3
"""Test script to verify hook is called with correct parameters"""

import json
import sys
from datetime import datetime

# This file will be modified to trigger the hook
print("Hook test file created at", datetime.now())

# Add some content that might trigger warnings
api_key = "test_key_for_hook_detection"  # Should trigger sensitive data warning
print("Debug statement for hook detection")  # Should trigger debug statement warning