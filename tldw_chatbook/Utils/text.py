# text.py
# Description: This file contains utility functions for text processing, including color formatting and text manipulation.
#
# Imports
#
# 3rd-party Libraries
#
# Local Imports
#
######################################################################################################################
#
# Functions:
import re



def sanitize_filename(filename):
    """
    Sanitizes the filename by:
      1) Removing forbidden characters entirely (rather than replacing them with '-')
      2) Collapsing consecutive whitespace into a single space
      3) Collapsing consecutive dashes into a single dash
    """
    # 1) Remove forbidden characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "", filename)
    # 2) Replace runs of whitespace with a single space
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    # 3) Replace consecutive dashes with a single dash
    sanitized = re.sub(r"-{2,}", "-", sanitized)
    return sanitized






def slugify(text: str) -> str:
    """Simple slugify function, robust for empty or non-string."""
    if not isinstance(text, str) or not text:
        return "unknown_type"  # Default slug for unexpected types
    return (
        text.lower()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("&", "and")
        .replace("(", "")
        .replace(")", "")
        .replace(":", "")
        .replace(",", "")
    )


#
# End of text.py
######################################################################################################################
