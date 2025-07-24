#!/usr/bin/env python3
"""
Test script to verify that the WebSearch_APIs.py configuration is loaded correctly.
"""

import sys
import os
from pprint import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the necessary modules
from tldw_chatbook.Web_Scraping.WebSearch_APIs import loaded_config_data, initialize_config

def main():
    """
    Test the WebSearch_APIs configuration loading.
    """
    print("Testing WebSearch_APIs configuration loading...")
    
    # Print the loaded configuration data
    print("\nLoaded configuration data:")
    pprint(loaded_config_data)
    
    # Verify that the 'search_engines' section exists
    if 'search_engines' in loaded_config_data:
        print("\n'search_engines' section exists.")
        
        # Print the keys in the 'search_engines' section
        print("\nKeys in 'search_engines' section:")
        pprint(list(loaded_config_data['search_engines'].keys()))
        
        # Check for specific keys that should be present
        required_keys = [
            'bing_search_api_key',
            'google_search_api_key',
            'brave_search_api_key',
            'bing_search_api_url',
            'google_search_api_url',
            'search_result_max',
            'bing_country_code'
        ]
        
        print("\nChecking for required keys:")
        for key in required_keys:
            if key in loaded_config_data['search_engines']:
                print(f"  ✓ {key}")
            else:
                print(f"  ✗ {key} - MISSING")
    else:
        print("\nERROR: 'search_engines' section does not exist in loaded_config_data.")
    
    # Test reinitializing the configuration
    print("\nReinitializing configuration...")
    reinitialized_config = initialize_config()
    
    # Verify that the reinitialized configuration matches the loaded configuration
    if reinitialized_config == loaded_config_data:
        print("Reinitialized configuration matches loaded configuration.")
    else:
        print("ERROR: Reinitialized configuration does not match loaded configuration.")
        print("\nDifferences:")
        for key in set(reinitialized_config.keys()) | set(loaded_config_data.keys()):
            if key not in reinitialized_config:
                print(f"  Key '{key}' in loaded_config_data but not in reinitialized_config")
            elif key not in loaded_config_data:
                print(f"  Key '{key}' in reinitialized_config but not in loaded_config_data")
            elif reinitialized_config[key] != loaded_config_data[key]:
                print(f"  Values for key '{key}' differ")

if __name__ == "__main__":
    main()