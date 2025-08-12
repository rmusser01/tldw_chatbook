#!/usr/bin/env python3
"""
Script to enable MediaWindowV88 in the config file.
"""

import toml
from pathlib import Path

# Find config file
config_path = Path.home() / ".config" / "tldw_cli" / "config.toml"

if config_path.exists():
    # Load config
    with open(config_path, 'r') as f:
        config = toml.load(f)
    
    # Enable new media UI
    config['use_new_media_ui'] = True
    
    # Save config
    with open(config_path, 'w') as f:
        toml.dump(config, f)
    
    print(f"✓ MediaWindowV88 enabled in {config_path}")
else:
    print(f"Config file not found at {config_path}")
    print("Creating config with MediaWindowV88 enabled...")
    
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {'use_new_media_ui': True}
    
    with open(config_path, 'w') as f:
        toml.dump(config, f)
    
    print(f"✓ Created config with MediaWindowV88 enabled at {config_path}")