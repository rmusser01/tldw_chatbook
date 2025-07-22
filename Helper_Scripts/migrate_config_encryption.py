#!/usr/bin/env python3
"""
Migration script for config encryption format.

This script helps migrate from the old encryption format (AES-256-CBC with PBKDF2)
to the new format (AES-256-GCM with scrypt).

Usage:
    python migrate_config_encryption.py [--config-path PATH]
"""
import argparse
import base64
import copy
import hashlib
import hmac
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import toml
from Cryptodome.Cipher import AES
from Cryptodome.Protocol.KDF import PBKDF2
from Cryptodome.Util.Padding import unpad
from loguru import logger

# Import the new encryption module
from tldw_chatbook.Utils.config_encryption import ConfigEncryption


class LegacyDecryptor:
    """Handles decryption of the old encryption format."""
    
    ENCRYPTION_PREFIX = "enc:"
    SALT_SIZE = 32
    KEY_SIZE = 32
    HMAC_KEY_SIZE = 32
    BLOCK_SIZE = 16
    ITERATIONS = 100000
    VERSION = 2
    
    def derive_keys(self, password: str, salt: bytes) -> Tuple[bytes, bytes]:
        """Derive encryption and HMAC keys from password and salt using PBKDF2."""
        master_key = PBKDF2(
            password.encode('utf-8'),
            salt,
            dkLen=self.KEY_SIZE + self.HMAC_KEY_SIZE,
            count=self.ITERATIONS,
            hmac_hash_module=hashlib.sha256
        )
        
        encryption_key = master_key[:self.KEY_SIZE]
        hmac_key = master_key[self.KEY_SIZE:]
        
        return encryption_key, hmac_key
    
    def verify_password_with_hash(self, password: str, stored_hash: str, salt: bytes) -> bool:
        """Verify password using the old hash method."""
        hash_bytes = PBKDF2(
            password.encode('utf-8'),
            salt,
            dkLen=32,
            count=self.ITERATIONS,
            hmac_hash_module=hashlib.sha256
        )
        computed_hash = base64.b64encode(hash_bytes).decode('utf-8')
        return computed_hash == stored_hash
    
    def verify_password_with_verifier(self, password: str, verifier: str, salt: bytes) -> bool:
        """Verify password using the newer verifier method."""
        try:
            decrypted = self.decrypt_value(verifier, password, salt)
            return decrypted == "CONFIG_ENCRYPTION_PASSWORD_VERIFICATION_V2"
        except ValueError:
            return False
    
    def decrypt_value(self, encrypted_value: str, password: str, salt: bytes) -> str:
        """Decrypt a value using the old format."""
        if encrypted_value.startswith(self.ENCRYPTION_PREFIX):
            encrypted_value = encrypted_value[len(self.ENCRYPTION_PREFIX):]
        
        try:
            combined = base64.b64decode(encrypted_value)
            
            # Check for version 2 format with HMAC
            if len(combined) > 33 and combined[0] == self.VERSION:
                # Version 2 format: VERSION || IV || ENCRYPTED_DATA || HMAC
                stored_mac = combined[-32:]
                message = combined[:-32]
                
                encryption_key, hmac_key = self.derive_keys(password, salt)
                
                # Verify HMAC
                h = hmac.new(hmac_key, message, hashlib.sha256)
                expected_mac = h.digest()
                
                if not hmac.compare_digest(stored_mac, expected_mac):
                    raise ValueError("HMAC verification failed")
                
                # Extract components
                iv = message[1:1+self.BLOCK_SIZE]
                encrypted_data = message[1+self.BLOCK_SIZE:]
            else:
                # Legacy format (version 1): IV || ENCRYPTED_DATA
                iv = combined[:self.BLOCK_SIZE]
                encrypted_data = combined[self.BLOCK_SIZE:]
                
                # Derive key (old method)
                encryption_key = PBKDF2(
                    password.encode('utf-8'),
                    salt,
                    dkLen=self.KEY_SIZE,
                    count=self.ITERATIONS,
                    hmac_hash_module=hashlib.sha256
                )
            
            # Decrypt
            cipher = AES.new(encryption_key, AES.MODE_CBC, iv)
            padded_plaintext = cipher.decrypt(encrypted_data)
            plaintext = unpad(padded_plaintext, self.BLOCK_SIZE)
            
            return plaintext.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Failed to decrypt value")
    
    def is_encrypted(self, value: str) -> bool:
        """Check if a value is encrypted."""
        return isinstance(value, str) and value.startswith(self.ENCRYPTION_PREFIX)
    
    def decrypt_config_section(self, config_dict: Dict[str, Any], password: str, salt: bytes) -> Dict[str, Any]:
        """Decrypt all encrypted values in a config section."""
        decrypted_dict = {}
        
        for key, value in config_dict.items():
            if self.is_encrypted(value):
                try:
                    decrypted_dict[key] = self.decrypt_value(value, password, salt)
                except ValueError:
                    logger.warning(f"Failed to decrypt value for key: {key}")
                    decrypted_dict[key] = value
            else:
                decrypted_dict[key] = value
        
        return decrypted_dict


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load the TOML configuration file."""
    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


def save_config(config_path: Path, config_data: Dict[str, Any]):
    """Save the configuration to TOML file."""
    try:
        # Create backup
        backup_path = config_path.with_suffix('.toml.backup')
        if config_path.exists():
            import shutil
            shutil.copy2(config_path, backup_path)
            logger.info(f"Created backup at: {backup_path}")
        
        # Write new config
        with open(config_path, "w", encoding="utf-8") as f:
            toml.dump(config_data, f)
        
        logger.info(f"Saved new config to: {config_path}")
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        raise


def get_password(prompt: str) -> str:
    """Get password from user with masking."""
    import getpass
    return getpass.getpass(prompt)


def migrate_config(config_path: Path):
    """Main migration function."""
    logger.info("Starting config encryption migration...")
    
    # Load config
    config_data = load_config(config_path)
    
    # Check if encryption is enabled
    encryption_config = config_data.get("encryption", {})
    if not encryption_config.get("enabled", False):
        logger.info("Encryption is not enabled. Nothing to migrate.")
        return
    
    # Get salt
    salt_b64 = encryption_config.get("salt", "")
    if not salt_b64:
        logger.error("No salt found in encryption config.")
        return
    
    salt = base64.b64decode(salt_b64)
    
    # Get password
    print("\n=== Config Encryption Migration ===")
    print("This will migrate your encrypted config to the new format.")
    print("You'll need your current encryption password.\n")
    
    password = get_password("Enter current encryption password: ")
    
    # Verify password
    legacy_decryptor = LegacyDecryptor()
    
    # Try verifier first, then hash
    password_verifier = encryption_config.get("password_verifier", "")
    password_hash = encryption_config.get("password_hash", "")
    
    password_valid = False
    if password_verifier:
        password_valid = legacy_decryptor.verify_password_with_verifier(password, password_verifier, salt)
    elif password_hash:
        password_valid = legacy_decryptor.verify_password_with_hash(password, password_hash, salt)
    
    if not password_valid:
        logger.error("Invalid password.")
        return
    
    logger.info("Password verified successfully.")
    
    # Decrypt all API settings
    decrypted_config = copy.deepcopy(config_data)
    decrypted_count = 0
    
    for section_name, section_value in config_data.items():
        if section_name.startswith('api_settings.') and isinstance(section_value, dict):
            logger.info(f"Decrypting section: {section_name}")
            decrypted_section = legacy_decryptor.decrypt_config_section(section_value, password, salt)
            decrypted_config[section_name] = decrypted_section
            decrypted_count += sum(1 for k, v in section_value.items() if legacy_decryptor.is_encrypted(v))
    
    logger.info(f"Decrypted {decrypted_count} values.")
    
    # Remove old encryption metadata
    if "encryption" in decrypted_config:
        del decrypted_config["encryption"]
    
    # Re-encrypt with new format
    new_encryptor = ConfigEncryption()
    
    # Get new password (or use same)
    print("\nYou can either:")
    print("1. Use the same password with the new encryption")
    print("2. Set a new password")
    
    choice = input("\nUse same password? (y/n): ").lower().strip()
    
    if choice != 'y':
        new_password = get_password("Enter new encryption password: ")
        confirm_password = get_password("Confirm new password: ")
        
        if new_password != confirm_password:
            logger.error("Passwords don't match.")
            return
    else:
        new_password = password
    
    # Set up new encryption metadata
    encryption_config = {
        "enabled": True,
        "method": "AES-256-GCM-scrypt",
        "version": 1,
        "password_verifier": new_encryptor.create_password_verifier(new_password)
    }
    decrypted_config["encryption"] = encryption_config
    
    # Encrypt API settings with new format
    encrypted_count = 0
    for section_name, section_value in decrypted_config.items():
        if section_name.startswith('api_settings.') and isinstance(section_value, dict):
            logger.info(f"Re-encrypting section: {section_name}")
            encrypted_section = new_encryptor.encrypt_config(section_value, new_password)
            decrypted_config[section_name] = encrypted_section
            encrypted_count += sum(1 for k, v in encrypted_section.items() if new_encryptor.is_encrypted(v))
    
    logger.info(f"Re-encrypted {encrypted_count} values with new format.")
    
    # Save the migrated config
    save_config(config_path, decrypted_config)
    
    print("\n✅ Migration completed successfully!")
    print(f"   - Decrypted {decrypted_count} values from old format")
    print(f"   - Re-encrypted {encrypted_count} values with new format")
    print(f"   - Backup saved as: {config_path.with_suffix('.toml.backup')}")
    print("\nYour config now uses AES-256-GCM with scrypt key derivation.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Migrate config encryption to new format")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path.home() / ".config" / "tldw_cli" / "config.toml",
        help="Path to config file (default: ~/.config/tldw_cli/config.toml)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    
    # Check if config exists
    if not args.config_path.exists():
        logger.error(f"Config file not found: {args.config_path}")
        sys.exit(1)
    
    try:
        migrate_config(args.config_path)
    except KeyboardInterrupt:
        print("\n\nMigration cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        if args.verbose:
            logger.exception("Full error trace:")
        sys.exit(1)


if __name__ == "__main__":
    main()