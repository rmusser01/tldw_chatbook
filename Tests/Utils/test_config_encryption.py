"""
Comprehensive test suite for config_encryption module.

Tests AES-256-GCM encryption with scrypt key derivation.
"""
import base64
import json
import pytest
from unittest.mock import patch, MagicMock

from tldw_chatbook.Utils.config_encryption import ConfigEncryption, config_encryption


class TestConfigEncryption:
    """Test suite for ConfigEncryption class."""
    
    @pytest.fixture
    def encryptor(self):
        """Create a fresh ConfigEncryption instance for each test."""
        return ConfigEncryption()
    
    @pytest.fixture
    def test_password(self):
        """Standard test password."""
        return "test_password_123!@#"
    
    @pytest.fixture
    def test_config(self):
        """Sample configuration dictionary."""
        return {
            "api_key": "sk-1234567890abcdef",
            "api_url": "https://api.example.com",
            "timeout": 30,
            "enabled": True,
            "nested": {
                "secret": "nested_secret_value",
                "public": "public_value"
            }
        }
    
    # Basic Encryption/Decryption Tests
    
    def test_encrypt_decrypt_simple_string(self, encryptor, test_password):
        """Test basic encryption and decryption of a simple string."""
        plaintext = "Hello, World!"
        
        encrypted = encryptor.encrypt_value(plaintext, test_password)
        assert encrypted.startswith(encryptor.ENCRYPTION_PREFIX)
        assert encrypted != plaintext
        
        decrypted = encryptor.decrypt_value(encrypted, test_password)
        assert decrypted == plaintext
    
    def test_encrypt_decrypt_empty_string(self, encryptor, test_password):
        """Test encryption of empty string."""
        plaintext = ""
        
        encrypted = encryptor.encrypt_value(plaintext, test_password)
        decrypted = encryptor.decrypt_value(encrypted, test_password)
        assert decrypted == plaintext
    
    def test_encrypt_decrypt_unicode(self, encryptor, test_password):
        """Test encryption of Unicode characters."""
        plaintext = "Hello 世界! 🌍 émojis"
        
        encrypted = encryptor.encrypt_value(plaintext, test_password)
        decrypted = encryptor.decrypt_value(encrypted, test_password)
        assert decrypted == plaintext
    
    def test_encrypt_decrypt_long_string(self, encryptor, test_password):
        """Test encryption of long strings."""
        plaintext = "A" * 10000  # 10KB string
        
        encrypted = encryptor.encrypt_value(plaintext, test_password)
        decrypted = encryptor.decrypt_value(encrypted, test_password)
        assert decrypted == plaintext
    
    def test_encrypt_different_passwords_different_output(self, encryptor):
        """Test that different passwords produce different ciphertexts."""
        plaintext = "test data"
        password1 = "password1"
        password2 = "password2"
        
        encrypted1 = encryptor.encrypt_value(plaintext, password1)
        encrypted2 = encryptor.encrypt_value(plaintext, password2)
        
        assert encrypted1 != encrypted2
    
    def test_encrypt_same_plaintext_different_outputs(self, encryptor, test_password):
        """Test that encrypting same plaintext twice produces different outputs (due to random salt/nonce)."""
        plaintext = "test data"
        
        encrypted1 = encryptor.encrypt_value(plaintext, test_password)
        encrypted2 = encryptor.encrypt_value(plaintext, test_password)
        
        # Different ciphertexts due to random salt and nonce
        assert encrypted1 != encrypted2
        
        # But both decrypt to the same value
        assert encryptor.decrypt_value(encrypted1, test_password) == plaintext
        assert encryptor.decrypt_value(encrypted2, test_password) == plaintext
    
    # Error Handling Tests
    
    def test_decrypt_with_wrong_password(self, encryptor, test_password):
        """Test decryption with wrong password raises ValueError."""
        plaintext = "secret data"
        encrypted = encryptor.encrypt_value(plaintext, test_password)
        
        with pytest.raises(ValueError, match="Failed to decrypt value"):
            encryptor.decrypt_value(encrypted, "wrong_password")
    
    def test_decrypt_corrupted_data(self, encryptor, test_password):
        """Test decryption of corrupted data raises ValueError."""
        # Create valid encrypted data
        encrypted = encryptor.encrypt_value("test", test_password)
        
        # Corrupt the base64 data
        if encrypted.startswith(encryptor.ENCRYPTION_PREFIX):
            encrypted_b64 = encrypted[len(encryptor.ENCRYPTION_PREFIX):]
            corrupted = encryptor.ENCRYPTION_PREFIX + encrypted_b64[:-5] + "XXXXX"
        
        with pytest.raises(ValueError, match="Failed to decrypt value"):
            encryptor.decrypt_value(corrupted, test_password)
    
    def test_decrypt_invalid_base64(self, encryptor, test_password):
        """Test decryption of invalid base64 raises ValueError."""
        invalid_encrypted = encryptor.ENCRYPTION_PREFIX + "not-valid-base64!!!"
        
        with pytest.raises(ValueError, match="Failed to decrypt value"):
            encryptor.decrypt_value(invalid_encrypted, test_password)
    
    def test_decrypt_too_short_data(self, encryptor, test_password):
        """Test decryption of data that's too short raises ValueError."""
        # Create data that's too short to contain all required components
        short_data = base64.b64encode(b"short").decode('utf-8')
        invalid_encrypted = encryptor.ENCRYPTION_PREFIX + short_data
        
        with pytest.raises(ValueError, match="Failed to decrypt value"):
            encryptor.decrypt_value(invalid_encrypted, test_password)
    
    def test_decrypt_wrong_version(self, encryptor, test_password):
        """Test decryption of data with wrong version raises ValueError."""
        # Create data with wrong version byte
        wrong_version_data = bytes([99]) + b"x" * 100  # Version 99 instead of 1
        invalid_encrypted = encryptor.ENCRYPTION_PREFIX + base64.b64encode(wrong_version_data).decode('utf-8')
        
        with pytest.raises(ValueError, match="Failed to decrypt value"):
            encryptor.decrypt_value(invalid_encrypted, test_password)
    
    # Password Verification Tests
    
    def test_create_and_verify_password(self, encryptor, test_password):
        """Test password verification system."""
        verifier = encryptor.create_password_verifier(test_password)
        
        assert encryptor.verify_password(test_password, verifier) is True
        assert encryptor.verify_password("wrong_password", verifier) is False
    
    def test_verify_password_with_invalid_verifier(self, encryptor, test_password):
        """Test password verification with invalid verifier."""
        assert encryptor.verify_password(test_password, "invalid_verifier") is False
        assert encryptor.verify_password(test_password, encryptor.ENCRYPTION_PREFIX + "invalid") is False
    
    # Config Encryption Tests
    
    def test_encrypt_config_simple(self, encryptor, test_password):
        """Test encryption of a simple config dictionary."""
        config = {
            "api_key": "secret_key",
            "timeout": 30,
            "enabled": True
        }
        
        encrypted_config = encryptor.encrypt_config(config, test_password)
        
        # String values should be encrypted
        assert encryptor.is_encrypted(encrypted_config["api_key"])
        # Non-string values should remain unchanged
        assert encrypted_config["timeout"] == 30
        assert encrypted_config["enabled"] is True
    
    def test_encrypt_config_nested(self, encryptor, test_password, test_config):
        """Test encryption of nested config dictionary."""
        encrypted_config = encryptor.encrypt_config(test_config, test_password)
        
        # Top-level strings encrypted
        assert encryptor.is_encrypted(encrypted_config["api_key"])
        assert encryptor.is_encrypted(encrypted_config["api_url"])
        
        # Nested strings encrypted
        assert encryptor.is_encrypted(encrypted_config["nested"]["secret"])
        assert encryptor.is_encrypted(encrypted_config["nested"]["public"])
        
        # Non-strings unchanged
        assert encrypted_config["timeout"] == 30
        assert encrypted_config["enabled"] is True
    
    def test_encrypt_config_already_encrypted(self, encryptor, test_password):
        """Test that already encrypted values are not re-encrypted."""
        # First encryption
        config = {"api_key": "secret"}
        encrypted_once = encryptor.encrypt_config(config, test_password)
        
        # Second encryption
        encrypted_twice = encryptor.encrypt_config(encrypted_once, test_password)
        
        # Should be the same (not double-encrypted)
        assert encrypted_once["api_key"] == encrypted_twice["api_key"]
    
    def test_decrypt_config_simple(self, encryptor, test_password):
        """Test decryption of encrypted config."""
        config = {"api_key": "secret_key", "timeout": 30}
        
        encrypted_config = encryptor.encrypt_config(config, test_password)
        decrypted_config = encryptor.decrypt_config(encrypted_config, test_password)
        
        assert decrypted_config == config
    
    def test_decrypt_config_nested(self, encryptor, test_password, test_config):
        """Test decryption of nested encrypted config."""
        encrypted_config = encryptor.encrypt_config(test_config, test_password)
        decrypted_config = encryptor.decrypt_config(encrypted_config, test_password)
        
        assert decrypted_config == test_config
    
    def test_decrypt_config_partial_failure(self, encryptor, test_password):
        """Test that partial decryption failure doesn't crash."""
        config = {"api_key": "secret", "backup_key": "backup_secret"}
        encrypted_config = encryptor.encrypt_config(config, test_password)
        
        # Corrupt one encrypted value
        encrypted_config["api_key"] = encryptor.ENCRYPTION_PREFIX + "corrupted_data"
        
        # Should decrypt what it can
        with patch('loguru.logger.warning') as mock_warning:
            decrypted_config = encryptor.decrypt_config(encrypted_config, test_password)
            
            # Corrupted value kept as-is
            assert decrypted_config["api_key"] == encryptor.ENCRYPTION_PREFIX + "corrupted_data"
            # Valid value decrypted
            assert decrypted_config["backup_key"] == "backup_secret"
            
            # Warning logged
            mock_warning.assert_called_once()
    
    # API Key Detection Tests
    
    def test_detect_api_keys_simple(self, encryptor):
        """Test detection of API keys in config."""
        config = {"api_key": "sk-12345"}
        assert encryptor.detect_api_keys(config) is True
        
        config = {"regular_setting": "value"}
        assert encryptor.detect_api_keys(config) is False
    
    def test_detect_api_keys_variations(self, encryptor):
        """Test detection of various API key patterns."""
        # Different key names
        for key_name in ["api_key", "apikey", "api-key", "secret", "token", "password"]:
            config = {key_name: "secret_value"}
            assert encryptor.detect_api_keys(config) is True
        
        # Case insensitive
        config = {"API_KEY": "secret"}
        assert encryptor.detect_api_keys(config) is True
    
    def test_detect_api_keys_nested(self, encryptor):
        """Test detection of API keys in nested structures."""
        config = {
            "database": {
                "host": "localhost",
                "password": "db_password"
            }
        }
        assert encryptor.detect_api_keys(config) is True
    
    def test_detect_api_keys_placeholder(self, encryptor):
        """Test that placeholders are not detected as API keys."""
        config = {
            "api_key": "<your-api-key-here>",
            "token": "<placeholder>"
        }
        assert encryptor.detect_api_keys(config) is False
    
    def test_detect_api_keys_already_encrypted(self, encryptor, test_password):
        """Test that already encrypted values are not detected as needing encryption."""
        config = {"api_key": "secret"}
        encrypted_config = encryptor.encrypt_config(config, test_password)
        
        assert encryptor.detect_api_keys(encrypted_config) is False
    
    def test_detect_api_keys_empty_values(self, encryptor):
        """Test that empty values are not detected as API keys."""
        config = {
            "api_key": "",
            "token": None,
            "secret": "   "  # Whitespace only
        }
        assert encryptor.detect_api_keys(config) is False
    
    # Utility Method Tests
    
    def test_is_encrypted(self, encryptor):
        """Test is_encrypted method."""
        assert encryptor.is_encrypted("enc:abcdef") is True
        assert encryptor.is_encrypted("regular_value") is False
        assert encryptor.is_encrypted("") is False
        assert encryptor.is_encrypted(None) is False
        assert encryptor.is_encrypted(123) is False
    
    # Format Validation Tests
    
    def test_encrypted_format_structure(self, encryptor, test_password):
        """Test that encrypted format follows expected structure."""
        plaintext = "test"
        encrypted = encryptor.encrypt_value(plaintext, test_password)
        
        # Remove prefix
        encrypted_b64 = encrypted[len(encryptor.ENCRYPTION_PREFIX):]
        
        # Decode base64
        combined = base64.b64decode(encrypted_b64)
        
        # Verify structure
        assert len(combined) >= 1 + encryptor.SALT_SIZE + encryptor.NONCE_SIZE + encryptor.TAG_SIZE
        assert combined[0] == encryptor.VERSION
    
    # Performance Tests
    
    def test_encryption_performance(self, encryptor, test_password):
        """Test that encryption completes in reasonable time."""
        import time
        
        plaintext = "A" * 1000  # 1KB
        iterations = 100
        
        start = time.time()
        for _ in range(iterations):
            encrypted = encryptor.encrypt_value(plaintext, test_password)
            decrypted = encryptor.decrypt_value(encrypted, test_password)
            assert decrypted == plaintext
        
        duration = time.time() - start
        avg_time = duration / iterations
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert avg_time < 0.1  # 100ms per operation
    
    # Integration Tests
    
    def test_full_config_roundtrip(self, encryptor, test_password):
        """Test full encryption/decryption cycle with complex config."""
        config = {
            "api_settings": {
                "openai": {
                    "api_key": "sk-openai-key",
                    "model": "gpt-4",
                    "temperature": 0.7
                },
                "anthropic": {
                    "api_key": "sk-anthropic-key",
                    "model": "claude-3"
                }
            },
            "general": {
                "debug": True,
                "log_level": "INFO"
            }
        }
        
        # Encrypt
        encrypted = encryptor.encrypt_config(config, test_password)
        
        # Verify encryption happened
        assert encryptor.is_encrypted(encrypted["api_settings"]["openai"]["api_key"])
        assert encryptor.is_encrypted(encrypted["api_settings"]["anthropic"]["api_key"])
        
        # Decrypt
        decrypted = encryptor.decrypt_config(encrypted, test_password)
        
        # Verify roundtrip
        assert decrypted == config
    
    def test_global_instance(self):
        """Test that global instance works correctly."""
        from tldw_chatbook.Utils.config_encryption import config_encryption
        
        password = "test123"
        plaintext = "test data"
        
        encrypted = config_encryption.encrypt_value(plaintext, password)
        decrypted = config_encryption.decrypt_value(encrypted, password)
        
        assert decrypted == plaintext


class TestConfigEncryptionEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_long_password(self):
        """Test encryption with very long password."""
        encryptor = ConfigEncryption()
        password = "x" * 10000  # 10KB password
        plaintext = "test"
        
        encrypted = encryptor.encrypt_value(plaintext, password)
        decrypted = encryptor.decrypt_value(encrypted, password)
        assert decrypted == plaintext
    
    def test_special_characters_in_password(self):
        """Test encryption with special characters in password."""
        encryptor = ConfigEncryption()
        password = "!@#$%^&*()_+-=[]{}|;':\",./<>?`~"
        plaintext = "test"
        
        encrypted = encryptor.encrypt_value(plaintext, password)
        decrypted = encryptor.decrypt_value(encrypted, password)
        assert decrypted == plaintext
    
    def test_binary_data_rejection(self):
        """Test that binary data is properly handled."""
        encryptor = ConfigEncryption()
        password = "test"
        
        # encrypt_value expects string, not bytes
        with pytest.raises(AttributeError):
            encryptor.encrypt_value(b"binary data", password)
    
    @patch('os.urandom')
    def test_random_generation_failure(self, mock_urandom):
        """Test handling of random generation failure."""
        encryptor = ConfigEncryption()
        mock_urandom.side_effect = OSError("Random generation failed")
        
        with pytest.raises(OSError):
            encryptor.encrypt_value("test", "password")
    
    def test_concurrent_encryption(self):
        """Test thread safety of encryption operations."""
        import threading
        import queue
        
        encryptor = ConfigEncryption()
        password = "test123"
        results = queue.Queue()
        errors = queue.Queue()
        
        def encrypt_worker(text, idx):
            try:
                encrypted = encryptor.encrypt_value(f"{text}_{idx}", password)
                decrypted = encryptor.decrypt_value(encrypted, password)
                results.put((idx, decrypted))
            except Exception as e:
                errors.put((idx, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=encrypt_worker, args=("test", i))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check results
        assert errors.empty(), "Encryption errors occurred in threads"
        assert results.qsize() == 10, "Not all threads completed"
        
        # Verify all results
        results_dict = {}
        while not results.empty():
            idx, decrypted = results.get()
            results_dict[idx] = decrypted
        
        for i in range(10):
            assert results_dict[i] == f"test_{i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])