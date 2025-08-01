# site_config_manager.py
# Description: Per-site configuration management for subscriptions
#
# This module manages site-specific configurations including:
# - Rate limiting per domain
# - Custom headers and authentication
# - JavaScript rendering settings
# - Content extraction rules
# - Ignore patterns and selectors
#
# Imports
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse
import threading
from collections import defaultdict
#
# Third-Party Imports
from loguru import logger
#
# Local Imports
from ..DB.ChaChaNotes_DB import ChaChaNotes_DB
from ..Security.config_encryption import ConfigEncryption
from ..Metrics.metrics_logger import log_counter, log_histogram
#
########################################################################################################################
#
# Site Configuration Manager
#
########################################################################################################################

class SiteConfig:
    """Configuration for a specific site/domain."""
    
    def __init__(self, domain: str, config_data: Optional[Dict[str, Any]] = None):
        """
        Initialize site configuration.
        
        Args:
            domain: The domain name (e.g., 'example.com')
            config_data: Optional existing configuration data
        """
        self.domain = domain
        
        # Load from config_data or use defaults
        config = config_data or {}
        
        # Rate limiting
        self.rate_limit_requests = config.get('rate_limit_requests', 60)  # requests per minute
        self.rate_limit_period = config.get('rate_limit_period', 60)  # seconds
        self.concurrent_requests = config.get('concurrent_requests', 1)
        
        # Request settings
        self.timeout = config.get('timeout', 30)  # seconds
        self.retry_count = config.get('retry_count', 3)
        self.retry_delay = config.get('retry_delay', 1)  # seconds
        
        # Headers and authentication
        self.custom_headers = config.get('custom_headers', {})
        self.auth_type = config.get('auth_type')  # 'basic', 'bearer', 'api_key'
        self.auth_credentials = config.get('auth_credentials', {})
        self.cookies = config.get('cookies', {})
        
        # JavaScript rendering
        self.requires_javascript = config.get('requires_javascript', False)
        self.wait_for_selector = config.get('wait_for_selector')
        self.wait_timeout = config.get('wait_timeout', 10)  # seconds
        self.viewport_width = config.get('viewport_width', 1920)
        self.viewport_height = config.get('viewport_height', 1080)
        
        # Content extraction
        self.content_selector = config.get('content_selector')
        self.title_selector = config.get('title_selector')
        self.date_selector = config.get('date_selector')
        self.author_selector = config.get('author_selector')
        self.exclude_selectors = config.get('exclude_selectors', [])
        
        # Change detection
        self.ignore_selectors = config.get('ignore_selectors', [])
        self.ignore_patterns = config.get('ignore_patterns', [])
        self.change_threshold = config.get('change_threshold', 0.1)
        
        # Content processing
        self.remove_scripts = config.get('remove_scripts', True)
        self.remove_styles = config.get('remove_styles', True)
        self.preserve_links = config.get('preserve_links', True)
        self.extract_images = config.get('extract_images', False)
        
        # Metadata
        self.notes = config.get('notes', '')
        self.created_at = config.get('created_at', datetime.now().isoformat())
        self.updated_at = config.get('updated_at', datetime.now().isoformat())
        self.last_error = config.get('last_error')
        self.success_count = config.get('success_count', 0)
        self.error_count = config.get('error_count', 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'domain': self.domain,
            'rate_limit_requests': self.rate_limit_requests,
            'rate_limit_period': self.rate_limit_period,
            'concurrent_requests': self.concurrent_requests,
            'timeout': self.timeout,
            'retry_count': self.retry_count,
            'retry_delay': self.retry_delay,
            'custom_headers': self.custom_headers,
            'auth_type': self.auth_type,
            'auth_credentials': self.auth_credentials,
            'cookies': self.cookies,
            'requires_javascript': self.requires_javascript,
            'wait_for_selector': self.wait_for_selector,
            'wait_timeout': self.wait_timeout,
            'viewport_width': self.viewport_width,
            'viewport_height': self.viewport_height,
            'content_selector': self.content_selector,
            'title_selector': self.title_selector,
            'date_selector': self.date_selector,
            'author_selector': self.author_selector,
            'exclude_selectors': self.exclude_selectors,
            'ignore_selectors': self.ignore_selectors,
            'ignore_patterns': self.ignore_patterns,
            'change_threshold': self.change_threshold,
            'remove_scripts': self.remove_scripts,
            'remove_styles': self.remove_styles,
            'preserve_links': self.preserve_links,
            'extract_images': self.extract_images,
            'notes': self.notes,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'last_error': self.last_error,
            'success_count': self.success_count,
            'error_count': self.error_count
        }
    
    def get_headers(self, base_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Get merged headers for requests.
        
        Args:
            base_headers: Base headers to merge with
            
        Returns:
            Merged headers dictionary
        """
        headers = base_headers.copy() if base_headers else {}
        headers.update(self.custom_headers)
        
        # Add authentication headers
        if self.auth_type == 'bearer' and 'token' in self.auth_credentials:
            headers['Authorization'] = f"Bearer {self.auth_credentials['token']}"
        elif self.auth_type == 'api_key':
            key_name = self.auth_credentials.get('key_name', 'X-API-Key')
            key_value = self.auth_credentials.get('key_value', '')
            if key_value:
                headers[key_name] = key_value
        
        return headers
    
    def get_auth(self) -> Optional[Tuple[str, str]]:
        """Get basic auth tuple if configured."""
        if self.auth_type == 'basic':
            username = self.auth_credentials.get('username')
            password = self.auth_credentials.get('password')
            if username and password:
                return (username, password)
        return None
    
    def record_success(self):
        """Record a successful request."""
        self.success_count += 1
        self.updated_at = datetime.now().isoformat()
    
    def record_error(self, error: str):
        """Record a failed request."""
        self.error_count += 1
        self.last_error = error
        self.updated_at = datetime.now().isoformat()


class RateLimiter:
    """Simple rate limiter for domain-specific request throttling."""
    
    def __init__(self):
        """Initialize rate limiter."""
        self._locks = defaultdict(threading.Lock)
        self._request_times = defaultdict(list)
    
    def check_rate_limit(self, domain: str, config: SiteConfig) -> Tuple[bool, float]:
        """
        Check if a request can be made according to rate limits.
        
        Args:
            domain: The domain to check
            config: Site configuration with rate limits
            
        Returns:
            Tuple of (can_request, wait_time_seconds)
        """
        with self._locks[domain]:
            now = datetime.now()
            cutoff = now - timedelta(seconds=config.rate_limit_period)
            
            # Clean old request times
            self._request_times[domain] = [
                t for t in self._request_times[domain]
                if t > cutoff
            ]
            
            # Check if under limit
            if len(self._request_times[domain]) < config.rate_limit_requests:
                self._request_times[domain].append(now)
                return True, 0.0
            
            # Calculate wait time
            oldest_request = min(self._request_times[domain])
            wait_until = oldest_request + timedelta(seconds=config.rate_limit_period)
            wait_seconds = (wait_until - now).total_seconds()
            
            return False, max(0, wait_seconds)
    
    def clear_domain(self, domain: str):
        """Clear rate limit history for a domain."""
        with self._locks[domain]:
            self._request_times[domain].clear()


class SiteConfigManager:
    """Manages per-site configurations for subscriptions."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize site config manager.
        
        Args:
            db_path: Optional path to database
        """
        self.db = ChaChaNotes_DB(db_path)
        self.encryption = ConfigEncryption()
        self.rate_limiter = RateLimiter()
        self._config_cache = {}
        self._cache_lock = threading.Lock()
        
        # Create site_configs table if it doesn't exist
        self._create_tables()
    
    def _create_tables(self):
        """Create site configuration tables."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS site_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT UNIQUE NOT NULL,
                    config_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_site_configs_domain
                ON site_configs(domain)
            """)
            
            conn.commit()
    
    def get_config(self, url: str) -> SiteConfig:
        """
        Get configuration for a URL, creating default if needed.
        
        Args:
            url: The URL to get configuration for
            
        Returns:
            Site configuration
        """
        domain = self._extract_domain(url)
        
        # Check cache first
        with self._cache_lock:
            if domain in self._config_cache:
                return self._config_cache[domain]
        
        # Load from database
        config = self._load_config(domain)
        if not config:
            # Create default config
            config = SiteConfig(domain)
            self._save_config(config)
        
        # Cache it
        with self._cache_lock:
            self._config_cache[domain] = config
        
        return config
    
    def save_config(self, config: SiteConfig) -> bool:
        """
        Save site configuration.
        
        Args:
            config: Site configuration to save
            
        Returns:
            Success status
        """
        success = self._save_config(config)
        
        if success:
            # Update cache
            with self._cache_lock:
                self._config_cache[config.domain] = config
        
        return success
    
    def _save_config(self, config: SiteConfig) -> bool:
        """Save configuration to database."""
        try:
            # Update timestamp
            config.updated_at = datetime.now().isoformat()
            
            # Encrypt sensitive data
            config_data = config.to_dict()
            if config_data.get('auth_credentials'):
                config_data['auth_credentials'] = self._encrypt_dict(config_data['auth_credentials'])
            if config_data.get('cookies'):
                config_data['cookies'] = self._encrypt_dict(config_data['cookies'])
            
            config_json = json.dumps(config_data)
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO site_configs 
                    (domain, config_data, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (config.domain, config_json))
                
                conn.commit()
                
                log_counter("site_config_saved", labels={"domain": config.domain})
                return True
                
        except Exception as e:
            logger.error(f"Error saving site config: {str(e)}")
            return False
    
    def _load_config(self, domain: str) -> Optional[SiteConfig]:
        """Load configuration from database."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT config_data
                    FROM site_configs
                    WHERE domain = ?
                """, (domain,))
                
                row = cursor.fetchone()
                if row:
                    config_data = json.loads(row[0])
                    
                    # Decrypt sensitive data
                    if config_data.get('auth_credentials'):
                        config_data['auth_credentials'] = self._decrypt_dict(config_data['auth_credentials'])
                    if config_data.get('cookies'):
                        config_data['cookies'] = self._decrypt_dict(config_data['cookies'])
                    
                    return SiteConfig(domain, config_data)
                
                return None
                
        except Exception as e:
            logger.error(f"Error loading site config: {str(e)}")
            return None
    
    def delete_config(self, domain: str) -> bool:
        """
        Delete site configuration.
        
        Args:
            domain: Domain to delete config for
            
        Returns:
            Success status
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM site_configs
                    WHERE domain = ?
                """, (domain,))
                
                conn.commit()
                
                # Clear from cache
                with self._cache_lock:
                    self._config_cache.pop(domain, None)
                
                # Clear rate limit history
                self.rate_limiter.clear_domain(domain)
                
                return True
                
        except Exception as e:
            logger.error(f"Error deleting site config: {str(e)}")
            return False
    
    def list_configs(self) -> List[Dict[str, Any]]:
        """List all site configurations."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT domain, config_data, created_at, updated_at
                    FROM site_configs
                    ORDER BY domain
                """)
                
                configs = []
                for row in cursor.fetchall():
                    config_data = json.loads(row[1])
                    configs.append({
                        'domain': row[0],
                        'created_at': row[2],
                        'updated_at': row[3],
                        'rate_limit': f"{config_data.get('rate_limit_requests', 60)}/min",
                        'requires_js': config_data.get('requires_javascript', False),
                        'has_auth': bool(config_data.get('auth_type')),
                        'success_count': config_data.get('success_count', 0),
                        'error_count': config_data.get('error_count', 0)
                    })
                
                return configs
                
        except Exception as e:
            logger.error(f"Error listing site configs: {str(e)}")
            return []
    
    def get_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get preset configurations for common sites."""
        return {
            'github.com': {
                'rate_limit_requests': 60,
                'custom_headers': {
                    'Accept': 'application/vnd.github.v3+json'
                },
                'content_selector': '.markdown-body',
                'requires_javascript': False
            },
            'reddit.com': {
                'rate_limit_requests': 60,
                'custom_headers': {
                    'User-Agent': 'SubscriptionBot/1.0'
                },
                'requires_javascript': True,
                'wait_for_selector': '[data-testid="post-container"]'
            },
            'twitter.com': {
                'rate_limit_requests': 15,
                'requires_javascript': True,
                'wait_for_selector': 'article',
                'viewport_height': 2000
            },
            'medium.com': {
                'rate_limit_requests': 30,
                'content_selector': 'article',
                'exclude_selectors': ['.pw-responses', '.js-postShareWidget'],
                'requires_javascript': True
            },
            'substack.com': {
                'rate_limit_requests': 60,
                'content_selector': '.available-content',
                'requires_javascript': False
            },
            'arxiv.org': {
                'rate_limit_requests': 30,
                'content_selector': '.ltx_document',
                'requires_javascript': False
            }
        }
    
    def apply_preset(self, domain: str, preset_name: str) -> bool:
        """
        Apply a preset configuration to a domain.
        
        Args:
            domain: Domain to configure
            preset_name: Name of preset to apply
            
        Returns:
            Success status
        """
        presets = self.get_presets()
        if preset_name not in presets:
            logger.error(f"Unknown preset: {preset_name}")
            return False
        
        # Get existing config or create new
        config = self.get_config(f"https://{domain}")
        
        # Apply preset values
        preset_data = presets[preset_name]
        for key, value in preset_data.items():
            setattr(config, key, value)
        
        return self.save_config(config)
    
    def check_rate_limit(self, url: str) -> Tuple[bool, float]:
        """
        Check if a request can be made to a URL.
        
        Args:
            url: URL to check
            
        Returns:
            Tuple of (can_request, wait_time_seconds)
        """
        domain = self._extract_domain(url)
        config = self.get_config(url)
        
        return self.rate_limiter.check_rate_limit(domain, config)
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split('/')[0]
        
        # Remove www prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        
        return domain.lower()
    
    def _encrypt_dict(self, data: dict) -> dict:
        """Encrypt sensitive values in dictionary."""
        encrypted = {}
        for key, value in data.items():
            if isinstance(value, str) and value:
                encrypted[key] = self.encryption.encrypt_value(value)
            else:
                encrypted[key] = value
        return encrypted
    
    def _decrypt_dict(self, data: dict) -> dict:
        """Decrypt sensitive values in dictionary."""
        decrypted = {}
        for key, value in data.items():
            if isinstance(value, str) and value.startswith('encrypted:'):
                try:
                    decrypted[key] = self.encryption.decrypt_value(value)
                except Exception:
                    decrypted[key] = ''  # Failed to decrypt
            else:
                decrypted[key] = value
        return decrypted
    
    def export_configs(self, export_path: Path) -> bool:
        """
        Export all configurations to file.
        
        Args:
            export_path: Path to export to
            
        Returns:
            Success status
        """
        try:
            configs = []
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT domain, config_data
                    FROM site_configs
                    ORDER BY domain
                """)
                
                for row in cursor.fetchall():
                    # Don't export encrypted credentials
                    config_data = json.loads(row[1])
                    config_data.pop('auth_credentials', None)
                    config_data.pop('cookies', None)
                    
                    configs.append({
                        'domain': row[0],
                        'config': config_data
                    })
            
            # Write export file
            export_data = {
                'version': '1.0',
                'exported_at': datetime.now().isoformat(),
                'configs': configs
            }
            
            export_path.write_text(json.dumps(export_data, indent=2))
            logger.info(f"Exported {len(configs)} site configs")
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting configs: {str(e)}")
            return False
    
    def import_configs(self, import_path: Path, merge: bool = True) -> int:
        """
        Import configurations from file.
        
        Args:
            import_path: Path to import from
            merge: Whether to merge with existing configs
            
        Returns:
            Number of configs imported
        """
        try:
            import_data = json.loads(import_path.read_text())
            configs = import_data.get('configs', [])
            
            imported = 0
            for config_entry in configs:
                domain = config_entry['domain']
                config_data = config_entry['config']
                
                if merge:
                    # Merge with existing
                    existing = self.get_config(f"https://{domain}")
                    for key, value in config_data.items():
                        if key not in ['auth_credentials', 'cookies']:  # Skip sensitive
                            setattr(existing, key, value)
                    
                    if self.save_config(existing):
                        imported += 1
                else:
                    # Replace entirely
                    new_config = SiteConfig(domain, config_data)
                    if self.save_config(new_config):
                        imported += 1
            
            logger.info(f"Imported {imported} site configs")
            return imported
            
        except Exception as e:
            logger.error(f"Error importing configs: {str(e)}")
            return 0


# Global instance
_site_config_manager = None

def get_site_config_manager(db_path: Optional[str] = None) -> SiteConfigManager:
    """Get global site config manager instance."""
    global _site_config_manager
    if _site_config_manager is None:
        _site_config_manager = SiteConfigManager(db_path)
    return _site_config_manager


# End of site_config_manager.py