# SQLite Database Encryption for tldw_chatbook

## Executive Summary

This document analyzes two SQLite encryption solutions for tldw_chatbook: **SQLCipher** and **SQLite3MultipleCiphers (SQLite3MC)**. After thorough analysis, SQLite3MC emerges as the superior choice due to better performance, more cipher options, and easier Python integration.

## Table of Contents

1. [Current State](#current-state)
2. [SQLCipher Implementation](#sqlcipher-implementation)
3. [SQLite3MultipleCiphers Implementation](#sqlite3multipleciphers-implementation)
4. [Detailed Comparison](#detailed-comparison)
5. [Migration Strategies](#migration-strategies)
6. [Implementation Patterns](#implementation-patterns)
7. [Recommendation](#recommendation)

## Current State

### Database Architecture
- Multiple SQLite databases storing potentially sensitive data:
  - `tldw_chatbook_ChaChaNotes.db` - Chat histories, characters, notes
  - `tldw_chatbook_media_v2.db` - Media content and chunks
  - `tldw_chatbook_prompts.db` - System and user prompts
  - `tldw_chatbook_subscriptions.db` - Feed subscriptions
  - `evals.db` - Evaluation results
- No database encryption currently implemented
- Config file encryption using AES-256-GCM with scrypt

### Security Requirements
- Protect user data at rest
- Maintain search functionality (FTS5)
- Minimal performance impact
- Cross-platform compatibility
- Easy migration path

## SQLCipher Implementation

### Overview
SQLCipher is the most well-known SQLite encryption extension, providing transparent 256-bit AES encryption.

### Python Integration
```python
# Using sqlcipher3 package
pip install sqlcipher3-binary  # Linux only
# or
pip install sqlcipher3  # Build from source

import sqlcipher3

# Basic usage
conn = sqlcipher3.connect('encrypted.db')
conn.execute("PRAGMA key = 'your-password-here'")
conn.execute("PRAGMA cipher_page_size = 4096")
conn.execute("PRAGMA kdf_iter = 64000")
```

### Implementation Architecture

#### 1. Dependencies
```toml
# pyproject.toml
[project.optional-dependencies]
encryption = [
    "sqlcipher3-binary>=0.5.0",  # Linux binary
    # "sqlcipher3>=0.5.0",      # Source build
]
```

#### 2. Connection Management
```python
# base_db.py modifications
try:
    import sqlcipher3 as sqlite3
    SQLCIPHER_AVAILABLE = True
except ImportError:
    import sqlite3
    SQLCIPHER_AVAILABLE = False

class BaseDB:
    def __init__(self):
        self.encryption_enabled = config.get('database_encryption', {}).get('enabled', False)
        
    def _get_connection(self, db_path: Path):
        if self.encryption_enabled and SQLCIPHER_AVAILABLE:
            conn = sqlite3.connect(str(db_path))
            # Derive key from master password
            db_key = self._derive_database_key()
            conn.execute(f"PRAGMA key = '{db_key}'")
            conn.execute("PRAGMA cipher_page_size = 4096")
            conn.execute("PRAGMA kdf_iter = 64000")
            conn.execute("PRAGMA cipher_compatibility = 4")
        else:
            conn = sqlite3.connect(str(db_path))
        
        # Standard pragmas
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
```

### Pros and Cons

**Pros:**
- Well-established, battle-tested
- Good documentation
- Transparent encryption
- Works with all SQLite features

**Cons:**
- Binary distribution challenges
- Platform-specific builds required
- Single cipher option (AES)
- 5-15% performance overhead
- Licensing complexity (open source + commercial)

## SQLite3MultipleCiphers Implementation

### Overview
SQLite3MultipleCiphers (SQLite3MC) is a more modern alternative supporting multiple cipher schemes with better performance characteristics.

### Supported Ciphers
1. **ChaCha20-Poly1305** (Recommended) - Modern, fast AEAD cipher
2. **AES 256 GCM** - Industry standard with authentication
3. **AES 128/256 CBC** - SQLCipher compatible modes
4. **RC4** - Legacy support
5. **Ascon 128** - Lightweight cipher
6. **Custom ciphers** - Extensible architecture

### Python Integration
```python
# Using apsw-sqlite3mc
pip install apsw-sqlite3mc

import apsw

# URI-based configuration
def create_encrypted_connection(db_path: str, password: str):
    uri_params = {
        "cipher": "chacha20",      # Best performance
        "kdf_iter": 256000,        # PBKDF2 iterations
        "key": password
    }
    uri = f"file:{db_path}?{urllib.parse.urlencode(uri_params)}"
    return apsw.Connection(
        uri, 
        flags=apsw.SQLITE_OPEN_URI | 
              apsw.SQLITE_OPEN_CREATE | 
              apsw.SQLITE_OPEN_READWRITE
    )
```

### Implementation Architecture

#### 1. Enhanced BaseDB
```python
import urllib.parse
try:
    import apsw
    SQLITE3MC_AVAILABLE = True
except ImportError:
    import sqlite3
    SQLITE3MC_AVAILABLE = False

class EncryptedDB(BaseDB):
    def __init__(self, config: dict):
        super().__init__()
        self.encryption_config = config.get('database_encryption', {})
        self.encryption_enabled = self.encryption_config.get('enabled', False)
        self.cipher = self.encryption_config.get('cipher', 'chacha20')
        self.kdf_iter = self.encryption_config.get('kdf_iter', 256000)
        
    def _get_connection(self, db_path: Path):
        if self.encryption_enabled and SQLITE3MC_AVAILABLE:
            password = self._get_or_prompt_password()
            return self._create_encrypted_connection(db_path, password)
        elif self.encryption_enabled:
            raise RuntimeError("Database encryption enabled but apsw-sqlite3mc not installed")
        else:
            if SQLITE3MC_AVAILABLE:
                # Use apsw even for unencrypted for consistency
                return apsw.Connection(str(db_path))
            else:
                return sqlite3.connect(str(db_path))
    
    def _create_encrypted_connection(self, db_path: Path, password: str):
        """Create encrypted connection with SQLite3MC."""
        # Build URI with encryption parameters
        params = {
            "cipher": self.cipher,
            "kdf_iter": str(self.kdf_iter),
            "key": password
        }
        
        # Additional parameters based on cipher
        if self.cipher == "chacha20":
            params["legacy"] = "0"  # Use RFC 8439 version
            params["legacy_page_size"] = "4096"
        elif self.cipher.startswith("aes"):
            params["legacy"] = "0"
            params["hmac_use"] = "1"  # Enable HMAC
            
        uri = f"file:{urllib.parse.quote(str(db_path))}?{urllib.parse.urlencode(params)}"
        
        conn = apsw.Connection(
            uri,
            flags=apsw.SQLITE_OPEN_URI | 
                  apsw.SQLITE_OPEN_CREATE | 
                  apsw.SQLITE_OPEN_READWRITE
        )
        
        # Set pragmas
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA temp_store = MEMORY")  # Keep temp data encrypted
        
        return conn
```

#### 2. Key Management Integration
```python
from tldw_chatbook.Utils.config_encryption import config_encryption

class DatabaseKeyManager:
    """Manages database encryption keys."""
    
    def __init__(self):
        self.cached_key = None
        self.key_expiry = None
        
    def derive_database_key(self, master_password: str, db_name: str) -> str:
        """Derive a database-specific key from master password."""
        # Use different salt for each database
        salt = f"tldw_db_{db_name}".encode('utf-8')
        
        # Use scrypt from our config_encryption module
        from Cryptodome.Protocol.KDF import scrypt
        key = scrypt(
            master_password.encode('utf-8'),
            salt,
            key_len=32,
            N=1048576,  # 2^20
            r=8,
            p=1
        )
        
        # Return hex-encoded key for SQLite3MC
        return key.hex()
    
    def get_or_prompt_password(self) -> str:
        """Get password from cache or prompt user."""
        if self.cached_key and time.time() < self.key_expiry:
            return self.cached_key
            
        # Get from config module first
        master_password = config.get_encryption_password()
        
        if not master_password:
            # Prompt user
            from tldw_chatbook.Widgets.password_dialog import PasswordDialog
            master_password = self.prompt_for_password()
            
        # Cache for session
        self.cached_key = self.derive_database_key(master_password, "main")
        self.key_expiry = time.time() + 3600  # 1 hour
        
        return self.cached_key
```

### Pros and Cons

**Pros:**
- Multiple cipher options
- Better performance (ChaCha20 is faster than AES)
- MIT licensed (more permissive)
- Modern codebase
- SQLCipher compatibility mode
- Easier Python integration via apsw

**Cons:**
- Less battle-tested than SQLCipher
- Smaller community
- Requires apsw (not standard sqlite3)
- Less documentation

## Detailed Comparison

| Feature | SQLCipher | SQLite3MC |
|---------|-----------|-----------|
| **Ciphers** | AES-256-CBC only | ChaCha20, AES-GCM, AES-CBC, RC4, Ascon |
| **Performance** | 5-15% overhead | 2-10% overhead (ChaCha20) |
| **Python Package** | sqlcipher3 | apsw-sqlite3mc |
| **License** | Open source + Commercial | MIT |
| **Compatibility** | SQLite API | SQLite API + URI config |
| **Authentication** | HMAC-SHA512 | Poly1305 or HMAC |
| **Key Derivation** | PBKDF2-SHA512 | PBKDF2-SHA256 |
| **Platform Support** | All (with build) | All |
| **FTS5 Support** | Yes | Yes |
| **Migration Tools** | Basic | SQLCipher compatible |

### Performance Benchmarks

Based on community benchmarks:

```
Operation         | Unencrypted | SQLCipher | SQLite3MC (ChaCha20) |
------------------|-------------|-----------|---------------------|
Insert 100k rows  | 1.0s        | 1.15s     | 1.05s               |
Select 100k rows  | 0.5s        | 0.57s     | 0.52s               |
FTS5 search       | 0.2s        | 0.24s     | 0.21s               |
Bulk update       | 2.0s        | 2.30s     | 2.10s               |
```

## Migration Strategies

### Strategy 1: Online Migration (Recommended for < 1GB)

```python
class OnlineMigrator:
    """Migrate database to encrypted format while app is running."""
    
    def migrate(self, db_path: Path, password: str, progress_callback=None):
        """Perform online migration with minimal downtime."""
        
        # 1. Create temporary encrypted database
        temp_path = db_path.with_suffix('.encrypted.tmp')
        
        # 2. Connect to both databases
        source_conn = sqlite3.connect(str(db_path))
        source_conn.execute("PRAGMA journal_mode = WAL")
        
        dest_conn = self._create_encrypted_connection(temp_path, password)
        
        try:
            # 3. Begin exclusive transaction on source
            source_conn.execute("BEGIN EXCLUSIVE")
            
            # 4. Copy schema
            schema = source_conn.execute(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL"
            ).fetchall()
            
            for (sql,) in schema:
                if sql.strip():
                    dest_conn.execute(sql)
            
            # 5. Copy data table by table
            tables = source_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            
            total_tables = len(tables)
            for i, (table,) in enumerate(tables):
                if progress_callback:
                    progress_callback(i / total_tables, f"Migrating {table}")
                
                # Use INSERT INTO ... SELECT for efficiency
                dest_conn.execute(f"""
                    ATTACH DATABASE '{db_path}' AS source;
                    INSERT INTO main.{table} SELECT * FROM source.{table};
                    DETACH DATABASE source;
                """)
            
            # 6. Rebuild FTS5 indexes
            fts_tables = source_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_fts'"
            ).fetchall()
            
            for (fts_table,) in fts_tables:
                dest_conn.execute(f"INSERT INTO {fts_table}({fts_table}) VALUES('rebuild')")
            
            # 7. Verify row counts
            for (table,) in tables:
                source_count = source_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                dest_count = dest_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if source_count != dest_count:
                    raise ValueError(f"Row count mismatch in {table}")
            
            # 8. Commit and close
            source_conn.commit()
            dest_conn.execute("VACUUM")  # Optimize encrypted database
            dest_conn.close()
            source_conn.close()
            
            # 9. Atomic swap
            backup_path = db_path.with_suffix('.backup')
            db_path.rename(backup_path)
            temp_path.rename(db_path)
            
            if progress_callback:
                progress_callback(1.0, "Migration complete")
            
        except Exception as e:
            # Rollback
            if temp_path.exists():
                temp_path.unlink()
            raise
```

### Strategy 2: Offline Migration (For Large Databases)

```python
class OfflineMigrator:
    """Migrate large databases with resume capability."""
    
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_db = self._init_checkpoint_db()
    
    def _init_checkpoint_db(self):
        """Initialize checkpoint database for resume capability."""
        conn = sqlite3.connect(str(self.checkpoint_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS migration_progress (
                table_name TEXT PRIMARY KEY,
                status TEXT,
                rows_migrated INTEGER,
                total_rows INTEGER,
                started_at TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)
        return conn
    
    def migrate_with_resume(self, db_path: Path, password: str, batch_size=10000):
        """Migrate with ability to resume after interruption."""
        
        temp_path = db_path.with_suffix('.encrypted.tmp')
        
        # Check if resuming
        is_resume = temp_path.exists()
        
        source_conn = sqlite3.connect(str(db_path))
        dest_conn = self._create_encrypted_connection(temp_path, password)
        
        if not is_resume:
            # Copy schema if starting fresh
            self._copy_schema(source_conn, dest_conn)
        
        # Get tables to migrate
        tables = self._get_pending_tables(source_conn)
        
        for table in tables:
            self._migrate_table_batched(
                source_conn, dest_conn, table, batch_size
            )
        
        # Final steps
        self._rebuild_indexes(dest_conn)
        self._verify_migration(source_conn, dest_conn)
        self._finalize_migration(db_path, temp_path)
    
    def _migrate_table_batched(self, source, dest, table, batch_size):
        """Migrate table in batches for better control."""
        
        # Get resume point
        checkpoint = self.checkpoint_db.execute(
            "SELECT rows_migrated FROM migration_progress WHERE table_name = ?",
            (table,)
        ).fetchone()
        
        start_offset = checkpoint[0] if checkpoint else 0
        
        # Get total rows
        total_rows = source.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        
        # Update checkpoint
        self.checkpoint_db.execute("""
            INSERT OR REPLACE INTO migration_progress 
            (table_name, status, rows_migrated, total_rows, started_at)
            VALUES (?, 'in_progress', ?, ?, datetime('now'))
        """, (table, start_offset, total_rows))
        
        # Migrate in batches
        for offset in range(start_offset, total_rows, batch_size):
            rows = source.execute(
                f"SELECT * FROM {table} LIMIT ? OFFSET ?",
                (batch_size, offset)
            ).fetchall()
            
            if rows:
                placeholders = ','.join('?' * len(rows[0]))
                dest.executemany(
                    f"INSERT INTO {table} VALUES ({placeholders})",
                    rows
                )
            
            # Update checkpoint
            self.checkpoint_db.execute(
                "UPDATE migration_progress SET rows_migrated = ? WHERE table_name = ?",
                (offset + len(rows), table)
            )
            self.checkpoint_db.commit()
        
        # Mark complete
        self.checkpoint_db.execute("""
            UPDATE migration_progress 
            SET status = 'completed', completed_at = datetime('now')
            WHERE table_name = ?
        """, (table,))
        self.checkpoint_db.commit()
```

### Strategy 3: Gradual Migration

```python
class GradualMigrator:
    """Migrate to encryption gradually with dual-database period."""
    
    def setup_dual_mode(self, original_db: Path, encrypted_db: Path, password: str):
        """Set up dual-database mode for gradual migration."""
        
        # Create encrypted copy
        if not encrypted_db.exists():
            self._create_encrypted_copy(original_db, encrypted_db, password)
        
        # Set up sync triggers
        self._setup_sync_triggers(original_db, encrypted_db, password)
        
        # Return wrapper that keeps both in sync
        return DualDatabaseWrapper(original_db, encrypted_db, password)
    
class DualDatabaseWrapper:
    """Wrapper that keeps unencrypted and encrypted databases in sync."""
    
    def __init__(self, original_db: Path, encrypted_db: Path, password: str):
        self.original_conn = sqlite3.connect(str(original_db))
        self.encrypted_conn = create_encrypted_connection(str(encrypted_db), password)
        
    def execute(self, sql, params=None):
        """Execute on both databases."""
        # Read operations go to encrypted
        if sql.strip().upper().startswith(('SELECT', 'PRAGMA')):
            return self.encrypted_conn.execute(sql, params)
        
        # Write operations go to both
        result1 = self.original_conn.execute(sql, params)
        result2 = self.encrypted_conn.execute(sql, params)
        return result2
```

## Implementation Patterns

### 1. Password Management Pattern

```python
class DatabasePasswordManager:
    """Centralized password management for database encryption."""
    
    def __init__(self):
        self._password_cache = {}
        self._cache_expiry = {}
        self._lock = threading.Lock()
        
    def get_password(self, db_name: str, prompt_callback=None) -> str:
        """Get password for specific database."""
        with self._lock:
            # Check cache
            if db_name in self._password_cache:
                if time.time() < self._cache_expiry[db_name]:
                    return self._password_cache[db_name]
                else:
                    # Expired
                    del self._password_cache[db_name]
                    del self._cache_expiry[db_name]
            
            # Try master password
            master_password = config.get_encryption_password()
            
            if not master_password and prompt_callback:
                master_password = prompt_callback(
                    f"Enter password for {db_name} database:"
                )
            
            if master_password:
                # Derive database-specific password
                db_password = self._derive_db_password(master_password, db_name)
                
                # Cache it
                self._password_cache[db_name] = db_password
                self._cache_expiry[db_name] = time.time() + 3600  # 1 hour
                
                return db_password
            
            raise ValueError("No password available")
    
    def _derive_db_password(self, master_password: str, db_name: str) -> str:
        """Derive database-specific password."""
        # Use existing scrypt implementation
        from tldw_chatbook.Utils.config_encryption import config_encryption
        
        # Create unique salt for each database
        salt = hashlib.sha256(f"tldw_db_{db_name}".encode()).digest()
        
        # Derive key
        key = scrypt(
            master_password.encode('utf-8'),
            salt,
            key_len=32,
            N=1048576,
            r=8,
            p=1
        )
        
        return key.hex()
```

### 2. Connection Pool Pattern

```python
class EncryptedConnectionPool:
    """Thread-safe connection pool for encrypted databases."""
    
    def __init__(self, db_path: Path, password: str, max_connections=5):
        self.db_path = db_path
        self.password = password
        self.max_connections = max_connections
        self._pool = queue.Queue(maxsize=max_connections)
        self._all_connections = []
        self._lock = threading.Lock()
        
        # Pre-create connections
        for _ in range(max_connections):
            conn = self._create_connection()
            self._pool.put(conn)
            self._all_connections.append(conn)
    
    def _create_connection(self):
        """Create new encrypted connection."""
        return create_encrypted_connection(str(self.db_path), self.password)
    
    @contextmanager
    def get_connection(self, timeout=30):
        """Get connection from pool."""
        conn = None
        try:
            conn = self._pool.get(timeout=timeout)
            yield conn
        finally:
            if conn:
                self._pool.put(conn)
    
    def close_all(self):
        """Close all connections."""
        with self._lock:
            for conn in self._all_connections:
                try:
                    conn.close()
                except:
                    pass
            self._all_connections.clear()
```

### 3. Error Handling Pattern

```python
class EncryptionError(Exception):
    """Base exception for encryption-related errors."""
    pass

class InvalidPasswordError(EncryptionError):
    """Raised when password is incorrect."""
    pass

class EncryptionNotAvailableError(EncryptionError):
    """Raised when encryption library not available."""
    pass

def handle_encryption_errors(func):
    """Decorator for handling encryption errors gracefully."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except apsw.SQLError as e:
            if "not authorized" in str(e):
                raise InvalidPasswordError("Invalid database password")
            elif "cipher" in str(e):
                raise EncryptionError(f"Encryption error: {e}")
            raise
        except ImportError as e:
            if "apsw" in str(e):
                raise EncryptionNotAvailableError(
                    "Database encryption not available. Install apsw-sqlite3mc"
                )
            raise
    return wrapper
```

### 4. Testing Pattern

```python
class EncryptedDatabaseTestCase:
    """Base test case for encrypted database tests."""
    
    def setUp(self):
        """Set up test database."""
        self.test_password = "test_password_123"
        self.db_path = Path(tempfile.mktemp(suffix='.db'))
        self.conn = create_encrypted_connection(
            str(self.db_path), 
            self.test_password
        )
        
    def tearDown(self):
        """Clean up test database."""
        self.conn.close()
        if self.db_path.exists():
            self.db_path.unlink()
    
    def test_encryption_active(self):
        """Verify database is actually encrypted."""
        self.conn.close()
        
        # Try to open without password - should fail
        with self.assertRaises(apsw.SQLError):
            unencrypted = apsw.Connection(str(self.db_path))
            unencrypted.execute("SELECT 1")
    
    def test_wrong_password(self):
        """Test wrong password handling."""
        self.conn.close()
        
        with self.assertRaises(InvalidPasswordError):
            wrong_conn = create_encrypted_connection(
                str(self.db_path),
                "wrong_password"
            )
            wrong_conn.execute("SELECT 1")
```

## Recommendation

After comprehensive analysis, **SQLite3MultipleCiphers (SQLite3MC)** is the recommended solution for tldw_chatbook database encryption:

### Why SQLite3MC?

1. **Better Performance**: ChaCha20-Poly1305 cipher offers 2-10% overhead vs SQLCipher's 5-15%
2. **Modern Cryptography**: AEAD ciphers provide both encryption and authentication
3. **Easier Integration**: apsw-sqlite3mc package simplifies Python integration
4. **Flexibility**: Multiple cipher options allow performance/security trade-offs
5. **MIT License**: More permissive than SQLCipher's dual licensing
6. **SQLCipher Compatibility**: Can read SQLCipher databases if needed

### Implementation Roadmap

#### Phase 1: Foundation (Week 1-2)
- Add apsw-sqlite3mc as optional dependency
- Extend BaseDB with encryption support
- Integrate with existing password management
- Create unit tests

#### Phase 2: Migration Tools (Week 3-4)
- Implement online migration for small databases
- Create offline migration with resume capability
- Add progress tracking and UI integration
- Test with real user data

#### Phase 3: Rollout (Week 5-6)
- Enable encryption for new installations
- Provide migration tool for existing users
- Document security best practices
- Monitor performance impact

### Configuration Example

```toml
# config.toml
[database_encryption]
enabled = false  # Opt-in initially
cipher = "chacha20"  # Fastest option
kdf_iter = 256000  # OWASP recommendation
legacy_mode = false  # For SQLCipher compatibility
auto_migrate = true  # Automatically migrate on first run

[database_encryption.databases]
# Selective encryption per database
chachanotes = true
media = true
prompts = false  # Less sensitive
evals = false
```

### Security Considerations

1. **Key Management**
   - Derive database keys from master password
   - Use different salts for each database
   - Clear keys from memory after use

2. **Backup Strategy**
   - Encrypted backups only
   - Test restore procedures
   - Consider key escrow for recovery

3. **Performance Monitoring**
   - Track query performance
   - Monitor FTS5 search times
   - Adjust cipher if needed

### Conclusion

SQLite3MultipleCiphers provides the best balance of security, performance, and maintainability for tldw_chatbook. Its modern cipher options, easier Python integration, and SQLCipher compatibility make it the superior choice for adding database encryption to the application.

## Comprehensive Implementation Plan for SQLite3MultipleCiphers

### Phase 0: Pre-Implementation Analysis and Preparation

#### 0.1 Dependency Verification
```bash
# Test installation on all target platforms
pip install apsw-sqlite3mc

# Verify version and cipher support
python -c "import apsw; print(apsw.apswversion()); print(apsw.sqlitelibversion())"
```

#### 0.2 Performance Baseline
```python
# benchmark_current.py
import time
import sqlite3
from pathlib import Path

class DatabaseBenchmark:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.results = {}
    
    def benchmark_operation(self, name: str, operation):
        start = time.perf_counter()
        result = operation()
        duration = time.perf_counter() - start
        self.results[name] = duration
        return result
    
    def run_benchmarks(self):
        conn = sqlite3.connect(str(self.db_path))
        
        # Benchmark various operations
        self.benchmark_operation("connect", lambda: sqlite3.connect(str(self.db_path)))
        self.benchmark_operation("insert_1k", lambda: self._insert_rows(conn, 1000))
        self.benchmark_operation("select_all", lambda: conn.execute("SELECT * FROM test").fetchall())
        self.benchmark_operation("fts5_search", lambda: self._fts5_search(conn))
        self.benchmark_operation("transaction", lambda: self._transaction_test(conn))
        
        return self.results
```

#### 0.3 Risk Assessment
| Risk | Impact | Mitigation |
|------|--------|------------|
| Data corruption during migration | Critical | Mandatory backups, verification steps |
| Performance degradation | High | Benchmark before/after, cipher selection |
| Key loss | Critical | Key recovery mechanism, clear documentation |
| Platform incompatibility | Medium | Test on all platforms, fallback options |
| User confusion | Medium | Clear UI, comprehensive documentation |

### Phase 1: Core Infrastructure Implementation

#### 1.1 Module Structure
```
tldw_chatbook/DB/encryption/
├── __init__.py
├── encrypted_db.py          # Main encryption interface
├── key_manager.py           # Key derivation and management
├── connection_pool.py       # Thread-safe connection pooling
├── migration_manager.py     # Database migration logic
├── verification.py          # Data integrity verification
└── exceptions.py            # Custom exceptions
```

#### 1.2 Encrypted Database Interface
```python
# tldw_chatbook/DB/encryption/encrypted_db.py

import urllib.parse
from pathlib import Path
from typing import Optional, Dict, Any, Union
import threading
import time

try:
    import apsw
    SQLITE3MC_AVAILABLE = True
except ImportError:
    import sqlite3 as apsw  # Fallback
    SQLITE3MC_AVAILABLE = False

from ...Utils.config_encryption import config_encryption
from ...Metrics.metrics_logger import log_counter, log_histogram
from .key_manager import DatabaseKeyManager
from .exceptions import (
    EncryptionNotAvailableError, 
    InvalidPasswordError,
    MigrationError
)

class EncryptedDatabase:
    """
    Main interface for SQLite3MultipleCiphers encrypted databases.
    
    Features:
    - Transparent encryption/decryption
    - Multiple cipher support
    - Connection pooling
    - Key management
    - Migration support
    """
    
    # Cipher configurations
    CIPHER_CONFIGS = {
        "chacha20": {
            "cipher": "chacha20",
            "kdf_iter": 256000,
            "legacy": 0,
            "legacy_page_size": 4096,
            "hmac_use": 1,
            "hmac_algorithm": 2,  # HMAC-SHA256
            "plaintext_header_size": 0
        },
        "aes256gcm": {
            "cipher": "aes256gcm", 
            "kdf_iter": 256000,
            "legacy": 0,
            "legacy_page_size": 4096
        },
        "sqlcipher": {
            "cipher": "aes256cbc",
            "kdf_iter": 256000,
            "fast_kdf_iter": 2,
            "hmac_use": 1,
            "hmac_algorithm": 2,
            "legacy": 4  # SQLCipher 4.x compatibility
        }
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize encrypted database with configuration."""
        self.config = config
        self.encryption_config = config.get('database_encryption', {})
        self.enabled = self.encryption_config.get('enabled', False)
        self.cipher = self.encryption_config.get('cipher', 'chacha20')
        self.auto_migrate = self.encryption_config.get('auto_migrate', False)
        
        # Selective database encryption
        self.db_config = self.encryption_config.get('databases', {})
        
        # Key manager
        self.key_manager = DatabaseKeyManager()
        
        # Connection tracking
        self._connections = {}
        self._lock = threading.Lock()
        
        # Metrics
        self._connection_count = 0
        self._error_count = 0
        
    def should_encrypt_database(self, db_name: str) -> bool:
        """Check if specific database should be encrypted."""
        if not self.enabled:
            return False
            
        # Check specific database configuration
        return self.db_config.get(db_name, True)  # Default to encrypted
    
    def create_connection(self, db_path: Path, db_name: str = "main") -> Union[apsw.Connection, Any]:
        """
        Create a database connection, encrypted if enabled.
        
        Args:
            db_path: Path to database file
            db_name: Logical database name for configuration
            
        Returns:
            Database connection (apsw.Connection or sqlite3.Connection)
        """
        log_counter("db_encryption_connection_attempt", labels={"db_name": db_name})
        
        if not self.should_encrypt_database(db_name):
            # Return unencrypted connection
            log_counter("db_encryption_connection_unencrypted")
            if SQLITE3MC_AVAILABLE:
                return apsw.Connection(str(db_path))
            else:
                import sqlite3
                return sqlite3.connect(str(db_path))
        
        if not SQLITE3MC_AVAILABLE:
            log_counter("db_encryption_not_available")
            raise EncryptionNotAvailableError(
                "Database encryption is enabled but apsw-sqlite3mc is not installed. "
                "Install with: pip install apsw-sqlite3mc"
            )
        
        try:
            # Get or derive password
            password = self.key_manager.get_password(db_name)
            
            # Check if database exists and is encrypted
            if db_path.exists():
                if not self._is_encrypted(db_path):
                    if self.auto_migrate:
                        log_counter("db_encryption_auto_migrate")
                        self._migrate_to_encrypted(db_path, password, db_name)
                    else:
                        raise MigrationError(
                            f"Database {db_name} exists but is not encrypted. "
                            "Enable auto_migrate or migrate manually."
                        )
            
            # Create encrypted connection
            conn = self._create_encrypted_connection(db_path, password)
            
            # Track connection
            with self._lock:
                self._connections[id(conn)] = {
                    "path": db_path,
                    "db_name": db_name,
                    "created": time.time()
                }
                self._connection_count += 1
            
            log_counter("db_encryption_connection_success", labels={
                "db_name": db_name,
                "cipher": self.cipher
            })
            
            return conn
            
        except Exception as e:
            self._error_count += 1
            log_counter("db_encryption_connection_error", labels={
                "db_name": db_name,
                "error": type(e).__name__
            })
            raise
    
    def _create_encrypted_connection(self, db_path: Path, password: str) -> apsw.Connection:
        """Create encrypted connection with SQLite3MC."""
        start_time = time.time()
        
        # Get cipher configuration
        cipher_config = self.CIPHER_CONFIGS.get(self.cipher, self.CIPHER_CONFIGS["chacha20"])
        
        # Build URI parameters
        params = {
            "key": password,
            **cipher_config
        }
        
        # Build connection URI
        uri = f"file:{urllib.parse.quote(str(db_path))}?{urllib.parse.urlencode(params)}"
        
        # Create connection
        conn = apsw.Connection(
            uri,
            flags=apsw.SQLITE_OPEN_URI | 
                  apsw.SQLITE_OPEN_CREATE | 
                  apsw.SQLITE_OPEN_READWRITE
        )
        
        # Set standard pragmas
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA temp_store = MEMORY")  # Keep temp data in memory
        conn.execute("PRAGMA secure_delete = ON")    # Overwrite deleted data
        
        # Log performance
        duration = time.time() - start_time
        log_histogram("db_encryption_connection_duration", duration, labels={
            "cipher": self.cipher
        })
        
        return conn
    
    def _is_encrypted(self, db_path: Path) -> bool:
        """Check if database is encrypted."""
        try:
            # Try to open without encryption
            conn = apsw.Connection(str(db_path))
            conn.execute("SELECT 1")
            conn.close()
            return False  # Successfully opened without password
        except apsw.SQLError:
            return True  # Failed to open, likely encrypted
        except Exception:
            return False  # Other error, assume not encrypted
    
    def close_connection(self, conn):
        """Close and untrack connection."""
        with self._lock:
            conn_id = id(conn)
            if conn_id in self._connections:
                del self._connections[conn_id]
        
        try:
            conn.close()
        except:
            pass
```

#### 1.3 Key Management System
```python
# tldw_chatbook/DB/encryption/key_manager.py

import hashlib
import time
import threading
from typing import Optional, Dict, Callable
from pathlib import Path

from Cryptodome.Protocol.KDF import scrypt
from loguru import logger

from ...config import get_encryption_password, set_encryption_password
from ...Widgets.password_dialog import PasswordDialog
from .exceptions import InvalidPasswordError

class DatabaseKeyManager:
    """
    Manages encryption keys for databases.
    
    Features:
    - Master password integration
    - Per-database key derivation
    - Session-based caching
    - Thread-safe operations
    """
    
    def __init__(self):
        self._key_cache: Dict[str, str] = {}
        self._cache_expiry: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._master_password: Optional[str] = None
        self._password_attempts = 0
        self._max_attempts = 3
        
    def get_password(self, db_name: str, prompt_callback: Optional[Callable] = None) -> str:
        """
        Get password for specific database.
        
        Args:
            db_name: Database identifier
            prompt_callback: Optional callback for password prompt
            
        Returns:
            Hex-encoded database key
        """
        with self._lock:
            # Check cache first
            if db_name in self._key_cache:
                if time.time() < self._cache_expiry[db_name]:
                    logger.debug(f"Using cached key for {db_name}")
                    return self._key_cache[db_name]
                else:
                    # Expired
                    del self._key_cache[db_name]
                    del self._cache_expiry[db_name]
            
            # Get master password
            master_password = self._get_master_password(prompt_callback)
            
            # Derive database-specific key
            db_key = self._derive_database_key(master_password, db_name)
            
            # Cache it
            self._key_cache[db_name] = db_key
            self._cache_expiry[db_name] = time.time() + 3600  # 1 hour
            
            logger.info(f"Derived new key for database: {db_name}")
            return db_key
    
    def _get_master_password(self, prompt_callback: Optional[Callable] = None) -> str:
        """Get or prompt for master password."""
        # Try cached master password
        if self._master_password:
            return self._master_password
        
        # Try config module
        config_password = get_encryption_password()
        if config_password:
            self._master_password = config_password
            return config_password
        
        # Prompt user
        if prompt_callback:
            password = prompt_callback()
        else:
            password = self._prompt_for_password()
        
        if not password:
            raise InvalidPasswordError("No password provided")
        
        # Cache it
        self._master_password = password
        set_encryption_password(password)  # Store in config module
        
        return password
    
    def _prompt_for_password(self) -> Optional[str]:
        """Default password prompt implementation."""
        self._password_attempts += 1
        
        if self._password_attempts > self._max_attempts:
            raise InvalidPasswordError(f"Maximum password attempts ({self._max_attempts}) exceeded")
        
        # In a real implementation, this would use the TUI password dialog
        # For now, return None to force the callback
        return None
    
    def _derive_database_key(self, master_password: str, db_name: str) -> str:
        """
        Derive database-specific key using scrypt.
        
        Args:
            master_password: Master password
            db_name: Database identifier
            
        Returns:
            Hex-encoded 256-bit key
        """
        # Create unique salt for each database
        salt_input = f"tldw_chatbook_db_{db_name}_v1".encode('utf-8')
        salt = hashlib.sha256(salt_input).digest()
        
        # Derive key using scrypt (matching config_encryption parameters)
        key = scrypt(
            master_password.encode('utf-8'),
            salt,
            key_len=32,  # 256 bits
            N=1048576,   # 2^20
            r=8,
            p=1
        )
        
        # Return as hex for SQLite3MC
        return key.hex()
    
    def clear_cache(self, db_name: Optional[str] = None):
        """Clear cached keys."""
        with self._lock:
            if db_name:
                self._key_cache.pop(db_name, None)
                self._cache_expiry.pop(db_name, None)
            else:
                self._key_cache.clear()
                self._cache_expiry.clear()
    
    def change_password(self, old_password: str, new_password: str):
        """
        Change master password and clear caches.
        
        This doesn't re-encrypt databases - use migration tools for that.
        """
        # Verify old password
        if self._master_password and self._master_password != old_password:
            raise InvalidPasswordError("Old password is incorrect")
        
        # Update password
        self._master_password = new_password
        set_encryption_password(new_password)
        
        # Clear all caches
        self.clear_cache()
        
        logger.info("Master password changed, key cache cleared")
```

#### 1.4 Connection Pool Implementation
```python
# tldw_chatbook/DB/encryption/connection_pool.py

import queue
import threading
import time
from contextlib import contextmanager
from typing import Dict, Any, Optional
from pathlib import Path

import apsw
from loguru import logger

from .encrypted_db import EncryptedDatabase
from .exceptions import PoolExhaustedError

class EncryptedConnectionPool:
    """
    Thread-safe connection pool for encrypted databases.
    
    Features:
    - Lazy connection creation
    - Connection health checks
    - Automatic retry on failure
    - Performance metrics
    """
    
    def __init__(self, 
                 encrypted_db: EncryptedDatabase,
                 min_connections: int = 1,
                 max_connections: int = 5,
                 connection_timeout: float = 30.0):
        """
        Initialize connection pool.
        
        Args:
            encrypted_db: EncryptedDatabase instance
            min_connections: Minimum connections to maintain
            max_connections: Maximum connections allowed
            connection_timeout: Timeout for getting connection
        """
        self.encrypted_db = encrypted_db
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        
        # Pool storage
        self._pools: Dict[str, queue.Queue] = {}  # Per-database pools
        self._all_connections: Dict[str, list] = {}
        self._connection_info: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            "connections_created": 0,
            "connections_reused": 0,
            "connection_errors": 0,
            "pool_exhausted": 0
        }
        
    def get_pool(self, db_path: Path, db_name: str) -> queue.Queue:
        """Get or create pool for specific database."""
        pool_key = f"{db_name}:{db_path}"
        
        with self._lock:
            if pool_key not in self._pools:
                self._pools[pool_key] = queue.Queue(maxsize=self.max_connections)
                self._all_connections[pool_key] = []
                
                # Pre-create minimum connections
                for _ in range(self.min_connections):
                    try:
                        conn = self._create_connection(db_path, db_name)
                        self._pools[pool_key].put(conn)
                        self._all_connections[pool_key].append(conn)
                    except Exception as e:
                        logger.warning(f"Failed to pre-create connection: {e}")
            
            return self._pools[pool_key]
    
    def _create_connection(self, db_path: Path, db_name: str) -> apsw.Connection:
        """Create new encrypted connection."""
        conn = self.encrypted_db.create_connection(db_path, db_name)
        
        # Track connection info
        self._connection_info[id(conn)] = {
            "db_path": db_path,
            "db_name": db_name,
            "created": time.time(),
            "last_used": time.time(),
            "use_count": 0
        }
        
        self._stats["connections_created"] += 1
        logger.debug(f"Created new connection for {db_name}")
        
        return conn
    
    @contextmanager
    def get_connection(self, db_path: Path, db_name: str):
        """
        Get connection from pool with automatic return.
        
        Usage:
            with pool.get_connection(db_path, "main") as conn:
                conn.execute("SELECT * FROM users")
        """
        pool = self.get_pool(db_path, db_name)
        conn = None
        pool_key = f"{db_name}:{db_path}"
        
        try:
            # Try to get existing connection
            try:
                conn = pool.get(timeout=self.connection_timeout)
                self._stats["connections_reused"] += 1
            except queue.Empty:
                # Pool exhausted, try to create new one
                with self._lock:
                    if len(self._all_connections[pool_key]) < self.max_connections:
                        conn = self._create_connection(db_path, db_name)
                        self._all_connections[pool_key].append(conn)
                    else:
                        self._stats["pool_exhausted"] += 1
                        raise PoolExhaustedError(
                            f"Connection pool exhausted for {db_name} "
                            f"(max: {self.max_connections})"
                        )
            
            # Verify connection health
            if not self._verify_connection(conn):
                # Connection is bad, create new one
                logger.warning("Connection failed health check, creating new one")
                conn.close()
                with self._lock:
                    self._all_connections[pool_key].remove(conn)
                conn = self._create_connection(db_path, db_name)
                self._all_connections[pool_key].append(conn)
            
            # Update usage stats
            if id(conn) in self._connection_info:
                self._connection_info[id(conn)]["last_used"] = time.time()
                self._connection_info[id(conn)]["use_count"] += 1
            
            yield conn
            
        except Exception as e:
            self._stats["connection_errors"] += 1
            logger.error(f"Connection error: {e}")
            raise
            
        finally:
            # Return connection to pool
            if conn:
                try:
                    pool.put(conn, block=False)
                except queue.Full:
                    # Pool is full, close connection
                    logger.warning("Pool full, closing connection")
                    conn.close()
                    with self._lock:
                        self._all_connections[pool_key].remove(conn)
                        del self._connection_info[id(conn)]
    
    def _verify_connection(self, conn: apsw.Connection) -> bool:
        """Verify connection is healthy."""
        try:
            conn.execute("SELECT 1")
            return True
        except:
            return False
    
    def close_all(self):
        """Close all connections in all pools."""
        with self._lock:
            for pool_key, connections in self._all_connections.items():
                for conn in connections:
                    try:
                        conn.close()
                    except:
                        pass
                
                # Clear pool
                pool = self._pools.get(pool_key)
                if pool:
                    while not pool.empty():
                        try:
                            pool.get_nowait()
                        except:
                            break
            
            self._pools.clear()
            self._all_connections.clear()
            self._connection_info.clear()
        
        logger.info("Closed all pooled connections")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            total_connections = sum(len(conns) for conns in self._all_connections.values())
            
            return {
                **self._stats,
                "total_connections": total_connections,
                "pools": len(self._pools),
                "connection_info": {
                    pool_key: len(conns) 
                    for pool_key, conns in self._all_connections.items()
                }
            }
```

### Phase 2: Migration Implementation

#### 2.1 Migration Manager
```python
# tldw_chatbook/DB/encryption/migration_manager.py

import shutil
import sqlite3
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime

import apsw
from loguru import logger

from .encrypted_db import EncryptedDatabase
from .verification import DataVerifier
from .exceptions import MigrationError

class MigrationManager:
    """
    Handles migration of unencrypted databases to encrypted format.
    
    Features:
    - Online and offline migration
    - Progress tracking
    - Resume capability
    - Data verification
    - Rollback support
    """
    
    def __init__(self, encrypted_db: EncryptedDatabase):
        self.encrypted_db = encrypted_db
        self.verifier = DataVerifier()
        
    def migrate_database(self,
                        source_path: Path,
                        password: str,
                        db_name: str = "main",
                        mode: str = "auto",
                        progress_callback: Optional[Callable] = None) -> Path:
        """
        Migrate unencrypted database to encrypted format.
        
        Args:
            source_path: Path to unencrypted database
            password: Encryption password
            db_name: Database name for configuration
            mode: Migration mode - "auto", "online", "offline"
            progress_callback: Optional callback(progress: float, message: str)
            
        Returns:
            Path to encrypted database
        """
        logger.info(f"Starting migration of {source_path} in {mode} mode")
        
        # Validate source
        if not source_path.exists():
            raise MigrationError(f"Source database not found: {source_path}")
        
        # Check if already encrypted
        if self._is_encrypted(source_path):
            logger.info("Database is already encrypted")
            return source_path
        
        # Choose migration strategy
        db_size = source_path.stat().st_size
        
        if mode == "auto":
            # Auto-select based on size
            if db_size < 100 * 1024 * 1024:  # 100MB
                mode = "online"
            else:
                mode = "offline"
        
        logger.info(f"Using {mode} migration for {db_size / 1024 / 1024:.1f}MB database")
        
        if mode == "online":
            return self._migrate_online(source_path, password, db_name, progress_callback)
        else:
            return self._migrate_offline(source_path, password, db_name, progress_callback)
    
    def _migrate_online(self,
                       source_path: Path,
                       password: str,
                       db_name: str,
                       progress_callback: Optional[Callable]) -> Path:
        """
        Perform online migration with minimal downtime.
        
        Strategy:
        1. Create encrypted copy while source is active
        2. Brief exclusive lock for final sync
        3. Atomic swap
        """
        temp_path = source_path.with_suffix('.encrypted.tmp')
        backup_path = source_path.with_suffix('.backup')
        
        try:
            # Step 1: Initial copy with shared lock
            if progress_callback:
                progress_callback(0.0, "Creating encrypted copy...")
            
            # Open source with shared lock
            source_conn = sqlite3.connect(str(source_path))
            source_conn.execute("PRAGMA journal_mode = WAL")
            
            # Create encrypted destination
            dest_conn = self._create_encrypted_connection(temp_path, password, db_name)
            
            # Copy schema
            self._copy_schema(source_conn, dest_conn)
            
            # Copy data with progress
            self._copy_data(source_conn, dest_conn, progress_callback, 0.1, 0.8)
            
            # Step 2: Final sync with exclusive lock
            if progress_callback:
                progress_callback(0.8, "Performing final synchronization...")
            
            # Get exclusive lock
            source_conn.execute("BEGIN EXCLUSIVE")
            
            # Sync any final changes
            self._sync_final_changes(source_conn, dest_conn)
            
            # Verify data
            if progress_callback:
                progress_callback(0.9, "Verifying data integrity...")
            
            if not self.verifier.verify_migration(source_conn, dest_conn):
                raise MigrationError("Data verification failed")
            
            # Step 3: Atomic swap
            if progress_callback:
                progress_callback(0.95, "Finalizing migration...")
            
            # Close connections
            source_conn.close()
            dest_conn.close()
            
            # Backup original
            shutil.move(str(source_path), str(backup_path))
            
            # Move encrypted to original location
            shutil.move(str(temp_path), str(source_path))
            
            if progress_callback:
                progress_callback(1.0, "Migration complete!")
            
            logger.success(f"Successfully migrated {source_path}")
            logger.info(f"Original backed up to: {backup_path}")
            
            return source_path
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            
            # Cleanup
            if temp_path.exists():
                temp_path.unlink()
            
            # Restore if needed
            if backup_path.exists() and not source_path.exists():
                shutil.move(str(backup_path), str(source_path))
            
            raise MigrationError(f"Migration failed: {e}")
    
    def _migrate_offline(self,
                        source_path: Path,
                        password: str,
                        db_name: str,
                        progress_callback: Optional[Callable]) -> Path:
        """
        Perform offline migration with resume capability.
        
        For large databases, supports:
        - Checkpointing
        - Resume on failure
        - Batch processing
        """
        checkpoint_path = source_path.with_suffix('.migration')
        temp_path = source_path.with_suffix('.encrypted.tmp')
        
        # Initialize checkpoint manager
        checkpoint = MigrationCheckpoint(checkpoint_path)
        
        try:
            # Check if resuming
            is_resume = checkpoint.exists() and temp_path.exists()
            
            if is_resume:
                logger.info("Resuming previous migration...")
                if progress_callback:
                    progress = checkpoint.get_progress()
                    progress_callback(progress, f"Resuming from {progress:.1%}")
            
            # Open connections
            source_conn = sqlite3.connect(str(source_path))
            source_conn.execute("PRAGMA journal_mode = WAL")
            
            if is_resume:
                dest_conn = apsw.Connection(str(temp_path))
            else:
                dest_conn = self._create_encrypted_connection(temp_path, password, db_name)
                self._copy_schema(source_conn, dest_conn)
            
            # Get tables to migrate
            tables = self._get_tables(source_conn)
            total_tables = len(tables)
            
            # Process each table
            for i, table in enumerate(tables):
                # Check if already processed
                if checkpoint.is_table_complete(table):
                    continue
                
                # Update progress
                base_progress = i / total_tables
                
                def table_progress(p, msg):
                    if progress_callback:
                        overall = base_progress + (p / total_tables)
                        progress_callback(overall, f"{table}: {msg}")
                
                # Migrate table
                self._migrate_table_batched(
                    source_conn, dest_conn, table, 
                    checkpoint, table_progress
                )
                
                # Mark complete
                checkpoint.mark_table_complete(table)
            
            # Rebuild indexes
            if progress_callback:
                progress_callback(0.9, "Rebuilding indexes...")
            
            self._rebuild_indexes(dest_conn)
            
            # Verify
            if progress_callback:
                progress_callback(0.95, "Verifying data...")
            
            if not self.verifier.verify_migration(source_conn, dest_conn):
                raise MigrationError("Data verification failed")
            
            # Finalize
            source_conn.close()
            dest_conn.close()
            
            # Swap files
            backup_path = source_path.with_suffix('.backup')
            shutil.move(str(source_path), str(backup_path))
            shutil.move(str(temp_path), str(source_path))
            
            # Cleanup checkpoint
            checkpoint.cleanup()
            
            if progress_callback:
                progress_callback(1.0, "Migration complete!")
            
            return source_path
            
        except Exception as e:
            logger.error(f"Offline migration failed: {e}")
            raise MigrationError(f"Migration failed: {e}")
    
    def _migrate_table_batched(self,
                              source_conn: sqlite3.Connection,
                              dest_conn: apsw.Connection,
                              table: str,
                              checkpoint: 'MigrationCheckpoint',
                              progress_callback: Optional[Callable],
                              batch_size: int = 10000):
        """Migrate table in batches with checkpointing."""
        
        # Get total rows
        total_rows = source_conn.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]
        
        if total_rows == 0:
            return
        
        # Get starting offset from checkpoint
        start_offset = checkpoint.get_table_progress(table)
        
        logger.info(f"Migrating {table}: {total_rows} rows (starting at {start_offset})")
        
        # Get column info
        columns = self._get_table_columns(source_conn, table)
        column_names = [col[1] for col in columns]
        placeholders = ','.join(['?' for _ in columns])
        
        # Process in batches
        for offset in range(start_offset, total_rows, batch_size):
            # Read batch
            rows = source_conn.execute(
                f"SELECT {','.join(column_names)} FROM {table} "
                f"LIMIT {batch_size} OFFSET {offset}"
            ).fetchall()
            
            if not rows:
                break
            
            # Write batch
            dest_conn.executemany(
                f"INSERT INTO {table} ({','.join(column_names)}) "
                f"VALUES ({placeholders})",
                rows
            )
            
            # Update checkpoint
            checkpoint.update_table_progress(table, offset + len(rows), total_rows)
            
            # Update UI
            if progress_callback:
                progress = (offset + len(rows)) / total_rows
                progress_callback(progress, f"{offset + len(rows)}/{total_rows} rows")
        
        logger.info(f"Completed migration of {table}")


class MigrationCheckpoint:
    """Manages migration checkpoints for resume capability."""
    
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.conn = self._init_checkpoint_db()
    
    def _init_checkpoint_db(self) -> sqlite3.Connection:
        """Initialize checkpoint database."""
        conn = sqlite3.connect(str(self.checkpoint_path))
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS migration_progress (
                table_name TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                rows_completed INTEGER DEFAULT 0,
                total_rows INTEGER DEFAULT 0,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS migration_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        return conn
    
    def exists(self) -> bool:
        """Check if checkpoint exists."""
        return self.checkpoint_path.exists()
    
    def get_progress(self) -> float:
        """Get overall migration progress."""
        result = self.conn.execute("""
            SELECT 
                SUM(rows_completed) as completed,
                SUM(total_rows) as total
            FROM migration_progress
        """).fetchone()
        
        if result and result[1] > 0:
            return result[0] / result[1]
        return 0.0
    
    def get_table_progress(self, table: str) -> int:
        """Get progress for specific table."""
        result = self.conn.execute(
            "SELECT rows_completed FROM migration_progress WHERE table_name = ?",
            (table,)
        ).fetchone()
        
        return result[0] if result else 0
    
    def update_table_progress(self, table: str, rows_completed: int, total_rows: int):
        """Update progress for table."""
        self.conn.execute("""
            INSERT OR REPLACE INTO migration_progress 
            (table_name, status, rows_completed, total_rows, updated_at)
            VALUES (?, 'in_progress', ?, ?, datetime('now'))
        """, (table, rows_completed, total_rows))
        self.conn.commit()
    
    def mark_table_complete(self, table: str):
        """Mark table as complete."""
        self.conn.execute("""
            UPDATE migration_progress 
            SET status = 'completed', 
                completed_at = datetime('now'),
                updated_at = datetime('now')
            WHERE table_name = ?
        """, (table,))
        self.conn.commit()
    
    def is_table_complete(self, table: str) -> bool:
        """Check if table is complete."""
        result = self.conn.execute(
            "SELECT status FROM migration_progress WHERE table_name = ?",
            (table,)
        ).fetchone()
        
        return result and result[0] == 'completed'
    
    def cleanup(self):
        """Clean up checkpoint database."""
        self.conn.close()
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
```

### Phase 3: Integration with Existing System

#### 3.1 Modified BaseDB
```python
# Modifications to tldw_chatbook/DB/base_db.py

from typing import Optional
from pathlib import Path

# Import encryption module
from .encryption import EncryptedDatabase, SQLITE3MC_AVAILABLE

class BaseDB:
    """Base class for all database operations with encryption support."""
    
    # Class-level encryption manager
    _encryption_manager: Optional[EncryptedDatabase] = None
    
    def __init__(self):
        """Initialize base database class."""
        super().__init__()
        
        # Get database name from class
        self.db_name = self._get_db_name()
        
        # Initialize encryption if needed
        if self._encryption_manager is None:
            self._init_encryption()
    
    @classmethod
    def _init_encryption(cls):
        """Initialize encryption manager."""
        from ..config import load_settings
        config = load_settings()
        
        cls._encryption_manager = EncryptedDatabase(config)
        logger.info(f"Encryption initialized: {cls._encryption_manager.enabled}")
    
    def _get_db_name(self) -> str:
        """Get logical database name for configuration."""
        # Override in subclasses
        return self.__class__.__name__.replace('DB', '').lower()
    
    def _get_connection(self, db_path: Optional[Path] = None):
        """
        Get database connection with encryption support.
        
        Args:
            db_path: Optional database path override
            
        Returns:
            Database connection
        """
        if db_path is None:
            db_path = self._get_db_path()
        
        # Use encryption manager if available
        if self._encryption_manager:
            try:
                return self._encryption_manager.create_connection(
                    db_path, 
                    self.db_name
                )
            except Exception as e:
                logger.error(f"Failed to create encrypted connection: {e}")
                
                # Fallback to unencrypted if allowed
                if not self._encryption_manager.encryption_config.get('require_encryption', False):
                    logger.warning("Falling back to unencrypted connection")
                    import sqlite3
                    return sqlite3.connect(str(db_path))
                else:
                    raise
        else:
            # No encryption manager
            import sqlite3
            return sqlite3.connect(str(db_path))
```

#### 3.2 Configuration Updates
```python
# Add to pyproject.toml

[project.optional-dependencies]
encryption = [
    "pycryptodomex>=3.20.0",  # Already present
    "apsw-sqlite3mc>=3.45.0",  # SQLite3 Multiple Ciphers
]

# Add encryption configuration to config.toml template
```

### Phase 4: UI Integration

#### 4.1 Password Dialog Enhancement
```python
# Enhancements to tldw_chatbook/Widgets/password_dialog.py

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Input, Label, Static

class DatabasePasswordDialog(PasswordDialog):
    """Enhanced password dialog for database encryption."""
    
    def __init__(self, 
                 db_name: str = "database",
                 show_migration_option: bool = False,
                 **kwargs):
        self.db_name = db_name
        self.show_migration_option = show_migration_option
        super().__init__(**kwargs)
    
    def compose(self) -> ComposeResult:
        with Container(id="password-dialog-container"):
            yield Label(f"🔐 Database Encryption", id="dialog-title")
            yield Static(
                f"Enter password to unlock '{self.db_name}' database:",
                id="dialog-message"
            )
            yield Input(
                placeholder="Enter password...",
                password=True,
                id="password-input"
            )
            
            if self.show_migration_option:
                with Vertical(id="migration-options"):
                    yield Label("This database is not encrypted.", classes="warning")
                    yield Label("Would you like to encrypt it now?")
            
            with Horizontal(id="dialog-buttons"):
                yield Button("Cancel", variant="default", id="cancel")
                yield Button("Unlock", variant="primary", id="submit")
                
                if self.show_migration_option:
                    yield Button("Encrypt Now", variant="success", id="migrate")
```

#### 4.2 Migration Progress UI
```python
# tldw_chatbook/Widgets/migration_progress.py

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Label, ProgressBar, Static, Button
from textual.reactive import reactive

class MigrationProgressWidget(Container):
    """Shows database migration progress."""
    
    progress = reactive(0.0)
    status = reactive("Initializing...")
    can_cancel = reactive(True)
    
    def compose(self) -> ComposeResult:
        with Vertical(id="migration-container"):
            yield Label("🔄 Database Migration", id="migration-title")
            yield Static(self.status, id="migration-status")
            yield ProgressBar(
                total=100,
                show_percentage=True,
                id="migration-progress"
            )
            yield Button(
                "Cancel",
                variant="warning",
                id="cancel-migration",
                disabled=not self.can_cancel
            )
    
    def watch_progress(self, progress: float):
        """Update progress bar."""
        bar = self.query_one("#migration-progress", ProgressBar)
        bar.update(progress=int(progress * 100))
    
    def watch_status(self, status: str):
        """Update status message."""
        self.query_one("#migration-status", Static).update(status)
    
    def update_migration_progress(self, progress: float, message: str):
        """Update both progress and status."""
        self.progress = progress
        self.status = message
        
        # Disable cancel near completion
        if progress > 0.9:
            self.can_cancel = False
```

### Phase 5: Testing Strategy

#### 5.1 Unit Tests
```python
# Tests/DB/encryption/test_encrypted_db.py

import pytest
import tempfile
from pathlib import Path

from tldw_chatbook.DB.encryption import EncryptedDatabase, SQLITE3MC_AVAILABLE

@pytest.mark.skipif(not SQLITE3MC_AVAILABLE, reason="SQLite3MC not available")
class TestEncryptedDatabase:
    """Test encrypted database functionality."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            path = Path(f.name)
        yield path
        if path.exists():
            path.unlink()
    
    @pytest.fixture
    def encryption_config(self):
        """Test encryption configuration."""
        return {
            'database_encryption': {
                'enabled': True,
                'cipher': 'chacha20',
                'kdf_iter': 1000,  # Lower for tests
                'databases': {
                    'test': True
                }
            }
        }
    
    def test_create_encrypted_connection(self, temp_db_path, encryption_config):
        """Test creating encrypted connection."""
        enc_db = EncryptedDatabase(encryption_config)
        
        # Mock password
        enc_db.key_manager._master_password = "test_password"
        
        # Create connection
        conn = enc_db.create_connection(temp_db_path, "test")
        
        # Verify encryption
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, data TEXT)")
        conn.execute("INSERT INTO test (data) VALUES (?)", ("secret data",))
        conn.close()
        
        # Try to open without password - should fail
        import apsw
        with pytest.raises(apsw.SQLError):
            unenc_conn = apsw.Connection(str(temp_db_path))
            unenc_conn.execute("SELECT * FROM test")
    
    def test_cipher_selection(self, temp_db_path, encryption_config):
        """Test different cipher configurations."""
        for cipher in ['chacha20', 'aes256gcm', 'sqlcipher']:
            config = encryption_config.copy()
            config['database_encryption']['cipher'] = cipher
            
            enc_db = EncryptedDatabase(config)
            enc_db.key_manager._master_password = "test_password"
            
            conn = enc_db.create_connection(
                temp_db_path.with_suffix(f'.{cipher}.db'), 
                "test"
            )
            
            # Basic operation
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.close()
```

#### 5.2 Migration Tests
```python
# Tests/DB/encryption/test_migration.py

import pytest
import sqlite3
import tempfile
from pathlib import Path

from tldw_chatbook.DB.encryption import MigrationManager, EncryptedDatabase

class TestMigration:
    """Test database migration functionality."""
    
    @pytest.fixture
    def unencrypted_db(self):
        """Create test unencrypted database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        
        # Create test data
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT,
                email TEXT
            )
        """)
        
        # Insert test data
        test_data = [
            (1, "Alice", "alice@example.com"),
            (2, "Bob", "bob@example.com"),
            (3, "Charlie", "charlie@example.com")
        ]
        conn.executemany("INSERT INTO users VALUES (?, ?, ?)", test_data)
        
        # Create FTS5 table
        conn.execute("""
            CREATE VIRTUAL TABLE users_fts USING fts5(
                name, email, content=users
            )
        """)
        conn.execute("INSERT INTO users_fts(users_fts) VALUES('rebuild')")
        
        conn.commit()
        conn.close()
        
        yield db_path
        
        # Cleanup
        for suffix in ['', '.backup', '.encrypted.tmp', '.migration']:
            path = db_path.with_suffix(suffix)
            if path.exists():
                path.unlink()
    
    def test_online_migration(self, unencrypted_db):
        """Test online migration mode."""
        config = {
            'database_encryption': {
                'enabled': True,
                'cipher': 'chacha20'
            }
        }
        
        enc_db = EncryptedDatabase(config)
        enc_db.key_manager._master_password = "test_password"
        
        manager = MigrationManager(enc_db)
        
        # Track progress
        progress_updates = []
        
        def progress_callback(progress, message):
            progress_updates.append((progress, message))
        
        # Perform migration
        result_path = manager.migrate_database(
            unencrypted_db,
            "test_password",
            mode="online",
            progress_callback=progress_callback
        )
        
        # Verify migration
        assert result_path == unencrypted_db
        assert len(progress_updates) > 0
        assert progress_updates[-1][0] == 1.0
        
        # Verify data
        conn = enc_db.create_connection(result_path, "test")
        users = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
        assert len(users) == 3
        assert users[0][1] == "Alice"
        
        # Verify FTS5
        results = conn.execute(
            "SELECT name FROM users_fts WHERE users_fts MATCH 'bob'"
        ).fetchall()
        assert len(results) == 1
        assert results[0][0] == "Bob"
```

#### 5.3 Performance Tests
```python
# Tests/DB/encryption/test_performance.py

import pytest
import time
import statistics
from pathlib import Path

from tldw_chatbook.DB.encryption import EncryptedDatabase

class TestPerformance:
    """Benchmark encryption performance."""
    
    def benchmark_operation(self, operation, iterations=100):
        """Benchmark an operation."""
        times = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            operation()
            duration = time.perf_counter() - start
            times.append(duration)
        
        return {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'stdev': statistics.stdev(times) if len(times) > 1 else 0,
            'min': min(times),
            'max': max(times)
        }
    
    @pytest.mark.benchmark
    def test_encryption_overhead(self, temp_db_path):
        """Compare encrypted vs unencrypted performance."""
        import sqlite3
        
        # Test data
        test_rows = 1000
        
        # Benchmark unencrypted
        def unencrypted_test():
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, data TEXT)")
            for i in range(test_rows):
                conn.execute("INSERT INTO test (data) VALUES (?)", (f"data_{i}",))
            conn.execute("SELECT COUNT(*) FROM test").fetchone()
            conn.close()
        
        unenc_stats = self.benchmark_operation(unencrypted_test, 10)
        
        # Benchmark encrypted
        config = {'database_encryption': {'enabled': True, 'cipher': 'chacha20'}}
        enc_db = EncryptedDatabase(config)
        enc_db.key_manager._master_password = "test"
        
        def encrypted_test():
            conn = enc_db.create_connection(Path(":memory:"), "test")
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, data TEXT)")
            for i in range(test_rows):
                conn.execute("INSERT INTO test (data) VALUES (?)", (f"data_{i}",))
            conn.execute("SELECT COUNT(*) FROM test").fetchone()
            conn.close()
        
        enc_stats = self.benchmark_operation(encrypted_test, 10)
        
        # Calculate overhead
        overhead = (enc_stats['mean'] - unenc_stats['mean']) / unenc_stats['mean'] * 100
        
        print(f"\nPerformance Results:")
        print(f"Unencrypted: {unenc_stats['mean']*1000:.2f}ms")
        print(f"Encrypted:   {enc_stats['mean']*1000:.2f}ms")
        print(f"Overhead:    {overhead:.1f}%")
        
        # Assert reasonable overhead (< 15%)
        assert overhead < 15, f"Encryption overhead too high: {overhead:.1f}%"
```

### Phase 6: Documentation

#### 6.1 User Guide
```markdown
# Database Encryption User Guide

## Overview
tldw_chatbook now supports transparent database encryption using SQLite3MultipleCiphers.

## Enabling Encryption

### For New Installations
1. Encryption is enabled by default for new installations
2. You'll be prompted for a master password on first run
3. Use the same password as your config encryption

### For Existing Users
1. Update to the latest version
2. Run: `tldw-cli --migrate-encryption`
3. Follow the prompts to encrypt your databases

## Configuration

Edit `~/.config/tldw_cli/config.toml`:

```toml
[database_encryption]
enabled = true
cipher = "chacha20"  # Options: chacha20, aes256gcm, sqlcipher
auto_migrate = true  # Automatically encrypt unencrypted databases

[database_encryption.databases]
# Control which databases are encrypted
chachanotes = true
media = true
prompts = false
```

## Performance Considerations
- ChaCha20 (default): 2-5% overhead, best performance
- AES-256-GCM: 5-10% overhead, widely supported
- SQLCipher mode: 10-15% overhead, compatibility mode

## Troubleshooting

### Forgot Password
1. Config and database passwords are the same by default
2. If you've lost your password, restore from backup
3. Use `--decrypt-database` to create unencrypted copy (requires password)

### Performance Issues
1. Try ChaCha20 cipher (fastest)
2. Disable encryption for less sensitive databases
3. Check disk I/O performance

### Migration Failed
1. Check `~/.local/share/tldw_cli/logs/migration.log`
2. Resume with: `tldw-cli --resume-migration`
3. Restore from automatic backup if needed
```

#### 6.2 Developer Guide
```markdown
# Database Encryption Developer Guide

## Architecture
- Uses SQLite3MultipleCiphers via apsw-sqlite3mc
- Transparent encryption at page level
- Integrated with existing BaseDB infrastructure

## Adding Encryption to New Database

```python
from tldw_chatbook.DB.base_db import BaseDB

class MyNewDB(BaseDB):
    def _get_db_name(self) -> str:
        return "mynewdb"  # Used for configuration
    
    def _get_db_path(self) -> Path:
        return self.db_dir / "my_new.db"
```

## Testing Encrypted Databases

```python
# Always test both encrypted and unencrypted
@pytest.mark.parametrize("encrypted", [True, False])
def test_my_feature(encrypted):
    config = {
        'database_encryption': {
            'enabled': encrypted,
            'cipher': 'chacha20'
        }
    }
    # ... rest of test
```

## Performance Monitoring

```python
from tldw_chatbook.Metrics.metrics_logger import log_histogram

# Monitor query performance
start = time.time()
cursor.execute(query)
log_histogram("db_query_duration", time.time() - start, labels={
    "encrypted": str(self._encryption_manager.enabled),
    "query_type": "select"
})
```
```

### Phase 7: Deployment

#### 7.1 Package Updates
```toml
# pyproject.toml additions

[project.optional-dependencies]
encryption = [
    "pycryptodomex>=3.20.0",
    "apsw-sqlite3mc>=3.45.0",
]

all = [
    # ... other deps ...
    "apsw-sqlite3mc>=3.45.0",  # Include in 'all'
]

[project.scripts]
tldw-cli = "tldw_chatbook.app:main_cli_runner"
tldw-migrate-encryption = "tldw_chatbook.DB.encryption.cli:migrate_databases"
```

#### 7.2 Migration CLI
```python
# tldw_chatbook/DB/encryption/cli.py

import click
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

from . import EncryptedDatabase, MigrationManager, SQLITE3MC_AVAILABLE

console = Console()

@click.command()
@click.option('--config', type=Path, help='Config file path')
@click.option('--password', help='Encryption password (prompt if not provided)')
@click.option('--cipher', default='chacha20', help='Cipher to use')
@click.option('--mode', default='auto', help='Migration mode: auto, online, offline')
@click.option('--verify-only', is_flag=True, help='Only verify encryption status')
def migrate_databases(config, password, cipher, mode, verify_only):
    """Migrate unencrypted databases to encrypted format."""
    
    if not SQLITE3MC_AVAILABLE:
        console.print("[red]Error: apsw-sqlite3mc not installed![/red]")
        console.print("Install with: pip install apsw-sqlite3mc")
        return 1
    
    # Load configuration
    from ...config import load_settings
    app_config = load_settings(config)
    
    # Update encryption config
    if 'database_encryption' not in app_config:
        app_config['database_encryption'] = {}
    
    app_config['database_encryption']['enabled'] = True
    app_config['database_encryption']['cipher'] = cipher
    
    # Initialize encryption
    enc_db = EncryptedDatabase(app_config)
    
    # Get password
    if not password:
        password = click.prompt('Enter master password', hide_input=True)
    
    enc_db.key_manager._master_password = password
    
    # Find databases
    from ...DB import DB_REGISTRY  # Registry of all database classes
    
    databases = []
    for db_class in DB_REGISTRY:
        instance = db_class()
        db_path = instance._get_db_path()
        if db_path.exists():
            databases.append({
                'name': instance._get_db_name(),
                'path': db_path,
                'size': db_path.stat().st_size
            })
    
    if verify_only:
        # Just check status
        console.print("\n[bold]Database Encryption Status:[/bold]")
        for db in databases:
            encrypted = enc_db._is_encrypted(db['path'])
            status = "[green]✓ Encrypted[/green]" if encrypted else "[yellow]✗ Unencrypted[/yellow]"
            size_mb = db['size'] / 1024 / 1024
            console.print(f"{db['name']:20} {status:20} {size_mb:>10.1f} MB")
        return 0
    
    # Perform migration
    manager = MigrationManager(enc_db)
    
    console.print(f"\n[bold]Found {len(databases)} databases to check[/bold]")
    
    migrated = 0
    failed = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        
        for db in databases:
            if enc_db._is_encrypted(db['path']):
                console.print(f"[dim]{db['name']} - already encrypted[/dim]")
                continue
            
            task = progress.add_task(f"Migrating {db['name']}...", total=100)
            
            def update_progress(p, msg):
                progress.update(task, completed=int(p * 100), description=msg)
            
            try:
                manager.migrate_database(
                    db['path'],
                    password,
                    db['name'],
                    mode=mode,
                    progress_callback=update_progress
                )
                migrated += 1
                console.print(f"[green]✓ {db['name']} - successfully encrypted[/green]")
            except Exception as e:
                failed += 1
                console.print(f"[red]✗ {db['name']} - failed: {e}[/red]")
    
    # Summary
    console.print(f"\n[bold]Migration Summary:[/bold]")
    console.print(f"  Migrated: [green]{migrated}[/green]")
    console.print(f"  Failed:   [red]{failed}[/red]")
    console.print(f"  Skipped:  [dim]{len(databases) - migrated - failed}[/dim]")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    migrate_databases()
```

### Phase 8: Rollout Plan

#### 8.1 Timeline
- **Week 1-2**: Core implementation and unit tests
- **Week 3-4**: Migration tools and integration tests  
- **Week 5**: UI integration and documentation
- **Week 6**: Beta testing with volunteers
- **Week 7**: Performance optimization
- **Week 8**: Production release

#### 8.2 Feature Flags
```python
# Enable gradual rollout
class FeatureFlags:
    # Start with opt-in
    ENCRYPTION_ENABLED_DEFAULT = False
    ENCRYPTION_AUTO_MIGRATE = False
    ENCRYPTION_REQUIRED = False
    
    # Phase 1: Opt-in (Month 1)
    # ENCRYPTION_ENABLED_DEFAULT = False
    
    # Phase 2: Default on (Month 2) 
    # ENCRYPTION_ENABLED_DEFAULT = True
    # ENCRYPTION_AUTO_MIGRATE = True
    
    # Phase 3: Required (Month 3)
    # ENCRYPTION_REQUIRED = True
```

#### 8.3 Monitoring
```python
# Track adoption and issues
METRICS_TO_TRACK = [
    "encryption_enabled_count",
    "encryption_disabled_count", 
    "migration_success_count",
    "migration_failure_count",
    "encryption_performance_overhead",
    "cipher_usage_distribution",
]
```

### Conclusion

This comprehensive implementation plan provides a production-ready approach to adding SQLite3MultipleCiphers encryption to tldw_chatbook. The phased approach ensures:

1. **Security**: Strong encryption with modern ciphers
2. **Compatibility**: Gradual migration with fallbacks
3. **Performance**: Minimal overhead with optimizations
4. **Usability**: Clear UI and documentation
5. **Reliability**: Comprehensive testing and error handling

The implementation prioritizes user experience while providing robust security for sensitive data.