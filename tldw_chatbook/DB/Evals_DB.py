# Evals_DB.py
# Description: DB Library for LLM Evaluation Management
#
"""
Evals_DB.py
-----------

A comprehensive SQLite-based library for managing LLM evaluation data including:
- Evaluation tasks and configurations
- Model performance results
- Benchmark datasets
- Comparative analysis

This library provides:
- Schema management with versioning
- Thread-safe database connections using `threading.local`
- CRUD operations for evaluation entities
- Support for multiple evaluation formats (Eleuther, custom)
- Full-Text Search (FTS5) capabilities
- Results analysis and aggregation
"""

import sqlite3
import json
import uuid
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Tuple
from loguru import logger

from .sql_validation import validate_table_name, validate_column_name
from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram

# Database Schema Version
SCHEMA_VERSION = 2

class EvalsDBError(Exception):
    """Base exception for EvalsDB related errors."""
    pass

class SchemaError(EvalsDBError):
    """Exception for schema version mismatches or migration failures."""
    pass

class InputError(ValueError):
    """Custom exception for input validation errors."""
    pass

class ConflictError(EvalsDBError):
    """Indicates a conflict due to concurrent modification or unique constraint violation."""
    
    def __init__(self, message="Conflict detected.", entity: Optional[str] = None, entity_id: Any = None):
        super().__init__(message)
        self.entity = entity
        self.entity_id = entity_id

class EvalsDB:
    """Database manager for LLM evaluation data and results."""
    
    def __init__(self, db_path: Union[str, Path] = "evals.db", client_id: str = "default_client"):
        """
        Initialize the EvalsDB with a database path and client ID.
        
        Args:
            db_path: Path to the SQLite database file
            client_id: Identifier for the client making changes (for audit trail)
        """
        # Handle special case for in-memory database
        if db_path == ":memory:":
            self.db_path = db_path
        else:
            self.db_path = Path(db_path)
            # Ensure database directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.client_id = client_id
        self._local = threading.local()
        
        # Initialize database schema
        self._init_schema()
        
        logger.info(f"EvalsDB initialized with path: {self.db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            # Convert Path to string if necessary, but keep :memory: as is
            db_path_str = self.db_path if isinstance(self.db_path, str) else str(self.db_path)
            conn = sqlite3.connect(db_path_str, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            self._local.connection = conn
        return self._local.connection
    
    def get_connection(self) -> sqlite3.Connection:
        """Public method to get thread-local database connection."""
        return self._get_connection()
    
    def _init_schema(self):
        """Initialize database schema."""
        conn = self._get_connection()
        try:
            with conn:
                # Check current schema version
                cursor = conn.execute("PRAGMA user_version")
                current_version = cursor.fetchone()[0]
                
                if current_version == 0:
                    self._create_schema(conn)
                elif current_version < SCHEMA_VERSION:
                    self._migrate_schema(conn, current_version)
                elif current_version > SCHEMA_VERSION:
                    raise SchemaError(f"Database version {current_version} is newer than supported version {SCHEMA_VERSION}")
                    
        except Exception as e:
            logger.error(f"Schema initialization failed: {e}")
            raise SchemaError(f"Failed to initialize schema: {e}")
    
    def _create_schema(self, conn: sqlite3.Connection):
        """Create the initial database schema."""
        
        # Task definitions table
        conn.execute("""
            CREATE TABLE eval_tasks (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                task_type TEXT NOT NULL CHECK (task_type IN ('question_answer', 'logprob', 'generation', 'classification')),
                config_format TEXT NOT NULL CHECK (config_format IN ('eleuther', 'custom')),
                config_data TEXT NOT NULL, -- JSON configuration
                dataset_id TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                version INTEGER NOT NULL DEFAULT 1,
                client_id TEXT NOT NULL,
                deleted_at TEXT,
                FOREIGN KEY (dataset_id) REFERENCES eval_datasets (id)
            )
        """)
        
        # Dataset management table
        conn.execute("""
            CREATE TABLE eval_datasets (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                format TEXT NOT NULL CHECK (format IN ('huggingface', 'json', 'csv', 'custom')),
                source_path TEXT NOT NULL,
                metadata TEXT, -- JSON metadata
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                version INTEGER NOT NULL DEFAULT 1,
                client_id TEXT NOT NULL,
                deleted_at TEXT
            )
        """)
        
        # Model configurations table
        conn.execute("""
            CREATE TABLE eval_models (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                name TEXT NOT NULL,
                provider TEXT NOT NULL,
                model_id TEXT NOT NULL,
                config TEXT, -- JSON configuration for model parameters
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                version INTEGER NOT NULL DEFAULT 1,
                client_id TEXT NOT NULL,
                deleted_at TEXT,
                UNIQUE(name, provider, model_id)
            )
        """)
        
        # Evaluation runs table
        conn.execute("""
            CREATE TABLE eval_runs (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                name TEXT NOT NULL,
                task_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')) DEFAULT 'pending',
                start_time TEXT,
                end_time TEXT,
                total_samples INTEGER,
                completed_samples INTEGER DEFAULT 0,
                config_overrides TEXT, -- JSON overrides for task config
                error_message TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                version INTEGER NOT NULL DEFAULT 1,
                client_id TEXT NOT NULL,
                deleted_at TEXT,
                FOREIGN KEY (task_id) REFERENCES eval_tasks (id),
                FOREIGN KEY (model_id) REFERENCES eval_models (id)
            )
        """)
        
        # Individual sample results table
        conn.execute("""
            CREATE TABLE eval_results (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                run_id TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                input_data TEXT NOT NULL, -- JSON input data
                expected_output TEXT,
                actual_output TEXT,
                logprobs TEXT, -- JSON log probabilities if available
                metrics TEXT, -- JSON metrics for this sample
                metadata TEXT, -- JSON additional metadata
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                client_id TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES eval_runs (id),
                UNIQUE(run_id, sample_id)
            )
        """)
        
        # Aggregated run metrics table
        conn.execute("""
            CREATE TABLE eval_run_metrics (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                run_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_type TEXT NOT NULL CHECK (metric_type IN ('accuracy', 'f1', 'rouge', 'bleu', 'perplexity', 'custom')),
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                client_id TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES eval_runs (id),
                UNIQUE(run_id, metric_name)
            )
        """)
        
        # Create indexes for performance
        conn.execute("CREATE INDEX idx_eval_tasks_type ON eval_tasks (task_type)")
        conn.execute("CREATE INDEX idx_eval_tasks_deleted ON eval_tasks (deleted_at)")
        conn.execute("CREATE INDEX idx_eval_runs_status ON eval_runs (status)")
        conn.execute("CREATE INDEX idx_eval_runs_task ON eval_runs (task_id)")
        conn.execute("CREATE INDEX idx_eval_runs_model ON eval_runs (model_id)")
        conn.execute("CREATE INDEX idx_eval_results_run ON eval_results (run_id)")
        conn.execute("CREATE INDEX idx_eval_run_metrics_run ON eval_run_metrics (run_id)")
        
        # Create FTS5 tables for search
        conn.execute("""
            CREATE VIRTUAL TABLE eval_tasks_fts USING fts5(
                id UNINDEXED,
                name,
                description,
                content='eval_tasks',
                content_rowid='rowid'
            )
        """)
        
        conn.execute("""
            CREATE VIRTUAL TABLE eval_datasets_fts USING fts5(
                id UNINDEXED,
                name,
                description,
                content='eval_datasets',
                content_rowid='rowid'
            )
        """)
        
        # Create triggers to maintain FTS5 tables
        conn.execute("""
            CREATE TRIGGER eval_tasks_fts_insert AFTER INSERT ON eval_tasks BEGIN
                INSERT INTO eval_tasks_fts (id, name, description) 
                VALUES (new.id, new.name, new.description);
            END
        """)
        
        conn.execute("""
            CREATE TRIGGER eval_tasks_fts_update AFTER UPDATE ON eval_tasks BEGIN
                UPDATE eval_tasks_fts SET name = new.name, description = new.description 
                WHERE id = new.id;
            END
        """)
        
        conn.execute("""
            CREATE TRIGGER eval_tasks_fts_delete AFTER DELETE ON eval_tasks BEGIN
                DELETE FROM eval_tasks_fts WHERE id = old.id;
            END
        """)
        
        conn.execute("""
            CREATE TRIGGER eval_datasets_fts_insert AFTER INSERT ON eval_datasets BEGIN
                INSERT INTO eval_datasets_fts (id, name, description) 
                VALUES (new.id, new.name, new.description);
            END
        """)
        
        conn.execute("""
            CREATE TRIGGER eval_datasets_fts_update AFTER UPDATE ON eval_datasets BEGIN
                UPDATE eval_datasets_fts SET name = new.name, description = new.description 
                WHERE id = new.id;
            END
        """)
        
        conn.execute("""
            CREATE TRIGGER eval_datasets_fts_delete AFTER DELETE ON eval_datasets BEGIN
                DELETE FROM eval_datasets_fts WHERE id = old.id;
            END
        """)
        
        # A/B Testing tables
        conn.execute("""
            CREATE TABLE ab_tests (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                test_id TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                description TEXT,
                task_id TEXT NOT NULL,
                model_a_id TEXT NOT NULL,
                model_b_id TEXT NOT NULL,
                config TEXT NOT NULL, -- JSON configuration
                status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')) DEFAULT 'pending',
                winner TEXT CHECK (winner IN ('model_a', 'model_b', 'tie', NULL)),
                result_data TEXT, -- JSON result data
                started_at TEXT,
                completed_at TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                version INTEGER NOT NULL DEFAULT 1,
                client_id TEXT NOT NULL,
                deleted_at TEXT,
                FOREIGN KEY (task_id) REFERENCES eval_tasks (id),
                FOREIGN KEY (model_a_id) REFERENCES eval_models (id),
                FOREIGN KEY (model_b_id) REFERENCES eval_models (id)
            )
        """)
        
        conn.execute("""
            CREATE TABLE ab_test_runs (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                ab_test_id TEXT NOT NULL,
                run_a_id TEXT NOT NULL,
                run_b_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                client_id TEXT NOT NULL,
                FOREIGN KEY (ab_test_id) REFERENCES ab_tests (id),
                FOREIGN KEY (run_a_id) REFERENCES eval_runs (id),
                FOREIGN KEY (run_b_id) REFERENCES eval_runs (id)
            )
        """)
        
        # Create indexes for A/B tests
        conn.execute("CREATE INDEX idx_ab_tests_status ON ab_tests (status)")
        conn.execute("CREATE INDEX idx_ab_tests_task ON ab_tests (task_id)")
        conn.execute("CREATE INDEX idx_ab_tests_models ON ab_tests (model_a_id, model_b_id)")
        conn.execute("CREATE INDEX idx_ab_test_runs_test ON ab_test_runs (ab_test_id)")
        
        # Set schema version
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        
        logger.info(f"Created EvalsDB schema version {SCHEMA_VERSION}")
    
    def _migrate_schema(self, conn: sqlite3.Connection, current_version: int):
        """Migrate schema from current_version to SCHEMA_VERSION."""
        logger.info(f"Migrating EvalsDB schema from version {current_version} to {SCHEMA_VERSION}")
        
        # Migrate from version 1 to 2
        if current_version == 1 and SCHEMA_VERSION >= 2:
            logger.info("Migrating to version 2: Adding A/B testing tables")
            
            # A/B Testing tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ab_tests (
                    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                    test_id TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    description TEXT,
                    task_id TEXT NOT NULL,
                    model_a_id TEXT NOT NULL,
                    model_b_id TEXT NOT NULL,
                    config TEXT NOT NULL, -- JSON configuration
                    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')) DEFAULT 'pending',
                    winner TEXT CHECK (winner IN ('model_a', 'model_b', 'tie', NULL)),
                    result_data TEXT, -- JSON result data
                    started_at TEXT,
                    completed_at TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                    version INTEGER NOT NULL DEFAULT 1,
                    client_id TEXT NOT NULL,
                    deleted_at TEXT,
                    FOREIGN KEY (task_id) REFERENCES eval_tasks (id),
                    FOREIGN KEY (model_a_id) REFERENCES eval_models (id),
                    FOREIGN KEY (model_b_id) REFERENCES eval_models (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ab_test_runs (
                    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                    ab_test_id TEXT NOT NULL,
                    run_a_id TEXT NOT NULL,
                    run_b_id TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                    client_id TEXT NOT NULL,
                    FOREIGN KEY (ab_test_id) REFERENCES ab_tests (id),
                    FOREIGN KEY (run_a_id) REFERENCES eval_runs (id),
                    FOREIGN KEY (run_b_id) REFERENCES eval_runs (id)
                )
            """)
            
            # Create indexes for A/B tests
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ab_tests_status ON ab_tests (status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ab_tests_task ON ab_tests (task_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ab_tests_models ON ab_tests (model_a_id, model_b_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ab_test_runs_test ON ab_test_runs (ab_test_id)")
        
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    
    # --- Task Management ---
    
    def create_task(self, name: str, task_type: str, config_format: str, config_data: Dict[str, Any], 
                   description: str = None, dataset_id: str = None) -> str:
        """Create a new evaluation task."""
        start_time = time.time()
        
        # Clean control characters from name and description
        name = ''.join(c for c in name if c.isprintable() and ord(c) != 0)
        if description:
            description = ''.join(c for c in description if c.isprintable() and ord(c) != 0)
        
        if not name or not name.strip():
            raise InputError("Task name cannot be empty")
        
        if task_type not in ['question_answer', 'logprob', 'generation', 'classification']:
            raise InputError(f"Invalid task_type: {task_type}")
        
        if config_format not in ['eleuther', 'custom']:
            raise InputError(f"Invalid config_format: {config_format}")
        
        task_id = str(uuid.uuid4())
        config_json = json.dumps(config_data)
        
        # Log DB operation start
        log_counter("eval_db_operation_started", labels={
            "operation": "create_task",
            "table": "eval_tasks"
        })
        
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("""
                    INSERT INTO eval_tasks (id, name, description, task_type, config_format, 
                                          config_data, dataset_id, client_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (task_id, name.strip(), description, task_type, config_format, 
                     config_json, dataset_id, self.client_id))
                
                # Log successful operation
                duration = time.time() - start_time
                log_histogram("eval_db_operation_duration", duration, labels={
                    "operation": "create_task",
                    "table": "eval_tasks",
                    "status": "success"
                })
                log_counter("eval_db_operation_success", labels={
                    "operation": "create_task",
                    "table": "eval_tasks"
                })
                log_histogram("eval_db_record_size", len(config_json), labels={
                    "table": "eval_tasks",
                    "field": "config_data"
                })
                
                logger.info(f"Created eval task: {name} ({task_id})")
                return task_id
                
        except sqlite3.IntegrityError as e:
            duration = time.time() - start_time
            log_histogram("eval_db_operation_duration", duration, labels={
                "operation": "create_task",
                "table": "eval_tasks",
                "status": "error"
            })
            log_counter("eval_db_operation_error", labels={
                "operation": "create_task",
                "table": "eval_tasks",
                "error_type": "integrity_error"
            })
            if "UNIQUE constraint failed" in str(e):
                raise ConflictError(f"Task with name '{name}' already exists", "eval_tasks", name)
            raise EvalsDBError(f"Failed to create task: {e}")
    
    def update_task(self, task_id: str, name: str = None, description: str = None,
                   config_data: Dict[str, Any] = None) -> bool:
        """Update an existing task."""
        updates = []
        params = []
        
        if name is not None:
            updates.append("name = ?")
            params.append(name.strip())
        
        if description is not None:
            updates.append("description = ?")
            params.append(description)
            
        if config_data is not None:
            updates.append("config_data = ?")
            params.append(json.dumps(config_data))
        
        if not updates:
            return True  # Nothing to update
        
        updates.append("updated_at = datetime('now', 'utc')")
        updates.append("version = version + 1")
        
        query = f"UPDATE eval_tasks SET {', '.join(updates)} WHERE id = ? AND deleted_at IS NULL"
        params.append(task_id)
        
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.execute(query, params)
                if cursor.rowcount > 0:
                    logger.info(f"Updated eval task: {task_id}")
                    return True
                return False
                
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise ConflictError(f"Task name already exists", "eval_tasks", task_id)
            raise EvalsDBError(f"Failed to update task: {e}")
    
    def delete_task(self, task_id: str) -> bool:
        """Soft delete a task."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.execute("""
                    UPDATE eval_tasks 
                    SET deleted_at = datetime('now', 'utc'),
                        updated_at = datetime('now', 'utc')
                    WHERE id = ? AND deleted_at IS NULL
                """, (task_id,))
                
                if cursor.rowcount > 0:
                    logger.info(f"Deleted eval task: {task_id}")
                    return True
                return False
                
        except Exception as e:
            raise EvalsDBError(f"Failed to delete task: {e}")
    
    def get_task(self, task_id: str, include_deleted: bool = False) -> Optional[Dict[str, Any]]:
        """Get task by ID."""
        start_time = time.time()
        
        # Log DB operation start
        log_counter("eval_db_operation_started", labels={
            "operation": "get_task",
            "table": "eval_tasks"
        })
        
        conn = self._get_connection()
        
        query = "SELECT * FROM eval_tasks WHERE id = ?"
        if not include_deleted:
            query += " AND deleted_at IS NULL"
        
        cursor = conn.execute(query, (task_id,))
        
        row = cursor.fetchone()
        
        # Log operation metrics
        duration = time.time() - start_time
        status = "found" if row else "not_found"
        log_histogram("eval_db_operation_duration", duration, labels={
            "operation": "get_task",
            "table": "eval_tasks",
            "status": status
        })
        log_counter("eval_db_operation_success", labels={
            "operation": "get_task",
            "table": "eval_tasks",
            "status": status
        })
        
        if row:
            task = dict(row)
            task['config_data'] = json.loads(task['config_data'])
            return task
        return None
    
    def list_tasks(self, task_type: str = None, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List evaluation tasks with optional filtering."""
        conn = self._get_connection()
        
        query = "SELECT * FROM eval_tasks WHERE deleted_at IS NULL"
        params = []
        
        if task_type:
            query += " AND task_type = ?"
            params.append(task_type)
        
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor = conn.execute(query, params)
        tasks = []
        for row in cursor.fetchall():
            task = dict(row)
            task['config_data'] = json.loads(task['config_data'])
            tasks.append(task)
        
        return tasks
    
    def search_tasks(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search tasks using FTS5."""
        conn = self._get_connection()
        
        # Remove null bytes and other control characters
        query = ''.join(c for c in query if c.isprintable() and ord(c) != 0)
        
        # For special characters or very short queries, use LIKE instead of FTS
        if len(query) <= 2 or any(c in query for c in ':*"\'[]()'):
            cursor = conn.execute("""
                SELECT * FROM eval_tasks 
                WHERE (name LIKE ? OR description LIKE ?) AND deleted_at IS NULL
                ORDER BY created_at DESC LIMIT ?
            """, (f'%{query}%', f'%{query}%', limit))
        else:
            # For normal queries, use FTS5 with proper escaping
            # Escape double quotes in the query
            escaped_query = query.replace('"', '""')
            safe_query = f'"{escaped_query}"' if escaped_query else '""'
            cursor = conn.execute("""
                SELECT t.* FROM eval_tasks t
                JOIN eval_tasks_fts fts ON t.id = fts.id
                WHERE eval_tasks_fts MATCH ? AND t.deleted_at IS NULL
                ORDER BY rank LIMIT ?
            """, (safe_query, limit))
        
        tasks = []
        for row in cursor.fetchall():
            task = dict(row)
            task['config_data'] = json.loads(task['config_data'])
            tasks.append(task)
        
        return tasks
    
    # --- Dataset Management ---
    
    def create_dataset(self, name: str, format: str, source_path: str, 
                      description: str = None, metadata: Dict[str, Any] = None) -> str:
        """Create a new evaluation dataset."""
        if not name or not name.strip():
            raise InputError("Dataset name cannot be empty")
        
        if format not in ['huggingface', 'json', 'csv', 'custom']:
            raise InputError(f"Invalid format: {format}")
        
        if not source_path or not source_path.strip():
            raise InputError("Source path cannot be empty")
        
        dataset_id = str(uuid.uuid4())
        metadata_json = json.dumps(metadata or {})
        
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("""
                    INSERT INTO eval_datasets (id, name, description, format, source_path, 
                                             metadata, client_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (dataset_id, name.strip(), description, format, source_path.strip(), 
                     metadata_json, self.client_id))
                
                logger.info(f"Created eval dataset: {name} ({dataset_id})")
                return dataset_id
                
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise ConflictError(f"Dataset with name '{name}' already exists", "eval_datasets", name)
            raise EvalsDBError(f"Failed to create dataset: {e}")
    
    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset by ID."""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT * FROM eval_datasets 
            WHERE id = ? AND deleted_at IS NULL
        """, (dataset_id,))
        
        row = cursor.fetchone()
        if row:
            dataset = dict(row)
            dataset['metadata'] = json.loads(dataset['metadata'])
            return dataset
        return None
    
    def list_datasets(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List evaluation datasets."""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT * FROM eval_datasets 
            WHERE deleted_at IS NULL
            ORDER BY created_at DESC LIMIT ? OFFSET ?
        """, (limit, offset))
        
        datasets = []
        for row in cursor.fetchall():
            dataset = dict(row)
            dataset['metadata'] = json.loads(dataset['metadata'])
            datasets.append(dataset)
        
        return datasets
    
    def search_datasets(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search datasets using FTS5."""
        # Escape special characters in FTS5 query by wrapping in quotes
        safe_query = f'"{query}"' if query else '""'
        
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT d.* FROM eval_datasets d
            JOIN eval_datasets_fts fts ON d.id = fts.id
            WHERE eval_datasets_fts MATCH ? AND d.deleted_at IS NULL
            ORDER BY rank LIMIT ?
        """, (safe_query, limit))
        
        datasets = []
        for row in cursor.fetchall():
            dataset = dict(row)
            dataset['metadata'] = json.loads(dataset['metadata'])
            datasets.append(dataset)
        
        return datasets
    
    # --- Model Management ---
    
    def create_model(self, name: str, provider: str, model_id: str, 
                    config: Dict[str, Any] = None) -> str:
        """Create a new model configuration."""
        if not all([name.strip(), provider.strip(), model_id.strip()]):
            raise InputError("Name, provider, and model_id cannot be empty")
        
        model_uuid = str(uuid.uuid4())
        config_json = json.dumps(config or {})
        
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("""
                    INSERT INTO eval_models (id, name, provider, model_id, config, client_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (model_uuid, name.strip(), provider.strip(), model_id.strip(), 
                     config_json, self.client_id))
                
                logger.info(f"Created eval model: {name} ({model_uuid})")
                return model_uuid
                
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise ConflictError(f"Model with name '{name}', provider '{provider}', and model_id '{model_id}' already exists", 
                                  "eval_models", f"{name}:{provider}:{model_id}")
            raise EvalsDBError(f"Failed to create model: {e}")
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model by ID."""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT * FROM eval_models 
            WHERE id = ? AND deleted_at IS NULL
        """, (model_id,))
        
        row = cursor.fetchone()
        if row:
            model = dict(row)
            model['config'] = json.loads(model['config'])
            return model
        return None
    
    def list_models(self, provider: str = None, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List evaluation models with optional provider filtering."""
        conn = self._get_connection()
        
        query = "SELECT * FROM eval_models WHERE deleted_at IS NULL"
        params = []
        
        if provider:
            query += " AND provider = ?"
            params.append(provider)
        
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor = conn.execute(query, params)
        models = []
        for row in cursor.fetchall():
            model = dict(row)
            model['config'] = json.loads(model['config'])
            models.append(model)
        
        return models
    
    # --- Evaluation Run Management ---
    
    def create_run(self, name: str, task_id: str, model_id: str, 
                  config_overrides: Dict[str, Any] = None) -> str:
        """Create a new evaluation run."""
        start_time = time.time()
        
        if not all([name.strip(), task_id.strip(), model_id.strip()]):
            raise InputError("Name, task_id, and model_id cannot be empty")
        
        # Verify task and model exist
        if not self.get_task(task_id):
            raise InputError(f"Task {task_id} not found")
        if not self.get_model(model_id):
            raise InputError(f"Model {model_id} not found")
        
        run_id = str(uuid.uuid4())
        config_json = json.dumps(config_overrides or {})
        
        # Log DB operation start
        log_counter("eval_db_operation_started", labels={
            "operation": "create_run",
            "table": "eval_runs"
        })
        
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("""
                    INSERT INTO eval_runs (id, name, task_id, model_id, config_overrides, client_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (run_id, name.strip(), task_id, model_id, config_json, self.client_id))
                
                # Log successful operation
                duration = time.time() - start_time
                log_histogram("eval_db_operation_duration", duration, labels={
                    "operation": "create_run",
                    "table": "eval_runs",
                    "status": "success"
                })
                log_counter("eval_db_operation_success", labels={
                    "operation": "create_run",
                    "table": "eval_runs"
                })
                
                logger.info(f"Created eval run: {name} ({run_id})")
                return run_id
                
        except Exception as e:
            duration = time.time() - start_time
            log_histogram("eval_db_operation_duration", duration, labels={
                "operation": "create_run",
                "table": "eval_runs",
                "status": "error"
            })
            log_counter("eval_db_operation_error", labels={
                "operation": "create_run",
                "table": "eval_runs",
                "error_type": type(e).__name__
            })
            raise
    
    def update_run_status(self, run_id: str, status: str, error_message: str = None):
        """Update run status."""
        if status not in ['pending', 'running', 'completed', 'failed', 'cancelled']:
            raise InputError(f"Invalid status: {status}")
        
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()
        
        with conn:
            if status == 'running' and not error_message:
                conn.execute("""
                    UPDATE eval_runs 
                    SET status = ?, start_time = ?, updated_at = ?
                    WHERE id = ?
                """, (status, now, now, run_id))
            elif status in ['completed', 'failed', 'cancelled']:
                conn.execute("""
                    UPDATE eval_runs 
                    SET status = ?, end_time = ?, error_message = ?, updated_at = ?
                    WHERE id = ?
                """, (status, now, error_message, now, run_id))
            else:
                conn.execute("""
                    UPDATE eval_runs 
                    SET status = ?, error_message = ?, updated_at = ?
                    WHERE id = ?
                """, (status, error_message, now, run_id))
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get evaluation run by ID."""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT r.*, t.name as task_name, m.name as model_name
            FROM eval_runs r
            JOIN eval_tasks t ON r.task_id = t.id
            JOIN eval_models m ON r.model_id = m.id
            WHERE r.id = ? AND r.deleted_at IS NULL
        """, (run_id,))
        
        row = cursor.fetchone()
        if row:
            run = dict(row)
            run['config_overrides'] = json.loads(run['config_overrides'])
            return run
        return None
    
    def list_runs(self, status: str = None, task_id: str = None, model_id: str = None,
                 limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List evaluation runs with optional filtering."""
        conn = self._get_connection()
        
        query = """
            SELECT r.*, t.name as task_name, m.name as model_name
            FROM eval_runs r
            JOIN eval_tasks t ON r.task_id = t.id
            JOIN eval_models m ON r.model_id = m.id
            WHERE r.deleted_at IS NULL
        """
        params = []
        
        if status:
            query += " AND r.status = ?"
            params.append(status)
        if task_id:
            query += " AND r.task_id = ?"
            params.append(task_id)
        if model_id:
            query += " AND r.model_id = ?"
            params.append(model_id)
        
        query += " ORDER BY r.created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor = conn.execute(query, params)
        runs = []
        for row in cursor.fetchall():
            run = dict(row)
            run['config_overrides'] = json.loads(run['config_overrides'])
            runs.append(run)
        
        return runs
    
    # --- Results Management ---
    
    def store_result(self, run_id: str, sample_id: str, input_data: Dict[str, Any],
                    actual_output: str, expected_output: str = None, 
                    logprobs: Dict[str, Any] = None, metrics: Dict[str, Any] = None,
                    metadata: Dict[str, Any] = None) -> str:
        """Store individual evaluation result."""
        start_time = time.time()
        result_id = str(uuid.uuid4())
        
        # Log DB operation start
        log_counter("eval_db_operation_started", labels={
            "operation": "store_result",
            "table": "eval_results"
        })
        
        # Calculate data sizes
        input_size = len(json.dumps(input_data))
        output_size = len(actual_output) if actual_output else 0
        metrics_size = len(json.dumps(metrics or {}))
        
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("""
                    INSERT INTO eval_results 
                    (id, run_id, sample_id, input_data, expected_output, actual_output, 
                     logprobs, metrics, metadata, client_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (result_id, run_id, sample_id, json.dumps(input_data), expected_output,
                     actual_output, json.dumps(logprobs or {}), json.dumps(metrics or {}),
                     json.dumps(metadata or {}), self.client_id))
                
                # Update completed samples count
                conn.execute("""
                    UPDATE eval_runs 
                    SET completed_samples = completed_samples + 1, updated_at = ?
                    WHERE id = ?
                """, (datetime.now(timezone.utc).isoformat(), run_id))
                
                # Log successful operation
                duration = time.time() - start_time
                log_histogram("eval_db_operation_duration", duration, labels={
                    "operation": "store_result",
                    "table": "eval_results",
                    "status": "success"
                })
                log_counter("eval_db_operation_success", labels={
                    "operation": "store_result",
                    "table": "eval_results"
                })
                
                # Log data sizes
                log_histogram("eval_db_result_input_size", input_size, labels={
                    "table": "eval_results"
                })
                log_histogram("eval_db_result_output_size", output_size, labels={
                    "table": "eval_results"
                })
                log_histogram("eval_db_result_metrics_size", metrics_size, labels={
                    "table": "eval_results"
                })
                
                return result_id
                
        except Exception as e:
            duration = time.time() - start_time
            log_histogram("eval_db_operation_duration", duration, labels={
                "operation": "store_result",
                "table": "eval_results",
                "status": "error"
            })
            log_counter("eval_db_operation_error", labels={
                "operation": "store_result",
                "table": "eval_results",
                "error_type": type(e).__name__
            })
            raise
    
    def store_run_metrics(self, run_id: str, metrics: Dict[str, Tuple[float, str]]):
        """Store aggregated metrics for a run.
        
        Args:
            run_id: ID of the evaluation run
            metrics: Dict mapping metric_name to (value, type) tuples
        """
        conn = self._get_connection()
        with conn:
            for metric_name, (value, metric_type) in metrics.items():
                conn.execute("""
                    INSERT OR REPLACE INTO eval_run_metrics 
                    (run_id, metric_name, metric_value, metric_type, client_id)
                    VALUES (?, ?, ?, ?, ?)
                """, (run_id, metric_name, value, metric_type, self.client_id))
    
    def get_run_results(self, run_id: str, limit: int = 1000, offset: int = 0) -> List[Dict[str, Any]]:
        """Get results for a specific run."""
        start_time = time.time()
        
        # Log DB operation start
        log_counter("eval_db_operation_started", labels={
            "operation": "get_run_results",
            "table": "eval_results"
        })
        
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT * FROM eval_results 
            WHERE run_id = ?
            ORDER BY created_at ASC LIMIT ? OFFSET ?
        """, (run_id, limit, offset))
        
        results = []
        for row in cursor.fetchall():
            result = dict(row)
            result['input_data'] = json.loads(result['input_data'])
            result['logprobs'] = json.loads(result['logprobs'])
            result['metrics'] = json.loads(result['metrics'])
            result['metadata'] = json.loads(result['metadata'])
            results.append(result)
        
        # Log operation metrics
        duration = time.time() - start_time
        log_histogram("eval_db_operation_duration", duration, labels={
            "operation": "get_run_results",
            "table": "eval_results",
            "status": "success"
        })
        log_counter("eval_db_operation_success", labels={
            "operation": "get_run_results",
            "table": "eval_results"
        })
        log_histogram("eval_db_result_count", len(results), labels={
            "operation": "get_run_results",
            "table": "eval_results"
        })
        
        return results
    
    def get_results_for_run(self, run_id: str, limit: int = 1000, offset: int = 0) -> List[Dict[str, Any]]:
        """Alias for get_run_results for backward compatibility."""
        return self.get_run_results(run_id, limit, offset)
    
    def get_run_metrics(self, run_id: str) -> Dict[str, Any]:
        """Get aggregated metrics for a run."""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT metric_name, metric_value, metric_type 
            FROM eval_run_metrics 
            WHERE run_id = ?
        """, (run_id,))
        
        metrics = {}
        for row in cursor.fetchall():
            metrics[row['metric_name']] = {
                'value': row['metric_value'],
                'type': row['metric_type']
            }
        
        return metrics
    
    # --- Analysis Methods ---
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare metrics across multiple runs."""
        if not run_ids:
            return {}
        
        conn = self._get_connection()
        placeholders = ','.join(['?'] * len(run_ids))
        
        cursor = conn.execute(f"""
            SELECT r.id, r.name, m.metric_name, m.metric_value, m.metric_type
            FROM eval_runs r
            JOIN eval_run_metrics m ON r.id = m.run_id
            WHERE r.id IN ({placeholders})
            ORDER BY r.name, m.metric_name
        """, run_ids)
        
        comparison = {}
        for row in cursor.fetchall():
            run_name = row['name']
            if run_name not in comparison:
                comparison[run_name] = {'run_id': row['id'], 'metrics': {}}
            
            comparison[run_name]['metrics'][row['metric_name']] = {
                'value': row['metric_value'],
                'type': row['metric_type']
            }
        
        return comparison
    
    # --- A/B Testing Methods ---
    
    def create_ab_test(self, test_id: str, name: str, description: str, task_id: str,
                      model_a_id: str, model_b_id: str, config: Dict[str, Any]) -> str:
        """Create a new A/B test."""
        if not all([test_id.strip(), name.strip(), task_id.strip(), 
                   model_a_id.strip(), model_b_id.strip()]):
            raise InputError("Required fields cannot be empty")
        
        # Verify task and models exist
        if not self.get_task(task_id):
            raise InputError(f"Task {task_id} not found")
        if not self.get_model(model_a_id):
            raise InputError(f"Model {model_a_id} not found")
        if not self.get_model(model_b_id):
            raise InputError(f"Model {model_b_id} not found")
        
        ab_id = str(uuid.uuid4())
        config_json = json.dumps(config)
        
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("""
                    INSERT INTO ab_tests 
                    (id, test_id, name, description, task_id, model_a_id, model_b_id, 
                     config, client_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (ab_id, test_id, name.strip(), description, task_id, 
                     model_a_id, model_b_id, config_json, self.client_id))
                
                logger.info(f"Created A/B test: {name} ({ab_id})")
                return ab_id
                
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise ConflictError(f"A/B test with ID '{test_id}' already exists", 
                                  "ab_tests", test_id)
            raise EvalsDBError(f"Failed to create A/B test: {e}")
    
    def update_ab_test_status(self, test_id: str, status: str, error_message: str = None):
        """Update A/B test status."""
        if status not in ['pending', 'running', 'completed', 'failed', 'cancelled']:
            raise InputError(f"Invalid status: {status}")
        
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()
        
        with conn:
            if status == 'running':
                conn.execute("""
                    UPDATE ab_tests 
                    SET status = ?, started_at = ?, updated_at = ?
                    WHERE test_id = ?
                """, (status, now, now, test_id))
            elif status in ['completed', 'failed', 'cancelled']:
                conn.execute("""
                    UPDATE ab_tests 
                    SET status = ?, completed_at = ?, updated_at = ?
                    WHERE test_id = ?
                """, (status, now, now, test_id))
    
    def update_ab_test_results(self, test_id: str, result: 'ABTestResult'):
        """Update A/B test with results."""
        from ..Evals.ab_testing import ABTestResult
        
        conn = self._get_connection()
        result_data = {
            'model_a_metrics': result.model_a_metrics,
            'model_b_metrics': result.model_b_metrics,
            'statistical_tests': result.statistical_tests,
            'winner': result.winner,
            'confidence_intervals': result.confidence_intervals,
            'model_a_latency': result.model_a_latency,
            'model_b_latency': result.model_b_latency,
            'model_a_cost': result.model_a_cost,
            'model_b_cost': result.model_b_cost,
            'sample_size': result.sample_size
        }
        
        winner = result.winner if result.winner else 'tie'
        
        with conn:
            conn.execute("""
                UPDATE ab_tests 
                SET result_data = ?, winner = ?, status = 'completed', 
                    completed_at = ?, updated_at = ?
                WHERE test_id = ?
            """, (json.dumps(result_data), winner, 
                 result.completed_at.isoformat() if result.completed_at else datetime.now(timezone.utc).isoformat(),
                 datetime.now(timezone.utc).isoformat(), test_id))
            
            logger.info(f"Updated A/B test results: {test_id}, winner: {winner}")
    
    def get_ab_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get A/B test by test_id."""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT a.*, 
                   t.name as task_name,
                   ma.name as model_a_name,
                   mb.name as model_b_name
            FROM ab_tests a
            JOIN eval_tasks t ON a.task_id = t.id
            JOIN eval_models ma ON a.model_a_id = ma.id
            JOIN eval_models mb ON a.model_b_id = mb.id
            WHERE a.test_id = ? AND a.deleted_at IS NULL
        """, (test_id,))
        
        row = cursor.fetchone()
        if row:
            test = dict(row)
            test['config'] = json.loads(test['config'])
            if test['result_data']:
                test['result_data'] = json.loads(test['result_data'])
            return test
        return None
    
    def get_ab_test_status(self, test_id: str) -> Dict[str, Any]:
        """Get the status of an A/B test."""
        test = self.get_ab_test(test_id)
        if test:
            return {
                'status': test['status'],
                'started_at': test['started_at'],
                'completed_at': test['completed_at'],
                'winner': test['winner'],
                'result_data': test.get('result_data')
            }
        return {'status': 'not_found'}
    
    def list_ab_tests(self, status: str = None, task_id: str = None,
                     limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List A/B tests with optional filtering."""
        conn = self._get_connection()
        
        query = """
            SELECT a.*, 
                   t.name as task_name,
                   ma.name as model_a_name,
                   mb.name as model_b_name
            FROM ab_tests a
            JOIN eval_tasks t ON a.task_id = t.id
            JOIN eval_models ma ON a.model_a_id = ma.id
            JOIN eval_models mb ON a.model_b_id = mb.id
            WHERE a.deleted_at IS NULL
        """
        params = []
        
        if status:
            query += " AND a.status = ?"
            params.append(status)
        if task_id:
            query += " AND a.task_id = ?"
            params.append(task_id)
        
        query += " ORDER BY a.created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor = conn.execute(query, params)
        tests = []
        for row in cursor.fetchall():
            test = dict(row)
            test['config'] = json.loads(test['config'])
            if test['result_data']:
                test['result_data'] = json.loads(test['result_data'])
            tests.append(test)
        
        return tests
    
    def create_ab_test_runs(self, ab_test_id: str, run_a_id: str, run_b_id: str) -> str:
        """Link evaluation runs to an A/B test."""
        run_link_id = str(uuid.uuid4())
        
        conn = self._get_connection()
        with conn:
            conn.execute("""
                INSERT INTO ab_test_runs (id, ab_test_id, run_a_id, run_b_id, client_id)
                VALUES (?, ?, ?, ?, ?)
            """, (run_link_id, ab_test_id, run_a_id, run_b_id, self.client_id))
        
        return run_link_id
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')