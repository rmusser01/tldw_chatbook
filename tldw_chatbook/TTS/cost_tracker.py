# cost_tracker.py
# Description: TTS usage and cost tracking for various providers
#
# Imports
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from loguru import logger

# Local imports
from tldw_chatbook.config import get_cli_setting

#######################################################################################################################
#
# Cost Tracking Data Models

class TTSProvider(Enum):
    """TTS provider identifiers"""
    OPENAI = "openai"
    ELEVENLABS = "elevenlabs"
    GOOGLE = "google"
    AMAZON = "amazon"
    AZURE = "azure"
    LOCAL = "local"


@dataclass
class TTSUsageRecord:
    """Record of a single TTS generation"""
    timestamp: datetime
    provider: str
    model: str
    characters: int
    estimated_cost: float
    voice: str
    format: str
    duration_seconds: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProviderCostInfo:
    """Cost information for a TTS provider"""
    provider: str
    cost_per_1k_chars: float  # Cost per 1000 characters
    free_tier_chars: int  # Free characters per month
    notes: str = ""


class CostTracker:
    """
    Track TTS usage and costs across providers.
    
    Stores usage data in SQLite database and provides cost estimates.
    """
    
    # Default cost information (prices as of 2024)
    DEFAULT_COSTS = {
        TTSProvider.OPENAI: ProviderCostInfo(
            provider="openai",
            cost_per_1k_chars=0.015,  # $0.015 per 1K characters for tts-1
            free_tier_chars=0,
            notes="tts-1-hd costs $0.030 per 1K chars"
        ),
        TTSProvider.ELEVENLABS: ProviderCostInfo(
            provider="elevenlabs",
            cost_per_1k_chars=0.18,  # ~$0.18 per 1K chars (varies by plan)
            free_tier_chars=10000,  # 10K chars/month free tier
            notes="Pricing varies by subscription plan"
        ),
        TTSProvider.GOOGLE: ProviderCostInfo(
            provider="google",
            cost_per_1k_chars=0.016,  # $0.016 per 1K chars for WaveNet
            free_tier_chars=1000000,  # 1M chars/month free
            notes="Standard voices are $0.004 per 1K chars"
        ),
        TTSProvider.LOCAL: ProviderCostInfo(
            provider="local",
            cost_per_1k_chars=0.0,  # Free for local models
            free_tier_chars=0,
            notes="No cost for local models"
        ),
    }
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize cost tracker.
        
        Args:
            db_path: Path to SQLite database (defaults to user data dir)
        """
        if db_path is None:
            # Use default location
            from tldw_chatbook.Utils.utils import get_project_root
            data_dir = get_project_root() / "user_data"
            data_dir.mkdir(exist_ok=True)
            db_path = data_dir / "tts_usage.db"
        
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_database()
        self._load_custom_costs()
    
    def _init_database(self):
        """Initialize the usage database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tts_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    characters INTEGER NOT NULL,
                    estimated_cost REAL NOT NULL,
                    voice TEXT NOT NULL,
                    format TEXT NOT NULL,
                    duration_seconds REAL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON tts_usage(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_provider ON tts_usage(provider)")
            
            # Create cost configuration table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS provider_costs (
                    provider TEXT PRIMARY KEY,
                    cost_per_1k_chars REAL NOT NULL,
                    free_tier_chars INTEGER NOT NULL,
                    notes TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def _load_custom_costs(self):
        """Load custom cost configurations from config"""
        # Check for custom costs in config
        custom_costs = get_cli_setting("tts_costs", "providers", {})
        
        with sqlite3.connect(self.db_path) as conn:
            for provider, cost_info in custom_costs.items():
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO provider_costs 
                        (provider, cost_per_1k_chars, free_tier_chars, notes)
                        VALUES (?, ?, ?, ?)
                    """, (
                        provider,
                        cost_info.get("cost_per_1k_chars", 0.0),
                        cost_info.get("free_tier_chars", 0),
                        cost_info.get("notes", "")
                    ))
                except Exception as e:
                    logger.error(f"Failed to load custom cost for {provider}: {e}")
            conn.commit()
    
    def track_usage(
        self,
        provider: str,
        model: str,
        text: str,
        voice: str,
        format: str,
        duration_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TTSUsageRecord:
        """
        Track a TTS generation.
        
        Args:
            provider: Provider name (e.g., "openai", "elevenlabs")
            model: Model used (e.g., "tts-1", "tts-1-hd")
            text: Text that was converted
            voice: Voice used
            format: Audio format
            duration_seconds: Duration of generated audio
            metadata: Additional metadata
            
        Returns:
            Usage record with cost estimate
        """
        characters = len(text)
        estimated_cost = self.estimate_cost(provider, model, characters)
        
        record = TTSUsageRecord(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            characters=characters,
            estimated_cost=estimated_cost,
            voice=voice,
            format=format,
            duration_seconds=duration_seconds,
            metadata=metadata
        )
        
        # Store in database
        with self._lock:
            self._store_record(record)
        
        return record
    
    def _store_record(self, record: TTSUsageRecord):
        """Store usage record in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO tts_usage 
                (timestamp, provider, model, characters, estimated_cost, 
                 voice, format, duration_seconds, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.timestamp.isoformat(),
                record.provider,
                record.model,
                record.characters,
                record.estimated_cost,
                record.voice,
                record.format,
                record.duration_seconds,
                json.dumps(record.metadata) if record.metadata else None
            ))
            conn.commit()
    
    def estimate_cost(self, provider: str, model: str, characters: int) -> float:
        """
        Estimate cost for TTS generation.
        
        Args:
            provider: Provider name
            model: Model name
            characters: Number of characters
            
        Returns:
            Estimated cost in USD
        """
        # Get cost info from database or defaults
        cost_info = self._get_cost_info(provider)
        
        if not cost_info:
            logger.warning(f"No cost info for provider {provider}")
            return 0.0
        
        # Special handling for different models
        cost_per_1k = cost_info.cost_per_1k_chars
        if provider == "openai" and model == "tts-1-hd":
            cost_per_1k = 0.030  # HD model costs more
        
        # Calculate cost
        cost = (characters / 1000.0) * cost_per_1k
        
        # Check free tier
        if cost_info.free_tier_chars > 0:
            # Get monthly usage so far
            monthly_chars = self.get_monthly_usage(provider)
            if monthly_chars < cost_info.free_tier_chars:
                # Still in free tier
                remaining_free = cost_info.free_tier_chars - monthly_chars
                if characters <= remaining_free:
                    return 0.0  # Fully covered by free tier
                else:
                    # Partially covered
                    billable_chars = characters - remaining_free
                    cost = (billable_chars / 1000.0) * cost_per_1k
        
        return round(cost, 4)
    
    def _get_cost_info(self, provider: str) -> Optional[ProviderCostInfo]:
        """Get cost information for a provider"""
        # First check database for custom costs
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT cost_per_1k_chars, free_tier_chars, notes
                FROM provider_costs
                WHERE provider = ?
            """, (provider,))
            row = cursor.fetchone()
            
            if row:
                return ProviderCostInfo(
                    provider=provider,
                    cost_per_1k_chars=row[0],
                    free_tier_chars=row[1],
                    notes=row[2] or ""
                )
        
        # Fall back to defaults
        for p in TTSProvider:
            if p.value == provider:
                return self.DEFAULT_COSTS.get(p)
        
        return None
    
    def get_monthly_usage(self, provider: Optional[str] = None) -> int:
        """
        Get total characters used this month.
        
        Args:
            provider: Optional provider filter
            
        Returns:
            Total characters used
        """
        start_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        with sqlite3.connect(self.db_path) as conn:
            if provider:
                cursor = conn.execute("""
                    SELECT SUM(characters) FROM tts_usage
                    WHERE timestamp >= ? AND provider = ?
                """, (start_of_month.isoformat(), provider))
            else:
                cursor = conn.execute("""
                    SELECT SUM(characters) FROM tts_usage
                    WHERE timestamp >= ?
                """, (start_of_month.isoformat(),))
            
            result = cursor.fetchone()[0]
            return result or 0
    
    def get_monthly_cost(self, provider: Optional[str] = None) -> float:
        """
        Get total estimated cost this month.
        
        Args:
            provider: Optional provider filter
            
        Returns:
            Total estimated cost in USD
        """
        start_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        with sqlite3.connect(self.db_path) as conn:
            if provider:
                cursor = conn.execute("""
                    SELECT SUM(estimated_cost) FROM tts_usage
                    WHERE timestamp >= ? AND provider = ?
                """, (start_of_month.isoformat(), provider))
            else:
                cursor = conn.execute("""
                    SELECT SUM(estimated_cost) FROM tts_usage
                    WHERE timestamp >= ?
                """, (start_of_month.isoformat(),))
            
            result = cursor.fetchone()[0]
            return result or 0.0
    
    def get_usage_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get usage statistics for a date range.
        
        Args:
            start_date: Start date (defaults to 30 days ago)
            end_date: End date (defaults to now)
            provider: Optional provider filter
            
        Returns:
            Dictionary with usage statistics
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            # Build query
            query_parts = ["WHERE timestamp >= ? AND timestamp <= ?"]
            params = [start_date.isoformat(), end_date.isoformat()]
            
            if provider:
                query_parts.append("AND provider = ?")
                params.append(provider)
            
            where_clause = " ".join(query_parts)
            
            # Get totals
            cursor = conn.execute(f"""
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(characters) as total_characters,
                    SUM(estimated_cost) as total_cost,
                    AVG(characters) as avg_characters,
                    AVG(estimated_cost) as avg_cost
                FROM tts_usage
                {where_clause}
            """, params)
            
            totals = cursor.fetchone()
            
            # Get per-provider breakdown
            cursor = conn.execute(f"""
                SELECT 
                    provider,
                    COUNT(*) as requests,
                    SUM(characters) as characters,
                    SUM(estimated_cost) as cost
                FROM tts_usage
                {where_clause}
                GROUP BY provider
                ORDER BY cost DESC
            """, params)
            
            provider_breakdown = [
                {
                    "provider": row[0],
                    "requests": row[1],
                    "characters": row[2],
                    "cost": row[3]
                }
                for row in cursor.fetchall()
            ]
            
            # Get per-model breakdown
            cursor = conn.execute(f"""
                SELECT 
                    provider,
                    model,
                    COUNT(*) as requests,
                    SUM(characters) as characters,
                    SUM(estimated_cost) as cost
                FROM tts_usage
                {where_clause}
                GROUP BY provider, model
                ORDER BY cost DESC
            """, params)
            
            model_breakdown = [
                {
                    "provider": row[0],
                    "model": row[1],
                    "requests": row[2],
                    "characters": row[3],
                    "cost": row[4]
                }
                for row in cursor.fetchall()
            ]
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "totals": {
                "requests": totals[0] or 0,
                "characters": totals[1] or 0,
                "cost": round(totals[2] or 0.0, 2),
                "avg_characters": round(totals[3] or 0.0, 1),
                "avg_cost": round(totals[4] or 0.0, 4)
            },
            "by_provider": provider_breakdown,
            "by_model": model_breakdown
        }
    
    def get_recent_usage(self, limit: int = 50) -> List[TTSUsageRecord]:
        """
        Get recent usage records.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of recent usage records
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp, provider, model, characters, estimated_cost,
                       voice, format, duration_seconds, metadata
                FROM tts_usage
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            records = []
            for row in cursor.fetchall():
                records.append(TTSUsageRecord(
                    timestamp=datetime.fromisoformat(row[0]),
                    provider=row[1],
                    model=row[2],
                    characters=row[3],
                    estimated_cost=row[4],
                    voice=row[5],
                    format=row[6],
                    duration_seconds=row[7],
                    metadata=json.loads(row[8]) if row[8] else None
                ))
            
            return records
    
    def update_cost_info(self, provider: str, cost_per_1k_chars: float, 
                        free_tier_chars: int = 0, notes: str = "") -> None:
        """
        Update cost information for a provider.
        
        Args:
            provider: Provider name
            cost_per_1k_chars: Cost per 1000 characters
            free_tier_chars: Free tier limit (characters/month)
            notes: Additional notes
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO provider_costs 
                (provider, cost_per_1k_chars, free_tier_chars, notes)
                VALUES (?, ?, ?, ?)
            """, (provider, cost_per_1k_chars, free_tier_chars, notes))
            conn.commit()
        
        logger.info(f"Updated cost info for {provider}")
    
    def export_usage_report(self, output_path: Path, format: str = "json") -> None:
        """
        Export usage report to file.
        
        Args:
            output_path: Path to save report
            format: Export format ("json" or "csv")
        """
        stats = self.get_usage_stats()
        
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(stats, f, indent=2, default=str)
        elif format == "csv":
            # Export as CSV
            import csv
            records = self.get_recent_usage(limit=None)
            
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "timestamp", "provider", "model", "characters",
                    "estimated_cost", "voice", "format", "duration_seconds"
                ])
                writer.writeheader()
                
                for record in records:
                    writer.writerow({
                        "timestamp": record.timestamp.isoformat(),
                        "provider": record.provider,
                        "model": record.model,
                        "characters": record.characters,
                        "estimated_cost": record.estimated_cost,
                        "voice": record.voice,
                        "format": record.format,
                        "duration_seconds": record.duration_seconds
                    })
        
        logger.info(f"Exported usage report to {output_path}")


# Global instance
_cost_tracker_instance: Optional[CostTracker] = None

def get_cost_tracker() -> CostTracker:
    """Get the global CostTracker instance"""
    global _cost_tracker_instance
    if _cost_tracker_instance is None:
        _cost_tracker_instance = CostTracker()
    return _cost_tracker_instance

#
# End of cost_tracker.py
#######################################################################################################################