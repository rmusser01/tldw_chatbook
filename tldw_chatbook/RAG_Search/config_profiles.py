"""
Configuration profiles for different RAG use cases.

This module provides preset configurations optimized for various scenarios,
along with utilities for experiment tracking and A/B testing.
"""

import json
import os
import re
import time
from typing import Dict, Any, Optional, List, Literal, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import hashlib
from loguru import logger

from .simplified.config import RAGConfig
from .reranker import RerankingConfig
from .parallel_processor import ProcessingConfig
from ..config import get_user_data_dir
from ..Metrics.metrics_logger import log_counter, log_histogram


def _slugify(name: str) -> str:
    """Filename-safe stable slug for a profile display name."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return slug or "profile"


# Filenames under profiles_dir that are never treated as user-profile files.
_RESERVED_PROFILE_FILES = {"custom_profiles.json", "custom_profiles.json.migrated"}


# Profile types
ProfileType = Literal[
    "fast_search",  # Optimized for speed
    "high_accuracy",  # Optimized for accuracy
    "balanced",  # Balance between speed and accuracy
    "long_context",  # For documents requiring large context
    "technical_docs",  # For technical documentation
    "conversational",  # For chat/conversation data
    "research_papers",  # For academic papers
    "code_search",  # For code repositories
    "multimodal",  # For mixed text/image content
    "custom",  # User-defined configuration
]


@dataclass
class ExperimentConfig:
    """Configuration for A/B testing and experiment tracking."""

    experiment_id: str = field(default_factory=lambda: f"exp_{int(time.time())}")
    name: str = "Unnamed Experiment"
    description: str = ""

    # A/B test settings
    enable_ab_testing: bool = False
    control_profile: str = "balanced"
    test_profiles: List[str] = field(default_factory=list)
    traffic_split: Dict[str, float] = field(
        default_factory=dict
    )  # Profile -> percentage

    # Tracking settings
    track_metrics: bool = True
    metrics_to_track: List[str] = field(
        default_factory=lambda: [
            "search_latency",
            "retrieval_accuracy",
            "reranking_improvement",
            "user_satisfaction",
            "context_quality",
        ]
    )

    # Persistence
    save_results: bool = True
    results_dir: Optional[Path] = None


@dataclass
class ProfileConfig:
    """Complete configuration profile including all components."""

    name: str
    description: str
    profile_type: ProfileType

    # Component configurations
    rag_config: RAGConfig
    reranking_config: Optional[RerankingConfig] = None
    processing_config: Optional[ProcessingConfig] = None

    # Profile metadata
    created_at: float = field(default_factory=time.time)
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)

    # Performance expectations
    expected_latency_ms: Optional[float] = None
    expected_accuracy: Optional[float] = None

    # Stable identity / mutability marker
    id: Optional[str] = None
    read_only: bool = False

    def __post_init__(self):
        if not self.id:
            self.id = _slugify(self.name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "id": self.id,
            "read_only": self.read_only,
            "name": self.name,
            "description": self.description,
            "profile_type": self.profile_type,
            "rag_config": asdict(self.rag_config),
            "reranking_config": asdict(self.reranking_config)
            if self.reranking_config
            else None,
            "processing_config": asdict(self.processing_config)
            if self.processing_config
            else None,
            "created_at": self.created_at,
            "version": self.version,
            "tags": list(self.tags),
            "expected_latency_ms": self.expected_latency_ms,
            "expected_accuracy": self.expected_accuracy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProfileConfig":
        """Create profile from dictionary."""
        rag_config = (
            RAGConfig.from_dict(data["rag_config"])
            if isinstance(data["rag_config"], dict)
            else data["rag_config"]
        )

        reranking_config = None
        if data.get("reranking_config"):
            reranking_config = (
                RerankingConfig(**data["reranking_config"])
                if isinstance(data["reranking_config"], dict)
                else data["reranking_config"]
            )

        processing_config = None
        if data.get("processing_config"):
            processing_config = (
                ProcessingConfig(**data["processing_config"])
                if isinstance(data["processing_config"], dict)
                else data["processing_config"]
            )

        return cls(
            name=data["name"],
            description=data["description"],
            profile_type=data["profile_type"],
            rag_config=rag_config,
            reranking_config=reranking_config,
            processing_config=processing_config,
            created_at=data.get("created_at", time.time()),
            version=data.get("version", "1.0"),
            tags=data.get("tags", []),
            expected_latency_ms=data.get("expected_latency_ms"),
            expected_accuracy=data.get("expected_accuracy"),
            id=data.get("id"),
            read_only=data.get("read_only", False),
        )


class ConfigProfileManager:
    """Manages RAG configuration profiles.

    Builtins are read-only clone-seeds; user profiles are stored one JSON file
    per profile under ``rag_profiles/`` keyed by a stable ``id`` (rename-safe).
    NOTE: the active-profile pointer and config-resolution wiring live in SP2b,
    not here — this class is storage + CRUD only.
    """

    def __init__(self, profiles_dir: Optional[Path] = None):
        self.profiles_dir = profiles_dir or (get_user_data_dir() / "rag_profiles")
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

        self._profiles: Dict[str, ProfileConfig] = {}
        self._current_experiment: Optional[ExperimentConfig] = None
        self._experiment_results: Dict[str, List[Dict[str, Any]]] = {}

        # Load built-in profiles
        self._load_builtin_profiles()

        # Immutable, un-flippable record of which ids are builtins. Mutation
        # guards (save/delete/rename) check THIS set, not the mutable
        # `ProfileConfig.read_only` field on a shared instance -- a caller
        # cannot bypass the guard by doing
        # `manager.get_profile("x").read_only = False`.
        self._builtin_ids = frozenset(self._profiles.keys())

        # Load custom profiles
        self._load_custom_profiles()

    def _load_builtin_profiles(self):
        """Load predefined configuration profiles."""

        # BM25 Only Profile (Pure keyword search)
        bm25_rag = RAGConfig()
        bm25_rag.vector_store.type = "memory"  # No vector DB needed
        bm25_rag.embedding.model = "all-MiniLM-L6-v2"  # Still needed for interface
        bm25_rag.chunking.chunk_size = 512
        bm25_rag.chunking.chunk_overlap = 64
        bm25_rag.search.default_search_mode = "plain"
        bm25_rag.search.default_top_k = 20
        bm25_rag.search.include_citations = True

        self._profiles["bm25_only"] = ProfileConfig(
            id="bm25_only",
            read_only=True,
            name="BM25 Only",
            description="Pure keyword/BM25 search without semantic vectors",
            profile_type="fast_search",
            rag_config=bm25_rag,
            expected_latency_ms=50,
            expected_accuracy=0.75,
            tags=["keyword", "bm25", "fast"],
        )

        # Vector Only Profile (Pure semantic search)
        vector_rag = RAGConfig()
        vector_rag.embedding.model = "sentence-transformers/all-mpnet-base-v2"
        vector_rag.embedding.batch_size = 32
        vector_rag.chunking.chunk_size = 384
        vector_rag.chunking.chunk_overlap = 64
        vector_rag.search.default_search_mode = "semantic"
        vector_rag.search.default_top_k = 10
        vector_rag.search.include_citations = True

        self._profiles["vector_only"] = ProfileConfig(
            id="vector_only",
            read_only=True,
            name="Vector Only",
            description="Pure semantic/vector search without keyword matching",
            profile_type="fast_search",
            rag_config=vector_rag,
            expected_latency_ms=150,
            expected_accuracy=0.85,
            tags=["semantic", "vector", "embedding"],
        )

        # Hybrid Basic Profile (No enhanced features)
        hybrid_basic_rag = RAGConfig()
        hybrid_basic_rag.embedding.model = "all-MiniLM-L6-v2"
        hybrid_basic_rag.embedding.batch_size = 32
        hybrid_basic_rag.chunking.chunk_size = 384
        hybrid_basic_rag.chunking.chunk_overlap = 64
        hybrid_basic_rag.search.default_search_mode = "hybrid"
        hybrid_basic_rag.search.default_top_k = 15
        hybrid_basic_rag.search.include_citations = True

        self._profiles["hybrid_basic"] = ProfileConfig(
            id="hybrid_basic",
            read_only=True,
            name="Hybrid Basic",
            description="Combined keyword and semantic search without enhancements",
            profile_type="balanced",
            rag_config=hybrid_basic_rag,
            expected_latency_ms=200,
            expected_accuracy=0.88,
            tags=["hybrid", "basic", "balanced"],
        )

        # Hybrid Enhanced Profile (With parent retrieval)
        hybrid_enhanced_rag = RAGConfig()
        hybrid_enhanced_rag.embedding.model = "sentence-transformers/all-mpnet-base-v2"
        hybrid_enhanced_rag.embedding.batch_size = 32
        hybrid_enhanced_rag.chunking.chunk_size = 384
        hybrid_enhanced_rag.chunking.chunk_overlap = 64
        hybrid_enhanced_rag.chunking.enable_parent_retrieval = True
        hybrid_enhanced_rag.chunking.parent_size_multiplier = 3
        hybrid_enhanced_rag.search.default_search_mode = "hybrid"
        hybrid_enhanced_rag.search.default_top_k = 10
        hybrid_enhanced_rag.search.include_citations = True
        hybrid_enhanced_rag.search.include_parent_docs = True
        hybrid_enhanced_rag.search.parent_size_threshold = 5000
        hybrid_enhanced_rag.search.parent_inclusion_strategy = "size_based"

        self._profiles["hybrid_enhanced"] = ProfileConfig(
            id="hybrid_enhanced",
            read_only=True,
            name="Hybrid Enhanced",
            description="Hybrid search with parent document retrieval",
            profile_type="balanced",
            rag_config=hybrid_enhanced_rag,
            expected_latency_ms=250,
            expected_accuracy=0.92,
            tags=["hybrid", "enhanced", "parent-retrieval"],
        )

        # Hybrid Full Profile (All features)
        hybrid_full_rag = RAGConfig()
        hybrid_full_rag.embedding.model = "BAAI/bge-base-en-v1.5"
        hybrid_full_rag.embedding.batch_size = 32
        hybrid_full_rag.chunking.chunk_size = 512
        hybrid_full_rag.chunking.chunk_overlap = 128
        hybrid_full_rag.chunking.chunking_method = "hierarchical"
        hybrid_full_rag.chunking.enable_parent_retrieval = True
        hybrid_full_rag.chunking.parent_size_multiplier = 3
        hybrid_full_rag.chunking.clean_artifacts = True
        hybrid_full_rag.chunking.preserve_structure = True
        hybrid_full_rag.search.default_search_mode = "hybrid"
        hybrid_full_rag.search.default_top_k = 20
        hybrid_full_rag.search.include_citations = True
        hybrid_full_rag.search.include_parent_docs = True
        hybrid_full_rag.search.parent_size_threshold = 8000
        hybrid_full_rag.search.parent_inclusion_strategy = "size_based"
        hybrid_full_rag.search.max_context_size = 32000

        hybrid_full_rerank = RerankingConfig(
            strategy="cross_encoder", top_k_to_rerank=15, include_reasoning=False
        )

        self._profiles["hybrid_full"] = ProfileConfig(
            id="hybrid_full",
            read_only=True,
            name="Hybrid Full",
            description="All features enabled for maximum accuracy",
            profile_type="high_accuracy",
            rag_config=hybrid_full_rag,
            reranking_config=hybrid_full_rerank,
            processing_config=ProcessingConfig(batch_size=32, num_workers=4),
            expected_latency_ms=400,
            expected_accuracy=0.95,
            tags=["hybrid", "full", "maximum"],
        )

        # Fast Search Profile (Legacy, kept for compatibility)
        fast_rag = RAGConfig()
        fast_rag.embedding.model = "all-MiniLM-L6-v2"
        fast_rag.embedding.batch_size = 64
        fast_rag.chunking.chunk_size = 256
        fast_rag.chunking.chunk_overlap = 32
        fast_rag.search.default_top_k = 5
        fast_rag.search.enable_cache = True
        fast_rag.search.cache_size = 200

        self._profiles["fast_search"] = ProfileConfig(
            id="fast_search",
            read_only=True,
            name="Fast Search",
            description="Optimized for low latency with acceptable accuracy",
            profile_type="fast_search",
            rag_config=fast_rag,
            processing_config=ProcessingConfig(batch_size=64, num_workers=None),
            expected_latency_ms=100,
            expected_accuracy=0.8,
            tags=["speed", "low-latency", "legacy"],
        )

        # High Accuracy Profile
        accurate_rag = RAGConfig()
        accurate_rag.embedding.model = "BAAI/bge-large-en-v1.5"  # Larger, more accurate
        accurate_rag.embedding.batch_size = 16
        accurate_rag.chunking.chunk_size = 512
        accurate_rag.chunking.chunk_overlap = 128
        accurate_rag.search.default_top_k = 20
        accurate_rag.search.include_citations = True
        accurate_rag.search.score_threshold = 0.7

        accurate_rerank = RerankingConfig(
            model_provider="openai",
            model_name="gpt-3.5-turbo",
            strategy="pointwise",
            top_k_to_rerank=15,
            include_reasoning=True,
        )

        self._profiles["high_accuracy"] = ProfileConfig(
            id="high_accuracy",
            read_only=True,
            name="High Accuracy",
            description="Optimized for maximum retrieval accuracy",
            profile_type="high_accuracy",
            rag_config=accurate_rag,
            reranking_config=accurate_rerank,
            processing_config=ProcessingConfig(batch_size=16, show_progress=True),
            expected_latency_ms=500,
            expected_accuracy=0.95,
            tags=["accuracy", "quality"],
        )

        # Balanced Profile
        balanced_rag = RAGConfig()
        balanced_rag.embedding.model = "sentence-transformers/all-mpnet-base-v2"
        balanced_rag.chunking.chunk_size = 384
        balanced_rag.chunking.chunk_overlap = 64
        balanced_rag.search.default_top_k = 10

        self._profiles["balanced"] = ProfileConfig(
            id="balanced",
            read_only=True,
            name="Balanced",
            description="Balance between speed and accuracy",
            profile_type="balanced",
            rag_config=balanced_rag,
            processing_config=ProcessingConfig(batch_size=32),
            expected_latency_ms=250,
            expected_accuracy=0.88,
            tags=["balanced", "general-purpose"],
        )

        # Long Context Profile
        long_context_rag = RAGConfig()
        long_context_rag.embedding.model = "BAAI/bge-base-en-v1.5"
        long_context_rag.chunking.chunk_size = 1024  # Larger chunks
        long_context_rag.chunking.chunk_overlap = 256
        long_context_rag.chunking.enable_parent_retrieval = True
        long_context_rag.chunking.parent_size_multiplier = 3
        long_context_rag.search.default_top_k = 5  # Fewer but larger chunks

        self._profiles["long_context"] = ProfileConfig(
            id="long_context",
            read_only=True,
            name="Long Context",
            description="For documents requiring extended context",
            profile_type="long_context",
            rag_config=long_context_rag,
            expected_latency_ms=300,
            expected_accuracy=0.9,
            tags=["context", "long-form"],
        )

        # Technical Documentation Profile
        tech_rag = RAGConfig()
        tech_rag.embedding.model = "sentence-transformers/all-mpnet-base-v2"
        tech_rag.chunking.chunk_size = 512
        tech_rag.chunking.chunking_method = "structural"  # Preserve document structure
        tech_rag.chunking.clean_artifacts = True
        tech_rag.chunking.preserve_tables = True
        tech_rag.chunking.preserve_structure = True
        tech_rag.search.include_citations = True

        self._profiles["technical_docs"] = ProfileConfig(
            id="technical_docs",
            read_only=True,
            name="Technical Documentation",
            description="Optimized for technical content with tables and code",
            profile_type="technical_docs",
            rag_config=tech_rag,
            expected_accuracy=0.92,
            tags=["technical", "documentation"],
        )

        # Research Papers Profile
        research_rag = RAGConfig()
        research_rag.embedding.model = (
            "allenai-specter"  # Specialized for scientific text
        )
        research_rag.chunking.chunk_size = 512
        research_rag.chunking.chunking_method = "hierarchical"
        research_rag.chunking.clean_artifacts = True  # Clean PDF artifacts
        research_rag.chunking.preserve_structure = True
        research_rag.search.include_citations = True
        research_rag.search.default_top_k = 15
        research_rag.search.include_parent_docs = True
        research_rag.search.parent_size_threshold = 10000  # Larger for research papers
        research_rag.search.parent_inclusion_strategy = "size_based"

        research_rerank = RerankingConfig(
            strategy="listwise", top_k_to_rerank=10, include_reasoning=True
        )

        self._profiles["research_papers"] = ProfileConfig(
            id="research_papers",
            read_only=True,
            name="Research Papers",
            description="Optimized for academic papers and citations",
            profile_type="research_papers",
            rag_config=research_rag,
            reranking_config=research_rerank,
            expected_accuracy=0.94,
            tags=["academic", "research", "citations"],
        )

        # Code Search Profile
        code_rag = RAGConfig()
        code_rag.embedding.model = "microsoft/codebert-base"  # Code-specific embeddings
        code_rag.chunking.chunk_size = 256  # Smaller chunks for code
        code_rag.chunking.chunking_method = "words"  # Preserve code structure
        code_rag.search.default_top_k = 20  # More results for code search

        self._profiles["code_search"] = ProfileConfig(
            id="code_search",
            read_only=True,
            name="Code Search",
            description="Optimized for searching code repositories",
            profile_type="code_search",
            rag_config=code_rag,
            expected_accuracy=0.85,
            tags=["code", "programming"],
        )

        logger.info(f"Loaded {len(self._profiles)} built-in profiles")

    def _profile_path(self, profile_id: str) -> Path:
        """Path to a user profile's own JSON file.

        Args:
            profile_id: Candidate profile id. Must already be a bare,
                filesystem-safe slug (i.e. equal to ``_slugify(profile_id)``).
                This is defense-in-depth against a crafted or hand-edited
                profile id (e.g. ``"../../etc/passwd"``) escaping
                ``profiles_dir`` via any id-keyed file operation
                (``_save_one``, ``delete_profile``, legacy-blob migration).

        Returns:
            The per-profile JSON file path under ``profiles_dir``.

        Raises:
            ValueError: If ``profile_id`` is not already a bare slug.
        """
        if profile_id != _slugify(profile_id):
            raise ValueError(f"unsafe profile id: {profile_id!r}")
        return self.profiles_dir / f"{profile_id}.json"

    def _save_one(self, profile: "ProfileConfig") -> None:
        """Write a single user profile to its own file (never builtins)."""
        if profile.read_only:
            return
        with open(self._profile_path(profile.id), "w") as f:
            json.dump(profile.to_dict(), f, indent=2, default=str)

    def _load_custom_profiles(self):
        """Load user profiles from per-file JSON; migrate a legacy blob once.

        The FILENAME (stem) is authoritative for a profile's id -- never the
        JSON's internal ``"id"`` field, which may be missing, hand-edited,
        stale, or (if untrusted) even crafted for path traversal. If the
        stem-derived id collides with an already-registered profile (a
        builtin, or a profile loaded earlier in this same pass), it is
        reassigned to a unique id. Whenever the canonical filename for the
        resolved id doesn't match the file it was loaded from -- because the
        stem needed slugifying, or the id was reassigned -- the profile is
        self-healed on disk: written under its canonical name first, then
        the stale file is removed, so a crash between the two steps never
        loses data.
        """
        self._migrate_legacy_blob()
        for path in self.profiles_dir.glob("*.json"):
            if path.name in _RESERVED_PROFILE_FILES:
                continue
            try:
                with open(path, "r") as f:
                    profile = ProfileConfig.from_dict(json.load(f))
                profile.read_only = False

                desired_id = _slugify(path.stem)
                if desired_id in self._profiles:
                    # Collides with a builtin or an already-loaded profile
                    # (e.g. a hand-placed <builtin_id>.json, or two files
                    # whose stems slugify to the same id). Never let it win:
                    # reassign to a unique id, checking disk too, since the
                    # in-memory dict is only partially populated mid-glob.
                    old_id = desired_id
                    desired_id = self._unique_id_reserving_disk(desired_id)
                    logger.warning(
                        f"Profile file {path.name} collided with id "
                        f"'{old_id}'; reassigned to id '{desired_id}'"
                    )

                profile.id = desired_id
                self._profiles[desired_id] = profile

                canonical_path = self._profile_path(desired_id)
                if canonical_path != path:
                    # Self-heal: write under the canonical name FIRST, then
                    # remove the stale/incorrectly-named file.
                    self._save_one(profile)
                    try:
                        if path.exists():
                            path.unlink()
                    except OSError as e:
                        logger.warning(
                            f"Failed to remove stale profile file {path}: {e}"
                        )
                    logger.warning(
                        f"Profile file {path.name} did not match its "
                        f"canonical id '{desired_id}'; self-healed to "
                        f"{canonical_path.name}"
                    )
            except Exception as e:
                logger.error(f"Failed to load profile {path.name}: {e}")

    def _migrate_legacy_blob(self):
        """One-time split of the old custom_profiles.json blob into per-file.

        Each profile entry is migrated independently: a single bad or
        unparseable entry (e.g. an old blob with a renamed/unknown
        ``rag_config`` sub-key, raising ``TypeError`` from
        ``ProfileConfig.from_dict``) is logged and skipped rather than
        aborting the whole migration -- otherwise one corrupt entry would
        silently take every OTHER valid custom profile in the blob down with
        it. The blob is renamed to ``.migrated`` unconditionally once it has
        been successfully read, so it is never reprocessed on a later boot
        even if every entry in it failed to migrate.
        """
        blob = self.profiles_dir / "custom_profiles.json"
        if not blob.exists():
            return
        try:
            with open(blob, "r") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Legacy profile blob migration failed to read {blob}: {e}")
            return

        for pdata in data.get("profiles", []):
            try:
                profile = ProfileConfig.from_dict(pdata)
                profile.read_only = False
                existing = self._profiles.get(profile.id)
                if existing is not None and existing.read_only:
                    # Never let a migrated user profile shadow a read-only
                    # builtin id -- write it to a uniquified file instead.
                    profile.id = self._unique_id_reserving_disk(profile.id)
                target = self._profile_path(profile.id)
                if not target.exists():  # never clobber an existing per-file profile
                    with open(target, "w") as out:
                        json.dump(profile.to_dict(), out, indent=2, default=str)
            except Exception as e:
                entry_name = pdata.get("name", "<unknown>") if isinstance(pdata, dict) else "<unknown>"
                logger.error(
                    f"Skipping unmigratable legacy profile entry '{entry_name}': {e}"
                )

        try:
            # os.replace is an atomic overwrite on both POSIX and Windows
            # (Path.rename raises on Windows if the destination exists).
            os.replace(str(blob), str(blob.parent / "custom_profiles.json.migrated"))
            logger.info("Migrated legacy custom_profiles.json to per-file profiles")
        except OSError as e:
            logger.error(f"Failed to rename migrated legacy blob {blob}: {e}")

    def get_profile(self, name: str) -> Optional[ProfileConfig]:
        """Get a configuration profile by name."""
        return self._profiles.get(name)

    def list_profiles(self) -> List[str]:
        """List all available profile names."""
        return list(self._profiles.keys())

    def _unique_id(self, base_slug: str) -> str:
        candidate, n = base_slug, 2
        while candidate in self._profiles:
            candidate = f"{base_slug}_{n}"
            n += 1
        return candidate

    def _unique_id_reserving_disk(self, base_slug: str) -> str:
        """Like _unique_id but also avoids ids whose file already exists on disk
        (the in-memory dict is only partially populated during load)."""
        candidate, n = base_slug, 2
        while candidate in self._profiles or self._profile_path(candidate).exists():
            candidate = f"{base_slug}_{n}"
            n += 1
        return candidate

    def save_profile(self, profile: "ProfileConfig") -> "ProfileConfig":
        """Persist a user profile.

        Refuses to persist a read-only builtin, whether the incoming
        ``profile`` object is itself read-only or its id merely collides
        with an existing read-only builtin.

        Args:
            profile: The profile to save. Its ``id`` determines both the
                in-memory key and the on-disk filename it is written to.

        Returns:
            The same ``profile`` instance, now registered and persisted.

        Raises:
            ValueError: If ``profile.read_only`` is ``True``, or
                ``profile.id`` names a builtin (checked against the
                immutable ``self._builtin_ids`` set, which cannot be
                defeated by flipping ``read_only`` on a shared instance), or
                ``profile.id`` collides with a different, read-only,
                already-registered profile.
        """
        if profile.read_only or profile.id in self._builtin_ids:
            raise ValueError(f"Profile '{profile.id}' is read-only")
        existing = self._profiles.get(profile.id)
        if existing is not None and existing is not profile and existing.read_only:
            raise ValueError(
                f"Profile id '{profile.id}' collides with read-only builtin '{existing.name}'"
            )
        self._profiles[profile.id] = profile
        self._save_one(profile)
        return profile

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a user profile by id, removing both its in-memory entry
        and its on-disk file.

        Args:
            profile_id: The id of the profile to delete.

        Returns:
            ``True`` if a profile was found and deleted, ``False`` if no
            profile with that id is registered.

        Raises:
            ValueError: If ``profile_id`` names a builtin profile. Checked
                against both ``self._builtin_ids`` (immutable, un-flippable)
                and the profile's own ``read_only`` flag.
        """
        if profile_id in self._builtin_ids:
            raise ValueError(f"Builtin profile '{profile_id}' cannot be deleted")
        prof = self._profiles.get(profile_id)
        if prof is None:
            return False
        if prof.read_only:
            raise ValueError(f"Builtin profile '{profile_id}' cannot be deleted")
        self._profiles.pop(profile_id, None)
        path = self._profile_path(profile_id)
        if path.exists():
            path.unlink()
        return True

    def rename_profile(self, profile_id: str, new_name: str) -> "ProfileConfig":
        """Rename a user profile's display name in place.

        The profile's ``id`` (and therefore its on-disk filename) is left
        unchanged -- only ``name`` is updated -- so the profile remains
        reachable under the same id before and after the rename.

        Args:
            profile_id: The id of the profile to rename.
            new_name: The new display name.

        Returns:
            The renamed profile.

        Raises:
            ValueError: If no profile with ``profile_id`` is registered, or
                it names a builtin profile (checked against both
                ``self._builtin_ids`` and ``read_only``).
        """
        if profile_id in self._builtin_ids:
            raise ValueError(f"Builtin profile '{profile_id}' cannot be renamed")
        prof = self._profiles.get(profile_id)
        if prof is None:
            raise ValueError(f"Profile '{profile_id}' not found")
        if prof.read_only:
            raise ValueError(f"Builtin profile '{profile_id}' cannot be renamed")
        prof.name = new_name  # id + filename stay the same (rename-safe)
        self._save_one(prof)
        return prof

    def clone_profile(self, source_id: str, new_name: str) -> "ProfileConfig":
        """Clone an existing profile (builtin or user) into a new, writable profile.

        Args:
            source_id: Id of the profile to clone from.
            new_name: Display name for the clone. Its id is derived from
                this name via ``_slugify`` and uniquified against existing
                ids, so it never collides with the source (or any other
                profile) even if the display name matches.

        Returns:
            The newly created, persisted, writable clone.

        Raises:
            ValueError: If ``source_id`` does not resolve to a registered
                profile.
        """
        src = self._profiles.get(source_id)
        if src is None:
            raise ValueError(f"Source profile '{source_id}' not found")
        new_id = self._unique_id(_slugify(new_name))
        clone = ProfileConfig.from_dict({
            **src.to_dict(),
            "id": new_id,
            "name": new_name,
            "read_only": False,
            "profile_type": "custom",
        })
        return self.save_profile(clone)

    def create_custom_profile(
        self, name: str, base_profile: str = "balanced", **overrides
    ) -> ProfileConfig:
        """
        Create a custom profile based on an existing one.

        Args:
            name: Name for the custom profile
            base_profile: Base profile to start from
            **overrides: Configuration overrides

        Returns:
            New custom profile
        """
        base = self.get_profile(base_profile)
        if not base:
            raise ValueError(f"Base profile '{base_profile}' not found")

        # Create a copy
        custom_config = ProfileConfig(
            name=name,
            description=f"Custom profile based on {base_profile}",
            profile_type="custom",
            rag_config=RAGConfig.from_dict(asdict(base.rag_config)),
            reranking_config=RerankingConfig(**asdict(base.reranking_config))
            if base.reranking_config
            else None,
            processing_config=ProcessingConfig(**asdict(base.processing_config))
            if base.processing_config
            else None,
            tags=["custom"] + base.tags,
        )

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(custom_config, key):
                setattr(custom_config, key, value)
            elif hasattr(custom_config.rag_config, key):
                setattr(custom_config.rag_config, key, value)
            # Add more override logic as needed

        # Uniquify the id up front so a user-chosen name that happens to match
        # a builtin's display name (e.g. "High Accuracy" -> "high_accuracy")
        # doesn't collide with the read-only builtin's id.
        custom_config.id = self._unique_id(_slugify(name))

        # Save the custom profile (keyed by profile.id, same as its filename,
        # so it's reachable by the same key before and after a restart)
        self.save_profile(custom_config)

        logger.info(f"Created custom profile: {name}")

        return custom_config

    def start_experiment(self, config: ExperimentConfig):
        """Start an A/B testing experiment."""
        self._current_experiment = config
        self._experiment_results[config.experiment_id] = []

        if not config.results_dir:
            config.results_dir = (
                self.profiles_dir / "experiments" / config.experiment_id
            )

        config.results_dir.mkdir(parents=True, exist_ok=True)

        # Save experiment configuration
        with open(config.results_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)

        logger.info(f"Started experiment: {config.name} (ID: {config.experiment_id})")

        log_counter(
            "rag_experiment_started", labels={"experiment_id": config.experiment_id}
        )

    def select_profile_for_experiment(
        self, user_id: Optional[str] = None
    ) -> Tuple[str, ProfileConfig]:
        """
        Select a profile for the current experiment based on traffic split.

        Args:
            user_id: Optional user ID for consistent assignment

        Returns:
            Tuple of (profile_name, profile_config)
        """
        if (
            not self._current_experiment
            or not self._current_experiment.enable_ab_testing
        ):
            # No experiment running, use default
            return "balanced", self.get_profile("balanced")

        # Ensure consistent assignment for user
        if user_id:
            hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            rand_val = (hash_val % 100) / 100.0
        else:
            import random

            rand_val = random.random()

        # Select profile based on traffic split
        cumulative = 0.0
        for profile_name, percentage in self._current_experiment.traffic_split.items():
            cumulative += percentage
            if rand_val < cumulative:
                profile = self.get_profile(profile_name)
                if profile:
                    log_counter(
                        "rag_experiment_profile_selected",
                        labels={
                            "experiment_id": self._current_experiment.experiment_id,
                            "profile": profile_name,
                        },
                    )
                    return profile_name, profile

        # Fallback to control
        return self._current_experiment.control_profile, self.get_profile(
            self._current_experiment.control_profile
        )

    def record_experiment_result(
        self, profile_name: str, query: str, metrics: Dict[str, Any]
    ):
        """Record results from an experiment run."""
        if not self._current_experiment or not self._current_experiment.track_metrics:
            return

        result = {
            "timestamp": time.time(),
            "profile": profile_name,
            "query": query,
            "metrics": metrics,
        }

        self._experiment_results[self._current_experiment.experiment_id].append(result)

        # Log metrics
        for metric_name, value in metrics.items():
            if metric_name in self._current_experiment.metrics_to_track:
                log_histogram(
                    f"rag_experiment_{metric_name}",
                    value,
                    labels={
                        "experiment_id": self._current_experiment.experiment_id,
                        "profile": profile_name,
                    },
                )

    def end_experiment(self) -> Dict[str, Any]:
        """End the current experiment and return results summary."""
        if not self._current_experiment:
            return {}

        exp_id = self._current_experiment.experiment_id
        results = self._experiment_results.get(exp_id, [])

        # Calculate summary statistics
        summary = {
            "experiment_id": exp_id,
            "name": self._current_experiment.name,
            "total_queries": len(results),
            "profiles": {},
        }

        # Group by profile
        profile_results = {}
        for result in results:
            profile = result["profile"]
            if profile not in profile_results:
                profile_results[profile] = []
            profile_results[profile].append(result["metrics"])

        # Calculate statistics for each profile
        for profile, metrics_list in profile_results.items():
            profile_stats = {}

            # Calculate mean for each metric
            for metric_name in self._current_experiment.metrics_to_track:
                values = [
                    m.get(metric_name, 0) for m in metrics_list if metric_name in m
                ]
                if values:
                    profile_stats[metric_name] = {
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values),
                    }

            summary["profiles"][profile] = {
                "query_count": len(metrics_list),
                "metrics": profile_stats,
            }

        # Save results if configured
        if self._current_experiment.save_results:
            results_file = self._current_experiment.results_dir / "results.json"
            with open(results_file, "w") as f:
                json.dump(
                    {
                        "summary": summary,
                        "detailed_results": results,
                        "completed_at": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )

            logger.info(f"Saved experiment results to {results_file}")

        # Clear experiment
        self._current_experiment = None

        log_counter("rag_experiment_completed", labels={"experiment_id": exp_id})

        return summary

    def validate_profile(self, profile: ProfileConfig) -> List[str]:
        """
        Validate a configuration profile for compatibility and correctness.

        Returns:
            List of validation warnings/errors (empty if valid)
        """
        warnings = []

        # Check embedding model availability
        if profile.rag_config.embedding.model not in self._get_available_models():
            warnings.append(
                f"Embedding model '{profile.rag_config.embedding.model}' may not be available"
            )

        # Check chunk size vs overlap
        if profile.rag_config.chunking.chunk_overlap >= profile.rag_config.chunking.chunk_size:
            warnings.append("Chunk overlap should be less than chunk size")

        # Check reranking configuration
        if profile.reranking_config:
            if (
                profile.reranking_config.top_k_to_rerank
                > profile.rag_config.search.default_top_k
            ):
                warnings.append("Reranking top_k should not exceed search top_k")

        # Check processing configuration
        if profile.processing_config:
            if (
                profile.processing_config.num_workers
                and profile.processing_config.num_workers > 32
            ):
                warnings.append("Using more than 32 workers may cause resource issues")

        return warnings

    def _get_available_models(self) -> List[str]:
        """Get list of known available embedding models."""
        # This is a simplified list - in practice, you'd check actual availability
        return [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "BAAI/bge-base-en-v1.5",
            "BAAI/bge-large-en-v1.5",
            "sentence-transformers/all-mpnet-base-v2",
            "allenai-specter",
            "microsoft/codebert-base",
        ]


# Convenience functions


def get_profile_manager(profiles_dir: Optional[Path] = None) -> ConfigProfileManager:
    """Get or create the global profile manager."""
    return ConfigProfileManager(profiles_dir)


def quick_profile(use_case: ProfileType) -> ProfileConfig:
    """Get a profile for a specific use case."""
    manager = get_profile_manager()
    profile = manager.get_profile(use_case)

    if not profile:
        # Fall back to balanced
        logger.warning(f"Profile '{use_case}' not found, using 'balanced'")
        profile = manager.get_profile("balanced")

    return profile
