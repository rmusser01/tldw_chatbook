# RAG System Improvements Plan

This document outlines the implementation plan for four major improvements to the tldw_chatbook RAG (Retrieval-Augmented Generation) system:

1. Multi-modal RAG support
2. Custom pipeline configurations
3. A/B testing capabilities
4. Comprehensive RAG metrics

## Table of Contents

- [Current State Analysis](#current-state-analysis)
- [1. Multi-modal RAG](#1-multi-modal-rag)
- [2. Custom Pipeline Configurations](#2-custom-pipeline-configurations)
- [3. A/B Testing Framework](#3-ab-testing-framework)
- [4. RAG Metrics System](#4-rag-metrics-system)
- [Implementation Roadmap](#implementation-roadmap)
- [Performance Considerations](#performance-considerations)

## Current State Analysis

### Existing Infrastructure

The tldw_chatbook RAG system currently has:

- **Text-only embeddings** using sentence-transformers
- **Modular architecture** with clear separation of retrievers, processors, and generators
- **TOML-based configuration** with comprehensive settings
- **Basic metrics collection** via MetricsCollector class
- **Multiple data sources**: Media DB, Chat History, Notes, Character Cards
- **Dual implementation**: Legacy (default) and new modular system (opt-in)

### Architecture Overview

```
tldw_chatbook/
├── RAG_Search/
│   └── Services/
│       ├── rag_service/         # New modular implementation
│       │   ├── app.py          # Orchestrator
│       │   ├── retrieval.py    # Retrieval strategies
│       │   ├── processing.py   # Document processing
│       │   ├── generation.py   # Response generation
│       │   └── types.py        # Core data types
│       └── embeddings_service.py
├── Embeddings/
│   ├── Embeddings_Lib.py       # Embedding models
│   └── Chroma_Lib.py           # Vector storage
└── Event_Handlers/
    └── Chat_Events/
        └── chat_rag_events.py  # Legacy implementation
```

## 1. Multi-modal RAG

### Overview

Extend the RAG system to support images, audio, and video content beyond just text transcriptions.

### Implementation Plan

#### 1.1 Update Core Data Types

```python
# In tldw_chatbook/RAG_Search/Services/rag_service/types.py

from enum import Enum, auto
from typing import Union, Optional
import numpy as np

class ModalityType(Enum):
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()
    MULTIMODAL = auto()  # Combined modalities

@dataclass
class MultiModalContent:
    """Container for multi-modal content."""
    text: Optional[str] = None
    image_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    video_path: Optional[Path] = None
    video_frame_timestamps: Optional[List[float]] = None
    
    @property
    def modalities(self) -> List[ModalityType]:
        """Get list of present modalities."""
        modalities = []
        if self.text: modalities.append(ModalityType.TEXT)
        if self.image_path: modalities.append(ModalityType.IMAGE)
        if self.audio_path: modalities.append(ModalityType.AUDIO)
        if self.video_path: modalities.append(ModalityType.VIDEO)
        return modalities

@dataclass
class MultiModalDocument(Document):
    """Extended document supporting multiple modalities."""
    content_multimodal: Optional[MultiModalContent] = None
    embeddings_by_modality: Dict[ModalityType, np.ndarray] = field(default_factory=dict)
    cross_modal_embedding: Optional[np.ndarray] = None  # Unified embedding
```

#### 1.2 Multi-modal Embedding Service

Create a new service for multi-modal embeddings:

```python
# New file: tldw_chatbook/RAG_Search/Services/multimodal_embeddings_service.py

import torch
from transformers import CLIPModel, CLIPProcessor, Wav2Vec2Model, Wav2Vec2Processor
from PIL import Image
import librosa
from typing import Dict, List, Optional, Union
import numpy as np

class MultiModalEmbeddingService:
    """Service for generating embeddings from multiple modalities."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        
        # Initialize models lazily
        self._clip_model = None
        self._clip_processor = None
        self._audio_model = None
        self._audio_processor = None
        
    def _load_clip(self):
        """Lazy load CLIP model for image/text."""
        if self._clip_model is None:
            model_name = self.config.get('clip_model', 'openai/clip-vit-base-patch32')
            self._clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
            self._clip_processor = CLIPProcessor.from_pretrained(model_name)
            
    def _load_audio_model(self):
        """Lazy load audio embedding model."""
        if self._audio_model is None:
            model_name = self.config.get('audio_model', 'facebook/wav2vec2-base')
            self._audio_model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
            self._audio_processor = Wav2Vec2Processor.from_pretrained(model_name)
    
    def embed_image(self, image_path: Path) -> np.ndarray:
        """Generate embedding for an image."""
        self._load_clip()
        image = Image.open(image_path).convert('RGB')
        inputs = self._clip_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self._clip_model.get_image_features(**inputs)
            
        return image_features.cpu().numpy().squeeze()
    
    def embed_text_with_clip(self, text: str) -> np.ndarray:
        """Generate CLIP text embedding (for cross-modal search)."""
        self._load_clip()
        inputs = self._clip_processor(text=text, return_tensors="pt", 
                                     padding=True, truncation=True, 
                                     max_length=77).to(self.device)
        
        with torch.no_grad():
            text_features = self._clip_model.get_text_features(**inputs)
            
        return text_features.cpu().numpy().squeeze()
    
    def embed_audio(self, audio_path: Path, sample_rate: int = 16000) -> np.ndarray:
        """Generate embedding for audio."""
        self._load_audio_model()
        
        # Load audio
        waveform, sr = librosa.load(audio_path, sr=sample_rate)
        
        # Process with wav2vec2
        inputs = self._audio_processor(waveform, sampling_rate=sample_rate, 
                                      return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self._audio_model(**inputs)
            # Use mean pooling over time dimension
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings.cpu().numpy().squeeze()
    
    def embed_video_frames(self, video_path: Path, 
                          sample_interval: float = 1.0) -> List[np.ndarray]:
        """Extract and embed frames from video."""
        import cv2
        
        self._load_clip()
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * sample_interval)
        
        frame_embeddings = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Get embedding
                inputs = self._clip_processor(images=pil_image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    features = self._clip_model.get_image_features(**inputs)
                
                frame_embeddings.append(features.cpu().numpy().squeeze())
            
            frame_count += 1
            
        cap.release()
        return frame_embeddings
    
    def create_unified_embedding(self, embeddings_by_modality: Dict[ModalityType, np.ndarray]) -> np.ndarray:
        """Create a unified embedding from multiple modalities."""
        # Simple approach: concatenate and normalize
        # More sophisticated: use a fusion model
        
        all_embeddings = []
        for modality, embedding in sorted(embeddings_by_modality.items()):
            # Normalize each embedding
            normalized = embedding / np.linalg.norm(embedding)
            all_embeddings.append(normalized)
        
        if len(all_embeddings) == 1:
            return all_embeddings[0]
        
        # Concatenate and reduce dimensionality if needed
        unified = np.concatenate(all_embeddings)
        
        # Optional: Use PCA or learned projection to reduce dimensions
        if len(unified) > self.config.get('max_unified_dim', 768):
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.config.get('unified_dim', 768))
            unified = pca.fit_transform(unified.reshape(1, -1)).squeeze()
        
        return unified / np.linalg.norm(unified)
```

#### 1.3 Update Retrieval Strategies

Extend retrievers to handle multi-modal queries:

```python
# In tldw_chatbook/RAG_Search/Services/rag_service/retrieval.py

class MultiModalRetriever(BaseRetriever):
    """Retriever supporting multi-modal search."""
    
    def __init__(self, 
                 vector_store: ChromaDB,
                 embedding_service: MultiModalEmbeddingService,
                 config: Dict[str, Any] = None):
        super().__init__(DataSource.MEDIA_DB, config)
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        
    async def retrieve(self,
                      query: Union[str, MultiModalContent],
                      filters: Optional[Dict[str, Any]] = None,
                      top_k: int = 10) -> SearchResult:
        """Retrieve documents using multi-modal query."""
        
        # Generate query embeddings based on modalities
        if isinstance(query, str):
            # Text-only query
            query_embedding = self.embedding_service.embed_text_with_clip(query)
            search_modality = ModalityType.TEXT
        else:
            # Multi-modal query
            embeddings = {}
            if query.text:
                embeddings[ModalityType.TEXT] = self.embedding_service.embed_text_with_clip(query.text)
            if query.image_path:
                embeddings[ModalityType.IMAGE] = self.embedding_service.embed_image(query.image_path)
            if query.audio_path:
                embeddings[ModalityType.AUDIO] = self.embedding_service.embed_audio(query.audio_path)
                
            query_embedding = self.embedding_service.create_unified_embedding(embeddings)
            search_modality = ModalityType.MULTIMODAL
        
        # Search in appropriate collection
        collection_name = self._get_collection_name(search_modality)
        results = self.vector_store.similarity_search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )
        
        # Convert to Documents
        documents = []
        for result in results:
            doc = MultiModalDocument(
                id=result['id'],
                content=result['metadata'].get('text_content', ''),
                metadata=result['metadata'],
                source=DataSource.MEDIA_DB,
                score=result['score'],
                content_multimodal=self._reconstruct_multimodal_content(result['metadata'])
            )
            documents.append(doc)
        
        return SearchResult(
            documents=documents,
            query=str(query),
            search_type="multimodal",
            metadata={"modality": search_modality.name}
        )
```

#### 1.4 Storage Schema Updates

Update ChromaDB collections for multi-modal embeddings:

```python
# In tldw_chatbook/Embeddings/Chroma_Lib.py

def create_multimodal_collection(self, name: str, embedding_dim: int):
    """Create a collection for multi-modal embeddings."""
    
    metadata = {
        "embedding_type": "multimodal",
        "embedding_dim": embedding_dim,
        "modalities": ["text", "image", "audio", "video"]
    }
    
    collection = self.client.create_collection(
        name=name,
        metadata=metadata,
        embedding_function=None  # We'll provide embeddings directly
    )
    
    return collection

def add_multimodal_documents(self, collection_name: str, documents: List[MultiModalDocument]):
    """Add multi-modal documents to collection."""
    
    collection = self.client.get_collection(collection_name)
    
    ids = []
    embeddings = []
    metadatas = []
    
    for doc in documents:
        ids.append(doc.id)
        
        # Use unified embedding if available, otherwise primary modality
        if doc.cross_modal_embedding is not None:
            embeddings.append(doc.cross_modal_embedding.tolist())
        else:
            # Use first available embedding
            for modality, embedding in doc.embeddings_by_modality.items():
                embeddings.append(embedding.tolist())
                break
        
        # Store all metadata including paths to original media
        metadata = doc.metadata.copy()
        if doc.content_multimodal:
            if doc.content_multimodal.image_path:
                metadata['image_path'] = str(doc.content_multimodal.image_path)
            if doc.content_multimodal.audio_path:
                metadata['audio_path'] = str(doc.content_multimodal.audio_path)
            if doc.content_multimodal.video_path:
                metadata['video_path'] = str(doc.content_multimodal.video_path)
        
        metadatas.append(metadata)
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
    )
```

### Integration Points

1. **Media Ingestion**: Update `Media_DB_v2.py` to extract and store multi-modal features
2. **UI Updates**: Enhance `SearchRAGWindow.py` to support image/audio queries
3. **Configuration**: Add multi-modal settings to `config.toml`

```toml
[rag.multimodal]
enabled = false  # Opt-in feature
clip_model = "openai/clip-vit-base-patch32"
audio_model = "facebook/wav2vec2-base"
video_sample_interval = 1.0  # seconds
max_image_size = 1024
device = "cpu"  # or "cuda", "mps"
```

## 2. Custom Pipeline Configurations

### Overview

Implement a YAML-based pipeline configuration system that allows users to define, version, and switch between different RAG pipeline configurations.

### Implementation Plan

#### 2.1 Pipeline Configuration Schema

Create a schema for pipeline definitions:

```yaml
# Example: pipelines.yaml
version: "1.0"
pipelines:
  default:
    description: "Standard RAG pipeline with balanced performance"
    retriever:
      strategy: "hybrid"
      components:
        - type: "media_db_fts"
          weight: 0.3
          config:
            top_k: 20
        - type: "vector_search"
          weight: 0.7
          config:
            top_k: 20
            collection: "media_embeddings"
    processor:
      reranker:
        enabled: true
        model: "flashrank"
        top_k: 10
      deduplication:
        enabled: true
        threshold: 0.85
      context_builder:
        max_tokens: 4096
        strategy: "relevance_ordered"
    generator:
      model: "gpt-3.5-turbo"
      temperature: 0.7
      system_prompt_template: "rag_default"
      
  multimodal_experimental:
    description: "Multi-modal pipeline for image and text search"
    retriever:
      strategy: "multimodal"
      components:
        - type: "clip_unified"
          config:
            collections: ["multimodal_media", "multimodal_notes"]
            top_k: 15
    processor:
      reranker:
        enabled: true
        model: "clip_rerank"  # Custom CLIP-based reranker
      context_builder:
        max_tokens: 3072
        include_images: true
        max_images: 3
    generator:
      model: "gpt-4-vision"
      temperature: 0.5
      
  fast_local:
    description: "Optimized for speed with local models"
    retriever:
      strategy: "vector_only"
      components:
        - type: "vector_search"
          config:
            top_k: 5
    processor:
      reranker:
        enabled: false  # Skip for speed
      context_builder:
        max_tokens: 2048
    generator:
      model: "llama-2-7b"
      temperature: 0.7
      api_type: "local_llama_cpp"
```

#### 2.2 Pipeline Builder

```python
# New file: tldw_chatbook/RAG_Search/Services/pipeline_builder.py

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from loguru import logger

from .rag_service.app import RAGApplication
from .rag_service.types import RetrieverStrategy, ProcessingStrategy, GenerationStrategy
from .service_factory import ServiceFactory

@dataclass
class PipelineConfig:
    """Configuration for a single pipeline."""
    name: str
    description: str
    retriever_config: Dict[str, Any]
    processor_config: Dict[str, Any]
    generator_config: Dict[str, Any]
    metadata: Dict[str, Any] = None

class PipelineBuilder:
    """Builds RAG pipelines from YAML configurations."""
    
    def __init__(self, 
                 service_factory: ServiceFactory,
                 base_config: Dict[str, Any]):
        self.service_factory = service_factory
        self.base_config = base_config
        self._pipeline_cache = {}
        
    def load_pipeline_configs(self, config_path: Path) -> Dict[str, PipelineConfig]:
        """Load all pipeline configurations from YAML file."""
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            
            pipelines = {}
            for name, config in data.get('pipelines', {}).items():
                pipelines[name] = PipelineConfig(
                    name=name,
                    description=config.get('description', ''),
                    retriever_config=config.get('retriever', {}),
                    processor_config=config.get('processor', {}),
                    generator_config=config.get('generator', {}),
                    metadata=config.get('metadata', {})
                )
            
            logger.info(f"Loaded {len(pipelines)} pipeline configurations")
            return pipelines
            
        except Exception as e:
            logger.error(f"Failed to load pipeline configs: {e}")
            return {}
    
    def build_pipeline(self, pipeline_config: PipelineConfig) -> RAGApplication:
        """Build a RAG pipeline from configuration."""
        
        # Check cache
        if pipeline_config.name in self._pipeline_cache:
            return self._pipeline_cache[pipeline_config.name]
        
        try:
            # Build retriever(s)
            retrievers = self._build_retrievers(pipeline_config.retriever_config)
            
            # Build processor
            processor = self._build_processor(pipeline_config.processor_config)
            
            # Build generator
            generator = self._build_generator(pipeline_config.generator_config)
            
            # Create RAG application
            app_config = self.base_config.copy()
            app_config.update({
                'pipeline_name': pipeline_config.name,
                'pipeline_description': pipeline_config.description
            })
            
            pipeline = RAGApplication(
                retrievers=retrievers,
                processor=processor,
                generator=generator,
                config=app_config
            )
            
            # Cache the pipeline
            self._pipeline_cache[pipeline_config.name] = pipeline
            
            logger.info(f"Built pipeline: {pipeline_config.name}")
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to build pipeline {pipeline_config.name}: {e}")
            raise
    
    def _build_retrievers(self, config: Dict[str, Any]) -> List[RetrieverStrategy]:
        """Build retriever components from config."""
        strategy = config.get('strategy', 'hybrid')
        components = config.get('components', [])
        
        retrievers = []
        for component in components:
            retriever_type = component['type']
            retriever_config = component.get('config', {})
            weight = component.get('weight', 1.0)
            
            # Create retriever using factory
            retriever = self.service_factory.create_retriever(
                retriever_type, 
                retriever_config
            )
            
            # Wrap with weight if needed
            if strategy == 'hybrid' and weight != 1.0:
                retriever = WeightedRetriever(retriever, weight)
            
            retrievers.append(retriever)
        
        return retrievers
    
    def _build_processor(self, config: Dict[str, Any]) -> ProcessingStrategy:
        """Build processor from config."""
        processor_class = self.service_factory.get_processor_class(
            config.get('type', 'default')
        )
        
        return processor_class(config)
    
    def _build_generator(self, config: Dict[str, Any]) -> GenerationStrategy:
        """Build generator from config."""
        generator_class = self.service_factory.get_generator_class(
            config.get('type', 'default')
        )
        
        return generator_class(config)
```

#### 2.3 Pipeline Manager

```python
# New file: tldw_chatbook/RAG_Search/Services/pipeline_manager.py

from typing import Dict, Optional, List
from pathlib import Path
import json
from datetime import datetime
from loguru import logger

class PipelineManager:
    """Manages multiple RAG pipelines and their lifecycle."""
    
    def __init__(self, 
                 pipeline_builder: PipelineBuilder,
                 config_dir: Path):
        self.pipeline_builder = pipeline_builder
        self.config_dir = config_dir
        self.pipelines = {}
        self.active_pipeline = None
        self.pipeline_history = []
        
    def load_all_pipelines(self):
        """Load all pipeline configurations from directory."""
        pipeline_files = list(self.config_dir.glob("*.yaml")) + \
                        list(self.config_dir.glob("*.yml"))
        
        for file in pipeline_files:
            configs = self.pipeline_builder.load_pipeline_configs(file)
            for name, config in configs.items():
                self.register_pipeline(name, config)
    
    def register_pipeline(self, name: str, config: PipelineConfig):
        """Register a pipeline configuration."""
        self.pipelines[name] = {
            'config': config,
            'instance': None,  # Lazy instantiation
            'created_at': datetime.now(),
            'last_used': None,
            'usage_count': 0
        }
        logger.info(f"Registered pipeline: {name}")
    
    def get_pipeline(self, name: str) -> Optional[RAGApplication]:
        """Get a pipeline instance by name."""
        if name not in self.pipelines:
            logger.error(f"Pipeline not found: {name}")
            return None
        
        pipeline_info = self.pipelines[name]
        
        # Lazy instantiation
        if pipeline_info['instance'] is None:
            pipeline_info['instance'] = self.pipeline_builder.build_pipeline(
                pipeline_info['config']
            )
        
        # Update usage stats
        pipeline_info['last_used'] = datetime.now()
        pipeline_info['usage_count'] += 1
        
        return pipeline_info['instance']
    
    def set_active_pipeline(self, name: str) -> bool:
        """Set the active pipeline."""
        if name not in self.pipelines:
            logger.error(f"Cannot set active pipeline: {name} not found")
            return False
        
        old_pipeline = self.active_pipeline
        self.active_pipeline = name
        
        # Track history
        self.pipeline_history.append({
            'timestamp': datetime.now(),
            'from_pipeline': old_pipeline,
            'to_pipeline': name
        })
        
        logger.info(f"Active pipeline changed: {old_pipeline} -> {name}")
        return True
    
    def get_active_pipeline(self) -> Optional[RAGApplication]:
        """Get the currently active pipeline."""
        if self.active_pipeline is None:
            # Use default if available
            if 'default' in self.pipelines:
                self.set_active_pipeline('default')
            else:
                # Use first available
                if self.pipelines:
                    first_name = list(self.pipelines.keys())[0]
                    self.set_active_pipeline(first_name)
        
        if self.active_pipeline:
            return self.get_pipeline(self.active_pipeline)
        
        return None
    
    def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all registered pipelines with metadata."""
        pipeline_list = []
        for name, info in self.pipelines.items():
            pipeline_list.append({
                'name': name,
                'description': info['config'].description,
                'is_active': name == self.active_pipeline,
                'created_at': info['created_at'].isoformat(),
                'last_used': info['last_used'].isoformat() if info['last_used'] else None,
                'usage_count': info['usage_count']
            })
        return pipeline_list
    
    def save_pipeline_config(self, name: str, config: PipelineConfig, 
                           version: Optional[str] = None):
        """Save a pipeline configuration to file."""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"pipeline_{name}_v{version}.yaml"
        filepath = self.config_dir / filename
        
        data = {
            'version': '1.0',
            'pipelines': {
                name: {
                    'description': config.description,
                    'retriever': config.retriever_config,
                    'processor': config.processor_config,
                    'generator': config.generator_config,
                    'metadata': config.metadata or {}
                }
            }
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        logger.info(f"Saved pipeline config: {filepath}")
        
    def get_pipeline_metrics(self, name: str) -> Dict[str, Any]:
        """Get metrics for a specific pipeline."""
        pipeline = self.get_pipeline(name)
        if pipeline and hasattr(pipeline, 'metrics_collector'):
            return pipeline.metrics_collector.get_summary()
        return {}
```

#### 2.4 Integration with UI

Add pipeline selection to the RAG UI:

```python
# In tldw_chatbook/UI/SearchRAGWindow.py

from textual.widgets import Select, Button
from ..RAG_Search.Services.pipeline_manager import PipelineManager

class PipelineSelector(Container):
    """Widget for selecting and managing RAG pipelines."""
    
    def __init__(self, pipeline_manager: PipelineManager):
        super().__init__()
        self.pipeline_manager = pipeline_manager
        
    def compose(self):
        pipelines = self.pipeline_manager.list_pipelines()
        options = [(p['name'], f"{p['name']} - {p['description']}") 
                   for p in pipelines]
        
        yield Label("Active Pipeline:")
        yield Select(
            options=options,
            value=self.pipeline_manager.active_pipeline,
            id="pipeline-select"
        )
        yield Button("Pipeline Info", id="pipeline-info-btn")
        yield Button("Compare Pipelines", id="pipeline-compare-btn")
    
    @on(Select.Changed, "#pipeline-select")
    def on_pipeline_changed(self, event):
        """Handle pipeline selection change."""
        new_pipeline = event.value
        if self.pipeline_manager.set_active_pipeline(new_pipeline):
            self.notify(f"Switched to pipeline: {new_pipeline}")
```

## 3. A/B Testing Framework

### Overview

Implement a framework for comparing different RAG pipelines side-by-side and analyzing their performance.

### Implementation Plan

#### 3.1 A/B Test Manager

```python
# New file: tldw_chatbook/RAG_Search/Services/ab_testing.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import hashlib
import json
from enum import Enum, auto
from loguru import logger

class ExperimentStatus(Enum):
    DRAFT = auto()
    RUNNING = auto()
    COMPLETED = auto()
    ABORTED = auto()

@dataclass
class ABTestConfig:
    """Configuration for an A/B test."""
    name: str
    description: str
    pipeline_a: str  # Control pipeline
    pipeline_b: str  # Treatment pipeline
    traffic_split: float = 0.5  # Fraction going to pipeline B
    min_samples: int = 100
    max_duration_hours: float = 24.0
    metrics_to_track: List[str] = None

@dataclass
class ABTestResult:
    """Result from a single A/B test comparison."""
    query: str
    timestamp: datetime
    pipeline_a_response: RAGResponse
    pipeline_b_response: RAGResponse
    pipeline_a_metrics: Dict[str, float]
    pipeline_b_metrics: Dict[str, float]
    user_preference: Optional[str] = None  # 'A', 'B', or None

class ABTestManager:
    """Manages A/B testing between different RAG pipelines."""
    
    def __init__(self, 
                 pipeline_manager: PipelineManager,
                 storage_path: Path):
        self.pipeline_manager = pipeline_manager
        self.storage_path = storage_path
        self.active_experiments = {}
        self.results_cache = {}
        
    def create_experiment(self, config: ABTestConfig) -> str:
        """Create a new A/B test experiment."""
        experiment_id = f"exp_{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.active_experiments[experiment_id] = {
            'config': config,
            'status': ExperimentStatus.DRAFT,
            'created_at': datetime.now(),
            'started_at': None,
            'results': [],
            'metrics_summary': {}
        }
        
        logger.info(f"Created A/B test experiment: {experiment_id}")
        return experiment_id
    
    def start_experiment(self, experiment_id: str):
        """Start an A/B test experiment."""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        exp = self.active_experiments[experiment_id]
        exp['status'] = ExperimentStatus.RUNNING
        exp['started_at'] = datetime.now()
        
        logger.info(f"Started experiment: {experiment_id}")
    
    def route_query(self, query: str, experiment_id: Optional[str] = None) -> str:
        """Route a query to appropriate pipeline based on experiment."""
        if experiment_id and experiment_id in self.active_experiments:
            exp = self.active_experiments[experiment_id]
            if exp['status'] == ExperimentStatus.RUNNING:
                # Use consistent hashing for deterministic routing
                query_hash = int(hashlib.md5(query.encode()).hexdigest(), 16)
                if (query_hash % 100) / 100.0 < exp['config'].traffic_split:
                    return exp['config'].pipeline_b
                else:
                    return exp['config'].pipeline_a
        
        # Default to active pipeline
        return self.pipeline_manager.active_pipeline
    
    async def run_comparison(self, 
                           query: str, 
                           experiment_id: str,
                           context: Optional[Dict[str, Any]] = None) -> ABTestResult:
        """Run both pipelines and compare results."""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        exp = self.active_experiments[experiment_id]
        config = exp['config']
        
        # Get pipelines
        pipeline_a = self.pipeline_manager.get_pipeline(config.pipeline_a)
        pipeline_b = self.pipeline_manager.get_pipeline(config.pipeline_b)
        
        if not pipeline_a or not pipeline_b:
            raise ValueError("One or both pipelines not available")
        
        # Run both pipelines in parallel
        import asyncio
        
        results = await asyncio.gather(
            self._run_pipeline_with_metrics(pipeline_a, query, context),
            self._run_pipeline_with_metrics(pipeline_b, query, context)
        )
        
        response_a, metrics_a = results[0]
        response_b, metrics_b = results[1]
        
        # Create result
        result = ABTestResult(
            query=query,
            timestamp=datetime.now(),
            pipeline_a_response=response_a,
            pipeline_b_response=response_b,
            pipeline_a_metrics=metrics_a,
            pipeline_b_metrics=metrics_b
        )
        
        # Store result
        exp['results'].append(result)
        
        return result
    
    async def _run_pipeline_with_metrics(self, 
                                       pipeline: RAGApplication,
                                       query: str,
                                       context: Optional[Dict[str, Any]] = None) -> Tuple[RAGResponse, Dict[str, float]]:
        """Run a pipeline and collect metrics."""
        import time
        
        start_time = time.time()
        
        # Run pipeline
        response = await pipeline.arun(
            query=query,
            conversation_context=context
        )
        
        end_time = time.time()
        
        # Collect metrics
        metrics = {
            'total_latency': end_time - start_time,
            'retrieval_latency': pipeline.last_metrics.get('retrieval_time', 0),
            'processing_latency': pipeline.last_metrics.get('processing_time', 0),
            'generation_latency': pipeline.last_metrics.get('generation_time', 0),
            'num_documents_retrieved': len(response.sources),
            'context_size': response.context.total_tokens,
            'response_length': len(response.answer)
        }
        
        # Add any custom metrics from the pipeline
        if hasattr(pipeline, 'get_custom_metrics'):
            metrics.update(pipeline.get_custom_metrics())
        
        return response, metrics
    
    def record_user_preference(self, 
                              experiment_id: str,
                              result_index: int,
                              preference: str):
        """Record user preference for a comparison result."""
        if experiment_id not in self.active_experiments:
            return
        
        exp = self.active_experiments[experiment_id]
        if 0 <= result_index < len(exp['results']):
            exp['results'][result_index].user_preference = preference
            logger.info(f"Recorded preference: {preference} for result {result_index}")
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze results of an A/B test experiment."""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        exp = self.active_experiments[experiment_id]
        results = exp['results']
        
        if not results:
            return {'error': 'No results to analyze'}
        
        # Calculate aggregate metrics
        metrics_a = []
        metrics_b = []
        preferences_a = 0
        preferences_b = 0
        
        for result in results:
            metrics_a.append(result.pipeline_a_metrics)
            metrics_b.append(result.pipeline_b_metrics)
            
            if result.user_preference == 'A':
                preferences_a += 1
            elif result.user_preference == 'B':
                preferences_b += 1
        
        # Statistical analysis
        import numpy as np
        from scipy import stats
        
        analysis = {
            'experiment_id': experiment_id,
            'num_comparisons': len(results),
            'pipeline_a': exp['config'].pipeline_a,
            'pipeline_b': exp['config'].pipeline_b,
            'user_preferences': {
                'pipeline_a': preferences_a,
                'pipeline_b': preferences_b,
                'no_preference': len(results) - preferences_a - preferences_b
            }
        }
        
        # Compare metrics
        metric_comparisons = {}
        for metric in exp['config'].metrics_to_track or ['total_latency']:
            values_a = [m.get(metric, 0) for m in metrics_a]
            values_b = [m.get(metric, 0) for m in metrics_b]
            
            if values_a and values_b:
                # T-test for statistical significance
                t_stat, p_value = stats.ttest_ind(values_a, values_b)
                
                metric_comparisons[metric] = {
                    'pipeline_a_mean': np.mean(values_a),
                    'pipeline_a_std': np.std(values_a),
                    'pipeline_b_mean': np.mean(values_b),
                    'pipeline_b_std': np.std(values_b),
                    'difference': np.mean(values_b) - np.mean(values_a),
                    'relative_change': (np.mean(values_b) - np.mean(values_a)) / np.mean(values_a) * 100,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        analysis['metric_comparisons'] = metric_comparisons
        
        # Winner determination
        winner = self._determine_winner(analysis)
        analysis['recommendation'] = winner
        
        return analysis
    
    def _determine_winner(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the winning pipeline based on analysis."""
        points_a = 0
        points_b = 0
        
        # User preference
        prefs = analysis['user_preferences']
        if prefs['pipeline_a'] > prefs['pipeline_b']:
            points_a += 2
        elif prefs['pipeline_b'] > prefs['pipeline_a']:
            points_b += 2
        
        # Metrics
        for metric, comparison in analysis['metric_comparisons'].items():
            if comparison['significant']:
                # Lower is better for latency metrics
                if 'latency' in metric:
                    if comparison['difference'] > 0:  # B is slower
                        points_a += 1
                    else:
                        points_b += 1
                # Higher is better for quality metrics
                else:
                    if comparison['difference'] > 0:  # B is better
                        points_b += 1
                    else:
                        points_a += 1
        
        if points_a > points_b:
            return {
                'winner': analysis['pipeline_a'],
                'confidence': 'high' if points_a - points_b > 2 else 'medium',
                'reason': 'Better performance and user preference'
            }
        elif points_b > points_a:
            return {
                'winner': analysis['pipeline_b'],
                'confidence': 'high' if points_b - points_a > 2 else 'medium',
                'reason': 'Better performance and user preference'
            }
        else:
            return {
                'winner': 'tie',
                'confidence': 'low',
                'reason': 'No significant difference detected'
            }
    
    def export_experiment_results(self, experiment_id: str, format: str = 'json') -> str:
        """Export experiment results in specified format."""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        exp = self.active_experiments[experiment_id]
        analysis = self.analyze_experiment(experiment_id)
        
        export_data = {
            'experiment': {
                'id': experiment_id,
                'name': exp['config'].name,
                'description': exp['config'].description,
                'created_at': exp['created_at'].isoformat(),
                'started_at': exp['started_at'].isoformat() if exp['started_at'] else None,
                'status': exp['status'].name
            },
            'config': {
                'pipeline_a': exp['config'].pipeline_a,
                'pipeline_b': exp['config'].pipeline_b,
                'traffic_split': exp['config'].traffic_split,
                'min_samples': exp['config'].min_samples
            },
            'analysis': analysis,
            'raw_results': [
                {
                    'query': r.query,
                    'timestamp': r.timestamp.isoformat(),
                    'pipeline_a_metrics': r.pipeline_a_metrics,
                    'pipeline_b_metrics': r.pipeline_b_metrics,
                    'user_preference': r.user_preference
                }
                for r in exp['results']
            ]
        }
        
        if format == 'json':
            return json.dumps(export_data, indent=2)
        elif format == 'csv':
            # Convert to CSV format
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=[
                'query', 'timestamp', 'pipeline',
                'total_latency', 'retrieval_latency', 
                'num_documents', 'user_preference'
            ])
            writer.writeheader()
            
            for r in exp['results']:
                # Write row for pipeline A
                writer.writerow({
                    'query': r.query,
                    'timestamp': r.timestamp.isoformat(),
                    'pipeline': exp['config'].pipeline_a,
                    'total_latency': r.pipeline_a_metrics.get('total_latency', 0),
                    'retrieval_latency': r.pipeline_a_metrics.get('retrieval_latency', 0),
                    'num_documents': r.pipeline_a_metrics.get('num_documents_retrieved', 0),
                    'user_preference': 'selected' if r.user_preference == 'A' else 'not_selected'
                })
                
                # Write row for pipeline B
                writer.writerow({
                    'query': r.query,
                    'timestamp': r.timestamp.isoformat(),
                    'pipeline': exp['config'].pipeline_b,
                    'total_latency': r.pipeline_b_metrics.get('total_latency', 0),
                    'retrieval_latency': r.pipeline_b_metrics.get('retrieval_latency', 0),
                    'num_documents': r.pipeline_b_metrics.get('num_documents_retrieved', 0),
                    'user_preference': 'selected' if r.user_preference == 'B' else 'not_selected'
                })
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
```

#### 3.2 A/B Testing UI

```python
# New file: tldw_chatbook/UI/ABTestingWindow.py

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Label, Button, Select, DataTable, Sparkline
from textual.reactive import reactive
from typing import Optional

class ABTestingWindow(Container):
    """UI for managing and viewing A/B tests."""
    
    def __init__(self, ab_test_manager: ABTestManager):
        super().__init__()
        self.ab_test_manager = ab_test_manager
        self.current_experiment = None
        
    def compose(self) -> ComposeResult:
        with Vertical():
            # Experiment controls
            with Horizontal(id="experiment-controls"):
                yield Button("New Experiment", id="new-exp-btn")
                yield Select(
                    options=self._get_experiment_options(),
                    id="experiment-select",
                    placeholder="Select experiment"
                )
                yield Button("Start", id="start-exp-btn")
                yield Button("Stop", id="stop-exp-btn")
                yield Button("Analyze", id="analyze-exp-btn")
            
            # Comparison view
            with Horizontal(id="comparison-view"):
                # Pipeline A results
                with Vertical(id="pipeline-a-view", classes="pipeline-view"):
                    yield Label("Pipeline A", classes="pipeline-header")
                    yield ScrollableContainer(id="pipeline-a-results")
                
                # Pipeline B results
                with Vertical(id="pipeline-b-view", classes="pipeline-view"):
                    yield Label("Pipeline B", classes="pipeline-header")
                    yield ScrollableContainer(id="pipeline-b-results")
            
            # Metrics dashboard
            with Container(id="metrics-dashboard"):
                yield Label("Experiment Metrics", classes="section-header")
                yield DataTable(id="metrics-table")
                
                # Performance charts
                with Horizontal(id="performance-charts"):
                    yield Sparkline(
                        data=[],
                        id="latency-chart",
                        title="Latency Comparison"
                    )
                    yield Sparkline(
                        data=[],
                        id="preference-chart",
                        title="User Preference"
                    )
    
    async def on_button_pressed(self, event: Button.Pressed):
        """Handle button presses."""
        if event.button.id == "new-exp-btn":
            await self._create_new_experiment()
        elif event.button.id == "start-exp-btn":
            await self._start_experiment()
        elif event.button.id == "analyze-exp-btn":
            await self._analyze_experiment()
    
    async def _run_comparison(self, query: str):
        """Run a comparison for the current experiment."""
        if not self.current_experiment:
            return
        
        result = await self.ab_test_manager.run_comparison(
            query=query,
            experiment_id=self.current_experiment
        )
        
        # Update UI with results
        await self._display_comparison_result(result)
    
    async def _display_comparison_result(self, result: ABTestResult):
        """Display comparison results in the UI."""
        # Update Pipeline A view
        pipeline_a_container = self.query_one("#pipeline-a-results")
        await self._add_result_to_container(
            pipeline_a_container,
            result.pipeline_a_response,
            result.pipeline_a_metrics
        )
        
        # Update Pipeline B view
        pipeline_b_container = self.query_one("#pipeline-b-results")
        await self._add_result_to_container(
            pipeline_b_container,
            result.pipeline_b_response,
            result.pipeline_b_metrics
        )
        
        # Update metrics
        await self._update_metrics_display()
```

## 4. RAG Metrics System

### Overview

Implement comprehensive metrics for evaluating RAG pipeline performance, including retrieval quality, context relevance, and answer quality.

### Implementation Plan

#### 4.1 RAG-Specific Metrics

```python
# New file: tldw_chatbook/RAG_Search/Services/rag_metrics.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics import ndcg_score
from loguru import logger

@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality."""
    precision_at_k: Dict[int, float]  # Precision@1, @5, @10
    recall_at_k: Dict[int, float]     # Recall@1, @5, @10
    mrr: float                        # Mean Reciprocal Rank
    ndcg: float                       # Normalized Discounted Cumulative Gain
    coverage: float                   # Fraction of queries with results

@dataclass
class ContextMetrics:
    """Metrics for context quality."""
    relevance_score: float           # Average relevance of context
    diversity_score: float           # Diversity of sources in context
    completeness_score: float        # How well context covers the query
    coherence_score: float          # Logical flow of context
    redundancy_score: float         # Amount of duplicate information

@dataclass
class AnswerMetrics:
    """Metrics for answer quality."""
    factuality_score: float         # Consistency with context
    relevance_score: float          # Relevance to query
    completeness_score: float       # Comprehensiveness
    fluency_score: float           # Language quality
    citation_accuracy: float       # Correct source attribution

@dataclass
class EndToEndMetrics:
    """End-to-end RAG metrics."""
    total_latency: float
    retrieval_metrics: RetrievalMetrics
    context_metrics: ContextMetrics
    answer_metrics: AnswerMetrics
    user_satisfaction: Optional[float] = None

class MetricCalculator(ABC):
    """Base class for metric calculators."""
    
    @abstractmethod
    def calculate(self, *args, **kwargs) -> float:
        """Calculate the metric."""
        pass

class RetrievalMetricCalculator:
    """Calculates retrieval-specific metrics."""
    
    def __init__(self, ground_truth_relevance: Optional[Dict[str, List[str]]] = None):
        self.ground_truth = ground_truth_relevance or {}
    
    def calculate_precision_at_k(self, 
                                retrieved_docs: List[Document],
                                relevant_doc_ids: List[str],
                                k: int) -> float:
        """Calculate Precision@K."""
        retrieved_ids = [doc.id for doc in retrieved_docs[:k]]
        relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in relevant_doc_ids)
        return relevant_retrieved / k if k > 0 else 0.0
    
    def calculate_recall_at_k(self,
                             retrieved_docs: List[Document],
                             relevant_doc_ids: List[str],
                             k: int) -> float:
        """Calculate Recall@K."""
        if not relevant_doc_ids:
            return 0.0
        
        retrieved_ids = [doc.id for doc in retrieved_docs[:k]]
        relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in relevant_doc_ids)
        return relevant_retrieved / len(relevant_doc_ids)
    
    def calculate_mrr(self,
                     retrieved_docs: List[Document],
                     relevant_doc_ids: List[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, doc in enumerate(retrieved_docs):
            if doc.id in relevant_doc_ids:
                return 1.0 / (i + 1)
        return 0.0
    
    def calculate_ndcg(self,
                      retrieved_docs: List[Document],
                      relevance_scores: Dict[str, float],
                      k: Optional[int] = None) -> float:
        """Calculate NDCG@K."""
        if k is None:
            k = len(retrieved_docs)
        
        # Get relevance scores for retrieved documents
        y_true = [relevance_scores.get(doc.id, 0.0) for doc in retrieved_docs[:k]]
        
        if not y_true or all(score == 0 for score in y_true):
            return 0.0
        
        # Ideal ranking would sort by relevance
        y_ideal = sorted(y_true, reverse=True)
        
        # Calculate NDCG
        return ndcg_score([y_ideal], [y_true])
    
    def calculate_all_metrics(self,
                            retrieved_docs: List[Document],
                            query: str) -> RetrievalMetrics:
        """Calculate all retrieval metrics."""
        # Get ground truth if available
        relevant_doc_ids = self.ground_truth.get(query, [])
        
        # Calculate metrics at different K values
        k_values = [1, 5, 10]
        precision_at_k = {}
        recall_at_k = {}
        
        for k in k_values:
            precision_at_k[k] = self.calculate_precision_at_k(
                retrieved_docs, relevant_doc_ids, k
            )
            recall_at_k[k] = self.calculate_recall_at_k(
                retrieved_docs, relevant_doc_ids, k
            )
        
        # Calculate other metrics
        mrr = self.calculate_mrr(retrieved_docs, relevant_doc_ids)
        
        # For NDCG, use document scores as relevance
        relevance_scores = {doc.id: doc.score for doc in retrieved_docs}
        ndcg = self.calculate_ndcg(retrieved_docs, relevance_scores)
        
        # Coverage is 1.0 if we have results, 0.0 otherwise
        coverage = 1.0 if retrieved_docs else 0.0
        
        return RetrievalMetrics(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            mrr=mrr,
            ndcg=ndcg,
            coverage=coverage
        )

class ContextMetricCalculator:
    """Calculates context quality metrics."""
    
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
    
    def calculate_relevance(self,
                          context: RAGContext,
                          query: str) -> float:
        """Calculate context relevance to query."""
        # Use embedding similarity as a proxy for relevance
        query_embedding = self.embedding_service.embed_text(query)
        
        relevance_scores = []
        for doc in context.documents:
            if doc.embedding is not None:
                similarity = self._cosine_similarity(query_embedding, doc.embedding)
                relevance_scores.append(similarity)
        
        return np.mean(relevance_scores) if relevance_scores else 0.0
    
    def calculate_diversity(self, context: RAGContext) -> float:
        """Calculate diversity of sources in context."""
        # Source diversity
        sources = set(doc.source.value for doc in context.documents)
        source_diversity = len(sources) / len(DataSource) if context.documents else 0.0
        
        # Content diversity (using embeddings)
        if len(context.documents) < 2:
            return source_diversity
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(context.documents)):
            for j in range(i + 1, len(context.documents)):
                if context.documents[i].embedding is not None and \
                   context.documents[j].embedding is not None:
                    sim = self._cosine_similarity(
                        context.documents[i].embedding,
                        context.documents[j].embedding
                    )
                    similarities.append(sim)
        
        # Diversity is inverse of average similarity
        content_diversity = 1.0 - np.mean(similarities) if similarities else 0.5
        
        # Combine source and content diversity
        return (source_diversity + content_diversity) / 2
    
    def calculate_completeness(self,
                             context: RAGContext,
                             query: str) -> float:
        """Calculate how well context covers the query."""
        # Extract key terms from query
        query_terms = set(query.lower().split())
        
        # Check coverage in context
        context_terms = set(context.combined_text.lower().split())
        
        covered_terms = query_terms.intersection(context_terms)
        coverage = len(covered_terms) / len(query_terms) if query_terms else 0.0
        
        # Also consider token count relative to typical needs
        optimal_tokens = 2000  # Assume this is optimal context size
        token_ratio = min(context.total_tokens / optimal_tokens, 1.0)
        
        return (coverage + token_ratio) / 2
    
    def calculate_coherence(self, context: RAGContext) -> float:
        """Calculate logical flow and coherence of context."""
        if len(context.documents) < 2:
            return 1.0
        
        # Calculate sequential similarity (documents should flow logically)
        sequential_similarities = []
        for i in range(len(context.documents) - 1):
            if context.documents[i].embedding is not None and \
               context.documents[i + 1].embedding is not None:
                sim = self._cosine_similarity(
                    context.documents[i].embedding,
                    context.documents[i + 1].embedding
                )
                sequential_similarities.append(sim)
        
        # Moderate similarity indicates good flow (not too similar, not too different)
        if sequential_similarities:
            avg_similarity = np.mean(sequential_similarities)
            # Optimal similarity around 0.6-0.8
            coherence = 1.0 - abs(avg_similarity - 0.7) * 2
            return max(0.0, min(1.0, coherence))
        
        return 0.5
    
    def calculate_redundancy(self, context: RAGContext) -> float:
        """Calculate redundancy in context."""
        if len(context.documents) < 2:
            return 0.0
        
        # Check for near-duplicate content
        high_similarity_pairs = 0
        total_pairs = 0
        
        for i in range(len(context.documents)):
            for j in range(i + 1, len(context.documents)):
                if context.documents[i].embedding is not None and \
                   context.documents[j].embedding is not None:
                    sim = self._cosine_similarity(
                        context.documents[i].embedding,
                        context.documents[j].embedding
                    )
                    total_pairs += 1
                    if sim > 0.9:  # High similarity threshold
                        high_similarity_pairs += 1
        
        return high_similarity_pairs / total_pairs if total_pairs > 0 else 0.0
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def calculate_all_metrics(self,
                            context: RAGContext,
                            query: str) -> ContextMetrics:
        """Calculate all context metrics."""
        return ContextMetrics(
            relevance_score=self.calculate_relevance(context, query),
            diversity_score=self.calculate_diversity(context),
            completeness_score=self.calculate_completeness(context, query),
            coherence_score=self.calculate_coherence(context),
            redundancy_score=self.calculate_redundancy(context)
        )

class AnswerMetricCalculator:
    """Calculates answer quality metrics."""
    
    def __init__(self, llm_service=None):
        self.llm_service = llm_service  # For LLM-based evaluation
    
    async def calculate_factuality(self,
                                 answer: str,
                                 context: RAGContext) -> float:
        """Calculate factuality/faithfulness to context."""
        if not self.llm_service:
            # Simple keyword-based approach
            context_terms = set(context.combined_text.lower().split())
            answer_terms = set(answer.lower().split())
            
            # Check what fraction of answer content is grounded in context
            grounded_terms = answer_terms.intersection(context_terms)
            return len(grounded_terms) / len(answer_terms) if answer_terms else 0.0
        
        # LLM-based evaluation
        prompt = f"""
        Context: {context.combined_text[:2000]}
        
        Answer: {answer}
        
        Rate the factual accuracy of the answer based solely on the provided context.
        Score from 0.0 to 1.0, where 1.0 means completely factual and grounded in context.
        Respond with only the numerical score.
        """
        
        response = await self.llm_service.generate(prompt, temperature=0.0)
        try:
            return float(response.strip())
        except:
            return 0.5
    
    async def calculate_relevance(self,
                                answer: str,
                                query: str) -> float:
        """Calculate answer relevance to query."""
        if not self.llm_service:
            # Keyword overlap approach
            query_terms = set(query.lower().split())
            answer_terms = set(answer.lower().split())
            
            overlap = query_terms.intersection(answer_terms)
            return len(overlap) / len(query_terms) if query_terms else 0.0
        
        # LLM-based evaluation
        prompt = f"""
        Question: {query}
        
        Answer: {answer}
        
        Rate how well the answer addresses the question.
        Score from 0.0 to 1.0, where 1.0 means perfectly relevant.
        Respond with only the numerical score.
        """
        
        response = await self.llm_service.generate(prompt, temperature=0.0)
        try:
            return float(response.strip())
        except:
            return 0.5
    
    def calculate_fluency(self, answer: str) -> float:
        """Calculate language fluency and readability."""
        # Simple heuristics for fluency
        
        # Check sentence structure
        sentences = answer.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        # Optimal sentence length is 15-20 words
        sentence_score = 1.0 - abs(avg_sentence_length - 17.5) / 17.5
        sentence_score = max(0.0, min(1.0, sentence_score))
        
        # Check for common issues
        issues = 0
        if '...' in answer: issues += 1
        if answer.count('(') != answer.count(')'): issues += 1
        if answer.count('[') != answer.count(']'): issues += 1
        
        issue_penalty = issues * 0.1
        
        return max(0.0, sentence_score - issue_penalty)
    
    def calculate_citation_accuracy(self,
                                  answer: str,
                                  sources: List[Document]) -> float:
        """Calculate accuracy of source citations."""
        # Look for citation patterns [1], [2], etc.
        import re
        
        citations = re.findall(r'\[(\d+)\]', answer)
        if not citations:
            # No citations found - check if answer claims facts
            if any(phrase in answer.lower() for phrase in 
                   ['according to', 'based on', 'states that', 'shows that']):
                return 0.0  # Should have citations
            return 1.0  # No factual claims requiring citations
        
        # Check if citations are valid
        valid_citations = 0
        for citation in citations:
            try:
                idx = int(citation) - 1
                if 0 <= idx < len(sources):
                    valid_citations += 1
            except:
                pass
        
        return valid_citations / len(citations) if citations else 1.0
    
    async def calculate_all_metrics(self,
                                  answer: str,
                                  query: str,
                                  context: RAGContext,
                                  sources: List[Document]) -> AnswerMetrics:
        """Calculate all answer metrics."""
        return AnswerMetrics(
            factuality_score=await self.calculate_factuality(answer, context),
            relevance_score=await self.calculate_relevance(answer, query),
            completeness_score=0.0,  # TODO: Implement
            fluency_score=self.calculate_fluency(answer),
            citation_accuracy=self.calculate_citation_accuracy(answer, sources)
        )

class RAGMetricsCollector:
    """Collects and aggregates all RAG metrics."""
    
    def __init__(self,
                 retrieval_calculator: RetrievalMetricCalculator,
                 context_calculator: ContextMetricCalculator,
                 answer_calculator: AnswerMetricCalculator):
        self.retrieval_calc = retrieval_calculator
        self.context_calc = context_calculator
        self.answer_calc = answer_calculator
        self.metrics_history = []
    
    async def collect_metrics(self,
                            query: str,
                            search_results: List[SearchResult],
                            context: RAGContext,
                            response: RAGResponse,
                            latency_info: Dict[str, float]) -> EndToEndMetrics:
        """Collect all metrics for a RAG query."""
        
        # Retrieval metrics
        all_retrieved_docs = []
        for result in search_results:
            all_retrieved_docs.extend(result.documents)
        
        retrieval_metrics = self.retrieval_calc.calculate_all_metrics(
            all_retrieved_docs, query
        )
        
        # Context metrics
        context_metrics = self.context_calc.calculate_all_metrics(
            context, query
        )
        
        # Answer metrics
        answer_metrics = await self.answer_calc.calculate_all_metrics(
            response.answer, query, context, response.sources
        )
        
        # Combine into end-to-end metrics
        metrics = EndToEndMetrics(
            total_latency=latency_info.get('total', 0.0),
            retrieval_metrics=retrieval_metrics,
            context_metrics=context_metrics,
            answer_metrics=answer_metrics
        )
        
        # Store in history
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'query': query,
            'metrics': metrics
        })
        
        return metrics
    
    def get_aggregate_metrics(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Get aggregate metrics over recent queries."""
        if last_n:
            recent_history = self.metrics_history[-last_n:]
        else:
            recent_history = self.metrics_history
        
        if not recent_history:
            return {}
        
        # Aggregate each metric type
        aggregate = {
            'num_queries': len(recent_history),
            'avg_latency': np.mean([h['metrics'].total_latency for h in recent_history]),
            'retrieval': {
                'avg_precision@5': np.mean([
                    h['metrics'].retrieval_metrics.precision_at_k.get(5, 0)
                    for h in recent_history
                ]),
                'avg_mrr': np.mean([
                    h['metrics'].retrieval_metrics.mrr
                    for h in recent_history
                ]),
                'avg_ndcg': np.mean([
                    h['metrics'].retrieval_metrics.ndcg
                    for h in recent_history
                ])
            },
            'context': {
                'avg_relevance': np.mean([
                    h['metrics'].context_metrics.relevance_score
                    for h in recent_history
                ]),
                'avg_diversity': np.mean([
                    h['metrics'].context_metrics.diversity_score
                    for h in recent_history
                ])
            },
            'answer': {
                'avg_factuality': np.mean([
                    h['metrics'].answer_metrics.factuality_score
                    for h in recent_history
                ]),
                'avg_fluency': np.mean([
                    h['metrics'].answer_metrics.fluency_score
                    for h in recent_history
                ])
            }
        }
        
        return aggregate
```

#### 4.2 Benchmark Datasets

```python
# New file: tldw_chatbook/RAG_Search/Services/benchmarks.py

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import yaml

@dataclass
class BenchmarkQuery:
    """A single benchmark query with ground truth."""
    id: str
    query: str
    relevant_documents: List[str]  # Document IDs
    expected_answer: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class BenchmarkDataset:
    """A collection of benchmark queries."""
    name: str
    description: str
    queries: List[BenchmarkQuery]
    metadata: Dict[str, Any] = None

class BenchmarkManager:
    """Manages benchmark datasets for RAG evaluation."""
    
    def __init__(self, benchmarks_dir: Path):
        self.benchmarks_dir = benchmarks_dir
        self.datasets = {}
        
    def load_dataset(self, name: str) -> BenchmarkDataset:
        """Load a benchmark dataset."""
        if name in self.datasets:
            return self.datasets[name]
        
        # Try different file formats
        for ext in ['.json', '.yaml', '.yml']:
            filepath = self.benchmarks_dir / f"{name}{ext}"
            if filepath.exists():
                with open(filepath, 'r') as f:
                    if ext == '.json':
                        data = json.load(f)
                    else:
                        data = yaml.safe_load(f)
                
                dataset = self._parse_dataset(data)
                self.datasets[name] = dataset
                return dataset
        
        raise ValueError(f"Benchmark dataset not found: {name}")
    
    def create_dataset(self, 
                      name: str,
                      description: str,
                      queries: List[Dict[str, Any]]) -> BenchmarkDataset:
        """Create a new benchmark dataset."""
        benchmark_queries = []
        for q in queries:
            benchmark_queries.append(BenchmarkQuery(
                id=q.get('id', f"q_{len(benchmark_queries)}"),
                query=q['query'],
                relevant_documents=q.get('relevant_documents', []),
                expected_answer=q.get('expected_answer'),
                metadata=q.get('metadata', {})
            ))
        
        dataset = BenchmarkDataset(
            name=name,
            description=description,
            queries=benchmark_queries
        )
        
        self.datasets[name] = dataset
        self._save_dataset(dataset)
        
        return dataset
    
    def run_benchmark(self,
                     dataset_name: str,
                     pipeline: RAGApplication,
                     metrics_collector: RAGMetricsCollector) -> Dict[str, Any]:
        """Run a benchmark on a pipeline."""
        dataset = self.load_dataset(dataset_name)
        results = []
        
        for query in dataset.queries:
            # Run pipeline
            response = pipeline.run(query.query)
            
            # Collect metrics
            metrics = metrics_collector.collect_metrics(
                query=query.query,
                search_results=response.search_results,
                context=response.context,
                response=response,
                latency_info=response.metadata.get('latency', {})
            )
            
            results.append({
                'query_id': query.id,
                'metrics': metrics,
                'expected_vs_actual': {
                    'expected_docs': query.relevant_documents,
                    'retrieved_docs': [doc.id for doc in response.sources],
                    'expected_answer': query.expected_answer,
                    'actual_answer': response.answer
                }
            })
        
        # Aggregate results
        return {
            'dataset': dataset_name,
            'pipeline': pipeline.config.get('pipeline_name', 'unknown'),
            'num_queries': len(results),
            'aggregate_metrics': metrics_collector.get_aggregate_metrics(),
            'detailed_results': results
        }
```

#### 4.3 Metrics Dashboard

```python
# New file: tldw_chatbook/UI/MetricsDashboard.py

from textual.app import ComposeResult
from textual.containers import Container, Grid, ScrollableContainer
from textual.widgets import Label, DataTable, Sparkline, ProgressBar
from textual.reactive import reactive
from typing import Dict, Any, List
import asyncio

class MetricsDashboard(Container):
    """Dashboard for displaying RAG metrics."""
    
    metrics_data = reactive({})
    
    def __init__(self, metrics_collector: RAGMetricsCollector):
        super().__init__()
        self.metrics_collector = metrics_collector
        self.update_interval = 5.0  # seconds
        
    def compose(self) -> ComposeResult:
        with Grid(id="metrics-grid"):
            # Summary cards
            yield self._create_metric_card(
                "avg_latency",
                "Avg Latency",
                "0.0s"
            )
            yield self._create_metric_card(
                "avg_precision",
                "Avg Precision@5",
                "0.0"
            )
            yield self._create_metric_card(
                "avg_relevance",
                "Avg Relevance",
                "0.0"
            )
            yield self._create_metric_card(
                "total_queries",
                "Total Queries",
                "0"
            )
            
            # Detailed metrics table
            with Container(id="detailed-metrics"):
                yield Label("Detailed Metrics", classes="section-header")
                yield DataTable(id="metrics-table")
            
            # Performance charts
            with Container(id="performance-charts"):
                yield Label("Performance Trends", classes="section-header")
                yield Sparkline(
                    [],
                    id="latency-trend",
                    title="Latency (ms)"
                )
                yield Sparkline(
                    [],
                    id="quality-trend",
                    title="Quality Score"
                )
            
            # Component breakdown
            with Container(id="component-breakdown"):
                yield Label("Component Performance", classes="section-header")
                yield ProgressBar(
                    total=100,
                    id="retrieval-performance"
                )
                yield ProgressBar(
                    total=100,
                    id="processing-performance"
                )
                yield ProgressBar(
                    total=100,
                    id="generation-performance"
                )
    
    def _create_metric_card(self, 
                           metric_id: str,
                           title: str,
                           initial_value: str) -> Container:
        """Create a metric display card."""
        return Container(
            Label(title, classes="metric-title"),
            Label(initial_value, id=f"{metric_id}-value", classes="metric-value"),
            id=f"{metric_id}-card",
            classes="metric-card"
        )
    
    async def on_mount(self):
        """Start metrics update loop when mounted."""
        self.update_task = asyncio.create_task(self._update_metrics_loop())
    
    async def _update_metrics_loop(self):
        """Continuously update metrics display."""
        while True:
            try:
                # Get latest metrics
                aggregate = self.metrics_collector.get_aggregate_metrics(last_n=100)
                
                # Update display
                await self._update_display(aggregate)
                
                # Wait before next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _update_display(self, metrics: Dict[str, Any]):
        """Update the dashboard display with new metrics."""
        # Update summary cards
        self.query_one("#avg_latency-value").update(
            f"{metrics.get('avg_latency', 0):.2f}s"
        )
        self.query_one("#avg_precision-value").update(
            f"{metrics.get('retrieval', {}).get('avg_precision@5', 0):.2%}"
        )
        self.query_one("#avg_relevance-value").update(
            f"{metrics.get('context', {}).get('avg_relevance', 0):.2%}"
        )
        self.query_one("#total_queries-value").update(
            str(metrics.get('num_queries', 0))
        )
        
        # Update detailed table
        table = self.query_one("#metrics-table", DataTable)
        table.clear()
        
        # Add metrics rows
        for category, values in metrics.items():
            if isinstance(values, dict):
                for metric, value in values.items():
                    table.add_row(category, metric, f"{value:.3f}")
```

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
1. Implement multi-modal data types and storage updates
2. Create pipeline configuration schema and builder
3. Set up basic metrics infrastructure

### Phase 2: Multi-modal Support (Weeks 3-4)
1. Integrate CLIP and audio embedding models
2. Update retrieval strategies for multi-modal search
3. Modify UI to support image/audio input
4. Test with sample multi-modal data

### Phase 3: Pipeline Management (Weeks 5-6)
1. Implement pipeline manager and registry
2. Add pipeline switching UI
3. Create pipeline versioning system
4. Build configuration hot-reload

### Phase 4: A/B Testing (Weeks 7-8)
1. Implement A/B test manager
2. Create comparison UI
3. Add statistical analysis
4. Build export functionality

### Phase 5: Metrics & Benchmarks (Weeks 9-10)
1. Implement all metric calculators
2. Create benchmark datasets
3. Build metrics dashboard
4. Add continuous monitoring

### Phase 6: Integration & Polish (Weeks 11-12)
1. Integrate all components
2. Performance optimization
3. Documentation
4. Testing and bug fixes

## Performance Considerations

Since tldw_chatbook is a single-user TUI application, we can optimize for:

1. **Lazy Loading**: Load models only when needed
2. **Model Caching**: Keep frequently used models in memory
3. **Local Storage**: Use SQLite and local ChromaDB efficiently
4. **Async Operations**: Non-blocking UI during long operations
5. **Resource Limits**: Configurable memory and CPU limits
6. **Incremental Updates**: Only process new/changed content

### Memory Management

```python
# Example memory-aware model loading
class ModelPool:
    def __init__(self, max_memory_gb: float = 4.0):
        self.max_memory = max_memory_gb * 1024 * 1024 * 1024
        self.loaded_models = {}
        self.model_sizes = {}
        
    def get_model(self, model_name: str):
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # Check if we need to evict models
        model_size = self._estimate_model_size(model_name)
        while self._current_memory_usage() + model_size > self.max_memory:
            self._evict_least_recently_used()
        
        # Load model
        model = self._load_model(model_name)
        self.loaded_models[model_name] = model
        self.model_sizes[model_name] = model_size
        
        return model
```

### UI Responsiveness

```python
# Example async operation with progress
async def index_multimodal_content(self, paths: List[Path]):
    progress = self.query_one("#indexing-progress", ProgressBar)
    progress.total = len(paths)
    
    for i, path in enumerate(paths):
        # Process in background
        await self._process_file(path)
        
        # Update UI
        progress.advance(1)
        self.query_one("#status").update(f"Processing: {path.name}")
        
        # Yield to UI thread
        await asyncio.sleep(0)
```

## Conclusion

This plan provides a comprehensive roadmap for implementing multi-modal RAG, custom pipeline configurations, A/B testing, and detailed metrics in the tldw_chatbook system. The modular architecture of the existing codebase makes these additions feasible while maintaining backward compatibility.

Key benefits:
- **Multi-modal RAG** enables searching across different content types
- **Pipeline configs** allow easy experimentation and customization
- **A/B testing** provides data-driven pipeline optimization
- **Comprehensive metrics** ensure quality and performance monitoring

The implementation is designed specifically for a single-user TUI application, optimizing for local performance and resource efficiency rather than distributed scalability.