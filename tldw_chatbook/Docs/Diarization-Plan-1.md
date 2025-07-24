# Speaker Diarization Implementation Plan for tldw_chatbook

## Executive Summary

This document outlines the comprehensive plan for implementing speaker diarization in tldw_chatbook using a modern vector embeddings approach. The implementation will enhance the existing transcription backends (faster-whisper and nemo/parakeet/canary) with the ability to identify and label different speakers in audio content.

## Table of Contents
1. [Overview](#overview)
2. [Architecture Design](#architecture-design)
3. [Implementation Phases](#implementation-phases)
4. [Architecture Decision Records (ADRs)](#architecture-decision-records-adrs)
5. [Technical Specifications](#technical-specifications)
6. [Integration Points](#integration-points)
7. [Testing Strategy](#testing-strategy)
8. [Performance Considerations](#performance-considerations)
9. [Future Enhancements](#future-enhancements)

## Overview

### Goals
- Add speaker diarization capability to existing transcription services
- Implement using vector embeddings approach for flexibility and maintainability
- Maintain backward compatibility with existing transcription workflows
- Provide configurable and optional diarization features
- Support both real-time and batch processing scenarios

### Non-Goals
- Replace existing transcription functionality
- Require diarization for all transcriptions
- Support video-based speaker identification
- Implement custom embedding models (will use pre-trained)

## Architecture Design

### High-Level Flow

```
Audio Input → VAD → Segmentation → Embedding Extraction → Clustering → Speaker Assignment → Output
     ↓                                      ↓                    ↓
[Silero VAD]                    [SpeechBrain ECAPA]    [Spectral Clustering]
```

### Component Architecture

```
TranscriptionService (existing)
    ├── DiarizationService (new)
    │   ├── VADProcessor
    │   ├── EmbeddingExtractor
    │   ├── SpeakerClusterer
    │   └── SegmentMerger
    ├── Faster-Whisper Backend
    ├── Parakeet/Canary Backend
    └── Other Backends...
```

### Data Flow

1. **Input**: Raw audio file (WAV format, 16kHz)
2. **VAD Processing**: Extract speech segments, filter silence
3. **Segmentation**: Create overlapping segments (1.5-3s duration)
4. **Embedding**: Generate 192-dimensional speaker vectors
5. **Clustering**: Group embeddings by speaker similarity
6. **Alignment**: Map speaker IDs to transcription segments
7. **Output**: Enhanced transcription with speaker labels

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)

#### 1.1 Create DiarizationService Module
```python
# tldw_chatbook/Local_Ingestion/diarization_service.py
class DiarizationService:
    def __init__(self, config: Dict[str, Any]):
        self.vad_model = None
        self.embedding_model = None
        self.config = config
        self._model_cache = {}
        self._model_lock = threading.RLock()
    
    def diarize(self, audio_path: str, 
                transcription_segments: List[Dict],
                progress_callback: Optional[Callable] = None) -> List[Dict]:
        """Main diarization pipeline"""
        pass
```

#### 1.2 VAD Integration
- Integrate Silero VAD for speech detection
- Implement configurable parameters (threshold, min_duration)
- Add progress reporting for long files

#### 1.3 Model Management
- Implement lazy loading for models
- Add model caching similar to TranscriptionService
- Support both CPU and GPU inference

### Phase 2: Embedding & Clustering (Week 2)

#### 2.1 Speaker Embedding Extraction
```python
def extract_embeddings(self, audio_segments: List[Dict]) -> np.ndarray:
    """Extract speaker embeddings using SpeechBrain ECAPA-TDNN"""
    embeddings = []
    for segment in audio_segments:
        # Process segment waveform
        embedding = self.embedding_model.encode_batch(segment['waveform'])
        embeddings.append(embedding)
    return np.array(embeddings)
```

#### 2.2 Clustering Implementation
- Spectral clustering with cosine similarity
- Automatic speaker count estimation
- Support for min/max speaker constraints

#### 2.3 Segment Processing
- Implement overlapping segment creation
- Add segment merging for same-speaker sequences
- Handle edge cases (single speaker, silence)

### Phase 3: Integration (Week 3)

#### 3.1 TranscriptionService Updates
```python
def transcribe(self, audio_path: str, ..., diarize: bool = False):
    # Existing transcription logic
    result = self._transcribe_with_provider(...)
    
    if diarize and result.get('segments'):
        # Apply diarization
        diarization_result = self._diarization_service.diarize(
            audio_path=audio_path,
            transcription_segments=result['segments'],
            progress_callback=progress_callback
        )
        result['segments'] = diarization_result
        result['diarization_performed'] = True
    
    return result
```

#### 3.2 Configuration Updates
- Add diarization section to config.toml
- Implement settings validation
- Add feature flags for gradual rollout

### Phase 4: UI & Export (Week 4)

#### 4.1 Display Enhancements
- Update transcript display widgets
- Add speaker labels and color coding
- Implement speaker statistics view

#### 4.2 Export Formats
- Enhance JSON export with speaker info
- Update text export with speaker prefixes
- Add SRT/VTT support with speaker labels

## Architecture Decision Records (ADRs)

### ADR-001: Vector Embeddings Approach

**Status**: Accepted  
**Date**: 2025-07-23

**Context**
- Traditional diarization libraries (pyannote) have complex dependencies
- Need flexible solution that integrates with existing ML pipeline
- Want to avoid version conflicts with other components

**Decision**
Use vector embeddings approach with:
- Silero VAD for voice activity detection
- SpeechBrain ECAPA-TDNN for speaker embeddings
- Spectral clustering for speaker grouping

**Consequences**
- ✅ More control over the pipeline
- ✅ Better integration with existing infrastructure
- ✅ Easier to maintain and debug
- ❌ Need to implement clustering logic
- ❌ Potentially lower accuracy than specialized solutions

### ADR-002: Model Selection

**Status**: Accepted  
**Date**: 2025-07-23

**Context**
- Need reliable pre-trained models
- Must support both CPU and GPU
- Should have reasonable model sizes

**Decision**
- **VAD**: Silero VAD (8MB model, high accuracy)
- **Embeddings**: SpeechBrain spkrec-ecapa-voxceleb (17MB model)
- **Clustering**: Scikit-learn spectral clustering

**Consequences**
- ✅ Small model sizes, fast loading
- ✅ Good accuracy for general use cases
- ✅ Active maintenance and community
- ❌ May need different models for specific domains

### ADR-003: Progressive Implementation

**Status**: Accepted  
**Date**: 2025-07-23

**Context**
- Existing users rely on current transcription
- Diarization adds computational overhead
- Not all use cases need speaker identification

**Decision**
- Implement as opt-in feature
- Default to disabled in configuration
- Graceful fallback if dependencies missing
- Per-transcription toggle option

**Consequences**
- ✅ No breaking changes
- ✅ Users can adopt gradually
- ✅ Performance impact only when needed
- ❌ Additional code complexity

### ADR-004: Storage Strategy

**Status**: Accepted  
**Date**: 2025-07-23

**Context**
- Need to store speaker information
- Must maintain backward compatibility
- Should support export/import

**Decision**
- Store speaker_id in segment JSON structure
- No schema changes to database
- Optional fields for compatibility

**Consequences**
- ✅ No database migration required
- ✅ Backward compatible
- ✅ Easy to export/share
- ❌ Slightly larger JSON payloads

### ADR-005: Clustering Method

**Status**: Accepted  
**Date**: 2025-07-23

**Context**
- Unknown number of speakers in most cases
- Need automatic speaker count estimation
- Must handle various audio qualities

**Decision**
Use spectral clustering with:
- Cosine similarity for affinity matrix
- Eigen-gap method for speaker count estimation
- Configurable min/max speaker bounds

**Consequences**
- ✅ Automatic speaker detection
- ✅ Works well with embeddings
- ✅ Handles non-convex clusters
- ❌ More complex than k-means
- ❌ Requires similarity computation

## Technical Specifications

### Dependencies

```toml
# pyproject.toml additions
[project.optional-dependencies]
diarization = [
    "silero-vad>=4.0",
    "speechbrain>=0.5.16",
    "scikit-learn>=1.3.0",  # Already in base deps
    "numpy>=1.24.0",        # Already in base deps
]
```

### Configuration Schema

```toml
# ~/.config/tldw_cli/config.toml
[diarization]
# Enable diarization by default
enabled = false

# VAD settings
vad_threshold = 0.5
vad_min_speech_duration = 0.25
vad_min_silence_duration = 0.25

# Segmentation settings
segment_duration = 2.0
segment_overlap = 0.5
min_segment_duration = 1.0
max_segment_duration = 3.0

# Embedding model
embedding_model = "speechbrain/spkrec-ecapa-voxceleb"
embedding_device = "auto"  # auto, cuda, cpu

# Clustering settings
clustering_method = "spectral"
similarity_threshold = 0.85
min_speakers = 1
max_speakers = 10

# Post-processing
merge_threshold = 0.5  # seconds between segments to merge
min_speaker_duration = 3.0  # minimum total duration per speaker
```

### Segment Data Structure

```python
# Enhanced segment structure
{
    "start": 1.5,
    "end": 4.2,
    "text": "Hello, how are you today?",
    "speaker_id": 0,              # New field
    "speaker_label": "SPEAKER_0",  # New field
    "confidence": 0.92,           # Optional
    
    # Legacy fields maintained
    "Time_Start": 1.5,
    "Time_End": 4.2,
    "Text": "Hello, how are you today?"
}
```

### API Examples

```python
# Basic usage
result = transcription_service.transcribe(
    audio_path="meeting.wav",
    provider="faster-whisper",
    diarize=True
)

# With progress callback
def progress_handler(progress: float, status: str, data: Optional[Dict]):
    if data and 'diarization_stage' in data:
        print(f"Diarization: {data['diarization_stage']} - {progress}%")

result = transcription_service.transcribe(
    audio_path="podcast.mp3",
    provider="parakeet",
    diarize=True,
    progress_callback=progress_handler
)

# Custom diarization settings
result = transcription_service.transcribe(
    audio_path="interview.wav",
    diarize=True,
    diarization_config={
        "min_speakers": 2,
        "max_speakers": 3,
        "clustering_method": "spectral"
    }
)
```

## Integration Points

### 1. TranscriptionService
- Add `_diarization_service` instance
- Modify `transcribe()` method
- Update progress reporting

### 2. Audio Processing
- No changes needed (uses same WAV format)
- Leverage existing audio loading

### 3. Database
- No schema changes required
- Speaker info stored in JSON segments
- Optional metadata table for future

### 4. UI Components
- Update `chat_message_enhanced.py` for speaker display
- Modify transcript export dialogs
- Add speaker color preferences

### 5. Configuration
- Update `config.py` with diarization defaults
- Add validation for new settings
- Document in CLAUDE.md

## Testing Strategy

### Unit Tests
```python
# Tests/Diarization/test_diarization_service.py
def test_vad_processing():
    """Test VAD extracts speech segments correctly"""
    
def test_embedding_extraction():
    """Test embedding generation for audio segments"""
    
def test_clustering_accuracy():
    """Test speaker clustering with known data"""
    
def test_segment_merging():
    """Test consecutive segment merging logic"""
```

### Integration Tests
```python
# Tests/Diarization/test_diarization_integration.py
def test_transcription_with_diarization():
    """Test full pipeline with real audio"""
    
def test_progress_reporting():
    """Test progress callbacks during diarization"""
    
def test_fallback_behavior():
    """Test graceful degradation without deps"""
```

### Performance Tests
- Benchmark on various audio lengths
- Memory usage profiling
- GPU vs CPU performance comparison

## Performance Considerations

### Memory Usage
- Embedding model: ~17MB
- VAD model: ~8MB
- Audio chunks: ~10MB per minute (16kHz mono)
- Embeddings: ~1MB per 100 segments

### Processing Time (estimates)
- VAD: ~0.1x real-time
- Embedding extraction: ~0.5x real-time
- Clustering: O(n²) for n segments
- Total: ~1-2x real-time on GPU

### Optimization Strategies
1. **Batch Processing**: Process multiple segments together
2. **Model Caching**: Keep models loaded between calls
3. **Parallel Processing**: VAD and embedding in parallel
4. **Chunking**: Process long audio in chunks
5. **GPU Utilization**: Automatic GPU detection

## Future Enhancements

### Phase 5: Advanced Features
1. **Multi-language Support**
   - Language-specific embedding models
   - Cross-lingual speaker tracking

2. **Real-time Diarization**
   - Streaming audio support
   - Online clustering algorithms

3. **Speaker Recognition**
   - Known speaker enrollment
   - Speaker database integration

4. **Quality Improvements**
   - Overlapped speech handling
   - Background noise robustness
   - Acoustic condition adaptation

### Phase 6: Integration Enhancements
1. **RAG Integration**
   - Speaker-aware chunking
   - Speaker-based retrieval

2. **Export Formats**
   - RTTM format for research
   - Audacity labels
   - Transcription APIs compatibility

3. **Analytics**
   - Speaker talk time statistics
   - Conversation dynamics analysis
   - Meeting insights

## Conclusion

This implementation plan provides a robust foundation for adding speaker diarization to tldw_chatbook. The modular design ensures maintainability, while the progressive implementation approach minimizes risk. The use of modern embedding-based techniques offers flexibility and room for future enhancements while maintaining compatibility with the existing transcription infrastructure.

## Appendix: Example Implementation

### Sample Diarization Pipeline
```python
import torch
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize

class SimpleDiarizationPipeline:
    def __init__(self):
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
        self.get_speech_timestamps = utils[0]
        
    def process(self, audio_path: str) -> List[Dict]:
        # Load audio
        wav = self.load_audio(audio_path)
        
        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            wav, self.vad_model, sampling_rate=16000
        )
        
        # Create segments
        segments = self.create_segments(wav, speech_timestamps)
        
        # Extract embeddings
        embeddings = self.extract_embeddings(segments)
        
        # Cluster speakers
        labels = self.cluster_speakers(embeddings)
        
        # Assign labels to segments
        for segment, label in zip(segments, labels):
            segment['speaker_id'] = int(label)
            
        return segments
```

This plan provides a clear roadmap for implementing speaker diarization while maintaining the quality and architectural integrity of the tldw_chatbook project.

## Implementation Status

### Completed Items
✅ Core diarization service module (`diarization_service.py`)
✅ VAD integration with Silero
✅ Speaker embedding extraction with SpeechBrain
✅ Clustering and segment merging algorithms
✅ TranscriptionService integration
✅ Configuration in config.toml
✅ Unit and integration tests
✅ Example usage scripts

### Usage Examples

#### Basic Usage
```python
from tldw_chatbook.Local_Ingestion.transcription_service import TranscriptionService

service = TranscriptionService()

# Transcribe with speaker diarization
result = service.transcribe(
    audio_path="meeting.wav",
    provider="faster-whisper",
    diarize=True  # Enable speaker identification
)

# Access results
print(f"Speakers found: {result['num_speakers']}")
for segment in result['segments']:
    print(f"{segment['speaker_label']}: {segment['text']}")
```

#### With Custom Settings
```python
result = service.transcribe(
    audio_path="interview.wav",
    diarize=True,
    num_speakers=2,  # If known
    diarization_config={
        'min_speakers': 2,
        'max_speakers': 3,
        'clustering_method': 'spectral'
    }
)
```

#### Configuration File
Add to `~/.config/tldw_cli/config.toml`:
```toml
[diarization]
enabled = true
embedding_model = "speechbrain/spkrec-ecapa-voxceleb"
clustering_method = "spectral"
min_speakers = 1
max_speakers = 10
```

### Installation

The diarization feature requires additional dependencies:

```bash
# Install with diarization support
pip install -e ".[diarization]"

# This installs:
# - torch (PyTorch)
# - speechbrain
# - silero-vad (included with torch.hub)
```

### Next Steps

1. **Performance Optimization**
   - Implement batch processing for embeddings
   - Add GPU memory management
   - Cache VAD results for repeated processing

2. **Enhanced Features**
   - Overlapped speech detection
   - Speaker change detection
   - Real-time diarization support

3. **UI Integration**
   - Add speaker colors in transcript display
   - Speaker statistics dashboard
   - Export with speaker labels

4. **Advanced Capabilities**
   - Speaker enrollment/recognition
   - Cross-file speaker tracking
   - Emotion/sentiment per speaker