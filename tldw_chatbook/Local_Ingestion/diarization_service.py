# diarization_service.py
"""
Speaker diarization service for tldw_chatbook.
Implements speaker identification using vector embeddings approach.
"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable, Tuple, TYPE_CHECKING
import json
from loguru import logger
from contextlib import contextmanager

# Type checking imports (not loaded at runtime)
if TYPE_CHECKING:
    import numpy as np
    import torch

# Check availability of required modules without importing
def _check_torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False

def _check_speechbrain_available():
    try:
        import speechbrain
        return True
    except ImportError:
        return False

def _check_sklearn_available():
    try:
        import sklearn
        return True
    except ImportError:
        return False

def _check_torchaudio_available():
    try:
        import torchaudio
        return True
    except ImportError:
        return False

# Module availability flags
TORCH_AVAILABLE = _check_torch_available()
SPEECHBRAIN_AVAILABLE = _check_speechbrain_available()
SKLEARN_AVAILABLE = _check_sklearn_available()
TORCHAUDIO_AVAILABLE = _check_torchaudio_available()

# Lazy-loaded modules (will be imported only when needed)
_torch = None
_numpy = None
_silero_vad_model = None
_silero_vad_utils = None
_speechbrain_encoder = None
_sklearn_modules = None
_torchaudio = None

# Local imports
from ..config import get_cli_setting


def _sanitize_path_component(name: str) -> str:
    """Sanitize a string to be safe for use as a directory/file name.
    
    Args:
        name: The string to sanitize
        
    Returns:
        A sanitized string safe for use in file paths
    """
    # Replace path separators and other unsafe characters
    safe_name = name.replace('/', '_').replace('\\', '_').replace(':', '_')
    safe_name = safe_name.replace('..', '_')  # Prevent directory traversal
    
    # Keep only alphanumeric, underscore, hyphen, and dot
    safe_name = ''.join(c if c.isalnum() or c in ('_', '-', '.') else '_' for c in safe_name)
    
    # Remove leading/trailing dots and underscores
    safe_name = safe_name.strip('._')
    
    # Ensure it's not empty
    if not safe_name:
        safe_name = 'model'
    
    return safe_name


def _lazy_import_torch():
    """Lazy import torch."""
    global _torch
    if _torch is None and TORCH_AVAILABLE:
        try:
            import torch
            _torch = torch
        except ImportError as e:
            logger.warning(f"Failed to import torch: {e}")
            _torch = None
    return _torch


def _lazy_import_numpy():
    """Lazy import numpy."""
    global _numpy
    if _numpy is None:
        try:
            import numpy
            _numpy = numpy
        except ImportError:
            logger.warning("NumPy not available")
            return None
    return _numpy


def _lazy_import_silero_vad():
    """Lazy import Silero VAD model.
    
    This function loads the Silero VAD model from torch hub.
    The model returns a tuple of (model, utils) where utils is a tuple of functions.
    
    Returns:
        tuple: (model, utils) or (None, None) if loading fails
        
    Note:
        The utils tuple order is critical and can break between versions:
        - utils[0]: get_speech_timestamps - Main VAD function
        - utils[1]: save_audio - Save audio to file
        - utils[2]: read_audio - Read audio from file  
        - utils[3]: VADIterator - Class for streaming VAD
        - utils[4]: collect_chunks - Collect speech chunks
    """
    global _silero_vad_model, _silero_vad_utils
    
    # Check if already loaded
    if _silero_vad_model is not None:
        return _silero_vad_model, _silero_vad_utils
    
    # Check torch availability
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available, cannot load Silero VAD")
        return None, None
    
    torch = _lazy_import_torch()
    if not torch:
        logger.warning("Failed to import torch for Silero VAD")
        return None, None
    
    try:
        logger.info("Loading Silero VAD model from torch hub...")
        
        # Set torch hub directory if not set
        default_hub_dir = str(Path.home() / '.cache' / 'torch' / 'hub')
        hub_dir = Path(os.environ.get('TORCH_HUB', default_hub_dir))
        hub_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model with explicit parameters
        result = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,  # Use cached version if available
            trust_repo=True,     # Required for loading
            verbose=False        # Reduce output noise
        )
        
        # Validate the result format
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            logger.error(
                f"Unexpected Silero VAD return format. Expected (model, utils) tuple, "
                f"got {type(result).__name__} with length {len(result) if hasattr(result, '__len__') else 'unknown'}"
            )
            return None, None
        
        model, utils = result
        
        # Validate model
        if model is None:
            logger.error("Silero VAD model is None")
            return None, None
        
        # Validate utils format
        if not isinstance(utils, (tuple, list)) or len(utils) < 5:
            logger.error(
                f"Unexpected Silero VAD utils format. Expected tuple/list with 5+ items, "
                f"got {type(utils).__name__} with {len(utils) if hasattr(utils, '__len__') else 'unknown'} items"
            )
            return None, None
        
        # Store globally for future use
        _silero_vad_model = model
        _silero_vad_utils = utils
        
        logger.info("Silero VAD loaded successfully")
        logger.debug(f"Silero VAD utils count: {len(utils)}")
        
        return model, utils
        
    except Exception as e:
        logger.error(f"Failed to load Silero VAD: {type(e).__name__}: {e}")
        logger.debug("Full error:", exc_info=True)
        
        # Reset globals on failure
        _silero_vad_model = None
        _silero_vad_utils = None
        
        return None, None


def _lazy_import_speechbrain():
    """Lazy import SpeechBrain encoder."""
    global _speechbrain_encoder
    if _speechbrain_encoder is None and SPEECHBRAIN_AVAILABLE:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            _speechbrain_encoder = EncoderClassifier
        except ImportError:
            try:
                # Fallback for older versions
                from speechbrain.pretrained import EncoderClassifier
                _speechbrain_encoder = EncoderClassifier
            except ImportError as e:
                logger.warning(f"Failed to import SpeechBrain EncoderClassifier: {e}")
                _speechbrain_encoder = None
    return _speechbrain_encoder


def _lazy_import_sklearn():
    """Lazy import sklearn modules."""
    global _sklearn_modules
    if _sklearn_modules is None and SKLEARN_AVAILABLE:
        try:
            from sklearn.cluster import SpectralClustering, AgglomerativeClustering
            from sklearn.preprocessing import normalize
            from sklearn.metrics import silhouette_score
            _sklearn_modules = {
                'SpectralClustering': SpectralClustering,
                'AgglomerativeClustering': AgglomerativeClustering,
                'normalize': normalize,
                'silhouette_score': silhouette_score
            }
        except ImportError as e:
            logger.warning(f"Failed to import sklearn modules: {e}")
            _sklearn_modules = None
    return _sklearn_modules


def _lazy_import_torchaudio():
    """Lazy import torchaudio."""
    global _torchaudio
    if _torchaudio is None and TORCHAUDIO_AVAILABLE:
        try:
            import torchaudio
            _torchaudio = torchaudio
        except ImportError as e:
            logger.warning(f"Failed to import torchaudio: {e}")
            _torchaudio = None
    return _torchaudio


class DiarizationError(Exception):
    """Base exception for diarization errors."""
    pass


class DiarizationService:
    """
    Speaker diarization service using vector embeddings approach.
    
    Pipeline:
    1. Voice Activity Detection (VAD) to find speech segments
    2. Split speech into fixed-length overlapping segments
    3. Extract speaker embeddings for each segment
    4. Cluster embeddings to identify speakers
    5. Merge consecutive segments from same speaker
    
    Attributes:
        is_available (bool): Whether all required dependencies are available.
                           Can be accessed directly or via is_diarization_available().
        config (dict): Configuration parameters for diarization.
    """
    
    def __init__(self):
        """Initialize the diarization service."""
        logger.info("Initializing DiarizationService...")
        
        # Configuration
        self.config = {
            # VAD settings
            'vad_threshold': get_cli_setting('diarization.vad_threshold', 0.5) or 0.5,
            'vad_min_speech_duration': get_cli_setting('diarization.vad_min_speech_duration', 0.25) or 0.25,
            'vad_min_silence_duration': get_cli_setting('diarization.vad_min_silence_duration', 0.25) or 0.25,
            
            # Segmentation settings
            'segment_duration': get_cli_setting('diarization.segment_duration', 2.0) or 2.0,
            'segment_overlap': get_cli_setting('diarization.segment_overlap', 0.5) or 0.5,
            'min_segment_duration': get_cli_setting('diarization.min_segment_duration', 1.0) or 1.0,
            'max_segment_duration': get_cli_setting('diarization.max_segment_duration', 3.0) or 3.0,
            
            # Embedding model
            'embedding_model': get_cli_setting('diarization.embedding_model', 'speechbrain/spkrec-ecapa-voxceleb') or 'speechbrain/spkrec-ecapa-voxceleb',
            'embedding_device': get_cli_setting('diarization.embedding_device', 'auto') or 'auto',
            
            # Clustering settings
            'clustering_method': get_cli_setting('diarization.clustering_method', 'spectral') or 'spectral',
            'similarity_threshold': get_cli_setting('diarization.similarity_threshold', 0.85) or 0.85,
            'min_speakers': get_cli_setting('diarization.min_speakers', 1) or 1,
            'max_speakers': get_cli_setting('diarization.max_speakers', 10) or 10,
            
            # Post-processing
            'merge_threshold': get_cli_setting('diarization.merge_threshold', 0.5) or 0.5,
            'min_speaker_duration': get_cli_setting('diarization.min_speaker_duration', 3.0) or 3.0,
        }
        
        logger.debug(f"Diarization service configuration: {self.config}")
        
        # Model storage (lazy loaded)
        self._vad_model = None
        self._vad_utils = None
        self._embedding_model = None
        self._model_lock = threading.RLock()
        
        # Check availability (without importing)
        # Public attribute - can be accessed directly by callers
        self.is_available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """Check if all required dependencies are available."""
        required = [
            (TORCH_AVAILABLE, "PyTorch"),
            (SPEECHBRAIN_AVAILABLE, "SpeechBrain"),
            (SKLEARN_AVAILABLE, "scikit-learn"),
        ]
        
        missing = [name for available, name in required if not available]
        if missing:
            logger.warning(f"Diarization unavailable. Missing: {', '.join(missing)}")
            return False
        
        return True
    
    def _get_device(self) -> str:
        """Determine the device to use for inference."""
        if self.config['embedding_device'] == 'auto':
            torch = _lazy_import_torch()
            if torch:
                try:
                    if hasattr(torch, 'cuda') and torch.cuda.is_available():
                        return 'cuda'
                except (AttributeError, RuntimeError) as e:
                    logger.debug(f"Error checking CUDA availability: {e}")
            return 'cpu'
        return self.config['embedding_device']
    
    def _load_embedding_model(self):
        """Load the speaker embedding model (lazy loading)."""
        with self._model_lock:
            if self._embedding_model is None:
                logger.info(f"Loading embedding model: {self.config['embedding_model']}")
                try:
                    EncoderClassifier = _lazy_import_speechbrain()
                    if not EncoderClassifier:
                        raise DiarizationError("SpeechBrain EncoderClassifier not available")
                    
                    device = self._get_device()
                    # Sanitize model name for safe directory creation
                    model_name = self.config['embedding_model']
                    safe_model_name = _sanitize_path_component(model_name)
                    
                    # Use pathlib for path construction
                    model_dir = Path('pretrained_models') / safe_model_name
                    
                    self._embedding_model = EncoderClassifier.from_hparams(
                        source=self.config['embedding_model'],
                        savedir=str(model_dir),
                        run_opts={"device": device}
                    )
                    logger.info(f"Embedding model loaded successfully on {device}")
                except Exception as e:
                    logger.error(f"Failed to load embedding model: {e}")
                    raise DiarizationError(f"Failed to load embedding model: {e}")
    
    def _load_vad_model(self):
        """Load the VAD model (lazy loading).
        
        This method loads the Silero VAD model and its utility functions.
        The VAD utilities are particularly brittle as they return as a tuple
        in a specific order that can change between versions.
        """
        if self._vad_model is None:
            try:
                model, utils = _lazy_import_silero_vad()
                if not model or not utils:
                    raise DiarizationError("Silero VAD model or utilities not available")
                
                # Validate that we have the expected utilities
                # Silero returns (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)
                # but this order is not guaranteed and can break between versions
                if not isinstance(utils, (list, tuple)) or len(utils) < 5:
                    raise DiarizationError(
                        f"Unexpected Silero VAD utils format. Expected tuple/list with 5+ items, "
                        f"got {type(utils).__name__} with {len(utils) if hasattr(utils, '__len__') else 'unknown'} items"
                    )
                
                # Store model
                self._vad_model = model
                
                # Map utilities with extensive validation
                # NOTE: This mapping is fragile and depends on Silero's return order
                try:
                    self._vad_utils = {
                        'get_speech_timestamps': utils[0],  # Main VAD function
                        'save_audio': utils[1],             # Audio saving utility
                        'read_audio': utils[2],             # Audio loading utility
                        'VADIterator': utils[3],            # Streaming VAD class
                        'collect_chunks': utils[4]          # Chunk collection utility
                    }
                    
                    # Validate that each utility is callable (except VADIterator which is a class)
                    for name, func in self._vad_utils.items():
                        if name != 'VADIterator' and not callable(func):
                            raise DiarizationError(f"VAD utility '{name}' is not callable")
                    
                    logger.debug("VAD utilities loaded and validated successfully")
                    
                except (IndexError, TypeError) as e:
                    raise DiarizationError(
                        f"Failed to map Silero VAD utilities. The utility order may have changed. Error: {e}"
                    )
                    
            except Exception as e:
                logger.error(f"Failed to load VAD model: {e}")
                self._vad_model = None
                self._vad_utils = None
                raise DiarizationError(f"Failed to load Silero VAD model: {e}")
    
    def _get_vad_utility(self, name: str) -> Callable:
        """Safely get a VAD utility function with validation.
        
        Args:
            name: Name of the utility ('get_speech_timestamps', 'read_audio', etc.)
            
        Returns:
            The utility function
            
        Raises:
            DiarizationError: If utility is not available or not callable
        """
        if not self._vad_utils:
            raise DiarizationError("VAD utilities not loaded")
            
        if name not in self._vad_utils:
            raise DiarizationError(
                f"VAD utility '{name}' not found. Available: {list(self._vad_utils.keys())}"
            )
            
        utility = self._vad_utils[name]
        
        # Special case for VADIterator which is a class, not a function
        if name == 'VADIterator':
            return utility
            
        if not callable(utility):
            raise DiarizationError(
                f"VAD utility '{name}' is not callable. Got type: {type(utility).__name__}"
            )
            
        return utility
    
    def diarize(
        self,
        audio_path: str,
        transcription_segments: Optional[List[Dict]] = None,
        num_speakers: Optional[int] = None,
        progress_callback: Optional[Callable[[float, str, Optional[Dict]], None]] = None
    ) -> Dict[str, Any]:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file (should be WAV format, 16kHz)
            transcription_segments: Optional transcription segments to align with
            num_speakers: Optional number of speakers (if known)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict with diarization results including segments with speaker IDs
        """
        if not self.is_available:
            raise DiarizationError("Diarization service is not available due to missing dependencies")
        
        start_time = time.time()
        logger.info(f"Starting diarization for: {audio_path}")
        
        try:
            # Load audio
            if progress_callback:
                progress_callback(0, "Loading audio file...", None)
            
            waveform = self._load_audio(audio_path)
            sample_rate = 16000  # Assuming 16kHz as standard
            
            # Step 1: Voice Activity Detection
            if progress_callback:
                progress_callback(10, "Detecting speech segments...", None)
            
            speech_timestamps = self._detect_speech(waveform, sample_rate)
            logger.info(f"Found {len(speech_timestamps)} speech segments")
            
            if not speech_timestamps:
                logger.warning("No speech detected in audio")
                return {
                    'segments': [],
                    'speakers': [],
                    'duration': len(waveform) / sample_rate,
                    'num_speakers': 0
                }
            
            # Step 2: Create overlapping segments
            if progress_callback:
                progress_callback(20, "Creating analysis segments...", None)
            
            segments = self._create_segments(waveform, speech_timestamps, sample_rate)
            logger.info(f"Created {len(segments)} analysis segments")
            
            # Step 3: Extract embeddings
            if progress_callback:
                progress_callback(30, "Extracting speaker embeddings...", None)
            
            embeddings = self._extract_embeddings(segments, progress_callback)
            logger.info(f"Extracted {len(embeddings)} embeddings")
            
            # Step 4: Cluster speakers
            if progress_callback:
                progress_callback(70, "Clustering speakers...", None)
            
            speaker_labels = self._cluster_speakers(
                embeddings, 
                num_speakers=num_speakers
            )
            
            # Count unique speakers
            unique_speakers = len(set(speaker_labels))
            logger.info(f"Identified {unique_speakers} speakers")
            
            # Step 5: Assign speakers to segments
            for segment, label in zip(segments, speaker_labels):
                segment['speaker_id'] = int(label)
                segment['speaker_label'] = f"SPEAKER_{label}"
            
            # Step 6: Merge consecutive segments
            if progress_callback:
                progress_callback(85, "Merging segments...", None)
            
            merged_segments = self._merge_segments(segments)
            
            # Step 7: Align with transcription if provided
            if transcription_segments:
                if progress_callback:
                    progress_callback(90, "Aligning with transcription...", None)
                
                aligned_segments = self._align_with_transcription(
                    merged_segments, 
                    transcription_segments
                )
            else:
                aligned_segments = merged_segments
            
            # Calculate speaker statistics
            speaker_stats = self._calculate_speaker_stats(aligned_segments)
            
            duration = time.time() - start_time
            logger.info(f"Diarization completed in {duration:.2f} seconds")
            
            if progress_callback:
                progress_callback(100, "Diarization complete", {
                    'num_speakers': unique_speakers,
                    'duration': duration
                })
            
            return {
                'segments': aligned_segments,
                'speakers': speaker_stats,
                'duration': len(waveform) / sample_rate,
                'num_speakers': unique_speakers,
                'processing_time': duration
            }
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}", exc_info=True)
            raise DiarizationError(f"Diarization failed: {str(e)}")
    
    def _load_audio(self, audio_path: str):
        """Load audio file and convert to correct format."""
        torchaudio = _lazy_import_torchaudio()
        torch = _lazy_import_torch()
        
        if torchaudio and torch:
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                return waveform.squeeze()
            except Exception as e:
                logger.warning(f"Failed to load audio with torchaudio: {e}")
                # Fall through to Silero VAD fallback
        else:
            # Fallback to read_audio from Silero VAD utilities
            logger.info("Falling back to Silero VAD read_audio function")
            
            # Ensure VAD utilities are loaded
            if not self._vad_utils:
                try:
                    self._load_vad_model()
                except Exception as e:
                    logger.error(f"Failed to load VAD model for audio reading: {e}")
                    raise DiarizationError(f"Cannot load audio: VAD model load failed: {e}")
            
            # Validate read_audio function exists and is callable
            if not self._vad_utils or 'read_audio' not in self._vad_utils:
                raise DiarizationError(
                    "VAD utilities missing 'read_audio' function. "
                    "Neither torchaudio nor Silero VAD audio loading available."
                )
            
            # Get read_audio function using safe getter
            read_audio = self._get_vad_utility('read_audio')
            
            try:
                # Call read_audio with proper parameters
                # NOTE: Silero's read_audio expects 'sampling_rate' not 'sample_rate'
                waveform = read_audio(audio_path, sampling_rate=16000)
                
                # Validate the loaded waveform
                if waveform is None:
                    raise DiarizationError("read_audio returned None")
                    
                return waveform
                
            except Exception as e:
                logger.error(f"Failed to load audio with Silero read_audio: {e}")
                raise DiarizationError(
                    f"Failed to load audio file '{audio_path}': {str(e)}"
                )
    
    def _detect_speech(self, waveform, sample_rate: int) -> List[Dict]:
        """Detect speech segments using VAD.
        
        Args:
            waveform: Audio waveform tensor
            sample_rate: Sample rate of the audio
            
        Returns:
            List of speech segments with start/end times
            
        Raises:
            DiarizationError: If VAD fails or utilities are not properly loaded
        """
        # Ensure VAD model is loaded
        if not self._vad_model:
            self._load_vad_model()
        
        # Validate VAD utilities are loaded
        if not self._vad_utils or 'get_speech_timestamps' not in self._vad_utils:
            raise DiarizationError(
                "VAD utilities not properly loaded. Missing 'get_speech_timestamps' function."
            )
        
        # Get the speech detection function using safe getter
        get_speech_timestamps = self._get_vad_utility('get_speech_timestamps')
        
        try:
            # Call the VAD function with proper parameters
            # NOTE: Parameter names and order are critical for Silero VAD
            speech_timestamps = get_speech_timestamps(
                waveform,
                self._vad_model,
                sampling_rate=sample_rate,  # Must be 'sampling_rate', not 'sample_rate'
                threshold=self.config['vad_threshold'],
                min_speech_duration_ms=int(self.config['vad_min_speech_duration'] * 1000),
                min_silence_duration_ms=int(self.config['vad_min_silence_duration'] * 1000)
            )
            
            # Validate the output format
            if not isinstance(speech_timestamps, list):
                raise DiarizationError(
                    f"Expected list of timestamps, got {type(speech_timestamps).__name__}"
                )
            
            # Validate each timestamp has required fields
            for i, ts in enumerate(speech_timestamps):
                if not isinstance(ts, dict):
                    raise DiarizationError(
                        f"Timestamp {i} is not a dict: {type(ts).__name__}"
                    )
                if 'start' not in ts or 'end' not in ts:
                    raise DiarizationError(
                        f"Timestamp {i} missing 'start' or 'end' field: {ts.keys()}"
                    )
                    
        except Exception as e:
            logger.error(f"VAD detection failed: {e}")
            raise DiarizationError(f"Speech detection failed: {str(e)}")
        
        # Convert to seconds
        for ts in speech_timestamps:
            ts['start'] = ts['start'] / sample_rate
            ts['end'] = ts['end'] / sample_rate
        
        return speech_timestamps
    
    def _create_segments(
        self, 
        waveform: "torch.Tensor", 
        speech_timestamps: List[Dict],
        sample_rate: int
    ) -> List[Dict]:
        """Create fixed-length overlapping segments from speech regions."""
        segments = []
        segment_samples = int(self.config['segment_duration'] * sample_rate)
        overlap_samples = int(self.config['segment_overlap'] * sample_rate)
        step_samples = segment_samples - overlap_samples
        
        for speech in speech_timestamps:
            start_sample = int(speech['start'] * sample_rate)
            end_sample = int(speech['end'] * sample_rate)
            
            # Create overlapping segments within this speech region
            for i in range(start_sample, end_sample - segment_samples + 1, step_samples):
                segment_waveform = waveform[i:i + segment_samples]
                
                segments.append({
                    'start': i / sample_rate,
                    'end': (i + segment_samples) / sample_rate,
                    'waveform': segment_waveform,
                    'speech_region': speech  # Keep reference to original speech region
                })
        
        return segments
    
    def _extract_embeddings(
        self, 
        segments: List[Dict],
        progress_callback: Optional[Callable] = None
    ) -> "np.ndarray":
        """Extract speaker embeddings for each segment."""
        # Load embedding model if not already loaded
        self._load_embedding_model()
        
        embeddings = []
        total_segments = len(segments)
        
        for i, segment in enumerate(segments):
            # Convert waveform to correct format
            waveform = segment['waveform'].unsqueeze(0)
            
            # Extract embedding
            torch = _lazy_import_torch()
            try:
                if torch and hasattr(torch, 'no_grad'):
                    try:
                        with torch.no_grad():
                            embedding = self._embedding_model.encode_batch(waveform)
                    except (AttributeError, RuntimeError) as e:
                        logger.debug(f"Failed to use torch.no_grad(): {e}")
                        # Fallback without no_grad context
                        embedding = self._embedding_model.encode_batch(waveform)
                else:
                    # No torch available or no_grad not available, run without context manager
                    embedding = self._embedding_model.encode_batch(waveform)
            except Exception as e:
                logger.error(f"Failed to extract embedding for segment {i}: {e}")
                raise DiarizationError(f"Embedding extraction failed: {e}")
            
            # Convert to numpy and flatten
            embedding = embedding.squeeze().cpu().numpy()
            embeddings.append(embedding)
            
            # Progress update
            if progress_callback and i % 10 == 0:
                progress = 30 + (40 * i / total_segments)  # 30-70% range
                progress_callback(
                    progress, 
                    f"Processing segment {i+1}/{total_segments}",
                    {'current': i+1, 'total': total_segments}
                )
        
        np = _lazy_import_numpy()
        if not np:
            raise DiarizationError("NumPy not available for creating embedding array")
        
        try:
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Failed to create numpy array from embeddings: {e}")
            raise DiarizationError(f"Failed to create embedding array: {e}")
    
    def _cluster_speakers(
        self, 
        embeddings: "np.ndarray",
        num_speakers: Optional[int] = None
    ) -> "np.ndarray":
        """Cluster embeddings to identify speakers."""
        sklearn_modules = _lazy_import_sklearn()
        if not sklearn_modules:
            raise DiarizationError("scikit-learn modules not available for clustering")
        
        # Normalize embeddings
        normalize = sklearn_modules['normalize']
        embeddings = normalize(embeddings, axis=1, norm='l2')
        
        if num_speakers is None:
            # Estimate number of speakers
            num_speakers = self._estimate_num_speakers(embeddings)
            logger.info(f"Estimated {num_speakers} speakers")
        
        # Ensure num_speakers is within bounds
        num_speakers = max(self.config['min_speakers'], 
                          min(num_speakers, self.config['max_speakers']))
        
        if self.config['clustering_method'] == 'spectral':
            SpectralClustering = sklearn_modules['SpectralClustering']
            clustering = SpectralClustering(
                n_clusters=num_speakers,
                affinity='cosine',
                assign_labels='kmeans',
                random_state=42
            )
        else:  # agglomerative
            AgglomerativeClustering = sklearn_modules['AgglomerativeClustering']
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers,
                affinity='cosine',
                linkage='average'
            )
        
        labels = clustering.fit_predict(embeddings)
        return labels
    
    def _estimate_num_speakers(self, embeddings: "np.ndarray") -> int:
        """Estimate the number of speakers using silhouette analysis."""
        sklearn_modules = _lazy_import_sklearn()
        if not sklearn_modules:
            # Default to 2 speakers if sklearn not available
            return 2
        
        max_score = -1
        best_n = 2
        
        SpectralClustering = sklearn_modules['SpectralClustering']
        silhouette_score = sklearn_modules['silhouette_score']
        
        # Try different numbers of speakers
        for n in range(2, min(len(embeddings), self.config['max_speakers'] + 1)):
            try:
                clustering = SpectralClustering(
                    n_clusters=n,
                    affinity='cosine',
                    assign_labels='kmeans',
                    random_state=42
                )
                labels = clustering.fit_predict(embeddings)
                
                # Calculate silhouette score
                score = silhouette_score(embeddings, labels, metric='cosine')
                
                if score > max_score:
                    max_score = score
                    best_n = n
                    
            except Exception as e:
                logger.warning(f"Failed to test {n} speakers: {e}")
        
        return best_n
    
    def _merge_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge consecutive segments from the same speaker."""
        if not segments:
            return []
        
        # Sort by start time
        segments = sorted(segments, key=lambda x: x['start'])
        
        merged = []
        current = segments[0].copy()
        
        for segment in segments[1:]:
            # Check if same speaker and close enough in time
            same_speaker = segment['speaker_id'] == current['speaker_id']
            close_enough = segment['start'] - current['end'] <= self.config['merge_threshold']
            
            if same_speaker and close_enough:
                # Extend current segment
                current['end'] = segment['end']
            else:
                # Save current and start new
                merged.append(current)
                current = segment.copy()
        
        # Don't forget the last segment
        merged.append(current)
        
        return merged
    
    def _align_with_transcription(
        self,
        diarization_segments: List[Dict],
        transcription_segments: List[Dict]
    ) -> List[Dict]:
        """Align diarization results with transcription segments."""
        aligned = []
        
        for trans_seg in transcription_segments:
            # Find overlapping diarization segments
            overlaps = []
            
            for diar_seg in diarization_segments:
                # Check for overlap
                overlap_start = max(trans_seg['start'], diar_seg['start'])
                overlap_end = min(trans_seg['end'], diar_seg['end'])
                
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    overlaps.append((diar_seg['speaker_id'], overlap_duration))
            
            # Assign speaker based on maximum overlap
            if overlaps:
                # Sort by overlap duration
                overlaps.sort(key=lambda x: x[1], reverse=True)
                speaker_id = overlaps[0][0]
                
                # Create aligned segment
                aligned_seg = trans_seg.copy()
                aligned_seg['speaker_id'] = speaker_id
                aligned_seg['speaker_label'] = f"SPEAKER_{speaker_id}"
                aligned.append(aligned_seg)
            else:
                # No overlap found, keep original
                aligned.append(trans_seg)
        
        return aligned
    
    def _calculate_speaker_stats(self, segments: List[Dict]) -> List[Dict]:
        """Calculate statistics for each speaker."""
        speaker_times = {}
        
        for segment in segments:
            speaker_id = segment.get('speaker_id', -1)
            duration = segment['end'] - segment['start']
            
            if speaker_id not in speaker_times:
                speaker_times[speaker_id] = {
                    'total_time': 0,
                    'segment_count': 0,
                    'first_appearance': segment['start'],
                    'last_appearance': segment['end']
                }
            
            stats = speaker_times[speaker_id]
            stats['total_time'] += duration
            stats['segment_count'] += 1
            stats['last_appearance'] = segment['end']
        
        # Convert to list format
        speakers = []
        for speaker_id, stats in speaker_times.items():
            speakers.append({
                'speaker_id': speaker_id,
                'speaker_label': f"SPEAKER_{speaker_id}",
                'total_time': stats['total_time'],
                'segment_count': stats['segment_count'],
                'first_appearance': stats['first_appearance'],
                'last_appearance': stats['last_appearance']
            })
        
        # Sort by total time (most talkative first)
        speakers.sort(key=lambda x: x['total_time'], reverse=True)
        
        return speakers
    
    def is_diarization_available(self) -> bool:
        """Check if diarization is available.
        
        Returns:
            bool: True if all required dependencies are available
            
        Note:
            You can also directly access the `is_available` attribute
            for the same information.
        """
        return self.is_available
    
    def get_requirements(self) -> Dict[str, bool]:
        """Get the status of required dependencies."""
        return {
            'torch': TORCH_AVAILABLE,
            'speechbrain': SPEECHBRAIN_AVAILABLE,
            'sklearn': SKLEARN_AVAILABLE,
            'torchaudio': TORCHAUDIO_AVAILABLE
        }