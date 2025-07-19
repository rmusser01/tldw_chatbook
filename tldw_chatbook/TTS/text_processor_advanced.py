# text_processor_advanced.py
# Description: Advanced text processing for TTS with SSML support and intelligent formatting
#
# Imports
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger

# Third-party imports
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    detect = None
    LangDetectException = None

#######################################################################################################################
#
# SSML Support Classes

class SSMLTag(Enum):
    """Supported SSML tags"""
    SPEAK = "speak"
    PROSODY = "prosody"
    EMPHASIS = "emphasis"
    BREAK = "break"
    SAY_AS = "say-as"
    PHONEME = "phoneme"
    SUB = "sub"
    LANG = "lang"
    VOICE = "voice"
    AUDIO = "audio"
    PARAGRAPH = "p"
    SENTENCE = "s"

@dataclass
class PronunciationRule:
    """Custom pronunciation rule"""
    text: str
    phoneme: str
    ipa: Optional[str] = None
    context: Optional[str] = None  # Regex pattern for context

@dataclass
class EmotionMarker:
    """Emotion or emphasis marker"""
    pattern: str  # Regex pattern to match
    emotion: str  # Emotion type: happy, sad, excited, etc.
    intensity: float  # 0.0 to 1.0

#######################################################################################################################
#
# Advanced Text Processor

class AdvancedTextProcessor:
    """
    Advanced text processor with SSML support and intelligent formatting.
    
    Features:
    - SSML generation for supported TTS providers
    - Automatic punctuation and formatting cleanup
    - Intelligent sentence breaking for natural pauses
    - Pronunciation dictionary with context awareness
    - Language detection and switching
    - Emotion and emphasis markup support
    - Number and date formatting
    - Abbreviation expansion
    """
    
    def __init__(self):
        # Common abbreviations to expand
        self.abbreviations = {
            "Mr.": "Mister",
            "Mrs.": "Misses",
            "Ms.": "Miss",
            "Dr.": "Doctor",
            "Prof.": "Professor",
            "St.": "Street",
            "Ave.": "Avenue",
            "Blvd.": "Boulevard",
            "Ltd.": "Limited",
            "Inc.": "Incorporated",
            "Corp.": "Corporation",
            "Co.": "Company",
            "vs.": "versus",
            "etc.": "et cetera",
            "i.e.": "that is",
            "e.g.": "for example",
            "Ph.D.": "Doctor of Philosophy",
            "M.D.": "Doctor of Medicine",
            "B.A.": "Bachelor of Arts",
            "M.A.": "Master of Arts",
            "B.S.": "Bachelor of Science",
            "M.S.": "Master of Science",
            "Jr.": "Junior",
            "Sr.": "Senior",
            "U.S.": "United States",
            "U.K.": "United Kingdom",
            "U.N.": "United Nations",
            "A.M.": "AM",
            "P.M.": "PM",
            "a.m.": "AM",
            "p.m.": "PM",
        }
        
        # Pronunciation dictionary
        self.pronunciation_rules: List[PronunciationRule] = [
            PronunciationRule("GIF", "jif", "dʒɪf"),
            PronunciationRule("SQL", "sequel", "ˈsiːkwəl"),
            PronunciationRule("JSON", "jason", "ˈdʒeɪsən"),
            PronunciationRule("UUID", "you-you-eye-dee", "juːjuːaɪdiː"),
            PronunciationRule("OAuth", "oh-auth", "oʊɔːθ"),
            PronunciationRule("WiFi", "why-fie", "waɪfaɪ"),
            PronunciationRule("GitHub", "git-hub", "ɡɪthʌb"),
            PronunciationRule("PyTorch", "pie-torch", "paɪtɔːrtʃ"),
            PronunciationRule("NumPy", "num-pie", "nʌmpaɪ"),
            PronunciationRule("TensorFlow", "tensor-flow", "ˈtɛnsərfloʊ"),
        ]
        
        # Emotion markers
        self.emotion_markers = [
            EmotionMarker(r"!\s*$", "excited", 0.8),
            EmotionMarker(r"\?\s*$", "questioning", 0.6),
            EmotionMarker(r"\.{3,}", "thoughtful", 0.5),
            EmotionMarker(r"^(Oh|Ah|Wow|Amazing)", "surprised", 0.7),
            EmotionMarker(r"(unfortunately|sadly|regret)", "sad", 0.6),
            EmotionMarker(r"(wonderful|fantastic|great|excellent)", "happy", 0.7),
            EmotionMarker(r"(terrible|awful|horrible)", "angry", 0.6),
        ]
        
        # Sentence boundary patterns
        self.sentence_boundaries = [
            r'(?<=[.!?])\s+(?=[A-Z])',  # Standard sentence end
            r'(?<=[.!?])\s*\n',  # Sentence end at line break
            r'\n\n+',  # Paragraph break
            r'(?<=:)\s+',  # After colon
            r'(?<=;)\s+',  # After semicolon
        ]
    
    def process_text(
        self,
        text: str,
        enable_ssml: bool = False,
        target_provider: Optional[str] = None,
        expand_abbreviations: bool = True,
        apply_pronunciation: bool = True,
        detect_language: bool = True,
        detect_emotions: bool = False,
        smart_punctuation: bool = True
    ) -> str:
        """
        Process text with advanced features.
        
        Args:
            text: Input text
            enable_ssml: Generate SSML markup
            target_provider: Target TTS provider for SSML compatibility
            expand_abbreviations: Expand common abbreviations
            apply_pronunciation: Apply pronunciation rules
            detect_language: Detect and mark language changes
            detect_emotions: Detect and mark emotions
            smart_punctuation: Apply smart punctuation fixes
            
        Returns:
            Processed text or SSML
        """
        # Clean and normalize text
        processed_text = self._normalize_text(text)
        
        # Apply smart punctuation if enabled
        if smart_punctuation:
            processed_text = self._fix_punctuation(processed_text)
        
        # Expand abbreviations
        if expand_abbreviations:
            processed_text = self._expand_abbreviations(processed_text)
        
        # Generate SSML if enabled
        if enable_ssml:
            return self._generate_ssml(
                processed_text,
                target_provider=target_provider,
                apply_pronunciation=apply_pronunciation,
                detect_language=detect_language,
                detect_emotions=detect_emotions
            )
        
        # Apply pronunciation rules for plain text
        if apply_pronunciation:
            processed_text = self._apply_pronunciation_plain(processed_text)
        
        return processed_text
    
    def _normalize_text(self, text: str) -> str:
        """Normalize whitespace and clean text"""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common encoding issues
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('–', '-').replace('—', '-')
        
        # Remove zero-width characters
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text.strip()
    
    def _fix_punctuation(self, text: str) -> str:
        """Apply smart punctuation fixes"""
        # Add space after punctuation if missing
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        # Fix multiple punctuation
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        # Fix comma spacing
        text = re.sub(r'\s*,\s*', ', ', text)
        
        # Fix quotes
        text = re.sub(r'"\s+', '"', text)
        text = re.sub(r'\s+"', '"', text)
        
        # Add periods to sentences that don't end with punctuation
        lines = text.split('\n')
        fixed_lines = []
        for line in lines:
            line = line.strip()
            if line and not re.search(r'[.!?:;]$', line):
                # Check if it looks like a sentence
                if re.match(r'^[A-Z].*[a-z]', line) and len(line.split()) > 3:
                    line += '.'
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations"""
        for abbrev, expansion in self.abbreviations.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def _apply_pronunciation_plain(self, text: str) -> str:
        """Apply pronunciation rules to plain text"""
        for rule in self.pronunciation_rules:
            # Check if context matches
            if rule.context:
                if not re.search(rule.context, text):
                    continue
            
            # Replace with phonetic spelling
            pattern = r'\b' + re.escape(rule.text) + r'\b'
            text = re.sub(pattern, rule.phoneme, text, flags=re.IGNORECASE)
        
        return text
    
    def _generate_ssml(
        self,
        text: str,
        target_provider: Optional[str] = None,
        apply_pronunciation: bool = True,
        detect_language: bool = True,
        detect_emotions: bool = False
    ) -> str:
        """
        Generate SSML markup.
        
        Note: Different TTS providers support different SSML features.
        This generates a subset that works with most providers.
        """
        # Start with root speak element
        ssml_parts = ['<speak>']
        
        # Split into sentences for processing
        sentences = self._split_sentences(text)
        
        current_language = None
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Detect language if enabled
            if detect_language and LANGDETECT_AVAILABLE:
                try:
                    detected_lang = detect(sentence)
                    if detected_lang != current_language:
                        if current_language:
                            ssml_parts.append('</lang>')
                        ssml_parts.append(f'<lang xml:lang="{detected_lang}">')
                        current_language = detected_lang
                except LangDetectException:
                    pass
            
            # Start sentence
            ssml_parts.append('<s>')
            
            # Detect and apply emotions
            if detect_emotions:
                emotion_tag = self._get_emotion_tag(sentence)
                if emotion_tag:
                    ssml_parts.append(emotion_tag)
            
            # Process words for pronunciation
            if apply_pronunciation:
                sentence = self._apply_ssml_pronunciation(sentence)
            
            # Add the sentence text
            ssml_parts.append(self._escape_xml(sentence))
            
            # Close emotion tag if opened
            if detect_emotions and '</prosody>' in ssml_parts[-3:]:
                ssml_parts.append('</prosody>')
            
            # End sentence
            ssml_parts.append('</s>')
            
            # Add appropriate pause
            if sentence.rstrip().endswith('?'):
                ssml_parts.append('<break time="300ms"/>')
            elif sentence.rstrip().endswith('!'):
                ssml_parts.append('<break time="250ms"/>')
            else:
                ssml_parts.append('<break time="200ms"/>')
        
        # Close language tag if open
        if current_language:
            ssml_parts.append('</lang>')
        
        # Close speak element
        ssml_parts.append('</speak>')
        
        return ''.join(ssml_parts)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences intelligently"""
        sentences = []
        
        # Use multiple patterns for sentence splitting
        for pattern in self.sentence_boundaries:
            text = re.sub(pattern, '\n<SENTENCE_BREAK>\n', text)
        
        # Split on our marker
        parts = text.split('<SENTENCE_BREAK>')
        
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
        
        return sentences
    
    def _get_emotion_tag(self, sentence: str) -> Optional[str]:
        """Get SSML emotion tag for sentence"""
        for marker in self.emotion_markers:
            if re.search(marker.pattern, sentence, re.IGNORECASE):
                # Map emotion to prosody attributes
                if marker.emotion == "excited":
                    return f'<prosody rate="110%" pitch="+10%">'
                elif marker.emotion == "sad":
                    return f'<prosody rate="90%" pitch="-10%">'
                elif marker.emotion == "happy":
                    return f'<prosody rate="105%" pitch="+5%">'
                elif marker.emotion == "angry":
                    return f'<prosody rate="95%" volume="+10%">'
                elif marker.emotion == "thoughtful":
                    return f'<prosody rate="85%">'
                elif marker.emotion == "questioning":
                    return f'<prosody pitch="+15%">'
        
        return None
    
    def _apply_ssml_pronunciation(self, text: str) -> str:
        """Apply SSML pronunciation tags"""
        for rule in self.pronunciation_rules:
            # Check if context matches
            if rule.context:
                if not re.search(rule.context, text):
                    continue
            
            # Replace with SSML phoneme tag
            pattern = r'\b' + re.escape(rule.text) + r'\b'
            
            if rule.ipa:
                # Use IPA if available
                replacement = f'<phoneme alphabet="ipa" ph="{rule.ipa}">{rule.text}</phoneme>'
            else:
                # Use alias/substitute
                replacement = f'<sub alias="{rule.phoneme}">{rule.text}</sub>'
            
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _escape_xml(self, text: str) -> str:
        """Escape text for XML/SSML"""
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&apos;')
        return text
    
    def add_pronunciation_rule(
        self,
        text: str,
        phoneme: str,
        ipa: Optional[str] = None,
        context: Optional[str] = None
    ):
        """Add a custom pronunciation rule"""
        self.pronunciation_rules.append(
            PronunciationRule(text, phoneme, ipa, context)
        )
    
    def add_abbreviation(self, abbreviation: str, expansion: str):
        """Add a custom abbreviation expansion"""
        self.abbreviations[abbreviation] = expansion
    
    def add_emotion_marker(self, pattern: str, emotion: str, intensity: float = 0.5):
        """Add a custom emotion marker"""
        self.emotion_markers.append(
            EmotionMarker(pattern, emotion, intensity)
        )
    
    def format_for_tts(self, text: str, provider: str = "openai") -> str:
        """
        Format text optimally for a specific TTS provider.
        
        Args:
            text: Input text
            provider: TTS provider name
            
        Returns:
            Formatted text optimized for the provider
        """
        # Provider-specific formatting
        if provider == "openai":
            # OpenAI handles most formatting well, just clean up
            return self.process_text(
                text,
                enable_ssml=False,
                expand_abbreviations=True,
                smart_punctuation=True
            )
        elif provider == "elevenlabs":
            # ElevenLabs supports SSML
            return self.process_text(
                text,
                enable_ssml=True,
                target_provider="elevenlabs",
                expand_abbreviations=True,
                detect_emotions=True
            )
        elif provider in ["kokoro", "chatterbox"]:
            # Local models benefit from pronunciation help
            return self.process_text(
                text,
                enable_ssml=False,
                expand_abbreviations=True,
                apply_pronunciation=True,
                smart_punctuation=True
            )
        else:
            # Default processing
            return self.process_text(
                text,
                enable_ssml=False,
                expand_abbreviations=True,
                smart_punctuation=True
            )

#######################################################################################################################
#
# Utility Functions

def preprocess_for_audiobook(
    text: str,
    enable_ssml: bool = False,
    provider: str = "openai"
) -> str:
    """
    Convenience function to preprocess text for audiobook generation.
    
    Args:
        text: Input text
        enable_ssml: Whether to generate SSML
        provider: Target TTS provider
        
    Returns:
        Processed text
    """
    processor = AdvancedTextProcessor()
    return processor.format_for_tts(text, provider)

def add_chapter_pauses(text: str, pause_duration: float = 2.0) -> str:
    """
    Add SSML pauses between chapters.
    
    Args:
        text: Text with chapter markers
        pause_duration: Pause duration in seconds
        
    Returns:
        Text with SSML pause markers
    """
    # Common chapter patterns
    chapter_patterns = [
        r'^Chapter\s+\d+',
        r'^CHAPTER\s+\d+',
        r'^Part\s+\d+',
        r'^Section\s+\d+',
    ]
    
    for pattern in chapter_patterns:
        # Add pause before chapter headings
        replacement = f'<break time="{pause_duration}s"/>\n\\g<0>'
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE | re.IGNORECASE)
    
    return text

#
# End of text_processor_advanced.py
#######################################################################################################################