"""
Test cases for TTS improvements
"""
import asyncio
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSEventHandler, TTSRequestEvent, TTSCompleteEvent, 
    TTSProgressEvent, TTSExportEvent, play_audio_file
)
from tldw_chatbook.TTS.audio_player import SimpleAudioPlayer, PlaybackState
from tldw_chatbook.TTS.cost_tracker import CostTracker, TTSProvider


class TestTTSEventHandler:
    """Test TTS event handler improvements"""
    
    @pytest.fixture
    def handler(self):
        """Create a test handler"""
        class TestHandler(TTSEventHandler):
            def __init__(self):
                super().__init__()
                self.messages = []
                
            async def post_message(self, message):
                self.messages.append(message)
                
            def notify(self, message, severity="info"):
                pass
        
        return TestHandler()
    
    @pytest.mark.asyncio
    async def test_cooldown_cleanup(self, handler):
        """Test that cooldown dictionary is cleaned up"""
        # Add some old entries
        old_time = asyncio.get_event_loop().time() - 400  # More than 5 minutes ago
        handler._request_cooldown["old_message"] = old_time
        handler._request_cooldown["recent_message"] = asyncio.get_event_loop().time()
        
        # Trigger cleanup
        handler._cleanup_cooldown_dict(asyncio.get_event_loop().time())
        
        # Old entry should be removed
        assert "old_message" not in handler._request_cooldown
        assert "recent_message" in handler._request_cooldown
    
    @pytest.mark.asyncio
    async def test_cooldown_max_entries(self, handler):
        """Test that cooldown dictionary respects max entries"""
        # Fill up with max entries
        base_time = asyncio.get_event_loop().time()
        for i in range(handler.MAX_COOLDOWN_ENTRIES + 100):
            handler._request_cooldown[f"msg_{i}"] = base_time + i
        
        # Create a request that triggers cleanup
        event = TTSRequestEvent("Test text", f"msg_new")
        with patch.object(handler, '_tts_service', None):
            await handler.handle_tts_request(event)
        
        # Should have removed oldest entries
        assert len(handler._request_cooldown) <= handler.MAX_COOLDOWN_ENTRIES
        # Oldest entries should be gone
        assert "msg_0" not in handler._request_cooldown
    
    @pytest.mark.asyncio
    async def test_progress_events(self, handler):
        """Test that progress events are sent during generation"""
        # Mock TTS service
        mock_service = AsyncMock()
        chunks = [b"chunk1", b"chunk2", b"chunk3", b"chunk4", b"chunk5"]
        
        async def mock_stream(*args):
            for chunk in chunks:
                yield chunk
        
        mock_service.generate_audio_stream = mock_stream
        handler._tts_service = mock_service
        handler._tts_config = {
            "default_model": "tts-1",
            "default_voice": "alloy",
            "default_format": "mp3",
            "default_speed": 1.0
        }
        
        # Generate TTS
        await handler._generate_tts("Test text", "test_msg", "alloy")
        
        # Check for progress events
        progress_events = [m for m in handler.messages if isinstance(m, TTSProgressEvent)]
        assert len(progress_events) >= 2  # At least initial and final
        assert progress_events[0].progress == 0.0
        assert progress_events[-1].progress == 1.0
        assert progress_events[-1].status == "Audio generation complete"
    
    @pytest.mark.asyncio
    async def test_export_functionality(self, handler, tmp_path):
        """Test audio export with custom naming"""
        # Create a mock audio file
        test_audio = tmp_path / "test_audio.mp3"
        test_audio.write_bytes(b"fake audio data")
        
        # Add to handler's audio files
        handler._audio_files["test_msg"] = test_audio
        
        # Export to custom location
        export_path = tmp_path / "exports" / "my_audio.mp3"
        event = TTSExportEvent("test_msg", export_path, include_metadata=True)
        
        await handler.handle_tts_export(event)
        
        # Check file was exported
        assert export_path.exists()
        assert export_path.read_bytes() == b"fake audio data"
        
        # Check metadata was created
        metadata_path = export_path.with_suffix(".mp3.json")
        assert metadata_path.exists()


class TestAudioPlayer:
    """Test audio player improvements"""
    
    @pytest.fixture
    def player(self):
        """Create test player"""
        return SimpleAudioPlayer()
    
    def test_play_stop(self, player, tmp_path):
        """Test play and stop functionality"""
        # Create test audio file
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"RIFF" + b"\x00" * 40)  # Minimal WAV header
        
        # Test play
        if player._player_cmd:  # Only test if player is available
            assert player.play(test_file)
            assert player.get_state() == PlaybackState.PLAYING
            
            # Test stop
            assert player.stop()
            assert player.get_state() == PlaybackState.IDLE
        else:
            # No player available on this system
            assert not player.play(test_file)
    
    def test_state_tracking(self, player):
        """Test state tracking"""
        assert player.get_state() == PlaybackState.IDLE
        assert not player.is_playing()


class TestCostTracker:
    """Test cost tracking functionality"""
    
    @pytest.fixture
    def tracker(self, tmp_path):
        """Create test tracker with temporary database"""
        db_path = tmp_path / "test_usage.db"
        return CostTracker(db_path)
    
    def test_cost_estimation(self, tracker):
        """Test cost estimation for different providers"""
        # OpenAI standard model
        cost = tracker.estimate_cost("openai", "tts-1", 1000)
        assert cost == 0.015  # $0.015 per 1K chars
        
        # OpenAI HD model
        cost = tracker.estimate_cost("openai", "tts-1-hd", 1000)
        assert cost == 0.030  # $0.030 per 1K chars
        
        # Local model (free)
        cost = tracker.estimate_cost("local", "kokoro", 10000)
        assert cost == 0.0
    
    def test_usage_tracking(self, tracker):
        """Test usage tracking and statistics"""
        # Track some usage
        record1 = tracker.track_usage(
            provider="openai",
            model="tts-1",
            text="Hello world",
            voice="alloy",
            format="mp3"
        )
        
        record2 = tracker.track_usage(
            provider="local",
            model="kokoro",
            text="This is a longer text for testing purposes",
            voice="af",
            format="wav"
        )
        
        # Check records
        assert record1.characters == 11
        assert record1.estimated_cost > 0
        assert record2.characters == 42
        assert record2.estimated_cost == 0.0  # Local is free
        
        # Check monthly usage
        monthly_chars = tracker.get_monthly_usage()
        assert monthly_chars == 53  # 11 + 42
        
        # Check monthly cost
        monthly_cost = tracker.get_monthly_cost()
        assert monthly_cost == record1.estimated_cost
    
    def test_free_tier_calculation(self, tracker):
        """Test free tier calculation"""
        # Update Google costs with free tier
        tracker.update_cost_info(
            provider="google",
            cost_per_1k_chars=0.016,
            free_tier_chars=1000000  # 1M free chars
        )
        
        # First request should be free (under free tier)
        cost = tracker.estimate_cost("google", "wavenet", 50000)
        assert cost == 0.0
        
        # Track the usage
        tracker.track_usage(
            provider="google",
            model="wavenet",
            text="x" * 50000,
            voice="en-US-Wavenet-A",
            format="mp3"
        )
        
        # Next request partially in free tier
        cost = tracker.estimate_cost("google", "wavenet", 1000000)
        expected = (50000 / 1000.0) * 0.016  # Only 50K billable
        assert abs(cost - expected) < 0.001


def test_play_audio_file_security():
    """Test that play_audio_file is secure"""
    # Test path validation
    test_path = Path("/tmp/test.mp3")
    
    # Should handle non-existent files gracefully
    play_audio_file(test_path)  # Should not raise
    
    # Should reject non-audio extensions
    bad_path = Path("/tmp/test.exe")
    play_audio_file(bad_path)  # Should not raise, just log error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])