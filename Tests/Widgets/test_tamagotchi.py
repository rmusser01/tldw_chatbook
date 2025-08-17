"""
Unit tests for the Tamagotchi module.

Tests validation, rate limiting, state recovery, and core functionality.
"""

import pytest
import time
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
from tldw_chatbook.Widgets.Tamagotchi import (
    BaseTamagotchi,
    Tamagotchi,
    CompactTamagotchi,
    BehaviorEngine,
    Personality,
    PERSONALITIES,
    SpriteManager,
    MemoryStorage,
    JSONStorage,
    SQLiteStorage,
    TamagotchiInteraction,
    TamagotchiDeath,
    TamagotchiStateChange
)
from tldw_chatbook.Widgets.Tamagotchi.validators import (
    TamagotchiValidator,
    StateValidator,
    RateLimiter,
    ValidationError
)


class TestValidators:
    """Test input validation functionality."""
    
    def test_name_validation_valid(self):
        """Test valid name validation."""
        assert TamagotchiValidator.validate_name("Pixel") == "Pixel"
        assert TamagotchiValidator.validate_name("Pet-123") == "Pet-123"
        assert TamagotchiValidator.validate_name("My_Pet") == "My_Pet"
        assert TamagotchiValidator.validate_name("  Fluffy  ") == "Fluffy"
    
    def test_name_validation_invalid(self):
        """Test invalid name validation."""
        with pytest.raises(ValidationError):
            TamagotchiValidator.validate_name("")
        
        with pytest.raises(ValidationError):
            TamagotchiValidator.validate_name("   ")
        
        with pytest.raises(ValidationError):
            TamagotchiValidator.validate_name("a" * 21)  # Too long
        
        with pytest.raises(ValidationError):
            TamagotchiValidator.validate_name("Pet@123")  # Invalid character
    
    def test_update_interval_validation(self):
        """Test update interval validation."""
        assert TamagotchiValidator.validate_update_interval(30.0) == 30.0
        assert TamagotchiValidator.validate_update_interval(1.0) == 1.0
        assert TamagotchiValidator.validate_update_interval(3600.0) == 3600.0
        
        with pytest.raises(ValidationError):
            TamagotchiValidator.validate_update_interval(0.5)  # Too small
        
        with pytest.raises(ValidationError):
            TamagotchiValidator.validate_update_interval(3601)  # Too large
        
        with pytest.raises(ValidationError):
            TamagotchiValidator.validate_update_interval("not a number")
    
    def test_personality_validation(self):
        """Test personality validation."""
        assert TamagotchiValidator.validate_personality("balanced", PERSONALITIES) == "balanced"
        assert TamagotchiValidator.validate_personality("ENERGETIC", PERSONALITIES) == "energetic"
        assert TamagotchiValidator.validate_personality("", PERSONALITIES) == "balanced"
        
        with pytest.raises(ValidationError):
            TamagotchiValidator.validate_personality("invalid", PERSONALITIES)
    
    def test_size_validation(self):
        """Test size validation."""
        assert TamagotchiValidator.validate_size("normal") == "normal"
        assert TamagotchiValidator.validate_size("compact") == "compact"
        assert TamagotchiValidator.validate_size("minimal") == "minimal"
        assert TamagotchiValidator.validate_size("") == "normal"
        
        with pytest.raises(ValidationError):
            TamagotchiValidator.validate_size("huge")
    
    def test_stat_validation(self):
        """Test stat value validation."""
        assert TamagotchiValidator.validate_stat(50, "happiness") == 50
        assert TamagotchiValidator.validate_stat(-10, "hunger") == 0
        assert TamagotchiValidator.validate_stat(150, "energy") == 100
        assert TamagotchiValidator.validate_stat(75.5, "health") == 75.5


class TestStateValidator:
    """Test state validation and recovery."""
    
    def test_valid_state(self):
        """Test validation of valid state."""
        state = {
            'name': 'Pixel',
            'happiness': 50,
            'hunger': 30,
            'energy': 70,
            'health': 100,
            'age': 5.5
        }
        is_valid, error = StateValidator.validate_state(state)
        assert is_valid
        assert error is None
    
    def test_invalid_state_missing_fields(self):
        """Test validation with missing required fields."""
        state = {
            'name': 'Pixel',
            'happiness': 50
        }
        is_valid, error = StateValidator.validate_state(state)
        assert not is_valid
        assert "Missing required fields" in error
    
    def test_invalid_state_wrong_types(self):
        """Test validation with wrong field types."""
        state = {
            'name': 123,  # Should be string
            'happiness': 50,
            'hunger': 30,
            'energy': 70,
            'health': 100,
            'age': 5
        }
        is_valid, error = StateValidator.validate_state(state)
        assert not is_valid
        assert "Invalid name" in error
    
    def test_invalid_state_out_of_range(self):
        """Test validation with out-of-range values."""
        state = {
            'name': 'Pixel',
            'happiness': 150,  # Out of range
            'hunger': 30,
            'energy': 70,
            'health': 100,
            'age': 5
        }
        is_valid, error = StateValidator.validate_state(state)
        assert not is_valid
        assert "out of range" in error
    
    def test_state_repair(self):
        """Test state repair functionality."""
        corrupted = {
            'name': 'Pixel',
            'happiness': 150,  # Out of range
            'hunger': -20,  # Negative
            'energy': 'not a number',  # Wrong type
            # Missing health
            'age': 5,
            'extra_field': 'ignored'
        }
        
        repaired = StateValidator.repair_state(corrupted)
        
        assert repaired['name'] == 'Pixel'
        assert repaired['happiness'] == 100  # Clamped
        assert repaired['hunger'] == 0  # Clamped
        assert repaired['energy'] == 50  # Default
        assert repaired['health'] == 100  # Default
        assert repaired['age'] == 5
    
    def test_create_default_state(self):
        """Test default state creation."""
        state = StateValidator.create_default_state("TestPet")
        
        assert state['name'] == "TestPet"
        assert state['happiness'] == 50
        assert state['hunger'] == 50
        assert state['energy'] == 50
        assert state['health'] == 100
        assert state['age'] == 0
        assert state['is_alive'] == True


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    def test_global_cooldown(self):
        """Test global interaction cooldown."""
        limiter = RateLimiter(global_cooldown=1.0)
        
        # First interaction should be allowed
        allowed, cooldown = limiter.can_interact(1.0)
        assert allowed
        assert cooldown == 0
        
        # Record the interaction
        limiter.record_interaction(1.0)
        
        # Immediate second interaction should be blocked
        allowed, cooldown = limiter.can_interact(1.5)
        assert not allowed
        assert cooldown == pytest.approx(0.5, rel=0.1)
        
        # After cooldown, should be allowed
        allowed, cooldown = limiter.can_interact(2.1)
        assert allowed
        assert cooldown == 0
    
    def test_action_specific_cooldown(self):
        """Test per-action cooldowns."""
        action_cooldowns = {'feed': 2.0, 'play': 1.0}
        limiter = RateLimiter(global_cooldown=0.5, action_cooldowns=action_cooldowns)
        
        # First feed should be allowed (initial state)
        allowed, cooldown = limiter.can_interact(10.0, 'feed')
        assert allowed
        limiter.record_interaction(10.0, 'feed')
        
        # Immediate second feed should be blocked
        allowed, cooldown = limiter.can_interact(10.5, 'feed')
        assert not allowed
        assert cooldown == pytest.approx(1.5, rel=0.1)
        
        # Play should still be allowed after global cooldown (different action)
        allowed, cooldown = limiter.can_interact(10.6, 'play')
        assert allowed
        limiter.record_interaction(10.6, 'play')
        
        # After feed cooldown, feed should be allowed
        allowed, cooldown = limiter.can_interact(12.1, 'feed')
        assert allowed


class TestBehaviorEngine:
    """Test behavior engine functionality."""
    
    def test_personality_initialization(self):
        """Test initialization with different personalities."""
        for personality_name in PERSONALITIES:
            engine = BehaviorEngine(personality_name)
            assert engine.personality.name == personality_name
    
    def test_decay_calculation(self):
        """Test stat decay over time."""
        engine = BehaviorEngine("balanced")
        
        # Test 1 minute decay
        decay = engine.calculate_decay(60)
        assert decay['happiness'] == pytest.approx(-0.5, rel=0.01)
        assert decay['hunger'] == pytest.approx(1.0, rel=0.01)
        assert decay['energy'] == pytest.approx(-0.3, rel=0.01)
        
        # Test different personality
        engine = BehaviorEngine("energetic")
        decay = engine.calculate_decay(60)
        assert decay['happiness'] == pytest.approx(-0.3, rel=0.01)
        assert decay['hunger'] == pytest.approx(1.5, rel=0.01)
    
    def test_action_processing(self):
        """Test action processing."""
        engine = BehaviorEngine("balanced")
        
        stats = {'happiness': 50, 'hunger': 50, 'energy': 50, 'health': 100}
        
        # Test feed action
        result = engine.process_action('feed', stats)
        assert result['success']
        assert 'changes' in result
        assert result['changes']['hunger'] < 0  # Hunger decreases
        
        # Test invalid action
        result = engine.process_action('invalid_action', stats)
        assert not result['success']
    
    def test_situational_modifiers(self):
        """Test situational modifiers on actions."""
        engine = BehaviorEngine("balanced")
        
        # Test sick pet modifier
        sick_stats = {'happiness': 50, 'hunger': 50, 'energy': 50, 'health': 20}
        result = engine.process_action('play', sick_stats)
        # Happiness gain should be reduced when sick
        assert 'happiness' in result['changes']
        assert result['changes']['happiness'] < 20
        
        # Test tired pet modifier  
        tired_stats = {'happiness': 50, 'hunger': 50, 'energy': 15, 'health': 100}
        result = engine.process_action('play', tired_stats)
        # Play should be less effective when tired
        assert 'happiness' in result['changes']
        assert result['changes']['happiness'] < 20


class TestSpriteManager:
    """Test sprite management functionality."""
    
    def test_sprite_themes(self):
        """Test different sprite themes."""
        # Test emoji theme
        manager = SpriteManager("emoji")
        sprite = manager.get_sprite("happy")
        assert sprite in manager.EMOJI_SPRITES["happy"]
        
        # Test ASCII theme
        manager = SpriteManager("ascii")
        sprite = manager.get_sprite("happy")
        assert sprite in manager.ASCII_SPRITES["happy"]
    
    def test_custom_sprites(self):
        """Test custom sprite registration."""
        manager = SpriteManager()
        custom_sprites = ["^_^", "^o^"]
        
        manager.register_sprite("custom_mood", custom_sprites)
        sprite = manager.get_sprite("custom_mood")
        assert sprite in custom_sprites
    
    def test_animation_frames(self):
        """Test animation frame retrieval."""
        manager = SpriteManager()
        
        frames = manager.get_animation("eating")
        assert len(frames) > 0
        assert all(isinstance(f, str) for f in frames)
        
        # Test non-existent animation
        frames = manager.get_animation("non_existent")
        assert frames == []


class TestStorageAdapters:
    """Test storage adapter functionality."""
    
    def test_memory_storage(self):
        """Test in-memory storage."""
        storage = MemoryStorage()
        
        state = {'name': 'Test', 'happiness': 75, 'hunger': 25, 
                'energy': 50, 'health': 100, 'age': 1}
        
        # Test save
        assert storage.save('pet1', state)
        
        # Test load
        loaded = storage.load('pet1')
        assert loaded == state
        
        # Test list
        assert 'pet1' in storage.list_pets()
        
        # Test delete
        assert storage.delete('pet1')
        assert storage.load('pet1') is None
    
    def test_json_storage(self):
        """Test JSON file storage."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            storage = JSONStorage(filepath)
            
            state = {'name': 'Test', 'happiness': 75, 'hunger': 25,
                    'energy': 50, 'health': 100, 'age': 1}
            
            # Test save
            assert storage.save('pet1', state)
            
            # Verify file exists
            assert Path(filepath).exists()
            
            # Test load
            loaded = storage.load('pet1')
            assert loaded['name'] == 'Test'
            assert 'last_saved' in loaded
            
            # Test persistence (new instance)
            storage2 = JSONStorage(filepath)
            loaded2 = storage2.load('pet1')
            assert loaded2['name'] == 'Test'
            
        finally:
            Path(filepath).unlink(missing_ok=True)
    
    def test_json_storage_backup(self):
        """Test JSON storage backup functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "pets.json"
            storage = JSONStorage(str(filepath), max_backups=2)
            
            # Save multiple times to create backups
            for i in range(4):
                state = {'name': f'Pet{i}', 'happiness': 50, 'hunger': 50,
                        'energy': 50, 'health': 100, 'age': i}
                storage.save(f'pet{i}', state)
                time.sleep(0.1)  # Ensure different timestamps
            
            # Check that backups were created
            backups = list(Path(tmpdir).glob("pets.backup_*.json"))
            assert len(backups) <= 2  # Max 2 backups
    
    def test_storage_recovery(self):
        """Test storage with state recovery."""
        storage = MemoryStorage(enable_recovery=True)
        
        # Test loading corrupted state
        corrupted = {'name': 'Test', 'happiness': 200}  # Invalid/incomplete
        storage.data['pet1'] = corrupted
        
        # Should recover and return valid state
        loaded = storage.load_with_recovery('pet1', 'DefaultName')
        assert loaded is not None
        assert 0 <= loaded['happiness'] <= 100
        assert loaded['health'] == 100  # Default value


class TestBaseTamagotchi:
    """Test the base Tamagotchi widget."""
    
    @pytest.fixture
    def mock_app(self):
        """Create a mock Textual app."""
        app = MagicMock()
        app.log = MagicMock()
        return app
    
    def test_initialization_valid(self, mock_app):
        """Test valid initialization."""
        pet = BaseTamagotchi(
            name="Pixel",
            personality="balanced",
            update_interval=30.0,
            sprite_theme="emoji",
            size="normal"
        )
        pet.app = mock_app
        
        assert pet.pet_name == "Pixel"
        assert pet.personality_type == "balanced"
        assert pet._update_interval == 30.0
        assert pet.display_size == "normal"
    
    def test_initialization_invalid(self, mock_app):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValidationError):
            BaseTamagotchi(
                name="",  # Invalid empty name
                personality="balanced"
            )
        
        with pytest.raises(ValidationError):
            BaseTamagotchi(
                name="Valid",
                personality="invalid_personality"
            )
        
        with pytest.raises(ValidationError):
            BaseTamagotchi(
                name="Valid",
                update_interval=0.1  # Too small
            )
    
    def test_rate_limiting(self, mock_app):
        """Test rate limiting functionality."""
        pet = BaseTamagotchi(
            name="Pixel",
            enable_rate_limiting=True,
            global_cooldown=1.0
        )
        pet.app = mock_app
        pet.notify = MagicMock()
        
        # Mock time
        with patch('tldw_chatbook.Widgets.Tamagotchi.base_tamagotchi.time.time') as mock_time:
            mock_time.return_value = 0
            
            # First interaction should work
            pet.interact("feed")
            
            # Immediate second interaction should be rate limited
            mock_time.return_value = 0.5
            pet.interact("feed")
            
            # Check that rate limit message was posted
            calls = [call for call in pet.post_message.call_args_list
                    if isinstance(call[0][0], TamagotchiInteraction)]
            assert len(calls) >= 1
            last_interaction = calls[-1][0][0]
            assert not last_interaction.success
            assert "wait" in last_interaction.message.lower()
    
    def test_stat_validation_on_interact(self, mock_app):
        """Test that stats are properly validated when changed."""
        pet = BaseTamagotchi(
            name="Pixel",
            enable_rate_limiting=False
        )
        pet.app = mock_app
        
        # Set extreme initial values
        pet.happiness = 95
        pet.hunger = 5
        
        # Mock behavior engine to return extreme changes
        with patch.object(pet.behavior_engine, 'process_action') as mock_process:
            mock_process.return_value = {
                'success': True,
                'changes': {'happiness': 20, 'hunger': -20},
                'message': 'Test'
            }
            
            pet.interact("feed")
            
            # Stats should be clamped to valid range
            assert pet.happiness == 100  # Clamped at max
            assert pet.hunger == 0  # Clamped at min
    
    def test_state_persistence(self, mock_app):
        """Test state saving and loading."""
        storage = MemoryStorage()
        
        # Create and save pet
        pet1 = BaseTamagotchi(name="Pixel", storage=storage)
        pet1.app = mock_app
        pet1.id = "test_pet"
        pet1.happiness = 75
        pet1.hunger = 25
        pet1._save_state()
        
        # Create new pet with same storage and load
        pet2 = BaseTamagotchi(name="Different", storage=storage)
        pet2.app = mock_app
        pet2.id = "test_pet"
        pet2._load_state()
        
        # Should have loaded the saved state
        assert pet2.happiness == 75
        assert pet2.hunger == 25


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_full_lifecycle(self):
        """Test a complete pet lifecycle."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            # Create pet with persistent storage
            storage = JSONStorage(filepath)
            pet = Tamagotchi(
                name="TestPet",
                personality="energetic",
                storage=storage,
                update_interval=1.0,
                enable_rate_limiting=False
            )
            pet.app = MagicMock()
            pet.id = "lifecycle_pet"
            
            # Initial state
            assert pet._is_alive
            assert pet.happiness == 50
            
            # Interact with pet
            pet.interact("feed")
            assert pet.hunger < 50
            
            pet.interact("play") 
            assert pet.happiness > 50
            
            # Save state
            pet._save_state()
            
            # Create new pet and load state
            pet2 = Tamagotchi(
                name="Different",
                storage=storage
            )
            pet2.app = MagicMock()
            pet2.id = "lifecycle_pet"
            pet2._load_state()
            
            # Should have same state
            assert pet2.happiness == pet.happiness
            assert pet2.hunger == pet.hunger
            
        finally:
            Path(filepath).unlink(missing_ok=True)
    
    def test_corruption_recovery(self):
        """Test recovery from corrupted save data."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            # Write corrupted data directly
            corrupted_data = {
                "corrupt_pet": {
                    "name": "Corrupted",
                    "happiness": "not_a_number",
                    "invalid_field": True
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(corrupted_data, f)
            
            # Try to load with recovery enabled
            storage = JSONStorage(filepath, enable_recovery=True)
            pet = Tamagotchi(
                name="Recovery",
                storage=storage
            )
            pet.app = MagicMock()
            pet.id = "corrupt_pet"
            
            # Should recover and load with defaults
            pet._load_state()
            assert pet._is_alive
            assert 0 <= pet.happiness <= 100
            assert 0 <= pet.hunger <= 100
            
        finally:
            Path(filepath).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])