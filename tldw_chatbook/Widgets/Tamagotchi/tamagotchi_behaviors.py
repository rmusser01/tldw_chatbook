"""
Behavior Engine for Tamagotchi

Manages personality types, actions, and stat calculations.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Personality:
    """
    Defines personality traits that affect tamagotchi behavior.
    
    All decay/increase values are in points per minute.
    """
    name: str
    happiness_decay: float  # Points lost per minute
    hunger_increase: float  # Points gained per minute  
    energy_decay: float     # Points lost per minute
    social_need: float      # Multiplier for interaction effects (1.0 = normal)
    resilience: float       # Resistance to negative effects (1.0 = normal)
    metabolism: float       # Food processing speed (1.0 = normal)
    playfulness: float      # Energy cost of play (1.0 = normal)


# Predefined personality types
PERSONALITIES: Dict[str, Personality] = {
    'balanced': Personality(
        name='balanced',
        happiness_decay=-0.5,
        hunger_increase=1.0,
        energy_decay=-0.3,
        social_need=1.0,
        resilience=1.0,
        metabolism=1.0,
        playfulness=1.0
    ),
    'energetic': Personality(
        name='energetic',
        happiness_decay=-0.3,
        hunger_increase=1.5,  # Burns energy faster
        energy_decay=-0.6,     # Gets tired quicker
        social_need=1.2,       # Loves interaction
        resilience=0.8,        # More sensitive
        metabolism=1.3,        # Fast metabolism
        playfulness=1.5        # Loves to play
    ),
    'lazy': Personality(
        name='lazy',
        happiness_decay=-0.2,
        hunger_increase=0.8,
        energy_decay=-0.1,     # Rarely gets tired
        social_need=0.7,       # Less social
        resilience=1.2,        # Hardy
        metabolism=0.7,        # Slow metabolism
        playfulness=0.5        # Doesn't like too much activity
    ),
    'needy': Personality(
        name='needy',
        happiness_decay=-1.0,  # Gets sad quickly
        hunger_increase=1.2,
        energy_decay=-0.4,
        social_need=2.0,       # Needs lots of attention
        resilience=0.6,        # Fragile
        metabolism=1.0,
        playfulness=1.2
    ),
    'independent': Personality(
        name='independent',
        happiness_decay=-0.3,
        hunger_increase=0.9,
        energy_decay=-0.2,
        social_need=0.5,       # Prefers solitude
        resilience=1.5,        # Very hardy
        metabolism=0.9,
        playfulness=0.8
    ),
    'playful': Personality(
        name='playful',
        happiness_decay=-0.4,
        hunger_increase=1.3,
        energy_decay=-0.5,
        social_need=1.5,
        resilience=0.9,
        metabolism=1.1,
        playfulness=2.0        # Loves playing
    )
}


def register_personality(name: str, personality: Personality) -> None:
    """
    Register a custom personality type.
    
    Args:
        name: Unique name for the personality
        personality: Personality instance
    """
    PERSONALITIES[name] = personality


class BehaviorEngine:
    """
    Manages tamagotchi behavior based on personality.
    
    Processes actions, calculates stat changes, and handles
    personality-based modifications.
    """
    
    def __init__(self, personality_type: str = "balanced"):
        """
        Initialize behavior engine with a personality.
        
        Args:
            personality_type: Name of personality type
        """
        self.personality = PERSONALITIES.get(
            personality_type, 
            PERSONALITIES['balanced']
        )
        
        # Track cooldowns for actions
        self.action_cooldowns: Dict[str, float] = {}
        
        # Initialize action effects based on personality
        self.action_effects = self._init_action_effects()
    
    def _init_action_effects(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize action effect mappings with personality modifiers.
        
        Returns:
            Dictionary of action effects
        """
        return {
            'feed': {
                'changes': {
                    'hunger': -30 * self.personality.metabolism,
                    'happiness': 5,
                    'energy': 10,
                    'health': 5
                },
                'animation': 'eating',
                'cooldown': 60,
                'message': 'Yum! That was delicious!'
            },
            'play': {
                'changes': {
                    'happiness': 20 * self.personality.playfulness,
                    'energy': -15 * self.personality.playfulness,
                    'hunger': 5
                },
                'animation': 'bounce',
                'cooldown': 30,
                'message': 'That was fun!'
            },
            'pet': {
                'changes': {
                    'happiness': 10 * self.personality.social_need
                },
                'animation': 'heart',
                'cooldown': 10,
                'message': 'Feels nice!'
            },
            'sleep': {
                'changes': {
                    'energy': 50,
                    'happiness': 5,
                    'hunger': 10  # Gets hungry while sleeping
                },
                'animation': 'sleeping',
                'cooldown': 300,
                'message': 'Zzz... Sweet dreams!'
            },
            'medicine': {
                'changes': {
                    'health': 30,
                    'happiness': -10,  # Medicine tastes bad
                    'energy': -5
                },
                'animation': 'healing',
                'cooldown': 120,
                'message': 'Yuck! But feeling better...'
            },
            'clean': {
                'changes': {
                    'health': 10,
                    'happiness': 15
                },
                'animation': 'sparkle',
                'cooldown': 60,
                'message': 'All clean and fresh!'
            },
            'treat': {
                'changes': {
                    'happiness': 30,
                    'hunger': -10,
                    'health': -5  # Too many treats aren't healthy
                },
                'animation': 'heart',
                'cooldown': 120,
                'message': 'Wow! A special treat!'
            }
        }
    
    def calculate_decay(self, time_delta: float) -> Dict[str, float]:
        """
        Calculate stat changes over time based on personality.
        
        Args:
            time_delta: Time passed in seconds
        
        Returns:
            Dictionary of stat changes
        """
        minutes = time_delta / 60.0
        
        return {
            'happiness': self.personality.happiness_decay * minutes,
            'hunger': self.personality.hunger_increase * minutes,
            'energy': self.personality.energy_decay * minutes,
            'health': 0  # Health doesn't decay naturally
        }
    
    def process_action(
        self, 
        action: str, 
        current_stats: Dict[str, float],
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Process an action and return its effects.
        
        Args:
            action: The action to perform
            current_stats: Current tamagotchi stats
            force: Whether to ignore cooldowns
        
        Returns:
            Dictionary with changes, animation, message, and success status
        """
        # Check if action exists
        if action not in self.action_effects:
            return {
                'changes': {},
                'success': False,
                'message': 'Unknown action'
            }
        
        # Check cooldown unless forced
        if not force and action in self.action_cooldowns:
            remaining = self.action_cooldowns[action]
            if remaining > 0:
                return {
                    'changes': {},
                    'success': False,
                    'message': f'Too soon! Wait {int(remaining)} seconds.',
                    'cooldown_remaining': remaining
                }
        
        # Get base effect
        effect = self.action_effects[action].copy()
        changes = effect['changes'].copy()
        
        # Apply situational modifiers
        changes = self._apply_situational_modifiers(
            action, changes, current_stats
        )
        
        # Apply personality resilience to all changes
        for stat in changes:
            if stat == 'happiness':
                changes[stat] *= self.personality.resilience
        
        # Set cooldown
        self.action_cooldowns[action] = effect.get('cooldown', 0)
        
        return {
            'changes': changes,
            'animation': effect.get('animation'),
            'message': effect.get('message', ''),
            'success': True
        }
    
    def _apply_situational_modifiers(
        self,
        action: str,
        changes: Dict[str, float],
        current_stats: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply situational modifiers based on current state.
        
        Args:
            action: The action being performed
            changes: Base stat changes
            current_stats: Current stats
        
        Returns:
            Modified stat changes
        """
        modified = changes.copy()
        
        # Sick pets respond poorly to most actions
        if current_stats.get('health', 100) < 30:
            if action not in ['medicine', 'sleep']:
                modified['happiness'] = modified.get('happiness', 0) * 0.5
        
        # Tired pets need sleep
        if current_stats.get('energy', 50) < 20:
            if action == 'play':
                modified['happiness'] = modified.get('happiness', 0) * 0.3
                modified['energy'] = modified.get('energy', 0) * 1.5  # Extra tiring
            elif action == 'sleep':
                modified['energy'] = modified.get('energy', 0) * 1.2  # Extra effective
        
        # Very hungry pets are less happy
        if current_stats.get('hunger', 50) > 80:
            if action != 'feed':
                modified['happiness'] = modified.get('happiness', 0) * 0.7
        
        # Very happy pets get bonus effects
        if current_stats.get('happiness', 50) > 80:
            modified['health'] = modified.get('health', 0) + 2  # Happiness boosts health
        
        # Overfed pets
        if current_stats.get('hunger', 50) < 10 and action == 'feed':
            modified['happiness'] = -5  # Don't like being overfed
            modified['health'] = -5
        
        return modified
    
    def update_cooldowns(self, time_delta: float) -> None:
        """
        Update action cooldowns.
        
        Args:
            time_delta: Time passed in seconds
        """
        for action in list(self.action_cooldowns.keys()):
            self.action_cooldowns[action] -= time_delta
            if self.action_cooldowns[action] <= 0:
                del self.action_cooldowns[action]
    
    def get_mood_from_stats(self, stats: Dict[str, float]) -> str:
        """
        Determine mood based on current stats.
        
        Args:
            stats: Current tamagotchi stats
        
        Returns:
            Mood string
        """
        happiness = stats.get('happiness', 50)
        hunger = stats.get('hunger', 50)
        energy = stats.get('energy', 50)
        health = stats.get('health', 100)
        
        # Priority order for mood determination
        if health <= 0:
            return 'dead'
        elif health < 30:
            return 'sick'
        elif energy < 20:
            return 'sleepy'
        elif hunger > 80:
            return 'hungry'
        elif happiness > 80:
            return 'excited' if energy > 70 else 'happy'
        elif happiness > 50:
            return 'neutral'
        elif happiness > 25:
            return 'sad'
        else:
            return 'very_sad'
    
    def get_recommended_action(self, stats: Dict[str, float]) -> Optional[str]:
        """
        Get recommended action based on current stats.
        
        Args:
            stats: Current tamagotchi stats
        
        Returns:
            Recommended action name or None
        """
        # Priority-based recommendations
        if stats.get('health', 100) < 30:
            return 'medicine'
        elif stats.get('hunger', 50) > 70:
            return 'feed'
        elif stats.get('energy', 50) < 30:
            return 'sleep'
        elif stats.get('happiness', 50) < 30:
            return 'play'
        elif stats.get('health', 100) < 70:
            return 'clean'
        
        return None
    
    def calculate_interaction_bonus(
        self,
        interaction_count: int,
        time_since_last: float
    ) -> float:
        """
        Calculate happiness bonus for regular interaction.
        
        Args:
            interaction_count: Total number of interactions
            time_since_last: Time since last interaction in seconds
        
        Returns:
            Happiness bonus multiplier
        """
        # Regular interaction bonus
        if time_since_last < 300:  # Within 5 minutes
            return 1.2 * self.personality.social_need
        elif time_since_last < 3600:  # Within 1 hour
            return 1.0
        else:  # Long time without interaction
            return 0.8