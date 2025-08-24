"""
Message System for Tamagotchi Events

Defines custom messages for tamagotchi events following Textual patterns.
"""

from textual.message import Message
from typing import Any, Dict, Optional


class TamagotchiMessage(Message):
    """
    Base message class for all tamagotchi events.
    
    Provides common attributes for identifying the source tamagotchi.
    """
    
    def __init__(self, tamagotchi: 'BaseTamagotchi') -> None:
        """
        Initialize base tamagotchi message.
        
        Args:
            tamagotchi: The tamagotchi widget that sent the message
        """
        super().__init__()
        self.tamagotchi = tamagotchi
        self.pet_id = tamagotchi.id or tamagotchi.pet_name
        self.pet_name = tamagotchi.pet_name
    
    @property
    def control(self) -> 'BaseTamagotchi':
        """Alias for tamagotchi for consistency with Textual patterns."""
        return self.tamagotchi


class TamagotchiInteraction(TamagotchiMessage):
    """
    Message sent when user interacts with the tamagotchi.
    
    Contains details about the action performed and its results.
    """
    
    def __init__(
        self, 
        tamagotchi: 'BaseTamagotchi',
        action: str,
        result: Dict[str, Any]
    ) -> None:
        """
        Initialize interaction message.
        
        Args:
            tamagotchi: The tamagotchi widget
            action: The action performed (feed, play, etc.)
            result: Dictionary containing action results
        """
        super().__init__(tamagotchi)
        self.action = action
        self.result = result
        self.success = result.get('success', False)
        self.changes = result.get('changes', {})
        self.message = result.get('message', '')
    
    def __repr__(self) -> str:
        return f"<TamagotchiInteraction action={self.action} success={self.success}>"


class TamagotchiStateChange(TamagotchiMessage):
    """
    Message sent when tamagotchi state changes significantly.
    
    Examples: sleeping/awake, healthy/sick transitions.
    """
    
    def __init__(
        self,
        tamagotchi: 'BaseTamagotchi',
        old_state: str,
        new_state: str
    ) -> None:
        """
        Initialize state change message.
        
        Args:
            tamagotchi: The tamagotchi widget
            old_state: Previous state
            new_state: New state
        """
        super().__init__(tamagotchi)
        self.old_state = old_state
        self.new_state = new_state
    
    def __repr__(self) -> str:
        return f"<TamagotchiStateChange {self.old_state} -> {self.new_state}>"


class TamagotchiEvolution(TamagotchiMessage):
    """
    Message sent when tamagotchi evolves to a new life stage.
    
    Used for growth systems where pets change forms over time.
    """
    
    def __init__(
        self,
        tamagotchi: 'BaseTamagotchi',
        from_stage: str,
        to_stage: str
    ) -> None:
        """
        Initialize evolution message.
        
        Args:
            tamagotchi: The tamagotchi widget
            from_stage: Previous evolution stage
            to_stage: New evolution stage
        """
        super().__init__(tamagotchi)
        self.from_stage = from_stage
        self.to_stage = to_stage
        self.evolution_time = tamagotchi.age
    
    def __repr__(self) -> str:
        return f"<TamagotchiEvolution {self.from_stage} -> {self.to_stage}>"


class TamagotchiAchievement(TamagotchiMessage):
    """
    Message sent when an achievement is unlocked.
    
    Used for gamification and milestone tracking.
    """
    
    def __init__(
        self,
        tamagotchi: 'BaseTamagotchi',
        achievement_id: str,
        achievement_name: str,
        description: str,
        reward: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize achievement message.
        
        Args:
            tamagotchi: The tamagotchi widget
            achievement_id: Unique achievement identifier
            achievement_name: Display name of achievement
            description: Achievement description
            reward: Optional reward details
        """
        super().__init__(tamagotchi)
        self.achievement_id = achievement_id
        self.achievement_name = achievement_name
        self.description = description
        self.reward = reward or {}
    
    def __repr__(self) -> str:
        return f"<TamagotchiAchievement {self.achievement_name}>"


class TamagotchiDeath(TamagotchiMessage):
    """
    Message sent when tamagotchi dies.
    
    Contains information about the cause and lifetime statistics.
    """
    
    def __init__(
        self,
        tamagotchi: 'BaseTamagotchi',
        cause: str,
        age: float,
        stats: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize death message.
        
        Args:
            tamagotchi: The tamagotchi widget
            cause: Cause of death (neglect, starvation, old_age, etc.)
            age: Age at death in hours
            stats: Optional lifetime statistics
        """
        super().__init__(tamagotchi)
        self.cause = cause
        self.age = age
        self.stats = stats or {}
        
        # Add final stats
        self.final_happiness = tamagotchi.happiness
        self.final_hunger = tamagotchi.hunger
        self.final_energy = tamagotchi.energy
        self.final_health = tamagotchi.health
    
    def __repr__(self) -> str:
        return f"<TamagotchiDeath cause={self.cause} age={self.age:.1f}h>"


class TamagotchiRevive(TamagotchiMessage):
    """
    Message sent when tamagotchi is revived/reborn.
    
    Used when implementing rebirth or resurrection mechanics.
    """
    
    def __init__(
        self,
        tamagotchi: 'BaseTamagotchi',
        previous_life: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize revive message.
        
        Args:
            tamagotchi: The tamagotchi widget
            previous_life: Information about previous life
        """
        super().__init__(tamagotchi)
        self.previous_life = previous_life or {}
        self.generation = self.previous_life.get('generation', 0) + 1
    
    def __repr__(self) -> str:
        return f"<TamagotchiRevive generation={self.generation}>"


class TamagotchiStatCritical(TamagotchiMessage):
    """
    Message sent when a stat reaches critical levels.
    
    Used to alert parent widgets of urgent attention needed.
    """
    
    def __init__(
        self,
        tamagotchi: 'BaseTamagotchi',
        stat_name: str,
        value: float,
        threshold: float,
        severity: str = "warning"
    ) -> None:
        """
        Initialize critical stat message.
        
        Args:
            tamagotchi: The tamagotchi widget
            stat_name: Name of the critical stat
            value: Current value
            threshold: Threshold that was crossed
            severity: Severity level (info, warning, error, critical)
        """
        super().__init__(tamagotchi)
        self.stat_name = stat_name
        self.value = value
        self.threshold = threshold
        self.severity = severity
    
    def __repr__(self) -> str:
        return f"<TamagotchiStatCritical {self.stat_name}={self.value} severity={self.severity}>"


class TamagotchiMoodChange(TamagotchiMessage):
    """
    Message sent when tamagotchi's mood changes.
    
    More granular than state changes, tracks emotional states.
    """
    
    def __init__(
        self,
        tamagotchi: 'BaseTamagotchi',
        old_mood: str,
        new_mood: str,
        trigger: Optional[str] = None
    ) -> None:
        """
        Initialize mood change message.
        
        Args:
            tamagotchi: The tamagotchi widget
            old_mood: Previous mood
            new_mood: New mood
            trigger: What triggered the mood change
        """
        super().__init__(tamagotchi)
        self.old_mood = old_mood
        self.new_mood = new_mood
        self.trigger = trigger
    
    def __repr__(self) -> str:
        return f"<TamagotchiMoodChange {self.old_mood} -> {self.new_mood}>"


class TamagotchiRequest(TamagotchiMessage):
    """
    Message sent when tamagotchi requests something.
    
    Used for implementing need-based interactions.
    """
    
    def __init__(
        self,
        tamagotchi: 'BaseTamagotchi',
        request_type: str,
        urgency: float = 0.5,
        options: Optional[list[str]] = None
    ) -> None:
        """
        Initialize request message.
        
        Args:
            tamagotchi: The tamagotchi widget
            request_type: Type of request (food, play, sleep, etc.)
            urgency: Urgency level (0.0 to 1.0)
            options: Available response options
        """
        super().__init__(tamagotchi)
        self.request_type = request_type
        self.urgency = urgency
        self.options = options or ['fulfill', 'ignore']
    
    def __repr__(self) -> str:
        return f"<TamagotchiRequest {self.request_type} urgency={self.urgency:.1f}>"