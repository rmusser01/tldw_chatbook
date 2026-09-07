"""
Sprite Management System for Tamagotchi

Handles visual representations including ASCII art and emoji sprites.
"""

from typing import Dict, List, Optional
import random


class SpriteManager:
    """
    Manages visual representations of the tamagotchi.

    Provides both emoji and ASCII art sprites with animation support.
    """

    # Default emoji sprites for different moods
    EMOJI_SPRITES: Dict[str, List[str]] = {
        "happy": ["😊", "😄", "🥰", "😃", "🤗"],
        "neutral": ["😐", "🙂", "😑", "😶", "🤔"],
        "sad": ["😢", "😭", "😞", "😔", "☹️"],
        "very_sad": ["😰", "😥", "😿", "💔", "😖"],
        "hungry": ["😋", "🤤", "😫", "🥺", "😩"],
        "sleepy": ["😴", "😪", "🥱", "💤", "😌"],
        "sick": ["🤢", "🤒", "😷", "🤧", "🤕"],
        "dead": ["💀", "👻", "⚰️", "🪦", "☠️"],
        "baby": ["🥚", "🐣", "🐥", "🐤", "🍼"],
        "teen": ["🐦", "🦆", "🐧", "🦜", "🦅"],
        "adult": ["🐓", "🦅", "🦜", "🦚", "🦉"],
        "excited": ["🤩", "🥳", "🎉", "✨", "🌟"],
        "angry": ["😠", "😡", "🤬", "👿", "💢"],
        "love": ["😍", "🥰", "💕", "💖", "💗"],
    }

    # ASCII art sprites for terminal compatibility
    ASCII_SPRITES: Dict[str, List[str]] = {
        "happy": ["^_^", "^o^", "(◕‿◕)", "(˘▾˘)", "(｡◕‿◕｡)"],
        "neutral": ["-_-", "o_o", "(._.|", ":|", "•_•"],
        "sad": ["T_T", ";_;", "(╥﹏╥)", ":'(", "Q_Q"],
        "very_sad": ["(T⌓T)", "｡･ﾟﾟ･(>_<)･ﾟﾟ･｡", "(ಥ﹏ಥ)", "(っ˘̩╭╮˘̩)っ", "(._.)"],
        "hungry": ["@_@", "*o*", "(｡◕‿◕｡)", "(°o°)", "(*￣▽￣)"],
        "sleepy": ["u_u", "-.-", "(－ω－) zzZ", "(-_-) zzZ", "(˘ω˘)"],
        "sick": ["x_x", "+_+", "(×﹏×)", "(*_*)", "@_@"],
        "dead": ["X_X", "✝_✝", "(✖╭╮✖)", "x.x", "[*_*]"],
        "excited": ["\\(^o^)/", "＼(◎o◎)／", "ヽ(´▽`)/", "＼(★^∀^★)／", "╰(*°▽°*)╯"],
        "angry": [">_<", "(╬ಠ益ಠ)", "ヽ(`⌒´)ﾉ", "(｀ε´)", "(╯°□°）╯"],
        "love": ["♥‿♥", "(♥ω♥)", "(´∀｀)♡", "(*♥‿♥*)", "♥(ˆ⌣ˆԅ)"],
    }

    # Animation sequences for actions
    ANIMATIONS: Dict[str, List[str]] = {
        "eating": ["😐", "😮", "😋", "😊"],
        "bounce": ["😊", "🙃", "😊", "🙃"],
        "spin": ["😊", "🙂", "😊", "🙃"],
        "heart": ["😊", "💕", "💖", "💕", "😊"],
        "sleeping": ["😊", "😪", "😴", "💤"],
        "healing": ["🤒", "💊", "💉", "😊"],
        "sparkle": ["😊", "✨", "🌟", "✨", "😊"],
        "dance": ["🕺", "💃", "🕺", "💃"],
        "jump": ["😊", "⬆️", "😄", "⬇️", "😊"],
        "wink": ["😊", "😉", "😊", "😜"],
    }

    def __init__(self, theme: str = "emoji"):
        """
        Initialize the sprite manager.

        Args:
            theme: Visual theme ('emoji', 'ascii', or custom)
        """
        self.theme = theme
        self.custom_sprites: Dict[str, List[str]] = {}
        self.custom_animations: Dict[str, List[str]] = {}

        # Select appropriate sprite set
        if theme == "emoji":
            self.sprites = self.EMOJI_SPRITES.copy()
        elif theme == "ascii":
            self.sprites = self.ASCII_SPRITES.copy()
        else:
            # Start with emoji as default for custom themes
            self.sprites = self.EMOJI_SPRITES.copy()

    def get_sprite(self, mood: str, variation: Optional[int] = None) -> str:
        """
        Get a sprite for the specified mood.

        Args:
            mood: The mood/state to get sprite for
            variation: Optional specific variation index

        Returns:
            String representation of the sprite
        """
        # Check custom sprites first
        if mood in self.custom_sprites:
            sprite_list = self.custom_sprites[mood]
        elif mood in self.sprites:
            sprite_list = self.sprites[mood]
        else:
            # Fallback to neutral if mood not found
            sprite_list = self.sprites.get("neutral", ["?"])

        if not sprite_list:
            return "?"

        # Select variation
        if variation is not None:
            index = variation % len(sprite_list)
        else:
            # Random variation for variety
            index = random.randint(0, len(sprite_list) - 1)

        return sprite_list[index]

    def register_sprite(self, mood: str, sprites: List[str]) -> None:
        """
        Register custom sprites for a mood.

        Args:
            mood: The mood to register sprites for
            sprites: List of sprite strings
        """
        self.custom_sprites[mood] = sprites

    def register_animation(self, action: str, frames: List[str]) -> None:
        """
        Register a custom animation sequence.

        Args:
            action: The action name
            frames: List of animation frame strings
        """
        self.custom_animations[action] = frames

    def get_animation(self, action: str) -> List[str]:
        """
        Get animation frames for an action.

        Args:
            action: The action to animate

        Returns:
            List of animation frame strings
        """
        # Check custom animations first
        if action in self.custom_animations:
            return self.custom_animations[action]
        elif action in self.ANIMATIONS:
            return self.ANIMATIONS[action]
        else:
            # Default simple animation
            return []

    def set_theme(self, theme: str) -> None:
        """
        Change the sprite theme.

        Args:
            theme: New theme name ('emoji', 'ascii', or custom)
        """
        self.theme = theme

        if theme == "emoji":
            self.sprites = self.EMOJI_SPRITES.copy()
        elif theme == "ascii":
            self.sprites = self.ASCII_SPRITES.copy()

        # Preserve custom sprites
        self.sprites.update(self.custom_sprites)

    def add_mood(self, mood: str, sprites: List[str]) -> None:
        """
        Add a new mood with sprites.

        Args:
            mood: New mood name
            sprites: List of sprites for the mood
        """
        self.sprites[mood] = sprites

    def get_available_moods(self) -> List[str]:
        """
        Get list of available moods.

        Returns:
            List of mood names
        """
        all_moods = set(self.sprites.keys())
        all_moods.update(self.custom_sprites.keys())
        return sorted(list(all_moods))

    def get_available_animations(self) -> List[str]:
        """
        Get list of available animations.

        Returns:
            List of animation names
        """
        all_animations = set(self.ANIMATIONS.keys())
        all_animations.update(self.custom_animations.keys())
        return sorted(list(all_animations))


class ThemePresets:
    """Predefined sprite themes for different styles."""

    RETRO_GAMING = {
        "happy": ["(^o^)", "\\(^_^)/", "o(^▽^)o"],
        "sad": ["(T_T)", "(;_;)", "(ToT)"],
        "hungry": ["(@_@)", "(>_<)", "(o_O)"],
        "sleepy": ["(-_-)zzz", "(-.-)Zzz", "(=_=)"],
        "sick": ["(x_x)", "(@_@)", "(+_+)"],
        "dead": ["[x_x]", "[X_X]", "(✖_✖)"],
    }

    KAWAII = {
        "happy": ["(◡ ‿ ◡)", "(´｡• ᵕ •｡`)", "(ﾉ◕ヮ◕)ﾉ*:･ﾟ✧"],
        "sad": ["(｡•́︿•̀｡)", "(っ˘̩╭╮˘̩)っ", "(｡ŏ﹏ŏ)"],
        "hungry": ["(｡♥‿♥｡)", "(｡･ω･｡)", "(っ˘ڡ˘ς)"],
        "sleepy": ["(｡-ω-)zzz", "(－ω－) zzZ", "(_ _).｡o○"],
        "sick": ["(｡>﹏<｡)", "(×_×)", "(｡•́︿•̀｡)"],
        "dead": ["(✖╭╮✖)", "✝(▀̿Ĺ̯▀̿ ̿)✝", "(҂◡_◡)"],
    }

    MINIMALIST = {
        "happy": [":)", ":D", "c:"],
        "sad": [":(", "D:", ":c"],
        "hungry": [":o", ":O", "o:"],
        "sleepy": ["-_-", "z_z", "._. "],
        "sick": [":/", ":|", ":\\"],
        "dead": ["x_x", "X_X", "*_*"],
    }

    @classmethod
    def get_theme(cls, name: str) -> Dict[str, List[str]]:
        """
        Get a predefined theme by name.

        Args:
            name: Theme name

        Returns:
            Dictionary of mood to sprite lists
        """
        themes = {
            "retro": cls.RETRO_GAMING,
            "kawaii": cls.KAWAII,
            "minimal": cls.MINIMALIST,
        }
        return themes.get(name, {})
