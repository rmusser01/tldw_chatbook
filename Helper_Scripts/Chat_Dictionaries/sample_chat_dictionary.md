# Sample Chat Dictionary
# This file demonstrates various chat dictionary features

# === Basic Text Replacements ===
# Simple word substitutions (case-sensitive by default)
AI: artificial intelligence
ML: machine learning
NLP: natural language processing
API: application programming interface
UI: user interface
UX: user experience

# === Writing Style Consistency ===
# Common phrases to maintain consistent tone
utilize: use
implement: create
leverage: use
commence: start
terminate: end
facilitate: help
endeavor: try

# === Technical Terminology ===
# Standardize technical terms
/\bdatabase\b/i: database
/\bdata base\b/i: database
/\bweb site\b/i: website
/\be-mail\b/i: email
/\bon-line\b/i: online

# === Emoji Replacements ===
# Add emojis to certain words
happy: happy 😊
sad: sad 😢
excited: excited 🎉
thinking: thinking 🤔
/\b(love|loves|loved|loving)\b/i: $1 ❤️

# === Professional Communication ===
# Make communication more formal
/\basap\b/i: as soon as possible
/\bfyi\b/i: for your information
/\bbtw\b/i: by the way
gonna: going to
wanna: want to
gotta: have to
kinda: kind of

# === Character Voice Example: Pirate ===
# Transform speech to pirate dialect (with group exclusions)
[pirate]you: ye|50
[pirate]your: yer|50
[pirate]the: th'|30
[pirate]my: me|40
hello: ahoy
friend: matey
yes: aye

# === Character Voice Example: Medieval ===
# Medieval/fantasy speech patterns
[medieval]you: thou|40
[medieval]your: thy|40
hello: hail and well met
goodbye: fare thee well
thank you: my gratitude

# === World-Building Example: Sci-Fi ===
# Replace modern terms with sci-fi equivalents
Earth: Terra Prime
president: Supreme Chancellor
car: hovercraft
phone: commlink
computer: neural interface
money: credits
/\bguns?\b/i: plasma rifle

# === Dynamic Greetings (using probability) ===
# Different greetings with equal chance
[greet]hello: Hey there!|33
[greet]hello: Howdy!|33
[greet]hello: What's up!|34

# === Time-based replacements ===
# Morning/evening greetings (would need time detection)
good morning: Top of the morning to you!
good evening: Pleasant evening to you!
good night: Sweet dreams!

# === Error Correction ===
# Fix common typos
teh: the
recieve: receive
occured: occurred
seperate: separate
definately: definitely

# === Multiline Example ===
# Complex replacements with formatting
simple_explanation: |
Let me break this down for you:
1. First, we need to understand the basics
2. Then, we can explore the details
3. Finally, we'll put it all together
---@@@---

technical_disclaimer: |
**Technical Note**: The following explanation includes advanced concepts.
If you need clarification on any terms, please don't hesitate to ask.
---@@@---

# === Regex with Backreferences ===
# More complex pattern matching
/\b(\w+)\s+\1\b/i: $1
# This removes duplicate words (e.g., "the the" becomes "the")

/\bvery\s+(good|bad|happy|sad)\b/i: extremely $1
# Intensifies certain adjectives

# === Context-Sensitive Replacements ===
# Professional vs casual contexts
[professional]thanks: Thank you for your consideration
[casual]thanks: Thanks a bunch!

[professional]sorry: I apologize for any inconvenience
[casual]sorry: My bad!

# === Mathematical Notation ===
# Convert text to symbols
plus or minus: ±
approximately: ≈
not equal: ≠
greater than or equal: ≥
less than or equal: ≤
infinity: ∞

# === Code Formatting ===
# Format code mentions
/`([^`]+)`/: **`$1`**
# Makes inline code bold

# === URL Shortening ===
# Shorten common URLs (example)
https://github.com/: GitHub:
https://stackoverflow.com/: SO:

# === Emphasis Patterns ===
# Add emphasis to important words
/\bIMPORTANT\b/: **IMPORTANT**
/\bNOTE\b/: _NOTE_
/\bWARNING\b/: ⚠️ **WARNING**
/\bTIP\b/: 💡 **TIP**

# === Language Translation Example ===
# Simple word translations (Spanish)
[spanish]hello: hola
[spanish]goodbye: adiós
[spanish]please: por favor
[spanish]thank you: gracias
[spanish]yes: sí
[spanish]no: no

# === Timed Effects Examples ===
# Sticky effect (remains for multiple messages)
excited_mode: SUPER EXCITED MODE ACTIVATED!!! 🎉🎉🎉|100|sticky:5

# Cooldown effect (can't trigger again for N messages)
special_ability: *uses special ability*|100|cooldown:10

# Delay effect (only triggers after N messages)
plot_twist: But wait... there's more to this story...|100|delay:5