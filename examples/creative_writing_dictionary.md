# Creative Writing Chat Dictionary
# Transform your writing with style, atmosphere, and character

# === Writing Style Enhancement ===
# Make descriptions more vivid
walked: strode|25
walked: ambled|25
walked: trudged|25
walked: sauntered|25

said: whispered|20
said: exclaimed|20
said: muttered|20
said: declared|20
said: remarked|20

looked: gazed|33
looked: peered|33
looked: glanced|34

big: enormous|25
big: massive|25
big: colossal|25
big: immense|25

small: tiny|25
small: minuscule|25
small: diminutive|25
small: petite|25

# === Atmosphere Building ===
# Add atmospheric details
/\bthe sun\b/: the golden sun
/\bthe moon\b/: the silvery moon
/\bthe stars\b/: the twinkling stars
/\bthe sky\b/: the vast sky
/\bthe wind\b/: the whispering wind
/\bthe rain\b/: the pattering rain
/\bthe forest\b/: the ancient forest
/\bthe mountain\b/: the towering mountain

# === Fantasy World Building ===
[fantasy]sword: enchanted blade
[fantasy]magic: arcane arts
[fantasy]wizard: mage
[fantasy]castle: fortress
[fantasy]dragon: wyrm
[fantasy]king: sovereign
[fantasy]queen: empress
[fantasy]knight: paladin
[fantasy]forest: enchanted woods
[fantasy]potion: elixir

# === Sci-Fi Terminology ===
[scifi]gun: plasma rifle
[scifi]car: hover vehicle
[scifi]phone: comm device
[scifi]computer: neural interface
[scifi]Earth: Terra
[scifi]human: Terran
[scifi]alien: xenomorph
[scifi]spaceship: stellar craft
[scifi]robot: synthetic
[scifi]city: metroplex

# === Character Voice: Old Sage ===
[sage]I think: In my considerable experience,
[sage]maybe: Perhaps, if the fates allow,
[sage]yes: Indeed, young one,
[sage]no: Alas, that cannot be,
[sage]hello: Greetings, seeker of wisdom,

# === Character Voice: Noir Detective ===
[noir]woman: dame
[noir]man: joe
[noir]money: dough
[noir]gun: heater
[noir]police: cops
[noir]criminal: perp
[noir]city: this rotten town
[noir]night: another dark night
[noir]rain: the kind of rain that washes away sins

# === Emotional Intensifiers ===
happy: overjoyed|20
happy: elated|20
happy: ecstatic|20
happy: jubilant|20
happy: blissful|20

sad: devastated|20
sad: heartbroken|20
sad: melancholic|20
sad: despondent|20
sad: forlorn|20

angry: furious|25
angry: enraged|25
angry: livid|25
angry: incensed|25

# === Sensory Descriptions ===
/\bsmelled\b/: caught the scent of
/\bheard\b/: perceived the sound of
/\btasted\b/: savored the flavor of
/\bfelt\b/: experienced the sensation of
/\bsaw\b/: witnessed

# === Time Transitions ===
then: Subsequently,|25
then: Moments later,|25
then: In the next instant,|25
then: Without warning,|25

suddenly: In a heartbeat,|33
suddenly: Without precedent,|33
suddenly: Like lightning,|34

# === Scene Setting ===
/^The room/: The dimly lit chamber
/^The house/: The weathered dwelling
/^The street/: The cobblestone avenue
/^The door/: The heavy oaken portal

# === Mystery/Thriller Elements ===
[mystery]clue: cryptic evidence
[mystery]suspect: person of interest
[mystery]murder: heinous crime
[mystery]detective: investigator
[mystery]witness: observer
[mystery]alibi: whereabouts

# === Romance Elements ===
[romance]love: deep affection
[romance]kiss: tender embrace
[romance]heart: soul
[romance]beautiful: breathtaking
[romance]handsome: devastatingly attractive
[romance]eyes: windows to the soul

# === Horror Atmosphere ===
[horror]dark: pitch black
[horror]quiet: deathly silent
[horror]cold: bone-chilling
[horror]shadow: lurking darkness
[horror]sound: unnatural noise
[horror]door: creaking portal
[horror]house: abandoned manor

# === Action Sequences ===
hit: struck with devastating force|33
hit: delivered a crushing blow|33
hit: connected with brutal impact|34

ran: sprinted at breakneck speed|33
ran: dashed with urgent purpose|33
ran: bolted like lightning|34

jumped: leaped with feline grace|50
jumped: vaulted effortlessly|50

# === Metaphorical Language ===
# Add poetic comparisons
/\bfast\b/: swift as an arrow
/\bquiet\b/: silent as the grave
/\bbrave\b/: courageous as a lion
/\bbeautiful\b/: lovely as a rose
/\bcold\b/: frigid as arctic ice

# === Chapter Beginnings ===
/^Chapter \d+$/: |
Chapter $1

The world held its breath as...
---@@@---

# === Dramatic Endings ===
/^The End$/: |
The End

...or was it merely the beginning?
---@@@---

# === Writing Crutch Removal ===
# Remove overused words
very: [removed]|50
really: [removed]|50
just: [removed]|30
actually: [removed]|30
basically: [removed]|40

# === Dialogue Tags ===
# Vary dialogue attribution
/(\w+) said,/: $1 noted,|20
/(\w+) said,/: $1 observed,|20
/(\w+) said,/: $1 commented,|20
/(\w+) said,/: $1 added,|20
/(\w+) said,/: $1 continued,|20