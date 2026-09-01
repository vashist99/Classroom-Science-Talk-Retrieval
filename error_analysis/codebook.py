"""False-positive taxonomy.

Codebook (mutually exclusive; first matching rule wins, then residual
hand review). Designed for *model-flagged, human-rejected* utterances.

deictic_attention
    Attention-getting or pointing without science content.
    "Look." / "Look at that!" / "Can you see?" / "Watch."

counting_math
    Rote number talk or "how many" with no investigation.
    "Two." / "One, two, three." / "How many?"

shape_color
    Naming a shape or color, or asking what color, as the whole point.
    "Circle." / "That's a rectangle." / "What color?" / "Orange."

object_naming
    Labeling a everyday object or animal with no scientific framing
    (not a science-concept word like habitat / pollen / predator).
    "Chicken?" / "Trash." / "Apple." / "Airplane."

classroom_management
    Directives, transitions, praise-as-control, procedural talk.
    "Sit down." / "Try." / "Everybody ready?" / "Wash your hands."

literacy
    Letters, writing, reading, names-as-text, storybook talk.

play_materials
    Play-Doh, paper, toys, coloring-as-craft (not observing a phenomenon).

social_affect
    Possession, feelings, social bids without science content.
    "Mine." / "I'm scared." / "That's yours."

borderline_science
    Reviewer said no, but the utterance could reasonably be science
    practice (heat, living/dead, what happened, a science content word
    used in context). Kept so we do not treat every reject as model-crazy.

other
    Does not fit the buckets above.
"""
from __future__ import annotations

import re

# Science-concept hold-out: one-word (or short) TPs we must not gate.
SCIENCE_CONTENT_LEXICON = {
    "habitat", "habitats", "pollen", "predator", "predators", "prey",
    "mammal", "mammals", "insect", "insects", "energy", "volcano",
    "volcanoes", "planet", "planets", "comet", "comets", "solar",
    "mosquito", "mosquitoes", "iguana", "iguanas", "jellyfish",
    "caterpillar", "caterpillars", "cockroach", "cockroaches",
    "grasshopper", "grasshoppers", "lizard", "lizards", "turtle",
    "turtles", "whale", "whales", "shadow", "shadows", "sphere",
    "spheres", "weather", "cloudy", "windy", "winter", "earth", "sun",
    "stars", "plant", "plants", "animal", "animals", "living",
    "seed", "seeds", "root", "roots", "stem", "stems",
    "leaf", "leaves", "larva", "larvae", "cocoon", "chrysalis",
    "metamorphosis", "evaporation", "melt", "melting", "freeze",
    "frozen", "magnet", "magnets", "force", "gravity", "oxygen",
    "carbon", "recycle", "recycling", "lifecycle", "life-cycle",
    "hibernation", "hibernate", "elephant", "kangaroo", "mouse",
    "moose", "skunk", "butterfly", "butterflies", "mantis",
    "ladybug", "ladybugs", "seahorse", "ocean", "honey", "bee",
    "bees", "fly", "fruit", "fruits", "vegetable", "vegetables",
    "carrot", "carrots", "strawberry", "carnivore", "herbivore",
    "omnivore", "dinosaur", "dino", "observe", "observation",
    "veterinarian", "nest", "eggs", "hatch", "hatched",
    "grow", "growing", "bird", "birds", "owl", "bug", "bugs",
    "worm", "worms", "hot", "cold", "temperature",
}

LOOK_EXACT = re.compile(
    r"^\s*(look|look!|look\.|look, look\.?|look, look, look\.?"
    r"|look at (that|this|it|them|here)\.?!?"
    r"|look,? look at (that|this|it|them)\.?!?"
    r"|watch\.?!?|watch this\.?!?|see\?|can you see\??"
    r"|see that\??|lookie\.?!?"
    r"|i (can't |can )?see\.?"
    r"|i saw that\.?"
    r"|we can't see\.?"
    r"|what do you see\??"
    r"|let me see\.?)\s*$",
    re.I,
)
LOOK_ANY = re.compile(r"\b(look|looking at|i saw|can you see|what do you see)\b", re.I)
HAPPEN_WHY = re.compile(
    r"\b(how does that work|how('s| is) it gonna happen|what('s| is) going to happen|"
    r"how did that happen|it('s| is) getting bigger|dry out)\b",
    re.I,
)

COUNT_EXACT = re.compile(
    r"^\s*(how many\??|"
    r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)"
    r"([,.\s]+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+))*"
    r"\.?)\s*$",
    re.I,
)
COUNT_CUES = re.compile(
    r"\b(how many|count(?:ing|ed)?|number[s]?|one two three)\b",
    re.I,
)

SHAPE_COLOR_EXACT = re.compile(
    r"^\s*(what color\??|what shape\??|"
    r"(?:it(?:'s| is) a |that(?:'s| is) a |a )?"
    r"(circle|circles|square|squares|triangle|triangles|rectangle|rectangles|"
    r"oval|ovals|diamond|diamonds|star|stars|"
    r"red|blue|green|yellow|orange|purple|pink|white|black|brown|gray|grey)"
    r"\.?)\s*$",
    re.I,
)
SHAPE_COLOR_CUES = re.compile(
    r"\b(circle|circles|square|squares|triangle|triangles|rectangle|"
    r"rectangles|what color|what shape|color it)\b",
    re.I,
)

MANAGEMENT = re.compile(
    r"\b(please|sit(?: down)?|stand(?: up)?|line up|wash|hands|"
    r"everybody|everyone|ready\??|try(?: it)?|come here|go sit|"
    r"clean up|quiet|listen|wait|stop|don't|do not|excuse me|"
    r"i'll|hold on|put it|give me|let's go)\b",
    re.I,
)
MANAGEMENT_EXACT = re.compile(
    r"^\s*(try\.?!?|ready\??|please\.?!?|sit\.?!?|okay\.?!?|ok\.?!?"
    r"|come on\.?!?|good job\.?!?|there you go\.?!?)\s*$",
    re.I,
)

LITERACY = re.compile(
    r"\b(write|writing|wrote|letter|letters|read|reading|book|story|"
    r"spell|spelling|name|abc|alphabet|word|words|page)\b",
    re.I,
)

PLAY = re.compile(
    r"\b(play-?doh|playdough|play doh|crayon|crayons|marker|markers|"
    r"paper|glue|sticker|stickers|toy|toys|color(?:ing)? it)\b",
    re.I,
)

SOCIAL = re.compile(
    r"^\s*(mine\.?!?|yours\.?!?|me\.?!?|no\.?!?|yes\.?!?|yeah\.?!?"
    r"|thank you\.?!?|thanks\.?!?|i don't know\.?!?|"
    r"i'm scared\.?!?|scared\.?!?)\s*$",
    re.I,
)
SOCIAL_CUES = re.compile(
    r"\b(mine|yours|scared|my turn|your turn|i love|friend)\b",
    re.I,
)

BORDERLINE = re.compile(
    r"\b(what happened|why (?:are|is|did|do)|because|alive|dead|"
    r"living|hot|cold|melt|melting|grow|growing|shadow|habitat|"
    r"pollen|predator|insect|plant|animal|weather|rain|sun|"
    r"mosquito|caterpillar|bee|bug|bugs|worm|seeds?)\b",
    re.I,
)

# Everyday object names that are not themselves science concepts.
OBJECT_EXACT = re.compile(
    r"^\s*(?:(?:it(?:'s| is) (?:a |an )?|that(?:'s| is) (?:a |an )?|a |the |my )?)?"
    r"(chicken|chickens|trash|apple|apples|monkey|monkeys|airplane|airplanes|"
    r"ball|balls|door|doors|teeth|tooth|hair|fingers?|hand|hands|"
    r"car|cars|truck|trucks|baby|babies|girl|boy|dog|cat|cats|"
    r"christmas|angelina|bee|bees|grasshopper)\??\.?\s*$",
    re.I,
)


def _norm(utt: str) -> str:
    return (utt or "").strip()


def is_science_content_name(utt: str) -> bool:
    """True if the utterance is essentially a science-concept name (hold-out)."""
    tokens = re.findall(r"[a-z']+", _norm(utt).lower())
    # strip trivial determiners
    tokens = [t for t in tokens if t not in {"a", "an", "the", "it's", "its", "thats", "that's", "my"}]
    if not tokens:
        return False
    if len(tokens) <= 3 and all(t in SCIENCE_CONTENT_LEXICON for t in tokens):
        return True
    if len(tokens) == 1 and tokens[0] in SCIENCE_CONTENT_LEXICON:
        return True
    return False


def assign_fp_label(utt: str) -> str:
    """Assign a taxonomy label. First match wins."""
    s = _norm(utt)
    if not s or s.lower() in {"nan", "none", "-"}:
        return "other"
    if LOOK_EXACT.match(s):
        return "deictic_attention"
    if COUNT_EXACT.match(s):
        return "counting_math"
    if SHAPE_COLOR_EXACT.match(s):
        return "shape_color"
    if MANAGEMENT_EXACT.match(s):
        return "classroom_management"
    if SOCIAL.match(s):
        return "social_affect"
    if OBJECT_EXACT.match(s) and not is_science_content_name(s):
        return "object_naming"
    if PLAY.search(s) and not BORDERLINE.search(s):
        return "play_materials"
    if LITERACY.search(s) and not BORDERLINE.search(s):
        return "literacy"
    if MANAGEMENT.search(s) and not BORDERLINE.search(s):
        return "classroom_management"
    if SHAPE_COLOR_CUES.search(s) and not BORDERLINE.search(s):
        return "shape_color"
    if COUNT_CUES.search(s) and not BORDERLINE.search(s):
        return "counting_math"
    if LOOK_ANY.search(s) and not BORDERLINE.search(s):
        return "deictic_attention"
    if SOCIAL_CUES.search(s) and not BORDERLINE.search(s) and len(s.split()) <= 6:
        return "social_affect"
    if HAPPEN_WHY.search(s) or BORDERLINE.search(s) or is_science_content_name(s):
        return "borderline_science"
    # short leftover names
    words = s.split()
    if len(words) <= 2 and s.endswith((".", "?", "!")) or len(words) <= 2:
        if not is_science_content_name(s):
            return "object_naming"
    return "other"


def gate_flags(utt: str) -> dict[str, bool]:
    """Boolean gates proposed for Track B pre-filter."""
    s = _norm(utt)
    n_words = len(s.split()) if s else 0
    holdout = is_science_content_name(s)
    look_exact = bool(LOOK_EXACT.match(s))
    count_exact = bool(COUNT_EXACT.match(s))
    shape_exact = bool(SHAPE_COLOR_EXACT.match(s))
    short = n_words <= 2
    return {
        "look_exact": look_exact and not holdout,
        "count_exact": count_exact and not holdout,
        "shape_color_exact": shape_exact and not holdout,
        "short_1_2": short and not holdout,
        "conservative_exact": (
            (look_exact or count_exact or shape_exact) and not holdout
        ),
        "any_proposed_gate": (
            (look_exact or count_exact or shape_exact or short) and not holdout
        ),
        "science_holdout": holdout,
    }
