"""Authoritative Sept 2026 TeLI Coding Manual (v0.3) rules, features, and prompts.

Source: 2026-09-28_TeLI_Science_Talk_Coding_Manual.docx.
Key alignments vs legacy v1/v2:
  - Route A (Science Content): 5 explicit domains (Life, Physical, Earth/Space,
    Environmental/Ecological, Functional Engineering/Design).
  - Route B (Science Practice): Observing/describing, comparing/classifying,
    predicting, testing/manipulating, explaining why/how, measuring/evidence —
    ONLY when directed at a natural/physical phenomenon or functional design.
  - Rule 6 STRICT EXCLUSION: Math, rote counting, shape naming, and geometry
    alone are NOT science (removes legacy STRUCTURE_SHAPE_SPACE leakage).
  - Rule 10 INCLUSION: Science content read aloud from books/videos IS science.
  - Bilingual support: English + Spanish / code-switched science cues for
    South Florida pre-K classrooms (24% Spanish-speaking children, 37% teachers).
"""
from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Section 3 & 6.1: 5 Valid Content Domains (zero shape/geometry leakage!)
# ---------------------------------------------------------------------------

LIFE_SCIENCE_RE = re.compile(
    r"\b("
    r"living|non-?living|alive|dead|organism|animal|animals|mammal|mammals|"
    r"reptile|reptiles|amphibian|amphibians|insect|insects|bug|bugs|arachnid|"
    r"bird|birds|fish|whale|whales|shark|sharks|dolphin|octopus|jellyfish|"
    r"seahorse|sea\s+horse|turtle|turtles|iguana|iguanas|lizard|lizards|"
    r"snake|snakes|frog|frogs|tadpole|toad|butterfly|butterflies|caterpillar|"
    r"caterpillars|chrysalis|cocoon|larva|larvae|metamorphosis|bee|bees|"
    r"bumblebee|ladybug|ladybugs|ant|ants|mosquito|mosquitoes|grasshopper|"
    r"mantis|spider|spiders|worm|worms|snail|snails|bat|bats|owl|owls|"
    r"elephant|kangaroo|skunk|wolf|deer|bear|moose|mouse|squirrel|raccoon|"
    r"penguin|giraffe|zebra|monkey|hippo|rhino|dinosaur|fossil|fossils|"
    r"carnivore|herbivore|omnivore|predator|predators|prey|nocturnal|"
    r"hibernat\w*|migrat\w*|camouflag\w*|adapt\w*|extinct\w*|endangered|"
    r"life\s*cycle|hatch\w*|eggs?|nest|nests|wings?|feathers?|scales?|fur|"
    r"fins?|gills?|beak|claws?|tail|tails|legs?|teeth|tooth|antenn\w*|thorax|"
    r"abdomen|shell|skeleton|grow|grows|growing|grown|growth|decay\w*|mold|fungus|"
    r"plant|plants|seed|seeds|seedling|sprout\w*|germinat\w*|root|roots|"
    r"stem|stems|trunk|branch|branches|leaf|leaves|petal|petals|flower|"
    r"flowers|blossom|pollen|pollinat\w*|nectar|photosynthes\w*|chlorophyll|"
    r"plant\s+bulb|vine|vines|bark|sap|fruit|fruits|vegetable|vegetables|nutrients?|"
    r"nutrition|healthy|vitamins?|protein|digestion|digest\w*|"
    r"heart|heartbeat|pulse|blood|veins?|arter\w*|bones?|spine|skull|rib|ribs|"
    r"lungs?|breath\w*|oxygen|muscle|muscles|brain|nerves?|stomach|skin|"
    r"senses?|five\s+senses|sense\s+of|germs?|bacteria|virus|microscope|"
    r"veterinarian|botanist|biologist|scientist|scientists"
    r")\b",
    re.I,
)

PHYSICAL_SCIENCE_RE = re.compile(
    r"\b("
    r"force|forces|push(?:es|ed|ing)?|pull(?:s|ed|ing)?|motion|gravity|friction|inertia|"
    r"momentum|magnet|magnets|magnetic|attract\w*|repel\w*|poles?|metal|iron|"
    r"steel|solid|solids|liquid|liquids|gas|gases|matter|states?\s+of\s+matter|"
    r"melt|melts|melting|melted|freeze|freezes|freezing|frozen|evaporat\w*|"
    r"condens\w*|boil\w*|steam|vapor|ice\s+cube|absorb\w*|dissolv\w*|mix|mixture|"
    r"solution|chemical|reaction|bubble|bubbles|fizz\w*|foam|float|floats|"
    r"floating|sink|sinks|sinking|sank|buoyan\w*|dense|density|heavy|heavier|"
    r"heaviest|light|lighter|lightest|weight|weigh\w*|mass|volume|balance|"
    r"energy|heat|thermal|electric\w*|battery|circuit|wire|bulb|light\s+bulb|"
    r"shadow|shadows|reflect\w*|prism|rainbow|transparent|opaque|translucent|"
    r"sound|vibrat\w*|pitch|loud|louder|echo|wave|waves|"
    r"material|materials|property|properties|texture|rough|smooth|bumpy|"
    r"hard|soft|flexible|rigid|elastic|stretch\w*|absorbent|waterproof|"
    r"sturdy|fragile|slope|ramp|incline|inclined|speed|faster|slower|"
    r"roll(?:s|ed|ing)?|bounce(?:s|d|ing)?|stick(?:s|ing)?|stuck"
    r")\b",
    re.I,
)

EARTH_SPACE_RE = re.compile(
    r"\b("
    r"weather|sunny|cloudy|clouds?|windy|wind|rain|raining|rainy|storm|storms|"
    r"thunder|lightning|tornado|hurricane|snow|snowing|hail|fog|foggy|mist|"
    r"temperature|thermometer|degrees?|fahrenheit|celsius|forecast|climate|"
    r"season|seasons|spring|summer|autumn|fall\s+leaves|winter|frost|drought|"
    r"flood|erosion|water\s+cycle|precipitation|puddle|"
    r"earth|sun|sunlight|solar|moon|lunar|crescent|full\s+moon|"
    r"constellation|planet|planets|mercury|venus|mars|jupiter|saturn|uranus|"
    r"neptune|orbit|comet|comets|asteroid|meteor|galaxy|universe|"
    r"astronaut|telescope|space|rocket|crater|eclipse|rotate|rotation|"
    r"volcano|volcanoes|magma|lava|eruption|erupt\w*|earthquake|rock|rocks|"
    r"mineral|minerals|crystal|crystals|geode|pebble|sand|soil|dirt|mud|clay|"
    r"ocean|oceans|river|lake|pond|glacier|mountain|valley|canyon|island|desert"
    r")\b",
    re.I,
)

ENV_ECOLOGY_RE = re.compile(
    r"\b("
    r"habitat|habitats|ecosystem|environment|environmental|ecology|ecological|"
    r"food\s+chain|food\s+web|compost\w*|recycl\w*|reus\w*|"
    r"conserv\w*|pollut\w*|litter\w*|forest|rainforest|jungle|wetland|swamp|"
    r"marsh|savanna|tundra|arctic|coral\s+reef|prairie|meadow|pond\s+life|"
    r"decompos\w*|earthworm|shelter|burrow|hive|beehive|web|anthill|"
    r"wild|wildlife|nature|natural\s+world|species"
    r")\b",
    re.I,
)

ENGINEERING_RE = re.compile(
    r"\b("
    r"engineer\w*|design\w*|invent\w*|prototype|blueprint|structure|structures|"
    r"build(?:s|ing)?\s+(?:a\s+)?(?:bridge|tower|ramp|rocket|structure|machine|device)|"
    r"bridge|tower|pulley|lever|wheel|axle|gear|gears|wedge|screw|"
    r"simple\s+machine|machine|mechanism|tool|tools|device|stable|stability|"
    r"support|foundation|collapse|topple|tumble|stronger|"
    r"hold\s+up|fit\s+together|how\s+does\s+.{1,25}\s+work|how\s+it\s+works"
    r")\b",
    re.I,
)

# Spanish & code-switched science cues (South Florida pre-K classrooms)
SPANISH_SCIENCE_RE = re.compile(
    r"\b("
    r"planta|plantas|semilla|semillas|ra[ií]z|ra[ií]ces|hoja|hojas|flor|flores|"
    r"polen|[aá]rbol|[aá]rboles|crecer|creciendo|vivo|vivos|viva|no\s+vivo|"
    r"animal|animales|insecto|insectos|mariposa|oruga|abeja|hormiga|ara[nñ]a|"
    r"p[aá]jaro|pez|peces|ballena|tortuga|rana|serpiente|lagarto|iguana|"
    r"h[aá]bitat|depredador|hibernaci[oó]n|clima|tiempo|lluvia|lloviendo|"
    r"nublado|soleado|viento|temperatura|caliente|fr[ií]o|hielo|derretir|"
    r"derritiendo|congelar|l[ií]quido|s[oó]lido|agua|tierra|sol|luna|estrella|"
    r"planeta|volc[aá]n|lava|roca|piedra|im[aá]n|fuerza|empujar|jalar|"
    r"gravedad|flotar|flota|hundir|hunde|pesado|liviano|observar|predecir|"
    r"experimento|investigar|cuerpo|coraz[oó]n|huesos|pulmones|sentidos|"
    r"naturaleza|medio\s+ambiente|reciclar"
    r")\b",
    re.I,
)

SPANISH_GENERAL_RE = re.compile(
    r"\b("
    r"qu[eé]|para|c[oó]mo|est[aá]|aqu[ií]|vamos|mira|ni[nñ]o|ni[nñ]a|por\s+favor|"
    r"gracias|donde|d[oó]nde|porque|por\s+qu[eé]|bueno|muy|m[aá]s|esto|eso|"
    r"tengo|tiene|m[ií]o|tuyo|agua|sol|planta|animal|grande|peque[nñ]o"
    r")\b",
    re.I,
)

# ---------------------------------------------------------------------------
# Section 6.2: Science Practice Cue Phrases
# ---------------------------------------------------------------------------

PRACTICE_OBSERVE_RE = re.compile(
    r"\b("
    r"observe|observing|observation|notice|noticing|noticed|"
    r"look\s+closely|take\s+a\s+closer\s+look|what\s+do\s+you\s+notice|"
    r"what\s+does\s+it\s+(?:look|feel|smell|sound|taste)\s+like|"
    r"what\s+changed|describe|microscope|examine|inspect"
    r")\b",
    re.I,
)

PRACTICE_COMPARE_RE = re.compile(
    r"\b("
    r"compare|comparing|classify|classifying|sort\s+by|group\s+by|"
    r"same\s+(?:or|and)\s+different|similar|difference|differences|"
    r"bigger\s+than|smaller\s+than|heavier|lighter|longer|shorter|"
    r"warmer|colder|faster|slower|what\s+kind\s+of|what\s+type\s+of|"
    r"living\s+or\s+non-?living|category|categories|characteristics?"
    r")\b",
    re.I,
)

PRACTICE_PREDICT_RE = re.compile(
    r"\b("
    r"predict|predicting|prediction|predictions|hypothesis|hypothesize|"
    r"what\s+do\s+you\s+think\s+(?:will|is\s+going\s+to|might|would)\s+happen|"
    r"what\s+(?:will|would|might)\s+happen\s+if|"
    r"what\s+do\s+you\s+expect|were\s+you\s+right\s+in\s+your\s+prediction|"
    r"make\s+a\s+(?:prediction|guess)|do\s+you\s+think\s+it\s+(?:will|can)"
    r")\b",
    re.I,
)

PRACTICE_TEST_RE = re.compile(
    r"\b("
    r"experiment|experiments|investigat\w*|let'?s\s+test|test\s+it|test\s+our|"
    r"try\s+it\s+and\s+see|find\s+out\s+(?:if|what|how|why)|"
    r"what\s+happens\s+(?:if|when)|change\s+the|add\s+more|does\s+it\s+still"
    r")\b",
    re.I,
)

PRACTICE_CAUSAL_RE = re.compile(
    r"\b("
    r"why\s+(?:do|does|did|is|are|was|were|would|can't|cannot|isn't|aren't)|"
    r"how\s+(?:do|does|did|come)|"
    r"because|causes?|caused\s+by|makes\s+it|turns?\s+into|so\s+that|"
    r"explain|explanation|reason|that'?s\s+why|happened\s+because"
    r")\b",
    re.I,
)

PRACTICE_EVIDENCE_RE = re.compile(
    r"\b("
    r"evidence|how\s+do\s+you\s+know|what\s+makes\s+you\s+(?:think|say)\s+that|"
    r"prove|proof|measure|measuring|measurement|data|chart|graph|record|"
    r"results?|according\s+to|my\s+research|our\s+observation|"
    r"ruler|scale|magnifying"
    r")\b",
    re.I,
)

# ---------------------------------------------------------------------------
# Section 6.3 & Rules 1–9: High-Risk Non-Science Traps (FP Diagnostics/Features)
# ---------------------------------------------------------------------------

TRAP_DEICTIC_BARE_RE = re.compile(
    r"^\s*(look|look!|look\.|look,?\s+look\.?!?|look\s+at\s+(?:that|this|it|me|them|here)\.?!?|"
    r"watch\.?!?|watch\s+(?:this|me)\.?!?|see\??|can\s+you\s+see\??|see\s+that\??|"
    r"i\s+(?:can'?t\s+|can\s+)?see\.?|i\s+saw\s+that\.?|what\s+do\s+you\s+see\??|"
    r"let\s+me\s+see\.?|lookie\.?!?)\s*$",
    re.I,
)

TRAP_COUNT_BARE_RE = re.compile(
    r"(^\s*(how\s+many\??|"
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)"
    r"(?:[,.\s]+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+))*"
    r"[.?!]?)\s*$|"
    r"\bhow\s+many\s+(?:friends|kids|children|words|letters|cookies|blocks|stickers|days|minutes)\b)",
    re.I,
)

TRAP_SHAPE_COLOR_RE = re.compile(
    r"\b(what\s+shape|what\s+color|circle|circles|square|squares|triangle|triangles|"
    r"rectangle|rectangles|oval|diamond|rhombus|pentagon|hexagon|octagon|trapezoid|"
    r"cylinder|cone|cube|star\s+shape|heart\s+shape|2d|3d|"
    r"color\s+(?:it|this|the)|crayon|crayons|marker|markers|"
    r"(?:is\s+it|that'?s|it'?s|the\s+color)\s+(?:red|blue|green|yellow|orange|purple|pink|black|white|brown|gray))\b",
    re.I,
)

TRAP_PRETEND_FOOD_RE = re.compile(
    r"\b(pretend|make\s+believe|once\s+upon\s+a\s+time|in\s+the\s+story|"
    r"chicken\s+nuggets?|eat\s+your|on\s+my\s+shirt|my\s+shirt\s+has|sticker\s+star|"
    r"happy\s+birthday|superhero|princess|monster|dragon)\b",
    re.I,
)

TRAP_MANAGEMENT_RE = re.compile(
    r"\b(sit\s+down|have\s+a\s+seat|criss\s*cross|line\s+up|clean\s+up|put\s+.{0,20}\s+away|"
    r"wash\s+your\s+hands|hands\s+(?:to\s+yourself|in\s+your\s+lap|up)|"
    r"eyes\s+on\s+me|look\s+at\s+me|look\s+me\s+in\s+(?:my|the)\s+eye|"
    r"listening\s+ears|quiet|stop|wait\s+your\s+turn|excuse\s+me|bathroom|potty|"
    r"backpack|cubby|nap\s+time|snack|lunch|shoes\s+on|coat\s+on|"
    r"science\s+(?:center\s+)?is\s+open|go\s+back\s+to\s+science|play\s+science)\b",
    re.I,
)

TRAP_LITERACY_CRAFT_RE = re.compile(
    r"\b(write\s+your\s+name|spell|alphabet|abc|letter\s+[a-z]|phonics|rhyme|"
    r"rhyming|syllable|uppercase|lowercase|trace\s+the|cut\s+it|glue\s+it|"
    r"play-?doh|playdough|sticker|stickers|paint\s+it|draw\s+a)\b",
    re.I,
)

TRAP_TEMP_CASUAL_RE = re.compile(
    r"\b(hot|cold|warm|wet|dry|freezing|chilly)\b",
    re.I,
)

QUESTION_RE = re.compile(r"\?|^\s*(what|why|how|where|when|which|who|is|are|do|does|did|can|could|will|would)\b", re.I)


def extract_codebook_features(text: str, speaker: str = "unknown") -> dict[str, float]:
    """Extract Manual-v0.3 aligned lexical, practice, and trap features."""
    s = (text or "").strip()
    words = re.findall(r"[a-zA-Z\u00C0-\u017F']+", s)
    n_words = len(words)
    n_chars = len(s)

    c_life = len(LIFE_SCIENCE_RE.findall(s))
    c_phys = len(PHYSICAL_SCIENCE_RE.findall(s))
    c_earth = len(EARTH_SPACE_RE.findall(s))
    c_env = len(ENV_ECOLOGY_RE.findall(s))
    c_eng = len(ENGINEERING_RE.findall(s))
    c_es_sci = len(SPANISH_SCIENCE_RE.findall(s))
    c_content_total = c_life + c_phys + c_earth + c_env + c_eng + c_es_sci

    p_obs = len(PRACTICE_OBSERVE_RE.findall(s))
    p_comp = len(PRACTICE_COMPARE_RE.findall(s))
    p_pred = len(PRACTICE_PREDICT_RE.findall(s))
    p_test = len(PRACTICE_TEST_RE.findall(s))
    p_caus = len(PRACTICE_CAUSAL_RE.findall(s))
    p_evid = len(PRACTICE_EVIDENCE_RE.findall(s))
    p_total = p_obs + p_comp + p_pred + p_test + p_caus + p_evid

    t_deictic = 1.0 if TRAP_DEICTIC_BARE_RE.match(s) else 0.0
    t_count = 1.0 if TRAP_COUNT_BARE_RE.search(s) else 0.0
    t_shape = float(len(TRAP_SHAPE_COLOR_RE.findall(s)))
    t_mgmt = float(len(TRAP_MANAGEMENT_RE.findall(s)))
    t_lit = float(len(TRAP_LITERACY_CRAFT_RE.findall(s)))
    t_temp = float(len(TRAP_TEMP_CASUAL_RE.findall(s)))
    t_pretend = float(len(TRAP_PRETEND_FOOD_RE.findall(s)))

    spk = (speaker or "unknown").strip().lower()
    trap_sum = t_deictic + t_count + t_shape + t_mgmt + t_lit + t_pretend
    return {
        "n_words": float(n_words),
        "log_n_words": float(min(n_words, 60)) / 10.0,
        "n_chars": float(min(n_chars, 300)) / 100.0,
        "is_1_2_words": 1.0 if n_words <= 2 else 0.0,
        "is_question": 1.0 if QUESTION_RE.search(s) else 0.0,
        "is_adult": 1.0 if spk == "adult" else 0.0,
        "is_child": 1.0 if spk == "child" else 0.0,
        "is_spanish_cue": 1.0 if SPANISH_GENERAL_RE.search(s) or c_es_sci > 0 else 0.0,
        "c_life": float(c_life),
        "c_phys": float(c_phys),
        "c_earth": float(c_earth),
        "c_env": float(c_env),
        "c_eng": float(c_eng),
        "c_es_sci": float(c_es_sci),
        "c_content_total": float(c_content_total),
        "has_content": 1.0 if c_content_total > 0 else 0.0,
        "p_obs": float(p_obs),
        "p_comp": float(p_comp),
        "p_pred": float(p_pred),
        "p_test": float(p_test),
        "p_caus": float(p_caus),
        "p_evid": float(p_evid),
        "p_total": float(p_total),
        "has_practice": 1.0 if p_total > 0 else 0.0,
        "has_both_content_and_practice": 1.0 if (c_content_total > 0 and p_total > 0) else 0.0,
        "trap_deictic_bare": t_deictic,
        "trap_count_bare": t_count,
        "trap_shape_color": t_shape,
        "trap_mgmt": t_mgmt,
        "trap_lit_craft": t_lit,
        "trap_temp_casual": t_temp,
        "trap_pretend_food": t_pretend,
        "has_any_trap": 1.0 if trap_sum > 0 else 0.0,
        "content_minus_traps": float(c_content_total) - trap_sum,
    }


def infer_subtype_rule(text: str) -> str:
    """Assign Conceptual Subtype tags (Manual Section 9) cleanly for workbook output."""
    s = (text or "").strip()
    tags: list[str] = []
    if (
        LIFE_SCIENCE_RE.search(s)
        or PHYSICAL_SCIENCE_RE.search(s)
        or EARTH_SPACE_RE.search(s)
        or ENV_ECOLOGY_RE.search(s)
        or ENGINEERING_RE.search(s)
        or SPANISH_SCIENCE_RE.search(s)
    ):
        tags.append("content")
    if PRACTICE_OBSERVE_RE.search(s) or PRACTICE_COMPARE_RE.search(s):
        tags.append("observation")
    if PRACTICE_PREDICT_RE.search(s):
        tags.append("prediction")
    if PRACTICE_CAUSAL_RE.search(s):
        tags.append("causal_reasoning")
    if PRACTICE_EVIDENCE_RE.search(s) or PRACTICE_TEST_RE.search(s):
        tags.append("evidence")
    return "|".join(tags) if tags else "content"


# ---------------------------------------------------------------------------
# Direct Codebook-Grounded System Prompt (Manual v0.3, Zero Pairwise Bias!)
# ---------------------------------------------------------------------------

MANUAL_V3_SLIDING_WINDOW_SYSTEM_PROMPT = """You are an expert preschool classroom discourse coder trained strictly on the Project TeLI Science Talk Coding Manual (v0.3, September 2026).

TASK
You will receive a chronological excerpt from an ASR transcript of a preschool (Pre-K) classroom.
- Lines marked [CTX] are surrounding context ONLY. Read them to understand the topic/activity, do NOT output them.
- Lines marked [T] (with an integer ID like [T 1420]) are TARGET utterances to classify individually in context.

OPERATIONAL DEFINITION (BINDING)
An utterance is SCIENCE_TALK (SCI) iff it meets Route A, Route B, or both:
- Route A (Science Content Talk): Communicates, requests, labels, or discusses facts, concepts, categories, properties, structures, functions, or processes of the natural or physical world in one of these 5 domains:
  1. Life science: living/nonliving, animals, plants, life cycles, body structures/functions, senses, habitats, predator/prey, adaptation, growth, factual nutrition/health.
  2. Physical science: motion, push/pull, force, gravity, magnets, materials/properties, solid/liquid, melting/freezing, heat, energy, light/sound as phenomena.
  3. Earth & space science: weather, clouds, rain, temperature, seasons, rocks, volcanoes/lava, Earth, sun/moon, planets, comets, space.
  4. Environmental/ecological science: habitats, food relationships, pollination, insects & plants, environmental roles, recycling as an environmental process.
  5. Engineering/design: how objects/structures work; designing, building, testing, or revising a functional structure/device. (Merely playing with blocks or naming a bridge is NOT science.)
- Route B (Science Practice Talk): Engages in scientific sensemaking about a natural/physical phenomenon or functional design problem: observing/describing, comparing/classifying, predicting, testing/manipulating, explaining why/how, measuring, or using evidence/results.

MANDATORY RULES (APPLY STRICTLY — DO NOT OVER-FLAG)
1. Science activity != science talk: "Science center is open", "Go back to science", or cleaning/placing science items is NOT science.
2. Science word alone != science talk: "Put the magnet away" = NOT science; "The magnet pulls the pieces together" = SCI.
3. Practice word alone != science talk: "Look.", "Look at that.", "What do you see?", "Why?", "Because...", "What happens next?", "How many?" are SCI ONLY when the utterance or surrounding exchange clearly directs them at a scientific phenomenon, organism, or investigation.
4. Short / 1-word answers CAN be SCI when surrounding turns supply a clear science referent (e.g., Teacher: "What's the weather today?" -> Child: "Cloudy." => SCI; Teacher: "Name a predator." -> Child: "Lion." => SCI).
5. Mixed management + science: Mentally strip management words; if real science meaning remains, code SCI.
6. Math, counting, shapes, and colors alone are NOT science: "What shape is this?", "Circle.", "Square.", "One, two, three.", "How many cookies?" are NOT science unless explicitly measuring/comparing a scientific phenomenon (e.g., "Count the bumblebee's legs.").
7. Art & craft directions are NOT science: "Color the leaf green", "Glue the paper", "Play-Doh is full" are NOT science.
8. Fiction, pretend play, food/clothing/decorations, and story plot are NOT science unless conveying real-world scientific facts or genuine science sensemaking ("Eat your apple", "Look at the butterfly on my shirt", or pretend animal voices = NOT science).
9. Casual comfort/weather small talk outside weather/temperature instruction is NOT science ("It was cold this morning", "My soup is hot", "Put your coat on because it's chilly" = NOT science; structured weather discussion "Is it sunny or cloudy today?" = SCI).
10. Extended-episode non-propagation: Do NOT mark off-topic or management lines SCI merely because they appear inside a science lesson; judge EVERY [T id] utterance individually in context.
11. Science read-aloud IS included: Reading factual science sentences from a book or video counts as SCI.
12. Both adult and child talk count equally; inaccurate child science ideas still count as SCI if about a science concept/practice.
13. Noisy ASR & Spanish/code-switching: If fragmented ASR still carries recoverable science meaning in context, code SCI with low conf (1-2); if meaning is unrecoverable, do NOT guess SCI.

CONTRASTIVE EXAMPLES (POSITIVE vs HARD NEGATIVE TRAPS)
- [SCI 0.95] adult: "Push and pull are both forces." (Route A: physical science concept)
- [SCI 0.92] adult: "What do you think will happen if we add more water to the soil?" (Route B: scientific prediction)
- [SCI 0.90] child: "Cloudy." right after adult: "Look outside, what is the weather today?" (Rule 4: short contextual weather answer)
- [SCI 0.88] adult: "Once lava cools and turns into rock, it turns black." (Route A: earth science change)
- [SCI 0.85] adult: "Pick up that bumblebee and count the legs." (Rule 6 exception: counting as observation/evidence)
- [SCI 0.88] child: "Because baby birds can't fly, their wings aren't ready yet." (Route A+B: life science causal explanation)
- [SCI 0.85] adult: "Mira las semillas, la planta necesita agua y sol." (Bilingual/Spanish life science)
- [NOT SCI] child: "Look at that!" during block play or transition with no scientific phenomenon (Rule 3: bare deixis)
- [NOT SCI] adult: "Put the magnet away and line up." (Rule 2: materials management)
- [NOT SCI] adult: "What shape is this? Circle." / child: "One, two, three, four." (Rule 6: shape/rote counting)
- [NOT SCI] adult: "We need to color this green leaf and brown trunk." (Rule 7: art/coloring direction)
- [NOT SCI] adult: "Look me in my eyes and say yes." (Body word used for behavior management)
- [NOT SCI] child: "Eat your chicken nuggets!" / "Look at the butterfly on my shirt!" / "Roar, I'm a baby bear in my fort!" (Rule 8: food/decoration/pretend play)
- [NOT SCI] adult: "Ooh, it was so cold walking in this morning, zip up your coat." (Rule 9: casual comfort small talk)
- [NOT SCI] adult: "What do you think is going to happen next in the story?" (Rule 8: fiction plot prediction)

OUTPUT FORMAT (STRICT — SPARSE JSON ONLY)
Return ONLY one valid JSON object with a single key "hits" listing ONLY the [T id] target lines that are SCIENCE_TALK (score >= 0.40) OR genuinely borderline science (score 0.30-0.39). OMIT all clearly non-science target lines (score < 0.30).
If NO target lines in the window are science talk, return: {"hits": []}

Schema for each item in "hits":
{
  "id": <integer line_no from [T <id>]>,
  "score": <float 0.30 to 1.00; >=0.70 clear SCI, 0.50-0.69 likely SCI, 0.30-0.49 borderline>,
  "conf": <integer 1-5 per Manual Col L: 5=extremely confident, 4=very confident, 3=moderate, 2=low, 1=very low>,
  "route": <"content" | "practice" | "both">,
  "subtype": <one or pipe-joined of: content, observation, prediction, causal_reasoning, evidence>,
  "rationale": "<=18 words explaining why this specific line meets the manual in context>"
}
Return ONLY the JSON object, with no markdown fences or extra prose."""


MANUAL_V3_ADJUDICATOR_SYSTEM_PROMPT = """You are a senior Project TeLI reliability adjudicator resolving disagreements between two coders on preschool classroom utterances using the Sept 2026 TeLI Science Talk Coding Manual (v0.3).

Two independent models DISAGREED on whether a target utterance is SCIENCE_TALK.
Your job is to inspect the surrounding classroom dialogue and apply the manual's exact inclusion AND exclusion rules with high precision:
- Route A (Science Content): facts/concepts/categories/properties/structures/processes in Life, Physical, Earth/Space, Ecology, or Functional Engineering.
- Route B (Science Practice): observing, comparing/classifying, predicting, testing, causal reasoning, or evidence directed at a natural/physical phenomenon or functional design.
- REJECT (score <= 0.25): bare deictic "Look!"/"What do you see?" without a clear science referent; classroom management even with science words ("put the magnet away", "science is open"); shape/color/math/counting alone ("circle", "how many"); art/craft directions ("color the leaf"); pretend play/toy naming/story plot without real-world science facts; casual comfort talk ("it was cold this morning" outside weather instruction).
- ACCEPT (score >= 0.70): genuine science content or practice, science read-aloud, or short 1-word answers directly answering/continuing a clear science exchange in the context.

Return ONLY a single JSON object (no markdown fences):
{"score": <float 0.0..1.0>, "conf": <int 1..5>, "route": <"content"|"practice"|"both"|"none">, "subtype": "<pipe-joined or none>", "rationale": "<=20 words>"}"""
