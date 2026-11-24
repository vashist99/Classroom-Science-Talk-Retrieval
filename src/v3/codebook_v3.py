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
    r"living|non-?living|alive|dead|organisms?|animals?|mammals?|"
    r"reptiles?|amphibians?|insects?|bugs?|arachnids?|"
    r"birds?|fish|whales?|sharks?|dolphins?|octopus(?:es)?|jellyfish|"
    r"seahorses?|sea\s+horses?|turtles?|iguanas?|lizards?|"
    r"snakes?|frogs?|tadpoles?|toads?|butterfl(?:y|ies)|caterpillars?|"
    r"chrysalis|cocoons?|larva|larvae|metamorphosis|bees?|"
    r"bumblebees?|ladybugs?|ants?|mosquito(?:es)?|grasshoppers?|crickets?|"
    r"mantis|spiders?|worms?|snails?|bats?|owls?|"
    r"elephants?|kangaroos?|skunks?|wol(?:f|ves)|deer|bears?|moose|mouse|mice|squirrels?|raccoons?|"
    r"penguins?|giraffes?|zebras?|monkeys?|hippos?|rhinos?|dinosaurs?|fossils?|"
    r"guinea\s+pigs?|opossums?|possums?|lions?|tigers?|"
    r"carnivores?|herbivores?|omnivores?|predators?|prey|nocturnal|"
    r"hibernat\w*|migrat\w*|camouflag\w*|adapt\w*|extinct\w*|endangered|"
    r"life\s*cycle|hatch\w*|eggs?|nests?|wings?|feathers?|scales?|fur|"
    r"fins?|gills?|beaks?|claws?|tails?|legs?|paws?|hooves?|teeth|tooth|antenn\w*|thorax|"
    r"abdomen|shells?|skeletons?|grow|grows|growing|grown|growth|decay\w*|mold|fungus|"
    r"plants?|seeds?|seedlings?|sprout\w*|germinat\w*|roots?|"
    r"stems?|trunks?|branch(?:es)?|leaf|leaves|petals?|flowers?|"
    r"blossoms?|pollen|pollinat\w*|nectar|photosynthes\w*|chlorophyll|"
    r"plant\s+bulb|pine\s+cones?|vines?|bark|sap|fruits?|vegetables?|nutrients?|"
    r"nutrition|healthy|vitamins?|protein|digestion|digest\w*|"
    r"hearts?|heartbeat|pulse|blood|veins?|arter\w*|bones?|spine|skull|ribs?|"
    r"lungs?|breath\w*|oxygen|muscles?|brains?|nerves?|stomach|skin|"
    r"senses?|five\s+senses|sense\s+of|germs?|bacteria|virus|microscopes?|"
    r"veterinarians?|botanists?|biologists?|scientists?"
    r")\b",
    re.I,
)

PHYSICAL_SCIENCE_RE = re.compile(
    r"\b("
    r"forces?|push(?:es|ed|ing)?|pull(?:s|ed|ing)?|motion|gravity|friction|inertia|"
    r"momentum|magnets?|magnetic|attract\w*|repel\w*|poles?|metals?|iron|"
    r"steel|solids?|liquids?|gas|gases|atoms?|molecules?|matter|states?\s+of\s+matter|"
    r"melt|melts|melting|melted|freeze|freezes|freezing|frozen|evaporat\w*|"
    r"condens\w*|boil\w*|steam|vapor|ice\s+cubes?|absorb\w*|dissolv\w*|mix|mixture|"
    r"solution|chemical|reaction|bubbles?|fizz\w*|foam|float|floats|"
    r"floating|sink|sinks|sinking|sank|buoyan\w*|dense|density|heavy|heavier|"
    r"heaviest|light|lighter|lightest|weight|weigh\w*|mass|volume|balance|"
    r"energy|heat|thermal|electric\w*|batter(?:y|ies)|circuits?|wires?|bulbs?|light\s+bulbs?|"
    r"shadows?|reflect\w*|prisms?|rainbows?|transparent|opaque|translucent|"
    r"sounds?|vibrat\w*|pitch|loud|louder|echo|waves?|"
    r"materials?|propert(?:y|ies)|textures?|rough|smooth|bumpy|"
    r"hard|soft|flexible|rigid|elastic|stretch\w*|absorbent|waterproof|"
    r"sturdy|fragile|slopes?|ramps?|incline|inclined|speed|faster|slower|"
    r"roll(?:s|ed|ing)?|bounce(?:s|d|ing)?|stick(?:s|ing)?|stuck"
    r")\b",
    re.I,
)

EARTH_SPACE_RE = re.compile(
    r"\b("
    r"weather|sunny|cloudy|clouds?|windy|wind|rain|raining|rainy|raindrops?|storms?|"
    r"thunder|lightning|tornado(?:es)?|hurricanes?|snow|snowing|snowflakes?|hail|fog|foggy|mist|"
    r"temperature|thermometers?|degrees?|fahrenheit|celsius|forecast|climate|"
    r"seasons?|spring|summer|autumn|fall\s+leaves|winter|frost|drought|"
    r"floods?|erosion|water\s+cycle|precipitation|puddles?|"
    r"earth|sun|sunlight|sunshine|solar|moons?|lunar|crescent|full\s+moon|"
    r"constellations?|planets?|mercury|venus|mars|jupiter|saturn|uranus|"
    r"neptune|orbit|comets?|asteroids?|meteors?|galax(?:y|ies)|universe|"
    r"astronauts?|telescopes?|space|rockets?|craters?|eclipse|rotate|rotation|"
    r"volcano(?:es)?|magma|lava|eruptions?|erupt\w*|earthquakes?|rocks?|"
    r"minerals?|crystals?|geodes?|pebbles?|sand|soil|dirt|mud|clay|"
    r"oceans?|rivers?|lakes?|ponds?|glaciers?|mountains?|valleys?|canyons?|islands?|deserts?"
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
    r"\bhow\s+many\s+(?:friends|kids|children|words|letters|cookies|blocks|stickers|days|minutes|cubes|sides|corners|shapes)\b)",
    re.I,
)

# USF Feedback Strict Exclusion Detectors (Shapes, Broad Counting, Look/Observe without Content)
TRAP_SHAPE_ONLY_RE = re.compile(
    r"(?<!pine\s)(?<!in\s)\b(what\s+shape|shapes?|circles?|squares?|triangles?|"
    r"rectangles?|ovals?|diamonds?|rhombus(?:es)?|pentagons?|hexagons?|octagons?|trapezoids?|"
    r"cylinders?|cones?|cubes?|spheres?|2d|3d)\b",
    re.I,
)

TRAP_COUNT_BROAD_RE = re.compile(
    r"(\bhow\s+many\b|\blet'?s\s+count\b|\bcount(?:ing|ed|s)?\b|"
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s*[,\s]+\s*(?:two|three|four|five|six|seven|eight|nine|ten)\b)",
    re.I,
)

LOOK_OBS_ANY_RE = re.compile(
    r"\b(look|looking|looks|looked|lookie|observe|observing|observation|notice|noticing|noticed|see|seeing|watch|watching)\b",
    re.I,
)

TRAP_SHAPE_COLOR_RE = re.compile(
    r"(?<!pine\s)(?<!in\s)\b(what\s+shape|what\s+color|shapes?|circles?|squares?|triangles?|"
    r"rectangles?|ovals?|diamonds?|rhombus(?:es)?|pentagons?|hexagons?|octagons?|trapezoids?|"
    r"cylinders?|cones?|cubes?|spheres?|star\s+shape|heart\s+shape|2d|3d|"
    r"color\s+(?:it|this|the)|crayons?|markers?|"
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

    p_obs_raw = len(PRACTICE_OBSERVE_RE.findall(s))
    # USF strict rule: looking/observing only counts as positive observation cue when paired with a science content word!
    p_obs = p_obs_raw if c_content_total > 0 else 0
    p_comp = len(PRACTICE_COMPARE_RE.findall(s))
    p_pred = len(PRACTICE_PREDICT_RE.findall(s))
    p_test = len(PRACTICE_TEST_RE.findall(s))
    p_caus = len(PRACTICE_CAUSAL_RE.findall(s))
    p_evid = len(PRACTICE_EVIDENCE_RE.findall(s))
    p_total = p_obs + p_comp + p_pred + p_test + p_caus + p_evid

    t_deictic = 1.0 if TRAP_DEICTIC_BARE_RE.match(s) else 0.0
    t_count = 1.0 if TRAP_COUNT_BARE_RE.search(s) else 0.0
    t_shape = float(len(TRAP_SHAPE_COLOR_RE.findall(s)))
    t_shape_only = 1.0 if (TRAP_SHAPE_ONLY_RE.search(s) and c_content_total == 0) else 0.0
    t_count_only = 1.0 if (TRAP_COUNT_BROAD_RE.search(s) and c_content_total == 0) else 0.0
    t_look_no_content = 1.0 if (LOOK_OBS_ANY_RE.search(s) and c_content_total == 0) else 0.0
    t_mgmt = float(len(TRAP_MANAGEMENT_RE.findall(s)))
    t_lit = float(len(TRAP_LITERACY_CRAFT_RE.findall(s)))
    t_temp = float(len(TRAP_TEMP_CASUAL_RE.findall(s)))
    t_pretend = float(len(TRAP_PRETEND_FOOD_RE.findall(s)))

    spk = (speaker or "unknown").strip().lower()
    trap_sum = t_deictic + t_count + t_shape + t_shape_only + t_count_only + 0.5 * t_look_no_content + t_mgmt + t_lit + t_pretend
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
        "trap_shape_only": t_shape_only,
        "trap_count_only": t_count_only,
        "trap_look_no_content": t_look_no_content,
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
3. STRICT Looking / Observing Rule: Talk about "looking", "seeing", "watching", "noticing", or "observing" (e.g., "Look.", "Look at that.", "What do you see?", "Does yours look like mine?", "Let's look closely.") MUST be EXCLUDED (NOT science) UNLESS EITHER (a) the utterance itself includes a real science content word/organism/phenomenon (e.g., "Look at the caterpillar's legs", "Observe how the ice melts"), OR (b) it sits directly inside an unmistakable >=3-utterance active science investigation of a specific scientific phenomenon. Likewise, bare "Why?", "Because...", "What happens next?" require a clear science referent.
4. Short / 1-word answers CAN be SCI when surrounding turns supply a clear science referent (e.g., Teacher: "What's the weather today?" -> Child: "Cloudy." => SCI; Teacher: "Name a predator." -> Child: "Lion." => SCI).
5. Mixed management + science: Mentally strip management words; if real science meaning remains, code SCI.
6. STRICT Shapes & Counting Exclusion: Talk about shapes/geometry ("What shape is this?", "Circle.", "Square.", "Triangle.", "Rectangle.", "Cone.", "Sphere.", "Aaron made a rectangle.") and counting ("How many?", "One, two, three, four...", "How many cubes/blocks/cookies?") MUST be EXCLUDED (NOT science). Counting counts ONLY when explicitly measuring/counting an anatomical or physical science property (e.g., "Count the bumblebee's legs.", "How many seeds sprouted?").
7. Art & craft directions are NOT science: "Color the leaf green", "Glue the paper", "Play-Doh is full" are NOT science.
8. Fiction, pretend play, food/clothing/decorations, and story plot are NOT science unless conveying real-world scientific facts or genuine science sensemaking ("Eat your apple", "Look at the butterfly on my shirt", or pretend animal voices = NOT science).
9. Casual comfort/weather small talk outside weather/temperature instruction is NOT science ("It was cold this morning", "My soup is hot", "Put your coat on because it's chilly" = NOT science; structured weather discussion "Is it sunny or cloudy today?" = SCI).
10. Extended-episode non-propagation: Do NOT mark off-topic or management lines SCI merely because they appear inside a science lesson; judge EVERY [T id] utterance individually in context.
11. Science read-aloud IS included: Reading factual science sentences from a book or video counts as SCI.
12. Both adult and child talk count equally; inaccurate child science ideas still count as SCI if about a science concept/practice.
13. Noisy ASR & Spanish/code-switching: If fragmented ASR still carries recoverable science meaning in context, code SCI with low conf (1-2); if meaning is unrecoverable, do NOT guess SCI.
14. Episode Cluster Prior: True preschool science talk rarely occurs as a single stand-alone utterance — it typically appears in clusters of >=3 science-related utterances together. Be extra conservative on isolated 1-turn candidates unless they unmistakably state a clear science concept.

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
- REJECT (score <= 0.25): talk about "looking"/"seeing"/"watching"/"observing" without a science content word (unless inside an unmistakable >=3-utterance active science investigation); classroom management even with science words ("put the magnet away", "science is open"); ALL shapes/geometry ("circle", "square", "rectangle", "cone", "what shape?"); counting ("how many?", "one, two, three, four...") unless explicitly counting/measuring a biological or physical science property; art/craft directions ("color the leaf"); pretend play/toy naming/story plot without real-world science facts; casual comfort talk ("it was cold this morning" outside weather instruction); weak isolated 1-turn hits without episode support.
- ACCEPT (score >= 0.70): genuine science content or practice, science read-aloud, or short 1-word answers directly answering/continuing a clear >=3-utterance science exchange in the context.

Return ONLY a single JSON object (no markdown fences):
{"score": <float 0.0..1.0>, "conf": <int 1..5>, "route": <"content"|"practice"|"both"|"none">, "subtype": "<pipe-joined or none>", "rationale": "<=20 words>"}"""


# ---------------------------------------------------------------------------
# USF Feedback Helpers: Dirty Legacy TP Cleanup + Soft ±2 Cluster & Stage-2 Veto
# ---------------------------------------------------------------------------

def identify_legacy_excluded_tps(
    df: Any,
    *,
    pause_cut_s: float = 30.0,
) -> Any:
    """Return boolean mask of legacy `Human coded` label==1 rows that violate USF's updated exclusions.

    Used ONLY to clean/downweight Stage-1 training folds (NEVER alters evaluation ground truth!):
      1. Pure shape/geometry TPs with zero science content word (e.g., "Aaron made a rectangle.").
      2. Pure counting TPs with zero science content word (e.g., "How many cubes do you have?").
      3. Look/observe TPs with zero science content word AND fewer than 2 content-bearing turns in ±2
         (e.g., "Does yours look like mine?", "Look at this, you get different colors.").
    """
    import numpy as np

    n = len(df)
    labels = df["label"].to_numpy()
    shape_only = df["cb_trap_shape_only"].to_numpy(dtype=np.float32) > 0
    count_only = df["cb_trap_count_only"].to_numpy(dtype=np.float32) > 0
    look_noc = df["cb_trap_look_no_content"].to_numpy(dtype=np.float32) > 0
    has_c = df["cb_has_content"].to_numpy(dtype=np.float32) > 0

    # Check if a look_no_content turn sits inside a clear content-rich ±2 exchange (>=2 content turns in ±2)
    tx_ids = df["transcript_id"].to_numpy()
    gaps = df["gap_prev_s"].fillna(0.0).to_numpy(dtype=np.float32)
    starts = np.flatnonzero(np.r_[True, tx_ids[1:] != tx_ids[:-1]])
    ends = np.r_[starts[1:], n]
    pm2_content_cnt = np.zeros(n, dtype=np.int32)
    for s, e in zip(starts, ends):
        gp = gaps[s:e]
        eb = np.flatnonzero(gp > pause_cut_s)
        for es, ee in zip(np.r_[0, eb], np.r_[eb, e - s]):
            gs = s + es
            L = ee - es
            hc = has_c[gs:gs + L]
            for i in range(L):
                lo, hi = max(0, i - 2), min(L, i + 3)
                pm2_content_cnt[gs + i] = int(hc[lo:hi].sum())

    dirty = (labels == 1) & (shape_only | count_only | (look_noc & (pm2_content_cnt < 2)))
    return dirty


def clean_training_labels_for_usf(df: Any) -> Any:
    """Return a copy of `df['label']` cleaned STRICTLY for Stage-1/PGKD training folds.

    NEVER used for evaluation (evaluation always scores against untouched original `df['label']`).
      - Pure shape-only & counting-only legacy TPs (e.g., "Aaron made a rectangle.",
        "How many cubes do you have?") -> -2 (masked out of positive training so Stage 1
        stops upweighting legacy shape/counting errors by 4.5x without causing degenerate
        linear feature separation on Y1!).
      - Isolated bare deictic/look-without-content legacy TPs that lack content neighbors
        AND match bare look/see traps -> -2 (masked out of positive training).
    """
    import numpy as np

    train_lb = df["label"].to_numpy(dtype=np.int8).copy()
    labels = df["label"].to_numpy()
    shape_only = df["cb_trap_shape_only"].to_numpy(dtype=np.float32) > 0
    count_only = df["cb_trap_count_only"].to_numpy(dtype=np.float32) > 0
    bare_deictic = df["cb_trap_deictic_bare"].to_numpy(dtype=np.float32) > 0
    # Mask ONLY unambiguous dirty legacy TPs (never flip to 0 with degenerate feature col!)
    dirty_mask = (labels == 1) & (shape_only | count_only | bare_deictic)
    train_lb[dirty_mask] = -2
    return train_lb


def apply_usf_cluster_and_veto_rules(
    df: Any,
    cal_probs: Any,
    *,
    llm_scores: Any | None = None,
    llm_eval: Any | None = None,
    support_thr: float = 0.05,
    min_cluster: int = 3,
    penalty_logit: float = 0.60,
    pause_cut_s: float = 30.0,
) -> tuple[Any, Any]:
    """Apply USF-aligned ±2 chronological cluster rule & Stage-2 LLM exclusion veto (zero leakage!).

    1. Pause-bounded ±2 window `[t-2, t-1, t, t+1, t+2]` inside each transcript:
       Computes `pm2_sci_count` = count of turns in ±2 with `cal_probs >= support_thr`.
       If `pm2_sci_count < min_cluster` (3), applies soft `-penalty_logit` (-0.60) in log-odds space.
    2. Stage-2 LLM Veto Gate (fixes Screening-mode Stage-1 overrides!):
       When Stage-2 LLMs evaluated a turn (`llm_eval == True`) and cleanly rejected it (`llm_scores <= 0.15`),
       ANY USF exclusion trap (`trap_shape_only`, `trap_count_only`, or `trap_look_no_content` with `<3` cluster)
       receives a decisive `-3.5` logit veto so Stage 1 cannot drag LLM-rejected shapes/counting/look turns
       above the Screening threshold.
    3. Local-only Trap Guardrail (when `llm_eval` is None):
       Applies `-2.5` logit damping on pure `trap_shape_only` / `trap_count_only` turns outside a ≥3 cluster,
       and `-1.2` logit damping on `trap_look_no_content` turns outside a ≥3 cluster.

    Returns:
      (adjusted_probs: float64[N], pm2_sci_count: int32[N])
    """
    import numpy as np
    from scipy.special import expit, logit

    n = len(df)
    p = np.clip(np.asarray(cal_probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    z = logit(p)

    # 1. Pause-bounded ±2 chronological support count
    pm2_cnt = np.zeros(n, dtype=np.int32)
    tx_ids = df["transcript_id"].to_numpy()
    gaps = df["gap_prev_s"].fillna(0.0).to_numpy(dtype=np.float64)
    starts = np.flatnonzero(np.r_[True, tx_ids[1:] != tx_ids[:-1]])
    ends = np.r_[starts[1:], n]
    for s, e in zip(starts, ends):
        gp = gaps[s:e]
        eb = np.flatnonzero(gp > pause_cut_s)
        for es, ee in zip(np.r_[0, eb], np.r_[eb, e - s]):
            gs = s + es
            L = ee - es
            pw = p[gs:gs + L]
            hi_flags = (pw >= support_thr).astype(np.int32)
            cs = np.r_[0, np.cumsum(hi_flags)]
            idx = np.arange(L)
            lo = np.maximum(0, idx - 2)
            hi = np.minimum(L, idx + 3)
            pm2_cnt[gs:gs + L] = cs[hi] - cs[lo]

    isolated = pm2_cnt < min_cluster
    z = np.where(isolated, z - penalty_logit, z)

    # 2. Extract USF trap flags (from df columns if present, else compute cleanly on the fly)
    if "cb_trap_shape_only" in df.columns and "cb_trap_count_only" in df.columns and "cb_trap_look_no_content" in df.columns:
        shape_only = df["cb_trap_shape_only"].to_numpy(dtype=np.float32) > 0
        count_only = df["cb_trap_count_only"].to_numpy(dtype=np.float32) > 0
        look_noc = df["cb_trap_look_no_content"].to_numpy(dtype=np.float32) > 0
    else:
        utts = df["utterance"].fillna("").astype(str).tolist()
        spks = df["speaker"].fillna("unknown").astype(str).tolist()
        feats = [extract_codebook_features(u, s) for u, s in zip(utts, spks)]
        shape_only = np.array([f["trap_shape_only"] > 0 for f in feats], dtype=bool)
        count_only = np.array([f["trap_count_only"] > 0 for f in feats], dtype=bool)
        look_noc = np.array([f["trap_look_no_content"] > 0 for f in feats], dtype=bool)

    usf_trap_strict = shape_only | count_only | (look_noc & isolated)

    if llm_scores is not None and llm_eval is not None:
        ev = np.asarray(llm_eval, dtype=bool)
        ls = np.asarray(llm_scores, dtype=np.float64)
        # Stage-2 LLM Veto: both LLMs rejected (score <= 0.15) on an explicit USF trap -> veto Stage-1 override!
        llm_veto = ev & (ls <= 0.15) & (shape_only | count_only | look_noc)
        z = np.where(llm_veto, z - 3.5, z)
        # Also damp un-evaluated pure shape/count traps outside a ≥3 cluster
        local_fallback = (~ev) & (shape_only | count_only) & isolated
        z = np.where(local_fallback, z - 2.5, z)
    else:
        # Local-only ($0-API) USF trap guardrail
        z = np.where((shape_only | count_only) & isolated, z - 2.5, z)
        z = np.where(look_noc & isolated, z - 1.2, z)

    return expit(np.clip(z, -30.0, 30.0)), pm2_cnt


def extract_pm2_context_columns(
    df: Any,
    *,
    pause_cut_s: float = 30.0,
    max_chars_per_line: int = 200,
) -> tuple[list[str], list[str]]:
    """Build chronological ±2 context strings (`context_before_2`, `context_after_2`) from full raw transcripts.

    Respects single-transcript boundaries and stops at `>pause_cut_s` silence breaks (marking
    `[>30s pause]` so human coders immediately see why fewer than 2 lines appear).
    Includes `unknown`/low-confidence turns in chronological order so dialogue flow is preserved.
    """
    import numpy as np

    n = len(df)
    ctx_before = [""] * n
    ctx_after = [""] * n
    if n == 0:
        return ctx_before, ctx_after

    tx_ids = df["transcript_id"].to_numpy()
    spks = df["speaker"].fillna("unknown").astype(str).tolist()
    utts = df["utterance"].fillna("").astype(str).tolist()
    gaps = df["gap_prev_s"].fillna(0.0).to_numpy(dtype=np.float64)
    confs = df["asr_conf"].to_numpy(dtype=np.float64) if "asr_conf" in df.columns else np.full(n, np.nan)

    def _fmt(idx: int) -> str:
        spk = spks[idx]
        c = confs[idx]
        tag = f"{spk} (low-conf)" if (np.isfinite(c) and c < 0.40) else spk
        txt = " ".join(utts[idx].split())[:max_chars_per_line]
        return f"{tag}: {txt}"

    starts = np.flatnonzero(np.r_[True, tx_ids[1:] != tx_ids[:-1]])
    ends = np.r_[starts[1:], n]
    for s, e in zip(starts, ends):
        for i in range(s, e):
            # Up to 2 turns immediately before `i`
            pre: list[str] = []
            j = i - 1
            while j >= s and len(pre) < 2:
                if gaps[j + 1] > pause_cut_s:
                    pre.append(f"[>{int(pause_cut_s)}s pause]")
                    break
                pre.append(_fmt(j))
                j -= 1
            pre.reverse()
            ctx_before[i] = "\n".join(pre)

            # Up to 2 turns immediately after `i`
            post: list[str] = []
            j = i + 1
            while j < e and len(post) < 2:
                if gaps[j] > pause_cut_s:
                    post.append(f"[>{int(pause_cut_s)}s pause]")
                    break
                post.append(_fmt(j))
                j += 1
            ctx_after[i] = "\n".join(post)

    return ctx_before, ctx_after
