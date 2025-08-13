import streamlit as st
import pandas as pd
import re
import io
import json
import time
from typing import List, Dict, Tuple
from openai import OpenAI, RateLimitError

# =========================
# OpenAI client (v1 syntax)
# =========================
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))

# =========================
# Config / Options
# =========================
DEFAULT_MAX_ROWS = 100  # default cap for testing; user can change in UI
LEARNING_STORE_FILENAME = "address_learning_store.jsonl"  # downloadable/uploadable
SUGGESTIONS_EXPORT = "mismatch_suggestions.csv"

# USPS state codes (for stripping city/state/ZIP tails)
USPS_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','IA','ID','IL','IN','KS','KY','LA','MA','MD','ME','MI','MN','MO','MS','MT','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OK','OR','PA','PR','RI','SC','SD','TN','TX','UT','VA','VI','VT','WA','WI','WV','WY'
}

# USPS-style street types (expanded)
USPS_STREET_TYPES = {
    # Common
    "alley": "aly", "allee": "aly", "ally": "aly",
    "annex": "anx", "annx": "anx",
    "arcade": "arc", "av": "ave", "aven": "ave", "avenue": "ave", "avenu": "ave", "avnue": "ave",
    "bayou": "byu", "beach": "bch", "bend": "bnd", "bluff": "blf", "bluffs": "blfs",
    "boulevard": "blvd", "boul": "blvd", "boulv": "blvd",
    "branch": "br", "bridge": "brg", "brook": "brk", "brooks": "brks",
    "burg": "bg", "burgs": "bgs", "bypass": "byp",
    "camp": "cp", "canyon": "cyn", "cape": "cpe", "causeway": "cswy",
    # IMPORTANT: Do not blindly map 'center/centre' everywhere; only if terminal suffix handling applies
    "center": "ctr", "centers": "ctrs",
    "circle": "cir", "circl": "cir", "circ": "cir", "circles": "cirs",
    "cliff": "clf", "cliffs": "clfs", "club": "clb",
    "common": "cmn", "commons": "cmns",
    "corner": "cor", "corners": "cors",
    "course": "crse", "court": "ct", "courts": "cts",
    "cove": "cv", "coves": "cvs", "creek": "crk", "crescent": "cres",
    "crossing": "xing", "dale": "dl", "dam": "dm", "divide": "dv", "drive": "dr",
    "estates": "ests", "estate": "est",
    "expressway": "expy",
    "extension": "ext", "extensions": "exts",
    "falls": "fls", "fall": "fl", "ferry": "fry", "field": "fld", "fields": "flds",
    "flat": "flt", "flats": "flts", "ford": "frd", "fords": "frds",
    "forest": "frst", "forge": "fgr", "forges": "fgrs",
    "fork": "frk", "forks": "frks", "fort": "ft",
    "freeway": "fwy", "garden": "gdn", "gardens": "gdns", "gateway": "gtwy",
    "glen": "gln", "glens": "glns", "green": "grn", "greens": "grns",
    "grove": "grv", "groves": "grvs",
    "harbor": "hbr", "harbour": "hbr", "harbors": "hbrs", "haven": "hvn",
    "heights": "hts", "highway": "hwy", "hill": "hl", "hills": "hls",
    "hollow": "holw",
    "inlet": "inlt", "island": "is", "islands": "iss", "isle": "isle",
    "junction": "jct", "junctions": "jcts",
    "key": "ky", "keys": "kys",
    "knoll": "knl", "knolls": "knls",
    "lake": "lk", "lakes": "lks", "landing": "lndg", "lane": "ln",
    "light": "lgt", "lights": "lgts",
    "loaf": "lf", "lock": "lck", "locks": "lcks",
    "lodge": "ldg", "mall": "mall", "manor": "mnr", "manors": "mnrs",
    "meadow": "mdw", "meadows": "mdws", "mews": "mews",
    "mill": "ml", "mills": "mls", "mission": "msn",
    "mount": "mt", "mountain": "mtn", "mountains": "mtns",
    "neck": "nck", "orchard": "orch", "oval": "oval",
    "overpass": "opas", "park": "park", "parks": "park",
    "parkway": "pkwy",
    "pass": "pass", "passage": "psge",
    "path": "path", "pike": "pike", "pine": "pne", "pines": "pnes",
    "place": "pl", "plain": "pln", "plains": "plns",
    "plaza": "plz", "point": "pt", "points": "pts",
    "port": "prt", "ports": "prts", "prairie": "pr", "radial": "radl",
    "ramp": "ramp", "ranch": "rnch", "rapid": "rpd", "rapids": "rpds",
    "rest": "rst", "ridge": "rdg", "ridges": "rdgs",
    "river": "riv", "road": "rd", "roads": "rds",
    "route": "rte", "row": "row", "run": "run",
    "shoal": "shl", "shoals": "shls",
    "shore": "shr", "shores": "shrs",
    "skyway": "skwy", "spring": "spg", "springs": "spgs",
    "spur": "spur", "spurs": "spur",
    "square": "sq", "squares": "sqs",
    "station": "sta", "stravenue": "stra", "stream": "strm",
    "street": "st", "streets": "sts",
    "summit": "smt", "terrace": "ter",  # NOTE: your reference data may prefer 'terr'; handle in learning
    "throughway": "trwy", "trace": "trce", "track": "trak",
    "trafficway": "trfy", "trail": "trl", "trailer": "trlr",
    "tunnel": "tunl", "turnpike": "tpke",
    "underpass": "upas", "union": "union", "valley": "vly", "valleys": "vlys",
    "viaduct": "via", "view": "vw", "views": "vws",
    "village": "vlg", "villages": "vlgs",
    "ville": "vl", "vista": "vis",
    "walk": "walk", "walks": "walk",
    "wall": "wall", "way": "way",
    "well": "wl", "wells": "wls"
}

# Directionals
DIRECTIONALS = {
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
}
DIR_TOKENS = set(["n","s","e","w","ne","nw","se","sw"])  # for quick checks

# Proper-name directional streets to preserve (do NOT abbreviate)
PROPER_DIRECTIONAL_STREETS = {
    "west st", "east st", "north st", "south st",
    "west ave", "east ave", "north ave", "south ave",
}

# Compound names to preserve (extendable via sidebar input)
DEFAULT_COMPOUNDS = [
    "northwood", "southgate", "eastwood", "westfield",
    "northridge", "southridge", "eastview", "westpark",
    "mount carmel", "green valley", "ridgeview", "lake shore",
    "pine hill", "maple grove", "cedar point", "oak ridge",
    "willow bend", "river road", "sunset boulevard", "highland avenue",
    "park place", "garden lane", "college drive", "union square",
    "harbor view", "brook stone", "fox run", "mill creek",
    "spring meadow", "pleasant hill", "fair view", "meadow brook",
    "stone bridge", "bridge way", "silver spring", "golden gate",
] + list(PROPER_DIRECTIONAL_STREETS)

# =========================
# Helpers: compound handling
# =========================

def protect_compounds(s: str, compounds: List[str]) -> str:
    s_low = str(s)
    for name in compounds:
        token = name.replace(" ", "")
        parts = name.split()
        patterns = [rf"\b{re.escape(name)}\b"]
        if len(parts) == 2:
            patterns.append(rf"\b{re.escape(parts[0])}\s+{re.escape(parts[1])}\b")
        for pat in patterns:
            s_low = re.sub(pat, token, s_low, flags=re.IGNORECASE)
    return s_low


def restore_compounds(s: str, compounds: List[str]) -> str:
    out = str(s)
    for name in compounds:
        token = name.replace(" ", "")
        # Restore with the original spacing/case titleized
        canonical = name.title()
        out = re.sub(rf"\b{re.escape(token)}\b", canonical, out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    return out

# =========================
# Adaptive Learning Store
# =========================

def get_store() -> Dict:
    if "learning_pairs" not in st.session_state:
        st.session_state.learning_pairs = []  # list of {raw, predicted, correct}
    if "custom_rules" not in st.session_state:
        # each rule: {"pattern": str, "replacement": str, "where": "pre"|"post"}
        st.session_state.custom_rules = []
    if "fewshot_examples" not in st.session_state:
        # list of (input_line, output_line)
        st.session_state.fewshot_examples = []
    return {
        "learning_pairs": st.session_state.learning_pairs,
        "custom_rules": st.session_state.custom_rules,
        "fewshot_examples": st.session_state.fewshot_examples,
    }


def load_learning_store(uploaded_file):
    store = get_store()
    if uploaded_file is None:
        return store
    # Accept either JSONL or JSON
    content = uploaded_file.read().decode("utf-8")
    if uploaded_file.name.endswith(".jsonl"):
        for line in content.splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            _ingest_store_obj(obj)
    else:
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                _ingest_store_obj(data)
            elif isinstance(data, list):
                for obj in data:
                    _ingest_store_obj(obj)
        except Exception:
            pass
    return get_store()


def _ingest_store_obj(obj: Dict):
    store = get_store()
    for k in ["learning_pairs", "custom_rules", "fewshot_examples"]:
        if k in obj and isinstance(obj[k], list):
            existing = store[k]
            for item in obj[k]:
                if item not in existing:
                    existing.append(item)


def export_learning_store() -> str:
    store = get_store()
    buf = io.StringIO()
    buf.write(json.dumps(store) + "\n")
    return buf.getvalue()

# =========================
# Light similarity for few-shot selection
# =========================

def _token_set(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def most_similar_examples(inp: str, k: int = 6) -> List[Tuple[str, str]]:
    exs = get_store()["fewshot_examples"]
    base = _token_set(inp)
    scored = []
    for item in exs:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        u, v = item[0], item[1]
        s = _token_set(str(u))
        j = (len(base & s) / max(1, len(base | s))) if (base or s) else 0.0
        scored.append((j, str(u), str(v)))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [(u, v) for (j, u, v) in scored[:k]]

# =========================
# Adaptive rules application
# =========================

def apply_custom_rules(s: str, where: str = "pre") -> str:
    for rule in get_store()["custom_rules"]:
        if rule.get("where", "pre") in (where, "both"):
            try:
                s = re.sub(rule["pattern"], rule["replacement"], s, flags=re.IGNORECASE)
            except re.error:
                continue
    return s

# =========================
# Utilities
# =========================

def strip_city_state_zip(s: str) -> str:
    """Remove trailing ZIP (5/9), and trailing state codes like 'NY' possibly repeated.
    Examples handled: '... 10462 NY NY' → drop trailing parts; '... NY 10462' → drop; '... 10462-1234' → drop.
    """
    tokens = re.findall(r"[A-Za-z0-9-]+", s)
    if not tokens:
        return s
    i = len(tokens)
    # Drop trailing ZIP or ZIP+4
    if i and re.fullmatch(r"\d{5}(-\d{4})?", tokens[i-1] or ""):
        i -= 1
    # Drop up to two trailing state codes
    removed_any = True
    count = 0
    while i and count < 2 and (tokens[i-1] or "").upper() in USPS_STATES:
        i -= 1
        count += 1
    # Rebuild from remaining tokens plus original spacing approximated
    return " ".join(tokens[:i]) if i else ""


def abbrev_terminal_suffix(s: str) -> str:
    """Only abbreviate the terminal street-type token (before unit/PO Box),
    to avoid changing parts of the proper street name (e.g., 'Centre Avenue' should keep 'Centre')."""
    # Split on spaces to inspect tail
    parts = s.split()
    if not parts:
        return s
    # Identify tail index that is part of unit or PO Box
    unit_labels = {"apt", "ste", "rm", "unit", "bsmt", "ph", "fl", "po", "box", "#"}
    stop = len(parts)
    # If a PO Box appears, leave suffixes alone
    try:
        po_index = [p.lower() for p in parts].index("po")
        # if 'PO Box' present, don't try to find a street suffix
        stop = min(stop, po_index)
    except ValueError:
        pass
    # Skip past any trailing unit chunks
    while stop > 0 and parts[stop-1].lower().strip(".,#") in unit_labels:
        stop -= 1
    # Candidate suffix at stop-1
    if stop == 0:
        return s
    cand = parts[stop-1].lower().strip(".,")
    if cand in USPS_STREET_TYPES:
        parts[stop-1] = USPS_STREET_TYPES[cand].title()
        return " ".join(parts)
    return s


def normalize_unit_token(tok: str) -> str:
    # Uppercase alphanum units like 4c → 4C, 24d → 24D, 4fe → 4FE
    if re.fullmatch(r"[0-9]+[a-z0-9-]*", tok.lower()):
        # keep digits and hyphens; uppercase letters
        return re.sub(r"[a-z]", lambda m: m.group(0).upper(), tok)
    return tok

# =========================
# Rule-based cleaner (updated, safer)
# =========================

def rule_clean(address: str, compounds: List[str], *,
               unit_style: str = "preserve",
               infer_unlabeled_unit: bool = False,
               keep_hash_if_present: bool = True,
               preserve_proper_directionals: bool = True) -> str:
    """
    unit_style: one of ['preserve','apt','hash','none']
    infer_unlabeled_unit: only infer when trailing token begins with a digit (avoid N/E/S/W)
    keep_hash_if_present: if raw has '#', preserve it as '# <val>' unless unit_style overrides
    preserve_proper_directionals: do not abbreviate 'West St', 'North Ave', etc.
    """
    if address is None or str(address).strip() == "":
        return ""

    raw_s = str(address)

    # Apply user/learned PRE rules early
    raw_s = apply_custom_rules(raw_s, where="pre")

    # Strip any trailing city/state/ZIP noise first
    s = strip_city_state_zip(raw_s)

    # Lower for normalization workflow
    s = s.lower().strip()

    # Normalize punctuation spacing early
    s = re.sub(r"\s*[.,]\s*", " ", s)

    # Protect compounds (and proper directional streets) from abbreviation
    s = protect_compounds(s, compounds)

    # Normalize inline '#<unit>' → mark for later formatting if preserving hash
    # Keep a flag if raw contained a hash
    had_hash = bool(re.search(r"(?:^|\s)#\s*[a-z0-9-]+\b", s, flags=re.IGNORECASE))
    s = re.sub(r'(?:^|\s)#\s*([a-z0-9-]+)\b', r' # \1', s, flags=re.IGNORECASE)

    # Unit synonyms, room, basement, floor, penthouse
    s = re.sub(r'\b(apartment|apt)\b\.?', 'apt', s, flags=re.IGNORECASE)
    s = re.sub(r'\b(suite|ste)\b\.?', 'ste', s, flags=re.IGNORECASE)
    s = re.sub(r'\b(room|rm)\b', 'rm', s, flags=re.IGNORECASE)
    s = re.sub(r'\bbasement\b', 'bsmt', s, flags=re.IGNORECASE)
    s = re.sub(r'\bfloor\b', 'fl', s, flags=re.IGNORECASE)
    s = re.sub(r'\bpenthouse\b', 'ph', s, flags=re.IGNORECASE)
    s = re.sub(r'\bph\b', 'ph', s, flags=re.IGNORECASE)  # keep PH/Ph as unit label

    # Remove stray commas/periods; keep hash placeholder we added above
    s = re.sub(r"[,.]", "", s)

    # Directionals (standalone only). If preserving proper directional streets, they are protected already.
    for word, abbr in DIRECTIONALS.items():
        s = re.sub(rf"\b{word}\b", abbr, s)

    # Apply terminal street-type abbreviation only
    s = abbrev_terminal_suffix(s)

    # Move leading unit to end (e.g., 'apt a 19204 39th ave' -> '19204 39th ave apt a')
    m = re.match(r'^(?:\s*(apt|ste|rm|unit|ph|fl|bsmt)\s+([a-z0-9-]+)\s+)(.+)$', s)
    if m:
        unit_lbl, unit_val, rest = m.groups()
        s = f"{rest.strip()} {unit_lbl} {unit_val}"

    # Optionally infer unlabeled trailing unit if token begins with a digit and is not a directional token
    if infer_unlabeled_unit and not re.search(r'\b(apt|ste|rm|unit|bsmt|ph|fl)\b', s, flags=re.IGNORECASE):
        m2 = re.search(r"\b([0-9][a-z0-9-]{0,7})$", s, flags=re.IGNORECASE)
        if m2:
            unit_tok = m2.group(1)
            # Avoid misreading directionals like 'N','S','E','W' which don't start with digit; safe here
            s = re.sub(r"\b([0-9][a-z0-9-]{0,7})$", r"apt " if unit_style in ("apt","preserve") else (r"# " if unit_style=="hash" else r""), s)

    # If we chose not to infer, but there is a trailing unit token, normalize its case (e.g., '4fe' → '4FE')
    # Also keep 'ph' as title case 'Ph'
    parts = s.split()
    if parts:
        last = parts[-1]
        if last not in {"apt","ste","rm","unit","ph","fl","bsmt","#"}:
            normalized_last = normalize_unit_token(last)
            # Don't uppercase 'ph' or 'fl' when they are labels
            parts[-1] = normalized_last
            s = " ".join(parts)

    # Unit label style conversion (post-process)
    if unit_style in ("apt","hash","none"):
        # Replace any existing style to requested
        # Patterns: 'apt X', 'ste X', 'rm X', '# X'
        if unit_style == "apt":
            s = re.sub(r"\b(ste|rm|unit)\s+([a-z0-9-]+)\b", r"apt ", s)
            s = re.sub(r"\s#\s*([a-z0-9-]+)\b", r" apt ", s)
        elif unit_style == "hash":
            s = re.sub(r"\b(apt|ste|rm|unit)\s+([a-z0-9-]+)\b", r"# ", s)
            if not had_hash and " # " not in s:
                # If there's already a trailing unlabeled unit token following a suffix, convert it to '#'
                s = re.sub(r"\b(ave|st|blvd|dr|rd|ln|ter|pl|ct|cir|pkwy|hwy|way|loop|trl|expy)\b\s+([0-9][a-z0-9-]{0,7})$", r" # ", s, flags=re.IGNORECASE)
        elif unit_style == "none":
            s = re.sub(r"\b(apt|ste|rm|unit)\s+([a-z0-9-]+)\b", r"", s)
            s = re.sub(r"\s#\s*([a-z0-9-]+)\b", r" ", s)

    # Spacing & title case & restore compounds
    s = re.sub(r"\s+", " ", s).strip()
    s = s.title()
    s = restore_compounds(s, compounds)
    # Ensure label capitalization for known units
    for lbl in ["Apt","Ste","Rm","Unit","Bsmt","Ph","Fl"]:
        s = re.sub(rf"\b{lbl}\b", lbl, s)

    # Apply user/learned POST rules
    s = apply_custom_rules(s, where="post")

    return s

# =========================
# LLM correction (address line) with adaptive few-shot
# =========================

def llm_correct(address_line: str, compounds: List[str], model_name: str, *,
                unit_style: str,
                infer_unlabeled_unit: bool,
                preserve_proper_directionals: bool) -> str:
    pre = protect_compounds(address_line, compounds)

    # Build few-shot block from learned examples (most similar first)
    fewshots = most_similar_examples(pre, k=6)
    ex_lines = []
    for ui, vo in fewshots:
        ex_lines.append(f"Input: '{ui}' → Output: '{vo}'")
    base_examples = [
        "Input: '1515 Summer St Unit 503' → Output: '1515 Summer St Unit 503'",
        "Input: '111 Centre Avenue unit 416' → Output: '111 Centre Ave 416'",  # unit style may remove label
        "Input: '337 Packman Avenue Bsmt' → Output: '337 Packman Ave Bsmt'",
        "Input: 'Apt. A 19204 39th Ave' → Output: '19204 39th Ave Apt A'",
        "Input: '2187 Cruger Ave #5C' → Output: '2187 Cruger Ave # 5C'",  # hash style example
        "Input: '2720 Grand Concourse 201' → Output: '2720 Grand Concourse 201'",
        "Input: '3154 Randall Avenue 3154 Randall Avenue' → Output: '3154 Randall Ave'",
        "Input: '42 Herriot St 4fe' → Output: '42 Herriot St 4FE'",
        "Input: '4126 Barnes Ave Ph' → Output: '4126 Barnes Ave Ph'",
        "Input: '195-197 Lembeck Ave Ground Floor' → Output: '195-197 Lembeck Ave Ground Fl'",
        "Input: '67 Circle Dr W' → Output: '67 Circle Dr W'",
        "Input: '261 West St Apt 2' → Output: '261 West St Apt 2'",
    ]
    fewshot_block = "\n".join(base_examples + ex_lines)

    # Dynamic policy derived from Word doc + dataset:
    unit_policy = {
        "preserve": "Preserve the unit style from input; if input used '#', keep '#'; if unlabeled trailing unit like '4C', keep unlabeled.",
        "apt": "Use 'Apt <value>' for units (convert '# <value>' and unlabeled units to 'Apt <value>').",
        "hash": "Use '# <value>' for units (convert 'Apt/Ste/Rm <value>' and unlabeled units to '# <value>').",
        "none": "Do not label units; keep as trailing token (e.g., '55 Halley St 4C').",
    }[unit_style]

    sys_prompt = (
        "You are a strict address-line normalizer for U.S. addresses. Output ONE line only (no city/state/ZIP).\n"
        "Follow these rules (Word doc standards + dataset conventions):\n"
        "• Strip any trailing city/state or ZIP if present in input.\n"
        "• Cardinal directions: abbreviate N, S, E, W, NE, NW, SE, SW when they are directionals; DO NOT change proper names like 'West St' or 'North Ave'.\n"
        "• Use USPS street-type abbreviations only on the terminal street-type word (St, Ave, Blvd, Rd, Dr, Ln, Ter, Pl, Ct, Cir, Pkwy, Hwy, etc.). Do not alter interior proper-name words like 'Centre' in 'Centre Ave'.\n"
        "• PO Boxes must be 'PO Box <value>'.\n"
        "• {unit_policy}\n"
        "• Keep 'Bsmt' as-is; 'Floor' → 'Fl'; 'Penthouse'/'PH' → 'Ph'.\n"
        "• Remove special characters except an accepted '# <unit>' pattern when used for unit style.\n"
        "• Deduplicate accidental repeats, title case, single spaces only. Do NOT hallucinate or add data.\n\n"
        "Examples:\n" + fewshot_block + "\n"
    )

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": pre},
            ],
            temperature=0.1,
        )
        out = resp.choices[0].message.content.strip()
    except RateLimitError:
        time.sleep(5)
        return llm_correct(address_line, compounds, model_name,
                           unit_style=unit_style,
                           infer_unlabeled_unit=infer_unlabeled_unit,
                           preserve_proper_directionals=preserve_proper_directionals)
    except Exception as e:
        return f"[LLM ERROR] {e}"

    out = restore_compounds(out, compounds)
    out = re.sub(r"\s+", " ", out).strip()
    return out

# =========================
# Canonicalization for comparison
# =========================

def canonicalize_for_compare(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[.,#]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

# =========================
# Mismatch analysis → suggestions (LLM-assisted, structured JSON)
# =========================

def explain_mismatch(raw: str, predicted: str, correct: str, model_name: str) -> Dict:
    """Return a dict with reason, and safe rule/few-shot suggestions."""
    prompt = (
        "You are auditing address line normalization mismatches. "
        "Given RAW, PREDICTED, and CORRECT (reference), explain why the mismatch occurred in one sentence. "
        "Then propose at most one safe regex rule if helpful (KEEP IT SIMPLE), indicating whether to apply PRE (before main rules) or POST (after), "
        "and optionally propose a (input→output) few-shot hint. Output STRICT JSON with keys: reason, rule_suggestion, few_shot_hint.\n\n"
        f"RAW: {raw}\nPREDICTED: {predicted}\nCORRECT: {correct}\n"
    )
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        txt = resp.choices[0].message.content.strip()
        data = json.loads(txt)
        if not isinstance(data, dict):
            raise ValueError("bad json")
        data.setdefault("reason", "")
        rs = data.get("rule_suggestion") or {}
        if not isinstance(rs, dict):
            rs = {}
        rule = {
            "pattern": str(rs.get("pattern", ""))[:300],
            "replacement": str(rs.get("replacement", ""))[:200],
            "where": str(rs.get("where", "post")).lower() if rs.get("where") in ("pre", "post", "both") else "post",
        }
        fh = data.get("few_shot_hint") or {}
        if not isinstance(fh, dict):
            fh = {}
        few = {
            "input": str(fh.get("input", ""))[:200],
            "output": str(fh.get("output", ""))[:200],
        }
        return {"reason": data.get("reason", ""), "rule_suggestion": rule, "few_shot_hint": few}
    except Exception:
        return {"reason": "Token/ordering difference or abbreviation variance.", "rule_suggestion": {"pattern": "", "replacement": "", "where": "post"}, "few_shot_hint": {"input": raw, "output": correct}}

# =========================
# UI
# =========================

st.title("📬 Address Cleaner & Formatter (CSV + Adaptive Learning)")
st.write(
    "Rules + LLM pipeline: merge lines → strip city/state/ZIP noise → apply deterministic rules → apply LLM with your standard → output ONE corrected address line. "
    "Use **Evaluation** to compare vs ground-truth, capture **non-exact matches**, and optionally learn from them. Use **Production** to generate a corrected CSV."
)

st.sidebar.header("Options")
mode = st.sidebar.radio("Mode", [
    "Evaluation (compare vs. correct CSV)",
    "Production (generate corrected CSV)",
])
max_rows = st.sidebar.number_input("Max rows to process", min_value=1, max_value=5000, value=DEFAULT_MAX_ROWS, step=1)
use_llm = st.sidebar.checkbox("Use LLM enhancement (after rules)", value=True)
compound_extra = st.sidebar.text_area("Extra compound street names to preserve (comma-separated)", value="")
sort_by_id = st.sidebar.checkbox("Sort results by ID (when ID is used)", value=True)
model_name = st.sidebar.selectbox("LLM model", ["gpt-4o-mini", "gpt-4", "gpt-4o"], index=0)

# Formatting policy toggles (tuned to your reference dataset by default)
st.sidebar.subheader("Formatting policy")
unit_style = st.sidebar.selectbox("Unit label style", ["preserve", "hash", "apt", "none"], index=0,
                                  help="How to render unit numbers by default")
infer_unlabeled_unit = st.sidebar.checkbox("Infer unlabeled trailing tokens as unit", value=False,
                                           help="If OFF, do not auto-insert 'Apt' for trailing tokens like '4C'.")
preserve_proper_directionals = st.sidebar.checkbox("Keep proper directional streets spelled out (West St, North Ave)", value=True)

st.sidebar.subheader("Adaptive Learning")
enable_learning = st.sidebar.checkbox("Enable adaptive learning", value=True)
learn_file = st.sidebar.file_uploader("Upload learning store (.jsonl or .json)", type=["jsonl", "json"], key="learn_upl")
if learn_file is not None:
    load_learning_store(learn_file)
if st.sidebar.button("Export learning store"):
    st.sidebar.download_button(
        "Download", data=export_learning_store(), file_name=LEARNING_STORE_FILENAME, mime="application/jsonl"
    )

compounds = DEFAULT_COMPOUNDS.copy()
if compound_extra.strip():
    compounds.extend([c.strip().lower() for c in compound_extra.split(",") if c.strip()])


def normalize_id_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r"\.0$", "", regex=True)
    )

# =========================
# EVALUATION MODE
# =========================
if mode.startswith("Evaluation"):
    st.header("🧪 Evaluation: RAW vs. CORRECT CSV")
    raw_file = st.file_uploader("Upload RAW addresses (CSV)", type=["csv"], key="raw_eval")
    correct_file = st.file_uploader("Upload CORRECT addresses (CSV)", type=["csv"], key="correct_eval")

    if raw_file and correct_file:
        raw_df = pd.read_csv(raw_file)
        cor_df = pd.read_csv(correct_file)

        st.subheader("Select Columns")
        raw_cols = list(raw_df.columns)
        cor_cols = list(cor_df.columns)

        raw_line1_col = st.selectbox("Raw: Address Line 1 column", options=raw_cols, index=min(0, len(raw_cols)-1), key="re_l1")
        raw_line2_col = st.selectbox("Raw: Address Line 2 column (optional)", options=["<none>"] + raw_cols, index=0, key="re_l2")
        cor_addr_col = st.selectbox("Correct: Address column", options=cor_cols, index=min(0, len(cor_cols)-1), key="re_coraddr")

        id_col_options = ["<none>"] + [c for c in raw_cols if c in cor_cols]
        id_col = st.selectbox("Optional: ID column present in BOTH files (for joining)", options=id_col_options, index=0, key="re_id")

        raw_df = raw_df.head(int(max_rows)).copy()
        cor_df = cor_df.head(int(max_rows)).copy()

        # Matching summary (normalized IDs to match join behavior)
        if id_col != "<none>":
            raw_df["_id_norm"] = normalize_id_series(raw_df[id_col])
            cor_df["_id_norm"] = normalize_id_series(cor_df[id_col])

            if sort_by_id:
                raw_df = raw_df.sort_values("_id_norm").reset_index(drop=True)
                cor_df = cor_df.sort_values("_id_norm").reset_index(drop=True)

            raw_ids = set(raw_df["_id_norm"].dropna().tolist())
            cor_ids = set(cor_df["_id_norm"].dropna().tolist())
            matched_ids = raw_ids & cor_ids
            only_raw = sorted(list(raw_ids - cor_ids))
            only_cor = sorted(list(cor_ids - raw_ids))

            c1, c2, c3 = st.columns(3)
            c1.metric("Matched IDs", len(matched_ids))
            c2.metric("Only in RAW", len(only_raw))
            c3.metric("Only in CORRECT", len(only_cor))

            if only_raw or only_cor:
                st.info(
                    "Unmatched examples — RAW only: "
                    + ", ".join(map(str, only_raw[:10]))
                    + (" …" if len(only_raw) > 10 else "")
                    + " | CORRECT only: "
                    + ", ".join(map(str, only_cor[:10]))
                    + (" …" if len(only_cor) > 10 else "")
                )
                or_buf = io.StringIO(); pd.DataFrame({id_col: only_raw}).to_csv(or_buf, index=False); or_buf.seek(0)
                oc_buf = io.StringIO(); pd.DataFrame({id_col: only_cor}).to_csv(oc_buf, index=False); oc_buf.seek(0)
                d1, d2 = st.columns(2)
                d1.download_button("⬇️ IDs only in RAW (CSV)", data=or_buf.getvalue(), file_name="ids_only_in_raw.csv", mime="text/csv")
                d2.download_button("⬇️ IDs only in CORRECT (CSV)", data=oc_buf.getvalue(), file_name="ids_only_in_correct.csv", mime="text/csv")
        else:
            matched_n = min(len(raw_df), len(cor_df))
            extra_raw = max(0, len(raw_df) - matched_n)
            extra_cor = max(0, len(cor_df) - matched_n)
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows aligned by order", matched_n)
            c2.metric("Extra in RAW", extra_raw)
            c3.metric("Extra in CORRECT", extra_cor)

        # Build RawAddress
        if raw_line2_col != "<none>" and raw_line2_col in raw_df.columns:
            raw_df["RawAddress"] = raw_df.apply(
                lambda r: f"{r[raw_line1_col]} {r[raw_line2_col]}".strip() if pd.notnull(r[raw_line2_col]) and str(r[raw_line2_col]).strip() != "" else str(r[raw_line1_col]),
                axis=1,
            )
        else:
            raw_df["RawAddress"] = raw_df[raw_line1_col].astype(str)

        # Ensure normalized IDs exist before processing for keyed predictions
        if id_col != "<none>":
            raw_df["_id_norm"] = normalize_id_series(raw_df[id_col])
            cor_df["_id_norm"] = normalize_id_series(cor_df[id_col])

        st.subheader("Processing")
        placeholder = st.empty()
        results = []
        total = len(raw_df)
        for i, raw in enumerate(raw_df["RawAddress"].astype(str).tolist(), start=1):
            interim = rule_clean(raw, compounds,
                                 unit_style=unit_style,
                                 infer_unlabeled_unit=infer_unlabeled_unit,
                                 preserve_proper_directionals=preserve_proper_directionals)
            pred = llm_correct(interim, compounds, model_name,
                               unit_style=unit_style,
                               infer_unlabeled_unit=infer_unlabeled_unit,
                               preserve_proper_directionals=preserve_proper_directionals) if use_llm else interim
            pred = re.sub(r"\s+", " ", pred).strip()
            results.append(pred)
            placeholder.progress(min(i / max(1, total), 1.0))

        # Predictions keyed by normalized ID when available
        if id_col != "<none>":
            pred_df = pd.DataFrame({"_id_norm": raw_df["_id_norm"].tolist(), "PredictedAddress": results})
        else:
            pred_df = pd.DataFrame({"PredictedAddress": results})

        # Align and compare
        if id_col != "<none>":
            base_cols = [id_col, "_id_norm", "RawAddress"]
            combo = (
                raw_df[base_cols]
                  .reset_index(drop=True)
                  .merge(
                      cor_df[[id_col, "_id_norm", cor_addr_col]].reset_index(drop=True),
                      on="_id_norm",
                      how="inner",
                      suffixes=("_RAW", "_COR")
                  )
            )
            combo = combo.merge(pred_df, on="_id_norm", how="left")
            combo = combo.rename(columns={cor_addr_col: "CorrectAddress"})
            if sort_by_id:
                combo = combo.sort_values("_id_norm").reset_index(drop=True)
            st.caption(f"Joined rows used for accuracy: {len(combo)}")
        else:
            matched_n = min(len(raw_df), len(cor_df))
            raw_sub = raw_df.head(matched_n)
            cor_sub = cor_df.head(matched_n)
            combo = pd.concat([
                raw_sub[["RawAddress"]].reset_index(drop=True),
                cor_sub[[cor_addr_col]].rename(columns={cor_addr_col: "CorrectAddress"}).reset_index(drop=True),
                pred_df.head(matched_n).reset_index(drop=True),
            ], axis=1)
            st.caption(f"Rows aligned by order and used for accuracy: {matched_n}")

        # Canonical comparison
        combo["_pred_norm"] = combo["PredictedAddress"].apply(canonicalize_for_compare)
        combo["_corr_norm"] = combo["CorrectAddress"].apply(canonicalize_for_compare)
        combo["ExactMatch"] = combo["_pred_norm"] == combo["_corr_norm"]

        accuracy = (combo["ExactMatch"].sum() / max(1, len(combo))) * 100.0
        st.success(f"✅ Accuracy: {accuracy:.2f}% (exact normalized match)")

        # Outputs (include clean ID when available)
        if id_col != "<none>":
            combo["ID"] = combo["_id_norm"]

        # Preview
        st.subheader("Results Preview")
        preview_cols = [c for c in ["ID" if "ID" in combo.columns else None, "RawAddress", "PredictedAddress", "CorrectAddress", "ExactMatch"] if c in combo.columns]
        st.dataframe(combo[preview_cols].head(50), use_container_width=True)

        # Non-exact matches + Learning widgets
        non_matches = combo.loc[~combo["ExactMatch"]].copy()
        st.subheader(f"❗ Non-exact matches: {len(non_matches)}")
        if len(non_matches) > 0:
            nm_cols = [c for c in ["ID" if "ID" in non_matches.columns else None, "RawAddress", "PredictedAddress", "CorrectAddress"] if c in non_matches.columns]
            st.dataframe(non_matches[nm_cols].head(25), use_container_width=True)

            # LLM-assisted mismatch explanations
            if enable_learning and use_llm:
                st.markdown("**Analyze mismatches and propose learning rules/examples**")
                sugg_rows = []
                max_analyze = st.slider("How many non-matches to analyze now?", 0, min(100, len(non_matches)), value=min(25, len(non_matches)))
                go = st.button("🔎 Analyze selected non-matches")
                if go and max_analyze > 0:
                    sub_nm = non_matches.head(max_analyze)
                    for _, r in sub_nm.iterrows():
                        expl = explain_mismatch(str(r.get("RawAddress", "")), str(r.get("PredictedAddress", "")), str(r.get("CorrectAddress", "")), model_name)
                        sugg_rows.append({
                            "RawAddress": r.get("RawAddress", ""),
                            "PredictedAddress": r.get("PredictedAddress", ""),
                            "CorrectAddress": r.get("CorrectAddress", ""),
                            "Reason": expl.get("reason", ""),
                            "RulePattern": expl.get("rule_suggestion", {}).get("pattern", ""),
                            "RuleReplacement": expl.get("rule_suggestion", {}).get("replacement", ""),
                            "RuleWhere": expl.get("rule_suggestion", {}).get("where", "post"),
                            "FewShotInput": expl.get("few_shot_hint", {}).get("input", ""),
                            "FewShotOutput": expl.get("few_shot_hint", {}).get("output", ""),
                            "ApproveRule": False,
                            "ApproveFewShot": True if expl.get("few_shot_hint", {}).get("output") else False,
                        })
                    sugdf = pd.DataFrame(sugg_rows)
                    st.dataframe(sugdf.head(100), use_container_width=True)

                    # Export suggestions CSV
                    buf = io.StringIO(); sugdf.to_csv(buf, index=False); buf.seek(0)
                    st.download_button("⬇️ Download suggestions (CSV)", data=buf.getvalue(), file_name=SUGGESTIONS_EXPORT, mime="text/csv")

                    # Apply approved suggestions immediately
                    if st.button("✅ Apply approved suggestions now"):
                        store = get_store()
                        applied_rules = 0
                        added_few = 0
                        for _, row in sugdf.iterrows():
                            if row.get("ApproveRule") and row.get("RulePattern"):
                                rule_obj = {
                                    "pattern": str(row.get("RulePattern")),
                                    "replacement": str(row.get("RuleReplacement", "")),
                                    "where": str(row.get("RuleWhere", "post")).lower() if str(row.get("RuleWhere", "post")).lower() in ("pre", "post", "both") else "post",
                                }
                                if rule_obj not in store["custom_rules"]:
                                    store["custom_rules"].append(rule_obj)
                                    applied_rules += 1
                            if row.get("ApproveFewShot") and row.get("FewShotInput") and row.get("FewShotOutput"):
                                pair = (str(row.get("FewShotInput")), str(row.get("FewShotOutput")))
                                if pair not in store["fewshot_examples"]:
                                    store["fewshot_examples"].append(pair)
                                    added_few += 1
                            lp = {"raw": row.get("RawAddress", ""), "predicted": row.get("PredictedAddress", ""), "correct": row.get("CorrectAddress", "")}
                            if lp not in store["learning_pairs"]:
                                store["learning_pairs"].append(lp)
                        st.success(f"Applied: {applied_rules} rules, added {added_few} few-shot examples. Export the learning store from the sidebar to persist.")

            # Download non-exact matches CSV
            nm_buf = io.StringIO(); non_matches[nm_cols].to_csv(nm_buf, index=False)
            st.download_button("⬇️ Download Non-Exact Matches (CSV)", data=nm_buf.getvalue(), file_name="address_non_exact_matches.csv", mime="text/csv")

        # Full evaluation CSV
        csv_buf = io.StringIO()
        out_cols = [c for c in ["ID" if "ID" in combo.columns else None, "RawAddress", "PredictedAddress", "CorrectAddress", "ExactMatch"] if c in combo.columns]
        combo[out_cols].to_csv(csv_buf, index=False)
        st.download_button("📥 Download Evaluation Results (CSV)", data=csv_buf.getvalue(), file_name="address_evaluation_results.csv", mime="text/csv")

# =========================
# PRODUCTION MODE
# =========================
else:
    st.header("🚀 Production: Generate Corrected Addresses CSV")
    raw_file_p = st.file_uploader("Upload RAW addresses (CSV)", type=["csv"], key="raw_prod")
    if raw_file_p:
        raw_df = pd.read_csv(raw_file_p)
        raw_cols = list(raw_df.columns)

        st.subheader("Select Columns")
        raw_line1_col = st.selectbox("Raw: Address Line 1 column", options=raw_cols, index=min(0, len(raw_cols)-1), key="rp_l1")
        raw_line2_col = st.selectbox("Raw: Address Line 2 column (optional)", options=["<none>"] + raw_cols, index=0, key="rp_l2")
        id_col = st.selectbox("Optional: ID column (e.g., CWID)", options=["<none>"] + raw_cols, index=0, key="rp_id")

        raw_df = raw_df.head(int(DEFAULT_MAX_ROWS if max_rows is None else max_rows)).copy()

        # Build RawAddress
        if raw_line2_col != "<none>" and raw_line2_col in raw_df.columns:
            raw_df["RawAddress"] = raw_df.apply(
                lambda r: f"{r[raw_line1_col]} {r[raw_line2_col]}".strip()
                if pd.notnull(r[raw_line2_col]) and str(r[raw_line2_col]).strip() != ""
                else str(r[raw_line1_col]),
                axis=1,
            )
        else:
            raw_df["RawAddress"] = raw_df[raw_line1_col].astype(str)

        # Normalize ID for clean output
        if id_col != "<none>":
            raw_df["ID"] = (
                raw_df[id_col]
                .astype(str)
                .str.strip()
                .str.replace(r"\.0$", "", regex=True)
            )

        st.subheader("Processing")
        placeholder = st.empty()
        results = []
        total = len(raw_df)
        for i, raw in enumerate(raw_df["RawAddress"].astype(str).tolist(), start=1):
            interim = rule_clean(
                raw,
                compounds,
                unit_style=unit_style,
                infer_unlabeled_unit=infer_unlabeled_unit,
                preserve_proper_directionals=preserve_proper_directionals,
            )
            pred = (
                llm_correct(
                    interim,
                    compounds,
                    model_name,
                    unit_style=unit_style,
                    infer_unlabeled_unit=infer_unlabeled_unit,
                    preserve_proper_directionals=preserve_proper_directionals,
                )
                if use_llm
                else interim
            )
            pred = re.sub(r"\s+", " ", pred).strip()
            results.append(pred)
            placeholder.progress(min(i / max(1, total), 1.0))

        # Build output CSV (ID + CorrectedAddress, plus Raw for audit)
        out_df = pd.DataFrame({"CorrectedAddress": results})
        if id_col != "<none>":
            out_df = pd.concat([raw_df[["ID"]].reset_index(drop=True), out_df], axis=1)
        out_df = pd.concat([out_df, raw_df[["RawAddress"]].reset_index(drop=True)], axis=1)

        st.subheader("Preview (first 50)")
        st.dataframe(out_df.head(50), use_container_width=True)

        csv_buf = io.StringIO()
        out_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download Corrected Addresses (CSV)",
            data=csv_buf.getvalue(),
            file_name="corrected_addresses.csv",
            mime="text/csv",
        )

    st.caption(
        "In Production mode, the app outputs the corrected address line per record using the Rules → LLM → Adaptive layer. "
        "Export and re-upload the learning store to persist behavior across sessions."
    )
