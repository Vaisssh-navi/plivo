import re
import json
from pathlib import Path
from typing import List
from rapidfuzz import process, fuzz

# ----------------------------------------------------------
# Load misspelling correction map
# ----------------------------------------------------------
MISSPELL_PATH = Path(__file__).parent.parent / "data" / "misspell_map.json"
MISSPELL_MAP = {}
if MISSPELL_PATH.exists():
    with open(MISSPELL_PATH, "r", encoding="utf-8") as f:
        MISSPELL_MAP = json.load(f)

def correct_common_misspellings(s: str) -> str:
    return " ".join(MISSPELL_MAP.get(t.lower(), t) for t in s.split())


# ----------------------------------------------------------
# Email normalization with lexicon-aware username splitting
# ----------------------------------------------------------
_EMAIL_USER = r"[a-z0-9][a-z0-9._-]{0,63}"
_EMAIL_DOM  = r"[a-z0-9][a-z0-9.-]{0,253}"
DOMAIN_BASES = ("gmail", "yahoo", "outlook", "hotmail")
DOMAIN_FIXES = {"g mail": "gmail", "hot mail": "hotmail"}

EMAIL_REGEX = re.compile(r"\b([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})\b", re.IGNORECASE)

def collapse_spelled_letters(s: str) -> str:
    """Collapse sequences like 'g m a i l' -> 'gmail'."""
    tokens, out, i = s.split(), [], 0
    while i < len(tokens):
        j = i
        while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalpha():
            j += 1
        if j - i >= 3:
            out.append("".join(tokens[i:j]))
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)

def _clean_domain_words(s: str) -> str:
    """Normalize domain words, collapse duplicates, and ensure '.com' endings."""
    s = re.sub(r"\s*([@.])\s*", r"\1", s)
    for bad, good in DOMAIN_FIXES.items():
        s = re.sub(bad, good, s, flags=re.IGNORECASE)
    s = re.sub(r"\bdot\b", ".", s, flags=re.IGNORECASE)
    s = re.sub(r"\.(?=\.)", "", s)
    s = re.sub(r"@(" + "|".join(DOMAIN_BASES) + r")com\b", r"@\1.com", s, flags=re.IGNORECASE)
    s = re.sub(r"@(" + "|".join(DOMAIN_BASES) + r")\b(?!\.)", r"@\1.com", s, flags=re.IGNORECASE)
    s = re.sub(r"\.com\.com\b", ".com", s, flags=re.IGNORECASE)
    return s

def _split_concatenated_name_in_email_with_lexicon(s: str, names_lex: List[str]) -> str:
    """
    Split concatenated usernames (varunsingh@gmail.com → varun.singh@gmail.com)
    using a first-name lexicon (case-insensitive). We only split once per email.
    """
    lex = {n.lower() for n in names_lex if n.strip()}

    def _add_dot(m):
        full = m.group(0)
        user, domain = m.group(1).lower(), m.group(2).lower()
        if any(sep in user for sep in (".", "_", "-")):
            return f"{user}@{domain}"
        for i in range(3, len(user) - 2):  # plausible split
            left, right = user[:i], user[i:]
            if left in lex and re.fullmatch(r"[a-z]+", right):
                return f"{left}.{right}@{domain}"
        return f"{user}@{domain}"

    return re.sub(r"\b([a-z]{6,})@([a-z0-9.-]+\.[a-z]{2,})\b", _add_dot, s, flags=re.IGNORECASE)

def _fix_spurious_at_before_email(s: str) -> str:
    verbs = r"(?:email|reply|contact|reach|mail|message)"
    return re.sub(
        rf"({verbs}\s+me?)\s*@\s*(?={_EMAIL_USER}@)",
        r"\1 ",
        s,
        flags=re.IGNORECASE,
    )

def _insert_at_for_readability(s: str) -> str:
    """Insert ' at ' after verbs when directly followed by an email (readability)."""
    full_email = rf"{_EMAIL_USER}@{_EMAIL_DOM}\.[a-z]{{2,}}"
    return re.sub(
        rf"\b(email|reach|reply|contact|mail|message)\b(\s+me)?\s+(?={full_email}\b)",
        lambda m: (m.group(0).rstrip() + " at "),
        s,
        flags=re.IGNORECASE,
    )

def _lowercase_emails_only(s: str) -> str:
    """Lower-case only the email substrings, keep rest of text as-is."""
    def _lc(m):
        return m.group(1).lower()
    return EMAIL_REGEX.sub(_lc, s)

def normalize_email_tokens(s: str, names_lex: List[str]) -> str:
    s2 = collapse_spelled_letters(correct_common_misspellings(s))
    s2 = _clean_domain_words(s2)
    s2 = _fix_spurious_at_before_email(s2)
    s2 = _insert_at_for_readability(s2)
    s2 = _split_concatenated_name_in_email_with_lexicon(s2, names_lex)
    s2 = _clean_domain_words(s2)
    s2 = _lowercase_emails_only(s2)
    return s2.strip()


# ----------------------------------------------------------
# Spoken-number → numeric (Indian scales + tolerant mixing)
# ----------------------------------------------------------
# Units / teens / tens
_UNITS = {
    "zero":0, "oh":0, "o":0, "one":1, "two":2, "three":3, "four":4, "five":5,
    "six":6, "seven":7, "eight":8, "nine":9, "ten":10, "eleven":11, "twelve":12,
    "thirteen":13, "fourteen":14, "fifteen":15, "sixteen":16, "seventeen":17,
    "eighteen":18, "nineteen":19
}
_TENS = {
    "twenty":20, "thirty":30, "forty":40, "fifty":50,
    "sixty":60, "seventy":70, "eighty":80, "ninety":90
}
_SCALES = {
    "hundred":100,
    "thousand":1000,
    "lakh":100000,
    "lac":100000,
    "crore":10000000,
    "cr":10000000,
}
_NUM_TOKEN = set(list(_UNITS.keys()) + list(_TENS.keys()) + list(_SCALES.keys()) + ["and"])

def _token_is_numberlike(tok: str) -> bool:
    t = re.sub(r"[^\w-]", "", tok.lower())
    return t.isdigit() or (t in _NUM_TOKEN)

def _word_value(tok: str) -> int:
    t = tok.lower()
    if t in _UNITS: return _UNITS[t]
    if t in _TENS: return _TENS[t]
    return None  # not a direct unit/tens

def _spoken_group_to_int(tokens: List[str]) -> int:
    """
    Convert a sequence of number-like tokens (including digits) into an int.
    Supports 'and', scales, and mixed small digits (e.g., 'sixty 3 lakh').
    """
    total = 0
    current = 0
    i = 0
    while i < len(tokens):
        tok = re.sub(r"[^\w-]", "", tokens[i].lower())
        if tok == "and":
            i += 1
            continue

        # direct digits
        if tok.isdigit():
            current += int(tok)
            i += 1
            continue

        # tens/units/teens
        val = _word_value(tok)
        if val is not None:
            current += val
            i += 1
            continue

        # hyphenated tens (e.g., 'twenty-three')
        if "-" in tok:
            parts = tok.split("-")
            val_sum = 0
            ok = True
            for p in parts:
                pv = _word_value(p)
                if pv is None:
                    ok = False
                    break
                val_sum += pv
            if ok:
                current += val_sum
                i += 1
                continue

        # scales
        if tok in _SCALES:
            scale = _SCALES[tok]
            if current == 0:
                current = 1
            current *= scale
            total += current
            current = 0
            i += 1
            continue

        # not a recognizable number token
        break

    return total + current

def _replace_spoken_numbers(s: str) -> str:
    """
    Scan text; whenever we see a run of number-like tokens (words/digits)
    of length >= 1 and the resolved value >= 10, replace that span by digits.
    """
    tokens = s.split()
    out = []
    i = 0
    while i < len(tokens):
        j = i
        span = []
        while j < len(tokens) and _token_is_numberlike(tokens[j]):
            span.append(tokens[j])
            j += 1
        if span:
            val = _spoken_group_to_int(span)
            # Only replace when the value is 10 or more (price-ish),
            # else keep tiny numbers to avoid accidental replacements.
            if val >= 10:
                out.append(str(val))
                i = j
                continue
        out.append(tokens[i])
        i += 1
    return " ".join(out)

def _indian_group(num: str) -> str:
    if len(num) <= 3: return num
    last3, rest = num[-3:], num[:-3]
    parts = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return ",".join(parts + [last3])

def add_rupee_before_numbers(s: str) -> str:
    """Add ₹ before plain numbers that look like prices (2–6 digits) and not part of emails."""
    def _repl(m):
        num = m.group(0)
        if re.match(r"^[0-9]{2,6}$", num):
            return "₹" + num
        return num
    return re.sub(r"(?<![@.])\b[0-9]{2,6}\b(?!@)", _repl, s)

def normalize_currency(s: str) -> str:
    s = re.sub(r"\brupees?\b", "₹", s, flags=re.IGNORECASE)
    s = _replace_spoken_numbers(s)           # convert spoken → digits first
    s = add_rupee_before_numbers(s)          # then add ₹ for bare prices
    def repl(m):
        raw = re.sub(r"[^0-9]", "", m.group(0))
        return "₹" + _indian_group(raw) if raw else m.group(0)
    return re.sub(r"₹\s*[0-9][0-9,\.]*", repl, s)


# ----------------------------------------------------------
# Name correction, capitalization, sentence case & greetings
# ----------------------------------------------------------
def correct_names_with_lexicon(s: str, names_lex: List[str], threshold: int = 90) -> str:
    """Correct and capitalize names based on the lexicon."""
    tokens, out = s.split(), []
    name_set = {n.lower() for n in names_lex}
    for t in tokens:
        best = process.extractOne(t, names_lex, scorer=fuzz.ratio)
        word = best[0] if best and best[1] >= threshold else t
        if word.lower() in name_set:
            word = word.capitalize()
        out.append(word)
    return " ".join(out)

def capitalize_sentences(s: str) -> str:
    s = s.strip()
    if not s: return s
    s = re.sub(r'([.!?])(\s*)([a-z])', lambda m: m.group(1) + m.group(2) + m.group(3).upper(), s)
    return s[0].upper() + s[1:] if s else s

# limit greeting comma to the first greeting occurrence only
_GREETING = re.compile(r'\b(hi|hello|hey)\s+([A-Za-z]+)\b', re.IGNORECASE)
def add_single_greeting_comma(s: str, names_lex: List[str]) -> str:
    used_once = False
    def _repl(m):
        nonlocal used_once
        greet = m.group(1).capitalize()
        name  = m.group(2).capitalize()
        if not used_once:
            used_once = True
            return f"{greet} {name},"
        else:
            return f"{greet} {name}"
    s = _GREETING.sub(_repl, s, count=0)
    # Also add comma after FIRST lexicon name if it's in position 1 and not already followed by comma
    if not used_once:
        for name in names_lex:
            cap = name.capitalize()
            s2 = re.sub(rf'^\s*({re.escape(cap)})\b(?!,)', r'\1,', s)
            if s2 != s:
                return s2
    return s

def _remove_trailing_punct_after_email(s: str) -> str:
    """Remove trailing '.' or '?' if the sentence ends with an email address."""
    return re.sub(r"([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})([.?\s]*)$", r"\1", s.strip(), flags=re.IGNORECASE)


# ----------------------------------------------------------
# Candidate generation
# ----------------------------------------------------------
EMAIL_VALID = re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")

def generate_candidates(text: str, names_lex: List[str]) -> List[tuple]:
    cands, seen = [], set()
    base = correct_common_misspellings(text.strip())

    # Full normalization
    t1 = normalize_email_tokens(base, names_lex)
    t1 = normalize_currency(t1)
    t1 = correct_names_with_lexicon(t1, names_lex)
    t1 = _remove_trailing_punct_after_email(t1)
    t1 = add_single_greeting_comma(capitalize_sentences(t1), names_lex)
    cands.append((t1, {"validated": bool(EMAIL_VALID.search(t1)) or ("₹" in t1), "why": "full"}))

    # Email-only
    t2 = normalize_email_tokens(base, names_lex)
    t2 = _remove_trailing_punct_after_email(t2)
    t2 = add_single_greeting_comma(capitalize_sentences(t2), names_lex)
    cands.append((t2, {"validated": bool(EMAIL_VALID.search(t2)), "why": "email"}))

    # Numbers + currency only
    t3 = normalize_currency(base)
    t3 = correct_names_with_lexicon(t3, names_lex)
    t3 = _remove_trailing_punct_after_email(t3)
    t3 = add_single_greeting_comma(capitalize_sentences(t3), names_lex)
    cands.append((t3, {"validated": ("₹" in t3), "why": "numbers"}))

    # Names only
    t4 = correct_names_with_lexicon(base, names_lex)
    t4 = _remove_trailing_punct_after_email(t4)
    t4 = add_single_greeting_comma(capitalize_sentences(t4), names_lex)
    cands.append((t4, {"validated": False, "why": "names"}))

    # Dedup & cap
    uniq = []
    for s2, m in cands:
        s2 = s2.strip()
        if s2 and s2 not in seen:
            uniq.append((s2, m))
            seen.add(s2)
        if len(uniq) >= 5:
            break
    return uniq
