"""Algerian advertiser entity matching on (noisy, multilingual) ASR transcripts.

Transcripts of Algerian ads mix Darija, Modern Standard Arabic and French, and Whisper often
renders brand names in either script or misspells them. Matching therefore:
  * normalises both scripts (lower-case, accents stripped; Arabic alef forms, ta marbuta, alef maqsura,
    diacritics and tatweel unified),
  * matches alias token sequences on word boundaries (no "mobil" inside "mobilis"),
  * accepts close misspellings of distinctive single-word names (difflib ratio >= 0.86, length >= 5),
  * weights ambiguous aliases low. Some brand names are also common words or names
    ("total", "géant", "stream", "iris", "ramy", سفينة = ship, يسير = easy, "chérie" vs Chery), so
    on their own they never produce a confident suggestion,
  * phonetic Darija matching (Phase 11.4): an Arabic-script word is reduced to a consonant skeleton and
    compared with the skeleton of each Latin brand name (موبيليس -> MBLS <- mobilis), so brands without a
    hand-written Arabic alias are still heard. Counted like a misspelling (one mention is not enough).

Phase 11.5 (zero manual input): Darija/French carrier phrases ("la promotion", "كاين جديد", "ما تفوتش") are removed
before matching; close misspellings are scored with RapidFuzz token_set_ratio (> FUZZY_MIN) when it is installed
(difflib otherwise); and resolve_brand() fuses the packshot-OCR and speech evidence of one spot into a confidence
(OCR + speech 0.99, OCR on a silent/music spot 0.95, OCR with unrelated speech 0.90, speech alone 0.80). Matches at or
above CONFIRM_MIN (0.85) are CONFIRMED without a click; the rest stay suggestions. Everything is deterministic lookup
tables: no model weights, no GPU. The gazetteer is a curated list of Algerian advertisers by sector, not an export of
a research corpus.
Directory entries marked "seen on El-Heddaf" were added from packshots captured by Spotly (Phase 11.3).
"""
import math
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

STRONG, WEAK = 1.0, 0.35
SUGGEST_MIN_CONFIDENCE = 0.55
FUZZY_MIN = 82                 # RapidFuzz token_set_ratio must EXCEED this for a noisy-string match
CONFIRM_MIN = 0.85             # fused confidence at/above which a match is CONFIRMED automatically


@dataclass(frozen=True)
class Brand:
    name: str
    category: str
    industry: str
    aliases: tuple = ()            # distinctive spellings, Latin or Arabic (weight STRONG)
    weak_aliases: tuple = ()       # ambiguous spellings: common words / names (weight WEAK)
    fuzzy: bool = True             # allow close misspellings / phonetic matches of strong aliases
    product: str = ""              # what is advertised, for display ("Mineral water")


BRANDS: tuple[Brand, ...] = (
    # ---------------- Telecom
    Brand("Mobilis", "Telecom", "Telecom", ("mobilis", "mobilisse", "موبيليس", "موبليس")),
    # Arabic spellings that are also ordinary words or first names are WEAK (2026-10-13: Whisper wrote
    # "جيلي جيدي" for "جيل جديد" = "new generation" in a TV ad, which a strong "جيلي" turned into Geely).
    Brand("Djezzy", "Telecom", "Telecom", ("djezzy", "djezy", "jezzy", "dzezzy", "جيزي"), ("جازي",)),
    Brand("Ooredoo", "Telecom", "Telecom", ("ooredoo", "oredoo", "ooredo", "أوريدو", "اوريدو", "أريدو")),
    Brand("Algerie Telecom", "Telecom", "Telecom",
          ("algerie telecom", "algeria telecom", "idoom", "اتصالات الجزائر", "ايدوم")),
    # ---------------- Electronics & Home
    Brand("Condor", "Electronics & Home", "Electronics", ("condor", "كوندور", "كندور")),
    Brand("Iris", "Electronics & Home", "Electronics", ("iris sat", "iris electronique"), ("iris", "ايريس", "إيريس")),
    Brand("Brandt", "Electronics & Home", "Electronics", ("brandt", "براندت", "برانت")),
    Brand("Geant", "Electronics & Home", "Electronics", ("geant electronics", "geant electronique"),
          ("geant", "جيان", "جيون")),
    Brand("Stream", "Electronics & Home", "Electronics", ("stream system", "stream systems"), ("stream", "ستريم")),
    Brand("Beko", "Electronics & Home", "Electronics", ("beko",), ("بيكو",)),
    # ---------------- Food & FMCG
    Brand("Cevital", "Food & FMCG", "Food & FMCG", ("cevital", "سيفيتال", "سفيتال")),
    Brand("Bimo", "Food & FMCG", "Food & FMCG", ("bimo", "بيمو")),
    Brand("Soummam", "Food & FMCG", "Food & FMCG", ("laiterie soummam", "yaourt soummam"),
          ("soummam", "صومام", "الصومام")),
    Brand("Candia", "Food & FMCG", "Food & FMCG", ("candia", "كانديا", "كنديا")),
    Brand("Ifri", "Food & FMCG", "Food & FMCG", ("ifri", "إفري", "افري")),
    Brand("Ramy", "Food & FMCG", "Food & FMCG", ("jus ramy", "ramy jus"), ("ramy", "رامي")),
    Brand("Amor Benamor", "Food & FMCG", "Food & FMCG",
          ("amor benamor", "amor ben amor", "عمر بن عمر"), ("benamor", "بن عمر")),
    Brand("Palmary", "Food & FMCG", "Food & FMCG", ("palmary", "بالماري")),
    Brand("Safina", "Food & FMCG", "Food & FMCG", ("safina huile",), ("safina", "سفينة")),
    # ---------------- Automotive & Energy
    Brand("TotalEnergies", "Automotive & Energy", "Automotive & Energy",
          ("totalenergies", "total energies", "توتال إنرجي", "توتال انرجي"), ("total", "توتال")),
    Brand("Naftal", "Automotive & Energy", "Automotive & Energy", ("naftal", "نفطال")),
    Brand("Mobil", "Automotive & Energy", "Automotive & Energy", ("mobil 1", "mobil super", "موبيل 1"),
          ("mobil", "موبيل"), fuzzy=False),
    Brand("Castrol", "Automotive & Energy", "Automotive & Energy", ("castrol", "كاسترول")),
    Brand("Chery", "Automotive & Energy", "Automotive & Energy", ("chery tiggo", "tiggo"), ("chery", "شيري")),
    Brand("Geely", "Automotive & Energy", "Automotive & Energy", ("geely",), ("جيلي",)),
    Brand("Fiat", "Automotive & Energy", "Automotive & Energy", ("fiat", "فيات")),
    # ---------------- Services & Banking
    Brand("Yassir", "Services & Banking", "Banking & Services", ("yassir", "yassir express"),
          ("ياسير", "يسير")),
    Brand("Heetch", "Services & Banking", "Banking & Services", ("heetch", "هيتش")),
    Brand("BNA", "Services & Banking", "Banking & Services", ("banque nationale d algerie", "البنك الوطني الجزائري"),
          ("bna",), fuzzy=False),
    Brand("CPA", "Services & Banking", "Banking & Services", ("credit populaire d algerie", "القرض الشعبي الجزائري"),
          ("cpa",), fuzzy=False),
    Brand("BEA", "Services & Banking", "Banking & Services", ("banque exterieure d algerie", "بنك الجزائر الخارجي"),
          ("bea",), fuzzy=False),
    Brand("CNEP", "Services & Banking", "Banking & Services", ("cnep", "cnep banque", "الصندوق الوطني للتوفير")),
    Brand("BaridiMob", "Services & Banking", "Banking & Services",
          ("baridimob", "baridi mob", "بريدي موب", "بريدي موبايل", "بريدي موبيل")),
    # ---------------- seen on El-Heddaf packshots (2026-10-13); spellings as printed + OCR variants
    Brand("Messad", "Food & FMCG", "Food & FMCG", ("messad", "messaid", "eau messad"), ("مسعد",),
          product="Mineral water"),
    Brand("Amiral", "Food & FMCG", "Food & FMCG", ("amiral", "optilla", "moon excellence", "اميرال", "اوبتيلا"),
          ("moon", "مون"), product="Biscuits"),
    Brand("Flux Care", "Food & FMCG", "Food & FMCG", ("flux care", "fluxcare", "fluxo care"), ("flux",),
          product="Men's toiletries"),
    Brand("Dayzy", "Food & FMCG", "Food & FMCG", ("dayzy", "دايزي"), product="Snacks"),
    Brand("Ricamar", "Food & FMCG", "Food & FMCG", ("ricamar", "ريكامار")),
    Brand("Bellat", "Food & FMCG", "Food & FMCG", ("bellat",), ("بلاط",), product="Cold cuts"),
    # ---------------- other major Algerian advertisers (common words / place names kept WEAK)
    Brand("Hamoud Boualem", "Food & FMCG", "Food & FMCG", ("hamoud boualem", "حمود بوعلام"), ("hamoud",),
          product="Soft drinks"),
    Brand("Rouiba", "Food & FMCG", "Food & FMCG", ("jus rouiba", "nca rouiba"), ("rouiba", "رويبة"), product="Juices"),
    Brand("Samsung", "Electronics & Home", "Electronics", ("samsung", "سامسونج", "سامسونغ")),
    Brand("Oppo", "Electronics & Home", "Electronics", ("oppo reno",), ("oppo", "اوبو"), fuzzy=False),
    Brand("Xiaomi", "Electronics & Home", "Electronics", ("xiaomi", "redmi", "شاومي")),
    Brand("Infinix", "Electronics & Home", "Electronics", ("infinix", "انفينيكس")),
    Brand("Realme", "Electronics & Home", "Electronics", ("realme", "ريلمي")),
    # ---------------- Phase 11.5 gazetteer: further Algerian advertisers by sector (curated, deterministic; spellings
    # are the printed names plus common ASR/OCR variants. Names that are also ordinary words are WEAK.)
    Brand("Danone Djurdjura", "Food & FMCG", "Food & FMCG", ("danone djurdjura", "djurdjura", "دانون", "جرجرة"),
          ("danone",), product="Dairy"),
    Brand("Saida", "Food & FMCG", "Food & FMCG", ("eau saida", "ماء سعيدة"), ("saida", "سعيدة"), product="Mineral water"),
    Brand("Lalla Khedidja", "Food & FMCG", "Food & FMCG", ("lalla khedidja", "lala khedidja", "لالة خديجة"),
          product="Mineral water"),
    Brand("Guedila", "Food & FMCG", "Food & FMCG", ("guedila", "قديلة"), product="Mineral water"),
    Brand("Fleurial", "Food & FMCG", "Food & FMCG", ("fleurial", "فلوريال"), product="Edible oil"),
    Brand("Elio", "Food & FMCG", "Food & FMCG", ("huile elio",), ("elio", "إيليو", "اليو"), product="Edible oil"),
    Brand("Lesieur Cristal", "Food & FMCG", "Food & FMCG", ("lesieur cristal", "lesieur"), ("cristal",),
          product="Edible oil"),
    Brand("Safia", "Food & FMCG", "Food & FMCG", ("eau safia",), ("safia", "صافية"), product="Mineral water"),
    Brand("Sim", "Food & FMCG", "Food & FMCG", ("semoule sim", "groupe sim"), ("sim",), fuzzy=False, product="Semolina & pasta"),
    Brand("Tchin-Lait", "Food & FMCG", "Food & FMCG", ("tchin lait", "tchinlait", "تشين ليت"), product="Dairy"),
    Brand("Youkous", "Food & FMCG", "Food & FMCG", ("youkous", "يوكوس"), product="Mineral water"),
    # ---- telecom / banking extras
    Brand("BADR", "Services & Banking", "Banking & Services", ("badr banque", "بنك الفلاحة والتنمية الريفية"),
          ("badr", "بدر"), fuzzy=False, product="Banking"),
    Brand("BDL", "Services & Banking", "Banking & Services", ("banque de developpement local", "بنك التنمية المحلية"),
          ("bdl",), fuzzy=False, product="Banking"),
    Brand("Societe Generale Algerie", "Services & Banking", "Banking & Services",
          ("societe generale algerie", "sg algerie", "سوسيتي جنرال")),
    Brand("BNP Paribas El Djazair", "Services & Banking", "Banking & Services",
          ("bnp paribas", "bnp paribas el djazair", "بي ان بي باريبا")),
    Brand("AGB", "Services & Banking", "Banking & Services", ("gulf bank algeria", "agb bank", "بنك الخليج الجزائر"),
          ("agb",), fuzzy=False, product="Banking"),
    # ---- home appliances
    Brand("Enie", "Electronics & Home", "Electronics", ("enie electronics", "enie"), fuzzy=False,
          product="Home appliances"),
    Brand("Starlight", "Electronics & Home", "Electronics", ("starlight",), ("ستارلايت",), product="Home appliances"),
    Brand("LG", "Electronics & Home", "Electronics", ("lg electronics", "ال جي"), ("lg",), fuzzy=False),
    Brand("Haier", "Electronics & Home", "Electronics", ("haier", "هاير")),
    Brand("Hisense", "Electronics & Home", "Electronics", ("hisense", "هايسنس")),
    Brand("TCL", "Electronics & Home", "Electronics", ("tcl",), fuzzy=False),
    # ---- automotive
    Brand("Renault", "Automotive & Energy", "Automotive & Energy", ("renault", "رونو", "رينو"), product="Cars"),
    Brand("Peugeot", "Automotive & Energy", "Automotive & Energy", ("peugeot", "بيجو"), product="Cars"),
    Brand("Hyundai", "Automotive & Energy", "Automotive & Energy", ("hyundai", "هيونداي", "هيوندا"), product="Cars"),
    Brand("Kia", "Automotive & Energy", "Automotive & Energy", ("kia motors", "كيا موتورز"), ("kia", "كيا"),
          fuzzy=False, product="Cars"),
    Brand("Toyota", "Automotive & Energy", "Automotive & Energy", ("toyota", "تويوتا"), product="Cars"),
    Brand("Suzuki", "Automotive & Energy", "Automotive & Energy", ("suzuki", "سوزوكي"), product="Cars"),
    Brand("Dacia", "Automotive & Energy", "Automotive & Energy", ("dacia", "داسيا"), product="Cars"),
    Brand("Volkswagen", "Automotive & Energy", "Automotive & Energy", ("volkswagen", "فولكسفاغن", "فولكسفاجن"),
          product="Cars"),
    Brand("Citroen", "Automotive & Energy", "Automotive & Energy", ("citroen", "سيتروين"), product="Cars"),
)

_AR_DIACRITICS = re.compile("[ً-ٰٕـ]")          # harakat, combining madda/hamza (NFKD), superscript alef, tatweel
_NON_WORD = re.compile(r"[^\w؀-ۿ]+")


def normalize(text: str) -> str:
    """Lower-case, strip Latin accents, unify Arabic letter variants, collapse to space-separated words."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not (unicodedata.category(ch) == "Mn" and ord(ch) < 0x0600))
    t = _AR_DIACRITICS.sub("", t.lower())
    t = (t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
          .replace("ة", "ه").replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي"))
    t = t.replace("_", " ")
    return " ".join(_NON_WORD.sub(" ", t).split())


def _stem(tok: str) -> str:
    """Light Arabic stemming for matching: drop a leading 'و' (and) and/or the article 'ال'."""
    if tok[:1] == "و" and len(tok) > 3 and "؀" <= tok[1:2] <= "ۿ":
        tok = tok[1:]
    if tok.startswith("ال") and len(tok) > 4:
        tok = tok[2:]
    return tok


def _tokens(text: str) -> list[str]:
    return [_stem(t) for t in normalize(text).split()]


def _norm_alias(a: str) -> tuple[str, ...]:
    return tuple(_tokens(a))


# Darija/French colloquial carriers of promotional speech: they hold no brand information, only noise that the
# fuzzy and phonetic matchers could mistake for one.
CARRIER_PHRASES = (
    "لا بروموسيون", "بروموسيون", "la promotion", "promotion", "promo", "كاين جديد", "ما تفوتش", "ماتفوتش",
    "تخفيضات", "تخفيض", "على خاطر", "علاخاطر", "nouveau", "nouvelle offre", "offre speciale",
)
_CARRIER_SEQS = tuple(sorted({_norm_alias(p) for p in CARRIER_PHRASES}, key=len, reverse=True))


def strip_carriers(tokens: list[str]) -> list[str]:
    """Remove carrier-phrase token sequences (longest first)."""
    out, i = [], 0
    while i < len(tokens):
        for seq in _CARRIER_SEQS:
            n = len(seq)
            if tuple(tokens[i:i + n]) == seq:
                i += n
                break
        else:
            out.append(tokens[i])
            i += 1
    return out


def clean_carriers(text: str) -> str:
    """Normalised text without carrier phrases."""
    return " ".join(strip_carriers(_tokens(text)))


try:
    from rapidfuzz import fuzz as _rf
except ImportError:                                    # optional: difflib gives the same answer for single words
    _rf = None


def fuzzy_ratio(a: str, b: str) -> float:
    """0-100 similarity of two strings: RapidFuzz token_set_ratio, else difflib.

    token_set_ratio alone scores 100 when one string's words are a subset of the other's ("total total" vs
    "total energies"), so the plain ratio must clear the bar too."""
    if _rf is not None:
        return float(min(_rf.token_set_ratio(a, b), _rf.ratio(a, b)))
    return 100.0 * SequenceMatcher(None, a, b).ratio()


_ALL_ALIASES: dict[str, set[str]] = {}                 # normalised single-token alias -> brands that own it


def _alias_owners() -> dict[str, set[str]]:
    if not _ALL_ALIASES:
        for b in BRANDS:
            for alias in b.aliases + b.weak_aliases:
                seq = _norm_alias(alias)
                if len(seq) == 1:
                    _ALL_ALIASES.setdefault(seq[0], set()).add(b.name)
    return _ALL_ALIASES


@dataclass
class BrandMatch:
    brand: str
    category: str
    industry: str
    score: float
    confidence: float
    evidence: list = field(default_factory=list)       # (alias, method, count)
    strong: bool = False                               # at least one distinctive (non-ambiguous) hit
    product: str = ""


# ---- phonetic skeletons: both scripts reduced to the same consonant classes (vowels/long vowels dropped)
_AR_CLASS = {"ب": "B", "ت": "T", "ط": "T", "ث": "T", "ج": "J", "خ": "K", "د": "D", "ض": "D", "ذ": "D", "ر": "R",
             "ز": "Z", "س": "S", "ص": "S", "ش": "X", "غ": "G", "ف": "F", "ڤ": "F", "ق": "K", "ك": "K", "ل": "L",
             "م": "M", "ن": "N", "پ": "B"}
_LAT_RULES = [("tion", "SN"), ("dj", "J"), ("sh", "X"), ("ch", "X"), ("ph", "F"), ("ck", "K"), ("qu", "K"),
              ("th", "T"), ("x", "KS")]
_LAT_CLASS = {"b": "B", "p": "B", "t": "T", "d": "D", "j": "J", "k": "K", "q": "K", "f": "F", "v": "F", "r": "R",
              "z": "Z", "s": "S", "l": "L", "m": "M", "n": "N", "g": "G"}
PHONETIC_MIN = 4                                       # shorter skeletons collide with ordinary words


def _collapse(seq: str) -> str:
    return "".join(c for i, c in enumerate(seq) if i == 0 or c != seq[i - 1])


def skeleton_ar(tok: str) -> str:
    return _collapse("".join(_AR_CLASS.get(ch, "") for ch in tok))


def skeleton_lat(word: str) -> str:
    w = word.lower()
    for a, b in _LAT_RULES:
        w = w.replace(a, b)
    out = []
    for i, ch in enumerate(w):
        if ch.isupper():
            out.append(ch)
        elif ch == "c":                                # soft c before e/i/y (cevital), else K
            out.append("S" if w[i + 1:i + 2] in ("e", "i", "y") else "K")
        elif ch == "g" and w[i + 1:i + 2] in ("e", "i", "y"):
            out.append("J")                            # French soft g (Algerian pronunciation)
        else:
            out.append(_LAT_CLASS.get(ch, ""))
    return _collapse("".join(out))


def _is_arabic(tok: str) -> bool:
    return any("؀" <= ch <= "ۿ" for ch in tok)


def _find_sequence(tokens, seq) -> int:
    n = len(seq)
    return sum(1 for i in range(len(tokens) - n + 1) if tuple(tokens[i:i + n]) == seq) if n else 0


def extract_brands(text: str) -> list[BrandMatch]:
    """All brands mentioned in `text`, best first (empty if none)."""
    tokens = strip_carriers(_tokens(text))
    if not tokens:
        return []
    owners = _alias_owners()
    ar_skel = [skeleton_ar(t) if _is_arabic(t) else "" for t in tokens]
    out = []
    for b in BRANDS:
        best_at: dict[int, tuple[float, str, str]] = {}      # token position -> (weight, alias, method)

        def mark(pos, weight, alias, method):
            if weight > best_at.get(pos, (0.0,))[0]:
                best_at[pos] = (weight, alias, method)
        for weight, aliases in ((STRONG, b.aliases), (WEAK, b.weak_aliases)):
            for alias in aliases:
                seq = _norm_alias(alias)
                n = len(seq)
                for i in range(len(tokens) - n + 1):
                    if n and tuple(tokens[i:i + n]) == seq:
                        mark(i, weight, alias, "exact" if weight == STRONG else "exact (ambiguous word)")
                if weight == STRONG and b.fuzzy and n >= 2 and len(" ".join(seq)) >= 8:
                    joined = " ".join(seq)                   # noisy multi-word names: "fluxo kare", "algerie telekom"
                    for i in range(len(tokens) - n + 1):
                        win = " ".join(tokens[i:i + n])
                        if win != joined and fuzzy_ratio(win, joined) > FUZZY_MIN:
                            mark(i, 0.6 * STRONG, alias, f"fuzzy ({win})")
                if weight == STRONG and b.fuzzy and n == 1 and len(seq[0]) >= 5:
                    for i, t in enumerate(tokens):
                        # a token that is another brand's exact name (mobil vs mobilis) is never a misspelling
                        if t != seq[0] and abs(len(t) - len(seq[0])) <= 2 and t[:1] == seq[0][:1]                                 and not (owners.get(t, set()) - {b.name}) and fuzzy_ratio(t, seq[0]) > FUZZY_MIN:
                            mark(i, 0.6 * STRONG, alias, f"fuzzy ({t})")
                    if not _is_arabic(seq[0]):                 # phonetic: Arabic-script word ~ Latin name
                        sk = skeleton_lat(seq[0])
                        if len(sk) >= PHONETIC_MIN:
                            for i, (t, tsk) in enumerate(zip(tokens, ar_skel)):
                                if tsk == sk and len(t) >= 4:
                                    mark(i, 0.6 * STRONG, alias, f"phonetic ({t})")
        if not best_at:
            continue
        # each word position counts once (its best match); at most 3 mentions
        top = sorted(best_at.values(), key=lambda x: -x[0])[:3]
        score = sum(w for w, _, _ in top)
        evidence = [(alias, method, sum(1 for v in best_at.values() if v[1] == alias and v[2] == method))
                    for alias, method in dict.fromkeys((a, m) for _, a, m in top)]
        out.append(BrandMatch(b.name, b.category, b.industry, round(score, 3), round(1 - math.exp(-1.2 * score), 3),
                              evidence, strong=any(w > WEAK for w, _, _ in best_at.values()), product=b.product))
    out.sort(key=lambda m: (-m.score, m.brand))
    return out


# A channel's own promos recur like ads but are not paid airtime. Token sequences after normalisation and
# light stemming ('الهداف' -> 'هداف', 'التطبيق' -> 'تطبيق'). Whisper often hears الهداف as الهدف ("the goal",
# common in sports talk), so that spelling only counts next to 'app' / 'channel'.
STATION_SELF_REFERENCES = {
    "El-Heddaf TV": (("هداف",), ("تطبيق", "هدف"), ("قناه", "هدف"), ("el", "heddaf"), ("elheddaf",), ("heddaf",),
                     ("el", "heddaf", "tv")),
}


def detect_station_promo(text: str, channel: str) -> str | None:
    """'station self-promo (<channel>)' if the transcript names the channel itself, else None."""
    tokens = _tokens(text)
    for seq in STATION_SELF_REFERENCES.get(channel, ()):
        if _find_sequence(tokens, seq):
            return f"station self-promo ({channel})"
    return None


def suggest_brand(text: str) -> BrandMatch | None:
    """The single brand worth suggesting, or None (no match, too weak, or two brands too close to call)."""
    matches = extract_brands(text)
    if not matches or matches[0].confidence < SUGGEST_MIN_CONFIDENCE or not matches[0].strong:
        return None                                    # ambiguous words alone never make a suggestion
    if len(matches) > 1 and matches[1].score >= 0.8 * matches[0].score:
        return None                                    # ambiguous between two brands: let a human decide
    return matches[0]


# ---------------------------------------------------------------- Phase 11.5: image + audio fusion
_CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])")   # OOREDOO4G
MIN_SPEECH_WORDS = 3            # a checked transcript shorter than this is a silent / music-only spot
CONF_OCR_AND_SPEECH, CONF_OCR_SILENT, CONF_OCR_OTHER_SPEECH, CONF_SPEECH_ONLY = 0.99, 0.95, 0.90, 0.80
_BRAND_BY_NAME = {b.name: b for b in BRANDS}


def sector_of(brand: str) -> str | None:
    """Sector used for competitor separation: the advertised product ("Mineral water") else the industry.
    Messad and Amiral are both FMCG but not competitors, so the industry alone would be too coarse."""
    b = _BRAND_BY_NAME.get(brand)
    return (b.product or b.industry) if b else None


@dataclass
class Resolution:
    brand: str | None = None
    sector: str | None = None
    industry: str | None = None
    confidence: float = 0.0
    status: str = "NONE"                 # CONFIRMED | SUGGESTED | NONE
    basis: str = ""
    evidence: list = field(default_factory=list)


def _clear_match(text: str) -> BrandMatch | None:
    return suggest_brand(" ".join([text, _CAMEL_SPLIT.sub(" ", text)])) if text and text.strip() else None


def resolve_brand(ocr_text: str | None, transcript: str | None, *, speech_checked: bool,
                  programme: bool = False, station_promo: bool = False) -> Resolution:
    """Cross-check what the packshot shows with what the soundtrack says (pure; no I/O).

    OCR + matching speech 0.99 | OCR on a silent/music spot 0.95 | OCR, speech names no brand 0.90 |
    speech only 0.80. Programme frames (channel ticker on screen) and station promos never resolve: their
    on-screen sponsors are not the advertised brand. `speech_checked` False = no transcript yet, so silence and
    absence of a spoken brand are unknown: OCR alone stays a suggestion until the transcript exists.
    """
    if station_promo:
        return Resolution(basis="station self-promo")
    if programme:
        return Resolution(basis="programme frame (channel ticker): on-screen text is not the advertised brand")
    ocr = _clear_match(ocr_text or "")
    speech = suggest_brand(transcript or "") if speech_checked else None
    words = len(strip_carriers(_tokens(transcript or ""))) if speech_checked else None
    silent = speech_checked and words < MIN_SPEECH_WORDS
    if ocr and speech and ocr.brand != speech.brand:
        return Resolution(basis=f"conflict: packshot shows {ocr.brand}, speech names {speech.brand}",
                          evidence=[dict(source="packshot", brand=ocr.brand), dict(source="speech", brand=speech.brand)])
    if ocr and speech:
        conf, basis = CONF_OCR_AND_SPEECH, "packshot text and speech agree"
    elif ocr and silent:
        conf, basis = CONF_OCR_SILENT, "packshot text on a silent / music-only spot"
    elif ocr and speech_checked:
        conf, basis = CONF_OCR_OTHER_SPEECH, "packshot text; speech names no other brand"
    elif ocr:
        conf, basis = CONF_SPEECH_ONLY, "packshot text; transcript not available yet"
    elif speech:
        conf, basis = CONF_SPEECH_ONLY, "speech only, no visual confirmation"
    else:
        return Resolution(basis="no brand in packshot text or speech")
    hit = ocr or speech
    evidence = [dict(source=src, brand=m.brand, evidence=[list(e) for e in m.evidence])
                for src, m in (("packshot", ocr), ("speech", speech)) if m]
    return Resolution(hit.brand, sector_of(hit.brand), hit.industry, conf,
                      "CONFIRMED" if conf >= CONFIRM_MIN else "SUGGESTED", basis, evidence)


# ------------------------------------------------------------------------------------- Phase 22: voice / music ratio
VOICE_SR = 11025
MUSIC_DOMINANT_BELOW = 0.20            # voice ratio under this: music-dominant creative -> packshot confirmation mandatory
_VF_N, _VF_HOP = 512, 256              # 46 ms frames, 23 ms hop
_VF_WIN = 43                           # ~1 s decision windows (50 % overlap)
_VF_FLAT_MAX = 0.03                    # window median spectral flatness (90 Hz..4.3 kHz) below this reads as vocal-peaky
_VF_ZCR_MAX = 0.25                     # hiss / cymbal-wash windows are not speech however flat-peaky they look
_VF_FLOOR, _VF_SPAN = 0.25, 0.50       # share of "vocal-peaky" windows a pure music bed shows anyway, and the range above it


def voice_frame_features(pcm):
    """Per 46 ms frame: (energy, zero-crossing rate, spectral flatness 90 Hz..4.3 kHz)."""
    import numpy as np
    x = np.asarray(pcm, dtype=np.float32)
    k = 1 + (x.size - _VF_N) // _VF_HOP
    if k < 2:
        return None
    idx = np.arange(_VF_N)[None, :] + _VF_HOP * np.arange(k)[:, None]
    fr = x[idx]
    zcr = np.abs(np.diff(np.signbit(fr).astype(np.int8), axis=1)).sum(1) / _VF_N
    fr = fr - fr.mean(1, keepdims=True)
    energy = (fr ** 2).mean(1) + 1e-12
    band = np.abs(np.fft.rfft(fr * np.hanning(_VF_N), axis=1))[:, 4:200] ** 2 + 1e-12
    return energy, zcr, np.exp(np.log(band).mean(1)) / band.mean(1)


def compute_voice_ratio(pcm_data) -> float:
    """Estimated share (0..1) of the creative that carries a spoken voice-over, from mono 11.025 kHz audio.

    Short-time spectral flatness and zero-crossing rate over 1 s windows: speech concentrates energy in harmonic / formant
    peaks (low flatness) at a speech-range crossing rate. A music bed shows such windows too (about a quarter of them on
    the 15 stored El-Heddaf creatives), so that baseline is subtracted: ratio = (share - 0.25) / 0.5, clipped to 0..1.
    Returns 0.0 for silence / too-short input.

    Heuristic, calibrated on those 15 creatives (transcript present = voice-over): all 6 music-only spots score < 0.20, 7 of 9
    spoken spots score above it, 2 (a noisy and a sparse mix) fall under it. The error direction is the safe one - an
    uncertain spot gets the extra packshot check - and register-ad has an explicit override for a known creative."""
    import numpy as np
    f = voice_frame_features(pcm_data)
    if f is None:
        return 0.0
    energy, zcr, flat = f
    active = energy > energy.max() * 1e-4
    hits = total = 0
    for i in range(0, energy.size - _VF_WIN + 1, _VF_WIN // 2):
        sl = slice(i, i + _VF_WIN)
        if active[sl].mean() < 0.5:
            continue
        total += 1
        hits += float(np.median(flat[sl])) < _VF_FLAT_MAX and float(np.median(zcr[sl])) < _VF_ZCR_MAX
    if not total:
        return 0.0
    return round(float(np.clip((hits / total - _VF_FLOOR) / _VF_SPAN, 0.0, 1.0)), 4)


def is_music_dominant(voice_ratio: float | None) -> bool:
    return voice_ratio is not None and voice_ratio < MUSIC_DOMINANT_BELOW


# ------------------------------------------------------------------------------- naming hygiene (hotfix 2026-10-21)
# Internal bookkeeping strings that must never reach a user-facing table: "Auto-discovered ELHEDDAF_AD_117",
# "Auto-resolved ELHEDDAF_AD_013", a bare asset id, or MASTER_0007_slug.
_INTERNAL_ID = re.compile(r"\b(?:[A-Z][A-Z0-9]*_AD_\d+|MASTER_\d+(?:_[A-Za-z0-9]+)?)\b")
_AUTO_PREFIX = re.compile(r"\bAuto[-\s]?(?:discovered|resolved|detected|classified)\b[:\s-]*", re.IGNORECASE)
_DASHES = re.compile(r"\s*[–—·|�]\s*")


def strip_internal(text: str | None) -> str:
    """Remove 'Auto-discovered' / 'Auto-resolved' and internal asset ids from a display string."""
    t = _AUTO_PREFIX.sub(" ", _INTERNAL_ID.sub(" ", text or ""))
    return " ".join(t.split()).strip(" -–—·|:")


_KNOWN_ALIASES: dict[str, str] | None = None


def _alias_table() -> dict[str, str]:
    """normalized single-word STRONG alias / brand name -> canonical brand name."""
    global _KNOWN_ALIASES
    if _KNOWN_ALIASES is None:
        t = {}
        for b in BRANDS:
            for a in (b.name, *b.aliases):
                n = normalize(a)
                if n and " " not in n:
                    t[n] = b.name
        _KNOWN_ALIASES = t
    return _KNOWN_ALIASES


def canonical_brand(name: str | None) -> str:
    """The registry spelling of a brand ('amiral', 'AMIRAL ' -> 'Amiral'); unknown names are only tidied."""
    tidy = " ".join(strip_internal(name).split())
    if not tidy:
        return ""
    n = normalize(tidy)
    for b in BRANDS:
        if n == normalize(b.name) or n in {normalize(a) for a in b.aliases}:
            return b.name
    return tidy


def repair_stitched_brand(brand: str | None, campaign: str | None) -> tuple[str, str]:
    """Undo name stitching: an OCR / speech fragment became the brand ('MON') while the real brand lost its first letters and
    ended up in the campaign ('miral NOW EXCELLENCE MON' is 'Amiral NOW EXCELLENCE' + 'MON'). When a campaign word is a
    leading-truncated spelling of a registry brand (1-2 letters missing) and the stored brand is not itself a registry brand,
    the registry brand wins and the fragment words are dropped from the campaign. -> (brand, campaign)."""
    brand, campaign = canonical_brand(brand), strip_internal(campaign)
    known = {b.name for b in BRANDS}
    if brand in known:
        return brand, campaign
    table = _alias_table()
    for word in campaign.split():
        w = normalize(word)
        if len(w) < 4:
            continue
        for alias, canon in table.items():
            if len(alias) >= 5 and alias != w and alias.endswith(w) and len(alias) - len(w) <= 2:
                rest = [x for x in campaign.split() if x != word]
                return canon, " ".join(rest)
    return brand, campaign


def clean_campaign(brand: str | None, campaign: str | None, duration: float | None = None) -> str:
    """User-facing campaign name: internal ids removed, never the brand repeated; falls back to '<n> s spot'."""
    c = strip_internal(campaign)
    if _OCR_SEGMENT.search(c) or _OCR_ASPECT.search(c):      # raw OCR scrape ("new X/ X MEN EDITION care"): purify first
        c = sanitize_ocr(c)
    b = " ".join((brand or "").split())
    if b and c.casefold().startswith(b.casefold()):
        c = c[len(b):].strip(" -–—·|:")
    if not c and duration:
        c = f"{round(float(duration))} s spot"
    return c or "Spot"


def display_label(brand: str | None, campaign: str | None, duration: float | None = None) -> str:
    """'Amiral · 45 s spot' - the one string every table shows for an ad."""
    b = canonical_brand(brand) or "Unknown advertiser"
    return f"{b} · {clean_campaign(b, campaign, duration)}"


# ------------------------------------------------------------------------------- deterministic OCR sanitizer (Phase 24)
# Raw packshot OCR is a stream of fragments: "new SILICONE FLUX/ FLUX MEN EDITION care 16:9". The sanitizer works in
# stages - noise strip, slash segmentation, taxonomy match (RapidFuzz token_set_ratio > TAXONOMY_MIN, every brand word
# present), variant extraction - and falls back to a tidied string when no canonical advertiser is recognised.
TAXONOMY_MIN = 85
_OCR_ASPECT = re.compile(r"(?<!\d)\d{1,2}\s*[:x×]\s*\d{1,2}(?!\d)|\b(?:hd|4k|uhd|1080p?|720p?)\b", re.IGNORECASE)
_OCR_URL = re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+\.(?:dz|com|net|org)\b", re.IGNORECASE)
_OCR_SEGMENT = re.compile(r"[/\|]+")
_OCR_JUNK = re.compile(r"[^\w\s؀-ۿ&'.-]|_", re.UNICODE)
_OCR_FRAGMENT = re.compile(r"^[^\W\d_]$")             # a lone letter is an OCR fragment (Arabic letters count too)
_VARIANT_NOISE = {"new", "nouveau", "nouvelle", "neu", "جديد", "جديدة", "the", "de", "du", "by", "et"}


def _title(tok: str) -> str:
    if _is_arabic(tok):
        return tok
    return tok.upper() if any(c.isdigit() for c in tok) else tok[:1].upper() + tok[1:].lower()


def _word_hit(word: str, token: str) -> bool:
    return token == word or (min(len(word), len(token)) >= 4 and fuzzy_ratio(word, token) > TAXONOMY_MIN)


def _brand_words(brand: Brand) -> tuple[str, ...]:
    return tuple(_tokens(brand.name))


def match_taxonomy(text: str) -> Brand | None:
    """The one canonical advertiser whose every name word appears in `text` (exactly or RapidFuzz > 85) and whose full name
    clears token_set_ratio > 85. Ties between brands are ambiguous -> None (never guess between advertisers)."""
    toks = _tokens(text)
    if not toks:
        return None
    flat = " ".join(toks)
    scored = []
    for b in BRANDS:
        words = _brand_words(b)
        if not words or not all(any(_word_hit(w, t) for t in toks) for w in words):
            continue
        if fuzzy_ratio(" ".join(words), flat) <= TAXONOMY_MIN and _rf is not None and \
                _rf.token_set_ratio(" ".join(words), flat) <= TAXONOMY_MIN:
            continue
        scored.append((sum(len(w) for w in words), b))
    if not scored:                                        # multi-word misspelling of an alias ("fluxo kare")
        strong = [m for m in extract_brands(flat) if m.strong]
        return next(b for b in BRANDS if b.name == strong[0].brand) if strong else None
    scored.sort(key=lambda x: -x[0])
    return scored[0][1] if len(scored) == 1 or scored[0][0] > scored[1][0] else None


def sanitize_ocr(raw: str | None) -> str:
    """Canonical display name from a raw OCR string: 'Flux Care - Men Edition'. Arabic and Latin names are preserved; a string
    that matches no advertiser is returned tidied (noise stripped, repeated words collapsed), never invented."""
    text = _OCR_URL.sub(" ", _OCR_ASPECT.sub(" ", raw or ""))
    segments = [" ".join(_OCR_JUNK.sub(" ", seg).split()) for seg in _OCR_SEGMENT.split(text)]
    segments = [" ".join(w for w in seg.split() if not _OCR_FRAGMENT.match(w)) for seg in segments if seg.strip()]
    flat = " ".join(segments)
    brand = match_taxonomy(flat)
    if brand is None:
        seen, out = set(), []
        for w in flat.split():
            if normalize(w) not in seen:
                seen.add(normalize(w))
                out.append(w)
        return " ".join(out)
    words = _brand_words(brand)
    # the variant lives in the last segment that carries the brand, after the brand's first word ("new SILICONE" before it is
    # a lead-in descriptor, "care" after it is the rest of the brand name)
    for seg in reversed(segments):
        toks = seg.split()
        norm = [_stem(normalize(t)) for t in toks]
        first = next((i for i, n in enumerate(norm) if any(_word_hit(w, n) for w in words)), None)
        if first is None:
            continue
        variant, seen = [], set(norm[first:first + 1])
        for t, n in list(zip(toks, norm))[first + 1:]:
            if not n or n in seen or n in _VARIANT_NOISE or any(_word_hit(w, n) or fuzzy_ratio(w, n) >= 70 for w in words):
                continue
            seen.add(n)
            variant.append(_title(t))
        return brand.name + (" - " + " ".join(variant) if variant else "")
    return brand.name
