from django.shortcuts import render, redirect
from django.http import JsonResponse
from openai import OpenAI
import os
import re
import difflib
import unicodedata

# --- Paste normalization (Google Docs, Word, etc.) ---
ZERO_WIDTH_RE = re.compile(r"[\u200B\u200C\u200D\u2060\uFEFF]")  # ZWSP/ZWNJ/ZWJ/WJ/BOM

def normalize_pasted_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", s)

    # Normalize all common "line break" variants to \n
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u2028", "\n").replace("\u2029", "\n")  # Unicode LS/PS
    s = s.replace("\u000b", "\n").replace("\u000c", "\n")  # VT/FF

    # Normalize non-breaking spaces to normal spaces
    s = s.replace("\u00a0", " ").replace("\u202f", " ").replace("\u2007", " ")

    # Remove invisible zero-width characters (they break matching/diffs)
    s = ZERO_WIDTH_RE.sub("", s)

    return s


WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # letters only (handles åäö)

def extract_words(s: str):
    # "words" = sequences of letters only; punctuation/hyphens/spaces ignored
    return WORD_RE.findall(unicodedata.normalize("NFC", s or ""))

def edit_distance_leq1(a: str, b: str) -> bool:
    """
    True if Levenshtein distance <= 1 (fast path).
    Allows 1 insert/delete/replace.
    """
    a = (a or "")
    b = (b or "")
    if a == b:
        return True

    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False

    # Same length: at most 1 substitution
    if la == lb:
        mismatches = sum(1 for x, y in zip(a, b) if x != y)
        return mismatches <= 1

    # Ensure a is the shorter
    if la > lb:
        a, b = b, a
        la, lb = lb, la

    # lb = la + 1: at most 1 insertion
    i = j = 0
    edits = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            edits += 1
            if edits > 1:
                return False
            j += 1  # skip one char in longer string
    return True


# Allow-list for common Swedish confusions that are NOT "synonyms"
ALLOWED_WORD_SWAPS = {
    "de": {"dem"},
    "dem": {"de"},
    # Add more later only if you truly want to allow them.
    # Keep this minimal to avoid “rewrites”.
}

def is_small_word_edit(a: str, b: str) -> bool:
    """
    Allows:
    - spelling tweaks (1–2 char edits)
    - short grammar swaps like de/dem (allow-list)
    Rejects:
    - real rewrites/synonyms (low similarity / large changes)
    """
    a0 = (a or "").lower()
    b0 = (b or "").lower()

    if a0 == b0:
        return True

    if a0 in ALLOWED_WORD_SWAPS and b0 in ALLOWED_WORD_SWAPS[a0]:
        return True

    maxlen = max(len(a0), len(b0))

    # For short words, use edit distance (SequenceMatcher ratio is misleading here)
    if maxlen <= 4:
        return edit_distance_leq1(a0, b0)

    # For normal words, allow typical misspellings
    if maxlen <= 4:
        return edit_distance_leq1(a0, b0)
    elif maxlen <= 7:
        return difflib.SequenceMatcher(a=a0, b=b0).ratio() >= 0.85
    else:
        return difflib.SequenceMatcher(a=a0, b=b0).ratio() >= 0.90



def violates_no_word_add_remove(original: str, corrected: str) -> bool:
    """
    True if model added/removed/replaced whole words (not just spelling).
    """
    ow = extract_words(original)
    cw = extract_words(corrected)

    # Added/removed words
    if len(ow) != len(cw):
        return True

    # Word-by-word substitution (synonyms / rewrites)
    for a, b in zip(ow, cw):
        if not is_small_word_edit(a, b):
            return True

    return False



def project_safe_word_corrections(original: str, corrected: str) -> str:
    """
    Salvage mode:
    - Never adds/removes/reorders words
    - Applies ONLY 1-to-1 small spelling edits to existing words in the original text
    - Preserves original whitespace/punctuation exactly
    """
    orig = unicodedata.normalize("NFC", original or "")
    corr = unicodedata.normalize("NFC", corrected or "")

    orig_matches = list(WORD_RE.finditer(orig))
    corr_words = extract_words(corr)

    orig_words = [m.group(0) for m in orig_matches]
    if not orig_words or not corr_words:
        return original

    sm = difflib.SequenceMatcher(
        a=[w.lower() for w in orig_words],
        b=[w.lower() for w in corr_words],
        autojunk=False
    )

    # Collect replacements as (start, end, new_word)
    reps = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue
        if (i2 - i1) == 1 and (j2 - j1) == 1:
            a = orig_words[i1]
            b = corr_words[j1]
            if is_small_word_edit(a, b):  # spelling-level only
                m = orig_matches[i1]
                reps.append((m.start(), m.end(), b))

    if not reps:
        return original

    # Apply from end → start so offsets don't shift
    out = orig
    for s, e, nw in sorted(reps, key=lambda x: x[0], reverse=True):
        out = out[:s] + nw + out[e:]

    return out


# =================================================
# OPENAI CLIENT
# =================================================

# Uses OPENAI_API_KEY from environment (systemd/gunicorn env or .env you load elsewhere)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =================================================
# MAIN VIEW
# =================================================


def chunk_text_preserve(text: str, max_chars: int = 1800):
    """
    Split text i chunks (bevarar whitespace), så långa texter inte kollapsar till 0 diffs.
    Primärt på meningar/linjebryt, fallback till hård split.
    """
    if not text:
        return [""]

    # Split på meningar + större dubbla linjebryt, men behåll delimiters i output
    units = re.findall(r".*?(?:[.!?]+(?:\s+|$)|\n{2,}|$)", text, flags=re.S)
    units = [u for u in units if u]  # ta bort tomma

    if not units:
        units = [text]

    chunks = []
    buf = ""
    for u in units:
        if len(buf) + len(u) <= max_chars:
            buf += u
        else:
            if buf:
                chunks.append(buf)
                buf = ""
            # Om en enskild unit är för stor, split hårt
            if len(u) > max_chars:
                for i in range(0, len(u), max_chars):
                    chunks.append(u[i:i + max_chars])
            else:
                buf = u

    if buf:
        chunks.append(buf)

    return chunks


def correct_with_openai_chunked(text: str, max_chars: int = 1800) -> str:
    """
    Kör korrektur i chunks för långa texter.
    """
    parts = chunk_text_preserve(text, max_chars=max_chars)
    out = []
    for p in parts:
        if p.strip():
            out.append(correct_with_openai(p))
        else:
            out.append(p)
    return "".join(out)


def index(request):
    if request.method == "POST" and request.headers.get("x-requested-with") == "XMLHttpRequest":
        text = normalize_pasted_text(request.POST.get("text", ""))

        if not text.strip():
            return JsonResponse({
                "original_text": "",
                "corrected_text": "",
                "differences": [],
                "error_count": 0,
            })

        # ✅ Chunk correction när texten är lång
        if len(text) > 2000:
            corrected_text = correct_with_openai_chunked(text, max_chars=1800)
        else:
            corrected_text = correct_with_openai(text)

        # Normal diff (stram)
        differences = find_differences_charwise(text, corrected_text)

        # ✅ Om det FINNS ändringar men 0 diffs (typiskt vid långa texter / många kommatecken)
        if not differences and corrected_text.strip() != text.strip():
            differences = find_differences_charwise(
                text,
                corrected_text,
                max_block_tokens=80,
                max_block_chars=1200,
                max_diffs=300,
            )

        return JsonResponse({
            "original_text": text,
            "corrected_text": corrected_text,
            "differences": differences,
            "error_count": len(differences),
        })

    return render(request, "checker/index.html")



def same_words_exact(a: str, b: str) -> bool:
    # Komma-pass får INTE ändra bokstäver/ord — bara komma/whitespace.
    return extract_words(a) == extract_words(b)

ONLY_COMMA_WS_RE = re.compile(r"^[\s,]*$", re.UNICODE)

def _adjacent_has_comma(orig: str, i1: int, i2: int, cand: str, j1: int, j2: int) -> bool:
    def ch(s: str, idx: int) -> str:
        return s[idx] if 0 <= idx < len(s) else ""
    # Check character right before/after the edited segment in either string
    return (
        ch(orig, i1 - 1) == "," or ch(orig, i2) == "," or
        ch(cand, j1 - 1) == "," or ch(cand, j2) == ","
    )

def keep_only_comma_changes(original: str, candidate: str) -> str:
    """
    Keep ONLY:
    - comma insert/remove
    - whitespace changes that are directly adjacent to a comma (space after comma etc.)
    Reject whitespace changes elsewhere (prevents merges like 'privat livet' -> 'privatlivet').
    """
    orig = unicodedata.normalize("NFC", original or "")
    cand = unicodedata.normalize("NFC", candidate or "")

    if not orig or not cand:
        return original

    sm = difflib.SequenceMatcher(a=orig, b=cand, autojunk=False)
    out = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.append(orig[i1:i2])
            continue

        oseg = orig[i1:i2]
        cseg = cand[j1:j2]

        # Only allow edits that are commas/whitespace
        if ONLY_COMMA_WS_RE.fullmatch(oseg) and ONLY_COMMA_WS_RE.fullmatch(cseg):
            # Always allow if a comma is involved, or the edit touches a comma
            if ("," in oseg) or ("," in cseg) or _adjacent_has_comma(orig, i1, i2, cand, j1, j2):
                out.append(cseg)
            else:
                # Disallow whitespace-only edits away from commas (prevents word merges/splits)
                out.append(oseg)
        else:
            # Revert anything else (word changes, hyphens, etc.)
            out.append(oseg)

    return "".join(out)



def insert_commas_with_openai(text: str) -> str:
    """
    Inserts/removes commas ONLY.
    Must not change words/letters/case. Only commas + whitespace around commas.
    """
    try:
        system_prompt = (
            "Du är en svensk komma-korrekturläsare.\n\n"
            "REGLER (MÅSTE FÖLJAS):\n"
            "- Du får en text och du ska ENDAST rätta kommatecken.\n"
            "- ÄNDRA INTE stavning, versaler/gemener eller ordval.\n"
            "- LÄGG INTE TILL eller TA INTE BORT ord.\n"
            "- ÄNDRA INTE ordens ordning.\n"
            "- Du får ENDAST sätta in/ta bort kommatecken.\n"
            "- Du får ENDAST ändra mellanslag direkt före eller direkt efter ett kommatecken.\n"
            "- Du får ALDRIG ta bort/lägga till mellanslag mellan två ord (slå inte ihop eller dela upp ord).\n\n"
            "Returnera ENDAST texten."
        )

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        out = (resp.choices[0].message.content or "").rstrip(" \t")
        if not out:
            return text

        # Keep only comma + whitespace changes (prevents word merges/splits)
        safe = keep_only_comma_changes(text, out)
        safe = undo_space_merges(text, safe)  # extra safety if model still tries to merge words
        return safe

    except Exception as e:
        print("❌ OpenAI comma-only error:", e)
        return text


# =================================================
# OPENAI – SWEDISH (MIRRORS NORWEGIAN STYLE)
# =================================================
def correct_with_openai(text: str) -> str:
    """
    Hard constraints:
    - never add/remove/reorder words
    - allow spelling + punctuation + spacing
    - we also undo pure word-merges like "privat livet" -> "privatlivet"
    """
    try:
        base_prompt = (
            "Du är en professionell svensk korrekturläsare.\n\n"
            "MÅL: Rätta ALLA stavfel och ALL interpunktion i texten, särskilt kommatecken, "
            "utan att ändra innehåll, ordval eller ordföljd.\n\n"

            "ABSOLUTA REGLER (MÅSTE FÖLJAS):\n"
            "- LÄGG INTE TILL nya ord\n"
            "- TA INTE BORT ord\n"
            "- ÄNDRA INTE ordens ordning\n"
            "- Skriv INTE om meningar och använd INTE synonymer\n"
            "- Ändra ENDAST bokstäver INUTI befintliga ord för att rätta stavfel\n"
            "- Du får rätta interpunktion (komma, punkt, kolon, citattecken) och mellanslag\n"
            "- Bevara radbrytningar och stycken EXAKT som i input\n\n"

            "OBLIGATORISK FELKONTROLL (UTFÖRS TYST INNAN DU SVARAR):\n"
            "För VARJE mening måste du kontrollera ALLA punkter nedan. "
            "Hoppa inte över någon punkt, även om meningen ser korrekt ut.\n\n"

            "A) STAVNING:\n"
            "- Kontrollera varje ord för felstavning\n"
            "- Kontrollera dubbelteckning, sammansättningar och vanliga förväxlingar\n\n"

            "B) KOMMATECKEN (MYCKET VIKTIGT):\n"
            "1) Inledande bisats → KOMMA KRÄVS\n"
            "   (Om, När, Eftersom, Medan, Sedan, För att, Ifall, Då)\n"
            "2) Inskjutna bisatser / parentetiska inskott → KOMMA RUNT\n"
            "3) Två huvudsatser med 'och', 'men', 'eller':\n"
            "   - Har båda subjekt + verb → KOMMA KRÄVS\n"
            "4) Uppräkningar → KOMMA där det krävs för korrekt grammatik\n"
            "5) Enkel huvudsats → SÄTT ALDRIG komma mellan subjekt och verb\n\n"

            "C) SLUTKONTROLL:\n"
            "- Om ett komma saknas enligt reglerna är det ALLTID ett fel\n"
            "- Om ett stavfel finns måste det rättas\n"
            "- Returnera ALDRIG identisk text om något fel finns\n\n"

            "Returnera ENDAST den korrigerade texten. Ingen förklaring."
        )


        def call_llm(system_prompt: str, user_text: str) -> str:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                temperature=0,
            )
            return (resp.choices[0].message.content or "").rstrip(" \t")

        # 1) First attempt
        corrected = call_llm(base_prompt, text)
        if not corrected:
            return text
        
                # 🚨 HARD FAIL: text had errors but model returned unchanged output
        if corrected.strip() == text.strip():
            raise RuntimeError("Model returned unchanged text despite required corrections")


        corrected = undo_space_merges(text, corrected)

        # 2) If unchanged, retry once with a nudge
        if corrected.strip() == text.strip():
            nudge_prompt = base_prompt + (
                "\n\nTEXTEN INNEHÅLLER FEL.\n"
                "Du måste rätta alla tydliga stavfel OCH alla kommateckenfel inom reglerna.\n"
                "Kör KOMMA-KONTROLLEN (punkt 1–5) mening för mening och returnera inte identisk text om något komma saknas/är fel."
            )

            corrected2 = call_llm(nudge_prompt, text)
            if corrected2:
                corrected2 = undo_space_merges(text, corrected2)
                corrected = corrected2

        # 3) Validate: if model added/removed/substituted whole words → retry strict once
# 3) Validate: if model added/removed/substituted whole words → retry strict once
        if violates_no_word_add_remove(text, corrected):
            strict_prompt = base_prompt + (
                "\n\nEXTRA STRIKT:\n"
                "- Antalet ord i svaret MÅSTE vara identiskt med input\n"
                "- Varje ord i output ska vara samma ord som input (endast små stavningsändringar är tillåtna)\n"
                "- Förbättra inte meningar eller flyt; rätta endast skrivfel och interpunktion.\n"
            )
            corrected2 = call_llm(strict_prompt, text)
            if corrected2:
                corrected2 = undo_space_merges(text, corrected2)

                # ✅ IMPORTANT: don't return yet — let comma-only pass run later
                if not violates_no_word_add_remove(text, corrected2):
                    corrected = corrected2
                else:
                    # 4) Salvage instead of returning original:
                    salvaged = project_safe_word_corrections(text, corrected2)
                    if not salvaged:
                        salvaged = project_safe_word_corrections(text, corrected)

                    if salvaged:
                        salvaged = insert_commas_with_openai(salvaged)
                        return salvaged

            # If strict failed, keep going with whatever we had (and run comma-only pass)


        # ✅ second pass: comma-only (won't change words)
        corrected = insert_commas_with_openai(corrected)
        return corrected

    except Exception as e:
        print("❌ OpenAI error:", e)
        return text


WS_TOKEN_RE = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)

def undo_space_merges(original: str, corrected: str, max_merge_words: int = 3) -> str:
    """
    Reverts corrections that ONLY merge multiple letter-words by removing spaces:
      "privat livet" -> "privatlivet"

    It does NOT touch hyphenations like:
      "e - post" -> "e-post"
    because that's not a pure space-removal merge.

    Keeps original whitespace between the words (so line breaks stay line breaks).
    """
    if not original or not corrected:
        return corrected

    orig_full = WS_TOKEN_RE.findall(unicodedata.normalize("NFC", original))
    corr_full = WS_TOKEN_RE.findall(unicodedata.normalize("NFC", corrected))

    def is_ws(t: str) -> bool:
        return t.isspace()

    # Letters-only (no digits/underscore). Works for Swedish letters too.
    def is_word(t: str) -> bool:
        return bool(re.fullmatch(r"[^\W\d_]+", t, re.UNICODE))

    # Build "significant token" lists (no whitespace) + map sig-index -> full-index
    orig_sig, orig_map = [], []
    for idx, tok in enumerate(orig_full):
        if not is_ws(tok):
            orig_sig.append(tok)
            orig_map.append(idx)

    corr_sig, corr_map = [], []
    for idx, tok in enumerate(corr_full):
        if not is_ws(tok):
            corr_sig.append(tok)
            corr_map.append(idx)

    # Lowercase for matching
    sm = difflib.SequenceMatcher(
        a=[t.lower() for t in orig_sig],
        b=[t.lower() for t in corr_sig],
        autojunk=False,
    )

    replacements = {}  # corr_full_index -> replacement_string

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue

        # We only care about N words -> 1 word
        if (j2 - j1) != 1:
            continue
        n = (i2 - i1)
        if not (2 <= n <= max_merge_words):
            continue

        corr_tok = corr_sig[j1]
        if not is_word(corr_tok):
            continue
        if not all(is_word(t) for t in orig_sig[i1:i2]):
            continue

        # Pure merge check: join original words equals corrected word (case-insensitive)
        if "".join(orig_sig[i1:i2]).lower() != corr_tok.lower():
            continue

        # Make sure the original region between these word tokens contains ONLY whitespace
        start_full = orig_map[i1]
        end_full = orig_map[i2 - 1]
        between = orig_full[start_full:end_full + 1]
        if sum(1 for t in between if not is_ws(t)) != n:
            continue

        replacement_str = "".join(between)  # preserves original whitespace between words
        corr_full_index = corr_map[j1]
        replacements[corr_full_index] = replacement_str

    if not replacements:
        return corrected

    # Apply replacements (no index shifting; we replace token content only)
    for idx, rep in replacements.items():
        corr_full[idx] = rep

    return "".join(corr_full)


# =================================================
# DIFF ENGINE (IDENTICAL TO NORWEGIAN/DANISH)
# =================================================
def find_differences_charwise(original: str, corrected: str, max_block_tokens: int = 14, max_block_chars: int = 180, max_diffs: int = 250):
    """
    Robust token diff that:
    - handles merges/splits (e.g., 'e - post' -> 'e-post')
    - returns original-string char spans (start/end) so frontend can highlight precisely
    - groups adjacent diffs into larger 'areas' to avoid highlighting every single word
    """
    orig_text = unicodedata.normalize("NFC", (original or "").replace("\r\n", "\n").replace("\r", "\n"))
    corr_text = unicodedata.normalize("NFC", (corrected or "").replace("\r\n", "\n").replace("\r", "\n"))

    if not orig_text and not corr_text:
        return []

    token_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def tokens_with_spans(s: str):
        toks, spans = [], []
        for m in token_re.finditer(s):
            toks.append(m.group(0))
            spans.append((m.start(), m.end()))
        return toks, spans

    orig_tokens, orig_spans = tokens_with_spans(orig_text)
    corr_tokens, corr_spans = tokens_with_spans(corr_text)

    def span_for_token_range(spans, i1, i2, text_len):
        """Char span from first token start to last token end, including any whitespace between."""
        if not spans:
            return 0, 0
        if i1 >= len(spans):
            return text_len, text_len
        if i1 == i2:
            # insertion point: before token i1
            return spans[i1][0], spans[i1][0]
        return spans[i1][0], spans[i2 - 1][1]

    def norm_no_space(s: str) -> str:
        # remove whitespace only; keep punctuation so 'e - post' ~ 'e-post'
        return re.sub(r"\s+", "", s.lower())

    def similarity(a: str, b: str) -> float:
        return difflib.SequenceMatcher(a=a, b=b).ratio()

    def is_pure_punct(s: str) -> bool:
        # punctuation-only string (commas, periods, hyphens, etc.)
        return bool(re.fullmatch(r"[^\w\s]+", s, re.UNICODE))

    # Important: disable autojunk (it can behave oddly on short/repetitive text)
    sm = difflib.SequenceMatcher(a=orig_tokens, b=corr_tokens, autojunk=False)

    raw_diffs = []

    # Build raw diffs from opcodes
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        o_start, o_end = span_for_token_range(orig_spans, i1, i2, len(orig_text))
        c_start, c_end = span_for_token_range(corr_spans, j1, j2, len(corr_text))

        o_chunk = orig_text[o_start:o_end]
        c_chunk = corr_text[c_start:c_end]

        # Keep things local (prevents huge “rewrite” highlights)
        o_tok_count = i2 - i1
        c_tok_count = j2 - j1
        if (o_tok_count + c_tok_count) > max_block_tokens:
            continue
        if (len(o_chunk) + len(c_chunk)) > max_block_chars:
            continue

        if tag == "replace":
            # Accept if it's basically a local correction OR a whitespace-merge/split
            if norm_no_space(o_chunk) == norm_no_space(c_chunk) or similarity(o_chunk.lower(), c_chunk.lower()) >= 0.55:
                raw_diffs.append({
                    "type": "replace",
                    "start": o_start,
                    "end": o_end,
                    "original": o_chunk,
                    "suggestion": c_chunk,
                    "c_start": c_start,
                    "c_end": c_end,
                })

        elif tag == "insert":
            # Punctuation-only inserts (like a missing comma) often become "0-length" diffs,
            # which the frontend can't highlight/click. Convert them into a small REPLACE
            # around nearby tokens.
            if c_chunk and is_pure_punct(c_chunk):

                # Expand original context to 1 token left + 1 token right (when possible)
                left_i = max(i1 - 1, 0)
                right_i = min(i1 + 1, len(orig_tokens))  # range end is exclusive

                # Expand corrected context similarly (token before + inserted punct + token after)
                left_j = max(j1 - 1, 0)
                right_j = min(j2 + 1, len(corr_tokens))

                o_start2, o_end2 = span_for_token_range(orig_spans, left_i, right_i, len(orig_text))
                c_start2, c_end2 = span_for_token_range(corr_spans, left_j, right_j, len(corr_text))

                o_chunk2 = orig_text[o_start2:o_end2]
                c_chunk2 = corr_text[c_start2:c_end2]

                # Keep things local (same safety limits)
                if ((right_i - left_i) + (right_j - left_j)) > max_block_tokens:
                    continue
                if (len(o_chunk2) + len(c_chunk2)) > max_block_chars:
                    continue

                if o_chunk2 != c_chunk2:
                    raw_diffs.append({
                        "type": "replace",
                        "start": o_start2,
                        "end": o_end2,
                        "original": o_chunk2,
                        "suggestion": c_chunk2,
                        "c_start": c_start2,
                        "c_end": c_end2,
                    })

        elif tag == "delete":
            # Only surface small deletes (usually punctuation / tiny tokens)
            if o_chunk and (is_pure_punct(o_chunk) or len(o_chunk.strip()) <= 2):
                raw_diffs.append({
                    "type": "delete",
                    "start": o_start,
                    "end": o_end,
                    "original": o_chunk,
                    "suggestion": "",
                    "c_start": c_start,
                    "c_end": c_end,
                })

    if not raw_diffs:
        return []

    # Sort and GROUP into “areas” (merge diffs separated only by whitespace)
    raw_diffs.sort(key=lambda d: (d["start"], d["end"]))

    grouped = [raw_diffs[0]]
    for d in raw_diffs[1:]:
        prev = grouped[-1]

        gap = orig_text[prev["end"]:d["start"]]
        gap_is_only_ws = (gap.strip() == "")
        gap_has_parabreak = ("\n\n" in gap)

        # Merge only if there is no paragraph break between
        if gap_is_only_ws and (not gap_has_parabreak) and (d["start"] <= prev["end"] + 2):

            prev["end"] = max(prev["end"], d["end"])
            prev["start"] = min(prev["start"], d["start"])

            # Rebuild the displayed chunks
            prev["original"] = orig_text[prev["start"]:prev["end"]]

            # Best-effort: if we have corrected spans, merge them too
            if "c_start" in prev and "c_start" in d:
                prev["c_start"] = min(prev["c_start"], d["c_start"])
                prev["c_end"] = max(prev["c_end"], d["c_end"])
                prev["suggestion"] = corr_text[prev["c_start"]:prev["c_end"]]

            prev["type"] = "replace"
        else:
            grouped.append(d)

    # Optional: dedupe identical spans
    out = []
    seen = set()
    for d in grouped:
        key = (d["start"], d["end"], d.get("suggestion", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "type": d["type"],
            "start": d["start"],
            "end": d["end"],
            "original": d["original"],
            "suggestion": d["suggestion"],
        })

    return out[:max_diffs]


# =================================================
# AUTH (UNCHANGED LOGIC, SWEDISH MESSAGES)
# =================================================
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from django.contrib.auth.models import User


def register(request):
    if request.method != "POST":
        return redirect("index")

    email = request.POST.get("email")
    password = request.POST.get("password")
    name = request.POST.get("name")

    if User.objects.filter(username=email).exists():
        messages.error(request, "E-postadressen används redan.")
        return redirect("/")

    user = User.objects.create_user(
        username=email,
        email=email,
        password=password,
        first_name=name,
    )

    login(request, user)
    return redirect("/")


def login_view(request):
    if request.method != "POST":
        return redirect("/")

    user = authenticate(
        request,
        username=request.POST.get("email"),
        password=request.POST.get("password"),
    )

    if user is None:
        messages.error(request, "Fel e-post eller lösenord.")
        return redirect("/")

    login(request, user)
    return redirect("/")


def logout_view(request):
    if request.method == "POST":
        logout(request)
    return redirect("/")
