from django.shortcuts import render, redirect
from django.http import JsonResponse
from openai import OpenAI
import logging
import re
import difflib
import unicodedata

client = OpenAI()
logger = logging.getLogger(__name__)





def correct_with_openai_sv(text: str) -> str:
    try:
        # 🔹 Collapse all whitespace FIRST
        text = re.sub(r"\s+", " ", text).strip()

        system_prompt = (
            "Du är en professionell svensk språkkorrekturläsare.\n\n"
            "VIKTIGA REGLER (OBLIGATORISKT):\n"
            "- LÄGG INTE TILL nya ord\n"
            "- TA INTE BORT ord\n"
            "- ÄNDRA INTE ordens ordning\n"
            "- DELA INTE upp eller slå ihop ord\n"
            "- ÄNDRA INTE mellanslag eller radbrytningar\n\n"
            "Du FÅR ENDAST:\n"
            "- korrigera stavfel INUTI befintliga ord\n"
            "- lägga till eller ta bort skiljetecken SOM ÄR DEL AV ORDET "
            "(t.ex. 'att' → 'att,')\n\n"
            "- När du lägger till skiljetecken: skriv det DIREKT efter ordet utan mellanslag (t.ex. 'men' → 'men,').\n"
            "Om en ändring kräver omformulering, LÄMNA DEN OÄNDRAD.\n\n"
            "Returnera ENDAST texten, utan förklaringar."

        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )

        corrected = (resp.choices[0].message.content or "").strip()

        # 🔐 Absolute safety: word count must match
        if len(corrected.split()) != len(text.split()):
            logger.warning("Word count mismatch – falling back to original")
            return text

        return corrected if corrected else text

    except Exception:
        logger.exception("OpenAI error")
        return text
    

import re
import difflib
import unicodedata

def find_differences_charwise(original: str, corrected: str):
    """
    Token-level diff with alignment.

    Surfaces:
    - small spelling tweaks inside a word
    - punctuation edits (especially comma insert/remove)

    Ignores:
    - bigger rewrites / multi-token replacements (so you don't mark everything red)
    """

    diffs_out = []

    orig_text = unicodedata.normalize("NFC", original)
    corr_text = unicodedata.normalize("NFC", corrected)

    # Merge standalone punctuation into previous token
    def merge_punctuation(tokens):
        merged = []
        for tok in tokens:
            if merged and re.fullmatch(r"[,.:;!?]", tok):
                merged[-1] += tok
            else:
                merged.append(tok)
        return merged

    # Tokenize into words OR single punctuation chars
    orig_tokens = merge_punctuation(re.findall(r"\w+|[^\w\s]", orig_text, flags=re.UNICODE))
    corr_tokens = merge_punctuation(re.findall(r"\w+|[^\w\s]", corr_text, flags=re.UNICODE))

    # Map each original token back to (start, end) in original string
    orig_positions = []
    cursor = 0
    for tok in orig_tokens:
        start = orig_text.find(tok, cursor)
        if start == -1:
            # If we ever fail mapping, bail out safely
            return []
        end = start + len(tok)
        orig_positions.append((start, end))
        cursor = end

    def span_for_range(i_start, i_end_exclusive):
        if i_start >= len(orig_positions):
            return len(orig_text), len(orig_text)
        if i_start == i_end_exclusive:
            start_i, _ = orig_positions[i_start]
            return start_i, start_i
        start_char = orig_positions[i_start][0]
        end_char = orig_positions[i_end_exclusive - 1][1]
        return start_char, end_char

    def is_pure_punctuation(tok: str) -> bool:
        return bool(re.fullmatch(r"[,.:;!?]+", tok))

    def tokens_are_small_edit(a: str, b: str) -> bool:
        a_low = a.lower()
        b_low = b.lower()

        # allow punctuation-only or case-only diffs
        if a_low.strip(",.;:!?") == b_low.strip(",.;:!?"):
            return True

        ratio = difflib.SequenceMatcher(a=a_low, b=b_low).ratio()
        return ratio >= 0.8  # stricter to avoid “everything is a change”

    sm = difflib.SequenceMatcher(a=orig_tokens, b=corr_tokens)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        if tag == "replace":
            # Only accept 1-to-1 small edits (word->word, / stavning)
            if (i2 - i1) == 1 and (j2 - j1) == 1:
                orig_tok = orig_tokens[i1]
                corr_tok = corr_tokens[j1]
                if tokens_are_small_edit(orig_tok, corr_tok):
                    start_char, end_char = span_for_range(i1, i2)
                    diffs_out.append({
                        "type": "replace",
                        "start": start_char,
                        "end": end_char,
                        "original": orig_tok,
                        "suggestion": corr_tok,
                    })
            continue

        if tag == "delete":
            # Only surface deletion if it's punctuation-ish
            if (i2 - i1) == 1:
                orig_tok = orig_tokens[i1]
                if is_pure_punctuation(orig_tok) or is_pure_punctuation(orig_tok[-1:]):
                    start_char, end_char = span_for_range(i1, i2)
                    diffs_out.append({
                        "type": "delete",
                        "start": start_char,
                        "end": end_char,
                        "original": orig_tok,
                        "suggestion": "",
                    })
            continue

        if tag == "insert":
            # Only surface insertion if it’s punctuation
            if (j2 - j1) == 1:
                corr_tok = corr_tokens[j1]
                if is_pure_punctuation(corr_tok) or is_pure_punctuation(corr_tok[-1:]):
                    start_char, _ = span_for_range(i1, i1)
                    diffs_out.append({
                        "type": "insert",
                        "start": start_char,
                        "end": start_char,
                        "original": "",
                        "suggestion": corr_tok,
                    })
            continue

    return diffs_out

def index(request):
    if request.method == "POST" and request.headers.get("x-requested-with") == "XMLHttpRequest":
        raw_text = (request.POST.get("text") or "").strip()

        if not raw_text:
            return JsonResponse({
                "original_text": "",
                "corrected_text": "",
                "differences": [],
                "error_count": 0,
            })

        # ✅ COLLAPSE ONCE, EARLY
        collapsed_text = re.sub(r"\s+", " ", raw_text).strip()

        corrected = correct_with_openai_sv(collapsed_text)
        differences = find_differences_charwise(collapsed_text, corrected)

        return JsonResponse({
            "original_text": collapsed_text,
            "corrected_text": corrected,
            "differences": differences,
            "error_count": len(differences),
        })

    return render(request, "checker/index.html")



from django.contrib.auth.models import User
from django.contrib.auth import login
from django.contrib import messages

def register(request):
    if request.method != "POST":
        return redirect("index")

    name = request.POST.get("name")
    email = request.POST.get("email")
    password = request.POST.get("password")

    if User.objects.filter(username=email).exists():
        messages.error(request, "E-postadressen används redan.")
        return redirect(request.POST.get("next", "/"))

    user = User.objects.create_user(
        username=email,
        email=email,
        password=password,
        first_name=name,
    )

    login(request, user)
    return redirect(request.POST.get("next", "/"))


from django.contrib.auth import authenticate, login

def login_view(request):
    if request.method != "POST":
        return redirect("/")

    email = request.POST.get("email")
    password = request.POST.get("password")

    if not email or not password:
        messages.error(request, "Fyll i både e-post och lösenord.")
        return redirect(request.POST.get("next", "/"))

    user = authenticate(
        request,
        username=email,
        password=password
    )

    if user is None:
        messages.error(request, "Fel e-post eller lösenord.")
        return redirect(request.POST.get("next", "/"))

    login(request, user)
    return redirect(request.POST.get("next", "/"))


from django.contrib.auth import logout

def logout_view(request):
    if request.method == "POST":
        logout(request)
    return redirect("/")
