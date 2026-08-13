
NOT_MENTIONED = "not mentioned"

SPATIAL_PREPS = {
    "on", "under", "underneath", "beneath", "below", "above", "over", "behind",
    "beside", "near", "by", "between", "against", "inside", "outside", "atop",
    "alongside", "opposite", "facing", "around", "amongst", "among", "within",
    "toward", "towards", "across", "at", "in", "next", "adjacent", "close", "atop",
}

SPATIAL_PHRASES = [
    "next to", "in front of", "across from", "close to", "adjacent to",
    "on top of", "to the left of", "to the right of", "in the middle of",
    "in the corner of", "at the end of", "in between", "far from",
    "to the side of", "on the side of", "in the back of", "at the back of",
]

_PRONOUNS = {
    "there", "it", "this", "that", "these", "those", "they", "he", "she",
    "you", "i", "we", "one", "ones", "something", "anything", "which", "who",
}

_LEADING_DROP_DEPS = {"det", "poss", "predet"}


def load_parser(model="en_core_web_sm"):
    import spacy

    if model not in load_parser._cache:
        try:
            load_parser._cache[model] = spacy.load(model, exclude=["ner", "lemmatizer"])
        except OSError as exc:
            raise OSError(
                f"spaCy model '{model}' is not installed. Install it with:\n"
                f"    python -m spacy download {model}"
            ) from exc
    return load_parser._cache[model]


load_parser._cache = {}


def _phrase_ending_at(doc, token):
    start = max(0, token.i - 3)
    return " ".join(t.text.lower() for t in doc[start:token.i + 1])


def _is_spatial_prep(doc, token):
    if token.text.lower() in SPATIAL_PREPS:
        return True
    window = _phrase_ending_at(doc, token)
    return any(window.endswith(phrase) for phrase in SPATIAL_PHRASES)


def _strip_span(doc, start, end):
    while start < end and (doc[start].dep_ in _LEADING_DROP_DEPS
                           or doc[start].is_punct
                           or doc[start].is_space):
        start += 1
    while end > start and (doc[end - 1].is_punct or doc[end - 1].is_space):
        end -= 1
    return start, end


def _span_text(doc, start, end):
    start, end = _strip_span(doc, start, end)
    if start >= end:
        return ""
    return " ".join(doc[start:end].text.split())


def _select_target(doc):
    chunks = list(doc.noun_chunks)
    if not chunks:
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN") and token.text.lower() not in _PRONOUNS:
                return token
        return None

    for chunk in chunks:
        root = chunk.root
        if root.text.lower() in _PRONOUNS or root.pos_ == "PRON":
            continue
        if root.dep_ == "pobj" and root.head is not None and _is_spatial_prep(doc, root.head):
            continue
        return root

    return chunks[0].root


def _target_span(doc, target):
    start = target.i
    compounds = sorted((c for c in target.children if c.dep_ == "compound"),
                       key=lambda t: t.i, reverse=True)
    for child in compounds:
        if child.i == start - 1:
            start = child.i
    return start, target.i + 1


def _adjective_text(doc, target, target_start):
    idxs = []
    for child in target.children:
        if child.dep_ in ("amod", "nummod"):
            idxs.append(child.i)
            for grand in child.children:
                if grand.dep_ in ("amod", "advmod", "compound", "conj", "cc"):
                    idxs.append(grand.i)
            for conj in child.conjuncts:
                idxs.append(conj.i)

    parts = []
    if idxs:
        lo, hi = min(idxs), max(idxs)
        hi = min(hi + 1, target_start)
        text = _span_text(doc, lo, hi)
        if text:
            parts.append(text)

    if target.dep_ in ("nsubj", "nsubjpass") and target.head is not None and \
            target.head.lower_ in ("is", "are", "was", "were", "be", "being", "been",
                                   "looks", "appears", "seems"):
        for child in target.head.children:
            if child.dep_ == "acomp":
                sub = list(child.subtree)
                text = _span_text(doc, sub[0].i, sub[-1].i + 1)
                if text:
                    parts.append(text)

    return ", ".join(p for p in parts if p)


def _neighbor_text(doc, target):
    preps = []
    for token in doc:
        if token.pos_ != "ADP" and token.dep_ not in ("prep", "agent"):
            continue
        if not _is_spatial_prep(doc, token):
            continue
        if not any(c.dep_ in ("pobj", "pcomp") for c in token.children):
            continue
        preps.append(token)

    tops = []
    for prep in preps:
        ancestors = set(a.i for a in prep.ancestors)
        if any(other.i in ancestors for other in preps):
            continue
        tops.append(prep)

    parts, seen = [], set()
    for prep in tops:
        sub = list(prep.subtree)
        start, end = sub[0].i, sub[-1].i + 1
        window = _phrase_ending_at(doc, prep)
        for phrase in SPATIAL_PHRASES:
            if window.endswith(phrase):
                back = len(phrase.split()) - 1
                start = min(start, max(0, prep.i - back))
                break
        if target is not None and start <= target.i < end:
            continue
        text = _span_text(doc, start, end)
        if text and text not in seen:
            seen.add(text)
            parts.append(text)

    return ", ".join(parts)


def empty_parse():
    return {"target": NOT_MENTIONED,
            "adjectives": NOT_MENTIONED,
            "neighbors": NOT_MENTIONED}


def parse_doc(doc):
    if doc is None or len(doc) == 0:
        return empty_parse()

    target = _select_target(doc)
    if target is None:
        return empty_parse()

    tgt_start, tgt_end = _target_span(doc, target)
    return {
        "target": _span_text(doc, tgt_start, tgt_end) or NOT_MENTIONED,
        "adjectives": _adjective_text(doc, target, tgt_start) or NOT_MENTIONED,
        "neighbors": _neighbor_text(doc, target) or NOT_MENTIONED,
    }


def parse_with_spacy(description, model="en_core_web_sm", nlp=None):
    if not description or not description.strip():
        return empty_parse()

    if nlp is None:
        nlp = load_parser(model)
    return parse_doc(nlp(description.lower()))
