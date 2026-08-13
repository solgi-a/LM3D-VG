"""
Rule-based dependency parser -- variant B of the parser ablation.

Drop-in replacement for the GPT-4o-mini parsing stage, emitting exactly the schema of
``data_parsing/final_parsing/parsed_result_{split}.json``, so swapping parsers needs no
change to the language module or the fusion network.

Output schema, derived from the real files
------------------------------------------
GPT-4o-mini writes raw phrase strings, one per field::

    {"target":     "chair",
     "adjectives": "dark brown wooden and leather",
     "neighbors":  "in the kitchen, placed in the table"}

Two properties of that file are load-bearing and reproduced here:

1. **Fields are surface phrases, not head-word lists** -- they keep determiners,
   prepositions, conjunctions and commas. Bare head nouns would make the ablation compare
   representation format rather than parser quality.
2. **An absent field is the literal string "not mentioned"**, never ``null``, ``""`` or
   ``[]``. Verified across all 9,508 val and 36,665 train annotations: 1,970 absent
   adjectives, 113 absent neighbors, 36 absent targets, no empty values of any other form.
   After tokenisation it becomes ``["not", "mentioned"]``, which is what the GRU consumes.

Tokenisation and the 7/17/75 caps are applied afterwards by ``tokenize_parse.py``, shared
by every parser variant.
"""

#: Literal used by the GPT-4o-mini pipeline for a field the description does not mention.
NOT_MENTIONED = "not mentioned"

#: Single-token spatial prepositions.
SPATIAL_PREPS = {
    "on", "under", "underneath", "beneath", "below", "above", "over", "behind",
    "beside", "near", "by", "between", "against", "inside", "outside", "atop",
    "alongside", "opposite", "facing", "around", "amongst", "among", "within",
    "toward", "towards", "across", "at", "in", "next", "adjacent", "close", "atop",
}

#: Multi-token spatial cues, matched on the surface string ending at the preposition.
SPATIAL_PHRASES = [
    "next to", "in front of", "across from", "close to", "adjacent to",
    "on top of", "to the left of", "to the right of", "in the middle of",
    "in the corner of", "at the end of", "in between", "far from",
    "to the side of", "on the side of", "in the back of", "at the back of",
]

#: Never treated as the target even when they head the first noun chunk.
_PRONOUNS = {
    "there", "it", "this", "that", "these", "those", "they", "he", "she",
    "you", "i", "we", "one", "ones", "something", "anything", "which", "who",
}

#: Determiner-ish dependencies stripped from the front of an extracted span.
_LEADING_DROP_DEPS = {"det", "poss", "predet"}


# ======================================================================================
# Pipeline
# ======================================================================================

def load_parser(model="en_core_web_sm"):
    """Load and cache a spaCy pipeline.

    ``en_core_web_sm`` is the default. ``en_core_web_trf`` is more accurate but roughly
    30-50x slower; over ScanRefer's ~46k descriptions that is the difference between
    about a minute and most of an hour on CPU, so the small model is the sensible
    default and the transformer is worth trying only if target accuracy disappoints.

    NER and the lemmatizer are excluded for speed. Everything the rules below read
    (POS tags, dependency labels, noun chunks, ``.lower_``) comes from the tagger,
    parser and attribute_ruler, which are kept. Nothing here touches ``.lemma_``.
    """
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


# ======================================================================================
# Helpers
# ======================================================================================

def _phrase_ending_at(doc, token):
    """Surface string of the up-to-4-token window ending at ``token``, lowercased."""
    start = max(0, token.i - 3)
    return " ".join(t.text.lower() for t in doc[start:token.i + 1])


def _is_spatial_prep(doc, token):
    if token.text.lower() in SPATIAL_PREPS:
        return True
    window = _phrase_ending_at(doc, token)
    return any(window.endswith(phrase) for phrase in SPATIAL_PHRASES)


def _strip_span(doc, start, end):
    """Trim determiners and stray punctuation from both ends of doc[start:end]."""
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


# ======================================================================================
# Target
# ======================================================================================

def _select_target(doc):
    """Head noun of the first genuine noun chunk.

    Skips existential/pronominal chunks ("there is a chair...") and, when the sentence
    opens with an imperative or descriptive verb ("find the chair..."), the first noun
    chunk after that verb is taken. Noun chunks sitting inside a spatial prepositional
    phrase describe a neighbour, not the target, so they are skipped too.
    """
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
    """The head noun plus any contiguous compound modifiers ("kitchen cabinet")."""
    start = target.i
    compounds = sorted((c for c in target.children if c.dep_ == "compound"),
                       key=lambda t: t.i, reverse=True)
    for child in compounds:
        if child.i == start - 1:
            start = child.i
    return start, target.i + 1


# ======================================================================================
# Adjectives
# ======================================================================================

def _adjective_text(doc, target, target_start):
    """Surface span covering the target's adjectival modifiers.

    Taking the span (rather than the individual tokens) is what reproduces GPT's output:
    "dark brown wooden and leather chair" yields "dark brown wooden and leather",
    conjunction included, because the span between the first and last modifier is kept
    intact.
    """
    idxs = []
    for child in target.children:
        if child.dep_ in ("amod", "nummod"):
            idxs.append(child.i)
            # "dark brown chair": "dark" attaches to "brown", not to "chair".
            for grand in child.children:
                if grand.dep_ in ("amod", "advmod", "compound", "conj", "cc"):
                    idxs.append(grand.i)
            for conj in child.conjuncts:
                idxs.append(conj.i)

    parts = []
    if idxs:
        lo, hi = min(idxs), max(idxs)
        hi = min(hi + 1, target_start)  # never swallow the target itself
        text = _span_text(doc, lo, hi)
        if text:
            parts.append(text)

    # Predicative adjectives: "the chair is brown".
    # Matched on the surface form, not .lemma_, because load_parser drops the lemmatizer.
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


# ======================================================================================
# Neighbors
# ======================================================================================

def _neighbor_text(doc, target):
    """Surface spans of the spatial prepositional phrases, joined with ", ".

    Nested PPs are folded into their outermost parent ("on the side of the table facing
    the ovens" stays one phrase) so the output reads like GPT's.
    """
    preps = []
    for token in doc:
        if token.pos_ != "ADP" and token.dep_ not in ("prep", "agent"):
            continue
        if not _is_spatial_prep(doc, token):
            continue
        # The phrase must actually govern a nominal.
        if not any(c.dep_ in ("pobj", "pcomp") for c in token.children):
            continue
        preps.append(token)

    # Drop any preposition contained inside another one we already took.
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
        # Multi-word cues ("next to", "in front of") start before the ADP itself.
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


# ======================================================================================
# Public entry point
# ======================================================================================

def empty_parse():
    """The record emitted when nothing could be extracted."""
    return {"target": NOT_MENTIONED,
            "adjectives": NOT_MENTIONED,
            "neighbors": NOT_MENTIONED}


def parse_doc(doc):
    """Apply the extraction rules to an already-parsed spaCy ``Doc``.

    Split out from ``parse_with_spacy`` so callers can use ``nlp.pipe`` over the whole
    dataset, which is several times faster than parsing one description at a time.
    """
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
    """Parse one referring expression into the GPT-4o-mini output schema.

    Returns ``{"target": str, "adjectives": str, "neighbors": str}`` where every field is
    a lowercase surface phrase, or the literal ``"not mentioned"`` when the description
    does not supply it.
    """
    if not description or not description.strip():
        return empty_parse()

    if nlp is None:
        nlp = load_parser(model)
    return parse_doc(nlp(description.lower()))
