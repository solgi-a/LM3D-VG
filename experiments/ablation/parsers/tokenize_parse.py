"""
Shared tokenizer for every parser variant.

`lib/dataset.py` reads the tokenized form, not the raw `parsed_result_*.json`::

    { scene_id: { object_id: { ann_id: {"target":      [tok, ...],
                                        "adjectives":  [tok, ...],
                                        "neighbors":   [tok, ...]} } } }

Every variant goes through this module, so the only thing differing between ablation arms
is the parser itself.

The tokenisation reproduces the convention of the existing caches, verified against
`final_parsing/` vs `final_parsing_tokenized/`: lowercase, whitespace split, punctuation
as its own token. It matches ScanRefer's ``token`` field, e.g.

    "in the kitchen, placed in the table"
        -> ['in','the','kitchen',',','placed','in','the','table']
    "at the table."
        -> ['at','the','table','.']

Two load-bearing constraints
----------------------------
1. **Token caps: target 7, adjectives 17, neighbors 75.** ``_transform_parsed`` allocates
   ``(num_token, 300)`` with num_token = 7/17/75, but ``__getitem__`` sets
   ``tgt_len = len(tokens)`` unclipped (lib/dataset.py:173-175). A longer list makes
   ``pack_padded_sequence`` receive a length beyond the sequence dimension and raise
   mid-training. The GPT caches are clipped at these values; the LLaMA cache is not (see
   clip_parse_cache.py).

2. **No field may be empty.** ``pack_padded_sequence`` rejects length 0. An absent field
   carries the tokens of ``"not mentioned"``, the GPT pipeline's convention.
"""

import re

#: Maximum tokens per field, fixed by _transform_parsed in lib/dataset.py.
FIELD_MAX_TOKENS = {"target": 7, "adjectives": 17, "neighbors": 75}

#: What an absent field tokenizes to.
NOT_MENTIONED_TOKENS = ["not", "mentioned"]

#: Words and punctuation, matching ScanRefer's own `token` field.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")


def tokenize(text):
    """Lowercase word/punctuation tokenisation matching the existing parse caches."""
    if text is None:
        return []
    return _TOKEN_RE.findall(str(text).lower())


def tokenize_field(text, field):
    """Tokenize one field and enforce its cap; never returns an empty list."""
    tokens = tokenize(text)
    if not tokens:
        tokens = list(NOT_MENTIONED_TOKENS)
    return tokens[: FIELD_MAX_TOKENS[field]]


def tokenize_record(parsed):
    """Tokenize a {target, adjectives, neighbors} record of raw strings."""
    return {field: tokenize_field(parsed.get(field), field)
            for field in FIELD_MAX_TOKENS}


def clip_tokens(tokens, field):
    """Enforce the cap on an already-tokenized list; never returns an empty list."""
    if not tokens:
        return list(NOT_MENTIONED_TOKENS)
    return list(tokens)[: FIELD_MAX_TOKENS[field]]


def validate_tokenized(data, label=""):
    """Assert the invariants lib/dataset.py relies on. Returns (ok, problems, checked)."""
    problems, checked = [], 0
    for scene_id, objects in data.items():
        for object_id, anns in objects.items():
            for ann_id, parsed in anns.items():
                checked += 1
                where = f"{label}{scene_id}/{object_id}/{ann_id}"
                if set(parsed) != set(FIELD_MAX_TOKENS):
                    problems.append(f"{where}: wrong keys {sorted(parsed)}")
                    continue
                for field, cap in FIELD_MAX_TOKENS.items():
                    tokens = parsed[field]
                    if not isinstance(tokens, list):
                        problems.append(f"{where}.{field}: not a list")
                    elif len(tokens) == 0:
                        problems.append(
                            f"{where}.{field}: empty -> pack_padded_sequence will fail")
                    elif len(tokens) > cap:
                        problems.append(
                            f"{where}.{field}: {len(tokens)} tokens > cap {cap}")
                    elif not all(isinstance(t, str) for t in tokens):
                        problems.append(f"{where}.{field}: non-string token")
    return (not problems), problems, checked
