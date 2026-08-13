
import re

FIELD_MAX_TOKENS = {"target": 7, "adjectives": 17, "neighbors": 75}

NOT_MENTIONED_TOKENS = ["not", "mentioned"]

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")


def tokenize(text):
    if text is None:
        return []
    return _TOKEN_RE.findall(str(text).lower())


def tokenize_field(text, field):
    tokens = tokenize(text)
    if not tokens:
        tokens = list(NOT_MENTIONED_TOKENS)
    return tokens[: FIELD_MAX_TOKENS[field]]


def tokenize_record(parsed):
    return {field: tokenize_field(parsed.get(field), field)
            for field in FIELD_MAX_TOKENS}


def clip_tokens(tokens, field):
    if not tokens:
        return list(NOT_MENTIONED_TOKENS)
    return list(tokens)[: FIELD_MAX_TOKENS[field]]


def validate_tokenized(data, label=""):
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
