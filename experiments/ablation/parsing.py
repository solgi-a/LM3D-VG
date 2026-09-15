"""
Every parse cache the ablation feeds on, and the scorer for the target field. CPU.

    python experiments/ablation/parsing.py <command> --help

Commands: spacy, noparse, clip, corrupt, target-accuracy. The small-LM variant lives in
parsing_smalllm.py, kept apart because it is the only one needing torch and transformers.

Every variant emits the same schema through the same tokenizer, so the only thing
differing between arms is the parser. Token caps and the "not mentioned" fallback are
load-bearing: details.md §4.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

FIELDS = ("target", "adjectives", "neighbors")

#: Fixed by _transform_parsed in lib/dataset.py.
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

#: What the GPT-4o-mini pipeline writes for a field the description does not mention.
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



def load_parser(model="en_core_web_sm"):
    """Load and cache a spaCy pipeline.

    en_core_web_trf is more accurate but 30-50x slower, which over ~46k descriptions is
    the difference between a minute and most of an hour. NER and the lemmatizer are
    excluded for speed; nothing below reads .lemma_.
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



def _select_target(doc):
    """Head noun of the first genuine noun chunk.

    Skips existential/pronominal chunks, verbs that open the sentence, and chunks inside
    a spatial prepositional phrase -- those describe a neighbour, not the target.
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



def _adjective_text(doc, target, target_start):
    """Surface span covering the target's adjectival modifiers.

    The span rather than the individual tokens, so "dark brown wooden and leather chair"
    yields the conjunction too, as GPT's output does.
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



def _neighbor_text(doc, target):
    """Surface spans of the spatial prepositional phrases, joined with ", ".

    Nested PPs fold into their outermost parent, so they stay one phrase.
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



def empty_parse():
    """The record emitted when nothing could be extracted."""
    return {"target": NOT_MENTIONED,
            "adjectives": NOT_MENTIONED,
            "neighbors": NOT_MENTIONED}


def parse_doc(doc):
    """Apply the extraction rules to an already-parsed spaCy ``Doc``.

    Separate from parse_with_spacy so callers can use nlp.pipe over the whole dataset.
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
    """Parse one referring expression into the GPT-4o-mini output schema."""
    if not description or not description.strip():
        return empty_parse()

    if nlp is None:
        nlp = load_parser(model)
    return parse_doc(nlp(description.lower()))


# ---- spacy ----

_DOC_SPACY = """
Run the rule-based spaCy parser over ScanRefer and write the files lib/dataset.py reads.

    python experiments/ablation/parsing.py spacy --splits train val

Writes data_parsing/spacy_parsing{,_tokenized}/, mirroring the GPT-4o-mini layout exactly.
Then pass --parsing_folder spacy_parsing_tokenized to training; nothing else changes.
"""


def load_split(split, data_root):
    """Load ScanRefer, tolerating the repo-root copy of the val file."""
    candidates = [
        os.path.join(data_root, f"ScanRefer_filtered_{split}.json"),
        os.path.join(os.getcwd(), f"ScanRefer_filtered_{split}.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f), path
    raise FileNotFoundError(
        f"ScanRefer_filtered_{split}.json not found. Looked in:\n  " + "\n  ".join(candidates)
    )


def _nested():
    return defaultdict(lambda: defaultdict(dict))


def _plain(nested):
    return {s: {o: dict(a) for o, a in objs.items()} for s, objs in nested.items()}


def run_split(split, args):
    records, source = load_split(split, args.data_root)
    if args.limit:
        records = records[: args.limit]
    print(f"[{split}] {len(records)} descriptions from {source}")

    nlp = load_parser(args.model)

    raw_out, tok_out = _nested(), _nested()
    stats = {"records": 0}
    for field in FIELDS:
        stats[f"not_mentioned_{field}"] = 0
        stats[f"capped_{field}"] = 0

    t0 = time.time()
    # nlp.pipe is markedly faster than calling nlp() per description.
    texts = [r["description"].lower() if r["description"] else "" for r in records]
    docs = nlp.pipe(texts, batch_size=256)

    for i, (record, doc) in enumerate(zip(records, docs)):
        parsed = parse_doc(doc)
        tokenized = tokenize_record(parsed)

        for field in FIELDS:
            if parsed[field] == NOT_MENTIONED:
                stats[f"not_mentioned_{field}"] += 1
            if len(tokenized[field]) == FIELD_MAX_TOKENS[field]:
                stats[f"capped_{field}"] += 1

        scene_id, object_id, ann_id = record["scene_id"], record["object_id"], record["ann_id"]
        raw_out[scene_id][object_id][ann_id] = parsed
        tok_out[scene_id][object_id][ann_id] = tokenized
        stats["records"] += 1

        if (i + 1) % 5000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{len(records)}  ({rate:.0f} desc/s)")

    elapsed = time.time() - t0

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    raw_path = os.path.join(args.raw_dir, f"parsed_result_{split}.json")
    tok_path = os.path.join(args.output_dir, f"tokenized_parsed_result_{split}.json")
    with open(raw_path, "w") as f:
        json.dump(_plain(raw_out), f)
    with open(tok_path, "w") as f:
        json.dump(_plain(tok_out), f)

    size_mb = os.path.getsize(tok_path) / (1024 ** 2)
    print(f"[{split}] parsed {stats['records']} in {elapsed:.1f}s "
          f"({stats['records'] / max(elapsed, 1e-9):.0f} desc/s)")
    print(f"[{split}] -> {raw_path}")
    print(f"[{split}] -> {tok_path} ({size_mb:.1f} MB)")
    print(f"[{split}] not mentioned: " + "  ".join(
        f"{f}={stats[f'not_mentioned_{f}']}" for f in FIELDS))
    print(f"[{split}] hit token cap: " + "  ".join(
        f"{f}={stats[f'capped_{f}']}" for f in FIELDS))

    stats["elapsed_sec"] = round(elapsed, 2)
    stats["desc_per_sec"] = round(stats["records"] / max(elapsed, 1e-9), 1)
    stats["raw_output"] = raw_path
    stats["tokenized_output"] = tok_path

    with open(tok_path) as f:
        ok, problems, checked = validate_tokenized(json.load(f))
    if ok:
        print(f"[{split}] schema check passed over {checked} entries "
              f"(keys, non-empty, within caps, all strings)")
    else:
        print(f"\n[{split}] SCHEMA CHECK FAILED ({len(problems)} problems):")
        for problem in problems[:20]:
            print("  -", problem)

    return stats, ok


def _add_spacy_args(parser):
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--raw-dir", dest="raw_dir",
                        default=os.path.join("data_parsing", "spacy_parsing"))
    parser.add_argument("--output-dir", dest="output_dir",
                        default=os.path.join("data_parsing", "spacy_parsing_tokenized"))
    parser.add_argument("--model", default="en_core_web_sm",
                        help="spaCy model: en_core_web_sm (fast) or en_core_web_trf "
                             "(more accurate, ~30-50x slower)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Parse only the first N descriptions (smoke test).")
    return parser


def run_spacy(args):

    summary, ok = {}, True
    for split in args.splits:
        summary[split], split_ok = run_split(split, args)
        ok &= split_ok
        print()

    with open(os.path.join(args.output_dir, "parse_run_summary.json"), "w") as f:
        json.dump({"parser": "spacy", "model": args.model, "splits": summary}, f, indent=2)

    if not ok:
        return 1
    print("All splits parsed and schema-validated.")
    print("To use this parse in training, set in experiments/ablation/ablation_config.py:")
    print(f"    ABLATION.PARSING_FOLDER = {os.path.basename(args.output_dir)!r}")
    print("(the folder is resolved under data/scannet/)")
    return 0


# ---- noparse ----

_DOC_NOPARSE = """
Parser ablation, variant D -- no parser at all. CPU, seconds.

    python experiments/ablation/parsing.py noparse --splits train val

Writes a cache in which every annotation carries the same content-free placeholder in all
three fields. The architecture is unchanged -- A2F and TAF still run -- so the difference
against the other variants is attributable to the parser alone.

The placeholder is a token, not zeros: an empty field crashes pack_padded_sequence and no
token maps to a zero GloVe row. ``--encoding unk`` (default) writes ["unk"], which is what
lib/dataset.py already writes for "no parse"; ``--encoding not_mentioned`` writes the GPT
pipeline's convention instead. See details.md §4.
"""


#: Placeholder token list per --encoding. Both are length >= 1 by construction, which is
#: the constraint pack_padded_sequence imposes.
ENCODINGS = {
    "unk": ["unk"],
    "not_mentioned": list(NOT_MENTIONED_TOKENS),
}

#: Where each encoding is written by default, so both can coexist on disk.
DEFAULT_DIRS = {
    "unk": os.path.join("data_parsing", "noparse_tokenized"),
    "not_mentioned": os.path.join("data_parsing", "noparse_notmentioned_tokenized"),
}


def build_split(split, placeholder, data_root, limit=None):
    """Return the nested {scene: {object: {ann: parsed}}} dict for one split."""
    records, source = load_split(split, data_root)
    if limit:
        records = records[:limit]

    out = defaultdict(lambda: defaultdict(dict))
    for record in records:
        # A fresh list per field per record: json.dump would happily share one object,
        # but a shared mutable would be a trap for anything that post-processes this.
        out[record["scene_id"]][record["object_id"]][record["ann_id"]] = {
            field: list(placeholder) for field in FIELDS
        }
    return {s: {o: dict(a) for o, a in objs.items()} for s, objs in out.items()}, \
        len(records), source


def _add_noparse_args(parser):
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--encoding", default="unk", choices=sorted(ENCODINGS),
                        help="Placeholder for every field. Default 'unk' matches the "
                             "no-parse encoding already in lib/dataset.py:687-689.")
    parser.add_argument("--output-dir", dest="output_dir", default=None,
                        help="Defaults to data_parsing/noparse_tokenized for --encoding "
                             "unk, data_parsing/noparse_notmentioned_tokenized otherwise.")
    parser.add_argument("--both", action="store_true",
                        help="Write both encodings, each to its own default directory.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only the first N descriptions per split (smoke test).")
    return parser


def run_noparse(args):

    encodings = sorted(ENCODINGS) if args.both else [args.encoding]
    if args.both and args.output_dir:
        args._parser.error("--both writes two directories; --output-dir cannot be used with it.")

    ok = True
    for encoding in encodings:
        placeholder = ENCODINGS[encoding]
        output_dir = args.output_dir or DEFAULT_DIRS[encoding]
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n=== variant D, encoding={encoding!r} -> {placeholder} ===")
        summary = {}
        for split in args.splits:
            data, n, source = build_split(split, placeholder, args.data_root, args.limit)
            path = os.path.join(output_dir, f"tokenized_parsed_result_{split}.json")
            with open(path, "w") as f:
                json.dump(data, f)

            size_mb = os.path.getsize(path) / (1024 ** 2)
            print(f"[{split}] {n} descriptions from {os.path.basename(source)}")
            print(f"[{split}] -> {path} ({size_mb:.1f} MB)")

            with open(path) as f:
                passed, problems, checked = validate_tokenized(json.load(f))
            if passed:
                print(f"[{split}] schema check passed over {checked} entries "
                      f"(keys, non-empty, within caps {dict(FIELD_MAX_TOKENS)}, all strings)")
            else:
                ok = False
                print(f"[{split}] SCHEMA CHECK FAILED ({len(problems)} problems):")
                for problem in problems[:20]:
                    print("  -", problem)

            summary[split] = {"records": n, "checked": checked, "schema_ok": passed,
                              "output": path}

        meta_path = os.path.join(output_dir, "parse_run_summary.json")
        with open(meta_path, "w") as f:
            json.dump({
                "parser": "none (variant D)",
                "encoding": encoding,
                "placeholder_tokens": placeholder,
                "rationale": "pack_padded_sequence rejects length 0 and no GloVe token "
                             "maps to a zero row, so the minimal content-free field is a "
                             "single placeholder token.",
                "splits": summary,
            }, f, indent=2)
        print(f"[meta] -> {meta_path}")

        print("\nTo train this variant:")
        print(f"    python scripts/ScanRefer_train.py --parsing_folder {os.path.basename(output_dir)} \\")
        print("        --use_cached_scenes --tag ABL-PARSER-NONE --use_color --use_normal")
        print("or:  python experiments/ablation/train_arms.py run parser-none")

    if not ok:
        return 1
    return 0


# ---- clip ----

_DOC_CLIP = """
Enforce the 7 / 17 / 75 token caps on an existing tokenized parse cache.

    python experiments/ablation/parsing.py clip         --input  data_parsing/final_parsing_tokenized_llama         --output data_parsing/llama_parsing_tokenized_clipped

__getitem__ sets the sequence lengths unclipped, so a field longer than its cap hands
pack_padded_sequence a length beyond the sequence dimension and it raises mid-training.
The GPT caches are already clipped; the LLaMA cache is not. See details.md §4.
"""

def clip_split(in_dir, out_dir, split):
    in_path = os.path.join(in_dir, f"tokenized_parsed_result_{split}.json")
    if not os.path.isfile(in_path):
        print(f"[{split}] skipped, no such file: {in_path}")
        return None

    with open(in_path) as f:
        data = json.load(f)

    clipped = {field: 0 for field in FIELD_MAX_TOKENS}
    filled = {field: 0 for field in FIELD_MAX_TOKENS}
    total = 0
    out = {}

    for scene_id, objects in data.items():
        out[scene_id] = {}
        for object_id, anns in objects.items():
            out[scene_id][object_id] = {}
            for ann_id, parsed in anns.items():
                total += 1
                record = {}
                for field in FIELD_MAX_TOKENS:
                    tokens = parsed.get(field) or []
                    if len(tokens) > FIELD_MAX_TOKENS[field]:
                        clipped[field] += 1
                    elif len(tokens) == 0:
                        filled[field] += 1
                    record[field] = clip_tokens(tokens, field)
                out[scene_id][object_id][ann_id] = record

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"tokenized_parsed_result_{split}.json")
    with open(out_path, "w") as f:
        json.dump(out, f)

    print(f"[{split}] {total} annotations -> {out_path}")
    print(f"[{split}] clipped: " + "  ".join(f"{f}={clipped[f]}" for f in FIELD_MAX_TOKENS))
    if any(filled.values()):
        print(f"[{split}] empty fields filled with 'not mentioned': "
              + "  ".join(f"{f}={filled[f]}" for f in FIELD_MAX_TOKENS))

    ok, problems, checked = validate_tokenized(out, label=f"{split}:")
    if ok:
        print(f"[{split}] schema check passed over {checked} entries")
    else:
        print(f"[{split}] SCHEMA CHECK FAILED ({len(problems)} problems):")
        for problem in problems[:20]:
            print("  -", problem)
    return ok


def _add_clip_args(parser):
    parser.add_argument("--input", required=True,
                        help="Folder holding tokenized_parsed_result_{split}.json")
    parser.add_argument("--output", required=True,
                        help="Destination folder (must differ from --input)")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    return parser


def run_clip(args):

    if os.path.abspath(args.input) == os.path.abspath(args.output):
        args._parser.error("--output must differ from --input; this script never edits in place")

    ok = True
    for split in args.splits:
        result = clip_split(args.input, args.output, split)
        if result is not None:
            ok &= result
        print()

    if not ok:
        return 1
    print("Done. To use this cache set in experiments/ablation/ablation_config.py:")
    print(f"    ABLATION.PARSING_FOLDER = {os.path.basename(args.output.rstrip('/'))!r}")
    return 0


# ---- corrupt ----

_DOC_CORRUPT = """
Deliberately corrupt a parse cache, to measure whether parse errors propagate. CPU, seconds.

    python experiments/ablation/parsing.py corrupt --splits val --rates 0.10 0.25 0.50 --mode all

A seeded fraction of annotations is rewritten: ``swap`` gives the target a different
annotation's target, ``drop`` blanks all three fields, ``shuffle`` replaces neighbors, and
``all`` (the default) draws one of the three per annotation.

Output goes to ``<source>_corrupt_<mode>_<pct>/`` with a manifest naming which annotations
were touched, so parse_error_propagation.py can compare the corrupted subset against the
untouched one -- the global curve mixes them and dilutes the effect by (1 - rate).

Needs no training, only re-evaluation. Next step:

    python experiments/ablation/train_arms.py corruption-sweep
"""

MODES = ("swap", "drop", "shuffle", "all")

#: How many times to redraw a donor before giving up and leaving the entry unchanged.
_MAX_DONOR_TRIES = 12


def flat_keys(data):
    """Sorted [(scene_id, object_id, ann_id)] so sampling is reproducible across runs."""
    keys = []
    for scene_id, objects in data.items():
        for object_id, anns in objects.items():
            for ann_id in anns:
                keys.append((scene_id, object_id, ann_id))
    keys.sort()
    return keys


def _get(data, key):
    scene_id, object_id, ann_id = key
    return data[scene_id][object_id][ann_id]


def _draw_donor(data, keys, key, field, rng):
    """A different annotation's value for ``field``, guaranteed to differ from key's."""
    original = _get(data, key)[field]
    for _ in range(_MAX_DONOR_TRIES):
        donor_key = keys[rng.randrange(len(keys))]
        if donor_key == key:
            continue
        candidate = _get(data, donor_key)[field]
        if candidate != original:
            return candidate
    return None


def corrupt(data, rate, mode, seed):
    """Return (corrupted_copy, manifest_entries, stats). ``data`` is not modified."""
    rng = random.Random(seed)
    keys = flat_keys(data)
    num_target = int(round(rate * len(keys)))
    selected = rng.sample(keys, num_target) if num_target else []

    # Deep-ish copy: the field lists are replaced wholesale, never mutated in place, so
    # copying one level below the annotation dict is sufficient and much cheaper.
    out = {
        scene_id: {
            object_id: {ann_id: {f: list(parsed[f]) for f in FIELDS}
                        for ann_id, parsed in anns.items()}
            for object_id, anns in objects.items()
        }
        for scene_id, objects in data.items()
    }

    manifest, stats = [], {"selected": len(selected), "applied": 0, "no_op": 0}
    for m in ("swap", "drop", "shuffle"):
        stats[f"applied_{m}"] = 0

    for key in sorted(selected):
        applied = mode if mode != "all" else rng.choice(("swap", "drop", "shuffle"))
        scene_id, object_id, ann_id = key
        entry = out[scene_id][object_id][ann_id]
        before = {f: list(entry[f]) for f in FIELDS}

        if applied == "drop":
            for field in FIELDS:
                entry[field] = list(NOT_MENTIONED_TOKENS)
        else:
            field = "target" if applied == "swap" else "neighbors"
            donor = _draw_donor(data, keys, key, field, rng)
            if donor is None:
                stats["no_op"] += 1
                continue
            entry[field] = clip_tokens(list(donor), field)

        if all(entry[f] == before[f] for f in FIELDS):
            # Can happen for `drop` when the parse was already all "not mentioned".
            stats["no_op"] += 1
            continue

        stats["applied"] += 1
        stats[f"applied_{applied}"] += 1
        manifest.append({
            "scene_id": scene_id, "object_id": object_id, "ann_id": ann_id,
            "applied": applied,
            "before": before,
            "after": {f: list(entry[f]) for f in FIELDS},
        })

    stats["total"] = len(keys)
    stats["effective_rate"] = round(stats["applied"] / max(len(keys), 1), 6)
    return out, manifest, stats


def _add_corrupt_args(parser):
    parser.add_argument("--splits", nargs="+", default=["val"],
                        help="Only 'val' is needed for the evaluation-only experiment.")
    parser.add_argument("--source", default="final_parsing_tokenized",
                        help="Parse cache to corrupt, under data_parsing/.")
    parser.add_argument("--parsing-root", dest="parsing_root", default="data_parsing")
    parser.add_argument("--rates", nargs="+", type=float, default=[0.10, 0.25, 0.50])
    parser.add_argument("--mode", default="all", choices=MODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="Keep only the first N annotations (smoke test).")
    parser.add_argument("--out-root", dest="out_root", default=None,
                        help="Defaults to --parsing-root.")
    return parser


def run_corrupt(args):

    out_root = args.out_root or args.parsing_root
    ok = True

    for rate in args.rates:
        if not 0.0 <= rate <= 1.0:
            args._parser.error(f"--rates must be in [0, 1]; got {rate}")
        pct = int(round(rate * 100))
        out_dir = os.path.join(out_root, f"{args.source}_corrupt_{args.mode}_{pct:02d}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n=== corruption {pct}% | mode={args.mode} | seed={args.seed} ===")
        manifest_all, stats_all = {}, {}

        for split in args.splits:
            src = os.path.join(args.parsing_root, args.source,
                               f"tokenized_parsed_result_{split}.json")
            if not os.path.isfile(src):
                print(f"ERROR: source cache not found: {src}")
                return 1
            with open(src) as f:
                data = json.load(f)

            if args.limit:
                keep = set(flat_keys(data)[: args.limit])
                data = {
                    s: {o: {a: p for a, p in anns.items() if (s, o, a) in keep}
                        for o, anns in objs.items()}
                    for s, objs in data.items()
                }
                data = {s: {o: a for o, a in objs.items() if a}
                        for s, objs in data.items()}
                data = {s: o for s, o in data.items() if o}

            # Per-split seed offset, so val and train are not corrupted in lockstep.
            corrupted, manifest, stats = corrupt(
                data, rate, args.mode, args.seed + hash(split) % 1000)

            path = os.path.join(out_dir, f"tokenized_parsed_result_{split}.json")
            with open(path, "w") as f:
                json.dump(corrupted, f)

            passed, problems, checked = validate_tokenized(corrupted)
            status = "passed" if passed else f"FAILED ({len(problems)} problems)"
            print(f"[{split}] {stats['applied']}/{stats['total']} corrupted "
                  f"(effective {100 * stats['effective_rate']:.2f}%, "
                  f"no-op {stats['no_op']})")
            print(f"[{split}]   swap={stats['applied_swap']}  drop={stats['applied_drop']}"
                  f"  shuffle={stats['applied_shuffle']}")
            print(f"[{split}] -> {path}")
            print(f"[{split}] schema check {status} over {checked} entries")
            if not passed:
                ok = False
                for problem in problems[:10]:
                    print("  -", problem)

            manifest_all[split] = manifest
            stats_all[split] = stats

        manifest_path = os.path.join(out_dir, "corruption_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump({
                "source": args.source,
                "mode": args.mode,
                "requested_rate": rate,
                "seed": args.seed,
                "stats": stats_all,
                "corrupted": manifest_all,
            }, f, indent=2)
        print(f"[manifest] -> {manifest_path}")

    print("\nEvaluate each level against the trained checkpoint (GPU):")
    print("    python experiments/ablation/train_arms.py corruption-sweep")
    print("then aggregate (CPU):")
    print("    python experiments/analysis.py error-propagation --run-dir outputs/<folder>/corruption")
    return 0 if ok else 1


# ---- target-accuracy ----

_DOC_TARGET_ACCURACY = """
Score a parser's target extraction against ScanRefer's ``object_name`` field.

    # spaCy (variant B) -- parses on the fly
    python experiments/ablation/parsing.py target-accuracy --splits train --parser spacy

    # score an existing cache
    python experiments/ablation/parsing.py target-accuracy --splits train         --parsed-dir data_parsing/final_parsing_tokenized --tag gpt4o-mini

object_name (e.g. "trash_can") is the annotated class of the referred object, so it is free
ground truth for the target slot. Three criteria are reported: ``exact`` (normalised
strings identical), ``substring`` (credits "trash can" vs "can") and ``fuzzy`` (token
overlap or edit distance <= 1 on the head word, crediting plurals and spelling drift).

A sample file for manual review is written alongside, mixing random draws with failures.
"""

def normalise(text):
    return " ".join(str(text).replace("_", " ").lower().split())


def levenshtein(a, b, cap=2):
    """Edit distance, short-circuited once it provably exceeds ``cap``."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        if min(current) > cap:
            return cap + 1
        previous = current
    return previous[-1]


def match_kinds(predicted, truth):
    """Return which of (exact, substring, fuzzy) hold for this pair."""
    pred, gold = normalise(predicted), normalise(truth)
    if not pred or pred == " ".join(NOT_MENTIONED_TOKENS):
        return {"exact": False, "substring": False, "fuzzy": False}

    exact = pred == gold
    substring = exact or pred in gold or gold in pred

    pred_tokens, gold_tokens = set(pred.split()), set(gold.split())
    overlap = bool(pred_tokens & gold_tokens)
    head_close = levenshtein(pred.split()[-1], gold.split()[-1]) <= 1
    fuzzy = substring or overlap or head_close

    return {"exact": exact, "substring": substring, "fuzzy": fuzzy}


def _iter_parsed(records, preloaded, nlp):
    """Yield (record, parsed_dict_of_token_lists) for each annotation."""
    if preloaded is not None:
        for record in records:
            try:
                yield record, preloaded[record["scene_id"]][record["object_id"]][record["ann_id"]]
            except KeyError:
                continue
        return

    from experiments.ablation.parsing import parse_doc
    from experiments.ablation.parsing import tokenize_record

    texts = [r["description"].lower() if r["description"] else "" for r in records]
    for record, doc in zip(records, nlp.pipe(texts, batch_size=256)):
        yield record, tokenize_record(parse_doc(doc))


def _add_target_accuracy_args(parser):
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--parser", default="spacy", choices=["spacy"],
                        help="Parser to run when --parsed-dir is not given.")
    parser.add_argument("--model", default="en_core_web_sm")
    parser.add_argument("--parsed-dir", dest="parsed_dir", default=None,
                        help="Score an existing tokenized_parsed_result_*.json instead of "
                             "re-parsing. Use this for the GPT-4o-mini and LLaMA caches.")
    parser.add_argument("--tag", default=None,
                        help="Name for this parser in the report filenames.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-samples", dest="num_samples", type=int, default=30,
                        help="How many parsed examples to dump for manual review.")
    parser.add_argument("--report-dir", dest="report_dir", default="outputs/parser_eval")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def run_target_accuracy(args):

    tag = args.tag or (os.path.basename(args.parsed_dir.rstrip("/")) if args.parsed_dir
                       else args.parser)

    random.seed(args.seed)
    os.makedirs(args.report_dir, exist_ok=True)
    report = {"tag": tag, "parsed_dir": args.parsed_dir, "model": args.model, "splits": {}}

    nlp = None
    if args.parsed_dir is None:
        from experiments.ablation.parsing import load_parser
        nlp = load_parser(args.model)

    for split in args.splits:
        records, source = load_split(split, args.data_root)
        if args.limit:
            records = records[: args.limit]

        preloaded = None
        if args.parsed_dir:
            path = os.path.join(args.parsed_dir, f"tokenized_parsed_result_{split}.json")
            with open(path) as f:
                preloaded = json.load(f)

        counts = Counter()
        total = 0
        rows, failures = [], []
        confusion = Counter()

        for record, parsed in _iter_parsed(records, preloaded, nlp):
            predicted = " ".join(parsed["target"])
            kinds = match_kinds(predicted, record["object_name"])

            total += 1
            for kind, hit in kinds.items():
                counts[kind] += int(hit)
            if parsed["target"] == NOT_MENTIONED_TOKENS:
                counts["no_target"] += 1

            row = {
                "scene_id": record["scene_id"],
                "object_id": record["object_id"],
                "ann_id": record["ann_id"],
                "description": record["description"],
                "object_name (ground truth)": record["object_name"],
                "parsed": parsed,
                "match": kinds,
            }
            rows.append(row)
            if not kinds["fuzzy"]:
                failures.append(row)
                confusion[(normalise(record["object_name"]), normalise(predicted))] += 1

        pct = {k: 100.0 * counts[k] / max(total, 1) for k in ("exact", "substring", "fuzzy")}
        print(f"\n=== {tag} | {split}  ({total} descriptions from {os.path.basename(source)}) ===")
        print(f"  exact match      : {counts['exact']:>6} / {total}  = {pct['exact']:.2f}%")
        print(f"  substring match  : {counts['substring']:>6} / {total}  = {pct['substring']:.2f}%")
        print(f"  fuzzy match      : {counts['fuzzy']:>6} / {total}  = {pct['fuzzy']:.2f}%")
        print(f"  no target found  : {counts['no_target']:>6} / {total}  "
              f"= {100.0 * counts['no_target'] / max(total, 1):.2f}%")

        if confusion:
            print("\n  most frequent target errors (ground truth -> predicted):")
            for (gold, pred), n in confusion.most_common(10):
                print(f"    {n:>5}x  {gold!r} -> {pred!r}")

        # Sample file for manual review: random draws plus failures, because a
        # qualitative section built only from successes is not a fair evaluation.
        num_fail = min(len(failures), max(1, args.num_samples // 3)) if failures else 0
        num_rand = max(0, args.num_samples - num_fail)
        samples = random.sample(rows, min(num_rand, len(rows)))
        if num_fail:
            samples += random.sample(failures, num_fail)

        sample_path = os.path.join(args.report_dir, f"samples_{split}_{tag}.json")
        with open(sample_path, "w") as f:
            json.dump({
                "note": "Random draws followed by failure cases, for manual qualitative review.",
                "tag": tag,
                "num_random": len(samples) - num_fail,
                "num_failures": num_fail,
                "samples": samples,
            }, f, indent=2)

        txt_path = os.path.join(args.report_dir, f"samples_{split}_{tag}.txt")
        with open(txt_path, "w") as f:
            f.write(f"Parser samples - {tag} - split={split}\n")
            f.write("=" * 88 + "\n\n")
            for i, row in enumerate(samples, 1):
                verdict = "OK " if row["match"]["fuzzy"] else "MISS"
                f.write(f"[{i:02d}] {verdict}  {row['scene_id']} "
                        f"obj={row['object_id']} ann={row['ann_id']}\n")
                f.write(f"  description : {row['description']}\n")
                f.write(f"  object_name : {row['object_name (ground truth)']}\n")
                f.write(f"  target      : {row['parsed']['target']}\n")
                f.write(f"  adjectives  : {row['parsed']['adjectives']}\n")
                f.write(f"  neighbors   : {row['parsed']['neighbors']}\n\n")

        print(f"\n  wrote {sample_path}")
        print(f"  wrote {txt_path}")

        report["splits"][split] = {
            "total": total,
            "exact": counts["exact"], "exact_pct": round(pct["exact"], 2),
            "substring": counts["substring"], "substring_pct": round(pct["substring"], 2),
            "fuzzy": counts["fuzzy"], "fuzzy_pct": round(pct["fuzzy"], 2),
            "no_target": counts["no_target"],
            "top_errors": [
                {"ground_truth": g, "predicted": p, "count": n}
                for (g, p), n in confusion.most_common(20)
            ],
        }

    report_path = os.path.join(args.report_dir, f"target_accuracy_{tag}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {report_path}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python experiments/ablation/parsing.py",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    for key, doc, adder, runner in (
            ("spacy", _DOC_SPACY, _add_spacy_args, run_spacy),
            ("noparse", _DOC_NOPARSE, _add_noparse_args, run_noparse),
            ("clip", _DOC_CLIP, _add_clip_args, run_clip),
            ("corrupt", _DOC_CORRUPT, _add_corrupt_args, run_corrupt),
            ("target-accuracy", _DOC_TARGET_ACCURACY, _add_target_accuracy_args,
             run_target_accuracy)):
        p = sub.add_parser(key, help=doc.strip().splitlines()[0], description=doc,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        adder(p)
        p.set_defaults(func=runner, _parser=p)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
