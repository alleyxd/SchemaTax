"""Tasks, prompt conditions and scoring for the schema-tax study.

Every task is asked under several output conditions:

    natural          plain text answer in the task's usual format, parsed with a regex
    json_free        JSON requested and the schema shown in the prompt, decoding unconstrained
    json_constrained JSON schema shown in the prompt and enforced by constrained decoding
    json_blind       JSON schema enforced by constrained decoding but not shown to the model

GSM8K adds two field-order variants of the constrained condition (answer before reasoning,
and answer only), since the order of keys decides whether the model reasons before committing.
"""
import json
import random
import re

from datasets import load_dataset

AG_LABELS = ["World", "Sports", "Business", "Sci/Tech"]
CONLL_TYPES = {"PER": "persons", "ORG": "organizations", "LOC": "locations", "MISC": "misc"}
CONLL_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def _schema(props: dict, order: list[str]) -> dict:
    return {"type": "object", "properties": {k: props[k] for k in order}, "required": order,
            "additionalProperties": False}


# ---------------------------------------------------------------- GSM8K

GSM_PROPS = {"reasoning": {"type": "string"}, "answer": {"type": "number"}}
GSM_SCHEMAS = {
    "reason_first": _schema(GSM_PROPS, ["reasoning", "answer"]),
    "answer_first": _schema(GSM_PROPS, ["answer", "reasoning"]),
    "answer_only": _schema(GSM_PROPS, ["answer"]),
}


def gsm8k(n: int, seed: int = 0) -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = random.Random(seed).sample(range(len(ds)), n)
    return [{"id": i, "question": ds[i]["question"],
             "gold": float(ds[i]["answer"].split("####")[-1].strip().replace(",", ""))} for i in idx]


def gsm8k_requests(doc: dict) -> dict:
    q = doc["question"]
    base = "Solve this math word problem.\n\n" + q
    reqs = {"natural": (base + "\n\nThink step by step, then finish with a line 'The answer is N.'", None)}
    for name, schema in GSM_SCHEMAS.items():
        shown = (base + "\n\nRespond with a JSON object that matches this schema:\n" + json.dumps(schema))
        if name == "reason_first":
            reqs["json_free"] = (shown, None)
            reqs["json_constrained"] = (shown, schema)
            reqs["json_blind"] = (base + "\n\nRespond in JSON.", schema)
        else:
            reqs[f"json_constrained_{name}"] = (shown, schema)
    return reqs


def _number(text: str) -> float | None:
    m = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def gsm8k_score(doc: dict, cond: str, text: str) -> tuple[bool, float]:
    """Returns (parsed_ok, score)."""
    if cond == "natural":
        m = re.search(r"answer is\s*\$?(-?[\d,]*\.?\d+)", text, re.I)
        val = _number(m.group(1)) if m else _number(text)
        return val is not None, float(val is not None and abs(val - doc["gold"]) < 1e-6)
    obj = _json(text)
    if not isinstance(obj, dict) or "answer" not in obj:
        return False, 0.0
    val = obj["answer"]
    if isinstance(val, str):
        val = _number(val)
    ok = isinstance(val, (int, float)) and not isinstance(val, bool)  # json true would otherwise count as 1
    return ok, float(ok and abs(float(val) - doc["gold"]) < 1e-6)


# ---------------------------------------------------------------- AG News

AG_SCHEMA = _schema({"label": {"type": "string", "enum": AG_LABELS}}, ["label"])


def agnews(n: int, seed: int = 0) -> list[dict]:
    ds = load_dataset("fancyzhx/ag_news", split="test")
    idx = random.Random(seed).sample(range(len(ds)), n)
    return [{"id": i, "text": ds[i]["text"], "gold": AG_LABELS[ds[i]["label"]]} for i in idx]


def agnews_requests(doc: dict) -> dict:
    base = ("Classify the topic of this news article as one of: World, Sports, Business, Sci/Tech.\n\n"
            "Article: " + doc["text"])
    shown = base + "\n\nRespond with a JSON object that matches this schema:\n" + json.dumps(AG_SCHEMA)
    return {
        "natural": (base + "\n\nAnswer with the label only.", None),
        "json_free": (shown, None),
        "json_constrained": (shown, AG_SCHEMA),
        "json_blind": (base + "\n\nRespond in JSON.", AG_SCHEMA),
    }


def agnews_score(doc: dict, cond: str, text: str) -> tuple[bool, float]:
    if cond == "natural":
        t = text.strip().lower()
        hits = [lab for lab in AG_LABELS if lab.lower() in t or (lab == "Sci/Tech" and "sci" in t)]
        pred = hits[0] if len(hits) == 1 else None
    else:
        obj = _json(text)
        pred = obj.get("label") if isinstance(obj, dict) else None
        pred = pred if pred in AG_LABELS else None
    return pred is not None, float(pred == doc["gold"])


# ---------------------------------------------------------------- CoNLL-2003 NER

NER_PROPS = {v: {"type": "array", "items": {"type": "string"}} for v in CONLL_TYPES.values()}
NER_SCHEMA = _schema(NER_PROPS, list(CONLL_TYPES.values()))


def _spans(tokens: list[str], tags: list[int]) -> set[tuple[str, str]]:
    out, cur, typ = set(), [], None
    for tok, t in zip(tokens, tags):
        tag = CONLL_TAGS[t]
        if tag.startswith("B-") or (tag.startswith("I-") and typ != tag[2:]):
            if cur:
                out.add((CONLL_TYPES[typ], " ".join(cur)))
            cur, typ = [tok], tag[2:]
        elif tag.startswith("I-"):
            cur.append(tok)
        else:
            if cur:
                out.add((CONLL_TYPES[typ], " ".join(cur)))
            cur, typ = [], None
    if cur:
        out.add((CONLL_TYPES[typ], " ".join(cur)))
    return out


def conll(n: int, seed: int = 0) -> list[dict]:
    ds = load_dataset("eriktks/conll2003", split="test", revision="refs/convert/parquet")
    pool = [i for i in range(len(ds)) if len(ds[i]["tokens"]) >= 6 and any(ds[i]["ner_tags"])]
    idx = random.Random(seed).sample(pool, n)
    return [{"id": i, "text": " ".join(ds[i]["tokens"]),
             "gold": sorted(_spans(ds[i]["tokens"], ds[i]["ner_tags"]))} for i in idx]


def conll_requests(doc: dict) -> dict:
    base = ("Extract the named entities from this sentence, grouped as persons, organizations, locations "
            "and misc (nationalities, events, other names). Copy each entity exactly as written.\n\n"
            "Sentence: " + doc["text"])
    shown = base + "\n\nRespond with a JSON object that matches this schema:\n" + json.dumps(NER_SCHEMA)
    natural = base + ("\n\nWrite four lines, each starting with the group name and a colon, and the "
                      "entities separated by semicolons, for example:\npersons: A; B\norganizations:\n"
                      "locations: C\nmisc:")
    return {
        "natural": (natural, None),
        "json_free": (shown, None),
        "json_constrained": (shown, NER_SCHEMA),
        "json_blind": (base + "\n\nRespond in JSON.", NER_SCHEMA),
    }


def conll_score(doc: dict, cond: str, text: str) -> tuple[bool, float]:
    gold = {tuple(x) for x in doc["gold"]}
    pred = set()
    if cond == "natural":
        ok = False
        for line in text.splitlines():
            m = re.match(r"\s*\**\s*(persons|organizations|locations|misc)\s*\**\s*:(.*)", line, re.I)
            if m:
                ok = True
                pred |= {(m.group(1).lower(), e.strip()) for e in m.group(2).split(";") if e.strip()}
    else:
        obj = _json(text)
        ok = isinstance(obj, dict)
        if ok:
            for key in CONLL_TYPES.values():
                vals = obj.get(key) or []
                if isinstance(vals, list):
                    pred |= {(key, str(v).strip()) for v in vals if str(v).strip()}
    if not gold and not pred:
        return ok, 1.0
    tp = len(gold & pred)
    f1 = 2 * tp / (len(gold) + len(pred)) if (gold or pred) else 0.0
    return ok, f1


# ---------------------------------------------------------------- shared

def _json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


TASKS = {
    "gsm8k": (gsm8k, gsm8k_requests, gsm8k_score, 1024),
    "agnews": (agnews, agnews_requests, agnews_score, 64),
    "conll": (conll, conll_requests, conll_score, 256),
}
