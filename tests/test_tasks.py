from schematax.analyze import paired_ci
from schematax.tasks import _json, _spans, agnews_score, conll_score, gsm8k_score

GSM = {"gold": 18.0}


def test_gsm8k_natural():
    assert gsm8k_score(GSM, "natural", "So she makes 9 * 2 = 18.\nThe answer is 18.") == (True, 1.0)
    assert gsm8k_score(GSM, "natural", "The answer is $1,018.") == (True, 0.0)
    assert gsm8k_score(GSM, "natural", "I don't know") == (False, 0.0)


def test_gsm8k_json():
    assert gsm8k_score(GSM, "json_constrained", '{"reasoning": "9 * 2", "answer": 18}') == (True, 1.0)
    assert gsm8k_score(GSM, "json_free", '```json\n{"answer": "18"}\n```') == (True, 1.0)
    assert gsm8k_score(GSM, "json_free", '{"reasoning": "no answer"}') == (False, 0.0)
    assert gsm8k_score({"gold": 1.0}, "json_free", '{"answer": true}') == (False, 0.0)


def test_json_extraction():
    assert _json('Here you go: {"label": "Sports"} hope that helps') == {"label": "Sports"}
    assert _json("not json") is None


def test_agnews():
    doc = {"gold": "Sci/Tech"}
    assert agnews_score(doc, "natural", "Sci/Tech") == (True, 1.0)
    assert agnews_score(doc, "natural", "World or Sports") == (False, 0.0)
    assert agnews_score(doc, "json_constrained", '{"label": "Sci/Tech"}') == (True, 1.0)
    assert agnews_score(doc, "json_constrained", '{"label": "Tech"}') == (False, 0.0)


def test_conll_spans():
    tokens = ["John", "Smith", "visited", "New", "York", "."]
    tags = [1, 2, 0, 5, 6, 0]
    assert _spans(tokens, tags) == {("persons", "John Smith"), ("locations", "New York")}


def test_conll_score():
    doc = {"gold": [["locations", "New York"], ["persons", "John Smith"]]}
    natural = "persons: John Smith\norganizations:\nlocations: New York\nmisc:"
    assert conll_score(doc, "natural", natural) == (True, 1.0)
    ok, f1 = conll_score(doc, "json_free", '{"persons": ["John Smith"], "locations": []}')
    assert ok and abs(f1 - 2 / 3) < 1e-9


def test_paired_ci():
    a = {i: {"score": 1.0} for i in range(10)}
    b = {i: {"score": float(i % 2)} for i in range(10)}
    mean, lo, hi = paired_ci(a, b)
    assert mean == 0.5 and lo <= mean <= hi
