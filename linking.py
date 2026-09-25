from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

RELATIONS = ["equals_total", "equals_opening_balance", "equals_movement", "breakdown_component", "unrelated"]
# method A: label similarity + amount match + period match
WEIGHTS_A = {"semantic": 0.5, "value": 0.3, "period": 0.2}
VALUE_TOLERANCE = 1

LLM_INSTRUCTIONS = """You link one line item of a summary financial statement to the rows of a note
in a Turkish audit report (KAP). Amounts are in TL. For EVERY candidate row choose one relation:
- equals_total: the row reports the same balance as the item at the same date
  (closing balance, net book value, or the total of a table).
- equals_opening_balance: the row is the opening balance of the following period and equals
  the item's balance at the end of its period.
- equals_movement: the row is the same flow (income, expense, gain, loss) for the same period.
- breakdown_component: the row is one part of a breakdown whose parts add up to the item.
- unrelated: anything else, including rows of the right table but for another period.
The item is either a balance at a date (balance sheet) or a flow over a period (income
statement); a balance is never "equals_movement" and a flow is never "equals_total".
The three equals_* relations require the same amount (sign may differ); rows whose amounts
only explain how a balance changed during the period are "unrelated" to that balance.
Use labels, column headers, periods and amounts; labels may contain OCR errors.
Return the column index ("col") of the value you used, or -1 for unrelated rows, a confidence
between 0 and 1, and a short reason."""

LLM_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["decisions"],
    "properties": {"decisions": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["row_id", "relation", "col", "confidence", "reason"],
        "properties": {"row_id": {"type": "string"}, "relation": {"type": "string", "enum": RELATIONS},
                       "col": {"type": "integer"}, "confidence": {"type": "number"},
                       "reason": {"type": "string"}}}}},
}


def summary_items(summary, note_number):
    items = []
    for t in summary["tables"]:
        periods = {c["col_id"]: c["period"] for c in t["columns"]}
        for r in t["rows"]:
            if note_number not in r["note_refs"]:
                continue
            for v in r["values"]:
                if v["kind"] in ("number", "zero"):
                    items.append({"item_id": f"{r['row_id']}:{v['col_id']}", "summary_row": r["row_id"],
                                  "col": v["col_id"], "label": r["label"], "statement": t["statement_type"],
                                  "period": periods[v["col_id"]], "value": v["value"]})
    return items


def value_match(a, b):
    if a is None or b is None or isinstance(b, list):
        return 0.0
    if abs(a - b) <= VALUE_TOLERANCE:
        return 1.0
    return 0.8 if abs(abs(a) - abs(b)) <= VALUE_TOLERANCE else 0.0


# 1 same date, 0.8 opening balance of the next period, 0.6 same year
def period_match(item_period, value_period):
    end = item_period["end"]
    if value_period["date"]:
        if value_period["date"] == end:
            return 1.0
        next_day = (date.fromisoformat(end) + timedelta(days=1)).isoformat()
        return 0.8 if item_period["type"] == "instant" and value_period["date"] == next_day else 0.0
    if value_period["year"]:
        return 0.6 if value_period["year"] == int(end[:4]) else 0.0
    return 0.3


def candidates_for(item, note_rows):
    cands = []
    for r in note_rows["rows"]:
        options = []
        for v in r["values"]:
            pm = period_match(item["period"], v["period"])
            if pm > 0 and v["kind"] in ("number", "zero"):
                vm = value_match(item["value"], v["value"])
                options.append((vm, pm, "toplam" in v["header"].lower(), v["col"], v))
        if options:
            vm, pm, _, col, v = max(options, key=lambda o: o[:4])
            cands.append({"item_id": item["item_id"], "note_row": r["row_id"], "block_id": r["block_id"],
                          "label": r["label"], "col": col, "header": v["header"], "value": v["value"],
                          "period": v["period"], "value_match": vm, "period_match": pm})
    return cands


def score_method_a(items, cands, note_rows, model_name):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    title = note_rows["title"] or ""
    item_text = {i["item_id"]: i["label"] for i in items}
    cand_text = [f"{title} - {c['label'] or 'toplam'} - {c['header']}" for c in cands]
    emb_items = {k: e for k, e in zip(item_text, model.encode(list(item_text.values()), normalize_embeddings=True))}
    emb_cands = model.encode(cand_text, normalize_embeddings=True)
    for c, e in zip(cands, emb_cands):
        c["semantic"] = round(max(0.0, float(np.dot(emb_items[c["item_id"]], e))), 4)
        c["score_a"] = round(WEIGHTS_A["semantic"] * c["semantic"] + WEIGHTS_A["value"] * c["value_match"]
                             + WEIGHTS_A["period"] * c["period_match"], 4)


def links_method_a(cands, note_rows, threshold):
    links = [dict(link_of(c), relation=rule_relation(c), score=c["score_a"], method="A")
             for c in cands if c["score_a"] >= threshold]
    return links + breakdown_links(links, note_rows)


def rule_relation(c):
    if c["period_match"] == 0.8 or "açılış" in c["label"].lower():
        return "equals_opening_balance"
    if c["period"]["source"] == "block":
        return "equals_movement"
    return "equals_total"


# parts at the same date that add up to a linked total; movement rows are not parts
def breakdown_links(links, note_rows):
    rows = {r["row_id"]: r for r in note_rows["rows"]}
    blocks = {b["block_id"]: b["row_ids"] for b in note_rows["blocks"]}
    extra = []
    for ln in links:
        ids = blocks[rows[ln["note_row"]]["block_id"]]
        if ln["relation"] != "equals_total" or ids[-1] != ln["note_row"] or len(ids) < 3:
            continue
        total_date = rows[ln["note_row"]]["values"][ln["col"]]["period"]["date"]
        cells = [rows[i]["values"][ln["col"]] if ln["col"] < len(rows[i]["values"]) else None for i in ids[:-1]]
        parts = [c["value"] if c and c["period"]["date"] == total_date else None for c in cells]
        if total_date and all(isinstance(p, (int, float)) for p in parts) \
                and abs(sum(parts) - ln["value"]) <= VALUE_TOLERANCE:
            extra += [{"item_id": ln["item_id"], "note_row": i, "col": ln["col"], "value": p,
                       "relation": "breakdown_component", "score": ln["score"], "method": ln["method"]}
                      for i, p in zip(ids[:-1], parts)]
    return extra


def link_of(c):
    return {"item_id": c["item_id"], "note_row": c["note_row"], "col": c["col"], "value": c["value"]}


# best-F1 cut on the hand-labelled control links
def fit_threshold(cands, gold, default):
    direct = {(g["item_id"], g["note_row"]) for g in gold if g["relation"] != "breakdown_component"}
    if not direct:
        return default, "config_default"
    scored = sorted({c["score_a"] for c in cands})
    best = (-1, default)
    for lo, hi in zip(scored, scored[1:]):
        t = (lo + hi) / 2
        pred = {(c["item_id"], c["note_row"]) for c in cands if c["score_a"] >= t}
        tp = len(pred & direct)
        f1 = 2 * tp / (len(pred) + len(direct)) if pred else 0
        if f1 > best[0]:
            best = (f1, round(t, 4))
    return best[1], "control_examples"


def run_method_b(items, cands, note_rows, model_name, cache_dir, fallback_links, max_candidates=30):
    links, fallback_items, calls = [], [], {"api": 0, "cached": 0}
    for item in items:
        mine = sorted([c for c in cands if c["item_id"] == item["item_id"]],
                      key=lambda c: c.get("score_a", 0), reverse=True)[:max_candidates]
        rows = {r["row_id"]: r for r in note_rows["rows"]}
        payload = {
            "summary_item": {"statement": item["statement"], "label": item["label"], "value": item["value"],
                             "kind": "balance at a date" if item["period"]["type"] == "instant" else "flow over a period",
                             "period": item["period"]["end"] if item["period"]["type"] == "instant"
                             else f"{item['period']['start']} - {item['period']['end']}"},
            "note": {"number": note_rows["note_number"], "title": note_rows["title"]},
            "candidates": [{"row_id": c["note_row"], "label": c["label"] or "(unlabelled total row)",
                            "values": [{"col": v["col"], "column": v["header"], "value": v["value"],
                                        "period": v["period"]["date"] or v["period"]["year"]}
                                       for v in rows[c["note_row"]]["values"] if v["kind"] in ("number", "zero")]}
                           for c in mine],
        }
        try:
            answer, cached = ask_llm(model_name, json.dumps(payload, ensure_ascii=False), cache_dir)
            calls["cached" if cached else "api"] += 1
            valid = {c["note_row"] for c in mine}
            for d in answer["decisions"]:
                if d["relation"] == "unrelated" or d["row_id"] not in valid:
                    continue
                values = rows[d["row_id"]]["values"]
                col = d["col"] if 0 <= d["col"] < len(values) else 0
                links.append({"item_id": item["item_id"], "note_row": d["row_id"], "col": col,
                              "value": values[col]["value"], "relation": d["relation"],
                              "score": round(min(max(d["confidence"], 0.0), 1.0), 4), "method": "B",
                              "reason": d["reason"]})
        except Exception as e:  # no key, API error, bad answer -> method A
            fallback_items.append({"item_id": item["item_id"], "error": f"{type(e).__name__}: {e}"[:200]})
            links += [dict(ln, method="A_fallback") for ln in fallback_links if ln["item_id"] == item["item_id"]]
    return links, fallback_items, calls


def ask_llm(model_name, user_text, cache_dir):
    key = hashlib.sha1(f"{model_name}\n{LLM_INSTRUCTIONS}\n{user_text}".encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), True
    from openai import OpenAI
    response = OpenAI().chat.completions.create(
        model=model_name, temperature=0,
        messages=[{"role": "system", "content": LLM_INSTRUCTIONS}, {"role": "user", "content": user_text}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "link_decisions", "strict": True, "schema": LLM_SCHEMA}},
    )
    answer = json.loads(response.choices[0].message.content)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(answer, ensure_ascii=False, indent=2), encoding="utf-8")
    return answer, False


def compare(links_a, links_b):
    a = {(ln["item_id"], ln["note_row"]): ln["relation"] for ln in links_a}
    b = {(ln["item_id"], ln["note_row"]): ln["relation"] for ln in links_b}
    fmt = lambda k: f"{k[0]} -> {k[1]}"
    return {
        "both_same_relation": sorted(fmt(k) for k in a.keys() & b.keys() if a[k] == b[k]),
        "both_different_relation": sorted(f"{fmt(k)} (A: {a[k]}, B: {b[k]})" for k in a.keys() & b.keys() if a[k] != b[k]),
        "only_a": sorted(f"{fmt(k)} ({a[k]})" for k in a.keys() - b.keys()),
        "only_b": sorted(f"{fmt(k)} ({b[k]})" for k in b.keys() - a.keys()),
    }


def evaluate(links, gold):
    pred = {(ln["item_id"], ln["note_row"]): ln["relation"] for ln in links}
    true = {(g["item_id"], g["note_row"]): g["relation"] for g in gold}
    tp = pred.keys() & true.keys()
    precision = len(tp) / len(pred) if pred else 0.0
    recall = len(tp) / len(true) if true else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(2 * precision * recall / (precision + recall), 3) if tp else 0.0,
            "relation_accuracy": round(sum(pred[k] == true[k] for k in tp) / len(tp), 3) if tp else 0.0,
            "false_positives": sorted(f"{k[0]} -> {k[1]}" for k in pred.keys() - true.keys()),
            "missed": sorted(f"{k[0]} -> {k[1]}" for k in true.keys() - pred.keys())}


def link_note(cfg, root, summary, note_rows):
    note = cfg["note_number"]
    items = summary_items(summary, note)
    cands = [c for item in items for c in candidates_for(item, note_rows)]
    control_path = root / cfg["control_links"]
    gold = json.loads(control_path.read_text(encoding="utf-8")).get(str(note), []) if control_path.exists() else []
    for g in gold:
        g["item_id"] = f"{g['summary_row']}:{g['col']}"

    score_method_a(items, cands, note_rows, cfg["embedding_model"])
    threshold, source = fit_threshold(cands, gold, cfg["link_threshold"])
    links_a = links_method_a(cands, note_rows, threshold)
    links_b, fallback_items, calls = run_method_b(items, cands, note_rows, cfg["llm_model"],
                                                  root / cfg["llm_cache_dir"], links_a)
    return {
        "note_number": note,
        "items": items,
        "candidates": cands,
        "method_a": {"model": cfg["embedding_model"], "weights": WEIGHTS_A, "threshold": threshold,
                     "threshold_source": source, "links": links_a},
        "method_b": {"model": cfg["llm_model"], "calls": calls, "fallback_items": fallback_items, "links": links_b},
        "comparison": compare(links_a, links_b),
        "evaluation": {"control_links": len(gold), "method_a": evaluate(links_a, gold),
                       "method_b": evaluate(links_b, gold)} if gold else None,
    }
